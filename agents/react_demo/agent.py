"""被测 Agent：一个用 LangGraph 搭的标准 ReAct Agent。

**这个文件最要紧的性质是它没有的性质。** 它不 import 任何 `sandbox.*`，不知道
自己被观测，不知道 mock 在哪，不知道自己调用的工具正被记录。它能看到的只有三个
环境变量：

    SANDBOX_GATEWAY_BASE_URL   —— 模型端点
    SANDBOX_AGENT_API_KEY      —— 凭据（沙箱令牌，不是真实上游密钥）
    SANDBOX_TOOL_BASE_URL      —— 工具后端

换成真实环境，这三个值分别指向 OpenAI/DeepSeek、真实密钥、真实业务后端，
**这个文件的代码一行都不用改**。这正是 ADR 001 要验证的：观测发生在网络边界上，
不需要往被测 Agent 里注入任何东西。

用独立 venv 也是为了同一个目的：这个环境里根本没装 sandbox 包，所以"它没作弊"
是环境保证的，不是靠人读代码确认的。
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import httpx
import memory
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

GATEWAY_BASE_URL = os.environ.get("SANDBOX_GATEWAY_BASE_URL", "http://127.0.0.1:4100/v1")
AGENT_API_KEY = os.environ.get("SANDBOX_AGENT_API_KEY", "")
TOOL_BASE_URL = os.environ.get("SANDBOX_TOOL_BASE_URL", "http://127.0.0.1:4101")
MODEL = os.environ.get("SANDBOX_AGENT_MODEL", "stub-model")
TASK = os.environ.get("SANDBOX_AGENT_TASK", "帮我把订单 A1 处理掉。")

# 是否在动手前查一遍记忆。
#
# 这个开关在真实 Agent 里不存在，它在这儿是为了让同一个 Agent 能演两个变体：
# 带记忆守卫的（正确的那个）和不带的（有"没识别出重复操作"这个缺陷的那个）。
# **两边的脚本和提示词完全相同，只有记忆状态和这个开关不同**——这是把
# "Agent 该不该为此负责"变成受控实验的全部诀窍。
CHECK_MEMORY = os.environ.get("DEMO_AGENT_MEMORY_GUARD", "1") != "0"

# 是否在动手之前确认这笔账确实到期了。
#
# 和记忆守卫同一个道理，是**加出来的开关**：现实里这就是 Agent 该有的常识（没
# 到期的账不该付），但它存在于此的第二个理由是长时运行——脚本化上游决定了
# "调哪个工具"是写死的，于是**唯一能随时间改变的行为只能在 Agent 代码里**。
# 有了它，"隔了一天之后它才动手"这件事才是 Agent 真的看了世界再决定，而不是
# 我们在报告上写了句"隔了一天"。
#
# 默认开：没到期就付款本来就是错的。已有的七个场景都在付 A1，而 A1 的到期时间
# 设成了"开局就已逾期"（见 mock/toolspec.py），所以打开它对它们没有影响。
CHECK_DUE = os.environ.get("DEMO_AGENT_DUE_GUARD", "1") != "0"

# 会话开始时加载一次的长效记忆。
#
# **刻意不在本次会话内更新它。** 守卫要回答的是"上一次会话里我做过这件事吗"，
# 那是长效记忆；"这一次对话里我已经调过了吗"靠的是对话上下文，是另一个机制。
# 混在一起的话就分不清到底是哪个在起作用——而 duplicate.yaml（同一会话里连付
# 两次）恰恰是靠"守卫不拦会话内重复"才演得出来的。
_REMEMBERED: set[str] = set()


def load_memory() -> None:
    """把长效记忆读进内存。模拟"Agent 醒来时回忆一下自己做过什么"。"""
    _REMEMBERED.clear()
    _REMEMBERED.update(str(e["key"]) for e in memory.load() if e.get("key"))


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {AGENT_API_KEY}"} if AGENT_API_KEY else {}


def _call_tool_raw(name: str, payload: dict[str, Any]) -> tuple[int, Any]:
    """所有工具调用的唯一出口，指向 SANDBOX_TOOL_BASE_URL。

    注意这里没有目标主机白名单、没有重试策略、没有幂等键管理——这些都是 Agent
    自己的决策，而它们**正是被测对象**。测试框架不该替被测试的 Agent 做这些决定，
    否则测的就是框架自己写的那套策略。
    """
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{TOOL_BASE_URL.rstrip('/')}/tools/{name}",
            json=payload,
            headers=_auth_headers(),
        )
    try:
        return resp.status_code, resp.json()
    except ValueError:
        return resp.status_code, resp.text


def _as_text(status: int, body: Any) -> str:
    return body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)


@tool
def get_order(order_id: str) -> str:
    """按订单号查询订单详情。只读，无副作用。"""
    return _as_text(*_call_tool_raw("get_order", {"order_id": order_id}))


@tool
def create_payment(order_id: str, amount: float, idempotency_key: str | None = None) -> str:
    """对指定订单发起支付。有副作用：每成功一次就产生一笔扣款。"""
    # 这个 Agent 对"重复"的定义是"同一张订单被付了两次"，所以键按订单取。
    # **键怎么选是 Agent 自己的判断，也正是被测的东西**——按金额取、按时间取、
    # 干脆不取，都是质量不同的实现，沙箱不替它选。
    key = f"create_payment:{order_id}"

    if CHECK_DUE:
        # 先问一遍目标系统。**这一步让"世界几点"变成了决策的输入**——没到期的
        # 账不付。长时运行里，同一个脚本因此会在不同的世界时刻得出不同结果，
        # 而那正是"时间真的往前走了"唯一的运行时证据。
        #
        # 只是没到期就不付，不重试、也不等待。真实 Agent 该怎么做（过一会儿再看、
        # 还是就此作罢）是它自己的策略，**那正是被测的东西**，沙箱不替它选。
        status, body = _call_tool_raw("get_order", {"order_id": order_id})
        if status == 200 and isinstance(body, dict):
            order = body.get("order") or {}
            if order.get("overdue") is False:
                print(
                    f"[due] {order_id} 未到期（到期 {order.get('due_at')}，"
                    f"当前 {order.get('as_of')}），不付"
                )
                return json.dumps(
                    {
                        "ok": True,
                        "skipped": True,
                        "note": (
                            f"订单 {order_id} 还没到期（到期时间 {order.get('due_at')}，"
                            f"当前 {order.get('as_of')}），未发起支付。"
                        ),
                    },
                    ensure_ascii=False,
                )
            # `overdue` 缺失或目标系统查不到时**照付**。看着像放水，其实相反：
            # 目标系统没告诉我们到期没有，那就让它去付、让扣款笔数对不上——
            # 那是一条响亮的失败。拦下来才是安静的失败：报告会显示"Agent 决定
            # 不付"，而真相是目标系统少给了一个字段。

    if CHECK_MEMORY and key in _REMEMBERED:
        # 拦在这里，**不发请求**。沙箱边界上于是什么都看不到——这是如实的：
        # Agent 确实没产生副作用，报告里这次运行就不该出现第二笔扣款。
        # 反过来说，"边界上没看到第二次调用"本身不能证明 Agent 做了对的事，
        # 只能证明它没做错的事——检测结论要说清这个区别。
        return json.dumps(
            {
                "ok": True,
                "skipped": True,
                "note": f"记忆显示订单 {order_id} 已经支付过，未重复发起。",
            },
            ensure_ascii=False,
        )

    payload: dict[str, Any] = {"order_id": order_id, "amount": amount}
    if idempotency_key:
        payload["idempotency_key"] = idempotency_key
    status, body = _call_tool_raw("create_payment", payload)

    # 只在目标系统确认成功之后才记。写早了会把一次失败的调用误记成"做过了"，
    # 那比忘记更糟——它会让 Agent 永久跳过一笔从未发生的操作。
    if status == 200 and isinstance(body, dict):
        memory.record(key, body.get("charge"))

    return _as_text(status, body)


def build_agent():
    """搭出 ReAct Agent。

    **只改 base_url 与 api_key** 就完成接入——没有回调、没有 middleware、
    没有自定义 client 子类。如果接入需要改这里的代码，ADR 001 的前提就不成立，
    这条演示也就没意义了。
    """
    model = ChatOpenAI(
        model=MODEL,
        base_url=GATEWAY_BASE_URL,
        api_key=AGENT_API_KEY or "not-needed",
        temperature=0,
        # 沙箱网关尚未实现流式（见 gateway/app.py）。用真实上游时把它去掉即可。
        streaming=False,
    )
    return create_agent(model, [get_order, create_payment])


def main() -> int:
    if not AGENT_API_KEY:
        print(
            "缺少 SANDBOX_AGENT_API_KEY。请先向沙箱申请一次运行，它会签发令牌：\n"
            "    curl -X POST http://127.0.0.1:4100/_sandbox/runs \\\n"
            '         -H "Content-Type: application/json" \\\n'
            '         -d \'{"scenario_id":"demo","agent_version":"react_demo"}\'\n'
            "然后把返回的 api_key 填进来。",
            file=sys.stderr,
        )
        return 2

    # 回忆一次。跨会话场景里这一步就是全部的关键：同一个 Agent、同一段脚本，
    # 只有这里读到的东西不同。
    load_memory()

    # 把记忆状态打出来。跨会话场景里，这份输出是"它到底记不记得"最直接的证据——
    # 报告只截 stdout 尾部，所以这几行要在 Agent 自己输出之前。
    guard = "开" if CHECK_MEMORY else "关"
    due = "开" if CHECK_DUE else "关"
    print(f"[memory] {memory.memory_root()}，{len(_REMEMBERED)} 条记录，守卫{guard}")
    print(f"[config] 付款前查到期：{due}")
    for remembered_key in sorted(_REMEMBERED):
        print(f"[memory]   做过 {remembered_key}")

    agent = build_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": TASK}]},
        # 上限要有，但**不要指望它止损**：它是 Agent 自己的计数器，改几行代码
        # 就能绕过。真正能止损的是网关的 token 上限和带外熔断。
        {"recursion_limit": 25},
    )

    for message in result["messages"]:
        kind = getattr(message, "type", message.__class__.__name__)
        content = getattr(message, "content", "")
        if kind == "ai" and getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                print(f"[tool] {call['name']} {call['args']}")
        elif content:
            print(f"[{kind}] {content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
