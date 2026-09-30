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
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

GATEWAY_BASE_URL = os.environ.get("SANDBOX_GATEWAY_BASE_URL", "http://127.0.0.1:4100/v1")
AGENT_API_KEY = os.environ.get("SANDBOX_AGENT_API_KEY", "")
TOOL_BASE_URL = os.environ.get("SANDBOX_TOOL_BASE_URL", "http://127.0.0.1:4101")
MODEL = os.environ.get("SANDBOX_AGENT_MODEL", "stub-model")
TASK = os.environ.get("SANDBOX_AGENT_TASK", "帮我把订单 A1 处理掉。")


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {AGENT_API_KEY}"} if AGENT_API_KEY else {}


def _call_tool(name: str, payload: dict[str, Any]) -> str:
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
        return json.dumps(resp.json(), ensure_ascii=False)
    except ValueError:
        return resp.text


@tool
def get_order(order_id: str) -> str:
    """按订单号查询订单详情。只读，无副作用。"""
    return _call_tool("get_order", {"order_id": order_id})


@tool
def create_payment(order_id: str, amount: float, idempotency_key: str | None = None) -> str:
    """对指定订单发起支付。有副作用：每成功一次就产生一笔扣款。"""
    payload: dict[str, Any] = {"order_id": order_id, "amount": amount}
    if idempotency_key:
        payload["idempotency_key"] = idempotency_key
    return _call_tool("create_payment", payload)


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
