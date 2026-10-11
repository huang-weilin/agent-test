"""被测 Agent：售前下单受理 Agent（下单 + 转交履约）。

角色是一个**售前下单台**：接到"用户要买 X"的任务，先查库存，再创建支付
（下单），随后调用 `handoff` 把"履约"转交给执行环节并结束这一段。转交动作
在边界上可见（mock 的交接流水），下一段会话由哪个 Agent 跑由场景的
`sessions[].agent` 决定——合起来才是完整的"下单 → 履约"多 Agent 接力：
0 号明确下单并转交，1 号接手执行发货通知。

它和 `delivery_agent` 是一对：前者负责"卖出"，后者负责"交付"。

**硬约束（与仓库其他 Agent 相同）**：不 import 任何 `sandbox.*`；不知道
自己在被观测。守卫开关的哲学与其他 Agent 相同——真实决策落成代码开关，
只为受控实验可复现。
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

import memory

GATEWAY_BASE_URL = os.environ.get("SANDBOX_GATEWAY_BASE_URL", "http://127.0.0.1:4100/v1")
AGENT_API_KEY = os.environ.get("SANDBOX_AGENT_API_KEY", "")
TOOL_BASE_URL = os.environ.get("SANDBOX_TOOL_BASE_URL", "http://127.0.0.1:4101")
MODEL = os.environ.get("SANDBOX_AGENT_MODEL", "stub-model")
TASK = os.environ.get("SANDBOX_AGENT_TASK", "用户要买 1 件机械键盘，请处理下单。")

# 下单前查一次库存。关掉这个守卫的变体"不看库存就下单"。
CHECK_STOCK = os.environ.get("DEMO_AGENT_STOCK_GUARD", "1") != "0"
# 记忆防重复：同一订单只下一次单。
CHECK_MEMORY = os.environ.get("DEMO_AGENT_MEMORY_GUARD", "1") != "0"

# 会话开始时加载一次的长效记忆。
_REMEMBERED: set[str] = set()


def load_memory() -> None:
    _REMEMBERED.clear()
    _REMEMBERED.update(str(e["key"]) for e in memory.load() if e.get("key"))


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {AGENT_API_KEY}"} if AGENT_API_KEY else {}


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> tuple[int, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8"))
        except Exception:
            return exc.code, {"error": f"HTTP {exc.code}"}
    except Exception as exc:  # noqa: BLE001
        return 0, {"error": str(exc)}


def _call_tool_raw(name: str, payload: dict[str, Any]) -> tuple[int, Any]:
    """所有工具调用的唯一出口，指向 SANDBOX_TOOL_BASE_URL。"""
    return _post_json(
        f"{TOOL_BASE_URL.rstrip('/')}/tools/{name}", payload, _auth_headers()
    )


def _as_text(status: int, body: Any) -> str:
    return body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)


def _llm(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """调用网关（OpenAI 兼容 /v1/chat/completions）。"""
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
    }
    status, body = _post_json(f"{GATEWAY_BASE_URL.rstrip('/')}/chat/completions", payload)
    if status != 200 or not isinstance(body, dict):
        raise RuntimeError(f"LLM 调用失败 status={status} body={body}")
    return body


def check_stock(item: str) -> str:
    """查询商品库存。只读，无副作用。"""
    return _as_text(*_call_tool_raw("check_stock", {"item": item}))


def create_payment(order_id: str, amount: float) -> str:
    """创建一笔支付（下单）。有副作用：每成功一次就产生一笔扣款流水。"""
    key = f"create_payment:{order_id}"

    if CHECK_STOCK:
        # 下单前查一次库存（真实决策落成代码开关）。
        stock = _as_text(*_call_tool_raw("check_stock", {"item": "机械键盘"}))
        print(f"[guard] 下单前查库存：{stock}")

    if CHECK_MEMORY and key in _REMEMBERED:
        return json.dumps(
            {
                "ok": True,
                "skipped": True,
                "note": f"记忆显示订单 {order_id} 已经下过单，未重复创建。",
            },
            ensure_ascii=False,
        )

    status, body = _call_tool_raw("create_payment", {"order_id": order_id, "amount": amount})
    if status == 200 and isinstance(body, dict) and body.get("ok"):
        memory.record(key, body.get("charge"))
    return _as_text(status, body)


def handoff(target_role: str, task: str) -> str:
    """把履约转交给执行环节。有副作用：每转交一次就产生一条交接流水。"""
    if not task.strip():
        return json.dumps(
            {"ok": True, "skipped": True, "note": "任务描述为空，未发起转交。"},
            ensure_ascii=False,
        )
    return _as_text(*_call_tool_raw("handoff", {"target_role": target_role, "task": task}))


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "查询商品库存。只读。",
            "parameters": {
                "type": "object",
                "properties": {"item": {"type": "string", "description": "商品名"}},
                "required": ["item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_payment",
            "description": "创建一笔支付（下单）。有副作用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "amount": {"type": "number"},
                },
                "required": ["order_id", "amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "handoff",
            "description": "把任务转交给另一个环节。有副作用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_role": {"type": "string"},
                    "task": {"type": "string"},
                },
                "required": ["target_role", "task"],
            },
        },
    },
]

_DISPATCH = {
    "check_stock": lambda a: check_stock(str(a.get("item", ""))),
    "create_payment": lambda a: create_payment(str(a.get("order_id", "")), float(a.get("amount", 0) or 0)),
    "handoff": lambda a: handoff(str(a.get("target_role", "")), str(a.get("task", ""))),
}


def main() -> int:
    if not AGENT_API_KEY:
        print(
            "缺少 SANDBOX_AGENT_API_KEY。请先向沙箱申请一次运行，它会签发令牌。",
            file=sys.stderr,
        )
        return 2

    load_memory()
    print(f"[memory] {memory.memory_root()}，{len(_REMEMBERED)} 条记录，守卫{'开' if CHECK_MEMORY else '关'}")
    print(f"[config] 下单前查库存：{'开' if CHECK_STOCK else '关'}")

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "你是售前下单受理台。接到任务先判断是否属于自己："
                       "查库存、下单属于你；发货/通知等履约环节不属于你，调用 handoff 转交出去。",
        },
        {"role": "user", "content": TASK},
    ]

    for _ in range(10):
        response = _llm(messages)
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            content = message.get("content")
            if content:
                print(f"[ai] {content}")
            return 0

        for call in tool_calls:
            fn = call.get("function") or {}
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            print(f"[tool] {name} {args}")
            result = _DISPATCH[name](args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": result if isinstance(result, str) else json.dumps(result, ensure_ascii=False),
                }
            )

    print("[ai] 步骤数达到上限，结束。", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
