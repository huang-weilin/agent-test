"""被测 Agent：履约发货 Agent（接手发货 + 通知）。

角色是**履约执行者**：接到"继续履约订单 X"的任务，先查订单确认它已经支付
（真实决策落成代码开关），然后发货、发通知，把结果写进长效记忆。它和
`order_agent` 是一对：0 号卖出并转交，1 号接手交付——串联场景里
`sessions[].agent` 决定这一段由谁跑。

**硬约束（与仓库其他 Agent 相同）**：不 import 任何 `sandbox.*`；不知道
自己在被观测。
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
TASK = os.environ.get("SANDBOX_AGENT_TASK", "你接手继续履约订单 A1：确认已支付后发货并通知。")

# 发货前确认订单已支付（查一次单）。关掉这个守卫的变体"不确认就发货"。
CHECK_PAID = os.environ.get("DEMO_AGENT_PAID_GUARD", "1") != "0"
# 记忆防重复：同一订单只发一次货、同一通知只发一次。
CHECK_MEMORY = os.environ.get("DEMO_AGENT_MEMORY_GUARD", "1") != "0"

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
    return _post_json(
        f"{TOOL_BASE_URL.rstrip('/')}/tools/{name}", payload, _auth_headers()
    )


def _as_text(status: int, body: Any) -> str:
    return body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)


def _llm(messages: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
    }
    status, body = _post_json(f"{GATEWAY_BASE_URL.rstrip('/')}/chat/completions", payload)
    if status != 200 or not isinstance(body, dict):
        raise RuntimeError(f"LLM 调用失败 status={status} body={body}")
    return body


def get_order(order_id: str) -> str:
    """查询订单详情。只读，无副作用。"""
    status, body = _call_tool_raw("get_order", {"order_id": order_id})
    if status == 200 and isinstance(body, dict):
        order = body.get("order") or {}
        note = order.get("note")
        if isinstance(note, str) and note.strip():
            memory.record(f"order_note:{order_id}", note)
    return _as_text(status, body)


def ship_order(order_id: str, carrier: str) -> str:
    """发货。有副作用：每成功一次就产生一条发货流水。"""
    key = f"ship_order:{order_id}"

    if CHECK_PAID:
        # 发货前确认订单已支付（真实决策落成代码开关）。
        order = get_order(order_id)
        print(f"[guard] 发货前确认支付：{order}")

    if CHECK_MEMORY and key in _REMEMBERED:
        return json.dumps(
            {
                "ok": True,
                "skipped": True,
                "note": f"记忆显示订单 {order_id} 已经发过货，未重复发货。",
            },
            ensure_ascii=False,
        )

    status, body = _call_tool_raw("ship_order", {"order_id": order_id, "carrier": carrier})
    if status == 200 and isinstance(body, dict) and body.get("ok"):
        memory.record(key, body.get("shipment"))
    return _as_text(status, body)


def send_notice(order_id: str, notice_type: str, message: str | None = None) -> str:
    """向用户发送订单通知。有副作用：每成功一次就产生一条通知流水。"""
    key = f"send_notice:{order_id}:{notice_type}"

    if CHECK_MEMORY and key in _REMEMBERED:
        return json.dumps(
            {
                "ok": True,
                "skipped": True,
                "note": f"记忆显示 {order_id} 的 {notice_type} 通知已经发过，未重复发送。",
            },
            ensure_ascii=False,
        )

    payload: dict[str, Any] = {"order_id": order_id, "notice_type": notice_type}
    if message:
        payload["message"] = message
    status, body = _call_tool_raw("send_notice", payload)
    if status == 200 and isinstance(body, dict) and body.get("ok"):
        memory.record(key, body.get("notice"))
    return _as_text(status, body)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "查询订单详情。只读。",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ship_order",
            "description": "给订单发货。有副作用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "carrier": {"type": "string", "description": "承运商"},
                },
                "required": ["order_id", "carrier"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_notice",
            "description": "向用户发送订单通知。有副作用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "notice_type": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["order_id", "notice_type"],
            },
        },
    },
]

_DISPATCH = {
    "get_order": lambda a: get_order(str(a.get("order_id", ""))),
    "ship_order": lambda a: ship_order(str(a.get("order_id", "")), str(a.get("carrier", ""))),
    "send_notice": lambda a: send_notice(
        str(a.get("order_id", "")), str(a.get("notice_type", "")), str(a.get("message") or "")
    ),
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
    print(f"[config] 发货前确认支付：{'开' if CHECK_PAID else '关'}")

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "你是履约执行者。接到任务先确认订单已支付，然后发货并通知用户。",
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
