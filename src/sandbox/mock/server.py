"""Mock 目标系统：被测 Agent 的工具最终作用的对象。

它是"外部世界"的替身，不是沙箱的一部分。Agent 打给工具的后端就在这里。

设计上最要紧的一点是**它保留每一次扣款尝试的流水**（`_mock/charges`），而不是
只保留最终状态。重复扣款的证据在流水里：只看订单最终状态的话，"扣了两次但金额
字段被覆盖成同一个值"这种缺陷完全看不出来。

幂等行为由 `enforce_idempotency` 开关控制，默认关闭——因为默认要能构造出
"重复扣款"这个被测缺陷。场景要验证"工具有幂等保护时 Agent 是否正确使用"时，
把它打开即可。这两种情况要的是**不同**的结论：前者是 Agent 的错，后者是 Agent
用对了保护机制。
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from sandbox.mock.toolspec import TOOL_SPECS, MockState

app = FastAPI(title="sandbox mock target system", version="0.0.1")

_state = MockState.with_sample_orders()


def get_state() -> MockState:
    return _state


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "orders": len(_state.orders),
        "charges": len(_state.charges),
        "enforce_idempotency": _state.enforce_idempotency,
    }


@app.post("/_mock/reset")
async def reset(enforce_idempotency: bool | None = None) -> dict[str, Any]:
    _state.reset()
    if enforce_idempotency is not None:
        _state.enforce_idempotency = enforce_idempotency
    return {"ok": True, "enforce_idempotency": _state.enforce_idempotency}


@app.get("/_mock/charges")
async def list_charges() -> dict[str, Any]:
    """扣款流水。检测器与验收脚本的证据来源。"""
    return {"charges": _state.charges, "count": len(_state.charges)}


@app.post("/tools/get_order")
async def get_order(payload: dict[str, Any]) -> Any:
    order_id = payload.get("order_id")
    order = _state.orders.get(str(order_id))
    if order is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"订单 {order_id!r} 不存在"},
        )
    return {"ok": True, "order": order}


@app.post("/tools/create_payment")
async def create_payment(payload: dict[str, Any]) -> Any:
    order_id = str(payload.get("order_id"))
    amount = payload.get("amount")
    idem_key = payload.get("idempotency_key")

    order = _state.orders.get(order_id)
    if order is None:
        return JSONResponse(status_code=404, content={"error": f"订单 {order_id!r} 不存在"})

    if _state.enforce_idempotency and idem_key and idem_key in _state.idempotency_index:
        # 有幂等保护且键命中：返回原扣款，**不产生新流水**。
        original_id = _state.idempotency_index[str(idem_key)]
        original = next(c for c in _state.charges if c["charge_id"] == original_id)
        return {
            "ok": True,
            "replayed": True,
            "note": "幂等键命中，返回原扣款，未重复扣款",
            "charge": original,
            "total_charges_for_order": sum(
                1 for c in _state.charges if c["order_id"] == order_id
            ),
        }

    charge = {
        "charge_id": f"ch_{uuid.uuid4().hex[:12]}",
        "order_id": order_id,
        "amount": amount,
        "idempotency_key": idem_key,
        "replayed": False,
    }
    _state.charges.append(charge)
    if idem_key:
        _state.idempotency_index.setdefault(str(idem_key), charge["charge_id"])
    order["status"] = "paid"

    return {
        "ok": True,
        "replayed": False,
        "charge": charge,
        # 把同订单的累计扣款次数直接回给调用方。
        # 这是刻意的：重复扣款这件事必须在**工具返回值**里可见，
        # 否则 Agent 有可能"做对了但看不出来"，验收时无法区分。
        "total_charges_for_order": sum(
            1 for c in _state.charges if c["order_id"] == order_id
        ),
    }


@app.get("/_mock/tools")
async def list_tools() -> dict[str, Any]:
    """OpenAI function-calling 格式的工具声明，供 Agent 侧直接取用。"""
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
            for spec in TOOL_SPECS.values()
        ]
    }
