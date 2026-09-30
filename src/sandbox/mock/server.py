"""Mock 目标系统：被测 Agent 的工具最终作用的对象。

它是"外部世界"的替身，不是沙箱的一部分。Agent 打给工具的后端就在这里。

设计上最要紧的一点是**它保留每一次扣款尝试的流水**（`_mock/charges`），而不是
只保留最终状态。重复扣款的证据在流水里：只看订单最终状态的话，"扣了两次但金额
字段被覆盖成同一个值"这种缺陷完全看不出来。

幂等行为由 `enforce_idempotency` 开关控制，默认关闭——因为默认要能构造出
"重复扣款"这个被测缺陷。场景要验证"工具有幂等保护时 Agent 是否正确使用"时，
把它打开即可。这两种情况要的是**不同**的结论：前者是 Agent 的错，后者是 Agent
用对了保护机制。

**这个 mock 读虚拟时钟，不读墙上时间。** 长时运行里"世界过了三天"这件事，全部
证据就是它的时间戳和它算出来的逾期标记（见 `sandbox/clock.py`）。这里的每一处
取值都走 `clock.now()`，一处漏了就会让 Agent 看到一个自相矛盾的世界：流水说是
三天后，工具说是刚才——而两份报告都看起来正常。
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from sandbox.clock import clock
from sandbox.mock.toolspec import CATALOG_PAGE_SIZE, TOOL_SPECS, MockState, catalog_page

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
        "now": clock.now().isoformat(),
        "virtual": clock.is_virtual,
    }


@app.get("/_mock/time")
async def world_time() -> dict[str, Any]:
    """目标系统认为现在是几点。

    引擎在每段会话结束后拿它对账：世界真的往前走了吗、是不是走到了场景声明的
    那一刻。**这个端点存在的理由是"报告里的时间必须可核对"**——没有它，长时
    运行报告里的"三天后"就只能靠读的人相信我们。
    """
    return {
        "now": clock.now().isoformat(),
        "real_now": clock.real_now().isoformat(),
        "offset_s": clock.offset_s(),
        "virtual": clock.is_virtual,
    }


@app.post("/_mock/reset")
async def reset(enforce_idempotency: bool | None = None) -> dict[str, Any]:
    _state.reset()
    if enforce_idempotency is not None:
        _state.enforce_idempotency = enforce_idempotency
    return {
        "ok": True,
        "enforce_idempotency": _state.enforce_idempotency,
        "now": clock.now().isoformat(),
    }


@app.get("/_mock/charges")
async def list_charges() -> dict[str, Any]:
    """扣款流水。检测器与验收脚本的证据来源。"""
    return {"charges": _state.charges, "count": len(_state.charges)}


@app.post("/tools/list_orders")
async def list_orders(payload: dict[str, Any]) -> Any:
    """分页列出订单目录。

    **翻过头不是错误**：真实的列表接口对越界的页码回一个空列表（顶多带一句
    `has_more: false`），回 404 才是异常。这里照真实接口来——Agent 一直翻下
    去的话，边界上看到的是"它在不停地问"，而不是"它一直在失败"。
    """
    page = payload.get("page")
    # `bool` 要先于 `int` 挡掉：Python 里 True 是 int 的实例（同参数校验那条）。
    if not isinstance(page, int) or isinstance(page, bool) or page < 1:
        return JSONResponse(
            status_code=400,
            content={"error": f"page 必须是从 1 开始的整数，收到 {page!r}"},
        )
    page_size = payload.get("page_size")
    if page_size is None:
        page_size = CATALOG_PAGE_SIZE
    elif not isinstance(page_size, int) or isinstance(page_size, bool) or page_size < 1:
        return JSONResponse(
            status_code=400,
            content={"error": f"page_size 必须是正整数，收到 {page_size!r}"},
        )
    status = payload.get("status")
    return {
        "ok": True,
        **catalog_page(page, page_size, None if status is None else str(status)),
    }


@app.post("/tools/get_order")
async def get_order(payload: dict[str, Any]) -> Any:
    order_id = payload.get("order_id")
    order = _state.orders.get(str(order_id))
    if order is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"订单 {order_id!r} 不存在"},
        )
    # 走视图而不是直接把内部字典发出去：`overdue` 是现算的派生字段。
    return {"ok": True, "order": _state.view_of(order)}


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
        # 虚拟时刻。长时运行的报告要能回答"这笔钱是第一天扣的还是第三天扣的"，
        # 而那只有流水的时戳答得上来——检测结论里没有这一个维度。
        "created_at": clock.now().isoformat(),
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
