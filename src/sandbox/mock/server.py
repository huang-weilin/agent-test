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
from sandbox.mock.toolspec import (
    CATALOG_PAGE_SIZE,
    SAMPLE_STOCK,
    TOOL_SPECS,
    MockState,
    catalog_page,
)

app = FastAPI(title="sandbox mock target system", version="0.0.1")

_state = MockState.with_sample_orders()


def get_state() -> MockState:
    return _state


def _state_stock(item: Any) -> dict[str, Any] | None:
    """查库存表。返回副本（只读视图），不认识的商品返回 None。"""
    stock = SAMPLE_STOCK.get(str(item))
    return dict(stock) if stock is not None else None


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

    charge_id = f"ch_{uuid.uuid4().hex[:12]}"
    charge = {
        "charge_id": charge_id,
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
        # 用局部变量而不是 `charge["charge_id"]`：那张流水表是混合值的字典，
        # 从里面取出来的东西类型上是 `object`，而索引表要的是字符串。
        _state.idempotency_index.setdefault(str(idem_key), charge_id)
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


@app.post("/tools/check_stock")
async def check_stock(payload: dict[str, Any]) -> Any:
    """查询商品库存。只读——它不改任何状态，重复查询是干净的。"""
    item = payload.get("item")
    stock = _state_stock(item)
    if stock is None:
        return JSONResponse(
            status_code=404, content={"error": f"商品 {item!r} 不在库存表里"}
        )
    return {"ok": True, "stock": dict(stock)}


@app.post("/tools/ship_order")
async def ship_order(payload: dict[str, Any]) -> Any:
    """对已支付订单发货。有副作用：每成功一次就产生一条发货流水。

    业务规则只有一条：**没付过钱的订单不能发货**（`status` 不是 `paid`/`shipped`
    就拒）。规则本身不是给检测器用的——它是"真实系统会怎么回"的模型：真实物流
    系统同样拒绝未支付订单，而**重复发货**（对已发货订单再发一次）真实系统往往
    不拦，这正是能构造出来的缺陷。
    """
    order_id = str(payload.get("order_id"))
    order = _state.orders.get(order_id)
    if order is None:
        return JSONResponse(status_code=404, content={"error": f"订单 {order_id!r} 不存在"})

    idem_key = payload.get("idempotency_key")
    if _state.enforce_idempotency and idem_key and idem_key in _state.shipment_idem:
        original_id = _state.shipment_idem[str(idem_key)]
        original = next(c for c in _state.shipments if c["shipment_id"] == original_id)
        return {
            "ok": True,
            "replayed": True,
            "note": "幂等键命中，返回原发货流水，未重复发货",
            "shipment": original,
            "total_shipments_for_order": sum(
                1 for c in _state.shipments if c["order_id"] == order_id
            ),
        }

    if order["status"] in ("unpaid", "refunded"):
        return JSONResponse(
            status_code=409,
            content={
                "error": f"订单 {order_id} 当前状态 {order['status']}，不能发货"
            },
        )

    shipment_id = f"sh_{uuid.uuid4().hex[:12]}"
    shipment = {
        "shipment_id": shipment_id,
        "order_id": order_id,
        "carrier": payload.get("carrier"),
        "tracking_no": payload.get("tracking_no"),
        "idempotency_key": idem_key,
        "replayed": False,
        "created_at": clock.now().isoformat(),
    }
    _state.shipments.append(shipment)
    if idem_key:
        _state.shipment_idem.setdefault(str(idem_key), shipment_id)
    order["status"] = "shipped"

    return {
        "ok": True,
        "replayed": False,
        "shipment": shipment,
        "total_shipments_for_order": sum(
            1 for c in _state.shipments if c["order_id"] == order_id
        ),
    }


@app.post("/tools/send_notice")
async def send_notice(payload: dict[str, Any]) -> Any:
    """向用户发送订单通知。有副作用：每成功一次就产生一条通知流水。

    通知不改订单状态，但它是写操作——真实的通知服务对同一件事发两次会真的
    发出两条消息。**重复通知**因此是能构造出来的缺陷（收件人被同一句话打扰
    两次），与"重复扣款"同类，只是副作用形态不同。
    """
    order_id = str(payload.get("order_id"))
    order = _state.orders.get(order_id)
    if order is None:
        return JSONResponse(status_code=404, content={"error": f"订单 {order_id!r} 不存在"})

    idem_key = payload.get("idempotency_key")
    if _state.enforce_idempotency and idem_key and idem_key in _state.notice_idem:
        original_id = _state.notice_idem[str(idem_key)]
        original = next(c for c in _state.notices if c["notice_id"] == original_id)
        return {
            "ok": True,
            "replayed": True,
            "note": "幂等键命中，返回原通知流水，未重复发送",
            "notice": original,
            "total_notices_for_order": sum(
                1 for c in _state.notices if c["order_id"] == order_id
            ),
        }

    notice_id = f"nt_{uuid.uuid4().hex[:12]}"
    notice = {
        "notice_id": notice_id,
        "order_id": order_id,
        "notice_type": payload.get("notice_type"),
        "message": payload.get("message"),
        "idempotency_key": idem_key,
        "replayed": False,
        "created_at": clock.now().isoformat(),
    }
    _state.notices.append(notice)
    if idem_key:
        _state.notice_idem.setdefault(str(idem_key), notice_id)

    return {
        "ok": True,
        "replayed": False,
        "notice": notice,
        "total_notices_for_order": sum(
            1 for c in _state.notices if c["order_id"] == order_id
        ),
    }


@app.post("/tools/create_refund")
async def create_refund(payload: dict[str, Any]) -> Any:
    """对已支付订单发起退款。有副作用：每成功一次就产生一笔退款流水。

    与 create_payment 同一纪律：**重复退款**（对同一笔已退款再退一次）真实
    系统往往不拦，于是它能被构造出来，而退款流水里的两笔记录就是证据。
    """
    order_id = str(payload.get("order_id"))
    order = _state.orders.get(order_id)
    if order is None:
        return JSONResponse(status_code=404, content={"error": f"订单 {order_id!r} 不存在"})

    idem_key = payload.get("idempotency_key")
    if _state.enforce_idempotency and idem_key and idem_key in _state.refund_idem:
        original_id = _state.refund_idem[str(idem_key)]
        original = next(c for c in _state.refunds if c["refund_id"] == original_id)
        return {
            "ok": True,
            "replayed": True,
            "note": "幂等键命中，返回原退款流水，未重复退款",
            "refund": original,
            "total_refunds_for_order": sum(
                1 for c in _state.refunds if c["order_id"] == order_id
            ),
        }

    if order["status"] == "unpaid":
        return JSONResponse(
            status_code=409,
            content={"error": f"订单 {order_id} 尚未支付，不能退款"},
        )

    refund_id = f"rf_{uuid.uuid4().hex[:12]}"
    refund = {
        "refund_id": refund_id,
        "order_id": order_id,
        "amount": payload.get("amount"),
        "reason": payload.get("reason"),
        "idempotency_key": idem_key,
        "replayed": False,
        "created_at": clock.now().isoformat(),
    }
    _state.refunds.append(refund)
    if idem_key:
        _state.refund_idem.setdefault(str(idem_key), refund_id)
    order["status"] = "refunded"

    return {
        "ok": True,
        "replayed": False,
        "refund": refund,
        "total_refunds_for_order": sum(
            1 for c in _state.refunds if c["order_id"] == order_id
        ),
    }


@app.post("/tools/create_plan")
async def create_plan(payload: dict[str, Any]) -> Any:
    """在任务计划板上创建一份计划。有副作用：同 id 重复创建会**覆盖**旧计划。

    覆盖是刻意的：真实工单系统里同 id 的计划只该有一份，第二次创建是"改写"
    不是"新增"——而一个 Agent 在不知道已有计划的情况下重新创建（覆盖掉别人
    写好的步骤），正是"规划-执行"类 Agent 的典型缺陷（计划被自己的无知冲掉）。
    场景要测它时，`plan_id` 就是判据：两次 create_plan 用同一个 id、步骤不同，
    计划板上只剩后一份。
    """
    plan_id = str(payload.get("plan_id"))
    steps = payload.get("steps")
    if not isinstance(steps, list) or not all(isinstance(x, str) for x in steps):
        return JSONResponse(
            status_code=400,
            content={"error": f"steps 必须是字符串数组，收到 {steps!r}"},
        )
    plan = {
        "plan_id": plan_id,
        "steps": list(steps),
        "done": [],
        "created_at": clock.now().isoformat(),
    }
    _state.plans[plan_id] = plan
    return {"ok": True, "plan": plan, "replaced": True}


@app.post("/tools/get_plan")
async def get_plan(payload: dict[str, Any]) -> Any:
    plan_id = str(payload.get("plan_id"))
    plan = _state.plans.get(plan_id)
    if plan is None:
        return JSONResponse(
            status_code=404, content={"error": f"计划 {plan_id!r} 不存在"}
        )
    return {"ok": True, "plan": dict(plan)}


@app.post("/tools/mark_step_done")
async def mark_step_done(payload: dict[str, Any]) -> Any:
    plan_id = str(payload.get("plan_id"))
    plan = _state.plans.get(plan_id)
    if plan is None:
        return JSONResponse(
            status_code=404, content={"error": f"计划 {plan_id!r} 不存在"}
        )
    step_index = payload.get("step_index")
    if not isinstance(step_index, int) or isinstance(step_index, bool):
        return JSONResponse(
            status_code=400,
            content={"error": f"step_index 必须是整数，收到 {step_index!r}"},
        )
    if step_index < 0 or step_index >= len(plan["steps"]):
        return JSONResponse(
            status_code=400,
            content={"error": f"step_index {step_index} 超出计划步骤范围"},
        )
    if step_index not in plan["done"]:
        plan["done"].append(step_index)
        plan["done"].sort()
    return {"ok": True, "plan": dict(plan), "already_done": step_index in plan["done"]}


@app.post("/tools/handoff")
async def handoff(payload: dict[str, Any]) -> Any:
    """把任务转交给另一个环节。有副作用：每转交一次就产生一条交接流水。

    交接本身不改订单状态——它记录的是"这一段的 Agent 决定把任务交给谁"。
    多 Agent 接力场景里，这条流水是**交接动作在边界上唯一的证据**：0 号调了
    handoff、下一段会话由 1 号跑，报告上才能说"0 号明确转交了，不是自己放弃"。
    重复转交（同一条任务转两次）是能构造出来的缺陷，流水里会有两条记录。
    """
    target_role = str(payload.get("target_role"))
    task = str(payload.get("task", ""))
    handoff_id = f"hd_{uuid.uuid4().hex[:12]}"
    record = {
        "handoff_id": handoff_id,
        "target_role": target_role,
        "task": task,
        "created_at": clock.now().isoformat(),
    }
    _state.handoffs.append(record)
    return {
        "ok": True,
        "handoff": record,
        "total_handoffs": len(_state.handoffs),
    }


@app.get("/_mock/handoffs")
async def list_handoffs() -> dict[str, Any]:
    """交接流水。多 Agent 接力场景的证据来源。"""
    return {"handoffs": _state.handoffs, "count": len(_state.handoffs)}


@app.get("/_mock/shipments")
async def list_shipments() -> dict[str, Any]:
    """发货流水。检测器与验收脚本的证据来源（与 `/_mock/charges` 同一纪律）。"""
    return {"shipments": _state.shipments, "count": len(_state.shipments)}


@app.get("/_mock/notices")
async def list_notices() -> dict[str, Any]:
    """通知流水。"""
    return {"notices": _state.notices, "count": len(_state.notices)}


@app.get("/_mock/refunds")
async def list_refunds() -> dict[str, Any]:
    """退款流水。"""
    return {"refunds": _state.refunds, "count": len(_state.refunds)}


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
