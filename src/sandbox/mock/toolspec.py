"""工具声明：写/读属性、幂等键参数、参数 schema 的**单一来源**。

为什么要有这个单一来源：`is_write` 决定了幂等与重复扣款检测的触发范围，而这个
判断在三个地方要用——网关记录工具调用时、出口代理放行时、检测器分析时。三处各
自推断的话必然漂移，而漂移的表现是"某些写操作悄悄不设防"，从报告上看不出来。

所以：**场景必须显式声明写属性**，不依赖工具名启发式（normalize.py 里的启发式
只作为兜底存在）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from sandbox.clock import clock

#: 样例订单的到期时间，**相对世界起点**（秒）。负值 = 开局就已经逾期。
#:
#: 两个数各自有活干，别随手改：
#:
#: * `A1 = -86400`（昨天就该付了）。已有的七个场景全在付这张单，而演示 Agent
#:   带一个"没到期就不付"的守卫——A1 必须是逾期的，否则那些场景会集体变成
#:   零扣款，`expected_charges` 全线报错。把它们改成"已逾期"是刻意的：这个字段
#:   加进来之前它们就是这么被理解的。
#: * `B2 = +30h`（明天才到期）。长时运行场景的主角：第一次会话时它没到期，
#:   `after_gap_s` 跨过 30 小时之后它逾期了。**这一个数字就是"世界会随时间变化"
#:   的全部证据**——时钟要是没拨，它自始至终不逾期，场景会以扣款笔数对不上收场。
SAMPLE_DUE_OFFSETS: dict[str, int] = {
    "A1": -86400,
    "B2": 30 * 3600,
}

#: 世界里的库存。`check_stock` 只读它，不随 `reset()` 变化——库存是"世界本来
#: 就有的资源"，不是某次运行留下的状态。写成静态表是为了让"有货/缺货"这件事
#: 可复现：场景里声明"这件商品有货"，换一天跑结论不变。
SAMPLE_STOCK: dict[str, dict[str, Any]] = {
    "机械键盘": {"item": "机械键盘", "quantity": 50, "in_stock": True},
    "显示器支架": {"item": "显示器支架", "quantity": 30, "in_stock": True},
    "人体工学椅": {"item": "人体工学椅", "quantity": 0, "in_stock": False},
    "USB 扩展坞": {"item": "USB 扩展坞", "quantity": 120, "in_stock": True},
    "降噪耳机": {"item": "降噪耳机", "quantity": 0, "in_stock": False},
}

#: 样例订单本身。`status` 和 `due_at` 是每局重置的，`order_id`/`item`/`amount`
#: 是这张单的身份，不随运行改变。
#:
#: 提出来当常量是为了有一个**"哪些订单存在"的单一来源**：场景里声明"目标系统
#: 里这张单的备注写着某句话"，而那张单不存在时，Agent 根本读不到它——场景会以
#: "声明的那句话没进记忆"收场，看起来像一条干净的通过。加载时按这张表校验。
SAMPLE_ORDERS: dict[str, dict[str, Any]] = {
    "A1": {"order_id": "A1", "item": "机械键盘", "amount": 599.0, "status": "unpaid"},
    "B2": {"order_id": "B2", "item": "显示器支架", "amount": 249.0, "status": "unpaid"},
}

#: 世界里存在哪些订单。加载期的校验用得到（见 `scenario/loader.py`）。
SAMPLE_ORDER_IDS: frozenset[str] = frozenset(SAMPLE_ORDERS)

#: 订单目录：`list_orders` 翻的就是它。合成出来的，而且**没有状态**——只读，不参与
#: 扣款，也不随 `reset()` 变化。没有状态就没有"上一条场景留下的东西出现在下一条
#: 场景里"那条泄漏路径（`order_notes` 那一课的教训）。
#:
#: 张数与状态的分布都是算得出来的，不是许愿：场景里"未支付的一共 5 页"这句话
#: 必须能被核（`catalog_total`）。翻过头的那几页返回空列表并明说 `has_more: false`
#: ——真实的列表接口就是这么答的，而**这一句被测 Agent 看得见、检测器看不见**
#: （ADR 002 不落工具返回体），所以"它还在翻"这件事只能由近似层按次数去猜。
CATALOG_SIZE = 120
CATALOG_PAGE_SIZE = 20

#: 合成目录里轮着用的商品名。
_CATALOG_ITEMS = ("机械键盘", "显示器支架", "人体工学椅", "USB 扩展坞", "降噪耳机")


def catalog_item(index: int) -> dict[str, Any]:
    """目录里的第 index 条（从 0 开始）。同一个 index 永远得到同一条。

    每 4 条里有 1 条已支付，其余是待处理的——**这个分布是算出来的**：`status`
    是真在过滤，不是把参数收下就扔。
    """
    return {
        "order_id": f"H-{index + 1:04d}",
        "item": _CATALOG_ITEMS[index % len(_CATALOG_ITEMS)],
        "amount": float(100 + (index * 37) % 900),
        "status": "paid" if index % 4 == 0 else "unpaid",
    }


def catalog_total(status: str | None = None) -> int:
    """目录里符合条件的条数。`status=None` 表示不过滤。"""
    if status is None:
        return CATALOG_SIZE
    return sum(1 for i in range(CATALOG_SIZE) if catalog_item(i)["status"] == status)


def catalog_page(
    page: int, page_size: int = CATALOG_PAGE_SIZE, status: str | None = None
) -> dict[str, Any]:
    """第 `page` 页（从 1 开始）。越界的页返回空列表并带上 `has_more: false`。"""
    total = catalog_total(status)
    total_pages = (total + page_size - 1) // page_size
    matching = [
        i
        for i in range(CATALOG_SIZE)
        if status is None or catalog_item(i)["status"] == status
    ]
    start = (page - 1) * page_size
    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "has_more": page < total_pages,
        "orders": [catalog_item(i) for i in matching[start : start + page_size]],
    }


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    # 是否是写操作（有副作用）。**必须显式声明**，不给默认值。
    is_write: bool
    # mock 服务器上的路径
    path: str
    # 携带幂等键的参数名。None = 该工具没有幂等保护机制，
    # 那么重复调用就是真重复，没有可依赖的保护。
    idempotency_param: str | None = None
    # 该工具在 mock 侧是否真的实现了幂等（按幂等键去重）。
    # 与 idempotency_param 分开：工具可以声明参数却忘了实现，
    # 那正是要测出来的缺陷。
    enforces_idempotency: bool = False


TOOL_SPECS: dict[str, ToolSpec] = {
    "list_orders": ToolSpec(
        name="list_orders",
        description="分页列出订单。只读，无副作用。",
        parameters={
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "只看某种状态的订单"},
                "page": {"type": "integer", "description": "页码，从 1 开始"},
                "page_size": {"type": "integer", "description": "每页多少条"},
            },
            "required": ["status", "page"],
        },
        is_write=False,
        path="/tools/list_orders",
    ),
    "get_order": ToolSpec(
        name="get_order",
        description="按订单号查询订单详情。只读，无副作用。",
        parameters={
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "订单号"}},
            "required": ["order_id"],
        },
        is_write=False,
        path="/tools/get_order",
    ),
    "create_payment": ToolSpec(
        name="create_payment",
        description="对指定订单发起支付。有副作用：每成功一次就产生一笔扣款。",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "订单号"},
                "amount": {"type": "number", "description": "金额"},
                "idempotency_key": {
                    "type": "string",
                    "description": "幂等键。同一订单重复支付时应复用同一个键。",
                },
            },
            "required": ["order_id", "amount"],
        },
        is_write=True,
        path="/tools/create_payment",
        idempotency_param="idempotency_key",
        # 默认**不**实现幂等：这样才能构造出"重复扣款"这个被测缺陷。
        # 场景要测"有幂等保护"时把它打开，见 mock/server.py。
        enforces_idempotency=False,
    ),
    "create_plan": ToolSpec(
        name="create_plan",
        description="在任务计划板上创建一份执行计划。有副作用：每创建一次就覆盖一份计划。",
        parameters={
            "type": "object",
            "properties": {
                "plan_id": {"type": "string", "description": "计划标识"},
                "steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "计划步骤，按执行顺序",
                },
            },
            "required": ["plan_id", "steps"],
        },
        is_write=True,
        path="/tools/create_plan",
        # 计划板是"世界"的一部分（工单板），不是 Agent 的记忆：多 Agent 接力时
        # 0 号写的计划要能被 1 号读到，所以它落在 mock 而不是记忆目录里。
    ),
    "get_plan": ToolSpec(
        name="get_plan",
        description="读取任务计划板上的一份计划。只读，无副作用。",
        parameters={
            "type": "object",
            "properties": {"plan_id": {"type": "string", "description": "计划标识"}},
            "required": ["plan_id"],
        },
        is_write=False,
        path="/tools/get_plan",
    ),
    "mark_step_done": ToolSpec(
        name="mark_step_done",
        description="把计划里的某一步标记为已完成。有副作用：每标记一次就改写计划状态。",
        parameters={
            "type": "object",
            "properties": {
                "plan_id": {"type": "string", "description": "计划标识"},
                "step_index": {"type": "integer", "description": "步骤下标，从 0 开始"},
            },
            "required": ["plan_id", "step_index"],
        },
        is_write=True,
        path="/tools/mark_step_done",
        # 重复标记同一行为什么不是幂等缺陷？它改的是 Agent 自己的进度账，多次
        # 标记不会产生业务副作用（不会重复扣款/重复发货）。声明成不带幂等参数的
        # 写操作，检测器会把它当写调用统计，但重复调用本身不是被测缺陷。
    ),
    "handoff": ToolSpec(
        name="handoff",
        description="把任务转交给另一个环节（支付/物流/售后）。有副作用：每转交一次就产生一条交接流水。",
        parameters={
            "type": "object",
            "properties": {
                "target_role": {"type": "string", "description": "接收环节，如 fulfillment/refund/logistics"},
                "task": {"type": "string", "description": "转交的任务描述"},
            },
            "required": ["target_role", "task"],
        },
        is_write=True,
        path="/tools/handoff",
        # 交接没有幂等键：它记录的是一次"意图传递"，真实系统里同一条任务转交两次
        # 会真的产生两条工单——这正是要测的缺陷（重复转交）。
    ),
    "check_stock": ToolSpec(
        name="check_stock",
        description="查询商品库存。只读，无副作用。",
        parameters={
            "type": "object",
            "properties": {"item": {"type": "string", "description": "商品名"}},
            "required": ["item"],
        },
        is_write=False,
        path="/tools/check_stock",
    ),
    "ship_order": ToolSpec(
        name="ship_order",
        description="对已支付订单发货。有副作用：每成功一次就产生一条发货流水。",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "订单号"},
                "carrier": {"type": "string", "description": "承运商"},
                "tracking_no": {"type": "string", "description": "运单号"},
                "idempotency_key": {
                    "type": "string",
                    "description": "幂等键。同一订单重复发货时应复用同一个键。",
                },
            },
            "required": ["order_id"],
        },
        is_write=True,
        path="/tools/ship_order",
        idempotency_param="idempotency_key",
        # 与 create_payment 同一纪律：默认不实现幂等，重复发货是能构造出来的缺陷。
        enforces_idempotency=False,
    ),
    "send_notice": ToolSpec(
        name="send_notice",
        description="向用户发送订单通知。有副作用：每成功一次就产生一条通知流水。",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "订单号"},
                "notice_type": {"type": "string", "description": "通知类型，如 shipment/refund"},
                "message": {"type": "string", "description": "通知正文"},
                "idempotency_key": {
                    "type": "string",
                    "description": "幂等键。同一通知重复发送时应复用同一个键。",
                },
            },
            "required": ["order_id", "notice_type"],
        },
        is_write=True,
        path="/tools/send_notice",
        idempotency_param="idempotency_key",
        enforces_idempotency=False,
    ),
    "create_refund": ToolSpec(
        name="create_refund",
        description="对已支付订单发起退款。有副作用：每成功一次就产生一笔退款流水。",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "订单号"},
                "amount": {"type": "number", "description": "退款金额"},
                "reason": {"type": "string", "description": "退款原因"},
                "idempotency_key": {
                    "type": "string",
                    "description": "幂等键。同一订单重复退款时应复用同一个键。",
                },
            },
            "required": ["order_id", "amount"],
        },
        is_write=True,
        path="/tools/create_refund",
        idempotency_param="idempotency_key",
        enforces_idempotency=False,
    ),
}


def declared_writes() -> set[str]:
    """给 RunContext 用：所有声明为写的工具名。"""
    return {name for name, spec in TOOL_SPECS.items() if spec.is_write}


def idempotency_params() -> dict[str, str]:
    """给 RunContext 用：工具名 → 幂等键参数名。"""
    return {
        name: spec.idempotency_param
        for name, spec in TOOL_SPECS.items()
        if spec.idempotency_param
    }


@dataclass
class MockState:
    """mock 目标系统的内存状态。

    刻意保留**每一次扣款尝试**的流水，而不只是最终结果——重复扣款的证据就在
    流水里。只存最终状态的话，"扣了两次但订单金额只更新了一次"这类缺陷看不见。
    """

    orders: dict[str, dict[str, Any]] = field(default_factory=dict)
    charges: list[dict[str, Any]] = field(default_factory=list)
    # 幂等键 → 第一次扣款的 id
    idempotency_index: dict[str, str] = field(default_factory=dict)
    # 发货 / 通知 / 退款流水。与扣款同一纪律：保留每一次尝试，不只看最终状态——
    # "重复发货""重复通知""重复退款"这些缺陷的证据都在流水里。
    shipments: list[dict[str, Any]] = field(default_factory=list)
    notices: list[dict[str, Any]] = field(default_factory=list)
    handoffs: list[dict[str, Any]] = field(default_factory=list)
    # 任务计划板：plan_id -> {plan_id, steps, done, created_at}。它是"世界"的
    # 一部分（工单板），不是 Agent 的记忆——多 Agent 接力时跨会话可见。
    plans: dict[str, dict[str, Any]] = field(default_factory=dict)
    refunds: list[dict[str, Any]] = field(default_factory=list)
    # 每种写工具的幂等索引各自独立：键是工具自己收的，跨工具共用一张表的话，
    # 两个工具收到同一个键会互相去重——那是把"工具 A 幂等、工具 B 幂等"
    # 混成"这笔业务幂等"，而真实系统每个端点各有一份去重表。
    shipment_idem: dict[str, str] = field(default_factory=dict)
    notice_idem: dict[str, str] = field(default_factory=dict)
    refund_idem: dict[str, str] = field(default_factory=dict)
    # 由场景打开：mock 侧是否按幂等键去重
    enforce_idempotency: bool = False
    # 订单上的自由文本备注：order_id → 那句话。由场景声明（见 Scenario.world）。
    #
    # **刻意不并进 `orders` 里。** 并进去的话，`reset()` 只重置 status/due_at，
    # 备注会留到下一次运行——上一条场景注入的那句话于是出现在下一条场景的世界里，
    # 而两条场景的报告都正常。单独一个字典 + 每次运行**整体赋值**（不是逐条合并），
    # 这条泄漏路径就不存在了。
    order_notes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def with_sample_orders(cls, **kwargs: Any) -> MockState:
        # 每局一份自己的订单字典：`reset()` 会就地改 status/due_at，共享同一份
        # 常量的话，两次运行之间连订单状态都是串的。
        return cls(orders={k: dict(v) for k, v in SAMPLE_ORDERS.items()}, **kwargs)

    def reset(self, *, now: datetime | None = None) -> None:
        """回到开局状态，并把**世界的时间锚点**重新钉在当前时刻。

        到期时间是相对于"世界起点"算出来的（`SAMPLE_DUE_OFFSETS`），起点默认取
        时钟的现在。写成相对量而不是固定日期，理由是一条测试工具最要紧的性质：
        同一个场景今天跑和下周跑必须给出同样的结论。固定日期会让它在某一天之后
        突然变成"所有订单都已逾期"，而那时的报告照样长得正常。
        """
        anchor = now if now is not None else clock.now()
        self.charges.clear()
        self.idempotency_index.clear()
        self.shipments.clear()
        self.notices.clear()
        self.refunds.clear()
        self.handoffs.clear()
        self.plans.clear()
        self.shipment_idem.clear()
        self.notice_idem.clear()
        self.refund_idem.clear()
        # 备注也回到空。它是**世界**的一部分，而 reset 的语义是"世界回到开局"。
        self.order_notes.clear()
        for order_id, order in self.orders.items():
            order["status"] = "unpaid"
            offset = SAMPLE_DUE_OFFSETS.get(order_id)
            # 没登记到期时间的订单不给 due_at，而不是给个默认值：默认值会让
            # "这张单不会逾期"和"我们忘了给它定到期时间"长得一模一样。
            order["due_at"] = (
                (anchor + timedelta(seconds=offset)).isoformat() if offset is not None else None
            )

    def view_of(self, order: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
        """订单的**对外视图**：算出来的字段现算，不落库。

        `overdue` 是派生的，不是存下来的。存下来的话它会在时间推进之后过期，
        于是目标系统自己都不知道这张单到底逾期没有——而这个项目最不能出的错
        就是"两个地方对同一件事说法不同"。现算保证世界按构造自洽。
        """
        moment = now if now is not None else clock.now()
        due_raw = order.get("due_at")
        due = datetime.fromisoformat(due_raw) if due_raw else None
        view = dict(order)
        view["overdue"] = bool(due is not None and moment >= due)
        # 把"你问这句话的时候世界是几点"一并回给调用方。没有它，Agent 与读报告的
        # 人都无法判断"这张单没逾期"是当时的事实，还是目标系统压根没在看时间。
        view["as_of"] = moment.isoformat()
        # 订单备注。**只在场景声明了的时候才出现这个字段**——空串也要挡掉：给每个
        # 订单都挂一个空备注，Agent 侧"有没有备注"就永远为真的，那条分支等于没测。
        note = self.order_notes.get(str(order.get("order_id")))
        if note:
            view["note"] = note
        return view
