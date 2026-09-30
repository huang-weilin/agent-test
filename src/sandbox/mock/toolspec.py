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
        description="对指定订单发起支付。**有副作用**：每成功一次就产生一笔扣款。",
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
    # 由场景打开：mock 侧是否按幂等键去重
    enforce_idempotency: bool = False

    @classmethod
    def with_sample_orders(cls, **kwargs: Any) -> MockState:
        return cls(
            orders={
                "A1": {"order_id": "A1", "item": "机械键盘", "amount": 599.0, "status": "unpaid"},
                "B2": {"order_id": "B2", "item": "显示器支架", "amount": 249.0, "status": "unpaid"},
            },
            **kwargs,
        )

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
        return view
