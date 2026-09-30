"""工具声明：写/读属性、幂等键参数、参数 schema 的**单一来源**。

为什么要有这个单一来源：`is_write` 决定了幂等与重复扣款检测的触发范围，而这个
判断在三个地方要用——网关记录工具调用时、出口代理放行时、检测器分析时。三处各
自推断的话必然漂移，而漂移的表现是"某些写操作悄悄不设防"，从报告上看不出来。

所以：**场景必须显式声明写属性**，不依赖工具名启发式（normalize.py 里的启发式
只作为兜底存在）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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

    def reset(self) -> None:
        self.charges.clear()
        self.idempotency_index.clear()
        for order in self.orders.values():
            order["status"] = "unpaid"
