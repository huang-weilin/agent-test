"""模型定价表。

**设计取向：宁可算不出价格，也不要猜。**

- 表里没有的模型 → 成本记 0.0 并标 `priced=False`，在报告里显式提示"未定价"。
  猜一个数字比留空更糟：它会污染成本对比，而用户不会发现。
- 价格是易变数据。每行带 `verified_on`，报告里超过 N 天要提示重新核对。

Token 数才是首要指标（网关权威计数），成本是从它派生出来的。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ModelPrice:
    """单位：美元 / 百万 Token。"""

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float = 0.0
    cache_write_per_mtok: float = 0.0
    verified_on: date | None = None


@dataclass(frozen=True)
class CostBreakdown:
    usd: float
    priced: bool
    reason: str = ""


# 注意：这些数值未经本项目核实，仅作为占位与结构示例。
# 接入真实计费前必须逐条核对官方定价页并更新 verified_on。
#
# 之所以仍然放具体数字而不是留空：walking skeleton 需要一个非零的成本读数来
# 验证成本聚合链路。但 `priced` 标志会一路带到报告里，避免被误当成账单。
_PRICES: dict[str, ModelPrice] = {
    "deepseek-chat": ModelPrice(
        input_per_mtok=0.27,
        output_per_mtok=1.10,
        cache_read_per_mtok=0.07,
        verified_on=None,  # 未核实 —— 报告中会标注
    ),
    "deepseek-reasoner": ModelPrice(
        input_per_mtok=0.55,
        output_per_mtok=2.19,
        cache_read_per_mtok=0.14,
        verified_on=None,
    ),
}


def compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> CostBreakdown:
    """按模型定价计算成本。未定价模型返回 0.0 且 priced=False。"""
    price = _PRICES.get(model)
    if price is None:
        return CostBreakdown(
            usd=0.0,
            priced=False,
            reason=f"模型 {model!r} 不在定价表中，成本未计入",
        )

    usd = (
        input_tokens * price.input_per_mtok
        + output_tokens * price.output_per_mtok
        + cache_read_tokens * price.cache_read_per_mtok
        + cache_write_tokens * price.cache_write_per_mtok
    ) / 1_000_000

    reason = "" if price.verified_on else "定价未经核实，仅供相对比较"
    return CostBreakdown(usd=usd, priced=price.verified_on is not None, reason=reason)


def known_models() -> list[str]:
    return sorted(_PRICES)
