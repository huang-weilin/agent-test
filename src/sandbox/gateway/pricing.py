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
    """按模型定价计算成本。未定价模型返回 0.0 且 priced=False。

    **`input_tokens` 是总 prompt，`cache_read_tokens` 是它的一个子集。**
    这句是算对账的前提，而且是量出来的，不是照着某个 SDK 的字段名猜的：
    对 DeepSeek 实测（2026-10-01，`prompt_cache_hit_tokens` 那天真的命中了
    1408 个）——`prompt_tokens` = `hit` + `miss`，命中那部分**已经包含在**
    总 prompt 里。所以下面的输入价只对**没命中的那部分**计费：

        prompt_tokens = 1645 = hit 1408 + miss 237
        真实计费     = 237 × 输入价 + 1408 × 缓存价
        修之前算的是 = 1645 × 输入价 + 1408 × 缓存价   ← 命中那 1408 被算了两次

    按表里的 deepseek-chat 单价（0.27 / 0.07），那一次调用被算成
    $0.000543 而实际是 $0.000163——**3.3 倍**。缓存命中率越高，这个偏差越大：
    全命中的调用会变成 4 倍。（这条路径此前一条断言都没有：`cache_read_tokens`
    全仓只有写入方，没有任何读它的测试。）

    **口径钉死在 DeepSeek 上，换上游必须重新核。** 写这三列的地方是
    `gateway/stream.py` 的 `settle_and_record`（`prompt_cache_hit_tokens` /
    `prompt_cache_miss_tokens`）——它记的是"未命中"，不是"写入缓存"。
    Anthropic 那种把缓存读写**另算**在 input 之外的口径，不能直接套这个减法。
    """
    price = _PRICES.get(model)
    if price is None:
        return CostBreakdown(
            usd=0.0,
            priced=False,
            reason=f"模型 {model!r} 不在定价表中，成本未计入",
        )

    # 夹在 0：上游若给出 read > input 的怪数据（或将来换口径换错），
    # 宁可让这一项归零，也不要算出一笔负数金额——负的成本会一路穿过聚合，
    # 变成"这轮省钱了"。
    uncached_input = max(0, input_tokens - cache_read_tokens)
    usd = (
        uncached_input * price.input_per_mtok
        + output_tokens * price.output_per_mtok
        + cache_read_tokens * price.cache_read_per_mtok
        + cache_write_tokens * price.cache_write_per_mtok
    ) / 1_000_000

    reason = "" if price.verified_on else "定价未经核实，仅供相对比较"
    return CostBreakdown(usd=usd, priced=price.verified_on is not None, reason=reason)


def known_models() -> list[str]:
    return sorted(_PRICES)
