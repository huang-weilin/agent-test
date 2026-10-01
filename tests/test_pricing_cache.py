"""缓存命中的 token 只许收一次钱。

**这是一条算错了也没人看得出来的账。** `compute_cost` 的四个入参里，
`input_tokens` 是**总 prompt**，`cache_read_tokens` 是它的**子集**——所以
把两者分别乘上各自的单价再加起来，命中那部分就被收了两次。

这不是从字段名推的：2026-10-01 在真上游上量到过命中
（`prompt_cache_hit_tokens = 1408`），而同一份 usage 里
`prompt_tokens = 1645 = 1408 + 237`。写这三列的地方是
`gateway/stream.py`——它把 `prompt_cache_miss_tokens` 记成了 `cache_write_tokens`。

修之前的偏差：那一次调用被算成 $0.000543，实际 $0.000163，**3.3 倍**；
命中率越高偏得越多，全命中时是 4 倍。而这条路径此前**一条断言都没有**
（`cache_read_tokens` 全仓只有写入方，没有任何测试读过它）。

下面第一条测试钉住"命中确实少收"，第二条用实测那份 usage 逐项对账。
"""

from __future__ import annotations

import pytest

from sandbox.gateway.pricing import compute_cost

#: 真上游（DeepSeek，2026-10-01）那次命中缓存的调用，字段原样抄下来。
LIVE_USAGE = {
    "prompt_tokens": 1645,
    "completion_tokens": 1,
    "prompt_cache_hit_tokens": 1408,
    "prompt_cache_miss_tokens": 237,
}


def test_命中缓存的调用比不命中便宜() -> None:
    """用户可见的那条性质：缓存命中率越高，同一次调用越便宜。

    修之前这条是**反的**——命中越多收得越贵（命中部分既按输入价收、
    又按缓存价再收一遍），而"缓存省不省钱"是个谁都会问的问题。
    """
    tokens = {"model": "deepseek-chat", "input_tokens": 10_000, "output_tokens": 0}
    uncached = compute_cost(**tokens, cache_read_tokens=0)
    half = compute_cost(**tokens, cache_read_tokens=5_000)
    all_cached = compute_cost(**tokens, cache_read_tokens=10_000)

    assert half.usd < uncached.usd, "命中一半竟然没有更便宜"
    assert all_cached.usd < half.usd, "全命中竟然没有更便宜"
    # 全命中时，剩下的只有缓存价那一项（0.07），不是输入价（0.27）。
    assert all_cached.usd == pytest.approx(10_000 * 0.07 / 1_000_000)


def test_实测那份_usage_逐项对账() -> None:
    """拿真上游那一次的字段算一遍，结果必须等于上游的计费口径：

        未命中 237 × 输入价 + 命中 1408 × 缓存价
    """
    cost = compute_cost(
        model="deepseek-chat",
        input_tokens=LIVE_USAGE["prompt_tokens"],
        output_tokens=LIVE_USAGE["completion_tokens"],
        cache_read_tokens=LIVE_USAGE["prompt_cache_hit_tokens"],
        cache_write_tokens=LIVE_USAGE["prompt_cache_miss_tokens"],
    )
    expected = (237 * 0.27 + 1408 * 0.07 + 1 * 1.10) / 1_000_000
    assert cost.usd == pytest.approx(expected)
    # 修之前那个值：命中的 1408 被多收了一遍输入价。
    assert cost.usd < (1645 * 0.27 + 1408 * 0.07 + 1 * 1.10) / 1_000_000

    # 一个防"两边都归零"的对照：这次调用不是 0 元，也不是负数。
    assert cost.usd > 0


def test_上游给怪数据时不出负数() -> None:
    """`read > input` 只可能来自口径变了或上游坏了。

    减法若不做夹取，这一项会算出一笔负数金额，而**负的成本会一路穿过聚合**：
    总额被拉低、按工具拆解时那一行变成"倒赚"，读报告的人只会以为工具很便宜。
    """
    cost = compute_cost(
        model="deepseek-chat",
        input_tokens=100,
        output_tokens=0,
        cache_read_tokens=1_000,
    )
    assert cost.usd >= 0
    assert cost.usd == pytest.approx(1_000 * 0.07 / 1_000_000)
