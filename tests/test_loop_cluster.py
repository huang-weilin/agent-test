"""近似层的聚类：改的是**算几次**，不是**算出什么**。

`LoopDetector._cluster` 原先在双重循环里现算规范化参数——每比较一对就把两边各
`normalize_args` 一遍。量于 2026-10-01，n=320 时那是 51,040 次比较 × 2 =
**102,080 次**，而它只依赖单条调用，真正需要的只有 320 次
（`../_probes/profile_loop_detector.py`）。改法是把它提到循环外。

这类"优化"最危险的地方在于它看起来总是对的：结果错了也只是某两条调用没被并进
同一个簇，而**簇小一点不会报错**——它只会让一条本该报出来的循环变成 pass。
所以这里两道闸门：

  * 语义：拿一份**独立写出来的朴素实现**（就是改之前那段代码的写法：每比较一对
    现算一遍）逐条对齐。它不引用被测代码，只是把"贪心聚类"这件事按定义重写一遍。
  * 机制：数 `normalize_args` 被调了几次。这一条必须是**可数的**而不是**计时的**
    ——时间随机器负载飘，飘了的闸门迟早被整体忽略。

至于"整段遍历的簇有多长"这类结论本身，归 `tests/test_loop_paging.py` 和
`tests/corpus/` 管——那两处盯着的是**判定**，这里盯的是**判定没被改动**。
"""

from __future__ import annotations

from typing import Any

import pytest

from sandbox.detectors.base import CallView, ExemptionSet
from sandbox.detectors.loop import LoopDetector
from sandbox.trace.normalize import args_similarity, normalize_args


def _call(i: int, tool: str, args: dict[str, Any]) -> CallView:
    return CallView(
        id=f"c{i}",
        seq=i + 1,
        tool_name=tool,
        args=args,
        fingerprint=f"{tool}:{sorted(args.items())}",
        is_write=False,
        session_ordinal=0,
        memory_horizon_ordinal=0,
    )


def _naive_cluster(
    detector: LoopDetector, group: list[CallView]
) -> list[list[str]]:
    """改之前那段代码的写法，逐字照抄，只是把结果换成 id 列表。

    **刻意不调用被测的 `_cluster`**：一份对照实现如果复用了被测代码的任何一步，
    它对齐的就不是语义，而是"两边犯了同一个错"。
    """
    clusters: list[list[CallView]] = []
    for call in group:
        placed = False
        for cluster in clusters:
            rep = cluster[0]
            sim = args_similarity(
                normalize_args(rep.args),
                normalize_args(call.args),
                key_weights=detector.key_weights,
            )
            if sim.score >= detector.similarity_threshold:
                cluster.append(call)
                placed = True
                break
        if not placed:
            clusters.append([call])
    return [[c.id for c in cluster] for cluster in clusters]


#: 几组刻意挑过的参数形状：全相同 / 全不同 / 两簇 / 正好卡在阈值上 /
#: 键集合不一致 / 嵌套字典 / 三种值类型混着。
SHAPES: dict[str, list[dict[str, Any]]] = {
    "全相同": [{"order_id": "A1", "amount": 599.0}] * 6,
    "全不同": [{"order_id": f"A{i}"} for i in range(6)],
    "两簇": [{"order_id": "A1"}, {"order_id": "A1"}, {"order_id": "Z9"}, {"order_id": "Z9"}],
    "长字符串很接近": [{"note": "订单 A1 需要人工复核"} for _ in range(3)]
    + [{"note": "订单 A1 需要人工复检"}],
    "键集合不一致": [{"a": 1}, {"b": 1}, {"a": 1, "b": 1}, {}],
    "嵌套": [{"q": {"page": 1, "size": 10}}, {"q": {"page": 2, "size": 10}}],
    "数值型紧挨着": [{"page": 1000 + i} for i in range(5)],
    # **刻意构造的一串：代表一换，划分就变。** 1000 与 1200 相似（0.833 ≥ 0.8），
    # 1200 与 1500 也相似（恰好 0.8），而 1000 与 1500 不相似（0.667）。
    # 所以第三条调用并进哪个簇，取决于拿谁当代表——贪心聚类的定义是**第一个**
    # （理由写在 `_cluster` 里：避免漂移）。没有这一组的话，"代表是谁"这件事
    # 在对照里根本看不出来。
    "代表一换划分就变": [{"page": 1000}, {"page": 1200}, {"page": 1500}],
    "布尔与类型混合": [{"flag": True}, {"flag": False}, {"flag": 1}, {"flag": "true"}],
    "空参数": [{}, {}, {}],
}


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_the_cluster_matches_a_naive_reference(name: str) -> None:
    """优化之后，簇的划分必须和朴素实现逐条一致。"""
    group = [_call(i, "list_orders", args) for i, args in enumerate(SHAPES[name])]
    detector = LoopDetector()

    got = [[c.id for c in cluster] for cluster in detector._cluster(group)]

    assert got == _naive_cluster(detector, group), f"{name} 这一组的簇划分变了"


def test_declared_field_weights_take_the_same_path() -> None:
    """声明了字段权重时也一致。

    权重是场景声明进来的，走的正是 `_cluster` 里那条比较路径——不覆盖它的话，
    "改了算几次"这件事在**带权重的场景**里就没有对照（而那正是
    `loop_paging_*` 两个场景在用的形状）。
    """
    detector = LoopDetector(key_weights={"page": 5.0, "query": 1.0})
    group = [
        _call(0, "search", {"query": "订单", "page": 1}),
        _call(1, "search", {"query": "订单", "page": 2}),
        _call(2, "search", {"query": "发票", "page": 2}),
        _call(3, "search", {"query": "订单", "page": 3}),
    ]

    got = [[c.id for c in cluster] for cluster in detector._cluster(group)]

    assert got == _naive_cluster(detector, group)


def test_normalization_happens_once_per_call_not_once_per_pair() -> None:
    """规范化次数是 O(n)，不是 O(n²)。

    `_cluster` 只该对每条调用规范化一次。改之前是每比较一对规范化两边，也就是
    `2·C(n,2) = n(n−1)` 次——n=60 时 3,540 次。

    这里数的是**次数**，不是毫秒：时间随机器飘，飘了的闸门迟早没人看。上限给得
    很松（`<= 12n`），因为它要拦的是"回到 O(n²)"那一种退化，不是几个常数。
    """
    import sandbox.detectors.loop as loop_module

    n = 60
    # **这些参数必须两两不相似**（"A0" 与 "A1" 的编辑距离相似度是 0.5 < 0.8），
    # 于是每条调用自成一簇、每次都要比过所有已有的簇 —— 那正是改之前的 C(n,2)。
    #
    # 这一条一开始踩过坑：先用的输入是页码 1000 起、连续递增，而那种数字**全都**
    # 并进同一个簇，比较次数只有 n−1 次。改前的写法在这种输入下也只多算一点点，
    # 于是"把规范化放回循环里"这个探针**没能把它弄红**——一条测不出退化的闸门，
    # 看起来和一条好闸门一模一样。
    group = [_call(i, "list_orders", {"order_id": f"A{i}"}) for i in range(n)]

    counter = {"n": 0}
    real = loop_module.normalize_args

    def counting(args, *rest, **kwargs):  # type: ignore[no-untyped-def]
        counter["n"] += 1
        return real(args, *rest, **kwargs)

    loop_module.normalize_args = counting
    try:
        detector = LoopDetector()
        detector._cluster(group)
    finally:
        loop_module.normalize_args = real

    assert counter["n"] <= 12 * n, (
        f"n={n} 时规范化了 {counter['n']} 次（上限 {12 * n}）。"
        f"每对比较各算一遍的话是 {n * (n - 1)} 次——这条闸门拦的就是那个。"
    )


def test_the_counting_gate_is_not_vacuous() -> None:
    """上面那条数次数的话，前提是**输入真的会造成很多次比较**。

    簇数掉到 1 的时候，比较次数只有 n−1 次，而那时"把规范化放回循环里"这个
    退化根本测不出来（上面那条数法会安静地变成一句永远为真的断言）。所以这里
    把前提钉住：这一串参数必须两两不相似、真的分成 n 个簇。

    ——这一条是**量出来的教训**，不是想出来的：第一版数法用的输入是连续页码，
    全都并进了一个簇，探针打上去是绿的。
    """
    group = [_call(i, "list_orders", {"order_id": f"A{i}"}) for i in range(60)]
    detector = LoopDetector()

    clusters = detector._cluster(group)

    assert len(clusters) == 60, (
        f"这一串本该分成 60 个簇（每条一个），实际 {len(clusters)} 个——"
        f"那意味着上面那条数法测不出 O(n²) 的退化"
    )
    assert _naive_cluster(detector, group) == [[c.id for c in cl] for cl in clusters]


def test_a_clean_trace_still_gets_no_approximate_finding() -> None:
    """改完之后，一次正常的（不重复的）轨迹仍然什么都不报。

    上面几条比的是"簇划分一致"，这一条比的是**端到端没被改动**：一份参数两两
    不同的轨迹走到 `run()` 里，产出的结论和改之前一模一样。
    """
    group = [_call(i, "get_order", {"order_id": f"A{i}"}) for i in range(10)]

    findings = LoopDetector().run(group, ExemptionSet())

    assert sorted(f.detector for f in findings) == ["loop", "loop.cross_session"]
    covering = [f for f in findings if f.detector == "loop"]
    assert covering[0].verdict.value == "pass"
    assert "未发现严格重复" in covering[0].summary
    # 一条辅助调用都没有的多会话轨迹，跨会话那一层照旧说不出结论。
    assert [f.verdict.value for f in findings if f.detector == "loop.cross_session"] == [
        "inconclusive"
    ]
