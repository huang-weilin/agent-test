"""循环检测器。对应验收标准之三：构造一次循环，必须报出来。

同时压误报——文档点名的四种"必须区分的情况"都在这里。
"""

from __future__ import annotations

from sandbox.db.models import DetectorVerdict
from sandbox.detectors.base import Exemption, ExemptionSet
from sandbox.detectors.loop import LoopDetector

# 场景声明"页码不是重要字段"。没有它，1→2 的相似度只有 0.75，聚不到一簇，
# 参数微调的循环就检测不出来。这正是 key_weights 存在的理由。
PAGE_WEIGHTS = {"page": 0.1}


def _detector(**kwargs) -> LoopDetector:
    return LoopDetector(key_weights=PAGE_WEIGHTS, **kwargs)


def _verdicts(findings) -> list[str]:
    return [f.verdict.value for f in findings]


def _of(findings, suffix: str):
    return [f for f in findings if f.detector.endswith(suffix)]


def test_strict_repeat_is_a_fail(make_call) -> None:
    """构造的循环：同一个调用原地重复。"""
    calls = [make_call(i, "create_payment", {"order_id": "A1", "amount": 599}) for i in range(1, 4)]
    findings = _detector().run(calls, ExemptionSet())

    strict = _of(findings, ".strict")
    assert len(strict) == 1
    assert strict[0].verdict == DetectorVerdict.FAIL
    assert strict[0].detail["count"] == 3


def test_two_repeats_is_not_yet_a_loop(make_call) -> None:
    """两次相同可能是正常重试或用户点了两下，不该报循环。"""
    calls = [make_call(i, "get_order", {"order_id": "A1"}) for i in range(1, 3)]
    findings = _detector().run(calls, ExemptionSet())
    assert not _of(findings, ".strict")


def test_bounded_retry_after_errors_is_not_a_loop(make_call) -> None:
    """正常重试：错误后重试，有限次 → 不算循环，但要单独统计。"""
    calls = [
        make_call(1, "create_payment", {"order_id": "A1"}, status="error"),
        make_call(2, "create_payment", {"order_id": "A1"}, status="error"),
        make_call(3, "create_payment", {"order_id": "A1"}, status="error"),
    ]
    findings = _detector().run(calls, ExemptionSet())

    assert not _of(findings, ".strict"), "正常重试被报成了循环"
    retry = _of(findings, ".retry")
    assert len(retry) == 1
    assert retry[0].verdict == DetectorVerdict.PASS
    assert retry[0].detail["retries"] == 2


def test_retries_beyond_the_bound_are_a_loop(make_call) -> None:
    """超过上界就不是重试，是卡住了。"""
    calls = [
        make_call(i, "create_payment", {"order_id": "A1"}, status="error") for i in range(1, 7)
    ]
    findings = _detector().run(calls, ExemptionSet())

    strict = _of(findings, ".strict")
    assert len(strict) == 1
    assert strict[0].verdict == DetectorVerdict.FAIL


def test_monotonic_pagination_is_not_a_loop(make_call) -> None:
    """正常分页：页码递增。"""
    calls = [
        make_call(i, "search", {"query": "键盘", "page": i}) for i in range(1, 4)
    ]
    findings = _detector().run(calls, ExemptionSet())

    approx = _of(findings, ".approximate")
    assert len(approx) == 1
    assert approx[0].verdict == DetectorVerdict.PASS, "正常翻页被报成了循环"
    assert approx[0].detail["values_monotonic"] is True


def test_page_churn_is_a_loop(make_call) -> None:
    """**关键用例：翻回头的页码。**

    1→2→3→1→2→3 在严格层看不出来（每次参数都不同），在近似层的差异只有 page，
    但它不单调——这就是"参数微调的循环"，也就是最容易被漏掉的那一类。
    """
    calls = [
        make_call(i, "search", {"query": "键盘", "page": p})
        for i, p in enumerate([1, 2, 3, 1, 2, 3], start=1)
    ]
    findings = _detector().run(calls, ExemptionSet())

    approx = _of(findings, ".approximate")
    assert len(approx) == 1
    assert approx[0].verdict == DetectorVerdict.FAIL
    assert approx[0].detail["values_monotonic"] is False


def test_similar_ids_in_a_batch_are_not_a_loop(make_call) -> None:
    """**负样本：一批不同的订单号。** 字符串字段没有序，谈不上"回退"。

    "在原地打转"说的是"它走回去了"，那要证据；字符串字段给不出这个证据，
    观测支持的只是"这些参数不一样"。修之前这一串报的是 fail
    「以相似参数重复调用 6 次…是在原地打转」——一条会被拿去查 Agent 行为缺陷的
    结论，而那里根本没有缺陷。

    **刻意用没有字段权重的检测器**：误报发生在默认路径上，而大多数用户不会去写
    那张权重表。形状也不许凑——近似的字符串才聚得成簇，参数各不相似时这一串
    压根走不到判定那一步，那样测的就成了空气（见 `tests/corpus/loop.yaml` 里
    那条样本的 `why`）。
    """
    calls = [
        make_call(i, "get_order", {"order_id": f"ORD-2024-{i:04d}"}) for i in range(1, 7)
    ]
    findings = LoopDetector().run(calls, ExemptionSet())

    assert not _of(findings, ".approximate"), (
        "一批不同的订单号被报成了循环："
        + "；".join(f.summary for f in _of(findings, ".approximate"))
    )


def test_string_cursor_paging_is_not_a_loop(make_call) -> None:
    """字符串游标翻页同样是正常遍历，不是循环。

    和上一条同源，但形状不同：上一条是**读不同的东西**，这一条是**读同一个
    结果集的下一页**。两条都落在"变化的字段是字符串"上。
    """
    calls = [
        make_call(i, "list_orders", {"status": "unpaid", "cursor": f"tok{i:04d}"})
        for i in range(6)
    ]
    findings = LoopDetector().run(calls, ExemptionSet())

    assert not _of(findings, ".approximate"), (
        "字符串游标翻页被报成了循环："
        + "；".join(f.summary for f in _of(findings, ".approximate"))
    )


def test_a_numeric_loop_still_reports_after_the_string_fix(make_call) -> None:
    """**保住检出。** 上面两条改的是"判不了"，不是"别报了"。

    少了这一条，"把近似层改成永远沉默"也能让那两条全绿——那是一个更坏的世界
    （真正的循环一条都报不出来），而它只看负样本是看不出来的。判据是数值字段
    有明确的回退（3 → 2）。

    形状和 `tests/corpus/loop.yaml` 的 `loop-approximate-non-monotonic` 一致
    （五个字段、四个常量）：这就是**下面那条"形状依赖"缺陷**——字段少了这个簇
    根本聚不起来，同一个循环会被判成 pass。那件事没在这一片里修，见
    `docs/detectors.md` 第三节末尾。
    """
    calls = [
        make_call(
            i,
            "list_orders",
            {"query": "book", "status": "paid", "sort": "asc", "region": "cn", "page": p},
        )
        for i, p in enumerate([1, 3, 2], start=1)
    ]
    findings = LoopDetector().run(calls, ExemptionSet())

    approx = _of(findings, ".approximate")
    assert [f.verdict for f in approx] == [DetectorVerdict.FAIL], (
        "数值字段明确回退的循环没报出来：" + str([f.summary for f in approx])
    )
    assert approx[0].detail["values_monotonic"] is False


def test_mixed_fields_where_one_has_no_order_are_not_judged(make_call) -> None:
    """一个数值字段 + 一个字符串字段：判不了，所以不报。

    判据是**只要有一个变化字段说不出序，就不能声称"它在打转"**——那句话是整个
    簇的结论，不是逐字段的。放开这一条的话，任何人往参数里加一个字符串字段
    （备注、租户号）就能把误报引回来。

    形状是量出来挑的：常量字段少于三个时这串聚不成簇，那样这条样本什么都没量到
    （探针打上去是绿的——第一版就是那样，改成这个形状才咬得住）。
    """
    calls = [
        make_call(
            i,
            "list_orders",
            {"query": "book", "status": "paid", "sort": "asc", "page": p, "note": n},
        )
        for i, (p, n) in enumerate([(1, "ready"), (3, "ready!"), (2, "ready.")], start=1)
    ]
    findings = LoopDetector().run(calls, ExemptionSet())

    assert not _of(findings, ".approximate")


def test_pagination_exemption_does_not_silence_a_strict_loop(make_call) -> None:
    """翻页豁免只能放过"页码在变"的情况。

    参数完全相同的严格重复不能被它静音——那恰恰是确定性最高的循环。
    """
    calls = [
        make_call(i, "search", {"query": "键盘", "page": 1}) for i in range(1, 4)
    ]
    exemptions = ExemptionSet(
        [
            Exemption(
                detector="loop.strict",
                tool="search",
                reason="翻页拉取是正常行为",
                match={"arg_differs_on": ["page"]},
            )
        ]
    )
    findings = _detector().run(calls, exemptions)

    strict = _of(findings, ".strict")
    assert strict[0].verdict == DetectorVerdict.FAIL, "严格循环被翻页豁免静音了"


def test_pagination_exemption_silences_when_only_the_page_differs(make_call) -> None:
    """页码在变且不单调、但场景显式声明豁免时，静音并留痕。"""
    calls = [
        make_call(i, "search", {"query": "键盘", "page": p})
        for i, p in enumerate([1, 2, 3, 1], start=1)
    ]
    exemptions = ExemptionSet(
        [
            Exemption(
                detector="loop.approximate",
                tool="search",
                reason="本场景刻意来回翻页，是预期行为",
                match={"arg_differs_on": ["page"]},
            )
        ]
    )
    findings = _detector().run(calls, exemptions)

    approx = _of(findings, ".approximate")
    assert approx[0].verdict == DetectorVerdict.PASS
    assert approx[0].exempted_by is not None
    assert approx[0].detail["would_have_been"] == "fail"


def test_long_monotonic_pagination_warns_instead_of_passing(make_call) -> None:
    """页码一直在涨但涨了十几页还没停 —— 分不清是遍历还是停不下来，报 warning。

    方向是刻意的：报错只是让人多看一眼，漏报是账单不封顶。
    """
    calls = [make_call(i, "search", {"query": "键盘", "page": i}) for i in range(1, 13)]
    findings = _detector().run(calls, ExemptionSet())

    approx = _of(findings, ".approximate")
    assert approx[0].verdict == DetectorVerdict.WARNING
    assert approx[0].detail["reason_code"] == "monotonic_unbounded"


def test_a_bare_cursor_reaches_the_warning_without_any_weight_table(make_call) -> None:
    """**唯一的字段就是光标**时，那条 warning 也要按声明的阈值出来。

    这条钉的是 2026-10-02 修的那个缺陷。原先判 warning 用的是**簇长**，而簇是拿
    "第一个成员"当代表比的加权平均相似度——代表不动，一串单调推进的调用涨到
    一定程度就自己断了。量出来的后果（`../_probes/measure_warn_threshold.py`）：
    声明"8 页"的那句阈值，实际要 `{page}` 36 页、`{status,page}` 19 页、
    四字段形状 13 页才报。**中间那些次报的是 `pass`「判为正常遍历」——一句做过
    的判定**，而错的方向正是代价不对称的那一头（停不下来却没告警）。

    `{page}` 这一格还有第二层：`field_weights: {page: 0.1}` **救不了它**。
    权重是相对的（`score = total/total_w`），只给一个字段声明等于没声明。
    所以最朴素的那种分页工具——参数里只有页码——用现成的逃生口也回不到 8 页。
    """
    calls = [make_call(i, "list_orders", {"page": i}) for i in range(1, 13)]
    # 刻意**不给权重表**：这一格要证明的正是"没有它也对"。
    findings = LoopDetector(key_weights={}).run(calls, ExemptionSet())

    approx = _of(findings, ".approximate")
    assert len(approx) == 1, "整组判：不该再切成好几条各自说一小段的结论"
    assert approx[0].verdict == DetectorVerdict.WARNING
    assert approx[0].detail["reason_code"] == "monotonic_unbounded"
    assert approx[0].detail["count"] == 12, "次数是整串的长度，不是某个簇的长度"
    assert approx[0].detail["differing_fields"] == ["page"]


def test_the_bare_cursor_path_sits_exactly_on_the_declared_threshold(make_call) -> None:
    """那条路也得**停在阈值上**，不是"看见一串翻页就报"。

    少了这条，一个对任何长度都报 warning 的实现也能让上面那条全绿——而那种检测器
    在真实使用里的下场是被关掉。
    """
    threshold = LoopDetector.monotonic_warn_after

    below = LoopDetector(key_weights={}).run(
        [make_call(i, "list_orders", {"page": i}) for i in range(1, threshold)], ExemptionSet()
    )
    assert all(
        f.verdict != DetectorVerdict.WARNING for f in _of(below, ".approximate")
    ), f"{threshold - 1} 次就报了 warning——阈值不在它声明的位置上"

    at = LoopDetector(key_weights={}).run(
        [make_call(i, "list_orders", {"page": i}) for i in range(1, threshold + 1)],
        ExemptionSet(),
    )
    approx = _of(at, ".approximate")
    assert [f.verdict for f in approx] == [DetectorVerdict.WARNING]
    assert approx[0].detail["count"] == threshold


def test_the_bare_cursor_path_stays_shut_on_the_shapes_it_cannot_read(make_call) -> None:
    """四条判据各自的**负样本**——少一条就是"整组判"变成一台误报机器。

    `_advancing_field` 认的是"键集合一致 + 恰好一个字段在变 + 它是数值 + 它非降"，
    这里把后三条各撞一次。字符串游标那一格尤其要紧：它在 `b0abfee` 里被定成
    "判不了"，沉默是对的（字符串上没有序），报 warning 就是凭空造一条观测。
    """
    n = LoopDetector.monotonic_warn_after + 4

    def warns(calls) -> bool:
        findings = LoopDetector(key_weights={}).run(calls, ExemptionSet())
        return any(
            f.verdict == DetectorVerdict.WARNING for f in _of(findings, ".approximate")
        )

    assert not warns(
        [make_call(i, "list_orders", {"cursor": f"tok{i:03d}"}) for i in range(1, n)]
    ), "字符串游标没有序——那是'判不了'那一档，不该报 warning"

    assert not warns(
        [make_call(i, "search", {"page": i, "size": 10 + i}) for i in range(1, n)]
    ), "两个字段一起变：认不出哪个是光标，这条路不表态"

    assert not warns(
        [make_call(i, "quote", {"amount": 30 - i}) for i in range(1, n)]
    ), "递减不是'往前推进'，那是 `_monotonic_state` 报 fail 的那一支"

    changing_keys = [make_call(i, "list_orders", {"page": i}) for i in range(1, n)]
    changing_keys[-1] = make_call(n, "list_orders", {"page": n, "filter": "x"})
    assert not warns(changing_keys), "中途多出一个筛选条件 = 换了次查询，不是同一串游标"


def test_an_exemption_actually_silences_the_unbounded_warning(make_call) -> None:
    """**给它写的豁免规则得真的静音得了它。**

    修之前不行：那条 warning 在 `exemptions.find(...)` 之前就返回了，于是给
    `loop.approximate` 写的规则只静音得了下面那条 pass——而 pass 本来就已经是
    "没告警"。量于 2026-10-02：写了规则和没写，40 页那串的 warning 一字不差地
    照出，`exempted_by` 是 `None`。**规则对这条 warning 完全无效**，而报告里
    "没有这条结论"和"这条结论被豁免了"长得一样——写规则的人以为自己处理过了。
    """
    calls = [make_call(i, "list_orders", {"page": i}) for i in range(1, 13)]
    silenced = LoopDetector(key_weights={}).run(
        calls,
        ExemptionSet(
            [
                Exemption(
                    detector="loop.approximate",
                    tool="list_orders",
                    reason="本场景就是要把整个目录翻完，已知没有终点",
                )
            ]
        ),
    )

    approx = _of(silenced, ".approximate")
    assert [f.verdict for f in approx] == [DetectorVerdict.PASS]
    assert approx[0].exempted_by == "exemption:loop.approximate/list_orders"
    # 记的是**这条结论本来是什么**。写死 "fail" 的话，一条被静音的 warning
    # 会说自己本来是个 fail——而这两档在这个项目里不是一回事
    # （warning 是"看一眼"，fail 是"这就是缺陷"）。
    assert approx[0].detail["would_have_been"] == "warning"


def test_a_narrow_exemption_that_does_not_match_lets_the_warning_through(make_call) -> None:
    """豁免的门是点名的：`arg_differs_on` 写别的字段就拦不住这一条。"""
    calls = [make_call(i, "list_orders", {"page": i, "status": "unpaid"}) for i in range(1, 13)]
    findings = LoopDetector(key_weights={}).run(
        calls,
        ExemptionSet(
            [
                Exemption(
                    detector="loop.approximate",
                    tool="list_orders",
                    reason="只想放行'筛选条件在变'的那一类",
                    match={"arg_differs_on": ["status"]},
                )
            ]
        ),
    )

    approx = _of(findings, ".approximate")
    assert [f.verdict for f in approx] == [DetectorVerdict.WARNING]
    assert approx[0].exempted_by is None


def test_ping_pong_is_a_fail(make_call) -> None:
    """委托乒乓：A→B→A→B 交替，两个工具互相触发。"""
    calls = [
        make_call(1, "get_order", {"order_id": "A1"}),
        make_call(2, "create_payment", {"order_id": "A1", "amount": 1}),
        make_call(3, "get_order", {"order_id": "A1"}),
        make_call(4, "create_payment", {"order_id": "A1", "amount": 2}),
        make_call(5, "get_order", {"order_id": "A1"}),
        make_call(6, "create_payment", {"order_id": "A1", "amount": 3}),
    ]
    findings = _detector().run(calls, ExemptionSet())

    pp = _of(findings, ".ping_pong")
    assert len(pp) == 1
    assert pp[0].verdict == DetectorVerdict.FAIL
    assert pp[0].detail["pair"] == ["get_order", "create_payment"]


def test_single_session_cross_session_check_is_inconclusive(make_call) -> None:
    """只有一个会话时，"跨会话循环"没测到就是没测到。"""
    calls = [make_call(1, "get_order", {"order_id": "A1"})]
    findings = _detector().run(calls, ExemptionSet())

    cs = _of(findings, ".cross_session")
    assert len(cs) == 1
    assert cs[0].verdict == DetectorVerdict.INCONCLUSIVE


def test_the_inconclusive_note_does_not_overclaim_how_many_sessions_ran(make_call) -> None:
    """检测器只看得见**留下了调用的**会话，措辞不能说得比观测更多。

    真实触发点是 cross_session_remembered：会话 #0 付了款，会话 #1 的脚本一字
    不差地又调了一次 create_payment，但记忆守卫在发起请求之前就把它拦下了——
    边界上于是只留得下会话 #0 的痕迹。那次运行确实涉及两个会话，检测器却看不见
    第二个，因为它的契约是"只吃 list[CallView]"。

    说成"本次只涉及 1 个会话"是假的，读报告的人会先怀疑场景配错了。判定本身
    不受影响（同一个 fingerprint 要出现在两个会话里才可比，只有一个会话有调用
    时本就无可比），要改的只是别让结论替观测说话。
    """
    calls = [make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0)]
    cs = _of(_detector().run(calls, ExemptionSet()), ".cross_session")

    assert cs[0].verdict == DetectorVerdict.INCONCLUSIVE
    assert "本次只涉及" not in cs[0].summary
    assert cs[0].detail["session_ordinals_with_calls"] == [0]


def test_cross_session_repeat_of_reads_is_flagged(make_call) -> None:
    """只读调用跨会话反复出现：报 warning，不报 fail。

    注意这个用例曾经用 `session_sequence_id="s1"/"s2"` 来表示"两个会话"——
    那是错的，同一个序列里的所有调用本来就共享一个 sequence id。区分会话的
    是**序号**。用 sequence id 去数会话会永远数出 1，于是永远 inconclusive。
    """
    calls = [
        make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0),
        make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=1, memory_horizon_ordinal=0),
    ]
    findings = _detector().run(calls, ExemptionSet())

    cs = _of(findings, ".cross_session")
    assert cs[0].verdict == DetectorVerdict.WARNING
    assert cs[0].detail["sessions"] == {"0": 1, "1": 1}


def test_cross_session_write_is_left_to_the_idempotency_detector(make_call) -> None:
    """写操作的跨会话重复不在这里报——否则同一个毛病会出现两条结论。"""
    calls = [
        make_call(1, "create_payment", {"order_id": "A1"}, is_write=True, session_ordinal=0),
        make_call(
            1,
            "create_payment",
            {"order_id": "A1"},
            is_write=True,
            session_ordinal=1,
            memory_horizon_ordinal=0,
        ),
    ]
    findings = _detector().run(calls, ExemptionSet())

    assert not _of(findings, ".cross_session") or all(
        f.verdict != DetectorVerdict.WARNING for f in _of(findings, ".cross_session")
    )


def test_cross_session_note_mentions_memory_loss_when_forgotten(make_call) -> None:
    """记忆没覆盖到首次调用时，结论里要提示"它可能根本不知道之前调过"。"""
    calls = [
        make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0),
        make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=2, memory_horizon_ordinal=-1),
    ]
    findings = _detector().run(calls, ExemptionSet())

    cs = _of(findings, ".cross_session")
    assert cs[0].verdict == DetectorVerdict.WARNING
    assert "记忆只覆盖到" in cs[0].summary


def test_all_blocked_calls_is_inconclusive(make_call) -> None:
    """全被出口代理拦下 = 一次都没到达目标系统。没测到不能读成通过。"""
    calls = [
        make_call(i, "create_payment", {"order_id": "A1"}, blocked_by_proxy=True, status="blocked")
        for i in range(1, 4)
    ]
    findings = _detector().run(calls, ExemptionSet())

    assert _verdicts(findings) == ["inconclusive"]
    assert "边界上被拒" in findings[0].summary
    # 被代理拦下的次数要单独说出来：全部被拒和大部分被拒是两件事，
    # 而"全是沙箱策略干的"意味着这份轨迹里 Agent 自己做的事一件都没留下。
    assert "3 次是出口代理按策略拦下的" in findings[0].summary
    assert findings[0].detail["blocked_by_proxy"] == 3


def test_a_rejected_call_is_inconclusive_even_without_the_proxy_flag(make_call) -> None:
    """判据是「到没到」，不是「谁拦的」——参数被 schema 拒掉的刻意不带那个标记。

    这一条和上面那条只差一点：三次调用同样一次都没发出去，但它们是 Agent 自己
    写错了参数，记录里 `blocked_by_proxy=False`。看标记的实现在这里会说
    「三次调用都在原地打转」，而那三次一次都没发生过。
    """
    calls = [
        make_call(i, "create_payment", {"order_id": "A1"}, status="invalid_args")
        for i in range(1, 4)
    ]
    findings = _detector().run(calls, ExemptionSet())

    assert _verdicts(findings) == ["inconclusive"]
    assert "边界上被拒" in findings[0].summary
    assert findings[0].detail["blocked_by_proxy"] == 0
    assert findings[0].detail["rejected_at_boundary"] == 3


def test_a_rejected_call_does_not_join_a_cross_session_group(make_call) -> None:
    """跨会话那一支也只吃到达了目标系统的调用。

    会话 #0 有一次真调用，会话 #1 那次在边界上被拒了——只有**一次**调用真的
    发生过，谈不上跨会话重复。原先跨会话那里连 `blocked_by_proxy` 都没过滤，
    传的是整份 `ordered`；同一个缺陷在三个地方各有一份。
    """
    calls = [
        make_call(1, "create_payment", {"order_id": "A1"}, session_ordinal=0),
        make_call(
            1,
            "create_payment",
            {"order_id": "A1"},
            session_ordinal=1,
            memory_horizon_ordinal=0,
            status="invalid_args",
        ),
    ]
    findings = _detector().run(calls, ExemptionSet())

    cross = _of(findings, ".cross_session")
    assert len(cross) == 1
    # 铁证在 detail 里：进分组的会话只有 #0，被拒的那次压根没进去。
    assert cross[0].detail["session_ordinals_with_calls"] == [0], (
        f"被拒的那次进了跨会话分组：{cross[0].detail}"
    )
    # 只有一个会话留下调用记录 -> 判不了。既不是"没发现跨会话循环"，也不是
    # "发现了"——「没测到」和「没问题」是两件事。
    assert cross[0].verdict == DetectorVerdict.INCONCLUSIVE
    assert "到达目标系统的会话只有 1 个" in cross[0].summary


def test_no_calls_is_inconclusive() -> None:
    findings = _detector().run([], ExemptionSet())
    assert findings[0].verdict == DetectorVerdict.INCONCLUSIVE


def test_snapshot_replays_are_reported_separately(make_call) -> None:
    """恢复导致的重放不是 Agent 的行为，但重放本身可能造成重复副作用。"""
    calls = [
        make_call(1, "create_payment", {"order_id": "A1"}, replay_reason="snapshot_restore"),
        make_call(2, "create_payment", {"order_id": "A1"}, replay_reason="snapshot_restore"),
        make_call(3, "create_payment", {"order_id": "A1"}, replay_reason="snapshot_restore"),
    ]
    findings = _detector().run(calls, ExemptionSet())

    assert not _of(findings, ".strict"), "恢复重放被当成了 Agent 的循环"
    replay = _of(findings, ".snapshot_replay")
    assert len(replay) == 1
    assert replay[0].verdict == DetectorVerdict.WARNING

    # 一次 Agent 自己发起的调用都没有 → 比对样本是 0，只能说"没测到"。
    # 报 PASS 会变成"已在 0 次工具调用中比对，未发现循环"——把没查说成查过。
    overall = [f for f in findings if f.detector == "loop"]
    assert len(overall) == 1
    assert overall[0].verdict == DetectorVerdict.INCONCLUSIVE
    assert overall[0].detail["snapshot_replays"] == 3


def test_blocked_calls_retry_does_not_become_an_agent_loop(make_call) -> None:
    """沙箱拦了它、再报它循环，等于自己制造误报（ADR 003 场景）。"""
    calls = [
        make_call(1, "create_payment", {"order_id": "A1"}, blocked_by_proxy=True, status="blocked"),
        make_call(2, "create_payment", {"order_id": "A1"}, status="ok"),
    ]
    findings = _detector().run(calls, ExemptionSet())
    assert not _of(findings, ".strict"), "被拦截的那次被算进了循环计数"
