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
    assert "出口代理拒绝" in findings[0].summary


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


def test_blocked_calls_retry_does_not_become_an_agent_loop(make_call) -> None:
    """沙箱拦了它、再报它循环，等于自己制造误报（ADR 003 场景）。"""
    calls = [
        make_call(1, "create_payment", {"order_id": "A1"}, blocked_by_proxy=True, status="blocked"),
        make_call(2, "create_payment", {"order_id": "A1"}, status="ok"),
    ]
    findings = _detector().run(calls, ExemptionSet())
    assert not _of(findings, ".strict"), "被拦截的那次被算进了循环计数"
