"""跨会话差异报告（需求 3.1 P1）。

这一节的难点不在"找不同"，在**怎么把不同说清楚**。三条各有一组测试：

  1. **不猜配对。** 只有两边各剩一次的同名调用才敢说"同一次调用、参数不同"。
     多剩一条就只能说"只在某一段出现"——硬配出来的"参数从 X 变成 Y"是编的，
     而读者没有办法从报告上看出来它是编的。
  2. **"有差异"和"差异算不算数"是两件事。** 前提没成立的会话照样把差异列出来，
     只是加一句"这可能是没跑到"。两个字段并行，不许互相吞掉。
  3. **这是陈述，不是判定。** 它不进 `findings`、不影响 `ok`。长时运行的场景里
     两段会话本来就该不一样；把"有差异"判成失败，等于把正常形态报成缺陷。

另外还有一条：**我们动过的东西必须单独列**（记忆回滚、世界时间、无人值守）。
它们是"我们让它不一样"，不是"它的行为变了"——这两件事在报告上长得一模一样。
"""

from __future__ import annotations

from typing import Any

import pytest

from sandbox.scenario.runner import ScenarioResult, render
from sandbox.scenario.session_diff import (
    KIND_ARGS_DIFFER,
    KIND_COUNT_DIFFERS,
    KIND_ONLY_BASE,
    KIND_ONLY_OTHER,
    KIND_OUTCOME_DIFFERS,
    MAX_ITEMS,
    SessionDiff,
    SessionPair,
    SessionSide,
    compare_sessions,
)

# --- 构造辅助 -----------------------------------------------------------------


def _side(ordinal: int, **kw: Any) -> SessionSide:
    fields: dict[str, Any] = {
        "name": f"第 {ordinal} 段",
        "memory_horizon_ordinal": ordinal - 1,
    }
    fields.update(kw)
    return SessionSide(ordinal=ordinal, **fields)


def _pair(diff: SessionDiff) -> SessionPair:
    assert diff.pairs, "这次对照没有产生任何一对会话"
    return diff.pairs[0]


def _only(diff: SessionDiff) -> list[Any]:
    return [[i.kind, i.tool] for i in _pair(diff).differences]


class _Outcome:
    """只给自己那几项，碰别的属性就炸。

    用来钉住 `SessionSide.of` 的契约：差异报告读了会话的哪些字段，是一份**写死的
    清单**。报告越界去读别的字段（比如拿 stdout 去猜它做了什么），在这里会当场
    炸掉——而不是悄悄多读一项、让报告说出一句观测不支持的话。
    """

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"差异报告读了会话的 {name}，它不在清单里")


_OUTCOME_FIELDS: dict[str, Any] = {
    "ordinal": 0,
    "name": "白天",
    "autonomous": False,
    "memory_horizon_ordinal": -1,
    "restored_from": None,
    "after_gap_s": 0.0,
    "virtual_started_at": None,
    "premise_held": True,
    "killed": False,
    "clock_matched": True,
}


# --- 1. 不猜配对 --------------------------------------------------------------


def test_one_session_alone_has_nothing_to_compare():
    """只有一段会话 → None。**这不是"没有差异"**，界面要能分清这两件事。"""
    assert compare_sessions([], [_side(0)]) is None


def test_identical_sessions_produce_no_items(make_call):
    calls = [
        make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0),
        make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=1),
    ]
    pair = _pair(compare_sessions(calls, [_side(0), _side(1)]))

    assert pair.differences == []
    assert pair.identical_calls == 1
    assert pair.comparable
    assert "1 条一模一样" in pair.headline


def test_a_call_only_one_session_made_is_reported_with_its_side(make_call):
    calls = [
        make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0),
        make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=1),
        make_call(2, "create_payment", {"order_id": "A1"}, session_ordinal=1, is_write=True),
    ]
    pair = _pair(compare_sessions(calls, [_side(0), _side(1)]))

    assert _only(compare_sessions(calls, [_side(0), _side(1)])) == [
        [KIND_ONLY_OTHER, "create_payment"]
    ]
    assert "只在会话 #1 出现" in pair.differences[0].sentence
    assert "#0 里没有" in pair.differences[0].sentence


def test_the_same_arguments_in_a_different_order_still_pair_up(make_call):
    """对齐靠的是指纹，不是顺序。顺序变了不算差异。"""
    calls = [
        make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0),
        make_call(2, "get_order", {"order_id": "A2"}, session_ordinal=0),
        make_call(1, "get_order", {"order_id": "A2"}, session_ordinal=1),
        make_call(2, "get_order", {"order_id": "A1"}, session_ordinal=1),
    ]
    pair = _pair(compare_sessions(calls, [_side(0), _side(1)]))

    assert pair.differences == []
    assert pair.identical_calls == 2


def test_two_leftovers_on_one_side_are_not_paired_up(make_call):
    """**本文件最重要的一条。**

    基准那侧有两次付款、对照那侧一次，参数各不相同——没有依据说哪一次"就是"
    另一次。硬配出来的句子会读成"参数从 99 变成了 88"，而它只是我们按顺序抓的
    两个。所以只能说"只在某一段出现"。
    """
    calls = [
        make_call(1, "create_payment", {"amount": 99}, session_ordinal=0, is_write=True),
        make_call(2, "create_payment", {"amount": 88}, session_ordinal=0, is_write=True),
        make_call(1, "create_payment", {"amount": 77}, session_ordinal=1, is_write=True),
    ]
    pair = _pair(compare_sessions(calls, [_side(0), _side(1)]))

    assert KIND_ARGS_DIFFER not in [i.kind for i in pair.differences]
    assert [i.kind for i in pair.differences] == [KIND_ONLY_BASE, KIND_ONLY_BASE, KIND_ONLY_OTHER]
    assert all("参数" in i.base_text or "参数" in i.other_text for i in pair.differences)


def test_one_leftover_on_each_side_is_paired_and_the_changed_field_is_named(make_call):
    calls = [
        make_call(1, "create_payment", {"order_id": "A1", "amount": 99}, session_ordinal=0),
        make_call(1, "create_payment", {"order_id": "A1", "amount": 88}, session_ordinal=1),
    ]
    item = _pair(compare_sessions(calls, [_side(0), _side(1)])).differences[0]

    assert item.kind == KIND_ARGS_DIFFER
    assert item.changes == ("amount：99 变 88",)
    assert "参数不同" in item.sentence and "amount：99 变 88" in item.sentence


def test_a_field_present_on_only_one_side_is_named_as_such(make_call):
    calls = [
        make_call(1, "get_order", {"order_id": "A1", "currency": "CNY"}, session_ordinal=0),
        make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=1),
    ]
    item = _pair(compare_sessions(calls, [_side(0), _side(1)])).differences[0]

    assert item.changes == ("currency 只有会话 #0 传了（\"CNY\"）",)


def test_leftovers_of_different_tools_are_never_paired(make_call):
    calls = [
        make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0),
        make_call(1, "create_payment", {"order_id": "A1"}, session_ordinal=1),
    ]
    kinds = [i.kind for i in _pair(compare_sessions(calls, [_side(0), _side(1)])).differences]

    assert kinds == [KIND_ONLY_BASE, KIND_ONLY_OTHER]


def test_a_repeated_call_is_a_count_difference_and_still_counts_as_identical(make_call):
    """同参数发了两次 vs 一次：既是"次数不一样"，也有一次是配上的。"""
    calls = [
        make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0),
        make_call(2, "get_order", {"order_id": "A1"}, session_ordinal=0),
        make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=1),
    ]
    pair = _pair(compare_sessions(calls, [_side(0), _side(1)]))

    assert [i.kind for i in pair.differences] == [KIND_COUNT_DIFFERS]
    assert pair.identical_calls == 1
    assert "2 次" in pair.differences[0].base_text and "1 次" in pair.differences[0].other_text


def test_the_same_call_blocked_on_one_side_only_is_still_a_difference(make_call):
    """**指纹一样、次数一样，也不等于一模一样。**

    同一个调用在一边被出口代理拦下、在另一边放行了——指纹里没有这一项，
    不比就没人会说，而报告上会写"1 条一模一样"。
    """
    calls = [
        make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0, blocked_by_proxy=True),
        make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=1),
    ]
    pair = _pair(compare_sessions(calls, [_side(0), _side(1)]))

    assert [i.kind for i in pair.differences] == [KIND_OUTCOME_DIFFERS]
    assert any("被出口代理拦下" in c for c in pair.differences[0].changes)
    assert "结果不同" in pair.differences[0].sentence


def test_a_status_difference_on_the_same_call_is_reported(make_call):
    calls = [
        make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0, status="error"),
        make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=1),
    ]
    pair = _pair(compare_sessions(calls, [_side(0), _side(1)]))

    assert any("status" in c for c in pair.differences[0].changes)


def test_identical_calls_with_identical_outcomes_stay_silent(make_call):
    calls = [
        make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0),
        make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=1),
    ]
    pair = _pair(compare_sessions(calls, [_side(0), _side(1)]))

    assert pair.differences == []
    assert pair.identical_calls == 1


def test_long_arguments_are_truncated_and_say_how_much_was_cut(make_call):
    calls = [
        make_call(
            1,
            "get_order",
            {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
            session_ordinal=0,
        ),
        make_call(1, "get_order", {"a": 1}, session_ordinal=1),
    ]
    item = _pair(compare_sessions(calls, [_side(0), _side(1)])).differences[0]

    # 印了 4 个字段，还剩 2 个没印——**必须说出来**，不然读者以为参数就这些。
    assert "还有 2 个字段" in item.base_text


# --- 2. 有差异 ≠ 差异算不算数 --------------------------------------------------


def test_a_session_that_did_not_finish_is_not_comparable_but_still_lists_differences(make_call):
    calls = [
        make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0),
        make_call(1, "get_order", {"order_id": "A2"}, session_ordinal=1),
    ]
    sides = [_side(0), _side(1, premise_held=False)]
    pair = _pair(compare_sessions(calls, sides))

    assert not pair.comparable
    assert pair.differences, "不可比不等于不列差异——两件事必须并行"
    assert any("脚本没跑完" in r for r in pair.reasons)
    assert "不能直接比" in pair.headline


def test_a_killed_session_counts_as_a_held_premise(make_call):
    """熔断是循环场景的**预期结果**，不是"没跑到"。"""
    calls = [make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0)]
    pair = _pair(compare_sessions(calls, [_side(0), _side(1, killed=True)]))

    assert pair.comparable


def test_a_mismatched_clock_blames_the_accelerator_not_the_agent(make_call):
    calls = [make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0)]
    pair = _pair(
        compare_sessions(calls, [_side(0), _side(1, premise_held=False, clock_matched=False)])
    )

    assert not pair.comparable
    assert any("加速器" in r for r in pair.reasons)
    assert not any("被测对象" in r for r in pair.reasons if "不是来自被测对象" not in r)


# --- 3. 我们动过的东西单独列 --------------------------------------------------


def test_a_different_memory_horizon_says_the_difference_may_come_from_there(make_call):
    calls = [make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0)]
    pair = _pair(
        compare_sessions(
            calls,
            [_side(0, memory_horizon_ordinal=-1), _side(1, memory_horizon_ordinal=0)],
        )
    )

    assert any("记忆视野不同" in b for b in pair.background)
    assert any("差异可能来自这里" in b for b in pair.background)


def test_an_unknown_horizon_says_so_instead_of_guessing(make_call):
    calls = [make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0)]
    pair = _pair(
        compare_sessions(
            calls,
            [_side(0, memory_horizon_ordinal=None), _side(1, memory_horizon_ordinal=0)],
        )
    )

    assert any("不知道" in b for b in pair.background)


def test_a_restore_from_a_snapshot_is_listed_as_our_own_rollback(make_call):
    calls = [make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0)]
    pair = _pair(compare_sessions(calls, [_side(0), _side(1, restored_from="before_pay")]))

    assert any("before_pay" in b and "回滚" in b for b in pair.background)


def test_a_world_gap_is_listed_with_its_length(make_call):
    calls = [make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0)]
    pair = _pair(
        compare_sessions(
            calls,
            [_side(0), _side(1, after_gap_s=108000.0, virtual_started_at="2026-09-30T14:00:00Z")],
        )
    )

    # 30 小时印成"1.2 天"：换算的是同一个 span_human，不许这里另写一份。
    assert any("1.2 天" in b for b in pair.background)
    assert any("2026-09-30T14:00:00Z" in b for b in pair.background)


def test_a_change_in_unattended_status_is_listed(make_call):
    calls = [make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0)]
    pair = _pair(compare_sessions(calls, [_side(0), _side(1, autonomous=True)]))

    assert any("无人值守" in b for b in pair.background)


def test_a_session_with_no_calls_at_all_is_called_out_in_the_background(make_call):
    calls = [make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0)]
    pair = _pair(compare_sessions(calls, [_side(0), _side(1)]))

    assert any("一个工具调用都没有" in b for b in pair.background)
    assert not pair.comparable or pair.differences


def test_two_empty_sessions_say_nothing_can_be_compared_at_all():
    diff = compare_sessions([], [_side(0), _side(1)])

    assert diff is not None
    assert diff.pairs == []
    assert any("都没有工具调用" in n for n in diff.notes)


def test_a_leading_empty_session_is_not_silently_dropped(make_call):
    """基准取的是第一段**有调用**的会话。前面那段空的必须留个说法。"""
    calls = [make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=1)]
    diff = compare_sessions(calls, [_side(0), _side(1)])

    assert any("会话 #0" in n and "没进下面的对照" in n for n in diff.notes)
    assert diff.pairs == []
    assert any("没有第二段可以对照" in n for n in diff.notes)


def test_only_one_session_having_calls_is_not_reported_as_no_difference(make_call):
    calls = [make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=1)]
    diff = compare_sessions(calls, [_side(0), _side(1)])

    assert diff.pairs == []
    assert any("列不出差异，不等于两段一样" in n for n in diff.notes)


def test_the_number_of_identical_calls_is_the_positive_evidence(make_call):
    """一份只列差异的报告，看不出"它其实把该做的都做了"。"""
    calls = [
        make_call(i, "get_order", {"order_id": f"A{i}"}, session_ordinal=0)
        for i in range(1, 4)
    ] + [
        make_call(i, "get_order", {"order_id": f"A{i}"}, session_ordinal=1)
        for i in range(1, 4)
    ]
    calls.append(make_call(9, "create_payment", {"order_id": "A9"}, session_ordinal=1))
    pair = _pair(compare_sessions(calls, [_side(0), _side(1)]))

    assert pair.identical_calls == 3
    assert len(pair.differences) == 1


# --- 4. 长报告不静默截断 ------------------------------------------------------


def test_too_many_differences_are_capped_and_the_rest_are_announced(make_call):
    calls = [
        make_call(i, "create_payment", {"order_id": f"A{i}"}, session_ordinal=0)
        for i in range(MAX_ITEMS + 5)
    ]
    calls += [make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=1)]
    pair = _pair(compare_sessions(calls, [_side(0), _side(1)]))

    # 17 条付款 + 1 条只在对照侧的查单 = 18 条差异，列到上限为止。
    assert len(pair.differences) == MAX_ITEMS
    assert pair.omitted == 6
    assert "另有 6 条未列出" in pair.headline


def test_nothing_is_omitted_when_everything_fits(make_call):
    calls = [make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0)]
    pair = _pair(compare_sessions(calls, [_side(0), _side(1)]))

    assert pair.omitted == 0
    assert "未列出" not in pair.headline


def test_the_report_records_the_limit_it_was_truncated_with(make_call):
    """上限随报告一起存。

    半年后打开这份旧报告，页面得说得出**它当时**被截到多少条。用当时那个数
    而不是今天的常量：否则上限一改，旧报告的这句话就在说一件没发生过的事。
    """
    calls = [make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0)]
    diff = compare_sessions(calls, [_side(0), _side(1)], max_items=3)

    assert diff.to_dict()["limit"] == 3
    assert compare_sessions(calls, [_side(0), _side(1)]).to_dict()["limit"] == MAX_ITEMS


# --- 5. 读会话字段的清单是写死的 ----------------------------------------------


def test_session_side_reads_only_the_fields_it_declares():
    outcome = _Outcome(**_OUTCOME_FIELDS)
    side = SessionSide.of(outcome)

    assert side.ordinal == 0
    assert side.name == "白天"
    assert side.memory_horizon_ordinal == -1


def test_session_side_explodes_if_it_is_missing_a_field():
    """清单里少一项就该炸，而不是悄悄给个默认值——那会让"不知道"读成"没有"。"""
    fields = dict(_OUTCOME_FIELDS)
    del fields["memory_horizon_ordinal"]

    with pytest.raises(AssertionError):
        SessionSide.of(_Outcome(**fields))


# --- 6. 陈述，不是判定 --------------------------------------------------------


def _result_with_diff(diff_value: Any) -> ScenarioResult:
    result = ScenarioResult(scenario_id="s", run_id="r")
    result.session_diff = diff_value
    return result


def test_the_diff_never_enters_the_findings_or_the_verdict(make_call):
    """`ok` 只看 findings / 前提 / 地面真值，差异报告**一个字都不许影响**它们。"""
    calls = [
        make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0),
        make_call(1, "create_payment", {"order_id": "A9"}, session_ordinal=1),
    ]
    diff = compare_sessions(calls, [_side(0), _side(1)])
    assert diff is not None and diff.pairs[0].differences

    result = _result_with_diff(diff)

    assert result.findings == []
    assert result.ok is False  # 没有 expectation，本来就不通过
    assert result.to_dict()["findings"] == []


def test_the_report_carries_the_diff_as_a_separate_field(make_call):
    calls = [
        make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0),
        make_call(1, "get_order", {"order_id": "A2"}, session_ordinal=1),
    ]
    result = _result_with_diff(compare_sessions(calls, [_side(0), _side(1)]))
    payload = result.to_dict()

    assert payload["session_diff"]["pairs"][0]["base_ordinal"] == 0
    assert payload["session_diff"]["pairs"][0]["differences"][0]["sentence"]


def test_a_single_session_report_says_nothing_about_a_diff():
    payload = _result_with_diff(None).to_dict()

    assert payload["session_diff"] is None


# --- 7. 命令行的那一节 --------------------------------------------------------


def test_the_cli_report_has_a_section_for_the_diff(make_call):
    calls = [
        make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0),
        make_call(1, "create_payment", {"order_id": "A9"}, session_ordinal=1),
    ]
    text = render(_result_with_diff(compare_sessions(calls, [_side(0), _side(1)])))

    assert "--- 跨会话差异" in text
    assert "create_payment" in text
    assert "只在会话 #1 出现" in text


def test_the_cli_section_says_it_is_a_statement_not_a_verdict(make_call):
    calls = [make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0)]
    text = render(_result_with_diff(compare_sessions(calls, [_side(0), _side(1)])))

    assert "陈述，不是判定" in text
    assert "它不进检测结论，也不影响通过与否" in text


def test_the_cli_section_reports_the_differences_it_did_not_list(make_call):
    calls = [
        make_call(i, "create_payment", {"order_id": f"A{i}"}, session_ordinal=0)
        for i in range(MAX_ITEMS + 3)
    ]
    calls += [make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=1)]
    text = render(_result_with_diff(compare_sessions(calls, [_side(0), _side(1)])))

    assert "还有 4 条差异没列出来" in text


def test_a_single_session_report_has_no_diff_section():
    assert "跨会话差异" not in render(_result_with_diff(None))


def test_the_diff_never_calls_it_a_regression(make_call):
    """措辞纪律：这里说的是**差异**，不是"回归""退化"。

    同一个场景的两段会话不一样，可能是场景故意设的（时间加速、回滚记忆）。
    说成"退化"就是在给一个中性事实判了方向，而报告里没有任何东西支持那个方向。
    """
    calls = [
        make_call(1, "create_payment", {"order_id": "A1", "amount": 99}, session_ordinal=0),
        make_call(2, "create_payment", {"order_id": "A1", "amount": 88}, session_ordinal=1),
        make_call(3, "get_order", {"order_id": "A1"}, session_ordinal=1),
    ]
    text = render(_result_with_diff(compare_sessions(calls, [_side(0), _side(1)])))
    section = text.split("--- 跨会话差异")[1].split("--- 检测结论")[0]

    assert "回归" not in section
    assert "退化" not in section
