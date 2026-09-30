"""调用链（需求 3.2 P1「调用链可视化」）。

需求原文是"展示错误如何在工具调用链中传播"。六个字里最要紧的是**传播**，
而它是个因果词——边界上看得见的只有**紧接着**。所以这一节的测试分四组，
每组的落点都是一句"报告不许说得比观测更多"：

  1. **顺序。** 链的全部价值就是顺序，而顺序只能按发生次序排。编号尤其不能
     用运行序号：那个计数器是工具调用和 LLM 调用共用的，印出来会跳号。
  2. **谁的错分开写。** 目标系统回的错是它自己的；被沙箱出口策略拦下、快照
     恢复时的重放是**我们**动的手。混在一起等于替被测对象背锅。
  3. **紧接着，不是因果。** 报告写得出"紧接着又是一次同样的调用"，写不出
     "它在重试"。后一句是替 Agent 下的结论，而且没有一处会因此报错。
  4. **陈述，不是判定。** 它不进 `findings`、不影响 `ok`。目标系统回了个错，
     是发生在这条链上的事，不是被测对象做错了事。
"""

from __future__ import annotations

from typing import Any

import pytest

from sandbox.scenario.call_chain import (
    MARK_INVALID_ARGS,
    MARK_OK,
    MARK_REPLAYED,
    MARK_SANDBOX_BLOCKED,
    MARK_TARGET_ERROR,
    MARK_UNDECLARED_TOOL,
    MARK_UNKNOWN,
    MAX_NODES,
    MAX_STEPS,
    build_chain,
    mark_of,
)
from sandbox.scenario.runner import ScenarioResult, render
from sandbox.scenario.session_diff import SessionSide

# --- 构造辅助 -----------------------------------------------------------------


def _side(ordinal: int, **kw: Any) -> SessionSide:
    return SessionSide(ordinal=ordinal, name=f"第 {ordinal} 段", **kw)


def _chains(calls, sides, **kw):
    chain = build_chain(calls, sides, **kw)
    assert chain is not None
    return chain.chains


def _first(calls, sides, **kw):
    return _chains(calls, sides, **kw)[0]


def _sentences(calls, sides, **kw) -> list[str]:
    return [s.sentence() for s in _first(calls, sides, **kw).steps]


def _all_text(calls, sides, **kw) -> str:
    """这份报告里所有给人看的字。措辞纪律那几条在它上面查。"""
    chain = _chains(calls, sides, **kw)
    lines = [c.headline for c in chain]
    lines += [n.line() for c in chain for n in c.nodes]
    lines += [s.sentence() for c in chain for s in c.steps]
    return "\n".join(lines)


def _result_with_chain(chain_value: Any) -> ScenarioResult:
    result = ScenarioResult(scenario_id="s", run_id="r")
    result.call_chain = chain_value
    return result


# --- 1. 顺序是链的全部 --------------------------------------------------------


def test_the_chain_lists_every_call_in_the_order_it_happened(make_call):
    """喂进去的顺序打乱，出来必须是发生顺序——链的意义就在这里。"""
    calls = [
        make_call(9, "create_payment", {"order_id": "A1"}, session_ordinal=0),
        make_call(2, "get_order", {"order_id": "NOPE"}, session_ordinal=0),
        make_call(5, "get_order", {"order_id": "A1"}, session_ordinal=0),
    ]
    chain = _first(calls, [_side(0)])

    assert [n.tool for n in chain.nodes] == ["get_order", "get_order", "create_payment"]
    assert [n.args_text for n in chain.nodes][0].startswith('order_id="NOPE"')


def test_the_number_shown_is_the_call_number_not_the_run_seq(make_call):
    """编号用"这一段里的第几次调用"，不用运行序号。

    运行序号是工具调用与 LLM 调用**共用**的计数器，一次工具调用前后各有一条
    LLM 调用，所以真实的序号是 2、4、6、9 这种跳号。原样印出来的话，读者会
    以为中间有几次没显示出来——而链上最不能出的错就是让人以为少了东西。
    """
    calls = [
        make_call(2, "get_order", {"order_id": "A1"}, session_ordinal=0),
        make_call(4, "get_order", {"order_id": "A2"}, session_ordinal=0),
        make_call(9, "get_order", {"order_id": "A3"}, session_ordinal=0),
    ]
    chain = _first(calls, [_side(0)])

    assert [n.position for n in chain.nodes] == [1, 2, 3]
    # 运行序号没有丢——要和检测结论里的"序号"对上时得靠它。
    assert [n.seq for n in chain.nodes] == [2, 4, 9]
    assert "第 1 次" in chain.nodes[0].line()
    assert "第 9 次" not in chain.nodes[2].line()


def test_calls_are_split_by_session(make_call):
    calls = [
        make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0),
        make_call(1, "create_payment", {"order_id": "A2"}, session_ordinal=1),
    ]
    chains = _chains(calls, [_side(0), _side(1)])

    assert [c.ordinal for c in chains] == [0, 1]
    assert [n.tool for n in chains[0].nodes] == ["get_order"]
    assert [n.tool for n in chains[1].nodes] == ["create_payment"]
    # 每段各自从 1 开始编号：跨段连号会让"第 5 次"指向另一段里的一次调用。
    assert chains[1].nodes[0].position == 1


def test_a_session_without_calls_still_gets_a_line(make_call):
    """空的那一段要说话。沉默会让它看起来和"没这一段"一样。"""
    calls = [make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0)]
    chains = _chains(calls, [_side(0), _side(1)])

    assert chains[1].nodes == []
    assert "一次工具调用都没有" in chains[1].headline


def test_no_sessions_means_no_chain():
    """一段会话都没有 = 没有链可看，不是"链是干净的"。"""
    assert build_chain([], []) is None


# --- 2. 谁的错分开写 ----------------------------------------------------------


@pytest.mark.parametrize(
    ("kw", "expected"),
    [
        ({}, MARK_OK),
        ({"status": "error"}, MARK_TARGET_ERROR),
        ({"status": "invalid_args"}, MARK_INVALID_ARGS),
        ({"status": "blocked"}, MARK_UNDECLARED_TOOL),
        ({"status": "blocked", "blocked_by_proxy": True}, MARK_SANDBOX_BLOCKED),
        ({"replay_reason": "snapshot_restore"}, MARK_REPLAYED),
    ],
)
def test_every_kind_of_failure_gets_its_own_mark(make_call, kw, expected):
    call = make_call(1, "get_order", {"order_id": "A1"}, **kw)
    assert mark_of(call) == expected


def test_a_replay_that_was_also_blocked_is_read_as_a_replay(make_call):
    """重放优先于"被拦下"：两件都是我们干的，但说法不一样——
    一条重放如果被读成"沙箱拦了它"，读者会去找一条其实没发生的拦截规则。"""
    call = make_call(
        1, "get_order", {"order_id": "A1"}, replay_reason="snapshot_restore", blocked_by_proxy=True
    )
    assert mark_of(call) == MARK_REPLAYED


def test_an_unknown_status_is_not_read_as_a_target_error(make_call):
    """认不出来的 status 不许猜成"目标系统报错"——猜错方向会冤枉目标系统。"""
    call = make_call(1, "get_order", {"order_id": "A1"}, status="weird")
    chain = _first([call], [_side(0)])

    assert chain.nodes[0].mark == MARK_UNKNOWN
    assert chain.nodes[0].failed  # 反正不是成功，但不说是谁的问题
    assert "weird" in chain.nodes[0].mark_text
    assert "目标系统" not in chain.nodes[0].mark_text


def test_the_blocks_we_did_are_not_counted_as_its_failures(make_call):
    """4 次调用里只有 1 次是它自己没成功，另外 1 次是我们拦的。"""
    calls = [
        make_call(1, "get_order", {"order_id": "NOPE"}, status="error", session_ordinal=0),
        make_call(2, "get_order", {"order_id": "A1"}, status="error", session_ordinal=0),
        make_call(3, "create_payment", {"order_id": "A1"}, session_ordinal=0, is_write=True),
        make_call(
            4,
            "create_payment",
            {"order_id": "A2"},
            session_ordinal=0,
            blocked_by_proxy=True,
            status="blocked",
        ),
    ]
    chain = _first(calls, [_side(0)])

    assert chain.total == 4
    assert chain.failed == 2
    assert chain.ours == 1
    assert "2 次没成功" in chain.headline
    assert "1 次是沙箱拦下或重放的" in chain.headline
    # 计数分开，措辞也要分开：那一句必须写明"不是它的失败"。
    blocked = chain.steps[-1].sentence()
    assert "这不是它的失败" in blocked


# --- 3. 紧接着，不是因果 ------------------------------------------------------


@pytest.mark.parametrize(
    ("second", "expected"),
    [
        ({"tool": "get_order", "args": {"order_id": "NOPE"}}, "又是同一个调用"),
        ({"tool": "get_order", "args": {"order_id": "A1"}}, "还是 get_order，参数变了"),
        ({"tool": "create_payment", "args": {"order_id": "A1"}}, "换成了别的工具"),
    ],
)
def test_the_call_right_after_a_failure_is_described_plainly(make_call, second, expected):
    calls = [
        make_call(1, "get_order", {"order_id": "NOPE"}, status="error", session_ordinal=0),
        make_call(2, second["tool"], second["args"], session_ordinal=0),
    ]
    assert expected in _sentences(calls, [_side(0)])[0]


def test_a_failure_at_the_end_says_the_calls_just_stopped(make_call):
    calls = [make_call(1, "get_order", {"order_id": "NOPE"}, status="error", session_ordinal=0)]
    assert "这一段的调用就结束了" in _sentences(calls, [_side(0)])[0]


def test_a_killed_session_says_the_calls_ended_because_of_the_kill(make_call):
    """被熔断终止和"它自己停下来了"必须分得开。

    前者是我们的看门狗动的手，后者可能是它做了决定——同一条链，两种含义。
    """
    calls = [make_call(1, "get_order", {"order_id": "NOPE"}, status="error", session_ordinal=0)]
    plain = _sentences(calls, [_side(0)])[0]
    killed = _sentences(calls, [_side(0, killed=True)])[0]

    assert "被熔断终止" not in plain
    assert "被熔断终止" in killed


def test_a_replay_after_a_failure_is_not_read_as_a_retry(make_call):
    """紧接着出现一次快照重放，看起来和"它原样又发了一次"一模一样——
    区别只在 `replay_reason`。不特判的话，恢复机制制造的调用会被算到它头上。"""
    calls = [
        make_call(1, "get_order", {"order_id": "NOPE"}, status="error", session_ordinal=0),
        make_call(2, "get_order", {"order_id": "NOPE"}, replay_reason="snapshot_restore"),
    ]
    sentence = _sentences(calls, [_side(0)])[0]

    assert "快照重放" in sentence
    assert "不是它主动重发" in sentence
    assert "又是同一个调用" not in sentence


def test_the_same_failure_twice_is_told_as_one_kind_with_two_occurrences(make_call):
    """同样的失败沿链出现了几次，是一句话里最值钱的部分。

    逐次列出来的话，一段循环会刷满屏，而"一共几次"恰恰是要看的东西。
    """
    calls = [
        make_call(1, "get_order", {"order_id": "NOPE"}, status="error", session_ordinal=0),
        make_call(2, "get_order", {"order_id": "NOPE"}, status="error", session_ordinal=0),
        make_call(3, "get_order", {"order_id": "A1"}, session_ordinal=0),
    ]
    steps = _first(calls, [_side(0)]).steps

    assert len(steps) == 1
    assert steps[0].times == 2
    assert steps[0].positions == (1, 2)
    sentence = steps[0].sentence()
    assert "一共出现了 2 次" in sentence and "第 1、2 次" in sentence
    # 第一次之后和最后一次之后各说各的：它们可以不一样，而这正是"沿链怎么走"。
    assert "第 1 次之后紧接着又是同一个调用" in sentence
    assert "第 2 次之后紧接着还是 get_order，参数变了" in sentence


def test_the_chain_never_says_because(make_call):
    """措辞纪律：写得出"紧接着"，写不出"因为……所以……"。

    那一句是替 Agent 下的结论——边界上看不见它的理由，而猜错了没有任何一处
    会报错。整份报告里给人看的字都不许出现因果连词。
    """
    calls = [
        make_call(1, "get_order", {"order_id": "NOPE"}, status="error", session_ordinal=0),
        make_call(2, "get_order", {"order_id": "NOPE"}, status="error", session_ordinal=0),
        make_call(3, "create_payment", {"order_id": "A1"}, session_ordinal=0),
    ]
    text = _all_text(calls, [_side(0)])

    assert "因为" not in text
    assert "所以" not in text
    assert "因此" not in text


# --- 4. 正面证据 --------------------------------------------------------------


def test_a_clean_session_says_nothing_failed(make_call):
    """干净的那一段要自己说出来。

    "没有一次不成功"和"这一节是空的"看起来一样，而前者是结论、后者是没查。
    """
    calls = [
        make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0),
        make_call(2, "create_payment", {"order_id": "A1"}, session_ordinal=0),
    ]
    chain = _first(calls, [_side(0)])

    assert chain.failed == 0 and chain.ours == 0
    assert "没有一次不成功" in chain.headline


def test_a_session_where_only_we_blocked_does_not_read_as_clean(make_call):
    """我们拦下的那一次不算它失败，但**也不能因此说这一段是干净的**。

    "没有一次不成功"是一句关于它的话；这一次它的调用根本没出去，
    说它成功、说它失败都是在说一件没发生的事。
    """
    calls = [
        make_call(
            1,
            "create_payment",
            {"order_id": "A1"},
            session_ordinal=0,
            blocked_by_proxy=True,
            status="blocked",
        )
    ]
    chain = _first(calls, [_side(0)])

    assert chain.failed == 0 and chain.ours == 1
    assert "没有一次不成功" not in chain.headline
    assert "1 次是沙箱拦下或重放的" in chain.headline


# --- 5. 长报告不静默截断 ------------------------------------------------------


def test_too_many_calls_are_capped_and_the_rest_are_announced(make_call):
    calls = [
        make_call(i, "get_order", {"order_id": f"A{i}"}, session_ordinal=0)
        for i in range(MAX_NODES + 4)
    ]
    chain = _first(calls, [_side(0)])

    assert len(chain.nodes) == MAX_NODES
    assert chain.total == MAX_NODES + 4
    assert chain.omitted_nodes == 4
    assert "还有 4 次没列出来" in chain.headline


def test_too_many_failures_are_capped_and_the_rest_are_announced(make_call):
    calls = [
        make_call(
            i, "get_order", {"order_id": f"A{i}"}, status="error", session_ordinal=0
        )
        for i in range(MAX_STEPS + 3)
    ]
    chain = _first(calls, [_side(0)])

    assert len(chain.steps) == MAX_STEPS
    assert chain.omitted_steps == 3


def test_nothing_is_omitted_when_everything_fits(make_call):
    calls = [make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0)]
    chain = _first(calls, [_side(0)])

    assert chain.omitted_nodes == 0 and chain.omitted_steps == 0
    assert "没列出来" not in chain.headline


def test_the_report_records_the_limits_it_was_truncated_with(make_call):
    """两个上限随报告一起存。

    半年后打开这份旧报告，页面得说得出**它当时**是按多少截的。用今天那个数，
    等于拿今天的常量去说一件过去的事。
    """
    calls = [make_call(i, "get_order", {"order_id": f"A{i}"}, session_ordinal=0) for i in range(6)]
    chain = build_chain(calls, [_side(0)], max_nodes=2, max_steps=1)

    assert chain is not None
    payload = chain.to_dict()
    assert payload["node_limit"] == 2 and payload["step_limit"] == 1
    assert build_chain(calls, [_side(0)]).to_dict()["node_limit"] == MAX_NODES


# --- 6. 陈述，不是判定 --------------------------------------------------------


def test_the_chain_never_enters_the_findings_or_the_verdict(make_call):
    """`ok` 只看 findings / 前提 / 地面真值，链**一个字都不许影响**它们。"""
    calls = [
        make_call(1, "get_order", {"order_id": "NOPE"}, status="error", session_ordinal=0),
        make_call(2, "get_order", {"order_id": "NOPE"}, status="error", session_ordinal=0),
    ]
    chain = build_chain(calls, [_side(0)])
    assert chain is not None and chain.chains[0].failed == 2

    result = _result_with_chain(chain)

    assert result.findings == []
    assert result.ok is False  # 没有 expectation，本来就不通过
    assert result.to_dict()["findings"] == []


def test_the_report_carries_the_chain_as_a_separate_field(make_call):
    calls = [make_call(1, "get_order", {"order_id": "NOPE"}, status="error", session_ordinal=0)]
    result = _result_with_chain(build_chain(calls, [_side(0)]))
    payload = result.to_dict()

    assert payload["call_chain"]["chains"][0]["nodes"][0]["mark"] == MARK_TARGET_ERROR
    assert payload["call_chain"]["chains"][0]["steps"][0]["sentence"]


def test_a_run_without_sessions_says_nothing_about_a_chain():
    payload = _result_with_chain(None).to_dict()

    assert payload["call_chain"] is None


# --- 7. 命令行的那一节 --------------------------------------------------------


def _chain_result(calls, sides, **kw) -> ScenarioResult:
    return _result_with_chain(build_chain(calls, sides, **kw))


def _chain_section(text: str) -> str:
    return text.split("--- 调用链")[1].split("--- 记忆观测")[0]


def test_the_cli_report_has_a_section_for_the_chain(make_call):
    calls = [
        make_call(1, "get_order", {"order_id": "NOPE"}, status="error", session_ordinal=0),
        make_call(2, "get_order", {"order_id": "A1"}, session_ordinal=0),
    ]
    text = render(_chain_result(calls, [_side(0)]))

    assert "--- 调用链" in text
    assert "第 1 次 get_order" in text
    assert "目标系统回了错" in text
    assert "接着又是" not in text  # 措辞是"紧接着"，不是"接着"
    assert "紧接着" in text


def test_the_cli_section_says_it_is_a_statement_not_a_verdict(make_call):
    calls = [make_call(1, "get_order", {"order_id": "NOPE"}, status="error", session_ordinal=0)]
    text = render(_chain_result(calls, [_side(0)]))

    section = _chain_section(text)
    assert "陈述，不是判定" in section
    assert "不进检测结论" in section
    assert "不是因果" in section


def test_the_cli_section_reports_the_failures_it_did_not_list(make_call):
    calls = [
        make_call(i, "get_order", {"order_id": f"A{i}"}, status="error", session_ordinal=0)
        for i in range(MAX_STEPS + 2)
    ]
    text = render(_chain_result(calls, [_side(0)]))

    assert "还有 2 种没成功的调用没列出来" in text
    assert f"最多列 {MAX_STEPS} 种" in text


def test_the_cli_section_separates_the_blocks_we_did_from_its_own_failures(make_call):
    calls = [
        make_call(1, "get_order", {"order_id": "NOPE"}, status="error", session_ordinal=0),
        make_call(
            2,
            "create_payment",
            {"order_id": "A1"},
            session_ordinal=0,
            blocked_by_proxy=True,
            status="blocked",
        ),
    ]
    section = _chain_section(render(_chain_result(calls, [_side(0)])))

    assert "1 次没成功" in section
    assert "1 次是沙箱拦下或重放的" in section
    assert "这不是它的失败" in section


def test_a_premise_that_did_not_hold_is_pointed_out(make_call):
    """前提没成立时链照列，但要指回「会话」那一节的解释——
    不指的话，一段脚本没跑完的轨迹会被当成它的行为读。"""
    calls = [make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0)]
    text = render(_chain_result(calls, [_side(0, premise_held=False)]))

    assert "这一段的前提没成立" in text
    assert "不指向被测对象的行为" in text


def test_a_run_without_sessions_has_no_chain_section():
    assert "调用链" not in render(_result_with_chain(None))
