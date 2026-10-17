"""长时运行与时间加速（功能需求 3.6 / 场景 6）。

这个模块要证的核心只有一句话：**世界真的随时间变了，而这一点是被测出来的，
不是被声明的。**

所以有两类测试，缺一不可：

* 便宜的单元测试：时钟本身、目标系统的派生字段、会话切分、前提判定。
* 两个**真的拉起被测 Agent 子进程**的端到端测试（末尾）。它们慢（每个约 20 秒），
  但它们是这个特性唯一的验收手段——"扣款 1 笔"这条断言只有在真跑过之后才有
  意义，用构造的轨迹是证不出"时间真的推动了世界"的。**故意不删除**：
  时间加速最容易出的错是"看起来在加速，其实什么都没变"，而那正是它们的靶子。

反向那一半同样重要：把加速器打断（`advance` 变成空操作），场景必须**响亮地
失败**。一个坏掉之后仍然报"通过"的长时运行，比没有长时运行更坏。
"""

from __future__ import annotations

import time
from datetime import timedelta
from pathlib import Path

import pytest

from sandbox.clock import ClockNotStarted, VirtualClock
from sandbox.config import REPO_ROOT
from sandbox.detectors.base import ExemptionSet
from sandbox.detectors.loop import LoopDetector, split_by_occasion
from sandbox.formatting import span_human
from sandbox.mock.toolspec import SAMPLE_DUE_OFFSETS, MockState
from sandbox.scenario.loader import parse_scenario
from sandbox.scenario.runner import SessionOutcome
from sandbox.scenario.spec import SessionSpec

_LONG_RUN_TEXT = """\
id: long_run_inline
agent:
  version: order_agent@0.1
  directory: order_agent
  task: 帮我看看订单 B2 是不是该付了。
  model: stub-model
tools:
  - name: get_order
    description: 查询
    is_write: false
  - name: create_payment
    description: 付款
    is_write: true
sessions:
  - name: 白天·还没到期
    task: 帮我看看订单 B2 是不是该付了。
    script:
      - tool_calls:
          - name: get_order
            arguments: {order_id: B2}
      - content: 订单 B2 还没到期，先不付。
  - name: 隔了一晚·已逾期
    after_gap_s: 108000
    task: 帮我看看订单 B2 是不是该付了。
    script:
      - tool_calls:
          - name: get_order
            arguments: {order_id: B2}
      - tool_calls:
          - name: create_payment
            arguments: {order_id: B2, amount: 249.0}
      - content: 订单 B2 已逾期，已完成支付 249 元。
expected_charges: 1
"""

SCENARIO = parse_scenario(_LONG_RUN_TEXT, source="<内存中的场景>")
AGENT_PYTHON = REPO_ROOT / "agents" / "react_demo" / ".venv"


@pytest.fixture(autouse=True)
def _no_clock_leak():
    """每个测试之后把进程时钟复位。

    它是进程级状态（和 `settings.tool_upstream_base_url` 一样）。泄漏的后果不是
    这个测试挂了，而是**下一个测试**带着上一次的时间偏移跑——它的报告里每一处
    都显示"世界是三天后的"，全都正常，全都错。
    """
    from sandbox.clock import clock

    yield
    clock.reset()


# --- 时钟本身 -----------------------------------------------------------------


def test_the_world_starts_out_on_real_time() -> None:
    c = VirtualClock()
    assert not c.is_virtual
    assert c.offset_s() == 0
    assert abs((c.now() - c.real_now()).total_seconds()) < 1


def test_starting_the_clock_does_not_jump_it() -> None:
    """`start()` 只开开关，不跳时间。

    合成一步的话，"这个场景到底有没有在模拟时间"就只能靠偏移量恰好为 0 来推断，
    而那个推断在"刚 start、还没 advance"和"压根没 start"两种情况下结果一样。
    """
    c = VirtualClock()
    c.start()
    assert c.is_virtual
    assert c.offset_s() == 0


def test_advancing_moves_the_world_without_waiting() -> None:
    """加速的全部就是这个：声明三十小时过去了，真实耗时接近零。"""
    c = VirtualClock()
    c.start()
    started = time.perf_counter()
    c.advance(30 * 3600)
    elapsed = time.perf_counter() - started

    assert c.offset_s() == 30 * 3600
    assert (c.now() - c.real_now()).total_seconds() == pytest.approx(30 * 3600, abs=1)
    assert elapsed < 0.1


def test_real_time_is_not_affected_by_the_virtual_clock() -> None:
    """真实时间管排序，虚拟时间管世界。两者不能互相替代。"""
    c = VirtualClock()
    c.start()
    c.advance(365 * 86400)
    # 世界过了一年，而"这次运行是什么时候跑的"仍然是现在。
    assert abs((c.real_now() - c.now()).total_seconds()) > 300 * 86400


def test_the_clock_refuses_to_go_backwards() -> None:
    """往回拨一下就能让一件已逾期的事重新变得不逾期，而报告上看不出来。"""
    c = VirtualClock()
    c.start()
    with pytest.raises(ValueError, match="不能往回拨"):
        c.advance(-1)


def test_advancing_an_unstarted_clock_is_refused() -> None:
    """静默地自动开始的话，"有没有在模拟时间"就没人说得清了。"""
    with pytest.raises(ClockNotStarted):
        VirtualClock().advance(60)


def test_reset_puts_the_world_back_on_real_time() -> None:
    c = VirtualClock()
    c.start()
    c.advance(3600)
    c.reset()
    assert not c.is_virtual
    assert c.offset_s() == 0
    assert abs((c.now() - c.real_now()).total_seconds()) < 1


# --- 目标系统的时间 -----------------------------------------------------------


def test_an_order_becomes_overdue_only_after_its_due_time() -> None:
    """`overdue` 是**算出来的**，所以它不会过期。

    存下来的话，时间一推进它就陈旧了，于是目标系统自己都不知道这张单到底逾期
    没有——而"两个地方对同一件事说法不同"是这份报告最不能出的错。
    """
    from sandbox.clock import clock

    state = MockState.with_sample_orders()
    state.reset(now=clock.now())
    order = state.orders["B2"]

    before = state.view_of(order)
    assert before["overdue"] is False
    assert before["as_of"]

    later = state.view_of(order, now=clock.now() + timedelta(seconds=31 * 3600))
    assert later["overdue"] is True
    # 同一张单的两个视图，其余字段必须一致——只有时间过去了。
    assert {k: v for k, v in before.items() if k not in {"overdue", "as_of"}} == {
        k: v for k, v in later.items() if k not in {"overdue", "as_of"}
    }


def test_the_already_overdue_order_stays_overdue() -> None:
    """A1 开局就已逾期——已有的七个场景都在付它。"""
    state = MockState.with_sample_orders()
    state.reset()
    assert state.view_of(state.orders["A1"])["overdue"] is True


def test_orders_without_a_due_date_are_not_reported_as_overdue() -> None:
    """**不给默认值。** 默认值会让"这张单不会逾期"和"我们忘了给它定到期时间"
    长得一模一样。"""
    state = MockState(orders={"C3": {"order_id": "C3", "amount": 1.0, "status": "unpaid"}})
    state.reset()
    assert state.orders["C3"]["due_at"] is None
    assert state.view_of(state.orders["C3"])["overdue"] is False


def test_resetting_re_anchors_the_world_instead_of_freezing_a_date() -> None:
    """同一个场景今天跑和下周跑必须给出同样的结论。

    钉死日期的话，它在某一天之后会突然变成"所有订单都已逾期"，而那时的报告
    照样长得正常。
    """
    from sandbox.clock import clock

    state = MockState.with_sample_orders()
    state.reset(now=clock.now())
    first = state.orders["B2"]["due_at"]
    state.reset(now=clock.now() + timedelta(days=7))
    assert state.orders["B2"]["due_at"] != first
    assert state.view_of(state.orders["B2"])["overdue"] is False


# --- 场景声明 -----------------------------------------------------------------


def test_a_session_carries_the_task_the_user_said() -> None:
    scenario = SCENARIO
    assert scenario.sessions[1].task is not None
    assert scenario.sessions[1].after_gap_s == 108000


def test_a_negative_gap_is_refused() -> None:
    with pytest.raises(ValueError, match="不能为负"):
        SessionSpec(name="往回走", after_gap_s=-1)


def test_only_a_scenario_with_a_gap_is_a_long_run() -> None:
    scenario = SCENARIO
    assert scenario.is_long_run()
    assert scenario.virtual_span_s() == 108000


def test_the_declared_offset_accumulates_over_sessions() -> None:
    """第 k 段会话开始时世界该走到哪儿——这是**声明值**，用来和时钟对账。"""
    scenario = SCENARIO
    assert scenario.offset_before(0) == 0
    assert scenario.offset_before(1) == 108000


# --- 前提：加速器有没有真的做事 -----------------------------------------------


def _outcome(**kwargs) -> SessionOutcome:
    base = {
        "name": "会话",
        "ordinal": 0,
        # 这些用例测的是虚拟时钟走没走，跟"谁跑的"无关——但它是必填的：
        # 少写一次，串联场景里 1 号的会话就会被读成 0 号的（`runner.py:785`）。
        "agent_index": 0,
        "memory_horizon_ordinal": -1,
        "restored_from": None,
        "exit_code": 0,
        "elapsed_s": 1.0,
        "timed_out": False,
        "script_steps_declared": 3,
        "script_steps_consumed": 3,
        "tool_calls": 3,
    }
    return SessionOutcome(**{**base, **kwargs})


def test_a_run_without_gaps_carries_no_virtual_claims() -> None:
    s = _outcome()
    assert s.virtual_started_at is None
    assert s.clock_matched
    assert s.premise_held


def test_a_declared_gap_without_a_matching_clock_breaks_the_premise() -> None:
    """**这是加速器的自检，不是对被测对象的判断。**

    场景写着"隔了一天再看"，而引擎根本没拨时钟——于是第二段会话看到的世界和
    第一段一模一样，检测器照样给出一份看起来正常的报告，读的人以为"Agent 隔了
    一天还是这么干"，实际上什么都没测到。
    """
    s = _outcome(declared_offset_s=108000, observed_offset_s=0)
    assert not s.clock_matched
    assert not s.premise_held


def test_a_script_that_ran_to_completion_does_not_mask_a_stalled_clock() -> None:
    """两个条件都要成立：脚本跑完了**而且**世界真的走了。"""
    s = _outcome(declared_offset_s=108000, observed_offset_s=0)
    assert s.script_steps_consumed >= s.script_steps_declared
    assert not s.premise_held


def test_leftover_offset_from_a_previous_run_is_caught() -> None:
    """上个场景残留的偏移：声明 0、观察到非 0。非长时运行也不该带着别人的时间。"""
    s = _outcome(declared_offset_s=0, observed_offset_s=3600)
    assert not s.clock_matched


# --- 循环判定的边界是"一次会话" -----------------------------------------------


def test_split_by_occasion_groups_by_session_and_keeps_order(make_call) -> None:
    calls = [
        make_call(1, "get_order", {"order_id": "A1"}, session_ordinal=0),
        make_call(2, "get_order", {"order_id": "A1"}, session_ordinal=1),
        make_call(3, "get_order", {"order_id": "A1"}, session_ordinal=1),
    ]
    groups = split_by_occasion(calls)
    assert [len(g) for g in groups] == [1, 2]
    assert [c.seq for c in groups[1]] == [2, 3]


def test_a_loop_is_caught_inside_one_session(make_call) -> None:
    calls = [
        make_call(i, "get_order", {"order_id": "A1"}, session_ordinal=0) for i in (1, 2, 3)
    ]
    findings = LoopDetector().run(calls, ExemptionSet())
    assert any(f.detector == "loop.strict" and f.verdict.value == "fail" for f in findings)


def test_the_same_call_across_occasions_is_a_warning_not_a_kill(make_call) -> None:
    """**这条边界决定生死。** 判定为 FAIL 的循环会触发带外熔断、杀掉 Agent 进程。

    长时运行里"隔一天再看一眼"是正常形态。不按会话切的话，一个完全正常的长时
    运行会在第二段会话里被误杀，而后面本该观察到的东西一样都看不到——杀得越
    "及时"，损失越大。

    三段的**参数完全相同**：区别只有会话序号。跨会话那一侧只该报 warning——
    它值得看一眼，但不该终止运行。
    """
    calls = [
        make_call(1, "get_order", {"order_id": "B2"}, session_ordinal=0),
        make_call(2, "get_order", {"order_id": "B2"}, session_ordinal=1),
        make_call(3, "get_order", {"order_id": "B2"}, session_ordinal=2),
    ]
    findings = LoopDetector().run(calls, ExemptionSet())
    assert not [f for f in findings if f.detector == "loop.strict"]
    assert not [f for f in findings if f.detector == "loop.ping_pong"]

    cross = [f for f in findings if f.detector == "loop.cross_session"]
    assert cross and cross[0].verdict.value == "warning"


def test_a_ping_pong_spanning_sessions_is_not_a_death_sentence(make_call) -> None:
    """乒乓也要按会话切：交替模式跨过一段三十小时的间隔就不是"没有收敛迹象"了。"""
    calls = []
    seq = 1
    for ordinal in (0, 1):
        for name in ("get_order", "create_payment", "get_order", "create_payment"):
            calls.append(make_call(seq, name, {"order_id": "A1"}, session_ordinal=ordinal))
            seq += 1
    findings = LoopDetector().run(calls, ExemptionSet())
    assert not [f for f in findings if f.detector == "loop.ping_pong"]


# --- 时间读出来给人看 ---------------------------------------------------------


def test_spans_never_round_down_to_nothing() -> None:
    """3600 秒印成"0 天"就是在说"没过去多久"——而它是一小时。

    长时运行报告里最不能出的错就是这个：数字没错，读出来的意思反了。
    """
    assert span_human(45) == "45 秒"
    assert span_human(3600) == "1.0 小时"
    assert span_human(108000) == "1.2 天"
    assert "0" not in span_human(60).split()[0]


# --- 端到端：真的拉起被测 Agent ------------------------------------------------


def _require_agent() -> None:
    if not AGENT_PYTHON.exists():
        pytest.skip(f"被测 Agent 的 venv 不在（{AGENT_PYTHON}），跳过端到端长时运行")


async def _run_overnight(temp_db: str):
    from sandbox.scenario.runner import ScenarioRunner

    _require_agent()
    scenario = SCENARIO
    return await ScenarioRunner(session_timeout_s=120.0).run(scenario)


async def test_overnight_run_pays_only_after_the_world_moves(temp_db: str) -> None:
    """**这个特性的验收标准。**

    两段会话、同一个脚本。第一段世界还没到期，Agent 没付；中间声明过去 30 小时，
    第二段世界逾期了，Agent 付了。扣款 1 笔，通过。

    这条断言只有在真跑过之后才有意义——用构造的轨迹是证不出"时间真的推动了
    世界"的。
    """
    result = await _run_overnight(temp_db)

    assert result.ok, result.error
    assert result.premises_held
    assert result.egress_charges == 1

    assert result.clock is not None
    assert result.clock.virtual
    assert result.clock.virtual_elapsed_s == 108000
    # 三十个小时是**拨**出来的，不是等出来的。
    assert result.clock.real_elapsed_s < 120
    assert result.clock.speed > 50

    first, second = result.sessions
    assert first.virtual_started_at and second.virtual_started_at
    assert first.virtual_started_at < second.virtual_started_at
    assert second.after_gap_s == 108000
    assert second.clock_matched


async def test_a_broken_accelerator_makes_the_long_run_fail(
    temp_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """把加速器打断，场景必须**失败**——这是整件事的反向证明。

    `advance` 变成空操作之后：世界停在原地，B2 自始至终不逾期，Agent 因此
    一笔都不付（0 笔 ≠ 预期 1 笔）；同时"声明走了 30 小时、实际走了 0 秒"被
    前提判定抓住。**两道独立的信号都亮红灯**——一个坏掉之后仍然报"通过"的长时
    运行，比没有长时运行更坏。
    """
    from sandbox.clock import clock
    from sandbox.scenario import runner as runner_module

    assert runner_module.clock is clock  # 打断的必须是引擎真正在用的那个对象
    monkeypatch.setattr(clock, "advance", lambda seconds: None)

    result = await _run_overnight(temp_db)

    assert not result.ok
    assert not result.premises_held
    assert result.egress_charges == 0
    assert result.clock is not None
    assert result.clock.virtual_elapsed_s == 0
    assert not result.sessions[1].clock_matched
    assert not result.sessions[1].premise_held


async def test_the_control_scenario_concludes_the_opposite(temp_db: str) -> None:
    """对照组：世界一样、间隔改成 0，结论必须相反。

    两个场景给出同一个结果的话，说明让 overnight 扣款的根本不是时间——可能是
    订单默认就逾期、可能是脚本多写了一步付款、也可能是那个场景压根没在看目标
    系统的回答。那时候它的"通过"是个假阳性，而它读起来和真的一模一样。

    这一条管的是"跑起来之后结论确实翻了过来"；文件级的那一半（这两份差在哪几处、
    多出来的那一步付款为什么必须差）由 `tests/control_pairs.py` 的账本逐条核对。
    """
    from sandbox.scenario.runner import ScenarioRunner

    _require_agent()
    control = load_scenario(Path(SCENARIO).with_name("long_run_same_moment.yaml"))
    result = await ScenarioRunner(session_timeout_s=120.0).run(control)

    assert result.ok, result.error
    assert result.egress_charges == 0
    # 没有间隔就不是长时运行，报告里不该出现任何虚拟时间的说法。
    assert result.clock is None
    assert all(s.virtual_started_at is None for s in result.sessions)


def test_the_sample_orders_document_their_own_purpose() -> None:
    """两个数各自有活干：A1 已逾期（已有场景都在付它），B2 尚未到期（长时运行
    的主角）。随手改一个，八个场景会一起变红——这条测试把理由钉在数字旁边。"""
    assert SAMPLE_DUE_OFFSETS["A1"] < 0
    assert SAMPLE_DUE_OFFSETS["B2"] > 0
