"""指标看板（需求 3.7）。

这一片的价值全在**口径**上，所以测试几乎都在钉"哪一类算哪里"，而不是钉数字
算得对不对：

* 结论按 `verdict_ok` 分，不按 `status` 分；
* 没留结论的运行不进达成率的分母；
* 一次不通过可能同时占好几行原因，也可能一行都占不上（报告没留存）；
* 延迟报的是中位数不是平均，三个口径不合并。

前一半是纯函数（喂 `RunSample` 就行），后一半要真的铺库——因为"所有栏位取同一
批运行"这条只有库在的时候才测得出来。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from sandbox.db.models import LlmCall, TestRun, ToolCall
from sandbox.metrics import (
    CHARGES,
    EXECUTION,
    EXPECTATION,
    PREMISE,
    LatencyStats,
    RunSample,
    report,
    run_wall_stats,
    summarize,
)

BEGIN = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)


def _sample(
    *,
    run_id: str = "r1",
    scenario: str = "s",
    status: str = "completed",
    verdict: bool | None = True,
    report_body: dict[str, object] | None = None,
    minutes: int = 0,
    duration_s: float | None = 30.0,
) -> RunSample:
    started = BEGIN + timedelta(minutes=minutes)
    return RunSample(
        run_id=run_id,
        scenario_id=scenario,
        status=status,
        verdict_ok=verdict,
        started_at=started,
        completed_at=None if duration_s is None else started + timedelta(seconds=duration_s),
        report=report_body,
    )


def _report(
    *,
    error: str | None = None,
    premises_held: bool = True,
    charges_ok: bool = True,
    missing: list[object] | None = None,
    unexpected: list[object] | None = None,
) -> dict[str, object]:
    """一份最小的结论快照，形状照 `ScenarioResult.to_dict()`。"""
    if error is not None and not missing and not unexpected:
        # 执行崩了的时候，预期那一项根本不会存在——照真实形状来。
        expectation: dict[str, object] | None = None
    else:
        expectation = {
            "matched": [],
            "missing": missing or [],
            "unexpected": unexpected or [],
            "uncovered": [],
        }
    return {
        "error": error,
        "premises_held": premises_held,
        "charges_ok": charges_ok,
        "expectation": expectation,
    }


# --- 结论分布 -----------------------------------------------------------------


def test_a_killed_but_passing_run_counts_as_passed() -> None:
    # 循环场景里进程是被我们亲手熔断杀掉的（status=killed），而那次运行是**通过**
    # 的：检测器报出来了、看门狗动手了，那正是场景想看到的东西。按状态分类的话，
    # 这条通过会被算成一个失败——这个测试钉的就是这件事。
    killed = _sample(status="killed", verdict=True)
    outcomes = summarize([killed]).outcomes
    assert outcomes.passed == 1
    assert outcomes.failed == 0
    assert outcomes.no_verdict == 0
    assert outcomes.rate == 1.0


def test_runs_without_a_verdict_are_not_in_the_denominator() -> None:
    samples = [
        _sample(run_id="a", verdict=True),
        _sample(run_id="b", verdict=True),
        _sample(run_id="c", verdict=False),
        # 还在跑的运行：没有结论，也不该被当成"不通过"。
        _sample(run_id="d", status="running", verdict=None),
        # 崩在收尾之前的运行：同样没留下结论。
        _sample(run_id="e", status="failed", verdict=None),
    ]
    outcomes = summarize(samples).outcomes
    assert outcomes.considered == 5
    assert outcomes.decided == 3
    # 分母是 3 不是 5。把它算成 3/5 就是在替两条没有结论的运行编结论。
    assert outcomes.rate == 2 / 3
    assert outcomes.no_verdict == 2
    assert dict(outcomes.no_verdict_by_status) == {"failed": 1, "running": 1}


def test_no_decided_runs_means_no_rate_at_all() -> None:
    outcomes = summarize([_sample(verdict=None)]).outcomes
    # None，不是 0，也不是 1——那两个数都是替"什么都没判"编出来的结论。
    assert outcomes.rate is None


# --- 不通过的原因 -------------------------------------------------------------


def test_a_passing_run_never_enters_the_ledger() -> None:
    ledger = summarize([_sample(verdict=True, report_body=_report())]).failures
    assert ledger.failed == 0
    assert all(cause.runs == 0 for cause in ledger.causes)


def test_one_failure_can_hold_several_causes() -> None:
    # 执行崩了 → 没跑到判定那一步；而"没跑到"本身又让预期那一项不存在。
    # 两行都算，因为两句话当时都成立。**所以各行之和不等于不通过的条数。**
    body = _report(error="RuntimeError: boom")
    ledger = summarize([_sample(verdict=False, report_body=body)]).failures
    counts = {cause.code: cause.runs for cause in ledger.causes}
    assert ledger.failed == 1
    assert counts[EXECUTION] == 1
    assert sum(counts.values()) == 1  # 这一条只占一行：没有会话就没有别的前提可谈


def test_premise_and_charges_mismatch_are_their_own_rows() -> None:
    body = _report(premises_held=False, charges_ok=False, missing=[{"detector": "x"}])
    ledger = summarize([_sample(verdict=False, report_body=body)]).failures
    counts = {cause.code: cause.runs for cause in ledger.causes}
    assert counts[PREMISE] == 1
    assert counts[CHARGES] == 1
    assert counts[EXPECTATION] == 1
    assert counts[EXECUTION] == 0
    # 三行都算，而这一批只有一次运行——这正是那句话说的事。
    assert sum(counts.values()) == 3


def test_the_expectation_row_splits_missing_from_unexpected() -> None:
    body = _report(
        missing=[{"detector": "loop"}],
        unexpected=[{"detector": "idempotency"}, {"detector": "memory"}],
    )
    ledger = summarize([_sample(verdict=False, report_body=body)]).failures
    # 两类各自数**运行条数**，不是结论条数——这条运行在两类上各算一次。
    # 预期没命中和"出现了没预料到的结论"是方向相反的两种成因，处理方式也
    # 完全不同（前者多半要去看用例）。
    assert ledger.missing_runs == 1
    assert ledger.unexpected_runs == 1


def test_a_failure_with_no_report_is_counted_but_not_attributed() -> None:
    ledger = summarize([_sample(verdict=False, report_body=None)]).failures
    assert ledger.failed == 1
    assert ledger.unreadable == 1
    assert all(cause.runs == 0 for cause in ledger.causes)


def test_a_report_missing_a_key_is_not_attributed_either() -> None:
    # 老报告（`charges_ok` 还不存在时写下的）：缺一项就整条读不出来。按缺的那项
    # 反推"它当时没问题"或"它当时有问题"，都是替一条没有证据的运行编理由。
    body = _report(missing=[{"detector": "loop"}])
    del body["charges_ok"]
    ledger = summarize([_sample(verdict=False, report_body=body)]).failures
    assert ledger.unreadable == 1
    assert all(cause.runs == 0 for cause in ledger.causes)


def test_the_ledger_only_reads_failed_runs() -> None:
    # 通过的那条报告里也带着一段 missing（比如后来被人手工改过），不该被算进
    # 原因表——原因表的定义是"不通过的那些为什么不过"。
    body = _report(missing=[{"detector": "loop"}])
    ledger = summarize(
        [
            _sample(run_id="a", verdict=True, report_body=body),
            _sample(run_id="b", verdict=False, report_body=body),
        ]
    ).failures
    assert ledger.failed == 1
    assert {cause.code: cause.runs for cause in ledger.causes}[EXPECTATION] == 1


# --- 按场景 -------------------------------------------------------------------


def test_scenarios_are_counted_apart_and_sorted() -> None:
    summary = summarize(
        [
            _sample(run_id="a", scenario="zeta", verdict=False),
            _sample(run_id="b", scenario="alpha", verdict=True),
            _sample(run_id="c", scenario="alpha", verdict=True),
        ]
    )
    assert [row.scenario_id for row in summary.by_scenario] == ["alpha", "zeta"]
    alpha = summary.by_scenario[0].tally
    assert (alpha.passed, alpha.failed) == (2, 0)


# --- 延迟 ---------------------------------------------------------------------


def test_the_median_ignores_the_long_tail() -> None:
    # 一次被超时强杀的工具调用（180 秒）能把平均拉到一个任何一次真实调用都没
    # 见过的数——而那正是读者最该看见的点。中位数不动，最大值把它摆出来。
    values = [10, 11, 12, 180_000]
    stats = LatencyStats.of(values)
    assert stats.p50_ms == 12
    assert stats.max_ms == 180_000
    assert stats.p50_ms != round(sum(values) / len(values))


def test_the_median_is_always_a_number_we_measured() -> None:
    # 偶数个取靠上的那一个，不取中间值：中间值可能是没人测到过的数
    # （10 与 11 的中间值是 10.5，四舍五入成 11 看着也像量到的，那就分不出来了）。
    assert LatencyStats.of([10, 11]).p50_ms == 11
    assert LatencyStats.of([10, 20, 30, 40]).p50_ms == 30
    assert LatencyStats.of([7]).p50_ms == 7


def test_no_samples_is_not_zero_milliseconds() -> None:
    stats = LatencyStats.of([])
    assert stats.count == 0
    assert stats.p50_ms == 0
    # 0 在这里只是空位，读的人要看的是 count——页面上也必须先看它。
    assert stats.count == 0


def test_a_run_without_completed_at_is_not_zero_seconds() -> None:
    wall, skipped = run_wall_stats(
        [
            _sample(run_id="a", duration_s=10.0),
            _sample(run_id="b", duration_s=20.0),
            _sample(run_id="c", duration_s=None),  # 还在跑
        ]
    )
    assert wall.count == 2
    assert wall.p50_ms == 20_000
    assert skipped == 1


# --- 读库：所有栏位取同一批运行 ------------------------------------------------


async def _insert(
    db: AsyncSession,
    *,
    run_id: str,
    scenario: str = "s",
    verdict: bool | None,
    minutes: int,
    status: str = "completed",
    report_body: dict[str, object] | None = None,
    cost: float = 0.0,
    llm_latency: int = 100,
    streaming: bool = False,
    tool_latency: int | None = None,
) -> None:
    started = BEGIN + timedelta(minutes=minutes)
    db.add(
        TestRun(
            id=run_id,
            scenario_id=scenario,
            agent_version="a@1",
            config_hash="h",
            status=status,
            verdict_ok=verdict,
            started_at=started,
            completed_at=started + timedelta(seconds=5),
            report_json=json.dumps(report_body) if report_body is not None else None,
        )
    )
    db.add(
        LlmCall(
            id=f"{run_id}-llm",
            test_run_id=run_id,
            seq=0,
            model="deepseek-chat",
            input_tokens=100,
            output_tokens=10,
            cost_usd=cost,
            latency_ms=llm_latency,
            streaming=streaming,
        )
    )
    if tool_latency is not None:
        db.add(
            ToolCall(
                id=f"{run_id}-tool",
                test_run_id=run_id,
                session_id=f"{run_id}-s0",
                session_sequence_id=f"seq-{run_id}",
                seq=1,
                tool_name="get_order",
                args_json="{}",
                arg_hash="x",
                fingerprint="y",
                latency_ms=tool_latency,
                ts=started,
            )
        )
    await db.flush()


async def test_every_column_covers_the_same_batch(db: AsyncSession, temp_db: str) -> None:
    # 三条运行，只看最近两条。结论、成本、延迟**必须都只算那两条**：
    # 一栏按最近 N 次算、另一栏按全量算的话，页面上两栏说的就不是一回事，
    # 而它们看起来同样正常。
    await _insert(
        db,
        run_id="old",
        verdict=False,
        minutes=0,
        cost=9.0,
        llm_latency=9_000,
        tool_latency=9_000,
        report_body=_report(missing=[{"detector": "loop"}]),
    )
    await _insert(db, run_id="new1", verdict=True, minutes=10, cost=1.0, llm_latency=100)
    await _insert(db, run_id="new2", verdict=False, minutes=20, cost=2.0, llm_latency=200)
    await db.commit()

    got = await report(limit=2)

    assert got.outcomes.considered == 2
    assert got.outcomes.failed == 1  # old 那条不在里面
    assert got.cost.cost_usd == 3.0  # 9.0 没有被算进来
    assert got.latency.model_calls.count == 2
    assert got.latency.model_calls.max_ms == 200
    assert "最近 2 次运行" in got.scope_text
    assert "实际取到 2 次" in got.scope_text


async def test_streaming_and_plain_model_calls_are_not_merged(
    db: AsyncSession, temp_db: str
) -> None:
    # 流式那一栏的"一次"是从请求发出到**最后一帧**，非流式是整个往返。
    # 两个数不是同一种耗时，合起来平均会得到一个没人定义过的值。
    await _insert(db, run_id="r1", verdict=True, minutes=0, llm_latency=80)
    await _insert(db, run_id="r2", verdict=True, minutes=10, llm_latency=5_000, streaming=True)
    await db.commit()

    got = await report(limit=10)
    assert got.latency.model_calls.count == 1
    assert got.latency.model_calls.p50_ms == 80
    assert got.latency.model_calls_streaming.count == 1
    assert got.latency.model_calls_streaming.p50_ms == 5_000


async def test_tool_latency_comes_from_tool_calls_not_model_calls(
    db: AsyncSession, temp_db: str
) -> None:
    await _insert(db, run_id="r1", verdict=True, minutes=0, llm_latency=80, tool_latency=40)
    await db.commit()

    got = await report(limit=10)
    assert got.latency.tool_calls.p50_ms == 40
    assert got.latency.run_wall.count == 1
    assert got.latency.run_wall.p50_ms == 5_000


async def test_an_empty_database_says_so_instead_of_reporting_zero(
    db: AsyncSession, temp_db: str
) -> None:
    got = await report(limit=10)
    assert got.outcomes.considered == 0
    assert got.outcomes.rate is None
    assert got.latency.model_calls.count == 0
    assert got.cost.cost_usd == 0.0
    assert "实际取到 0 次" in got.scope_text
