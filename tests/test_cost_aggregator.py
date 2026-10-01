"""成本聚合与预算预警（需求场景 4）。

这个模块的价值全在**跨运行**上：一次运行花了多少钱是明细，"这个月烧掉多少"
才是场景 4 的用户故事要问的。所以大部分测试都要先把若干次运行铺进库里。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from sandbox.cost.aggregator import (
    CostRollup,
    budget_status,
    by_agent,
    by_model,
    by_period,
    by_scenario,
    by_session,
    for_period,
    for_run,
    period_start,
)
from sandbox.db.models import LlmCall, Session, TestRun

# --- 造数据 -------------------------------------------------------------------


async def _run(
    db: AsyncSession,
    *,
    run_id: str,
    scenario: str = "s",
    agent: str = "a@1",
    started: datetime | None = None,
) -> TestRun:
    run = TestRun(
        id=run_id,
        scenario_id=scenario,
        agent_version=agent,
        config_hash="h",
        started_at=started or datetime(2026, 9, 30, 12, 0, tzinfo=UTC),
    )
    db.add(run)
    await db.flush()
    return run


async def _call(
    db: AsyncSession,
    *,
    run_id: str,
    model: str = "deepseek-chat",
    cost: float,
    inp: int = 100,
    out: int = 10,
    session_id: str | None = None,
    seq: int = 0,
    status: str = "ok",
) -> None:
    db.add(
        LlmCall(
            id=f"{run_id}-{seq}-{session_id or 'none'}",
            test_run_id=run_id,
            session_id=session_id,
            seq=seq,
            model=model,
            input_tokens=inp,
            output_tokens=out,
            cost_usd=cost,
            status=status,
        )
    )


async def _session(db: AsyncSession, run_id: str, ordinal: int) -> Session:
    sess = Session(
        id=f"{run_id}-s{ordinal}",
        test_run_id=run_id,
        session_sequence_id=f"seq-{run_id}",
        ordinal=ordinal,
    )
    db.add(sess)
    await db.flush()
    return sess


# --- 运行级 -------------------------------------------------------------------


async def test_run_cost_sums_the_detail_rows(db: AsyncSession, temp_db: str) -> None:
    await _run(db, run_id="r1")
    await _call(db, run_id="r1", cost=0.25, inp=1000, out=200, seq=0)
    await _call(db, run_id="r1", cost=0.75, inp=3000, out=400, seq=1)
    await db.commit()

    rollup = await for_run("r1")
    assert rollup.llm_calls == 2
    assert rollup.input_tokens == 4000
    assert rollup.output_tokens == 600
    assert rollup.total_tokens == 4600
    assert rollup.cost_usd == pytest.approx(1.0)


async def test_an_unknown_run_is_zero_not_an_error(db: AsyncSession, temp_db: str) -> None:
    # 空集合的和是 0，不是异常。"这个运行不存在"和"这个运行没花钱"在聚合的
    # 语境下是同一个答案，为它抛错只会让调用方到处写 try。
    assert await for_run("nope") == CostRollup()


async def test_runs_do_not_bleed_into_each_other(db: AsyncSession, temp_db: str) -> None:
    await _run(db, run_id="r1")
    await _run(db, run_id="r2")
    await _call(db, run_id="r1", cost=1.0, seq=0)
    await _call(db, run_id="r2", cost=9.0, seq=0)
    await db.commit()

    assert (await for_run("r1")).cost_usd == pytest.approx(1.0)
    assert (await for_run("r2")).cost_usd == pytest.approx(9.0)


# --- 会话级 -------------------------------------------------------------------


async def test_costs_split_by_session_ordinal(db: AsyncSession, temp_db: str) -> None:
    await _run(db, run_id="r1")
    s0 = await _session(db, "r1", 0)
    s1 = await _session(db, "r1", 1)
    await _call(db, run_id="r1", cost=1.0, session_id=s0.id, seq=0)
    await _call(db, run_id="r1", cost=2.0, session_id=s1.id, seq=1)
    await _call(db, run_id="r1", cost=4.0, session_id=s1.id, seq=2)
    await db.commit()

    per = await by_session("r1")
    assert per[0].cost_usd == pytest.approx(1.0)
    assert per[0].llm_calls == 1
    assert per[1].cost_usd == pytest.approx(6.0)  # 2.0 + 4.0
    assert per[1].llm_calls == 2


async def test_calls_outside_any_session_are_kept_not_dropped(
    db: AsyncSession, temp_db: str
) -> None:
    """没归到会话的调用要留下，归到 -1。

    丢掉它们的话，"各会话之和"和"运行总额"对不上，而那个差额看起来像是聚合
    算错了——查的人会去翻聚合代码，而真正的原因在别处。
    """
    await _run(db, run_id="r1")
    s0 = await _session(db, "r1", 0)
    await _call(db, run_id="r1", cost=1.0, session_id=s0.id, seq=0)
    await _call(db, run_id="r1", cost=3.0, session_id=None, seq=1)
    await db.commit()

    per = await by_session("r1")
    assert per[-1].cost_usd == pytest.approx(3.0)

    total = await for_run("r1")
    assert sum(r.cost_usd for r in per.values()) == pytest.approx(total.cost_usd)
    assert sum(r.llm_calls for r in per.values()) == total.llm_calls


async def test_session_breakdown_totals_match_the_run(db: AsyncSession, temp_db: str) -> None:
    # 上面那条测的是"有孤儿调用"的情形，这条测普通情形。两者都必须对得上。
    await _run(db, run_id="r1")
    s0 = await _session(db, "r1", 0)
    s1 = await _session(db, "r1", 1)
    for i, sid in enumerate([s0.id, s0.id, s1.id]):
        await _call(db, run_id="r1", cost=0.1 * (i + 1), session_id=sid, seq=i)
    await db.commit()

    per = await by_session("r1")
    assert set(per) == {0, 1}
    assert sum(r.cost_usd for r in per.values()) == pytest.approx(0.6)


# --- 跨运行的拆解 -------------------------------------------------------------


async def test_breakdown_by_scenario(db: AsyncSession, temp_db: str) -> None:
    for i, scenario in enumerate(["clean", "clean", "loop"]):
        await _run(db, run_id=f"r{i}", scenario=scenario)
        await _call(db, run_id=f"r{i}", cost=0.1, seq=0)
    await db.commit()

    by = await by_scenario()
    assert by["clean"].llm_calls == 2
    assert by["loop"].llm_calls == 1


async def test_breakdown_by_agent_version(db: AsyncSession, temp_db: str) -> None:
    await _run(db, run_id="r1", agent="react@1")
    await _run(db, run_id="r2", agent="react@2")
    await _call(db, run_id="r1", cost=1.0, seq=0)
    await _call(db, run_id="r2", cost=5.0, seq=0)
    await db.commit()

    by = await by_agent()
    assert by["react@1"].cost_usd == pytest.approx(1.0)
    assert by["react@2"].cost_usd == pytest.approx(5.0)


async def test_breakdown_by_model(db: AsyncSession, temp_db: str) -> None:
    await _run(db, run_id="r1")
    await _call(db, run_id="r1", model="deepseek-chat", cost=1.0, seq=0)
    await _call(db, run_id="r1", model="deepseek-reasoner", cost=7.0, seq=1)
    await db.commit()

    by = await by_model()
    assert by["deepseek-reasoner"].cost_usd == pytest.approx(7.0)


async def test_since_filters_out_older_runs(db: AsyncSession, temp_db: str) -> None:
    old = datetime(2026, 1, 1, tzinfo=UTC)
    await _run(db, run_id="old", started=old)
    await _run(db, run_id="new", started=datetime(2026, 9, 30, tzinfo=UTC))
    await _call(db, run_id="old", cost=1.0, seq=0)
    await _call(db, run_id="new", cost=2.0, seq=0)
    await db.commit()

    by = await by_scenario(since=datetime(2026, 6, 1, tzinfo=UTC))
    assert by["s"].llm_calls == 1
    assert by["s"].cost_usd == pytest.approx(2.0)


# --- 周期 ---------------------------------------------------------------------


def test_period_start_of_a_day_is_midnight() -> None:
    now = datetime(2026, 9, 30, 15, 45, 30, tzinfo=UTC)
    assert period_start("day", now) == datetime(2026, 9, 30, tzinfo=UTC)


def test_the_week_starts_on_monday() -> None:
    """ISO 周，周一为界。

    按周日切的话，周一早上看到的"本周花费"里混着上一个周末的账——而对"这周
    超支了吗"这个问题，两个答案不一样。
    """
    # 2026-09-30 是周三。
    now = datetime(2026, 9, 30, 15, 0, tzinfo=UTC)
    start = period_start("week", now)
    assert start == datetime(2026, 9, 28, tzinfo=UTC)  # 周一
    assert start.weekday() == 0


def test_period_start_of_a_month_is_the_first() -> None:
    now = datetime(2026, 9, 30, 15, 0, tzinfo=UTC)
    assert period_start("month", now) == datetime(2026, 9, 1, tzinfo=UTC)


def test_an_unknown_period_is_rejected() -> None:
    # 悄悄退回"天"的话，一个拼错的 "weekly" 会给出一个看起来正常的日预算数字。
    with pytest.raises(ValueError, match="未知的周期"):
        period_start("weekly", datetime.now(UTC))  # type: ignore[arg-type]


async def test_for_period_only_counts_the_current_window(
    db: AsyncSession, temp_db: str
) -> None:
    now = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
    await _run(db, run_id="yesterday", started=now - timedelta(days=1))
    await _run(db, run_id="today", started=now - timedelta(hours=1))
    await _call(db, run_id="yesterday", cost=5.0, seq=0)
    await _call(db, run_id="today", cost=2.0, seq=0)
    await db.commit()

    today = await for_period("day", now=now)
    assert today.cost_usd == pytest.approx(2.0)

    month = await for_period("month", now=now)
    assert month.cost_usd == pytest.approx(7.0)


async def test_a_run_is_bucketed_by_when_it_started(db: AsyncSession, temp_db: str) -> None:
    """跨零点的运行整个算在开始的那一天，不劈成两半。

    按 `LlmCall.ts` 归期的话，一次跑过零点的运行会一半算今天、一半算昨天，
    两边的数都对不上任何一次运行——而对不上账的账单比没有账单更糟。
    """
    now = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
    await _run(db, run_id="late", started=datetime(2026, 9, 29, 23, 50, tzinfo=UTC))
    # 这笔调用的时间戳落在第二天，但运行是 29 号起的。
    await _call(db, run_id="late", cost=3.0, seq=0)
    await db.commit()

    buckets = dict(await by_period("day", now=now, limit=7))
    assert buckets["2026-09-29"].cost_usd == pytest.approx(3.0)
    assert "2026-09-30" not in buckets


async def test_by_period_returns_newest_first(db: AsyncSession, temp_db: str) -> None:
    for day in (28, 29, 30):
        rid = f"r{day}"
        await _run(db, run_id=rid, started=datetime(2026, 9, day, 12, 0, tzinfo=UTC))
        await _call(db, run_id=rid, cost=float(day), seq=0)
    await db.commit()

    labels = [label for label, _ in await by_period("day", now=datetime(2026, 9, 30, tzinfo=UTC))]
    assert labels == ["2026-09-30", "2026-09-29", "2026-09-28"]


# --- 未定价 -------------------------------------------------------------------


async def test_unpriced_calls_are_counted_and_flagged(
    db: AsyncSession, temp_db: str
) -> None:
    """模型不在定价表里时，成本是**下界**，报告必须说出来。

    只印一个 `$0.00` 的话，读报告的人会以为这次运行不要钱。真实情况是我们
    一个字都没算——两者看起来完全一样，这是最容易被当成"正常"的错。
    """
    await _run(db, run_id="r1")
    await _call(db, run_id="r1", model="some-unknown-model", cost=0.0, seq=0)
    await _call(db, run_id="r1", model="deepseek-chat", cost=0.5, seq=1)
    await db.commit()

    rollup = await for_run("r1")
    assert rollup.unpriced_calls == 1
    assert rollup.is_lower_bound
    assert rollup.to_dict()["cost_is_lower_bound"] is True


async def test_calls_without_upstream_usage_are_counted_and_flagged(
    db: AsyncSession, temp_db: str
) -> None:
    """token 数是估的调用要被数出来，并且**不能**说成"下界"。

    两个标记各管一件事，合并的话两句话都会变成假的：

    - `is_lower_bound`：金额少算了（没有模型单价）；
    - `tokens_are_estimated`：token 数不是上游给的。

    估的数可能偏大也可能偏小，所以它不是下界——说成下界会让人以为"实际只会
    更贵"，而实际可能更便宜。
    """
    await _run(db, run_id="r1")
    await _call(db, run_id="r1", cost=1.0, seq=0)
    await _call(db, run_id="r1", cost=1.0, seq=1, status="usage_estimated")
    await db.commit()

    rollup = await for_run("r1")
    assert rollup.llm_calls == 2
    assert rollup.estimated_calls == 1
    assert rollup.tokens_are_estimated is True
    assert rollup.is_lower_bound is False, (
        "有单价就是有单价——「token 是估的」和「金额少算了」是两条独立的事实，"
        "把它们压成一个标记，报告就会说一句不成立的话"
    )
    assert rollup.to_dict()["tokens_are_estimated"] is True


async def test_the_estimate_flag_survives_grouping(db: AsyncSession, temp_db: str) -> None:
    """按会话拆解之后，估的那几次调用还要能被数出来——和各会话之和一起。

    拆解走的是另一条 SQL。漏了那一列的话，运行总数说"有 1 次是估的"，
    逐会话列出来却一处都看不出来，读的人会以为总额那行印错了。
    """
    await _run(db, run_id="r1")
    s0 = await _session(db, "r1", 0)
    await _call(db, run_id="r1", cost=1.0, session_id=s0.id, seq=0, status="usage_estimated")
    await _call(db, run_id="r1", cost=1.0, session_id=s0.id, seq=1)
    await db.commit()

    grouped = await by_session("r1")
    assert grouped[0].estimated_calls == 1


async def test_a_fully_priced_run_is_not_a_lower_bound(db: AsyncSession, temp_db: str) -> None:
    await _run(db, run_id="r1")
    await _call(db, run_id="r1", model="deepseek-chat", cost=0.5, seq=0)
    await db.commit()

    rollup = await for_run("r1")
    assert rollup.unpriced_calls == 0
    assert not rollup.is_lower_bound


async def test_the_unpriced_flag_survives_grouping(db: AsyncSession, temp_db: str) -> None:
    # 分组之后标记不能丢，否则"这个场景花得最多"的结论会把没算钱的调用
    # 混进排名里，而排名上看不出区别。
    await _run(db, run_id="r1", scenario="s1")
    await _call(db, run_id="r1", model="unknown-x", cost=0.0, seq=0)
    await db.commit()

    assert (await by_scenario())["s1"].is_lower_bound


# --- 预算 ---------------------------------------------------------------------


async def test_no_ceiling_configured_means_no_status(
    db: AsyncSession, temp_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """默认不设限。

    一个没配过的阈值一旦生效，就会在别人第一次跑的时候弹出一堆他不认识的告警
    ——而告警一多就会被整体忽略，包括真正要紧的那些。
    """
    from sandbox.config import settings

    monkeypatch.setattr(settings, "cost_budget_daily_usd", 0.0)
    monkeypatch.setattr(settings, "cost_budget_weekly_usd", 0.0)
    monkeypatch.setattr(settings, "cost_budget_monthly_usd", 0.0)

    assert await budget_status() == []


async def test_a_configured_ceiling_reports_spend_and_exceeded(
    db: AsyncSession, temp_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sandbox.config import settings

    monkeypatch.setattr(settings, "cost_budget_daily_usd", 1.0)
    monkeypatch.setattr(settings, "cost_budget_weekly_usd", 0.0)
    monkeypatch.setattr(settings, "cost_budget_monthly_usd", 0.0)

    now = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
    await _run(db, run_id="r1", started=now - timedelta(hours=1))
    await _call(db, run_id="r1", cost=1.5, seq=0)
    await db.commit()

    statuses = await budget_status(now=now)
    assert len(statuses) == 1
    assert statuses[0].period == "day"
    assert statuses[0].spent_usd == pytest.approx(1.5)
    assert statuses[0].exceeded
    assert statuses[0].ratio == pytest.approx(1.5)


async def test_exactly_on_the_ceiling_is_not_exceeded(
    db: AsyncSession, temp_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """正好花完不算超。

    判据是 `>` 而不是 `>=`：预算写 $1.00 的意思是"最多花 1 块"。把"正好用满"
    报成超支，会让人以为自己算错了账。
    """
    from sandbox.config import settings

    monkeypatch.setattr(settings, "cost_budget_daily_usd", 1.0)
    monkeypatch.setattr(settings, "cost_budget_weekly_usd", 0.0)
    monkeypatch.setattr(settings, "cost_budget_monthly_usd", 0.0)

    now = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
    await _run(db, run_id="r1", started=now - timedelta(hours=1))
    await _call(db, run_id="r1", cost=1.0, seq=0)
    await db.commit()

    statuses = await budget_status(now=now)
    assert not statuses[0].exceeded


async def test_all_three_periods_report_independently(
    db: AsyncSession, temp_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sandbox.config import settings

    monkeypatch.setattr(settings, "cost_budget_daily_usd", 10.0)
    monkeypatch.setattr(settings, "cost_budget_weekly_usd", 10.0)
    monkeypatch.setattr(settings, "cost_budget_monthly_usd", 10.0)

    now = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
    # 月初的一笔：算进月预算，不算今天/本周。
    await _run(db, run_id="r1", started=datetime(2026, 9, 2, tzinfo=UTC))
    await _call(db, run_id="r1", cost=12.0, seq=0)
    await db.commit()

    got = {b.period: b for b in await budget_status(now=now)}
    assert set(got) == {"day", "week", "month"}
    assert not got["day"].exceeded
    assert not got["week"].exceeded
    assert got["month"].exceeded
