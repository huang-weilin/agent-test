"""成本聚合：按运行、会话、场景、Agent、模型、时间周期统计 Token 与花费。

需求场景 4 的用户故事是"云端 Agent 夜间持续运行，成本是预期的 3 倍"。那个故事
里成本监控的主语从**单次运行**变成了**长期预算**——一次运行花了多少钱是明细，
"这个月已经烧掉多少"才是他要盯的东西。所以这个模块存在的理由是跨运行的聚合，
而不是把 `TestRun.total_cost_usd` 再加一遍。

三条设计约束，都是"不做就会静默出错"的那类：

1. **一律从 `LlmCall` 明细重算，不信累计字段。** `TestRun` 上确实有
   `total_cost_usd`，但那是运行结束时写一次的快照。用它做长期聚合的话，任何一次
   漏结算（网关崩了、进程被熔断杀掉）都会永久留在账上，而且对不上明细——这种
   不一致极难发现，因为两边都是"看起来正常的一个数"。明细是唯一的真相源。

2. **Token 计数以网关为准，不看 Agent 上报的 usage。** 见 `db/models.py` 里
   `LlmCall` 的注释。需求原文第 1 步写的是"在 Trace 采集层捕获每次 LLM 调用的
   usage_metadata"，这里**故意不照做**：usage_metadata 来自被测 Agent 的进程，
   是被测对象能改的东西。用被测对象自报的账去算它的成本，等于让它自己写考卷。

3. **预算状态不是检测结论。** 见 `budget_status()`——月预算超了不是"这次运行的
   行为有缺陷"，混进 findings 会让一次完全干净的运行报 fail。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from sqlalchemy import case, func, literal, select
from sqlalchemy.ext.asyncio import AsyncSession

from sandbox.config import settings
from sandbox.db.engine import session_scope
from sandbox.db.models import LlmCall, Session, TestRun
from sandbox.gateway.pricing import known_models

Period = Literal["day", "week", "month"]


@dataclass(frozen=True)
class CostRollup:
    """一段范围内的 Token 与花费。可相加，所以能任意分组后再汇总。"""

    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    # 定价表里查不到的模型，其调用数。见 `is_lower_bound`。
    unpriced_calls: int = 0
    # Token 数**不是上游给的**、而是网关自己估的调用数。见 `tokens_are_estimated`。
    estimated_calls: int = 0

    def __add__(self, other: CostRollup) -> CostRollup:
        return CostRollup(
            llm_calls=self.llm_calls + other.llm_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            unpriced_calls=self.unpriced_calls + other.unpriced_calls,
            estimated_calls=self.estimated_calls + other.estimated_calls,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def is_lower_bound(self) -> bool:
        """金额是不是**下界**（还有没算进钱的部分）。

        只要有未定价的调用，这个数就少算了东西。把它当账单报出去，用户会在
        收到真实账单时才发现，而那时已经调了好几轮参数——所以报告必须把
        "这是下界"说出来，不能只用 `$0.0000` 打发。
        """
        return self.unpriced_calls > 0

    @property
    def tokens_are_estimated(self) -> bool:
        """这里的 Token 数是不是**有一部分不是上游给的**。

        和 `is_lower_bound` 是两回事，别合并：那个说的是"金额少算了"（没单价），
        这个说的是"token 数是估的"（没拿到 usage 帧）。估的数可能偏大也可能偏小，
        所以它不是下界——它是一条"这几个数字的来源不一样"的备注，报告里必须
        印出来，否则读的人会把估的数当成账单上的数。
        """
        return self.estimated_calls > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "unpriced_calls": self.unpriced_calls,
            "estimated_calls": self.estimated_calls,
            "cost_is_lower_bound": self.is_lower_bound,
            "tokens_are_estimated": self.tokens_are_estimated,
        }


def _rollup(row: Any) -> CostRollup:
    return CostRollup(
        llm_calls=int(row[0]),
        input_tokens=int(row[1]),
        output_tokens=int(row[2]),
        cache_read_tokens=int(row[3]),
        cache_write_tokens=int(row[4]),
        cost_usd=float(row[5]),
        unpriced_calls=int(row[6]),
        estimated_calls=int(row[7]),
    )


def _sum_columns() -> list[Any]:
    """聚合列。顺序与 `_rollup` 一一对应，改一处必须改另一处。

    "未定价"是在**查询时**按当前定价表判定的，不是落库时存下来的标志。
    这样补一条新模型的单价之后，历史数据立刻跟着被重新定价——而"我们有没有
    这个模型的单价"本来就该是当下的属性，不是历史事实。

    "估的"反过来，是**落库时**写下的（`LlmCall.status`）：它是"那一次调用当时
    拿没拿到上游的用量"这个历史事实，事后改不了，也不该跟着定价表变。
    """
    priced = known_models()
    unpriced = case((LlmCall.model.in_(priced), 0), else_=1) if priced else literal(1)
    estimated = case((LlmCall.status == "usage_estimated", 1), else_=0)
    return [
        func.count(LlmCall.id),
        func.coalesce(func.sum(LlmCall.input_tokens), 0),
        func.coalesce(func.sum(LlmCall.output_tokens), 0),
        func.coalesce(func.sum(LlmCall.cache_read_tokens), 0),
        func.coalesce(func.sum(LlmCall.cache_write_tokens), 0),
        func.coalesce(func.sum(LlmCall.cost_usd), 0.0),
        func.coalesce(func.sum(unpriced), 0),
        func.coalesce(func.sum(estimated), 0),
    ]


# --- 单个运行 / 会话 ----------------------------------------------------------


async def for_run(run_id: str, *, db: AsyncSession | None = None) -> CostRollup:
    """一次运行的花费。从明细重算。"""

    async def _go(session: AsyncSession) -> CostRollup:
        stmt = select(*_sum_columns()).where(LlmCall.test_run_id == run_id)
        return _rollup((await session.execute(stmt)).one())

    return await _with_db(db, _go)


async def for_runs(
    run_ids: list[str], *, db: AsyncSession | None = None
) -> dict[str, CostRollup]:
    """多个运行各自的花费，一次查完。

    存在的理由是**不要 N+1**：运行列表要显示每一行的花费，逐行调 `for_run`
    就是几十次往返。但也不能因此退回 `TestRun.total_cost_usd`——那是运行结束
    时写一次的快照，用它做列表等于让列表和明细各说各话。分组聚合既是一次
    查询，又仍然是从明细算的。

    列表里没出现的运行 id 就是没有调用记录，调用方按"零"处理。
    """

    async def _go(session: AsyncSession) -> dict[str, CostRollup]:
        if not run_ids:
            return {}
        stmt = (
            select(LlmCall.test_run_id, *_sum_columns())
            .where(LlmCall.test_run_id.in_(run_ids))
            .group_by(LlmCall.test_run_id)
        )
        return {str(row[0]): _rollup(row[1:]) for row in (await session.execute(stmt)).all()}

    return await _with_db(db, _go)


async def by_session(run_id: str, *, db: AsyncSession | None = None) -> dict[int, CostRollup]:
    """按**会话序号**拆解一次运行的花费。

    键取序号而不是 `session_id`，与检测器一致：报告和其他地方谈的都是"第几个
    会话"，用 uuid 做键会让人对不上号。

    没有关联到会话的调用（会话还没开始就有请求进来）归到 `-1`。**不丢弃它们**：
    丢掉的话各会话之和对不上运行总额，而这种差额看起来像是聚合算错了。
    """

    async def _go(session: AsyncSession) -> dict[int, CostRollup]:
        stmt = (
            select(Session.ordinal, *_sum_columns())
            .join(Session, LlmCall.session_id == Session.id)
            .where(LlmCall.test_run_id == run_id)
            .group_by(Session.ordinal)
        )
        grouped = {int(row[0]): _rollup(row[1:]) for row in (await session.execute(stmt)).all()}

        # 会话之外的调用：单独算一次差，而不是再查一遍全部。
        total = await for_run(run_id, db=session)
        tracked = CostRollup()
        for rollup in grouped.values():
            tracked = tracked + rollup
        if total.llm_calls != tracked.llm_calls:
            grouped[-1] = CostRollup(
                llm_calls=total.llm_calls - tracked.llm_calls,
                input_tokens=total.input_tokens - tracked.input_tokens,
                output_tokens=total.output_tokens - tracked.output_tokens,
                cache_read_tokens=total.cache_read_tokens - tracked.cache_read_tokens,
                cache_write_tokens=total.cache_write_tokens - tracked.cache_write_tokens,
                cost_usd=total.cost_usd - tracked.cost_usd,
                unpriced_calls=total.unpriced_calls - tracked.unpriced_calls,
                estimated_calls=total.estimated_calls - tracked.estimated_calls,
            )
        return grouped

    return await _with_db(db, _go)


# --- 跨运行的拆解 -------------------------------------------------------------


async def by_scenario(
    *, since: datetime | None = None, db: AsyncSession | None = None
) -> dict[str, CostRollup]:
    """按场景拆解。哪些用例最烧钱——优化顺序就是它。"""

    async def _go(session: AsyncSession) -> dict[str, CostRollup]:
        stmt = (
            select(TestRun.scenario_id, *_sum_columns())
            .join(LlmCall, LlmCall.test_run_id == TestRun.id)
            .group_by(TestRun.scenario_id)
            .order_by(func.sum(LlmCall.cost_usd).desc())
        )
        return await _grouped(session, stmt, since)

    return await _with_db(db, _go)


async def by_agent(
    *, since: datetime | None = None, db: AsyncSession | None = None
) -> dict[str, CostRollup]:
    """按 Agent 版本拆解。回归时"新版本更贵了吗"看这个。"""

    async def _go(session: AsyncSession) -> dict[str, CostRollup]:
        stmt = (
            select(TestRun.agent_version, *_sum_columns())
            .join(LlmCall, LlmCall.test_run_id == TestRun.id)
            .group_by(TestRun.agent_version)
            .order_by(func.sum(LlmCall.cost_usd).desc())
        )
        return await _grouped(session, stmt, since)

    return await _with_db(db, _go)


async def by_model(
    *, since: datetime | None = None, db: AsyncSession | None = None
) -> dict[str, CostRollup]:
    """按模型拆解。换模型省没省钱，这里直接看得到。"""

    async def _go(session: AsyncSession) -> dict[str, CostRollup]:
        stmt = (
            select(LlmCall.model, *_sum_columns())
            .join(TestRun, LlmCall.test_run_id == TestRun.id)
            .group_by(LlmCall.model)
            .order_by(func.sum(LlmCall.cost_usd).desc())
        )
        return await _grouped(session, stmt, since)

    return await _with_db(db, _go)


async def _grouped(
    session: AsyncSession, stmt: Any, since: datetime | None
) -> dict[str, CostRollup]:
    if since is not None:
        stmt = stmt.where(TestRun.started_at >= since)
    return {str(row[0]): _rollup(row[1:]) for row in (await session.execute(stmt)).all()}


# --- 按时间周期 ---------------------------------------------------------------


def period_start(period: Period, now: datetime) -> datetime:
    """当前周期从哪一刻开始（UTC）。

    week 从**周一**起算，不是周日：ISO 周。需求写的是"周预算"，而中国的工作周
    从周一起算——按周日切的话，周一早上看到的"本周花费"里混着上一个周末的账。

    month 按自然月，不做"30 天滚动窗口"。两者对"这个月花超了吗"这个问题给出的
    答案不一样，而用户问的显然是自然月。
    """
    if period == "day":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "week":
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight - timedelta(days=midnight.weekday())
    if period == "month":
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    raise ValueError(f"未知的周期 {period!r}，只支持 day / week / month")


async def for_period(
    period: Period,
    *,
    now: datetime | None = None,
    db: AsyncSession | None = None,
) -> CostRollup:
    """本周期内花掉的钱。按 `TestRun.started_at` 归期。"""

    async def _go(session: AsyncSession) -> CostRollup:
        moment = now or datetime.now(UTC)
        stmt = (
            select(*_sum_columns())
            .join(TestRun, LlmCall.test_run_id == TestRun.id)
            .where(TestRun.started_at >= period_start(period, moment))
        )
        return _rollup((await session.execute(stmt)).one())

    return await _with_db(db, _go)


async def by_period(
    period: Period,
    *,
    limit: int = 30,
    now: datetime | None = None,
    db: AsyncSession | None = None,
) -> list[tuple[str, CostRollup]]:
    """按天/周/月分组的历史花费，从近到远。趋势看这个。

    分组在 Python 里做而不是 SQL：跨 SQLite/PostgreSQL 的日期截断函数写法不同
    （`strftime` vs `date_trunc`），而这里的数据量是"一台机器跑过的测试运行"，
    拉回内存分组完全够用。为省这点内存把两种方言都写一遍不划算。

    归期用 `TestRun.started_at` 而不是 `LlmCall.ts`：一次运行应当整个落在它开始
    的那个周期里，否则跨零点的运行会被劈成两半，两边的数都对不上任何一次运行。
    """
    moment = now or datetime.now(UTC)
    start = period_start(period, moment)

    async def _go(session: AsyncSession) -> list[tuple[str, CostRollup]]:
        stmt = (
            select(TestRun.started_at, *_sum_columns())
            .join(LlmCall, LlmCall.test_run_id == TestRun.id)
            .where(TestRun.started_at >= start - _lookback(period, limit))
            .group_by(TestRun.id, TestRun.started_at)
        )
        buckets: dict[str, CostRollup] = defaultdict(CostRollup)
        for row in (await session.execute(stmt)).all():
            key = _bucket_label(period, row[0])
            buckets[key] = buckets[key] + _rollup(row[1:])
        return sorted(buckets.items(), key=lambda kv: kv[0], reverse=True)

    return await _with_db(db, _go)


def _lookback(period: Period, limit: int) -> timedelta:
    # 往回拉足够多的天，保证 limit 个桶都能填满。周期越长，一个桶要的天数越多。
    per_bucket = {"day": 1, "week": 7, "month": 31}[period]
    return timedelta(days=per_bucket * max(limit, 1))


def _bucket_label(period: Period, moment: datetime) -> str:
    if period == "day":
        return moment.strftime("%Y-%m-%d")
    if period == "week":
        iso = moment.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    return moment.strftime("%Y-%m")


async def _with_db(db: AsyncSession | None, fn: Any) -> Any:
    """复用调用方的事务，没有就自己开一个。

    复用是必要的，不是省事：`by_session` 内部要连着查两次，各开一个事务的话，
    两次之间新落进来的调用会让"各会话之和"与"运行总额"对不上。
    """
    if db is not None:
        return await fn(db)
    async with session_scope() as session:
        return await fn(session)


# --- 预算 ---------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetStatus:
    period: Period
    spent_usd: float
    ceiling_usd: float

    @property
    def exceeded(self) -> bool:
        return self.spent_usd > self.ceiling_usd

    @property
    def ratio(self) -> float:
        return self.spent_usd / self.ceiling_usd if self.ceiling_usd > 0 else 0.0


async def budget_status(
    *,
    now: datetime | None = None,
    db: AsyncSession | None = None,
) -> list[BudgetStatus]:
    """日/周/月预算状态。阈值在 `config` 里，0 表示不设限。

    **这个结果不进 findings，这是刻意的。** findings 回答的是"这次运行的行为有
    没有缺陷"；"本月超支"和被测 Agent 这次做没做错事没有任何关系。混进去的后果
    是一次完全干净的运行因为上个月跑太多而报 fail——正是这个项目一直在防的那类
    范畴错误：把两件不同的事压成同一个结论，读报告的人无法分辨。

    所以它单独成一节，和"目标系统侧真实扣款"一样，是**独立于检测结论的另一条
    事实**，只陈述，不参与通过与否的判定。
    """
    ceilings: dict[Period, float] = {
        "day": settings.cost_budget_daily_usd,
        "week": settings.cost_budget_weekly_usd,
        "month": settings.cost_budget_monthly_usd,
    }

    async def _go(session: AsyncSession) -> list[BudgetStatus]:
        out: list[BudgetStatus] = []
        for period, ceiling in ceilings.items():
            if ceiling <= 0:
                continue
            rollup = await for_period(period, now=now, db=session)
            out.append(
                BudgetStatus(period=period, spent_usd=rollup.cost_usd, ceiling_usd=ceiling)
            )
        return out

    return await _with_db(db, _go)
