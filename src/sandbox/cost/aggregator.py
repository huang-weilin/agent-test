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

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from sqlalchemy import case, func, literal, select
from sqlalchemy.ext.asyncio import AsyncSession

from sandbox.config import settings
from sandbox.db.engine import session_scope
from sandbox.db.models import LlmCall, Session, TestRun, ToolCall
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


async def _with_db[T](
    db: AsyncSession | None, fn: Callable[[AsyncSession], Awaitable[T]]
) -> T:
    """复用调用方的事务，没有就自己开一个。

    复用是必要的，不是省事：`by_session` 内部要连着查两次，各开一个事务的话，
    两次之间新落进来的调用会让"各会话之和"与"运行总额"对不上。

    **类型参数不是装饰**：原来两个位置都是 `Any`，于是九个查询函数全都顶着
    `no-any-return`——`-> CostRollup` 的函数返回一个 `Any`，类型检查器就再也
    帮不上忙了（这正是"mypy 跑起来"要抓的那一类）。`_T` 一上，它们各自
    对上了自己的返回类型，一个改动修掉九条。
    """
    if db is not None:
        return await fn(db)
    async with session_scope() as session:
        return await fn(session)


# --- 按工具（需求 3.5 / 3.7）--------------------------------------------------


@dataclass(frozen=True)
class ToolCost:
    """某个工具在一段范围内的账。

    **这里没有"这个工具花了多少钱"。** 工具调用本身不花 token——它是打到出口
    代理的一次 HTTP 请求；token 花在 LLM 调用上。硬把两者配成一对（"谁发起的
    这次调用，就算在谁头上"）会得到一个错的数：钱不在那次调用上，在**返回的
    内容**上——它进了上下文，之后每一次请求都要为它再付一遍。

    所以这个类里是三类东西，各有各的口径：

    * **数出来的**：调用次数、状态分布、返回体字节数、延迟；
    * **量出来的**：`context_growth_bytes`——这次调用之后第一次请求比之前那次
      大了多少字节。它是**观测**，可能为负（Agent 自己裁过上下文）；
    * **标着假设的**：`rent_bytes`——见 `by_tool` 的注释，它是一个条件数字，
      报告里必须连着假设一起印。
    """

    tool_name: str
    calls: int = 0
    #: 库里 `status` 的原值分布。**不在这里重新分类**：多出来的值原样出现，
    #: 而不是掉进一个"其他"里消失。
    by_status: dict[str, int] = field(default_factory=dict)
    #: 被沙箱自己的策略挡下的次数（ADR 003）。**和 `status="blocked"` 不是一回
    #: 事**：未声明的工具也是 blocked，但那是被测对象做错了事，不是策略在生效。
    blocked_by_proxy_calls: int = 0
    write_calls: int = 0
    #: Σ 返回体字节数（边界上量到的）。只统计量到的那几次。
    total_result_bytes: int = 0
    measured_size_calls: int = 0
    largest_result_bytes: int = 0
    latency_ms: int = 0
    #: 量到的上下文增量：Σ(这次调用之后第一次请求的字节数 − 之前最后一次请求的
    #: 字节数)。两次请求都在同一个会话里量到才算，**所以它可能为负**。
    context_growth_bytes: int = 0
    growth_measured_calls: int = 0
    #: 条件数字：Σ(返回体字节 × 之后同会话里还剩几次请求)。见 `by_tool`。
    rent_bytes: int = 0
    calls_with_followup: int = 0
    #: 之后一次请求都没有的调用数。它们的租金记 0——**0 是"看不见"，不是"免费"**，
    #: 所以要单独数出来，报告里得说。
    calls_without_followup: int = 0

    @property
    def unmeasured_size_calls(self) -> int:
        """没量到返回体字节数的调用数（老库的行，或者根本没出去）。"""
        return self.calls - self.measured_size_calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "calls": self.calls,
            "by_status": dict(self.by_status),
            "blocked_by_proxy_calls": self.blocked_by_proxy_calls,
            "write_calls": self.write_calls,
            "total_result_bytes": self.total_result_bytes,
            "measured_size_calls": self.measured_size_calls,
            "unmeasured_size_calls": self.unmeasured_size_calls,
            "largest_result_bytes": self.largest_result_bytes,
            "latency_ms": self.latency_ms,
            "context_growth_bytes": self.context_growth_bytes,
            "growth_measured_calls": self.growth_measured_calls,
            "rent_bytes": self.rent_bytes,
            "calls_with_followup": self.calls_with_followup,
            "calls_without_followup": self.calls_without_followup,
        }


@dataclass(frozen=True)
class ToolCostReport:
    tools: dict[str, ToolCost]
    llm_calls: int = 0
    #: 其中记下了请求体字节数的调用数。**密度只由这些行算**——老行没有请求体
    #: 字节数，把它们的分母（输入 token）算进来，密度会被稀释成一个没有含义的数。
    llm_calls_measured: int = 0
    #: 实测密度：Σ请求体字节 ÷ Σ输入token（只算量到过的行）。None = 一次都没量到，
    #: 于是租金只能按字节报，折算不出 token。
    bytes_per_token: float | None = None

    def tokens_for(self, byte_count: int) -> int | None:
        """把字节折成 token。量不到密度就返回 None——**不猜一个数**。"""
        if not self.bytes_per_token or self.bytes_per_token <= 0:
            return None
        return round(byte_count / self.bytes_per_token)

    @property
    def total_rent_bytes(self) -> int:
        return sum(t.rent_bytes for t in self.tools.values())

    @property
    def rent_tokens(self) -> int | None:
        """全部工具租金折成的 token（量不到密度就是 None）。"""
        return self.tokens_for(self.total_rent_bytes)

    def to_dict(self) -> dict[str, Any]:
        """给报告和界面用的形状。

        `tools` 是**列表且按租金从大到小**：报告要回答"哪几个工具在吃上下文"，
        而字典的顺序不承诺任何东西（前端排序还得自己知道按哪个字段排）。

        `rent_tokens` 由这里**注进去**，不是 `ToolCost` 自己算的：折算密度是整份
        报告一个数（见 `bytes_per_token`），让每个工具各自拿密度算一遍，等于承认
        它们可以用不同的密度——而那正是"同一个库里的数互相矛盾"的起点。
        """
        tools: list[dict[str, Any]] = []
        for tool in sorted(self.tools.values(), key=lambda t: -t.rent_bytes):
            item = tool.to_dict()
            item["rent_tokens"] = self.tokens_for(tool.rent_bytes)
            tools.append(item)
        return {
            "tools": tools,
            "llm_calls": self.llm_calls,
            "llm_calls_measured": self.llm_calls_measured,
            "bytes_per_token": (
                round(self.bytes_per_token, 3) if self.bytes_per_token else None
            ),
            "total_rent_bytes": self.total_rent_bytes,
            "rent_tokens": self.rent_tokens,
        }


async def by_tool(
    *,
    run_id: str | None = None,
    since: datetime | None = None,
    db: AsyncSession | None = None,
) -> ToolCostReport:
    """按工具拆：调了多少次、返回了多大、那些返回体在上下文里又待了多久。

    ## 为什么不是"这个工具的 token 成本"

    工具调用不花 token。想归集到 token，只能沿时间线把"产生这次调用的那次 LLM
    调用"配上去——而那是个**错的**近似：钱不在那次调用上，在返回的内容上，它
    进了上下文，之后每一次请求都要为它再付一遍。所以这里报的是**上下文占用**：

    * `context_growth_bytes` 是**量出来的**（前后两次请求体的字节差）；
    * `rent_bytes` 是**条件下的**：假设返回体在此后每一次请求里**原样重发**，
      它一共占多少字节 = Σ(返回体字节 × 之后同会话里还剩几次请求)。

    第二个数字的前提是"原样重发"。真实情况里 Agent 可能摘要、可能丢历史——
    所以它是**条件数字，不是观测**，报告里必须连着这句话一起印。两个数之差
    （实测增量 vs 租金）恰好是这个问题唯一诚实的答案：差额就是"它裁没裁"。

    ## 几条口径

    * **按 (运行, 会话) 分组**。`seq` 是运行内全局单调序号（LLM 调用与工具调用
      共用一条时间线），所以"之后还剩几次请求"是数出来的，不靠时间戳——这个
      项目的墙钟会被快照恢复和时间加速拨动。分组必须带会话：一次长时运行里
      会有多段会话，而上下文是**按会话**从零开始的。
    * **之后一次请求都没有的调用，租金记 0 并单独计数**（`calls_without_followup`）。
      0 是"看不见"，不是"免费"。
    * **密度只由量到过请求体字节的行算**（`LlmCall.request_bytes` 非空）。
      老行没有这一列，把它们算进来会让密度变成一个混合口径的数。
    * 未出去的调用（被挡下、参数不合法）没有返回体，**不参与**租金；
      但它们仍然是"调了多少次"的一部分，也在调用次数里。

    `run_id` 与 `since` 二选一：给 `run_id` 是单次运行的报告，给 `since` 是
    跨运行（哪些工具的返回体在长期烧钱），都不给就是全部。
    """

    async def _go(session: AsyncSession) -> ToolCostReport:
        llm_stmt = select(
            LlmCall.test_run_id,
            LlmCall.session_id,
            LlmCall.seq,
            LlmCall.request_bytes,
            LlmCall.input_tokens,
        )
        tool_stmt = select(
            ToolCall.test_run_id,
            ToolCall.session_id,
            ToolCall.seq,
            ToolCall.tool_name,
            ToolCall.status,
            ToolCall.is_write,
            ToolCall.blocked_by_proxy,
            ToolCall.result_bytes,
            ToolCall.latency_ms,
        )
        if run_id is not None:
            llm_stmt = llm_stmt.where(LlmCall.test_run_id == run_id)
            tool_stmt = tool_stmt.where(ToolCall.test_run_id == run_id)
        elif since is not None:
            llm_stmt = llm_stmt.join(TestRun, LlmCall.test_run_id == TestRun.id).where(
                TestRun.started_at >= since
            )
            tool_stmt = tool_stmt.join(TestRun, ToolCall.test_run_id == TestRun.id).where(
                TestRun.started_at >= since
            )

        # (运行, 会话) → 该会话里的请求时间线。只有这两列加得起来：
        # `seq` 是运行内全局序号，工具调用与 LLM 调用共用它（见 recorder.py）。
        timeline: dict[tuple[str, str | None], list[tuple[int, int | None, int]]] = (
            defaultdict(list)
        )
        measured_bytes = 0
        measured_tokens = 0
        llm_calls = 0
        llm_calls_measured = 0
        for rid, sid, seq, req_bytes, in_tokens in (await session.execute(llm_stmt)).all():
            timeline[(str(rid), sid)].append((int(seq), req_bytes, int(in_tokens or 0)))
            llm_calls += 1
            if req_bytes is not None:
                llm_calls_measured += 1
                measured_bytes += int(req_bytes)
                measured_tokens += int(in_tokens or 0)
        for rows in timeline.values():
            rows.sort(key=lambda row: row[0])

        acc: dict[str, ToolCost] = {}
        for row in (await session.execute(tool_stmt)).all():
            rid, sid, seq, name, status, is_write, blocked, result_bytes, latency = row
            key = (str(rid), sid)
            rows = timeline.get(key, [])
            seqs = [r[0] for r in rows]

            # 这次调用之后还剩几次请求：第一个 seq 大于它的请求开始，一直到本段
            # 会话结束。**这就是"原样重发"假设下它被带的次数。**
            after = bisect_right(seqs, int(seq))
            remaining = len(seqs) - after
            before = bisect_left(seqs, int(seq)) - 1

            size = int(result_bytes) if result_bytes is not None else 0
            cur = acc.get(
                str(name),
                ToolCost(tool_name=str(name), by_status={}),
            )
            status_dict = dict(cur.by_status)
            status_dict[str(status)] = status_dict.get(str(status), 0) + 1
            growth = 0
            growth_measured = cur.growth_measured_calls
            if 0 <= after < len(rows) and before >= 0:
                b_new = rows[after][1]
                b_old = rows[before][1]
                if b_new is not None and b_old is not None:
                    growth = int(b_new) - int(b_old)
                    growth_measured += 1
            acc[str(name)] = ToolCost(
                tool_name=str(name),
                calls=cur.calls + 1,
                by_status=status_dict,
                blocked_by_proxy_calls=cur.blocked_by_proxy_calls + (1 if blocked else 0),
                write_calls=cur.write_calls + (1 if is_write else 0),
                total_result_bytes=cur.total_result_bytes + size,
                measured_size_calls=cur.measured_size_calls
                + (1 if result_bytes is not None else 0),
                largest_result_bytes=max(cur.largest_result_bytes, size),
                latency_ms=cur.latency_ms + int(latency or 0),
                context_growth_bytes=cur.context_growth_bytes + growth,
                growth_measured_calls=growth_measured,
                rent_bytes=cur.rent_bytes + size * remaining,
                calls_with_followup=cur.calls_with_followup + (1 if remaining else 0),
                calls_without_followup=cur.calls_without_followup + (0 if remaining else 1),
            )

        return ToolCostReport(
            tools=acc,
            llm_calls=llm_calls,
            llm_calls_measured=llm_calls_measured,
            bytes_per_token=(measured_bytes / measured_tokens) if measured_tokens else None,
        )

    return await _with_db(db, _go)


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
