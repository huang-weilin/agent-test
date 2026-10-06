"""界面要读的东西，都在这里查。

单独成模块是为了让路由只做 HTTP 的事（表单、重定向、状态码），而"历史运行
长什么样"是一个可以脱离 HTTP 测试的问题。

**成本一律现算，不问 `TestRun.total_cost_usd`。** 那个字段是运行结束时写一次
的快照，列表用它的话，列表和明细页会各说各话——而两边看起来都很正常。
见 `cost/aggregator.py` 的模块注释。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import func, select

from sandbox.cost import aggregator as cost
from sandbox.db.engine import session_scope
from sandbox.db.models import MemorySnapshot, Session, TestRun, ToolCall


@dataclass(frozen=True)
class RunRow:
    """运行列表里的一行。"""

    id: str
    scenario_id: str
    agent_version: str
    status: str
    # None = **结论未留存**（加这个字段之前跑的运行），不是"不通过"。
    verdict_ok: bool | None
    started_at: datetime
    completed_at: datetime | None
    cost: cost.CostRollup

    @property
    def finished(self) -> bool:
        return self.completed_at is not None


async def recent_runs(*, limit: int = 50, scenario_id: str | None = None) -> list[RunRow]:
    """最近的运行，从新到旧。"""
    async with session_scope() as db:
        stmt = select(TestRun).order_by(TestRun.started_at.desc()).limit(limit)
        if scenario_id is not None:
            stmt = stmt.where(TestRun.scenario_id == scenario_id)
        runs = list((await db.execute(stmt)).scalars())

        rollups = await cost.for_runs([r.id for r in runs], db=db)

    return [
        RunRow(
            id=r.id,
            scenario_id=r.scenario_id,
            agent_version=r.agent_version,
            status=r.status,
            verdict_ok=r.verdict_ok,
            started_at=r.started_at,
            completed_at=r.completed_at,
            cost=rollups.get(r.id, cost.CostRollup()),
        )
        for r in runs
    ]


async def get_run(run_id: str) -> TestRun | None:
    async with session_scope() as db:
        return await db.get(TestRun, run_id)


async def report_rows(*, limit: int = 50, scenario_id: str | None = None) -> list[dict[str, Any]]:
    """报告列表的行：结论 + 问题数 + 花费。

    `problems` 是这份报告的失败条目数（执行出错 / 前提不成立 / 计费不符 /
    预期没命中各算一条）。`None` = 那份报告没留存，界面显示成「未留存」。
    """
    async with session_scope() as db:
        stmt = select(TestRun).order_by(TestRun.started_at.desc()).limit(limit)
        if scenario_id is not None:
            stmt = stmt.where(TestRun.scenario_id == scenario_id)
        runs = list((await db.execute(stmt)).scalars())
        rollups = await cost.for_runs([r.id for r in runs], db=db)
    rows: list[dict[str, Any]] = []
    for r in runs:
        problems: int | None = None
        if r.report_json:
            try:
                rep = json.loads(r.report_json)
            except ValueError:
                rep = None
            if rep is not None:
                problems = 0
                if rep.get("error") is not None:
                    problems += 1
                if not rep.get("premises_held", True):
                    problems += 1
                if not rep.get("charges_ok", True):
                    problems += 1
                expectation = rep.get("expectation")
                if isinstance(expectation, dict):
                    problems += int(bool(expectation.get("missing")))
                    problems += int(bool(expectation.get("unexpected")))
        rows.append({
            "id": r.id,
            "scenario_id": r.scenario_id,
            "verdict_ok": r.verdict_ok,
            "started_at": r.started_at,
            "cost": rollups.get(r.id, cost.CostRollup()),
            "problems": problems,
        })
    return rows


def parse_report(run: TestRun) -> dict[str, Any] | None:
    """取出运行时存下的结论快照。None = 结论未留存。

    历史运行（在本字段存在之前跑的）没有这份快照，返回 None。**界面必须把它
    显示成"未留存"，而不是渲染成一份空的失败报告**——后者看起来和"跑了但
    什么都没测到"一模一样，而那是完全不同的两件事。
    """
    if not run.report_json:
        return None
    # `json.loads` 的返回是 Any；标注一下，别让 Any 从签名里漏出去。
    parsed: dict[str, Any] = json.loads(run.report_json)
    return parsed


def _shorten_args(args_json: str) -> str:
    """参数摘要。审计表是拿来**扫**的，一整块 JSON 会把表撑到没法看。

    截断了就带上省略号——不然读者会以为参数本来就长这样，
    而"参数里少了一个字段"和"参数被截断显示"是两回事。
    """
    if len(args_json) <= 140:
        return args_json
    return args_json[:140] + "…"


@dataclass(frozen=True)
class ToolCallRow:
    seq: int
    session_ordinal: int
    tool_name: str
    is_write: bool
    idempotency_key: str | None
    status: str
    blocked_by_proxy: bool
    replay_reason: str | None
    args_json: str
    error: str | None
    latency_ms: int
    #: 看门狗为这一次调用花掉的时间。**NULL 与 0 是两件事**：NULL = 这次调用没
    #: 真的出到目标系统，看门狗没跑过（被边界拦下的那几支，见 `proxy/app.py`
    #: 的 `_settle`）；0 = 跑了，快到一毫秒都量不出来。老库里的行也是 NULL
    #: ——那时没量过，补不出来。报告里这两种要分开显示。
    watchdog_ms: int | None
    ts: datetime

    @property
    def args_short(self) -> str:
        return _shorten_args(self.args_json)


async def tool_calls_for_run(run_id: str) -> list[ToolCallRow]:
    """工具调用审计。**这一节是需求里的"工具调用审计"模块。**"""
    async with session_scope() as db:
        stmt = (
            select(ToolCall, Session.ordinal)
            .join(Session, ToolCall.session_id == Session.id, isouter=True)
            .where(ToolCall.test_run_id == run_id)
            .order_by(ToolCall.seq)
        )
        return [
            ToolCallRow(
                seq=call.seq,
                # 关联不到会话时给 -1，与 cost.by_session 的约定一致：
                # 丢掉这些行会让审计表的行数对不上运行的工具调用总数。
                session_ordinal=ordinal if ordinal is not None else -1,
                tool_name=call.tool_name,
                is_write=call.is_write,
                idempotency_key=call.idempotency_key,
                status=call.status,
                blocked_by_proxy=call.blocked_by_proxy,
                replay_reason=call.replay_reason,
                args_json=call.args_json,
                error=call.error,
                latency_ms=call.latency_ms,
                watchdog_ms=call.watchdog_ms,
                ts=call.ts,
            )
            for call, ordinal in (await db.execute(stmt)).all()
        ]


@dataclass(frozen=True)
class SnapshotRow:
    id: str
    taken_at_ordinal: int
    taken_at_seq: int
    adapter: str
    digest: str
    entry_count: int
    size_bytes: int
    taken_at: datetime


async def snapshots_for_sequence(session_sequence_id: str | None) -> list[SnapshotRow]:
    """记忆快照。**记忆视图那一节。**

    按会话序列查而不是按运行：快照属于会话序列，这正是跨会话归因的依据
    （回滚到哪儿决定了 Agent 本可以知道什么）。同一个序列下可能不止一条
    运行，但正常路径是一次运行一个新序列。
    """
    if not session_sequence_id:
        return []
    async with session_scope() as db:
        stmt = (
            select(MemorySnapshot)
            .where(MemorySnapshot.session_sequence_id == session_sequence_id)
            .order_by(MemorySnapshot.taken_at_ordinal, MemorySnapshot.taken_at_seq)
        )
        return [
            SnapshotRow(
                id=s.id,
                taken_at_ordinal=s.taken_at_ordinal,
                taken_at_seq=s.taken_at_seq,
                adapter=s.adapter,
                digest=s.digest,
                entry_count=s.entry_count,
                size_bytes=s.size_bytes,
                taken_at=s.taken_at,
            )
            for s in (await db.execute(stmt)).scalars()
        ]


# --- 记忆视图（需求 3.4 / 界面模块「记忆视图」） -----------------------------
#
# 这一节要回答的问题只有一个：**这个会话本可以知道什么。**
#
# 记忆在 Agent 进程内，边界拦不到它（见 memory/snapshot.py 的模块注释）。
# 所以这里能显示的不是"Agent 记住了什么"——那看不见——而是**我们的操纵
# 让它的视野从哪里开始**。判据是 `Session.memory_horizon_ordinal`：
# 自然延续时它等于 `ordinal - 1`（记得此前发生的一切），从旧快照恢复时
# 会比这小，**缺的那一段就是 Agent 不可能知道的部分**。


@dataclass(frozen=True)
class SessionRow:
    id: str
    ordinal: int
    memory_horizon_ordinal: int
    restored_from_snapshot_id: str | None
    status: str
    started_at: datetime
    ended_at: datetime | None

    @property
    def expected_horizon(self) -> int:
        """自然延续的话，这个会话应当记得的最后一个序号。

        序号 0 的会话之前没有记忆，所以是 -1 而不是 0——保持一致，
        不然 `missing_ordinals` 会把"第一个会话"报成缺了一段。
        """
        return self.ordinal - 1

    @property
    def missing_ordinals(self) -> list[int]:
        """本会话**不可能知道**的那些序号。

        空列表 = 自然延续，Agent 记得此前一切。非空 = 我们回滚过记忆，
        落在这一段里的历史调用，Agent 不可能知道，重复的责任在恢复机制
        而不在 Agent。
        """
        if self.memory_horizon_ordinal >= self.expected_horizon:
            return []
        return list(range(self.memory_horizon_ordinal + 1, self.expected_horizon + 1))

    @property
    def was_rolled_back(self) -> bool:
        """是否从快照恢复过。

        注意判据是"有没有恢复动作"，不是"缺不缺序号"——恢复到一个
        **同一个**快照也可能缺序号，反过来恢复后视野恰好接上也是可能的。
        两者分开算，页面上才能说清楚是哪一种。
        """
        return self.restored_from_snapshot_id is not None


async def sessions_for_sequence(session_sequence_id: str | None) -> list[SessionRow]:
    if not session_sequence_id:
        return []
    async with session_scope() as db:
        stmt = (
            select(Session)
            .where(Session.session_sequence_id == session_sequence_id)
            .order_by(Session.ordinal)
        )
        return [
            SessionRow(
                id=s.id,
                ordinal=s.ordinal,
                memory_horizon_ordinal=s.memory_horizon_ordinal,
                restored_from_snapshot_id=s.restored_from_snapshot_id,
                status=s.status,
                started_at=s.started_at,
                ended_at=s.ended_at,
            )
            for s in (await db.execute(stmt)).scalars()
        ]


@dataclass(frozen=True)
class SequenceRow:
    """记忆视图列表页的一行。一个会话序列 = 一次跨会话实验。"""

    session_sequence_id: str
    scenario_id: str
    run_id: str
    started_at: datetime
    session_count: int
    snapshot_count: int
    rolled_back_sessions: int
    missing_ordinals: int


async def recent_sequences(*, limit: int = 30) -> list[SequenceRow]:
    """最近的会话序列。

    以 `TestRun` 为入口而不是以 `Session`：序列是运行的产物，用户是从
    "我跑过哪几次"找到它的。没有会话的失败运行不进这个列表——它们没有
    记忆视图可看。
    """
    async with session_scope() as db:
        runs = list(
            (
                await db.execute(
                    select(TestRun)
                    .where(TestRun.session_sequence_id.is_not(None))
                    .order_by(TestRun.started_at.desc())
                    .limit(limit)
                )
            ).scalars()
        )
        sequences = [r.session_sequence_id for r in runs if r.session_sequence_id]
        if not sequences:
            return []

        sessions: dict[str, list[Session]] = {}
        for session in (
            await db.execute(select(Session).where(Session.session_sequence_id.in_(sequences)))
        ).scalars():
            sessions.setdefault(session.session_sequence_id, []).append(session)

        snap_counts: dict[str, int] = {}
        for seq_id, count in (
            await db.execute(
                select(MemorySnapshot.session_sequence_id, func.count())
                .where(MemorySnapshot.session_sequence_id.in_(sequences))
                .group_by(MemorySnapshot.session_sequence_id)
            )
        ).all():
            snap_counts[seq_id] = count

    out: list[SequenceRow] = []
    for run in runs:
        seq = run.session_sequence_id
        if not seq:
            continue
        rows = [
            SessionRow(
                id=s.id,
                ordinal=s.ordinal,
                memory_horizon_ordinal=s.memory_horizon_ordinal,
                restored_from_snapshot_id=s.restored_from_snapshot_id,
                status=s.status,
                started_at=s.started_at,
                ended_at=s.ended_at,
            )
            for s in sessions.get(seq, [])
        ]
        out.append(
            SequenceRow(
                session_sequence_id=seq,
                scenario_id=run.scenario_id,
                run_id=run.id,
                started_at=run.started_at,
                session_count=len(rows),
                snapshot_count=snap_counts.get(seq, 0),
                rolled_back_sessions=sum(1 for r in rows if r.was_rolled_back),
                missing_ordinals=sum(len(r.missing_ordinals) for r in rows),
            )
        )
    return out


# --- 审计日志（需求 3.2 / 界面模块「审计日志」） -----------------------------
#
# 单次运行的审计在运行详情页里（`tool_calls_for_run`）。这一节是**跨运行的**
# 那一份：需求 3.2 的"调用拦截"说的是拦截 Agent 的每一次工具调用并记录
# 输入输出参数，"重放审计"说的是记录重试次数和重放路径。按运行看只能看一次
# 运行的，而"这个工具是不是老被拒绝"这种问题得横着看。


@dataclass(frozen=True)
class AuditRow:
    seq: int
    run_id: str
    scenario_id: str
    agent_version: str
    session_ordinal: int
    tool_name: str
    is_write: bool
    idempotency_key: str | None
    status: str
    blocked_by_proxy: bool
    replay_reason: str | None
    args_json: str
    error: str | None
    latency_ms: int
    ts: datetime

    @property
    def args_short(self) -> str:
        return _shorten_args(self.args_json)


#: 审计日志一页最多取多少行。**超了必须在页面上说**——静默截断会让
#: "这个工具从来没被拒绝过"变成一句基于残页的结论。
AUDIT_PAGE = 200


async def audit_log(
    *,
    scenario_id: str | None = None,
    tool_name: str | None = None,
    writes_only: bool = False,
    blocked_only: bool = False,
    limit: int = AUDIT_PAGE,
) -> tuple[list[AuditRow], int]:
    """跨运行的工具调用审计。返回 (这一页, 符合条件的总行数)。

    总数一起返回是为了让页面能说"显示 200 / 共 1734 条"。只给一页而不给
    总数的话，用户没法知道下面被截掉了多少。
    """
    async with session_scope() as db:
        joins = (
            select(ToolCall, TestRun, Session.ordinal)
            .join(TestRun, ToolCall.test_run_id == TestRun.id)
            .join(Session, ToolCall.session_id == Session.id, isouter=True)
        )
        counted = (
            select(func.count())
            .select_from(ToolCall)
            .join(TestRun, ToolCall.test_run_id == TestRun.id)
        )

        conditions = []
        if scenario_id:
            conditions.append(TestRun.scenario_id == scenario_id)
        if tool_name:
            conditions.append(ToolCall.tool_name == tool_name)
        if writes_only:
            conditions.append(ToolCall.is_write.is_(True))
        if blocked_only:
            conditions.append(ToolCall.blocked_by_proxy.is_(True))

        for cond in conditions:
            joins = joins.where(cond)
            counted = counted.where(cond)

        total = (await db.execute(counted)).scalar_one()
        rows = (
            await db.execute(joins.order_by(ToolCall.ts.desc()).limit(limit))
        ).all()

    return (
        [
            AuditRow(
                seq=call.seq,
                run_id=run.id,
                scenario_id=run.scenario_id,
                agent_version=run.agent_version,
                session_ordinal=ordinal if ordinal is not None else -1,
                tool_name=call.tool_name,
                is_write=call.is_write,
                idempotency_key=call.idempotency_key,
                status=call.status,
                blocked_by_proxy=call.blocked_by_proxy,
                replay_reason=call.replay_reason,
                args_json=call.args_json,
                error=call.error,
                latency_ms=call.latency_ms,
                ts=call.ts,
            )
            for call, run, ordinal in rows
        ],
        total,
    )


async def tool_names_seen() -> list[str]:
    """审计筛选下拉里的工具名。从实际记录里取，不从工具声明里取——
    声明过的工具可能一次都没被调用，而筛选一个永远为空的选项是白费力气。"""
    async with session_scope() as db:
        rows = (
            await db.execute(
                select(ToolCall.tool_name).distinct().order_by(ToolCall.tool_name)
            )
        ).scalars()
        return list(rows)


async def scenarios_seen() -> list[str]:
    """审计筛选下拉里的场景名。同样从运行记录里取而非从磁盘上的场景文件取：
    场景文件可能已经被删掉，而它跑出来的调用仍留在审计日志里。"""
    async with session_scope() as db:
        rows = (
            await db.execute(
                select(TestRun.scenario_id).distinct().order_by(TestRun.scenario_id)
            )
        ).scalars()
        return list(rows)


async def scenario_for_sequence(session_sequence_id: str) -> str | None:
    """这个会话序列是哪个场景跑出来的。

    正常路径下一条运行一个序列，所以取最近那条运行就够。真出现一个序列跨多条
    运行的情况，取最近的是**有信息量的选择**：序列的归属跟随它的最后一次运行。
    """
    async with session_scope() as db:
        # 显式标注而不是直接 return：`AsyncSession` 是泛型的，未参数化时
        # `execute()` 的返回是 Any，直接返回会把 Any 泄漏出这个函数的签名。
        found: str | None = (
            await db.execute(
                select(TestRun.scenario_id)
                .where(TestRun.session_sequence_id == session_sequence_id)
                .order_by(TestRun.started_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        return found
