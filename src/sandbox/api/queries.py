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

from sqlalchemy import select

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


def parse_report(run: TestRun) -> dict[str, Any] | None:
    """取出运行时存下的结论快照。None = 结论未留存。

    历史运行（在本字段存在之前跑的）没有这份快照，返回 None。**界面必须把它
    显示成"未留存"，而不是渲染成一份空的失败报告**——后者看起来和"跑了但
    什么都没测到"一模一样，而那是完全不同的两件事。
    """
    if not run.report_json:
        return None
    return json.loads(run.report_json)


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
    ts: datetime

    @property
    def args_short(self) -> str:
        """参数摘要。审计表是拿来**扫**的，一整块 JSON 会把表撑到没法看。

        截断了就带上省略号——不然读者会以为参数本来就长这样，
        而"参数里少了一个字段"和"参数被截断显示"是两回事。
        """
        if len(self.args_json) <= 140:
            return self.args_json
        return self.args_json[:140] + "…"


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
