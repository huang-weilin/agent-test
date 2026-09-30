"""把库里的调用轨迹读回成检测器能吃的视图。

检测器**不直接查库**，只接收 `list[CallView]`。这样带来两个好处：检测器可以
拿人为构造的轨迹直接测（验收标准二、三就是这么做的），以及"跑检测"这个动作
不依赖数据库连接——watchdog 想在任意时刻扫一遍都行。

两种读法，对应两个层次的检测：

- `load_call_views(run_id)` —— 单次运行。循环与幂等检测的常规入口。
- `load_session_sequence(seq_id)` —— **整个会话序列**，跨多次运行。
  跨会话检测必须走这个：一次运行只看得见自己那一段，看不到上一次会话做过什么，
  而"Agent 有没有识别出这是重复操作"恰恰要靠跨会话比对才能回答。
"""

from __future__ import annotations

import json

from sqlalchemy import select

from sandbox.db.engine import session_scope
from sandbox.db.models import Session, ToolCall
from sandbox.detectors.base import CallView


def _row_to_view(
    row: ToolCall,
    *,
    session_ordinal: int = 0,
    memory_horizon_ordinal: int = -1,
) -> CallView:
    try:
        args = json.loads(row.args_json) if row.args_json else {}
    except ValueError:
        # 参数解析不了也要让检测器看到这次调用，否则它在检测里"消失"了，
        # 而消失的调用会让循环计数偏低——漏报的方向。用一个显式标记代替。
        args = {"__unparsable__": True}
    return CallView(
        id=row.id,
        seq=row.seq,
        tool_name=row.tool_name,
        args=args,
        fingerprint=row.fingerprint,
        is_write=bool(row.is_write),
        session_sequence_id=row.session_sequence_id or "",
        session_id=row.session_id or "",
        session_ordinal=session_ordinal,
        memory_horizon_ordinal=memory_horizon_ordinal,
        idempotency_key=row.idempotency_key,
        replay_reason=row.replay_reason,
        blocked_by_proxy=bool(row.blocked_by_proxy),
        status=row.status,
    )


async def load_call_views(run_id: str) -> list[CallView]:
    """按序号读出一次运行的全部工具调用。"""
    async with session_scope() as db:
        rows = (
            (
                await db.execute(
                    select(ToolCall, Session)
                    .join(Session, ToolCall.session_id == Session.id, isouter=True)
                    .where(ToolCall.test_run_id == run_id)
                    .order_by(ToolCall.seq)
                )
            )
            .all()
        )
        return [
            _row_to_view(
                call,
                session_ordinal=session.ordinal if session else 0,
                memory_horizon_ordinal=session.memory_horizon_ordinal if session else -1,
            )
            for call, session in rows
        ]


async def load_session_sequence(session_sequence_id: str) -> list[CallView]:
    """读出一个会话序列里的全部工具调用，按 (会话序号, 运行内序号) 排序。

    **排序键必须带上会话序号。** `seq` 只是运行内单调的，跨运行会重复——
    只按 seq 排的话，第 2 次会话的第一次调用（seq=1）会排到第 1 次会话的
    第 50 次调用前面，而跨会话归因完全建立在"谁先谁后"上。

    也正因如此，这里的排序键就是跨会话的时间线，而不是墙钟时间：
    墙钟在恢复快照、时间加速等场景下会失真，序号不会。
    """
    async with session_scope() as db:
        rows = (
            (
                await db.execute(
                    select(ToolCall, Session)
                    .join(Session, ToolCall.session_id == Session.id, isouter=True)
                    .where(ToolCall.session_sequence_id == session_sequence_id)
                    .order_by(
                        Session.ordinal.asc().nullsfirst(),
                        ToolCall.seq.asc(),
                    )
                )
            )
            .all()
        )
        return [
            _row_to_view(
                call,
                session_ordinal=session.ordinal if session else 0,
                memory_horizon_ordinal=session.memory_horizon_ordinal if session else -1,
            )
            for call, session in rows
        ]


async def load_sessions(session_sequence_id: str) -> list[Session]:
    """按序号读出会话序列里的会话行。报告与归因解释用。"""
    async with session_scope() as db:
        rows = (
            (
                await db.execute(
                    select(Session)
                    .where(Session.session_sequence_id == session_sequence_id)
                    .order_by(Session.ordinal)
                )
            )
            .scalars()
            .all()
        )
        return list(rows)
