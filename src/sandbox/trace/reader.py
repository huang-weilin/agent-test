"""把库里的调用轨迹读回成检测器能吃的视图。

检测器**不直接查库**，只接收 `list[CallView]`。这样带来两个好处：检测器可以
拿人为构造的轨迹直接测（验收标准二、三就是这么做的），以及"跑检测"这个动作
不依赖数据库连接——watchdog 想在任意时刻扫一遍都行。
"""

from __future__ import annotations

import json

from sqlalchemy import select

from sandbox.db.engine import session_scope
from sandbox.db.models import ToolCall
from sandbox.detectors.base import CallView


def _row_to_view(row: ToolCall) -> CallView:
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
                    select(ToolCall).where(ToolCall.test_run_id == run_id).order_by(ToolCall.seq)
                )
            )
            .scalars()
            .all()
        )
        return [_row_to_view(r) for r in rows]
