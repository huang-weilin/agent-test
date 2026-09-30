"""验证数据层：建表、关键索引存在、跨会话幂等查询路径可用、四态语义不退化。"""

from __future__ import annotations

import uuid

from sqlalchemy import select

from sandbox.db.models import (
    DetectorFinding,
    DetectorVerdict,
    Session,
    TestRun,
    ToolCall,
)


async def test_can_persist_run_session_and_tool_calls(db) -> None:
    run_id, seq_id = str(uuid.uuid4()), str(uuid.uuid4())
    db.add(
        TestRun(
            id=run_id,
            scenario_id="duplicate_payment",
            agent_version="react_demo@0.0.1",
            config_hash="deadbeef",
            session_sequence_id=seq_id,
        )
    )
    await db.flush()

    sess = Session(id=str(uuid.uuid4()), test_run_id=run_id, session_sequence_id=seq_id)
    db.add(sess)
    await db.flush()

    # 同一指纹在同一会话序列内出现两次 —— 幂等检测要能查出来
    for i in range(2):
        db.add(
            ToolCall(
                id=str(uuid.uuid4()),
                test_run_id=run_id,
                session_id=sess.id,
                session_sequence_id=seq_id,
                seq=i,
                tool_name="create_payment",
                args_json='{"amount": 100, "order_id": "A1"}',
                arg_hash="h" * 64,
                fingerprint="f" * 64,
                is_write=True,
            )
        )
    await db.flush()

    found = (
        (
            await db.execute(
                select(ToolCall).where(
                    ToolCall.session_sequence_id == seq_id,
                    ToolCall.fingerprint == "f" * 64,
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(found) == 2, "跨会话幂等检测的主查询路径必须可用"
    assert all(tc.is_write for tc in found)


def test_tool_call_indexes_exist() -> None:
    """索引是设计的一部分：没有它们，跨会话查询在真实数据量下会崩。"""
    indexes = {ix.name for ix in ToolCall.__table__.indexes}
    assert "ix_tool_call_seq_fp" in indexes, "跨会话指纹索引缺失"
    assert "ix_tool_call_run_seq" in indexes, "运行内窗口索引缺失"


def test_detector_verdict_has_four_states() -> None:
    """四态语义不可退化。inconclusive 必须存在，见 docs/detectors.md。"""
    assert {v.value for v in DetectorVerdict} == {
        "pass",
        "fail",
        "warning",
        "inconclusive",
    }


async def test_inconclusive_is_storable(db) -> None:
    """'没测到' 必须能被如实记录，而不是被迫写成 pass。"""
    run_id = str(uuid.uuid4())
    db.add(TestRun(id=run_id, scenario_id="s", agent_version="a", config_hash="c"))
    await db.flush()
    db.add(
        DetectorFinding(
            id=str(uuid.uuid4()),
            test_run_id=run_id,
            detector="idempotency",
            verdict=DetectorVerdict.INCONCLUSIVE.value,
            summary="Agent 未调用任何写操作，无法判定幂等性",
        )
    )
    await db.flush()

    row = (
        await db.execute(select(DetectorFinding).where(DetectorFinding.test_run_id == run_id))
    ).scalar_one()
    assert row.verdict == "inconclusive"
