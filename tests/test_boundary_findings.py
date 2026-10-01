"""边界当场判的那两条结论，怎么回到这次运行里。

`tool_scope` / `tool_schema` 是出口代理在放行之前当场判的（`proxy/app.py`），
写进 `detector_finding` 表。在这次接线之前，**没有任何一条路把它们带回这次
运行**：`result.findings` 只装 `_run_detectors` 算出来的那几条，于是 `expect`
里写 `tool_schema` 永远"未命中"、报告里一条都看不到、`ok` 也不受影响——而
`docs/detectors.md`、`docs/scenarios.md`、两个页面都在说它们进了检测结论。

这个文件钉三件事：

  * 它们真的进得了 `result.findings`，`expect` 命得中，`ok` 受它影响；
  * 它们排在最前面（永远是 fail，不能被后面某条 warning 压住）；
  * **沙箱自己造成的那些不许进来**（`egress` / `token_budget` / 看门狗那条
    `loop`）——`blocked_by_proxy` 那一整套区分就是为了这个。
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from sqlalchemy import select

from sandbox.db.models import DetectorFinding, DetectorVerdict, Session, TestRun, ToolCall
from sandbox.scenario.expect import evaluate
from sandbox.scenario.loader import load_scenario
from sandbox.trace.reader import BOUNDARY_DETECTORS, load_boundary_findings

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_REJECTED = REPO_ROOT / "scenarios" / "schema_rejected.yaml"
AGENT_PYTHON = REPO_ROOT / "agents" / "react_demo" / ".venv" / "Scripts" / "python.exe"


# --- 白名单：哪些结论读得回来，哪些读不回来 -----------------------------------------


async def _row(
    db,
    *,
    run_id: str,
    detector: str,
    verdict: str = "fail",
    summary: str = "s",
) -> None:
    db.add(
        DetectorFinding(
            id=str(uuid.uuid4()),
            test_run_id=run_id,
            detector=detector,
            verdict=verdict,
            confidence="high",
            summary=summary,
        )
    )
    await db.commit()


async def _run_row(db, *, sequence_id: str, ordinal: int = 0) -> str:
    run_id = str(uuid.uuid4())
    db.add(TestRun(id=run_id, scenario_id="t", agent_version="a@1", config_hash="h"))
    db.add(
        Session(
            id=str(uuid.uuid4()),
            test_run_id=run_id,
            session_sequence_id=sequence_id,
            ordinal=ordinal,
        )
    )
    await db.commit()
    return run_id


async def test_only_the_two_boundary_conclusions_come_back(db) -> None:
    """**这是一张白名单，不是"库里所有的结论"。**

    表里还躺着看门狗熔断时写的 `loop`、网关的 `token_budget`、代理自己的
    `egress`。读回来的话：`loop` 会和跑完之后算出来的那条撞成"同一件事两条
    结论"；另外两条是沙箱自己的策略在生效，不是被测对象做错了事——混进
    `findings` 就是把账记到它头上。
    """
    seq = str(uuid.uuid4())
    run_id = await _run_row(db, sequence_id=seq)
    await _row(db, run_id=run_id, detector="tool_schema", summary="参数不合规")
    await _row(db, run_id=run_id, detector="tool_scope", summary="调了没声明的工具")
    await _row(db, run_id=run_id, detector="egress", summary="路由走反了")
    await _row(db, run_id=run_id, detector="token_budget", verdict="warning", summary="超预算")
    await _row(db, run_id=run_id, detector="loop", summary="看门狗熔断那条")

    back = await load_boundary_findings(seq)

    assert [f.detector for f in back] == ["tool_schema", "tool_scope"]
    assert set(BOUNDARY_DETECTORS) == {"tool_scope", "tool_schema"}


async def test_other_runs_in_the_database_are_not_mixed_in(db) -> None:
    """按会话序列过滤，不是"把整张表捞出来"。"""
    seq = str(uuid.uuid4())
    mine = await _run_row(db, sequence_id=seq)
    other = await _run_row(db, sequence_id=str(uuid.uuid4()))
    await _row(db, run_id=mine, detector="tool_schema", summary="我的")
    await _row(db, run_id=other, detector="tool_schema", summary="别人的")

    back = await load_boundary_findings(seq)

    assert [f.summary for f in back] == ["我的"]


async def test_the_sequence_is_ordered_by_session_ordinal(db) -> None:
    """跨会话的先后按**会话序号**排，不按写入时刻。

    写第一条结论时故意让它的 `ts` 更晚：墙钟这个项目里会被快照恢复和时间加速
    拨动，跨会话唯一可信的时间线是序号（和 `load_session_sequence` 同一条约定）。
    """
    seq = str(uuid.uuid4())
    first = await _run_row(db, sequence_id=seq, ordinal=1)
    second = await _run_row(db, sequence_id=seq, ordinal=0)
    await _row(db, run_id=first, detector="tool_schema", summary="第一次会话里的")
    await _row(db, run_id=second, detector="tool_schema", summary="第零次会话里的")

    back = await load_boundary_findings(seq)

    assert [f.summary for f in back] == ["第零次会话里的", "第一次会话里的"]


# --- 端到端：真的拉起 Agent 子进程 -------------------------------------------------


def _require_agent() -> None:
    if not AGENT_PYTHON.exists():
        pytest.skip(f"被测 Agent 的 venv 不在（{AGENT_PYTHON}），跳过端到端边界结论")


async def _run_scenario(path: Path):
    from sandbox.scenario.runner import ScenarioRunner

    _require_agent()
    return await ScenarioRunner(session_timeout_s=120.0).run(load_scenario(path))


async def test_a_rejected_call_reaches_the_report_and_can_be_expected(
    temp_db: str, db
) -> None:
    """这条特性的验收标准。

    四种参数写错的方式（enum / minimum / maximum / 金额上限）都在边界上被拒，
    四条结论必须原样进到这次运行的报告里、`expect` 命得中、而**钱一笔没动**——
    `expected_charges: 0` 说的就是"那笔违规的支付一次都没落到目标系统上"。
    """
    result = await _run_scenario(SCHEMA_REJECTED)

    assert result.ok, result.error
    boundary = [f for f in result.findings if f.detector in BOUNDARY_DETECTORS]
    assert [f.detector for f in boundary] == ["tool_schema"] * 4
    assert all(f.verdict == DetectorVerdict.FAIL for f in boundary)

    assert result.egress_charges == 0, "地面真值：被拒的调用没到达目标系统"
    assert result.expectation is not None
    assert len(result.expectation.matched) == 4
    assert result.expectation.unexpected == []

    # 被拒的调用在库里的样子：状态是 invalid_args，而 blocked_by_proxy 必须是
    # False——送错参数的是 Agent 自己，不是沙箱的策略在生效。标反了的话，
    # 循环检测会把它的重试当成沙箱制造的，审计页会说"不是被测对象的错"。
    rejected = (
        (await db.execute(select(ToolCall).where(ToolCall.status == "invalid_args")))
        .scalars()
        .all()
    )
    assert len(rejected) == 4
    assert all(c.blocked_by_proxy is False for c in rejected)


async def test_the_boundary_conclusions_come_first(temp_db: str) -> None:
    """排序：边界那两条排在聚合结论前面。

    它们是这次运行里最直接的事实（没有推断、没有阈值），而且**永远是 fail**；
    排在后面的话会被某条 warning 压到看不见的地方——同一条约定见
    `tests/test_autonomy.py::test_the_failures_come_before_the_attempts`。
    """
    result = await _run_scenario(SCHEMA_REJECTED)

    detectors = [f.detector for f in result.findings]
    first_aggregate = min(
        (i for i, d in enumerate(detectors) if d not in BOUNDARY_DETECTORS),
        default=len(detectors),
    )
    assert all(d in BOUNDARY_DETECTORS for d in detectors[:first_aggregate])
    assert first_aggregate > 0


async def test_a_boundary_failure_nobody_expected_fails_the_run(temp_db: str) -> None:
    """接线了才算数：没写预期的边界失败必须让这次运行不通过。

    只接线到"报告里看得见"是不够的——那样它仍然是一条**不参与判定**的陈列。
    这里把场景的预期清空（其余一字不改），同一个 `result.findings` 立刻该判成
    不通过，并且那四条都落在"意外"那一栏里。
    """
    result = await _run_scenario(SCHEMA_REJECTED)
    assert result.ok

    scenario = load_scenario(SCHEMA_REJECTED)
    stripped = scenario.model_copy(update={"expect": []})
    verdict = evaluate(stripped, result.findings)

    assert not verdict.ok
    assert len(verdict.unexpected) == 4
    assert all(f.detector in BOUNDARY_DETECTORS for f in verdict.unexpected)
