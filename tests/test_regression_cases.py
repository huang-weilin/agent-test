"""回归用例的存取：冻结下来的基线必须和运行那次**逐字一样**。

两侧的取数走同一个函数，这不是为了好看。基线侧和当前侧如果各写一份取数，
它们迟早会漂移成两套判据——于是"用哪种方式看"会改变结论，而那是最难被发现
的一类错误：两次运行明明一样，一个页面说没变、另一个说变了。
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from sandbox.db.engine import session_scope
from sandbox.db.models import RegressionCase, TestRun
from sandbox.regression import cases

REPORT = {
    "ok": True,
    "premises_held": True,
    "error": None,
    "sessions": [],
    "findings": [],
    "expectation": None,
    "egress_charges": 1,
    "expected_charges": 1,
    "cost": None,
}

CONFIG = {"prompt": {"task": "干活", "model": "stub-model"}}


async def _run(run_id: str = "run-1", **overrides) -> str:
    fields = {
        "id": run_id,
        "scenario_id": "clean",
        "agent_version": "a@1",
        "config_hash": "c" * 64,
        "config_scheme": "2",
        "session_sequence_id": f"seq-{run_id}",
        "status": "completed",
        "started_at": datetime.now(UTC),
        "verdict_ok": True,
        "report_json": json.dumps(REPORT),
        "config_parts_json": json.dumps(CONFIG),
        "env_json": json.dumps({"model": "stub-model"}),
    }
    fields.update(overrides)
    async with session_scope() as db:
        db.add(TestRun(**fields))
    return run_id


async def test_the_frozen_baseline_matches_the_run_it_came_from(temp_db: str) -> None:
    """抄下来的一侧和原运行经过**同一个**取数函数，得到的快照必须相等。"""
    run_id = await _run()
    case_id = await cases.create_case(run_id=run_id, name="用例", note="")

    async with session_scope() as db:
        run = await db.get(TestRun, run_id)
        case = await db.get(RegressionCase, case_id)
        from_run = cases.snapshot_of_run(run)
        from_case = cases.snapshot_of_case(case)

    # run_id 那一项本来就不同（基线记的是**出处**的 id），把它排除掉比
    # 让两个函数各自编一个占位值干净。
    assert replace(from_run, run_id="x") == replace(from_case, run_id="x")


async def test_a_run_with_a_conclusion_is_accepted(temp_db: str) -> None:
    run_id = await _run()
    assert await cases.create_case(run_id=run_id, name="用例", note="") is not None


async def test_a_run_with_no_conclusion_is_refused(temp_db: str) -> None:
    """拿"未留存"当基线，之后每次比对都会报出一个假的退化。"""
    run_id = await _run("run-bare", verdict_ok=None, report_json=None)
    with pytest.raises(cases.NoConclusionError):
        await cases.create_case(run_id=run_id, name="用例", note="")


async def test_a_run_with_only_a_report_is_accepted(temp_db: str) -> None:
    """结论列没存但报告在——`resolved_ok` 读得出来，所以这条能用。

    判据是"**有没有**结论"，不是"结论列填没填"。按列判的话，一批本来
    好好的历史运行会被无理由地拒掉。
    """
    run_id = await _run("run-report-only", verdict_ok=None)
    assert await cases.create_case(run_id=run_id, name="用例", note="") is not None


async def test_an_unknown_run_is_refused(temp_db: str) -> None:
    with pytest.raises(KeyError):
        await cases.create_case(run_id="nope", name="用例", note="")


async def test_an_empty_name_falls_back_to_the_scenario(temp_db: str) -> None:
    """名字留空是允许的，但不能存下一个没有名字的用例——列表里会认不出来。"""
    run_id = await _run()
    case_id = await cases.create_case(run_id=run_id, name="   ", note="")
    async with session_scope() as db:
        assert (await db.get(RegressionCase, case_id)).name == "clean 的基线"


# --- JSON 列的读取 -------------------------------------------------------------


def test_unreadable_json_reads_as_no_record() -> None:
    """一条读不出来的记录不该让整个页面打不开。

    它应该表现为"这条没有记录"，然后在界面上如实显示成缺失。抛异常的话，
    用户看到的是一片 500，而问题只是一行旧数据。
    """
    assert cases._loads("{不是 JSON") is None
    assert cases._loads("") is None
    assert cases._loads(None) is None


def test_a_json_array_is_not_a_dict() -> None:
    """这些列按约定存字典。存成数组是**另一份数据**，不是一份配置。

    退化成 None 比让 `.get()` 在后面炸掉好——那时报错的位置离原因很远。
    """
    assert cases._loads("[1, 2, 3]") is None
    assert cases._loads("null") is None
    assert cases._loads('{"a": 1}') == {"a": 1}
