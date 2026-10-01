"""「空表」和「没给」在边界上是两件事，这一对场景把它变成受控实验。

`toolschema` 的默认拒绝（`additionalProperties` 没写就是不许）本来只对**表过态的
声明**成立。旧行为把范围放宽到了没表态的声明上：`properties` 一个字没写
（`{type: object}`）时，`properties or {}` 把每个键都判成多出来的——一份没说键是
怎样的声明，被当成了说「一个键都不许」。后果不是漏检，是**凭空造了一条约束**：
这个工具的每一次调用都在边界上被拒，报告上写着 Agent 参数送错了，而真相是这份
声明让每一次调用都过不去。今天没有场景这么写过，所以当时没有一条测试会红。

这个文件钉三件事：

  * 那一对场景**只差 `properties` 这一行**（脚本、任务、世界、工具名一字不差）；
  * 没给键的名单时，调用照常到达目标系统，一条 `tool_schema` 都不产出；
  * 给了空表时，每一次调用都被拒——而且**不标成出口拦截**（送错参数的判定是
    Agent 的账，`blocked_by_proxy` 那一整套区分就是为这个）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from sandbox.db.models import ToolCall
from sandbox.scenario.loader import load_scenario
from sandbox.trace.reader import BOUNDARY_DETECTORS
from tests.control_pairs import assert_only_differs, entry_for

REPO_ROOT = Path(__file__).resolve().parents[1]
UNDECLARED = REPO_ROOT / "scenarios" / "schema_keys_undeclared.yaml"
EMPTY_TABLE = REPO_ROOT / "scenarios" / "schema_empty_properties.yaml"
AGENT_PYTHON = REPO_ROOT / "agents" / "react_demo" / ".venv" / "Scripts" / "python.exe"


# --- 受控对照：两个场景必须只差一个比特 ------------------------------------------


def test_the_controlled_pair_differs_in_exactly_one_bit() -> None:
    """脚本、任务、世界、工具名一字不差，唯一的差别是声明里有没有 `properties`。

    差别要是还有别的，结论的不同就可能是别的东西造成的——而两份报告读起来都
    很正常，谁都发现不了。这里两面都钉：抹掉那一行之后必须完全相等，而那一行
    必须真的在一边有、一边没有（否则上一句相等是句空话）。
    """
    undeclared = load_scenario(UNDECLARED)
    empty = load_scenario(EMPTY_TABLE)

    # 文件级的比对在账本里（`tests/control_pairs.py`）。这一对的形状特别值得
    # 记一笔：两支的差别是一个**空表**对**没有这个键**，而空字典在逐叶比对里
    # 没有叶子——账本里有一条测试专门盯着这一点（`test_an_empty_container_is_
    # not_the_same_as_a_missing_key`）。
    assert_only_differs(
        entry_for("schema_empty_properties", "schema_keys_undeclared")
    )

    assert "properties" not in undeclared.tools[0].parameters, "对照组不该有键的名单"
    assert empty.tools[0].parameters == {"type": "object", "properties": {}}, (
        "实验组给的是一个空表——不是没写"
    )


# --- 端到端：真的拉起 Agent 子进程 -------------------------------------------------


def _runner() -> Any:
    """拉起 Agent 子进程的执行器。venv 不在就跳过——这一条依赖真实子进程。"""
    if not AGENT_PYTHON.exists():
        pytest.skip(f"被测 Agent 的 venv 不在（{AGENT_PYTHON}），跳过端到端")

    from sandbox.scenario.runner import ScenarioRunner

    return ScenarioRunner(session_timeout_s=120.0)


async def test_a_key_set_nobody_declared_does_not_become_a_ban(temp_db: str, db) -> None:
    """没事先说明键是怎样的，就别替它发明一份名单。

    这条是修好之后才绿的：改之前，两次调用都会在边界上被拒（每个键都算多出来的），
    报告里两条 `tool_schema/fail`——而脚本送的正是这个工具本来就该收的参数。
    """
    result = await _runner().run(load_scenario(UNDECLARED))

    assert result.ok, result.error
    assert [f for f in result.findings if f.detector in BOUNDARY_DETECTORS] == [], (
        "键上没有可核对的东西，不该凭空产出一条参数不合规"
    )

    calls = (await db.execute(select(ToolCall))).scalars().all()
    assert len(calls) == 2, "两次调用都要留下记录"
    assert {c.status for c in calls} == {"ok"}, "两次都该到达目标系统，不是被拒"
    assert result.egress_charges == 0


async def test_an_empty_key_table_bans_every_key(temp_db: str, db) -> None:
    """空表是一句写下来的「一个键都不许」，照旧拒绝。

    这一条在修之前就是绿的，留着是因为对照组的说服力全在它身上：同一个脚本、
    同一个工具，只把 `properties: {}` 去掉，结论就翻了。
    """
    result = await _runner().run(load_scenario(EMPTY_TABLE))

    assert result.ok, result.error
    boundary = [f for f in result.findings if f.detector in BOUNDARY_DETECTORS]
    assert [f.detector for f in boundary] == ["tool_schema"] * 2
    assert result.expectation is not None
    assert len(result.expectation.matched) == 2
    assert result.expectation.unexpected == []

    calls = (await db.execute(select(ToolCall))).scalars().all()
    assert {c.status for c in calls} == {"invalid_args"}
    # 送错参数的判定是 Agent 的账，不是沙箱的策略在生效——标反了，循环检测会把
    # 它的重试当成沙箱制造的，审计页会说"不是被测对象的错"。
    assert all(c.blocked_by_proxy is False for c in calls)
