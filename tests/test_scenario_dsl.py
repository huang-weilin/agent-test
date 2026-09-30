"""场景 DSL：加载、校验、预期比对、以及"算不算过"的判定。

这些都不碰子进程，所以跑得很快——被测 Agent 那一路的成本不该压在每一次
校验逻辑的回归上。

分三块，用注释分段（项目约定不用测试类，见 pyproject 里 python_classes 的说明）：
  * 加载与校验
  * 预期比对
  * 前提成立与否
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from sandbox.db.models import Confidence, DetectorVerdict
from sandbox.detectors.base import Finding
from sandbox.scenario.expect import ExpectResult, evaluate
from sandbox.scenario.loader import ScenarioError, load_all, load_scenario
from sandbox.scenario.runner import ScenarioResult, SessionOutcome
from sandbox.scenario.spec import Expectation, Scenario

# --- 构造辅助 -----------------------------------------------------------------


def _tool(name: str, *, is_write: bool, **kw: Any) -> dict[str, Any]:
    decl: dict[str, Any] = {
        "name": name,
        "description": f"{name} 的说明",
        "is_write": is_write,
        "parameters": {"type": "object", "properties": {}},
    }
    decl.update(kw)
    return decl


def _scenario_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "t",
        "agent": {"version": "a@1", "task": "干活"},
        "tools": [_tool("read_thing", is_write=False), _tool("write_thing", is_write=True)],
        "sessions": [
            {
                "name": "第一次",
                "script": [{"tool_calls": [{"name": "write_thing", "arguments": {}}]}],
            }
        ],
    }
    base.update(overrides)
    return base


def _write(tmp_path: Path, data: Any, name: str = "s.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _finding(
    detector: str,
    verdict: DetectorVerdict,
    *,
    reason_code: str | None = None,
    summary: str = "s",
) -> Finding:
    detail = {"reason_code": reason_code} if reason_code else {}
    return Finding(
        detector=detector,
        verdict=verdict,
        summary=summary,
        confidence=Confidence.HIGH,
        detail=detail,
    )


def _scenario_model(expect: list[dict[str, Any]], **kw: Any) -> Scenario:
    return Scenario.model_validate(_scenario_dict(expect=expect, **kw))


def _outcome(**kw: Any) -> SessionOutcome:
    base: dict[str, Any] = {
        "name": "s",
        "ordinal": 0,
        "memory_horizon_ordinal": -1,
        "restored_from": None,
        "exit_code": 0,
        "elapsed_s": 1.0,
        "timed_out": False,
        "script_steps_declared": 3,
        "script_steps_consumed": 3,
        "tool_calls": 3,
    }
    base.update(kw)
    return SessionOutcome(**base)


# --- 加载与校验 ---------------------------------------------------------------


def test_a_well_formed_scenario_loads(tmp_path: Path) -> None:
    scenario = load_scenario(_write(tmp_path, _scenario_dict()))
    assert scenario.id == "t"
    assert scenario.declared_writes() == {"write_thing"}
    assert len(scenario.sessions) == 1


def test_missing_file_is_an_error_not_a_default(tmp_path: Path) -> None:
    # 静默返回一个空场景的话，跑完会报"什么都没发现"——一个不存在的场景
    # 会被读成一次通过。
    with pytest.raises(ScenarioError, match="不存在"):
        load_scenario(tmp_path / "nope.yaml")


def test_non_mapping_top_level_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ScenarioError, match="映射"):
        load_scenario(_write(tmp_path, ["不是映射"]))


def test_is_write_has_no_default(tmp_path: Path) -> None:
    """`is_write` 必须显式声明。

    给个 False 默认值的话，忘了写的那些工具会静默变成只读，于是重复扣款
    检测根本不会覆盖它们——而报告上完全看不出来。
    """
    data = _scenario_dict()
    del data["tools"][1]["is_write"]
    with pytest.raises(ScenarioError):
        load_scenario(_write(tmp_path, data))


def test_unknown_fields_are_rejected(tmp_path: Path) -> None:
    # 打错字段名如果能静默通过，场景作者会以为它生效了。
    with pytest.raises(ScenarioError):
        load_scenario(_write(tmp_path, _scenario_dict(typo_field=1)))


def test_duplicate_tool_names_are_rejected(tmp_path: Path) -> None:
    data = _scenario_dict(tools=[_tool("same", is_write=False), _tool("same", is_write=True)])
    with pytest.raises(ScenarioError, match="重复"):
        load_scenario(_write(tmp_path, data))


def test_a_scenario_with_no_sessions_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ScenarioError):
        load_scenario(_write(tmp_path, _scenario_dict(sessions=[])))


def test_restoring_from_a_later_snapshot_is_rejected(tmp_path: Path) -> None:
    """恢复到**之后**才拍的快照 = 让记忆回到未来。

    不拦的话它会跑起来，只是记忆视野算出一个没意义的数，而归因完全建立在
    视野上——结论会静静地反过来。
    """
    data = _scenario_dict(
        sessions=[
            {"name": "一", "restore_from": "future", "script": []},
            {"name": "二", "snapshot_after": "future", "script": []},
        ]
    )
    with pytest.raises(ScenarioError, match="记忆回到未来"):
        load_scenario(_write(tmp_path, data))


def test_restoring_from_an_earlier_snapshot_is_allowed(tmp_path: Path) -> None:
    data = _scenario_dict(
        sessions=[
            {"name": "一", "snapshot_before": "base", "script": []},
            {"name": "二", "restore_from": "base", "script": []},
        ]
    )
    scenario = load_scenario(_write(tmp_path, data))
    assert scenario.sessions[1].restore_from == "base"


def test_a_snapshot_from_the_same_session_is_not_yet_available(tmp_path: Path) -> None:
    # snapshot_after 要等会话跑完才拍，所以同一个会话里引用它是不成立的。
    data = _scenario_dict(
        sessions=[{"name": "一", "restore_from": "later", "snapshot_after": "later"}]
    )
    with pytest.raises(ScenarioError, match="记忆回到未来"):
        load_scenario(_write(tmp_path, data))


def test_script_calling_an_undeclared_tool_is_rejected(tmp_path: Path) -> None:
    """多半是打错字。

    不拦的话 Agent 会去调一个它自己没有的工具、直接报错退出，而报告显示
    "没有发现问题"——把一次跑崩读成一次通过。
    """
    data = _scenario_dict(
        sessions=[
            {"name": "一", "script": [{"tool_calls": [{"name": "typo_thing", "arguments": {}}]}]}
        ]
    )
    with pytest.raises(ScenarioError, match="未声明的工具"):
        load_scenario(_write(tmp_path, data))


def test_a_note_on_an_order_that_does_not_exist_is_rejected(tmp_path: Path) -> None:
    """备注挂在目标系统里没有的单上，Agent 永远读不到它。

    **这条错误最坏的地方是它不报错。** 备注挂在 Z9 上，而世界里只有 A1、B2：
    那一支无论怎么跑都会通过，报告上看起来像"Agent 没把世界里的东西记下来"——
    一个场景作者打错字造成的空白，读起来像被测对象的一个优点。
    """
    data = _scenario_dict(world={"order_notes": {"Z9": "客户备注：工作日送达。"}})
    with pytest.raises(ScenarioError, match="不存在的订单"):
        load_scenario(_write(tmp_path, data))


def test_a_blank_note_is_rejected(tmp_path: Path) -> None:
    """空备注在目标系统那边等同于没有备注。

    而同一条规矩在目标系统侧是"根本不给 `note` 键"（见 `view_of`）——于是这一支
    测的其实是"Agent 没记下一件不存在的事"，它必定通过，且必定什么都不说明。
    """
    data = _scenario_dict(world={"order_notes": {"A1": "   "}})
    with pytest.raises(ScenarioError, match="空的"):
        load_scenario(_write(tmp_path, data))


def test_a_note_on_a_real_order_passes_the_front_door(tmp_path: Path) -> None:
    """反向那一半：写对了的声明必须真的装进场景里。

    只有上面两条的话，一个"看到 world 就报错"的校验也能全绿。
    """
    data = _scenario_dict(world={"order_notes": {"A1": "客户备注：工作日送达。"}})
    scenario = load_scenario(_write(tmp_path, data))
    assert scenario.world.order_notes == {"A1": "客户备注：工作日送达。"}


def test_load_all_sorts_by_filename(tmp_path: Path) -> None:
    _write(tmp_path, _scenario_dict(id="b"), "b.yaml")
    _write(tmp_path, _scenario_dict(id="a"), "a.yaml")
    (tmp_path / "notes.txt").write_text("不是场景", encoding="utf-8")
    assert [s.id for s in load_all(tmp_path)] == ["a", "b"]


def test_load_all_on_a_missing_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ScenarioError):
        load_all(tmp_path / "nope")


# --- 预期比对 -----------------------------------------------------------------


def test_a_matching_expectation_is_matched() -> None:
    scenario = _scenario_model([{"detector": "idempotency", "verdict": "fail"}])
    result = evaluate(scenario, [_finding("idempotency", DetectorVerdict.FAIL)])
    assert result.ok
    assert len(result.matched) == 1


def test_an_unmet_expectation_is_missing_not_passed() -> None:
    """期望报错、实际没报 → 不能算通过。

    两种可能：Agent 这次没犯那个错，或者检测器漏报了。分不清，所以只能报
    "未达成"——折叠成通过等于给一个可能坏掉的检测器发绿灯。
    """
    scenario = _scenario_model([{"detector": "idempotency", "verdict": "fail"}])
    result = evaluate(scenario, [])
    assert not result.ok
    assert len(result.missing) == 1


def test_an_uncovered_failure_is_unexpected_and_fails_the_run() -> None:
    scenario = _scenario_model([])
    result = evaluate(scenario, [_finding("loop.strict", DetectorVerdict.FAIL)])
    assert not result.ok
    assert len(result.unexpected) == 1


def test_inconclusive_is_never_unexpected() -> None:
    """未覆盖（inconclusive）是信息，不是缺陷。

    它计入覆盖率，但不该让运行失败——否则任何只有一个会话的场景都会因为
    "跨会话判定做不了"而永远红着。
    """
    scenario = _scenario_model([])
    result = evaluate(scenario, [_finding("loop.cross_session", DetectorVerdict.INCONCLUSIVE)])
    assert result.ok
    assert len(result.uncovered) == 1
    assert result.unexpected == []


def test_unexpected_failures_can_be_downgraded_per_scenario() -> None:
    scenario = _scenario_model([], unexpected_failures="allow")
    result = evaluate(scenario, [_finding("loop.strict", DetectorVerdict.FAIL)])
    assert result.ok


def test_reason_code_is_part_of_the_match() -> None:
    scenario = _scenario_model(
        [{"detector": "idempotency", "verdict": "fail", "reason_code": "missing_key"}]
    )
    wrong = evaluate(
        scenario, [_finding("idempotency", DetectorVerdict.FAIL, reason_code="key_rotated")]
    )
    assert not wrong.ok and len(wrong.missing) == 1

    right = evaluate(
        scenario, [_finding("idempotency", DetectorVerdict.FAIL, reason_code="missing_key")]
    )
    assert right.ok


def test_expectations_do_not_compete_for_the_same_finding() -> None:
    """每条预期各自找一条没被占用的结论，不按顺序配对。

    按顺序配的话结果会依赖结论的排列顺序；而两条一模一样的预期，也只能
    各自都找到结论才算通过——一条结论不该同时满足两条预期。
    """
    scenario = _scenario_model(
        [
            {"detector": "idempotency", "verdict": "fail"},
            {"detector": "idempotency", "verdict": "fail"},
        ]
    )
    result = evaluate(scenario, [_finding("idempotency", DetectorVerdict.FAIL)])
    assert len(result.matched) == 1
    assert len(result.missing) == 1
    assert not result.ok


def test_verdict_is_part_of_the_match() -> None:
    scenario = _scenario_model([{"detector": "idempotency", "verdict": "warning"}])
    result = evaluate(scenario, [_finding("idempotency", DetectorVerdict.FAIL)])
    assert not result.ok
    # 它同时也是一条没被覆盖的失败，所以两栏都该有——但它是**同一个** finding，
    # 不是两条，所以 missing 和 unexpected 各计一次是对的。
    assert len(result.missing) == 1
    assert len(result.unexpected) == 1


# --- 前提成立与否 -------------------------------------------------------------


def test_full_consumption_holds_the_premise() -> None:
    assert _outcome(script_steps_consumed=3).premise_held


def test_over_consumption_holds_the_premise() -> None:
    # 循环场景会多消费一步：脚本耗尽后上游补一句强制终止。
    assert _outcome(script_steps_consumed=4).premise_held


def test_early_exit_breaks_the_premise() -> None:
    """Agent 提前退出 → 这一段根本没跑成，检测结论不指向被测对象。"""
    assert not _outcome(script_steps_consumed=1, exit_code=1).premise_held


def test_a_circuit_break_holds_the_premise() -> None:
    """被熔断时脚本必然没消费完，但那正是循环场景的预期结果。

    按"消费完才算数"判的话，一个完美触发的循环场景会被报成"没测到"——
    判据把成功的信号读成了失败。
    """
    outcome = _outcome(
        script_steps_consumed=3,
        script_steps_declared=8,
        killed=True,
        kill_reason="loop.strict: ...",
        exit_code=1,
    )
    assert outcome.premise_held


def test_a_killed_run_does_not_excuse_a_short_script_unrelatedly() -> None:
    # 熔断不是万能豁免：判据仍然是"消费完 或 被熔断"，不是"一律算数"。
    assert _outcome(script_steps_consumed=8, script_steps_declared=8, killed=True).premise_held


def test_ok_requires_expectations_to_be_present() -> None:
    # 没有预期就等于没有断言——不能算通过。
    result = ScenarioResult(scenario_id="t", run_id="r", sessions=[_outcome()])
    assert not result.ok


def test_a_broken_premise_fails_the_run_regardless_of_findings() -> None:
    """前提没成立时一律不通过，无论检测器说了什么。

    那些结论是在一段不符合场景描述的轨迹上跑出来的，不指向被测对象。
    让它们算数的话，"什么都没测到"会以"全部通过"的样子交出去——这是本
    工具最严重的失效方式。
    """
    result = ScenarioResult(
        scenario_id="t",
        run_id="r",
        sessions=[_outcome(script_steps_consumed=0, exit_code=1)],
        expectation=ExpectResult(),
    )
    assert not result.premises_held
    assert not result.ok


def test_an_execution_error_fails_the_run() -> None:
    result = ScenarioResult(
        scenario_id="t", run_id="r", sessions=[_outcome()], expectation=ExpectResult(), error="boom"
    )
    assert not result.ok


# --- 目标系统侧地面真值 -------------------------------------------------------


def _charged(egress: int, expected: int | None) -> ScenarioResult:
    return ScenarioResult(
        scenario_id="t",
        run_id="r",
        sessions=[_outcome()],
        expectation=ExpectResult(),
        egress_charges=egress,
        expected_charges=expected,
    )


def test_unset_expected_charges_is_not_a_check() -> None:
    # 没写就不检查。写成 0 是"我断言一笔都不该扣"，两者不能混。
    assert _charged(egress=99, expected=None).charges_ok


def test_charges_matching_the_expectation_pass() -> None:
    assert _charged(egress=1, expected=1).charges_ok


def test_too_many_charges_fail_the_run() -> None:
    assert not _charged(egress=2, expected=1).charges_ok


def test_too_few_charges_fail_the_run() -> None:
    """扣少了也是不对的。

    这个方向容易被当成"更安全所以放过"，但扣少了同样是没按场景描述执行：
    要么 Agent 中间挂了，要么该付的没付。场景写的是 1 笔，就得是 1 笔。
    """
    assert not _charged(egress=0, expected=1).charges_ok


def test_unreadable_charge_count_is_not_a_pass() -> None:
    """查不到 != 没问题。

    负数是"mock 没起来、这笔账根本没对上"。把它读成"没发现问题"，正是这个
    工具最严重的失效方式——报告会以绿灯的样子交出一段谁也没验过的结论。
    """
    assert not _charged(egress=-1, expected=1).charges_ok


def test_a_passing_detector_cannot_rescue_a_wrong_charge_count() -> None:
    """检测器全绿、预期也全命中，但钱确实扣多了 → 整体仍然不通过。

    这就是 `expected_charges` 存在的全部理由。幂等检测器只看边界上发生了什么，
    Agent 在进程内直接扣款、或者 mock 自己重放了请求，边界都可能很干净。
    地面真值必须独立成立，不能由检测结论反推。
    """
    expectation = ExpectResult(
        matched=[
            (
                Expectation(detector="idempotency", verdict=DetectorVerdict.PASS),
                _finding("idempotency", DetectorVerdict.PASS),
            )
        ],
    )
    result = ScenarioResult(
        scenario_id="t",
        run_id="r",
        sessions=[_outcome()],
        expectation=expectation,
        egress_charges=2,
        expected_charges=1,
    )
    assert result.expectation is not None and result.expectation.ok
    assert not result.ok
