"""场景里的豁免规则：能被写下来、能生效、能留痕、写错了当场报错。

豁免机制本身在检测器层早就有了（`sandbox.detectors.base.Exemption`），也有语料
覆盖。缺的是**前门**：`docs/detectors.md` 第二节把它写成一等配置并给了 YAML 例子，
而 `Scenario` 上根本没有这个字段——`extra="forbid"` 会把照文档写的场景直接判成
加载失败。也就是说，那份文档当时指向一个不存在的功能。

这个文件盯的就是那类"文档说有、代码里没有"的缺口，所以每一层都单独测：

  * **加载层**：三种"看着在、其实永远不生效"的写法必须当场报错；
  * **装配层**：runner 真的把它传下去了（不是只有 `run_all` 会认）；
  * **语义层**：规则只静音它该静音的那一条，不牵连别的；
  * **留痕**：被静音的结论照样在报告里，而且带着理由；
  * **指纹**：加一条豁免之后，前后两次运行不能再比"失败少了几个"。
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import yaml

from sandbox.corpus import load_corpus
from sandbox.db.models import DetectorVerdict
from sandbox.detectors.base import Exemption
from sandbox.scenario import identity
from sandbox.scenario.loader import ScenarioError, load_scenario
from sandbox.scenario.runner import _run_detectors
from sandbox.scenario.spec import Scenario

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "corpus"


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


def _rule(**overrides: Any) -> dict[str, Any]:
    """一条能命中的豁免规则：只读工具，按"差异只在 page 上"匹配。"""
    rule: dict[str, Any] = {
        "detector": "loop.approximate",
        "tool": "list_orders",
        "reason": "这个工具的排除法翻页本来就不单调，已人工确认无害",
        "match": {"arg_differs_on": ["page"]},
    }
    rule.update(overrides)
    return rule


def _scenario_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "t",
        "agent": {"version": "a@1", "task": "干活"},
        "tools": [_tool("list_orders", is_write=False)],
        "sessions": [
            {
                "name": "第一次",
                "script": [{"tool_calls": [{"name": "list_orders", "arguments": {}}]}],
            }
        ],
    }
    base.update(overrides)
    return base


def _write(tmp_path: Path, data: Any, name: str = "s.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _rejects(data: dict[str, Any], tmp_path: Path, *keywords: str) -> None:
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(_write(tmp_path, data))
    message = str(excinfo.value)
    for keyword in keywords:
        assert keyword in message, f"报错里没提到 {keyword!r}：\n{message}"


def _item(item_id: str):
    return next(i for i in load_corpus(CORPUS) if i.id == item_id)


# --- 加载层：三种"装上了但没接线"的写法 ---------------------------------------


def test_a_wellformed_exemption_loads(tmp_path: Path) -> None:
    scenario = load_scenario(_write(tmp_path, _scenario_dict(exemptions=[_rule()])))
    rules = scenario.exemption_set().rules
    assert len(rules) == 1
    assert rules[0].label == "exemption:loop.approximate/list_orders"


def test_no_exemptions_means_an_empty_rule_set(tmp_path: Path) -> None:
    """不写就是没有，而不是"有个默认规则"。"""
    scenario = load_scenario(_write(tmp_path, _scenario_dict()))
    assert scenario.exemption_set().rules == []


def test_an_exemption_for_an_unknown_detector_is_rejected(tmp_path: Path) -> None:
    """文档里那个 `loop_detection` 就是这么错的：检测器叫 `loop`。"""
    _rejects(
        _scenario_dict(exemptions=[_rule(detector="loop_detection")]),
        tmp_path,
        "loop_detection",
        "永远不会命中",
    )


def test_an_exemption_for_an_undeclared_tool_is_rejected(tmp_path: Path) -> None:
    _rejects(
        _scenario_dict(exemptions=[_rule(tool="list_order")]),
        tmp_path,
        "list_order",
        "永远不会命中",
    )


def test_an_exemption_without_a_reason_is_rejected(tmp_path: Path) -> None:
    _rejects(
        _scenario_dict(exemptions=[_rule(reason="   ")]),
        tmp_path,
        "reason",
    )


def test_an_exemption_with_an_unknown_match_key_is_rejected(tmp_path: Path) -> None:
    _rejects(
        _scenario_dict(exemptions=[_rule(match={"arg_differs_on_page": ["page"]})]),
        tmp_path,
        "不认识的键",
    )


def test_arg_differs_on_that_is_a_string_is_rejected(tmp_path: Path) -> None:
    """写成字符串的话，集合运算是按字符走的——"page" 会变成四个单字符字段名。"""
    _rejects(
        _scenario_dict(exemptions=[_rule(match={"arg_differs_on": "page"})]),
        tmp_path,
        "字段名列表",
    )


# --- 装配层：runner 真的把它传下去了吗 ----------------------------------------


def _calls_of(item) -> list[Any]:
    return item.calls


def test_the_runner_passes_the_scenarios_exemptions_to_the_detectors() -> None:
    """**这条是这个文件存在的理由。**

    字段加上去、检测器也认，但 runner 那一行忘了传，表现就是"场景里写了豁免，
    跑出来照报不误"——而所有分层的测试都还是绿的：加载测的是能不能读进来，
    检测器测的是拿到规则会不会静音，两边都不经过 runner 那一行。
    """
    item = _item("loop-approximate-non-monotonic")
    rule = Exemption(
        detector="loop.approximate",
        tool="list_orders",
        reason="人工确认过这个工具的翻页方式",
        match={"arg_differs_on": ["page"]},
    )
    scenario = Scenario.model_validate(
        _scenario_dict(exemptions=[asdict(rule)])
    )

    findings = _run_detectors(scenario, _calls_of(item))
    approximate = [f for f in findings if f.detector == "loop.approximate"]

    assert len(approximate) == 1, "近似层的结论条数变了"
    assert approximate[0].verdict == DetectorVerdict.PASS, "豁免没生效"
    assert approximate[0].exempted_by == rule.label


def test_both_front_doors_take_the_same_fields() -> None:
    """场景 YAML 和标注语料写豁免用的是同一组字段名。

    多一个少一个都要红：两个前门不一致的话，规则只在一处生效，而两边的测试
    都还是绿的（各自只测自己那套写法）。
    """
    from sandbox.corpus import EXEMPTION_KEYS
    from sandbox.scenario.spec import ExemptionDecl

    assert set(ExemptionDecl.model_fields) == set(EXEMPTION_KEYS)


def test_a_scenario_rule_reproduces_the_corpus_label() -> None:
    """同一份豁免、同一条轨迹：场景侧的结果必须等于语料那侧的标注。

    "一个机制、两个前门"的守卫。字段映射写错一处（比如场景用 `match`、检测器
    读 `params`），规则就只在一边生效，而两边的测试都还是绿的——语料只测语料
    那套写法，场景只测场景那套。
    """
    item = _item("loop-approximate-paging-exempted")
    labeled = {e.detector: e.verdict for e in item.expect}
    assert item.exemptions.rules, "对照样本应当带着一条豁免，否则这个测试是空转的"

    scenario = Scenario.model_validate(
        _scenario_dict(exemptions=[asdict(item.exemptions.rules[0])])
    )
    findings = _run_detectors(scenario, _calls_of(item))
    approximate = next(f for f in findings if f.detector == "loop.approximate")

    assert approximate.verdict == labeled["loop.approximate"]
    assert approximate.exempted_by is not None, "静音了却没留痕"


# --- 语义层 -------------------------------------------------------------------


def test_the_reason_travels_with_the_silenced_finding() -> None:
    """留痕要带理由。只印规则名的话，读报告的人还得翻回场景文件才知道为什么允许。"""
    item = _item("loop-approximate-non-monotonic")
    reason = "运营手动确认过，这个工具的翻页会来回"
    scenario = Scenario.model_validate(
        _scenario_dict(
            exemptions=[
                {
                    "detector": "loop.approximate",
                    "tool": "list_orders",
                    "reason": reason,
                    "match": {"arg_differs_on": ["page"]},
                }
            ]
        )
    )
    findings = _run_detectors(scenario, _calls_of(item))
    silenced = next(f for f in findings if f.exempted_by)

    assert reason in silenced.summary
    assert "静音" in silenced.summary


def test_a_rule_that_does_not_match_leaves_the_finding_alone() -> None:
    """规则写歪了不该顺手静音别的东西——匹配不上就是不生效，和没写一样。"""
    item = _item("loop-approximate-non-monotonic")
    scenario = Scenario.model_validate(
        _scenario_dict(
            exemptions=[
                _rule(match={"arg_differs_on": ["region"]})  # 实际变的是 page
            ]
        )
    )
    findings = _run_detectors(scenario, _calls_of(item))
    approximate = next(f for f in findings if f.detector == "loop.approximate")

    assert approximate.verdict == DetectorVerdict.FAIL
    assert approximate.exempted_by is None


def test_a_write_needs_allow_write_to_be_silenced() -> None:
    """写操作的豁免是静音重复扣款告警的最短路径，所以要多一道门。"""
    item = _item("idem-unprotected-write-repeat")
    rule = {
        "detector": "idempotency",
        "tool": "create_payment",
        "reason": "上游有去重",
    }

    without = Scenario.model_validate(
        _scenario_dict(
            tools=[_tool("create_payment", is_write=True)],
            sessions=[{"name": "s", "script": [{"tool_calls": []}]}],
            exemptions=[rule],
        )
    )
    silenced = Scenario.model_validate(
        _scenario_dict(
            tools=[_tool("create_payment", is_write=True)],
            sessions=[{"name": "s", "script": [{"tool_calls": []}]}],
            exemptions=[{**rule, "allow_write": True}],
        )
    )

    untouched = next(
        f for f in _run_detectors(without, _calls_of(item)) if f.detector == "idempotency"
    )
    muted = next(
        f for f in _run_detectors(silenced, _calls_of(item)) if f.detector == "idempotency"
    )

    assert untouched.verdict == DetectorVerdict.FAIL, "没写 allow_write 却静音了写操作"
    assert muted.verdict == DetectorVerdict.PASS
    assert muted.exempted_by is not None


# --- 指纹：静音不算修复 -------------------------------------------------------


def test_adding_an_exemption_is_a_change_to_the_grading_standard() -> None:
    """豁免改的是"什么算失败"，不是被测对象。落在指纹里，这次对比才不会被
    读成"失败少了几个"——那正是把静音读成修复。"""
    before = identity.sections(Scenario.model_validate(_scenario_dict()))
    after = identity.sections(
        Scenario.model_validate(_scenario_dict(exemptions=[_rule()]))
    )

    assert identity.changed_sections(before, after) == ["expect"]
    assert "expect" in identity.TEST_SECTIONS, (
        "豁免落在的节必须属于「测试本身变了」——否则加了豁免的运行会被判成可比"
    )
