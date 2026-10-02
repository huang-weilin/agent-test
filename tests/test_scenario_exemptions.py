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
from sandbox.detectors.base import Exemption, ExemptionSet, UnanswerableMatch
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


def test_an_empty_field_list_is_rejected(tmp_path: Path) -> None:
    """`arg_differs_on: []` 是一条写得出来、却永远不成立的条件。

    「差异只在零个字段上」就是「两次调用完全相同」，而豁免那一侧**刻意**不让
    这种重复命中（参数一模一样的重复是确定性最高的循环，静音它等于把最该报的
    那件事关掉，见 `ExemptionSet.find` 里那个 `not differing_fields`）。所以这条
    规则是死的：四种差异组合实测 0 次命中。

    与写在答不了的检测器上是同一种坏，所以同一种处理。**空表和没给是两件事**，
    这里也是：不写 `arg_differs_on` 是"不加这个条件"，写 `[]` 是"加一个永远不
    成立的条件"——后者是装饰，而装饰会让读报告的人以为这件事有人管过。
    """
    _rejects(
        _scenario_dict(exemptions=[_rule(match={"arg_differs_on": []})]),
        tmp_path,
        "空列表",
        "arg_differs_on",
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


# --- match 键：答不了它的检测器不许写 ------------------------------------------


#: 这三种写法在补上闸门之前都是**常量**：规则看着像条件，跑起来永远同一个结果。
#: 左边是检测器名，右边是它答不了的键。
#:
#: 具体是哪一种常量，实测过（拆掉两道闸门，每种写法跑三种调用组合）：**全是死规则，
#: 一次都不命中**。调用点没传那个事实，比对的那一边就是 `None`，而 `None` 跟
#: `True`、`False` 都不相等。写 `false` 并不会变成"无条件静音"——那种更刺激的说法
#: 一开始就写进了注释和文档里，是后来真的跑了一遍才改掉的。
CANNOT_ANSWER: list[tuple[str, dict[str, Any]]] = [
    ("loop.strict", {"tool_has_idempotency": False}),
    ("loop.strict", {"tool_has_idempotency": True}),
    ("loop.ping_pong", {"arg_differs_on": ["page"]}),
]


@pytest.mark.parametrize(("detector", "match"), CANNOT_ANSWER)
def test_a_key_the_detector_cannot_answer_is_refused_at_load(
    detector: str, match: dict[str, Any], tmp_path: Path
) -> None:
    """键认得，这个检测器答不了——加载时就拒，不等它跑。

    这两种写法的共同点是**报告上一个字都看不出来**：规则是死的（永远不命中，
    因为调用点没把那个事实传下来，比对的那一边是 `None`，跟 true/false 都不
    相等），而"没有这条结论"和"这条结论被豁免了"在报告里是同一个样子——
    写规则的人于是以为自己处理过了。

    一条不会命中的规则比没有这条规则更坏，所以在这里拒。报错要说清楚谁答得了——
    只写"不支持"的话，写规则的人只会把键删掉，而他要的可能是换个检测器。
    """
    _rejects(
        _scenario_dict(exemptions=[_rule(detector=detector, match=match)]),
        tmp_path,
        "答不了",
        sorted(match)[0],
    )


def test_the_refusal_names_a_detector_that_actually_answers(tmp_path: Path) -> None:
    """报错里点名的那些检测器，写上去必须真的能用。

    这条防的是"把话说一半"：报错说"换一个检测器"，而它点的那个检测器自己
    也答不了——比不提示更坏，用户会照着改，然后再撞一次墙。
    """
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(
            _write(
                tmp_path,
                _scenario_dict(
                    exemptions=[
                        _rule(detector="loop.ping_pong", match={"arg_differs_on": ["page"]})
                    ]
                ),
            )
        )
    message = str(excinfo.value)
    assert "loop.approximate" in message, message

    # 报错点名的那一个，照着写要能加载通过
    load_scenario(
        _write(
            tmp_path,
            _scenario_dict(
                exemptions=[
                    _rule(detector="loop.approximate", match={"arg_differs_on": ["page"]})
                ]
            ),
        )
    )


def test_an_answering_detector_really_answers() -> None:
    """表里写着"答得了"的那几对，跑一遍真的不许抛。

    上面那批测的是"拒得对不对"，这条测的是**拒的另一半**：表里的每一对都得是
    真的。少了它，把表清空（全部拒绝）也是绿的——一个只会的"不"的闸门看起来
    和一条严格的闸门一模一样。
    """
    item = _item("loop-approximate-non-monotonic")
    scenario = Scenario.model_validate(
        _scenario_dict(
            exemptions=[
                _rule(detector="loop.approximate", match={"arg_differs_on": ["page"]})
            ]
        )
    )
    findings = _run_detectors(scenario, _calls_of(item))
    assert any(f.detector == "loop.approximate" for f in findings), "这条轨迹该出近似层结论"

    idem = _item("idem-exemption-scoped-to-unprotected")
    scenario = Scenario.model_validate(
        _scenario_dict(
            tools=[_tool("create_payment", is_write=True)],
            exemptions=[
                _rule(
                    detector="idempotency",
                    tool="create_payment",
                    allow_write=True,
                    match={"tool_has_idempotency": False},
                )
            ],
        )
    )
    findings = _run_detectors(scenario, _calls_of(idem))
    silenced = next(f for f in findings if f.detector == "idempotency")
    assert silenced.exempted_by is not None, "幂等这一族答得了这个键，规则该命中"

    # 答得了"零个键"的那几个名字同样是**真的**：它们查豁免，只是没有任何
    # 当场知道的事可以当条件。这一条防的是把 `frozenset()` 读成"答不了任何东西
    # 所以拒了吧"——那样 `autonomy.unauthorized` 这一类就只能靠
    # `may_run_unattended: true` 表达，豁免里多出来的那个"为什么"就没了。
    ping = _item("loop-ping-pong")
    scenario = Scenario.model_validate(
        _scenario_dict(
            tools=[_tool("get_order", is_write=False), _tool("get_user", is_write=False)],
            exemptions=[
                _rule(detector="loop.ping_pong", tool="get_order", match={}),
            ],
        )
    )
    findings = _run_detectors(scenario, _calls_of(ping))
    ping_pong = [f for f in findings if f.detector == "loop.ping_pong"]
    assert ping_pong, "这条轨迹该出乒乓结论"
    assert ping_pong[0].exempted_by is not None, "乒乓这一路会查豁免，零个键也该命中"

    auto = _item("auto-unauthorized-write")
    scenario = Scenario.model_validate(
        _scenario_dict(
            tools=[_tool("create_payment", is_write=True, may_run_unattended=False)],
            # 第二段会话没人看着——"无人值守"是场景的显式声明，不是从调用里猜的。
            sessions=[{"name": "有人看着", "script": []}, {"name": "没人看着", "autonomous": True}],
            exemptions=[
                _rule(
                    detector="autonomy.unauthorized",
                    tool="create_payment",
                    allow_write=True,
                    match={},
                )
            ],
        )
    )
    findings = _run_detectors(scenario, _calls_of(auto))
    unauthorized = [f for f in findings if f.detector == "autonomy.unauthorized"]
    assert unauthorized, "这条轨迹该出越权结论"
    assert unauthorized[0].exempted_by is not None, "越权这一路会查豁免，零个键也该命中"


# --- 检测器名本身：不查豁免的那几条路 ------------------------------------------


#: `KNOWN_DETECTORS` 里**压根没有查豁免那一步**的名字。
#:
#: 这张表和 `EXEMPTION_AWARE` 是一件事的两面：那边记"查，而且答得了这些键"，
#: 这边记"不查"。分开写是因为它们的错法不同——那边写错会把规则变成常量，
#: 这边写错只是规则永远不命中。对写规则的人来说后果一样：**规则看起来在，
#: 实际不按他以为的走**，而报告里一个字都看不出来。
#:
#: 注意 `loop` 和 `loop.strict` 看着像父子，行为却不同：只有严格层那条路会去
#: 找豁免。凭名字猜是不行的，所以这张表必须逐条写下来、并且被测试钉住。
NO_EXEMPTION_STEP: list[str] = [
    "loop",
    "loop.retry",
    "loop.cross_session",
    "loop.snapshot_replay",
    "loop.unknown_replay",
    "idempotency.read_repeats",
]


@pytest.mark.parametrize("detector", NO_EXEMPTION_STEP)
def test_a_detector_that_never_asks_is_refused_at_load(
    detector: str, tmp_path: Path
) -> None:
    """这几条路上没有查豁免那一步，所以规则收下来也没人看。

    **收下一条永远不会命中的规则，比拒了更坏**：报告里"没有这条结论"和"这条
    结论被豁免了"是同一个样子，写规则的人于是以为自己处理过了。要豁免这一族，
    得先给它把那一小步接上——那是改代码，不是改 YAML。
    """
    _rejects(
        _scenario_dict(exemptions=[_rule(detector=detector, match={})]),
        tmp_path,
        "不查豁免",
        detector,
    )


def test_every_known_detector_is_classified() -> None:
    """15 个检测器名，每一个都得说清楚它属于哪一类。

    这条是防漂的：以后有人加一个检测器、或者把某条路接上豁免，这张分类就必须
    跟着动。少了它，新名字会静静地落进"没被任何一张表提到"的缝里——那时它是
    `EXTRA` 之外的东西，两边都不拦，规则写上去照样不生效，而没有任何测试会红。
    """
    from sandbox.detectors import KNOWN_DETECTORS
    from sandbox.detectors.base import EXEMPTION_AWARE, NOT_EXEMPTIBLE

    classified = set(EXEMPTION_AWARE) | set(NOT_EXEMPTIBLE) | set(NO_EXEMPTION_STEP)
    assert classified == set(KNOWN_DETECTORS), (
        f"没被分类的：{sorted(set(KNOWN_DETECTORS) - classified)}；"
        f"表里有、检测器列表里没有的：{sorted(classified - set(KNOWN_DETECTORS))}"
    )


def test_the_refusal_of_an_unaware_detector_is_not_an_unknown_detector(
    tmp_path: Path,
) -> None:
    """报错得说对是哪一种错——"不查豁免"和"没这个名字"要走完全相反的修法。

    说成"检测器名不认识"的话，写规则的人会去翻名字列表，然后换成另一个同样
    不查豁免的名字，再撞一次墙。所以报错要点名今天**真的会查**的那几个。
    """
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(
            _write(
                tmp_path,
                _scenario_dict(exemptions=[_rule(detector="loop.retry", match={})]),
            )
        )
    message = str(excinfo.value)
    assert "loop.retry" in message, message
    assert "loop.strict" in message, f"报错该点名真会查豁免的那几个：\n{message}"
    assert "不认识的检测器" not in message, message


# --- 运行时那道闸门：调用点忘了传事实 -----------------------------------------


def test_find_raises_when_the_call_site_did_not_answer() -> None:
    """加载期那张表拦的是**人挑错了检测器**；这里拦的是**调用点忘了传事实**。

    两件事不一样：表可以漂（有人加了一个新的 find 调用点，而那个调用点不传
    某个事实），而"传没传"在那个调用点上是一眼可见的。少了这一道，那条规则
    就会悄悄退化成常量，报告上什么都看不出来——而豁免机制是专门用来让结论
    消失的东西，它最不能出的就是这个毛病。
    """
    rules = ExemptionSet(
        [
            Exemption(
                detector="loop.approximate",
                tool="get_order",
                reason="人工确认过",
                match={"arg_differs_on": ["page"]},
            )
        ]
    )
    with pytest.raises(UnanswerableMatch) as excinfo:
        rules.find("loop.approximate", "get_order", is_write=False)
    assert "arg_differs_on" in str(excinfo.value)


def test_an_empty_answer_is_still_an_answer() -> None:
    """空元组是"答过了：没有差异"，不是"答不了"——判的是 `is None`，不是真假。

    这一条是上面那道闸门最容易写错的地方：写成判真假的话，"这次调用两次参数
    完全一样"会被读成"这个调用点答不了"，于是一次本该正常匹配的豁免直接炸掉
    整个运行。空表和没给是两件事，这里也是同一条。
    """
    rules = ExemptionSet(
        [
            Exemption(
                detector="loop.approximate",
                tool="get_order",
                reason="人工确认过",
                match={"arg_differs_on": ["page"]},
            )
        ]
    )
    assert rules.find("loop.approximate", "get_order", is_write=False, differing_fields=()) is None
    assert (
        rules.find("loop.approximate", "get_order", is_write=False, differing_fields=("page",))
        is not None
    )


def test_a_false_answer_is_still_an_answer() -> None:
    """`False` 同一条道理：工具没有声明幂等保护，是一个答案，不是"没答"。"""
    rules = ExemptionSet(
        [
            Exemption(
                detector="idempotency",
                tool="create_payment",
                reason="上游有去重",
                allow_write=True,
                match={"tool_has_idempotency": False},
            )
        ]
    )
    assert (
        rules.find("idempotency", "create_payment", is_write=True, tool_has_idempotency=False)
        is not None
    )
    assert (
        rules.find("idempotency", "create_payment", is_write=True, tool_has_idempotency=True)
        is None
    )
