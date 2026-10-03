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
from sandbox.scenario import identity
from sandbox.scenario.expect import ExpectResult, evaluate
from sandbox.scenario.loader import ScenarioError, load_all, load_scenario
from sandbox.scenario.runner import ScenarioResult, SessionOutcome, render
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


def test_the_warn_policy_is_not_a_value(tmp_path: Path) -> None:
    """`unexpected_failures: warn` 曾经是个合法值，而它和 fail 一模一样。

    它从加进来那天起就没接上：判定只看意外那一栏空不空，warn 落进和 fail 完全
    相同的分支。所以写了它的场景跑出来和 fail 一模一样——配置指纹里记着它、
    报告上什么都没有，而写的人以为"报了但不拦"。现在它在加载时就被拒。

    **这条测试红了，就说明有人把 warn 加回了类型里。** 加回去之前先回答那个问题：
    谁是它的诚实产出方？知道会失败就写 `expect`（note 里写为什么），不想被拦就
    写 `allow`，中间那一档的意思自相矛盾。
    """
    data = _scenario_dict(unexpected_failures="warn")
    with pytest.raises(ScenarioError, match="没有 warn"):
        load_scenario(_write(tmp_path, data))


def test_an_unknown_policy_is_rejected(tmp_path: Path) -> None:
    # 认不出来的取值静默通过的话，写错的人会以为它生效了——而它落进的其实是
    # "不是 allow 就是 fail"那条路，也就是默认那一个。
    data = _scenario_dict(unexpected_failures="banana")
    with pytest.raises(ScenarioError, match="unexpected_failures"):
        load_scenario(_write(tmp_path, data))


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


def test_an_expectation_on_an_unknown_detector_is_rejected(tmp_path: Path) -> None:
    """预期写错检测器名 = 这条判卷标准**根本不存在**。

    这是「装上了但没接线」最安静的一种：豁免写错了检测器，那条 fail 照样出现在
    报告里（用户看得见问题还在）；预期写错了检测器，那条预期既不在「命中」也不在
    「未命中」——报告上**完全没有它的位置**。用户以为自己声明过「这里不该报警」，
    而实际上这次运行对那件事一个字都没说过。
    """
    data = _scenario_dict(expect=[{"detector": "idempotancy", "verdict": "fail"}])
    with pytest.raises(ScenarioError, match="未知检测器"):
        load_scenario(_write(tmp_path, data))


@pytest.mark.parametrize(
    "name",
    [
        # 边界上那四个。**这条参数化是被一次真事故逼出来的**：先前的计划是照
        # `KNOWN_DETECTORS` 校验，而实测 29 份场景里有 6 处合法声明写的就是
        # `tool_schema`——那个集合里没有它。只照那一份清单判，会当场冤枉 6 处。
        "tool_scope",
        "tool_schema",
        "egress",
        "token_budget",
        # 一个普通的、`KNOWN_DETECTORS` 里的名字，作为对照。
        "idempotency",
    ],
)
def test_the_legal_detector_names_load(name: str, tmp_path: Path) -> None:
    """合法名字一个都不许被冤枉。

    只有反向那条（写错的被拒）的话，一个"什么都不放行"的实现也能过——而那种实现
    会让每一份现有场景都加载不了。
    """
    data = _scenario_dict(expect=[{"detector": name, "verdict": "fail"}])
    assert load_scenario(_write(tmp_path, data)).expect[0].detector == name


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


def _paging_tool(name: str, **kw: Any) -> dict[str, Any]:
    """一个只有 `page` 一个参数的工具。字段权重那一组测试都用它。"""
    tool = _tool(name, is_write=False)
    tool["parameters"] = {
        "type": "object",
        "properties": {"page": {"type": "integer", "description": "页码"}},
    }
    tool.update(kw)
    return tool


def _paging_scenario(*tools: dict[str, Any]) -> dict[str, Any]:
    """一份只调用**第一个**工具的场景，脚本一步不多。"""
    return _scenario_dict(
        tools=list(tools),
        sessions=[
            {
                "name": "翻页",
                "script": [
                    {
                        "tool_calls": [
                            {
                                "name": tools[0]["name"],
                                "arguments": {"page": 1},
                            }
                        ]
                    }
                ],
            }
        ],
    )


def test_a_weight_on_a_field_that_does_not_exist_is_rejected(tmp_path: Path) -> None:
    """权重挂在不存在的字段上：声明看着在，实际什么都没改。

    检测器拿到 `{"pge": 0.1}` 之后，`pge` 一次都不会出现在参数里，而 `page`
    照旧权重 1.0——一串翻页于是照样被切成好几个簇，够不着告警条数。
    **报告上完全看不出来**：它只是"没报警"。
    """
    data = _paging_scenario(_paging_tool("list_orders", field_weights={"pge": 0.1}))
    with pytest.raises(ScenarioError, match="没有这个字段"):
        load_scenario(_write(tmp_path, data))


def test_two_tools_disagreeing_on_one_weight_is_rejected(tmp_path: Path) -> None:
    """两个工具给同一个字段名写了不同的权重：合不出来，谁赢看书写顺序。

    检测器拿到的是一张扁表（字段名 → 权重），没有"按工具分别调参"这回事。
    静默地让后写的赢，等于**同一份场景挪一下工具顺序就换一种判法**。
    """
    data = _paging_scenario(
        _paging_tool("list_orders", field_weights={"page": 0.1}),
        _paging_tool("list_tickets", field_weights={"page": 0.5}),
    )
    with pytest.raises(ScenarioError, match="不同的权重"):
        load_scenario(_write(tmp_path, data))


def test_two_tools_agreeing_on_one_weight_is_fine(tmp_path: Path) -> None:
    """反向那一半：两边说的一样时没有歧义，合成一张表就是对的。

    只有上面那条的话，一个"见到 field_weights 就报错"的校验也能全绿。
    """
    data = _paging_scenario(
        _paging_tool("list_orders", field_weights={"page": 0.1}),
        _paging_tool("list_tickets", field_weights={"page": 0.1}),
    )
    scenario = load_scenario(_write(tmp_path, data))

    assert scenario.loop_field_weights() == {"page": 0.1}


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), 0, -1])
def test_a_weight_no_similarity_can_use_is_rejected(
    tmp_path: Path, weight: float
) -> None:
    """`nan` 最坏：`nan >= 阈值` 恒为假，于是**所有簇都不成立、近似层整体静默**。

    报告上看起来只是"这一段没发现循环"。`inf` 与"删掉那个字段"等价，`0` 与负数
    没有意义——都在加载时报错，不留给运行期。
    """
    data = _paging_scenario(_paging_tool("list_orders", field_weights={"page": weight}))
    with pytest.raises(ScenarioError, match="正的有限数"):
        load_scenario(_write(tmp_path, data))


def test_a_numeric_string_weight_is_coerced_by_the_model(tmp_path: Path) -> None:
    """`page: "0.1"` 在场景文件里等价于 `page: 0.1`——**这是 pydantic 的宽松模式**，
    不是这里的选择，所以把它写成一条测试而不是一句注释。

    真正危险的那个输入（`"abc"` 这种换不出数字的）在同一个地方被挡住，不用单测。
    字符串要在这里被认出来的话，得对这一个字段开严格模式，而严格模式会连
    `page: 1` 这种**整数写法**一起拒掉——那个写法是自然的，不该为了这个被拒。
    所以"权重必须是数字"那一条只对**不过模型的那条路**（运行注册的 JSON 载荷）
    生效，见 tests/test_loop_paging.py。
    """
    data = _paging_scenario(_paging_tool("list_orders", field_weights={"page": "0.1"}))
    scenario = load_scenario(_write(tmp_path, data))

    assert scenario.loop_field_weights() == {"page": 0.1}


def test_a_weight_that_is_not_a_number_at_all_is_rejected(tmp_path: Path) -> None:
    """反向：`"abc"` 换不出数字，模型这一层就报错——它不会是静默的。"""
    data = _paging_scenario(_paging_tool("list_orders", field_weights={"page": "abc"}))
    with pytest.raises(ScenarioError, match="场景定义不合法"):
        load_scenario(_write(tmp_path, data))


def test_the_weights_are_part_of_the_grading_standard(tmp_path: Path) -> None:
    """权重进配置指纹的 `tools` 节：改了它等于换了"什么算重复"。

    不进指纹的话，一次"把 page 压到 0.1"的改动会被读成"这一版失败少了几个"——
    而实际上是判卷标准变了。
    """
    with_weight = load_scenario(
        _write(
            tmp_path,
            _paging_scenario(_paging_tool("list_orders", field_weights={"page": 0.1})),
            "w.yaml",
        )
    )
    without = load_scenario(
        _write(tmp_path, _paging_scenario(_paging_tool("list_orders")), "n.yaml")
    )

    assert identity.changed_sections(
        identity.sections(without), identity.sections(with_weight)
    ) == ["tools"]


# --- 主键声明：判据放宽的那一档，声明必须是真的 ---------------------------------


def _booking_tool(**kw: Any) -> dict[str, Any]:
    """一个有 schema 的写工具——主键要对得上参数，所以这里得有字段表。"""
    tool = _tool("book_reservation", is_write=True)
    tool["parameters"] = {
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "flights": {"type": "array"},
            "payment_methods": {"type": "array"},
        },
    }
    tool.update(kw)
    return tool


def _booking_scenario(*tools: dict[str, Any]) -> dict[str, Any]:
    """一份只调用**第一个**工具的场景，参数里有主键。"""
    return _scenario_dict(
        tools=list(tools),
        sessions=[
            {
                "name": "订票",
                "script": [
                    {
                        "tool_calls": [
                            {"name": tools[0]["name"], "arguments": {"user_id": "U1"}}
                        ]
                    }
                ],
            }
        ],
    )


def test_an_identity_field_that_is_not_a_parameter_is_rejected(tmp_path: Path) -> None:
    """主键挂在不存在的字段上（这里是 `fligths` 打错了）：这一档判据一次都不生效。

    分组时取不到那个字段的值，于是每一组都退回逐字那一档——**报告上看不出来**，
    它只会是"没报警"，和「按主键比过了、没问题」长得一模一样。
    """
    data = _booking_scenario(_booking_tool(identity_fields=["user_id", "fligths"]))
    with pytest.raises(ScenarioError, match="参数里没有这些字段"):
        load_scenario(_write(tmp_path, data))


def test_an_identity_declaration_that_matches_the_parameters_loads(tmp_path: Path) -> None:
    """反向：字段对得上时正常加载。

    只有上面那条的话，一个"见到 identity_fields 就报错"的校验也能全绿。
    """
    data = _booking_scenario(_booking_tool(identity_fields=["user_id", "flights"]))
    scenario = load_scenario(_write(tmp_path, data))

    assert scenario.tools[0].identity_fields == ["user_id", "flights"]


def test_an_identity_declaration_without_a_schema_is_rejected(tmp_path: Path) -> None:
    """工具没声明 `parameters` 时报错——没有可以对照的字段表。

    这里的取舍要写下来：**没有 schema 就判不了对错**，而放行等于收下一份没人
    核对得了、也可能永远不生效的声明。想按主键判，先把参数声明出来。
    """
    tool = _booking_tool(identity_fields=["user_id"])
    del tool["parameters"]
    with pytest.raises(ScenarioError, match="没有声明 parameters"):
        load_scenario(_write(tmp_path, _booking_scenario(tool)))


def test_a_read_only_tool_cannot_declare_a_key(tmp_path: Path) -> None:
    """只读工具不许声明主键：读不改变世界，没有"重复执行"可言。

    这条钉的是**校验真的接在加载路径上**——`checked_identity_fields` 自己那几条
    在被调用之前什么都不算，而一份收了它的场景会让这个工具的每次**读**调用都被
    拿去分主键组，报出来的却是写操作那一族的话。
    """
    data = _paging_scenario(_paging_tool("list_orders", identity_fields=["page"]))
    with pytest.raises(ScenarioError, match="只读"):
        load_scenario(_write(tmp_path, data))


def test_the_declared_key_is_part_of_the_grading_standard(tmp_path: Path) -> None:
    """主键声明进配置指纹的 `tools` 节：改了它等于换了"什么算同一个操作"。

    不进指纹的话，一次「声明 user_id 是主键」的改动会被读成"失败少了几个"——
    而实际上判卷标准变了：原先配不上对的几次调用，现在配得上了。
    """
    declared = load_scenario(
        _write(
            tmp_path,
            _booking_scenario(_booking_tool(identity_fields=["user_id"])),
            "declared.yaml",
        )
    )
    bare = load_scenario(_write(tmp_path, _booking_scenario(_booking_tool()), "bare.yaml"))

    assert identity.changed_sections(
        identity.sections(bare), identity.sections(declared)
    ) == ["tools"]


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


def test_a_count_in_an_expectation_is_honored() -> None:
    """场景的 `expect` 也能写 `count`：**问几条就得有几条**。

    场景这一边本来就有兜底（多出来的 fail 进 `unexpected`），所以 `count` 在这里
    多半用不上。它仍然要能写、写得对：语料和场景共用同一套预期语义，一边认得、
    另一边不认的话，"同一个词在两处意思不同"就是这么长出来的。

    两半都要：够数时通过；**不够数时按缺的条数报**（要两条只来一条，不是"没报"）。
    """
    scenario = _scenario_model([{"detector": "loop.strict", "verdict": "fail", "count": 2}])
    enough = evaluate(
        scenario,
        [
            _finding("loop.strict", DetectorVerdict.FAIL, summary="a"),
            _finding("loop.strict", DetectorVerdict.FAIL, summary="b"),
        ],
    )
    assert enough.ok
    assert len(enough.matched) == 2

    short = evaluate(scenario, [_finding("loop.strict", DetectorVerdict.FAIL, summary="a")])
    assert not short.ok
    assert len(short.missing) == 1, "要两条、来了一条：缺的是一条，不是两条"
    assert len(short.matched) == 1


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


# --- 报告：预期那一节 -------------------------------------------------------------


def _result(scenario: Scenario, findings: list[Finding]) -> ScenarioResult:
    return ScenarioResult(
        scenario_id=scenario.id,
        run_id="r",
        sessions=[_outcome()],
        findings=findings,
        expectation=evaluate(scenario, findings),
    )


def _section(text: str, title: str) -> str:
    """报告里某一节的内容（报表用 `--- 标题 ---` 分段，到下一节为止）。"""
    return text.split(f"--- {title} ---")[1].split("\n--- ")[0]


def test_the_expectation_section_keeps_its_four_columns_apart() -> None:
    """四个栏位各有各的说法，一个都不能少——尤其是最后两个。

    「意外」是 fail，「未覆盖」是 inconclusive。把它们并成一栏（都是"没被预期
    覆盖的结论"）看着更简洁，实际是把"没测到"读成了"有问题"，反过来也一样。
    这一节此前没有任何渲染测试：合并了两栏也不会有条红的。
    """
    scenario = _scenario_model([{"detector": "idempotency", "verdict": "pass"}])
    text = render(
        _result(
            scenario,
            [
                _finding("idempotency", DetectorVerdict.PASS, summary="只付了一次"),
                _finding("loop.strict", DetectorVerdict.FAIL, summary="停不下来"),
                _finding(
                    "loop.cross_session", DetectorVerdict.INCONCLUSIVE, summary="只有一段会话"
                ),
            ],
        )
    )
    section = _section(text, "预期")
    assert "命中  idempotency/pass" in section
    assert "意外  [fail] loop.strict：停不下来" in section
    assert "未覆盖 [inconclusive] loop.cross_session：只有一段会话" in section
    assert "未命中" not in section


def test_an_allow_policy_stops_judging_not_reporting() -> None:
    """`allow` 的意思是"这次不判"，不是"这次不看"。

    那条结论仍然出现在「检测结论」一节里。两节的读法不一样：检测结论列的是
    **看见了什么**，预期一节判的是**该不该看见**。把 allow 实现成"从报告里
    拿掉"，一个真实的 Agent 缺陷会因为场景里写了一个词而消失。
    """
    scenario = _scenario_model([], unexpected_failures="allow")
    finding = _finding("loop.strict", DetectorVerdict.FAIL, summary="停不下来")
    result = _result(scenario, [finding])

    assert result.ok
    text = render(result)
    assert "loop.strict" in _section(text, "检测结论")
    assert "意外" not in _section(text, "预期")


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
