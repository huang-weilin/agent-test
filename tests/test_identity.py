"""配置指纹：改了配置认不认得出，认出了说得出改的是哪一块。

**为什么这个文件值得存在。** 指纹夹在两个方向相反的误报中间，两边都致命：

* 认不出来 —— "配置没变，但结论变了"这条最该被看到的信号直接消失。用户改了
  Prompt 重跑，报告说配置一致，于是他去查 Agent 的代码，而问题在他自己手上。
* 认错了 —— 调一下缩进、换一下 YAML 键序就被报成"配置变了"，于是真正的行为
  差异被归到配置头上，用户去查一个没动过的文件。

这两条都有对应的用例，而且它们必须同时成立：只满足一条的指纹是没用的。

不碰子进程、不碰数据库，纯函数，所以跑得飞快。
"""

from __future__ import annotations

import json

import pytest

from sandbox.scenario import identity
from sandbox.scenario.loader import parse_scenario
from sandbox.scenario.spec import Scenario
from sandbox.trace.recorder import CONFIG_SCHEME, compute_config_hash

# 底本。五个节各有一个可以单独改动的地方，下面的变体各自只动一处。
BASE = """
id: clean
description: 底本
agent:
  version: react_demo@0.1
  task: 查一下订单 A1。
  model: stub-model
tools:
  - name: get_order
    description: 查订单
    is_write: false
sessions:
  - name: 第一次
    script:
      - tool_calls: [{name: get_order, arguments: {order_id: A1}}]
expect:
  - detector: idempotency
    verdict: pass
"""

#: 语义相同、**字节不同**的一份写法。每一样都是真实会发生的事：编辑器自动
#: 调整缩进、格式化工具重排键、有人顺手加了行注释、YAML 里给标量加引号。
REFORMATTED = """
id: clean
description: 底本
agent:
  model: stub-model
  task: "查一下订单 A1。"    # 同样的任务，只是被加了引号
  version: react_demo@0.1
tools:
  - is_write: false
    name: get_order
    description: 查订单
sessions:
  # 会话一
  - script:
      - tool_calls: [{arguments: {order_id: "A1"}, name: get_order}]
    name: 第一次
expect:
  - verdict: pass
    detector: idempotency
"""

#: 只改一处的变体 → (底本里的原文, 换成什么, 期望被认出来的那一节)。
MUTATIONS: dict[str, tuple[str, str, str]] = {
    "改了 Prompt": ("查一下订单 A1。", "查一下订单 A2。", "prompt"),
    "改了模型": ("model: stub-model", "model: other-model", "prompt"),
    "改了工具声明": ("    is_write: false", "    is_write: true", "tools"),
    "改了脚本": ("order_id: A1", "order_id: B9", "script"),
    "改了期望": ("verdict: pass", "verdict: fail", "expect"),
    "换了 Agent 版本": ("version: react_demo@0.1", "version: react_demo@0.2", "agent"),
}


def _scenario(text: str = BASE, name: str = "<场景>") -> Scenario:
    return parse_scenario(text, source=name)


def _mutated(name: str) -> Scenario:
    old, new, _ = MUTATIONS[name]
    assert old in BASE, f"底本里找不到 {old!r}——变体定义已经和底本脱节了"
    return _scenario(BASE.replace(old, new), name=f"<{name}>")


def _hash(scenario: Scenario) -> str:
    return compute_config_hash(identity.sections(scenario))


# --- 不该报"变了" -------------------------------------------------------------
# 这些用例挡的是"误报成配置变更"。它们比下面那组更容易被写错：字节哈希能
# 通过全部"改了要认出来"的用例，却会在这里全军覆没。


def test_same_scenario_hashes_the_same() -> None:
    assert _hash(_scenario()) == _hash(_scenario())


def test_reformatting_is_not_a_config_change() -> None:
    """YAML 的键序、缩进、引号、注释都不改变它表达的意思。"""
    assert (
        identity.changed_sections(
            identity.sections(_scenario()), identity.sections(_scenario(REFORMATTED))
        )
        == []
    )
    assert _hash(_scenario()) == _hash(_scenario(REFORMATTED))


def test_survives_a_round_trip_through_the_database() -> None:
    """`config_parts_json` 的**真实路径**：写进去的是 JSON 文本，比对时读回来。

    如果 `sections()` 里混进了 JSON 装不下的类型（枚举、日期），写库时会被
    `default=str` 悄悄变成字符串，读回来就和新鲜解析出来的对象不相等——
    于是一份没动过的配置被报成"全变了"。这个往返必须逐字相等。
    """
    parts = identity.sections(_scenario())
    round_tripped = json.loads(json.dumps(parts))  # 不传 default，装不下就报错
    assert round_tripped == parts
    assert identity.changed_sections(round_tripped, parts) == []


def test_no_record_is_not_the_same_as_all_changed() -> None:
    """本字段存在之前的运行没存过配置。

    `None` 与空字典都是"没记录"，必须给出**同一个**答案。答案不能是"没变"
    （没有依据），也不能直说"五节全变了"（读起来像有人把配置全改了）——
    返回全部节名，由调用方判定"可比性未知"，那是第三种答案。
    """
    after = identity.sections(_scenario())
    assert identity.changed_sections(None, after) == list(identity.SECTIONS)
    assert identity.changed_sections({}, after) == list(identity.SECTIONS)
    assert identity.changed_sections(None, after) == identity.changed_sections({}, after)


# --- 该报"变了"，且要指出是哪一节 -------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected_section"), [(n, m[2]) for n, m in MUTATIONS.items()]
)
def test_a_change_is_detected_and_located_to_its_section(
    name: str, expected_section: str
) -> None:
    """一个总数只能回答"变了没有"，而用户要问的是"改哪了"。

    改 Prompt、改工具声明、改脚本、改期望，指向的修复动作完全不同，所以定位
    必须精确到节——**只**报出被改的那一节，不多不少。
    """
    base, mutated = _scenario(), _mutated(name)

    assert _hash(base) != _hash(mutated), f"{name}：指纹没认出来"
    assert identity.changed_sections(
        identity.sections(base), identity.sections(mutated)
    ) == [expected_section], f"{name}：定位到的节不对"


def test_editing_the_script_or_expectations_is_a_test_change() -> None:
    """`script` 与 `expect` 是**测试本身**，不是被测对象。

    它们变了而结论跟着变，什么都说明不了——把这种情况报成"行为回归"是本工具
    最不该犯的错。所以它们必须被认出来，让报告能说"这次比的是两套测试"。
    """
    base = _scenario()
    for name in ("改了脚本", "改了期望"):
        changed = identity.changed_sections(
            identity.sections(base), identity.sections(_mutated(name))
        )
        assert changed, f"{name}：测试定义变了却没被认出来"
        assert changed[0] in ("script", "expect")


def test_catches_what_the_old_scheme_missed() -> None:
    """回归这个指纹存在的理由本身。

    旧口径只哈希场景 id 与模型名，于是"id 和模型都没动，只改了脚本或 Prompt"
    给同一个哈希——而这一类比"改了 id"常见得多，也才是回归对比真正要看的。
    """

    def old(scenario: Scenario) -> str:
        return compute_config_hash({"scenario": scenario.id, "model": scenario.agent.model})

    base = _scenario()
    for name in ("改了 Prompt", "改了工具声明", "改了脚本", "改了期望"):
        mutated = _mutated(name)
        assert old(base) == old(mutated), f"{name}：旧口径本来就抓不到，用例前提不成立"
        assert _hash(base) != _hash(mutated), f"{name}：新口径也没抓到"


def test_several_changed_sections_are_all_listed() -> None:
    """改动不会那么客气地一次只来一处。"""
    text = BASE.replace("查一下订单 A1。", "查一下订单 A2。").replace(
        "    is_write: false", "    is_write: true"
    )
    changed = identity.changed_sections(
        identity.sections(_scenario()), identity.sections(_scenario(text))
    )
    assert changed == ["prompt", "tools"], f"多节改动只列出了 {changed}"


def test_section_order_is_fixed() -> None:
    """差异报告里各节的出现顺序由这个常量决定，顺序变了报告前后就不可比。"""
    assert identity.SECTIONS == ("prompt", "tools", "script", "agent", "expect")
    assert list(identity.sections(_scenario())) == list(identity.SECTIONS)


# --- 环境记录 -----------------------------------------------------------------


def test_environment_says_whether_the_upstream_is_scripted() -> None:
    """脚本化上游意味着"模型行为"是被我们写死的。

    这句话必须准确，否则结论会被外推到真实模型身上——而它们完全是两回事。
    """
    assert identity.environment(_scenario())["llm_upstream"] == "scripted"
    live = _scenario(BASE.replace("model: stub-model", "model: deepseek-chat"))
    assert identity.environment(live)["llm_upstream"] == "live"


def test_environment_carries_the_agents_own_env() -> None:
    """Agent 进程的额外环境变量会影响它怎么跑，属于环境的一部分。"""
    text = BASE.replace("  model: stub-model", "  model: stub-model\n  env: {TEMP: '0.1'}")
    env = identity.environment(_scenario(text))
    assert env["agent_env"] == {"TEMP": "0.1"}
    assert env["model"] == "stub-model"


def test_environment_is_json_storable() -> None:
    """`env_json` 同样要过一遍 JSON。装不下的类型会被 `default=str` 悄悄吃掉。"""
    env = identity.environment(_scenario())
    assert json.loads(json.dumps(env)) == env


def test_the_scheme_version_is_recorded() -> None:
    """口径必须和哈希一起落库。

    少了它，一次旧运行（按旧口径算的哈希）和一次新运行放一起比，会得出
    "配置变了"——那是一句基于不可比数据的结论。见 models.py 里那一列的注释。
    """
    assert CONFIG_SCHEME == "2"
