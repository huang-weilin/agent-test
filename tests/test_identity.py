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
from sandbox.scenario.spec import Scenario, SessionSpec, WorldDecl
from sandbox.trace.recorder import CONFIG_SCHEME, compute_config_hash

# 底本。六个节各有一个可以单独改动的地方，下面的变体各自只动一处。
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
world:
  order_notes:
    A1: 客户备注：工作日送达。
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
world:
  order_notes:
    A1: "客户备注：工作日送达。"    # 同一句话，只是被加了引号
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
    "改了世界里的备注": (
        "A1: 客户备注：工作日送达。",
        "A1: 客户备注：周末也送。",
        "world",
    ),
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


def test_renaming_a_session_is_not_a_test_change() -> None:
    """给会话改个叫法，这次运行一个字节都没变。

    会话名是报告里那句话的标签，不是被测对象拿到的东西。它留在指纹里的时候，
    改它会让回归报告说出"改的是测试本身，两次运行比的不是同一套测试"——而那句
    话会把真信号盖住：用户恰好在同一次重跑里改过 Prompt 的话，报告已经在说
    "比不了"，他就不会再去查那个 Prompt 了。

    两半都得有：**改名不许报**，而同一个会话里的脚本动一下**必须报**（否则
    上面那句"不报"只是因为这一节根本不看东西）。
    """
    renamed = _scenario(BASE.replace("- name: 第一次", "- name: 换个叫法"))
    assert _hash(_scenario()) == _hash(renamed), "改个会话名叫法不该换指纹"
    assert (
        identity.changed_sections(identity.sections(_scenario()), identity.sections(renamed))
        == []
    )

    assert identity.changed_sections(
        identity.sections(_scenario()), identity.sections(_mutated("改了脚本"))
    ) == ["script"], "同一节里的脚本动了必须报出来"


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
    """差异报告里各节的出现顺序由这个常量决定，顺序变了报告前后就不可比。

    `world` 是后加的第六节（世界初始状态：订单上的自由文本备注）。加节要动这
    一行，是刻意的——多一节或少一节，报告里的差异定位就不一样了。
    """
    assert identity.SECTIONS == ("prompt", "tools", "script", "agent", "world", "expect")
    assert list(identity.sections(_scenario())) == list(identity.SECTIONS)


def test_the_world_is_a_test_section_not_an_agent_section() -> None:
    """换了世界里的那句话 = 换了这次测试面对的布景。

    它归在"测试变了"那一类：换了世界之后"失败少了几个"什么都不说明，而把测试
    改动读成行为回归是本工具最不该犯的错。它和 `script` 是同一类东西——都是
    作者摆出来的布景，不是被测对象的构建。
    """
    assert "world" in identity.TEST_SECTIONS
    assert "world" not in ("prompt", "tools", "agent")


def test_every_scenario_field_enters_the_fingerprint_or_is_listed_as_prose() -> None:
    """场景模型上的每个顶层字段，要么进指纹，要么在散文名单里。

    这条盯的是**以后**：往 `Scenario` 上加一个字段（比如给循环权重找个新家），
    忘了把它接进 `sections()` 的话，症状是"配置没变，但结论变了"——最该被看到
    的信号直接消失，而用户会去查一个没动过的文件。这里比的是两张表，所以红的
    时候不用猜是谁漏了。

    两个方向都要成立：少一个字段（漏接线）报错，多一个名字（`SECTION_SOURCES`
    里写了一个模型上已经没有的字段）也报错——后者会让这张表慢慢变成小说。
    """
    declared = {name for names in identity.SECTION_SOURCES.values() for name in names}
    assert declared | set(identity.PROSE_FIELDS) == set(Scenario.model_fields), (
        "这些字段既没进指纹也不在散文名单里："
        f"{sorted(set(Scenario.model_fields) - declared - set(identity.PROSE_FIELDS))}；"
        f"这些名字在表里但模型上没有了：{sorted(declared - set(Scenario.model_fields))}"
    )
    assert identity.SECTION_SOURCES.keys() == set(identity.SECTIONS)


def test_the_session_name_is_the_only_thing_a_session_hides() -> None:
    """会话那一节只许抹掉一个字段——`name`。

    抹掉的动作是**减法**：`SessionSpec` 上以后再加字段（比如别的计数器），
    它会一声不响地落在指纹之外，而这一节看起来还是"什么都比"。所以抹掉的名单
    要和模型的字段表对上：`SessionSpec` 减去名单，剩下的必须是这一节里真的
    出现的字段。
    """
    payload = identity.sections(_scenario())["script"][0]
    hidden = set(identity.SESSION_FIELDS_NOT_IN_FINGERPRINT)
    assert set(SessionSpec.model_fields) - hidden == set(payload), (
        "会话那一节的字段和 `SESSION_FIELDS_NOT_IN_FINGERPRINT` 对不上"
    )
    assert "name" not in payload

    # 上面那两句是**同源**的：`hidden` 从同一个常量取、又被减了两次，所以它们看
    # 得见「名单里写了一个模型上已经没有的名字」，看不见「名单**长了一条**」——
    # 而后者才是真正危险的方向：多藏一个字段，那一节就少比一样东西，报告上却
    # 还是「这一节都核过了」。所以名单本身逐字钉在这里：往上面加字段是一个
    # **要动这一行**的动作。（这条是探针量出来的：把 `script` 加进名单，上面两句
    # 全绿，红的只有下面这一句。）
    assert identity.SESSION_FIELDS_NOT_IN_FINGERPRINT == ("name",), (
        "会话里被藏起来的东西不止 `name` 一个——多藏一个字段，这一节就少比一样"
        "东西，而报告上看起来还是「这一节比过了」。"
    )


def test_every_world_field_reaches_the_fingerprint() -> None:
    """世界那一节里的字段必须**全部**进指纹。

    漏一个的症状是"配置没变，但结论变了"——最该被看到的那个信号直接消失，而
    用户会去查一个没动过的文件。这里是字段表对字段表，以后往 `WorldDecl` 里加
    东西忘了接线，红的会是这一条。
    """
    payload = identity.sections(_scenario())["world"]
    missing = sorted(set(WorldDecl.model_fields) - set(payload))
    assert not missing, f"这些世界字段没进配置指纹：{missing}"


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

    **这是全项目唯一把口径的值写死的地方**，写死是故意的：升版必须是一个有人
    签字同意的动作，而不是一次顺手的重构。3 → 4 的场合是 `ToolDecl` 多了
    `may_run_unattended`、`SessionSpec` 多了 `autonomous`——两节的内容都变了，
    于是没动过的场景也会换哈希。不升版的话，那种不相等会被读成"测试改了"。
    4 → 5 是同一件事的小一号：`ToolDecl` 多了 `identity_fields`，`tools` 这一节
    的载荷（整份 `ToolDecl`）于是换了内容，一份没改过的场景也会换哈希。
    """
    assert CONFIG_SCHEME == "5"
