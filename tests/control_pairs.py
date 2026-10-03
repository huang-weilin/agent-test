"""受控对照的账本：每一句「只差一个比特」，都要有东西在核对它。

**为什么值得单独有一个模块。** 这个仓库到处写着同一句话：

    脚本、任务、工具声明一字不差，唯一的差别是 X。

它是这一整套测试的承重墙——「结论不同只可能是 X 造成的」，全靠它。这句话有三个
特点，凑在一起就出事：写起来便宜（一句话）、读起来可信（不像会错的样子）、错了
**没有任何症状**（两份报告都正常，差异被归到别的地方去了）。

所以它得变成机械可比的东西，而口径只能有一处。以前它被手写了六遍，每遍都只在
自己那一对上有东西核对；量下来，十二对里有两对**根本没有任何东西在核对它**，
其中一对的说法是假的（`long_run_overnight` 说「一字不差，只把间隔改成 0」，实测
被测侧差六处，包括第二段脚本里多出来的那一步付款）。

**比对用哪把尺子。** 用 `identity.sections`——生产上判「配置变没变」的那把。它自己
再手搓一份 `model_dump` 的话，就会出现「测试说这两份只差一个比特，而生产会把它读成
配置变了」这种自相矛盾的两句话。今天这一条是硬的：`tests/test_identity.py` 里有一条
断言盯着「场景模型上的每个字段要么进指纹、要么在散文名单里」。

**逐叶比对有一个天生的盲点：空容器没有叶子。** `properties: {}` 和「没有
`properties` 这个键」在这张表里长得一模一样——而这两件事的区别正是
`schema_empty_properties` 那一对在演的东西。所以空字典、空列表在这里**按值处理**：
它们是叶子，不是「没有子树」。漏了这一条，账本会给那一对判「被测侧一处都不差」。

**账本长什么样。** 每条记录是一组场景 + 一句「这一对在证明什么」+ 一份**路径形状**
清单（`[n]` 表示任意下标）。核对分三步，缺一不可：

1. 把清单上的形状**抹掉**之后，两份必须逐字相等（多一处差别就红——那才是真正要抓
   的漂移：有人往其中一份里偷偷加了变量）；
2. 每个形状至少要命中一条真的差别（否则那是一句空话：写在清单上、而两支之间根本
   没有这个差别，它比漏写一条更隐蔽）；
3. 清单为空时另有一条：被测侧必须**一处都不差**，而两份文件不能完全一样
   （唯一变的是尺子，不是没有变量）。

`tests/test_control_pairs.py` 跑这三条，外加两个反向断言：**每份场景要么在账本里
做对照、要么在 `UNPAIRED` 里写一句为什么**；以及近邻绊线（两份场景被测侧只差几处
却没进账本，那多半是又抄了一份场景）。
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sandbox.scenario import identity
from sandbox.scenario.loader import load_scenario
from sandbox.scenario.spec import Scenario

#: 场景文件所在处。账本里的名字都是**文件名的词干**——`id` 与文件名今天一一对应，
#: 但账本比的是文件，用词干更直白（改文件名要动这里，那正是应该被看见的动作）。
SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"

#: 受控对照要比的那几节：**被测对象面对的那份输入**。
#:
#: `expect` 不在里面：两边的尺子本来就必须不同——「必须给出相反的结论」正是对照组
#: 的全部内容。把它算进差异里，等于要求对照组放弃它的作用。
SUBJECT_SECTIONS: tuple[str, ...] = tuple(
    name for name in identity.SECTIONS if name != "expect"
)


@dataclass(frozen=True)
class Controlled:
    """一条受控对照：一组场景、一句说法、一份差别形状清单。"""

    #: 参与对照的场景文件名（不带 `.yaml`）。两个以上表示「这一族之间两两都不许有
    #: 清单之外的差别」。
    scenarios: tuple[str, ...]
    #: 这一对在证明什么。核对失败时原样打出来——读的人先要知道它本来想说什么。
    claim: str
    #: 差别只许落在这些**形状**上：`[n]` 表示任意下标，其余字符逐字相等。
    differs_in: tuple[str, ...]


def entry_for(*scenarios: str) -> Controlled:
    """按场景名取出账本里的那一条。

    各族的测试文件用它来引用自己那一对，而不是把比对再手写一遍。**取不到就报错**：
    测试里写错一个名字，不该悄悄退化成"这一条没有对照断言"——那正是这个账本要
    消灭的那种"看起来在核对"。
    """
    wanted = set(scenarios)
    found = [entry for entry in PAIRS if wanted <= set(entry.scenarios)]
    assert len(found) == 1, (
        f"{list(scenarios)} 在账本里对应 {len(found)} 条记录——应当恰好一条。"
    )
    return found[0]


def subject(scenario: Scenario) -> dict[str, Any]:
    """被测对象面对的那份输入：除 `expect`（尺子）之外的全部。

    分节用的是 `identity.sections`，也就是生产上判「配置变没变」的那把尺子。
    """
    parts = identity.sections(scenario)
    return {name: parts[name] for name in SUBJECT_SECTIONS}


def _shape(pattern: str) -> re.Pattern[str]:
    """把写给人看的形状变成正则：`[n]` = 任意下标。

    用正则而不是 `fnmatch`，是因为 `fnmatch` 里 `[...]` 是字符类——`tools[n]` 会
    被它读成「tools 后面跟一个 n」，而路径上那一格是数字。
    """
    return re.compile(re.escape(pattern).replace(r"\[n\]", r"\[\d+\]"))


def leaves(payload: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """把嵌套结构摊平成「路径 → 值」。

    **空字典与空列表按值处理**——它们是叶子，不是「没有子树」。这一条不是洁癖：
    `properties: {}` 与「没有 `properties`」的区别正是有一对场景在演的东西，而
    一个只认「有子节点才算有内容」的摊平器看不见它，于是那一对会被判成一处都不差。
    """
    if isinstance(payload, dict) and payload:
        for key, value in payload.items():
            yield from leaves(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(payload, list) and payload:
        for index, value in enumerate(payload):
            yield from leaves(value, f"{prefix}[{index}]")
    else:
        yield prefix, payload


#: 被抹掉的那些格子在两边都写成同一个值，于是它们在比对里既不算"相同"也不算"不同"。
_ERASED = "<抹掉的>"


def measured(a: Scenario, b: Scenario) -> list[str]:
    """两份场景之间**所有**不同的叶子路径（不抹任何东西）。"""
    left, right = dict(leaves(subject(a))), dict(leaves(subject(b)))
    return sorted(
        path
        for path in set(left) | set(right)
        if left.get(path, "<没给>") != right.get(path, "<没给>")
    )


def _residual(
    a: Scenario, b: Scenario, shapes: tuple[re.Pattern[str], ...]
) -> list[str]:
    """抹掉清单上的形状之后，两边仍然不一样的地方。

    抹的动作必须**两边对称**地做，包括那一侧根本没有这个路径的时候。否则
    「这一侧有、那一侧没有」的差别会以「抹掉的 vs 没给」的形式活下来——账本上
    写着"这里允许不同"，而残差里它仍然算一处差别，读报的人会以为自己看错了。
    """
    left, right = dict(leaves(subject(a))), dict(leaves(subject(b)))
    for path in set(left) | set(right):
        if any(shape.fullmatch(path) for shape in shapes):
            left[path] = right[path] = _ERASED
    return sorted(
        path
        for path in set(left) | set(right)
        if left.get(path, "<没给>") != right.get(path, "<没给>")
    )


def assert_only_differs(entry: Controlled) -> None:
    """核对一条对照记录。三条都过才算数，见模块 docstring。"""
    loaded = [
        (name, load_scenario(SCENARIOS / f"{name}.yaml")) for name in entry.scenarios
    ]
    shapes = tuple(_shape(pattern) for pattern in entry.differs_in)
    hit: set[str] = set()

    for (name_a, a), (name_b, b) in itertools.combinations(loaded, 2):
        residual = _residual(a, b, shapes)
        assert not residual, (
            f"{entry.claim}\n"
            f"  {name_a} 与 {name_b} 之间，除了账本声明的 {list(entry.differs_in)} "
            f"之外还有别的差别：{residual}\n"
            f"  差别要是还有别的，结论的不同就可能是别的东西造成的——而两份报告"
            f"读起来完全一样，谁都发现不了。"
        )

        paths = measured(a, b)
        for pattern, shape in zip(entry.differs_in, shapes, strict=True):
            if any(shape.fullmatch(path) for path in paths):
                hit.add(pattern)

    missing = [pattern for pattern in entry.differs_in if pattern not in hit]
    assert not missing, (
        f"{entry.claim}\n"
        f"  账本声明的 {missing} 在两支之间一条都没命中——这句声明是空的："
        f"写在清单上，而两支之间根本没有这个差别。"
    )

    if not entry.differs_in:
        # 唯一的变量是尺子。那就要两件事同时成立：被测侧一处都不差，而两份文件
        # 不能完全一样（完全一样的两份文件之间没有对照可言）。
        for (name_a, a), (name_b, b) in itertools.combinations(loaded, 2):
            assert identity.changed_sections(
                identity.sections(a), identity.sections(b)
            ), f"{entry.claim}\n  {name_a} 与 {name_b} 一模一样——那没有对照可言。"


# --- 账本 ---------------------------------------------------------------------
#
# 顺序跟着「哪一族」走，不跟字母走：读账本的人是在找某一族实验的说法。

PAIRS: tuple[Controlled, ...] = (
    Controlled(
        scenarios=("loop", "loop_exempted"),
        claim="同一串调用、同一份脚本，唯一变的是判卷标准里多了一条豁免（一边被杀、一边跑完）。",
        # 被测侧一处都不差：豁免是**声明**，不是行为——它落在 `expect` 那一节里。
        # 这一条同时说明"唯一变量"里也包含"尺子"这种变量，它照样是受控的。
        differs_in=(),
    ),
    Controlled(
        scenarios=("duplicate", "missing_key", "key_rotated", "protected"),
        claim="同一个订单、同一段脚本付两次：四支的差别只许来自工具声明与幂等键怎么写。",
        differs_in=(
            "tools[n].enforces_idempotency",
            "script[n].script[n].tool_calls[n].arguments.idempotency_key",
        ),
    ),
    Controlled(
        scenarios=("autonomous_night_authorized", "autonomous_night_unauthorized"),
        claim="无人值守的夜间写操作：唯一的变量是那一个授权位（它报越权、它报按授权办事）。",
        differs_in=("tools[n].may_run_unattended",),
    ),
    Controlled(
        scenarios=("autonomous_night_egress_allowed", "autonomous_night_egress_blocked"),
        claim="同一份脚本、同一台真实目标主机：唯一的变量是放行名单里有没有那台支付网关。",
        differs_in=(
            "world.egress.allow[n].host",
            "world.egress.allow[n].mock_path",
            "world.egress.allow[n].reason",
        ),
    ),
    Controlled(
        scenarios=(
            "autonomous_night_generic_egress_allowed",
            "autonomous_night_generic_egress_blocked",
        ),
        claim="与上面那一对同一件事，只是工具没写死目标主机：唯一的变量还是那份名单。",
        differs_in=(
            "world.egress.allow[n].host",
            "world.egress.allow[n].mock_path",
            "world.egress.allow[n].reason",
        ),
    ),
    Controlled(
        scenarios=("schema_empty_properties", "schema_keys_undeclared"),
        claim="空表和「没给」是两件事：唯一的变量是声明里有没有 `properties` 那一行。",
        differs_in=("tools[n].parameters.properties",),
    ),
    Controlled(
        scenarios=("clean", "stream_clean"),
        claim="同一段脚本、同一个世界，唯一的变量是请求走不走流式（`DEMO_AGENT_STREAM`）。",
        # 要证的是「换一种传输方式不改行为」：结论该一字不差，而账要从上游的
        # usage 来、不能是网关估的（那一半在 `tests/test_gateway_stream.py` 里
        # 真跑一遍来核，不在这个文件里）。
        #
        # `clean` 本来是 UNPAIRED 里的独苗（理由见那张表）——它不需要对照组，
        # 但它可以**当**人家的对照组：这里的变量是传输方式，而 clean 正是
        # 「不流式」那一支的全部内容。
        #
        # 两处一起列：`clean` 根本没有 `env` 那一行，`stream_clean` 有一个只装了
        # 这一个变量的 `env`。抹掉叶子之后，一边是空表、一边是「没有这个键」——
        # 而空表在这个摊平器里**是按值算的**（见文件头那段：空容器没有叶子，
        # 不特判就会把这一对判成「一处都不差」）。
        differs_in=("agent.env", "agent.env.DEMO_AGENT_STREAM"),
    ),
    Controlled(
        scenarios=("memory_poisoned", "memory_note_clean"),
        claim="记忆中毒：唯一的变量是世界里的那句话（订单备注上的自由文本）。",
        differs_in=("world.order_notes.A1",),
    ),
    Controlled(
        scenarios=("memory_tidy_archived", "memory_tidy_pruned"),
        claim="整理记忆：唯一的变量是整理时留不留原件（挪进 `archive/` 还是删掉）。",
        differs_in=("agent.env.DEMO_AGENT_MEMORY_TIDY",),
    ),
    Controlled(
        scenarios=("loop_paging_bounded", "loop_paging_unbounded"),
        claim="分页循环：唯一的变量是脚本往后多翻了几页、以及随之的那句汇报。",
        differs_in=(
            "script[n].script[n].content",
            "script[n].script[n].tool_calls[n].name",
            "script[n].script[n].tool_calls[n].arguments.page",
            "script[n].script[n].tool_calls[n].arguments.page_size",
            "script[n].script[n].tool_calls[n].arguments.status",
        ),
    ),
    Controlled(
        scenarios=("long_run_overnight", "long_run_same_moment"),
        claim=(
            "长时运行：世界与工具一模一样，两段会话的间隔从 30 小时变成 0。"
            "脚本的差别是**结论本身**——对照组不能写那一步付款（写了就会真的扣钱，"
            "和它自己的地面真值打架），所以这一对钉的不是「脚本一字不差」，是"
            "「除了时间与那一步之外，别的都一字不差」。"
        ),
        differs_in=(
            "script[n].after_gap_s",
            "script[n].script[n].content",
            "script[n].script[n].tool_calls[n].name",
            "script[n].script[n].tool_calls[n].arguments.order_id",
            "script[n].script[n].tool_calls[n].arguments.amount",
        ),
    ),
    Controlled(
        scenarios=(
            "cross_session_remembered",
            "cross_session_duplicate",
            "cross_session_memory_loss",
        ),
        claim=(
            "跨会话三方对照：脚本、任务、模型、工具声明都一样，只有两处——记忆回滚到"
            "哪儿（快照点），以及 Agent 查不查记忆（那一个环境变量）。"
        ),
        differs_in=(
            "agent.env",
            "agent.env.DEMO_AGENT_MEMORY_GUARD",
            "script[n].snapshot_before",
            "script[n].snapshot_after",
            "script[n].restore_from",
        ),
    ),
    Controlled(
        scenarios=("cross_session_handoff", "cross_session_same_agent"),
        claim=(
            "同一段脚本、同样三段会话、同一份记忆判据：唯一的变量是第二段换不换人"
            "（换 → 记忆不跟着走，重复扣款、归因指向接缝；不换 → 只扣一笔）。"
        ),
        # 五处，全是"名单里多了一个人"这一件事的不同侧面：`others[0]` 那四项是
        # 1 号那份 AgentSpec 的字段（directory/entry/env 在两边其实取同一个值，
        # 差别只在"有没有这一项"），`script[1].agent` 是会话上指名了谁。
        # 没有任何一处是行为或判据上的差别——那正是这一对要的：**归因的翻转
        # 只由"换人"这一个变量造成**。
        differs_in=(
            "agent.others[n].version",
            "agent.others[n].directory",
            "agent.others[n].entry",
            "agent.others[n].env",
            "script[n].agent",
        ),
    ),
)

#: 没进任何对照的场景，以及为什么。**每一份场景都要么在账本里、要么在这里**——
#: 反向断言盯的就是这个：加了一份新场景而它谁都不搭，红的是那一条。理由不必长，
#: 但要说出「为什么它没有对照组」，而不是「还没写」。
UNPAIRED: dict[str, str] = {
    "target_error": (
        "目标系统报错那一路的活样本。它的对照组要造一个「同样的调用、目标系统不报错」"
        "的世界，而那正是 clean 的常态——再抄一份只差一个布尔值的场景，比在测试里"
        "直接把返回体换掉更贵，也更难读。"
    ),
    "schema_rejected": (
        "参数校验的负样本。它没有正样本对照组：参数校验是每个请求上顺带做的一次判定，"
        "不存在「被覆盖到的路径」这回事（见 README 的「已知边界」一节），所以"
        "「两边结论相反」在这里没有可对照的那一半。"
    ),
}

#: 近邻清单：两份场景被测侧差得很少、但**不是**受控对照。写了理由才算数。
#: 今天它是空的——量过，全仓库被测侧差异 ≤ 6 的对子恰好都在账本里。空着不是
#: 摆设：它是一句「我看过了」的声明，绊线红了要往这里写，或者往账本里写。
NOT_A_PAIR: dict[tuple[str, str], str] = {}
