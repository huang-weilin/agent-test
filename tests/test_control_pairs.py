"""账本的执行者：把每一句「只差一个比特」跟文件对一遍。

账本本身（口径、记录、反向清单）在 `tests/control_pairs.py`。这个文件做三件事：
跑账本、盯覆盖、以及**证明这个账本真的会红**——最后那一类不能省：一个从来不会
失败的断言和一句注释没有区别，而它的失效方式和"接线没接上"一模一样，报告上
看不出来。
"""

from __future__ import annotations

import itertools
import re

import pytest

from sandbox.scenario import identity
from sandbox.scenario.loader import load_scenario, parse_scenario
from sandbox.scenario.spec import Scenario
from tests.control_pairs import (
    NOT_A_PAIR,
    PAIRS,
    SCENARIOS,
    SUBJECT_SECTIONS,
    UNPAIRED,
    Controlled,
    assert_only_differs,
    leaves,
    measured,
    subject,
)

ALL_SCENARIOS: list[str] = sorted(path.stem for path in SCENARIOS.glob("*.yaml"))


def load_of(name: str) -> Scenario:
    return load_scenario(SCENARIOS / f"{name}.yaml")


def paths_of(name: str) -> set[str]:
    """一份场景被测侧的叶子路径集合（不看值，只看形状）。"""
    return set(dict(leaves(subject(load_of(name)))))


# --- 账本 ---------------------------------------------------------------------


@pytest.mark.parametrize("entry", PAIRS, ids=[e.scenarios[0] for e in PAIRS])
def test_each_ledger_entry_holds(entry: Controlled) -> None:
    """每条记录：抹掉声明的差别之后必须逐字相等，且每个声明都要真的命中。"""
    assert_only_differs(entry)


def test_every_scenario_is_in_the_ledger_or_has_a_reason() -> None:
    """每一份场景要么在账本里做对照，要么写一句「为什么它没有对照组」。

    这条是账本的反向断言，也是这份账本作为**待办清单**的用法：新加一份场景而
    它谁都不搭，红的会是这一条——那时要么把它跟谁配一对，要么在 `UNPAIRED` 里
    说清楚它为什么是独苗。两处都不做的话，它就静静地躺在 `scenarios/` 里，
    而"这一族有活样本撑着"这句话慢慢变成假的。
    """
    in_pairs = {name for entry in PAIRS for name in entry.scenarios}
    missing = sorted(set(ALL_SCENARIOS) - in_pairs - set(UNPAIRED))
    assert not missing, (
        f"这些场景既不在账本里、也没写为什么没有对照组：{missing}\n"
        f"  加一份新场景而它谁都不搭，不会有人发现——这条就是那个「有人发现」。"
    )

    unknown = sorted((in_pairs | set(UNPAIRED)) - set(ALL_SCENARIOS))
    assert not unknown, f"账本里有这些名字，而 scenarios/ 下没有对应的文件：{unknown}"

    twice = sorted(
        name for name in in_pairs if sum(name in e.scenarios for e in PAIRS) > 1
    )
    assert not twice, f"这些场景同时出现在两条记录里，读的人不知道信哪条：{twice}"

    empty = sorted(name for name, why in UNPAIRED.items() if not why.strip())
    assert not empty, f"这些场景的理由是空的：{empty}"


def test_a_pair_that_is_named_in_the_prose_must_be_in_the_ledger() -> None:
    """场景散文里点名了另一份场景，那一对就必须在账本里。

    说法最自然的地方是场景自己的 `description`——「对照组是 X.yaml」就写在
    那儿。可它是**散文**：写得出来、读得通、而没有任何东西会去核它。这一条把
    那些点名收上来，要求它们同时出现在账本里（只有账本才有判据）。

    同一个名字出现在别的名字里面（`memory_note_clean.yaml` 里含着
    `clean.yaml`）不算点名，所以左边要挡住词字符。
    """
    patterns = {
        name: re.compile(r"(?<![A-Za-z0-9_-])" + re.escape(name) + r"\.yaml")
        for name in ALL_SCENARIOS
    }
    groups = [set(entry.scenarios) for entry in PAIRS]

    unledgered: list[str] = []
    for name in ALL_SCENARIOS:
        description = load_of(name).description or ""
        for other in ALL_SCENARIOS:
            if other == name or not patterns[other].search(description):
                continue
            if not any({name, other} <= group for group in groups):
                unledgered.append(f"{name} 点名了 {other}")

    assert not unledgered, (
        f"这些场景在散文里点了名，却没有进账本：{unledgered}\n"
        f"  点名就是在说「我们是一对」。说了就要有判据跟着——账本里那一条。"
    )


def test_no_scenario_pair_looks_like_a_copy_without_a_declaration() -> None:
    """近邻绊线：一份场景像是从另一份抄来只改了一处，就得说明白它是哪一对。

    判据是两条一起：**叶子路径集合一个包含另一个**（形状是同一份场景），
    且**不同的值只有一处**。这样写而不是写一个相似度阈值，是因为量过——
    这个仓库里"差异条数少"根本不是判据：`clean` 是一份最小的场景，它跟七份
    场景的差异都在 6 条以内，全是真的不同场景。绊线一旦开始误报，第一个动作
    就是把它删掉，所以宁可窄。
    """
    reported: list[str] = []
    for name, other in itertools.combinations(ALL_SCENARIOS, 2):
        if any({name, other} <= set(entry.scenarios) for entry in PAIRS):
            continue
        if (name, other) in NOT_A_PAIR or (other, name) in NOT_A_PAIR:
            continue
        left, right = paths_of(name), paths_of(other)
        if not (left <= right or right <= left):
            continue
        if len(measured(load_of(name), load_of(other))) <= 1:
            reported.append(f"{name} / {other}")

    assert not reported, (
        f"这些对子看起来像「抄了一份只改一处」，却既不在账本里、也没有解释：{reported}\n"
        f"  要么它们就是一对受控实验（写进账本），要么在 `NOT_A_PAIR` 里说清楚"
        f"它们为什么只是长得像。"
    )


# --- 账本自己的承重测试：它必须真的会红 --------------------------------------------


def test_the_ledger_catches_the_claim_that_used_to_be_written_here() -> None:
    """把过去那句话照原样写进账本，它必须红。

    `order_delivery_chain` 那一对的说法如果只交代「0 号那边多了一个环境变量开关」
    ——写在测试的 docstring 里，而没有任何东西核对过它——真去量一次：被测侧
    还差十来处，包括 0 号脚本里那一步多出来的 `create_payment`（参数里带着金额）
    与转交语句。这条测试把那种旧说法钉在墙上：**它必须失败**，而且要说清楚
    差在哪几处。
    """
    stale = Controlled(
        scenarios=("order_delivery_chain_clean", "order_delivery_chain_duplicate"),
        claim="两支只差下单侧那个环境变量开关。",
        differs_in=("agent.env",),
    )
    with pytest.raises(AssertionError) as caught:
        assert_only_differs(stale)

    message = str(caught.value)
    assert "agent.env" in message
    assert "arguments.amount" in message, (
        "0 号脚本里多出来的那一步付款必须被指出来——它正是旧说法漏掉的东西"
    )


def test_a_declared_difference_that_is_not_there_is_refused() -> None:
    """清单上写了、而两支之间根本没这个差别，必须红：那是一句空话。

    它比"漏写一条"更隐蔽——漏写会让残差那条红，而空话让整条记录看起来一切
    正常：清单看着挺长，读者以为差别都交代过了，其实其中一条纯属虚构。
    """
    padded = Controlled(
        scenarios=("order_delivery_chain_clean", "order_delivery_chain_duplicate"),
        claim="唯一的差别是下单侧多了一步付款与一个环境变量开关。",
        differs_in=(
            "agent.env",
            "agent.env.DEMO_AGENT_MEMORY_GUARD",
            "script[n].script[n].tool_calls[n].arguments.amount",
            "script[n].script[n].tool_calls[n].arguments.order_id",
            "script[n].script[n].tool_calls[n].arguments.target_role",
            "script[n].script[n].tool_calls[n].arguments.task",
            "script[n].script[n].tool_calls[n].name",
            "script[n].script[n].content",
            "world.order_notes.A1",
        ),
    )
    with pytest.raises(AssertionError, match="一条都没命中"):
        assert_only_differs(padded)


def test_a_pair_of_identical_files_is_refused() -> None:
    """清单为空 = 只说「被测侧一处都不差」。那两份文件就不能是完全一样的两份。

    少了这一条，把同一个文件名写两遍就会得到一条永远通过的记录——账本上多一条
    空的对照，而它看起来和真的一样。
    """
    self_pair = Controlled(
        scenarios=("order_delivery_chain_clean", "order_delivery_chain_clean"),
        claim="它自己跟自己。",
        differs_in=(),
    )
    with pytest.raises(AssertionError, match="一模一样"):
        assert_only_differs(self_pair)


def test_an_empty_container_is_not_the_same_as_a_missing_key() -> None:
    """摊平器必须看得见空表。

    同一份工具声明，只把 `parameters: {type: object, properties: {}}` 换成
    `parameters: {type: object}`——差别就是空表与「没有 `properties`」，而空
    字典在逐叶比对里没有叶子。一个只认"有子节点才算有内容"的摊平器会把这一对
    判成**一处都不差**。这一条直接盯着那一行。
    """
    empty = parse_scenario(
        """
id: t
agent: {version: v, task: 做点事}
tools:
  - name: get_order
    description: 查询
    is_write: false
    parameters: {type: object, properties: {}}
sessions:
  - name: 一段
    script: []
""",
        source="<内存中的场景>",
    )
    undeclared = parse_scenario(
        """
id: t
agent: {version: v, task: 做点事}
tools:
  - name: get_order
    description: 查询
    is_write: false
    parameters: {type: object}
sessions:
  - name: 一段
    script: []
""",
        source="<内存中的场景>",
    )

    assert measured(empty, undeclared) == ["tools[0].parameters.properties"], (
        "空表和没给必须在这一层的比对里看得见"
    )


def test_the_subject_leaves_out_the_ruler() -> None:
    """`expect` 不进被测侧——两边的尺子本来就必须不同。

    把它算进去的话，`loop` 与 `loop_exempted` 那条记录就写不出来：那一对的被测侧
    确实一处都不差，唯一的变量是判卷标准里多了一条豁免。
    """
    derived = tuple(name for name in identity.SECTIONS if name != "expect")
    assert derived == SUBJECT_SECTIONS, (
        "被测侧应当是「全部节减去 `expect`」。多算了 `expect` 的话，对照组就成了"
        "要求两边判卷标准一样——而「两边结论相反」正是它的全部内容。"
    )
