# -*- coding: utf-8 -*-
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


def assert_inline_only_differs(
    *scenarios: Scenario, claim: str, differs_in: tuple[str, ...]
) -> None:
    """核对一条「内联构造」的受控对照——三条与 `assert_only_differs` 相同。

    场景不在 `scenarios/` 目录里（是测试自己 `parse_scenario` 出来的内存对象），
    走不了 `load_scenario` 那一条。与文件对共用同一把尺子（`identity.sections`）
    和同一份残差口径；多支（如幂等四支）时按两两组合核，任何一对都不许出现
    声明之外的差别。
    """
    if len(scenarios) < 2:
        raise ValueError("受控对照至少要两支场景。")
    shapes = tuple(_shape(pattern) for pattern in differs_in)
    hit: set[str] = set()

    for a, b in itertools.combinations(scenarios, 2):
        residual = _residual(a, b, shapes)
        assert not residual, (
            f"{claim}\n"
            f"  除了声明的 {list(differs_in)} 之外还有别的差别：{residual}\n"
            f"  差别要是还有别的，结论的不同就可能是别的东西造成的——而两份报告"
            f"读起来完全一样，谁都发现不了。"
        )

        paths = measured(a, b)
        for pattern, shape in zip(differs_in, shapes, strict=True):
            if any(shape.fullmatch(path) for path in paths):
                hit.add(pattern)

    missing = [pattern for pattern in differs_in if pattern not in hit]
    assert not missing, (
        f"{claim}\n"
        f"  声明的 {missing} 在几支之间一条都没命中——这句声明是空的。"
    )

    if not differs_in:
        for a, b in itertools.combinations(scenarios, 2):
            assert identity.changed_sections(
                identity.sections(a), identity.sections(b)
            ), f"{claim}\n  两支一模一样——那没有对照可言。"


# --- 账本 ---
# --- 账本 ---------------------------------------------------------------------
#
# 顺序跟着「哪一族」走，不跟字母走：读账本的人是在找某一族实验的说法。

PAIRS: tuple[Controlled, ...] = (
    Controlled(
        scenarios=("order_delivery_chain_clean", "order_delivery_chain_duplicate"),
        claim=(
            "多 Agent 接力（下单→履约）：唯一的变量是 0 号（order_agent）下了一次单还是两次"
            "——0 号关掉记忆守卫后重复 create_payment，目标系统扣款 1 笔 vs 2 笔，"
            "幂等 fail 只出现在下单侧，1 号那侧一字不差。"
        ),
        differs_in=(
            "agent.env",
            "agent.env.DEMO_AGENT_MEMORY_GUARD",
            "script[n].script[n].content",
            "script[n].script[n].tool_calls[n].name",
            "script[n].script[n].tool_calls[n].arguments.order_id",
            "script[n].script[n].tool_calls[n].arguments.amount",
            "script[n].script[n].tool_calls[n].arguments.target_role",
            "script[n].script[n].tool_calls[n].arguments.task",
        ),
    ),
)

#: 没进任何对照的场景，以及为什么。**每一份场景都要么在账本里、要么在这里**——
#: 反向断言盯的就是这个：加了一份新场景而它谁都不搭，红的是那一条。理由不必长，
#: 但要说出「为什么它没有对照组」，而不是「还没写」。
UNPAIRED: dict[str, str] = {}

#: 近邻清单：两份场景被测侧差得很少、但**不是**受控对照。写了理由才算数。
#: 今天它是空的——量过，全仓库被测侧差异 ≤ 6 的对子恰好都在账本里。空着不是
#: 摆设：它是一句「我看过了」的声明，绊线红了要往这里写，或者往账本里写。
NOT_A_PAIR: dict[tuple[str, str], str] = {}
