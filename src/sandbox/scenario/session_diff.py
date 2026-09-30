"""跨会话差异报告（需求 3.1 P1）。

同一个场景里跑了好几段会话，报告要能回答一个问题：

> **它这一段做的事，和那一段做的事，一样吗？**

做法是把两段会话的**工具调用**逐项对齐：同工具、同参数指纹的算"同一件事"，
对不上的按"只在某一段出现"列出来。

## 与回归对比的分工

`regression/diff.py` 比的是**两次运行**（配置变没变、结论变没变）；这里比的是
**同一次运行里的两段会话**。两者的纪律是同一条：

> **"有差异"和"差异算不算数"是两件事。**

所以 `comparable` 与 `differences` 是两个并列的字段：不管可不可比，差异照列——
不可比的时候只是多一句"这份差异可能不指向被测对象"。

## 三条自我约束

1. **这是陈述，不是判定。** 它不进 `findings`，不影响 `ok`。同一个场景的两段
   会话本来就允许不一样——长时运行的场景里，第二段看到的世界都不一样了。
   把"有差异"判成失败，等于把长时运行的正常形态报成缺陷。

2. **不猜配对。** 只有"某一侧正好只剩一次、另一侧也正好只剩一次"的同名工具
   才敢说"这是同一次调用、参数不同"。其余的一律叫"只在某一段出现"——把剩下
   的硬凑成对，报告上会长出一句"参数从 X 变成了 Y"，而那个 Y 可能属于另一次
   调用。读者没有办法看出这句话是猜的。

3. **我们动过的东西单独列。** 记忆回滚、世界时间、无人值守——这些不一样是
   **我们**制造的。不列出来，读的人分不清"它的行为变了"和"我们让它不一样了"。
   而这两件事在报告上长得一模一样。
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sandbox.detectors.base import CallView
from sandbox.formatting import args_text, span_human, value_text

#: 差异的种类。**都是差异**——"一模一样"的那些不在这里列，只在计数里体现。
KIND_ONLY_BASE = "only_base"
KIND_ONLY_OTHER = "only_other"
KIND_COUNT_DIFFERS = "count_differs"
KIND_OUTCOME_DIFFERS = "outcome_differs"
KIND_ARGS_DIFFER = "args_differ"

#: 展示顺序。先列"只在某一段出现"（差异最大），再列次数、再列参数——
#: 参数只差一个字段的那条排最后，因为读者往往扫一眼上面几条就够了。
KIND_ORDER: tuple[str, ...] = (
    KIND_ONLY_BASE,
    KIND_ONLY_OTHER,
    KIND_COUNT_DIFFERS,
    KIND_OUTCOME_DIFFERS,
    KIND_ARGS_DIFFER,
)

KIND_LABELS: dict[str, str] = {
    KIND_ONLY_BASE: "只在基准那一段",
    KIND_ONLY_OTHER: "只在对照那一段",
    KIND_COUNT_DIFFERS: "次数不一样",
    KIND_OUTCOME_DIFFERS: "结果不一样",
    KIND_ARGS_DIFFER: "参数不一样",
}

#: 一对会话最多列多少条差异。超了**必须在报告里说**——静默截断会让"还有 20 条
#: 差异"读成"差异就这些"。审计页那条规矩在这里同样成立。
MAX_ITEMS = 12

#: 单条差异里最多展开几处字段变化。
MAX_CHANGES = 4


@dataclass(frozen=True)
class SessionSide:
    """会话里跟"能不能拿来比"有关的那几项。

    刻意不用鸭子类型一把抓：这份字段清单就是"差异报告读了会话的哪些属性"的
    全部，多读一项必须在这里加一行、并被 `of` 显式搬过来。报告越界去读会话的
    其它字段（比如拿 stdout 猜行为）正是在这里被拦住的。
    """

    ordinal: int
    name: str = ""
    autonomous: bool = False
    memory_horizon_ordinal: int | None = None
    restored_from: str | None = None
    after_gap_s: float = 0.0
    virtual_started_at: str | None = None
    premise_held: bool = True
    killed: bool = False
    clock_matched: bool = True

    @classmethod
    def of(cls, outcome: Any) -> SessionSide:
        """从 runner 的 `SessionOutcome` 搬过来。"""
        return cls(
            ordinal=outcome.ordinal,
            name=outcome.name,
            autonomous=outcome.autonomous,
            memory_horizon_ordinal=outcome.memory_horizon_ordinal,
            restored_from=outcome.restored_from,
            after_gap_s=outcome.after_gap_s,
            virtual_started_at=outcome.virtual_started_at,
            premise_held=outcome.premise_held,
            killed=outcome.killed,
            clock_matched=outcome.clock_matched,
        )


@dataclass(frozen=True)
class DiffItem:
    kind: str
    tool: str
    base_ordinal: int
    other_ordinal: int
    base_text: str
    other_text: str
    #: 只有 KIND_ARGS_DIFFER 会用：逐字段说明差在哪。
    changes: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return KIND_LABELS.get(self.kind, self.kind)

    @property
    def sentence(self) -> str:
        """一句话说清这条差异。命令行的报告直接印它。"""
        if self.kind == KIND_ONLY_BASE:
            return (
                f"{self.tool}：只在会话 #{self.base_ordinal} 出现"
                f"（{self.base_text}），会话 #{self.other_ordinal} 里没有"
            )
        if self.kind == KIND_ONLY_OTHER:
            return (
                f"{self.tool}：只在会话 #{self.other_ordinal} 出现"
                f"（{self.other_text}），会话 #{self.base_ordinal} 里没有"
            )
        if self.kind == KIND_COUNT_DIFFERS:
            return (
                f"{self.tool}：参数一模一样，但发的次数不同——"
                f"会话 #{self.base_ordinal} {self.base_text}，"
                f"会话 #{self.other_ordinal} {self.other_text}"
            )
        if self.kind == KIND_OUTCOME_DIFFERS:
            detail = "；".join(self.changes)
            return (
                f"{self.tool}：参数一模一样，但对上的调用结果不同——{detail}"
            )
        detail = "；".join(self.changes) if self.changes else "（字段级差异见下方明细）"
        return (
            f"{self.tool}：两段各只剩这一次，按同一次调用对齐，参数不同——"
            f"会话 #{self.base_ordinal} {self.base_text}；"
            f"会话 #{self.other_ordinal} {self.other_text}。{detail}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "tool": self.tool,
            "base_ordinal": self.base_ordinal,
            "other_ordinal": self.other_ordinal,
            "base_text": self.base_text,
            "other_text": self.other_text,
            "changes": list(self.changes),
            "sentence": self.sentence,
        }


@dataclass(frozen=True)
class SessionPair:
    """一段会话与基准会话的对照。"""

    base: SessionSide
    other: SessionSide
    base_calls: int
    other_calls: int
    #: 两边指纹一致、次数也对得上的调用条数。**这是"一样的部分"的唯一证据**：
    #: 一份只列差异的报告，看不出"它其实把该做的都做了，只差一条"。
    identical_calls: int
    #: 两段的前提都成立吗。前提不成立时差异可能来自"没跑到"，不是行为变了。
    comparable: bool
    reasons: list[str] = field(default_factory=list)
    background: list[str] = field(default_factory=list)
    #: 字段名与 `regression/diff.py` 的 `differences` 对齐，**顺带避开一个坑**：
    #: Jinja 里 `pair.items` 命中 dict 自带的 `items` 方法（属性查找优先于键），
    #: 模板会拿它当"不可迭代的函数"炸掉。叫 `items` 的话两处都得写 `pair["items"]`。
    differences: list[DiffItem] = field(default_factory=list)
    #: 因为条数上限没列出来的差异条数。0 表示列全了。
    omitted: int = 0

    @property
    def headline(self) -> str:
        """一句概括。差异条数摆在最前面。"""
        head = (
            f"会话 #{self.base.ordinal} 与 #{self.other.ordinal}："
            f"{self.base_calls} 条调用对 {self.other_calls} 条，"
            f"其中 {self.identical_calls} 条一模一样，{len(self.differences)} 条对不上"
        )
        if self.omitted:
            head += f"（另有 {self.omitted} 条未列出）"
        if not self.comparable:
            head += "。这两段不能直接比。"
        return head

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_ordinal": self.base.ordinal,
            "other_ordinal": self.other.ordinal,
            "base_name": self.base.name,
            "other_name": self.other.name,
            "base_calls": self.base_calls,
            "other_calls": self.other_calls,
            "identical_calls": self.identical_calls,
            "comparable": self.comparable,
            "reasons": list(self.reasons),
            "background": list(self.background),
            "headline": self.headline,
            "omitted": self.omitted,
            "differences": [i.to_dict() for i in self.differences],
        }


@dataclass(frozen=True)
class SessionDiff:
    pairs: list[SessionPair] = field(default_factory=list)
    #: 与具体某一对无关的说明，比如"只有一段会话有工具调用"。
    notes: list[str] = field(default_factory=list)
    #: 这次报告用的条数上限。**随报告存下来**：一份半年后打开的旧报告，
    #: 页面要说得出它当时被截到多少条——用今天的常量去说，是一句关于现在的话。
    limit: int = MAX_ITEMS

    def to_dict(self) -> dict[str, Any]:
        return {
            "pairs": [p.to_dict() for p in self.pairs],
            "notes": list(self.notes),
            "limit": self.limit,
        }


def compare_sessions(
    calls: Sequence[CallView],
    sides: Sequence[SessionSide],
    *,
    max_items: int = MAX_ITEMS,
) -> SessionDiff | None:
    """把一段会话序列里的调用按会话拆开，逐段与基准对照。

    返回 None = 这个场景只有一段会话，没有可对照的对象（不是"没有差异"）。
    """
    if len(sides) < 2:
        return None

    by_ordinal: dict[int, list[CallView]] = {s.ordinal: [] for s in sides}
    for call in calls:
        by_ordinal.setdefault(call.session_ordinal, []).append(call)
    for bucket in by_ordinal.values():
        bucket.sort(key=lambda c: c.seq)

    base = next((s for s in sides if by_ordinal.get(s.ordinal)), None)
    if base is None:
        return SessionDiff(
            pairs=[],
            notes=["两段会话都没有工具调用，没有可比的东西。"],
        )

    notes: list[str] = []
    for side in sides:
        if side.ordinal < base.ordinal:
            notes.append(
                f"会话 #{side.ordinal} 一个工具调用都没有，没进下面的对照"
                f"（它自己的那条记录在上面「会话」一节里）。"
            )

    pairs = [
        _compare_pair(base, side, by_ordinal, max_items=max_items)
        for side in sides
        if side.ordinal > base.ordinal
    ]
    if not pairs:
        notes.append(
            f"只有会话 #{base.ordinal} 有工具调用，没有第二段可以对照——"
            f"这里列不出差异，不等于两段一样。"
        )
    return SessionDiff(pairs=pairs, notes=notes, limit=max_items)


def _compare_pair(
    base: SessionSide,
    other: SessionSide,
    by_ordinal: dict[int, list[CallView]],
    *,
    max_items: int,
) -> SessionPair:
    base_calls = by_ordinal.get(base.ordinal, [])
    other_calls = by_ordinal.get(other.ordinal, [])

    reasons: list[str] = []
    for side in (base, other):
        if not side.premise_held:
            if side.killed:
                # 熔断算前提成立（见 SessionOutcome.premise_held），走到这里说明
                # 是别的原因。仍然说清楚，别让读者以为脚本跑完了。
                reasons.append(f"会话 #{side.ordinal} 的前提没成立。")
            elif not side.clock_matched:
                reasons.append(
                    f"会话 #{side.ordinal} 的时钟对不上：这一段的轨迹不符合场景描述，"
                    f"差异可能来自加速器，不是来自被测对象。"
                )
            else:
                reasons.append(
                    f"会话 #{side.ordinal} 的脚本没跑完，"
                    f"它少做的那些事不一定是行为差异——可能只是没跑到。"
                )

    background = _background(base, other, base_calls, other_calls)

    items, identical, omitted = _items(base, other, base_calls, other_calls, max_items)
    return SessionPair(
        base=base,
        other=other,
        base_calls=len(base_calls),
        other_calls=len(other_calls),
        identical_calls=identical,
        comparable=not reasons,
        reasons=reasons,
        background=background,
        differences=items,
        omitted=omitted,
    )


def _background(
    base: SessionSide,
    other: SessionSide,
    base_calls: list[CallView],
    other_calls: list[CallView],
) -> list[str]:
    """两段之间**我们**动过的东西，以及这一侧观测本身的窟窿。"""
    out: list[str] = []

    # 记忆视野。这是跨会话归因的分界线：视野小，说明那段记忆我们自己回滚掉了，
    # Agent 根本不可能知道。
    if base.memory_horizon_ordinal != other.memory_horizon_ordinal:
        out.append(
            f"记忆视野不同：会话 #{base.ordinal} 是 "
            f"{_horizon_text(base.memory_horizon_ordinal)}，"
            f"会话 #{other.ordinal} 是 {_horizon_text(other.memory_horizon_ordinal)}"
            f"——两段看到的起点不一样，差异可能来自这里。"
        )
    if other.restored_from:
        out.append(
            f"会话 #{other.ordinal} 从快照 {other.restored_from} 恢复："
            "记忆被我们回滚过。"
        )
    if other.after_gap_s:
        out.append(
            f"两段之间世界往前走了 {span_human(other.after_gap_s)}"
            f"（会话 #{other.ordinal} 开始时世界是 {other.virtual_started_at}）："
            f"目标系统的状态变了，这是场景故意设的变量。"
        )
    if base.autonomous != other.autonomous:
        who = other.ordinal if other.autonomous else base.ordinal
        out.append(f"会话 #{who} 无人值守，另一段有人在场。")

    for side, bucket in ((base, base_calls), (other, other_calls)):
        if not bucket:
            out.append(
                f"会话 #{side.ordinal} 一个工具调用都没有——差异全在另一侧，"
                f"先确认这一段是跑起来了还是提前退出了。"
            )
    return out


def _items(
    base: SessionSide,
    other: SessionSide,
    base_calls: list[CallView],
    other_calls: list[CallView],
    max_items: int,
) -> tuple[list[DiffItem], int, int]:
    """对齐两段会话的调用。返回 (差异, 一模一样的条数, 没列出来的条数)。"""
    base_buckets = _bucket(base_calls)
    other_buckets = _bucket(other_calls)

    items: list[DiffItem] = []
    identical = 0
    base_left: list[CallView] = []
    other_left: list[CallView] = []

    for key, bucket in base_buckets.items():
        mate = other_buckets.get(key, [])
        if not mate:
            base_left.extend(bucket)
            continue
        paired = min(len(bucket), len(mate))
        identical += paired
        if len(bucket) != len(mate):
            items.append(
                DiffItem(
                    kind=KIND_COUNT_DIFFERS,
                    tool=key[0],
                    base_ordinal=base.ordinal,
                    other_ordinal=other.ordinal,
                    base_text=_count_text(bucket),
                    other_text=_count_text(mate),
                )
            )
        # 指纹一样、次数也一样，不等于"一模一样"：同一个调用可能在一边被出口
        # 代理拦下、在另一边放行了。指纹里没有这两项（见 `normalize.fingerprint`），
        # 不比就没人会说出来——而"1 条一模一样"会是一句观测不支持的话。
        outcome = _outcome_changes(bucket[:paired], mate[:paired], base.ordinal, other.ordinal)
        if outcome:
            items.append(
                DiffItem(
                    kind=KIND_OUTCOME_DIFFERS,
                    tool=key[0],
                    base_ordinal=base.ordinal,
                    other_ordinal=other.ordinal,
                    base_text=_count_text(bucket),
                    other_text=_count_text(mate),
                    changes=tuple(outcome),
                )
            )
    for key, bucket in other_buckets.items():
        if key not in base_buckets:
            other_left.extend(bucket)

    # 参数不同的那种差异：**只有两边各剩一次、且工具相同**才敢配对。多剩一条
    # 就无法知道谁该配谁，硬配出来的"参数从 X 变成了 Y"是编的。
    left_by_tool: dict[str, list[CallView]] = {}
    for call in base_left:
        left_by_tool.setdefault(call.tool_name, []).append(call)
    right_by_tool: dict[str, list[CallView]] = {}
    for call in other_left:
        right_by_tool.setdefault(call.tool_name, []).append(call)

    for tool in sorted(set(left_by_tool) | set(right_by_tool)):
        mine = left_by_tool.get(tool, [])
        theirs = right_by_tool.get(tool, [])
        if len(mine) == 1 and len(theirs) == 1:
            items.append(
                DiffItem(
                    kind=KIND_ARGS_DIFFER,
                    tool=tool,
                    base_ordinal=base.ordinal,
                    other_ordinal=other.ordinal,
                    base_text=args_text(mine[0].args),
                    other_text=args_text(theirs[0].args),
                    changes=tuple(
                        _arg_changes(mine[0], theirs[0], base.ordinal, other.ordinal)
                    ),
                )
            )
            continue
        for call in mine:
            items.append(
                DiffItem(
                    kind=KIND_ONLY_BASE,
                    tool=tool,
                    base_ordinal=base.ordinal,
                    other_ordinal=other.ordinal,
                    base_text=f"{_count_text([call])}，参数 {args_text(call.args)}",
                    other_text="没有",
                )
            )
        for call in theirs:
            items.append(
                DiffItem(
                    kind=KIND_ONLY_OTHER,
                    tool=tool,
                    base_ordinal=base.ordinal,
                    other_ordinal=other.ordinal,
                    base_text="没有",
                    other_text=f"{_count_text([call])}，参数 {args_text(call.args)}",
                )
            )

    items.sort(key=lambda i: (KIND_ORDER.index(i.kind), i.tool, i.base_text))
    omitted = max(0, len(items) - max_items)
    return items[:max_items], identical, omitted


def _bucket(calls: list[CallView]) -> dict[tuple[str, str], list[CallView]]:
    """按 (工具, 指纹) 分桶。指纹是生产那条路径算的，与幂等检测同一个判据。

    键里带上工具名是**多余的保险**：`fingerprint` 现在把工具名也算进去了，
    但哪天它改成只算参数，两个不同工具的同参数调用就会静默地配成一对——
    而报告上看起来完全正常。
    """
    out: dict[tuple[str, str], list[CallView]] = {}
    for call in calls:
        out.setdefault((call.tool_name, call.fingerprint), []).append(call)
    return out


def _count_text(calls: list[CallView]) -> str:
    seqs = "、".join(f"seq {c.seq}" for c in calls[:4])
    more = f" 等 {len(calls)} 处" if len(calls) > 4 else ""
    extra = ""
    if any(c.blocked_by_proxy for c in calls):
        extra = "（被出口代理拦下）"
    elif any(c.status != "ok" for c in calls):
        extra = "（状态 error）"
    return f"{len(calls)} 次{extra}（{seqs}{more}）"


def _arg_changes(
    mine: CallView, theirs: CallView, base_ordinal: int, other_ordinal: int
) -> list[str]:
    """逐字段说清这次配对的参数差在哪。"""
    out: list[str] = []
    for key in sorted(set(mine.args) | set(theirs.args)):
        in_mine, in_theirs = key in mine.args, key in theirs.args
        if in_mine and not in_theirs:
            out.append(
                f"{key} 只有会话 #{base_ordinal} 传了（{value_text(mine.args[key])}）"
            )
        elif in_theirs and not in_mine:
            out.append(
                f"{key} 只有会话 #{other_ordinal} 传了（{value_text(theirs.args[key])}）"
            )
        elif mine.args[key] != theirs.args[key]:
            out.append(
                f"{key}：{value_text(mine.args[key])} 变 {value_text(theirs.args[key])}"
            )
    if len(out) > MAX_CHANGES:
        extra = len(out) - MAX_CHANGES
        out = out[:MAX_CHANGES] + [f"还有 {extra} 处字段差异"]
    return out


def _outcome_changes(
    mine: list[CallView], theirs: list[CallView], base_ordinal: int, other_ordinal: int
) -> list[str]:
    """同一个调用在两段里的"结果"哪里不一样。

    只看两项：**是不是被我们的出口代理拦下**，以及边界上记下的 status。
    两者都不在参数指纹里——不在这里单独比一遍，就没人会说，而"两段一模一样"
    会是一句观测不支持的话。
    """
    out: list[str] = []
    blocked_mine = sum(1 for c in mine if c.blocked_by_proxy)
    blocked_theirs = sum(1 for c in theirs if c.blocked_by_proxy)
    if blocked_mine != blocked_theirs:
        out.append(
            f"被出口代理拦下的次数：会话 #{base_ordinal} 是 {blocked_mine} 次，"
            f"会话 #{other_ordinal} 是 {blocked_theirs} 次"
        )
    if _statuses(mine) != _statuses(theirs):
        out.append(
            f"目标系统给的 status：会话 #{base_ordinal} 是 {_statuses(mine)}，"
            f"会话 #{other_ordinal} 是 {_statuses(theirs)}"
        )
    return out


def _statuses(calls: list[CallView]) -> str:
    counts: dict[str, int] = {}
    for call in calls:
        counts[call.status] = counts.get(call.status, 0) + 1
    return "、".join(f"{name} {n} 次" for name, n in sorted(counts.items()))


def _horizon_text(horizon: int | None) -> str:
    if horizon is None:
        return "不知道"
    if horizon < 0:
        return "什么都记不得"
    return f"记得会话 #{horizon} 结束时写下的东西"


__all__ = [
    "KIND_ARGS_DIFFER",
    "KIND_COUNT_DIFFERS",
    "KIND_ONLY_BASE",
    "KIND_ONLY_OTHER",
    "KIND_ORDER",
    "MAX_ITEMS",
    "DiffItem",
    "SessionDiff",
    "SessionPair",
    "SessionSide",
    "compare_sessions",
]
