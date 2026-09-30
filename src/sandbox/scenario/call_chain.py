"""调用链（需求 3.2 的 P1「调用链可视化」）。

要回答的是需求原文那句话：**错误如何在工具调用链中传播**。

## 边界上看得到什么

一次工具调用在边界上留下的只有几样：调了什么工具、带了什么参数、**是不是
成功了**、以及**没成功是谁的原因**。链就是把一段会话里的这些调用按发生顺序
排出来，把没成功的那几次标出来，再写清"那一次之后紧接着发生的是什么"。

## 我们能证明的是顺序，不是因果

需求里用的是"传播"，而"传播"是个因果词。边界上看得见的是**紧接着**：第 3 次
调用没成功，第 4 次调用还是同一个工具、参数一模一样。**它在重试**是读的人
下的结论；报告只写得出"紧接着的那一次和它是同一个调用"。所以这一节通篇写
"紧接着"，不写"因为……所以……"——两个词只差一点，但一个是我们观测到的，
另一个是我们替 Agent 解释的，而解释错了没有任何一处会报错。

## 四条自我约束

1. **没成功不是判定。** 目标系统回了个错，是发生在这条链上的事，不等于被测
   对象做错了事。这里列的东西**不进 `findings`、不影响 `ok`**：谁对谁错看
   检测结论那一节。这一节只负责把顺序摆出来。

2. **谁的错分开写。** 目标系统回的错 / 参数不合规被边界拒绝 / 工具没在场景里
   声明 / 被我们自己的出口策略拦下——四种在链上是四种标记。把"沙箱拦的"混进
   "目标系统报的错"里，等于替被测对象背锅。这与 `blocked_by_proxy` 是同一条
   规矩的两个入口。

3. **快照重放不算它重发。** 从快照恢复时我们会重放调用，那是恢复机制制造的
   （见 `docs/detectors.md` 的风险 R-replay）。紧接着出现一次重放的调用，
   链上要写清"这一条是重放"——否则它看起来正好像是它在原样重试。

4. **没出错的链也要有正面证据。** "9 次调用，没有一次不成功"和"这一节是空的"
   看起来一样，而前者是结论、后者是没查。干净的那一段要自己说出来。
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sandbox.detectors.base import CallView
from sandbox.formatting import args_text
from sandbox.scenario.session_diff import SessionSide

# --- 标记 -------------------------------------------------------------------
#: 成功。
MARK_OK = "ok"
#: 目标系统回了非 2xx。**最硬的一种"出错"**：真实世界的回答。
MARK_TARGET_ERROR = "target_error"
#: 参数不符合场景声明的 schema，在边界上被拒。真实 API 一样会拒，所以这是
#: 被测对象自己的问题——因此不标 `blocked_by_proxy`（见 `proxy/app.py`）。
MARK_INVALID_ARGS = "invalid_args"
#: 调了场景里没声明的工具，被边界拒绝。同样是它自己的问题。
MARK_UNDECLARED_TOOL = "undeclared_tool"
#: 被沙箱的出口策略拦下。**这不是它的失败**，是我们动的手。
MARK_SANDBOX_BLOCKED = "sandbox_blocked"
#: 快照恢复时的重放。也不是它主动发的。
MARK_REPLAYED = "replayed"
#: 认不出来的 status。不许猜成"目标系统报错"——猜错方向会冤枉目标系统。
MARK_UNKNOWN = "unknown"

MARK_LABELS: dict[str, str] = {
    MARK_OK: "成功",
    MARK_TARGET_ERROR: "没成功：目标系统回了错",
    MARK_INVALID_ARGS: "没成功：参数不符合声明的 schema，在边界上被拒",
    MARK_UNDECLARED_TOOL: "没成功：调了场景里没声明的工具",
    MARK_SANDBOX_BLOCKED: "被沙箱的出口策略拦下（这不是它的失败）",
    MARK_REPLAYED: "快照恢复时的重放（不是它主动发的）",
    MARK_UNKNOWN: "边界记下了一个认不出来的 status（不是成功）",
}

#: 它自己的失败。`ours` 的那些不算——那两种是我们制造的。
FAILED_MARKS: frozenset[str] = frozenset(
    {MARK_TARGET_ERROR, MARK_INVALID_ARGS, MARK_UNDECLARED_TOOL, MARK_UNKNOWN}
)
#: 我们制造的，不进"它失败了"的计数。
OURS_MARKS: frozenset[str] = frozenset({MARK_SANDBOX_BLOCKED, MARK_REPLAYED})

#: 每次调用前面的小标记。**不复用检测结论那套 `ok` / `!!` / `?` / `..`**：
#: 那些是判定，这里是陈述，同一套符号会让读者以为链上标了 `!!` 的调用已经被
#: 判成缺陷了。`·` 成功、`!!` 它自己没成功、`--` 我们动的手。
GLYPH_OK = "·"
GLYPH_FAILED = "!!"
GLYPH_OURS = "--"

# --- 紧接着的那一次调用的关系 ------------------------------------------------
REL_SAME_CALL = "same_call"
REL_REPLAYED = "replayed"
REL_SAME_TOOL = "same_tool"
REL_OTHER_TOOL = "other_tool"
REL_STOPPED = "stopped"

#: 一段会话最多列多少次调用。超了**必须在报告里说**——静默截断会让"就调了
#: 这几下"读成"后面什么都没发生"，而后面往往正是出事的地方。
MAX_NODES = 30
#: 一段会话最多列几条没成功的调用（按 (工具, 指纹, 标记) 合并之后）。
MAX_STEPS = 8


@dataclass(frozen=True)
class ChainNode:
    """链上的一次调用。"""

    #: 在这段会话里的位置：第几次工具调用。**报告里印的是它**——运行序号
    #: `seq` 与 LLM 调用共用一个计数器（一次工具调用前后各有一条 LLM 调用），
    #: 印出来会是 2、4、6、9 这种跳号，读者会以为中间有几次没显示出来。
    position: int
    tool: str
    args_text: str
    mark: str
    #: 运行序号。要和审计页、检测结论里的序号对上时用它（JSON 里带着）。
    seq: int = 0
    #: 边界原样记下的那个 status。只在 `MARK_UNKNOWN` 的措辞里用得上——
    #: 认不出来的时候要把原值印出来，不然读者连去哪儿查都不知道。
    status: str = "ok"
    is_write: bool = False

    @property
    def failed(self) -> bool:
        return self.mark in FAILED_MARKS

    @property
    def ours(self) -> bool:
        return self.mark in OURS_MARKS

    @property
    def glyph(self) -> str:
        if self.failed:
            return GLYPH_FAILED
        if self.ours:
            return GLYPH_OURS
        return GLYPH_OK

    @property
    def mark_text(self) -> str:
        """标记的人话，不带连接号。成功是空的——成功不需要解释。"""
        if self.mark == MARK_OK:
            return ""
        if self.mark == MARK_UNKNOWN:
            return f"没成功：边界记下了一个认不出来的 status（{self.status!r}）"
        return MARK_LABELS.get(self.mark, self.mark)

    @property
    def mark_label(self) -> str:
        text = self.mark_text
        return f"——{text}" if text else ""

    def line(self) -> str:
        """命令行里的一行。"""
        return (
            f"{self.glyph} 第 {self.position} 次 {self.tool}"
            f"（{self.args_text}）{self.mark_label}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "seq": self.seq,
            "tool": self.tool,
            "args_text": self.args_text,
            "mark": self.mark,
            "mark_text": self.mark_text,
            "glyph": self.glyph,
            "is_write": self.is_write,
            "line": self.line(),
        }


@dataclass(frozen=True)
class ChainStep:
    """一种没成功的调用，以及紧接着它发生的事。

    按 `(工具, 指纹, 标记)` 合并：同样的调用在链上出现了几次，就是"这条出错
    沿着链延续了几次"。合并是刻意的不精确——逐次列出来的话，一段循环会刷满
    屏，而读者要的恰恰是"同样的失败一共几次、中间隔着什么"。
    """

    node: ChainNode
    #: 同样的调用（同工具、同参数、同标记）在这段会话里出现的位置。
    positions: tuple[int, ...]
    #: 第一次之后紧接着的那次调用。None = 这一次之后这一段就没有调用了。
    after_first_tool: str | None
    after_first_relation: str
    #: 最后一次之后紧接着的那次调用。与上者不同的时候才单独说。
    after_last_tool: str | None
    after_last_relation: str
    #: 这一段是不是被熔断终止的。决定"结束了"该怎么读。
    killed: bool = False

    @property
    def times(self) -> int:
        return len(self.positions)

    @property
    def ours(self) -> bool:
        return self.node.ours

    def sentence(self) -> str:
        """一句话说清这一次没成功、之后又发生了什么。

        措辞上**不区分"它失败"和"我们拦下"**——两者的区别已经写在 `mark_label`
        里了（"这不是它的失败"）。这里再分一套句式，等于同一件事有两个写法。
        """
        head = (
            f"第 {self.node.position} 次调用 {self.node.tool}"
            f"（{self.node.args_text}）{self.node.mark_label}"
        )
        if self.times == 1:
            return f"{head}。紧接着{self._tail(self.after_first_tool, self.after_first_relation)}"

        places = "、".join(str(p) for p in self.positions)
        first_tail = self._tail(self.after_first_tool, self.after_first_relation)
        tail = f"第 {self.node.position} 次之后紧接着{first_tail}"
        last = self.positions[-1]
        if (self.after_last_tool, self.after_last_relation) != (
            self.after_first_tool,
            self.after_first_relation,
        ):
            last_tail = self._tail(self.after_last_tool, self.after_last_relation)
            tail += f"；第 {last} 次之后紧接着{last_tail}"
        return (
            f"{head}。同样的调用在这段会话里一共出现了 {self.times} 次"
            f"（第 {places} 次）：{tail}"
        )

    def _tail(self, tool: str | None, relation: str) -> str:
        if relation == REL_REPLAYED:
            return f"是一次快照重放（{tool}），不是它主动重发"
        if relation == REL_SAME_CALL:
            return f"又是同一个调用（{tool}），参数一模一样"
        if relation == REL_SAME_TOOL:
            return f"还是 {tool}，参数变了"
        if relation == REL_OTHER_TOOL:
            return f"换成了别的工具（{tool}）"
        if self.killed:
            return "这一段的调用就结束了（这一段被熔断终止）"
        return "这一段的调用就结束了"

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.node.position,
            "positions": list(self.positions),
            "times": self.times,
            "tool": self.node.tool,
            "mark": self.node.mark,
            "ours": self.ours,
            "sentence": self.sentence(),
        }


@dataclass(frozen=True)
class SessionChain:
    """一段会话的链。"""

    ordinal: int
    name: str = ""
    nodes: list[ChainNode] = field(default_factory=list)
    #: 这一段总共有多少次调用。与 `len(nodes)` 不同就是被截了。
    total: int = 0
    #: 它自己没成功的次数、我们动的手的次数。**分开计**，因为前者是行为，
    #: 后者是我们的观测设置。
    failed: int = 0
    ours: int = 0
    steps: list[ChainStep] = field(default_factory=list)
    omitted_nodes: int = 0
    omitted_steps: int = 0
    #: 这一段的前提没成立（脚本没跑完 / 时钟对不上）。链上看到的顺序依然是真的，
    #: 但它不指向被测对象的行为——由「会话」那节给理由，这里只指过去。
    premise_held: bool = True
    killed: bool = False

    @property
    def headline(self) -> str:
        if not self.total:
            return f"会话 #{self.ordinal}：一次工具调用都没有。"
        head = f"会话 #{self.ordinal}：{self.total} 次工具调用"
        parts = []
        if self.failed:
            parts.append(f"{self.failed} 次没成功")
        if self.ours:
            parts.append(f"{self.ours} 次是沙箱拦下或重放的")
        if parts:
            head += "，" + "，".join(parts)
        else:
            # 正面证据。不说的话，"这一节很短"会被读成"没查"。
            head += "，没有一次不成功"
        head += "。"
        if self.omitted_nodes:
            head += (
                f"下面只按顺序列了前 {len(self.nodes)} 次，"
                f"还有 {self.omitted_nodes} 次没列出来。"
            )
        return head

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "name": self.name,
            "total": self.total,
            "failed": self.failed,
            "ours": self.ours,
            "headline": self.headline,
            "omitted_nodes": self.omitted_nodes,
            "omitted_steps": self.omitted_steps,
            "premise_held": self.premise_held,
            "killed": self.killed,
            "nodes": [n.to_dict() for n in self.nodes],
            "steps": [s.to_dict() for s in self.steps],
        }


@dataclass(frozen=True)
class CallChain:
    chains: list[SessionChain] = field(default_factory=list)
    #: 这次报告用的两个上限。**随报告存下来**：一份旧报告要能说出它当时是按
    #: 多少截的——用今天的常量去说，是一句关于现在的话，没法核对。
    node_limit: int = MAX_NODES
    step_limit: int = MAX_STEPS

    def to_dict(self) -> dict[str, Any]:
        return {
            "chains": [c.to_dict() for c in self.chains],
            "node_limit": self.node_limit,
            "step_limit": self.step_limit,
        }


def build_chain(
    calls: Sequence[CallView],
    sides: Sequence[SessionSide],
    *,
    max_nodes: int = MAX_NODES,
    max_steps: int = MAX_STEPS,
) -> CallChain | None:
    """把一次运行的调用按会话排成链。返回 None = 这次运行一段会话都没有。

    与跨会话差异报告共用 `SessionSide`：那边读的是"能不能拿来比"，这里读的是
    "这段的顺序该怎么措辞"，两边都只认这几项。会话对象的其它字段（stdout、
    退出码……）这里一概不读——拿进程输出去解释它为什么这么调工具，是猜。
    """
    if not sides:
        return None

    by_ordinal: dict[int, list[CallView]] = {s.ordinal: [] for s in sides}
    for call in calls:
        by_ordinal.setdefault(call.session_ordinal, []).append(call)

    chains = [
        _session_chain(side, by_ordinal.get(side.ordinal, []), max_nodes, max_steps)
        for side in sides
    ]
    return CallChain(chains=chains, node_limit=max_nodes, step_limit=max_steps)


def _session_chain(
    side: SessionSide,
    calls: list[CallView],
    max_nodes: int,
    max_steps: int,
) -> SessionChain:
    ordered = sorted(calls, key=lambda c: c.seq)
    nodes = [
        ChainNode(
            position=index + 1,
            seq=call.seq,
            tool=call.tool_name,
            args_text=args_text(call.args),
            mark=mark_of(call),
            status=call.status,
            is_write=call.is_write,
        )
        for index, call in enumerate(ordered)
    ]
    steps = _steps(ordered, nodes, killed=side.killed)
    return SessionChain(
        ordinal=side.ordinal,
        name=side.name,
        nodes=nodes[:max_nodes],
        total=len(nodes),
        failed=sum(1 for n in nodes if n.failed),
        ours=sum(1 for n in nodes if n.ours),
        steps=steps[:max_steps],
        omitted_nodes=max(0, len(nodes) - max_nodes),
        omitted_steps=max(0, len(steps) - max_steps),
        premise_held=side.premise_held,
        killed=side.killed,
    )


def mark_of(call: CallView) -> str:
    """边界记下的这一次调用，算哪一种。

    顺序是有讲究的：**先认重放和沙箱拦截**（那两个是我们干的，会把 status
    盖成 blocked），再认它自己的失败。反过来先看 status 的话，一次被拦下的
    调用会先被读成"没成功"，再被读成"我们拦的"——最终结果对，但中间那句
    是错的。
    """
    if call.replay_reason:
        return MARK_REPLAYED
    if call.blocked_by_proxy:
        return MARK_SANDBOX_BLOCKED
    if call.status == "invalid_args":
        return MARK_INVALID_ARGS
    if call.status == "blocked":
        # 走到这里说明不是出口策略拦的，那就是调了没声明的工具（见 proxy/app.py）。
        return MARK_UNDECLARED_TOOL
    if call.status == "ok":
        return MARK_OK
    if call.status == "error":
        return MARK_TARGET_ERROR
    return MARK_UNKNOWN


def _steps(
    ordered: list[CallView],
    nodes: list[ChainNode],
    *,
    killed: bool,
) -> list[ChainStep]:
    """按 (工具, 指纹, 标记) 合并出"没成功的那些调用"。"""
    groups: dict[tuple[str, str, str], list[int]] = {}
    for index, (call, node) in enumerate(zip(ordered, nodes, strict=True)):
        if node.mark == MARK_OK:
            continue
        groups.setdefault((call.tool_name, call.fingerprint, node.mark), []).append(index)

    steps: list[ChainStep] = []
    for indexes in groups.values():
        first, last = indexes[0], indexes[-1]
        after_first_tool, after_first_rel = _after(ordered, nodes, first)
        after_last_tool, after_last_rel = _after(ordered, nodes, last)
        steps.append(
            ChainStep(
                node=nodes[first],
                positions=tuple(nodes[i].position for i in indexes),
                after_first_tool=after_first_tool,
                after_first_relation=after_first_rel,
                after_last_tool=after_last_tool,
                after_last_relation=after_last_rel,
                killed=killed,
            )
        )
    # 按**第一次出现的位置**排。链的意义就是顺序，按工具名字母序排会把它抹掉。
    steps.sort(key=lambda s: s.node.position)
    return steps


def _after(
    ordered: list[CallView], nodes: list[ChainNode], index: int
) -> tuple[str | None, str]:
    """第 index 次调用之后紧接着的那一次是什么关系。"""
    if index + 1 >= len(ordered):
        return None, REL_STOPPED
    nxt_call, nxt_node = ordered[index + 1], nodes[index + 1]
    if nxt_node.mark == MARK_REPLAYED:
        # 先认重放：一次重放和原调用指纹一模一样，不特判的话它会看起来
        # 正好像是"它在原样重试"——而那是恢复机制干的。
        return nxt_call.tool_name, REL_REPLAYED
    if nxt_call.fingerprint == ordered[index].fingerprint:
        return nxt_call.tool_name, REL_SAME_CALL
    if nxt_call.tool_name == ordered[index].tool_name:
        return nxt_call.tool_name, REL_SAME_TOOL
    return nxt_call.tool_name, REL_OTHER_TOOL


__all__ = [
    "FAILED_MARKS",
    "MARK_LABELS",
    "MAX_NODES",
    "MAX_STEPS",
    "OURS_MARKS",
    "CallChain",
    "ChainNode",
    "ChainStep",
    "SessionChain",
    "build_chain",
    "mark_of",
]
