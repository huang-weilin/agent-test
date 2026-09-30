"""记忆内容的观测：会话边界上，Agent 的长效记忆里都有些什么。

**这是观测面，不是操纵面。** `snapshot.py` 那一套（capture / restore / store）是为了
主动把记忆回滚到某个时点——回滚到哪儿由我们决定，于是"Agent 本可以知道吗"有确定答案，
跨会话归因不需要猜。这里要回答的是另一个问题：**记忆的内容自己有没有变**——
上一段会话写下的结论还在不在、有没有多出一条本不该有的东西。

三条边界必须说清楚，否则这个模块会被读出它没说过的话：

1. **观测点是会话边界，不是"写入的那一刻"。** 记忆活在 Agent 进程里（见 ADR 001：
   网络边界拦不到进程内的文件）。我们只能在它每次进出时看一眼。所以"某个条目在
   会话中途被改过又改回去"这一类是看不见的——报告里不许暗示看得见。
2. **内容取自适配器自己的快照格式**，不另开一条读取路径。多一条读取路径就多一个
   "它和快照里的内容不一致"的可能，而那种不一致在报告上看不出来。
3. **观测不做判定。** 这里只负责"看见了什么"，至于算不算漂移、算不算中毒，由
   `sandbox.detectors.memory` 按场景**声明**的判据来决定。理由是全项目统一的套路：
   观测能被被测对象影响，判定依据必须来自声明。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

#: 会话边界上的两个观测点。`start` 是这段会话**开始时**记忆的样子（恢复之后、
#: Agent 跑起来之前），`end` 是 Agent 退出时（下一段会话恢复之前）。
PHASE_START = "start"
PHASE_END = "end"

#: 排序用。同一序号下先看 start 再看 end——顺序错了，"哪个是基准"就会反。
_PHASE_ORDER = {PHASE_START: 0, PHASE_END: 1}


def phase_label(ordinal: int, phase: str) -> str:
    """会话边界上某个时点的说法："会话 0 结束时"。

    **只有这一个地方拼它。** 检测结论、命令行报告、界面三处都要说出同一个时点
    名字，各拼一份的结果是它们迟早不一致——而且不一致的方式很难看：少一个空格
    （"会话 0结束时"），读的人不会觉得那是 bug，只会觉得报告马虎。
    """
    when = "开始" if phase == PHASE_START else "结束"
    return f"会话 {ordinal} {when}时"


@dataclass(frozen=True)
class MemoryEntry:
    """记忆里的一个条目。`text` 是解码后的内容，供人读与做标记检索。"""

    path: str
    sha256: str
    text: str


@dataclass(frozen=True)
class MemoryObservation:
    """某个会话边界上，记忆的一个切面。

    `horizon` 是这段会话开始时记忆覆盖到哪个序号（-1 = 什么都不记得），也就是
    `RunContext.memory_horizon_ordinal`。它和 `entries` 一起观测，理由和
    `CallView` 冗余那两个会话字段一样：**归因需要它**。

    少了它，"这条记忆不见了"就只能有一种解释——Agent 弄丢了。而实际上更常见的
    是我们自己回滚过：把记忆恢复到某个快照之后，快照之后写下的东西按设计就不该
    还在。两件事在报告上长得一样，但该修的地方完全不同（一个是 Prompt，一个是
    回滚点）。这正是幂等那边跨会话归因用的同一个判据。
    """

    ordinal: int
    phase: str
    entries: tuple[MemoryEntry, ...] = ()
    horizon: int | None = None

    @property
    def label(self) -> str:
        return phase_label(self.ordinal, self.phase)

    def get(self, path: str) -> MemoryEntry | None:
        for entry in self.entries:
            if entry.path == path:
                return entry
        return None

    def under(self, prefix: str) -> tuple[MemoryEntry, ...]:
        return tuple(e for e in self.entries if e.path.startswith(prefix))

    def digest_of(self, target: str) -> str | None:
        """一个文件或一个目录前缀在当前切面里的摘要。没有 → None。

        **目录用合并摘要而不是逐个比对**：把 `notes/` 全目录声明成"必须保持"时，
        要判的是这一整块内容没变，而逐个比对会在"文件改名"上给出一个和"内容变了"
        混在一起的结果——那两件事要分开说（见检测器里的说明）。

        `sha256:` 前缀是刻意的：这个值会出现在报告里，而报告里同时还有单文件的
        摘要，两者格式不区分的话读的人分不出哪个是哪个。
        """
        if target.endswith("/"):
            members = sorted(self.under(target), key=lambda e: e.path)
            if not members:
                return None
            payload = "\n".join(f"{e.path}\0{e.sha256}" for e in members)
            return f"sha256:{_sha256(payload)}"
        entry = self.get(target)
        return None if entry is None else f"sha256:{entry.sha256}"

    def texts(self) -> list[tuple[MemoryEntry, str]]:
        """(条目, 内容) 的列表。标记检索用。"""
        return [(entry, entry.text) for entry in self.entries]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chronological(observations: list[MemoryObservation]) -> list[MemoryObservation]:
    """按 (会话序号, start 先于 end) 排好。**判定前一律先过这一道**：

    语料是手写的，观测的书写顺序不保证；而"哪个是基准、哪个是后来"完全建立在
    顺序上——顺序反了，漂移判定会指向相反的方向，而报告看起来同样正常。
    """
    return sorted(
        observations, key=lambda o: (o.ordinal, _PHASE_ORDER.get(o.phase, 99))
    )


def observe_memory(
    adapter, *, ordinal: int, phase: str, horizon: int | None = None
) -> MemoryObservation:
    """在会话边界上拍一张记忆切面。

    `adapter` 只要满足 `MemoryAdapter` 协议即可——内容怎么读是适配器自己的事，
    这里不碰文件系统。适配器不支持枚举条目的话会在 `entries_of` 上报错，
    而不是悄悄返回一份空观测：**空观测会被读成"记忆里什么都没有"**。
    """
    return MemoryObservation(
        ordinal=ordinal,
        phase=phase,
        entries=tuple(adapter.entries_of(adapter.capture())),
        horizon=horizon,
    )


def summarize_observations(observations: list[MemoryObservation]) -> list[dict[str, object]]:
    """进报告的形状：**只说文件与摘要，不带内容**。

    报告要跨运行长期保留，而记忆内容里可能有用户真实数据（Agent 把订单号、
    地址写进笔记是很正常的）。要判"有没有变"，摘要就够了；真要看内容，快照
    就在本地库里，按摘要去取。

    另附 `changed`：相对上一个观测点，哪些条目出现了、哪些消失了、哪些内容变了。
    这一栏**不是判定**——Agent 往记忆里写东西是正常的。它的用处是让读报告的人
    看到"这段会话动过哪些文件"，而判定只发生在场景声明过的判据上。
    """
    ordered = chronological(observations)
    out: list[dict[str, object]] = []
    previous: dict[str, str] = {}
    for observation in ordered:
        digests = {entry.path: entry.sha256 for entry in observation.entries}
        changed = sorted(
            path
            for path, digest in digests.items()
            if path in previous and previous[path] != digest
        )
        added = sorted(path for path in digests if path not in previous)
        removed = sorted(path for path in previous if path not in digests)
        out.append(
            {
                "ordinal": observation.ordinal,
                "phase": observation.phase,
                # 这段会话开始时记忆覆盖到哪儿。**跨会话结论的归因靠它**：
                # horizon 比某件事发生的序号小，就说明那件事被回滚掉了，
                # "记忆不见了"得记在恢复机制头上而不是 Agent 头上。
                "horizon": observation.horizon,
                "entries": len(observation.entries),
                "paths": sorted(digests),
                "added": added,
                "removed": removed,
                "changed": changed,
            }
        )
        previous = digests
    return out
