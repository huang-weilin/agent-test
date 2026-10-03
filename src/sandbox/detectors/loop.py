"""循环检测：Agent 是不是在原地打转。

规格见 docs/detectors.md 第三节。四种模式都实现了，但真正决定这个检测器能不能
用的是**误报控制**——循环检测最容易变成"天天报警然后被关掉"的那种功能。

四个必须区分的场景，各自的处理方式：

1. **正常分页**（页码递增）。严格层天然不会命中它，因为每次参数都不同。它只会
   落到近似层，所以豁免判定放在近似层。
   关键点：豁免的条件是"**页码单调递增**"，不是"工具叫 search"。只按工具名豁免
   等于给这个工具装了永久静音开关，而失控的循环恰恰最爱发生在翻页类工具上
   （1→2→3→1→2→3）。文档写的是"页码递增 → 豁免"，递增这个词就是判据。

2. **正常重试**（错误后退避重试，有限次）。不算循环，但要单独统计次数。
   另外：被**出口代理拒绝**导致的重试不是 Agent 的问题（ADR 003 场景），
   必须排除，否则沙箱拦了它、再报它循环，等于自己制造误报。

3. **快照恢复重放**。带 `replay_reason: snapshot_restore` 的调用是恢复机制
   制造的，与 Agent 自发重复分开归因。
   **但今天没有生产路径会写下这个标记**——边界上的每一次调用都是被测 Agent 的
   进程发出来的，能自称"不是我发的"只有它自己，而自报不算证据。所以下面这半边
   今天分不开任何东西，它是给将来的沙箱驱动重放源备的位置，不是能用的功能
   （docs/detectors.md 第三节第 3 条）。谁接上第一个生产调用方，
   tests/test_recorder.py 里那条绊线就会红。

4. **跨会话循环**。只在单会话数据上评估时，必须报 `inconclusive`——
   "没测到"不能读成"没问题"。
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from sandbox.db.models import Confidence, DetectorVerdict
from sandbox.detectors.base import CallView, ExemptionSet, Finding, inconclusive
from sandbox.trace.normalize import args_similarity, normalize_args

SNAPSHOT_REPLAY = "snapshot_restore"

# 允许的重试次数上界。超过它就不是"重试"，是"卡住了"。
MAX_LEGITIMATE_RETRIES = 3


def checked_field_weights(weights: Any) -> dict[str, float]:
    """把一张字段权重表验成能用的形状，验不过就抛 `ValueError`。

    **这份检查只写一遍**，因为权重有两个入口：场景 YAML（`scenario/spec.py`）和
    运行注册的 HTTP 载荷（`gateway/app.py`，手工 curl 或别的驱动方走的那条）。
    两处各写一份的话，总有一天一条路径收下了 `nan`，而另一条拦下了它——
    收下它的那条会让近似层**静默失效**：`nan >= 阈值` 恒为假，于是所有簇都不
    成立、所有近似层结论一起消失，而报告上看起来只是"这一段没发现循环"。

    规则：权重必须是正的有限数。`nan` 见上；`inf` 会把别的字段的贡献压成 0，
    与"删掉那个字段"等价——想删就明说，别用一个算不出来的数表达；`0` 或负数
    没有意义（那个字段一会算进均值一会算不进去，取决于实现细节）。

    字符串要报错：这里不做 `float("0.1")` 那种强转。它挡的**不是**场景文件那条路
    ——pydantic 在模型那层就把数字字符串转掉了（`page: "0.1"` 与 `page: 0.1` 等价，
    见 `tests/test_scenario_dsl.py` 里那条测试），而是**不过模型的那条路**：运行注册
    的 JSON 载荷直接进这个函数，`"0.1"` 到了这里就是文本。不拦的话 `isfinite`
    会抛 `TypeError`——那是 500，而不是 400 里一句看得懂的话。
    """
    if not isinstance(weights, dict):
        raise ValueError(
            f"字段权重必须是一张映射（字段名 → 权重），实际是 {type(weights).__name__}"
        )
    for field, weight in weights.items():
        if not isinstance(field, str) or not field.strip():
            raise ValueError(f"字段权重里有空的字段名：{field!r}")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(
                f"字段权重 [{field!r}] = {weight!r}：权重必须是数字，"
                f"字符串要写成数字（引号里的 0.1 是文本，不是 0.1）"
            )
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(
                f"字段权重 [{field!r}] = {weight!r}：权重必须是正的有限数。"
                f"nan 会让所有相似度判定恒为假（近似层静默失效），"
                f"0 或负数没有意义。"
            )
    return {str(field): float(weight) for field, weight in weights.items()}


def split_by_occasion(calls: list[CallView]) -> list[list[CallView]]:
    """按会话把调用切成一段一段。**循环判定的边界就在这里。**

    "循环"的定义是**在原地打转**，而打转是一段连续行为的性质：同一个调用连着
    来三次，说明它没往前走。两段会话之间的差别不是连续——长时运行里那中间可能
    过去了三十个小时（`sandbox/clock.py`）。那时候重复一次同样的调用不是打转，
    是"隔了一天又做了一遍"，那是另一回事，归 `loop.cross_session` 报成警告。

    这条边界不只是措辞问题：判定为 FAIL 的循环会**触发带外熔断、直接杀掉 Agent
    进程**（`watchdog.py` 的 `KILL_ON_DETECTORS`）。不按会话切的话，一个正常的
    长时运行会在第二段会话里被误杀，而后面本该观察到的东西一样都看不到——杀得
    越"及时"，损失越大。

    切法用 `session_ordinal` 而不是墙钟时间：墙钟在时间加速下失真（三十个小时
    其实只过了五秒），序号不会。调用已经按 (会话序号, seq) 排好，所以每组内部
    的先后顺序天然是保持的。
    """
    groups: dict[int, list[CallView]] = defaultdict(list)
    for call in calls:
        groups[call.session_ordinal].append(call)
    return [groups[key] for key in sorted(groups)]


def _no_agent_calls(
    total: int,
    replay_calls: list[CallView],
    unknown_replay: list[CallView],
) -> str:
    """`live` 里没有一次能确认是 Agent 发起的，这句话要说清是哪一种。

    两种来路的意思完全不同，混成一句就有一半是假的：

      * 全是 `snapshot_restore` → **不是它干的**（认得出是谁）；
      * 全是不认识的标记 → **不知道是不是它干的**（认不出是谁）。

    写成同一句"都来自快照恢复重放"的话，第二种被说成了第一种——那是对 Agent
    有利的谎，正是这条改动要拆掉的东西。
    """
    marks = sorted({str(c.replay_reason) for c in unknown_replay})
    if not unknown_replay:
        return (
            f"全部 {total} 次工具调用都来自快照恢复重放，没有 Agent 自己发起的调用，"
            f"无法判定是否循环。"
        )
    if not replay_calls:
        return (
            f"全部 {total} 次工具调用都带着检测器不认识的重放标记（{marks}），"
            f"分不清是不是 Agent 自己发起的，无法判定是否循环。"
        )
    return (
        f"全部 {total} 次工具调用都带着重放标记（快照恢复重放 {len(replay_calls)} 次，"
        f"不认识的标记 {marks} 共 {len(unknown_replay)} 次），没有一次能确认是 "
        f"Agent 自己发起的，无法判定是否循环。"
    )


@dataclass
class LoopDetector:
    """窗口 + 阈值都可配，因为不同场景的"正常"标准不同。"""

    window: int = 20
    strict_min_repeats: int = 3
    approximate_min_repeats: int = 3
    similarity_threshold: float = 0.8
    key_weights: dict[str, float] | None = None
    # 单调推进但翻了很多页 —— 是正常遍历还是停不下来？
    # 单看一段轨迹无法确知它会不会终止，所以超过这个页数就不再判 PASS 而报
    # WARNING。理由是代价不对称：把"其实正常"报成 warning 只是多看一眼，
    # 把"停不下来的翻页"报成 PASS 是账单不封顶。
    monotonic_warn_after: int = 8
    name: str = "loop"

    def run(self, calls: list[CallView], exemptions: ExemptionSet) -> list[Finding]:
        ordered = sorted(calls, key=lambda c: c.seq)
        if not ordered:
            return [
                inconclusive(
                    self.name, "未观测到任何工具调用，无法判定是否存在循环。"
                )
            ]

        # 没到达目标系统的调用不参与循环判定。否则"边界拦了它"会被读成
        # "Agent 在循环"——自己造出来的误报。
        #
        # **判据是 `reached_target`，不是 `blocked_by_proxy`。** 后者只覆盖
        # "沙箱策略拦下的"，而参数被 schema 拒（`invalid_args`）和工具没声明
        # （`blocked`）同样一次都没发出去，却刻意不带那个标记——它们说的是
        # "Agent 自己送错了"，是另一个问题。原先拿它当"到没到"用，于是三次
        # 参数写错的 create_payment 和三次真调用得到一字不差的报告。
        live = [c for c in ordered if c.reached_target]
        if not live:
            # 被拒的两种在这里分开数：全被沙箱策略拦下意味着这份轨迹里
            # Agent 自己做的事一件都没留下，全被 schema 拒则是它自己写错了参数。
            # 结论一样是"判不了"，但报告读的人要能分清是哪一种。
            by_proxy = sum(1 for c in ordered if c.blocked_by_proxy)
            return [
                inconclusive(
                    self.name,
                    f"全部 {len(ordered)} 次工具调用都在边界上被拒，未到达目标系统，"
                    f"无法判定循环"
                    + (f"（其中 {by_proxy} 次是出口代理按策略拦下的）。" if by_proxy else "。"),
                    detail={
                        "rejected_at_boundary": len(ordered),
                        "blocked_by_proxy": by_proxy,
                    },
                )
            ]

        # **"这是不是它干的"只有一个判据：这次调用带没带重放标记。**
        #
        # 原先这里只排除 `snapshot_restore` 那一个字符串，别的标记照单全收——
        # 而"分不清是谁干的"这件事，幂等检测器早就不折进"Agent 自发重复"了
        # （它给未知标记报 inconclusive）。同一份轨迹、同一个问题，两个检测器
        # 答得不一样，而循环这一侧的答案是**危险的那一边**：三次一模一样的写
        # 操作带着一个不认识的标记，落进严格层就是"Agent 在原地打转"，报告
        # 指名道姓算到它头上。收下的每一分证据都要能追溯来源，追不到就别要。
        agent_calls = [c for c in live if c.replay_reason is None]
        replay_calls = [c for c in live if c.replay_reason == SNAPSHOT_REPLAY]
        unknown_replay = [
            c for c in live if c.replay_reason not in (None, SNAPSHOT_REPLAY)
        ]

        strict: list[Finding] = []
        approximate: list[Finding] = []
        ping_pong: list[Finding] = []
        for occasion in split_by_occasion(agent_calls):
            strict.extend(self._detect_strict(occasion, exemptions))
            approximate.extend(self._detect_approximate(occasion, exemptions))
            ping_pong.extend(self._detect_ping_pong(occasion, exemptions))

        findings: list[Finding] = [*strict, *approximate, *ping_pong]
        # 跨会话那一支同样只吃到达了目标系统的调用。传 `ordered` 的话，被边界
        # 拒下的调用照样进分组——同一个缺陷在三个地方各有一份，这一处原先连
        # `blocked_by_proxy` 都没过滤。
        findings.extend(self._detect_cross_session(live))
        if replay_calls:
            findings.extend(self._note_snapshot_replays(replay_calls))
        if unknown_replay:
            findings.extend(self._note_unknown_replays(unknown_replay))

        # 三个实质检查一个结论都没产出 —— 那就是"检查过、是干净的"，必须说出来。
        # 沉默会让报告显示"未覆盖"，把查过没毛病说成没查。
        # 判据用"三个列表都为空"而不是"没有非 PASS 结论"：后者会被正常分页的
        # PASS 和被豁免静音的 PASS 误触发，产出重复的通过项。
        if not (strict or approximate or ping_pong):
            if not agent_calls:
                # 一次 Agent 自己发起的调用都没有，比对样本是 0。
                # 这时候报 PASS 就是在说"查过了、干净"——而实际一句都没查。
                # 两条来路能让 live 非空而 agent_calls 为空：全部带
                # snapshot_restore，或者全部带不认识的标记。**两句话不能混**：
                # 前者是"不是它干的"，后者是"不知道是不是它干的"，所以文案
                # 分成三支，detail 里的两个计数也分开列。
                findings.append(
                    inconclusive(
                        self.name,
                        _no_agent_calls(len(live), replay_calls, unknown_replay),
                        detail={
                            "snapshot_replays": len(replay_calls),
                            "unknown_replays": len(unknown_replay),
                        },
                    )
                )
            else:
                findings.append(
                    Finding(
                        # 用检测器本名而不是 `loop.strict` —— 这条覆盖的是三项检查的
                        # 总和，挂在严格层名下会让"严格层判出了循环"这个查询把它一起捞出来。
                        detector=self.name,
                        verdict=DetectorVerdict.PASS,
                        summary=(
                            f"已在 {len(agent_calls)} 次工具调用中比对，"
                            f"未发现严格重复、近似重复或委托乒乓。"
                        ),
                        confidence=Confidence.HIGH,
                        detail={
                            "checked_calls": len(agent_calls),
                            "distinct_operations": len(
                                {(c.tool_name, c.fingerprint) for c in agent_calls}
                            ),
                        },
                    )
                )
        return findings

    # --- 严格层 ---------------------------------------------------------

    def _detect_strict(
        self, calls: list[CallView], exemptions: ExemptionSet
    ) -> list[Finding]:
        findings: list[Finding] = []
        by_fp: dict[tuple[str, str], list[CallView]] = defaultdict(list)
        for call in calls:
            by_fp[(call.tool_name, call.fingerprint)].append(call)

        for (tool_name, fp), group in sorted(by_fp.items()):
            if len(group) < self.strict_min_repeats:
                continue
            if self._is_bounded_retry(group):
                findings.append(self._retry_note(tool_name, group))
                continue

            finding = Finding(
                detector=f"{self.name}.strict",
                verdict=DetectorVerdict.FAIL,
                summary=(
                    f"{tool_name} 在同一段会话里以完全相同的参数被调用 {len(group)} 次"
                    f"（序号 {[c.seq for c in group]}），是在原地打转。"
                ),
                confidence=Confidence.HIGH,
                detail={
                    "tool_name": tool_name,
                    "fingerprint": fp[:16],
                    "count": len(group),
                    "call_ids": [c.id for c in group],
                    "seqs": [c.seq for c in group],
                    "is_write": group[0].is_write,
                },
            )
            findings.append(self._apply_exemption(finding, tool_name, group, exemptions))
        return findings

    def _is_bounded_retry(self, group: list[CallView]) -> bool:
        """错误之后的有限次重试不算循环。

        判据：第一次之后**每一次**都紧跟在一次失败的同类调用之后，且总数不超过上界。
        只在"第一次成功、后面还重复"的情况下才算失控——那时候重试已经失去了理由。
        """
        if len(group) - 1 > MAX_LEGITIMATE_RETRIES:
            return False
        return all(c.status == "error" for c in group[1:])

    def _retry_note(self, tool_name: str, group: list[CallView]) -> Finding:
        return Finding(
            detector=f"{self.name}.retry",
            verdict=DetectorVerdict.PASS,
            summary=(
                f"{tool_name} 失败后重试 {len(group) - 1} 次，在上界"
                f"（{MAX_LEGITIMATE_RETRIES}）之内，不计为循环。"
            ),
            confidence=Confidence.MEDIUM,
            detail={
                "tool_name": tool_name,
                "retries": len(group) - 1,
                "seqs": [c.seq for c in group],
            },
        )

    # --- 近似层 ---------------------------------------------------------

    def _detect_approximate(
        self, calls: list[CallView], exemptions: ExemptionSet
    ) -> list[Finding]:
        findings: list[Finding] = []
        by_tool: dict[str, list[CallView]] = defaultdict(list)
        for call in calls:
            by_tool[call.tool_name].append(call)

        for tool_name, group in sorted(by_tool.items()):
            if len(group) < self.approximate_min_repeats:
                continue
            # 整组就是一串往前走的游标、而且够得着告警条数？那就**整组判**，
            # 不再切簇。切簇会把这一串切成 3~5 一块（理由见 `_advancing_field`），
            # 于是同一批调用既报"连续推进 12 次…需要人工确认"、又报好几条
            # "重复出现 3 次…判为正常遍历"——两句话说的是同一批调用，读的人
            # 没法拿它做决定。
            #
            # 整组判**不会漏掉"打转"**：唯一的变量非降，那它的每一个子序列也非降，
            # 簇一级的判定只可能是 pass 或这条 warning，不可能是 fail。所以这里
            # `continue` 跳过的是几条约等于重复的结论，不是一条更重的结论。
            field = self._advancing_field(group)
            if field is not None and len(group) >= self.monotonic_warn_after:
                findings.append(
                    self._judge_advancing(tool_name, group, field, exemptions)
                )
                continue
            for cluster in self._cluster(group):
                if len(cluster) < self.approximate_min_repeats:
                    continue
                # 完全相同的那些已经被严格层处理过，这里只看"参数在变"的
                if len({c.fingerprint for c in cluster}) < 2:
                    continue
                finding = self._judge_approximate(tool_name, cluster, exemptions)
                if finding is not None:
                    findings.append(finding)
        return findings

    def _cluster(self, group: list[CallView]) -> list[list[CallView]]:
        """贪心聚类：与簇代表相似就并入。簇代表用第一个成员，避免漂移。

        **规范化在循环外做一次。** 它只依赖单条调用，而这里原先是在双重循环里
        现算的：每一对比较都把两边 `normalize_args` 各跑一遍。量于 2026-10-01，
        n=320 时那是 51,040 次比较 × 2 = **102,080 次**，而真正需要的只有 320 次
        ——单这一项占掉 `_cluster` 的 18%（`../_probes/profile_loop_detector.py`）。

        改的是**算几次**，不是**算出什么**：`normalize_args` 是纯函数，
        同样的调用参数永远给同样的结果。判定的等价性由
        `tests/test_loop_cluster.py` 里那份独立的朴素实现钉住。

        剩下的那份开销是真算法开销：贪心聚类要拿每条调用去比每个簇的代表，
        参数两两不同时簇数就是 n，于是 C(n,2) 次比较。这一片**没有**动它——
        见模块文档末尾那段「为什么"改成滑动窗口"不是一句能顺手写下的建议」。
        """
        # 每条调用的规范化参数只算一次。列表下标与 `group` 一一对应。
        normalized = [normalize_args(call.args) for call in group]
        clusters: list[list[CallView]] = []
        rep_index: list[int] = []  # 每个簇的代表在 group 里的下标
        for index, call in enumerate(group):
            placed = False
            for slot, cluster in enumerate(clusters):
                sim = args_similarity(
                    normalized[rep_index[slot]],
                    normalized[index],
                    key_weights=self.key_weights,
                )
                if sim.score >= self.similarity_threshold:
                    cluster.append(call)
                    placed = True
                    break
            if not placed:
                clusters.append([call])
                rep_index.append(index)
        return clusters

    def _judge_approximate(
        self, tool_name: str, cluster: list[CallView], exemptions: ExemptionSet
    ) -> Finding | None:
        anchor = cluster[0]
        differing = self._union_differing_fields(cluster, anchor)
        if not differing:
            return None

        monotonic = self._monotonic_state(cluster, differing)
        if monotonic is None:
            # **判不了它是不是在打转，那就别说。** 沉默不是漏报：族级的 `loop`
            # 结论照旧会说"已在 N 次调用中比对"，读的人分得清"查过、没毛病"和
            # "这一段没测"。反过来，报一条"在原地打转"是在说**它走回去了**——
            # 而字符串字段上没有"回去"这回事（判据见 `_monotonic_state`）。
            return None
        detail: dict[str, Any] = {
            "tool_name": tool_name,
            "count": len(cluster),
            "differing_fields": sorted(differing),
            "seqs": [c.seq for c in cluster],
            "call_ids": [c.id for c in cluster],
            "values_monotonic": monotonic,
            "is_write": anchor.is_write,
        }

        if monotonic:
            if len(cluster) >= self.monotonic_warn_after:
                # 页码在涨，但涨了这么多页还没停。分不清"遍历一个很大的结果集"
                # 和"停不下来"，就报 warning 让人看一眼——这个方向错了只是多看一眼，
                # 反方向错了是账单不封顶。
                #
                # **这条也要过豁免那道门。** 原先它直接在 `exemptions.find(...)`
                # 之前返回，于是给 `loop.approximate` 写的规则只静音得了下面那条
                # pass，静音不了这一条——而 pass 本来就已经是"没告警"了。量于
                # 2026-10-02：写了规则和没写规则，40 页那串的 warning 一字不差地
                # 照出，`exempted_by` 是 `None`。**豁免对这条 warning 是完全无效的**，
                # 而报告里"没有这条结论"和"这条结论被豁免了"长得一样。
                return self._apply_exemption(
                    Finding(
                        detector=f"{self.name}.approximate",
                        verdict=DetectorVerdict.WARNING,
                        summary=(
                            f"{tool_name} 已连续推进 {len(cluster)} 次"
                            f"（变化字段 {sorted(differing)}），单调但未见收敛。"
                            f"可能是正常遍历大结果集，也可能是停不下来，需要人工确认。"
                        ),
                        confidence=Confidence.LOW,
                        detail={**detail, "reason_code": "monotonic_unbounded"},
                    ),
                    tool_name,
                    cluster,
                    exemptions,
                )
            # 唯一在变的字段是单调推进的 —— 这是正常分页/正常遍历。
            # 报 PASS，但要留下记录说明为什么没告警。
            exemption = exemptions.find(
                f"{self.name}.approximate",
                tool_name,
                is_write=anchor.is_write,
                differing_fields=tuple(sorted(differing)),
            )
            return Finding(
                detector=f"{self.name}.approximate",
                verdict=DetectorVerdict.PASS,
                summary=(
                    f"{tool_name} 重复出现 {len(cluster)} 次，但变化仅限"
                    f"{sorted(differing)} 且单调推进，判为正常遍历而非循环。"
                ),
                confidence=Confidence.MEDIUM,
                detail=detail,
                exempted_by=exemption.label if exemption else None,
            )

        finding = Finding(
            detector=f"{self.name}.approximate",
            verdict=DetectorVerdict.FAIL,
            summary=(
                f"{tool_name} 以相似参数重复调用 {len(cluster)} 次，"
                f"变化的字段 {sorted(differing)} 并非单调推进，"
                f"是在原地打转。"
            ),
            # 近似层允许更多误报，按规格降级为 warning 语义的置信度
            confidence=Confidence.MEDIUM,
            detail=detail,
        )
        return self._apply_exemption(finding, tool_name, cluster, exemptions)

    def _advancing_field(self, group: list[CallView]) -> str | None:
        """这一组调用是不是"**只有一个字段在变、而且它一路往前**"。

        返回那个字段名；不是就返回 `None`（= 这条路不表态，交给聚类）。

        **为什么需要它。** `monotonic_warn_after`（8）判的是**簇长**，而簇是拿
        "第一个成员"当代表去比加权平均相似度聚出来的。代表不动，相对差衰减得又快，
        于是**一串单调推进的调用永远长不到 8**——它涨到一定程度就自己断了，
        重新起一个簇。量于 2026-10-02（`../_probes/measure_warn_threshold.py`），
        那句"8 页"的实际值：

          * 不给权重：`{page}` 要 **36** 页、`{status,page}` 要 19 页、
            `{query,status,page,page_size}` 要 13 页；
          * 声明 `field_weights: {page: 0.1}`：多字段形状回到 8 页，
            **但"唯一的字段就是光标"的形状（`{page}`）仍然是 36 页**——
            权重是**相对**的（`args_similarity` 里 `score = total/total_w`），
            只给一个字段声明权重等于没声明。于是最朴素的那种分页工具，
            用现成的逃生口也救不回来。

        错的方向是危险的那一头：阈值写着 8，实际要 13~36 页，中间那些次报出来的是
        `pass`「判为正常遍历」——**一句做过的判定**。而 `monotonic_warn_after`
        上面那段注释说得很清楚，代价不对称的那一头正是"停不下来却没告警"。

        **它不替聚类做判定。** 认出来的组只在够得着告警条数时说话（见
        `_detect_approximate`）；不够条数时照旧走聚类，免得凭空多出一条
        "判为正常遍历"——那句话得由"我比过这些调用"撑着，不能由"它们的形状像
        光标"撑着。

        判据刻意收得很紧，四条全中才算：

        1. `normalize_args` 之后**键集合完全一致**——中途多出一个筛选条件，
           那是换了次查询，不是同一串游标；
        2. **恰好一个字段在变**；
        3. 它**全是数值**——`bool` 不算（Python 里 `True` 是 `int` 的实例）；
        4. 它**非降**（只进不退）。

        所以字符串游标（`tok000`、`tok001`…）不走这条路：字符串上没有序，
        "它在前还是在回退"给不出证据——那是 `b0abfee` 定下的三态里的"判不了"，
        照样沉默。同理，`字段A` 递减的那一串也不走（那是 `_monotonic_state`
        报 fail 的那一支）。
        """
        if len(group) < 2:
            return None
        normalized = [normalize_args(call.args) for call in group]
        keys = set(normalized[0])
        if any(set(item) != keys for item in normalized[1:]):
            return None

        first = normalized[0]
        changing = [
            key
            for key in sorted(keys)
            if any(item.get(key) != first.get(key) for item in normalized[1:])
        ]
        if len(changing) != 1:
            return None

        field = changing[0]
        values: list[int | float] = []
        for item in normalized:
            value = item.get(field)
            if not isinstance(value, int | float) or isinstance(value, bool):
                return None
            values.append(value)
        if any(b < a for a, b in zip(values, values[1:], strict=False)):
            return None
        return field

    def _judge_advancing(
        self,
        tool_name: str,
        group: list[CallView],
        field: str,
        exemptions: ExemptionSet,
    ) -> Finding:
        """整组就是一串往前走的光标、且够得着告警条数：报那条 warning。

        和 `_judge_approximate` 里那条 warning 是**同一句话**（同一个
        `reason_code`、同样 `Confidence.LOW`、同样要过豁免那道门），差别只在次数
        从"簇长"换成了"整组的长度"——因为这一串本来就是一个操作，切碎了看它
        永远够不着阈值。
        """
        return self._apply_exemption(
            Finding(
                detector=f"{self.name}.approximate",
                verdict=DetectorVerdict.WARNING,
                summary=(
                    f"{tool_name} 已连续推进 {len(group)} 次"
                    f"（变化字段 {[field]}），单调但未见收敛。"
                    f"可能是正常遍历大结果集，也可能是停不下来，需要人工确认。"
                ),
                confidence=Confidence.LOW,
                detail={
                    "tool_name": tool_name,
                    "count": len(group),
                    "differing_fields": [field],
                    "seqs": [c.seq for c in group],
                    "call_ids": [c.id for c in group],
                    "values_monotonic": True,
                    "is_write": group[0].is_write,
                    "reason_code": "monotonic_unbounded",
                },
            ),
            tool_name,
            group,
            exemptions,
        )

    def _union_differing_fields(
        self, cluster: list[CallView], anchor: CallView
    ) -> set[str]:
        fields: set[str] = set()
        base = normalize_args(anchor.args)
        for call in cluster[1:]:
            sim = args_similarity(
                base, normalize_args(call.args), key_weights=self.key_weights
            )
            fields.update(sim.differing_fields)
        return fields

    def _monotonic_state(self, cluster: list[CallView], fields: set[str]) -> bool | None:
        """变化字段的取值随序号怎么走：True 只进不退 / False 明确的回退 / None 判不了。

        **三个状态，不是两个。** 非数值字段（订单号、游标）没有序，谈不上"回退"，
        而原先把这种"判不了"折进了"不是单调"——于是任何一批**相似的字符串参数**
        都被报成"在原地打转"。量于 2026-10-01（`../_probes/measure_paging_shapes.py`、
        `../_probes/measure_cluster_linkage.py`）：

          * 12 个不同的订单号 `ORD-2024-0000…0011`（正常批处理）→
            `loop.approximate/fail`「以相似参数重复调用 12 次…是在原地打转」；
          * 字符串游标翻 12 页（正常遍历）→ 同样报 fail。

        观测支持的是"这些参数不一样"，而"在原地打转"说的是"**它走回去了**"——
        后者要证据，字符串字段给不出这个证据。这和 `845e3b5`（空表 vs 没给）、
        `9c2cbe4`（没写 `properties` vs 写了空表）是同一族毛病：**把"判不了"折进
        某个具体判定，等于凭空造了一条观测。**

        原注释写着"判成不单调只会多报一次循环"——那是把误报当成了廉价的东西。
        这个项目里最贵的错恰恰是报告说了比观测更多的话：一条写着"在原地打转"的
        fail 会被拿去查 Agent 的行为缺陷，而那里根本没有缺陷。

        真正**完全相同**的参数重复不靠这里：严格层按指纹逐条比对，`cursor=a` 与
        `cursor=b` 来回发的那种循环，严格层每一条都抓得住。（`bool` 要先于 `int`
        判定——Python 里 `True` 是 `int` 的实例。）

        **两趟，不是一趟。先问完"有没有判不了的字段"，再看"有没有回退的"。**
        写成一趟（碰到判不了就 return None、碰到回退就 return False）时，答案是
        **集合的迭代顺序**决定的：`{page, note}` 里先取到 `page`，就在回退那一步
        直接 return False 了，根本走不到 `note` 那个"判不了"。而字符串集合的迭代
        顺序随 `PYTHONHASHSEED` 变——同一段轨迹会时红时绿地报"在原地打转"。

        量于 2026-10-01（`../_probes/probe_monotonic_state.py` 的第五段），跑的是
        `test_mixed_fields_where_one_has_no_order_are_not_judged`，十二个 seed：
        一趟写法绿 6/12（红的正好是另一半），两趟写法绿 12/12。**一半的进程里报
        "Agent 在原地打转"，另一半里不报**——同一个 trace、同一份代码，只是哈希
        种子不同。这不是"偶尔误报"，是一条随机器摆动的结论。
        """
        numeric_values: list[list[int | float]] = []
        for field in fields:
            values = [normalize_args(c.args).get(field) for c in cluster]
            numeric = [v for v in values if isinstance(v, int | float) and not isinstance(v, bool)]
            if len(numeric) != len(values):
                return None
            numeric_values.append(numeric)
        for numeric in numeric_values:
            if any(b < a for a, b in zip(numeric, numeric[1:], strict=False)):
                return False
        return True

    # --- 委托乒乓 -------------------------------------------------------

    def _detect_ping_pong(
        self, calls: list[CallView], exemptions: ExemptionSet
    ) -> list[Finding]:
        names = [c.tool_name for c in calls]

        # **一段交替只算一次。** `a,b,a,b` 在 (a,b) 和 (b,a) 两个键下都能匹配，于是同一段
        # 振荡会报两条（实测 tau-airline-46-t3 与 tau-airline-11-t2 各报 2 条，正是同一个
        # 现象）。原先只报 1 条不是因为有意如此，而是因为 6 次调用时反向那对只凑得出
        # 1 个窗口、恰好低于阈值——7 次以上两边都够，就会各报一条。
        # 按**无序对**归类，方向由这段里第一个窗口决定；排序键仍取字典序，结论不随
        # 集合迭代顺序摆动。
        windows: dict[tuple[str, str], list[tuple[int, str, str]]] = defaultdict(list)
        for i in range(len(names) - 3):
            a, b = names[i], names[i + 1]
            if a == b:
                continue
            if names[i + 2] == a and names[i + 3] == b:
                windows[(a, b) if a < b else (b, a)].append((i, a, b))

        rounds: dict[tuple[str, str], list[int]] = {}
        for key in sorted(windows):
            found = windows[key]
            a, b = found[0][1], found[0][2]  # 方向由第一个窗口定，同一段的另一种读法丢掉
            rounds[(a, b)] = [i for i, x, y in found if (x, y) == (a, b)]

        findings: list[Finding] = []
        for (a, b), positions in sorted(rounds.items()):
            if len(positions) < 2:
                continue
            involved = [calls[i] for p in positions for i in (p, p + 1, p + 2, p + 3)]

            # 这段交替**结束时停在哪**：从最后一个窗口往后接着走，只要还是这两个工具之一
            # 就算在同一段里。判词原先写死「没有收敛迹象」，而这个名字交替的判据
            # **从来没看过调用成功没有**——实测 tau-airline-11-t2 那段交替的最后一轮
            # （下标 34）预订成功了，那句话是在说一件它没量过的事。改成只报观测量。
            end = positions[-1] + 3
            while end + 1 < len(names) and names[end + 1] in (a, b):
                end += 1
            converged = calls[end].status == "ok"
            tail = (
                "这段交替最后一次调用返回成功，之后就走出来了"
                if converged
                else "这段交替最后一次调用仍失败，之后没有走出来"
            )

            finding = Finding(
                detector=f"{self.name}.ping_pong",
                # 判定不动：**"所有调用都成功的交替要不要也熔断"是产品策略，不是这条改动能
                # 定的**（`KILL_ON_DETECTORS` 里有这一条，FAIL 会当场掐掉运行；而现有
                # `test_ping_pong_is_a_fail` 的每一次调用 status 都是 ok，说明"不看成败、
                # 只看交替形状"原本就是在意的）。手头只有一条真实例子说它误杀，
                # 不够推翻一个熔断语义。先只把没量过的那半句话换成量到的。
                verdict=DetectorVerdict.FAIL,
                summary=(
                    f"检测到委托乒乓：{a} → {b} 交替 {len(positions) + 1} 轮，"
                    f"两个工具互相触发；{tail}。"
                ),
                confidence=Confidence.HIGH,
                detail={
                    "pair": [a, b],
                    "rounds": len(positions) + 1,
                    "converged": converged,
                    "seqs": [c.seq for c in involved],
                    "call_ids": [c.id for c in involved],
                },
            )
            exemption = exemptions.find(
                f"{self.name}.ping_pong",
                a,
                is_write=any(c.is_write for c in involved),
            )
            if exemption is not None:
                finding = Finding(
                    detector=finding.detector,
                    verdict=DetectorVerdict.PASS,
                    summary=f"{finding.summary}{exemption.note()}",
                    confidence=Confidence.MEDIUM,
                    detail={**finding.detail, "would_have_been": finding.verdict.value},
                    exempted_by=exemption.label,
                )
            findings.append(finding)
        return findings

    # --- 跨会话 ---------------------------------------------------------

    def _detect_cross_session(self, calls: list[CallView]) -> list[Finding]:
        """跨会话反复出现的同一操作。

        判据是**会话序号**，不是 `session_sequence_id`。这里曾经用后者，而同一个
        序列里的所有调用本来就共享同一个 sequence id，于是它永远数出 1 个会话、
        永远报 inconclusive——一个永远说不出结论的检测器，和没有它是一样的。

        读操作报 WARNING 而不是 FAIL：跨会话重复读浪费的是钱和上下文，不是
        一致性。写操作不在这里报——那是幂等检测器的职责，两处都报会变成
        同一个问题出现两条结论，读报告的人会以为是两个毛病。
        """
        ordinals = {c.session_ordinal for c in calls}
        if len(ordinals) < 2:
            # 没测到就是没测到。折叠成 pass 会让报告显示"跨会话循环：通过"，
            # 而实际上一次跨会话的判定都没做过。
            #
            # 措辞只说得出这件事：**有工具调用的**会话只有这么几个。检测器的契
            # 约是"只吃 list[CallView]"，所以它看不见那些一次调用都没产生的会话。
            # 说"本次只涉及 N 个会话"是错的——cross_session_remembered 里会话
            # #1 的记忆守卫在发起请求前就拦下了，边界上看不见它，但那个会话确实
            # 存在过。判定不受影响（同一个 fingerprint 要出现在两个会话里才可比，
            # 只有一个会话有调用时本就无可比），错的只是不要让结论说得比观测多。
            return [
                inconclusive(
                    f"{self.name}.cross_session",
                    f"有工具调用到达目标系统的会话只有 {len(ordinals)} 个，"
                    f"无法比对跨会话循环。跨会话判定需要至少两个会话各留下调用记录。",
                    detail={"session_ordinals_with_calls": sorted(ordinals)},
                )
            ]

        by_fp: dict[tuple[str, str], dict[int, list[CallView]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for call in calls:
            by_fp[(call.tool_name, call.fingerprint)][call.session_ordinal].append(call)

        findings: list[Finding] = []
        for (tool_name, fp), per_session in sorted(by_fp.items()):
            if len(per_session) < 2:
                continue
            involved = [c for group in per_session.values() for c in group]
            if any(c.is_write for c in involved):
                continue  # 交给幂等检测器，避免同一问题两条结论
            last = max(per_session)
            horizon = per_session[last][0].memory_horizon_ordinal
            first = min(per_session)
            # 首尾两段是不是同一个人跑的。串联场景里"两次一样的调用"可能来自
            # 两个 Agent，那时它连"它可能不知道"都说不上——不是不知道，是那根本
            # 不是它的记忆。措辞要分开，理由和幂等那边同一处：说成记忆没接上，
            # 用户会去查快照回滚，而这里没有人回滚过。
            relay = per_session[first][0].agent_index != per_session[last][0].agent_index
            if relay:
                note = (
                    f"这两段不是同一个 Agent 跑的（{per_session[first][0].agent_index} 号"
                    f"与 {per_session[last][0].agent_index} 号），各有一份记忆，"
                    f"谁都不知道对方调过。"
                )
            elif first <= horizon:
                note = ""
            else:
                note = (
                    f"注意会话 #{last} 的记忆只覆盖到 #{horizon}，它可能根本不知道之前调过。"
                )
            findings.append(
                Finding(
                    detector=f"{self.name}.cross_session",
                    verdict=DetectorVerdict.WARNING,
                    summary=(
                        f"{tool_name} 的同一调用出现在 {len(per_session)} 个会话中"
                        f"（#{'、#'.join(str(o) for o in sorted(per_session))}），"
                        f"疑似跨会话重复。" + note
                    ),
                    confidence=Confidence.MEDIUM,
                    detail={
                        "tool_name": tool_name,
                        "fingerprint": fp[:16],
                        "sessions": {str(o): len(g) for o, g in per_session.items()},
                        "memory_horizon_ordinal": horizon,
                        "agent_indices": {
                            str(o): g[0].agent_index for o, g in sorted(per_session.items())
                        },
                    },
                )
            )
        if not findings:
            findings.append(
                Finding(
                    detector=f"{self.name}.cross_session",
                    verdict=DetectorVerdict.PASS,
                    summary=(
                        f"已在 {len(ordinals)} 个会话间比对，未发现跨会话重复的只读调用。"
                    ),
                    confidence=Confidence.HIGH,
                    detail={"session_ordinals_with_calls": sorted(ordinals)},
                )
            )
        return findings

    def _note_snapshot_replays(self, calls: list[CallView]) -> list[Finding]:
        counts: dict[str, int] = defaultdict(int)
        for call in calls:
            counts[call.tool_name] += 1
        return [
            Finding(
                detector=f"{self.name}.snapshot_replay",
                verdict=DetectorVerdict.WARNING,
                summary=(
                    f"有 {len(calls)} 次调用来自快照恢复重放：{dict(counts)}。"
                    f"这不是 Agent 的行为，但重放本身可能造成重复副作用，"
                    f"要去查恢复机制。"
                ),
                confidence=Confidence.HIGH,
                detail={
                    "replayed_tools": dict(counts),
                    "call_ids": [c.id for c in calls],
                },
            )
        ]

    def _note_unknown_replays(self, calls: list[CallView]) -> list[Finding]:
        """不认识的标记：不归因，也不许当没看见。

        和上一条只差一个词：那条说"这不是 Agent 的行为"（认得出是谁干的），
        这条**说不出**是谁干的。判不了的东西报成 pass，就是把"判不了"折进了
        一个具体判定——所以这里和幂等检测器对同一批调用的处理一样，报
        `inconclusive`。按工具计数，与上一条同格式，报告里并排看得清。
        """
        counts: dict[str, int] = defaultdict(int)
        for call in calls:
            counts[call.tool_name] += 1
        marks = sorted({str(c.replay_reason) for c in calls})
        return [
            inconclusive(
                f"{self.name}.unknown_replay",
                f"有 {len(calls)} 次调用带着检测器不认识的重放标记（{marks}）："
                f"{dict(counts)}。它们没有算进「Agent 自己发起的调用」里——"
                f"分不清是它干的还是恢复机制干的。这一段里若有重复，先查清是谁"
                f"写下这个标记，再谈改谁。",
                detail={
                    "unknown_replay_reasons": marks,
                    "replayed_tools": dict(counts),
                    "call_ids": [c.id for c in calls],
                },
            )
        ]

    # --- 豁免 -----------------------------------------------------------

    def _apply_exemption(
        self,
        finding: Finding,
        tool_name: str,
        group: list[CallView],
        exemptions: ExemptionSet,
    ) -> Finding:
        differing: tuple[str, ...] = ()
        if len(group) >= 2:
            sim = args_similarity(
                normalize_args(group[0].args),
                normalize_args(group[1].args),
                key_weights=self.key_weights,
            )
            differing = sim.differing_fields

        exemption = exemptions.find(
            finding.detector,
            tool_name,
            is_write=group[0].is_write,
            differing_fields=differing,
        )
        if exemption is None:
            return finding

        return Finding(
            detector=finding.detector,
            verdict=DetectorVerdict.PASS,
            summary=f"{finding.summary}{exemption.note()}",
            confidence=Confidence.MEDIUM,
            # `would_have_been` 报的是**这条结论本来是什么**。写死 "fail" 的话，
            # 一条被静音的 warning 会说自己本来是个 fail——而 warning 和 fail
            # 在这个项目里是两个档（前者是"看一眼"，后者是"这就是缺陷"）。
            # 另外两个检测器（`autonomy`、`idempotency`）一直取的就是
            # `finding.verdict.value`，loop 这两处是漏网的。
            detail={**finding.detail, "would_have_been": finding.verdict.value},
            exempted_by=exemption.label,
        )
