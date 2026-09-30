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

4. **跨会话循环**。只在单会话数据上评估时，必须报 `inconclusive`——
   "没测到"不能读成"没问题"。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from sandbox.db.models import Confidence, DetectorVerdict
from sandbox.detectors.base import CallView, ExemptionSet, Finding, inconclusive
from sandbox.trace.normalize import args_similarity, normalize_args

SNAPSHOT_REPLAY = "snapshot_restore"

# 允许的重试次数上界。超过它就不是"重试"，是"卡住了"。
MAX_LEGITIMATE_RETRIES = 3


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

        # 被代理拒绝的调用没有到达目标系统，不参与循环判定。
        # 否则"沙箱拦了它"会被读成"Agent 在循环"——自己造出来的误报。
        live = [c for c in ordered if not c.blocked_by_proxy]
        if not live:
            return [
                inconclusive(
                    self.name,
                    f"全部 {len(ordered)} 次工具调用都被出口代理拒绝，未到达目标系统，"
                    f"无法判定循环。",
                    detail={"blocked_by_proxy": len(ordered)},
                )
            ]

        agent_calls = [c for c in live if c.replay_reason != SNAPSHOT_REPLAY]
        replay_calls = [c for c in live if c.replay_reason == SNAPSHOT_REPLAY]

        strict = self._detect_strict(agent_calls, exemptions)
        approximate = self._detect_approximate(agent_calls, exemptions)
        ping_pong = self._detect_ping_pong(agent_calls, exemptions)

        findings: list[Finding] = [*strict, *approximate, *ping_pong]
        findings.extend(self._detect_cross_session(ordered))
        if replay_calls:
            findings.extend(self._note_snapshot_replays(replay_calls))

        # 三个实质检查一个结论都没产出 —— 那就是"检查过、是干净的"，必须说出来。
        # 沉默会让报告显示"未覆盖"，把查过没毛病说成没查。
        # 判据用"三个列表都为空"而不是"没有非 PASS 结论"：后者会被正常分页的
        # PASS 和被豁免静音的 PASS 误触发，产出重复的通过项。
        if not (strict or approximate or ping_pong):
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
                    f"{tool_name} 以完全相同的参数被调用 {len(group)} 次"
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
        """贪心聚类：与簇代表相似就并入。簇代表用第一个成员，避免漂移。"""
        clusters: list[list[CallView]] = []
        for call in group:
            placed = False
            for cluster in clusters:
                rep = cluster[0]
                sim = args_similarity(
                    normalize_args(rep.args),
                    normalize_args(call.args),
                    key_weights=self.key_weights,
                )
                if sim.score >= self.similarity_threshold:
                    cluster.append(call)
                    placed = True
                    break
            if not placed:
                clusters.append([call])
        return clusters

    def _judge_approximate(
        self, tool_name: str, cluster: list[CallView], exemptions: ExemptionSet
    ) -> Finding | None:
        anchor = cluster[0]
        differing = self._union_differing_fields(cluster, anchor)
        if not differing:
            return None

        monotonic = self._all_monotonic(cluster, differing)
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
            exemption = exemptions.find(
                f"{self.name}.approximate",
                tool_name,
                is_write=anchor.is_write,
                differing_fields=tuple(sorted(differing)),
            )
            if len(cluster) >= self.monotonic_warn_after:
                # 页码在涨，但涨了这么多页还没停。分不清"遍历一个很大的结果集"
                # 和"停不下来"，就报 warning 让人看一眼——这个方向错了只是多看一眼，
                # 反方向错了是账单不封顶。
                return Finding(
                    detector=f"{self.name}.approximate",
                    verdict=DetectorVerdict.WARNING,
                    summary=(
                        f"{tool_name} 已连续推进 {len(cluster)} 次"
                        f"（变化字段 {sorted(differing)}），单调但未见收敛。"
                        f"可能是正常遍历大结果集，也可能是停不下来，需要人工确认。"
                    ),
                    confidence=Confidence.LOW,
                    detail={**detail, "reason_code": "monotonic_unbounded"},
                )
            # 唯一在变的字段是单调推进的 —— 这是正常分页/正常遍历。
            # 报 PASS，但要留下记录说明为什么没告警。
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
                f"变化的字段 {sorted(differing)} **不是**单调推进，"
                f"是在原地打转。"
            ),
            # 近似层允许更多误报，按规格降级为 warning 语义的置信度
            confidence=Confidence.MEDIUM,
            detail=detail,
        )
        return self._apply_exemption(finding, tool_name, cluster, exemptions)

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

    def _all_monotonic(self, cluster: list[CallView], fields: set[str]) -> bool:
        """变化字段的取值是否随序号单调推进（允许相等，不允许回退）。"""
        for field in fields:
            values = [normalize_args(c.args).get(field) for c in cluster]
            numeric = [v for v in values if isinstance(v, int | float) and not isinstance(v, bool)]
            if len(numeric) != len(values):
                # 非数值字段无法谈单调性，保守判为不单调。
                # 保守在这里是安全的：判成不单调只会多报一次循环，
                # 判成单调会让真正的循环静默通过。
                return False
            if any(b < a for a, b in zip(numeric, numeric[1:], strict=False)):
                return False
        return True

    # --- 委托乒乓 -------------------------------------------------------

    def _detect_ping_pong(
        self, calls: list[CallView], exemptions: ExemptionSet
    ) -> list[Finding]:
        names = [c.tool_name for c in calls]
        rounds: dict[tuple[str, str], list[int]] = defaultdict(list)

        for i in range(len(names) - 3):
            a, b = names[i], names[i + 1]
            if a == b:
                continue
            if names[i + 2] == a and names[i + 3] == b:
                rounds[(a, b)].append(i)

        findings: list[Finding] = []
        for (a, b), positions in sorted(rounds.items()):
            if len(positions) < 2:
                continue
            involved = [calls[i] for p in positions for i in (p, p + 1, p + 2, p + 3)]
            finding = Finding(
                detector=f"{self.name}.ping_pong",
                verdict=DetectorVerdict.FAIL,
                summary=(
                    f"检测到委托乒乓：{a} → {b} 交替 {len(positions) + 1} 轮，"
                    f"两个工具互相触发，没有收敛迹象。"
                ),
                confidence=Confidence.HIGH,
                detail={
                    "pair": [a, b],
                    "rounds": len(positions) + 1,
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
                    summary=f"{finding.summary}（已被豁免规则 {exemption.label} 静音）",
                    confidence=Confidence.MEDIUM,
                    detail={**finding.detail, "would_have_been": "fail"},
                    exempted_by=exemption.label,
                )
            findings.append(finding)
        return findings

    # --- 跨会话 ---------------------------------------------------------

    def _detect_cross_session(self, calls: list[CallView]) -> list[Finding]:
        sessions = {c.session_sequence_id for c in calls if c.session_sequence_id}
        if len(sessions) < 2:
            # 没测到就是没测到。折叠成 pass 会让报告显示"跨会话循环：通过"，
            # 而实际上一次跨会话的判定都没做过。
            return [
                inconclusive(
                    f"{self.name}.cross_session",
                    f"本次运行只涉及 {len(sessions)} 个会话，无法判定跨会话循环。",
                    detail={"sessions": sorted(sessions)},
                )
            ]

        by_fp: dict[tuple[str, str], set[str]] = defaultdict(set)
        for call in calls:
            by_fp[(call.tool_name, call.fingerprint)].add(call.session_sequence_id)

        findings: list[Finding] = []
        for (tool_name, fp), seen in sorted(by_fp.items()):
            if len(seen) < 2:
                continue
            findings.append(
                Finding(
                    detector=f"{self.name}.cross_session",
                    verdict=DetectorVerdict.WARNING,
                    summary=(
                        f"{tool_name} 的同一次调用出现在 {len(seen)} 个不同会话中，"
                        f"疑似跨会话重复。"
                    ),
                    confidence=Confidence.MEDIUM,
                    detail={
                        "tool_name": tool_name,
                        "fingerprint": fp[:16],
                        "sessions": sorted(seen),
                    },
                )
            )
        if not findings:
            findings.append(
                Finding(
                    detector=f"{self.name}.cross_session",
                    verdict=DetectorVerdict.PASS,
                    summary=f"已在 {len(sessions)} 个会话间比对，未发现跨会话重复调用。",
                    confidence=Confidence.HIGH,
                    detail={"sessions": sorted(sessions)},
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
            summary=f"{finding.summary}（已被豁免规则 {exemption.label} 静音）",
            confidence=Confidence.MEDIUM,
            detail={**finding.detail, "would_have_been": "fail"},
            exempted_by=exemption.label,
        )
