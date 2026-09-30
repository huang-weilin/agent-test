"""幂等检测：写操作被重复执行了吗。

规格见 docs/detectors.md 第四节。三处超出字面规格但必须做的判断：

**一、"工具有幂等保护"和"保护真的生效了"是两件事。** 需求只说"对无幂等保护的
写操作告警"。但如果工具声明了幂等键、Agent 却每次重试都生成一个新键，保护就
形同虚设，照样重复扣款。只看"工具是否声明了保护"会漏掉这一类——而它恰恰是
实践中最常见的失败方式：框架自动填幂等键，填入的是随机 UUID。
所以要看 `(指纹, 幂等键)` 这一对，而不只是指纹。

**二、被代理拦下的调用不算重复扣款。** 它根本没到目标系统。混进来会让"Agent
尝试越权被拦"变成"Agent 重复扣款"，性质完全不同。用 `blocked_by_proxy` 排除。

**三、快照恢复导致的重放要单独归因。** 重复扣款确实发生了，但制造它的是恢复
机制，不是 Agent。报告必须说清是谁的错——否则用户会去改一个没问题的 Prompt。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from sandbox.db.models import Confidence, DetectorVerdict
from sandbox.detectors.base import CallView, ExemptionSet, Finding, inconclusive

SNAPSHOT_REPLAY = "snapshot_restore"


@dataclass(frozen=True)
class ToolFacts:
    """检测器需要知道的工具属性。由场景的工具声明提供。"""

    name: str
    is_write: bool
    enforces_idempotency: bool = False


class IdempotencyDetector:
    name = "idempotency"

    def __init__(self, tool_facts: dict[str, ToolFacts] | None = None) -> None:
        self.tool_facts = tool_facts or {}

    def _facts(self, call: CallView) -> ToolFacts:
        return self.tool_facts.get(
            call.tool_name,
            ToolFacts(name=call.tool_name, is_write=call.is_write, enforces_idempotency=False),
        )

    def run(self, calls: list[CallView], exemptions: ExemptionSet) -> list[Finding]:
        # 被代理拒绝的调用没到达目标系统，不构成重复扣款。
        observed = [c for c in calls if not c.blocked_by_proxy]
        writes = [c for c in observed if c.is_write]

        if not writes:
            return [
                inconclusive(
                    self.name,
                    "未观测到任何写操作，无法判定幂等性。"
                    "注意这不等于通过——可能是 Agent 根本没走到写路径。",
                    detail={
                        "observed_calls": len(observed),
                        "blocked_by_proxy": sum(1 for c in calls if c.blocked_by_proxy),
                    },
                ),
                # 读重复的成本信息在没有写操作时同样有价值，不能跟着一起丢掉
                *self._read_repeat_note(observed),
            ]

        findings: list[Finding] = []
        groups: dict[tuple[str, str], list[CallView]] = defaultdict(list)
        for call in writes:
            groups[(call.tool_name, call.fingerprint)].append(call)

        judged = 0
        for (tool_name, fingerprint), group in sorted(groups.items()):
            if len(group) < 2:
                continue
            finding = self._judge(tool_name, fingerprint, group, exemptions)
            if finding is not None:
                judged += 1
                findings.append(finding)

        # 判据是"有没有判定过任何一组"，而不是"有没有非 PASS 结论"。
        # 用后者的话，一条被豁免静音成 PASS 的结论会再触发一次补充 PASS，
        # 报告里就会出现两条互相重复的通过项。
        if judged == 0:
            # 跑了、检查了、没发现问题 —— 这就是 pass，必须显式说出来。
            # 不说的后果是报告显示"覆盖率 0%"，把"测过了没毛病"读成"什么都没测"。
            # 那和把 inconclusive 读成 pass 是同一个错误的反面，一样有害。
            tools_checked = sorted({c.tool_name for c in writes})
            findings.append(
                Finding(
                    detector=self.name,
                    verdict=DetectorVerdict.PASS,
                    summary=(
                        f"已检查 {len(writes)} 次写操作（{tools_checked}），未发现重复执行。"
                    ),
                    confidence=Confidence.HIGH,
                    detail={
                        "checked_writes": len(writes),
                        "distinct_operations": len(groups),
                    },
                )
            )

        findings.extend(self._read_repeat_note(observed))
        return findings

    def _judge(
        self,
        tool_name: str,
        fingerprint: str,
        group: list[CallView],
        exemptions: ExemptionSet,
    ) -> Finding | None:
        facts = self._facts(group[0])
        replay = [c for c in group if c.replay_reason == SNAPSHOT_REPLAY]
        agent = [c for c in group if c.replay_reason is None]
        # 其他重放原因（未来的 resume/retry 之类）。目前只记录不判定——
        # 出现未知重放标记时应当能被看见，而不是被静默归进 Agent 自发重复。
        other_replay = [c for c in group if c.replay_reason not in (None, SNAPSHOT_REPLAY)]

        detail: dict[str, Any] = {
            "tool_name": tool_name,
            "fingerprint": fingerprint[:16],
            "total_calls": len(group),
            "agent_initiated": len(agent),
            "snapshot_replays": len(replay),
            "other_replay_reasons": sorted(
                {c.replay_reason for c in other_replay}, key=str
            ),
            "seqs": [c.seq for c in group],
            "call_ids": [c.id for c in group],
            "idempotency_keys": sorted({c.idempotency_key for c in group}, key=str),
            "enforces_idempotency": facts.enforces_idempotency,
        }

        # Agent 自发重复不足两次，但确实发生了重复 —— 那就是恢复机制制造的。
        # 顺序很重要：先判 Agent 自发重复，再判恢复重放。反过来的话，
        # 一次"Agent 重复 + 恢复重放"混合的组会被归因成纯恢复问题。
        if len(agent) < 2 and replay:
            return Finding(
                detector=self.name,
                verdict=DetectorVerdict.FAIL,
                summary=(
                    f"{tool_name} 被重复执行 {len(group)} 次，"
                    f"其中 {len(replay)} 次由快照恢复重放导致（非 Agent 行为）。"
                    f"重复扣款是真实的，但要去修恢复机制，不要改 Agent。"
                ),
                confidence=Confidence.HIGH,
                detail=detail,
            )

        if len(agent) < 2:
            return None

        # 这里开始是 Agent 自发重复——按保护情况分层判定
        if not facts.enforces_idempotency:
            finding = Finding(
                detector=self.name,
                verdict=DetectorVerdict.FAIL,
                summary=(
                    f"{tool_name} 是没有幂等保护的写操作，却被执行了 {len(agent)} 次"
                    f"（同一次调用）。这是重复扣款。"
                ),
                confidence=Confidence.HIGH,
                detail=detail,
            )
        else:
            keys = {c.idempotency_key for c in agent}
            if keys == {None}:
                finding = Finding(
                    detector=self.name,
                    verdict=DetectorVerdict.FAIL,
                    summary=(
                        f"{tool_name} 声明了幂等保护，但重复调用未携带幂等键，"
                        f"保护未生效，实际重复执行 {len(agent)} 次。"
                    ),
                    confidence=Confidence.HIGH,
                    detail={**detail, "reason_code": "missing_key"},
                )
            elif len(keys) > 1:
                finding = Finding(
                    detector=self.name,
                    verdict=DetectorVerdict.FAIL,
                    summary=(
                        f"{tool_name} 的重复调用每次用了不同的幂等键"
                        f"（{len(keys)} 个），保护被绕过，实际重复执行 {len(agent)} 次。"
                        f"工具声明保护不等于保护生效。"
                    ),
                    confidence=Confidence.HIGH,
                    detail={**detail, "reason_code": "key_rotated"},
                )
            else:
                # 同一个幂等键重复调用：保护生效了，没有重复扣款。
                # 报 warning 而不是 pass：Agent 确实做了无谓的重复调用，
                # 值得看见，但不该按缺陷对待。
                finding = Finding(
                    detector=self.name,
                    verdict=DetectorVerdict.WARNING,
                    summary=(
                        f"{tool_name} 被重复调用 {len(agent)} 次，但复用了同一幂等键，"
                        f"保护生效，未产生重复扣款。"
                    ),
                    confidence=Confidence.MEDIUM,
                    detail={**detail, "reason_code": "protected"},
                )

        exemption = exemptions.find(
            self.name,
            tool_name,
            # 两个来源任一说是写操作就按写处理。两处声明本应一致，不一致时
            # 往保守的方向取——否则一个不一致的声明就能把写操作静音掉。
            is_write=facts.is_write or group[0].is_write,
            tool_has_idempotency=facts.enforces_idempotency,
        )
        if exemption is None:
            return finding

        # 豁免不是静默的：结论降级但**必须留痕**，报告里要能看到"因豁免未告警"。
        return Finding(
            detector=finding.detector,
            verdict=DetectorVerdict.PASS,
            summary=f"{finding.summary}（已被豁免规则 {exemption.label} 静音）",
            confidence=Confidence.MEDIUM,
            detail={**finding.detail, "would_have_been": finding.verdict.value},
            exempted_by=exemption.label,
        )

    def _read_repeat_note(self, calls: list[CallView]) -> list[Finding]:
        """读操作的重复调用不告警，但要出现在报告里——它计入成本。"""
        reads = [c for c in calls if not c.is_write]
        groups: dict[tuple[str, str], list[CallView]] = defaultdict(list)
        for call in reads:
            groups[(call.tool_name, call.fingerprint)].append(call)

        repeated = {t: len(g) for (t, _), g in groups.items() if len(g) >= 3}
        if not repeated:
            return []
        return [
            Finding(
                detector=f"{self.name}.read_repeats",
                verdict=DetectorVerdict.PASS,
                summary=f"只读工具重复调用（不告警，但计入成本）：{repeated}",
                confidence=Confidence.HIGH,
                detail={"repeated_reads": repeated},
            )
        ]
