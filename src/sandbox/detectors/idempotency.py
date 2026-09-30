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

**但今天没有生产路径会写下这个标记。** 边界上的每一次调用都是被测 Agent 的
进程发出来的，自报不算证据，所以"这一次是恢复机制干的"今天说不出口：一次由
恢复机制制造的重复扣款会被记成 Agent 的重复扣款（**钱确实多扣了**，只是该去
修哪儿这句话今天说不出来）。这一条和循环检测共用一份说明，写在
docs/detectors.md 第三节第 3 条；tests/test_recorder.py 里有一条绊线盯着
"还没有生产调用方"这件事，谁接上第一个它就会红。

**四、跨会话重复的归因看记忆视野，不看直觉。** 需求里的用户故事是"用户周一让
Agent 订票，周三再次要求时，Agent 没有识别出这是重复操作"——Agent 有长效记忆，
本应认出来。所以判据是"重复发生时那件事在不在它的记忆里"，而记忆视野是我们
拍快照时定位的，不是推测的。在记忆里 → Agent 的锅；不在 → 恢复/记忆传递的锅。
两种情况都报 fail，因为钱确实扣了两次；区别在 summary 里说该去修哪儿。
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
            if other_replay:
                # 未知重放标记：既不归进"Agent 自发重复"，也不能当没看见。
                # 上面那段说明写的是"应当能被看见"——而在此之前它只出现在一个
                # 永远不会被返回的 detail 里，等于写了个注释就当实现了。
                # 报 inconclusive 而不是 None：None 会让这一组落进"已检查 N 次写操作、
                # 未发现重复执行"那句汇总 PASS 里，把一个判不了的重复说成查过没毛病。
                return inconclusive(
                    self.name,
                    f"{tool_name} 有 {len(other_replay)} 次重复调用带着未知的重放标记"
                    f"（{sorted({c.replay_reason for c in other_replay}, key=str)}），"
                    f"分不清是 Agent 自发重复还是恢复机制导致，无法判定。",
                    detail=detail,
                )
            return None

        # 跨会话重复：同一操作出现在序列中的多个会话里。
        # 判定在 _judge_cross_session，它用的是记忆视野而不是重放标记。
        by_ordinal: dict[int, list[CallView]] = defaultdict(list)
        for call in agent:
            by_ordinal[call.session_ordinal].append(call)
        if len(by_ordinal) >= 2:
            return self._judge_cross_session(tool_name, fingerprint, by_ordinal, detail, exemptions)

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
            summary=f"{finding.summary}{exemption.note()}",
            confidence=Confidence.MEDIUM,
            detail={**finding.detail, "would_have_been": finding.verdict.value},
            exempted_by=exemption.label,
        )

    def _judge_cross_session(
        self,
        tool_name: str,
        fingerprint: str,
        by_ordinal: dict[int, list[CallView]],
        detail: dict[str, Any],
        exemptions: ExemptionSet,
    ) -> Finding:
        """跨会话重复执行：谁该负责，由**记忆视野**决定。

        需求里的用户故事是"用户周一让 Agent 订票，周三再次要求时，Agent 没有
        识别出这是重复操作"。要点是：Agent 有长效记忆，本应认出来。

        于是归因就看一条——重复发生时，那件事在不在它的记忆里：

          * `首次发生的序号 <= 本次会话的记忆视野` → 在记忆里，Agent 本可以
            认出来却没认出来。**Agent 的锅。**
          * 否则 → 不在记忆里（我们回滚过，或记忆根本没带上）。它不可能知道。
            **恢复机制或记忆配置的锅，不是 Agent 的。**

        这里的判断不需要猜：视野是 start_session 依据快照位置写死的，而快照
        是我们自己拍的。这正是把记忆做成操纵面换来的东西。

        两种情况都报 FAIL，因为**钱是真的扣了两次**。区别在 summary 里说清楚
        该去修哪儿——否则用户会去改一个没问题的 Prompt。
        """
        ordinals = sorted(by_ordinal)
        first, last = ordinals[0], ordinals[-1]
        first_calls = by_ordinal[first]
        last_calls = by_ordinal[last]
        facts = self._facts(first_calls[0])
        # 视野是会话属性，同一会话内每条调用都一样，取第一条即可。
        horizon = last_calls[0].memory_horizon_ordinal
        should_have_known = first <= horizon

        session_detail = {
            **detail,
            "sessions": {str(o): [c.seq for c in cs] for o, cs in sorted(by_ordinal.items())},
            "first_ordinal": first,
            "last_ordinal": last,
            "memory_horizon_ordinal": horizon,
            "missed_ordinals": [
                o for o in range(horizon + 1, last) if o in by_ordinal
            ],
        }

        if should_have_known:
            finding = Finding(
                detector=self.name,
                verdict=DetectorVerdict.FAIL,
                summary=(
                    f"{tool_name} 的同一操作在会话 #{first} 与 #{last} 各执行了一次"
                    f"（记忆视野覆盖到 #{horizon}，Agent 本可以认出来）。"
                    f"这是跨会话重复扣款——它没有把'我做过了'这件事用起来。"
                ),
                confidence=Confidence.HIGH,
                detail={**session_detail, "reason_code": "cross_session_duplicate"},
            )
        else:
            forgotten = sorted(
                {str(o) for o in ordinals if o > horizon and o < last}
            )
            finding = Finding(
                detector=self.name,
                verdict=DetectorVerdict.FAIL,
                summary=(
                    f"{tool_name} 的同一操作在会话 #{first} 与 #{last} 各执行了一次，"
                    f"但会话 #{last} 的记忆只覆盖到 #{horizon}"
                    + (f"（#{'、#'.join(forgotten)} 那段被丢掉了）" if forgotten else "")
                    + "。重复扣款是真实的，但 Agent 不可能知道——"
                    "要去修记忆的恢复/传递，不要改 Agent。"
                ),
                confidence=Confidence.HIGH,
                detail={**session_detail, "reason_code": "cross_session_memory_loss"},
            )

        exemption = exemptions.find(
            self.name,
            tool_name,
            is_write=facts.is_write or first_calls[0].is_write,
            tool_has_idempotency=facts.enforces_idempotency,
        )
        if exemption is None:
            return finding
        return Finding(
            detector=finding.detector,
            verdict=DetectorVerdict.PASS,
            summary=f"{finding.summary}{exemption.note()}",
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
