"""熔断看门狗：检测到失控就地终止，而不是等运行结束再报告。

**为什么必须在这里做。** 检测器如果只在运行结束后跑，那它产出的是一份尸检
报告——"这次循环烧了 400 万 token"。要真正省钱，判定必须发生在运行**当中**。

**为什么"带外"是关键词。** 网关拒绝后续请求，只能让 Agent 的**下一次**调用
失败；如果它此刻正在一次长达几十秒的推理里，或者干脆卡在一个不经过网关的
循环里，钱还在烧。真正止损必须能终止 Agent 进程——那才是带外 kill。
所以 `RunRegistry.kill()` 会同时做两件事：置位熔断标记（带内表现）+
通知带外处理器（真正杀进程）。

**代价要讲清楚。** 每次工具调用后重扫一遍该运行的整条轨迹，是 O(n²)。
walking skeleton 的量级（几十次调用）完全没问题；真实场景要改成只扫滑动窗口。
这个取舍是有意的：先证明"熔断能止损"这件事成立，再优化它的常数。
"""

from __future__ import annotations

import structlog

from sandbox.db.models import DetectorVerdict
from sandbox.detectors.base import ExemptionSet, Finding
from sandbox.detectors.loop import LoopDetector
from sandbox.runtime import runs
from sandbox.trace.reader import load_call_views
from sandbox.trace.recorder import record_finding

log = structlog.get_logger(__name__)

# 这些检测器的 FAIL 才触发熔断。幂等检测刻意**不在**这个名单里：
# 重复扣款是要报给人看的缺陷，但它不该成为终止运行的理由——运行继续下去，
# 后面的检测器（记忆漂移、跨会话）才有数据可看。
# 循环和预算失控不一样：它们会一直烧钱，多跑一秒都是净损失。
KILL_ON_DETECTORS: frozenset[str] = frozenset({"loop.strict", "loop.ping_pong"})

_default_detector = LoopDetector()


async def inspect_run(
    run_id: str, detector: LoopDetector | None = None
) -> list[Finding]:
    """扫一遍该运行的轨迹并落库结论。返回本轮产生的 findings。"""
    calls = await load_call_views(run_id)
    if not calls:
        return []

    active = runs.get(run_id)
    # **豁免要跟着进来看门狗这一侧。** 检测器本来就会把命中的豁免降级成 pass
    # 并留下 `exempted_by`，所以这里只要把场景声明的那一套传进去，"一条被豁免的
    # 循环不算缺陷"这句话就同时在熔断和事后报告两边成立。传空集合的话，两边会
    # 给出相反结论：进程被杀，报告却说"已豁免、通过"。
    findings = (detector or _default_detector).run(
        calls, active.exemptions if active else ExemptionSet()
    )

    ctx = active.ctx if active else None
    if ctx is None:
        return findings

    for finding in findings:
        await record_finding(
            ctx,
            detector=finding.detector,
            verdict=finding.verdict,
            summary=finding.summary,
            confidence=finding.confidence,
            detail=finding.detail,
            exempted_by=finding.exempted_by,
        )
    return findings


async def enforce(run_id: str, detector: LoopDetector | None = None) -> str | None:
    """扫一遍，命中就熔断。返回熔断理由，未熔断返回 None。"""
    active = runs.get(run_id)
    if active is None or active.killed:
        return None

    findings = await inspect_run(run_id, detector)
    for finding in findings:
        if finding.detector not in KILL_ON_DETECTORS:
            continue
        if finding.verdict != DetectorVerdict.FAIL:
            if finding.exempted_by is not None:
                # 本该杀、被豁免拦下了。**这不是静音**——结论里带着
                # `exempted_by`，报告会照原样印出理由。但日志里要留一笔：
                # 一个确实在打转、而我们放它继续跑的运行，事后该能查得到。
                #
                # 每次工具调用都会重扫，所以同一条只报一次：反复刷同一行会把
                # "查得到"变成"翻不到"。
                key = f"{finding.detector}/{finding.exempted_by}"
                if active is not None and key not in active.warned_exemptions:
                    active.warned_exemptions.add(key)
                    log.warning(
                        "watchdog.exempted",
                        run_id=run_id,
                        detector=finding.detector,
                        exempted_by=finding.exempted_by,
                    )
            continue

        reason = f"{finding.detector}: {finding.summary}"
        # 先落库再熔断。反过来的话，进程被杀掉的那一瞬间结论可能还没写进去，
        # 报告里就只剩"运行被杀死"，看不到为什么——这正是最难排查的一类问题。
        log.warning("watchdog.kill", run_id=run_id, detector=finding.detector)
        runs.kill(run_id, reason)
        return reason
    return None


def summarize_verdicts(findings: list[Finding]) -> dict[str, float]:
    """按四态统计。

    `inconclusive` 单独计数、**不并入 pass**，并把覆盖率与"已覆盖部分的失败率"
    并列给出。报告只显示一个通过率的话，"啥也没测"会被读成"全绿"——
    这个工具最严重的失效方式就是让用户拿着空报告上生产。
    """
    counts = {"pass": 0, "fail": 0, "warning": 0, "inconclusive": 0}
    for finding in findings:
        counts[finding.verdict.value] += 1

    total = sum(counts.values())
    covered = total - counts["inconclusive"]
    return {
        **counts,
        "total": float(total),
        "covered": float(covered),
        "coverage_ratio": round(covered / total, 3) if total else 0.0,
        "failure_rate_among_covered": (
            round(counts["fail"] / covered, 3) if covered else 0.0
        ),
    }


__all__ = ["enforce", "inspect_run", "summarize_verdicts"]
