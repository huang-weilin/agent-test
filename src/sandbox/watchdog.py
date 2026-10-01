"""熔断看门狗：检测到失控就地终止，而不是等运行结束再报告。

**为什么必须在这里做。** 检测器如果只在运行结束后跑，那它产出的是一份尸检
报告——"这次循环烧了 400 万 token"。要真正省钱，判定必须发生在运行**当中**。

**为什么"带外"是关键词。** 网关拒绝后续请求，只能让 Agent 的**下一次**调用
失败；如果它此刻正在一次长达几十秒的推理里，或者干脆卡在一个不经过网关的
循环里，钱还在烧。真正止损必须能终止 Agent 进程——那才是带外 kill。
所以 `RunRegistry.kill()` 会同时做两件事：置位熔断标记（带内表现）+
通知带外处理器（真正杀进程）。

**代价要讲清楚，而且要**量**出来讲。** 每次工具调用后重扫一遍该运行的整条轨迹，
所以代价随调用次数超线性地涨。这句话原先写的是"几十次调用完全没问题"——那是
**没量过就写下的**。2026-10-01 量到的（`../_probes/measure_watchdog_scan.py`、
`../_probes/measure_watchdog_breakdown.py`，本机 SQLite）：

| 调用次数 | 读整条轨迹 | 跑一遍检测器 | 落一条结论 | 整个运行累计 |
|---|---|---|---|---|
| 40 | 3.7 ms | 5.8 ms | 10.5 ms | 1.1 s |
| 80 | 4.8 ms | 33.0 ms | 10.7 ms | 3.1 s |
| 160 | 7.8 ms | 163.9 ms | 11.5 ms | 12.4 s |
| 320 | 14.1 ms | 649.7 ms | 10.1 ms | **80.8 s** |

读是线性的，**检测器每翻一倍调用涨 4~5 倍**（n=320 时 51,040 次参数相似度比较
= C(320,2)，每次约 34 µs），而**落一条结论是常数 10 ms**——一次扫描落两条，
于是它在小运行里就是全部开销。

**这笔时间记在被测会话的墙钟上。** 看门狗是在出口代理的请求处理里 `await` 的
（`proxy/app.py`），而 Agent 正阻塞在这个请求上；单会话超时（`runner`）量的是
Agent 进程活了多久，两者之间没有任何东西把我们的开销摘出来。更糟的是它在轨迹里
**看不见**：`tool_call.latency_ms` 在 `enforce` 之前就量完了，所以一次因为看门狗
太慢而超时的运行，报告里长得和"Agent 自己慢"一模一样。这一片把两处**精确**的
浪费去掉了（见下），但"谁的错分开写"这条还没接上——**它是记在案的债**。

这一片改掉的两处，都不改变任何判定：

1. `_cluster` 里 `normalize_args` 挪出双重循环。它只依赖单条调用，原先每比较
   一对就重算两边——n=320 时是 102,080 次，实际只需要 320 次。
2. `inspect_run` 同一个 `(检测器, 四态)` 只落一行。n−1 次扫描给出的是同一个
   结论，原先照单全收——320 次调用会在库里留下 640 行副本。

**为什么"改成只扫滑动窗口"不是一句能顺手写下的建议。** 那是原先挂在这段末尾的
补救办法，看着显然，量一下就知道它会**静默地少报**（`../_probes/check_sliding_window.py`）：

  * 跨会话那一层要的是"同一个操作出现在两个会话里"。只看最后 3 次调用时，
    轨迹里只剩一个会话——量到的结论从 `loop.cross_session/warning` 变成
    `inconclusive`。它没报错，只是不判了。
  * 近似层那个"翻了很多页还没停"的 warning 判的是**整段有多长**
    （`monotonic_warn_after = 8`）。12 页的遍历报 warning；只看最后 7 次调用时
    同一段轨迹报的是 **pass**。整段没变，窗口大小把它判没了。

两处都不是实现细节，是"这个检测器看的是什么"。要动它，得先回答"窗口里那份
结论还算不算数"——那是检测器行为变更，得单独做、单独测。所以这里只写下欠账
的数目，不顺手改。
"""

from __future__ import annotations

import structlog

from sandbox.db.models import DetectorVerdict
from sandbox.detectors.base import ExemptionSet, Finding
from sandbox.detectors.loop import LoopDetector
from sandbox.runtime import ActiveRun, runs
from sandbox.trace.reader import load_call_views
from sandbox.trace.recorder import record_finding

log = structlog.get_logger(__name__)

# 这些检测器的 FAIL 才触发熔断。幂等检测刻意**不在**这个名单里：
# 重复扣款是要报给人看的缺陷，但它不该成为终止运行的理由——运行继续下去，
# 后面的检测器（记忆漂移、跨会话）才有数据可看。
# 循环和预算失控不一样：它们会一直烧钱，多跑一秒都是净损失。
KILL_ON_DETECTORS: frozenset[str] = frozenset({"loop.strict", "loop.ping_pong"})

def detector_for(active: ActiveRun | None) -> LoopDetector:
    """按这次运行声明的字段权重造一个检测器。

    **不能用模块级的单例。** 权重来自场景，而一个进程里会跑好几个场景：单例的
    话第二个场景要么沿用第一个场景的权重，要么得有人去改它——而"改它"在多进程
    与并发下就是串味，症状是同一段轨迹在不同运行里被判成不同的东西。现造一个
    的代价只是几个字典。

    运行不在（比如进程重启后扫一条旧轨迹）时用空表。这与"这份场景没声明权重"
    是同一条路：检测器照旧按参数形状猜，**这里不替它猜**。
    """
    return LoopDetector(key_weights=dict(active.loop_field_weights) if active else {})


async def inspect_run(
    run_id: str, detector: LoopDetector | None = None
) -> list[Finding]:
    """扫一遍该运行的轨迹并落库结论。返回本轮产生的 findings。

    `detector` 是给测试用的注入点（验"某种阈值下会怎样"）；生产路径不传，
    由 `detector_for(active)` 从这次运行的声明里现造一个。
    """
    calls = await load_call_views(run_id)
    if not calls:
        return []

    active = runs.get(run_id)
    # **豁免要跟着进来看门狗这一侧。** 检测器本来就会把命中的豁免降级成 pass
    # 并留下 `exempted_by`，所以这里只要把场景声明的那一套传进去，"一条被豁免的
    # 循环不算缺陷"这句话就同时在熔断和事后报告两边成立。传空集合的话，两边会
    # 给出相反结论：进程被杀，报告却说"已豁免、通过"。
    #
    # 字段权重同理、而且是同一个理由的另一面：**熔断与事后报告必须用同一套判卷
    # 标准**。报告那边拿的是场景声明的那张权重表（runner 直接读 scenario），
    # 看门狗这边只能从注册表里读——它的声明是过线来的，见 gateway/app.py。
    findings = (detector or detector_for(active)).run(
        calls, active.exemptions if active else ExemptionSet()
    )

    if active is None:
        return findings
    ctx = active.ctx

    for finding in findings:
        # **同一个结论只落一行。** 这个函数每次工具调用都要跑一遍，而它在
        # n−1 次扫描里给出的是同一个结论——照单全收的话，一次 320 次调用的运行
        # 会往库里写 640 行副本。见 `ActiveRun.recorded_conclusions`。
        key = (finding.detector, finding.verdict.value)
        if key in active.recorded_conclusions:
            continue
        active.recorded_conclusions.add(key)
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


__all__ = ["detector_for", "enforce", "inspect_run", "summarize_verdicts"]
