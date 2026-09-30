"""检测器的统一装配点。

**这个模块存在的唯一理由：语料和真实运行必须跑同一套检测器。**

`tests/corpus/` 的精度数字只有在"语料跑的就是线上跑的那一套"时才有意义。
两处各自 `[]` 拼一遍的话，总有一天会有一边多一个检测器或改了顺序，而两边的
测试都还是绿的——语料算出来的精确率于是变成一句关于另一套东西的话。
"""

from __future__ import annotations

from collections.abc import Sequence

from sandbox.detectors.autonomy import AutonomyDetector
from sandbox.detectors.base import CallView, ExemptionSet, Finding
from sandbox.detectors.idempotency import IdempotencyDetector, ToolFacts
from sandbox.detectors.loop import LoopDetector
from sandbox.detectors.memory import MemoryDetector, MemoryRules
from sandbox.memory.observe import MemoryObservation

#: 当前装进管线的检测器。语料加载时会拿它校验标注里写的检测器名，
#: 拼错了要当场失败——否则那条标注永远"未命中"，看起来像检测器漏报。
#:
#: 子名字（`loop.strict` 这种）也在这里，人工维护。漏登记的后果是语料加载失败，
#: 也就是往可见的一侧错。
KNOWN_DETECTORS: frozenset[str] = frozenset(
    {
        "loop",
        "loop.strict",
        "loop.approximate",
        "loop.ping_pong",
        "loop.cross_session",
        "loop.retry",
        "loop.snapshot_replay",
        "idempotency",
        "idempotency.read_repeats",
        "autonomy",
        "autonomy.unauthorized",
        "memory",
        "memory.drift",
        "memory.poisoning",
    }
)


def run_all(
    calls: list[CallView],
    *,
    tool_facts: dict[str, ToolFacts] | None = None,
    unattended_allowlist: frozenset[str] = frozenset(),
    autonomous_ordinals: frozenset[int] = frozenset(),
    exemptions: ExemptionSet | None = None,
    memories: Sequence[MemoryObservation] = (),
    memory_rules: MemoryRules | None = None,
) -> list[Finding]:
    """跑全套检测器。顺序固定：输出稳定，比对预期时不会因字典序漂移而时红时绿。

    声明从参数进来，观测从 `calls` 进来——这是全项目统一的套路：能被 Agent
    影响的字段不参与判定，判定依据全部来自场景/语料的显式声明。

    `memories` 是**第二条观测通道**：记忆内容不是边界上的调用，`CallView` 里没有
    它要的东西（见 detectors/memory.py 的说明）。两个通道都从参数进来，位置并列，
    这样"哪些检测器吃哪条通道"在调用处一眼可见。`MemoryDetector` 也**不接豁免**
    ——记忆族的例外机制是逐条声明的判据本身，见 `base.NOT_EXEMPTIBLE`。
    """
    ex = exemptions if exemptions is not None else ExemptionSet()
    findings: list[Finding] = []
    findings.extend(IdempotencyDetector(tool_facts or {}).run(calls, ex))
    findings.extend(LoopDetector().run(calls, ex))
    findings.extend(
        AutonomyDetector(
            allowlist=unattended_allowlist,
            autonomous_ordinals=autonomous_ordinals,
        ).run(calls, ex)
    )
    findings.extend(MemoryDetector(memory_rules).inspect(memories))
    return findings
