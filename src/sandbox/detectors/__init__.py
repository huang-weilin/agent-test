"""检测器的统一装配点。

**这个模块存在的唯一理由：语料和真实运行必须跑同一套检测器。**

`tests/corpus/` 的精度数字只有在"语料跑的就是线上跑的那一套"时才有意义。
两处各自 `[]` 拼一遍的话，总有一天会有一边多一个检测器或改了顺序，而两边的
测试都还是绿的——语料算出来的精确率于是变成一句关于另一套东西的话。
"""

from __future__ import annotations

from collections.abc import Sequence

from sandbox.detectors.autonomy import AutonomyDetector
from sandbox.detectors.base import (
    EXEMPTION_AWARE,
    NOT_EXEMPTIBLE,
    CallView,
    ExemptionSet,
    Finding,
)
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
        "loop.unknown_replay",
        "idempotency",
        "idempotency.read_repeats",
        "autonomy",
        "autonomy.unauthorized",
        "memory",
        "memory.drift",
        "memory.poisoning",
    }
)


#: 边界上那四个名字。**它们不在 `KNOWN_DETECTORS` 里，因为不是一个类产出的**：
#: `tool_scope` / `tool_schema` / `egress` 由代理当场判（`proxy/app.py`），
#: `token_budget` 由网关在熔断那一刻写（`gateway/app.py`）。它们写的是字面量，
#: 源码里没有一份可 import 的集合——所以这里是**唯一一处抄写**，代价由闸门兜着：
#: `tests/test_explain_coverage.py` 把「这个集合 ∪ `KNOWN_DETECTORS`」和
#: `explain.DETECTOR_HELP` 比对，而后者那边是**从源码现算**出来的
#: （AST 扫 `detector=` 与 `{"detector": ...}`）。两边不一致就红。
#:
#: **别和 `trace/reader.py` 的 `BOUNDARY_DETECTORS` 混。** 那是个同名不同义的集合
#: （"能从轨迹里读回来的那两条"），只有两个成员。名字像不代表管的是同一件事。
BOUNDARY_DETECTOR_NAMES: frozenset[str] = frozenset(
    {"tool_scope", "tool_schema", "egress", "token_budget"}
)

#: 场景 `expect[].detector` 上允许写的名字。
#:
#: 原先这里没有任何校验：`Expectation.detector` 是个裸 `str`，比对时拿它做裸字符串
#: 相等（`scenario/expect.py`），于是**写错一个字母，那条预期静默地永远配不上**——
#: 场景照样加载、照样跑、照样出报告，只是那一条从来不参与判定。用户以为自己
#: 声明过的判卷标准其实不存在。
#:
#: 豁免那边**是**校验的（`loader.py` 逐条查 `KNOWN_DETECTORS`），所以这是同一种
#: 防护只做了一半。补上另一半的时候**不能直接照抄 `KNOWN_DETECTORS`**：实测 29 份
#: 场景里有 6 处合法声明写的是 `tool_schema`，而它不在那 15 个名字里——只按
#: `KNOWN_DETECTORS` 判会当场冤枉 6 处。
SCENARIO_DETECTOR_NAMES: frozenset[str] = KNOWN_DETECTORS | BOUNDARY_DETECTOR_NAMES

#: 用户在报告页上，对一条结论能做什么。三选一，**不是一个"能不能豁免"的布尔值**：
#: 这三种情形对用户的下一步动作完全不同，合成一个真假值就必然要对其中两种撒谎。
EXEMPT_CAN = "can"
"""能豁免：这条路上真的会去查豁免规则。"""

EXEMPT_MEMORY = "memory"
"""记忆族：有例外机制，但不是豁免——是场景里逐条声明的 `must_hold` / `forbidden`。"""

EXEMPT_NONE = "none"
"""没有出口：那条路上压根没有查豁免那一步。今天就是办不到，如实说。"""


def exemption_kind(detector: str) -> str:
    """这个检测器上的结论，用户有没有办法说「这条我认可是误报」。

    **必须三种分开**。界面上写成"每条结论旁边都有个按钮"的话，用户点到后两种
    上就会撞见一个装上了但不生效的开关——豁免机制自己最不能出的就是这个毛病
    （它整个存在的意义就是"让结论消失"，一个静音不了的静音开关比没有更坏）。

    实测：19 个检测器名里能豁免的只有 6 个，记忆族 3 个，剩下 10 个没有出口。

    名字不认识就**报错**，不返回 `EXEMPT_NONE`——一个拼错的检测器名会安安静静地
    被归进"没有出口"，而它看起来和"这条确实办不到"一模一样。
    """
    if detector in EXEMPTION_AWARE:
        return EXEMPT_CAN
    if detector in NOT_EXEMPTIBLE:
        return EXEMPT_MEMORY
    if detector in SCENARIO_DETECTOR_NAMES:
        return EXEMPT_NONE
    raise ValueError(
        f"不认识的检测器名 {detector!r}。认得的名字在 `SCENARIO_DETECTOR_NAMES` 里。"
        f"（这里不接受没见过的名字：把它归进「没有出口」的话，"
        f"一个拼错的名字会看起来像一条真的办不到的结论。）"
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
    loop_field_weights: dict[str, float] | None = None,
) -> list[Finding]:
    """跑全套检测器。顺序固定：输出稳定，比对预期时不会因字典序漂移而时红时绿。

    声明从参数进来，观测从 `calls` 进来——这是全项目统一的套路：能被 Agent
    影响的字段不参与判定，判定依据全部来自场景/语料的显式声明。

    `loop_field_weights` 也是声明的一种，所以它也走参数：它说的是"这个工具的
    参数里，哪个字段是它的身份、哪个只是光标"。**少了它，近似层判出来的东西
    取决于参数形状而不是行为**——实测翻页到第 10 页会被切成 5+3 两个簇，
    于是"翻了很多页也不停"这个结论根本到不了告警阈值（见 `LoopDetector.key_weights`）。
    真实运行与看门狗两条路径都必须拿到**同一张表**，否则熔断与事后报告会各说各话。

    `memories` 是**第二条观测通道**：记忆内容不是边界上的调用，`CallView` 里没有
    它要的东西（见 detectors/memory.py 的说明）。两个通道都从参数进来，位置并列，
    这样"哪些检测器吃哪条通道"在调用处一眼可见。`MemoryDetector` 也**不接豁免**
    ——记忆族的例外机制是逐条声明的判据本身，见 `base.NOT_EXEMPTIBLE`。
    """
    ex = exemptions if exemptions is not None else ExemptionSet()
    findings: list[Finding] = []
    findings.extend(IdempotencyDetector(tool_facts or {}).run(calls, ex))
    findings.extend(LoopDetector(key_weights=loop_field_weights or {}).run(calls, ex))
    findings.extend(
        AutonomyDetector(
            allowlist=unattended_allowlist,
            autonomous_ordinals=autonomous_ordinals,
        ).run(calls, ex)
    )
    findings.extend(MemoryDetector(memory_rules).inspect(memories))
    return findings
