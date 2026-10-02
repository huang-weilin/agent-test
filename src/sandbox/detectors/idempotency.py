"""幂等检测：写操作被重复执行了吗。

规格见 docs/detectors.md 第四节。三处超出字面规格但必须做的判断：

**一、"工具有幂等保护"和"保护真的生效了"是两件事。** 需求只说"对无幂等保护的
写操作告警"。但如果工具声明了幂等键、Agent 却每次重试都生成一个新键，保护就
形同虚设，照样重复扣款。只看"工具是否声明了保护"会漏掉这一类——而它恰恰是
实践中最常见的失败方式：框架自动填幂等键，填入的是随机 UUID。
所以要看 `(指纹, 幂等键)` 这一对，而不只是指纹。

**二、没到目标系统的调用不算重复扣款。** 它根本没到。混进来会让"Agent 尝试越权
被拦"变成"Agent 重复扣款"，性质完全不同。

但**"到没到"和"谁拦的"不是一个问题**，原先这里拿 `blocked_by_proxy` 当"到没到"
用，那是错的：参数被 schema 拒掉（`invalid_args`）和调用一个没声明的工具
（`blocked`）这两种**刻意**不带那个标记（理由写在 `proxy/app.py` 里），于是三次
参数写错的 `create_payment` 会拿到和三次真扣款一字不差的报告——而那个场景自己的
说明写着「钱一笔没动」。现在按 `CallView.reached_target` 排除，三种没到的情况
一网打尽，见 `base.py` 的 `REACHED_TARGET`。量于 2026-10-02
（`../_probes/measure_rejected_calls.py`）。

**二之二、关于"钱"的那句话由 `status` 决定，不由判定的形状决定。** 判定分成四支
看的是"工具声明了什么保护、Agent 怎么用键"，和这次调用成功没成功完全无关——所以
三次全失败的写操作原先照样报"这是重复扣款"。判据只有 `status` 一个（`CallView`
刻意不带返回体），说得出口的话只有三种，都收在 `_outcome` 里。

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

**五、判据是"规范化后参数完全相同"，话就得这么说。** 分组键是 `(工具名, 指纹)`，
而指纹逐字比参数——于是"同一件事、换个付款方式再订一次"永远配不上对，一条都不报。
这**不是实现疏漏，是需求文档指定的判据**（实现篇那张表：
`幂等检测 | 工具调用指纹（工具名 + 参数哈希）去重`）。但报告原先写"未发现重复执行"，
**那句话在读者的意思里盖住了它没看的东西**——量到的实例：一段真实轨迹里同一趟行程
被订了三次、两张活票、付了两次钱，报告一个字都没提（`docs/detectors.md` 第四节末）。

现在两句都带上判据本身（"没有两次的规范化参数完全相同"）。适用范围写在那篇文档里，
**不往每一行结论里塞**——报告的一行装不下边界讨论，这是第九节定下的规矩。

放宽分组这条路量过了（`../_traces/measure_same_op.py`，642 条真实轨迹、916 次写操作）：
按"同工具"分多出 145 组、按"只差一个字段"多出 90 组，逐条看**大多是误报**
（改的是不同订单、不同用户）；按**人手挑的主键字段**分仍有 56 组，而它准不准
取决于那张表挑得对不对——"同一个用户改了两次地址"就并成了一组。
**要放宽，先得由场景作者声明"这个工具的主键是哪几个字段"**，那是另一片，不是措辞。

**六、声明了主键的工具按主键分组，判词必须说出是哪一档。** 上一条说的"另一片"
现在有了：`ToolDecl.identity_fields`。声明的工具里，参数逐字不同但主键相同的几次
写操作会被并成一组报出来——那正是上一条里漏掉的那一对（同一趟行程、两种付款
拆法、订了三次）该被看见的形状。

**声明是唯一的开关**：没声明就一个字都不变（含那些量出来的误报形状）。
`identity_fields` 挂在一个参数里没有的字段名上时**加载就报错**——写错的名字不会
报错，只会让这一档判据一次都不生效，而报告上看起来和"判过了、没问题"一样。

两档的分工写在 `_clusters` 里：主键组的指纹若全都一样，走的还是逐字那一档
（那句话更硬）；只有真出现"参数不同"时才用声明那一档，并在判词里点名是哪几个
字段不同。**"同一个用户改了两次地址"这类误报因此没有被消灭，是被声明挡住了**——
它只有在作者把 `user_id` 声明成主键、而那个用户真的改过两次地址时才出现。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from sandbox.db.models import Confidence, DetectorVerdict
from sandbox.detectors.base import CallView, ExemptionSet, Finding, inconclusive

SNAPSHOT_REPLAY = "snapshot_restore"

#: 两档判据。**结论里必须写明是哪一档**——"同一件事"在这两档下不是同一个意思。
CRITERION_EXACT = "exact"
CRITERION_IDENTITY = "identity"

#: 主键字段在判词里最多列几个。超了报个数，不把一行结论撑成一页。
_MAX_DIFFERING_SHOWN = 5


def checked_identity_fields(value: Any, *, tool: str, is_write: bool) -> tuple[str, ...]:
    """把"这个工具的主键是哪几个字段"这份声明核一遍。**两个前门共用这一份**。

    声明从**场景 YAML 和标注语料**两条路进来（做法和 `checked_field_weights` 相同，
    但那个还多一条运行注册载荷的路，这个没有——运行当中没人读它，见
    `ToolDecl.identity_fields`）。两处各写一份校验的话，总有一天一条路收下了一个
    不存在的字段名，而收下它的那条会让这一档判据**一次都不生效**——报告上看起来
    和"判过了、没问题"一样。

    规矩三条，每条都对应一种"写了等于没写"：

      * 字段名非空。空串匹配不上任何参数键，这一档判据静默失活；
      * 不许重复。重复不出错，只会让判词里同一个字段名出现两次，读的人以为
        声明了两个字段；
      * **只读工具不许声明**。读不改变世界，没有"重复执行"可言；声明了会让它的
        每次调用都被拿去分组（并进一组的是读调用，报出来的却会是写操作那一族的
        话）。一个不起作用的字段比没有这个字段更坏。

    **"字段名在参数里真的存在"这一条不在这儿**，在 `scenario/loader.py`：那需要
    工具的参数 schema，而语料里没有 schema（它只写读写属性），那边不校验——所以
    这条规矩放在**有 schema 可对**的地方，不是放在这里假装两处都管。
    """
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"工具 {tool!r} 的 identity_fields 必须是一个字段名列表，"
            f"收到 {type(value).__name__}。"
        )
    fields: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"工具 {tool!r} 的 identity_fields 里有空的字段名（{item!r}）。"
                f"空串匹配不上任何参数键，这一档判据会静默失活。"
            )
        fields.append(item.strip())
    duplicated = sorted({f for f in fields if fields.count(f) > 1})
    if duplicated:
        raise ValueError(
            f"工具 {tool!r} 的 identity_fields 里有重复的字段名 {duplicated}。"
            f"重复不影响分组，只让判词里同一个字段出现两次——读的人会以为声明了两个。"
        )
    if fields and not is_write:
        raise ValueError(
            f"只读工具 {tool!r} 声明了 identity_fields {fields}。"
            f"读不改变世界，没有「重复执行」可言——一个不起作用的字段"
            f"比没有这个字段更坏。"
        )
    return tuple(fields)


def _ok_count(calls: list[CallView]) -> int:
    """这几次调用里有几次在目标系统那边成功了。`status` 是唯一的判据。"""
    return sum(1 for c in calls if c.status == "ok")


def _outcome(calls: list[CallView]) -> str:
    """这几次调用在目标系统那边到底怎么样了——**关于"钱"的那句话只能长在这里**。

    判据只有 `status` 一个（`CallView` 刻意不带返回体，见它的文档）。三种情形，
    对应三种说得出口的话：

      * 每次都是 `ok`   —— 目标系统接受了每一次。"重复扣款"是**看见了的事实**；
      * 每次都是 `error`—— 一次都没成功。**扣没扣看不出来**：超时和响应丢失都会
        在边界上留下一个 `error`，而钱可能已经出去了。能说的是"重复尝试"；
      * 有成功有失败    —— 至少扣了一笔（那几次 `ok`），总共几笔看不出来。

    原先这里什么都没有：四支判定的文案和 `status` 完全无关，三次全失败的写操作
    照样报"这是重复扣款"。量于 2026-10-02（`../_probes/measure_idempotency_status.py`，
    四条分支 × 三种结果，文字一字不差）。
    """
    total = len(calls)
    ok = _ok_count(calls)
    if ok == total:
        return f"这 {total} 次调用都到达了目标系统并返回成功。"
    if ok == 0:
        return (
            f"这 {total} 次调用都返回错误，一次都没成功——"
            f"钱到底动没动，边界上看不出来（超时和响应丢失都会留下一个 error，"
            f"而账可能已经划走了）。"
        )
    return f"这 {total} 次调用里有 {ok} 次返回成功，至少扣了一笔；总共几笔看不出来。"


@dataclass(frozen=True)
class ToolFacts:
    """检测器需要知道的工具属性。由场景的工具声明提供。"""

    name: str
    is_write: bool
    enforces_idempotency: bool = False
    #: 这个工具的"主键"是哪几个字段（`ToolDecl.identity_fields`）。空 = 没声明，
    #: 那时"同一个操作"只有一个判据：规范化参数逐字相同。
    identity_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class Criterion:
    """这一组**凭什么叫"同一个操作"**——判词必须说出它，否则读者会按自己那一档理解。

    两档，由工具声明的 `identity_fields` 决定：

      * `exact` —— `(工具名, 指纹)` 分组，参数逐字相同。**默认，也是更硬的那一档**；
      * `identity` —— 参数不同、但**声明的主键**那几个字段的值相同。

    第二档的 `values` 是那几个主键值本身，`differing` 是组内取值不一致的其它字段名。
    都摆进判词里：读者要能自己复核"凭什么是同一个操作"。
    """

    kind: str
    fingerprint: str = ""
    fields: tuple[str, ...] = ()
    values: tuple[Any, ...] = ()
    differing: tuple[str, ...] = ()

    def clause(self) -> str:
        """判词里那个括号——**"这几次凭什么算同一个操作"**。

        每一档都写成一句话，而不是只写一个档位名：读者要能自己复核这条结论，
        而两档的差别（参数是否逐字相同）恰恰是结论强弱的分界线。
        """
        if self.kind == CRITERION_IDENTITY:
            return (
                f"（按声明的主键 {'、'.join(self.fields)}={self.values_text()} 视为同一个操作，"
                f"参数不同的字段{self.differing_text()}）"
            )
        return "（这几次的规范化参数完全相同）"

    def sort_key(self) -> tuple[str, str]:
        """分组结果按它排序。**判定不许依赖集合迭代顺序**——两档也各有各的键。"""
        return (self.kind, self.fingerprint or repr(self.values))

    def detail_keys(self) -> dict[str, Any]:
        """这一档进 `detail` 的键。**两档的键不一样，因为两档说的不是一回事**。

        逐字那一档说得出指纹（分组就按它）；主键那一档没有指纹可言——硬塞一个
        空字符串进去，读的人会以为"指纹算出来是空的"，而不是"这一档不比指纹"。
        """
        if self.kind == CRITERION_IDENTITY:
            return {"identity_fields": list(self.fields), "identity_values": list(self.values)}
        return {"fingerprint": self.fingerprint[:16]}

    def values_text(self) -> str:
        """主键值，给判词用。值可能很长，逐个截断。"""
        return "、".join(_short(v) for v in self.values)

    def differing_text(self) -> str:
        """参数里哪几个字段不一样。**这就是这一档判据与逐字那一档的全部差别。**"""
        if not self.differing:
            return ""
        shown = self.differing[:_MAX_DIFFERING_SHOWN]
        tail = (
            f" 等 {len(self.differing)} 个字段"
            if len(self.differing) > _MAX_DIFFERING_SHOWN
            else ""
        )
        return "：" + "、".join(shown) + tail


@dataclass(frozen=True)
class Ungroupable:
    """声明了主键、却有调用没带那几个字段——**这一档判据对它们一次都没跑过**。

    不能折进"判过了、没问题"里：那正是本项目最怕的那种句子（报告说了比观测多的
    话）。所以它单独报一条 `inconclusive`，理由和未知重放标记那一条一样。
    """

    tool_name: str
    fields: tuple[str, ...]
    calls: tuple[CallView, ...]

    def missing_text(self) -> str:
        """每个字段各缺在哪几次调用上——读者要能去改数据。"""
        parts = []
        for name in self.fields:
            seqs = [c.seq for c in self.calls if name not in c.args]
            if seqs:
                parts.append(f"{name}：第 {seqs} 次")
        return "；".join(parts)


def _short(value: Any, limit: int = 40) -> str:
    """把主键值压到一行装得下。截断要能看出来（`…`），不然读的人会以为值就这么短。"""
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


def _freeze(value: Any) -> Any:
    """参数值可能是列表/字典，先压成可哈希的形状再比。

    `{a, b}` 直接进集合会 TypeError；而换成 `str(value)` 的话，`1` 和 `"1"`
    会算成同一个值——判据是"值不同"，不是"打印出来不同"。
    """
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


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
        # 没到达目标系统的调用不构成重复扣款：边界上被拒的那一次，什么也没发生。
        # 判据是 `reached_target` 而不是 `blocked_by_proxy`，理由见模块文档第二条。
        observed = [c for c in calls if c.reached_target]
        writes = [c for c in observed if c.is_write]

        if not writes:
            rejected = len(calls) - len(observed)
            by_proxy = sum(1 for c in calls if c.blocked_by_proxy)
            return [
                inconclusive(
                    self.name,
                    "未观测到任何写操作，无法判定幂等性。"
                    "注意这不等于通过——可能是 Agent 根本没走到写路径。"
                    + (
                        f"（另有 {rejected} 次调用在边界上被拒，其中 {by_proxy} 次"
                        f"是出口代理按策略拦下的，其余是参数校验或未声明的工具。）"
                        if rejected
                        else ""
                    ),
                    detail={
                        "observed_calls": len(observed),
                        "rejected_at_boundary": rejected,
                        "blocked_by_proxy": by_proxy,
                    },
                ),
                # 读重复的成本信息在没有写操作时同样有价值，不能跟着一起丢掉
                *self._read_repeat_note(observed),
            ]

        findings: list[Finding] = []
        clusters, ungroupable = self._clusters(writes)

        # 声明了主键、却有调用没带那几个字段的：这一档判据对它们没跑过。
        # **单独报一条**，不折进下面那句 pass 里——见 `Ungroupable`。
        for miss in ungroupable:
            findings.append(
                inconclusive(
                    self.name,
                    f"{miss.tool_name} 声明了主键（{'、'.join(miss.fields)}），"
                    f"但有 {len(miss.calls)} 次写调用没带齐那几个字段，"
                    f"这些调用只按逐字判据看过，没按主键比过（{miss.missing_text()}）。",
                    detail={
                        "tool_name": miss.tool_name,
                        "identity_fields": list(miss.fields),
                        "seqs": [c.seq for c in miss.calls],
                        "call_ids": [c.id for c in miss.calls],
                    },
                )
            )

        judged = 0
        for criterion, group in clusters:
            if len(group) < 2:
                continue
            finding = self._judge(criterion, group, exemptions)
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
            declared = self._declared(writes)
            # 两句分开说：第二句只在真有工具声明了主键时才出现，而且说的是
            # **主键那条判据**跑没跑——不比逐字那句弱，也不比它强。
            by_identity = (
                f"其中 {declared} 次调用带的字段够按声明的主键比（"
                f"{'、'.join(sorted(self._ids_by_tool(writes)))}），"
                f"也没有两次是同一个操作。"
                if declared
                else ""
            )
            findings.append(
                Finding(
                    detector=self.name,
                    verdict=DetectorVerdict.PASS,
                    summary=(
                        f"已检查 {len(writes)} 次写操作（{tools_checked}），"
                        f"没有两次的规范化参数完全相同。{by_identity}"
                    ),
                    confidence=Confidence.HIGH,
                    detail={
                        "checked_writes": len(writes),
                        "distinct_operations": len(clusters),
                        "identity_checked_writes": declared,
                        "identity_tools": sorted(self._ids_by_tool(writes)),
                    },
                )
            )

        findings.extend(self._read_repeat_note(observed))
        return findings

    # --- 分组：判据由工具声明决定 --------------------------------------------

    def _ids_by_tool(self, calls: list[CallView]) -> dict[str, tuple[str, ...]]:
        """这次轨迹里，哪些工具**声明了非空的主键**。声明是唯一的开关。"""
        out: dict[str, tuple[str, ...]] = {}
        for call in calls:
            fields = self._facts(call).identity_fields
            if fields:
                out[call.tool_name] = fields
        return out

    def _declared(self, calls: list[CallView]) -> int:
        """够按主键比的写调用有多少次（三个条件：声明了、带齐了字段）。"""
        count = 0
        for call in calls:
            fields = self._facts(call).identity_fields
            if fields and all(f in call.args for f in fields):
                count += 1
        return count

    def _clusters(
        self, writes: list[CallView]
    ) -> tuple[list[tuple[Criterion, list[CallView]]], list[Ungroupable]]:
        """把写操作分成组，连同**判据**一起返回。一组最多出一条结论。

        分档的规矩只有一条：**工具声明了主键、这几次调用又都带齐了那几个字段，
        才按主键分**；否则退回逐字那一档。两个方向都是保守的：

          * 没声明 → 一个字都不变（含那些量出来的误报形状，见模块文档第五条）；
          * 声明了但字段没带齐 → 那几次**不按主键比**，并在 `Ungroupable` 里
            报出来。拿"字段缺失"当"值不同"用，会让缺字段的调用各自成组，
            于是这一档判据在数据不齐的时候静默变成逐字那一档。

        主键组的指纹若全都一样，**整组退回逐字那一档**：同样的事实，逐字那句话
        更硬（"这几次的规范化参数完全相同"比"主键相同"说得多），而且这样声明
        不会把一条已经存在的逐字结论变成另一条。
        """
        clusters: list[tuple[Criterion, list[CallView]]] = []
        ungroupable: list[Ungroupable] = []

        by_tool: dict[str, list[CallView]] = defaultdict(list)
        for call in writes:
            by_tool[call.tool_name].append(call)

        for tool_name, calls in by_tool.items():
            fields = self._facts(calls[0]).identity_fields
            rest: list[CallView] = []
            if fields:
                # 桶的键要可哈希，而参数里可能是列表/字典（τ-bench 的 `flights`
                # 就是嵌套列表）。**键用压过的那一份，判词用原值**：读者要按参数
                # 本来的样子复核这条结论，不能让他去认一个 `('F1',)`。
                buckets: dict[tuple[Any, ...], tuple[tuple[Any, ...], list[CallView]]] = {}
                misses: list[CallView] = []
                for call in calls:
                    if not all(f in call.args for f in fields):
                        misses.append(call)
                        continue
                    values = tuple(call.args[f] for f in fields)
                    key = tuple(_freeze(v) for v in values)
                    if key not in buckets:
                        buckets[key] = (values, [])
                    buckets[key][1].append(call)
                rest = misses
                if misses:
                    ungroupable.append(
                        Ungroupable(tool_name=tool_name, fields=tuple(fields), calls=tuple(misses))
                    )
                for values, group in buckets.values():
                    fingerprints = {c.fingerprint for c in group}
                    if len(fingerprints) == 1:
                        # 逐字那一档说得更多，交给它（下面按指纹再分一次会得到同一组）。
                        rest.extend(group)
                    else:
                        clusters.append(
                            (
                                Criterion(
                                    kind=CRITERION_IDENTITY,
                                    fields=tuple(fields),
                                    values=values,
                                    differing=self._differing(group, fields),
                                ),
                                group,
                            )
                        )
            else:
                rest = list(calls)
            by_fp: dict[str, list[CallView]] = defaultdict(list)
            for call in rest:
                by_fp[call.fingerprint].append(call)
            for fingerprint, group in by_fp.items():
                clusters.append((Criterion(kind=CRITERION_EXACT, fingerprint=fingerprint), group))

        clusters.sort(key=lambda pair: (pair[1][0].tool_name, pair[0].sort_key()))
        ungroupable.sort(key=lambda miss: miss.tool_name)
        return clusters, ungroupable

    @staticmethod
    def _differing(group: list[CallView], fields: tuple[str, ...]) -> tuple[str, ...]:
        """组内取值不一致的字段名（主键那几个不在其中，它们按定义是全等的）。

        只列**在这几次调用里都出现过**的字段：一次没带的字段不算"不同"，
        那会报出一个读者去参数里找不到的字段名。
        """
        common = set(group[0].args)
        for call in group[1:]:
            common &= set(call.args)
        return tuple(
            sorted(
                name
                for name in common - set(fields)
                if len({_freeze(c.args[name]) for c in group}) > 1
            )
        )

    def _judge(
        self,
        criterion: Criterion,
        group: list[CallView],
        exemptions: ExemptionSet,
    ) -> Finding | None:
        facts = self._facts(group[0])
        tool_name = group[0].tool_name
        replay = [c for c in group if c.replay_reason == SNAPSHOT_REPLAY]
        agent = [c for c in group if c.replay_reason is None]
        # 其他重放原因（未来的 resume/retry 之类）。目前只记录不判定——
        # 出现未知重放标记时应当能被看见，而不是被静默归进 Agent 自发重复。
        other_replay = [c for c in group if c.replay_reason not in (None, SNAPSHOT_REPLAY)]

        detail: dict[str, Any] = {
            "tool_name": tool_name,
            # **凭什么叫"同一个操作"**。两档判据的结论长得像、含义不同，所以
            # 每一行结论都带着它，读的人不用回去看工具声明。
            "criterion": criterion.kind,
            **criterion.detail_keys(),
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
            # summary 里关于"钱"的那句话依据什么，在这儿留下原始计数。
            "statuses": {
                status: sum(1 for c in group if c.status == status)
                for status in sorted({c.status for c in group})
            },
        }

        # Agent 自发重复不足两次，但确实发生了重复 —— 那就是恢复机制制造的。
        # 顺序很重要：先判 Agent 自发重复，再判恢复重放。反过来的话，
        # 一次"Agent 重复 + 恢复重放"混合的组会被归因成纯恢复问题。
        if len(agent) < 2 and replay:
            return Finding(
                detector=self.name,
                verdict=DetectorVerdict.FAIL,
                summary=(
                    f"{tool_name} 被重复执行 {len(group)} 次{criterion.clause()}，"
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
                # 没有两次的规范化参数完全相同"那句汇总 PASS 里，
                # 把一个判不了的重复说成查过没毛病。
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
            return self._judge_cross_session(criterion, by_ordinal, detail, exemptions)

        # 这里开始是 Agent 自发重复——按保护情况分层判定
        # 这些尾巴只由 `status` 决定，不由上面分了哪一支决定：判定分组看的是
        # "工具声明了什么保护、Agent 怎么用键"，那和这次调用成功没成功无关。
        ok = _ok_count(agent)
        if not facts.enforces_idempotency:
            if ok == len(agent):
                tail = "这是重复扣款。"
            elif ok == 0:
                tail = (
                    "这是重复尝试，不是已证实的重复扣款——失败之后重试一个没有"
                    "幂等保护的写操作，正是重复扣款最常见的成因。"
                )
            else:
                tail = "重复扣款是真的。"
            finding = Finding(
                detector=self.name,
                verdict=DetectorVerdict.FAIL,
                summary=(
                    f"{tool_name} 是没有幂等保护的写操作，被重复尝试了 {len(agent)} 次"
                    f"{criterion.clause()}。{_outcome(agent)}{tail}"
                ),
                confidence=Confidence.HIGH,
                detail=detail,
            )
        else:
            keys = {c.idempotency_key for c in agent}
            if keys == {None}:
                if ok == len(agent):
                    tail = "重复扣款发生了。"
                elif ok == 0:
                    tail = "重复扣款尚未证实，但保护确实没生效。"
                else:
                    tail = "重复扣款是真的。"
                finding = Finding(
                    detector=self.name,
                    verdict=DetectorVerdict.FAIL,
                    summary=(
                        f"{tool_name} 声明了幂等保护，但重复调用未携带幂等键，"
                        f"保护未生效，实际重复调用 {len(agent)} 次{criterion.clause()}。"
                        f"{_outcome(agent)}{tail}"
                    ),
                    confidence=Confidence.HIGH,
                    detail={**detail, "reason_code": "missing_key"},
                )
            elif len(keys) > 1:
                if ok == len(agent):
                    tail = "重复扣款发生了。"
                elif ok == 0:
                    tail = "重复扣款尚未证实，但保护确实被绕过了。"
                else:
                    tail = "重复扣款是真的。"
                finding = Finding(
                    detector=self.name,
                    verdict=DetectorVerdict.FAIL,
                    summary=(
                        f"{tool_name} 的重复调用每次用了不同的幂等键"
                        f"（{len(keys)} 个），保护被绕过，实际重复调用 {len(agent)} 次"
                        f"{criterion.clause()}。"
                        f"工具声明保护不等于保护生效。{_outcome(agent)}{tail}"
                    ),
                    confidence=Confidence.HIGH,
                    detail={**detail, "reason_code": "key_rotated"},
                )
            else:
                # 同一个幂等键重复调用：保护生效了，没有重复扣款。
                # 报 warning 而不是 pass：Agent 确实做了无谓的重复调用，
                # 值得看见，但不该按缺陷对待。
                #
                # 但"保护生效"本身也是要凭证的：目标系统接受了，才谈得上后面几次
                # 被它按幂等键去重掉。一次都没成功时，这几次到底有没有被去重、
                # 钱动没动，边界上看不出来——那时候说"保护生效，未产生重复扣款"
                # 就是把**声明**读成了**事实**。
                #
                # 这一支不走 `_outcome()`：它那句"至少扣了一笔；总共几笔看不出来"
                # 和"未产生重复扣款"读起来打架，而这里真正要说的不是"扣了几笔"，
                # 是"去重这件事有没有被证到"。
                if ok == len(agent):
                    observation = (
                        "这几次调用全部返回成功——目标系统接受了每一次，"
                        "重复的那几次应当被它按幂等键去重了，未产生重复扣款。"
                    )
                    confidence = Confidence.MEDIUM
                elif ok == 0:
                    observation = (
                        f"这 {len(agent)} 次调用都返回错误，一次都没成功——"
                        f"保护有没有真的去重、钱动没动，边界上看不出来。"
                    )
                    confidence = Confidence.LOW
                else:
                    observation = (
                        f"这 {len(agent)} 次调用里有 {ok} 次返回成功、"
                        f"{len(agent) - ok} 次返回错误。复用了同一个幂等键，"
                        f"成功的那几次应当被去重成一次；失败的那几次到底有没有落账，"
                        f"边界上看不出来。"
                    )
                    confidence = Confidence.LOW
                finding = Finding(
                    detector=self.name,
                    verdict=DetectorVerdict.WARNING,
                    summary=(
                        f"{tool_name} 被重复调用 {len(agent)} 次{criterion.clause()}，"
                        f"但复用了同一幂等键。{observation}"
                    ),
                    confidence=confidence,
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
        criterion: Criterion,
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

        两种情况都报 FAIL，因为**同一件事确实被做了两遍**。区别在 summary 里说清楚
        该去修哪儿——否则用户会去改一个没问题的 Prompt。

        归因和"钱"是两根轴，别混：归因看记忆视野，措辞看 `status`。原本这里两种
        情况都断言"跨会话重复扣款"/"钱是真的扣了两次"，而判据里根本没有成功与否
        ——两次都返回错误时，这句话就是报告说了比观测更多的话。
        """
        ordinals = sorted(by_ordinal)
        first, last = ordinals[0], ordinals[-1]
        tool_name = by_ordinal[first][0].tool_name
        first_calls = by_ordinal[first]
        last_calls = by_ordinal[last]
        facts = self._facts(first_calls[0])
        # 视野是会话属性，同一会话内每条调用都一样，取第一条即可。
        horizon = last_calls[0].memory_horizon_ordinal
        should_have_known = first <= horizon
        flat = [c for calls in by_ordinal.values() for c in calls]
        all_ok = _ok_count(flat) == len(flat)

        session_detail = {
            **detail,
            "sessions": {str(o): [c.seq for c in cs] for o, cs in sorted(by_ordinal.items())},
            "first_ordinal": first,
            "last_ordinal": last,
            "memory_horizon_ordinal": horizon,
            "missed_ordinals": [
                o for o in range(horizon + 1, last) if o in by_ordinal
            ],
            "statuses": {
                status: sum(1 for c in flat if c.status == status)
                for status in sorted({c.status for c in flat})
            },
        }

        if should_have_known:
            finding = Finding(
                detector=self.name,
                verdict=DetectorVerdict.FAIL,
                summary=(
                    f"{tool_name} 的同一操作在会话 #{first} 与 #{last} 各调用了一次"
                    f"{criterion.clause()}（记忆视野覆盖到 #{horizon}，Agent 本可以认出来）。"
                    f"{_outcome(flat)}"
                    + (
                        "这是跨会话重复扣款——它没有把'我做过了'这件事用起来。"
                        if all_ok
                        else "Agent 的锅照样是它的锅：'我做过了'这件事就在它的记忆里，"
                        "而它没有用起来。"
                    )
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
                    f"{tool_name} 的同一操作在会话 #{first} 与 #{last} 各调用了一次"
                    f"{criterion.clause()}，"
                    f"但会话 #{last} 的记忆只覆盖到 #{horizon}"
                    + (f"（#{'、#'.join(forgotten)} 那段被丢掉了）" if forgotten else "")
                    + f"。{_outcome(flat)}"
                    + (
                        "重复扣款是真实的，但 Agent 不可能知道——"
                        if all_ok
                        else "重复是真实的，但 Agent 不可能知道——"
                    )
                    + "要去修记忆的恢复/传递，不要改 Agent。"
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
