"""检测器基类：四态语义、豁免机制、以及检测器看到的调用视图。

规格见 docs/detectors.md。这里落实三条最容易被做错的：

1. **`inconclusive` 不是 `pass`。** 检测器必须能说出"我没测到"，而且调用方
   必须把它和通过分开统计。把"没走到该路径"折叠成绿色，是这类工具最严重的
   失效方式——用户会拿着一个什么都没测的报告去上生产。

2. **写操作默认不可豁免。** 只读工具的重复调用确实无害，可以豁免；但写操作
   一旦可豁免，豁免规则就成了把重复扣款静音的最短路径。所以豁免写操作需要
   在规则里显式声明 `allow_write: true`——让"我知道我在静音什么"成为一个
   必须写下来的动作。

3. **豁免不是静默的。** 命中的豁免要一路带到报告里（`exempted_by`），
   让"因 N 条豁免未告警"可见。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from sandbox.db.models import Confidence, DetectorVerdict

#: 边界上"这次调用有没有**到**目标系统"的判据：`status` 落在这里面的才算发生过。
#:
#: **它和 `blocked_by_proxy` 回答的不是同一个问题。** 后者回答"**是谁**拦的"：
#: True 是沙箱自己的策略，False 是 Agent 自己送错了参数。两个问题在四种 status
#: 上只重合两次：
#:
#:   * `ok` / `error`   —— 发出去了。`error` 是目标系统回的错，调用本身到了；
#:   * `invalid_args`   —— 参数被 schema 拒了，一次都没发出去，**`blocked_by_proxy=False`**；
#:   * `blocked`        —— 工具没声明（`proxy/app.py:151`），同样没发出去，同样是 False。
#:
#: 后两种**刻意**不带那个标记，理由写在 `proxy/app.py` 里，`tests/test_api.py` 的
#: `test_an_invalid_args_call_is_not_marked_as_a_proxy_block` 守着它。所以检测器
#: 不能拿 `blocked_by_proxy` 当"到没到"用：那样三次参数写错的 `create_payment`
#: 会拿到和三次真扣款**一字不差**的报告（量于 2026-10-02，
#: `../_probes/measure_rejected_calls.py`）。
#:
#: 取值来自 `proxy/app.py` 里那几处 `record_tool_call(...)`；
#: `tests/test_call_status_vocabulary.py` 盯着这份名单有没有走样。
REACHED_TARGET: frozenset[str] = frozenset({"ok", "error"})

#: 补集：**在边界上就被拦下、一次都没发出去**的那些。两种都没到目标系统，
#: 区别只在于是谁拦的——`blocked` 可能是沙箱策略（那时另有 `blocked_by_proxy=True`），
#: 也可能是工具压根没声明（那时是 False，Agent 自己的错）。
#:
#: 单独的常量而不是"剩下的都是"，是为了让"补集"这句话可以被机械核对：
#: 谁新加一种 status，谁就得在这里或在上面的名单里挑一边，挑漏了测试会红。
REJECTED_AT_BOUNDARY: frozenset[str] = frozenset({"blocked", "invalid_args"})

#: 边界上可能出现的一切 `status`。上面两个集合**不重不漏**地盖住它——
#: 两个方向都由 `tests/test_call_status_vocabulary.py` 拿 `proxy/app.py`
#: 的真实写法核对：生产里写过的每一种都在名单里，名单里的每一种生产里都写过。
CALL_STATUSES: frozenset[str] = REACHED_TARGET | REJECTED_AT_BOUNDARY


@dataclass(frozen=True)
class CallView:
    """检测器看到的工具调用。与 ORM 解耦，便于直接喂构造好的轨迹做测试。

    刻意不含返回体，**连它的字节数也不含**：检测器判的是**行为**（调了什么、
    调了几次、参数关系），不是返回值内容。让返回值进来会诱导写出"看返回里有
    没有报错"这种脆弱规则，而返回值恰恰是最容易被 Agent 影响的字段。
    （字节数在 `ToolCall.result_bytes` 里，那是给成本报告算"返回体在上下文里
    被重发了多少"用的，不是给检测器用的——两件事，别顺手并过来。）
    """

    id: str
    seq: int
    tool_name: str
    args: dict[str, Any]
    fingerprint: str
    is_write: bool
    session_sequence_id: str = ""
    session_id: str = ""
    # 会话序号与记忆视野。两者其实是**会话**的属性，冗余到每条调用上——
    # 与已经这么做的 session_id / session_sequence_id 同一套路。理由是检测器
    # 的契约是"只吃 list[CallView]"，为跨会话检测单独开一条参数通道会让
    # 所有检测器的签名一起变复杂，换来的只是省两个字段。
    #
    # 判断跨会话重复时用的就是 memory_horizon_ordinal：它比 session_ordinal
    # 小，说明这段记忆被回滚掉了，Agent 不可能知道——归因的分界线就在这里。
    session_ordinal: int = 0
    memory_horizon_ordinal: int = -1
    idempotency_key: str | None = None
    replay_reason: str | None = None
    blocked_by_proxy: bool = False
    status: str = "ok"

    @property
    def reached_target(self) -> bool:
        """这次调用到没到目标系统。没到的**一次都没发生过**，不进任何判定。

        见 `REACHED_TARGET`：它问的和 `blocked_by_proxy` 不是同一件事，别互相顶替。
        """
        return self.status in REACHED_TARGET


@dataclass(frozen=True)
class Finding:
    detector: str
    verdict: DetectorVerdict
    summary: str
    confidence: Confidence = Confidence.HIGH
    detail: dict[str, Any] = field(default_factory=dict)
    exempted_by: str | None = None


#: `match` 里认得的键。写错的键（`arg_differs_on_page` 这种）会让规则永远
#: 匹配不上——而规则看起来是在的，报告里也不会有任何异常。一条"装上了但没
#: 接线"的静音开关比没有规则更坏，所以在这里硬拦。
#:
#: 拦"这个键存在吗"只是一半，另一半是 `EXEMPTION_AWARE`：键认得、
#: 而**这个检测器答不了**的时候，规则会变成一条死规则，比不生效还坏。
KNOWN_MATCH_KEYS: frozenset[str] = frozenset({"arg_differs_on", "tool_has_idempotency"})

#: 哪些检测器**真的会去查豁免**，以及它各自答得了哪些 `match` 键。按**检测器名**
#: 写，不按类：同一个 `LoopDetector` 里 `loop.approximate` 答得了"这两次调用差在
#: 哪些字段上"，`loop.ping_pong` 答不了——乒乓判定压根不算那个。
#:
#: **不在表里的检测器，规则写上去永远不会命中。** `KNOWN_DETECTORS` 里有 15 个
#: 名字，**查豁免的就是下面这 6 个**（这张表的键）。另外 6 个——`loop`、`loop.retry`、
#: `loop.cross_session`、`loop.snapshot_replay`、`loop.unknown_replay`、
#: `idempotency.read_repeats`——那几条路上压根没有查豁免那一步；剩下的记忆三族是
#: 另一回事，见 `NOT_EXEMPTIBLE`。不查就不收——收下一条永远不会命中的规则，等于让
#: 写的人以为自己处理过了。
#:
#: **答不了某个键的时候，规则就是一条死规则——两种写法都是。** 调用点没把这个
#: 事实传下来，比对的那一边就是 `None`，而 `None` 跟 `True`、`False` 都不相等。
#: 所以 `tool_has_idempotency` 无论写 true 还是 false，写在 `loop.strict` 上都不会
#: 命中（实测：拆掉闸门之后，两种写法 × 三种调用组合，0 次命中）。
#:
#: 报告上完全看不出来：那条规则从来没提过，而"没有这条结论"和"这条结论被豁免了"
#: 是同一个样子。写规则的人以为自己处理过了。所以加载时就拒。
#:
#: 这是 `KNOWN_MATCH_KEYS` 那条纪律的另一半：那张表管"这个键存在吗"，这张管
#: "这个检测器查不查豁免、答得了吗"。
#:
#: 运行时还有第二道（`ExemptionSet.find` 按调用点实际传了哪些事实再问一遍）。
#: 两道都留着，因为它们防的不是同一件事：这张表防**人写规则时挑错了检测器**，
#: 运行时那道防**有人加了新的 find 调用点却忘了把事实传下去**——那恰恰是这张
#: 表会漂的地方，而它漂起来没有任何症状。
EXEMPTION_AWARE: dict[str, frozenset[str]] = {
    "idempotency": frozenset({"tool_has_idempotency"}),
    "loop.strict": frozenset({"arg_differs_on"}),
    "loop.approximate": frozenset({"arg_differs_on"}),
    "loop.ping_pong": frozenset(),
    "autonomy": frozenset(),
    "autonomy.unauthorized": frozenset(),
}


class UnanswerableMatch(ValueError):
    """一条规则在问一个这个检测器答不了的问题。加载时拒，运行时也拒。"""

#: **不吃豁免的检测器。** 记忆族的判据是场景里逐条声明的 `must_hold` / `forbidden`，
#: 声明本身就是那个"例外机制"——再叠一层豁免，等于给同一件事开两个开关。
#:
#: 所以在这里硬拦而不是"接受了但不生效"。后者正是一个"装上了但没接线"的静音开关：
#: 规则看起来在场景文件里，检测器也不会报错，只是那条 fail 照样出现在报告里——
#: 用户以为自己已经处理过这个问题了。豁免机制自己最不能出的就是这个毛病。
NOT_EXEMPTIBLE: frozenset[str] = frozenset({"memory", "memory.drift", "memory.poisoning"})


def check_match(match: dict[str, Any]) -> None:
    """校验 `match` 的形状。不合法就抛 `ValueError`。

    单独做成函数，是为了让**场景加载**（用户写的 YAML）和**检测器构造**报同一
    条错：两条入口各写一遍校验，迟早会有一条松掉，而松掉的那条正是用户走的那条。
    """
    unknown = set(match) - KNOWN_MATCH_KEYS
    if unknown:
        raise ValueError(
            f"match 里有不认识的键 {sorted(unknown)}，这条规则永远不会命中。"
            f"认得的键只有 {sorted(KNOWN_MATCH_KEYS)}。"
        )
    differs = match.get("arg_differs_on")
    if differs is not None and (
        isinstance(differs, str)
        or not isinstance(differs, (list, tuple))
        or not all(isinstance(name, str) for name in differs)
    ):
        raise ValueError(
            f"arg_differs_on 要的是字段名列表（例如 [page]），收到 {differs!r}。"
            f"写成字符串的话集合运算是按字符走的，看起来能跑，命中的是别的规则。"
        )
    if differs is not None and len(differs) == 0:
        raise ValueError(
            "arg_differs_on 给的是空列表——「差异只在零个字段上」就是「两次调用完全"
            "相同」，而这个条件永远不会成立：豁免那一侧专门拦着不让它命中（参数一模一样"
            "的重复是确定性最高的循环，见 ExemptionSet.find 里那个 not differing_fields）。"
            "所以这是一条写得出来、却永远不命中的规则，与写在答不了的检测器上是同一种坏。\n"
            "要连这种重复一起静音，就把 match 写成空的——一个条件都不加。但那是"
            "「这一族全部放行」，比看起来更像一个开关，写之前先确认你要的是这个。"
        )


def check_answerable(detector: str, match: dict[str, Any]) -> None:
    """这个检测器查豁免吗；查的话，这条规则要问的事它答得了吗。

    答不了（或者压根不查）就抛 `UnanswerableMatch`。两件事放在同一个函数里，
    因为它们对写规则的人来说是同一种后果：**规则看起来在，实际不按他以为的走**。

    和 `check_match` 分开是因为它们要的东西不一样：那个只看 `match` 的形状，
    这个还要看**规则指的是哪个检测器**。两个入口（场景 YAML、过线的规则字典）
    都从 `Exemption.__post_init__` 走，所以两条路报的是同一条错。
    """
    if detector not in EXEMPTION_AWARE:
        raise UnanswerableMatch(
            f"检测器 {detector} 不查豁免——这条规则永远不会命中。\n"
            f"今天查豁免的只有 {sorted(EXEMPTION_AWARE)}。\n"
            f"这不是键写错了：那条路上压根没有查豁免的那一步（记忆那几族是另一种"
            f"情况，它们的例外机制是逐条声明的判据本身，见 NOT_EXEMPTIBLE）。\n"
            f"一条永远不命中的规则比没有规则更坏——读报告的人会以为那件事已经"
            f"处理过了。要豁免这一族，得先给它接上。"
        )

    answers = EXEMPTION_AWARE[detector]
    unanswerable = sorted(set(match) - answers)
    if not unanswerable:
        return

    lines = []
    for key in unanswerable:
        names = sorted(name for name, keys in EXEMPTION_AWARE.items() if key in keys)
        if names:
            lines.append(f"  {key} 今天只有 {'、'.join(names)} 答得了")
        else:
            lines.append(f"  {key} 今天没有任何检测器答得了——它对谁都是一句空话")
    raise UnanswerableMatch(
        f"检测器 {detector} 答不了这些 match 键：{unanswerable}。\n"
        + "\n".join(lines)
        + "\n答不了的结果不是「这条规则不生效」，是它永远也不会命中：调用点没把这个"
        "事实传下来，比对的那一边是 None，而 None 跟 true、false 都不相等——所以"
        "两种写法都一样是死规则。听起来比恒真好一点，其实一样坏：报告里「没有这条"
        "结论」和「这条结论被豁免了」是同一个样子，你会以为自己处理过了。\n"
        "换一个检测器、换一个键，或者把那个条件去掉——"
        "不想要的条件应该删掉，而不是留下来当装饰。"
    )


@dataclass(frozen=True)
class Exemption:
    """一条豁免规则。

    `match` 目前支持：
      - `arg_differs_on`: [字段名] —— 仅当两次调用的差异**只**出现在这些字段上时命中
      - `tool_has_idempotency`: bool —— 仅当工具声明了幂等保护时命中

    **两个键都不是哪儿都能用的**：只有当场知道那件事的检测器才答得了它，见
    `EXEMPTION_AWARE`。写到答不了的检测器上是加载期错误。

    而且，**不是每个检测器都会去查豁免**。记忆那三族写在 `NOT_EXEMPTIBLE` 里，
    另有六条路（`loop`、`loop.retry`、`loop.cross_session`、`loop.snapshot_replay`、
    `loop.unknown_replay`、`idempotency.read_repeats`）压根没有查豁免那一步。
    两边都拒，理由不同：记忆族是"本就有另一套例外机制，别叠两个开关"，那六条是
    "收了也不会有人看"。但给写规则的人的后果是同一种——规则看起来在，实际不按
    他以为的走。
    """

    detector: str
    reason: str
    tool: str | None = None
    match: dict[str, Any] = field(default_factory=dict)
    # 显式声明"我知道这条规则会静音写操作"。默认不允许。
    allow_write: bool = False

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError(
                f"豁免规则 {self!r} 缺少 reason。"
                f"没有理由的豁免在报告里无法复核，等于永久静音。"
            )
        if self.detector in NOT_EXEMPTIBLE:
            raise ValueError(
                f"豁免规则 {self!r} 指向 {self.detector}，而记忆族的结论不吃豁免。\n"
                f"记忆族的判据就是场景里逐条声明的 must_hold / forbidden——"
                f"不想让某条算失败，就把那条声明去掉，而不是再叠一层静音。"
            )
        try:
            check_match(self.match)
        except ValueError as exc:
            raise ValueError(f"豁免规则 {self!r} 的 {exc}") from exc
        try:
            check_answerable(self.detector, self.match)
        except UnanswerableMatch as exc:
            raise UnanswerableMatch(f"豁免规则 {self.label}：{exc}") from exc

    @property
    def label(self) -> str:
        target = self.tool or "*"
        return f"exemption:{self.detector}/{target}"

    def note(self) -> str:
        """静音留痕的固定措辞，附在结论摘要后面。

        带上 reason 是刻意的：只印 label 的话，读报告的人还得翻回场景文件才知道
        "为什么允许"。豁免的全部代价就是被看见——省掉理由等于让它重新变成静音。
        """
        return f"（已被豁免规则 {self.label} 静音：{self.reason}）"

    def to_dict(self) -> dict[str, Any]:
        """过线用的形状。字段名与场景 YAML 里的写法一致，读的人不用学第二套。"""
        return {
            "detector": self.detector,
            "tool": self.tool,
            "reason": self.reason,
            "match": dict(self.match),
            "allow_write": self.allow_write,
        }

    @classmethod
    def of(cls, rule: dict[str, Any]) -> Exemption:
        """从过线的形状还原一条规则，**校验照走一遍**。

        直接 `Exemption(**rule)` 不行：字典里多一个键就是一个 `TypeError`，
        而那种报错读起来像程序崩了，不像"你的规则写错了"。少一个键、写错一个
        检测器名、`match` 里写错键——三条路都在这里变回同一种说法：
        这条规则错在哪。
        """
        if not isinstance(rule, dict):
            raise ValueError(f"豁免规则要是一个对象（字段名: 值），收到 {rule!r}。")
        unknown = set(rule) - {"detector", "tool", "reason", "match", "allow_write"}
        if unknown:
            raise ValueError(
                f"豁免规则里有不认识的字段 {sorted(unknown)}。"
                f"认得的只有 detector / tool / reason / match / allow_write。"
            )
        for required in ("detector", "reason"):
            if not rule.get(required):
                raise ValueError(f"豁免规则缺少 {required}：{rule!r}")
        return cls(
            detector=rule["detector"],
            tool=rule.get("tool"),
            reason=rule["reason"],
            match=dict(rule.get("match") or {}),
            allow_write=bool(rule.get("allow_write", False)),
        )


class ExemptionSet:
    def __init__(self, rules: list[Exemption] | None = None) -> None:
        self.rules = list(rules or [])

    def to_dicts(self) -> list[dict[str, Any]]:
        return [rule.to_dict() for rule in self.rules]

    @classmethod
    def of(cls, rules: list[dict[str, Any]] | None) -> ExemptionSet:
        """从过线的形状还原一整套规则。空/缺省 = 一条都没有。"""
        return cls([Exemption.of(rule) for rule in rules or []])

    def find(
        self,
        detector: str,
        tool: str,
        *,
        is_write: bool,
        differing_fields: tuple[str, ...] | None = None,
        tool_has_idempotency: bool | None = None,
    ) -> Exemption | None:
        """找一条命中的豁免。写操作且未显式允许 → 直接跳过，不豁免。

        两个事实参数写 `None` 的意思是**这个调用点答不了**（乒乓那一处不算
        "两次调用差在哪些字段上"），不是"值是空/假"。规则要问而事实是 `None`
        时**直接抛**，不按默认值算下去：

        * 加载期那张表（`EXEMPTION_AWARE`）拦的是**人挑错了检测器**；
        * 这里拦的是**调用点忘了把事实传下去**——表会漂，而"传没传"不会。

        两道闸门防的不是同一件事，所以两道都要有。少了这一道，一个新加的
        `find` 调用点就能让某条带条件的规则悄悄变成死规则（`None` 跟 true、false
        都不相等，写哪个值都不命中）——报告上什么都看不出来，而这恰恰是豁免机制
        （一个专门用来"让结论消失"的东西）最不能出的毛病。
        """
        for rule in self.rules:
            if rule.detector != detector:
                continue
            if rule.tool is not None and rule.tool != tool:
                continue
            if is_write and not rule.allow_write:
                continue

            expected_diff = rule.match.get("arg_differs_on")
            if expected_diff is not None:
                self._require_answered(rule, detector, "arg_differs_on", differing_fields)
                # "差异只在 page 上"要求**确实存在差异且只在 page 上**。
                # 不加 `differing_fields` 非空这个条件的话，参数完全相同的严格
                # 重复也会被翻页豁免静音掉——而那是确定性最高的循环。
                if not differing_fields:
                    continue
                if set(differing_fields) - set(expected_diff):
                    continue
            if "tool_has_idempotency" in rule.match:
                self._require_answered(
                    rule, detector, "tool_has_idempotency", tool_has_idempotency
                )
                if bool(rule.match["tool_has_idempotency"]) != tool_has_idempotency:
                    continue
            return rule
        return None

    @staticmethod
    def _require_answered(
        rule: Exemption, detector: str, key: str, answered: Any
    ) -> None:
        """规则问了、而这个调用点没答 → 抛。答了（哪怕是 False / 空元组）就放行。

        判的是 `is None` 而不是真假：`False` 和 `()` 都是**答过的答案**，
        `None` 才是"没人问过我这件事"。
        """
        if answered is not None:
            return
        raise UnanswerableMatch(
            f"规则 {rule.label} 拿 {key} 当条件，而 {detector} 这一处调豁免时"
            f"没有把这件事传下来。\n"
            f"按默认值（None）算下去的话这条规则永远不命中——None 跟 true、false"
            f"都不相等，写哪个值都一样。而它看起来完全正常：报告里「没有这条结论」"
            f"和「这条结论被豁免了」是同一个样子。\n"
            f"两条路：把这个检测器加进 EXEMPTION_AWARE 并在调用点把事实传下去，"
            f"或者让这条规则换个条件。"
        )


def inconclusive(detector: str, summary: str, detail: dict[str, Any] | None = None) -> Finding:
    """未覆盖该路径的统一构造。

    单独做成函数是**刻意的**：让"我没测到"在代码里有一个显式的、好搜索的出口，
    而不是靠某个分支忘了返回 fail 来实现。搜索 `inconclusive(` 就能审计出
    所有"宣布自己没测到"的地方。
    """
    return Finding(
        detector=detector,
        verdict=DetectorVerdict.INCONCLUSIVE,
        summary=summary,
        confidence=Confidence.LOW,
        detail=detail or {},
    )


class Detector(Protocol):
    name: str

    def run(self, calls: list[CallView], exemptions: ExemptionSet) -> list[Finding]: ...
