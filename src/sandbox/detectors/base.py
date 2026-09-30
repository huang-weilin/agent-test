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


@dataclass(frozen=True)
class CallView:
    """检测器看到的工具调用。与 ORM 解耦，便于直接喂构造好的轨迹做测试。

    刻意不含 result_json：检测器判的是**行为**（调了什么、调了几次、参数关系），
    不是返回值内容。让返回值进来会诱导写出"看返回里有没有报错"这种脆弱规则，
    而返回值恰恰是最容易被 Agent 影响的字段。
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
KNOWN_MATCH_KEYS: frozenset[str] = frozenset({"arg_differs_on", "tool_has_idempotency"})


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


@dataclass(frozen=True)
class Exemption:
    """一条豁免规则。

    `match` 目前支持：
      - `arg_differs_on`: [字段名] —— 仅当两次调用的差异**只**出现在这些字段上时命中
      - `tool_has_idempotency`: bool —— 仅当工具声明了幂等保护时命中
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
        try:
            check_match(self.match)
        except ValueError as exc:
            raise ValueError(f"豁免规则 {self!r} 的 {exc}") from exc

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


class ExemptionSet:
    def __init__(self, rules: list[Exemption] | None = None) -> None:
        self.rules = list(rules or [])

    def find(
        self,
        detector: str,
        tool: str,
        *,
        is_write: bool,
        differing_fields: tuple[str, ...] = (),
        tool_has_idempotency: bool = False,
    ) -> Exemption | None:
        """找一条命中的豁免。写操作且未显式允许 → 直接跳过，不豁免。"""
        for rule in self.rules:
            if rule.detector != detector:
                continue
            if rule.tool is not None and rule.tool != tool:
                continue
            if is_write and not rule.allow_write:
                continue

            expected_diff = rule.match.get("arg_differs_on")
            if expected_diff is not None:
                # "差异只在 page 上"要求**确实存在差异且只在 page 上**。
                # 不加 `differing_fields` 非空这个条件的话，参数完全相同的严格
                # 重复也会被翻页豁免静音掉——而那是确定性最高的循环。
                if not differing_fields:
                    continue
                if set(differing_fields) - set(expected_diff):
                    continue
            if "tool_has_idempotency" in rule.match and (
                bool(rule.match["tool_has_idempotency"]) != tool_has_idempotency
            ):
                continue
            return rule
        return None


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
