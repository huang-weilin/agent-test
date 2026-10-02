"""标注语料：把"检测器判得对不对"变成一件可以数出来的事。

`docs/detectors.md` 给每个检测器写了精度目标（精确率 ≥ 0.95 之类）。**没有语料，
那些数字就是占位**——它们既无法验收，也无法调优。这个模块让它们可算。

## 一条语料是什么

一段**手写的**轨迹 + 它的标注：

  * `calls` —— 边界上看到的那串调用（谁调的、什么参数、属于哪段会话、成功没有）
  * `tools` / `sessions` —— 两份**声明**（工具的读写属性与授权、哪段会话没人看着）
  * `expect` —— 每一族检测器该给出的结论

## 三件刻意的事

**一、轨迹是手写的，不是从某次运行里录回来的。** 录回来的话，标注只能照着检测器
当时给的结论写——那是同义反复：检测器说什么，语料就"期望"什么，精确率永远是 1。
手写才可能写下"我认为这是一次循环"这个**独立判断**，然后拿它去撞检测器。

**二、负样本才是重点。** 一个只会报警的检测器等于是没有检测器：天天误报的下一步
就是被人整体忽略，包括那些真正要紧的。所以每条轨迹都要对**每一族**检测器表态——
"这一族该报" 或 "这一族不许报"。少表态就加载失败，不能靠"没写等于不关心"滑过去：
没写等于没标，而没标的数据混进精确率里，数字就变成一句关于没标的空话。

**三、标注比对与场景 `expect` 共用 `match_expectations`。** 语料和场景用的是同一套
四态语义，两处各写一遍的话，总有一天会出现"语料判通过、场景判不通过"。

## 这不算什么

语料的精确率是**关于这份语料**的陈述，不是关于世界的。样本量小，一条误报就能让
精确率掉几个百分点。要更大的结论，得先把语料做大——而做大的前提是有人真的去
标注，这件事没法自动化。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sandbox.db.models import DetectorVerdict
from sandbox.detectors import KNOWN_DETECTORS, run_all
from sandbox.detectors.base import (
    CALL_STATUSES,
    CallView,
    Exemption,
    ExemptionSet,
    Finding,
)
from sandbox.detectors.idempotency import ToolFacts, checked_identity_fields
from sandbox.detectors.memory import ForbiddenText, MemoryRules, MustHold
from sandbox.memory.observe import PHASE_END, PHASE_START, MemoryEntry, MemoryObservation
from sandbox.scenario.expect import match_expectations
from sandbox.scenario.spec import Expectation
from sandbox.trace.normalize import fingerprint, normalize_args

#: 语料来源。**白名单**：来源是给读报告的人判断可信度的依据，写个没见过的词
#: 就等于那句话没人能核对。PERSIST / GhostWriter 是公开来源（见 docs/detectors.md
#: 第八节），"自造"是自己构造的，"访谈"需要授权才能收进仓库。
SOURCES: frozenset[str] = frozenset(
    {"自造", "公开-PERSIST", "公开-GhostWriter", "公开-tau-bench", "访谈"}
)

#: 检测器族。每一族都必须在语料里表态，缺一个就加载失败。
FAMILIES: tuple[str, ...] = ("loop", "idempotency", "autonomy", "memory")

#: `expect` 里表示"这一族不许报 fail/warning"的写法。
NONE = "none"

_ITEM_KEYS = {
    "id",
    "source",
    "why",
    "tools",
    "sessions",
    "calls",
    "exemptions",
    "memory",
    "expect",
    "absent",
}
_CALL_KEYS = {
    "tool",
    "args",
    "session",
    "status",
    "blocked_by_proxy",
    "replay_reason",
    "idempotency_key",
}
_SESSION_KEYS = {"ordinal", "name", "autonomous", "memory_horizon"}
_TOOL_KEYS = {"is_write", "enforces_idempotency", "may_run_unattended", "identity_fields"}

#: 语料里记忆判据能写的键。前两组和场景 YAML 那边是**同一组**（`MustHoldDecl` /
#: `ForbiddenTextDecl` 的字段），有测试钉着两边不许漂移。
#: `observations` 只属于语料：真实运行的观测由引擎在会话边界上拍，不是写出来的。
MEMORY_KEYS: frozenset[str] = frozenset({"observations", "must_hold", "forbidden"})
MUST_HOLD_KEYS: frozenset[str] = frozenset({"path", "reason"})
FORBIDDEN_KEYS: frozenset[str] = frozenset({"text", "reason"})
OBSERVATION_KEYS: frozenset[str] = frozenset({"ordinal", "phase", "entries", "horizon"})


class CorpusError(ValueError):
    """语料文件写错了。**一律在加载时抛，不在评分时悄悄跳过。**

    跳过的话，一条写坏的语料会以"未命中"的形式出现在结果里，而那看起来和
    "检测器漏报"一模一样——一个格式笔误会被读成检测器的缺陷。
    """


@dataclass(frozen=True)
class SessionSpec:
    ordinal: int
    name: str = ""
    autonomous: bool = False
    #: 这段会话开始时，Agent 的记忆覆盖到哪个序号。默认 = 上一个会话，
    #: 即"自然延续、记得此前一切"；回滚过的会话显式写小一些。
    memory_horizon: int | None = None

    def horizon(self) -> int:
        return self.ordinal - 1 if self.memory_horizon is None else self.memory_horizon


@dataclass(frozen=True)
class CorpusItem:
    id: str
    source: str
    why: str
    path: str
    calls: list[CallView]
    tool_facts: dict[str, ToolFacts]
    unattended_allowlist: frozenset[str]
    autonomous_ordinals: frozenset[int]
    exemptions: ExemptionSet
    #: 会话边界上的记忆观测。**第二条观测通道**——它不是边界上的调用，
    #: 所以不在 `calls` 里（见 detectors/memory.py 的说明）。
    memories: list[MemoryObservation] = field(default_factory=list)
    #: 记忆判据（must_hold / forbidden）。空 = 这一族只观测不判定。
    memory_rules: MemoryRules = field(default_factory=MemoryRules)
    #: 该报的结论，逐条比对（含 verdict 与 reason_code）。
    expect: list[Expectation] = field(default_factory=list)
    #: 每一族检测器"该不该报"。值为 True 表示这一族在这条轨迹上应当报出
    #: fail/warning；False 表示不许报。这是算精确率/召回率的依据。
    family_labels: dict[str, bool] = field(default_factory=dict)
    #: 每条调用在语料里出现的行号，报错时能指出是哪儿。
    source_lines: dict[str, int] = field(default_factory=dict)
    #: **不许出现**的结论，`(检测器名, 结论或 None)`。族级的 `none` 说的是
    #: "这一族不许报 fail/warning"，说的是**严重程度**；`absent` 说的是某个
    #: 具体结论不许存在。两者拦的不是一件事：一个"报了 fail 有道理、但同时
    #: 又报了一条 pass 说它很规矩"的检测器，族级标注拦不住。
    absent: list[tuple[str, DetectorVerdict | None]] = field(default_factory=list)


@dataclass(frozen=True)
class ItemResult:
    item: CorpusItem
    findings: list[Finding]
    matched: list[tuple[Expectation, Finding]]
    missing: list[Expectation]
    #: 标了"不许报"却报了 fail/warning 的结论。**这是最重要的一栏**——
    #: 精确率就是靠它压出来的。
    forbidden: list[Finding]
    uncovered: list[Finding]
    #: 命中了 `absent` 的结论：明说了不许出现，还是出现了。
    violated_absent: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing and not self.forbidden and not self.violated_absent

    def fired_families(self) -> set[str]:
        return {
            family_of(f.detector)
            for f in self.findings
            if f.verdict in (DetectorVerdict.FAIL, DetectorVerdict.WARNING)
        }


def family_of(detector: str) -> str:
    """`loop.strict` → `loop`。族是按检测器名的前缀切的。"""
    return detector.split(".", 1)[0]


def _require_mapping(value: Any, what: str, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CorpusError(f"{path}: {what}必须是一个对象，收到 {type(value).__name__}")
    return value


def _reject_unknown(keys: Any, allowed: set[str] | frozenset[str], what: str, path: str) -> None:
    unknown = sorted(set(keys) - set(allowed))
    if unknown:
        raise CorpusError(
            f"{path}: {what}里有不认识的字段 {unknown}。"
            f"支持的只有 {sorted(allowed)}。\n"
            f"拼错的字段会被静默忽略，而那条约束于是从来没被检查过——"
            f"所以这里直接让加载失败。"
        )


def _parse_memory(
    raw: Any, path: str
) -> tuple[list[MemoryObservation], MemoryRules]:
    """把 `memory` 解析成观测 + 判据。

    观测的 `entries` 写成 `{路径: 内容}`，**摘要由加载器算**——和 `calls` 的
    `seq`、指纹同一个理由：手写摘要就会手写错，而错的摘要在报告里表现为
    "内容变了"，指向一个根本没发生过的问题。

    判据部分（`must_hold` / `forbidden`）与场景 YAML 是同一组字段，两处各写一遍
    校验的话迟早会松掉一处，所以两处都从 `scenario.loader` 的同一套规矩来：
    空路径、空标记、缺理由都在加载时报错。
    """
    if raw is None:
        return [], MemoryRules()
    mapping = _require_mapping(raw, "memory", path)
    _reject_unknown(mapping, MEMORY_KEYS, "memory", path)

    observations: list[MemoryObservation] = []
    raw_observations = _require_list(mapping.get("observations") or [], "observations", path)
    for i, entry in enumerate(raw_observations):
        where = f"{path}: memory.observations[{i}]"
        obs = _require_mapping(entry, "记忆观测", where)
        _reject_unknown(obs, OBSERVATION_KEYS, "记忆观测", where)
        if "ordinal" not in obs:
            raise CorpusError(f"{where} 缺 ordinal。观测点是按 (会话, 边界) 排序的。")
        phase = str(obs.get("phase", ""))
        if phase not in (PHASE_START, PHASE_END):
            raise CorpusError(
                f"{where}: phase 只能是 {PHASE_START!r} 或 {PHASE_END!r}，收到 {phase!r}。"
                f"边界名写错的后果是这一条被排到时间线的另一端，"
                f"漂移会指向相反的方向，而报告看起来仍然正常。"
            )
        contents = _require_mapping(obs.get("entries") or {}, "entries", where)
        observations.append(
            MemoryObservation(
                ordinal=int(obs["ordinal"]),
                phase=phase,
                # 不写 = 不知道（不是"自然延续"）。**不许猜**：视野决定了
                # "键不见了"该算回滚还是算丢失，猜一个数就等于替标注人做归因。
                horizon=None if obs.get("horizon") is None else int(obs["horizon"]),
                entries=tuple(
                    MemoryEntry(
                        path=str(name),
                        sha256=hashlib.sha256(str(text).encode("utf-8")).hexdigest(),
                        text=str(text),
                    )
                    for name, text in contents.items()
                ),
            )
        )

    must_hold: list[MustHold] = []
    for i, entry in enumerate(_require_list(mapping.get("must_hold") or [], "must_hold", path)):
        where = f"{path}: memory.must_hold[{i}]"
        rule = _require_mapping(entry, "记忆判据", where)
        _reject_unknown(rule, MUST_HOLD_KEYS, "记忆判据", where)
        target = str(rule.get("path", ""))
        if not target.strip():
            raise CorpusError(f"{where} 的 path 是空的，它匹配不到任何东西。")
        reason = str(rule.get("reason", ""))
        if not reason.strip():
            raise CorpusError(f"{where}（{target}）缺 reason。没有理由的判据无法复核。")
        must_hold.append(MustHold(path=target, reason=reason))

    forbidden: list[ForbiddenText] = []
    for i, entry in enumerate(_require_list(mapping.get("forbidden") or [], "forbidden", path)):
        where = f"{path}: memory.forbidden[{i}]"
        rule = _require_mapping(entry, "禁用标记", where)
        _reject_unknown(rule, FORBIDDEN_KEYS, "禁用标记", where)
        text = str(rule.get("text", ""))
        if not text.strip():
            # 空串是任何内容的子串，会命中每一个条目——报一条满屏的中毒结论，
            # 把真正的问题淹在里面。
            raise CorpusError(
                f"{where} 的 text 是空的。空文本是任何内容的子串，"
                f"它会命中记忆里的每一个条目。"
            )
        reason = str(rule.get("reason", ""))
        if not reason.strip():
            raise CorpusError(f"{where}（{text!r}）缺 reason。")
        forbidden.append(ForbiddenText(text=text, reason=reason))

    return observations, MemoryRules(must_hold=tuple(must_hold), forbidden=tuple(forbidden))


def _parse_expectations(
    raw: Any, path: str, item_id: str
) -> tuple[list[Expectation], dict[str, bool]]:
    """把 `expect` 解析成逐条预期 + 每族的"该不该报"。

    每一族都必须在 `expect` 里出现（可以写在族名上，也可以写在子名字上）。
    缺一族的后果是那族的数据不进精确率——**覆盖率会悄悄下降，而结果看起来
    完好**，这正是本项目一再要防的那种失效。
    """
    mapping = _require_mapping(raw, "expect", path)
    expectations: list[Expectation] = []
    family_labels: dict[str, bool] = {}
    for key, value in mapping.items():
        if key not in KNOWN_DETECTORS:
            raise CorpusError(
                f"{path}: expect 里的 {key!r} 不是一个已知的检测器。"
                f"已知的有 {sorted(KNOWN_DETECTORS)}。\n"
                f"写错名字的后果是这条预期永远'未命中'，而它看起来和检测器漏报一模一样。"
            )
        family = family_of(key)
        if group := _verdict_of(value, path, key):
            expectations.append(
                Expectation(
                    detector=key,
                    verdict=DetectorVerdict(group["verdict"]),
                    reason_code=group.get("reason_code"),
                    note=group.get("note", ""),
                )
            )
        # 判 `is not None` 而不是真假：`_verdict_of` 要么给 None，要么给一个
        # **一定带 verdict 的**字典（写了对象却没写 verdict 它当场就抛了），
        # 所以空字典到不了这里。写成 `bool(group)` 的话，万一哪天它真返回了 `{}`，
        # 这一行会安静地算成"这一族没张口"——**统计里少一条样本，而结果看起来完好**。
        fires = group is not None and DetectorVerdict(group["verdict"]) in (
            DetectorVerdict.FAIL,
            DetectorVerdict.WARNING,
        )
        # 族级标注取**或**，不是"必须一致"。同一族的不同层完全可以合理地一报
        # 一不报：一段轨迹的跨会话层报 warning 而严格层确是干净的（所以另有一条
        # `loop: pass`），两句话同时为真。取或之后，族级统计问的还是那个问题——
        # "这一族在这条轨迹上张口了吗"。
        family_labels[family] = family_labels.get(family, False) or fires
    missing = [f for f in FAMILIES if f not in family_labels]
    if missing:
        raise CorpusError(
            f"{path}（{item_id}）: 没有对检测器族 {missing} 表态。\n"
            f"每条轨迹都要说清每一族该报还是不该报——少一族就等于那族的样本"
            f"没进统计，而结果看起来完好。不关心的写法是 `{NONE}`。"
        )
    return expectations, family_labels


def _verdict_of(value: Any, path: str, key: str) -> dict[str, Any] | None:
    if value is None or value == NONE:
        return None
    if isinstance(value, str):
        return {"verdict": value}
    if isinstance(value, dict):
        _reject_unknown(value, {"verdict", "reason_code", "note"}, f"expect.{key}", path)
        if "verdict" not in value:
            raise CorpusError(f"{path}: expect.{key} 写了对象却没写 verdict。")
        return dict(value)
    raise CorpusError(f"{path}: expect.{key} 的写法不认识：{value!r}。")


def _parse_absent(raw: Any, path: str) -> list[tuple[str, DetectorVerdict | None]]:
    out: list[tuple[str, DetectorVerdict | None]] = []
    for i, entry in enumerate(_require_list(raw or [], "absent", path)):
        where = f"{path}: absent[{i}]"
        mapping = _require_mapping(entry, "不许出现的结论", where)
        _reject_unknown(mapping, {"detector", "verdict"}, "不许出现的结论", where)
        detector = str(mapping.get("detector", ""))
        if detector not in KNOWN_DETECTORS:
            raise CorpusError(f"{where}: 未知检测器 {detector!r}，这条禁用永远不会生效。")
        verdict = None
        if "verdict" in mapping:
            try:
                verdict = DetectorVerdict(str(mapping["verdict"]))
            except ValueError as exc:
                raise CorpusError(f"{where}: 未知结论 {mapping['verdict']!r}。") from exc
        out.append((detector, verdict))
    return out


def _parse_sessions(raw: Any, path: str) -> list[SessionSpec]:
    sessions: list[SessionSpec] = []
    for i, entry in enumerate(_require_list(raw, "sessions", path)):
        where = f"{path}: sessions[{i}]"
        mapping = _require_mapping(entry, "会话", where)
        _reject_unknown(mapping, _SESSION_KEYS, "会话", where)
        if "ordinal" not in mapping:
            raise CorpusError(f"{where} 缺 ordinal。会话序号是跨会话结论的排序依据。")
        sessions.append(
            SessionSpec(
                ordinal=int(mapping["ordinal"]),
                name=str(mapping.get("name", "")),
                autonomous=bool(mapping.get("autonomous", False)),
                memory_horizon=(
                    None
                    if mapping.get("memory_horizon") is None
                    else int(mapping["memory_horizon"])
                ),
            )
        )
    if not sessions:
        raise CorpusError(f"{path}: sessions 不能为空。")
    ordinals = [s.ordinal for s in sessions]
    if len(set(ordinals)) != len(ordinals):
        raise CorpusError(f"{path}: 会话序号重复：{ordinals}。")
    return sessions


def _require_list(value: Any, what: str, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise CorpusError(f"{path}: {what} 必须是一个列表，收到 {type(value).__name__}")
    return value


def _parse_tools(raw: Any, path: str) -> dict[str, ToolFacts]:
    mapping = _require_mapping(raw, "tools", path)
    tools: dict[str, ToolFacts] = {}
    for name, decl in mapping.items():
        where = f"{path}: tools.{name}"
        decl = _require_mapping(decl, "工具声明", where)
        _reject_unknown(decl, _TOOL_KEYS, "工具声明", where)
        if "is_write" not in decl:
            # 与场景 DSL 同一条规矩：读写属性不给默认值。给了默认，忘记声明的
            # 写操作会静默变成只读，重复扣款检测根本不覆盖它。
            raise CorpusError(f"{where} 缺 is_write。读写属性没有默认值，必须写出来。")
        tools[name] = ToolFacts(
            name=name,
            is_write=bool(decl["is_write"]),
            enforces_idempotency=bool(decl.get("enforces_idempotency", False)),
            # 规则与场景 YAML 是同一份（`checked_identity_fields`）：语料要能
            # 复现"声明了主键"的场景形状，两处收的东西不一样的话，语料上量出来的
            # 精度就是关于另一套声明说的。
            identity_fields=checked_identity_fields(
                decl.get("identity_fields"), tool=name, is_write=bool(decl["is_write"])
            ),
        )
    return tools


def _parse_allowlist(raw: Any, path: str) -> frozenset[str]:
    mapping = _require_mapping(raw, "tools", path)
    # 只有**写**操作能声明授权：读不改变世界，允许声明只会让人以为自己限制了什么。
    for name, decl in mapping.items():
        if decl.get("may_run_unattended") and not decl.get("is_write"):
            raise CorpusError(
                f"{path}: 只读工具 {name} 声明了 may_run_unattended。"
                f"读不改变世界，不构成主动行为——一个不起作用的字段比没有这个字段更坏。"
            )
    return frozenset(name for name, decl in mapping.items() if decl.get("may_run_unattended"))


#: 语料里一条豁免规则能写的键。和场景 YAML 那边是**同一组**（`ExemptionDecl`
#: 的字段），有测试钉着两边不许漂移——两个前门写法不一致的话，规则只在一处生效，
#: 而两边的测试都是绿的。
EXEMPTION_KEYS: frozenset[str] = frozenset(
    {"detector", "tool", "reason", "match", "allow_write"}
)


def _parse_exemptions(raw: Any, path: str) -> ExemptionSet:
    rules: list[Exemption] = []
    for i, entry in enumerate(_require_list(raw or [], "exemptions", path)):
        where = f"{path}: exemptions[{i}]"
        mapping = _require_mapping(entry, "豁免规则", where)
        _reject_unknown(mapping, EXEMPTION_KEYS, "豁免规则", where)
        detector = str(mapping.get("detector", ""))
        if detector not in KNOWN_DETECTORS:
            raise CorpusError(f"{where}: 豁免指向未知检测器 {detector!r}，它永远不会命中。")
        reason = str(mapping.get("reason", "")).strip()
        if not reason:
            raise CorpusError(
                f"{where}: 豁免缺 reason。没有理由的豁免在报告里无法复核，等于永久静音。"
            )
        try:
            rules.append(
                Exemption(
                    detector=detector,
                    tool=mapping.get("tool"),
                    reason=reason,
                    match=dict(mapping.get("match", {})),
                    allow_write=bool(mapping.get("allow_write", False)),
                )
            )
        except ValueError as exc:
            # `Exemption.__post_init__` 是**唯一**的收口处，它拦的规矩（比如指向
            # 记忆族这种"不吃豁免"的检测器）必须原样变成加载期错误。漏了这一层，
            # 那些规则会在加载时通过、跑到检测器那里才炸，而语料加载期的报错是
            # 这个项目里唯一能让人看清自己写错了什么的地方。
            raise CorpusError(f"{where}: {exc}") from exc
    return ExemptionSet(rules)


def _parse_calls(
    raw: Any,
    path: str,
    item_id: str,
    tools: dict[str, ToolFacts],
    sessions: list[SessionSpec],
) -> tuple[list[CallView], dict[str, int]]:
    by_ordinal = {s.ordinal: s for s in sessions}
    calls: list[CallView] = []
    lines: dict[str, int] = {}
    for i, entry in enumerate(_require_list(raw, "calls", path)):
        where = f"{path}: calls[{i}]"
        mapping = _require_mapping(entry, "调用", where)
        _reject_unknown(mapping, _CALL_KEYS, "调用", where)

        tool = str(mapping.get("tool", ""))
        if tool not in tools:
            # 与场景的跨字段校验同一条：调一个没声明的工具多半是打错字，
            # 而不拦的后果是它的读写属性静默取默认值，判定从此建立在一个瞎猜上。
            raise CorpusError(
                f"{where}: 调用了没有声明的工具 {tool!r}（已声明：{sorted(tools)}）。"
            )
        ordinal = int(mapping.get("session", 0))
        if ordinal not in by_ordinal:
            raise CorpusError(
                f"{where}: 引用了没有声明的会话 #{ordinal}（已声明：{sorted(by_ordinal)}）。"
            )
        session = by_ordinal[ordinal]
        args = dict(mapping.get("args") or {})
        # `status` 的取值受 `CALL_STATUSES` 管辖，**和边界那边是同一份名单**。
        # 原先这里谁都没核：写个错字（`erorr`）会静默落进 `reached_target` 的
        # "不在名单里就一律算没到"那一支，于是这次调用被判成**一次都没发生过**，
        # 不进任何判定——而"漏判"和"判对了"在报告上长得一模一样。
        #
        # 拦它的理由和上面拦"调了没声明的工具"是同一条：不拦的后果是判定从此
        # 建立在一个瞎猜上。区别只在错字落在键上还是落在值上。
        status = str(mapping.get("status", "ok"))
        if status not in CALL_STATUSES:
            raise CorpusError(
                f"{where}: status {status!r} 不在词表里（{sorted(CALL_STATUSES)}）。"
                f"词表外的取值会被 `reached_target` 判成「没到目标系统」，也就是这次调用"
                f"一次都没发生过——多半是打错字，而不是你真想表达那个意思。"
            )
        call_id = f"{item_id}#{i}"
        lines[call_id] = i
        calls.append(
            CallView(
                id=call_id,
                seq=i,
                tool_name=tool,
                args=args,
                # 指纹从**和生产同一处**算：语料里的"同一操作"必须和生产里的
                # "同一操作"是同一个判据，否则精确率是在说另一套东西。
                fingerprint=fingerprint(tool, normalize_args(args)),
                is_write=tools[tool].is_write,
                session_sequence_id=item_id,
                session_id=f"{item_id}#s{ordinal}",
                session_ordinal=ordinal,
                memory_horizon_ordinal=session.horizon(),
                idempotency_key=mapping.get("idempotency_key"),
                replay_reason=mapping.get("replay_reason"),
                blocked_by_proxy=bool(mapping.get("blocked_by_proxy", False)),
                status=str(mapping.get("status", "ok")),
            )
        )
    return calls, lines


def parse_item(raw: Any, path: str) -> CorpusItem:
    mapping = _require_mapping(raw, "语料", path)
    _reject_unknown(mapping, _ITEM_KEYS, "语料", path)

    item_id = str(mapping.get("id", "")).strip()
    if not item_id:
        raise CorpusError(f"{path}: 语料缺 id。")
    source = str(mapping.get("source", "")).strip()
    if source not in SOURCES:
        raise CorpusError(
            f"{path}（{item_id}）: 来源 {source!r} 不在白名单里（{sorted(SOURCES)}）。"
            f"来源是读报告的人判断这条语料可信度的依据。"
        )
    why = str(mapping.get("why", "")).strip()
    if not why:
        # 没有依据的标注只是一句断言。标注是可复核的，靠的就是这一段。
        raise CorpusError(f"{path}（{item_id}）: 缺 why——这条轨迹凭什么该判成这个结论？")

    tools = _parse_tools(mapping.get("tools"), path)
    sessions = _parse_sessions(mapping.get("sessions"), path)
    calls, lines = _parse_calls(mapping.get("calls"), path, item_id, tools, sessions)
    expectations, labels = _parse_expectations(mapping.get("expect"), path, item_id)
    memories, memory_rules = _parse_memory(mapping.get("memory"), path)

    return CorpusItem(
        id=item_id,
        source=source,
        why=why,
        path=path,
        calls=calls,
        tool_facts=tools,
        unattended_allowlist=_parse_allowlist(mapping.get("tools"), path),
        autonomous_ordinals=frozenset(s.ordinal for s in sessions if s.autonomous),
        exemptions=_parse_exemptions(mapping.get("exemptions"), path),
        memories=memories,
        memory_rules=memory_rules,
        expect=expectations,
        family_labels=labels,
        source_lines=lines,
        absent=_parse_absent(mapping.get("absent"), path),
    )


def load_corpus(root: str | Path) -> list[CorpusItem]:
    """读一个目录下所有 `*.yaml`。文件之间用 id 去重——重名的两条语料会让
    统计里少一条，而结果看起来完好。"""
    root = Path(root)
    if not root.is_dir():
        raise CorpusError(f"语料目录不存在：{root}")
    items: list[CorpusItem] = []
    seen: dict[str, str] = {}
    for path in sorted(root.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        payload = yaml.safe_load(text)
        if payload is None:
            continue
        for entry in _require_list(payload.get("items"), "items", path.name):
            item = parse_item(entry, path.name)
            if item.id in seen:
                raise CorpusError(f"语料 id 重复：{item.id}（{seen[item.id]} 与 {path.name}）")
            seen[item.id] = path.name
            items.append(item)
    return items


def run_item(item: CorpusItem) -> ItemResult:
    """跑检测器并比对标注。**用的是线上那一套检测器**（`detectors.run_all`）。"""
    findings = run_all(
        item.calls,
        tool_facts=item.tool_facts,
        unattended_allowlist=item.unattended_allowlist,
        autonomous_ordinals=item.autonomous_ordinals,
        exemptions=item.exemptions,
        memories=item.memories,
        memory_rules=item.memory_rules,
    )
    matched, missing, remaining = match_expectations(item.expect, findings)
    forbidden = [
        f
        for f in remaining
        if not item.family_labels.get(family_of(f.detector), False)
        and f.verdict in (DetectorVerdict.FAIL, DetectorVerdict.WARNING)
    ]
    uncovered = [f for f in remaining if f.verdict == DetectorVerdict.INCONCLUSIVE]
    violated_absent = [
        f
        for f in findings
        if any(
            f.detector == detector and (verdict is None or f.verdict == verdict)
            for detector, verdict in item.absent
        )
    ]
    return ItemResult(
        item=item,
        findings=findings,
        matched=matched,
        missing=missing,
        forbidden=forbidden,
        uncovered=uncovered,
        violated_absent=violated_absent,
    )


def evaluate_all(items: list[CorpusItem]) -> list[ItemResult]:
    return [run_item(item) for item in items]


@dataclass(frozen=True)
class FamilyScore:
    family: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float | None:
        return None if self.tp + self.fp == 0 else self.tp / (self.tp + self.fp)

    @property
    def recall(self) -> float | None:
        return None if self.tp + self.fn == 0 else self.tp / (self.tp + self.fn)


def score(results: list[ItemResult]) -> dict[str, FamilyScore]:
    """按检测器族算精确率/召回率。

    一条轨迹对一个族只有两种标注：该报（fail/warning）或不该报。
    于是四格是完整的，而**"这条路根本没走到"不是负样本**——那种情况在语料里
    由 `inconclusive` 标注，且不计入任何一格（既不算误报也不算漏报）。
    """
    counters: dict[str, dict[str, int]] = {
        f: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for f in FAMILIES
    }
    for result in results:
        fired = result.fired_families()
        for family in FAMILIES:
            should = result.item.family_labels.get(family)
            if should is None:  # 加载时已保证不会发生，防御性保留
                continue
            did = family in fired
            if should and did:
                counters[family]["tp"] += 1
            elif should and not did:
                counters[family]["fn"] += 1
            elif not should and did:
                counters[family]["fp"] += 1
            else:
                counters[family]["tn"] += 1
    return {f: FamilyScore(f, **c) for f, c in counters.items()}
