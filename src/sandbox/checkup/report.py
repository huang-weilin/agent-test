"""把 N 次运行的结论合成一页体检报告。**纯函数：不碰库、不碰磁盘、不起进程。**

输入是一串 `ItemRun`（每一项挂着一份 `report_json` 解析出来的字典），输出是
一页分好档的结论。分开的理由和 `regression/diff.py` 一样：这里全部的逻辑都是
"怎么把已经记下来的事换个读法"，那是可以脱离数据库测的问题。

## 这一层的全部风险：换个读法的时候，替结论说了它没说过的话

同一份运行结论，排成"体检报告"时要经过四次改口的机会，每一处都写明了为什么
这么选：

1. **运行级的「通过」在体检里不是好话。** 一支演"重复扣款"的场景跑出「通过」，
   意思是**预期达成**——检测器正常抓到了那笔重复付款。它和"Agent 没问题"是
   两件事（见 `scenario/expect.py`：`ok` 判的是"实际发生的事和场景说的对不
   对得上"）。所以这一层**不用 `ok` 给任何一项定档**，只把它原样带出去。

2. **「没抓到」不折成「没问题」。** 一项该抓到的没抓到（`missing`），这一层
   报成"这项没结论"，绝不报成"这项正常"。谁的责任（检测器漏报还是 Agent 这次
   没犯）在数据上分不清，报告就不许说（`docs/metrics.md` 的不支持清单里有这条）。

3. **「判不了」单独成档**，而且按检测器汇总计数，不逐条铺开——实测里
   `uncovered` 每次运行都非空（通常是 `loop.cross_session`、`autonomy`、
   `memory` 那几条：这几支场景里没有跨会话、没有无人值守会话，所以那几格
   本来就没东西可判）。逐条铺 6 遍会把整页淹掉，而淹掉之后它的读法就变成了
   "一堆废话"——那正是它想避免的。

4. **执行失败 / 前提不成立 / 地面真值对不上，各自成档**，不和"没抓到"合并。
   它们说的都是"这一次的结论不成立"，而成立与否和抓到没抓到是两回事。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from sandbox.checkup.catalog import Checkup, CheckupItem

#: 一项体检的档。**每一档的名字都是照着"这一项能得出什么结论"起的**，
#: 不按严重程度起——严重程度由显示顺序定，见 `CheckupReport` 上的几条属性。
ItemKind = Literal[
    "not_run",  # 作业层就失败了，连运行记录都没有
    "failed",  # 跑起来了，但执行失败（报告里有 error）
    "no_report",  # 有运行记录，但没有留下结论快照
    "premises",  # 前提不成立：这一次没按剧本走完，结论不指向被测对象
    "charges",  # 地面真值对不上：目标系统上的扣款笔数和场景说的不符
    "unreadable",  # 结论快照读不出来（字段形状不是我们写的那个样子）
    "no_verdict",  # 读得出来，但一条结论都没判出来（见 `_classify` 末尾）
    "missed",  # 该抓到的没抓到
    "false_alarm",  # 报了场景没让它报的
    "caught",  # 抓到了（预期里有一条 fail 命中）
    "noted",  # 只有 warning：不太对，但还不算错
    "clear",  # 这一项什么都没发生
]

#: 结论页上"一项都没抓到"时必须跟着出现的那句话。放在代码里而不是模板里，
#: 是为了让它可测：`tests/test_checkup.py` 钉着它。
NO_FINDING_CAVEAT = (
    "这一次没抓到，不等于它不会犯。这份检查只看得到它自己那几支剧本里发生过的事，"
    "剧本没演到的情形它一概不知道。"
)


class ReportShapeError(ValueError):
    """`report_json` 的某个字段不是我们写进去的那个形状。"""


@dataclass(frozen=True)
class ItemRun:
    """一项体检的原始材料。**还没判过。**"""

    item: CheckupItem
    run_id: str | None = None
    #: `report_json` 解析出来的字典。None = 没有结论快照（或还没跑完）。
    report: dict[str, Any] | None = None
    #: 执行层报的错。非空 = 这一次没跑成。
    error: str = ""


@dataclass(frozen=True)
class FindingLine:
    """一条检测结论的摘要，够在体检页上印一行。

    `count` 是**一字不差的同一条结论出现了几次**（见 `_merge_identical`）。
    默认 1 = 只出现一次。它必须跟着显示：不显示的话，"报了九次"和"报了一次"
    在页面上长得一模一样，而次数正是读的人判断严重程度的第一眼。
    """

    detector: str
    verdict: str
    summary: str
    reason_code: str | None = None
    count: int = 1


@dataclass(frozen=True)
class UnknownLine:
    """一条"判不了"。

    `declared` = 场景**事先声明过**这一条这次应当是判不了（它出现在
    `matched` 里，判定为 `inconclusive`）。False = 这条结论没被任何预期覆盖
    （`uncovered`）。两种都要报，但它们不是一回事：前者是设计，后者是覆盖
    面在悄悄变窄，所以页面上分开写。
    """

    detector: str
    summary: str
    declared: bool


@dataclass(frozen=True)
class ItemOutcome:
    """一项体检跑完之后，我们对它知道什么。"""

    item: CheckupItem
    kind: ItemKind
    #: 为什么是这一档。人话，直接印在页面上。
    note: str
    run_id: str | None = None
    #: 运行级的结论（`report["ok"]`）。**它不定档**，只供跳转过去的人对照。
    #: None = 没留存 / 没跑成。
    run_ok: bool | None = None
    error: str = ""
    caught: tuple[FindingLine, ...] = ()
    noted: tuple[FindingLine, ...] = ()
    unknown: tuple[UnknownLine, ...] = ()
    missing: tuple[FindingLine, ...] = ()
    unexpected: tuple[FindingLine, ...] = ()
    #: 命中的 pass 有几条。**只数个数**：它们是这支场景顺带做的检查，
    #: 不是这一项要回答的问题。
    fine: int = 0

    @property
    def is_incomplete(self) -> bool:
        """这一项有没有给出任何可用结论。"""
        return self.kind in (
            "not_run",
            "failed",
            "no_report",
            "premises",
            "charges",
            "unreadable",
            "no_verdict",
        )

    @property
    def has_instrument_problem(self) -> bool:
        """检查自己有没有出问题（该抓的没抓 / 不该报的报了）。"""
        return bool(self.missing) or bool(self.unexpected)


@dataclass(frozen=True)
class UnknownRollup:
    """"判不了"按检测器汇总的一行。"""

    detector: str
    count: int
    #: 其中场景事先声明过的条数。剩下的（`count - declared`）是没被任何预期
    #: 覆盖到的，那是覆盖面在变窄的信号。
    declared: int
    #: 结论原文，取第一条。
    example: str


@dataclass(frozen=True)
class CheckupReport:
    """一页体检报告。"""

    checkup: Checkup
    items: tuple[ItemOutcome, ...] = field(default_factory=tuple)

    # --- 分档（顺序 = 页面上该读的顺序） ----------------------------------

    @property
    def incomplete(self) -> tuple[ItemOutcome, ...]:
        """没得出可用结论的项。**排在最前面**：这一页有多少是空的，先得说。"""
        return tuple(i for i in self.items if i.is_incomplete)

    @property
    def instrument_problems(self) -> tuple[ItemOutcome, ...]:
        """检查自身有异常的项（该抓的没抓 / 不该报的报了）。

        排在被检对象的问题之前：这一页给出的**覆盖面**由它决定。少报意味着
        这一页上有几格其实是空的，多报意味着别的格子里可能有冤枉。
        """
        return tuple(i for i in self.items if i.has_instrument_problem)

    @property
    def findings(self) -> tuple[ItemOutcome, ...]:
        """抓到的（受检对象的问题）。"""
        return tuple(i for i in self.items if i.kind == "caught")

    @property
    def notes(self) -> tuple[ItemOutcome, ...]:
        """值得看一眼（只有 warning）。"""
        return tuple(i for i in self.items if i.kind == "noted")

    @property
    def clear(self) -> tuple[ItemOutcome, ...]:
        """这一项什么都没发生。"""
        return tuple(i for i in self.items if i.kind == "clear")

    @property
    def unknown(self) -> tuple[UnknownRollup, ...]:
        """判不了的，按检测器汇总，条数多的在前。"""
        counts: dict[str, list[UnknownLine]] = {}
        for outcome in self.items:
            for line in outcome.unknown:
                counts.setdefault(line.detector, []).append(line)
        rolls = [
            UnknownRollup(
                detector=detector,
                count=len(lines),
                declared=sum(1 for line in lines if line.declared),
                example=lines[0].summary,
            )
            for detector, lines in counts.items()
        ]
        # 条数相同时按名字排：**不让字典的插入顺序决定页面长什么样**。
        # 插入顺序跟着场景的加载顺序走，而那是可以变的——两次体检排出两种
        # 顺序，读的人会以为结论变了。
        return tuple(sorted(rolls, key=lambda r: (-r.count, r.detector)))

    # --- 顶栏那几句话 ----------------------------------------------------

    @property
    def done_count(self) -> int:
        return len(self.items) - len(self.incomplete)

    @property
    def headline(self) -> str:
        """一句话总结。**只陈述计数**，不替结论下判断——判断在下面各节里。"""
        parts = [f"{len(self.items)} 项里跑成了 {self.done_count} 项。"]
        if self.findings:
            parts.append(f"抓到 {len(self.findings)} 类问题。")
        if self.notes:
            parts.append(f"{len(self.notes)} 项要留意但不严重。")
        if not self.findings and not self.notes:
            parts.append("这一类毛病一项都没抓到。")
        if self.instrument_problems:
            parts.append(f"{len(self.instrument_problems)} 项的检查过程本身有异常，先看那一节。")
        if self.incomplete:
            parts.append(f"{len(self.incomplete)} 项没得出可用结论。")
        return "".join(parts)


# --- 读报告 ----------------------------------------------------------------


#: 合并"一字不差"时用的键：四个字段全等才算同一条。见 `_merge_identical`。
_LineKey = tuple[str, str, str, str | None]


def _merge_identical(lines: Iterable[FindingLine]) -> tuple[FindingLine, ...]:
    """把**一字不差**的几条结论合并成一条，带上条数。

    实测里 `schema_rejected` 那一项会印出三行一模一样的「Agent 调用 'list_orders'
    的参数不符合场景声明的 schema」——检测器对每一次被拒的调用各报一条，而这几条
    的正文恰好一个字都不差。三行同样的字不是信息，是噪音，而**噪音一多，真结论
    就没人看了**（同一条理由见"判不了"那一节的汇总）。

    合并的边界卡死在**四个字段全等**（检测器 / 结论 / 正文 / 理由码）：措辞上差
    一个字的都不并。宁可多印一行不同的，也不要把两条不同的结论读成一条——那是
    "报告说了比观测更多的话"的另一种写法。

    这是**计数**，不是改口：合并之后正文一个字不动，次数印在旁边。同一条做法
    见调用链（`docs/call-chain.md`：同样的失败合并成一句带次数）。
    """
    counts: dict[_LineKey, int] = {}
    for line in lines:
        key = (line.detector, line.verdict, line.summary, line.reason_code)
        counts[key] = counts.get(key, 0) + line.count
    return tuple(
        FindingLine(
            detector=key[0],
            verdict=key[1],
            summary=key[2],
            reason_code=key[3],
            count=count,
        )
        for key, count in counts.items()
    )


def _list_of(value: Any, where: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ReportShapeError(
            f"报告里的 {where} 应该是一串，实际是 {type(value).__name__}。"
        )
    return value


def _dicts(value: Any, where: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in _list_of(value, where):
        if not isinstance(entry, dict):
            raise ReportShapeError(
                f"报告里的 {where} 里有一项不是字典（{type(entry).__name__}）。"
            )
        out.append(entry)
    return out


def _text(value: Any) -> str:
    """报告里的字符串字段。认不出来的原样转成字符串，不猜。"""
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _line_of(entry: dict[str, Any], *, expect_key: str | None = None) -> FindingLine:
    """从 `expectation` 的一节里读出一条结论。

    `matched` 的形状和另外三节不同：它是 `{"expect": {...}, "got": "…"}`，
    检测器名在 `expect` 里、结论原文在 `got` 里。这个不对称是 `ExpectResult`
    自己定的（见 `scenario/expect.py` 的 `to_dict`），这里跟着它走，不另外
    记一份"哪一栏有检测器名"的对照表。
    """
    if expect_key is not None:
        head = entry.get(expect_key)
        head = head if isinstance(head, dict) else {}
        summary = _text(entry.get("got"))
    else:
        head = entry
        summary = _text(entry.get("summary"))
    reason = head.get("reason_code")
    return FindingLine(
        detector=_text(head.get("detector")),
        verdict=_text(head.get("verdict")),
        summary=summary,
        reason_code=reason if isinstance(reason, str) else None,
    )


def _classify(run: ItemRun) -> ItemOutcome:
    item = run.item

    if run.run_id is None:
        return ItemOutcome(
            item=item,
            kind="not_run",
            note="这一次根本没跑起来，所以这一项一个字的结论都没有。",
            error=run.error or "执行器在建立运行记录之前就失败了。",
        )
    if run.report is None:
        return ItemOutcome(
            item=item,
            kind="no_report",
            note="运行记录建起来了，但没留下结论快照——所以这一项没有可读的结论。"
            "这不是「没问题」。",
            run_id=run.run_id,
            error=run.error,
        )

    report = run.report
    ok = report.get("ok")
    run_ok = ok if isinstance(ok, bool) else None
    exec_error = report.get("error")
    exec_error_text = _text(exec_error)

    expectation = report.get("expectation")
    if exec_error_text:
        return ItemOutcome(
            item=item,
            kind="failed",
            note="这次运行执行失败。执行失败和「没通过」是两回事：前者说明这次测试"
            "根本没跑成，结论不指向被测对象。",
            run_id=run.run_id,
            run_ok=run_ok,
            error=exec_error_text,
        )

    if expectation is None:
        return ItemOutcome(
            item=item,
            kind="unreadable",
            note="这份结论快照里没有「预期比对」那一节，所以这一项没法读出结论。"
            "多半是它在那一节存在之前跑的。这不是「没问题」。",
            run_id=run.run_id,
            run_ok=run_ok,
        )

    if not isinstance(expectation, dict):
        raise ReportShapeError("报告的 expectation 不是字典。")

    matched = [
        _line_of(e, expect_key="expect")
        for e in _dicts(expectation.get("matched"), "expectation.matched")
    ]
    missing = [_line_of(e) for e in _dicts(expectation.get("missing"), "expectation.missing")]
    unexpected = [
        _line_of(e) for e in _dicts(expectation.get("unexpected"), "expectation.unexpected")
    ]
    uncovered = [
        _line_of(e) for e in _dicts(expectation.get("uncovered"), "expectation.uncovered")
    ]

    # 四个桶都过一遍合并：同一个检测器对第一次、第二次、第三次被拒的调用各报一条
    # 时，这几条的正文往往一个字都不差。合并的是**一字不差**的那些，条数印在旁边。
    caught = _merge_identical(f for f in matched if f.verdict == "fail")
    noted = _merge_identical(f for f in matched if f.verdict == "warning")
    fine = sum(1 for f in matched if f.verdict == "pass")
    # 判不了的两个来源合到一处，但**标着来路**：命中预期里的 inconclusive 是
    # 场景事先说好的，落进 uncovered 的那些是没人管过的。
    unknown = tuple(
        [UnknownLine(f.detector, f.summary, True) for f in matched if f.verdict == "inconclusive"]
        + [UnknownLine(f.detector, f.summary, False) for f in uncovered]
    )

    common: dict[str, Any] = {
        "item": item,
        "run_id": run.run_id,
        "run_ok": run_ok,
        "caught": caught,
        "noted": noted,
        "unknown": unknown,
        "missing": _merge_identical(missing),
        "unexpected": _merge_identical(unexpected),
        "fine": fine,
    }

    # 前提和地面真值排在"抓到没抓到"前面：**它们说的是"这一次的结论成不成立"**，
    # 而成立与否比结论内容更靠前。这两条一出现，报告自己就已经说了"结论不指向
    # 被测对象"（见 `scenario/runner.py` 的 `_render`），这里只是不让它在体检页上
    # 被埋掉。
    if report.get("premises_held") is False:
        return ItemOutcome(
            kind="premises",
            note="这一次没按剧本走完（脚本没被消费光，或者上游给的答复不是场景写的那一支）。"
            "所以这次运行得出的任何结论都不指向被测对象——既不指向它有问题，也不指向它没问题。",
            **common,
        )
    if report.get("charges_ok") is False:
        return ItemOutcome(
            kind="charges",
            note="目标系统上真实的写操作笔数和场景说的对不上。这是这套工具能给出的最强的一类"
            "证据（它数的是目标系统那边实际发生了什么，不是 Agent 说了什么），"
            "但「这是谁的责任」分不清：可能被测对象多做了一次，也可能是这次运行的布景有问题。",
            **common,
        )
    if missing:
        return ItemOutcome(
            kind="missed",
            note="这一项该抓到的东西没抓到。两种可能——检查自己漏了，或者被测对象这次没照剧本"
            "做——这次运行的数据分不出是哪一种。所以这一项没有结论：既不算过关，也不算没过关。",
            **common,
        )
    if unexpected:
        return ItemOutcome(
            kind="false_alarm",
            note="这一项报了场景没让它报的东西。这可能是检查自己的误报，也可能是被测对象真的"
            "做了一件场景没预料到的事——两种情况都值得看，但不能当成同一件事。",
            **common,
        )
    if caught:
        return ItemOutcome(
            kind="caught",
            note="这一项抓到了。往下看它是凭什么判的。",
            **common,
        )
    if noted:
        return ItemOutcome(
            kind="noted",
            note="这一项不算错，但有值得看一眼的地方。",
            **common,
        )
    # **`clear` 要求至少有一条 `pass` 命中。**
    #
    # 这一条是给一种很安静的失效兜底的：菜单里挂一支 `expect: []` 的场景，
    # 或者一支只声明了 `inconclusive` 的场景，跑完什么都不命中也不落空——
    # 落到"没事"那一档的话，页面会写着「这项做完了，没发现问题」，而它
    # **一条都没有比过**。这正是本项目最怕的形状：报告说了比观测更多的话。
    #
    # 所以"没判过"必须自己成一档，和"判过、没问题"分开。
    if fine == 0:
        extra = f"另外还有 {len(unknown)} 格这次判不了，列在最后那一节。" if unknown else ""
        return ItemOutcome(
            kind="no_verdict",
            note="这一项读得出来，但一条结论都没判出来——没有一条预期命中，也没有落空。"
            "这一项这次什么都没判。它不等于「没问题」。" + extra,
            **common,
        )
    return ItemOutcome(
        kind="clear",
        note=f"这一项做完了，没发现问题（{fine} 条预期全部命中）。",
        **common,
    )


def build(checkup: Checkup, runs: list[ItemRun]) -> CheckupReport:
    """把每一项的原始材料合成一页报告。

    一项读不出来（结论快照的形状不是我们写的那种）**不会让整页打不开**：
    它自己变成"读不出来"那一档，其余各项照常。整页 500 的话，用户看到的是
    一片错误页，而问题只是一份旧格式的快照。
    """
    outcomes: list[ItemOutcome] = []
    for run in runs:
        try:
            outcomes.append(_classify(run))
        except ReportShapeError as exc:
            outcomes.append(
                ItemOutcome(
                    item=run.item,
                    kind="unreadable",
                    note="这份结论快照读不出来，这一项没有结论。这不是「没问题」。",
                    run_id=run.run_id,
                    error=str(exc),
                )
            )
    return CheckupReport(checkup=checkup, items=tuple(outcomes))
