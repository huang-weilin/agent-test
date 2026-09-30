"""回归对比：先说清两次运行**能不能比**，再说差在哪。

需求要的是"回归集管理"——把一次运行钉成基线，改完东西再跑，看结论变了没有。
这个模块存在的理由是其中最容易做错的一点：

**"有差异"和"差异算不算数"是两件事。**

两次运行的结论差了三处，能不能就此说"Agent 退化了"？只取决于一件事：**配置
动了没有**。配置没动，差异就是行为差异；配置动了（改了 Prompt、换了工具），
差异可能是改配置改出来的——那时候报"回归"，用户会去查一个他刚刚亲手改过的
东西。把这两件事揉成一句"发现 3 处差异"，用户只能自己猜哪几处算数。

所以 `Comparison` 里，**可比性**和**差异清单**是两个并列的字段，谁都替代不了谁。

还有一条同样重要：**测试定义变了**（`script` / `expect`）不是"配置变更"的一种，
它比配置变更更严重——那意味着两次运行比的是**两套测试**，结论差异什么都说明
不了。所以它是单独的一种可比性，不是"配置变了"的一个子类。

**为什么不做执行轨迹的向量化相似度。** 那会得到一个 0~1 的数字，而它回答不了
这个工具真正要回答的问题——"这次能不能判"。能判与不能判是**结论**上的区别，
不是相似度上的区别：轨迹相似度 0.97 的那次可能恰恰是钱多扣了一笔。这里做的是
**结论差分**：拿两个已经是结论的东西逐项对，对出来的每一处都指向一个具体的
事实（哪条检测结论翻了、钱扣了几次、前提成不成立）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sandbox.scenario import identity

# --- 可比性 ------------------------------------------------------------------
#: 配置一致，结论差异就是行为差异。
COMPARABLE = "comparable"
#: 配置变了（且不是测试定义）。
CONFIG_CHANGED = "config_changed"
#: **测试本身**变了：脚本或判卷标准。比的是两套测试。
TEST_CHANGED = "test_changed"
#: 没有依据判断配置动没动（本字段存在之前的运行、口径不同、没留下配置内容）。
UNKNOWN = "unknown"

# --- 结论走向 ----------------------------------------------------------------
#: 通过 → 不通过。
REGRESSED = "regressed"
#: 不通过 → 通过。
FIXED = "fixed"
SAME = "same"
#: 比不出来（不可比、有一侧没结论、或两次的前提不同）。
INDETERMINATE = "indeterminate"

# --- 差异的种类（机器可读） ---------------------------------------------------
#: 整体判定翻转。
KIND_VERDICT = "verdict"
#: 执行本身失败（崩了/超时），与"跑了但不通过"是两回事。
KIND_ERROR = "error"
#: 前提是否成立——"什么都没测到"必须与"表现得很好"分开。
KIND_PREMISE = "premise"
#: 某条检测结论变了。
KIND_FINDING = "finding"
#: 预期对上了没有（缺了哪条、多了哪条）。
KIND_EXPECTATION = "expectation"
#: 目标系统侧真实扣款笔数。**地面真值**，与检测结论互相独立。
KIND_CHARGES = "charges"
#: 某个会话的记忆视野变了（回滚位置变了）。
KIND_MEMORY_HORIZON = "memory_horizon"
#: 会话增删。
KIND_SESSION = "session"
#: 成本。信息，不是判定——花了多少钱是事实陈述，不是行为有没有缺陷的结论。
KIND_COST = "cost"

#: 差异分组的显示名。顺序就是它在页面上的顺序——**判定的排在最前**，
#: 因为那是用户打开这一页要看的东西；成本和会话排最后，它们是背景信息。
KIND_LABELS: dict[str, str] = {
    KIND_ERROR: "执行",
    KIND_VERDICT: "整体判定",
    KIND_PREMISE: "前提",
    KIND_FINDING: "检测结论",
    KIND_EXPECTATION: "预期比对",
    KIND_CHARGES: "目标系统（地面真值）",
    KIND_SESSION: "会话结构",
    KIND_MEMORY_HORIZON: "记忆视野",
    KIND_COST: "成本",
}

#: 分组的展示顺序。不属于任何一组的一律排在最后，而不是消失——
#: 一个"未知种类"的差异被静默丢掉，等于对比结果少了一行而没人知道。
KIND_ORDER: tuple[str, ...] = (
    KIND_ERROR,
    KIND_VERDICT,
    KIND_PREMISE,
    KIND_FINDING,
    KIND_EXPECTATION,
    KIND_CHARGES,
    KIND_SESSION,
    KIND_MEMORY_HORIZON,
    KIND_COST,
)

_VERDICT_LABEL = {True: "通过", False: "不通过", None: "未留存"}


def _label(ok: bool | None) -> str:
    return _VERDICT_LABEL[ok]


def _num(value: Any) -> str:
    """扣款笔数的显示。`-1` 不是"负一笔"，是**查不到**。

    把查不到印成 `-1` 会引出一个不存在的问题（"为什么是负的"），印成 `0`
    更糟——那是替一次没测到的运行断言"没扣钱"。
    """
    if value is None:
        return "未声明"
    if isinstance(value, int) and value < 0:
        return "查不到"
    return str(value)


@dataclass(frozen=True)
class Snapshot:
    """参与比较的一侧：一次运行的结论 + 配置 + 环境。

    TestRun 和 RegressionCase 都能造出一个来。**比较逻辑只认这个类型**，
    所以"基线"和"这次跑的"走的是同一条代码路径——两者若有各自的比较实现，
    迟早会漂移成两套判据。
    """

    run_id: str
    verdict_ok: bool | None
    report: dict[str, Any] | None
    config_scheme: str | None
    config_hash: str | None
    config_parts: dict[str, Any] | None
    env: dict[str, Any] | None

    @property
    def resolved_ok(self) -> bool | None:
        """这次运行到底通过没有。

        结论列优先；列里没存（本字段存在之前的运行）时退回报告里的 `ok`。
        两边都没有就是 `None`——**没记录**，不是"不通过"。
        """
        if self.verdict_ok is not None:
            return self.verdict_ok
        if self.report:
            ok = self.report.get("ok")
            if isinstance(ok, bool):
                return ok
        return None

    @property
    def premises_held(self) -> bool | None:
        if not self.report:
            return None
        value = self.report.get("premises_held")
        return value if isinstance(value, bool) else None

    @property
    def sessions(self) -> list[dict[str, Any]]:
        if not self.report:
            return []
        sessions = self.report.get("sessions")
        return [s for s in sessions if isinstance(s, dict)] if isinstance(sessions, list) else []

    @property
    def findings(self) -> dict[str, list[str]]:
        """检测器 → 它给出的结论（四态）。同一检测器可能给多条。"""
        if not self.report:
            return {}
        findings = self.report.get("findings")
        if not isinstance(findings, list):
            return {}
        out: dict[str, list[str]] = {}
        for f in findings:
            if not isinstance(f, dict):
                continue
            out.setdefault(str(f.get("detector", "?")), []).append(str(f.get("verdict", "?")))
        return {k: sorted(v) for k, v in out.items()}

    @property
    def expectation(self) -> dict[str, Any]:
        exp = (self.report or {}).get("expectation")
        return exp if isinstance(exp, dict) else {}


@dataclass(frozen=True)
class Difference:
    """一处具体的差异。**只陈述事实，不说谁对谁错。**

    `kind` 是机器可读的，界面按它分组；`subject` 是人看的定位——哪个检测器、
    哪次会话、还是目标系统。
    """

    kind: str
    subject: str
    before: str
    after: str

    @property
    def sentence(self) -> str:
        return f"{self.subject}：{self.before} → {self.after}"


@dataclass(frozen=True)
class Comparison:
    """一次回归对比的完整结论。"""

    comparability: str
    outcome: str
    headline: str
    changed_sections: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    differences: list[Difference] = field(default_factory=list)

    @property
    def counts(self) -> bool:
        """这次比较**算不算数**。

        界面靠它决定差异清单怎么显示：不算数时仍然列出来（人还是想看），
        但必须写明"这些差异可能只是改配置带来的"，而不是让一个红色清单
        自己说话。
        """
        return self.comparability == COMPARABLE

    def grouped(self) -> list[tuple[str, list[Difference]]]:
        """按种类分组，顺序由 `KIND_ORDER` 定。

        分组在**这里**做而不是在模板里：Jinja 里做分组要靠 `groupby` 加排序，
        而"顺序固定"这件事是结论的一部分（判定在前、背景信息在后），
        不该散在模板里。
        """
        order = {kind: n for n, kind in enumerate(KIND_ORDER)}
        buckets: dict[str, list[Difference]] = {}
        for d in self.differences:
            buckets.setdefault(d.kind, []).append(d)
        return sorted(buckets.items(), key=lambda kv: order.get(kv[0], len(order)))


def _section_labels(names: list[str]) -> str:
    return "、".join(identity.SECTION_LABELS.get(n, n) for n in names)


def comparability(baseline: Snapshot, current: Snapshot) -> tuple[str, list[str], list[str]]:
    """两次运行的结果能不能放在一起比。返回 (可比性, 变化了的节, 理由)。

    判据只有一条主线：**配置内容比对**。`config_scheme` 不是装饰——两种口径
    算出的哈希不相等，如果照着哈希判"配置变了"，一次旧运行和一次新运行放在
    一起就会得出一句基于不可比数据的结论。所以口径不同直接判不可比。
    """
    if baseline.config_scheme is None or current.config_scheme is None:
        return UNKNOWN, [], [
            "有一侧没留下配置口径——它跑在这个字段存在之前，"
            "当时的指纹只覆盖场景 id 与模型名，改了脚本或 Prompt 都不变。"
        ]
    if baseline.config_scheme != current.config_scheme:
        return UNKNOWN, [], [
            f"两次运行的配置口径不同（{baseline.config_scheme} 与 "
            f"{current.config_scheme}），指纹不可比。"
        ]

    before, after = baseline.config_parts, current.config_parts
    if not before or not after:
        # 只有一侧留下配置内容：**不能假设配置没变**。空配置的运行（测试、
        # 探活）和一次真运行之间没有可比性可言。
        if before == after:
            return COMPARABLE, [], []
        return UNKNOWN, [], [
            "有一侧没留下配置内容（那次运行没有带配置），无从判断配置动没动。"
        ]

    changed = identity.changed_sections(before, after)
    if not changed:
        return COMPARABLE, [], []

    labels = _section_labels(changed)
    if identity.TEST_SECTIONS.intersection(changed):
        return TEST_CHANGED, changed, [
            f"改的是**测试本身**（{labels}）。两次运行比的不是同一套测试，"
            f"结论差异说明不了被测对象的问题。"
        ]
    return CONFIG_CHANGED, changed, [
        f"配置变了：{labels}。结论差异可能就是这么来的，不能据此说行为回归。"
    ]


def _compare_verdict(baseline: Snapshot, current: Snapshot) -> list[Difference]:
    before, after = baseline.resolved_ok, current.resolved_ok
    if before == after:
        return []
    return [Difference(KIND_VERDICT, "整体判定", _label(before), _label(after))]


def _compare_error(baseline: Snapshot, current: Snapshot) -> list[Difference]:
    before = (baseline.report or {}).get("error")
    after = (current.report or {}).get("error")
    if before == after:
        return []
    return [
        Difference(
            KIND_ERROR,
            "执行",
            str(before) if before else "正常结束",
            str(after) if after else "正常结束",
        )
    ]


def _compare_premises(baseline: Snapshot, current: Snapshot) -> list[Difference]:
    """逐会话比前提。

    前提不成立意味着**什么都没测到**。它从 false 变成 true 不是"行为变好了"，
    是"这次终于测到了"——所以界面和结论都必须把它和检测结论分开。
    """
    diffs: list[Difference] = []
    for name, before, after in _zip_sessions(baseline, current):
        if before is None or after is None:
            continue
        b, a = before.get("premise_held"), after.get("premise_held")
        if b == a:
            continue
        diffs.append(
            Difference(
                KIND_PREMISE,
                f"{name}的前提",
                "成立" if b else "不成立（脚本没走完）",
                "成立" if a else "不成立（脚本没走完）",
            )
        )
    return diffs


def _zip_sessions(
    baseline: Snapshot, current: Snapshot
) -> list[tuple[str, dict[str, Any] | None, dict[str, Any] | None]]:
    """按会话序号对齐两次运行的会话。缺的一侧给 None，**不是跳过**。

    按序号而不是按名字：名字是场景文件里的字符串，改个名不该被读成"换了一个
    会话"。序号才是会话序列的位置。
    """
    b = {int(s.get("ordinal", 0)): s for s in baseline.sessions}
    c = {int(s.get("ordinal", 0)): s for s in current.sessions}
    out = []
    for ordinal in sorted(set(b) | set(c)):
        name = f"会话 #{ordinal}"
        out.append((name, b.get(ordinal), c.get(ordinal)))
    return out


def _compare_sessions(baseline: Snapshot, current: Snapshot) -> list[Difference]:
    diffs: list[Difference] = []
    for name, before, after in _zip_sessions(baseline, current):
        if before is not None and after is not None:
            continue
        diffs.append(
            Difference(
                KIND_SESSION,
                name,
                "存在" if before is not None else "没有这次会话",
                "存在" if after is not None else "没有这次会话",
            )
        )
    return diffs


def _compare_memory_horizons(baseline: Snapshot, current: Snapshot) -> list[Difference]:
    """比记忆视野。

    视野变了意味着**回滚位置变了**，也就是测试里"让它忘掉到哪一刻"的旋钮
    被拧过。这属于测试定义变更，所以它同时也是"这次比较不作数"的一条证据。
    界面要说清这一点，否则用户会把这行读成"Agent 的记忆出了问题"。
    """
    diffs: list[Difference] = []
    for name, before, after in _zip_sessions(baseline, current):
        if before is None or after is None:
            continue
        b, a = before.get("memory_horizon_ordinal"), after.get("memory_horizon_ordinal")
        if b == a:
            continue
        diffs.append(
            Difference(KIND_MEMORY_HORIZON, f"{name}的记忆视野", _horizon(b), _horizon(a))
        )
    return diffs


def _horizon(value: Any) -> str:
    """视野的显示。`-1` 表示"什么都记不得"，说成"覆盖到 -1"没人看得懂。"""
    if value is None:
        return "未记录"
    return "什么都记不得" if int(value) < 0 else f"覆盖到第 {int(value)} 次会话"


def _compare_findings(baseline: Snapshot, current: Snapshot) -> list[Difference]:
    """逐个检测器比结论。

    **只比四态结论，不比结论文案。** 文案是给人读的句子，改个措辞不该被报成
    行为差异——那样的假阳性会让这份清单很快没人看。代价是"同一个 fail、
    但细节分量不同"看不出来；那件事由地面真值那一行补上（钱扣了几次是数字，
    不是文案）。
    """
    diffs: list[Difference] = []
    b, c = baseline.findings, current.findings
    for detector in sorted(set(b) | set(c)):
        before, after = b.get(detector), c.get(detector)
        if before == after:
            continue
        diffs.append(
            Difference(KIND_FINDING, detector, _verdicts(before), _verdicts(after))
        )
    return diffs


def _verdicts(values: list[str] | None) -> str:
    if not values:
        return "没给出结论"
    return "、".join(values)


def _compare_expectations(baseline: Snapshot, current: Snapshot) -> list[Difference]:
    """比预期对上了没有：缺了哪条、多了哪条。

    这是"这次检测到底证明了什么"的直接对照。一条预期从对不上变成对上了，
    和从对上了变成对不上，指向的动作完全不同。
    """

    def keys(exp: dict[str, Any], field_name: str) -> list[str]:
        """`detector/verdict` 的集合。

        两个字段在两处的形状不完全一样（`missing` 是期望本身的 dump，
        `unexpected` 是检测结论的摘要），但这两项都有，所以取法相同。
        """
        items = exp.get(field_name)
        if not isinstance(items, list):
            return []
        return sorted(
            f"{item.get('detector')}/{item.get('verdict')}"
            for item in items
            if isinstance(item, dict)
        )

    b_missing = keys(baseline.expectation, "missing")
    c_missing = keys(current.expectation, "missing")
    b_unexpected = keys(baseline.expectation, "unexpected")
    c_unexpected = keys(current.expectation, "unexpected")

    diffs: list[Difference] = []
    if b_missing != c_missing:
        diffs.append(
            Difference(KIND_EXPECTATION, "没有对上的预期", _list(b_missing), _list(c_missing))
        )
    if b_unexpected != c_unexpected:
        diffs.append(
            Difference(KIND_EXPECTATION, "多出来的失败", _list(b_unexpected), _list(c_unexpected))
        )
    return diffs


def _list(values: list[str]) -> str:
    return "、".join(values) if values else "无"


def _compare_charges(baseline: Snapshot, current: Snapshot) -> list[Difference]:
    """比目标系统侧的真实扣款笔数。**这一行是本工具的核心断言。**

    检测器完全可以报出漂亮的 pass 而钱其实被扣了两次。所以这一行独立于
    检测结论单独比，而且"查不到"必须与"0 笔"分开说。
    """
    diffs: list[Difference] = []
    b_report, c_report = baseline.report or {}, current.report or {}
    b_egress, c_egress = b_report.get("egress_charges"), c_report.get("egress_charges")
    if b_egress != c_egress:
        diffs.append(
            Difference(
                KIND_CHARGES,
                "目标系统真实扣款",
                _num(b_egress),
                _num(c_egress),
            )
        )
    b_expected = b_report.get("expected_charges")
    c_expected = c_report.get("expected_charges")
    if b_expected != c_expected:
        # 期望值变了是**测试定义**变更的一个侧面，但它值得单独一行：用户可能
        # 只改了这一个数字，而它直接决定了"通过"的判据。
        diffs.append(
            Difference(
                KIND_CHARGES,
                "期望扣款笔数",
                _num(b_expected),
                _num(c_expected),
            )
        )
    return diffs


def _compare_cost(baseline: Snapshot, current: Snapshot) -> list[Difference]:
    """比成本。**信息，不是判定。**

    花了多少钱是事实陈述。成本变了不构成"行为回归"——把它算进判定，会让
    一个完全正常的运行因为多调了一次模型而变红。
    """
    b = (baseline.report or {}).get("cost") or {}
    c = (current.report or {}).get("cost") or {}
    b_usd, c_usd = b.get("total_usd"), c.get("total_usd")
    if b_usd is None and c_usd is None:
        return []
    if b_usd == c_usd:
        return []
    return [
        Difference(KIND_COST, "成本", f"${_money(b_usd)}", f"${_money(c_usd)}")
    ]


def _money(value: Any) -> str:
    return "未记录" if value is None else f"{float(value):.4f}"


def _outcome(
    baseline: Snapshot, current: Snapshot, comparability_: str
) -> tuple[str, str]:
    """结论走向 + 一句话理由。"""
    if comparability_ != COMPARABLE:
        return INDETERMINATE, ""
    before, after = baseline.resolved_ok, current.resolved_ok
    if before is None or after is None:
        return INDETERMINATE, "有一侧没留下结论，比不出走向。"
    if baseline.premises_held != current.premises_held:
        # 前提不同则"通过与否"根本不是同一个问题的答案。基线那次可能压根
        # 什么都没测到，于是它的"不通过"和这次的"通过"之间没有可比性——
        # 报成"已修复"是在说一件没发生过的事。
        return INDETERMINATE, (
            "两次运行的前提不同（有一次压根没测到），通过与否不是同一个问题的答案。"
        )
    if before == after:
        return SAME, ""
    return (REGRESSED if before else FIXED), ""


def _headline(
    comparability_: str, outcome: str, changed: list[str], notes: list[str]
) -> str:
    if comparability_ == TEST_CHANGED:
        return f"测试定义变了（{_section_labels(changed)}），本次比较不作数。"
    if comparability_ == CONFIG_CHANGED:
        return f"配置变了（{_section_labels(changed)}），本次比较不作数。"
    if comparability_ == UNKNOWN:
        return "不可比：" + (notes[0] if notes else "缺少判断配置动没动的依据。")
    if outcome == REGRESSED:
        return "配置一致，整体判定由通过变为不通过——这是行为差异。"
    if outcome == FIXED:
        return "配置一致，整体判定由不通过变为通过。"
    if outcome == INDETERMINATE:
        return "配置一致，" + (notes[0] if notes else "但结论走向比不出来。")
    return "配置一致，整体判定未变。"


def compare(baseline: Snapshot, current: Snapshot) -> Comparison:
    """把两次运行的结论放在一起看。

    **差异清单在不可比时照样给出**，不做条件分支隐藏。理由：用户点进这个
    页面就是想看两次差在哪，"不可比"是说这些差异不能拿去下结论，不是
    说它们不存在。藏起来会让人以为两次运行真的一模一样。
    """
    comparability_, changed, notes = comparability(baseline, current)
    outcome, outcome_note = _outcome(baseline, current, comparability_)
    if outcome_note:
        notes = [*notes, outcome_note]

    differences = [
        *_compare_error(baseline, current),
        *_compare_verdict(baseline, current),
        *_compare_premises(baseline, current),
        *_compare_findings(baseline, current),
        *_compare_expectations(baseline, current),
        *_compare_charges(baseline, current),
        *_compare_sessions(baseline, current),
        *_compare_memory_horizons(baseline, current),
        *_compare_cost(baseline, current),
    ]

    return Comparison(
        comparability=comparability_,
        outcome=outcome,
        headline=_headline(comparability_, outcome, changed, notes),
        changed_sections=changed,
        notes=notes,
        differences=differences,
    )
