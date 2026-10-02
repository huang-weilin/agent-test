"""回归对比：先说清能不能比，再说差在哪。

这个文件里最要紧的不是"差异能被列出来"——那是容易的部分。要紧的是**不可比
的情况必须报成不可比**。一个总是给出结论的对比工具比没有对比工具更坏：它会
把"我改了 Prompt"报成"Agent 退化了"，于是下一次没人再信它。
"""

from __future__ import annotations

from typing import Any

import pytest

from sandbox.cost.aggregator import CostRollup
from sandbox.regression import diff

#: 一份完整配置，五个节都在。改配置的用例在它上面改动。
CONFIG: dict[str, Any] = {
    "prompt": {"task": "把订单处理掉", "model": "stub-model"},
    "tools": [{"name": "create_payment", "is_write": True}],
    "script": [{"name": "第一次", "script": [{"tool_calls": []}]}],
    "agent": {"version": "a@1", "directory": "react_demo", "entry": "agent.py", "env": {}},
    "expect": {"expect": [{"detector": "idempotency", "verdict": "pass"}], "expected_charges": 1},
}


def _report(
    *,
    ok: bool = True,
    premises_held: bool = True,
    error: str | None = None,
    settings: list[dict[str, Any]] | None = None,
    findings: list[dict[str, Any]] | None = None,
    expectation: dict[str, Any] | None = None,
    egress_charges: int = 1,
    expected_charges: int | None = 1,
    cost: float | None = 0.01,
) -> dict[str, Any]:
    return {
        "ok": ok,
        "premises_held": premises_held,
        "error": error,
        "sessions": settings
        if settings is not None
        else [
            {
                "name": "第一次",
                "ordinal": 0,
                "memory_horizon_ordinal": -1,
                "premise_held": premises_held,
            }
        ],
        "findings": findings
        if findings is not None
        else [{"detector": "idempotency", "verdict": "pass", "summary": "只付了一次"}],
        "expectation": expectation
        if expectation is not None
        else {"matched": [], "missing": [], "unexpected": [], "uncovered": []},
        "egress_charges": egress_charges,
        "expected_charges": expected_charges,
        # **问运行器要形状，不手写字典。** 手写的话，夹具和被测代码会用同一个
        # 错键，于是成本对比在真实数据上什么都不报、而测试一直是绿的——这个
        # 坑已经踩过一次（代码读 `total_usd`，运行器写 `cost_usd`）。
        "cost": None if cost is None else CostRollup(cost_usd=cost).to_dict(),
    }


#: `_snap(report=...)` 不传时是"用默认报告"，所以"压根没有报告"要另一个哨兵。
#: 两者是不同的状态（一个是有结论、一个是没结论），共用一个 `None` 会把
#: "没留存"的用例悄悄测成"留存了"，于是那组用例全部假绿。
#:
#: 哨兵只活在测试里：`_snap` 把它翻成真正的 `None`（也就是库里那个 NULL），
#: 哨兵对象本身绝不进 `Snapshot`——它不是 JSON，`Snapshot` 里的每个字段都得
#: 和从库里读出来的一模一样，否则测的是另一套输入。
_NO_REPORT = object()
_DEFAULT_REPORT = object()


def _sessions(*horizons: int) -> list[dict[str, Any]]:
    """按序号构造会话列表。`horizons` 一项一个会话，值是它的记忆视野。"""
    return [
        {
            "name": f"第{n + 1}次",
            "ordinal": n,
            "memory_horizon_ordinal": h,
            "premise_held": True,
        }
        for n, h in enumerate(horizons)
    ]


def _snap(
    *,
    run_id: str = "run-a",
    report: Any = _DEFAULT_REPORT,
    verdict_ok: bool | None = True,
    scheme: str | None = "2",
    parts: dict[str, Any] | None = None,
    config_hash: str | None = "h",
) -> diff.Snapshot:
    return diff.Snapshot(
        run_id=run_id,
        verdict_ok=verdict_ok,
        report=_report()
        if report is _DEFAULT_REPORT
        else (None if report is _NO_REPORT else report),
        config_scheme=scheme,
        config_hash=config_hash,
        config_parts=CONFIG if parts is None else parts,
        env={"model": "stub-model"},
    )


def _kinds(comparison: diff.Comparison) -> list[str]:
    return [d.kind for d in comparison.differences]


def _find(comparison: diff.Comparison, kind: str) -> list[diff.Difference]:
    return [d for d in comparison.differences if d.kind == kind]


# --- 配置没变：差异就是行为差异 -------------------------------------------------


def test_identical_runs_report_no_change() -> None:
    result = diff.compare(_snap(), _snap(run_id="run-b"))
    assert result.comparability == diff.COMPARABLE
    assert result.outcome == diff.SAME
    assert result.differences == []
    assert result.counts


def test_a_passing_run_that_starts_failing_is_a_regression() -> None:
    """这就是回归工具存在的理由本身。"""
    baseline = _snap()
    current = _snap(run_id="run-b", verdict_ok=False, report=_report(ok=False))
    result = diff.compare(baseline, current)

    assert result.comparability == diff.COMPARABLE
    assert result.outcome == diff.REGRESSED
    assert result.counts
    verdict = _find(result, diff.KIND_VERDICT)
    assert len(verdict) == 1
    assert verdict[0].before == "通过"
    assert verdict[0].after == "不通过"


def test_a_failing_run_that_starts_passing_is_reported_as_fixed() -> None:
    baseline = _snap(verdict_ok=False, report=_report(ok=False))
    current = _snap(run_id="run-b")
    result = diff.compare(baseline, current)
    assert result.outcome == diff.FIXED
    assert _find(result, diff.KIND_VERDICT)[0].sentence == "整体判定：不通过 → 通过"


def test_only_the_verdict_is_compared_not_the_wording() -> None:
    """结论文案改了不算行为差异。

    文案是给人读的句子。把改措辞报成差异，这份清单很快就会没人看——
    而没人看的清单等于没有清单。代价是"同一个 fail、但细节不同"看不出来，
    那件事由地面真值那一行补上。
    """
    baseline = _snap(
        report=_report(
            findings=[
                {"detector": "idempotency", "verdict": "fail", "summary": "重复扣款 2 笔"}
            ]
        )
    )
    current = _snap(
        run_id="run-b",
        verdict_ok=False,
        report=_report(
            ok=False,
            findings=[
                {
                    "detector": "idempotency",
                    "verdict": "fail",
                    "summary": "订单 A1 被重复扣款，共 2 笔",
                }
            ],
        ),
    )
    result = diff.compare(baseline, current)
    assert _find(result, diff.KIND_FINDING) == []


# --- 配置变了：这次比较不作数 ---------------------------------------------------


@pytest.mark.parametrize(
    ("section", "value"),
    [
        ("prompt", {"task": "换个任务", "model": "stub-model"}),
        ("tools", [{"name": "create_payment", "is_write": False}]),
        ("agent", {"version": "a@2", "directory": "react_demo", "entry": "agent.py", "env": {}}),
    ],
)
def test_a_config_change_makes_the_comparison_not_count(section: str, value: Any) -> None:
    """配置变了，结论差异可能就是改配置改出来的。

    这时候报"回归"会让用户去查一个他刚刚亲手改过的东西。
    """
    current_parts = {**CONFIG, section: value}
    result = diff.compare(_snap(), _snap(run_id="run-b", parts=current_parts))

    assert result.comparability == diff.CONFIG_CHANGED
    assert result.outcome == diff.INDETERMINATE
    assert not result.counts
    assert result.changed_sections == [section]
    assert section in result.headline or "配置变了" in result.headline


def test_the_test_changing_is_reported_apart_from_the_config_changing() -> None:
    """改了脚本不是"配置变更"的一种，它比配置变更更严重。

    那意味着两次运行比的是**两套测试**。把它和"改了 Prompt"归成一类，
    等于允许用户拿两套不同的卷子对分数。
    """
    for section, value in (
        ("script", [{"name": "第一次", "script": [{"tool_calls": [{"name": "x"}]}]}]),
        (
            "expect",
            {
                "expect": [{"detector": "idempotency", "verdict": "fail"}],
                "expected_charges": 2,
            },
        ),
    ):
        result = diff.compare(_snap(), _snap(run_id="run-b", parts={**CONFIG, section: value}))
        assert result.comparability == diff.TEST_CHANGED, f"{section} 被当成了普通配置变更"
        assert result.outcome == diff.INDETERMINATE
        assert not result.counts


def test_differences_are_still_listed_when_they_do_not_count() -> None:
    """不可比不等于没有差异。

    藏起来会让人以为两次运行真的一模一样。列出来、并且写明"这些可能只是
    改配置带来的"，才是完整的回答。
    """
    result = diff.compare(
        _snap(),
        _snap(
            run_id="run-b",
            parts={**CONFIG, "prompt": {"task": "换个任务", "model": "stub-model"}},
            verdict_ok=False,
            report=_report(ok=False, egress_charges=2),
        ),
    )
    assert not result.counts
    assert diff.KIND_VERDICT in _kinds(result)
    assert diff.KIND_CHARGES in _kinds(result)


# --- 缺依据时必须说"不知道" -----------------------------------------------------


def test_a_run_without_a_config_scheme_is_not_comparable() -> None:
    """本字段存在之前的运行，指纹是按旧口径算的（改了脚本或 Prompt 都不变）。

    照哈希判"配置没变"会得出一句基于不可比数据的结论。
    """
    result = diff.compare(_snap(), _snap(run_id="run-b", scheme=None))
    assert result.comparability == diff.UNKNOWN
    assert not result.counts
    assert result.headline.startswith("不可比")


def test_two_different_schemes_are_not_comparable() -> None:
    """两种口径算出的哈希不相等，**不能读成"配置变了"**。

    这是 `config_scheme` 这一列存在的全部理由。
    """
    result = diff.compare(_snap(scheme="1"), _snap(run_id="run-b", scheme="2"))
    assert result.comparability == diff.UNKNOWN
    assert result.changed_sections == []
    assert "口径" in result.headline


def test_a_run_without_config_content_is_not_assumed_unchanged() -> None:
    """一次没带配置的运行和一个真运行之间没有可比性。"""
    result = diff.compare(_snap(), _snap(run_id="run-b", parts={}))
    assert result.comparability == diff.UNKNOWN
    assert not result.counts


def test_two_runs_without_config_content_are_comparable_to_each_other() -> None:
    """两个都没带配置，那是**一致**，不是未知。

    两种情况不同：跟一个真配置比是"无从判断"，两个探活运行互比是"确实一样"。
    """
    result = diff.compare(
        _snap(parts={}, verdict_ok=None, report=_NO_REPORT),
        _snap(run_id="run-b", parts={}, verdict_ok=None, report=_NO_REPORT),
    )
    assert result.comparability == diff.COMPARABLE


def test_missing_verdicts_do_not_become_failures() -> None:
    """没留存结论不是"不通过"。

    替一次没有结论的运行编一个结论，是四态语义里被明令禁止的那件事。
    """
    result = diff.compare(
        _snap(verdict_ok=None, report=_NO_REPORT),
        _snap(run_id="run-b", verdict_ok=None, report=_NO_REPORT),
    )
    assert result.outcome == diff.INDETERMINATE
    assert result.differences == []


def test_one_side_missing_a_verdict_is_indeterminate() -> None:
    result = diff.compare(_snap(verdict_ok=None, report=_NO_REPORT), _snap(run_id="run-b"))
    assert result.outcome == diff.INDETERMINATE
    assert "没留下结论" in "".join(result.notes)


# --- 前提不同则"通过与否"不是同一个问题 --------------------------------------------


def test_differing_premises_block_the_outcome() -> None:
    """基线那次什么都没测到，这次的"通过"和它的"不通过"之间没有可比性。

    报成"已修复"是在说一件没发生过的事——什么都没改，只是这次脚本走完了。
    这是本项目"没测到 ≠ 表现好"那条规矩在对比侧的样子。
    """
    baseline = _snap(verdict_ok=False, report=_report(ok=False, premises_held=False))
    current = _snap(run_id="run-b")
    result = diff.compare(baseline, current)

    assert result.comparability == diff.COMPARABLE
    assert result.outcome == diff.INDETERMINATE
    assert "前提不同" in "".join(result.notes)
    assert diff.KIND_PREMISE in _kinds(result)


def test_a_session_premise_is_reported_per_session() -> None:
    """两次都跑了两个会话，只有第二个的前提翻了。

    报成整条运行的"前提不成立"，用户就得自己去猜是哪一个会话没走完——
    而跨会话场景里"是第几次没走完"恰恰是唯一有用的信息。
    """
    baseline = _snap(report=_report(settings=_sessions(-1, 0)))
    current = _snap(
        run_id="run-b",
        report=_report(
            settings=[
                {**_sessions(-1, 0)[0]},
                {**_sessions(-1, 0)[1], "premise_held": False},
            ],
            premises_held=False,
        ),
    )
    result = diff.compare(baseline, current)
    premise = _find(result, diff.KIND_PREMISE)
    assert [d.subject for d in premise] == ["会话 #1的前提"]
    assert premise[0].after == "不成立（脚本没走完）"


def test_a_session_that_only_exists_in_one_run_is_reported_as_added_or_removed() -> None:
    """会话增删按序号对齐后报出来，不是静默跳过。

    静默跳过会让人以为两次运行的会话结构一样。
    """
    baseline = _snap()
    current = _snap(run_id="run-b", report=_report(settings=_sessions(-1, 0)))
    sessions = _find(diff.compare(baseline, current), diff.KIND_SESSION)
    assert len(sessions) == 1
    assert sessions[0].subject == "会话 #1"
    assert (sessions[0].before, sessions[0].after) == ("没有这次会话", "存在")


# --- 地面真值 -----------------------------------------------------------------


def test_charges_are_compared_even_when_the_verdict_does_not_move() -> None:
    """检测器可能报出漂亮的 pass 而钱被扣了两次。

    这正是本工具要抓的失效方式，所以这一行独立于检测结论单独比。
    """
    result = diff.compare(_snap(), _snap(run_id="run-b", report=_report(egress_charges=2)))
    charges = _find(result, diff.KIND_CHARGES)
    assert charges[0].before == "1"
    assert charges[0].after == "2"


def test_unreachable_target_system_is_not_shown_as_zero_charges() -> None:
    """`egress_charges < 0` 是**查不到**，不是"扣了负一笔"，更不是"没扣钱"。

    把查不到显示成 0，就是替一次没测到的运行断言"没问题"。
    """
    result = diff.compare(_snap(), _snap(run_id="run-b", report=_report(egress_charges=-1)))
    charges = _find(result, diff.KIND_CHARGES)
    assert charges[0].after == "查不到"


def test_changed_expectation_count_is_reported_separately() -> None:
    """期望笔数变了直接改变"通过"的判据，值得单独一行。"""
    result = diff.compare(
        _snap(), _snap(run_id="run-b", report=_report(expected_charges=2))
    )
    subjects = [d.subject for d in _find(result, diff.KIND_CHARGES)]
    assert "期望扣款笔数" in subjects


# --- 记忆视野 -----------------------------------------------------------------


def test_a_moved_memory_horizon_is_reported_as_a_test_change() -> None:
    """视野变了意味着回滚位置变了，那是**测试**的旋钮。

    不解释的话，用户会把这行读成"Agent 的记忆出了问题"。

    这里正是三条边界场景的差别所在：`cross_session_duplicate` 与
    `cross_session_memory_loss` 的事实完全相同——同一个重复扣款、同样的会话
    结构——只有视野不同，于是归因完全相反。回归对比必须看得见这个差别，
    否则"改了回滚位置"会被读成"Agent 退化了"。
    """
    baseline = _snap(report=_report(settings=_sessions(-1, 0)))
    current = _snap(run_id="run-b", report=_report(settings=_sessions(-1, -1)))
    result = diff.compare(baseline, current)
    horizon = _find(result, diff.KIND_MEMORY_HORIZON)
    assert [d.subject for d in horizon] == ["会话 #1的记忆视野"]
    assert horizon[0].sentence == "会话 #1的记忆视野：覆盖到第 0 次会话 → 什么都记不得"


# --- 检测结论与预期 ---------------------------------------------------------------


def test_a_detector_changing_its_conclusion_is_reported() -> None:
    current = _snap(
        run_id="run-b",
        verdict_ok=False,
        report=_report(
            ok=False,
            findings=[{"detector": "idempotency", "verdict": "fail", "summary": "扣了两次"}],
        ),
    )
    result = diff.compare(_snap(), current)
    finding = _find(result, diff.KIND_FINDING)
    assert len(finding) == 1
    assert finding[0].subject == "idempotency"
    assert (finding[0].before, finding[0].after) == ("pass", "fail")


def test_a_detector_that_only_appears_in_one_run_is_reported() -> None:
    current = _snap(
        run_id="run-b",
        report=_report(
            findings=[
                {"detector": "idempotency", "verdict": "pass", "summary": "s"},
                {"detector": "loop", "verdict": "warning", "summary": "s"},
            ]
        ),
    )
    finding = _find(diff.compare(_snap(), current), diff.KIND_FINDING)
    assert finding[0].before == "没给出结论"
    assert finding[0].after == "warning"


def test_expectations_that_stop_matching_are_reported() -> None:
    current = _snap(
        run_id="run-b",
        verdict_ok=False,
        report=_report(
            ok=False,
            expectation={
                "matched": [],
                "missing": [{"detector": "idempotency", "verdict": "pass"}],
                "unexpected": [{"detector": "loop", "verdict": "fail", "summary": "s"}],
                "uncovered": [],
            },
        ),
    )
    result = diff.compare(_snap(), current)
    subjects = [d.subject for d in _find(result, diff.KIND_EXPECTATION)]
    assert subjects == ["没有对上的预期", "多出来的失败"]


# --- 成本：信息，不是判定 -----------------------------------------------------------


def test_cost_differences_are_listed_but_do_not_drive_the_outcome() -> None:
    """花多少钱是事实陈述，不是"行为有没有缺陷"的结论。

    算进判定的话，一个完全正常的运行会因为多调了一次模型而变红。
    """
    result = diff.compare(_snap(), _snap(run_id="run-b", report=_report(cost=0.02)))
    assert diff.KIND_COST in _kinds(result)
    assert result.outcome == diff.SAME
    assert result.counts


def test_the_cost_comparison_reads_the_key_the_runner_actually_writes() -> None:
    """**形状从生产者那里要，不手写。**

    这一条是补的：原来的成本对比读的是 `total_usd`，而运行器写的是
    `cost_usd`——真实数据上这个对比从来没报过任何东西。三条测试全是绿的，
    因为夹具照着同一个错键造、代码照着同一个错键读，**两边一致地错**。

    所以这里不问测试自己（那样只会再骗一次），而是问 `CostRollup.to_dict()`
    ——运行器真正写进 `report_json` 的那个形状。它红了只可能是一个原因：
    读的键和写的键又不是同一个了。
    """
    cheap = CostRollup(llm_calls=1, cost_usd=0.01).to_dict()
    dear = CostRollup(llm_calls=3, cost_usd=0.02).to_dict()

    before = _report()
    before["cost"] = cheap
    after = _report()
    after["cost"] = dear

    result = diff.compare(_snap(), _snap(run_id="run-b", report=after))
    cost = _find(result, diff.KIND_COST)
    assert cost, "运行器写的那个键读不到，成本对比就是一段不报数的死代码"
    assert cost[0].before == "$0.0100"
    assert cost[0].after == "$0.0200"


def test_a_lower_bound_is_not_shown_as_a_precise_number() -> None:
    """未定价的调用让金额少算了一部分，报告不能把它印成一个精确值。

    这个项目对"下界被读成账单"的警惕写在 `docs/cost.md` 里；回归对比页是
    成本出现的第四个地方，同样要守。
    """
    before = _report()
    before["cost"] = CostRollup(llm_calls=1, cost_usd=0.01).to_dict()
    after = _report()
    after["cost"] = CostRollup(llm_calls=1, cost_usd=0.01, unpriced_calls=1).to_dict()
    after["cost"] = {**after["cost"], "cost_usd": 0.02}

    cost = _find(diff.compare(_snap(), _snap(run_id="run-b", report=after)), diff.KIND_COST)
    assert cost
    assert "下界" in cost[0].after
    assert "下界" not in cost[0].before


def test_a_ledger_gap_travels_with_the_money_in_the_diff() -> None:
    """账没对平的时候，这一格印的金额是少算的，得跟着一起说。

    和下面那条的分别在这里：`None` 是**没比过**，非零是**比过、而且明确知道
    缺了几条**。后者是真的信息——`$0.0100` 比实际少，不标出来就是拿下界冒充
    账单，而这是这个项目在成本上守得最紧的一条。
    """
    settled = {**_report(cost=0.01), "unrecorded_llm_calls": 0}
    short = {**_report(cost=0.01), "unrecorded_llm_calls": 2}
    early = {**_report(cost=0.01), "unrecorded_llm_calls": -1}

    def compare_to(after: dict[str, Any]) -> list[diff.Difference]:
        return _find(
            diff.compare(_snap(report=settled), _snap(run_id="run-b", report=after)),
            diff.KIND_COST,
        )

    missing = compare_to(short)
    assert missing, "缺了明细的那一侧金额是少算的，这一格不能和正常的一格长得一样"
    assert "明细缺 2 条" in missing[0].after
    assert "明细缺" not in missing[0].before

    ahead = compare_to(early)
    assert ahead
    assert "明细多 1 条" in ahead[0].after


def test_a_report_that_never_reconciled_does_not_invent_a_cost_difference() -> None:
    """`None`（没对过账）在两份报告之间**不作声**。

    `None` 和 `0` 是两件事——一个是"没比过"，一个是"比过且对上了"。但在**对比**
    这个场景里把 `None` 也印成一格，升级前后的两次运行（一份有这个键、一份
    没有）就会**每一次都凭空多出一条成本差**：那不是两次运行的差异，是版本的
    差异，混进"这次比上次变了什么"里就是一条假警报，而假警报一多真警报就没人看了。

    单看一份报告的地方才打这句（运行页、终端报告），那里没有"另一份"可赖。
    """
    legacy = _report(cost=0.01)  # 老版本写的报告，压根没有这个键
    assert "unrecorded_llm_calls" not in legacy

    for after in (
        {**_report(cost=0.01), "unrecorded_llm_calls": 0},
        {**_report(cost=0.01), "unrecorded_llm_calls": None},
    ):
        result = diff.compare(_snap(report=legacy), _snap(run_id="run-b", report=after))
        assert diff.KIND_COST not in _kinds(result), (
            "「没对过账」和「没这个键」在对比里不能算差异，否则每次升级都会多一条"
        )


def test_a_run_that_crashed_shows_the_error_not_just_a_red_verdict() -> None:
    """"没跑起来"和"跑起来了但没通过"是两回事，报告必须分得开。"""
    current = _snap(
        run_id="run-b", verdict_ok=False, report=_report(ok=False, error="连接被拒绝")
    )
    result = diff.compare(_snap(), current)
    error = _find(result, diff.KIND_ERROR)
    assert error[0].before == "正常结束"
    assert error[0].after == "连接被拒绝"
