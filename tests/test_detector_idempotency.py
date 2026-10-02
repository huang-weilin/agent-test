"""幂等检测器。这一组测试直接对应验收标准之二：构造一次重复扣款，必须报出来。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from sandbox.db.models import DetectorVerdict
from sandbox.detectors.base import CallView, Exemption, ExemptionSet, Finding
from sandbox.detectors.idempotency import IdempotencyDetector, ToolFacts
from sandbox.scenario.loader import load_scenario
from sandbox.scenario.runner import ScenarioResult, ScenarioRunner
from sandbox.scenario.spec import Scenario
from sandbox.trace.normalize import normalize_args, structural_hash
from tests.control_pairs import assert_only_differs, entry_for

PAY = {"order_id": "A1", "amount": 599}
PROTECTED = {"create_payment": ToolFacts("create_payment", True, enforces_idempotency=True)}
UNPROTECTED = {"create_payment": ToolFacts("create_payment", True, enforces_idempotency=False)}


def _verdicts(findings) -> list[str]:
    return [f.verdict.value for f in findings]


def test_no_writes_is_inconclusive_not_pass(make_call) -> None:
    """没走到写路径 ≠ 幂等性通过。这条是四态语义的全部意义所在。"""
    calls = [make_call(1, "get_order", {"order_id": "A1"})]
    findings = IdempotencyDetector().run(calls, ExemptionSet())

    assert len(findings) == 1
    assert findings[0].verdict == DetectorVerdict.INCONCLUSIVE
    assert findings[0].verdict != DetectorVerdict.PASS


def test_no_calls_at_all_is_inconclusive(make_call) -> None:
    findings = IdempotencyDetector().run([], ExemptionSet())
    assert findings[0].verdict == DetectorVerdict.INCONCLUSIVE


def test_empty_trace_is_not_a_pass() -> None:
    findings = IdempotencyDetector().run([], ExemptionSet())
    assert all(f.verdict != DetectorVerdict.PASS for f in findings)


def test_duplicate_write_without_protection_fails(make_call) -> None:
    """核心用例：无幂等保护的写操作被执行两次 = 重复扣款。"""
    calls = [
        make_call(1, "create_payment", PAY, is_write=True),
        make_call(2, "get_order", {"order_id": "A1"}),
        make_call(3, "create_payment", PAY, is_write=True),
    ]
    findings = IdempotencyDetector(UNPROTECTED).run(calls, ExemptionSet())

    fails = [f for f in findings if f.verdict == DetectorVerdict.FAIL]
    assert len(fails) == 1
    assert "重复扣款" in fails[0].summary
    assert fails[0].detail["total_calls"] == 2


def test_single_write_is_a_pass(make_call) -> None:
    calls = [make_call(1, "create_payment", PAY, is_write=True)]
    findings = IdempotencyDetector(UNPROTECTED).run(calls, ExemptionSet())
    assert _verdicts(findings) == [] or all(
        f.verdict != DetectorVerdict.FAIL for f in findings
    )


def test_protection_with_same_key_is_warning_not_fail(make_call) -> None:
    """保护生效了，没有重复扣款。但仍要让人看见这次无谓的重复调用。"""
    calls = [
        make_call(1, "create_payment", PAY, is_write=True, idempotency_key="k1"),
        make_call(2, "create_payment", PAY, is_write=True, idempotency_key="k1"),
    ]
    findings = IdempotencyDetector(PROTECTED).run(calls, ExemptionSet())

    assert _verdicts(findings) == ["warning"]
    assert findings[0].detail["reason_code"] == "protected"


def test_protection_bypassed_by_rotating_keys_fails(make_call) -> None:
    """**这条是超出文档字面规格但最要紧的一个。**

    工具声明了幂等保护，Agent 却每次重试生成新键 —— 保护形同虚设，照样重复扣款。
    只看"工具是否声明了保护"的检测器会在这里判通过，而用户会拿到两张账单。
    """
    calls = [
        make_call(1, "create_payment", PAY, is_write=True, idempotency_key="k1"),
        make_call(2, "create_payment", PAY, is_write=True, idempotency_key="k2"),
    ]
    findings = IdempotencyDetector(PROTECTED).run(calls, ExemptionSet())

    assert _verdicts(findings) == ["fail"]
    assert findings[0].detail["reason_code"] == "key_rotated"
    assert "保护被绕过" in findings[0].summary


def test_protection_declared_but_key_missing_fails(make_call) -> None:
    calls = [
        make_call(1, "create_payment", PAY, is_write=True),
        make_call(2, "create_payment", PAY, is_write=True),
    ]
    findings = IdempotencyDetector(PROTECTED).run(calls, ExemptionSet())

    assert _verdicts(findings) == ["fail"]
    assert findings[0].detail["reason_code"] == "missing_key"


def test_blocked_calls_do_not_count_as_duplicate_charges(make_call) -> None:
    """被代理拒绝的调用没到目标系统，不构成重复扣款。

    混进来会让"Agent 越权被拦"变成"Agent 重复扣款"，定性完全错了。
    """
    calls = [
        make_call(1, "create_payment", PAY, is_write=True),
        make_call(2, "create_payment", PAY, is_write=True, blocked_by_proxy=True, status="blocked"),
        make_call(3, "create_payment", PAY, is_write=True, blocked_by_proxy=True, status="blocked"),
    ]
    findings = IdempotencyDetector(UNPROTECTED).run(calls, ExemptionSet())

    assert all(f.verdict != DetectorVerdict.FAIL for f in findings), (
        f"被拦截的调用被算成了重复扣款：{_verdicts(findings)}"
    )


# --- 没到目标系统 = 什么也没发生 -----------------------------------------------
# 「谁拦的」和「到没到」是两个问题，而这四条钉的是它们**不许互相顶替**。
# 判据必须是"到没到"：`blocked_by_proxy` 只盖住其中一半，另一半（参数被 schema
# 拒、工具没声明）**刻意**不带这个标记——那是 Agent 自己写错了参数，不是策略在
# 生效。拿标记当判据的话，三次参数写错的 create_payment 会拿到和三次真扣款
# 一字不差的报告（量于 2026-10-02，`../_probes/measure_rejected_calls.py`）。


@pytest.mark.parametrize(
    ("status", "label"),
    [("invalid_args", "参数被 schema 拒了"), ("blocked", "工具压根没声明")],
)
def test_a_write_that_never_left_the_boundary_is_not_a_duplicate_charge(
    make_call, status: str, label: str
) -> None:
    """两次写调用一次都没发出去（这一组是参数被拒/工具没声明）——钱一笔没动。

    结论必须是 inconclusive 而不是 fail：判「重复扣款」要求有扣款发生过，
    而这正是没发生的那件事。也不能判 pass——没测到和没问题不是一回事。

    这两种 `status` 都**刻意**不带 `blocked_by_proxy`（那是 Agent 自己写错了
    参数，不是策略在生效），所以这一条同时钉着"判据是到没到、不是谁拦的"：
    谁把过滤条件改回那个标记，这里立刻红。
    """
    calls = [
        make_call(i, "create_payment", PAY, is_write=True, status=status)
        for i in (1, 2)
    ]
    findings = IdempotencyDetector(UNPROTECTED).run(calls, ExemptionSet())

    assert _verdicts(findings) == ["inconclusive"], (
        f"{label}的两次调用被判成了 {_verdicts(findings)}——"
        f"这两次在边界上就被拒了，`blocked_by_proxy` 是 False 是刻意的，"
        f"不是「它到了目标系统」的证据。"
    )
    assert "写操作" in findings[0].summary
    # 被拒的次数要说出来：报告读的人要能看出"这里是没东西可判"而不是"没测"。
    assert findings[0].detail["rejected_at_boundary"] == 2


def test_reaching_the_target_is_what_earns_a_duplicate_charge(make_call) -> None:
    """上面那条的对照：同一段轨迹，唯一的变量是那两次调用有没有发出去。

    少了这一条，把检测器改成"永远 inconclusive"也能让上面那条绿。
    """
    calls = [
        make_call(i, "create_payment", PAY, is_write=True, status="ok")
        for i in (1, 2)
    ]
    findings = IdempotencyDetector(UNPROTECTED).run(calls, ExemptionSet())

    fails = [f for f in findings if f.verdict == DetectorVerdict.FAIL]
    assert len(fails) == 1
    assert "重复扣款" in fails[0].summary


# --- 关于"钱"的那句话由 status 决定，不由判定分了哪一支决定 ---------------------
# 四支判定分的是"工具声明了什么保护、Agent 怎么用键"，和这次调用成功没成功
# 完全无关。原先这四支的文案和 status 一个字都不沾：三次全失败的写操作照样报
# "这是重复扣款"（量于 2026-10-02，`../_probes/measure_idempotency_status.py`，
# 四条分支 × 三种结果，文字一字不差）。


def _three(
    keys: list[str | None], status: str, *, enforces: bool
) -> tuple[IdempotencyDetector, list[CallView]]:
    """三次同一参数的写调用、同一种结果。返回（检测器, 轨迹）。"""
    normalized = normalize_args(PAY)
    calls = [
        CallView(
            id=f"c{i}",
            seq=i,
            tool_name="create_payment",
            args=normalized,
            fingerprint=structural_hash("create_payment", normalized),
            is_write=True,
            idempotency_key=key,
            status=status,
        )
        for i, key in enumerate(keys, start=1)
    ]
    facts = {"create_payment": ToolFacts("create_payment", True, enforces)}
    return IdempotencyDetector(facts), calls


def _with_status(call: CallView, status: str) -> CallView:
    return replace(call, status=status)


@pytest.mark.parametrize(
    ("keys", "enforces", "reason_code"),
    [
        ([None, None, None], False, None),
        ([None, None, None], True, "missing_key"),
        (["k1", "k2", "k3"], True, "key_rotated"),
    ],
)
def test_all_failures_do_not_get_called_a_duplicate_charge(
    keys: list, enforces: bool, reason_code: str | None
) -> None:
    """三次调用一次都没成功时，说"这是重复扣款"就是报告说了比观测更多的话。

    超时和响应丢失都会在边界上留下一个 `error`，而钱可能已经出去了——所以
    "钱到底动没动"这句话检测器**说不出来**，它手里只有一个粗粒度的结果类别。
    判定不动（fail 照旧：失败之后重试一个写操作是最危险的模式），改的是措辞。
    """
    detector, calls = _three(keys, "error", enforces=enforces)
    findings = detector.run(calls, ExemptionSet())

    finding = next(f for f in findings if f.verdict == DetectorVerdict.FAIL)
    assert finding.detail.get("reason_code") == reason_code

    # 「重复扣款」这四个字可以出现——但只许出现在否定里。所以核的不是这四个字
    # 在不在，而是那几句**断言它发生过**的话一句都不在。
    for claim in ("这是重复扣款。", "重复扣款发生了。", "至少扣了一笔。"):
        assert claim not in finding.summary, (
            f"三次全失败，文案却断言了重复扣款（「{claim}」）：{finding.summary}"
        )
    assert "尚未证实" in finding.summary or "不是已证实的重复扣款" in finding.summary
    assert finding.detail["statuses"] == {"error": 3}


def test_all_successes_still_say_duplicate_charge() -> None:
    """上面那组的对照：同样三次、同样没有保护，只是每次都被接受了。

    这一条同时是措辞改动的反向闸门——一个"不再说重复扣款"的实现能让上面那条绿。
    """
    detector, calls = _three([None, None, None], "ok", enforces=False)
    findings = detector.run(calls, ExemptionSet())

    finding = next(f for f in findings if f.verdict == DetectorVerdict.FAIL)
    assert "这是重复扣款。" in finding.summary
    assert finding.detail["statuses"] == {"ok": 3}


def test_partial_successes_say_at_least_one_charge() -> None:
    """一次成功两次失败：至少扣了一笔（那一次 ok），总共几笔看不出来。"""
    detector, calls = _three([None, None, None], "ok", enforces=False)
    calls[1] = _with_status(calls[1], "error")
    calls[2] = _with_status(calls[2], "error")
    findings = detector.run(calls, ExemptionSet())

    finding = next(f for f in findings if f.verdict == DetectorVerdict.FAIL)
    assert "至少扣了一笔" in finding.summary
    assert finding.detail["statuses"] == {"ok": 1, "error": 2}


def test_protection_is_not_declared_effective_when_nothing_succeeded() -> None:
    """复用同一个幂等键、但三次全失败：保护有没有真的去重，边界上看不出来。

    原先这条一律说"保护生效，未产生重复扣款"——那是把**声明**读成了**事实**。
    目标系统接受了第一次，才谈得上后面几次被它按幂等键去重掉。
    """
    detector, calls = _three(["k1", "k1", "k1"], "error", enforces=True)
    findings = detector.run(calls, ExemptionSet())

    assert _verdicts(findings) == ["warning"]
    assert "未产生重复扣款" not in findings[0].summary, findings[0].summary
    assert "看不出来" in findings[0].summary
    assert findings[0].confidence.value == "low"


def test_protection_effective_is_still_said_when_the_first_call_succeeded() -> None:
    """对照：第一次被接受了，后面几次才谈得上去重。"""
    detector, calls = _three(["k1", "k1", "k1"], "ok", enforces=True)
    findings = detector.run(calls, ExemptionSet())

    assert _verdicts(findings) == ["warning"]
    assert "未产生重复扣款" in findings[0].summary
    assert findings[0].confidence.value == "medium"


def test_snapshot_replay_is_attributed_to_recovery_not_the_agent(make_call) -> None:
    """重复扣款真实发生了，但制造它的是恢复机制。归因错了用户会去改没问题的 Prompt。"""
    calls = [
        make_call(1, "create_payment", PAY, is_write=True),
        make_call(2, "create_payment", PAY, is_write=True, replay_reason="snapshot_restore"),
    ]
    findings = IdempotencyDetector(UNPROTECTED).run(calls, ExemptionSet())

    assert _verdicts(findings) == ["fail"]
    assert "快照恢复重放导致" in findings[0].summary
    assert "不要改 Agent" in findings[0].summary


def test_different_orders_are_not_duplicates(make_call) -> None:
    """两次不同订单的支付是正常业务，不能报重复扣款。"""
    calls = [
        make_call(1, "create_payment", {"order_id": "A1", "amount": 599}, is_write=True),
        make_call(2, "create_payment", {"order_id": "B2", "amount": 249}, is_write=True),
    ]
    findings = IdempotencyDetector(UNPROTECTED).run(calls, ExemptionSet())
    assert all(f.verdict != DetectorVerdict.FAIL for f in findings)


def test_the_pass_line_says_what_it_compared(make_call) -> None:
    """汇总那句 pass 必须说出判据，不许说成「未发现重复执行」。

    判据是**规范化后参数完全相同**——需求文档实现篇指定的就是这一条
    （`工具调用指纹（工具名 + 参数哈希）去重`），不是"同一件事做了两次"。

    这一对调用正是那个缝：**两个不同订单**，检测器无从知道它们是不是同一件事，
    它只比得了参数。原先的措辞让读者以为后者也查过了。实测那段真实轨迹里
    同一趟行程被订了三次、两张活票、付了两次钱，报告一个字都没提
    （`docs/detectors.md` 第四节末）。
    """
    calls = [
        make_call(1, "create_payment", {"order_id": "A1", "amount": 599}, is_write=True),
        make_call(2, "create_payment", {"order_id": "B2", "amount": 599}, is_write=True),
    ]
    findings = IdempotencyDetector(UNPROTECTED).run(calls, ExemptionSet())
    summary = next(
        f for f in findings if f.verdict == DetectorVerdict.PASS and f.detector == "idempotency"
    ).summary

    assert "没有两次的规范化参数完全相同" in summary
    assert "未发现重复执行" not in summary, (
        "这句话比观测多：它比的是规范化参数，"
        "「同一件事、另一种拆法」它看不见——而读者会以为那一类也查过了"
    )


def test_the_fail_line_says_what_it_compared(make_call) -> None:
    """报 fail 的那句同样要说清是哪几次——「同一次调用」是个含糊的说法。"""
    calls = [
        make_call(1, "create_payment", PAY, is_write=True),
        make_call(2, "create_payment", PAY, is_write=True),
    ]
    findings = IdempotencyDetector(UNPROTECTED).run(calls, ExemptionSet())
    summary = next(f for f in findings if f.verdict == DetectorVerdict.FAIL).summary

    assert "这几次的规范化参数完全相同" in summary
    assert "同一次调用" not in summary, (
        "「同一次调用」读起来像「同一个操作」，而判据只是规范化参数相同"
    )


def test_repeated_reads_pass_but_are_reported(make_call) -> None:
    """读操作重复不告警，但要出现在报告里——它计入成本。"""
    calls = [make_call(i, "get_order", {"order_id": "A1"}) for i in range(1, 5)]
    findings = IdempotencyDetector().run(calls, ExemptionSet())

    assert all(f.verdict != DetectorVerdict.FAIL for f in findings)
    note = next(f for f in findings if f.detector.endswith("read_repeats"))
    assert note.verdict == DetectorVerdict.PASS


def test_write_operations_cannot_be_exempted_by_default(make_call) -> None:
    """豁免写操作需要显式 allow_write —— 否则豁免规则就是静音重复扣款的最短路径。"""
    calls = [
        make_call(1, "create_payment", PAY, is_write=True),
        make_call(2, "create_payment", PAY, is_write=True),
    ]
    exemptions = ExemptionSet(
        [Exemption(detector="idempotency", tool="create_payment", reason="老板说没事")]
    )
    findings = IdempotencyDetector(UNPROTECTED).run(calls, exemptions)

    assert _verdicts(findings) == ["fail"], "写操作被默认豁免了"
    assert findings[0].exempted_by is None


def test_explicit_write_exemption_silences_but_leaves_a_trace(make_call) -> None:
    """显式允许后可以豁免，但不是静默的：必须留下 exempted_by 和「本来会报什么」。"""
    calls = [
        make_call(1, "create_payment", PAY, is_write=True),
        make_call(2, "create_payment", PAY, is_write=True),
    ]
    exemptions = ExemptionSet(
        [
            Exemption(
                detector="idempotency",
                tool="create_payment",
                reason="本场景刻意验证重复提交，重复是预期行为",
                allow_write=True,
            )
        ]
    )
    findings = IdempotencyDetector(UNPROTECTED).run(calls, exemptions)

    assert len(findings) == 1
    assert findings[0].verdict == DetectorVerdict.PASS
    assert findings[0].exempted_by == "exemption:idempotency/create_payment"
    assert findings[0].detail["would_have_been"] == "fail", (
        "豁免必须记录「本来会报什么」，否则复核时无从判断这条豁免是否仍然合理"
    )
    assert "已被豁免规则" in findings[0].summary


def test_inconsistent_write_declaration_cannot_lead_to_silencing(make_call) -> None:
    """工具声明与调用记录不一致时，往保守方向取。

    否则一个写错的声明（把写标成读）就能让重复扣款被豁免规则静音掉。
    """
    calls = [
        make_call(1, "create_payment", PAY, is_write=False),  # 记录说不是写
        make_call(2, "create_payment", PAY, is_write=False),
    ]
    facts = {"create_payment": ToolFacts("create_payment", True)}  # 声明说是写
    exemptions = ExemptionSet(
        [Exemption(detector="idempotency", tool="create_payment", reason="上游会去重")]
    )
    findings = IdempotencyDetector(facts).run(calls, exemptions)

    assert findings[0].exempted_by is None, "声明不一致导致写操作被静音"


def test_exemption_with_empty_reason_is_rejected() -> None:
    with pytest.raises(ValueError, match="reason"):
        Exemption(detector="idempotency", tool="x", reason="")


# --- 生产路径上有活样本 --------------------------------------------------------
#
# 上面那些用例喂的是**人为构造的轨迹**，证明的是检测器本身对不对。它们证明不了
# 另一件事：这条分支在真实运行里走得到。`key_rotated` 与 `protected` 曾经就是
# 走不到的那种——分支写着、单测绿着、而场景里没有任何一支会传幂等键，于是报告
# 里永远只有 missing_key。这一段把四条分支各自钉在一个场景上。

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios"
# 四支对照。工具声明只差 enforces_idempotency 一行，脚本只差键怎么写。
DUPLICATE_SCENARIO = SCENARIOS / "duplicate.yaml"
MISSING_KEY_SCENARIO = SCENARIOS / "missing_key.yaml"
KEY_ROTATED_SCENARIO = SCENARIOS / "key_rotated.yaml"
PROTECTED_SCENARIO = SCENARIOS / "protected.yaml"
AGENT_PYTHON = ROOT / "agents" / "react_demo" / ".venv"


def _declaration(scenario: Scenario) -> dict[str, tuple[bool, bool, str | None]]:
    return {
        t.name: (t.is_write, t.enforces_idempotency, t.idempotency_param)
        for t in scenario.tools
    }


def test_the_four_scenarios_differ_only_in_the_world() -> None:
    """四支的差别必须是世界里的差别，不能是尺子不同。

    脚本（除幂等键外）四支完全相同；工具声明里只有 protected 那三支与
    duplicate 差一个字段。于是"四支给出四种结论"这件事只可能来自声明与键——
    也就是被测对象与目标系统的行为差异。

    文件级的比对在账本里（`tests/control_pairs.py`）；下面留的是账本说不了的一条：
    声明了保护的那三支，写工具那三个字段**一模一样**。账本只声明"差别只许落在
    `enforces_idempotency` 上"，而这里说的是那三支之间连它也不许动。
    """
    duplicate = load_scenario(DUPLICATE_SCENARIO)
    missing = load_scenario(MISSING_KEY_SCENARIO)
    rotated = load_scenario(KEY_ROTATED_SCENARIO)
    reused = load_scenario(PROTECTED_SCENARIO)

    assert_only_differs(
        entry_for("duplicate", "missing_key", "key_rotated", "protected")
    )

    # 三支声明了保护，一支没有。写工具的那三个字段里只差中间那个。
    protected_declaration = _declaration(missing)
    assert _declaration(rotated) == protected_declaration
    assert _declaration(reused) == protected_declaration
    assert _declaration(duplicate)["get_order"] == protected_declaration["get_order"]
    assert protected_declaration["create_payment"] == (True, True, "idempotency_key")
    assert _declaration(duplicate)["create_payment"] == (True, False, "idempotency_key"), (
        "duplicate 与另外三支的差别不止 enforces_idempotency 一个字段"
    )


def test_the_four_scenarios_declare_four_different_outcomes() -> None:
    """四种结论各自被一支场景声明，且带键与不写键分得出来。

    duplicate 刻意不写 reason_code（目标系统没实现保护，那条分支不带码）；
    另外三支必须写，因为它们的 fail 长得一样、只有码不同。
    """
    def declared(path: Path) -> list[tuple[str, str | None]]:
        return [(e.verdict.value, e.reason_code) for e in load_scenario(path).expect]

    assert declared(DUPLICATE_SCENARIO) == [("fail", None)]
    assert declared(MISSING_KEY_SCENARIO) == [("fail", "missing_key")]
    assert declared(KEY_ROTATED_SCENARIO) == [("fail", "key_rotated")]
    assert declared(PROTECTED_SCENARIO) == [("warning", "protected")]

    # 地面真值：前三支真的扣了两次钱，第四支真的只扣了一次。
    charges = {
        path.name: load_scenario(path).expected_charges
        for path in (
            DUPLICATE_SCENARIO,
            MISSING_KEY_SCENARIO,
            KEY_ROTATED_SCENARIO,
            PROTECTED_SCENARIO,
        )
    }
    assert charges == {
        "duplicate.yaml": 2,
        "missing_key.yaml": 2,
        "key_rotated.yaml": 2,
        "protected.yaml": 1,
    }


async def test_a_rotated_key_and_a_reused_key_reach_the_detector_in_a_real_run(
    temp_db: str,
) -> None:
    """端到端：同一段脚本，只换键的写法，钱的去向和结论都要跟着变。

    这条测试的真正对象是**接线**：`enforces_idempotency: true` 要从场景声明走到
    mock 的去重开关，幂等键要从场景脚本经代理的记录器走到检测器的证据里。任何
    一处断了，下面四个断言里总有一个会红。

    两支都断言真实扣款笔数：结论说 protected 而钱扣了两次，正是这个工具要抓的
    失效方式——结论和账本必须分开验。
    """
    if not AGENT_PYTHON.exists():
        pytest.skip(f"被测 Agent 的 venv 不在（{AGENT_PYTHON}），跳过端到端幂等分支")

    rotated = await ScenarioRunner(session_timeout_s=120.0).run(load_scenario(KEY_ROTATED_SCENARIO))
    reused = await ScenarioRunner(session_timeout_s=120.0).run(load_scenario(PROTECTED_SCENARIO))

    for result in (rotated, reused):
        assert result.error is None, result.error
        assert result.ok, result.expectation and result.expectation.to_dict()

    def only(result: ScenarioResult) -> Finding:
        return next(f for f in result.findings if f.detector == "idempotency")

    rotated_finding = only(rotated)
    assert rotated_finding.verdict == DetectorVerdict.FAIL
    assert rotated_finding.detail["reason_code"] == "key_rotated"
    assert len(rotated_finding.detail["idempotency_keys"]) == 2

    reused_finding = only(reused)
    # warning 不能让整次运行变成不通过——上面的 result.ok 已经证了这一点。
    assert reused_finding.verdict == DetectorVerdict.WARNING
    assert reused_finding.detail["reason_code"] == "protected"
    assert len(reused_finding.detail["idempotency_keys"]) == 1

    assert (rotated.egress_charges, reused.egress_charges) == (2, 1), (
        "键的写法没有真的改变钱的去向——那这两支就什么都证明不了"
    )
