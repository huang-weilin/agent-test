"""幂等检测器。这一组测试直接对应验收标准之二：构造一次重复扣款，必须报出来。"""

from __future__ import annotations

from pathlib import Path

import pytest

from sandbox.db.models import DetectorVerdict
from sandbox.detectors.base import Exemption, ExemptionSet, Finding
from sandbox.detectors.idempotency import IdempotencyDetector, ToolFacts
from sandbox.scenario.loader import load_scenario
from sandbox.scenario.runner import ScenarioResult, ScenarioRunner
from sandbox.scenario.spec import Scenario
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
