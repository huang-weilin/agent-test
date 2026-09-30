"""幂等检测器。这一组测试直接对应验收标准之二：构造一次重复扣款，必须报出来。"""

from __future__ import annotations

from sandbox.db.models import DetectorVerdict
from sandbox.detectors.base import Exemption, ExemptionSet
from sandbox.detectors.idempotency import IdempotencyDetector, ToolFacts

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
    import pytest

    with pytest.raises(ValueError, match="reason"):
        Exemption(detector="idempotency", tool="x", reason="")
