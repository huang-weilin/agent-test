"""幂等检测器。这一组测试直接对应验收标准之二：构造一次重复扣款，必须报出来。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from sandbox.db.models import DetectorVerdict
from sandbox.detectors.base import CallView, Exemption, ExemptionSet, Finding
from sandbox.detectors.idempotency import IdempotencyDetector, ToolFacts, checked_identity_fields
from sandbox.scenario.loader import parse_scenario
from sandbox.scenario.runner import ScenarioResult, ScenarioRunner
from sandbox.scenario.spec import Scenario
from sandbox.trace.normalize import normalize_args, structural_hash
from tests.control_pairs import assert_inline_only_differs, assert_only_differs, entry_for

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


def _replay_pair(make_call):
    """一对调用：一次 Agent 自己发起、一次快照重放——归因给恢复机制的那一支。"""
    return [
        make_call(1, "create_payment", PAY, is_write=True),
        make_call(2, "create_payment", PAY, is_write=True, replay_reason="snapshot_restore"),
    ]


def test_a_rule_silences_all_three_idempotency_branches(make_call) -> None:
    """同一条豁免规则，三条归因分支上都要管用。

    原先这条分支**提前 return**，"查豁免、降级、留痕"那一小段只在另外两条分支上
    各抄了一份——于是同一条规则管不管用，取决于检测器内部走了哪个 `if`。规则写的
    是检测器名，写规则的人看不见分支，只会看到"我写了规则，那条 fail 还在"。
    """
    exemptions = ExemptionSet(
        [
            Exemption(
                detector="idempotency",
                tool="create_payment",
                reason="本场景的重复由恢复机制制造，已在别处修",
                allow_write=True,
            )
        ]
    )
    findings = IdempotencyDetector(UNPROTECTED).run(_replay_pair(make_call), exemptions)

    assert "快照恢复重放导致" in findings[0].summary, "走的不是重放那条分支"
    assert findings[0].verdict == DetectorVerdict.PASS
    assert findings[0].exempted_by == "exemption:idempotency/create_payment"
    assert findings[0].detail["would_have_been"] == "fail", "豁免必须留痕"


def test_the_replay_branch_opens_no_default_hole_for_writes(make_call) -> None:
    """接上豁免不等于给写操作开默认口子：没有 `allow_write` 的规则照样不生效。"""
    exemptions = ExemptionSet(
        [Exemption(detector="idempotency", tool="create_payment", reason="老板说没事")]
    )
    findings = IdempotencyDetector(UNPROTECTED).run(_replay_pair(make_call), exemptions)

    assert findings[0].verdict == DetectorVerdict.FAIL
    assert findings[0].exempted_by is None


def test_the_replay_branch_answers_the_answered_facts(make_call) -> None:
    """新接的这个调用点必须把事实传下去：规则拿 `tool_has_idempotency` 当条件时，
    传没传决定它是命中还是死规则（`None` 跟 true、false 都不相等）。

    这一处要是漏传，`find` 会抛 `UnanswerableMatch`——那也得到有人试过才知道。
    """
    exemptions = ExemptionSet(
        [
            Exemption(
                detector="idempotency",
                tool="create_payment",
                reason="这个工具本来就没有幂等保护，重复由上游去重",
                allow_write=True,
                match={"tool_has_idempotency": False},
            )
        ]
    )
    findings = IdempotencyDetector(UNPROTECTED).run(_replay_pair(make_call), exemptions)

    assert findings[0].exempted_by is not None, (
        "规则按 tool_has_idempotency 挑工具，这个调用点没把事实传下去"
    )


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


# --- 声明的主键：第二档判据 ---------------------------------------------------
# 指纹逐字比参数，于是"同一件事、换个付款方式再订一次"配不上对。放宽分组这条路
# 量过（642 条真实轨迹、916 次写操作，`../_traces/measure_same_op.py`）：按
# "同工具"放宽多报 145 组、按"只差一个字段"多报 90 组，逐条看大多是误报；
# 按**人手挑的主键字段**放宽只剩 56 组——而那张表只有懂业务的人给得出来。
# 所以它不是猜出来的，是**声明**出来的：`ToolDecl.identity_fields`。
#
# 这一组测试钉三件事：声明能配上对（下面第一条）、**没声明就一个字都不变**
# （第二条，受控对照）、字段没带齐时不许折进"判过了"（最后一条）。

TRIP = {"user_id": "U1", "flights": ["F1"], "passengers": ["P1"]}
BOOK_IDENTITY = {
    "book_reservation": ToolFacts(
        "book_reservation", True, identity_fields=("user_id", "flights", "passengers")
    )
}


def test_a_declared_key_puts_two_ways_to_pay_in_one_group(make_call) -> None:
    """同一趟行程、两种付款拆法：指纹配不上对，声明的主键配得上。

    这就是那段真实轨迹里漏掉的一对：用户为同一段行程付了两次钱，而三次订票
    两两指纹不同，检测器一条都没报（`docs/detectors.md` 第四节末）。
    """
    calls = [
        make_call(
            1,
            "book_reservation",
            {**TRIP, "payment_methods": ["certificate-50", "card-50"]},
            is_write=True,
        ),
        make_call(2, "book_reservation", {**TRIP, "payment_methods": ["card-100"]}, is_write=True),
    ]
    findings = IdempotencyDetector(BOOK_IDENTITY).run(calls, ExemptionSet())

    fails = [f for f in findings if f.verdict == DetectorVerdict.FAIL]
    assert len(fails) == 1, _verdicts(findings)
    summary = fails[0].summary
    # 判词要说出**凭什么叫同一个操作**，还要说出哪几个字段不同——读者要能自己复核。
    assert "声明的主键" in summary and "user_id、flights、passengers" in summary
    assert "payment_methods" in summary, "没说是哪个字段不同，读者复核不了这条结论"
    assert fails[0].detail["criterion"] == "identity"
    # 主键那一档的 detail 里没有 `fingerprint` 这个键——它不比指纹，硬塞一个空串
    # 会被读成"指纹算出来是空的"。它报的是主键字段和取值。
    assert fails[0].detail["identity_fields"] == ["user_id", "flights", "passengers"]
    assert fails[0].detail["identity_values"] == ["U1", ["F1"], ["P1"]]
    assert "fingerprint" not in fails[0].detail


def test_without_the_declaration_the_same_two_calls_stay_clean(make_call) -> None:
    """受控对照：同一段轨迹，唯一的变量是**声明**。

    少了这一条，"声明"是不是承重的就没人核：把上面那条的实现换成"凡同工具即同
    操作"，两条都会绿，而真实数据上那会多报 145 组。
    """
    calls = [
        make_call(
            1,
            "book_reservation",
            {**TRIP, "payment_methods": ["certificate-50", "card-50"]},
            is_write=True,
        ),
        make_call(2, "book_reservation", {**TRIP, "payment_methods": ["card-100"]}, is_write=True),
    ]
    findings = IdempotencyDetector().run(calls, ExemptionSet())

    assert all(f.verdict != DetectorVerdict.FAIL for f in findings), (
        "没声明主键却把它们配成了一对——那就是量出来的那些误报形状"
    )


def test_a_declared_key_does_not_merge_different_targets(make_call) -> None:
    """声明了主键、主键却不同：各是各的，一条都不许报。

    连着改三张订单的地址（`idem-same-tool-distinct-targets`）在语料里是同一个形状：
    声明**不是**"凡同工具即同操作"的开关。
    """
    identity = {"modify_pending_order_address": ToolFacts(
        "modify_pending_order_address", True, identity_fields=("order_id",)
    )}
    calls = [
        make_call(
            i, "modify_pending_order_address", {"order_id": oid, "address1": "X1"}, is_write=True
        )
        for i, oid in enumerate(("A1", "B2", "C3"), start=1)
    ]
    findings = IdempotencyDetector(identity).run(calls, ExemptionSet())

    assert all(f.verdict != DetectorVerdict.FAIL for f in findings), _verdicts(findings)


def test_an_identical_pair_still_gets_the_exact_wording(make_call) -> None:
    """主键组里指纹若全都一样，走的还是逐字那一档——那句话更硬。

    声明的效果因此只可能是**多看见**：一条已经存在的逐字结论不会被换成措辞更弱的
    另一条（"主键相同"比"参数完全相同"说得少）。
    """
    calls = [
        make_call(1, "book_reservation", {**TRIP, "payment_methods": ["card-100"]}, is_write=True),
        make_call(2, "book_reservation", {**TRIP, "payment_methods": ["card-100"]}, is_write=True),
    ]
    findings = IdempotencyDetector(BOOK_IDENTITY).run(calls, ExemptionSet())

    fail = next(f for f in findings if f.verdict == DetectorVerdict.FAIL)
    assert fail.detail["criterion"] == "exact"
    assert "这几次的规范化参数完全相同" in fail.summary
    assert "声明的主键" not in fail.summary


def test_the_pass_line_says_whether_the_declared_key_was_checked(make_call) -> None:
    """声明了主键、又没报出任何一组时，那句 pass 要说出这一档也跑过了。

    不说的后果和"未发现重复执行"那句一样：读者以为主键这一档没查——或者更糟，
    以为"同一个操作做了两次"这一类也在这句话的覆盖面里，而它其实只比了逐字。
    """
    calls = [
        make_call(1, "book_reservation", {**TRIP, "payment_methods": ["card-100"]}, is_write=True),
        make_call(
            2,
            "book_reservation",
            {**TRIP, "flights": ["F2"], "payment_methods": ["card-100"]},
            is_write=True,
        ),
    ]
    findings = IdempotencyDetector(BOOK_IDENTITY).run(calls, ExemptionSet())

    summary = next(f for f in findings if f.verdict == DetectorVerdict.PASS).summary
    assert "没有两次的规范化参数完全相同" in summary
    assert "声明的主键" in summary, f"pass 那句没提主键这一档跑没跑：{summary}"
    assert "book_reservation" in summary


def test_a_call_missing_a_declared_key_is_inconclusive_not_pass(make_call) -> None:
    """声明了主键、调用却没带那个字段：这一档判据对它**一次都没跑过**。

    不能折进"判过了、没问题"里——那正是本项目最怕的那种句子。也不许拿"字段缺失"
    当"值不同"用：那会让缺字段的调用各自成组，判据在数据不齐时静默退化。
    """
    calls = [
        make_call(1, "book_reservation", {**TRIP, "payment_methods": ["card-100"]}, is_write=True),
        make_call(2, "book_reservation", {"user_id": "U1", "payment_methods": ["card-100"]},
                  is_write=True),
        make_call(3, "book_reservation", {**TRIP, "payment_methods": ["card-100"]}, is_write=True),
    ]
    findings = IdempotencyDetector(BOOK_IDENTITY).run(calls, ExemptionSet())

    # 1 和 3 参数逐字相同 → 逐字那一档照样报 fail（那一档对它们跑过了）；
    # 2 缺 flights/passengers，只按逐字看过，单独报一条 inconclusive。
    assert sorted(_verdicts(findings)) == ["fail", "inconclusive"], _verdicts(findings)
    note = next(f for f in findings if f.verdict == DetectorVerdict.INCONCLUSIVE)
    assert "flights" in note.summary and "passengers" in note.summary
    assert note.detail["seqs"] == [2]


@pytest.mark.parametrize(
    ("value", "is_write", "fragment"),
    [
        (["order_id", "order_id"], True, "重复"),
        (["order_id", "  "], True, "空"),
        ({"order_id": 1}, True, "列表"),
        (["order_id"], False, "只读"),
    ],
)
def test_a_bad_declaration_is_refused_at_load_time(value, is_write: bool, fragment: str) -> None:
    """写坏的声明必须**在加载时**炸，不能等它静默地不生效。

    四种都是"写了等于没写"：字段名重复、字段名为空、写成了字典、声明在只读工具上。
    """
    with pytest.raises(ValueError, match=fragment):
        checked_identity_fields(value, tool="t", is_write=is_write)


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
def _idem_text(
    scenario_id: str,
    *,
    protected: bool,
    arguments: list[str],
    expected_charges: int,
    verdict: str,
    reason_code: str | None,
) -> str:
    tool = (
        """  - name: create_payment
    description: 付款
    is_write: true
    enforces_idempotency: true
    idempotency_param: idempotency_key
"""
        if protected
        else """  - name: create_payment
    description: 付款
    is_write: true
    idempotency_param: idempotency_key
"""
    )
    steps = "".join(
        "      - tool_calls:\n          - name: create_payment\n"
        f"            arguments: {args}\n"
        for args in arguments
    )
    tail = f"reason_code: {reason_code}\n" if reason_code else ""
    return f"""\
id: {scenario_id}
agent:
  version: order_agent@0.1
  directory: order_agent
  task: 处理订单 A1。
  model: stub-model
tools:
  - name: get_order
    description: 查询
    is_write: false
{tool}sessions:
  - name: 一段
    script:
{steps}      - content: 完成。
expected_charges: {expected_charges}
expect:
  - detector: idempotency
    verdict: {verdict}
    {tail}"""


DUPLICATE_SCENARIO = parse_scenario(
    _idem_text(
        "duplicate",
        protected=False,
        arguments=["{order_id: A1, amount: 599.0}", "{order_id: A1, amount: 599.0}"],
        expected_charges=2,
        verdict="fail",
        reason_code=None,
    ),
    source="<内存中的场景>",
)
MISSING_KEY_SCENARIO = parse_scenario(
    _idem_text(
        "missing_key",
        protected=True,
        arguments=["{order_id: A1, amount: 599.0}", "{order_id: A1, amount: 599.0}"],
        expected_charges=2,
        verdict="fail",
        reason_code="missing_key",
    ),
    source="<内存中的场景>",
)
KEY_ROTATED_SCENARIO = parse_scenario(
    _idem_text(
        "key_rotated",
        protected=True,
        arguments=[
            "{order_id: A1, amount: 599.0, idempotency_key: pay-A1-attempt-1}",
            "{order_id: A1, amount: 599.0, idempotency_key: pay-A1-attempt-2}",
        ],
        expected_charges=2,
        verdict="fail",
        reason_code="key_rotated",
    ),
    source="<内存中的场景>",
)
PROTECTED_SCENARIO = parse_scenario(
    _idem_text(
        "protected",
        protected=True,
        arguments=[
            "{order_id: A1, amount: 599.0, idempotency_key: pay-A1-1}",
            "{order_id: A1, amount: 599.0, idempotency_key: pay-A1-1}",
        ],
        expected_charges=1,
        verdict="warning",
        reason_code="protected",
    ),
    source="<内存中的场景>",
)
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
    duplicate = DUPLICATE_SCENARIO
    missing = MISSING_KEY_SCENARIO
    rotated = KEY_ROTATED_SCENARIO
    reused = PROTECTED_SCENARIO

    assert_inline_only_differs(
        DUPLICATE_SCENARIO,
        MISSING_KEY_SCENARIO,
        KEY_ROTATED_SCENARIO,
        PROTECTED_SCENARIO,
        claim=(
            "四支的差别只有：写工具声明里有没有幂等保护、参数表里有没有那个键名、"
            "脚本里写的幂等键值。"
        ),
        differs_in=(
            "tools[n].enforces_idempotency",
            "script[n].script[n].tool_calls[n].arguments.idempotency_key",
        ),
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
    def declared(scenario: Scenario) -> list[tuple[str, str | None]]:
        return [(e.verdict.value, e.reason_code) for e in scenario.expect]

    assert declared(DUPLICATE_SCENARIO) == [("fail", None)]
    assert declared(MISSING_KEY_SCENARIO) == [("fail", "missing_key")]
    assert declared(KEY_ROTATED_SCENARIO) == [("fail", "key_rotated")]
    assert declared(PROTECTED_SCENARIO) == [("warning", "protected")]

    # 地面真值：前三支真的扣了两次钱，第四支真的只扣了一次。
    charges = {
        scenario.id: scenario.expected_charges
        for scenario in (
            DUPLICATE_SCENARIO,
            MISSING_KEY_SCENARIO,
            KEY_ROTATED_SCENARIO,
            PROTECTED_SCENARIO,
        )
    }
    assert charges == {
        "duplicate": 2,
        "missing_key": 2,
        "key_rotated": 2,
        "protected": 1,
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

    rotated = await ScenarioRunner(session_timeout_s=120.0).run(KEY_ROTATED_SCENARIO)
    reused = await ScenarioRunner(session_timeout_s=120.0).run(PROTECTED_SCENARIO)

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
