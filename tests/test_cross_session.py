"""跨会话幂等检测的端到端测试。

这个文件要证明的是需求里那条用户故事：

    用户周一让 Agent 订票，周三再次要求时，Agent 没有识别出这是重复操作。

**但真正要证明的是归因，不是检测。** "检测到重复"本身不难——难的是说清楚该去
修哪儿。同一个现象（A1 被扣了两次）有两种完全相反的成因：

  * Agent 有记忆却没认出来   → 改 Prompt / 改 Agent 的判重逻辑
  * 我们回滚了记忆，它不可能知道 → 改恢复机制，**别动 Agent**

分不清这两者的话，报告就是在指挥用户去改一个没问题的东西——比不报还糟。

分界线的依据是记忆视野，而记忆视野是我们拍快照时定位的，不是推测的。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sandbox.config import settings
from sandbox.db.models import DetectorVerdict
from sandbox.detectors.base import ExemptionSet
from sandbox.detectors.idempotency import IdempotencyDetector, ToolFacts
from sandbox.memory.adapters.file_memory import FileMemoryAdapter
from sandbox.memory.snapshot import SnapshotStore
from sandbox.trace.reader import load_session_sequence
from sandbox.trace.recorder import (
    capture_snapshot,
    record_tool_call,
    restore_snapshot,
    start_run,
    start_session,
)

PAY = {"order_id": "A1", "amount": 599.0}
TOOL_FACTS = {"create_payment": ToolFacts("create_payment", is_write=True)}
WRITES = {"create_payment"}


@pytest.fixture
def memory_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "mem"
    root.mkdir()
    monkeypatch.setattr(settings, "memory_root", root)
    return root


async def _pay(ctx, **kwargs) -> str:
    """扣一次款。

    `is_write` 不用（也不能）传：记录器从 `ctx.declared_writes` 推导，
    不让调用方自己声明——"这次算不算写操作"由工具声明决定，
    交给调用方声明的话，两个地方的说法迟早会不一致。
    """
    return await record_tool_call(ctx, tool_name="create_payment", args=PAY, **kwargs)


def _findings(calls, tool_facts=None):
    detector = IdempotencyDetector(tool_facts or TOOL_FACTS)
    return detector.run(calls, ExemptionSet())


def _fail(findings):
    fails = [f for f in findings if f.verdict == DetectorVerdict.FAIL]
    assert len(fails) == 1, f"期望恰好一条 fail，实际 {[f.summary for f in fails]}"
    return fails[0]


# --- 归因的两条路 -------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_intact_then_repeat_is_the_agents_fault(temp_db: str) -> None:
    """记忆完好，Agent 还是重复了 —— Agent 的锅。

    这是需求原文的场景：它有长效记忆，本应认出来。
    """
    ctx = await start_run(scenario_id="ticket", agent_version="v1", declared_writes=WRITES)
    await _pay(ctx)  # 周一

    await start_session(ctx)  # 自然延续：记得周一做过什么
    await _pay(ctx)  # 周三又来一次

    calls = await load_session_sequence(ctx.session_sequence_id)
    finding = _fail(_findings(calls))

    assert finding.detail["reason_code"] == "cross_session_duplicate"
    assert "本可以认出来" in finding.summary
    assert finding.detail["memory_horizon_ordinal"] == 0
    assert finding.detail["sessions"] == {"0": [1], "1": [2]}


@pytest.mark.asyncio
async def test_memory_rewound_then_repeat_is_the_recoverys_fault(
    temp_db: str, memory_root: Path, tmp_path: Path
) -> None:
    """记忆被回滚过，Agent 不可能知道 —— **别去改 Agent**。

    这是同一种现象的另一半，也是整个跨会话设计存在的理由：同样报 fail
    （钱确实扣了两次），但结论指向完全不同的地方去修。
    """
    store = SnapshotStore(tmp_path / "snaps")
    mem = memory_root / "agent"
    mem.mkdir(parents=True)
    (mem / "facts.md").write_text("无历史", encoding="utf-8")
    adapter = FileMemoryAdapter(mem)

    ctx = await start_run(scenario_id="ticket", agent_version="v1", declared_writes=WRITES)

    # 扣款**之前**拍快照 —— 这份记忆里没有那笔扣款
    before_payment = await capture_snapshot(ctx, adapter, store)

    await _pay(ctx)  # 周一：真的扣了
    (mem / "facts.md").write_text("周一：订了票 A1", encoding="utf-8")

    # 恢复：把记忆退回扣款之前，Agent 于是"忘了"
    await restore_snapshot(before_payment, adapter, store)
    assert "周一" not in (mem / "facts.md").read_text(encoding="utf-8")

    await start_session(ctx, restored_from=before_payment)
    await _pay(ctx)  # 周三：它以为还没订过，又扣了一次

    calls = await load_session_sequence(ctx.session_sequence_id)
    finding = _fail(_findings(calls))

    assert finding.detail["reason_code"] == "cross_session_memory_loss"
    assert "不要改 Agent" in finding.summary
    assert finding.detail["memory_horizon_ordinal"] == -1


@pytest.mark.asyncio
async def test_the_two_paths_differ_only_in_attribution(
    temp_db: str, memory_root: Path, tmp_path: Path
) -> None:
    """**这个文件最要紧的一条断言。**

    两种情形下发生的**事实完全相同**：同一个订单被扣了两次，两次都在库里，
    指纹一致，序号一致。唯一的差别是记忆视野——而结论必须截然不同。

    如果这条测试退化成一个只比 verdict 的断言，那就说明归因又变回了靠猜。
    """
    # 第一条路：记忆完好
    ctx_a = await start_run(scenario_id="a", agent_version="v1", declared_writes=WRITES)
    await _pay(ctx_a)
    await start_session(ctx_a)
    await _pay(ctx_a)
    fault_of_agent = _fail(_findings(await load_session_sequence(ctx_a.session_sequence_id)))

    # 第二条路：记忆被回滚
    store = SnapshotStore(tmp_path / "snaps")
    mem = memory_root / "agent"
    mem.mkdir(parents=True)
    (mem / "facts.md").write_text("无历史", encoding="utf-8")
    adapter = FileMemoryAdapter(mem)

    ctx_b = await start_run(scenario_id="b", agent_version="v1", declared_writes=WRITES)
    snap = await capture_snapshot(ctx_b, adapter, store)
    await _pay(ctx_b)
    await start_session(ctx_b, restored_from=snap)
    await _pay(ctx_b)
    fault_of_recovery = _fail(_findings(await load_session_sequence(ctx_b.session_sequence_id)))

    # 事实相同
    assert fault_of_agent.verdict == fault_of_recovery.verdict == DetectorVerdict.FAIL
    assert fault_of_agent.detail["tool_name"] == fault_of_recovery.detail["tool_name"]
    assert fault_of_agent.detail["fingerprint"] == fault_of_recovery.detail["fingerprint"]
    assert (
        fault_of_agent.detail["sessions"] == fault_of_recovery.detail["sessions"]
    ), "两次的调用分布必须一模一样，否则这个对比就不成立"

    # 归因不同
    assert fault_of_agent.detail["reason_code"] != fault_of_recovery.detail["reason_code"]
    assert "不要改 Agent" not in fault_of_agent.summary
    assert "不要改 Agent" in fault_of_recovery.summary


# --- 不能误报 -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_that_remembers_does_not_re_pay(temp_db: str) -> None:
    """负样本：Agent 认出了重复，没有第二次扣款 —— 一条 fail 都不该有。

    没有这条测试，"跨会话检测"很容易退化成"跨会话时只要调过同一工具就报警"，
    而那种检测器上线第一天就会被关掉。
    """
    ctx = await start_run(scenario_id="ticket", agent_version="v1", declared_writes=WRITES)
    await _pay(ctx)
    await start_session(ctx)
    # 只查了一下，没再扣
    await record_tool_call(ctx, tool_name="get_order", args={"order_id": "A1"})

    calls = await load_session_sequence(ctx.session_sequence_id)
    findings = _findings(calls)

    assert not [f for f in findings if f.verdict == DetectorVerdict.FAIL]
    assert any(f.verdict == DetectorVerdict.PASS for f in findings), (
        "查过没问题必须说出来，否则报告会显示覆盖率 0%，把'测过了'读成'什么都没测'"
    )


@pytest.mark.asyncio
async def test_different_orders_across_sessions_are_not_duplicates(temp_db: str) -> None:
    """不同订单各付一次是正常行为，不是跨会话重复。"""
    ctx = await start_run(scenario_id="ticket", agent_version="v1", declared_writes=WRITES)
    await record_tool_call(
        ctx, tool_name="create_payment", args={"order_id": "A1", "amount": 1.0}
    )
    await start_session(ctx)
    await record_tool_call(
        ctx, tool_name="create_payment", args={"order_id": "A2", "amount": 2.0}
    )

    findings = _findings(await load_session_sequence(ctx.session_sequence_id))
    assert not [f for f in findings if f.verdict == DetectorVerdict.FAIL]


@pytest.mark.asyncio
async def test_blocked_calls_do_not_count_across_sessions(temp_db: str) -> None:
    """被出口代理拦下的那次没到目标系统，不构成跨会话重复扣款。"""
    ctx = await start_run(scenario_id="ticket", agent_version="v1", declared_writes=WRITES)
    await _pay(ctx)
    await start_session(ctx)
    await _pay(ctx, blocked_by_proxy=True)

    findings = _findings(await load_session_sequence(ctx.session_sequence_id))
    assert not [f for f in findings if f.verdict == DetectorVerdict.FAIL]


# --- 视野与顺序：归因的地基 ---------------------------------------------------


@pytest.mark.asyncio
async def test_horizon_is_taken_from_the_restored_snapshot_not_the_ordinal(
    temp_db: str, memory_root: Path, tmp_path: Path
) -> None:
    """恢复不同时刻的快照，得到不同的视野——归因跟着快照走，不跟着会话号走。

    会话号是 2，但恢复的是"会话 0 之后"的快照，所以它记得会话 0、不记得会话 1。
    这正是"缺的那一段就是 Agent 不可能知道的部分"。
    """
    store = SnapshotStore(tmp_path / "snaps")
    mem = memory_root / "agent"
    mem.mkdir(parents=True)
    adapter = FileMemoryAdapter(mem)

    ctx = await start_run(scenario_id="t", agent_version="v1", declared_writes=WRITES)
    await _pay(ctx)  # 会话 0 的一次调用 → 视野到 (0, 1)
    after_session_0 = await capture_snapshot(ctx, adapter, store)
    assert after_session_0.position == (0, 1)

    await start_session(ctx)
    await _pay(ctx)  # 会话 1 又扣了一次（此时会报跨会话）

    session = await start_session(ctx, restored_from=after_session_0)
    assert session.ordinal == 2
    assert session.memory_horizon_ordinal == 0, "恢复的是会话 0 之后的快照 → 记得会话 0"

    ctx2 = await start_run(scenario_id="t2", agent_version="v1", declared_writes=WRITES)
    empty = await capture_snapshot(ctx2, adapter, store)  # (0, 0)
    await start_session(ctx2)
    third = await start_session(ctx2, restored_from=empty)
    assert third.memory_horizon_ordinal == -1, "恢复到会话 0 开始前 → 什么都不记得"


@pytest.mark.asyncio
async def test_missed_ordinals_are_named_in_the_report(
    temp_db: str, memory_root: Path, tmp_path: Path
) -> None:
    """报告要说清楚"丢的是哪一段"，而不是笼统地说记忆对不上。"""
    store = SnapshotStore(tmp_path / "snaps")
    mem = memory_root / "agent"
    mem.mkdir(parents=True)
    adapter = FileMemoryAdapter(mem)

    ctx = await start_run(scenario_id="t", agent_version="v1", declared_writes=WRITES)
    baseline = await capture_snapshot(ctx, adapter, store)  # (0, 0)，什么都不记得
    await _pay(ctx)  # 会话 0

    await start_session(ctx)
    await _pay(ctx)  # 会话 1
    await _pay(ctx)

    await start_session(ctx, restored_from=baseline)  # 会话 2：退回到最初
    await _pay(ctx)

    finding = _fail(_findings(await load_session_sequence(ctx.session_sequence_id)))
    assert finding.detail["missed_ordinals"] == [0, 1]
    assert "被丢掉了" in finding.summary


# --- 豁免仍然管用 -------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_session_write_cannot_be_silenced_without_allow_write(temp_db: str) -> None:
    """写操作的跨会话重复默认不可豁免——豁免规则是让重复扣款静音的最短路径。"""
    from sandbox.detectors.base import Exemption

    ctx = await start_run(scenario_id="t", agent_version="v1", declared_writes=WRITES)
    await _pay(ctx)
    await start_session(ctx)
    await _pay(ctx)

    calls = await load_session_sequence(ctx.session_sequence_id)
    findings = IdempotencyDetector(TOOL_FACTS).run(
        calls,
        ExemptionSet(
            [
                Exemption(
                    detector="idempotency",
                    reason="以为写操作也能豁免",
                    tool="create_payment",
                    # 没写 allow_write=True
                )
            ]
        ),
    )
    assert [f for f in findings if f.verdict == DetectorVerdict.FAIL]

    # 显式声明之后才允许静音，而且必须留痕
    findings = IdempotencyDetector(TOOL_FACTS).run(
        calls,
        ExemptionSet(
            [
                Exemption(
                    detector="idempotency",
                    reason="这个订单确实是用户要求重复支付的",
                    tool="create_payment",
                    allow_write=True,
                )
            ]
        ),
    )
    silenced = [f for f in findings if f.exempted_by]
    assert silenced and silenced[0].verdict == DetectorVerdict.PASS
    assert silenced[0].detail["would_have_been"] == "fail"


@pytest.mark.asyncio
async def test_snapshot_metadata_survives_the_round_trip(
    temp_db: str, memory_root: Path, tmp_path: Path
) -> None:
    """快照的元数据要能原样读回来——报告与人工排查都依赖它。"""
    store = SnapshotStore(tmp_path / "snaps")
    mem = memory_root / "agent"
    mem.mkdir(parents=True)
    (mem / "a.md").write_text("内容", encoding="utf-8")
    adapter = FileMemoryAdapter(mem)

    ctx = await start_run(scenario_id="t", agent_version="v1")
    ref = await capture_snapshot(ctx, adapter, store)

    listed = await store.list_for_sequence(ctx.session_sequence_id)
    assert [r.id for r in listed] == [ref.id]
    assert listed[0].entry_count == 1
    assert listed[0].size_bytes > 0

    payload = json.loads((await store.load(ref.id)).decode("utf-8"))
    assert [e["path"] for e in payload["entries"]] == ["a.md"]
