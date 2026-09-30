"""记录器是网关与工具代理共用的唯一写入口，它写错字段，所有检测器都跟着错。"""

from __future__ import annotations

import ast
import json

from sqlalchemy import select

from sandbox.config import REPO_ROOT
from sandbox.db.engine import session_scope
from sandbox.db.models import DetectorFinding, DetectorVerdict, LlmCall, Session, TestRun, ToolCall
from sandbox.detectors.idempotency import SNAPSHOT_REPLAY
from sandbox.trace.recorder import (
    CONFIG_SCHEME,
    RunContext,
    finish_run,
    recompute_totals,
    record_finding,
    record_llm_call,
    record_tool_call,
    start_run,
)


async def _make_run(**kwargs) -> RunContext:
    defaults = {
        "scenario_id": "scen-1",
        "agent_version": "react_demo@0.1",
        "config": {"prompt": "v1", "tools": ["create_payment"]},
    }
    return await start_run(**{**defaults, **kwargs})


async def test_start_run_persists_run_and_first_session(db) -> None:
    ctx = await _make_run()

    run = await db.get(TestRun, ctx.run_id)
    assert run is not None
    assert run.scenario_id == "scen-1"
    assert run.status == "running"
    assert len(run.config_hash) == 64

    sess = await db.get(Session, ctx.session_id)
    assert sess is not None
    assert sess.session_sequence_id == ctx.session_sequence_id


async def test_config_hash_is_content_addressed(temp_db: str) -> None:
    """同样的配置必须得到同样的哈希——两次运行可比的前提。"""
    a = await _make_run(config={"prompt": "v1", "tools": ["create_payment"]})
    b = await _make_run(config={"tools": ["create_payment"], "prompt": "v1"})  # 键序不同
    c = await _make_run(config={"prompt": "v2", "tools": ["create_payment"]})

    async with session_scope() as db_:
        ra, rb, rc = [await db_.get(TestRun, x.run_id) for x in (a, b, c)]
        assert ra.config_hash == rb.config_hash, "键序不该影响配置哈希"
        assert ra.config_hash != rc.config_hash, "改了 prompt 必须换哈希"


async def test_start_run_records_scheme_and_original_parts(temp_db: str) -> None:
    """口径与分节原值都要落库，缺一个回归对比就会说谎。

    只有哈希能说"变了"，但说不出"改的是哪一块"；只有原值能定位到节，但两节
    长得一样时又分不出是同一份还是两份。所以两个都存。

    口径也必须存：**两个不同口径算出的哈希不相等，不能读成"配置变了"**。

    比的是常量本身，不是写死的字面量。这里问的是"存进去了没有"，而写死的
    话，每一次口径升版都会让这条与升版无关的测试变红——那种红是噪音，而噪音
    会训练人无视红色。**口径的值在哪一处被钉住是刻意的**：只有
    `test_identity.py::test_the_scheme_version_is_recorded` 一处，理由写在
    那里（升版必须是有人签字同意的动作）。
    """
    ctx = await _make_run()
    async with session_scope() as db_:
        run = await db_.get(TestRun, ctx.run_id)
        assert run.config_scheme == CONFIG_SCHEME
        # 存的是**原值**，不是"给值算个哈希"——差异报告要引用它说出改了什么。
        assert json.loads(run.config_parts_json) == {
            "prompt": "v1",
            "tools": ["create_payment"],
        }


async def test_run_without_config_says_so_rather_than_claiming_unchanged(temp_db: str) -> None:
    """没带配置开出来的运行（测试、探活）不能假装自己有一份配置。

    它记的是空配置，于是和任何真配置比对都会得出"全变了"——那是**正确**的
    答案，因为确实无从比较。反过来若在这里塞一份假配置，就会凭空造出
    "配置没变"的依据。
    """
    ctx = await _make_run(config=None, env=None)
    async with session_scope() as db_:
        run = await db_.get(TestRun, ctx.run_id)
        assert json.loads(run.config_parts_json) == {}
        assert json.loads(run.env_json) == {}


async def test_tool_call_args_are_normalized_before_storage(db) -> None:
    """噪声字段必须被剔除后再入库，否则近似层比较会被噪声淹没。"""
    ctx = await _make_run(declared_writes={"create_payment"})
    await record_tool_call(
        ctx,
        tool_name="create_payment",
        args={"order_id": " A1 ", "amount": 100, "request_id": "req-abc", "timestamp": 123},
    )

    call = (await db.execute(select(ToolCall))).scalar_one()
    stored = json.loads(call.args_json)
    assert stored == {"order_id": "A1", "amount": 100}, (
        "request_id/timestamp 应被剔除，空白应被去除"
    )
    assert call.is_write is True
    assert len(call.arg_hash) == 64
    assert call.fingerprint == call.arg_hash, "当前指纹与结构哈希同构，见 normalize.py 的说明"


async def test_idempotency_key_survives_normalization(db) -> None:
    """幂等键必须在**原始**参数里找——规范化会把它当噪声剔掉。

    这是很容易写错的一处：先规范化再找幂等键，就永远找不到。
    """
    ctx = await _make_run()
    await record_tool_call(
        ctx,
        tool_name="create_payment",
        args={"order_id": "A1", "idempotency_key": "idem-777", "timestamp": 1},
    )

    call = (await db.execute(select(ToolCall))).scalar_one()
    assert call.idempotency_key == "idem-777"
    assert "idempotency_key" not in json.loads(call.args_json)


async def test_declared_idempotency_param_beats_alias_table(db) -> None:
    ctx = await _make_run(idempotency_key_params={"create_payment": "client_token"})
    await record_tool_call(ctx, tool_name="create_payment", args={"client_token": "tok-1"})

    call = (await db.execute(select(ToolCall))).scalar_one()
    assert call.idempotency_key == "tok-1"


async def test_seq_is_a_single_timeline_across_tables(db) -> None:
    """LLM 调用与工具调用共用一条时间线——顺序是归因的基础。"""
    ctx = await _make_run()
    await record_tool_call(ctx, tool_name="get_order", args={"order_id": "A1"})
    await record_llm_call(ctx, model="stub-model", input_tokens=100, output_tokens=20)
    await record_tool_call(ctx, tool_name="create_payment", args={"order_id": "A1", "amount": 100})

    calls = (await db.execute(select(ToolCall).order_by(ToolCall.seq))).scalars().all()
    llms = (await db.execute(select(LlmCall))).scalars().all()

    assert [c.seq for c in calls] == [1, 3]
    assert [x.seq for x in llms] == [2]


async def test_llm_call_cost_comes_from_pricing_not_caller(db) -> None:
    ctx = await _make_run()
    priced = await record_llm_call(
        ctx, model="deepseek-chat", input_tokens=1_000_000, output_tokens=0
    )

    # 定价未经核实，所以 priced 为 False，但金额仍要算出来供链路验证
    assert priced["usd"] > 0
    assert priced["priced"] is False

    row = (await db.execute(select(LlmCall))).scalar_one()
    assert row.cost_usd == priced["usd"]


async def test_unknown_model_is_unpriced_not_guessed(db) -> None:
    """未定价模型记 0.0 并说明原因——猜一个数字会污染成本对比且没人会发现。"""
    ctx = await _make_run()
    result = await record_llm_call(ctx, model="stub-model", input_tokens=500, output_tokens=100)

    assert result["usd"] == 0.0
    assert result["priced"] is False
    assert "不在定价表中" in result["reason"]


async def test_recompute_totals_matches_detail_rows(db) -> None:
    ctx = await _make_run()
    await record_llm_call(ctx, model="deepseek-chat", input_tokens=1000, output_tokens=200)
    await record_llm_call(ctx, model="deepseek-chat", input_tokens=3000, output_tokens=400)

    totals = await recompute_totals(ctx.run_id)
    assert totals["input_tokens"] == 4000
    assert totals["output_tokens"] == 600
    assert totals["llm_calls"] == 2

    await finish_run(ctx, totals=totals)
    run = await db.get(TestRun, ctx.run_id)
    assert run.total_input_tokens == 4000
    assert run.status == "completed"
    assert run.completed_at is not None


async def test_finding_carries_four_state_verdict_and_exemption(db) -> None:
    ctx = await _make_run()
    await record_finding(
        ctx,
        detector="idempotency",
        verdict=DetectorVerdict.INCONCLUSIVE,
        summary="未观测到任何写操作，无法判定幂等性",
        detail={"observed_writes": 0},
    )
    await record_finding(
        ctx,
        detector="loop",
        verdict=DetectorVerdict.PASS,
        summary="未发现循环",
        exempted_by="rule:pagination-whitelist",
    )

    rows = (
        (await db.execute(select(DetectorFinding).order_by(DetectorFinding.detector)))
        .scalars()
        .all()
    )
    assert [r.verdict for r in rows] == ["inconclusive", "pass"]
    assert rows[0].exempted_by is None
    assert rows[1].exempted_by == "rule:pagination-whitelist", "豁免必须留痕，不能静默"
    assert json.loads(rows[0].detail_json) == {"observed_writes": 0}


async def test_blocked_by_proxy_is_recorded_distinctly(db) -> None:
    """被出口代理拒绝的调用要能与 Agent 自身失败区分——否则污染检测结果。"""
    ctx = await _make_run(declared_writes={"create_payment"})
    await record_tool_call(
        ctx,
        tool_name="create_payment",
        args={"order_id": "A1"},
        status="blocked",
        error="egress denied: host not in allowlist",
        blocked_by_proxy=True,
    )

    call = (await db.execute(select(ToolCall))).scalar_one()
    assert call.blocked_by_proxy is True
    assert call.status == "blocked"


async def test_replay_reason_separates_agent_retry_from_snapshot_restore(db) -> None:
    """快照恢复导致的重放不是 Agent 的错——混在一起会冤枉被测 Agent。"""
    ctx = await _make_run()
    await record_tool_call(
        ctx, tool_name="create_payment", args={"order_id": "A1", "amount": 100}
    )
    await record_tool_call(
        ctx,
        tool_name="create_payment",
        args={"order_id": "A1", "amount": 100},
        replay_reason="snapshot_restore",
    )

    calls = (await db.execute(select(ToolCall).order_by(ToolCall.seq))).scalars().all()
    assert calls[0].fingerprint == calls[1].fingerprint, "严格层应认出这是同一次调用"
    assert calls[0].replay_reason is None
    assert calls[1].replay_reason == "snapshot_restore"


# --- 谁给"重放"打标记：今天没有生产调用方，这条是绊线不是断言 -------------------


def test_no_live_caller_writes_a_replay_marker() -> None:
    """绊线：`replay_reason` 今天没有任何生产调用方，谁接上第一个，这条就红。

    `record_tool_call` 是工具调用唯一的写入口，所以"有没有人写下这个标记"在这
    一个函数的调用点上就能问完。扫的是源码不是运行时数据：跑一遍场景只能证明
    **这一次**没有标记，而这条声明说的是**生产路径上根本没有那个调用方**。

    为什么要有这条线：`docs/detectors.md` 第三节第 3 条、`docs/longrun.md` 的
    不支持清单、`call_chain.py` 的 `MARK_REPLAYED`、以及上面那条直接构造带标记
    调用的单测，它们合起来在讲一句话——**"这一次是重放、不是它主动重发的"今天
    报不出来，那个标记的位置是给将来的沙箱驱动重放源备的**。谁真接上了第
    一个生产调用方，这句话就不成立了：报告会开始印"紧接着是一次快照重放"，而
    文档还在说这行印不出来。绊线在这里拦住他，逼他回去把三处说法一起改掉。

    反过来说，**不要为了让这条测试绿而把生产调用方挪到 `**kwargs` 里**——那只是
    让绊线抓不到，标记本身照样进了库，读报告的人照样会以为一次恢复机制制造的
    重复扣款是 Agent 自己干的。
    """
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        rel = path.relative_to(REPO_ROOT)
        text = path.read_text(encoding="utf-8")
        # 一：显式关键字 `record_tool_call(..., replay_reason=...)`。
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Call):
                func = node.func
                called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if called == "record_tool_call" and any(
                    kw.arg == "replay_reason" for kw in node.keywords
                ):
                    offenders.append(f"{rel}:{node.lineno}")
            # 二：把标记的常量从检测器包里引出来——接生产调用方时最常见的写法。
            elif isinstance(node, ast.ImportFrom) and "detectors" in (node.module or "").split("."):
                if any(alias.name == "SNAPSHOT_REPLAY" for alias in node.names):
                    offenders.append(f"{rel}:{node.lineno}（把重放标记的常量引了出来）")
        # 三：把标记的值直接写进源码。前两条都走的是"认名字"，这一条认值，给
        # 绕开名字的写法（比如塞进一个不叫 replay_reason 的字典）兜底。
        if path.parent.name != "detectors" and SNAPSHOT_REPLAY in text:
            offenders.append(f"{rel}（写下了 {SNAPSHOT_REPLAY!r}）")

    assert not offenders, (
        "src/ 里出现了写下重放标记的生产调用方：\n    "
        + "\n    ".join(offenders)
        + "\n先读 docs/detectors.md 第三节第 3 条：这条归因只在标记由我们打的时候"
        "才成立，被测对象的自报不算证据。改了它，docs/longrun.md 的不支持清单和"
        "call_chain.py 的 MARK_REPLAYED 那两处说法也要一起改。"
    )
