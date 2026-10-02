"""Trace 记录器：把边界上截获的调用落成结构化实体（ADR 002）。

**它是网关与工具代理共用的唯一写入口。** 两条边界拦截路径（ADR 001）都必须
经由这里入库，否则"Agent 到底做了什么"就会出现两个真相源。

两个设计点值得单独说明：

1. `seq` 是**运行内全局单调序号**，LLM 调用与工具调用共用同一条时间线。
   分表各自计数的话，"Agent 先查订单再重复扣款"这个顺序在报告里就拼不出来，
   而顺序恰恰是循环检测与归因的基础。两个表各有一个 `seq` 索引，共用计数器
   不会冲突——因为计数器只保证单调，不保证连号。

2. 记录器**不做任何判定**。它只负责忠实落库，判断留给检测器。唯一的例外是
   `is_write` 与 `fingerprint` 这类派生字段——它们是规范化结果，不是结论。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select

from sandbox.db.engine import session_scope
from sandbox.db.models import (
    Confidence,
    DetectorFinding,
    DetectorVerdict,
    LlmCall,
    RunStatus,
    Session,
    TestRun,
    ToolCall,
)
from sandbox.gateway.pricing import compute_cost
from sandbox.memory.snapshot import MemoryAdapter, SnapshotRef, SnapshotStore
from sandbox.trace.normalize import fingerprint, looks_like_write, normalize_args, structural_hash

# 工具参数里可能承载幂等键的字段名。场景可显式声明，这里是兜底别名表。
_IDEMPOTENCY_FIELD_ALIASES: tuple[str, ...] = (
    "idempotency_key",
    "idempotencykey",
    "idempotency-key",
    "idem_key",
)


#: `config_hash` 的口径版本。
#:
#: **改这个常量的定义时必须一起改它。** 只有口径不同，两个哈希不相等才不会被
#: 读成"配置变了"——而那是回归对比里最要命的一种误读：它会把"换了算法"报成
#: "改了 Prompt"，用户于是去查一个没动过的文件。
#:
#: 1（隐式，未记录）：只覆盖场景 id 与模型名。改了脚本或 Prompt 它都不变。
#: 2：覆盖 `scenario/identity.py` 分出的那五节，改动可定位到具体是哪一块。
#: 3：会话多了 `after_gap_s`，于是每一节的**内容**都变了（`script` 节直接
#:    序列化整个 `SessionSpec`）。**这正是这个常量存在的场合**——口径改了、
#:    哈希必然不相等，而它必须被读成"换算法了"而不是"改了脚本"。不升版的话，
#:    一个没动过的场景在升级前后各跑一次会被报成"测试变了"，是本末倒置的假警报。
#: 4：写操作多了 `may_run_unattended`（`tools` 节）、会话多了 `autonomous`
#:    （`script` 节），于是**又一次**每一节的内容都变了。与 3 同一个场合。
#: 5：工具声明多了 `identity_fields`（`tools` 节）。这一节的载荷是整份 `ToolDecl`
#:    序列化出来的，所以**每份场景**的 `tools` 节都换了内容，连一行都没改过的
#:    也一样。与 3、4 同一个场合，区别只在这次变的是一节：不升版的话，同一份
#:    场景在升级前后各跑一次会被报成"改了工具声明"，用户会去查一个没动过的文件。
CONFIG_SCHEME = "5"


def compute_config_hash(payload: dict[str, Any]) -> str:
    """对运行配置做内容寻址。

    需求要求"差异项可定位到具体的 Prompt 或工具改动"。没有这个哈希，
    两次运行的结果差异就只能靠人回忆上次改了什么。

    传进来的 payload 就是 `identity.sections()` 的那五节，所以这个哈希覆盖
    **整个配置**；单节的定位靠 `config_parts_json` 里的原值比对，不靠它。
    """
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class RunContext:
    """一次运行的记录上下文。网关与代理各自持有一个，指向同一个 run。"""

    run_id: str
    session_id: str
    session_sequence_id: str
    declared_writes: set[str] | None = None
    # 工具 → 幂等键参数名。由工具声明提供，缺省时走别名表。
    idempotency_key_params: dict[str, str] = field(default_factory=dict)
    # 当前会话在序列中的序号，以及**这个会话开始时 Agent 的记忆覆盖到哪**。
    # 后者由 start_session 依据快照位置写入，检测器的跨会话归因全靠它。
    session_ordinal: int = 0
    memory_horizon_ordinal: int = -1

    _seq: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @property
    def current_seq(self) -> int:
        """当前时间线位置。取快照时要记下它，快照才定位得到。"""
        return self._seq

    async def next_seq(self) -> int:
        """取下一个时间线序号。锁是必需的：网关是并发的。"""
        async with self._lock:
            self._seq += 1
            return self._seq

    def extract_idempotency_key(self, tool_name: str, raw_args: dict[str, Any]) -> str | None:
        """从原始参数里找幂等键。

        注意是在**原始**参数上找，不是规范化后的——`normalize_args` 会把
        `idempotency_key` 当噪声剔除，所以规范化之后再找就永远找不到了。
        """
        declared = self.idempotency_key_params.get(tool_name)
        candidates = (
            (declared, *_IDEMPOTENCY_FIELD_ALIASES) if declared else _IDEMPOTENCY_FIELD_ALIASES
        )
        lowered = {k.lower(): v for k, v in raw_args.items()}
        for name in candidates:
            if name and name.lower() in lowered:
                value = lowered[name.lower()]
                return str(value) if value is not None else None
        return None


async def start_run(
    *,
    scenario_id: str,
    agent_version: str,
    config: dict[str, Any] | None = None,
    env: dict[str, Any] | None = None,
    session_sequence_id: str | None = None,
    declared_writes: set[str] | None = None,
    idempotency_key_params: dict[str, str] | None = None,
) -> RunContext:
    """建运行并开第一个会话。"""
    run_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    seq_id = session_sequence_id or str(uuid.uuid4())

    async with session_scope() as db:
        db.add(
            TestRun(
                id=run_id,
                scenario_id=scenario_id,
                agent_version=agent_version,
                config_hash=compute_config_hash(config or {}),
                # 口径与分节原值一起落库：前者让"两种口径不相等"能和"配置真变了"
                # 分开，后者让差异报告能说出改的是哪一块。
                config_scheme=CONFIG_SCHEME,
                config_parts_json=json.dumps(config or {}, ensure_ascii=False, default=str),
                session_sequence_id=seq_id,
                status=RunStatus.RUNNING.value,
                env_json=json.dumps(env or {}, ensure_ascii=False, default=str),
            )
        )
        db.add(
            Session(
                id=session_id,
                test_run_id=run_id,
                session_sequence_id=seq_id,
                ordinal=0,
            )
        )

    return RunContext(
        run_id=run_id,
        session_id=session_id,
        session_sequence_id=seq_id,
        declared_writes=declared_writes,
        idempotency_key_params=idempotency_key_params or {},
    )


async def start_session(
    ctx: RunContext,
    *,
    ordinal: int | None = None,
    restored_from: SnapshotRef | None = None,
) -> Session:
    """进入会话序列的下一个会话。**跨会话测试的入口。**

    归因规则在这里落地，它是整个跨会话检测的地基：

    - 自然延续（`restored_from is None`）时，新会话记得此前发生的一切，
      `memory_horizon_ordinal = ordinal - 1`。此后再出现重复，Agent 本可以
      知道，所以是 Agent 的问题。
    - 从旧快照恢复时，记忆只覆盖到快照那一刻。**快照之后发生的调用，
      这个会话不可能知道**——再出现重复就是恢复机制制造的。

    也就是说"是谁的锅"不需要猜、也不需要启发式，它由我们回滚到了哪里
    唯一确定。这也正是把记忆快照做成操纵面（而不是观测面）换来的东西。

    当前的 `ctx.session_id` 会被切换成新会话，此后所有调用都记在新会话下。
    """
    new_ordinal = ctx.session_ordinal + 1 if ordinal is None else ordinal
    if new_ordinal <= ctx.session_ordinal:
        # 序号必须递增。允许回退的话，读者按 (ordinal, seq) 排序时会得到
        # 一个和真实时间线不符的顺序，而跨会话归因完全建立在顺序上。
        raise ValueError(
            f"新会话序号 {new_ordinal} 不大于当前序号 {ctx.session_ordinal}，"
            f"会话序列必须单调递增。"
        )

    horizon = new_ordinal - 1 if restored_from is None else horizon_of(restored_from)

    session = Session(
        id=str(uuid.uuid4()),
        test_run_id=ctx.run_id,
        session_sequence_id=ctx.session_sequence_id,
        ordinal=new_ordinal,
        memory_horizon_ordinal=horizon,
        restored_from_snapshot_id=restored_from.id if restored_from else None,
    )
    async with session_scope() as db:
        db.add(session)

    ctx.session_id = session.id
    ctx.session_ordinal = new_ordinal
    ctx.memory_horizon_ordinal = horizon
    return session


def horizon_of(ref: SnapshotRef) -> int:
    """快照反映到哪个会话序号为止。

    位置 (k, 0) 表示"序号 k 的会话开始**之前**"——那时序号 k 还什么都没发生，
    所以视野停在 k-1。位置 (k, s>0) 表示"序号 k 进行到 s 之后"，视野到 k。

    边界快照的两种取法正好对应这两支：会话开始前取 (k, 0)，会话结束后取 (k, 末序号)。
    """
    return ref.taken_at_ordinal if ref.taken_at_seq > 0 else ref.taken_at_ordinal - 1


async def capture_snapshot(
    ctx: RunContext,
    adapter: MemoryAdapter,
    store: SnapshotStore,
) -> SnapshotRef:
    """在当前位置给 Agent 的记忆拍一张快照。

    `taken_at_seq` 取当前时间线位置，所以这张快照的精确定位是
    "序号 ctx.session_ordinal 进行到 ctx.current_seq 之后"。恢复它意味着
    忘记这之后的一切。
    """
    payload = adapter.capture()
    return await store.save(
        session_sequence_id=ctx.session_sequence_id,
        taken_at_ordinal=ctx.session_ordinal,
        taken_at_seq=ctx.current_seq,
        adapter=adapter.name,
        payload=payload,
        entry_count=adapter.entry_count(),
    )


async def restore_snapshot(
    ref: SnapshotRef,
    adapter: MemoryAdapter,
    store: SnapshotStore,
    *,
    allow_create: bool = False,
) -> None:
    """把 Agent 的记忆回滚到快照状态。

    **这是破坏性的**，会清空并重建适配器指向的记忆目录。三道闸在适配器里
    （见 memory/adapters/file_memory.py），这里不重复判断。

    `allow_create` 默认 False：记忆目录不存在时拒绝恢复。这是对的默认——
    "目录不在那儿"最常见的原因是**路径搞错了**，而那时恢复等于在一个别的地方
    凭空造一份记忆出来，删掉的可能是无关的东西。

    场景执行器会传 True，因为那个目录是它自己开出来、自己注入给 Agent 的：
    它知道路径是对的，"目录还没有"只是因为 Agent 这次一个字都没往记忆里写。
    """
    payload = await store.load(ref.id)
    adapter.restore(payload, allow_create=allow_create)


async def finish_run(
    ctx: RunContext,
    *,
    status: RunStatus = RunStatus.COMPLETED,
    totals: dict[str, float] | None = None,
) -> None:
    """收尾：写状态、结束会话、汇总 token 与成本。

    汇总是从明细表**重算**的，不是由调用方传进来的累计值——传进来的话，
    漏记一次调用就会让总额与明细对不上，而这种不一致极难发现。
    """
    async with session_scope() as db:
        run = await db.get(TestRun, ctx.run_id)
        if run is None:
            raise KeyError(f"run {ctx.run_id} 不存在")

        totals = totals or {}
        run.status = status.value
        run.completed_at = datetime.now(UTC)
        run.total_input_tokens = int(totals.get("input_tokens", 0))
        run.total_output_tokens = int(totals.get("output_tokens", 0))
        run.total_cost_usd = float(totals.get("cost_usd", 0.0))

        sess = await db.get(Session, ctx.session_id)
        if sess is not None:
            sess.ended_at = datetime.now(UTC)
            sess.status = "ended"


async def mark_run_failed(run_id: str) -> None:
    """把运行标成失败。**给异常收尾用。**

    另一条路径是 `finish_run`，但它要一个 `RunContext`——执行中途抛异常的
    时候那个上下文可能压根还没建起来。没有这个函数，崩掉的运行会永远停在
    `running`，而界面会把"永远进行中"显示给用户。
    """
    async with session_scope() as db:
        run = await db.get(TestRun, run_id)
        if run is None:
            return
        run.status = RunStatus.FAILED.value
        run.completed_at = datetime.now(UTC)


async def record_verdict(run_id: str, *, ok: bool, report: dict[str, Any]) -> None:
    """把判定结果与完整报告写回运行记录。

    **刻意不在 `finish_run` 里做。** `finish_run` 发生在执行结束那一刻，而
    结论要等检测器跑完、预期比对完才知道——中间隔着"判断"这一步。合并的话，
    写下去的那个字段就同时背着"跑完了"和"通过了"两种含义，而它们会分开：
    `loop` 场景里进程是被我们亲手熔断杀掉的（执行失败），但那次运行是
    **通过**的。

    `report` 存的是 `ScenarioResult.to_dict()` 的 JSON。写入用 `default=str`
    兜底：报告是**记录**，里面有任何一个字段不是 JSON 原生类型，都不该让
    整次运行的结论丢在地上。兜底转出来的字符串不好看，但比 lost update 好。

    读侧（`api/queries.py`）必须把 NULL 当成"结论未留存"，不是"不通过"。
    """
    async with session_scope() as db:
        run = await db.get(TestRun, run_id)
        if run is None:
            raise KeyError(f"run {run_id} 不存在")
        run.verdict_ok = ok
        run.report_json = json.dumps(report, ensure_ascii=False, default=str)


async def recompute_totals(run_id: str) -> dict[str, float]:
    """从 LlmCall 明细重算运行总额。

    报告里的数字必须能追溯到明细行。这是"成本对得上"的唯一保证。
    """
    async with session_scope() as db:
        stmt = select(
            func.coalesce(func.sum(LlmCall.input_tokens), 0),
            func.coalesce(func.sum(LlmCall.output_tokens), 0),
            func.coalesce(func.sum(LlmCall.cost_usd), 0.0),
            func.count(LlmCall.id),
        ).where(LlmCall.test_run_id == run_id)
        row = (await db.execute(stmt)).one()
        return {
            "input_tokens": float(row[0]),
            "output_tokens": float(row[1]),
            "cost_usd": float(row[2]),
            "llm_calls": float(row[3]),
        }


async def record_tool_call(
    ctx: RunContext,
    *,
    tool_name: str,
    args: dict[str, Any] | None = None,
    result_bytes: int | None = None,
    status: str = "ok",
    error: str | None = None,
    latency_ms: int = 0,
    replay_reason: str | None = None,
    blocked_by_proxy: bool = False,
    seq: int | None = None,
    is_write: bool | None = None,
) -> str:
    """落一条工具调用。返回其 id，便于检测器在 detail_json 里引用。

    `is_write=None` 时按工具名推（声明优先，见 `looks_like_write`）。**传了就以
    传进来的为准**——那是路由层掌握的事实，名字里读不出来：通用出口转发的是一次
    POST，而它记在哪个名字下由被测对象说（甚至可能什么都没说）。按名字推的话，
    一次往任意地址发的 POST 会被记成读操作，于是幂等和主动行为审计都看不见它
    ——报告上完全正常，只是少了一次扣款。

    `result_bytes` 是**返回体的字节数，不是返回体**。参数名和类型都写成这样是
    故意的：签名里没有地方能塞正文，想塞的人会在 mypy 那一步就被拦下。体量由
    调用方量（它手里才有真正发出去的那份字节），见 `proxy/app.py` 里那句注释
    ——用自己 `json.dumps` 一遍的办法量的话，分隔符和转义的差异会让报告里的
    体量与 Agent 实际收到的永远对不上。
    """
    raw_args = args or {}
    normalized = normalize_args(raw_args)
    arg_hash = structural_hash(tool_name, normalized)

    call_id = str(uuid.uuid4())
    assigned_seq = seq if seq is not None else await ctx.next_seq()

    async with session_scope() as db:
        db.add(
            ToolCall(
                id=call_id,
                test_run_id=ctx.run_id,
                session_id=ctx.session_id,
                session_sequence_id=ctx.session_sequence_id,
                seq=assigned_seq,
                tool_name=tool_name,
                args_json=json.dumps(normalized, ensure_ascii=False, default=str),
                arg_hash=arg_hash,
                fingerprint=fingerprint(tool_name, normalized),
                result_bytes=result_bytes,
                status=status,
                error=error,
                is_write=(
                    is_write
                    if is_write is not None
                    else looks_like_write(tool_name, ctx.declared_writes)
                ),
                idempotency_key=ctx.extract_idempotency_key(tool_name, raw_args),
                replay_reason=replay_reason,
                blocked_by_proxy=blocked_by_proxy,
                latency_ms=latency_ms,
            )
        )
    return call_id


async def record_llm_call(
    ctx: RunContext,
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    latency_ms: int = 0,
    streaming: bool = False,
    status: str = "ok",
    request_bytes: int | None = None,
    seq: int | None = None,
) -> dict[str, Any]:
    """落一条 LLM 调用。token 数取自网关的权威计数。

    成本在**这里**算而不是由调用方传：让调用方传成本，等于允许两处算法漂移。
    返回 cost breakdown，供网关/报告标注"未定价"。

    `request_bytes` 由网关量（`gateway/app.py` 里那句 `len(await
    request.body())`）——它是**收到的那份字节**，不是重新序列化一遍的长度；
    重编码会改掉键序、空格和转义，量出来的就不是账单上的那个量了。None 表示
    没量到（老库的行），见 `db/models.py` 里这一列的注释。
    """
    cost = compute_cost(
        model,
        input_tokens,
        output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )
    assigned_seq = seq if seq is not None else await ctx.next_seq()

    async with session_scope() as db:
        db.add(
            LlmCall(
                id=str(uuid.uuid4()),
                test_run_id=ctx.run_id,
                session_id=ctx.session_id,
                seq=assigned_seq,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                cost_usd=cost.usd,
                latency_ms=latency_ms,
                streaming=streaming,
                status=status,
                request_bytes=request_bytes,
            )
        )

    return {"usd": cost.usd, "priced": cost.priced, "reason": cost.reason}


async def record_finding(
    ctx: RunContext,
    *,
    detector: str,
    verdict: DetectorVerdict,
    summary: str,
    confidence: Confidence = Confidence.HIGH,
    detail: dict[str, Any] | None = None,
    exempted_by: str | None = None,
) -> str:
    """落一条检测结论。四态语义见 docs/detectors.md。"""
    finding_id = str(uuid.uuid4())
    async with session_scope() as db:
        db.add(
            DetectorFinding(
                id=finding_id,
                test_run_id=ctx.run_id,
                detector=detector,
                verdict=verdict.value,
                confidence=confidence.value,
                summary=summary,
                detail_json=(
                    json.dumps(detail, ensure_ascii=False, default=str) if detail else None
                ),
                exempted_by=exempted_by,
            )
        )
    return finding_id
