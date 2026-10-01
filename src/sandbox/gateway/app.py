"""LLM 网关：OpenAI 兼容的边界拦截点（ADR 001）。

它同时干四件事，缺一不可：

1. **路由**——把请求关联到具体的 TestRun（见 runtime.py 关于 api_key 的说明）。
2. **观测**——落 LlmCall 明细。Agent 侧报的 usage 不可信；这里才是权威计数点。
3. **止损**——放行前预扣预算，超限直接拒绝。这是全系统唯一能真正省下钱的位置。
4. **凭据隔离**——真实上游密钥只活在这个进程里，被测 Agent 拿到的是沙箱令牌。

第 4 点容易被忽略但很重要：如果 Agent 手里是真实密钥，它就能绕过网关直连上游，
"完整观测"立刻失效——而且是静默失效，报告会显示一切正常，只是少了一半调用。

流式请求（`stream: true`）走的是同一套职责，只有一处不同：结算发生在转发
**结束之后**（客户端随时可能断开），所以它挂在生成器的 `finally` 上。转发与结算
的实现都在 `stream.py`，这个文件只负责路由和 HTTP 的形状。
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from sandbox.config import settings
from sandbox.db.models import Confidence, DetectorVerdict
from sandbox.detectors.base import ExemptionSet
from sandbox.detectors.loop import checked_field_weights
from sandbox.gateway.budget import BudgetExceededError, TokenBudget
from sandbox.gateway.stream import (
    UPSTREAM_TIMEOUT,
    Reserved,
    UpstreamError,
    chain_first,
    real_stream_lines,
    relay_stream,
    settle_and_record,
    stub_stream_lines,
    upstream_headers,
    upstream_url,
    usage_requested,
    with_usage_option,
)
from sandbox.gateway.stub import scripted_upstream
from sandbox.runtime import RunContext, runs
from sandbox.trace.recorder import record_finding

log = structlog.get_logger(__name__)

app = FastAPI(title="sandbox LLM gateway", version="0.0.1")


def _default_tool_schemas() -> dict[str, dict[str, Any]]:
    """未声明参数 schema 时的回落表：内置 mock 的**内置**工具声明。

    与 `_default_tools` 同一套路、同一个理由——手工起一个运行排查时不该被迫先把
    工具和它们的参数声明抄一遍。两者的来源是同一份 `TOOL_SPECS`，所以手工路径下
    的参数校验和它列给 Agent 的那份说明永远一致。
    """
    from sandbox.mock.toolspec import TOOL_SPECS

    return {name: spec.parameters for name, spec in TOOL_SPECS.items()}


def _default_tools() -> dict[str, str]:
    """未声明工具时的回落表：内置 mock 的工具 → 路径。

    这个耦合方向（网关 → mock）是**故意局限在兜底上**的，所以用延迟导入把它
    挡在模块导入图之外：网关的核心职责不认识 mock，只有"手工起一个运行来排查"
    这条便利路径需要一份现成的工具表。场景运行一律自带 tools，走不到这里。

    要是哪天这里被当成正常路径用了，说明场景文件没把工具声明传下来——那时
    该修的是传参，不是把这张表做大。
    """
    from sandbox.mock.toolspec import TOOL_SPECS

    return {name: spec.path for name, spec in TOOL_SPECS.items()}


async def _call_real_upstream(payload: dict[str, Any]) -> dict[str, Any]:
    """转发到真实上游（非流式）。

    密钥与 URL 的拼装都在 `stream` 里——**流式与非流式共用那一份**，所以
    "真实密钥只出现在网关进程的一个函数里"这句话只有一个实现处。
    """
    url = upstream_url(settings.upstream_base_url)
    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        resp = await client.post(url, json=payload, headers=upstream_headers())

    if resp.status_code >= 400:
        raise UpstreamError(resp.status_code, resp.text[:2000])
    return resp.json()


def _error(status: int, message: str, err_type: str = "sandbox_error") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "code": status}},
    )


def _upstream_failure(exc: Exception, ctx: Any) -> JSONResponse:
    """上游那几种失败 → 响应。**流式与非流式共用这一份。**

    分成两张表写的话，"脚本没注册"在一条路上是 400、在另一条路上是 502，
    而两条路指向的是同一件事故——排查的人会以为是两个不同的问题。
    """
    if isinstance(exc, StopAsyncIteration):
        return _error(
            502,
            "上游的流一帧都没有就结束了。空流会让客户端一直等下去，"
            "所以这里直接报错，而不是回一个 200。",
            err_type="empty_stream",
        )
    if isinstance(exc, KeyError):
        return _error(400, str(exc), err_type="stub_script_missing")
    if isinstance(exc, UpstreamError):
        return _error(exc.status, exc.detail, err_type="upstream_error")
    log.exception("gateway.upstream_failed", run_id=getattr(ctx, "run_id", None))
    return _error(502, f"上游调用失败：{exc}", err_type="upstream_error")


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "upstream_mode": settings.upstream_mode,
        "upstream_ready": settings.upstream_ready,
        "active_runs": runs.active_ids(),
    }


@app.get("/_sandbox/runs")
async def list_runs() -> dict[str, Any]:
    """给验收脚本看的运行/预算快照。"""
    return {"runs": runs.snapshot()}


@app.post("/_sandbox/runs", response_model=None)
async def create_run(payload: dict[str, Any]) -> dict[str, Any] | JSONResponse:
    """开一个运行并签发沙箱令牌。

    declared_writes 由场景工具声明提供——**必须显式声明**，不能只靠工具名启发式，
    因为写判定直接决定幂等/重复扣款检测的触发范围。
    """
    from sandbox.trace.recorder import start_run

    # 工具表与参数 schema 表，**要么两张都来自调用方，要么两张都用内置的**。
    #
    # 不传则回落到内置工具声明：手工 curl 起一个运行来排查时不该被迫先声明一遍
    # 工具和它们的参数。但只补其中一张是最坏的一种——同一个工具会被配上一份
    # 谁也不认识的声明，而报告上完全看不出来。
    #
    # 判据是字段**在不在**，不是它**是不是空的**——与下面 `declared_writes` 同一条
    # 规矩。空表和没给折叠成同一件事的话，一个声明了"这个场景没有工具"的场景会
    # 悄悄拿到内置的那张表：Agent 调一个它没被声明过的工具，本该报 `tool_scope`
    # 失败并被拦下，实际却被转发出去、**钱真的动了**。
    if "tools" in payload or "tool_schemas" in payload:
        tool_table = payload.get("tools") or {}
        schema_table = payload.get("tool_schemas") or {}
    else:
        tool_table = _default_tools()
        schema_table = _default_tool_schemas()

    # 写工具的声明有**三种**状态，少一种就是少一句话：
    #
    #   给了非空表 = 这些是写操作；给了空表 = **这个场景没有写操作**；没给 = 不知道。
    #
    # 空表和"没给"折叠成同一件事，是把一次**声明**换成一次**猜测**：场景作者写下
    # "这里没有写操作"，沙箱转头拿工具名去猜，还可能猜反——而写判定直接决定幂等/
    # 重复扣款检测的触发范围，猜错的代价是漏报或误报一整类缺陷。
    # 没给（`null`、字段不在）才是真的不知道，那时才轮到启发式兜底。
    raw_writes = payload.get("declared_writes")
    if raw_writes is not None and not (
        isinstance(raw_writes, list) and all(isinstance(n, str) for n in raw_writes)
    ):
        # 写错的声明不能静默降级成"没声明"：`set("create_payment")` 会拆成一堆
        # 单字符，每个工具都不在里面，于是整张声明表**看起来**生效了、实际从没
        # 匹配上——这一类"装上了但没接线"在报告上完全看不出来。
        return _error(
            400,
            "declared_writes 必须是字符串数组。空数组是有意义的：它说的是"
            "「这个场景没有写操作」；不传这个字段则等于「不知道」。",
            err_type="bad_declared_writes",
        )

    ctx = await start_run(
        scenario_id=payload.get("scenario_id", "adhoc"),
        agent_version=payload.get("agent_version", "unknown"),
        config=payload.get("config") or {},
        env=payload.get("env") or {},
        declared_writes=set(raw_writes) if raw_writes is not None else None,
        idempotency_key_params=payload.get("idempotency_key_params") or {},
    )
    # 豁免规则也要在场——**看门狗读的就是它**（见 `runtime.ActiveRun.exemptions`）。
    # 校验在这里照走一遍：线上的形状是 JSON，手工 curl 也会走这条路，规则写错
    # 要让调用方看见是哪一条错在哪，而不是等熔断那一刻才炸。
    try:
        exemptions = ExemptionSet.of(payload.get("exemptions"))
    except ValueError as exc:
        return _error(400, f"豁免规则有问题：{exc}", err_type="bad_exemptions")

    # 字段权重同理，也走看门狗（它决定"这串调用算不算在原地打转"）。同一条规则
    # 在场景加载那边也走一遍，实现是同一个函数——两处各写一份的话，一条路径
    # 会收下 `nan` 而另一条拦住，收下它的那条让近似层静默失效。
    raw_weights = payload.get("loop_field_weights")
    try:
        loop_field_weights = (
            {} if raw_weights is None else checked_field_weights(raw_weights)
        )
    except ValueError as exc:
        return _error(
            400, f"字段权重有问题：{exc}", err_type="bad_loop_field_weights"
        )

    active = runs.register(
        ctx,
        scenario_id=payload.get("scenario_id", "adhoc"),
        ceiling=payload.get("token_ceiling"),
        # 场景声明的工具 → 路径。代理据此判断调用是否越界，
        # 所以必须在开跑前定下来，不能等第一个请求到了再补。
        tools=tool_table,
        # 参数 schema，出口代理拿它核对每一次调用。与上面那张表同源同命，见上。
        tool_schemas=schema_table,
        exemptions=exemptions,
        loop_field_weights=loop_field_weights,
    )
    return {
        "run_id": ctx.run_id,
        "session_id": ctx.session_id,
        "session_sequence_id": ctx.session_sequence_id,
        "api_key": active.token,
        "base_url": f"http://{settings.gateway_host}:{settings.gateway_port}/v1",
        "token_ceiling": active.budget.ceiling,
    }


@app.post("/_sandbox/runs/{run_id}/finish")
async def finish_run_endpoint(run_id: str, status: str = "completed") -> dict[str, Any]:
    """收尾一个运行：注销路由并汇总。"""
    from sandbox.db.models import RunStatus
    from sandbox.trace.recorder import finish_run, recompute_totals

    active = runs.get(run_id)
    if active is None:
        return {"ok": False, "detail": f"run {run_id} 不在活跃表中"}

    totals = await recompute_totals(run_id)
    try:
        run_status = RunStatus(status)
    except ValueError:
        run_status = RunStatus.COMPLETED

    await finish_run(active.ctx, status=run_status, totals=totals)
    runs.unregister(run_id)
    return {"ok": True, "run_id": run_id, "status": run_status.value, "totals": totals}


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: str | None = Header(default=None),
    x_sandbox_run_id: str | None = Header(default=None),
) -> Any:
    started = time.perf_counter()
    payload = await request.json()

    active = runs.resolve(authorization, x_sandbox_run_id)
    if active is None:
        return _error(
            401,
            "无法把请求关联到任何运行。请把 api_key 设为网关签发的沙箱令牌"
            "（sk-sandbox-<run_id>）；若同时有多个运行活跃，无标识请求会被拒绝——"
            "因为猜错会把 A 的调用记到 B 头上，且事后无法发现。",
            err_type="unroutable_request",
        )

    if active.killed:
        # 熔断之后放行任何一个请求，都等于熔断没发生过。
        return _error(
            410,
            f"该运行已被沙箱带外熔断：{active.kill_reason}。后续请求一律拒绝。",
            err_type="run_killed",
        )

    ctx = active.ctx
    budget = active.budget
    model = payload.get("model") or "unknown"

    est = Reserved.of(payload)

    try:
        budget.reserve(est.total)
    except BudgetExceededError as exc:
        # 熔断是运行级事件，必须留痕。否则报告里只会看到"运行提前结束"，
        # 看不到为什么结束。
        await record_finding(
            ctx,
            detector="token_budget",
            verdict=DetectorVerdict.FAIL,
            summary=f"运行级 Token 预算耗尽，网关已拒绝请求：{exc}",
            confidence=Confidence.HIGH,
            detail={
                "used": exc.used,
                "ceiling": exc.ceiling,
                "requested": exc.requested,
                "model": model,
            },
        )
        return _error(429, str(exc), err_type="budget_exceeded")

    if payload.get("stream"):
        return await _streaming_response(
            payload, ctx=ctx, budget=budget, reserved=est, model=model, started=started
        )

    try:
        if settings.upstream_mode == "stub":
            response = scripted_upstream.next_response(
                ctx.run_id, model=model, approx_input_tokens=est.input_tokens
            )
        else:
            response = await _call_real_upstream(payload)
    except Exception as exc:  # noqa: BLE001 — 必须归还未预扣，否则预算会慢性泄漏
        budget.release(est.total)
        return _upstream_failure(exc, ctx)

    cost = await settle_and_record(
        ctx,
        budget,
        reserved=est,
        model=model,
        usage=response.get("usage") or None,
        # 非流式的内容不经过网关，没有可数的地方；没拿到 usage 时只能退回预扣的上界。
        observed_output=None,
        started=started,
        streaming=False,
    )

    # 响应头是给人和给验收脚本看的：一眼能看出这次请求落在哪个运行、预算还剩多少。
    headers = {
        "X-Sandbox-Run-Id": ctx.run_id,
        "X-Sandbox-Tokens-Used": str(budget.used),
        "X-Sandbox-Tokens-Remaining": str(budget.remaining),
        "X-Sandbox-Cost-USD": f"{cost['usd']:.6f}",
        "X-Sandbox-Priced": "1" if cost["priced"] else "0",
    }
    return JSONResponse(content=response, headers=headers)


async def _streaming_response(
    payload: dict[str, Any],
    *,
    ctx: RunContext,
    budget: TokenBudget,
    reserved: Reserved,
    model: str,
    started: float,
) -> JSONResponse | StreamingResponse:
    """流式：转发上游的帧，同时累计结算。

    **先取第一帧再返回 200。** 开流之前就会失败的情况（脚本没注册、上游 4xx、
    连不上）在这一步就暴露出来，那时还能回一个正常的错误响应；等 200 和第一帧
    发出去之后再失败，客户端只会看到一个断掉的流，而错误信息只剩下网关的日志。
    取出来的那一帧由 `chain_first` 放回去，客户端不会少收东西。
    """
    outbound, injected = with_usage_option(payload)
    lines: AsyncIterator[str]
    if settings.upstream_mode == "stub":
        lines = stub_stream_lines(
            ctx.run_id, model, reserved, include_usage=usage_requested(outbound)
        )
    else:
        lines = real_stream_lines(outbound)

    try:
        first = await anext(lines)
    except Exception as exc:  # noqa: BLE001 — 同上，失败必须归还预扣
        budget.release(reserved.total)
        return _upstream_failure(exc, ctx)

    # 流式响应**没有** X-Sandbox-Tokens-Used / Cost 这几个头：它们要等流结束
    # 才知道，而响应头在第一帧之前就发出去了。不知道就不写，别写一个"当前值"
    # 让读的人以为那是这次调用的结算数——结算数在 LlmCall 明细和报告里。
    headers = {
        "X-Sandbox-Run-Id": ctx.run_id,
        "X-Sandbox-Streaming": "1",
        "X-Sandbox-Token-Ceiling": str(budget.ceiling),
        "Cache-Control": "no-cache",
    }
    if injected:
        # 我们动过这次请求，就得让它在响应里留下痕迹（见 with_usage_option）。
        headers["X-Sandbox-Stream-Options-Added"] = "include_usage"

    return StreamingResponse(
        relay_stream(
            chain_first(first, lines),
            ctx=ctx,
            budget=budget,
            reserved=reserved,
            model=model,
            started=started,
        ),
        media_type="text/event-stream",
        headers=headers,
    )
