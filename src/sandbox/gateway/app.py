"""LLM 网关：OpenAI 兼容的边界拦截点（ADR 001）。

它同时干四件事，缺一不可：

1. **路由**——把请求关联到具体的 TestRun（见 runtime.py 关于 api_key 的说明）。
2. **观测**——落 LlmCall 明细。Agent 侧报的 usage 不可信；这里才是权威计数点。
3. **止损**——放行前预扣预算，超限直接拒绝。这是全系统唯一能真正省下钱的位置。
4. **凭据隔离**——真实上游密钥只活在这个进程里，被测 Agent 拿到的是沙箱令牌。

第 4 点容易被忽略但很重要：如果 Agent 手里是真实密钥，它就能绕过网关直连上游，
"完整观测"立刻失效——而且是静默失效，报告会显示一切正常，只是少了一半调用。
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
import structlog
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

from sandbox.config import settings
from sandbox.db.models import Confidence, DetectorVerdict
from sandbox.gateway.budget import BudgetExceededError
from sandbox.gateway.stub import scripted_upstream
from sandbox.runtime import runs
from sandbox.trace.recorder import record_finding, record_llm_call

log = structlog.get_logger(__name__)

app = FastAPI(title="sandbox LLM gateway", version="0.0.1")

# 保守的输入估算：CJK 约 1–2 字符/token，英文约 4。取 3 是往安全的方向估——
# 估多了只是提前熔断（可恢复），估少了会让超限的那一次已经花出去。
_CHARS_PER_TOKEN = 3
# 请求没给 max_tokens 时的补全上界。
_DEFAULT_MAX_COMPLETION = 4096


class UpstreamError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        self.detail = detail
        super().__init__(detail)


def estimate_input_tokens(payload: dict[str, Any]) -> int:
    """估算输入 token。**只用于预扣**，结算一律用上游返回的真实 usage。"""
    blob = json.dumps(
        {
            "messages": payload.get("messages", []),
            "tools": payload.get("tools"),
            "system": payload.get("system"),
        },
        ensure_ascii=False,
        default=str,
    )
    return max(1, len(blob) // _CHARS_PER_TOKEN)


def estimate_reserve(payload: dict[str, Any]) -> tuple[int, int]:
    """返回 (输入预估, 输出预估)。输出取 max_tokens，没给就按默认上界。"""
    max_out = payload.get("max_tokens") or payload.get("max_completion_tokens")
    if not isinstance(max_out, int) or max_out <= 0:
        max_out = _DEFAULT_MAX_COMPLETION
    return estimate_input_tokens(payload), max_out


def _upstream_url(base: str) -> str:
    """把 base_url 规范成 chat/completions 端点。两种写法都接受。"""
    base = base.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


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
    """转发到真实上游。真实密钥只在这里出现。"""
    if not settings.upstream_ready:
        raise UpstreamError(
            500,
            "upstream_mode=real 但没有配置 SANDBOX_UPSTREAM_API_KEY。"
            "拒绝静默降级到 stub——那会让测试结果看起来正常但毫无意义。",
        )

    url = _upstream_url(settings.upstream_base_url)
    headers = {
        "Authorization": f"Bearer {settings.upstream_api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, json=payload, headers=headers)

    if resp.status_code >= 400:
        raise UpstreamError(resp.status_code, resp.text[:2000])
    return resp.json()


def _error(status: int, message: str, err_type: str = "sandbox_error") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "code": status}},
    )


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


@app.post("/_sandbox/runs")
async def create_run(payload: dict[str, Any]) -> dict[str, Any]:
    """开一个运行并签发沙箱令牌。

    declared_writes 由场景工具声明提供——**必须显式声明**，不能只靠工具名启发式，
    因为写判定直接决定幂等/重复扣款检测的触发范围。
    """
    from sandbox.trace.recorder import start_run

    ctx = await start_run(
        scenario_id=payload.get("scenario_id", "adhoc"),
        agent_version=payload.get("agent_version", "unknown"),
        config=payload.get("config") or {},
        env=payload.get("env") or {},
        declared_writes=set(payload["declared_writes"]) if payload.get("declared_writes") else None,
        idempotency_key_params=payload.get("idempotency_key_params") or {},
    )
    active = runs.register(
        ctx,
        scenario_id=payload.get("scenario_id", "adhoc"),
        ceiling=payload.get("token_ceiling"),
        # 场景声明的工具 → 路径。代理据此判断调用是否越界，
        # 所以必须在开跑前定下来，不能等第一个请求到了再补。
        #
        # 不传则回落到内置工具表：手工 curl 起一个运行来排查时不该被迫先声明
        # 一遍工具。回落值是显式的默认，不是隐式行为——传了就完全以传的为准。
        tools=payload.get("tools") or _default_tools(),
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

    if payload.get("stream"):
        # 明确拒绝而不是静默走非流式：静默的话 Agent 侧解析会炸在别处，
        # 报错信息指向错误的位置，排查成本远高于这里直接说清楚。
        return _error(
            501,
            "网关尚未实现流式响应。请把请求的 stream 置为 false。"
            "（流式需要边转发边累计 usage，在结算正确性上另有一套逻辑，"
            "不走通不发布。）",
            err_type="not_implemented",
        )

    est_in, est_out = estimate_reserve(payload)
    reserved = est_in + est_out

    try:
        budget.reserve(reserved)
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

    try:
        if settings.upstream_mode == "stub":
            response = scripted_upstream.next_response(
                ctx.run_id, model=model, approx_input_tokens=est_in
            )
        else:
            response = await _call_real_upstream(payload)
    except KeyError as exc:
        budget.release(reserved)
        return _error(400, str(exc), err_type="stub_script_missing")
    except UpstreamError as exc:
        budget.release(reserved)
        return _error(exc.status, exc.detail, err_type="upstream_error")
    except Exception as exc:  # noqa: BLE001 — 必须归还未预扣，否则预算会慢性泄漏
        budget.release(reserved)
        log.exception("gateway.upstream_failed", run_id=ctx.run_id)
        return _error(502, f"上游调用失败：{exc}", err_type="upstream_error")

    usage = response.get("usage") or {}
    in_tok = int(usage.get("prompt_tokens") or 0)
    out_tok = int(usage.get("completion_tokens") or 0)
    cache_read = int(usage.get("prompt_cache_hit_tokens") or 0)
    cache_write = int(usage.get("prompt_cache_miss_tokens") or 0)

    budget.settle(reserved, in_tok, out_tok)
    latency_ms = int((time.perf_counter() - started) * 1000)

    cost = await record_llm_call(
        ctx,
        model=model,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        latency_ms=latency_ms,
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
