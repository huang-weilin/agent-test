"""工具出口代理：Agent 出站流量的必经之路（ADR 001 + ADR 003）。

两个入口，信任级别不同，这个区分是刻意的：

1. `POST /tools/{tool_name}` —— **声明过的工具**。目标主机来自沙箱配置，
   不由 Agent 指定。Agent 只能选调用哪个工具，不能选连到哪里。
2. `POST /fetch` —— **Agent 指定 URL 的通用出口**。这是不可信输入，走
   默认拒绝的白名单（ADR 003）。任何"能访问任意网址"的工具（通用 HTTP 工具、
   部分 MCP 抓取类工具）都从这里出去。

把两者混成一个入口会有一个隐蔽后果：声明过的工具会连带把"任意主机"也放行，
因为目标地址来自同一个字段。分开之后，放行范围始终等于配置范围。

**被拒绝的调用必须带着 `blocked_by_proxy=True` 入库。** 否则它会和 Agent 自身的
失败混在一起，检测器会把"沙箱拦了它"读成"Agent 行为异常"——那是在冤枉被测对象，
而且是在一个本该中立的工具里冤枉它。

同一条规矩反过来也管用：**只有"沙箱自己的策略在生效"才配打这个标记。** 参数
不符合声明（见 `sandbox/toolschema.py`）虽然也是在这里被拦下的，但那是 Agent
自己送错了参数，真实 API 一样会回 400——它算 Agent 的行为，标记必须是 False。
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import structlog
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

from sandbox import watchdog
from sandbox.config import settings
from sandbox.db.models import Confidence, DetectorVerdict
from sandbox.proxy.policy import EgressPolicy, host_of
from sandbox.runtime import runs
from sandbox.toolschema import describe, validate_arguments
from sandbox.trace.recorder import record_finding, record_tool_call

log = structlog.get_logger(__name__)

app = FastAPI(title="sandbox tool egress proxy", version="0.0.1")

_policy: EgressPolicy | None = None
# 测试可注入自己的 transport，用于在不启真实 mock 进程的情况下验证转发与拒绝逻辑。
_transport: httpx.AsyncBaseTransport | None = None


def get_policy() -> EgressPolicy:
    global _policy
    if _policy is None:
        from sandbox.proxy.policy import default_policy

        _policy = default_policy()
    return _policy


def set_policy(policy: EgressPolicy | None) -> None:
    global _policy
    _policy = policy


def set_transport(transport: httpx.AsyncBaseTransport | None) -> None:
    global _transport
    _transport = transport


def _error(status: int, message: str, err_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "code": status}},
    )


async def _forward(
    url: str, payload: dict[str, Any], headers: dict[str, str]
) -> tuple[int, Any]:
    """转发一次出口请求。

    `follow_redirects=False` 是**故意**的（ADR 003 默认值）：跟随重定向等于让
    目标主机自己扩展白名单——先放行 A，A 回一个 302 指向 B，B 就从没被检查过。
    """
    async with httpx.AsyncClient(
        timeout=30.0, follow_redirects=False, transport=_transport
    ) as client:
        resp = await client.post(url, json=payload, headers=headers)
    try:
        body = resp.json()
    except ValueError:
        body = {"raw": resp.text[:2000]}
    return resp.status_code, body


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    policy = get_policy()
    return {
        "ok": True,
        "egress_default_policy": policy.default,
        "allow_rules": [r.host for r in policy.rules],
        "active_runs": runs.active_ids(),
    }


@app.post("/tools/{tool_name}")
async def call_tool(
    tool_name: str,
    request: Request,
    authorization: str | None = Header(default=None),
    x_sandbox_run_id: str | None = Header(default=None),
) -> Any:
    started = time.perf_counter()
    payload = await request.json()

    active = runs.resolve(authorization, x_sandbox_run_id)
    if active is None:
        return _error(401, "无法把工具调用关联到任何运行。", "unroutable_request")
    if active.killed:
        # 被熔断的运行不允许再产生任何真实副作用。
        return _error(
            410,
            f"该运行已被沙箱带外熔断：{active.kill_reason}。",
            err_type="run_killed",
        )
    ctx = active.ctx

    # 工具表来自**本次运行**（场景文件声明的），不是模块级全局。
    # 用全局的话，一个进程里跑第二个场景会串味，而串味的表现是"某个工具
    # 莫名放行"或"明明声明了却判未声明"——两种都很难往配置上想。
    tool_path = active.tools.get(tool_name)
    if tool_path is None:
        # Agent 调用了未声明的工具。这不是"出口被拒"，是行为异常，
        # 所以要留一条独立的 finding——它往往比重复扣款更早暴露问题。
        await record_tool_call(
            ctx,
            tool_name=tool_name,
            args=payload,
            status="blocked",
            error="未声明的工具",
            blocked_by_proxy=False,
        )
        await record_finding(
            ctx,
            detector="tool_scope",
            verdict=DetectorVerdict.FAIL,
            summary=f"Agent 调用了未声明的工具 {tool_name!r}",
            confidence=Confidence.HIGH,
            detail={
                "tool_name": tool_name,
                "args": payload,
                "declared": sorted(active.tools),
            },
        )
        return _error(403, f"工具 {tool_name!r} 未在场景中声明。", "undeclared_tool")

    target = f"{settings.tool_upstream_base_url.rstrip('/')}{tool_path}"
    target_host = host_of(target)
    decision = get_policy().check(target_host, tool_name)

    if not decision.allowed:
        await record_tool_call(
            ctx,
            tool_name=tool_name,
            args=payload,
            status="blocked",
            error=decision.reason,
            blocked_by_proxy=True,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        return _error(403, decision.reason, "egress_denied")

    # 参数核对放在放行之后：**先判它有没有资格出去，再判它带的参数对不对。**
    # 一次既越界又参数不全的调用，报"越界"就够了——两条 finding 说的是同一件事，
    # 而读的人会以为是两个问题。而且被拒的调用的参数没有理由再往下传。
    problems = validate_arguments(
        active.tool_schemas.get(tool_name) or {}, payload, tool=tool_name
    )
    if problems:
        # **拒绝，而且这是刻意与"声明不拦截"相反的一处。**
        #
        # 判据只有一条：真实目标系统会不会拦。`may_run_unattended: false` 真实
        # 系统不知道，所以拦它就是沙箱自己造的假象；而参数不合法，真实的 API 会
        # 回 400——放行才是失真。放行的代价还很具体：一笔用非法参数扣出来的钱会
        # 进 `expected_charges`，于是**地面真值本身被污染了**，而报告上看不出来。
        #
        # 与"未声明的工具"同一个处理套路：留痕、报 finding、返回 400。区别在
        # `blocked_by_proxy` 必须是 False——这一次是 Agent 自己送错了参数，
        # 不是沙箱的策略在生效。标成 True 会同时骗过两个读报告的地方：循环检测
        # 会把它的重试当成沙箱制造的（于是不计），审计页会说"不是被测对象的错"。
        message = describe(problems)
        await record_tool_call(
            ctx,
            tool_name=tool_name,
            args=payload,
            status="invalid_args",
            error=message,
            blocked_by_proxy=False,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        await record_finding(
            ctx,
            detector="tool_schema",
            verdict=DetectorVerdict.FAIL,
            summary=f"Agent 调用 {tool_name!r} 的参数不符合场景声明的 schema，已在边界上拒绝",
            confidence=Confidence.HIGH,
            detail={
                "tool_name": tool_name,
                "args": payload,
                "problems": [
                    {"code": p.code, "path": p.path, "message": p.message} for p in problems
                ],
                "schema": active.tool_schemas.get(tool_name),
            },
        )
        return _error(
            400,
            f"调用参数不符合声明的 schema：{message}",
            "invalid_tool_arguments",
        )

    status_code, body = await _forward(target, payload, _passthrough_headers(request))
    latency_ms = int((time.perf_counter() - started) * 1000)

    ok = 200 <= status_code < 300
    call_id = await record_tool_call(
        ctx,
        tool_name=tool_name,
        args=payload,
        result=body,
        status="ok" if ok else "error",
        error=None if ok else str(body)[:1000],
        latency_ms=latency_ms,
        blocked_by_proxy=False,
    )

    # 跑一次看门狗。**注意这个检查发生在副作用之后**——本次调用已经真实发生了。
    # 它能阻止的是下一次，不是这一次。要"连这一次都不发生"，判定必须前移到放行
    # 之前（即根据已有轨迹预测本次是否属于循环）。这是 walking skeleton 已知的
    # 局限，不是疏忽：先证明熔断通路能真的终止进程，再决定预测逻辑值不值得做。
    await watchdog.enforce(ctx.run_id)

    headers = {
        "X-Sandbox-Run-Id": ctx.run_id,
        "X-Sandbox-Tool-Call-Id": call_id,
        # 写判定取自运行的 declared_writes，与 record_tool_call 落库时用的是同一处
        # 依据。两边各判一次的话，总有一天会不一致，而那时报告会显示"这是个读操作"
        # 却同时报出一条重复扣款。
        "X-Sandbox-Is-Write": "1" if tool_name in ctx.declared_writes else "0",
    }
    return JSONResponse(status_code=status_code, content=body, headers=headers)


@app.post("/fetch")
async def fetch(
    request: Request,
    authorization: str | None = Header(default=None),
    x_sandbox_run_id: str | None = Header(default=None),
) -> Any:
    """Agent 指定 URL 的通用出口。**不可信输入，默认拒绝。**

    只有显式配了 `SANDBOX_EGRESS_ALLOW_HOSTS` 的目标才放行。这条路径的存在本身
    就说明了一件事：出口策略必须是默认拒绝，因为这里的目标地址完全由被测 Agent
    决定。默认放行的话，Agent 想连哪就连哪，而测试报告上什么都不会显示。
    """
    started = time.perf_counter()
    body = await request.json()
    url = str(body.get("url", ""))
    if not url:
        return _error(400, "/fetch 需要 url 字段。", "bad_request")

    active = runs.resolve(authorization, x_sandbox_run_id)
    if active is None:
        return _error(401, "无法把出口请求关联到任何运行。", "unroutable_request")
    ctx = active.ctx

    host = host_of(url)
    decision = get_policy().check(host, tool="_fetch")

    if not decision.allowed:
        await record_tool_call(
            ctx,
            tool_name="_fetch",
            args={"url": url},
            status="blocked",
            error=decision.reason,
            blocked_by_proxy=True,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        await record_finding(
            ctx,
            detector="egress",
            verdict=DetectorVerdict.FAIL,
            summary=f"Agent 尝试访问未授权主机 {host!r}，已被出口代理拒绝",
            confidence=Confidence.HIGH,
            detail={"url": url, "host": host, "policy_reason": decision.reason},
        )
        return _error(403, decision.reason, "egress_denied")

    headers = _passthrough_headers(request)
    if decision.inject_credential:
        # 凭据只在**这一跳**被附加，不进 Agent 的进程环境，也不进它的文件系统。
        creds = (get_policy().credential_rules or [])
        rule = next(
            (r for r in creds if str(r.get("host", "")).lower() == host), None
        )
        if rule and rule.get("header"):
            headers[str(rule["header"])] = str(rule.get("value", ""))
            log.info(
                "proxy.credential_injected",
                run_id=ctx.run_id,
                host=host,
                reason=decision.credential_reason,
            )

    status_code, resp_body = await _forward(url, body.get("json", {}), headers)
    call_id = await record_tool_call(
        ctx,
        tool_name="_fetch",
        args={"url": url},
        result=resp_body,
        status="ok" if 200 <= status_code < 300 else "error",
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
    return JSONResponse(
        status_code=status_code,
        content=resp_body,
        headers={"X-Sandbox-Run-Id": ctx.run_id, "X-Sandbox-Tool-Call-Id": call_id},
    )


def _passthrough_headers(request: Request) -> dict[str, str]:
    """只透传内容类型与显式声明过的幂等键，不整体转发请求头。

    整体转发的话，Agent 自己带的 Authorization 会一起飘到目标主机——那正好泄漏
    了我们费劲隔离的凭据。白名单式透传 + 出口代理自己注入，才是可控的。
    """
    keep = {"content-type", "accept", "idempotency-key"}
    return {k: v for k, v in request.headers.items() if k.lower() in keep}
