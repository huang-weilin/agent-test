"""工具出口代理：Agent 出站流量的必经之路（ADR 001 + ADR 003）。

入口按**目标是谁定的**分成两类，信任级别不同，这个区分是刻意的：

1. `POST /tools/{tool_name}` —— **目标由沙箱配置定的工具**。目标主机来自场景
   声明（`egress_host`），不由 Agent 指定。Agent 只能选调用哪个工具，不能选
   连到哪里。
2. `POST /fetch` 与 `POST /fetch/{tool_name}` —— **目标由被测对象指定的出口**。
   URL 在请求体里，所以这是不可信输入，走默认拒绝的白名单（ADR 003）。任何
   "能访问任意网址"的工具（通用 HTTP 工具、部分 MCP 抓取类工具）都从这里出去。
   带名字的那条是**声明过的**开放目标工具（`egress_host: agent`），名字进路径，
   代理拿它去对工具表；不带名字的那条是匿名出口（手工 curl 起的那条路）。

把两类混成一个入口会有一个隐蔽后果：声明过的工具会连带把"任意主机"也放行，
因为目标地址来自同一个字段。分开之后，放行范围始终等于配置范围——而这正是
"这个工具的调用该走哪条路"必须在运行注册时就过线的原因（见 `ToolDecl.proxy_path`）。

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
from sandbox.proxy.policy import (
    GENERIC_EGRESS_BODY_FIELD,
    GENERIC_EGRESS_PREFIX,
    GENERIC_EGRESS_URL_FIELD,
    EgressPolicy,
    generic_egress_path,
    host_of,
)
from sandbox.runtime import runs
from sandbox.toolschema import describe, validate_arguments
from sandbox.trace.normalize import looks_like_write
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


async def _settle(ctx: Any) -> None:
    """一次调用**真的出到目标系统之后**，让看门狗扫一遍。

    **两条路由都要经过这里**：定目标的 `/tools/{name}`，和目标由调用方指定的
    `/fetch` 与 `/fetch/{name}`。原先只有前者接了，后者一次都不扫——而事后那轮
    检测照样会在整条会话序列上算出循环、把结论落进报告，所以**报告看起来是对的**，
    没发生的只是止损：一个用通用出口打转的 Agent 不会被杀，它会一直烧到 token
    上限或单会话超时（那两条都是兜底，不是"检测到循环就地终止"）。
    量它的那次对照在 `tests/test_open_route_parity.py`。

    **被拦下的那几支不扫**（未声明的工具、走错入口、出口被拒、参数不合 schema）：
    它们在目标系统上什么都没留下，而这条通路要防的正是"循环把目标系统打爆"；
    那些路径上的 token 失控由网关的预算熔断独立负责。这是写下来的边界，不是
    遗漏——`watchdog.py` 与 `docs/detectors.md` 里原先那句"每次工具调用后重扫"
    按这条改掉了。
    """
    await watchdog.enforce(ctx.run_id)


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

    if tool_path.startswith(GENERIC_EGRESS_PREFIX):
        # 这个工具的目标由调用方指定，它的调用必须走通用出口
        # （`POST /fetch/{名字}`，见 `fetch_for_tool`）——从这条固定入口进来的话，
        # 它连的会是配置里那条 mock 路径，而那不是它的目的地。
        #
        # 放过去的话，报告上会出现一次"记在它名下、实际发给别处"的调用：判定按
        # 声明的那台主机做，请求却落在 mock 上，两边说的不是一件事。
        await record_tool_call(
            ctx,
            tool_name=tool_name,
            args=payload,
            status="blocked",
            error=f"工具 {tool_name!r} 的入口是通用出口",
            blocked_by_proxy=True,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        await record_finding(
            ctx,
            detector="egress",
            verdict=DetectorVerdict.FAIL,
            summary=(
                f"工具 {tool_name!r} 的目标由调用方指定，这次调用却从固定入口进来，"
                f"已在边界上拒绝"
            ),
            confidence=Confidence.HIGH,
            detail={"tool_name": tool_name, "args": payload, "expected_path": tool_path},
        )
        return _error(
            403,
            f"工具 {tool_name!r} 的调用必须走通用出口 POST {tool_path}。",
            "tool_route_mismatch",
        )

    target = f"{settings.tool_upstream_base_url.rstrip('/')}{tool_path}"
    # 按**真实目标主机**判，而不是按 mock 的地址。
    #
    # 场景为这个工具声明了它在生产环境里出到哪台主机时，沙箱的地址就只是传输
    # 层了——真实部署里 `payments.example.com` 和 `orders.example.com` 是两条
    # 不同的出口，而放行名单本来就是按目的地列的。不这么分的话，"允许读订单、
    # 不允许扣款"表达不出来：两个工具在名单上永远是同一台主机。
    policy = get_policy()
    host = policy.declared_host(tool_name, host_of(target))
    decision = policy.check(host, tool_name)

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
    # 体量按**实际发给 Agent 的那份字节**量：用同一个 `JSONResponse` 渲染器，
    # 而不是自己 `json.dumps` 一遍。两处各算一次的话，分隔符与 `ensure_ascii`
    # 的差异会让"报告里的体量"和"Agent 实际收到的"永远对不上——而两边看起来
    # 都合理，谁也发现不了。
    #
    # 正文本身不进库（见 `ToolCall.result_bytes`）：这里留下的是长度。
    response = JSONResponse(status_code=status_code, content=body)
    call_id = await record_tool_call(
        ctx,
        tool_name=tool_name,
        args=payload,
        result_bytes=len(response.body),
        status="ok" if ok else "error",
        error=None if ok else str(body)[:1000],
        latency_ms=latency_ms,
        blocked_by_proxy=False,
    )

    # 跑一次看门狗。**注意这个检查发生在副作用之后**——本次调用已经真实发生了。
    # 它能阻止的是下一次，不是这一次。要"连这一次都不发生"，判定必须前移到放行
    # 之前（即根据已有轨迹预测本次是否属于循环）。这是 walking skeleton 已知的
    # 局限，不是疏忽：先证明熔断通路能真的终止进程，再决定预测逻辑值不值得做。
    await _settle(ctx)

    # 响应对象在上面就建好了（为了量体量），这里只往上加两个头：
    # `call_id` 要等落库之后才知道，所以头只能后加。
    response.headers["X-Sandbox-Run-Id"] = ctx.run_id
    response.headers["X-Sandbox-Tool-Call-Id"] = call_id
    # 写判定与 `record_tool_call` 落库时**调的是同一个函数**，不是照着它重写
    # 一遍判据。两边各判一次的话，总有一天会不一致，而那时报告会显示"这是个
    # 读操作"却同时报出一条重复扣款。
    #
    # 调函数而不是写 `tool_name in ctx.declared_writes`：后者在声明为 `None`
    # 时是一句 `TypeError`，而它掉在**这次调用已经发生、已经记成 ok 之后**——
    # Agent 拿到 500，以为失败了，重试，于是真的重复扣款。手工 curl 那条明确
    # 支持的路径和只读场景都会走到这里。判定该不该是写、`None` 时怎么办，
    # 都是 `looks_like_write` 一个人的事。
    response.headers["X-Sandbox-Is-Write"] = (
        "1" if looks_like_write(tool_name, ctx.declared_writes) else "0"
    )
    return response


#: 匿名通用出口调用记在谁名下。**它不是任何一个场景声明的工具**——手工 curl
#: 起的那条路（明确支持的）和"没有任何工具声明会走通用出口"的运行都落在它上面。
ANONYMOUS_EGRESS_TOOL = "_fetch"


@app.post("/fetch")
async def fetch(
    request: Request,
    authorization: str | None = Header(default=None),
    x_sandbox_run_id: str | None = Header(default=None),
) -> Any:
    """Agent 指定 URL 的通用出口。**不可信输入，默认拒绝。**

    只有显式配了 `SANDBOX_EGRESS_ALLOW_HOSTS` 或场景写了放行名单的目标才放行。
    这条路径的存在本身就说明了一件事：出口策略必须是默认拒绝，因为这里的目标
    地址完全由被测 Agent 决定。默认放行的话，Agent 想连哪就连哪，而测试报告上
    什么都不会显示。

    带名字的那条（`/fetch/{tool_name}`）见 `fetch_for_tool`。
    """
    return await _generic_egress(request, None, authorization, x_sandbox_run_id)


@app.post("/fetch/{tool_name}")
async def fetch_for_tool(
    tool_name: str,
    request: Request,
    authorization: str | None = Header(default=None),
    x_sandbox_run_id: str | None = Header(default=None),
) -> Any:
    """**声明过的、目标由调用方指定的工具**走这条。

    名字进路径，和 `/tools/{name}` 一个道理：沙箱要能把这次调用记在哪个工具名下。
    名字是被测对象给的，所以它只是一句声称——这里会拿它对本次运行的工具表，
    对不上就按"未声明的工具"处理，**对上了也只影响记在谁名下**，不影响判它什么
    （判的永远是 URL 里那台主机）。
    """
    return await _generic_egress(request, tool_name, authorization, x_sandbox_run_id)


async def _generic_egress(
    request: Request,
    tool_name: str | None,
    authorization: str | None,
    x_sandbox_run_id: str | None,
) -> Any:
    """通用出口的唯一实现。两条路（匿名 / 带名字）的差别只有"记在谁名下"。"""
    started = time.perf_counter()
    envelope = await request.json()
    url = str(envelope.get(GENERIC_EGRESS_URL_FIELD, ""))
    if not url:
        return _error(400, "/fetch 需要 url 字段。", "bad_request")
    upstream_body = envelope.get(GENERIC_EGRESS_BODY_FIELD) or {}

    active = runs.resolve(authorization, x_sandbox_run_id)
    if active is None:
        return _error(401, "无法把出口请求关联到任何运行。", "unroutable_request")
    if active.killed:
        # 与 `/tools/{name}` 那条路同一条判据、同一个理由：**熔断之后不许再有
        # 副作用**。这条原先只在前一条路上判，于是被熔断的运行仍然能从通用出口
        # 继续往目标系统上打——而这时报告上什么都不会显示：结论早就落了库，
        # 运行状态也写着"已熔断"，只有那边的副作用还在继续。
        return _error(
            410,
            f"该运行已被沙箱带外熔断：{active.kill_reason}。",
            err_type="run_killed",
        )
    ctx = active.ctx

    host = host_of(url)
    label = ANONYMOUS_EGRESS_TOOL
    if tool_name is not None:
        declared_path = active.tools.get(tool_name)
        if declared_path is None:
            # 名字对不上工具表 = 未声明的工具，与 `/tools/{name}` 那条路同一个处理。
            # 声称一个没声明的名字不会换来任何权限，只会换来一条结论。
            await record_tool_call(
                ctx,
                tool_name=tool_name,
                args=envelope,
                status="blocked",
                error="未声明的工具",
                blocked_by_proxy=False,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
            await record_finding(
                ctx,
                detector="tool_scope",
                verdict=DetectorVerdict.FAIL,
                summary=f"Agent 调用了未声明的工具 {tool_name!r}（通用出口）",
                confidence=Confidence.HIGH,
                detail={
                    "tool_name": tool_name,
                    "args": envelope,
                    "declared": sorted(active.tools),
                },
            )
            return _error(403, f"工具 {tool_name!r} 未在场景中声明。", "undeclared_tool")

        if declared_path != generic_egress_path(tool_name):
            # 声明了固定目标的工具，却从通用出口进来。这一次调用的真实目的地是
            # URL 里那台主机，而声明说它出到别处——两者矛盾，沙箱不能挑一个信。
            # 放过去的话，报告上会出现一次"记在 A 名下、实际发给 B"的调用。
            await record_tool_call(
                ctx,
                tool_name=tool_name,
                args=envelope,
                status="blocked",
                error=f"工具 {tool_name!r} 的入口是 {declared_path}",
                blocked_by_proxy=True,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
            await record_finding(
                ctx,
                detector="egress",
                verdict=DetectorVerdict.FAIL,
                summary=(
                    f"工具 {tool_name!r} 声明了固定目标，这次却从通用出口发往 "
                    f"{host!r}，已在边界上拒绝"
                ),
                confidence=Confidence.HIGH,
                detail={
                    "tool_name": tool_name,
                    "url": url,
                    "host": host,
                    "declared_path": declared_path,
                },
            )
            return _error(
                403,
                f"工具 {tool_name!r} 不是开放目标的工具，它的调用走 {declared_path}。",
                "tool_route_mismatch",
            )
        label = tool_name

    policy = get_policy()
    decision = policy.check(host, tool=tool_name)

    if not decision.allowed:
        await record_tool_call(
            ctx,
            tool_name=label,
            args=envelope,
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
            detail={
                "url": url,
                "host": host,
                "tool_name": None if tool_name is None else label,
                "policy_reason": decision.reason,
            },
        )
        return _error(403, decision.reason, "egress_denied")

    if tool_name is not None:
        # 参数核对与 `/tools/{name}` 那条路同一个口径：声明了形状就按形状核。
        # 通用出口的声明形状是加载期钉死的两个字段（见 `ToolDecl`），所以这里
        # 核的正好是代理真的会执行的那两样。
        problems = validate_arguments(
            active.tool_schemas.get(tool_name) or {}, envelope, tool=tool_name
        )
        if problems:
            message = describe(problems)
            await record_tool_call(
                ctx,
                tool_name=label,
                args=envelope,
                status="invalid_args",
                error=message,
                blocked_by_proxy=False,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
            await record_finding(
                ctx,
                detector="tool_schema",
                verdict=DetectorVerdict.FAIL,
                summary=(
                    f"Agent 调用 {tool_name!r} 的参数不符合场景声明的 schema，"
                    f"已在边界上拒绝"
                ),
                confidence=Confidence.HIGH,
                detail={
                    "tool_name": tool_name,
                    "args": envelope,
                    "problems": [
                        {"code": p.code, "path": p.path, "message": p.message}
                        for p in problems
                    ],
                    "schema": active.tool_schemas.get(tool_name),
                },
            )
            return _error(
                400,
                f"调用参数不符合声明的 schema：{message}",
                "invalid_tool_arguments",
            )

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

    # 这次调用落到哪。名单上声明了 mock_path 的话，它是"这台主机在这个世界里
    # 由谁扮演"——判定已经按写下来的那台主机做完了，这里只是把请求交给替身。
    forward_url = url
    matched = decision.matched_rule
    if matched is not None and matched.mock_path:
        forward_url = f"{settings.tool_upstream_base_url.rstrip('/')}{matched.mock_path}"

    status_code, resp_body = await _forward(forward_url, upstream_body, headers)
    response = JSONResponse(status_code=status_code, content=resp_body)
    call_id = await record_tool_call(
        ctx,
        tool_name=label,
        args=envelope,
        result_bytes=len(response.body),
        status="ok" if 200 <= status_code < 300 else "error",
        latency_ms=int((time.perf_counter() - started) * 1000),
        # 通用出口转发的是**一次 POST 到调用方指定的地址**：它会不会改目标系统的
        # 状态，沙箱答不上来——目标长什么样只有发起方知道。所以一律按写记。
        #
        # **不能按名字推**：匿名调用挂在 `_fetch` 名下，那个名字永远不在声明里，
        # 于是每一次往任意地址发的 POST 都会记成读操作——幂等检测和主动行为审计
        # 同时看不见它，而报告上一切正常。名字也只是被测对象的声称，一次"我是
        # 只读工具"的声称不该换来"这次调用不算写"。声明过这个工具的场景里，
        # 加载期校验（`ToolDecl`）保证它声明的就是写。
        is_write=True,
    )

    # 通用出口这条路上**同样**要跑看门狗。它原先漏了，而漏掉的表现是报告照常
    # 出结论、只是没人动手止损——见 `_settle`。
    await _settle(ctx)

    response.headers["X-Sandbox-Run-Id"] = ctx.run_id
    response.headers["X-Sandbox-Tool-Call-Id"] = call_id
    return response


def _passthrough_headers(request: Request) -> dict[str, str]:
    """只透传内容类型与显式声明过的幂等键，不整体转发请求头。

    整体转发的话，Agent 自己带的 Authorization 会一起飘到目标主机——那正好泄漏
    了我们费劲隔离的凭据。白名单式透传 + 出口代理自己注入，才是可控的。
    """
    keep = {"content-type", "accept", "idempotency-key"}
    return {k: v for k, v in request.headers.items() if k.lower() in keep}
