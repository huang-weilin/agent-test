"""流式转发，以及**一次调用的结算**。

这个模块干两件在代码上必须放在一起的事：把上游的帧**边收边转**给被测 Agent，
同时在转发的同时把这一串帧读成一次调用，然后结算、落库。

**为什么结算也在这里，而不是留在网关主文件里。** 因为流式的结算挂在生成器的
`finally` 上（客户端随时可能断开），而"上游没给 usage 时按什么记账"这条规则
**只有一份实现**——非流式的路径也调这里的 `settle_and_record`。两条路各写一份
的话，这条规则会漂成两个版本，而它们只在出问题的那一次才分叉：预算的账对不上
明细，两边的数看起来都正常。

**凭据**：真实上游的密钥只在这个文件里拼进请求头（`upstream_headers`），
与非流式路径共用同一个函数。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

from sandbox.config import settings
from sandbox.gateway.budget import TokenBudget
from sandbox.gateway.sse import (
    StreamAccumulator,
    chunks_of,
    done_frame,
    frame_of,
    parse_frame,
)
from sandbox.gateway.stub import scripted_upstream
from sandbox.trace.recorder import RunContext, record_llm_call

log = structlog.get_logger(__name__)

#: 真实上游的超时。流式下这是"两个帧之间"的上限，不是整个响应的上限。
UPSTREAM_TIMEOUT = 120.0

#: 输入/输出的字符→token 估算。**这里的注释原来写着"取 3 是往安全的方向估"，
#: 那句话对英文成立、对中文是反的**——实测（2026-10-01，真上游，四种内容各一次）：
#:
#:     纯中文      800 字符 →  480 token   1.67 字符/token
#:     中文 JSON  3582 字符 → 1636 token   2.19 字符/token   ← 工具返回体就长这样
#:     英文 JSON  1892 字符 →  623 token   3.04 字符/token
#:     纯英文     1980 字符 →  361 token   5.48 字符/token
#:
#: 取 3 的净效果是：**中文（本项目的常态）低估 27%–37%，英文高估约 45%**。
#: "低估"在止损上的方向是坏的那一边：预扣少于实际，熔断会晚一步。
#:
#: 为什么不改成一个"安全"的值：单个常数做不到同时对这两类内容安全——要让
#: 中文不低估就得取 ≤1.67，那英文会被高估 3 倍以上，预扣虚高会让预算在真实
#: 花费的三分之一处就熔断。而这一层估算**本来就只是粗筛**：一次请求预扣的大头
#: 是输出上界（没给 `max_tokens` 时 4096），字符估算那点偏差决定不了放不放行。
#: 所以这里选择把偏差**量出来、写在这儿**，并让 `tests/test_token_estimator.py`
#: 盯着它——而不是换一个看起来更顺眼的数。
_CHARS_PER_TOKEN = 3
#: 请求没给 max_tokens 时的补全上界。
_DEFAULT_MAX_COMPLETION = 4096


class UpstreamError(Exception):
    """真实上游返回了错误，或者根本没配好。"""

    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        self.detail = detail
        super().__init__(detail)


# --- 预扣 ---------------------------------------------------------------------


def estimate_input_tokens(payload: dict[str, Any]) -> int:
    """估算输入 token。**只用于预扣**，结算一律优先用上游返回的真实 usage。"""
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


@dataclass(frozen=True)
class Reserved:
    """一次请求预扣的两个估算值。

    打包成一个东西是因为它们**永远一起出现**：预扣、结算、归还三个动作拿到的
    必须是同一份数字。分开传的话，"归还时少减了一个"这种错法在调用点看不出来。
    """

    input_tokens: int
    output_tokens: int

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    @classmethod
    def of(cls, payload: dict[str, Any]) -> Reserved:
        """输出取 max_tokens，没给就按默认上界（宁可估多）。"""
        max_out = payload.get("max_tokens") or payload.get("max_completion_tokens")
        if not isinstance(max_out, int) or max_out <= 0:
            max_out = _DEFAULT_MAX_COMPLETION
        return cls(input_tokens=estimate_input_tokens(payload), output_tokens=max_out)


# --- 结算 ---------------------------------------------------------------------


@dataclass(frozen=True)
class Numbers:
    """这次调用按哪几个数记账，以及这些数是从哪来的。"""

    input_tokens: int
    output_tokens: int
    #: `ok` = 上游给的权威用量；`usage_estimated` = 没拿到，数字是网关估的。
    status: str


def numbers_for(
    usage: dict[str, Any] | None, reserved: Reserved, observed_output: int | None
) -> Numbers:
    """上游给了用量就用上游的；没给就按估算，**绝不记 0**。

    「没给就记 0」看着无害（反正我们不知道真实值），实际是这套止损里唯一真正
    致命的写法：`settle` 的语义是"退掉预扣、加上实际"，实际传 0 等于把预扣
    **全额退回**。一次跑飞了的流式循环会因此一分钱都不算——报告显示这次运行
    很便宜，Token 上限永远不会触发，而所有数字看起来都很正常。

    所以这里一律往估算的方向走，并且把这次调用记成 `usage_estimated`：
    **数字可以是估的，但不能是偷偷估的。**
    """
    if usage:
        return Numbers(
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            status="ok",
        )
    return Numbers(
        input_tokens=reserved.input_tokens,
        # 输出按**看到的内容**估：流式有机会数（每个分片都从我们手上过），
        # 非流式连内容都没经过网关，只能退回预扣时的上界。
        output_tokens=observed_output if observed_output is not None else reserved.output_tokens,
        status="usage_estimated",
    )


def estimate_output_tokens(acc: StreamAccumulator) -> int:
    """按累计到的内容估这次流产出了多少 token。

    正文和工具调用的参数分片都要数：一步只调工具、一个字正文都不产出的响应，
    只数正文会得到 0——而它恰恰是脚本化场景里最常见的那一种。
    """
    chars = len(acc.text)
    for slot in acc.tool_calls.values():
        chars += sum(len(fragment) for fragment in slot["arguments"])
    return max(1, chars // _CHARS_PER_TOKEN) if chars else 0


async def settle_and_record(
    ctx: RunContext,
    budget: TokenBudget,
    *,
    reserved: Reserved,
    model: str,
    usage: dict[str, Any] | None,
    observed_output: int | None,
    started: float,
    streaming: bool,
) -> dict[str, Any]:
    """结算 + 落库，返回 cost breakdown。

    **结算那一行必须是同步的、且排在最前面。** 它决定了被取消时会发生什么：
    客户端中途断开会让流式生成器被取消，取消点落在下面的 `await` 上时，
    预算的账**已经结清了**，丢的只是那条 DB 明细。反过来写（先落库再结算），
    一次断开会同时丢掉明细和账，而账是止损唯一的依据。
    """
    numbers = numbers_for(usage, reserved, observed_output)
    budget.settle(reserved.total, numbers.input_tokens, numbers.output_tokens)
    return await record_llm_call(
        ctx,
        model=model,
        input_tokens=numbers.input_tokens,
        output_tokens=numbers.output_tokens,
        cache_read_tokens=int((usage or {}).get("prompt_cache_hit_tokens") or 0),
        cache_write_tokens=int((usage or {}).get("prompt_cache_miss_tokens") or 0),
        latency_ms=int((time.perf_counter() - started) * 1000),
        streaming=streaming,
        status=numbers.status,
    )


# --- 上游 ---------------------------------------------------------------------


def upstream_url(base: str) -> str:
    """把 base_url 规范成 chat/completions 端点。两种写法都接受。"""
    base = base.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def upstream_headers() -> dict[str, str]:
    """真实上游的请求头。**真实密钥只在这一个地方拼进请求。**

    流式与非流式共用它，所以"密钥只出现在网关进程里"这句话只有一个实现处：
    哪天要换成别的鉴权方式，改的是这一行，而不是两处各改一遍。
    """
    if not settings.upstream_ready:
        raise UpstreamError(
            500,
            "upstream_mode=real 但没有配置 SANDBOX_UPSTREAM_API_KEY。"
            "拒绝静默降级到 stub——那会让测试结果看起来正常但毫无意义。",
        )
    return {
        "Authorization": f"Bearer {settings.upstream_api_key}",
        "Content-Type": "application/json",
    }


async def real_stream_lines(payload: dict[str, Any]) -> AsyncIterator[str]:
    """真实上游的流，一行一条。**不改写任何一帧**：客户端收到的就是上游发的。

    转发用的是原始行而不是解出来再重新编码的 JSON：重新编码会改掉键的顺序、
    空格和转义，而"我收到的东西和上游发的一模一样"是这条链路唯一值得信的地方。
    """
    url = upstream_url(settings.upstream_base_url)
    headers = upstream_headers()
    async with (
        httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client,
        client.stream("POST", url, json=payload, headers=headers) as resp,
    ):
        if resp.status_code >= 400:
            body = (await resp.aread()).decode("utf-8", "replace")
            raise UpstreamError(resp.status_code, body[:2000])
        async for line in resp.aiter_lines():
            yield line + "\n"


async def stub_stream_lines(
    run_id: str, model: str, reserved: Reserved, *, include_usage: bool
) -> AsyncIterator[str]:
    """脚本化上游的流：把一次脚本响应拆成帧。

    **游标与 `next_response` 是同一个**：流式与非流式消费脚本的方式完全一样，
    场景脚本第几步就是第几步——不会因为开了流式就整体错位。错位的话，
    "这个场景测的是重复付款"这句话就悄悄不成立了。
    """
    response = scripted_upstream.next_response(
        run_id, model=model, approx_input_tokens=reserved.input_tokens
    )
    for chunk in chunks_of(response, include_usage=include_usage):
        yield frame_of(chunk)
    yield done_frame()


# --- 补 usage 开关 -------------------------------------------------------------


def with_usage_option(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """给流式请求补上 `stream_options.include_usage`。返回 (新请求, 是否改过)。

    真实上游只有收到这个开关，才会在流末发**一帧 `choices` 为空、只带 usage 的
    帧**。没有它，网关的结算就永远只能靠估——而"唯一权威的计数点"是这个类存在
    的理由之一（见文件头的四条职责）。

    **调用方明确写了 `false` 的时候不覆盖。** 那是一次显式表态：它要么自己有
    另外的计数办法，要么就是想要一个干净的流。我们照做，但结算会退化成估算并
    把这件事写进 `LlmCall.status`，而不是假装数字是准的。

    返回的第二个值要**说出去**（网关会在响应头里写一行、文档里也写）：
    偷偷往调用方的请求里塞字段，而响应看起来一切正常，是那种事后没人查得出来
    的改动。
    """
    options = payload.get("stream_options")
    if isinstance(options, dict):
        if "include_usage" in options:
            # 它对这个开关表过态了，照它的来（哪怕是 `false`）。
            return payload, False
        # 它带了别的开关、但没提 usage：我们只添自己那一项，不动它的。
        return {**payload, "stream_options": {**options, "include_usage": True}}, True
    if options is not None:
        # 形状不对（不是个字典）。那是上游该去抱怨的东西，不该由我们替它改对。
        return payload, False
    return {**payload, "stream_options": {"include_usage": True}}, True


def usage_requested(payload: dict[str, Any]) -> bool:
    """补过之后，这次流到底要不要 usage 帧。"""
    options = payload.get("stream_options")
    return bool(isinstance(options, dict) and options.get("include_usage"))


# --- 转发 + 结算 ---------------------------------------------------------------


async def chain_first(first: str, rest: AsyncIterator[str]) -> AsyncIterator[str]:
    """把已经取出来的第一帧放回流里。

    存在只是因为"开流之前就会失败"的几种情况要能被挡在 200 之前：为了看清
    第一帧（脚本没注册、上游 4xx、连不上），必须先把流拉开一步，而拉出来的
    那一帧不能丢——丢了客户端会少收第一个分片，看起来像上游发的东西不完整。
    """
    yield first
    async for line in rest:
        yield line


async def relay_stream(
    lines: AsyncIterator[str],
    *,
    ctx: RunContext,
    budget: TokenBudget,
    reserved: Reserved,
    model: str,
    started: float,
) -> AsyncIterator[bytes]:
    """边转边读：把上游的帧发给客户端，同时累计成一次调用。

    **结算挂在 `finally` 上，这是这块代码的全部要点。** 客户端随时可能中途
    断开（被看门狗掐掉、自己崩了、用户关了窗口）：生成器被取消时 `finally`
    照样跑，于是这次调用的账照结。断开的是接收方，上游那次调用是真的发生了——
    不结的话，一套"把 Agent 打断在流中间"的测试会系统性地少算钱。
    """
    acc = StreamAccumulator()
    try:
        async for line in lines:
            chunk = parse_frame(line)
            if chunk is not None:
                acc.feed(chunk)
            # 转发的字节就是上游给的字节。看不懂的帧也照转（见 sse.parse_frame）。
            yield line.encode("utf-8")
    finally:
        if not acc.finish_reason:
            # 没见到收尾帧：要么客户端断开，要么上游半路没了。两种都值得留痕，
            # 因为它们都会让"这次调用产出了什么"变得不完整。
            log.warning("gateway.stream_cut_short", run_id=ctx.run_id, chunks=acc.chunks)
        if acc.unmerged_tool_fragments:
            # 拼不起来的工具分片。它不会让账算错，但会让**客户端**收到半个
            # 参数——而那个失败在 Agent 那边看起来是它自己解析错了。
            log.warning(
                "gateway.stream_unmerged_tool_fragments",
                run_id=ctx.run_id,
                count=acc.unmerged_tool_fragments,
            )
        try:
            await settle_and_record(
                ctx,
                budget,
                reserved=reserved,
                model=model,
                usage=acc.usage,
                observed_output=estimate_output_tokens(acc),
                started=started,
                streaming=True,
            )
        except asyncio.CancelledError:
            # 生成器正被取消。预算已经结清（上面那一行是同步的），**丢的是明细**：
            # 成本是从明细重算的，所以这次调用会少算一次。这条日志是它唯一的痕迹，
            # 别把它读成"没事"。
            log.warning("gateway.stream_record_lost", run_id=ctx.run_id)
            raise
