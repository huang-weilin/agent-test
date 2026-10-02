"""流式：边转发边结算。

网关原来对 `stream: true` 回 501，理由写在响应体里——「流式需要边转发边累计
usage，在结算正确性上另有一套逻辑，不走通不发布」。这个文件就是那套逻辑的
判据。它盯的**不是"流式能用"**，而是三件会静默出错的事：

1. 转给客户端的帧是不是上游原样的帧（少一帧、拼错一帧，Agent 那边报的是它
   自己解析失败）；
2. 结算用的是上游给的权威 usage，还是我们估的数——两者差一次真实的账单；
3. **流式这次调用的钱算不算得了、账记不记得全**。这条原先写的是"客户端中途
   断开时"，**那是把范围说小了**：2026-10-03 量到的是每一次被取消的流式写都会
   丢明细——结账那一半同步加数、取消吃不掉，明细那一半会被吃掉，于是报告里
   这几笔调用整个不见了（同一段脚本，非流式那支落 3 条、流式那支只落 1 条，
   而两个运行都是绿的）。现在明细的落库脱开了请求的生命周期，缺口另外用**对账**
   兜着：`tests/test_gateway_stream.py` 里那对受控对照比的是**账**（条数与
   `unrecorded_llm_calls`），不只是结论——原先那句 `assert rows` 分不出
   "3 次落了 3 条"和"3 次落了 1 条"。

那个缺口的来龙去脉与判据在 `docs/cost.md` 的「账没对平」一节；"这条链上拆掉
哪一处谁先红"由仓库外的 `_probes/probe_stream_records.py` 一次量八个方向
（其中一条是只改散文的绿灯对照：**只改散文必须什么都不红**）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import anyio
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from sandbox.db.models import LlmCall
from sandbox.gateway import app as gateway_app
from sandbox.gateway.sse import StreamAccumulator, parse_frame
from sandbox.gateway.stream import Reserved, relay_stream
from sandbox.gateway.stub import scripted_upstream
from sandbox.runtime import runs

STEPS = [
    {"tool_calls": [{"name": "create_payment", "arguments": {"order_id": "A1"}}]},
    {"content": "已经处理好了。"},
]


@pytest_asyncio.fixture
async def client(temp_db: str):
    transport = ASGITransport(app=gateway_app.app)
    async with AsyncClient(transport=transport, base_url="http://gw.test") as c:
        yield c


@pytest.fixture(autouse=True)
def _clean_registry():
    runs.clear()
    yield
    runs.clear()


async def _register(client: AsyncClient, **overrides) -> dict:
    payload = {
        "scenario_id": "stream-test",
        "agent_version": "react_demo@0.1",
        "declared_writes": ["create_payment"],
        "token_ceiling": 100_000,
        **overrides,
    }
    resp = await client.post("/_sandbox/runs", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _chat_body(**overrides) -> dict:
    return {
        "model": "stub-model",
        "messages": [{"role": "user", "content": "帮我下单"}],
        "stream": True,
        **overrides,
    }


def frames_of(text: str) -> list[str]:
    """把响应体切成一条条 SSE 记录。空行是记录分隔，不是内容。"""
    return [line for line in text.split("\n") if line.strip()]


# --- 转发 ---------------------------------------------------------------------


async def test_a_streaming_call_comes_back_as_frames(client) -> None:
    """流式请求要真的以 SSE 回来，而不是被降级成一次性 JSON。

    降级会**看起来能用**：客户端的解析器拿到一个非流式的 JSON body 通常不报错，
    只是"一个 token 都没流出来"。报告上这次运行一切正常，而"边收边渲染"这条
    路径一次都没被走过。
    """
    created = await _register(client)
    scripted_upstream.register(created["run_id"], STEPS)

    resp = await client.post("/v1/chat/completions", json=_chat_body())

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    frames = frames_of(resp.text)
    assert frames[0].startswith("data: {"), "第一帧就该是 data 帧"
    assert frames[-1] == "data: [DONE]", (
        "流必须以 [DONE] 收尾：少它的话，客户端的流式解析器会一直等下去"
    )


async def test_the_frames_carry_the_same_content_as_the_script(client) -> None:
    """帧拼起来的内容必须等于脚本这一步。

    这里做的正是**客户端要做的那件事**：逐帧解析、把分片拼回去。参数被切成
    8 个字符一片，所以"响应体里能搜到 `order_id`"这种断言是不成立的——真正
    要证明的是拼回来是一段合法 JSON、而且内容对得上。网关转发路上丢掉任何
    一片，这里都会散架。
    """
    created = await _register(client)
    scripted_upstream.register(created["run_id"], STEPS)

    resp = await client.post("/v1/chat/completions", json=_chat_body())

    assert "create_payment" in resp.text, "函数名不必分片，应当在某一帧里原样出现"

    acc = StreamAccumulator()
    for line in frames_of(resp.text):
        chunk = parse_frame(line)
        if chunk is not None:
            acc.feed(chunk)
    message = acc.to_response()["choices"][0]["message"]

    assert acc.to_response()["choices"][0]["finish_reason"] == "tool_calls"
    call = message["tool_calls"][0]
    assert call["function"]["name"] == "create_payment"
    assert json.loads(call["function"]["arguments"]) == {"order_id": "A1"}, (
        "分片拼回来必须是一段合法 JSON：拼错了的话，被测 Agent 那边报的是"
        "「工具参数解析失败」——看起来像它的缺陷"
    )


async def test_the_script_cursor_advances_once_per_streaming_call(client) -> None:
    """流式与非流式消费脚本的方式必须一样：第几次调用就是第几步。

    游标错位的话，"这个场景测的是重复付款"这句话会悄悄不成立——脚本还在，
    只是被读串了行，而报告上看不出任何异常。
    """
    created = await _register(client)
    scripted_upstream.register(created["run_id"], STEPS)
    run_id = created["run_id"]

    first = await client.post("/v1/chat/completions", json=_chat_body())
    second = await client.post("/v1/chat/completions", json=_chat_body())

    assert "create_payment" in first.text
    assert "已经处理好了" in second.text
    assert scripted_upstream.calls_made(run_id) == 2


# --- 结算 ---------------------------------------------------------------------


async def test_a_streaming_call_is_settled_with_the_upstream_usage(client, db) -> None:
    """拿到 usage 帧时，结算用上游的数，并把这次调用标成流式。

    `LlmCall.streaming` 这一列在流式做出来之前是个**没有产出方的字段**：
    写在那儿、默认 `False`、没有任何人来写它。这条断言就是它的产出方。
    """
    created = await _register(client)
    scripted_upstream.register(created["run_id"], STEPS)

    resp = await client.post("/v1/chat/completions", json=_chat_body())
    assert resp.status_code == 200

    rows = (await db.execute(select(LlmCall))).scalars().all()
    assert len(rows) == 1, "一次流式调用应当落一条明细"
    assert rows[0].streaming is True
    assert rows[0].status == "ok", "拿到了权威用量，就不该标成估算"
    assert rows[0].input_tokens > 0 and rows[0].output_tokens > 0

    snapshot = (await client.get("/_sandbox/runs")).json()["runs"][0]
    assert snapshot["used"] == rows[0].input_tokens + rows[0].output_tokens
    assert snapshot["remaining"] == 100_000 - snapshot["used"], "预扣必须结清，不能留着"


async def test_the_gateway_adds_the_usage_option_and_says_so(client) -> None:
    """调用方没提 usage 开关时，网关补上它——并且在响应里留下痕迹。

    补这个开关是**改动调用方的请求**。改动本身有充分理由（没有它结算永远只能
    靠估），但"偷偷改"是另一回事：响应头里那一行是这次改动唯一的痕迹。
    """
    created = await _register(client)
    scripted_upstream.register(created["run_id"], STEPS)

    resp = await client.post("/v1/chat/completions", json=_chat_body())

    assert resp.headers.get("X-Sandbox-Stream-Options-Added") == "include_usage"
    assert "usage" in resp.text, "补了开关，上游的 usage 帧就该出现在流里"


async def test_a_client_that_asked_for_a_clean_stream_gets_no_usage_and_we_say_so(
    client, db
) -> None:
    """调用方明确写了 `include_usage: false`：照它的来，然后**按估记账并标出来**。

    这一条是这整套里最要紧的一根钉子。没有 usage 帧时若按 0 结算，
    `settle` 会把预扣**全额退回**——这次调用一分钱不算，Token 上限永远不会
    触发，而报告上一切正常。所以：数字可以是估的，但不能是偷偷估的。
    """
    created = await _register(client)
    scripted_upstream.register(created["run_id"], STEPS)

    resp = await client.post(
        "/v1/chat/completions",
        json=_chat_body(stream_options={"include_usage": False}),
    )

    assert resp.headers.get("X-Sandbox-Stream-Options-Added") is None, (
        "调用方表过态了，我们不该再动它的请求"
    )
    assert "usage" not in resp.text

    row = (await db.execute(select(LlmCall))).scalars().one()
    assert row.status == "usage_estimated", (
        "没拿到权威用量的调用必须被标出来——报告里那一行「这几个数不是账单」"
        "读的就是它"
    )
    assert row.output_tokens > 0, (
        "输出 token 按**看到的内容**估：这一步只有工具调用、一个字正文都没有，"
        "只数正文会得到 0，而 0 会连带把预扣退回去"
    )

    snapshot = (await client.get("/_sandbox/runs")).json()["runs"][0]
    assert snapshot["used"] == row.input_tokens + row.output_tokens
    assert snapshot["remaining"] == 100_000 - snapshot["used"], "预扣必须结清"


async def test_a_client_disconnect_still_pays_for_the_call(client, db) -> None:
    """客户端中断：钱照付，预扣照结清，明细照落。

    断开的是接收方。上游那次调用是真的发生了、真的计费了——按"客户端不要了"
    就退账的话，一套专门测"把 Agent 掐在流中间"的用例会系统性地少算钱，
    而且是越打断越省钱，正好把止损的方向搞反。

    这里用的是 `aclose()`，也就是**优雅关闭**：`finally` 里的 `await` 会一路
    跑完，所以明细也落得下来。真的被取消时明细会丢、而账不能丢——那是下一条。
    """
    created = await _register(client)
    scripted_upstream.register(created["run_id"], STEPS[:1])
    active = runs.get(created["run_id"])
    assert active is not None
    budget = active.budget
    reserved = Reserved(input_tokens=100, output_tokens=200)
    budget.reserve(reserved.total)

    stream = relay_stream(
        _lines_from_stub(created["run_id"]),
        ctx=active.ctx,
        budget=budget,
        reserved=reserved,
        model="stub-model",
        started=time.perf_counter(),
    )

    seen = 0
    async for _frame in stream:
        seen += 1
        if seen == 3:
            break  # 客户端在第 3 帧之后走人
    await stream.aclose()

    assert budget.used > 0, "断开也要结账"
    assert budget.remaining == budget.ceiling - budget.used, (
        "预扣必须结清：`remaining` 里还压着预扣的话，预算会越过越少，"
        "而每一次都看起来正常"
    )

    row = (await db.execute(select(LlmCall))).scalars().one()
    assert row.streaming is True
    assert row.status == "usage_estimated", (
        "只转了三帧就被掐断，usage 帧根本没到——这次调用的数只能是估的"
    )


async def test_a_really_cancelled_stream_settles_the_account_and_records_the_call(
    client, db, monkeypatch
) -> None:
    """**真的取消**（不是优雅关闭）：账要结清，明细也要落下来。

    上一条用的是 `aclose()`——那是优雅关闭，`finally` 里的 `await` 会一路跑完，
    所以它证明不了取消时会发生什么。真实场景是**取消**：看门狗掐人、客户端读完
    最后一帧就关连接（starlette 随即取消整个任务组）。

    两半的待遇是分开定的，这是这块代码最要紧的一处：

    * **结算在前、而且是同步的** → 取消再狠，账也结清了。写成"先落库再结算"的话，
      取消点正好落在落库那个 `await` 上，**结算永远跑不到**，账停在预扣上：报告
      显示这次运行很便宜，Token 上限再也不会被触发。
    * **明细交给一个被 `shield` 护住的任务** → 外层的取消传不进去，它自己跑完。

    2026-10-03 之前这里没有 shield，断言写的是 `rows == []`——"明细会丢，这是
    已知的代价"。那句话当时是对的，却盖住了一个更大的事实：丢的不是"取消时"那
    一条，而是**每次调用收尾那一条**。同一份世界、同一段脚本，流式的
    `stream_clean` 只落 1 条明细、报告写「1 次 LLM 调用」，而非流式的孪生场景
    `clean` 落 3 条——两边在报告上都像正常的数字。

    改坏了不报错、不变慢：`used` 停在 0 是结算没跑到，明细那条不见了是 shield
    没了。所以两件事分开量。
    """
    from sandbox.gateway import stream as stream_mod

    created = await _register(client)
    scripted_upstream.register(created["run_id"], STEPS[:1])
    active = runs.get(created["run_id"])
    assert active is not None
    budget = active.budget
    reserved = Reserved(input_tokens=100, output_tokens=200)
    budget.reserve(reserved.total)

    events: list[str] = []
    monkeypatch.setattr(
        stream_mod.log, "warning", lambda event, **_: events.append(event)
    )

    seen = 0
    with anyio.CancelScope() as scope:
        stream = relay_stream(
            _lines_from_stub(created["run_id"]),
            ctx=active.ctx,
            budget=budget,
            reserved=reserved,
            model="stub-model",
            started=time.perf_counter(),
        )
        async for _frame in stream:
            seen += 1
            if seen == 3:
                scope.cancel()

    assert seen == 3, "取消要发生在流中间，否则测的是别的东西"
    assert budget.used > 0, (
        "取消也要结账。`used` 停在 0 = 结算那一步根本没跑到——"
        "取消点吃掉的应该是明细，不是账"
    )
    assert budget.remaining == budget.ceiling - budget.used, "预扣必须结清"

    # 被护住的那次落库还在后台跑。等它回来——运行收尾时做的是同一件事
    # （`scenario/runner.py` 里那个 drain），所以这里量的正是那条路径。
    assert await stream_mod.drain_detached_records() == 0, (
        "落库没在超时内跑完。真跑不完也不该把运行卡住，但那是另一条路"
        "（报告里的对账会照实说少了几笔），不该由这个测试静默放过"
    )

    rows = (await db.execute(select(LlmCall))).scalars().all()
    assert len(rows) == 1, (
        "取消吃掉的是**转发**，不是这次调用的明细：上游那次调用真的发生了、"
        "真的计费了，明细就得落下来。少这一条，报告里的次数和金额都会少算，"
        "而它看起来和一次干净的运行一模一样"
    )
    assert rows[0].streaming is True
    assert rows[0].status == "usage_estimated", (
        "只转了三帧就被掐断，usage 帧根本没到——这次调用的数只能是估的"
    )
    assert "gateway.stream_record_detached" in events, (
        "明细改由后台写完时日志里要留一条：这是「落库脱离了请求生命周期」"
        "唯一的痕迹，别把它读成丢明细"
    )


async def test_a_lost_detail_is_reported_as_a_lower_bound(client, db, monkeypatch) -> None:
    """明细真的丢了的时候，报告必须**自己说出来**，而不是照着一个偏小的数说话。

    这一条把"丢"做成真的：头一次落库就抛（库锁了、盘满了、连接拿不到——shield
    护得住取消，护不住"写就是写不成"）。以前这种失败没有出口：日志里一句之外，
    报告照旧写「1 次 LLM 调用，449 in / 20 out」——那个数看着和一次真的只调了
    一次的运行一模一样。

    判据是两件事一起看：**网关那本内存账**（`budget.calls`，在 `settle` 里同步
    加一，取消和异常都吃不掉）和**明细条数**。差额就是丢了几笔，而它就是报告里
    那句"下界"的由来。
    """
    from sandbox.cost import aggregator as cost
    from sandbox.gateway import stream as stream_mod
    from sandbox.scenario import runner as runner_mod

    created = await _register(client)
    run_id = created["run_id"]
    scripted_upstream.register(run_id, STEPS)
    active = runs.get(run_id)
    assert active is not None

    real = stream_mod.record_llm_call
    seen = {"n": 0}

    async def flaky(*args, **kwargs):  # type: ignore[no-untyped-def]
        seen["n"] += 1
        if seen["n"] == 1:
            raise RuntimeError("这一条测的就是落库失败这条路")
        return await real(*args, **kwargs)

    monkeypatch.setattr(stream_mod, "record_llm_call", flaky)

    first = await client.post("/v1/chat/completions", json=_chat_body())
    second = await client.post("/v1/chat/completions", json=_chat_body())
    assert first.status_code == 200 and second.status_code == 200, (
        "沙箱自己的账写不进去，不该把客户端的流带崩：帧已经发完了，那边没有任何错"
    )

    await stream_mod.drain_detached_records()

    result = runner_mod.ScenarioResult(scenario_id="stream-test", run_id=run_id)
    result.cost = await cost.for_run(run_id)
    assert result.cost.llm_calls == 1, "两次调用里丢了一条明细"
    assert active.budget.calls == 2, (
        "账是同步结的两次。它停在 1 的话，说明丢的那一条账面也没结——"
        "那是另一件事（止损会失效），比少一条明细严重得多"
    )
    result.unrecorded_llm_calls = runner_mod.reconcile_llm_calls(
        settled=active.budget.calls, recorded=result.cost.llm_calls
    )
    assert result.unrecorded_llm_calls == 1

    text = runner_mod.render(result)
    assert "有 1 次调用的账结了、明细没落" in text, (
        "少了一条明细就必须说出来。不说的话，上面那个「1 次 LLM 调用」和"
        "一次真的只调了一次的运行长得一模一样"
    )
    assert "是下界，不是账单" in text


async def _lines_from_stub(run_id: str):
    """直接拿脚本化上游的帧序列，绕开 HTTP。

    断开这件事在 ASGI 测试客户端里做不出来（它会把整个响应读完），所以这一条
    测的是转发器本身——而"谁在什么时候关掉这个生成器"正是它要防的那件事。
    """
    from sandbox.gateway.stream import stub_stream_lines

    async for line in stub_stream_lines(
        run_id, "stub-model", Reserved(input_tokens=100, output_tokens=200), include_usage=True
    ):
        # 真上游的帧之间一定有等待（网络），这里补一个落点：没有它，取消就
        # 无处可落——生成器会一路把整段脚本走完，"中途取消"这件事根本没发生。
        await anyio.sleep(0)
        yield line


# --- 开流之前就会失败的那几种 ---------------------------------------------------


async def test_a_stream_with_no_registered_script_is_refused_before_the_stream(
    client,
) -> None:
    """脚本没注册：要在**开流之前**报错，而不是回一个 200 再断掉。

    先发 200 再失败的话，客户端拿到的是一个断掉的流，错误信息只剩网关日志；
    这里还能给一个正常的错误响应，而且预扣必须归还。
    """
    await _register(client)
    resp = await client.post("/v1/chat/completions", json=_chat_body())

    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "stub_script_missing"

    snapshot = (await client.get("/_sandbox/runs")).json()["runs"][0]
    assert snapshot["remaining"] == 100_000, "开流失败也必须归还预扣"


async def test_streaming_respects_the_budget_ceiling(client) -> None:
    """熔断在流式下同样发生在**放行之前**。"""
    await _register(client, token_ceiling=50)
    resp = await client.post("/v1/chat/completions", json=_chat_body())

    assert resp.status_code == 429
    assert resp.json()["error"]["type"] == "budget_exceeded"


async def test_a_killed_run_refuses_streaming_requests_too(client) -> None:
    """熔断之后放行任何一个请求，都等于熔断没发生过——流式也不例外。"""
    created = await _register(client)
    runs.kill(created["run_id"], "测试用的带外熔断")

    resp = await client.post("/v1/chat/completions", json=_chat_body())
    assert resp.status_code == 410
    assert resp.json()["error"]["type"] == "run_killed"


# --- 活样本：真的跑一个开着流式的 Agent ------------------------------------------
#
# 上面那些测的是网关自己。这一个测的是**这条路径真的接上了**：`stream_clean`
# 与 `clean` 是同一份世界、同一段脚本，唯一的变量是 Agent 打网关时走不走流式
# （账本里那一条）。两边的结论必须一字不差，而账要从上游的 usage 来。

ROOT = Path(__file__).resolve().parents[1]
AGENT_PYTHON = ROOT / "agents" / "react_demo" / ".venv"


async def test_the_streaming_pair_really_runs_and_still_pays_once(
    temp_db: str, monkeypatch
) -> None:
    """真跑两遍：结论一模一样，各付一次款，流式那支的账是上游给的。

    这一条同时钉住两个方向。只有"结论一样"的话，一条把流式请求**静默降级**成
    非流式的网关也能全绿——而那样"边收边渲染"这条路径一次都没被走过；只有
    "落了一条 streaming 记录"的话，一个把工具参数拼坏的转发器也能全绿。
    """
    if not AGENT_PYTHON.exists():
        pytest.skip(f"被测 Agent 的 venv 不在（{AGENT_PYTHON}），跳过端到端流式对照")

    from sandbox.db.engine import session_scope
    from sandbox.db.models import LlmCall as Call
    from sandbox.gateway import stream as stream_mod
    from sandbox.scenario import runner as runner_mod
    from sandbox.scenario.loader import load_scenario
    from sandbox.scenario.runner import ScenarioRunner

    # 收尾前那个 drain 也要有人看着。它没有自己的断言对象：删掉它，这一条里
    # 两个运行照样是绿的（明细早就落好了），而真实情形——**运行紧跟着一次被取消
    # 的调用结束**——会把那几次调用整段算漏。所以这里放一个探针函数，量的是
    # "收尾时它被叫过"，不是"它做对了什么"。
    drained: list[float] = []
    real_drain = stream_mod.drain_detached_records

    async def spy(timeout: float = stream_mod.DRAIN_TIMEOUT) -> int:
        drained.append(timeout)
        return await real_drain(timeout)

    monkeypatch.setattr(runner_mod, "drain_detached_records", spy)

    runner = ScenarioRunner(session_timeout_s=120.0)
    streamed = await runner.run(load_scenario(ROOT / "scenarios" / "stream_clean.yaml"))
    plain = await runner.run(load_scenario(ROOT / "scenarios" / "clean.yaml"))

    assert streamed.error is None, streamed.error
    assert plain.error is None, plain.error
    assert streamed.ok, streamed.expectation and streamed.expectation.to_dict()
    assert plain.ok, plain.expectation and plain.expectation.to_dict()

    # 钱：两支各付一次。参数被拼坏的话，付款那一步根本走不到。
    assert streamed.egress_charges == 1
    assert plain.egress_charges == 1

    # 结论：同一批检测器、同样的判决。流式那支要是把工具调用的参数拼坏了，
    # 这里会先炸——那时 Agent 收到的是半个 JSON。
    assert [(f.detector, f.verdict) for f in streamed.findings] == [
        (f.detector, f.verdict) for f in plain.findings
    ]

    async with session_scope() as db:
        rows = (
            (await db.execute(select(Call).where(Call.test_run_id == streamed.run_id)))
            .scalars()
            .all()
        )
        plain_rows = (
            (await db.execute(select(Call).where(Call.test_run_id == plain.run_id)))
            .scalars()
            .all()
        )

    # **配对要比的是账，不只是结论。**
    #
    # 这一段是 2026-10-03 补的。原先这里只有 `assert rows`——它分不出"3 次调用
    # 落了 3 条"和"3 次调用落了 1 条"，而后者正是当时的事实：流式那支只落 1 条，
    # 报告写「1 次 LLM 调用，449 in / 20 out」，非流式那支落 3 条、写
    # 「3 次 LLM 调用，1,844 in / 55 out」。两个场景同一份世界、同一段脚本、
    # 同样的步数，调用次数必须一样——**断言"非空"的闸门对"少了大半"是瞎的**。
    assert len(rows) == len(plain_rows), (
        f"同一段脚本，流式落了 {len(rows)} 条明细、非流式落了 {len(plain_rows)} 条。"
        "两边调用次数必须一样：少的是被取消吃掉的明细，而报告只认明细"
    )
    # 账要对得上：网关那本内存账结了几次，明细就该有几条。
    assert streamed.unrecorded_llm_calls == 0, (
        "结账次数与明细条数对不上——这一栏是报告里那句「少算了这几笔」的由来"
    )
    assert plain.unrecorded_llm_calls == 0
    assert drained, (
        "运行收尾前要先把挂着的落库等回来（`drain_detached_records`）。这一步"
        "被删掉的话，这一次运行照样是绿的——绿的只是**这一次**：明细落完和收尾"
        "之间隔了好几秒。真正会漏的是「运行紧跟着一次被取消的调用结束」那种时刻，"
        "而它会以「报告算少了几笔」的样子出现，正是这一片修掉的那件事"
    )
    assert rows, "一次 LLM 调用都没落库"
    assert all(row.streaming for row in rows), (
        "这一支的 Agent 是开着流式跑的（`DEMO_AGENT_STREAM=1`），落下来的调用"
        "必须都是流式——否则测的是对照组，而报告上完全看不出来"
    )
    assert all(row.status == "ok" for row in rows), (
        "网关替调用方补了 `include_usage`，所以权威用量应当拿到了——"
        "拿到还标成估算的话，那条上报线（报告、界面、回归对比）说的都是假话"
    )
    assert all(row.input_tokens > 0 and row.output_tokens > 0 for row in rows)
