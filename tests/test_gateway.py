"""网关是 ADR 001 的承重墙：路由、观测、止损、凭据隔离，四项都要有断言。"""

from __future__ import annotations

from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from sandbox.config import settings
from sandbox.db.models import DetectorFinding, LlmCall
from sandbox.gateway import app as gateway_app
from sandbox.gateway.stub import scripted_upstream
from sandbox.runtime import make_run_token, parse_run_token, runs


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
        "scenario_id": "scen-idem",
        "agent_version": "react_demo@0.1",
        "config": {"prompt": "v1"},
        "declared_writes": ["create_payment"],
        **overrides,
    }
    resp = await client.post("/_sandbox/runs", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _chat_body(**overrides) -> dict:
    return {
        "model": "stub-model",
        "messages": [{"role": "user", "content": "帮我下单"}],
        "stream": False,
        **overrides,
    }


async def test_request_with_no_active_run_is_rejected(client) -> None:
    resp = await client.post("/v1/chat/completions", json=_chat_body())
    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "unroutable_request"


async def test_ambiguous_request_is_rejected_not_guessed(client) -> None:
    """两个运行同时活跃且请求无标识时必须拒绝。

    猜错会把 A 的调用记到 B 头上，而且事后从数据里看不出来——这类串台比报错
    危险得多。
    """
    await _register(client, scenario_id="A")
    await _register(client, scenario_id="B")

    resp = await client.post("/v1/chat/completions", json=_chat_body())
    assert resp.status_code == 401


async def test_single_active_run_allows_unidentified_request(client) -> None:
    """只有一个运行活跃时，连 api_key 都不必改——这是"只改 base_url"的兜底路径。"""
    created = await _register(client)
    scripted_upstream.register(created["run_id"], [{"content": "好的"}])
    resp = await client.post("/v1/chat/completions", json=_chat_body())

    assert resp.status_code == 200, resp.text
    assert resp.headers["X-Sandbox-Run-Id"] == created["run_id"]


async def test_token_routes_to_the_right_run_among_many(client) -> None:
    a = await _register(client, scenario_id="A")
    b = await _register(client, scenario_id="B")
    scripted_upstream.register(a["run_id"], [{"content": "我是 A"}])
    scripted_upstream.register(b["run_id"], [{"content": "我是 B"}])

    resp = await client.post(
        "/v1/chat/completions",
        json=_chat_body(),
        headers={"Authorization": f"Bearer {b['api_key']}"},
    )
    assert resp.status_code == 200
    assert resp.headers["X-Sandbox-Run-Id"] == b["run_id"]
    assert resp.headers["X-Sandbox-Run-Id"] != a["run_id"]
    # 不只看头，还要看内容确实来自 B 的脚本——头对但内容串台是最坏的情况
    assert resp.json()["choices"][0]["message"]["content"] == "我是 B"


async def test_run_token_roundtrip() -> None:
    assert parse_run_token(make_run_token("abc-123")) == "abc-123"
    assert parse_run_token("Bearer sk-sandbox-abc-123") == "abc-123"
    # 长得像真实上游密钥的字符串必须认不出——不能被误当成运行标识。
    # 这里用构造的假值：真实密钥只存在于 .env，任何受版本控制的文件里都不该出现。
    assert parse_run_token("Bearer sk-0123456789abcdef0123456789abcdef") is None
    assert parse_run_token(None) is None


async def test_llm_call_is_recorded_with_gateway_authoritative_counts(client, db) -> None:
    created = await _register(client)
    scripted_upstream.register(created["run_id"], [{"content": "好的"}])

    resp = await client.post("/v1/chat/completions", json=_chat_body())
    assert resp.status_code == 200

    rows = (await db.execute(select(LlmCall))).scalars().all()
    assert len(rows) == 1
    assert rows[0].model == "stub-model"
    assert rows[0].input_tokens > 0 and rows[0].output_tokens > 0
    assert rows[0].test_run_id == created["run_id"]


async def test_request_bytes_is_the_body_we_received_not_a_reserialization(client, db) -> None:
    """`request_bytes` 量的是**收到的那份字节**。

    按工具的租金要折算 token，用的就是这个数。自己把 payload `json.dumps` 一遍再取
    长度会得到另一个数——键序、空格、转义都变了——于是"报告里的字节数"和"Agent
    真正发出来的"永远对不上，而两边看起来都合理。

    这条测试的**输入里故意留着多余的空格**：重新序列化一遍，那个长度就变了。
    """
    created = await _register(client)
    scripted_upstream.register(created["run_id"], [{"content": "好的"}])

    # 注意末尾那几个空格：它们在 JSON 里没有意义，但它们是**发出来的字节**。
    raw = '{"model":   "stub-model",   "messages": [{"role": "user", "content": "帮我下单"}]}'
    resp = await client.post(
        "/v1/chat/completions",
        content=raw.encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 200

    row = (await db.execute(select(LlmCall))).scalars().one()
    assert row.request_bytes == len(raw.encode("utf-8"))
    # 对照：重新序列化会得到另一个长度。没有这一句，"等于 len(raw)"可能只是
    # 因为两种写法碰巧同长。
    import json as _json  # noqa: PLC0415

    assert len(_json.dumps(_json.loads(raw))) != len(raw)


async def test_streaming_request_bytes_is_what_the_caller_sent(client, db) -> None:
    """流式那条路也一样，而且这里还多一层：网关会往请求里补 `include_usage`。

    记下来的必须是**调用方发的那份**，不是补过字段之后的——补过的那份是网关自己
    造的东西，拿它当"Agent 发出的字节"是在报告里写一句不成立的话。
    """
    created = await _register(client)
    scripted_upstream.register(created["run_id"], [{"content": "好的"}])

    # 手写这一串、不走 `json.dumps`：要的就是"调用方发来的那份字节"里
    # 没有 `stream_options`，而网关补完之后长度会变。
    raw = (
        '{"model": "stub-model", "messages": [{"role": "user", "content": "帮我下单"}],'
        ' "stream": true}'
    )
    resp = await client.post(
        "/v1/chat/completions",
        content=raw.encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 200
    # 先说清这一局真的走了"补字段"那条路：没有它，下面那条断言的对照就不成立。
    assert resp.headers["X-Sandbox-Stream-Options-Added"] == "include_usage"

    row = (await db.execute(select(LlmCall))).scalars().one()
    assert row.request_bytes == len(raw.encode("utf-8"))
    assert "stream_options" not in raw  # 补过的那份比这份长


async def test_budget_ceiling_blocks_and_leaves_a_finding(client, db) -> None:
    """熔断必须发生在放行之前，且必须留痕。

    只在事后统计的话，超限的那一次已经花出去了——那就只是个仪表盘，不是熔断。
    """
    await _register(client, token_ceiling=50)
    # 每次预留 4096+ 输出，50 的池子第一次就装不下
    resp = await client.post("/v1/chat/completions", json=_chat_body())

    assert resp.status_code == 429
    assert resp.json()["error"]["type"] == "budget_exceeded"

    findings = (await db.execute(select(DetectorFinding))).scalars().all()
    assert len(findings) == 1
    assert findings[0].detector == "token_budget"
    assert findings[0].verdict == "fail"

    # 被拒绝的请求不该产生 LLM 调用记录
    assert (await db.execute(select(LlmCall))).scalars().all() == []


async def test_budget_is_settled_with_actual_usage_not_the_estimate(client) -> None:
    """预扣用估算，结算用真实值。只预留不结算的话预算会慢性泄漏。"""
    created = await _register(client, token_ceiling=100_000)
    scripted_upstream.register(created["run_id"], [{"content": "hi"}])

    resp = await client.post("/v1/chat/completions", json=_chat_body())
    assert resp.status_code == 200

    used = int(resp.headers["X-Sandbox-Tokens-Used"])
    est_input = len(str(_chat_body())) // 3
    assert used < est_input + 4096, "结算值应远小于预扣的估算值"
    assert int(resp.headers["X-Sandbox-Tokens-Remaining"]) == 100_000 - used


async def test_reservation_is_released_when_upstream_fails(client) -> None:
    """上游失败必须归还预扣，否则反复失败会把预算耗光而一次调用都没发生。"""
    await _register(client, token_ceiling=10_000)
    # 未注册脚本 → stub 抛 KeyError
    resp = await client.post("/v1/chat/completions", json=_chat_body())
    assert resp.status_code == 400

    snapshot = (await client.get("/_sandbox/runs")).json()["runs"][0]
    assert snapshot["used"] == 0
    assert snapshot["remaining"] == 10_000, "预扣未归还，预算泄漏了"


async def test_real_mode_without_key_refuses_instead_of_falling_back(client, monkeypatch) -> None:
    """real 模式缺密钥必须报错，绝不能静默退回 stub。

    静默降级最坏的地方是：结果看起来完全正常，但测的是假模型。
    """
    monkeypatch.setattr(settings, "upstream_mode", "real")
    monkeypatch.setattr(settings, "upstream_api_key", None)

    created = await _register(client)
    resp = await client.post("/v1/chat/completions", json=_chat_body())

    assert resp.status_code == 500
    assert "拒绝静默降级" in resp.json()["error"]["message"]
    assert created["run_id"]


# --- 真实上游那一条路 ---------------------------------------------------------
#
# 上面那条之外，测试里的 `upstream_mode` 一直是 stub，所以真正做转发的
# `_call_real_upstream` 平时一条断言都没有——而它是**唯一会把真实密钥拼进
# 请求**的地方。下面两条盯的是上游回来的东西形状不对时会怎样：边界上的输入
# 不受我们控制，而这条路要等配了真钥匙才有人走，那时候再发现就晚了。


def _fake_reply() -> dict[str, Any]:
    """一份像上游会回的对象。形状照着 OpenAI 的来，不是自己编的。"""
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "model": "fake-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "上游原样回来的那句话"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


class _FakeResponse:
    status_code = 200
    text = "（假的响应体，这两条测试都不看它）"

    def __init__(self, body: Any) -> None:
        self._body = body

    def json(self) -> Any:
        return self._body


class _FakeClient:
    def __init__(self, http: _FakeHttp) -> None:
        self._http = http

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self._http.sent.append({"url": url, **kwargs})
        return _FakeResponse(self._http.body)


class _FakeHttp:
    """替掉网关模块里的 `httpx`。

    **只实现 `_call_real_upstream` 真正用到的那两个方法**（`AsyncClient` 和
    `post`）。多出来的部分没人验，会自己长出和真 httpx 不一样的脾气，而那种
    差异只在出事那天才显形。
    """

    def __init__(self, body: Any) -> None:
        self.body = body
        self.sent: list[dict[str, Any]] = []

    def AsyncClient(self, **_kwargs: Any) -> _FakeClient:
        return _FakeClient(self)


def _fake_upstream(monkeypatch: pytest.MonkeyPatch, body: Any) -> _FakeHttp:
    fake = _FakeHttp(body)
    monkeypatch.setattr(gateway_app, "httpx", fake)
    return fake


async def test_real_mode_passes_an_object_reply_through(client, monkeypatch) -> None:
    """对照：上游回一个正常的对象时，这一趟必须成功。

    没有这一条的话，下面那条「顶层不是对象要报错」可能只是因为**这条路整个
    走不通**才红——一个永远报错的函数能通过所有「它报错了」的断言。
    """
    monkeypatch.setattr(settings, "upstream_mode", "real")
    monkeypatch.setattr(settings, "upstream_api_key", "test-key-not-real")
    fake = _fake_upstream(monkeypatch, _fake_reply())

    await _register(client)
    resp = await client.post("/v1/chat/completions", json=_chat_body())

    assert resp.status_code == 200, resp.text
    assert resp.json()["choices"][0]["message"]["content"] == "上游原样回来的那句话"
    assert len(fake.sent) == 1
    assert fake.sent[0]["url"].endswith("/chat/completions")


async def test_real_mode_with_a_non_object_reply_reports_an_upstream_error(
    client, monkeypatch
) -> None:
    """上游回的 JSON 顶层不是对象：报 502，而且那句话指向上游。

    不核这一下的后果很具体：结算那一步会拿它去 `response.get("usage")`，
    炸成「list 没有 get」——那是个 500，一个字都没提到上游。读日志的人会去
    查网关，而问题在上游。状态码也必须是 502 而不是照着上游的 200 抄：
    抄了的话，客户端收到的是一个"成功但解析不了"的 200。
    """
    monkeypatch.setattr(settings, "upstream_mode", "real")
    monkeypatch.setattr(settings, "upstream_api_key", "test-key-not-real")
    _fake_upstream(monkeypatch, [{"order_id": "A1"}])

    await _register(client)
    resp = await client.post("/v1/chat/completions", json=_chat_body())

    assert resp.status_code == 502, resp.text
    error = resp.json()["error"]
    assert error["type"] == "upstream_error"
    assert "顶层不是对象" in error["message"]
    # 这一趟的预扣要还回去。新加的这条岔路也在那个 `except` 下面，但"也在
    # 下面"是读出来的，这条断言是量出来的：少了它，预算会随着每次形状不
    # 对的上游响应慢性泄漏。
    snapshot = (await client.get("/_sandbox/runs")).json()["runs"][0]
    assert snapshot["used"] == 0, "预扣未归还，预算泄漏了"


# 流式那一族在 `tests/test_gateway_stream.py`。这里原本只有一条断言——501
# 明确拒绝，理由是"结算正确性上另有一套逻辑，不走通不发布"。那套逻辑做出来了，
# 断言也就搬到了它该在的地方。


async def test_stub_returns_openai_compatible_shape(client) -> None:
    """Agent 侧用的是标准 OpenAI 客户端，形状不一致会直接解析失败。"""
    created = await _register(client)
    scripted_upstream.register(
        created["run_id"],
        [{"tool_calls": [{"name": "create_payment", "arguments": {"order_id": "A1"}}]}],
    )

    resp = await client.post("/v1/chat/completions", json=_chat_body())
    body = resp.json()

    assert body["object"] == "chat.completion"
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "create_payment"
    # OpenAI 的 arguments 是 JSON 字符串而不是对象，这点错了客户端会炸
    assert isinstance(call["function"]["arguments"], str)


# --- 写工具的声明：空表和"没给"是两件事 ----------------------------------------


async def test_an_empty_declaration_is_kept_as_an_empty_declaration(client) -> None:
    """**空表说的是"这个场景没有写操作"，不是"我不知道"。**

    折叠成"没给"之后，场景作者的一句声明被换成了一次按工具名的猜测——而写判定
    直接决定幂等/重复扣款检测的触发范围，猜反的代价是漏报一整类缺陷。
    """
    created = await _register(client, declared_writes=[])
    assert runs.get(created["run_id"]).ctx.declared_writes == set()


async def test_no_declaration_at_all_stays_unknown(client) -> None:
    """反面：字段不在（或为 null）= 真的不知道，这才轮到启发式兜底。"""
    created = await _register(client, declared_writes=None)
    assert runs.get(created["run_id"]).ctx.declared_writes is None

    # 字段干脆不传，同一种意思
    resp = await client.post(
        "/_sandbox/runs", json={"scenario_id": "x", "agent_version": "v"}
    )
    assert runs.get(resp.json()["run_id"]).ctx.declared_writes is None


@pytest.mark.parametrize(
    "bad",
    [
        "create_payment",  # 字符串：set() 会把它拆成一堆单字符，等于一条都没声明
        {"create_payment": True},  # 字典
        [1, 2],  # 元素不是字符串
        [None],
    ],
)
async def test_a_malformed_declaration_is_refused_not_silently_dropped(
    client, bad
) -> None:
    """写错的声明**必须在开跑前退回**，不能静默降级成"没声明"。

    静默降级的形态最坏：整张声明表看起来生效了、实际从没匹配上，报告上什么都不显示。
    """
    resp = await client.post(
        "/_sandbox/runs",
        json={"scenario_id": "x", "agent_version": "v", "declared_writes": bad},
    )

    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "bad_declared_writes"
    assert runs.active_ids() == [], "退回的运行不许留在注册表里"


async def test_a_scenario_that_declares_no_tools_does_not_inherit_the_builtin_table(
    client,
) -> None:
    """同一族的另一处：**空工具表也不能回落到内置那张。**

    一个声明了"这个场景没有工具"的场景，如果拿到内置的 mock 工具表，Agent 调一个
    它压根没被声明过的工具会被**转发出去**——本该报 `tool_scope` 失败并在边界上
    拦下，实际是钱真的动了，而报告看起来正常。

    没给这个字段（手工 curl 那条路）仍然回落，那是这张表存在的理由。
    """
    resp = await client.post(
        "/_sandbox/runs",
        json={"scenario_id": "x", "agent_version": "v", "tools": {}, "tool_schemas": {}},
    )
    active = runs.get(resp.json()["run_id"])

    assert active.tools == {}, "空工具表被换成了内置那张——Agent 能调到没声明的工具"

    # 反面：字段不在，才轮到内置兜底
    resp = await client.post(
        "/_sandbox/runs", json={"scenario_id": "x", "agent_version": "v"}
    )
    assert "create_payment" in runs.get(resp.json()["run_id"]).tools, "手工路径的兜底没了"
