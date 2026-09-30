"""网关是 ADR 001 的承重墙：路由、观测、止损、凭据隔离，四项都要有断言。"""

from __future__ import annotations

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


async def test_streaming_is_explicitly_not_implemented(client) -> None:
    """明确拒绝优于静默走非流式——后者会让错误在 Agent 侧以别的方式爆出来。"""
    await _register(client)
    resp = await client.post("/v1/chat/completions", json=_chat_body(stream=True))
    assert resp.status_code == 501
    assert resp.json()["error"]["type"] == "not_implemented"


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
