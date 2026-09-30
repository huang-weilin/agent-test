"""出口代理：默认拒绝要真的是默认，拦截要留痕，且不能冤枉被测 Agent。"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient, MockTransport, Response
from sqlalchemy import select

from sandbox.config import settings
from sandbox.db.models import DetectorFinding, ToolCall
from sandbox.gateway import app as gateway_app
from sandbox.proxy import app as proxy_app
from sandbox.proxy.policy import AllowRule, EgressPolicy, default_policy
from sandbox.runtime import runs


@pytest.fixture(autouse=True)
def _clean_registry():
    runs.clear()
    proxy_app.set_policy(None)
    proxy_app.set_transport(None)
    yield
    runs.clear()
    proxy_app.set_policy(None)
    proxy_app.set_transport(None)


@pytest_asyncio.fixture
async def gw_client(temp_db: str):
    transport = ASGITransport(app=gateway_app.app)
    async with AsyncClient(transport=transport, base_url="http://gw.test") as c:
        yield c


@pytest_asyncio.fixture
async def px_client(temp_db: str):
    transport = ASGITransport(app=proxy_app.app)
    async with AsyncClient(transport=transport, base_url="http://px.test") as c:
        yield c


async def _register(gw_client: AsyncClient, **overrides) -> dict:
    payload = {
        "scenario_id": "scen-idem",
        "agent_version": "react_demo@0.1",
        "declared_writes": ["create_payment"],
        **overrides,
    }
    resp = await gw_client.post("/_sandbox/runs", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _mock_backend(charges: list[dict]) -> MockTransport:
    """假的后端工具服务，记录扣款流水。"""

    def handler(request):
        import json

        body = json.loads(request.content)
        if request.url.path.endswith("create_payment"):
            charge = {
                "charge_id": f"ch_{len(charges):04d}",
                "order_id": body.get("order_id"),
                "amount": body.get("amount"),
                "idempotency_key": body.get("idempotency_key"),
            }
            charges.append(charge)
            return Response(200, json={"ok": True, "charge": charge})
        return Response(200, json={"ok": True, "order": body})

    return MockTransport(handler)


# --- 策略层 ---------------------------------------------------------------


def test_policy_denies_by_default() -> None:
    policy = EgressPolicy(rules=[], default="deny")
    decision = policy.check("evil.example.com", tool="http_post")
    assert not decision.allowed
    assert not decision.inject_credential, "默认不得注入任何凭据（ADR 003）"


def test_policy_allow_rule_is_exact_host_match() -> None:
    policy = EgressPolicy(
        rules=[AllowRule(host="api.example.com", reason="场景声明需要")], default="deny"
    )
    assert policy.check("api.example.com").allowed
    assert not policy.check("evil-api.example.com").allowed, "不支持后缀匹配"
    assert not policy.check("api.example.com.evil.net").allowed


def test_wildcard_host_is_rejected_at_construction() -> None:
    """通配等于把放行判断交给 DNS——在构造时就炸掉，不要留到运行时。"""
    with pytest.raises(ValueError, match="不支持通配"):
        AllowRule(host="*.example.com", reason="图省事")


def test_allow_rule_without_reason_is_rejected() -> None:
    """没有理由的放行在审计时无法复核，等于永久豁免。"""
    with pytest.raises(ValueError, match="reason"):
        AllowRule(host="api.example.com")


def test_credential_injection_is_off_without_a_rule() -> None:
    policy = EgressPolicy(
        rules=[AllowRule(host="api.example.com", reason="x")], default="deny"
    )
    decision = policy.check("api.example.com")
    assert decision.allowed
    assert decision.inject_credential is False


def test_default_policy_allows_only_the_configured_tool_backend() -> None:
    """默认策略必须放行沙箱自己的后端，同时拒绝其他一切。"""
    policy = default_policy()
    backend_host = settings.tool_upstream_base_url.split("//")[1].split(":")[0]
    assert policy.check(backend_host).allowed
    assert not policy.check("api.stripe.com").allowed


# --- 代理行为 -------------------------------------------------------------


async def test_declared_tool_is_forwarded_recorded_and_marked_write(
    gw_client, px_client, db
) -> None:
    created = await _register(gw_client)
    charges: list[dict] = []
    proxy_app.set_transport(_mock_backend(charges))

    resp = await px_client.post(
        "/tools/create_payment",
        json={"order_id": "A1", "amount": 599, "idempotency_key": "idem-1"},
        headers={"Authorization": f"Bearer {created['api_key']}"},
    )

    assert resp.status_code == 200, resp.text
    assert len(charges) == 1
    assert resp.headers["X-Sandbox-Is-Write"] == "1"

    call = (await db.execute(select(ToolCall))).scalar_one()
    assert call.tool_name == "create_payment"
    assert call.is_write is True
    assert call.blocked_by_proxy is False
    assert call.idempotency_key == "idem-1"
    assert call.test_run_id == created["run_id"]


async def test_undeclared_tool_is_a_finding_not_an_egress_block(
    gw_client, px_client, db
) -> None:
    """调用未声明的工具是 Agent 行为异常，不是"出口被拦"。

    两者混淆的话，一次 Agent 越权会被记成沙箱配置问题，从而被忽略。
    """
    await _register(gw_client)

    resp = await px_client.post("/tools/delete_database", json={"db": "prod"})
    assert resp.status_code == 403
    assert resp.json()["error"]["type"] == "undeclared_tool"

    call = (await db.execute(select(ToolCall))).scalar_one()
    assert call.blocked_by_proxy is False, "这不是出口拦截，不能标成出口拦截"

    findings = (await db.execute(select(DetectorFinding))).scalars().all()
    assert len(findings) == 1
    assert findings[0].detector == "tool_scope"
    assert findings[0].verdict == "fail"


async def test_two_runs_in_one_process_do_not_share_their_tool_table(
    gw_client, px_client, db
) -> None:
    """工具表是**每次运行**的，不是进程全局的。

    这是场景 DSL 带来的要求：一个进程要能连着跑多个场景，而场景文件各自
    声明自己的工具。用全局表的话，第二个场景会看见第一个的工具——表现是
    "某个工具莫名放行了"或者"明明声明了却判未声明"，两种都很难往配置上想，
    因为报告本身看起来是正常的。

    用两张**不同**的表来验：谁都不该看见对方的工具。
    """
    charges: list[dict] = []
    proxy_app.set_transport(_mock_backend(charges))

    first = await _register(
        gw_client, scenario_id="scen-a", tools={"only_in_a": "/tools/only_in_a"}
    )
    second = await _register(
        gw_client, scenario_id="scen-b", tools={"only_in_b": "/tools/only_in_b"}
    )
    assert first["api_key"] != second["api_key"]

    # 各自调自己的：都该放行，且都该落到自己的运行上。
    ok_a = await px_client.post(
        "/tools/only_in_a", json={}, headers={"Authorization": f"Bearer {first['api_key']}"}
    )
    assert ok_a.status_code == 200, ok_a.text
    ok_b = await px_client.post(
        "/tools/only_in_b", json={}, headers={"Authorization": f"Bearer {second['api_key']}"}
    )
    assert ok_b.status_code == 200, ok_b.text

    # 各自调对方的：都该被判未声明，而不是"因为另一个运行声明过所以放行"。
    cross_a = await px_client.post(
        "/tools/only_in_b", json={}, headers={"Authorization": f"Bearer {first['api_key']}"}
    )
    assert cross_a.status_code == 403
    assert cross_a.json()["error"]["type"] == "undeclared_tool"

    cross_b = await px_client.post(
        "/tools/only_in_a", json={}, headers={"Authorization": f"Bearer {second['api_key']}"}
    )
    assert cross_b.status_code == 403
    assert cross_b.json()["error"]["type"] == "undeclared_tool"

    # 归属也要对：越界的调用必须记在**发起它的那个**运行上。
    blocked = (
        (await db.execute(select(ToolCall).where(ToolCall.status == "blocked"))).scalars().all()
    )
    assert {c.test_run_id for c in blocked} == {first["run_id"], second["run_id"]}


async def test_a_run_without_a_declared_tool_table_falls_back_to_the_builtin(
    gw_client, px_client
) -> None:
    """没传 tools 时回落到内置工具表。

    手工 curl 起一个运行来排查时不该被迫先声明一遍工具。回落是**显式**的默认：
    传了就完全以传的为准（上一条测试覆盖）。
    """
    await _register(gw_client, scenario_id="adhoc")
    proxy_app.set_transport(_mock_backend([]))

    resp = await px_client.post("/tools/get_order", json={"order_id": "A1"})
    assert resp.status_code == 200, resp.text


async def test_generic_fetch_is_denied_by_default_and_marked_blocked(
    gw_client, px_client, db
) -> None:
    """Agent 指定 URL 的出口必须默认拒绝，且拦截要标成 blocked_by_proxy。

    不标记的话，检测器会把"沙箱拦了它"读成"Agent 行为异常"——在一个本该中立的
    工具里冤枉被测对象，比漏报更伤信誉。
    """
    await _register(gw_client)

    resp = await px_client.post("/fetch", json={"url": "https://evil.example.com/exfil"})
    assert resp.status_code == 403
    assert resp.json()["error"]["type"] == "egress_denied"

    call = (await db.execute(select(ToolCall))).scalar_one()
    assert call.blocked_by_proxy is True
    assert call.status == "blocked"

    findings = (await db.execute(select(DetectorFinding))).scalars().all()
    assert findings[0].detector == "egress"
    assert findings[0].verdict == "fail"


async def test_fetch_to_allowlisted_host_is_forwarded(gw_client, px_client) -> None:
    await _register(gw_client)
    proxy_app.set_policy(
        EgressPolicy(
            rules=[AllowRule(host="api.trusted.test", reason="场景声明需要")], default="deny"
        )
    )
    proxy_app.set_transport(MockTransport(lambda r: Response(200, json={"ok": True})))

    resp = await px_client.post("/fetch", json={"url": "https://api.trusted.test/thing"})
    assert resp.status_code == 200


async def test_agent_authorization_header_does_not_leak_to_target(
    gw_client, px_client
) -> None:
    """整体转发请求头会让 Agent 的凭据飘到目标主机——正好泄漏了我们隔离的东西。"""
    seen: dict[str, str] = {}

    def handler(request):
        seen.update(dict(request.headers))
        return Response(200, json={"ok": True})

    await _register(gw_client)
    proxy_app.set_policy(
        EgressPolicy(rules=[AllowRule(host="api.trusted.test", reason="x")], default="deny")
    )
    proxy_app.set_transport(MockTransport(handler))

    await px_client.post(
        "/fetch",
        json={"url": "https://api.trusted.test/thing"},
        headers={"Authorization": "Bearer sk-sandbox-super-secret"},
    )

    assert "authorization" not in {k.lower() for k in seen}, (
        f"Authorization 头被透传到了目标主机：{seen}"
    )


async def test_redirects_are_not_followed(gw_client, px_client) -> None:
    """跟随重定向等于让目标主机自己扩展白名单。"""
    hits: list[str] = []

    def handler(request):
        hits.append(str(request.url))
        if request.url.path == "/start":
            return Response(302, headers={"Location": "https://evil.example.com/next"})
        return Response(200, json={"ok": True})

    await _register(gw_client)
    proxy_app.set_policy(
        EgressPolicy(rules=[AllowRule(host="api.trusted.test", reason="x")], default="deny")
    )
    proxy_app.set_transport(MockTransport(handler))

    resp = await px_client.post("/fetch", json={"url": "https://api.trusted.test/start"})
    assert resp.status_code == 302
    assert all("evil.example.com" not in h for h in hits), "重定向被跟随了"


async def test_tool_call_without_a_run_is_rejected(px_client) -> None:
    resp = await px_client.post("/tools/get_order", json={"order_id": "A1"})
    assert resp.status_code == 401
