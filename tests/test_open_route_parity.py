"""通用出口与定目标那条路：**同一套规矩**。

代理有两条出口路由（`proxy/app.py` 的模块注释）：`/tools/{name}`（目标由沙箱配置
定）和 `/fetch`、`/fetch/{name}`（目标由被测对象指定）。判定口径、写标记、参数核对
都已经在两处分别对齐过——而**运行级的两个保护**当初只接在前一条上：

1. **看门狗**：`watchdog.enforce` 全仓原先只有一个调用点，在 `call_tool` 末尾。
2. **熔断之后不许再有副作用**：`active.killed` 那个 410 原先只在 `call_tool` 开头。

两条漏掉的后果不一样，但形状一样——**报告上看不出来**。第一件：事后那轮检测照样
在整条会话序列上算出循环、结论一个字不少，少的只是"就地终止"；第二件：被熔断的
运行从此还能通过通用出口继续往目标系统上打，而熔断的全部意义就是止损。

这一组都是**受控对照**：同一份夹具、同样的调用、同一台主机、同一个检测器，只差
走哪条路。修之前，定目标那几条绿、通用出口那几条红。
"""

from __future__ import annotations

from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient, MockTransport, Response

from sandbox.gateway import app as gateway_app
from sandbox.proxy import app as proxy_app
from sandbox.proxy.policy import AllowRule, policy_with_rules
from sandbox.runtime import runs

_HTTP_POST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "url": {"type": "string"},
        "body": {"type": "object", "additionalProperties": True},
    },
    "required": ["url"],
}
_CREATE_PAYMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
    "required": ["order_id", "amount"],
}

#: 通用出口那次调用要**逐字相同**：`loop.strict` 的判据是严格重复，
#: `strict_min_repeats = 3`。值里不带任何随机会变的东西。
_ENVELOPE = {
    "url": "https://pay.example.com/v1/charges",
    "body": {"order_id": "A1", "amount": 599.0},
}
#: 定目标那条路上的同一次调用。两条路打的是同一个业务动作，参数形状不同是
#: 因为入口协议不同（见 `ToolDecl._an_open_target_declares_the_route_protocol`）。
_PAYLOAD = {"order_id": "A1", "amount": 599.0}

_REPEATS = 3


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


async def _register(gw_client: AsyncClient) -> dict:
    payload = {
        "scenario_id": "scen-open-route-parity",
        "agent_version": "react_demo@0.1",
        "declared_writes": ["http_post", "create_payment"],
        "tools": {
            "http_post": "/fetch/http_post",
            "create_payment": "/tools/create_payment",
        },
        "tool_schemas": {
            "http_post": _HTTP_POST_SCHEMA,
            "create_payment": _CREATE_PAYMENT_SCHEMA,
        },
    }
    resp = await gw_client.post("/_sandbox/runs", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _allow_and_spy(seen: list[Any] | None = None) -> None:
    """放行支付网关，并把转发截在一张假 transport 上（不出网）。

    传了 `seen` 就记下每一次真的转发——"不许到达目标系统"这种断言必须看这里，
    不能只看响应码。
    """
    proxy_app.set_policy(
        policy_with_rules(
            [AllowRule(host="pay.example.com", reason="场景声明", mock_path=None)],
            tool_hosts={},
        )
    )

    def handler(request: Any) -> Response:
        if seen is not None:
            seen.append(request)
        return Response(200, json={"ok": True})

    proxy_app.set_transport(MockTransport(handler))


async def _post_times(
    px_client: AsyncClient, path: str, payload: dict, key: str, times: int
) -> None:
    for _ in range(times):
        resp = await px_client.post(
            path, json=payload, headers={"Authorization": f"Bearer {key}"}
        )
        assert resp.status_code == 200, resp.text


def _kill_reason(run_id: str) -> str:
    active = runs.get(run_id)
    assert active is not None and active.killed is True, (
        "三次逐字相同的调用之后这个运行还活着——看门狗没扫到这条路上"
    )
    return active.kill_reason


# --- 第一件：看门狗两条路都要扫 ---------------------------------------------------


@pytest.mark.asyncio
async def test_the_fixed_route_kills_a_loop(gw_client, px_client, db) -> None:
    """对照组（修之前就是绿的，它是下面两条的标尺）。"""
    created = await _register(gw_client)
    _allow_and_spy()

    await _post_times(px_client, "/tools/create_payment", _PAYLOAD, created["api_key"], _REPEATS)

    assert _kill_reason(created["run_id"]).startswith("loop.strict")


@pytest.mark.asyncio
async def test_the_named_open_route_kills_a_loop(gw_client, px_client, db) -> None:
    """/fetch/{名字}：目标由调用方指定的那类工具。

    "出口范围写不进配置"的那类工具（通用 HTTP 工具、抓取类 MCP）恰恰是最容易
    打转的一类——它的目标由调用方给定，一次循环就是往同一台主机重复发同一份
    请求。看门狗接不到这条路上，"检测到循环就地终止"就不是一条运行时的保证。
    """
    created = await _register(gw_client)
    _allow_and_spy()

    await _post_times(px_client, "/fetch/http_post", _ENVELOPE, created["api_key"], _REPEATS)

    assert _kill_reason(created["run_id"]).startswith("loop.strict")


@pytest.mark.asyncio
async def test_the_anonymous_open_route_kills_a_loop(gw_client, px_client, db) -> None:
    """`/fetch`：连名字都没报的那条路。

    与前一条共用 `_generic_egress` 这一个实现，差别只在记在谁名下——但它是**独立
    的一个端点**，而"两条入口各自接一遍"正是原先漏掉的那件事的形状。
    """
    created = await _register(gw_client)
    _allow_and_spy()

    await _post_times(px_client, "/fetch", _ENVELOPE, created["api_key"], _REPEATS)

    assert _kill_reason(created["run_id"]).startswith("loop.strict")


# --- 第二件：熔断之后两条路都不许再有副作用 -----------------------------------------


@pytest.mark.asyncio
async def test_the_fixed_route_refuses_a_killed_run(gw_client, px_client, db) -> None:
    """对照组：熔断之后定目标那条路回 410，而且请求一次都没转发出去。"""
    created = await _register(gw_client)
    seen: list[Any] = []
    _allow_and_spy(seen)
    runs.kill(created["run_id"], "loop.strict: 测试里手动熔断")

    resp = await px_client.post(
        "/tools/create_payment",
        json=_PAYLOAD,
        headers={"Authorization": f"Bearer {created['api_key']}"},
    )

    assert resp.status_code == 410
    assert resp.json()["error"]["type"] == "run_killed"
    assert seen == [], "熔断之后这次调用还是发出去了"


@pytest.mark.asyncio
async def test_the_open_route_refuses_a_killed_run(gw_client, px_client, db) -> None:
    """通用出口：同样回 410，同样一次都不许转发。

    熔断的全部意义是止损，而这条路上漏掉那道判据时，**报告上什么都不会显示**：
    结论早就落了库，运行状态也写着"已熔断"，只有目标系统上的副作用还在继续。
    """
    created = await _register(gw_client)
    seen: list[Any] = []
    _allow_and_spy(seen)
    runs.kill(created["run_id"], "loop.strict: 测试里手动熔断")

    resp = await px_client.post(
        "/fetch/http_post",
        json=_ENVELOPE,
        headers={"Authorization": f"Bearer {created['api_key']}"},
    )

    assert resp.status_code == 410
    assert resp.json()["error"]["type"] == "run_killed"
    assert seen == [], "熔断之后这次调用还是发出去了"
