"""看门狗自己的时间记在谁的名下（「谁的错分开写」）。

看门狗是在出口代理的请求处理里 `await` 的，而 Agent 正阻塞在那个请求上等它扫完
——**这笔时间是我们的开销，记在被测会话的墙钟上**。而它原先在轨迹里看不见：
`tool_call.latency_ms` 在 `enforce` 之前就量完了，于是一次因为看门狗太慢而超时的
运行，报告里长得和"Agent 自己慢"一模一样。这一组钉住四件事：

1. 真的出到目标系统的调用，那笔耗时**记在它自己那一行上**（不是运行级的一个总数：
   代价随调用次数超线性涨，只有落在被叫的那一次上才分得开谁的时间是谁的）；
2. 记下来的**确实是 `enforce` 的耗时**，不是"请求总共花了多久"——所以这一组里
   有一对受控对照，分别把目标系统和看门狗做大，看它落在哪一列；
3. **被边界拦下的调用留 NULL，不是 0**。"没量到"与"量到 0"是两件事，把前者读成
   后者，等于替一次没发生过的测量报一个数；
4. 报告与看板**真的把它读出来了**。只钉"写进库了"是不够的——写在库里没人读，
   和没写一样，而那正是这一片要修的那种缺陷的形状。

第 4 条同时钉住了两处**完全不同**的渲染（运行详情页的审计表、指标看板的延迟表），
它们各自算各自的：一处漏了，另一处不会红。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient, MockTransport, Response

from sandbox import metrics
from sandbox.api import queries
from sandbox.gateway import app as gateway_app
from sandbox.proxy import app as proxy_app
from sandbox.proxy.policy import AllowRule, policy_with_rules
from sandbox.runtime import runs

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
    "required": ["order_id", "amount"],
}
#: 通用出口的声明形状是加载期钉死的那两个字段（见 `ToolDecl` 那条校验），
#: 和定目标那条路不是一份——这也是为什么两条路各自要接一遍记账。
_ENVELOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "url": {"type": "string"},
        "body": {"type": "object", "additionalProperties": True},
    },
    "required": ["url"],
}
_PAYLOAD = {"order_id": "A1", "amount": 599.0}
_ENVELOPE = {
    "url": "https://pay.example.com/v1/charges",
    "body": {"order_id": "A1", "amount": 599.0},
}

#: 受控对照里"做大"的那一笔。取得足够大，好让两列的数字拉开档次：看门狗在一两条
#: 调用的轨迹上是 10~25 ms 的量级（`watchdog.py` 那张表里"落一条结论"是常数 10 ms，
#: 一次扫描落两条），所以 150 ms 的睡不会和它撞上。
_SLOW = 0.15
_SLOW_MS = int(_SLOW * 1000)
#: 判定用的门槛比睡的时间低一档：`perf_counter` 量的是墙钟，机器忙起来会多，
#: 但不会少。少到门槛以下就说明量的不是这笔睡。
_FLOOR = _SLOW_MS - 30


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
    resp = await gw_client.post(
        "/_sandbox/runs",
        json={
            "scenario_id": "scen-watchdog-accounting",
            "agent_version": "react_demo@0.1",
            "declared_writes": ["http_post", "create_payment"],
            "tools": {
                "http_post": "/fetch/http_post",
                "create_payment": "/tools/create_payment",
            },
            "tool_schemas": {"http_post": _ENVELOPE_SCHEMA, "create_payment": _SCHEMA},
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _allow_host(*, target_sleep: float = 0.0) -> None:
    """放行支付网关，转发截在一张假 transport 上（不出网）。

    `target_sleep` 是给受控对照用的：**目标系统**慢，与看门狗慢是两笔时间，
    而报告必须把它们分到两列去。
    """
    proxy_app.set_policy(
        policy_with_rules(
            [AllowRule(host="pay.example.com", reason="场景声明", mock_path=None)],
            tool_hosts={},
        )
    )

    def handler(request: Any) -> Response:
        if target_sleep:
            time.sleep(target_sleep)
        return Response(200, json={"ok": True})

    proxy_app.set_transport(MockTransport(handler))


async def _post(px_client: AsyncClient, path: str, payload: dict, key: str) -> Any:
    return await px_client.post(
        path, json=payload, headers={"Authorization": f"Bearer {key}"}
    )


async def _rows(run_id: str) -> list[queries.ToolCallRow]:
    return await queries.tool_calls_for_run(run_id)


# --- 一：真的出到目标系统的调用，那一行上有耗时 ---------------------------------------


@pytest.mark.asyncio
async def test_a_call_that_reached_the_target_records_the_scan_cost(
    gw_client, px_client, db
) -> None:
    created = await _register(gw_client)
    _allow_host()

    resp = await _post(px_client, "/tools/create_payment", _PAYLOAD, created["api_key"])
    assert resp.status_code == 200, resp.text

    rows = await _rows(created["run_id"])
    assert len(rows) == 1
    assert rows[0].watchdog_ms is not None, (
        "这次调用真的出到了目标系统，看门狗跑过它——那一行上该有我们花掉的时间。"
        "留在 NULL 说明这一次的耗时没被记下来（或者写入方被摘掉了）"
    )


@pytest.mark.asyncio
async def test_the_open_route_records_it_too(gw_client, px_client, db) -> None:
    """两条出口路由各接一遍——上一片刚因为"只接了一条"出过事。

    `watchdog.enforce` 本身在两条路上都跑（`tests/test_open_route_parity.py`
    钉着），但**记账是另一处调用**，接漏了不会有人报错：报告照常出，只是那一列
    永远是 NULL。
    """
    created = await _register(gw_client)
    _allow_host()

    resp = await _post(px_client, "/fetch/http_post", _ENVELOPE, created["api_key"])
    assert resp.status_code == 200, resp.text

    rows = await _rows(created["run_id"])
    assert rows[0].watchdog_ms is not None


@pytest.mark.asyncio
async def test_a_failed_writeback_does_not_turn_a_done_call_into_a_failure(
    gw_client, px_client, db, monkeypatch
) -> None:
    """**记账失败不许改变这次调用的结果。**

    这句话不是修辞：写这一笔的时候，这次调用已经真的发生在目标系统上了、也已经
    记成 `ok` 了。让它变成 500 的话，Agent 多半会重试——于是真的多扣一次款，而
    起因只是我们记不上一个既不判通过、也不判熔断的列。

    所以记账那一句外面有兜底，日志里留一行 `proxy.watchdog_cost_lost`。
    """

    async def boom(call_id: str, watchdog_ms: int) -> None:
        raise RuntimeError("库写不进去")

    monkeypatch.setattr(proxy_app, "record_watchdog_cost", boom)

    created = await _register(gw_client)
    _allow_host()

    resp = await _post(px_client, "/tools/create_payment", _PAYLOAD, created["api_key"])
    assert resp.status_code == 200, "记账写不进去，却让一次已经发生的调用对 Agent 报失败"
    # 那一格留 NULL——**不是 0**。记不上就是没量到，不能替它编一个数。
    assert (await _rows(created["run_id"]))[0].watchdog_ms is None


# --- 二：这一对受控对照说明两列量的确实是两件事 ---------------------------------------


@pytest.mark.asyncio
async def test_a_slow_target_lands_in_latency_not_in_watchdog(
    gw_client, px_client, db
) -> None:
    """目标系统慢：那是 Agent 等目标系统的时间，**不是**我们的开销。"""
    created = await _register(gw_client)
    _allow_host(target_sleep=_SLOW)

    resp = await _post(px_client, "/tools/create_payment", _PAYLOAD, created["api_key"])
    assert resp.status_code == 200, resp.text

    row = (await _rows(created["run_id"]))[0]
    assert row.latency_ms >= _FLOOR, "目标系统睡了 150 ms，这段往返没被量进 latency_ms"
    assert row.watchdog_ms is not None and row.watchdog_ms < _FLOOR, (
        f"看门狗这一列记的是 {row.watchdog_ms} ms，而这次慢的是目标系统——"
        "两笔时间被记进了同一列"
    )


@pytest.mark.asyncio
async def test_a_slow_watchdog_lands_in_watchdog_not_in_latency(
    gw_client, px_client, db, monkeypatch
) -> None:
    """把 `enforce` 本身拖慢：这笔时间**必须**落在看门狗那一列上。

    对照组是它的反面。只测一条的话，"两列记的是同一笔时间"也能全绿——而那正是
    要修的缺陷：`latency_ms` 在 `enforce` **之前**就量完了，两个数一样大就说明
    有一列没在量它该量的东西。
    """

    async def slow_enforce(run_id: str, detector: Any = None) -> str | None:
        await asyncio.sleep(_SLOW)
        return None

    # 打在模块属性上，和 `proxy/app.py` 里 `watchdog.enforce(...)` 解析到的是同一个
    # 对象——这条路径才是生产路径。打在自己模块里的一个副本上测的是假路径。
    monkeypatch.setattr(proxy_app.watchdog, "enforce", slow_enforce)

    created = await _register(gw_client)
    _allow_host()

    resp = await _post(px_client, "/tools/create_payment", _PAYLOAD, created["api_key"])
    assert resp.status_code == 200, resp.text

    row = (await _rows(created["run_id"]))[0]
    assert row.watchdog_ms is not None and row.watchdog_ms >= _FLOOR, (
        f"`enforce` 睡了 150 ms，这一列只记到 {row.watchdog_ms} ms——"
        "它量的不是看门狗的耗时"
    )
    assert row.latency_ms < _FLOOR, (
        f"这次慢的是看门狗，`latency_ms` 却也记到了 {row.latency_ms} ms——"
        "两笔时间混在一起了"
    )


# --- 三：被拦下的调用留 NULL，不是 0 --------------------------------------------------


@pytest.mark.asyncio
async def test_an_undeclared_tool_leaves_it_null_not_zero(gw_client, px_client, db) -> None:
    """未声明的工具在边界上就被拒了，目标系统上什么都没留下——看门狗不为它扫。

    这一条防的是"没量到"被折成 0：**0 说的是"跑了，而且快到量不出来"**，
    两个意思混在一起之后，看板上的中位数会被一批根本没测过的调用拉低，而它
    看起来完全正常。
    """
    created = await _register(gw_client)
    _allow_host()

    resp = await _post(px_client, "/tools/nope", _PAYLOAD, created["api_key"])
    assert resp.status_code == 403, resp.text

    row = (await _rows(created["run_id"]))[0]
    assert row.status == "blocked"
    assert row.watchdog_ms is None, (
        f"这条调用没出到目标系统，看门狗一次都没跑，这一列该是 NULL，实际是 {row.watchdog_ms}"
    )


@pytest.mark.asyncio
async def test_a_denied_egress_leaves_it_null_not_zero(gw_client, px_client, db) -> None:
    """出口被ADR 003 默认拒绝的那一支：同样不扫，同样留 NULL。

    用通用出口走一台**没放行**的主机。它和上一条是两条不同的分支（一条判工具声明、
    一条判出口名单），而"被拦下的不扫"这条边界对两条都成立。
    """
    created = await _register(gw_client)
    _allow_host()

    envelope = {"url": "https://evil.example.com/v1/charges", "body": {"a": 1}}
    resp = await _post(px_client, "/fetch/http_post", envelope, created["api_key"])
    assert resp.status_code == 403, resp.text

    row = (await _rows(created["run_id"]))[0]
    assert row.blocked_by_proxy is True
    assert row.watchdog_ms is None


# --- 四：报告与看板真的把它读出来了 ---------------------------------------------------


@pytest.mark.asyncio
async def test_the_metrics_rollup_counts_ran_and_did_not_run_separately(
    gw_client, px_client, db
) -> None:
    """看板上这两栏必须分开：**跑过的**统计中位数，**没跑过的**单独计数。

    `LatencyStats.of` 只收到跑过的那批；如果把 NULL 当成 0 一起喂进去，中位数会
    被一批压根没测过的调用拉低——而页面上那一行看不出任何异样。所以除了
    `watchdog.count`，还得有一个 `calls_without_watchdog` 把缺口摆出来。
    """
    created = await _register(gw_client)
    _allow_host()

    hit = await _post(px_client, "/tools/create_payment", _PAYLOAD, created["api_key"])
    assert hit.status_code == 200
    # 第二条根本到不了目标系统：它进 `tool_calls` 那一栏（能算出耗时），
    # 但不进 `watchdog` 那一栏。
    missed = await _post(px_client, "/tools/nope", _PAYLOAD, created["api_key"])
    assert missed.status_code == 403

    got = await metrics.report(limit=10)

    assert got.latency.tool_calls.count == 2, "两条调用都记在工具调用那一栏（的分子里）"
    assert got.latency.watchdog.count == 1, "只有一条真的出到了目标系统"
    assert got.latency.calls_without_watchdog == 1, (
        "没跑看门狗的那条必须单独计数——并进上面那一栏就是把没量到读成 0"
    )
