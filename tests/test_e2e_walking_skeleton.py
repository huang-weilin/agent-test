"""端到端验收：README 里写明的三条标准，逐条证。

    1. 被测 Agent 只改配置（base_url + api_key）就能被网关完整观测
    2. 人为构造一次重复扣款 → 幂等检测器必须报出来
    3. 人为构造一次循环 → 循环检测器必须报出来

这里跑的是一条**真的 HTTP 链路**：agent 客户端 → 网关 → (stub 上游)
　　　　　　　　　　　　　　　　　　↘ 出口代理 → mock 目标系统
每一步都是真实的 HTTP 请求与真实的数据库写入，没有绕过任何一层。

用脚本化上游而不是真实 LLM，是因为这条测试要证的是**管道正确性**。真实模型
引入的不确定性会让"检测器没报出来"有两种解释——它坏了，或者模型这次没触发。
脚本化之后，"检测器报没报出来"是唯一变量。
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from sandbox.db.models import DetectorFinding, LlmCall, ToolCall
from sandbox.detectors.base import ExemptionSet
from sandbox.detectors.idempotency import IdempotencyDetector, ToolFacts
from sandbox.detectors.loop import LoopDetector
from sandbox.gateway import app as gateway_app
from sandbox.gateway.stub import scripted_upstream
from sandbox.mock import server as mock_server
from sandbox.mock.toolspec import idempotency_params
from sandbox.proxy import app as proxy_app
from sandbox.runtime import runs
from sandbox.trace.reader import load_call_views


@pytest.fixture(autouse=True)
def _reset_state():
    runs.clear()
    proxy_app.set_policy(None)
    proxy_app.set_transport(ASGITransport(app=mock_server.app))
    mock_server.get_state().reset()
    yield
    runs.clear()
    proxy_app.set_policy(None)
    proxy_app.set_transport(None)
    mock_server.get_state().reset()


@pytest_asyncio.fixture
async def agent(temp_db: str):
    """一个"被测 Agent"：只持有网关地址与沙箱令牌，没有任何沙箱内部知识。

    它不知道 mock 在哪，不知道代理的白名单，也不知道自己的调用正在被记录。
    这正是 ADR 001 要验证的那种观测方式。
    """
    gateway = AsyncClient(
        transport=ASGITransport(app=gateway_app.app), base_url="http://gateway.test"
    )
    proxy = AsyncClient(
        transport=ASGITransport(app=proxy_app.app), base_url="http://proxy.test"
    )

    class Agent:
        def __init__(self) -> None:
            self.base_url = str(gateway.base_url)
            self.api_key = ""
            self.tool_base_url = str(proxy.base_url)
            self.messages: list[dict] = []
            self.llm_calls = 0
            self.tool_results: list[dict] = []

        async def register(self, **overrides) -> dict:
            payload = {
                "scenario_id": "e2e",
                "agent_version": "react_demo@test",
                "declared_writes": ["create_payment"],
                "idempotency_key_params": idempotency_params(),
                **overrides,
            }
            resp = await gateway.post("/_sandbox/runs", json=payload)
            assert resp.status_code == 200, resp.text
            created = resp.json()
            self.api_key = created["api_key"]
            return created

        async def run_react_loop(self, max_steps: int = 10) -> None:
            """标准 ReAct 循环。工具调用走出口代理，模型调用走网关。"""
            self.messages = [{"role": "user", "content": "帮我把订单处理掉"}]
            for _ in range(max_steps):
                resp = await gateway.post(
                    "/v1/chat/completions",
                    json={
                        "model": "stub-model",
                        "messages": self.messages,
                        "stream": False,
                    },
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"网关拒绝：{resp.status_code} {resp.text}")
                self.llm_calls += 1

                message = resp.json()["choices"][0]["message"]
                self.messages.append(message)

                tool_calls = message.get("tool_calls")
                if not tool_calls:
                    return

                for call in tool_calls:
                    import json

                    args = json.loads(call["function"]["arguments"])
                    tool_resp = await proxy.post(
                        f"/tools/{call['function']['name']}",
                        json=args,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                    )
                    result = tool_resp.json()
                    self.tool_results.append(result)
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )

        async def close(self) -> None:
            await gateway.aclose()
            await proxy.aclose()

    a = Agent()
    yield a
    await a.close()


async def _script(run_id: str, steps: list[dict]) -> None:
    scripted_upstream.register(run_id, steps)


# --- 标准一：只改配置就能被完整观测 ---------------------------------------


async def test_agent_is_fully_observed_without_touching_its_code(agent, db) -> None:
    """Agent 只拿到 base_url 与 api_key，全部调用都被记录了。

    断言的是"完整"：Agent 发了几次模型调用、几次工具调用，库里就必须有
    同样多的行。少记一次就意味着报告会低估——而低估是看不见的。
    """
    created = await agent.register()
    await _script(
        created["run_id"],
        [
            {"tool_calls": [{"name": "get_order", "arguments": {"order_id": "A1"}}]},
            {"content": "订单已查到。"},
        ],
    )

    await agent.run_react_loop()

    # Agent 侧视角：它认为自己发了 2 次模型调用、1 次工具调用
    assert agent.llm_calls == 2

    # 沙箱侧视角：必须一致
    llm_rows = (await db.execute(select(LlmCall))).scalars().all()
    tool_rows = (await db.execute(select(ToolCall))).scalars().all()
    assert len(llm_rows) == agent.llm_calls, "网关漏记了模型调用"
    assert len(tool_rows) == 1, "出口代理漏记了工具调用"

    call = tool_rows[0]
    assert call.tool_name == "get_order"
    assert call.test_run_id == created["run_id"]
    assert call.is_write is False

    # 观测到了什么：token 数只有网关知道，Agent 从未上报过
    assert sum(r.input_tokens for r in llm_rows) > 0


async def test_agent_never_receives_the_real_upstream_key(agent) -> None:
    """Agent 手里的凭据是沙箱令牌，不是真实上游密钥。

    这点如果错了，"完整观测"会**静默**失效：Agent 有真实密钥就可以绕过网关
    直连上游，而报告上一切正常，只是少了一半调用。
    """
    created = await agent.register()
    assert created["api_key"].startswith("sk-sandbox-")

    from sandbox.config import settings

    if settings.upstream_api_key:
        assert settings.upstream_api_key not in created["api_key"]


# --- 标准二：构造的重复扣款必须被报出来 -----------------------------------


async def test_constructed_duplicate_charge_is_caught(agent, db) -> None:
    """构造：Agent 对同一个订单、同样金额付了两次。

    而且要让**真实副作用**发生两次——mock 的扣款流水必须真的有两条，
    否则检测器报的是个不存在的问题，测试也就没证明任何事。
    """
    created = await agent.register()
    pay = {"order_id": "A1", "amount": 599}
    await _script(
        created["run_id"],
        [
            {"tool_calls": [{"name": "create_payment", "arguments": pay}]},
            {"tool_calls": [{"name": "create_payment", "arguments": pay}]},
            {"content": "支付完成。"},
        ],
    )

    await agent.run_react_loop()

    # 先确认缺陷真实发生了
    charges = mock_server.get_state().charges
    assert len(charges) == 2, f"重复扣款没有真的发生，检测器无从谈起：{charges}"
    assert charges[0]["order_id"] == charges[1]["order_id"] == "A1"

    # 再确认检测器报了出来
    calls = await load_call_views(created["run_id"])
    facts = {"create_payment": ToolFacts("create_payment", True, enforces_idempotency=False)}
    findings = IdempotencyDetector(facts).run(calls, ExemptionSet())

    fails = [f for f in findings if f.verdict.value == "fail"]
    assert len(fails) == 1, f"检测器没报出重复扣款：{[f.verdict.value for f in findings]}"
    assert "重复扣款" in fails[0].summary


async def test_two_different_orders_do_not_trigger_a_false_alarm(agent) -> None:
    """负样本：两次不同订单的支付是正常业务。

    没有这条，"检测器总能报出重复扣款"这个断言是没意义的。
    """
    created = await agent.register()
    await _script(
        created["run_id"],
        [
            {
                "tool_calls": [
                    {"name": "create_payment", "arguments": {"order_id": "A1", "amount": 599}}
                ]
            },
            {
                "tool_calls": [
                    {"name": "create_payment", "arguments": {"order_id": "B2", "amount": 249}}
                ]
            },
            {"content": "两单都付好了。"},
        ],
    )

    await agent.run_react_loop()

    assert len(mock_server.get_state().charges) == 2
    calls = await load_call_views(created["run_id"])
    findings = IdempotencyDetector().run(calls, ExemptionSet())
    assert all(f.verdict.value != "fail" for f in findings), "正常业务被报成了重复扣款"


# --- 标准三：构造的循环必须被报出来，且熔断能真的终止它 --------------------


async def test_constructed_loop_is_caught_and_killed(agent, db) -> None:
    """构造：Agent 卡在循环里，反复查同一个订单。

    熔断要证明两件事，缺一不可：
      - 循环被判定出来了（检测器那半）
      - 熔断之后运行**真的停了**，不是"报了警但继续烧钱"
    """
    created = await agent.register()
    await _script(
        created["run_id"],
        [
            {"tool_calls": [{"name": "get_order", "arguments": {"order_id": "A1"}}]}
            for _ in range(8)
        ],
    )

    with pytest.raises(RuntimeError, match="410|拒绝"):
        await agent.run_react_loop()

    calls = await load_call_views(created["run_id"])
    findings = LoopDetector().run(calls, ExemptionSet())
    strict = [f for f in findings if f.detector == "loop.strict"]

    assert strict, "反复调用同一个工具没有被判为循环"
    assert strict[0].verdict.value == "fail"

    # 熔断确实生效：后续请求被网关拒绝
    resp = await _post_chat(agent.api_key)
    assert resp.status_code == 410
    assert resp.json()["error"]["type"] == "run_killed"


async def test_kill_actually_stops_further_tool_side_effects(agent) -> None:
    """熔断之后不能再产生任何真实副作用。

    只在网关上拦截是不够的——工具调用走的是另一条边界。两条都拦住，
    "终止"才成立。
    """
    created = await agent.register()
    await _script(
        created["run_id"],
        [
            {"tool_calls": [{"name": "get_order", "arguments": {"order_id": "A1"}}]}
            for _ in range(8)
        ],
    )

    with pytest.raises(RuntimeError):
        await agent.run_react_loop()

    charges_before = len(mock_server.get_state().charges)
    resp = await _post_tool(agent.api_key, "create_payment", {"order_id": "A1", "amount": 599})

    assert resp.status_code == 410
    assert len(mock_server.get_state().charges) == charges_before, (
        "熔断后工具调用仍然到达了目标系统"
    )


async def test_kill_handlers_fire_for_out_of_band_termination(agent) -> None:
    """带外那半：熔断要通知外部去**终止 Agent 进程**。

    只靠网关拒绝请求是不够的——Agent 可能正在一次长推理里，或者压根不再经过
    网关。真正止损必须能杀掉进程，所以熔断必须发出一个进程外的信号。
    """
    created = await agent.register()
    fired: list[tuple[str, str]] = []
    runs.on_kill(lambda rid, reason: fired.append((rid, reason)))

    await _script(
        created["run_id"],
        [
            {"tool_calls": [{"name": "get_order", "arguments": {"order_id": "A1"}}]}
            for _ in range(8)
        ],
    )

    with pytest.raises(RuntimeError):
        await agent.run_react_loop()

    assert len(fired) == 1
    assert fired[0][0] == created["run_id"]
    assert "loop.strict" in fired[0][1]


# --- 四态语义：不能把"没测到"读成"通过" ----------------------------------


async def test_agent_that_never_writes_yields_inconclusive_not_pass(agent, db) -> None:
    """只读的 Agent 走完一遍，幂等性**没有被验证过**。

    报告必须说 inconclusive。折叠成 pass 的话，用户会以为"幂等性检查通过了"，
    而实际上一次写操作都没发生过。
    """
    created = await agent.register()
    await _script(
        created["run_id"],
        [
            {"tool_calls": [{"name": "get_order", "arguments": {"order_id": "A1"}}]},
            {"content": "查完了。"},
        ],
    )
    await agent.run_react_loop()

    calls = await load_call_views(created["run_id"])
    findings = IdempotencyDetector().run(calls, ExemptionSet())

    assert any(f.verdict.value == "inconclusive" for f in findings)
    assert all(f.verdict.value != "pass" for f in findings), (
        "没有任何写操作，幂等性判定却报了通过"
    )


async def test_findings_are_persisted_with_four_state_verdicts(agent, db) -> None:
    """结论要落库，且四态取值原样保存。"""
    from sandbox.watchdog import inspect_run

    created = await agent.register()
    await _script(
        created["run_id"],
        [
            {"tool_calls": [{"name": "get_order", "arguments": {"order_id": "A1"}}]},
            {"content": "查完了。"},
        ],
    )
    await agent.run_react_loop()

    findings = await inspect_run(created["run_id"])
    assert findings

    rows = (await db.execute(select(DetectorFinding))).scalars().all()
    assert rows, "看门狗没有把结论落库"
    for row in rows:
        assert row.verdict in {"pass", "fail", "warning", "inconclusive"}


async def _post_chat(api_key: str):
    gateway = AsyncClient(
        transport=ASGITransport(app=gateway_app.app), base_url="http://gateway.test"
    )
    try:
        return await gateway.post(
            "/v1/chat/completions",
            json={"model": "stub-model", "messages": [], "stream": False},
            headers={"Authorization": f"Bearer {api_key}"},
        )
    finally:
        await gateway.aclose()


async def _post_tool(api_key: str, tool: str, args: dict):
    proxy = AsyncClient(
        transport=ASGITransport(app=proxy_app.app), base_url="http://proxy.test"
    )
    try:
        return await proxy.post(
            f"/tools/{tool}", json=args, headers={"Authorization": f"Bearer {api_key}"}
        )
    finally:
        await proxy.aclose()
