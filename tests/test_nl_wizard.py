"""自然语言录入：一句想验证的话 → 确认页 → 一份能跑的场景。

与向导测试同源的两种错法要在这里各自钉住：

**一、LLM 回来的是不可信输入。** 模型输出是自由文本，只能当用户输入对待：
非法症状键、非法取值、不存在的工具、非数字的扣款笔数——清洗层必须丢掉或
回退，不能让它流进 `Answers`。`answers_from_json` 的每一条兜底都要有测试。

**二、没有模型时不许假装。** `answers_from_text` 的契约是"永远不抛异常，只
降级到内置关键词规则，并明说这是草稿"。测试把 `upstream_ready` 钉成 False，
断言它确实走了规则、确实带上了那句"内置规则猜的"警告——静默降级（不警告）
等于把用户话里的意思悄悄换掉。

端到端那条不联网：monkeypatch `nl_wizard.answers_from_text` 返回一份固定答案，
验证 GET/POST 页面、确认页判卷标准、保存按钮指向与「返回改一改」回带原文。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import pytest_asyncio

from sandbox.api import nl_wizard, wizard
from sandbox.api.app import create_app
from sandbox.db.engine import init_db

FACTS = wizard.tool_facts()


def _rule_answers(text: str) -> wizard.Answers:
    return nl_wizard._rule_answers(text, FACTS)[0]


# --- 一、本地关键词规则（降级路径） ------------------------------------------


def test_rule_duplicate_charge() -> None:
    a = _rule_answers("同一笔钱会被付两次，应该被抓到")
    assert a.choices == {"charge_twice": "expect_fail"}
    assert a.shape == "twice"
    assert "create_payment" in a.tools


def test_rule_loop_words() -> None:
    a = _rule_answers("它翻页一直翻不到头，停不下来")
    assert a.choices == {"loop_any": "expect_fail"}


def test_rule_autonomy_overnight() -> None:
    a = _rule_answers("半夜没人看着的时候，它会不会自己动手？")
    assert a.choices == {"autonomy_any": "expect_fail"}
    assert a.shape == "overnight"


def test_rule_handoff_relay() -> None:
    a = _rule_answers("换一个人接手，上次做过的事会不会又做一遍？")
    assert a.shape == "relay"


def test_rule_clean_default() -> None:
    a = _rule_answers("一切正常，不该报警")
    assert a.choices == {"charge_twice": "expect_pass"}
    assert a.shape == "once"


def test_rule_keeps_task_text() -> None:
    a = _rule_answers("帮我查一下订单 A1")
    assert a.agent_task == "帮我查一下订单 A1"


# --- 二、LLM JSON 清洗 ------------------------------------------------------


def test_answers_from_json_cleans_bad_keys_and_values() -> None:
    obj = {
        "scenario_id": "My Test!!",
        "agent_version": "order_agent@0.1",
        "shape": "twice",
        "tools": ["get_order", "not_a_tool", "get_order"],
        "choices": {
            "charge_twice": "expect_fail",
            "not_a_symptom": "expect_fail",
            "loop_any": "whatever",
        },
        "expected_charges": "2",
    }
    answers, warnings = nl_wizard.answers_from_json(obj, FACTS)
    assert answers.scenario_id == "my-test"
    assert answers.tools == ("get_order",)
    assert answers.choices == {"charge_twice": "expect_fail"}
    assert answers.expected_charges == 2


def test_answers_from_json_bad_shape_falls_back_with_warning() -> None:
    obj = {"scenario_id": "x", "shape": "sideways"}
    answers, warnings = nl_wizard.answers_from_json(obj, FACTS)
    assert answers.shape == "once"
    assert any("跑法没认出来" in w for w in warnings)


def test_answers_from_json_bad_charges_warns_and_drops() -> None:
    obj = {"scenario_id": "x", "expected_charges": "两笔"}
    answers, warnings = nl_wizard.answers_from_json(obj, FACTS)
    assert answers.expected_charges is None
    assert any("扣款笔数不是数字" in w for w in warnings)


def test_answers_from_json_missing_task_warns() -> None:
    obj = {"scenario_id": "x"}
    answers, warnings = nl_wizard.answers_from_json(obj, FACTS)
    assert answers.agent_task == ""
    assert any("没有提炼出给 Agent 的指令" in w for w in warnings)


# --- 三、主入口：降级与 LLM 两条路径 ------------------------------------------


def test_answers_from_text_empty_raises() -> None:
    with pytest.raises(nl_wizard.PlainTextError):
        import asyncio

        asyncio.run(nl_wizard.answers_from_text("   ", FACTS, []))


def test_answers_from_text_falls_back_when_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    monkeypatch.setattr(
        nl_wizard.config,
        "settings",
        SimpleNamespace(
            upstream_ready=False, upstream_api_key=None, upstream_base_url=""
        ),
    )
    monkeypatch.setattr(nl_wizard.user_llm, "effective", lambda: None)
    monkeypatch.setattr(nl_wizard.user_llm, "ready", lambda: False)
    answers, warnings = asyncio.run(
        nl_wizard.answers_from_text("同一笔钱会被付两次", FACTS, ["order_agent@0.1"])
    )
    assert answers.choices == {"charge_twice": "expect_fail"}
    assert any("内置关键词规则" in w for w in warnings)


def test_answers_from_text_uses_llm_json(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    async def fake_llm(text: str, facts: list[wizard.ToolFact], versions: list[str], selected_agents: list[str] | None = None) -> dict[str, Any]:
        return {
            "scenario_id": "pay-check",
            "agent_what": "付款助手",
            "agent_goal": "同一笔钱只付一次",
            "agent_task": "帮我把订单 A1 处理掉",
            "tools": ["get_order", "create_payment"],
            "shape": "twice",
            "choices": {"charge_twice": "expect_fail"},
            "expected_charges": 2,
        }

    monkeypatch.setattr(nl_wizard, "_llm_json", fake_llm)
    monkeypatch.setattr(nl_wizard.user_llm, "ready", lambda: True)
    answers, warnings = asyncio.run(
        nl_wizard.answers_from_text("同一笔钱会不会付两次", FACTS, ["order_agent@0.1"])
    )
    assert answers.scenario_id == "pay-check"
    assert answers.choices == {"charge_twice": "expect_fail"}
    assert answers.expected_charges == 2
    assert warnings == []


# --- 四、路由端到端（不联网） ------------------------------------------------


@pytest_asyncio.fixture
async def client(
    temp_db: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[httpx.AsyncClient]:
    await init_db()
    app = create_app(scenarios_root=tmp_path / "scenarios")

    async def fake_answers(text: str, facts: list[wizard.ToolFact], versions: list[str], selected_agents: list[str] | None = None):
        answers = wizard.Answers(
            scenario_id="pay-check",
            agent_what="付款助手",
            agent_goal="同一笔钱只付一次",
            agent_version="order_agent@0.1",
            agent_task="帮我把订单 A1 处理掉",
            tools=("get_order", "create_payment"),
            shape="twice",
            choices={"charge_twice": "expect_fail"},
            expected_charges=2,
        )
        return answers, ["（测试用的说明）"]

    monkeypatch.setattr(nl_wizard, "answers_from_text", fake_answers)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://nl") as client:
        yield client


async def test_nl_page_renders(client: httpx.AsyncClient) -> None:
    r = await client.get("/scenarios/plain")
    assert r.status_code == 200
    assert "用一句话建一个测试" in r.text
    assert "帮我生成测试" in r.text
    assert "data-fill" in r.text


async def test_nl_parse_shows_confirm(client: httpx.AsyncClient) -> None:
    r = await client.post("/scenarios/plain", data={"text": "同一笔钱会被付两次"})
    assert r.status_code == 200
    assert "判卷标准" in r.text
    assert "存下来" in r.text
    assert "/scenarios/pay-check/save" in r.text
    assert "同一笔钱会被付两次" in r.text  # 原文回显


async def test_nl_back_keeps_text(client: httpx.AsyncClient) -> None:
    r = await client.post(
        "/scenarios/plain", data={"text": "同一笔钱会被付两次", "submit": "back"}
    )
    assert r.status_code == 200
    assert "用一句话建一个测试" in r.text
    assert 'name="text"' in r.text
    assert "同一笔钱会被付两次" in r.text


async def test_nl_save_pipeline_is_the_same_save_route(client: httpx.AsyncClient) -> None:
    """确认页的存盘按钮必须指向现有 `/scenarios/{id}/save`（先解析再落盘）。"""
    r = await client.post("/scenarios/plain", data={"text": "同一笔钱会被付两次"})
    assert 'action="/scenarios/pay-check/save"' in r.text


# --- 五、首页产品化 hero -----------------------------------------------------


@pytest_asyncio.fixture
async def home_client(temp_db: str, tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    await init_db()
    app = create_app(scenarios_root=tmp_path / "scenarios")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://home") as client:
        yield client


async def test_home_has_plain_language_entry(home_client: httpx.AsyncClient) -> None:
    """首页必须给不懂技术的人一个「用一句话建测试」的入口。"""
    r = await home_client.get("/")
    assert r.status_code == 200
    assert "用一句话建测试" in r.text
    assert "/scenarios/plain" in r.text
    assert "现成测试包" not in r.text
    assert "测试任务" in r.text


# --- 五、选 Agent 与连接设置（用户配置层） ----------------------------------


async def test_nl_page_shows_agent_picker(client: httpx.AsyncClient) -> None:
    """NL 页第二步必须列出可被测的 Agent（从 agents/ 目录读，不是写死的）。"""
    r = await client.get("/scenarios/plain")
    assert r.status_code == 200
    assert "agent-pick" in r.text
    assert "order_agent@0.1" in r.text
    assert "delivery_agent@0.1" in r.text
    assert "第 2 步" in r.text


def test_two_selected_agents_force_relay(monkeypatch: pytest.MonkeyPatch) -> None:
    """勾了 2 个 Agent：shape 强制 relay，0 号第一段、1 号第二段。"""
    import asyncio

    async def fake_llm(text, facts, versions, selected_agents=None):
        return {
            "scenario_id": "relay-check",
            "agent_what": "接力",
            "agent_goal": "换人接手",
            "agent_task": "第一段",
            "agent2_task": "第二段",
            "tools": ["get_order", "create_payment"],
            "shape": "once",  # 模型想跑单程，但用户勾了 2 个 → 必须打回 relay
            "choices": {"charge_twice": "expect_fail"},
        }

    monkeypatch.setattr(nl_wizard, "_llm_json", fake_llm)
    monkeypatch.setattr(nl_wizard.user_llm, "ready", lambda: True)
    answers, _ = asyncio.run(
        nl_wizard.answers_from_text(
            "换一个人接手，上次做过的事会不会又做一遍",
            FACTS,
            ["order_agent@0.1", "delivery_agent@0.1"],
            selected_agents=["order_agent@0.1", "delivery_agent@0.1"],
        )
    )
    assert answers.shape == "relay"
    assert answers.agent_version == "order_agent@0.1"
    assert answers.agent2_version == "delivery_agent@0.1"
    assert answers.agent2_task == "第二段"


def test_one_selected_agent_locks_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """勾 1 个 Agent：agent_version 锁死，模型想换人也不行。"""
    import asyncio

    async def fake_llm(text, facts, versions, selected_agents=None):
        return {
            "scenario_id": "lock-check",
            "agent_what": "履约",
            "agent_goal": "同一单发一次",
            "agent_task": "处理订单",
            "tools": ["get_order", "create_payment"],
            "shape": "twice",
            "agent_version": "delivery_agent@0.1",  # 模型不听话，想换
            "choices": {"charge_twice": "expect_fail"},
        }

    monkeypatch.setattr(nl_wizard, "_llm_json", fake_llm)
    monkeypatch.setattr(nl_wizard.user_llm, "ready", lambda: True)
    answers, _ = asyncio.run(
        nl_wizard.answers_from_text(
            "同一单会不会发两次",
            FACTS,
            ["order_agent@0.1"],
            selected_agents=["order_agent@0.1"],
        )
    )
    assert answers.agent_version == "order_agent@0.1"


async def test_settings_save_and_clear(client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """/settings 能保存用户连接、能清除；保存后 user_llm 读得回来。"""
    import sandbox.user_llm as ul

    monkeypatch.setattr(ul, "FILE", tmp_path / "local.json")
    r = await client.post(
        "/settings",
        data={"base_url": "https://api.example.com/v1", "api_key": "sk-test-123", "model": "my-model"},
    )
    assert r.status_code == 303
    cfg = ul.load()
    assert cfg and cfg["api_key"] == "sk-test-123"
    assert cfg["model"] == "my-model"

    r2 = await client.get("/settings?saved=1")
    assert r2.status_code == 200
    assert "连接设置" in r2.text
    assert "已保存" in r2.text
    assert "api.example.com" in r2.text

    r3 = await client.post("/settings", data={"submit": "clear"})
    assert r3.status_code == 303
    assert ul.load() is None


async def test_settings_rejects_empty_key(client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """空密钥不许保存：要填接口地址和密钥，缺一不可。"""
    import sandbox.user_llm as ul

    monkeypatch.setattr(ul, "FILE", tmp_path / "local.json")
    r = await client.post("/settings", data={"base_url": "https://api.example.com", "api_key": "", "model": "m"})
    assert r.status_code == 200
    assert "接口地址和密钥都要填" in r.text
    assert ul.load() is None
