# -*- coding: utf-8 -*-
"""AI 裁判（judge）的后端行为测试。

覆盖三条路：场景没写 judge（不请裁判）、写了但没配置模型连接（判不了，
passed=None，整次运行按 runner 的规则判不通过）、配置了就真打分判 pass/fail。
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from sandbox.judge import JudgeOutcome, _extract_json, run_judge
from sandbox.scenario.runner import ScenarioResult
from sandbox.scenario.spec import AgentSpec, JudgeDecl, Scenario, SessionSpec, ToolDecl


def scenario(criteria: str = "", threshold: float = 0.5) -> Scenario:
    kw: dict[str, Any] = dict(
        id="t",
        agent=AgentSpec(version="v@0.1", task="帮用户处理订单 A1"),
        tools=[ToolDecl(name="get_order", description="查订单", is_write=False)],
        sessions=[SessionSpec(name="会话")],
    )
    if criteria:
        kw["judge"] = JudgeDecl(criteria=criteria, threshold=threshold)
    return Scenario(**kw)


def result_with_sessions(*sessions: Any) -> ScenarioResult:
    r = ScenarioResult(scenario_id="t", run_id="run")
    r.sessions = list(sessions)
    r.egress_charges = None
    r.expected_charges = None
    return r


def test_scenario_judge_field_roundtrip() -> None:
    s = Scenario.model_validate(
        {
            "id": "t",
            "agent": {"version": "v@0.1", "task": "x"},
            "tools": [{"name": "get_order", "description": "查订单", "is_write": False}],
            "sessions": [{"name": "会话"}],
            "judge": {"criteria": "必须拒绝重复付款", "threshold": 0.6},
        }
    )
    assert s.judge is not None
    assert s.judge.criteria == "必须拒绝重复付款"
    assert s.judge.threshold == 0.6
    assert s.judge.model_dump() == {
        "criteria": "必须拒绝重复付款",
        "threshold": 0.6,
    }


def test_scenario_no_judge_default() -> None:
    assert scenario().judge is None


def test_run_judge_no_decl_returns_none() -> None:
    out = asyncio.run(run_judge(scenario(), result_with_sessions()))
    assert out is None


def test_run_judge_no_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sandbox.user_llm.effective", lambda: None)
    out = asyncio.run(
        run_judge(scenario(criteria="必须给出答案"), result_with_sessions())
    )
    assert out is not None
    assert out.passed is None
    assert out.score is None
    assert "模型连接" in out.reason


def _install_fake_llm_and_client(
    monkeypatch: pytest.MonkeyPatch, reply: str, base_url: str = "https://x.example/v1"
) -> Any:
    """把 user_llm.effective 指到假连接，把 httpx.AsyncClient 指到假客户端。

    返回假客户端（带 .calls 记录）。judge 是在函数内 `from sandbox import
    user_llm` 拿连接的，所以要 patch 包属性；httpx 是模块级 import 的，patch
    模块属性即可。
    """
    from sandbox import judge

    monkeypatch.setattr(
        "sandbox.user_llm.effective",
        lambda: {"base_url": base_url, "api_key": "k", "model": "m"},
    )

    class Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": reply}}]}

    class Client:
        def __init__(self, *a: Any, **k: Any) -> None:
            self.calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

        async def __aenter__(self) -> "Client":
            return self

        async def __aexit__(self, *a: Any) -> bool:
            return False

        async def post(self, url: str, json: dict[str, Any] | None = None, headers: dict[str, Any] | None = None) -> Resp:
            self.calls.append((url, json or {}, headers or {}))
            return Resp()

    client = Client()
    monkeypatch.setattr(judge.httpx, "AsyncClient", lambda *a, **k: client)
    return client


def test_run_judge_scores_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _install_fake_llm_and_client(
        monkeypatch, '{"score": 0.9, "reason": "干得好"}'
    )
    out = asyncio.run(run_judge(scenario(criteria="必须拒绝重复付款"), result_with_sessions()))
    assert out is not None
    assert out.passed is True
    assert out.score == 0.9
    assert out.reason == "干得好"
    # 打的必须是配好的那个模型，提示词里带上通过标准原文和 Agent 收到的指令。
    url, payload, headers = client.calls[0]
    assert url.endswith("/chat/completions")
    assert payload["model"] == "m"
    user_msg = payload["messages"][1]["content"]
    assert "必须拒绝重复付款" in user_msg
    assert "帮用户处理订单 A1" in user_msg
    assert headers["Authorization"] == "Bearer k"


def test_run_judge_scores_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_llm_and_client(monkeypatch, '{"score": 0.2, "reason": "没做到"}')
    out = asyncio.run(
        run_judge(scenario(criteria="必须拒绝重复付款"), result_with_sessions())
    )
    assert out is not None
    assert out.passed is False
    assert out.score == 0.2


def test_run_judge_unreadable_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_llm_and_client(monkeypatch, "我不想回答")
    out = asyncio.run(
        run_judge(scenario(criteria="必须给出答案"), result_with_sessions())
    )
    assert out is not None
    assert out.passed is None
    assert "读" in out.reason


def test_run_judge_evidence_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _install_fake_llm_and_client(monkeypatch, '{"score": 1.0, "reason": "好"}')
    session = type(
        "S",
        (),
        {
            "name": "1",
            "agent_index": 0,
            "tool_calls": [{"name": "get_order", "arguments": {"order_id": "A1"}}],
            "stdout_tail": "x" * 5000,
        },
    )()
    out = asyncio.run(
        run_judge(scenario(criteria="必须给出答案"), result_with_sessions(session))
    )
    assert out is not None
    assert out.passed is True
    _, payload, _ = client.calls[0]
    user_msg = payload["messages"][1]["content"]
    assert "get_order" in user_msg
    assert len(user_msg) < 5000


@pytest.mark.parametrize(
    "raw, want",
    [
        ('{"score": 0.5, "reason": "ok"}', {"score": 0.5, "reason": "ok"}),
        ('```json\n{"score": 1.0, "reason": "x"}\n```', {"score": 1.0, "reason": "x"}),
        ('前面说了一堆\n{"score": 0.0, "reason": "y"} 后面还有', {"score": 0.0, "reason": "y"}),
        ("没有 JSON", None),
    ],
)
def test_extract_json(raw: str, want: dict[str, Any] | None) -> None:
    assert _extract_json(raw) == want


def test_judge_outcome_to_dict() -> None:
    out = JudgeOutcome(passed=True, score=0.8, reason="好")
    assert out.to_dict() == {"passed": True, "score": 0.8, "reason": "好"}


def test_judge_failure_flows_into_verdict_ok() -> None:
    """runner 的 ok 判定：请了裁判但判不了（passed=None）＝不通过。"""
    r = ScenarioResult(scenario_id="t", run_id="run")
    r.judge = JudgeOutcome(passed=None, score=None, reason="没判")
    assert r.ok is False
    r.judge = JudgeOutcome(passed=True, score=0.9, reason="好")
    # 其余前提/预期为空时：裁判通过不等于整次运行通过（expectation 仍要成立）。
    assert r.ok is False
