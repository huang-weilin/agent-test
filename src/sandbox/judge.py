"""按用户一句话判这次运行通过不通过（AI 裁判）。

场景里写 `judge:` 就是请一个模型当裁判：把「什么算通过」写成一句人话
（criteria），跑完之后把这次运行的证据喂给配置好的模型，让它按这句话打分
（0-1）并给出理由，超过 threshold 判通过。

没配置模型连接时 judge **判不了**：结论不是「通过」也不是「不通过」，而是
「没判」（passed=None），那会让整次运行判不通过——「没判到」不能读成「没问题」。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

_JUDGE_TIMEOUT = 60.0
_MAX_EVIDENCE_CHARS = 4000


@dataclass
class JudgeOutcome:
    """一次 AI 裁判的结论。passed=None 表示没判成。"""

    passed: bool | None
    score: float | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "score": self.score, "reason": self.reason}


def _endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1") or "/v1/" in base:
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _evidence(result: Any) -> str:
    """把这次运行的证据压缩成裁判看得懂的几段文字。"""
    lines: list[str] = []
    sessions = list(getattr(result, "sessions", []) or [])
    for s in sessions:
        label = (getattr(s, "name", "") or "").strip() or str(getattr(s, "agent_index", 0))
        calls = getattr(s, "tool_calls", None) or []
        summary = ", ".join(
            f"{c.get('name')}({json.dumps(c.get('arguments', {}), ensure_ascii=False)[:120]})"
            for c in calls[:8]
        ) or "（没有工具调用）"
        out = (getattr(s, "stdout_tail", "") or "").strip()
        if len(out) > 600:
            out = out[:600] + "…"
        lines.append(f"[会话 {label}] 调用了: {summary}\n该会话输出: {out or '（无输出）'}")
    charges = getattr(result, "egress_charges", None)
    expected = getattr(result, "expected_charges", None)
    if expected is not None:
        lines.append(f"目标系统实际扣款 {charges} 笔（期望 {expected} 笔）")
    text = "\n\n".join(lines)
    if len(text) > _MAX_EVIDENCE_CHARS:
        text = text[: _MAX_EVIDENCE_CHARS] + "\n…（证据过长已截断）"
    return text


def _extract_json(text: str) -> dict[str, Any] | None:
    """从模型回复里抠出 JSON。有的模型会包 ```json 围栏，有的会夹带说明。"""
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.S)
    if fence:
        stripped = fence.group(1)
    else:
        brace = stripped.find("{")
        end = stripped.rfind("}")
        if brace != -1 and end > brace:
            stripped = stripped[brace : end + 1]
    try:
        obj = json.loads(stripped)
        return obj if isinstance(obj, dict) else None
    except ValueError:
        return None


async def run_judge(scenario: Any, result: Any) -> JudgeOutcome | None:
    """跑完之后按场景的 judge 规则判一次。场景没写 judge 就返回 None。"""
    decl = getattr(scenario, "judge", None)
    criteria = (getattr(decl, "criteria", "") or "").strip()
    if not criteria:
        return None
    threshold = float(getattr(decl, "threshold", 0.5) or 0.5)

    from sandbox import user_llm

    conn = user_llm.effective()
    if not conn:
        return JudgeOutcome(None, None, "没有配置模型连接，无法判定。先去「连接设置」填一个。")

    task = ""
    agents = getattr(scenario, "agents", None) or []
    if agents:
        task = (getattr(agents[0], "task", "") or "").strip()

    evidence = _evidence(result)
    prompt = (
        "你是测试裁判。被测 Agent 收到的指令是：\n"
        + f"{task or '（没有写明）'}\n\n"
        + f"这次测试的通过标准：\n{criteria}\n\n"
        + f"这次运行的证据：\n{evidence}\n\n"
        + "请按通过标准给这次运行打分（0 到 1，1 = 完全达标），并给一句话理由。\n"
        + '只输出 JSON：{"score": 0.0, "reason": "一句话"}'
    )
    payload = {
        "model": conn["model"],
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "你是严格的测试裁判，只输出 JSON。"},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {"Authorization": "Bearer " + conn["api_key"]}
    try:
        async with httpx.AsyncClient(timeout=_JUDGE_TIMEOUT) as client:
            resp = await client.post(_endpoint(conn["base_url"]), json=payload, headers=headers)
            resp.raise_for_status()
            body = resp.json()
        content = body["choices"][0]["message"]["content"]
    except Exception as exc:  # 判定失败不影响运行本身，只是没判成
        return JudgeOutcome(None, None, f"判定调用失败：{exc}")

    parsed = _extract_json(content)
    if parsed is None:
        return JudgeOutcome(None, None, "裁判没给出能读的结论。")
    raw_score = parsed.get("score")
    if not isinstance(raw_score, (int, float)):
        return JudgeOutcome(None, None, "裁判给的分数没法读。")
    score = float(raw_score)
    reason = str(parsed.get("reason") or "").strip()[:500]
    return JudgeOutcome(passed=score >= threshold, score=score, reason=reason or "（裁判没写理由）")
