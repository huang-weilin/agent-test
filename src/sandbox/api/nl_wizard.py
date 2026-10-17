# -*- coding: utf-8 -*-
"""自然语言录入：把一句大白话变成一份场景的答案。

两条路径：

1. **LLM 解析（首选）**：调用配置里的上游（`SANDBOX_UPSTREAM_*`，默认 DeepSeek），
   让模型把用户的话翻译成 `wizard.Answers` 的 JSON。这只在 `settings.upstream_ready`
   时尝试——没配 key 就直接走下面那条，不假装。

2. **本地关键词规则（降级）**：没有 key、网络失败、或者模型返回的东西拼不成
   Answers 时，用关键词把最常见的几类毛病映射出来。它只覆盖最常见的几类，
   并且一定在警告里说明"这是内置规则猜的，请在确认页检查"。

两条路径的产出走**同一条确认页**：`nl_confirm.html` 复用向导确认页的判卷标准
展示与 `/scenarios/{id}/save` 存盘管道——自然语言建出来的场景和手写的没有区别，
也没有第二条真相源。

**为什么宁可降级也不报错**：这一页是给不懂技术的人用的第一入口。模型不可用时
让他空手而归，比给一个"可能不完全准、但明说了要检查"的草稿要糟得多——前者是
死路，后者是待审的草稿，而确认页存在的意义就是审它。
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

import httpx

from sandbox import config, user_llm
from sandbox.api import wizard

_NL_MODEL = "deepseek-chat"
_NL_TIMEOUT = 30.0


#: LLM 返回 JSON 时可能裹在 ```json ... ``` 里，或者带了前后废话。剥到只剩
#: 第一个 `{` 到最后一个 `}`。
_JSON_SKIN = re.compile(r"\{.*\}", re.S)


class PlainTextError(ValueError):
    """这段大白话拼不出一份能跑的场景。消息直接给用户看。"""


def _llm_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _tool_summary(facts: list[wizard.ToolFact]) -> list[dict[str, str]]:
    return [
        {
            "name": f.name,
            "描述": f.description,
            "副作用": "会扣钱/写东西" if f.is_write else "只读",
            "参数": json.dumps(dict(f.properties), ensure_ascii=False),
        }
        for f in facts
    ]


def _build_prompt(
    text: str,
    facts: list[wizard.ToolFact],
    agent_versions: list[str],
    selected_agents: list[str] | None = None,
) -> str:
    """给 LLM 的指令。用户的话是唯一事实来源，清单是它要遵守的选项。"""
    symptoms = "\n".join(
        f"- {s.key}：「{s.question}」——{s.when}"
        for s in wizard.SYMPTOMS
    )
    shapes = "\n".join(f"- {k}：{v}" for k, v in wizard.SHAPES.items())
    choices = "\n".join(f"- {k}：{v}" for k, v in wizard.CHOICES.items())
    tools = "\n".join(f"- {t['name']}：{t['描述']}（{'会扣钱/写东西' if t['副作用'] != '只读' else '只读'}）"
                     for t in _tool_summary(facts))
    return f"""你是「AI Agent 测试场景」的配置助手。用户会用大白话描述一个想给 AI Agent 出的考题，
你要把它翻译成一份结构化的场景答案 JSON。只输出 JSON，不要解释。

用户的话：{text}

可选症状（key：问法——触发它需要的剧情）：
{symptoms}

可选跑法（shape）：
{shapes}

每个症状可选的值（choice）：
{choices}

可用工具：
{tools}

可用的 Agent 版本：{', '.join(agent_versions) or 'react_demo@0.1'}
    用户指定的被测 Agent（必须从这里选，且只能选这些）：{', '.join(selected_agents) if selected_agents else '（未指定，你来定）'}
    选了 2 个时 shape 用 relay（0 号跑第一段、1 号跑第二段），agent2_version 填第二个。

输出 JSON 结构（全部字段）：
{{
  "scenario_id": "小写字母数字连字符，从用户的话里提炼，没有就用测试",
  "agent_version": "Agent 版本号，默认 react_demo@0.1",
  "agent_what": "这个 Agent 是干什么的，一句话（人话，给人看）",
  "agent_goal": "这次要验证什么，一句话（人话，给人看）",
  "agent_task": "这一趟对它说的指令（从用户的话提炼，若用户描述的是测试意图而非指令，就用一句合理的任务）",
  "tools": ["只用上面清单里存在的工具名"],
  "shape": "once/twice/cross/relay/overnight 之一",
  "choices": {{
    "症状key": "skip/expect_fail/expect_pass 之一"
  }},
  "expected_charges": "扣款笔数，用户没提钱就 null",
  "unattended_allowed": false,
  "agent2_version": "shape 是 relay 时才需要，否则省略",
  "agent2_task": "shape 是 relay 时才需要，否则省略"
}}

规则：
- choices 里只出现用户明确提到或明显暗示的症状；没提到的不要放。
- 用户说"该报警/会犯/抓得到"→ expect_fail；说"不该报警/正常/对照"→ expect_pass。
- tools 只从清单里选，且只选用户话里用得到的那几个。
- expected_charges 只有用户提到钱/扣款次数时才填数字，否则 null。
"""


async def _llm_json(
    text: str,
    facts: list[wizard.ToolFact],
    agent_versions: list[str],
    selected_agents: list[str] | None = None,
) -> dict[str, Any]:
    """调上游，返回解析出的 JSON 对象。任何失败都抛异常（由调用方降级）。

    连接只有一份来源：用户界面里配的 key（`user_llm.effective()`）。
    没有内置默认（`.env` 的 `SANDBOX_UPSTREAM_*` 是给被测 Agent 的上游，
    不是解析用的连接）。没配就走本地规则。
    """
    conn = user_llm.effective()
    if not conn:
        raise PlainTextError("没有可用的 LLM 连接（界面里没配 key），走本地规则。")
    payload = {
        "model": conn["model"],
        "messages": [
            {"role": "system", "content": _build_prompt(text, facts, agent_versions)},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {conn['api_key']}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=_NL_TIMEOUT) as client:
        resp = await client.post(_llm_endpoint(conn["base_url"]), json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    content = data["choices"][0]["message"]["content"]
    match = _JSON_SKIN.search(content)
    if not match:
        raise PlainTextError("模型返回的不是 JSON。")
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise PlainTextError("模型返回的不是对象。")
    return obj


def _clean_id(raw: str) -> str:
    """把任意字符串收拾成合法场景 id；空的给一个随机后缀，避免撞名。"""
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", raw.strip().lower()).strip("-_")
    if not cleaned:
        cleaned = f"plain-{uuid.uuid4().hex[:6]}"
    if not re.match(r"^[a-z0-9]", cleaned):
        cleaned = f"t-{cleaned}"
    return cleaned[:48]


def _clean_choices(raw: Any) -> dict[str, str]:
    """只认 16 个合法症状键和 3 个合法值；其余一律丢掉。"""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    valid_keys = {s.key for s in wizard.SYMPTOMS}
    for key, value in raw.items():
        if key in valid_keys and value in wizard.CHOICES:
            out[key] = str(value)
    return out


def _clean_tools(raw: Any, facts: list[wizard.ToolFact]) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    valid = {f.name for f in facts}
    seen: list[str] = []
    for name in raw:
        name = str(name)
        if name in valid and name not in seen:
            seen.append(name)
    return tuple(seen)


def answers_from_json(
    obj: dict[str, Any],
    facts: list[wizard.ToolFact],
    selected_agents: list[str] | None = None,
) -> tuple[wizard.Answers, list[str]]:
    """把 LLM 的 JSON 清洗成 `wizard.Answers`。拼不出的字段用安全默认值 + 警告。

    `selected_agents` 非空时，被测 Agent 锁定在用户选的那几个里：1 个就是它，
    2 个强制 relay 接力（0 号第一段、1 号第二段）。LLM 想跑出圈会直接被打回。
    """
    warnings: list[str] = []
    scenario_id = _clean_id(str(obj.get("scenario_id") or ""))
    agent_version = str(obj.get("agent_version") or "").strip() or "react_demo@0.1"
    shape = str(obj.get("shape") or "once")

    if selected_agents:
        agent_version = selected_agents[0]
        if len(selected_agents) >= 2:
            shape = "relay"
            agent2_version = selected_agents[1]
            agent2_task = str(obj.get("agent2_task") or "").strip()
        else:
            agent2_version = ""
            agent2_task = ""
    else:
        agent2_version = str(obj.get("agent2_version") or "").strip()
        agent2_task = str(obj.get("agent2_task") or "").strip()

    if shape not in wizard.SHAPES:
        shape = "once"
        warnings.append("跑法没认出来，按「正常做一次」处理。")

    tools = _clean_tools(obj.get("tools"), facts)
    choices = _clean_choices(obj.get("choices"))

    expected_charges: int | None = None
    raw_charges = obj.get("expected_charges")
    if raw_charges is not None and raw_charges != "":
        try:
            expected_charges = int(raw_charges)
        except (TypeError, ValueError):
            warnings.append("扣款笔数不是数字，这次不验钱。")

    unattended_allowed = bool(obj.get("unattended_allowed", False))

    answers = wizard.Answers(
        scenario_id=scenario_id,
        agent_what=str(obj.get("agent_what") or "（还没写）"),
        agent_goal=str(obj.get("agent_goal") or "（还没写）"),
        agent_version=agent_version,
        agent_task=str(obj.get("agent_task") or "").strip(),
        tools=tools,
        shape=shape,
        choices=choices,
        agent2_version=agent2_version,
        agent2_task=agent2_task,
        expected_charges=expected_charges,
        unattended_allowed=unattended_allowed,
    )
    if not answers.agent_task:
        warnings.append("没有提炼出给 Agent 的指令，这一趟它没有任务可做。")
    if not tools:
        warnings.append("没选任何工具，脚本会是空的。")
    return answers, warnings


# --- 本地关键词规则（降级路径） ----------------------------------------------


def _rule_answers(
    text: str, facts: list[wizard.ToolFact], selected_agents: list[str] | None = None
) -> tuple[wizard.Answers, list[str]]:
    """没有 LLM 时用关键词猜。只覆盖最常见几类，警告里明说。"""
    t = text
    choices: dict[str, str] = {}
    shape = "once"
    tools: list[str] = []
    writes = [f.name for f in facts if f.is_write]
    reads = [f.name for f in facts if not f.is_write]

    if re.search(r"付两次|扣两次|重复扣|重复付|多扣|又付|重复支付|重复发货|发两次", t):
        choices["charge_twice"] = "expect_fail"
        shape = "twice"
        tools += writes[:1] + reads[:1]
    if re.search(r"循环|打转|停不下来|反复|重试|翻页|翻不到头|一模一样", t):
        choices["loop_any"] = "expect_fail"
        tools += reads[:1]
    if re.search(r"半夜|没人|无人值守|自主|越权|自己动手|背着", t):
        choices["autonomy_any"] = "expect_fail"
        shape = "overnight"
    if re.search(r"授权|允许它|我同意", t):
        choices["autonomy_any"] = "expect_pass"
    if re.search(r"记不住|忘记|记忆|忘了", t):
        choices["memory_lost"] = "expect_fail"
    if re.search(r"污染|塞了|不该记|记住不该|中毒", t):
        choices["memory_poisoned"] = "expect_fail"
    if re.search(r"传错|参数|缺字段|类型不对|不合规", t):
        choices["tool_schema"] = "expect_fail"
    if re.search(r"没允许|不该调|不该连|越权调用|白名单外", t):
        choices["tool_scope"] = "expect_fail"
    if re.search(r"换人|接手|接力|第二个人|另一个 agent", t):
        shape = "relay"
    elif re.search(r"两段|跨会话|第二天|隔天|上次做过", t):
        shape = "cross"
    if re.search(r"正常|对照|不该报警|一切正常|干净|不报警", t):
        shape = "once"
        choices = {"charge_twice": "expect_pass"} if not choices else choices

    if not tools:
        tools = writes[:1] + reads[:1] if writes else reads[:1]
    if not choices:
        choices = {"charge_twice": "expect_pass"}

    if selected_agents and len(selected_agents) >= 2:
        shape = "relay"
    agent_version = (selected_agents[0] if selected_agents else "react_demo@0.1")
    agent2_version = (selected_agents[1] if selected_agents and len(selected_agents) >= 2 else "")

    answers = wizard.Answers(
        scenario_id="",
        agent_what="（待确认）",
        agent_goal="（待确认）",
        agent_version=agent_version,
        agent2_version=agent2_version,
        agent2_task=t if agent2_version else "",
        agent_task=t,
        tools=tuple(tools),
        shape=shape,
        choices=choices,
    )
    warnings = [
        "这次没有连上解析用的模型，下面这份是用内置关键词规则猜的草稿——"
        "它只认最常见的几类说法，请在确认页逐条检查再存。",
    ]
    return answers, warnings


async def answers_from_text(
    text: str,
    facts: list[wizard.ToolFact],
    agent_versions: list[str],
    selected_agents: list[str] | None = None,
) -> tuple[wizard.Answers, list[str]]:
    """主入口：一句话 → (Answers, warnings)。永远不抛异常，只降级。

    `selected_agents`：用户在界面上勾选的被测 Agent（可 1 个或 2 个）。
    传了就必须照选——LLM 只能在这几个里选版本，2 个时强制 relay 接力。
    """
    text = text.strip()
    if not text:
        raise PlainTextError("先写一句话，说清楚你想让测试验证什么。")

    if user_llm.ready():
        try:
            obj = await _llm_json(text, facts, agent_versions, selected_agents)
            answers, warnings = answers_from_json(obj, facts, selected_agents)
            answers = _finalize_id(answers, text)
            return answers, warnings
        except (PlainTextError, httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError):
            pass  # 降级到规则

    answers, warnings = _rule_answers(text, facts, selected_agents)
    answers = _finalize_id(answers, text)
    return answers, warnings


def _finalize_id(answers: wizard.Answers, text: str) -> wizard.Answers:
    """把 scenario_id 定下来：从用户的话里提一个短词，否则随机。"""
    if answers.scenario_id and not answers.scenario_id.startswith("t-"):
        return answers
    words = re.findall(r"[a-zA-Z\u4e00-\u9fff]{2,12}", text)
    base = ""
    for w in reversed(words):
        if w not in ("这个", "那个", "一个", "一下", "应该", "会不会", "的时候"):
            base = w
            break
    raw = base if base else "plain"
    cleaned = _clean_id(raw)
    if cleaned == "plain" or cleaned.startswith("t-"):
        cleaned = f"plain-{uuid.uuid4().hex[:6]}"
    return wizard.Answers(
        scenario_id=cleaned,
        agent_what=answers.agent_what,
        agent_goal=answers.agent_goal,
        agent_version=answers.agent_version,
        agent_task=answers.agent_task,
        tools=answers.tools,
        shape=answers.shape,
        choices=answers.choices,
        agent2_version=answers.agent2_version,
        agent2_task=answers.agent2_task,
        expected_charges=answers.expected_charges,
        unattended_allowed=answers.unattended_allowed,
    )


def agent_versions() -> list[str]:
    """agents/ 目录下可用的版本号（用于 prompt）。"""
    return [f"{a['name']}@0.1" for a in agents_list()] or ["react_demo@0.1"]


def agents_list() -> list[dict[str, str]]:
    """agents/ 目录下可被测的 Agent，带一句人话描述（给人选的）。

    描述从每个 agent 自己的 README.md 里提取标题行下的第一段：那是作者写给
    人看的「这个 Agent 是干嘛的」。提取失败就给一句通用兜底，绝不整页空白。
    """
    from sandbox.config import REPO_ROOT

    root = REPO_ROOT / "agents"
    out: list[dict[str, str]] = []
    if not root.exists():
        return out
    for d in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda x: x.name):
        desc = ""
        meta = d / "AGENT.json"
        if meta.exists():
            # 用户导入的 Agent：描述写在注册里，优先于 README。
            try:
                obj = json.loads(meta.read_text(encoding="utf-8"))
                desc = str(obj.get("desc") or "")
            except (OSError, ValueError):
                desc = ""
        readme = d / "README.md"
        if readme.exists():
            try:
                lines = readme.read_text(encoding="utf-8").splitlines()
            except (OSError, ValueError):
                lines = []
            # 标题行之后的第一段非空文字（去掉标题本身）
            for line in lines[1:]:
                stripped = line.strip().lstrip("#").strip()
                if stripped:
                    desc = stripped
                    break
        out.append({
            "name": d.name,
            "version": f"{d.name}@0.1",
            "desc": desc or "一个可以拿来测的 Agent（README 没写简介）。",
        })
    return out
