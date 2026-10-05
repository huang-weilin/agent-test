# -*- coding: utf-8 -*-
"""用户自己的 LLM 连接（界面里配的 key）。

对发布出去给别人下载的产品来说，「配好就能用」的默认值是不存在的——
每个用户的模型、密钥都是他自己的。这一层把界面里填的连接存到
`.agent_sandbox_local.json`（仓库根，已 gitignore），让「配 key → 选 agent →
大白话建测试」的第一步有个真实的落点。

**生效优先级**：用户界面里配的连接（本文件）> 仓库 `.env` 的
`SANDBOX_UPSTREAM_*`（开发环境内置）。两层都没有时，自然语言录入降级到
本地关键词规则，并在页面上明说「这是草稿」。

**安全边界**：这个 key 只用于「把大白话翻译成场景」这一步，跑在界面进程里。
它**绝不**被注入给被测 Agent 的进程——被测 Agent 在沙盒里拿到的永远是假凭据
（这是 `gateway` 的 Credential Vault 设计）。翻译和被测是两条路。
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any

from sandbox import config
from sandbox.config import REPO_ROOT

FILE = REPO_ROOT / ".agent_sandbox_local.json"
_LOCK = threading.Lock()

#: 没配模型名时用的默认值（DeepSeek 的 OpenAI 兼容接口）。
_DEFAULT_MODEL = "deepseek-chat"


def load() -> dict[str, Any] | None:
    """读用户配置。文件不存在或坏了就返回 None（不会炸页面）。"""
    with _LOCK:
        try:
            raw = FILE.read_text(encoding="utf-8")
        except (OSError, ValueError):
            return None
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    api_key = str(obj.get("api_key") or "").strip()
    base_url = str(obj.get("base_url") or "").strip()
    if not api_key or not base_url:
        return None
    model = str(obj.get("model") or "").strip() or _DEFAULT_MODEL
    return {"base_url": base_url, "api_key": api_key, "model": model}


def save(base_url: str, api_key: str, model: str) -> None:
    """写入用户配置。三个字段都去空白，api_key 只存不打印。"""
    base_url = base_url.strip().rstrip("/")
    api_key = api_key.strip()
    model = model.strip() or _DEFAULT_MODEL
    payload = {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with _LOCK:
        FILE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def clear() -> None:
    """删掉用户配置，回到 .env 的内置连接（如果有）。"""
    with _LOCK:
        try:
            FILE.unlink()
        except OSError:
            pass


def effective() -> dict[str, str] | None:
    """真正会拿去调解析模型的连接。用户配置优先，退回 .env 内置。"""
    user = load()
    if user:
        return {
            "base_url": user["base_url"],
            "api_key": user["api_key"],
            "model": user["model"],
            "source": "user",
        }
    if config.settings.upstream_api_key:
        return {
            "base_url": config.settings.upstream_base_url,
            "api_key": config.settings.upstream_api_key,
            "model": _DEFAULT_MODEL,
            "source": "builtin",
        }
    return None


def ready() -> bool:
    return effective() is not None


def status_text() -> str:
    """给人看的连接状态（页面顶部那行小字）。"""
    conn = effective()
    if not conn:
        return "还没连上解析用的模型。没连也能用，只是机器会用内置规则先猜一版草稿，你确认时看仔细点。"
    model = conn.get("model") or _DEFAULT_MODEL
    base = conn.get("base_url") or ""
    host = base.split("//")[-1].split("/")[0] if "//" in base else base
    return f"已连上 {model}（{host}）。它只负责把大白话翻成测试，不会碰被测 Agent。"
