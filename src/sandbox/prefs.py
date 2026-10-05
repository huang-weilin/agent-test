# -*- coding: utf-8 -*-
"""界面偏好（简单版 / 专业版）。

存到仓库根的 `.agent_sandbox_prefs.json`（已 gitignore，不会随项目发出去）。

**为什么单独一个文件**：连接设置（`user_llm`）存的是密钥，属于敏感配置；
界面模式只是显示偏好，两者生命周期和敏感度都不同，不该共用一份文件。
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

from sandbox.config import REPO_ROOT

FILE = REPO_ROOT / ".agent_sandbox_prefs.json"
_LOCK = threading.Lock()

_MODES = ("simple", "pro")


def load_mode() -> str:
    """当前界面模式：`simple`（默认，给不懂技术的人）或 `pro`（给工程师）。

    文件不存在、坏了、或字段非法时一律回到 `simple`——新产品默认是简单版，
    这是产品定位的一部分，不是可选项。
    """
    with _LOCK:
        try:
            raw = FILE.read_text(encoding="utf-8")
        except (OSError, ValueError):
            return "simple"
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return "simple"
    if not isinstance(obj, dict):
        return "simple"
    mode = str(obj.get("ui_mode") or "").strip()
    return mode if mode in _MODES else "simple"


def save_mode(mode: str) -> str:
    """写入界面模式。非法值被归一化到 `simple`，返回实际生效的值。"""
    mode = mode if mode in _MODES else "simple"
    payload = {
        "ui_mode": mode,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with _LOCK:
        FILE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return mode
