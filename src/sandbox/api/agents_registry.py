# -*- coding: utf-8 -*-
"""用户自己的 Agent 注册表（「导入我的 Agent」）。

导入 = 在 `agents/<名字>/` 下写一份 `AGENT.json`，指向用户本机的 Agent 目录
和启动方式。运行器（`scenario/runner.py`）拉起时先读这份注册：目录、入口文件
和 Python 命令都以注册为准，而不是仓库里的 `agents/<名字>/`。

`AGENT.json` 的字段：

    name        Agent 名字（= 场景里 `agent_version` 的 `@` 之前那一段）
    desc        一句话描述（显示在测试任务和自然语言录入里）
    source_dir  Agent 所在的绝对路径
    entry       入口文件（相对 source_dir，默认 agent.py）
    python      Python 命令；留空 = 自动找 source_dir 下的 .venv，找不到用系统 python
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from sandbox.config import REPO_ROOT

AGENTS_ROOT: Path = REPO_ROOT / "agents"
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class RegistryError(Exception):
    """导入失败，消息直接给用户看。"""


def agent_dir(name: str) -> Path:
    return AGENTS_ROOT / name


def list_registered() -> list[dict[str, str]]:
    """agents/ 下所有带 AGENT.json 的目录。"""
    out: list[dict[str, str]] = []
    if not AGENTS_ROOT.exists():
        return out
    for d in sorted((p for p in AGENTS_ROOT.iterdir() if p.is_dir()), key=lambda x: x.name):
        meta = d / "AGENT.json"
        if not meta.exists():
            continue
        try:
            obj = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            obj = {}
        out.append({
            "name": str(obj.get("name") or d.name),
            "desc": str(obj.get("desc") or ""),
            "source_dir": str(obj.get("source_dir") or ""),
            "entry": str(obj.get("entry") or "agent.py"),
            "python": str(obj.get("python") or ""),
        })
    return out


def register(*, name: str, desc: str, source_dir: str,
             entry: str = "agent.py", python: str = "") -> str:
    """登记一个 Agent，返回规范化后的名字。"""
    name = name.strip()
    if not _NAME_RE.match(name):
        raise RegistryError("名字只能用小写字母、数字、连字符或下划线，且以字母或数字开头。")
    src = Path(source_dir.strip())
    if not src.is_dir():
        raise RegistryError(f"目录不存在或不是文件夹：{source_dir}")
    meta = {
        "name": name,
        "desc": desc.strip(),
        "source_dir": str(src.resolve()),
        "entry": (entry.strip() or "agent.py"),
        "python": python.strip(),
    }
    d = agent_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "AGENT.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return name


def unregister(name: str) -> None:
    """删掉注册。只动 AGENT.json；目录里如果还有别的东西就不碰。"""
    if not _NAME_RE.match(name):
        return
    d = agent_dir(name)
    meta = d / "AGENT.json"
    if meta.exists():
        meta.unlink()
    try:
        d.rmdir()
    except OSError:
        pass
