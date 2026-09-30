"""场景文件的加载与校验。

校验里有两条是**跨字段**的，单看任何一个字段都发现不了，但它们出错的方式都很安静：

1. `restore_from` 引用了一张**后面**才拍的快照。这意味着"回到未来"，
   记忆视野会算出一个没意义的数，而归因完全建立在视野上。必须报错，
   不能让它跑起来——跑起来的结果看起来是正常的，只是结论悄悄反了。

2. 脚本里调用了**没声明**的工具。多半是打错字。不报错的话，Agent 会去调一个
   它自己没有的工具，ReAct 循环直接报错退出，而报告会显示"没有发现问题"。

第二条的取舍值得说明：确实存在"故意让 Agent 调用未声明工具来测出口代理"的场景，
但那种场景应当显式声明该工具再把它放到 allowlist 之外，而不是靠拼错名字。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from sandbox.scenario.spec import Scenario

SCENARIO_SUFFIXES = (".yaml", ".yml", ".json")


class ScenarioError(ValueError):
    """场景文件不合法。消息里要带上文件名与位置，否则用户无从下手。"""


def load_scenario(path: str | Path) -> Scenario:
    """从文件加载一个场景。"""
    path = Path(path)
    if not path.exists():
        raise ScenarioError(f"场景文件不存在：{path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ScenarioError(f"{path} 不是合法的 YAML：{exc}") from exc

    if not isinstance(raw, dict):
        raise ScenarioError(f"{path} 的顶层必须是一个映射（mapping），实际是 {type(raw).__name__}")

    try:
        scenario = Scenario.model_validate(raw)
    except ValidationError as exc:
        raise ScenarioError(f"{path} 场景定义不合法：\n{exc}") from exc

    _check_references(scenario, path)
    return scenario


def _check_references(scenario: Scenario, path: Path) -> None:
    tool_names = set(scenario.tool_index())

    # --- 快照引用必须指向**之前**拍下的快照 ---
    defined: set[str] = set()
    for index, session in enumerate(scenario.sessions):
        if session.restore_from is not None and session.restore_from not in defined:
            raise ScenarioError(
                f"{path}: 会话 #{index}（{session.name}）要恢复到快照 "
                f"{session.restore_from!r}，但那个快照此时还不存在。\n"
                f"到这一步为止可用的快照：{sorted(defined) or '（无）'}\n"
                f"恢复到**之后**才拍的快照等于让记忆回到未来，"
                f"记忆视野会算错，而归因完全建立在视野上。"
            )
        # 本会话拍下的快照，供**后面的**会话引用。所以校验在循环开头、
        # 登记在循环末尾——一个会话只能恢复到它之前拍下的状态。
        for name in (session.snapshot_before, session.snapshot_after):
            if name:
                defined.add(name)

    # --- 脚本只能调用声明过的工具 ---
    for index, session in enumerate(scenario.sessions):
        for step_no, step in enumerate(session.script):
            for call in step.get("tool_calls", []) or []:
                name = call.get("name")
                if name not in tool_names:
                    raise ScenarioError(
                        f"{path}: 会话 #{index}（{session.name}）第 {step_no} 步调用了 "
                        f"未声明的工具 {name!r}。\n"
                        f"已声明的工具：{sorted(tool_names) or '（无）'}\n"
                        f"若确实要测「调用未声明工具被代理拦下」，"
                        f"请把该工具声明出来并排除在放行名单之外。"
                    )


def load_all(directory: str | Path) -> list[Scenario]:
    """加载一个目录下的全部场景，按文件名排序。"""
    directory = Path(directory)
    if not directory.is_dir():
        raise ScenarioError(f"场景目录不存在：{directory}")
    paths = sorted(
        p for p in directory.iterdir() if p.suffix.lower() in SCENARIO_SUFFIXES
    )
    return [load_scenario(p) for p in paths]


def dump_skeleton() -> dict[str, Any]:
    """一份可复制的最小场景骨架。`--init` 用。"""
    return {
        "id": "my-scenario",
        "description": "描述这个场景要验证什么行为。",
        "agent": {
            "version": "react_demo@0.1",
            "task": "帮我把订单 A1 处理掉。",
            "model": "stub-model",
        },
        "tools": [
            {
                "name": "get_order",
                "description": "按订单号查询订单详情。只读。",
                "is_write": False,
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            }
        ],
        "sessions": [
            {
                "name": "第一次",
                "script": [
                    {"tool_calls": [{"name": "get_order", "arguments": {"order_id": "A1"}}]},
                    {"content": "完成了。"},
                ],
            }
        ],
        "expect": [
            {"detector": "idempotency", "verdict": "pass", "note": "这个场景不该报警"}
        ],
    }
