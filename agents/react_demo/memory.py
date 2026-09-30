"""被测 Agent 的记忆：一个文件型记忆目录。

**这个文件只用标准库。** 它属于被测 Agent，不属于沙箱，所以不能依赖任何
框架侧的包；同时它也不 import `sandbox.*`——Agent 在物理上看不到沙箱实现，
这一点由独立 venv 保证。

记忆就是"我做过哪些有副作用的操作"，一行一条，追加写 JSONL：

    {"key": "create_payment:A1", "result": {"charge_id": "ch_..."}}

**刻意不写时间戳。** 时间戳会让同一段经历每次产出不同的字节，于是内容寻址
失效、"这次恢复其实什么都没变"就判断不出来了。真实 Agent 通常想要时间戳，
那是它自己的取舍；演示 Agent 选可复现。

这个模块的定位是"参考实现 + 可拆掉的零件"：真实的被测 Agent 有自己的记忆
方案（LangGraph checkpointer、向量库、业务数据库……），沙箱不关心是哪种，
只要求它落在 `SANDBOX_AGENT_MEMORY_DIR` 指向的目录里——因为回滚要靠它。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

MEMORY_FILE = "actions.jsonl"


def memory_root() -> Path:
    """记忆目录。由沙箱创建并通过环境变量交给 Agent。

    Agent 不自己猜路径：猜错了，测试框架的回滚就会回滚到一处它根本不读的地方，
    而报告会显示"回滚过了它还是重复扣款"——把 Agent 冤枉成不读记忆。
    """
    return Path(os.environ.get("SANDBOX_AGENT_MEMORY_DIR", "./.agent-memory"))


def load(root: Path | None = None) -> list[dict[str, Any]]:
    """读出全部记忆条目。文件不存在 = 什么都没做过，不是错误。

    损坏的行**跳过而不是抛错**：一条写了一半的记录不该让整个 Agent 起不来，
    那会把"记忆里有一条坏数据"放大成"Agent 完全不可用"。跳过的行会被之后的
    写入覆盖掉位置关系，所以这里只做到"不崩"，不假装能修复。
    """
    path = (root or memory_root()) / MEMORY_FILE
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except ValueError:
            continue
    return entries


def already_done(key: str, root: Path | None = None) -> bool:
    """这个操作以前成功做过吗。"""
    return any(entry.get("key") == key for entry in load(root))


def record(key: str, result: Any, root: Path | None = None) -> None:
    """记下"这个操作做成了"。

    **调用时机是这里唯一的难点。** 现在是在副作用返回成功之后写。这中间有个
    窗口：调用成功了但还没写进记忆就崩了——那笔操作就被忘掉了，下次会重做。
    反过来先写后调用，则会在调用失败时误记成做过了。

    这是经典的 dual-write 问题，没有免费的解。演示 Agent 选"后写"，因为对
    "重复扣款"这个被测缺陷来说，后写偏保守（宁可重做也不误记）。真实系统要用
    事务、幂等键或 outbox 来解——**而那正是被测对象的一部分，沙箱不该替它决定。**
    """
    path = (root or memory_root()) / MEMORY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "result": result}, ensure_ascii=False) + "\n")
