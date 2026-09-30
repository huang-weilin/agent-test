"""被测 Agent 的记忆模块。

**这里的加载方式和别的测试不一样，是刻意的。** `agents/react_demo/memory.py`
不属于 `sandbox` 包——它属于被测 Agent，跑在另一个 venv 里，物理上看不到沙箱。
测试直接按文件路径把它拉进来加载，就是为了不为了测试的方便去动它的这一性质。
一旦哪天有人把它挪进 `src/sandbox/`，这些测试会一起红。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

MEMORY_PY = Path(__file__).resolve().parents[1] / "agents" / "react_demo" / "memory.py"


def _load_memory_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_agent_memory_under_test", MEMORY_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # 注册进 sys.modules 再 exec：dataclass / 循环导入之类的东西依赖它。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def memory() -> ModuleType:
    return _load_memory_module()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "memory"


# --- 路径 ---------------------------------------------------------------------


def test_memory_root_follows_the_sandbox_env_var(
    memory: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """记忆目录由沙箱指定，Agent 不自己猜。

    猜错了，回滚就会回滚到一处它根本不读的地方，报告于是显示"回滚过了它还是
    重复扣款"——把 Agent 冤枉成不读记忆。
    """
    target = tmp_path / "somewhere"
    monkeypatch.setenv("SANDBOX_AGENT_MEMORY_DIR", str(target))
    assert memory.memory_root() == target


def test_memory_root_falls_back_when_unset(
    memory: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 单独手动跑这个 Agent 时没有沙箱注入环境变量，得能起来。
    monkeypatch.delenv("SANDBOX_AGENT_MEMORY_DIR", raising=False)
    assert memory.memory_root() == Path("./.agent-memory")


# --- 读 -----------------------------------------------------------------------


def test_a_missing_file_means_nothing_was_done(memory: ModuleType, root: Path) -> None:
    # 第一次醒来就是这个状态。空记忆不是错误。
    assert memory.load(root) == []


def test_blank_lines_are_skipped(memory: ModuleType, root: Path) -> None:
    root.mkdir(parents=True)
    (root / memory.MEMORY_FILE).write_text(
        '\n{"key": "a", "result": null}\n\n\n{"key": "b", "result": null}\n',
        encoding="utf-8",
    )
    assert [e["key"] for e in memory.load(root)] == ["a", "b"]


def test_a_corrupt_line_does_not_take_the_agent_down(memory: ModuleType, root: Path) -> None:
    """写了一半的记录跳过，不抛。

    抛的话，"记忆里有一条坏数据"就升级成了"Agent 完全起不来"——修一个小问题
    要付大代价。这里只承诺"不崩"，不假装能修复：跳过的行之后会被新的写入挤掉
    位置关系。
    """
    root.mkdir(parents=True)
    (root / memory.MEMORY_FILE).write_text(
        '{"key": "good1", "result": null}\n'
        '{"key": "truncated", "resu\n'  # 崩在写一半
        'not json at all\n'
        '{"key": "good2", "result": null}\n',
        encoding="utf-8",
    )
    assert [e["key"] for e in memory.load(root)] == ["good1", "good2"]


# --- 写 -----------------------------------------------------------------------


def test_record_then_load_round_trips(memory: ModuleType, root: Path) -> None:
    memory.record("create_payment:A1", {"charge_id": "ch_1"}, root)
    assert memory.load(root) == [
        {"key": "create_payment:A1", "result": {"charge_id": "ch_1"}}
    ]


def test_record_creates_the_directory(memory: ModuleType, root: Path) -> None:
    # 沙箱会先建好目录，但单独跑这个 Agent 时不一定有。
    assert not root.exists()
    memory.record("k", None, root)
    assert (root / memory.MEMORY_FILE).exists()


def test_records_are_append_only(memory: ModuleType, root: Path) -> None:
    # 记忆是"做过什么"的日志，不是"现在是什么状态"的快照。
    # 覆盖写会让回滚后的差异变得看不出来。
    memory.record("a", 1, root)
    memory.record("b", 2, root)
    assert [e["key"] for e in memory.load(root)] == ["a", "b"]


def test_non_ascii_survives_the_round_trip(memory: ModuleType, root: Path) -> None:
    memory.record("支付:A1", "成功", root)
    assert memory.load(root)[0]["result"] == "成功"


# --- 可复现性 -----------------------------------------------------------------


def test_the_same_history_produces_identical_bytes(
    memory: ModuleType, root: Path, tmp_path: Path
) -> None:
    """**没有时间戳**，所以同一段经历每次产出同样的字节。

    这是快照能被内容寻址的前提。带上时间戳的话，两次内容完全相同的恢复会得到
    不同的哈希，于是"这次恢复其实什么都没变"就判断不出来了——而那恰恰是跨会话
    实验里要分辨的东西。
    """
    other = tmp_path / "memory2"
    for target in (root, other):
        memory.record("create_payment:A1", {"charge_id": "ch_1"}, target)
        memory.record("create_payment:A2", {"charge_id": "ch_2"}, target)

    assert (root / memory.MEMORY_FILE).read_bytes() == (other / memory.MEMORY_FILE).read_bytes()


def test_the_key_comes_first_so_the_file_stays_greppable(
    memory: ModuleType, root: Path
) -> None:
    memory.record("create_payment:A1", {"charge_id": "ch_1"}, root)
    line = (root / memory.MEMORY_FILE).read_text(encoding="utf-8").strip()
    assert json.loads(line).keys() == {"key", "result"}
    assert line.index('"key"') < line.index('"result"')


# --- 查询 ---------------------------------------------------------------------


def test_already_done_is_false_before_and_true_after(memory: ModuleType, root: Path) -> None:
    assert not memory.already_done("create_payment:A1", root)
    memory.record("create_payment:A1", None, root)
    assert memory.already_done("create_payment:A1", root)


def test_already_done_matches_the_whole_key_not_a_prefix(
    memory: ModuleType, root: Path
) -> None:
    """键是整体比对的。

    子串匹配会让 `create_payment:A1` 被 `create_payment:A10` 命中——一张订单
    付过款，另一张就永远不敢付了，而报告只会显示"钱少扣了一笔"。
    """
    memory.record("create_payment:A10", None, root)
    assert not memory.already_done("create_payment:A1", root)
    assert memory.already_done("create_payment:A10", root)


def test_a_missing_key_field_is_not_a_match(memory: ModuleType, root: Path) -> None:
    # 损坏或旧格式的条目不该把任何查询判成"做过"。
    root.mkdir(parents=True)
    (root / memory.MEMORY_FILE).write_text('{"result": 1}\n', encoding="utf-8")
    assert not memory.already_done("anything", root)
