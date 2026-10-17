"""被测 Agent 的记忆模块。

**这里的加载方式和别的测试不一样，是刻意的。** `agents/order_agent/memory.py`
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

MEMORY_PY = Path(__file__).resolve().parents[1] / "agents" / "order_agent" / "memory.py"


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


# --- 整理 ---------------------------------------------------------------------
#
# 这个动作是**被测对象侧**的，不是沙箱的：真实 Agent 整理记忆时留不留原件是它
# 自己的取舍。演示 Agent 把两种取舍做成一个开关，为了让记忆检测里"声明的路径
# 不在了"那两条分支有活样本——`memory_tidy_archived` / `memory_tidy_pruned`。


def test_archive_moves_the_file_without_touching_a_byte(
    memory: ModuleType, root: Path
) -> None:
    """`archive` 之后根目录空了，但内容一个字节都没变——只是换了文件名。

    这一条是 `memory_moved` 那条分支的地基：检测器判"内容还在别处"判的就是
    逐字节相等（`_moved_to` 比的是内容），所以整理必须原样搬运，不能顺手改写。
    """
    memory.record("create_payment:A1", {"charge_id": "ch_1"}, root)
    before = (root / memory.MEMORY_FILE).read_bytes()

    done = memory.tidy("archive", root)

    assert done == [f"{memory.MEMORY_FILE} → archive/{memory.MEMORY_FILE}"]
    assert not (root / memory.MEMORY_FILE).exists(), "原件该不在原位了"
    assert (root / "archive" / memory.MEMORY_FILE).read_bytes() == before


def test_prune_leaves_nothing_behind(memory: ModuleType, root: Path) -> None:
    """`prune` 是把记忆真的丢掉——包括之前归档进去的那些。

    只删根目录的话，"删干净"就名不副实：内容还躺在 archive/ 里，检测器会
    把它找出来报成"内容换了地方"（warning），而这一支要演的恰恰是"找不到"。
    """
    memory.record("create_payment:A1", {"charge_id": "ch_1"}, root)
    memory.tidy("archive", root)
    memory.record("create_payment:A2", {"charge_id": "ch_2"}, root)
    assert (root / "archive").exists()

    done = memory.tidy("prune", root)

    assert sorted(done) == ["actions.jsonl 已删除", "archive/actions.jsonl 已删除"]
    assert list(root.rglob("*")) == [], "记忆目录该是空的"


def test_tidying_an_empty_memory_is_a_no_op(memory: ModuleType, root: Path) -> None:
    """第一次醒来就是这个状态：还没建过记忆目录。整理不该因此报错。

    它不是"没事找事"——`memory_tidy_*` 场景的第一段会话就会走到这里（那时候
    确实什么都还没有），Agent 崩在这儿的话整个场景跑不起来。
    """
    assert memory.tidy("archive", root) == []
    assert memory.tidy("prune", root) == []


def test_an_unknown_mode_is_refused_loudly(memory: ModuleType, root: Path) -> None:
    """写错的模式名要吵。

    安静地不整理的话，场景会以"预期未达成"收场，而报告上看不出这是配置写错了
    还是 Agent 这次没做那件事——这正是这个项目反复要防的那类结论。
    """
    memory.record("a", 1, root)
    with pytest.raises(ValueError, match="archive / prune"):
        memory.tidy("archvie", root)  # 手滑
    assert (root / memory.MEMORY_FILE).exists(), "报错了就不该已经动过文件"
