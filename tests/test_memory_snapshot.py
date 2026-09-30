"""记忆快照的测试。

这个模块的重点不是"功能对不对"，而是**破坏性操作有没有被闸住**。恢复快照会
清空并重建 Agent 的记忆目录，闸门失效的代价是删掉不该删的东西，而那种错误
在测试里往往表现为"一跑就过"——因为目录本来就空的。

所以这里大量篇幅在测**拒绝**：路径越界要拒绝、适配器不匹配要拒绝、内容损坏要
拒绝、路径逃逸要拒绝。每一条都对应一种真实的删库方式。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sandbox.config import settings
from sandbox.memory.adapters.file_memory import FileMemoryAdapter, RestoreRefused
from sandbox.memory.snapshot import SnapshotStore, digest_of
from sandbox.trace.reader import load_call_views, load_session_sequence, load_sessions
from sandbox.trace.recorder import (
    capture_snapshot,
    horizon_of,
    restore_snapshot,
    start_run,
    start_session,
)

# --- 适配器：capture / restore ------------------------------------------------


@pytest.fixture
def memory_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把记忆根指到临时目录。

    必须指走：`memory_restore_guard` 校验的是路径在不在 `settings.memory_root`
    之下。不指走的话，测试要么因为越界被拒（测了个寂寞），要么真去动
    data/memory（污染开发数据）。
    """
    root = tmp_path / "mem"
    root.mkdir()
    monkeypatch.setattr(settings, "memory_root", root)
    return root


def _seed(directory: Path, entries: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in entries.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_capture_is_deterministic(memory_root: Path) -> None:
    """同一份记忆两次 capture 必须产出同样的字节。

    内容寻址全靠这条。不确定的话，摘要就会漂，于是"恢复到的是不是同一个状态"
    判断不出来，"这次恢复其实什么都没改"也会被当成一次真实的故障注入。
    """
    mem = memory_root / "agent"
    _seed(mem, {"facts.md": "用户偏好中文", "history/1.txt": "订了 A1"})
    adapter = FileMemoryAdapter(mem)

    assert adapter.capture() == adapter.capture()
    assert digest_of(adapter.capture()) == digest_of(adapter.capture())


def test_capture_then_restore_is_identity(memory_root: Path) -> None:
    """restore(capture()) 必须是恒等操作。

    破坏这条的典型方式很隐蔽：capture 时漏掉某种文件，restore 就把它们删了。
    所以这里放了一个嵌套目录和一个空文件——两种容易在遍历里被漏掉的形态。
    """
    mem = memory_root / "agent"
    _seed(mem, {"facts.md": "用户偏好中文", "history/1.txt": "订了 A1", "empty": ""})
    adapter = FileMemoryAdapter(mem)

    before = adapter.capture()
    adapter.restore(before)

    assert adapter.capture() == before


def test_restore_brings_back_deleted_content(memory_root: Path) -> None:
    """恢复要能把删除和新增都抹平——这才是"回滚到周三之前"。"""
    mem = memory_root / "agent"
    _seed(mem, {"facts.md": "只有周一的事"})
    adapter = FileMemoryAdapter(mem)
    snapshot = adapter.capture()

    (mem / "facts.md").write_text("周一和周三的事", encoding="utf-8")
    (mem / "new.md").write_text("周三新增", encoding="utf-8")
    adapter.restore(snapshot)

    assert (mem / "facts.md").read_text(encoding="utf-8") == "只有周一的事"
    assert not (mem / "new.md").exists()


def test_missing_directory_captures_as_empty_not_error(memory_root: Path) -> None:
    """还没建过记忆目录是合法状态（新 Agent 第一次跑），不是错误。

    当成错误的话，会话边界的第一次快照就会失败——而那次快照正是用来做基线的。
    """
    adapter = FileMemoryAdapter(memory_root / "not-yet")
    payload = adapter.capture()
    assert json.loads(payload)["entries"] == []
    assert adapter.entry_count() == 0


def test_restore_into_missing_dir_is_refused_without_explicit_consent(memory_root: Path) -> None:
    adapter = FileMemoryAdapter(memory_root / "agent")
    payload = FileMemoryAdapter(memory_root / "agent").capture()

    with pytest.raises(RestoreRefused, match="不存在"):
        adapter.restore(payload)


def test_restore_outside_memory_root_is_refused(memory_root: Path, tmp_path: Path) -> None:
    """路径写错时必须拒绝，而不是把那个目录清空。"""
    outsider = tmp_path / "somewhere-else"
    _seed(outsider, {"important.txt": "别删我"})
    payload = FileMemoryAdapter(outsider).capture()

    adapter = FileMemoryAdapter(outsider)
    with pytest.raises(RestoreRefused, match="memory_restore_guard"):
        adapter.restore(payload)
    assert (outsider / "important.txt").exists(), "拒绝之后原目录必须原封不动"


def test_guard_can_be_turned_off_deliberately(
    memory_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """闸门是默认值，不是硬编码——测真实 Agent 时它的记忆目录不在我们管辖内。"""
    outsider = tmp_path / "real-agent-memory"
    _seed(outsider, {"a.md": "原始"})
    adapter = FileMemoryAdapter(outsider)
    snapshot = adapter.capture()

    (outsider / "a.md").write_text("被改过", encoding="utf-8")
    monkeypatch.setattr(settings, "memory_restore_guard", False)
    adapter.restore(snapshot)

    assert (outsider / "a.md").read_text(encoding="utf-8") == "原始"


def test_restore_refuses_adapter_mismatch(memory_root: Path) -> None:
    """拿 A 格式的字节去覆盖 B 格式的存储会把记忆砸坏。"""
    mem = memory_root / "agent"
    _seed(mem, {"a.md": "x"})
    payload = FileMemoryAdapter(mem).capture()
    doc = json.loads(payload)
    doc["adapter"] = "langgraph_checkpointer"

    with pytest.raises(RestoreRefused, match="适配器"):
        FileMemoryAdapter(mem).restore(json.dumps(doc).encode())


def test_restore_refuses_format_version_drift(memory_root: Path) -> None:
    mem = memory_root / "agent"
    _seed(mem, {"a.md": "x"})
    doc = json.loads(FileMemoryAdapter(mem).capture())
    doc["version"] = 99

    with pytest.raises(RestoreRefused, match="格式版本"):
        FileMemoryAdapter(mem).restore(json.dumps(doc).encode())


def test_restore_refuses_corrupted_entry(memory_root: Path) -> None:
    """单个条目内容与摘要不符就整体拒绝。

    部分写入比整体失败严重得多：Agent 的记忆会变成一半新一半旧的混合态，
    而那种状态在真实系统里根本不存在，据此测出的结论没有意义。
    """
    mem = memory_root / "agent"
    _seed(mem, {"a.md": "原始内容"})
    doc = json.loads(FileMemoryAdapter(mem).capture())
    doc["entries"][0]["content_b64"] = "5pS55LqG"  # "改了"

    with pytest.raises(RestoreRefused, match="摘要不符"):
        FileMemoryAdapter(mem).restore(json.dumps(doc).encode())


def test_restore_refuses_path_escape(memory_root: Path, tmp_path: Path) -> None:
    """快照里的相对路径带 `..` = 往记忆目录外面写。"""
    mem = memory_root / "agent"
    _seed(mem, {"a.md": "x"})
    doc = json.loads(FileMemoryAdapter(mem).capture())
    doc["entries"][0]["path"] = "../escaped.md"

    with pytest.raises(RestoreRefused, match="之外"):
        FileMemoryAdapter(mem).restore(json.dumps(doc).encode())
    assert not (memory_root / "escaped.md").exists()


def test_restore_refuses_non_json_payload(memory_root: Path) -> None:
    mem = memory_root / "agent"
    _seed(mem, {"a.md": "x"})
    with pytest.raises(RestoreRefused, match="不是合法"):
        FileMemoryAdapter(mem).restore(b"\x00\x01\x02 not json")


def test_symlinked_directories_are_not_walked_into(memory_root: Path, tmp_path: Path) -> None:
    """链接**目录**不能被走进去。

    这是写这个测试时发现的一个真漏洞：早先的实现是 `rglob("*")` 再逐个查
    `is_symlink()`，那只挡得住链接**文件**。rglob 会先走进链接目录，再把它
    里面的普通文件列出来——而那些文件本身不是链接，逐个检查根本查不出来。
    结果就是把目录外的一整棵树吸进快照。

    Windows 上普通用户建不了符号链接（要 SeCreateSymbolicLinkPrivilege），
    但**目录联接（junction）不需要提权**，所以这条路在 Windows 上也测得到。
    """
    outside = tmp_path / "outside-tree"
    _seed(outside, {"secret.md": "真实上游密钥不该进快照"})

    mem = memory_root / "agent"
    _seed(mem, {"a.md": "x"})
    link = mem / "linked-dir"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        # 退到 junction。Windows 上 mklink /J 不需要管理员权限。
        import subprocess

        # errors="replace"：mklink 的输出走 Windows 控制台代码页（中文机器上
        # 是 GBK），按 UTF-8 硬解会在 reading 线程里抛 UnicodeDecodeError。
        # 那个异常不影响 returncode，但会让每一次测试运行都挂一条告警——
        # 一条永远在的告警会训练人忽略告警。
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
            capture_output=True,
            text=True,
            errors="replace",
        )
        if result.returncode != 0:
            pytest.skip(f"本机既建不了符号链接也建不了 junction：{result.stderr.strip()}")

    payload = FileMemoryAdapter(mem).capture()
    paths = [e["path"] for e in json.loads(payload)["entries"]]
    assert paths == ["a.md"], f"链接目录被走进去了，快照里出现了 {paths}"
    assert not any("secret" in p for p in paths)


def test_symlinked_files_are_not_captured(memory_root: Path, tmp_path: Path) -> None:
    """链接文件同样不存。少一个链接，好过把边界捅穿。"""
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("真实上游密钥不该进快照", encoding="utf-8")

    mem = memory_root / "agent"
    _seed(mem, {"a.md": "x"})
    try:
        (mem / "link.txt").symlink_to(outside)
    except OSError:
        pytest.skip("本机不支持创建文件符号链接（Windows 需要提权或开发者模式）")

    payload = FileMemoryAdapter(mem).capture()
    paths = [e["path"] for e in json.loads(payload)["entries"]]
    assert paths == ["a.md"]


# --- 快照存储 -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_content_is_stored_once(temp_db: str, tmp_path: Path) -> None:
    """内容寻址：同一份记忆存两次只占一个文件，但元数据是两行。"""
    store = SnapshotStore(tmp_path / "snaps")
    payload = b'{"a":1}'

    first = await store.save(
        session_sequence_id="seq-1",
        taken_at_ordinal=0,
        taken_at_seq=1,
        adapter="hermes_file",
        payload=payload,
    )
    second = await store.save(
        session_sequence_id="seq-1",
        taken_at_ordinal=0,
        taken_at_seq=1,
        adapter="hermes_file",
        payload=payload,
    )

    assert first.digest == second.digest
    assert first.payload_path == second.payload_path
    assert first.id != second.id
    assert len(list((tmp_path / "snaps").rglob("*.bin"))) == 1


@pytest.mark.asyncio
async def test_load_detects_corruption(temp_db: str, tmp_path: Path) -> None:
    """内容文件被改动过就拒绝返回——拿截断的记忆去恢复比恢复失败严重得多。"""
    store = SnapshotStore(tmp_path / "snaps")
    ref = await store.save(
        session_sequence_id="seq-1",
        taken_at_ordinal=0,
        taken_at_seq=1,
        adapter="hermes_file",
        payload=b"original",
    )
    Path(ref.payload_path).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="内容损坏"):
        await store.load(ref.id)


@pytest.mark.asyncio
async def test_find_covering_picks_latest_snapshot_before_a_point(
    temp_db: str, tmp_path: Path
) -> None:
    """`find_covering` 是"回到某个时刻之前"的实现——取早于等于该位置的最后一个。"""
    store = SnapshotStore(tmp_path / "snaps")
    for ordinal, seq in [(0, 3), (1, 7), (2, 2)]:
        await store.save(
            session_sequence_id="seq-1",
            taken_at_ordinal=ordinal,
            taken_at_seq=seq,
            adapter="hermes_file",
            payload=f"{ordinal}-{seq}".encode(),
        )

    covering = await store.find_covering("seq-1", (1, 7))
    assert covering is not None
    assert covering.position == (1, 7)

    before_all = await store.find_covering("seq-1", (0, 2))
    assert before_all is None, "没有任何快照早于这个位置时必须返回 None，不能瞎给一个"


@pytest.mark.asyncio
async def test_snapshots_are_scoped_to_their_sequence(temp_db: str, tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "snaps")
    await store.save(
        session_sequence_id="seq-1",
        taken_at_ordinal=0,
        taken_at_seq=1,
        adapter="hermes_file",
        payload=b"a",
    )
    await store.save(
        session_sequence_id="seq-2",
        taken_at_ordinal=0,
        taken_at_seq=1,
        adapter="hermes_file",
        payload=b"b",
    )

    assert len(await store.list_for_sequence("seq-1")) == 1
    assert len(await store.list_for_sequence("seq-2")) == 1
    assert await store.list_for_sequence("seq-3") == []


# --- 视野换算 -----------------------------------------------------------------


def test_horizon_of_boundary_snapshots() -> None:
    """快照的两取法对应两种视野。

    会话 k 开始**之前**取（seq=0）：序号 k 还什么都没发生，视野停在 k-1。
    会话 k 结束**之后**取（seq>0）：整个序号 k 都在记忆里，视野到 k。

    这个换算决定了跨会话归因的每一条结论，所以单独测。
    """
    from sandbox.memory.snapshot import SnapshotRef

    def ref(ordinal: int, seq: int) -> SnapshotRef:
        return SnapshotRef(
            id="s",
            session_sequence_id="seq-1",
            taken_at_ordinal=ordinal,
            taken_at_seq=seq,
            digest="d",
            adapter="hermes_file",
            payload_path="p",
            size_bytes=0,
            entry_count=0,
        )

    assert horizon_of(ref(3, 0)) == 2, "会话 3 开始前 → 只知道 0..2"
    assert horizon_of(ref(3, 12)) == 3, "会话 3 结束后 → 知道 0..3"
    assert horizon_of(ref(0, 0)) == -1, "会话 0 开始前 → 什么都不知道"


# --- 与记录器的联动 -----------------------------------------------------------


@pytest.mark.asyncio
async def test_start_session_records_horizon_from_snapshot(
    temp_db: str, memory_root: Path, tmp_path: Path
) -> None:
    """从快照恢复开的会话，其记忆视野必须等于快照的视野——不能是"自然延续"。"""
    store = SnapshotStore(tmp_path / "snaps")
    adapter = FileMemoryAdapter(memory_root / "agent")

    ctx = await start_run(scenario_id="x", agent_version="v1")
    assert ctx.session_ordinal == 0 and ctx.memory_horizon_ordinal == -1

    baseline = await capture_snapshot(ctx, adapter, store)
    assert baseline.position == (0, 0)

    await start_session(ctx)
    assert ctx.session_ordinal == 1
    assert ctx.memory_horizon_ordinal == 0, "自然延续应当记得会话 0"

    rewind = await start_session(ctx, restored_from=baseline)
    assert ctx.session_ordinal == 2
    assert rewind.memory_horizon_ordinal == -1, "恢复到会话 0 之前 = 什么都不记得"
    assert rewind.restored_from_snapshot_id == baseline.id


@pytest.mark.asyncio
async def test_session_ordinals_must_increase(temp_db: str) -> None:
    """序号回退会让 (ordinal, seq) 排序与真实时间线不符，而归因建立在顺序上。"""
    ctx = await start_run(scenario_id="x", agent_version="v1")
    await start_session(ctx, ordinal=5)
    with pytest.raises(ValueError, match="单调递增"):
        await start_session(ctx, ordinal=5)
    with pytest.raises(ValueError, match="单调递增"):
        await start_session(ctx, ordinal=2)


@pytest.mark.asyncio
async def test_calls_are_attributed_to_the_current_session(temp_db: str) -> None:
    """切换会话后，新调用必须记在新会话下——否则归因全错。"""
    from sandbox.trace.recorder import record_tool_call

    ctx = await start_run(scenario_id="x", agent_version="v1")
    await record_tool_call(ctx, tool_name="get_order", args={"order_id": "A1"})
    first_session = ctx.session_id

    await start_session(ctx)
    await record_tool_call(ctx, tool_name="get_order", args={"order_id": "A2"})

    calls = await load_session_sequence(ctx.session_sequence_id)
    assert [c.session_ordinal for c in calls] == [0, 1]
    assert calls[0].session_id == first_session
    assert calls[1].session_id != first_session

    sessions = await load_sessions(ctx.session_sequence_id)
    assert [s.ordinal for s in sessions] == [0, 1]


@pytest.mark.asyncio
async def test_sequence_orders_by_session_then_run_seq(temp_db: str) -> None:
    """**排序键必须带会话序号。**

    `seq` 只是运行内单调的，跨会话会重复。只按 seq 排的话，第二次会话的第一次
    调用（seq=1）会排到第一次会话的第 50 次调用前面——而跨会话归因完全建立在
    "谁先谁后"上，排错了结论就反了。
    """
    from sandbox.trace.recorder import record_tool_call

    ctx = await start_run(scenario_id="x", agent_version="v1")
    for i in range(5):
        await record_tool_call(ctx, tool_name="get_order", args={"order_id": f"A{i}"})

    await start_session(ctx)
    await record_tool_call(ctx, tool_name="get_order", args={"order_id": "B0"})

    calls = await load_session_sequence(ctx.session_sequence_id)
    assert [c.args["order_id"] for c in calls] == ["A0", "A1", "A2", "A3", "A4", "B0"]


@pytest.mark.asyncio
async def test_load_call_views_still_reads_a_single_run(temp_db: str) -> None:
    """单次运行的读法不能被跨会话改造破坏——它是常规检测的入口。"""
    from sandbox.trace.recorder import record_tool_call

    ctx = await start_run(scenario_id="x", agent_version="v1")
    await record_tool_call(ctx, tool_name="create_payment", args={"order_id": "A1"})
    await record_tool_call(ctx, tool_name="create_payment", args={"order_id": "A1"})

    calls = await load_call_views(ctx.run_id)
    assert len(calls) == 2
    assert all(c.session_sequence_id == ctx.session_sequence_id for c in calls)


@pytest.mark.asyncio
async def test_restore_snapshot_end_to_end(
    temp_db: str, memory_root: Path, tmp_path: Path
) -> None:
    """从记录器到适配器的完整一轮：拍快照 → 改记忆 → 恢复 → 回到原状。"""
    store = SnapshotStore(tmp_path / "snaps")
    mem = memory_root / "agent"
    _seed(mem, {"facts.md": "周一：订了票 A1"})
    adapter = FileMemoryAdapter(mem)

    ctx = await start_run(scenario_id="x", agent_version="v1")
    ref = await capture_snapshot(ctx, adapter, store)

    (mem / "facts.md").write_text("周三：又订了票 A1", encoding="utf-8")
    (mem / "extra.md").write_text("周三新增", encoding="utf-8")

    await restore_snapshot(ref, adapter, store)

    assert (mem / "facts.md").read_text(encoding="utf-8") == "周一：订了票 A1"
    assert not (mem / "extra.md").exists()
    assert digest_of(adapter.capture()) == ref.digest, "恢复后内容摘要必须与快照一致"
