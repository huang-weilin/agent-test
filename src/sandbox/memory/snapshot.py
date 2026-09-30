"""记忆快照：存储与检索。

**先把这个模块的定位说清楚，因为它和项目里其他模块都不一样。**

ADR 001 的观测发生在网络边界上。记忆不在那里——它是 Agent 进程内的一个目录、
一个 checkpointer 库、或一段远程状态。边界拦不到它。

所以这里做的**不是观测，是操纵**：在会话边界把 Agent 的记忆整体存下来，
之后可以把它回滚回去，看 Agent 会不会因为"忘了"而把同一件事再做一遍。

这区分直接决定了归因。重复扣款确实发生了，但制造它的是我们的回滚——
报告必须说清楚，否则用户会去改一个没问题的 Prompt。

内容寻址是刻意的：同一份记忆存两次只占一份空间，而且摘要相同即内容相同，
让"这次恢复其实什么都没改"变成一个可判定的问题（空操作不该被当成故障注入）。
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import select

from sandbox.config import settings
from sandbox.db.engine import session_scope
from sandbox.db.models import MemorySnapshot


class MemoryAdapter(Protocol):
    """一种记忆载体的读写方式。Hermes 文件型、LangGraph checkpointer 各实现一个。

    `capture`/`restore` 必须**对称**：restore(capture()) 应当是个恒等操作。
    这条不做成断言而是写在这里，是因为破坏它的方式往往很隐蔽——比如 capture
    时漏掉了隐藏文件，restore 就会把它们删掉。
    """

    name: str

    def capture(self) -> bytes: ...

    def restore(self, payload: bytes) -> None: ...

    def entry_count(self) -> int: ...


@dataclass(frozen=True)
class SnapshotRef:
    """快照的元数据。不含内容——内容按需加载，避免把大对象一直捧在内存里。"""

    id: str
    session_sequence_id: str
    taken_at_ordinal: int
    taken_at_seq: int
    digest: str
    adapter: str
    payload_path: str
    size_bytes: int
    entry_count: int

    @property
    def position(self) -> tuple[int, int]:
        """快照在会话序列里的位置 (ordinal, seq)。跨会话归因就是靠它排序的。"""
        return (self.taken_at_ordinal, self.taken_at_seq)


class SnapshotStore:
    """快照存储。lite profile 落本地文件；生产期元数据进 PostgreSQL、大对象进 S3。

    接口按"可替换为对象存储"来设计：调用方只拿 `SnapshotRef` 和 `bytes`，
    永远不碰路径。
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else settings.memory_root / "snapshots"

    def _path_for(self, digest: str) -> Path:
        # 两级目录：单目录堆几万个文件在 Windows 上会明显变慢。
        return self.root / digest[:2] / f"{digest}.bin"

    async def save(
        self,
        *,
        session_sequence_id: str,
        taken_at_ordinal: int,
        taken_at_seq: int,
        adapter: str,
        payload: bytes,
        entry_count: int = 0,
    ) -> SnapshotRef:
        """存一份快照。相同内容重复存会复用同一个文件，但元数据每次都是新行。"""
        digest = hashlib.sha256(payload).hexdigest()
        path = self._path_for(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            # 先写临时文件再改名：中途失败不会留下一个内容截断、但摘要对不上的
            # 文件——那种文件在恢复时才会暴露，而那时已经覆盖掉 Agent 的记忆了。
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(payload)
            tmp.replace(path)

        snapshot_id = str(uuid.uuid4())
        async with session_scope() as db:
            db.add(
                MemorySnapshot(
                    id=snapshot_id,
                    session_sequence_id=session_sequence_id,
                    taken_at_ordinal=taken_at_ordinal,
                    taken_at_seq=taken_at_seq,
                    digest=digest,
                    adapter=adapter,
                    payload_path=str(path),
                    size_bytes=len(payload),
                    entry_count=entry_count,
                )
            )
        return SnapshotRef(
            id=snapshot_id,
            session_sequence_id=session_sequence_id,
            taken_at_ordinal=taken_at_ordinal,
            taken_at_seq=taken_at_seq,
            digest=digest,
            adapter=adapter,
            payload_path=str(path),
            size_bytes=len(payload),
            entry_count=entry_count,
        )

    async def ref(self, snapshot_id: str) -> SnapshotRef:
        async with session_scope() as db:
            row = await db.get(MemorySnapshot, snapshot_id)
            if row is None:
                raise KeyError(f"快照 {snapshot_id} 不存在")
            return _row_to_ref(row)

    async def load(self, snapshot_id: str) -> bytes:
        """取快照内容。读到一半失败会**抛错**，不返回截断的字节。

        记忆快照截断之后拿去恢复，等于把 Agent 的记忆砸掉一半——比恢复失败
        严重得多，所以这里宁可让调用方拿到异常。
        """
        ref = await self.ref(snapshot_id)
        path = Path(ref.payload_path)
        if not path.exists():
            raise FileNotFoundError(f"快照 {snapshot_id} 的内容文件已丢失：{path}")
        payload = path.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != ref.digest:
            raise ValueError(
                f"快照 {snapshot_id} 内容损坏：摘要应为 {ref.digest[:16]}，实际 {actual[:16]}。"
                f"拒绝用损坏的内容恢复记忆。"
            )
        return payload

    async def list_for_sequence(self, session_sequence_id: str) -> list[SnapshotRef]:
        """按位置列出某会话序列的全部快照。报告里的"记忆版本追踪"用它。"""
        async with session_scope() as db:
            rows = (
                (
                    await db.execute(
                        select(MemorySnapshot)
                        .where(MemorySnapshot.session_sequence_id == session_sequence_id)
                        .order_by(
                            MemorySnapshot.taken_at_ordinal, MemorySnapshot.taken_at_seq
                        )
                    )
                )
                .scalars()
                .all()
            )
        return [_row_to_ref(r) for r in rows]

    async def find_covering(
        self, session_sequence_id: str, position: tuple[int, int]
    ) -> SnapshotRef | None:
        """找位置 **早于等于** `position` 的最后一个快照。

        "回到某个时刻之前的状态"就靠它：跨会话重放场景里，我们把一个早于
        首次扣款的快照恢复回去，于是 Agent 的记忆里不再有那笔扣款。
        """
        refs = await self.list_for_sequence(session_sequence_id)
        covering = [r for r in refs if r.position <= position]
        return covering[-1] if covering else None


def _row_to_ref(row: MemorySnapshot) -> SnapshotRef:
    return SnapshotRef(
        id=row.id,
        session_sequence_id=row.session_sequence_id,
        taken_at_ordinal=row.taken_at_ordinal,
        taken_at_seq=row.taken_at_seq,
        digest=row.digest,
        adapter=row.adapter,
        payload_path=row.payload_path,
        size_bytes=row.size_bytes,
        entry_count=row.entry_count,
    )


def digest_of(payload: bytes) -> str:
    """内容摘要。用于判断两次快照是否同一状态。"""
    return hashlib.sha256(payload).hexdigest()


def summarize(ref: SnapshotRef) -> dict[str, Any]:
    """给报告用的快照摘要。"""
    return {
        "snapshot_id": ref.id,
        "position": list(ref.position),
        "adapter": ref.adapter,
        "digest": ref.digest[:16],
        "size_bytes": ref.size_bytes,
        "entry_count": ref.entry_count,
    }
