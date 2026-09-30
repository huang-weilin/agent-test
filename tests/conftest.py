"""共享测试夹具。

每个测试拿一个独立的临时 SQLite 文件，互不污染，也不碰开发用的 data/sandbox.db。

关键点：夹具替换的是 **`sandbox.db.engine` 里那个全局 SessionFactory**，而不是
另起一个并行的引擎。记录器、检测器都经由 `session_scope()` 取会话——如果测试
自建引擎，测的就是夹具的库、跑的就是开发库，测试会假绿。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import sandbox.db.engine as engine_module
from sandbox.db.models import Base


@pytest_asyncio.fixture
async def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[str]:
    """把全局引擎指到临时库，并建好表。"""
    url = f"sqlite+aiosqlite:///{(tmp_path / 'test.db').as_posix()}"
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    monkeypatch.setattr(engine_module, "SessionFactory", factory)
    monkeypatch.setattr(engine_module, "engine", engine)

    yield url

    await engine.dispose()


@pytest.fixture
async def db(temp_db: str) -> AsyncIterator[AsyncSession]:
    """直接拿一个会话，供 schema 类断言使用。"""
    factory = engine_module.SessionFactory
    async with factory() as session:
        yield session


@pytest.fixture
def make_call():
    """构造 CallView 的工厂。

    检测器测试要能直接喂**人为构造的轨迹**——这正是"构造一次重复扣款、
    构造一次循环"的做法。走真实 Agent 去撞出这些行为既慢又不确定。
    """
    from typing import Any

    from sandbox.detectors.base import CallView
    from sandbox.trace.normalize import normalize_args, structural_hash

    def _make(
        seq: int,
        tool_name: str,
        args: dict[str, Any] | None = None,
        *,
        is_write: bool = False,
        fingerprint: str | None = None,
        idempotency_key: str | None = None,
        replay_reason: str | None = None,
        blocked_by_proxy: bool = False,
        status: str = "ok",
        session_sequence_id: str = "seq-1",
        session_id: str = "sess-1",
    ) -> CallView:
        normalized = normalize_args(args or {})
        return CallView(
            id=f"call-{seq:03d}",
            seq=seq,
            tool_name=tool_name,
            args=normalized,
            fingerprint=fingerprint or structural_hash(tool_name, normalized),
            is_write=is_write,
            session_sequence_id=session_sequence_id,
            session_id=session_id,
            idempotency_key=idempotency_key,
            replay_reason=replay_reason,
            blocked_by_proxy=blocked_by_proxy,
            status=status,
        )

    return _make
