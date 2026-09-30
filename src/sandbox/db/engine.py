"""数据库引擎与会话工厂。开发期 SQLite，生产期 PostgreSQL（ADR 002）。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from sandbox.config import settings
from sandbox.db.models import Base


def _ensure_sqlite_parent_dir(url: str) -> None:
    """SQLite 不会自动创建父目录——不建的话首次连接直接报 unable to open database file。"""
    if not url.startswith("sqlite"):
        return
    _, _, raw_path = url.partition(":///")
    if raw_path and raw_path != ":memory:":
        Path(raw_path).parent.mkdir(parents=True, exist_ok=True)


_ensure_sqlite_parent_dir(settings.database_url)

engine = create_async_engine(settings.database_url, echo=False, future=True)
SessionFactory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_db() -> None:
    """建表。生产环境应改为 Alembic 迁移（依赖已声明，尚未启用）。"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_all() -> None:
    """仅供测试与本地重置使用。"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    async with SessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
