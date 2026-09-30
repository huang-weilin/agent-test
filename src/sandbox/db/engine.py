"""数据库引擎与会话工厂。开发期 SQLite，生产期 PostgreSQL（ADR 002）。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.schema import CreateColumn

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


def _missing_column_ddl(sync_conn: Any) -> list[str]:
    """找出"模型里有、库里没有"的列，生成 ALTER TABLE 语句。"""
    inspector = inspect(sync_conn)
    existing_tables = set(inspector.get_table_names())
    statements: list[str] = []
    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue  # 表本身缺，create_all 会整张建出来
        present = {col["name"] for col in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in present:
                continue
            rendered = CreateColumn(column).compile(dialect=sync_conn.dialect)
            statements.append(f"ALTER TABLE {table.name} ADD COLUMN {rendered}")
    return statements


async def _add_missing_columns() -> None:
    """给**已存在**的表补上新加的列。

    `create_all` 只建缺的表，不会给已存在的表加列。所以给模型加一个字段之后，
    老库会在第一次查询时炸在 `no such column: test_run.verdict_ok` 上——而
    在此之前 README 给的解法是"删掉 data/sandbox.db"。那对一个把**历史记录**
    当作卖点的工具是不可接受的：为了用上新版本，用户得先扔掉全部的运行历史。

    **这不是迁移系统。** 它不处理删列、改名、改类型、数据回填，也不留版本号。
    它只保证一件事：新加的字段在旧库上存在。真正的迁移走 Alembic（依赖已声明，
    尚未启用）。

    新加的列必须是可空的，或者带 server_default——否则 SQLite 的
    `ALTER TABLE ADD COLUMN` 会直接拒绝。这不是限制，是**故意的**：一个
    NOT NULL 的新列在旧行上填什么值都是替历史编造事实。可空 + 在读取侧
    显式处理 NULL，才说得清"这条运行是加这个字段之前跑的"。
    """
    async with engine.begin() as conn:
        for statement in await conn.run_sync(_missing_column_ddl):
            await conn.execute(text(statement))


async def init_db() -> None:
    """建表 + 补齐新增的列。生产环境应改为 Alembic 迁移（依赖已声明，尚未启用）。"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await _add_missing_columns()


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
