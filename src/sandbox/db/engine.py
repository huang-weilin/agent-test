"""数据库引擎与会话工厂。开发期 SQLite，生产期 PostgreSQL（ADR 002）。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
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


def make_engine(url: str) -> AsyncEngine:
    """建引擎。**生产那个和测试那个都从这一行出来，没有第二个构造处。**

    理由不是"少写几行"，而是池的参数必须两边一致：测试用的夹具自建一个池的话，
    "取消之后连接池还好不好用"这件事测的是夹具的池，而跑的是另一个。

    `pool_pre_ping=True` 是为一件实测出来的事加的：**流式请求被取消时，落库那个
    `await` 停在 SQLAlchemy 归还连接的路径上**，那条连接会以"已关闭但还在池里"
    的状态留下来。紧接着的第一次取连接就会拿到它（`sqlite3.OperationalError:
    no active connection`），而**那一次可能是另一个运行、另一个场景的请求**——
    报出来的错和它做的事毫无关系。

    pre-ping 不阻止尸体产生，只是不让下一个人拿到它：取连接时先探一下，探不通
    就扔掉换一条。这对 SQLite 是本地的、对生产期的 PostgreSQL 也是标准配置——
    数据库重启、连接被中间件掐掉、空闲超时都会留下同样的东西。

    实测见 `tests/test_gateway_stream.py` 里那条"取消之后"，它量的是同一个
    构造路径（夹具走的就是这个函数）。
    """
    return create_async_engine(
        url,
        echo=False,
        future=True,
        pool_pre_ping=True,
    )


engine = make_engine(settings.database_url)
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
