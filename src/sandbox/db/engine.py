"""数据库引擎与会话工厂。开发期 SQLite，生产期 PostgreSQL（ADR 002）。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
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

log = structlog.get_logger(__name__)


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

    `pool_pre_ping=True` 这个开关是**留着**的，但它的理由在 2026-10-03 降了一级：
    原来它是**实测逼出来的**——流式请求被取消时，落库那个 `await` 停在 SQLAlchemy
    归还连接的路径上，那条连接会以"已关闭但还在池里"的状态留下来，紧接着的第一次
    取连接就会拿到它（`sqlite3.OperationalError: no active connection`），而**那
    一次可能是另一个运行、另一个场景的请求**——报出来的错和它做的事毫无关系。

    那件事的**制造者已经没了**：落库现在脱开请求的生命周期（`gateway/stream.py`
    里 shield 住的那个任务），不再被取消打断。量到的是：**把 pre-ping 关掉之后
    `tests/test_gateway_stream.py` 全绿**——所以当年那句"关掉它就红"已经不成立，
    那条测试也不再是这个开关的判据。留着它的理由改成**推理的，不是量到的**：
    能停在归还连接半路上的 `await` 不止流式那一处（进程被杀、任务被取消、写库
    超时都可能），而"池里躺着一条关掉的连接"是这一层的通用问题；再说它对 SQLite
    是本地的、对生产期的 PostgreSQL 也是标准配置——数据库重启、连接被中间件掐掉、
    空闲超时都会留下同样的东西。

    **推理不是测量。** 换成"有证据说它挡了什么"之前，这一行按推理记，不按实测记。
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


def _legacy_content_to_erase(sync_conn: Any) -> list[str]:
    """要把**内容**清掉的旧列（这是擦除，不是补列）。

    整个机制只加不删，这里是唯一的例外，理由值得写下来：`tool_call.result_json`
    曾经存的是工具返回体的正文。各处对用户的承诺是它不落库（README 与四份
    docs 九处，连场景/测试/源码注释共十三处，数法见 `db/models.py` 那段注），
    而代码一直在写——两边对齐之后，承诺这边赢了，正文不再写。
    **但旧库里那些已经写进去的正文不会自己消失。** 留着它们，等于让旧库继续
    持有一个我们已经说不持有的东西，而且没人会再去看它（读取方只有两处注释）。

    只清空、不删列：删列要走真正的迁移（Alembic 未启用），而在这一点上清空与
    删列对"库里还有没有返回体正文"是等价的。语句是幂等的——第二次跑时
    `WHERE result_json IS NOT NULL` 已经匹配不到任何行。
    """
    inspector = inspect(sync_conn)
    if "tool_call" not in set(inspector.get_table_names()):
        return []
    present = {col["name"] for col in inspector.get_columns("tool_call")}
    if "result_json" not in present:
        return []  # 新库：这一列根本没建过
    return ["UPDATE tool_call SET result_json = NULL WHERE result_json IS NOT NULL"]


async def _erase_legacy_content() -> int:
    """跑一次擦除，返回擦掉的行数。

    **擦除必须出声。** 静默地清掉别人库里的数据，和静默地不记某次调用是同一类
    问题——只是方向相反：前者会让"我还留着历史"这句话不成立，而用户不会知道。
    """
    erased = 0
    async with engine.begin() as conn:
        for statement in await conn.run_sync(_legacy_content_to_erase):
            result = await conn.execute(text(statement))
            erased += result.rowcount or 0
    if erased:
        log.warning(
            "db.legacy_tool_result_bodies_erased",
            rows=erased,
            column="tool_call.result_json",
            why="返回体正文不再落库，旧库里的存量一并清空",
        )
    return erased


async def init_db() -> None:
    """建表 + 补齐新增的列 + 擦掉承诺不再持有的旧内容。

    生产环境应改为 Alembic 迁移（依赖已声明，尚未启用）。
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await _add_missing_columns()
    await _erase_legacy_content()


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
