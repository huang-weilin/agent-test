"""边界上留下的东西：体量留下，正文不留。

**这条承诺以前是假的。** `ToolCall` 上有一列 `result_json`，从第一次提交
（`4b290d6`）起就一直在写工具返回体的**全文**；与此同时仓库里一共十三处写着
「返回体不落库（只留参数指纹）」，其中几处还挂着「按 ADR 002」——ADR 002 里
没有这句话（它说的是 span 级数据不重复落库）。十三处的数法记在
`db/models.py` 那段注里。

两边都不是凭空写的：写下那句话的人记得的是 `CallView`（检测器看到的视图）
确实不含返回值，而列确实在写。**「检测器看不到」和「库里没有」是两句不同的
话**，它们在这个仓库里被当成了一句。

现在两边对齐到中间：留下体量（`result_bytes`，成本报告要算「这个工具的返回体
在上下文里被重发了多少」），不留正文。下面盯的就是后半句——**整个库里有没有
那些字节**，而不是「我们没去读那个字段」。

断言「库里没有某个字符串」有个天然的坑：**扫描本身坏掉时它照样绿**。所以
第二条用例是那条扫描的对照，第三条是「承诺改了，存量怎么办」。
"""

from __future__ import annotations

import json

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient, MockTransport, Response
from sqlalchemy import select, text

from sandbox.db.engine import init_db
from sandbox.db.models import Session, TestRun, ToolCall
from sandbox.gateway import app as gateway_app
from sandbox.proxy import app as proxy_app
from sandbox.proxy.policy import AllowRule, EgressPolicy
from sandbox.runtime import runs

#: 混在返回体里的标记。够独特，别处不会自然出现。
MARKER = "SANDBOX-CANARY-3f9a1c"


@pytest.fixture(autouse=True)
def _clean_registry():
    runs.clear()
    proxy_app.set_policy(None)
    proxy_app.set_transport(None)
    yield
    runs.clear()
    proxy_app.set_policy(None)
    proxy_app.set_transport(None)


@pytest_asyncio.fixture
async def gw_client(temp_db: str):
    async with AsyncClient(
        transport=ASGITransport(app=gateway_app.app), base_url="http://gw.test"
    ) as c:
        yield c


@pytest_asyncio.fixture
async def px_client(temp_db: str):
    async with AsyncClient(
        transport=ASGITransport(app=proxy_app.app), base_url="http://px.test"
    ) as c:
        yield c


async def _register(gw_client: AsyncClient) -> dict:
    resp = await gw_client.post(
        "/_sandbox/runs",
        json={
            "scenario_id": "storage",
            "agent_version": "react_demo@0.1",
            "tools": {"get_order": "/tools/get_order"},
            "tool_schemas": {
                "get_order": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                }
            },
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _body_with_marker() -> dict:
    return {"ok": True, "order": {"order_id": "A1", "note": MARKER}, "filler": "x" * 200}


async def _scan_whole_db(db, needle: str) -> list[str]:
    """在**每一张表的每一个格子**里找这个字符串。

    不按列名挑：挑列名的话，「哪个字段存了正文」这个判断就由写测试的人做，
    而存进去的人可以不认账——他大可以新加一列。

    表与列都从**库本身**读（`sqlite_master` + `PRAGMA table_info`），不是从
    `Base.metadata`。后者看着更整洁，但它扫的是「模型认识的列」：一条用原生
    DDL 加的列在它眼里不存在。**这不是假想的漏洞**——`result_json` 这一列本身
    就是这么一路被读到又被漏掉的，写这条扫描时第一版正是栽在这里（老库用例
    红了一次：正文确实在库里，扫描说没有）。
    """
    tables = [
        row[0]
        for row in (
            await db.execute(
                text(
                    "SELECT name FROM sqlite_master"
                    " WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                )
            )
        ).all()
    ]
    hits: list[str] = []
    for table in tables:
        columns = [
            row[1] for row in (await db.execute(text(f"PRAGMA table_info({table})"))).all()
        ]
        rows = (await db.execute(text(f'SELECT * FROM "{table}"'))).all()
        for row in rows:
            for column, value in zip(columns, row, strict=True):
                if value is not None and needle in str(value):
                    hits.append(f"{table}.{column}")
    return hits


async def _check_one_call(db, resp, tool_name: str) -> None:
    """两条转发路共用的三条检查。放在一处，是因为它们必须**逐字相同**：
    分头写的话，两条路会各自漂移，而报告上它们是同一列数字。
    """
    # 先证明这次调用**真的带回了正文**——否则下面那句「库里没有」可能只是因为
    # 上游压根没回东西，那是一条什么都不成立的绿。
    assert resp.status_code == 200, resp.text
    assert MARKER in resp.text, "假上游的返回体里没有标记，这条用例就没在测东西"

    call = (
        await db.execute(select(ToolCall).where(ToolCall.tool_name == tool_name))
    ).scalar_one()
    # 体量按 Agent 实际收到的那份字节记：同一个渲染器量出来的，逐字节相同。
    assert call.result_bytes == len(resp.content)

    hits = await _scan_whole_db(db, MARKER)
    assert hits == [], (
        f"工具返回体的正文出现在库里的 {hits}。这不是「检测器读不到」的问题——"
        "是**存下来了**：被测对象引出来的内容不该在库里长住，而十三处说法也对"
        "用户承诺过它不落库。"
    )


# --- 两条写入路，各自一条绊线 ---------------------------------------------
# `result_bytes` 有两个写入方（`/tools/{name}` 与通用出口）。只盯一条的话，
# 另一条上删掉那个参数不会有任何东西变红。


async def test_声明工具的返回体不落库_体量留下(gw_client, px_client, db) -> None:
    created = await _register(gw_client)
    proxy_app.set_transport(MockTransport(lambda r: Response(200, json=_body_with_marker())))

    resp = await px_client.post(
        "/tools/get_order",
        json={"order_id": "A1"},
        headers={"Authorization": f"Bearer {created['api_key']}"},
    )

    await _check_one_call(db, resp, "get_order")


async def test_通用出口的返回体同样不落库(gw_client, px_client, db) -> None:
    """第二条写入路：目标地址由被测对象指定，返回体更是它引出来的。"""
    await _register(gw_client)
    proxy_app.set_policy(
        EgressPolicy(
            rules=[AllowRule(host="api.trusted.test", reason="用例需要")], default="deny"
        )
    )
    proxy_app.set_transport(MockTransport(lambda r: Response(200, json=_body_with_marker())))

    resp = await px_client.post("/fetch", json={"url": "https://api.trusted.test/thing"})

    await _check_one_call(db, resp, proxy_app.ANONYMOUS_EGRESS_TOOL)


# --- 上面那些「没有」是可信的 ---------------------------------------------


async def test_全库扫描本身找得到东西(db) -> None:
    """上面两条说的是「没有」。这条证明「找得到」是可能的——否则一个写坏的
    扫描会让上面两条永远绿。

    标记写进 `test_run.scenario_id`：一个本来就该保存的字段。跨表是刻意的
    ——真正要防的是「新加了一张表/一列，把正文存在那里」。
    """
    db.add(
        TestRun(id="run-canary", scenario_id=MARKER, agent_version="x@1", config_hash="h")
    )
    await db.commit()

    hits = await _scan_whole_db(db, MARKER)
    assert hits == ["test_run.scenario_id"], f"扫描没找到它该找到的东西：{hits}"


# --- 承诺改了，存量跟着走 -------------------------------------------------


async def test_旧库里存过的返回体正文会被清掉(db) -> None:
    """以前写进去的正文不会自己消失。

    旧库的形状用「在现库上加回那一列」来模拟——要复现的是**那一列存在、且里面有
    内容**这一个事实，不是当年那张表的每一个字节。真正的补列只加不删（见
    `db/engine.py`），擦除是唯一的例外，所以它需要一条自己的绊线。
    """
    db.add(TestRun(id="run-old", scenario_id="old", agent_version="x@1", config_hash="h"))
    db.add(Session(id="s-old", test_run_id="run-old", session_sequence_id="sq-old"))
    await db.commit()

    await db.execute(text("ALTER TABLE tool_call ADD COLUMN result_json TEXT"))
    # 老代码写进来的那一行。`ts` 也手写：这一列在模型上是 NOT NULL，而原生
    # INSERT 不走 Python 侧的 default。
    await db.execute(
        text(
            "INSERT INTO tool_call"
            " (id, test_run_id, session_id, session_sequence_id, seq, tool_name,"
            "  args_json, arg_hash, fingerprint, result_json, status, is_write,"
            "  blocked_by_proxy, latency_ms, ts)"
            " VALUES ('old-call', 'run-old', 's-old', 'sq-old', 1, 'get_order',"
            "  '{}', 'h', 'fp', :body, 'ok', 0, 0, 3, :ts)"
        ),
        {"body": json.dumps({"note": MARKER}), "ts": "2026-01-01 00:00:00.000000"},
    )
    await db.commit()
    assert await _scan_whole_db(db, MARKER) == ["tool_call.result_json"], (
        "旧库没造成——先看看列加上了没有"
    )

    await init_db()  # 就是应用启动时跑的那一趟

    assert await _scan_whole_db(db, MARKER) == [], "旧库里的返回体正文没有被清掉"

    # **擦的是内容，不是历史。** 行还在、其余字段原样——把行删掉就成了另一种谎：
    # 调用记录没了，报告上会显示「这个工具没被调过」。
    row = (
        await db.execute(text("SELECT id, tool_name, seq, status FROM tool_call"))
    ).mappings().all()
    assert [dict(r) for r in row] == [
        {"id": "old-call", "tool_name": "get_order", "seq": 1, "status": "ok"}
    ]

    # 体量**不补**。老行里的正文是当年的 `json.dumps`，和今天发给 Agent 的那份
    # 渲染未必逐字节相同（`ensure_ascii` 一类的差别就够），照它的长度填一个数
    # 会得到一个看起来像字节数、其实另一个口径的数字——报告说了比观测更多的话，
    # 正是这个项目最不能犯的错。NULL = 这行没有体量记录。
    size = (
        await db.execute(text("SELECT result_bytes FROM tool_call WHERE id='old-call'"))
    ).scalar_one()
    assert size is None

    # 新列在老库上也补上了（老库跑新版本的第一件事）。
    columns = {r[1] for r in (await db.execute(text("PRAGMA table_info(tool_call)"))).all()}
    assert "result_bytes" in columns


async def test_擦除是幂等的(db) -> None:
    """启动会跑很多次。第二次跑时旧列里已经没有任何非空值，不能再报一遍
    「擦掉了 N 行」——那不是擦除，是每次启动都吓人一跳。
    """
    db.add(TestRun(id="run-two", scenario_id="x", agent_version="x@1", config_hash="h"))
    await db.commit()
    await db.execute(text("ALTER TABLE tool_call ADD COLUMN result_json TEXT"))
    await db.execute(
        text(
            "INSERT INTO tool_call"
            " (id, test_run_id, session_id, session_sequence_id, seq, tool_name,"
            "  args_json, arg_hash, fingerprint, result_json, status, is_write,"
            "  blocked_by_proxy, latency_ms, ts)"
            " VALUES ('c2', 'run-two', 's2', 'sq2', 1, 'get_order', '{}', 'h', 'fp',"
            "  :body, 'ok', 0, 0, 3, :ts)"
        ),
        {"body": "{}", "ts": "2026-01-01 00:00:00.000000"},
    )
    await db.commit()

    from sandbox.db.engine import _erase_legacy_content

    assert await _erase_legacy_content() == 1
    assert await _erase_legacy_content() == 0


async def test_没出去的调用体量是空_不是零(gw_client, px_client, db) -> None:
    """和「出去了、但返回体是空的」分开。

    两个数在报告上长得一模一样的话，「这个工具每次都返回空」会和「这些调用压根
    没出去」混成一句，而它们指向的结论正相反：前者该去看工具，后者该去看策略。
    """
    await _register(gw_client)

    resp = await px_client.post("/fetch", json={"url": "https://evil.example.com/exfil"})

    assert resp.status_code == 403
    call = (await db.execute(select(ToolCall))).scalar_one()
    assert call.status == "blocked"
    assert call.blocked_by_proxy is True
    assert call.result_bytes is None, "被挡下的调用没有返回体，不是返回了一个空体"
