"""按工具拆解：报的是上下文占用，不是"这个工具花了多少钱"（需求 3.5 / 3.7）。

这一片的每一个数都要能读成一句成立的话，所以相机分成两类：

* **口径**：三个数（返回体 / 实测增量 / 租金）各自是怎么数出来的，以及
  `NULL` 与 `0` 的区别——`0 B` 说的是"它返回了空体"，而老库的行与没出去的调用
  说的是"没量到"，报告上这两句话完全不同；
* **不漂**：正文只有一份实现，两个终端出口（场景报告、跨运行成本报告）拿到的
  是同一串字符。有一条测试直接把脚本跑起来读它的 stdout——只断言"脚本 import
  了那个函数"是没用的，那正是"装上了但没接线"。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from sandbox.cost.aggregator import CostRollup, ToolCost, ToolCostReport, by_tool
from sandbox.db.models import LlmCall, ToolCall
from sandbox.formatting import TOOL_COST_TITLE, byte_size, display_width, pad_left, tool_cost_lines

# --- 造数据 -------------------------------------------------------------------


async def _llm(
    db: AsyncSession,
    *,
    run_id: str,
    seq: int,
    session_id: str | None,
    request_bytes: int | None = None,
    input_tokens: int = 100,
    ts: datetime | None = None,
) -> None:
    db.add(
        LlmCall(
            id=f"{run_id}-llm-{session_id or 'none'}-{seq}",
            test_run_id=run_id,
            session_id=session_id,
            seq=seq,
            model="stub-model",
            input_tokens=input_tokens,
            output_tokens=1,
            status="ok",
            request_bytes=request_bytes,
            ts=ts or datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
        )
    )


async def _tool(
    db: AsyncSession,
    *,
    run_id: str,
    seq: int,
    session_id: str | None,
    name: str = "list_orders",
    result_bytes: int | None = 1000,
    status: str = "ok",
    blocked: bool = False,
    is_write: bool = False,
    ts: datetime | None = None,
) -> None:
    db.add(
        ToolCall(
            id=f"{run_id}-tool-{session_id or 'none'}-{seq}",
            test_run_id=run_id,
            session_id=session_id,
            # 会话内序号，落库时必填。租金按 `seq` 分组，这一列不参与计算——
            # 但行得写得出来，否则测的是"造不出的数据"。
            session_sequence_id=f"{session_id or 'none'}-{seq}",
            seq=seq,
            args_json="{}",
            arg_hash="0" * 64,
            fingerprint="1" * 64,
            tool_name=name,
            status=status,
            blocked_by_proxy=blocked,
            is_write=is_write,
            result_bytes=result_bytes,
            latency_ms=5,
            ts=ts or datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
        )
    )


async def _run_row(db: AsyncSession, run_id: str) -> None:
    from sandbox.db.models import TestRun

    db.add(
        TestRun(
            id=run_id,
            scenario_id="s",
            agent_version="a@1",
            config_hash="h",
            started_at=datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
        )
    )
    await db.flush()


@pytest.fixture
async def seeded(db: AsyncSession) -> AsyncSession:
    """一次运行、一段会话、5 次请求、2 次工具调用。

    形状是照着 `loop_paging_bounded` 真跑出来的那份摆的：先两次空转请求，然后
    第一次工具调用，中间两次请求，第二次工具调用，最后一次请求。所以两次调用的
    「之后还剩几次请求」分别是 3 和 1。
    """
    await _run_row(db, "r1")
    for seq in (1, 2, 5, 6, 9):
        await _llm(db, run_id="r1", seq=seq, session_id="s1", request_bytes=100 * seq)
    await _tool(db, run_id="r1", seq=3, session_id="s1", result_bytes=500)
    await _tool(db, run_id="r1", seq=7, session_id="s1", result_bytes=800)
    await db.commit()
    return db


# --- 顺序 ---------------------------------------------------------------------


#: 夹具那一局的正确答案：序号 5、6、9 三次请求在第一次调用之后（500 × 3），
#: 序号 7 那次调用之后只剩序号 9（800 × 1）。
_RENT_BY_SEQ = 500 * 3 + 800 * 1


async def _set_clock(db: AsyncSession, *, descending: bool) -> None:
    """把所有行的 `ts` 铺成与序号同向（或反向）的一条时间线。

    两个方向都要能铺出来：被测对象的墙钟会被快照恢复和时间加速拨动，所以
    "哪一行先发生"不能由时间戳回答。
    """
    from sqlalchemy import update

    base = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
    # 逐行改，不塞一条 SQL 表达式：这一局统共七行，逐个序号写出来的东西读得懂，
    # 而且"序号 3 与 7 上各有一个工具调用"这件事在代码里看得见。
    for seq in (1, 2, 3, 5, 6, 7, 9):
        offset = timedelta(seconds=(-seq if descending else seq))
        for table in (LlmCall, ToolCall):
            await db.execute(
                update(table)
                .where(table.test_run_id == "r1", table.seq == seq)
                .values(ts=base + offset)
            )
    await db.commit()


async def test_时间戳倒过来也不影响租金(seeded: AsyncSession, db: AsyncSession) -> None:
    """`seq` 是运行内**全局单调**序号（LLM 调用与工具调用共用一条时间线）。

    这一局把墙钟**故意铺成反的**：序号越大时间越早。按时间戳数的实现会得出
    "这两次工具调用之后几乎什么都没有"，租金算成另一个数（500×2 + 800×4 = 4200）。

    顺序只能有一个来源，而墙钟不是——这个项目的墙钟会被快照恢复和时间加速拨动。
    """
    await _set_clock(db, descending=True)
    tool = (await by_tool(run_id="r1")).tools["list_orders"]
    assert tool.rent_bytes == _RENT_BY_SEQ
    assert tool.calls_with_followup == 2


async def test_时间戳正着铺也一样(seeded: AsyncSession, db: AsyncSession) -> None:
    """上一条的**对照**：把时间戳铺成与序号同向，结果必须一模一样。

    没有这条，上一条红了也说不清是"读对了 seq"还是"那两个数碰巧不等"。
    """
    await _set_clock(db, descending=False)
    tool = (await by_tool(run_id="r1")).tools["list_orders"]
    assert tool.rent_bytes == _RENT_BY_SEQ
    assert tool.calls_with_followup == 2


# --- 之后没有请求 -----------------------------------------------------------------


async def test_之后一次请求都没有的调用租金记零并单独计数(db: AsyncSession) -> None:
    """0 是"看不见"，不是"免费"——两个数都要有，报告里也要印。"""
    await _run_row(db, "r1")
    await _llm(db, run_id="r1", seq=1, session_id="s1", request_bytes=500)
    await _tool(db, run_id="r1", seq=2, session_id="s1", result_bytes=9000)
    await db.commit()

    report = await by_tool(run_id="r1")
    tool = report.tools["list_orders"]
    assert tool.rent_bytes == 0
    assert tool.calls_without_followup == 1
    assert tool.calls_with_followup == 0
    text = "\n".join(tool_cost_lines(report))
    assert "0 是「看不见」，不是「免费」" in text


# --- 会话要分开 ---------------------------------------------------------------


async def test_另一个会话的请求不算在这一次调用头上(db: AsyncSession) -> None:
    """上下文是**按会话**从零开始的，所以租金也必须按会话数。

    一次长时运行里有多段会话；不分组的话，上一段会话末尾的一次工具调用会被
    下一段的请求"续上"，租金凭空变大。
    """
    await _run_row(db, "r1")
    await _llm(db, run_id="r1", seq=1, session_id="s1", request_bytes=500)
    await _tool(db, run_id="r1", seq=2, session_id="s1", result_bytes=1000)
    for seq in (3, 4, 5):
        await _llm(db, run_id="r1", seq=seq, session_id="s2", request_bytes=500)
    await db.commit()

    tool = (await by_tool(run_id="r1")).tools["list_orders"]
    assert tool.rent_bytes == 0
    assert tool.calls_without_followup == 1


# --- 三个数的口径 ---------------------------------------------------------------


async def test_实测增量是前后两次请求的差(seeded: AsyncSession) -> None:
    """`context_growth_bytes` 是观测：这次调用之后第一次请求 − 之前最后一次请求。

    序号 1、2 两次请求各 100、200 字节（夹具里 request_bytes = 100 × seq）。
    第一次工具调用在序号 3 之后，前一次请求是序号 2（200），后一次是序号 5
    （500）→ +300。第二次工具调用夹在 6（600）与 9（900）之间 → +300。
    """
    tool = (await by_tool(run_id="r1")).tools["list_orders"]
    assert tool.growth_measured_calls == 2
    assert tool.context_growth_bytes == 300 + 300


async def test_量不到的增量不写零(db: AsyncSession) -> None:
    """前后有一次没量到，这次调用就不进「实测增量」——写 0 是在说它没变。"""
    await _run_row(db, "r1")
    await _llm(db, run_id="r1", seq=1, session_id="s1", request_bytes=None)
    await _tool(db, run_id="r1", seq=2, session_id="s1", result_bytes=10)
    await _llm(db, run_id="r1", seq=3, session_id="s1", request_bytes=800)
    await db.commit()

    tool = (await by_tool(run_id="r1")).tools["list_orders"]
    assert tool.growth_measured_calls == 0
    assert tool.context_growth_bytes == 0
    assert "实测增量" in "\n".join(tool_cost_lines(await by_tool(run_id="r1")))


async def test_密度只用记下请求体字节的那些行(db: AsyncSession) -> None:
    """老行的输入 token 要是进了分母，密度会被稀释成一个没有含义的数。"""
    await _run_row(db, "r1")
    await _llm(db, run_id="r1", seq=1, session_id="s1", request_bytes=1000, input_tokens=200)
    await _llm(db, run_id="r1", seq=2, session_id="s1", request_bytes=None, input_tokens=999_999)
    await _llm(db, run_id="r1", seq=3, session_id="s1", request_bytes=None, input_tokens=999_999)
    await db.commit()

    report = await by_tool(run_id="r1")
    assert report.llm_calls == 3
    assert report.llm_calls_measured == 1
    assert report.bytes_per_token == pytest.approx(5.0)


async def test_一条都没量到时不许猜一个密度(db: AsyncSession) -> None:
    """量不到就报字节、不折算——**不猜一个数**。"""
    await _run_row(db, "r1")
    await _llm(db, run_id="r1", seq=1, session_id="s1", request_bytes=None)
    await _tool(db, run_id="r1", seq=2, session_id="s1", result_bytes=4000)
    await db.commit()

    report = await by_tool(run_id="r1")
    assert report.bytes_per_token is None
    assert report.tokens_for(4000) is None
    text = "\n".join(tool_cost_lines(report))
    assert "折算 token 的密度无从量起" in text
    # 字节数照报——量不到的只是折算。
    assert byte_size(4000) in text


# --- NULL 与 0 是两件事 ---------------------------------------------------------


async def test_返回空体和没量到在报告上是两句话(db: AsyncSession) -> None:
    """一对**只差一个取值**的输入，报告必须给出两句不同的话。

    `result_bytes=0` = 这个工具返回了一个空体（是真的量到了，量到的是 0）；
    `result_bytes=NULL` = 这一行没有体量记录（老库的行，或者调用根本没出去）。
    两者印成同一句 `0 B` 的话，读的人会把"历史数据不全"读成"这个工具不返回东西"
    ——报告说了比观测更多的话。
    """
    await _run_row(db, "r1")
    await _tool(db, run_id="r1", seq=1, session_id="s1", name="empty_body", result_bytes=0)
    await _tool(db, run_id="r1", seq=2, session_id="s1", name="legacy", result_bytes=None)
    await db.commit()

    report = await by_tool(run_id="r1")
    lines = {line.strip().split()[0]: line for line in tool_cost_lines(report) if " 次  " in line}

    def column(line: str, name: str) -> str:
        """取「返回体」那一列的值。整行里还可能有别的「量不到」（实测增量），
        所以只能按列名切，不能整行去 in。"""
        return line.split(name, 1)[1].split("实测增量", 1)[0].strip()

    assert column(lines["empty_body"], "返回体") == "0 B"
    assert column(lines["legacy"], "返回体") == "量不到"

    assert report.tools["empty_body"].measured_size_calls == 1
    assert report.tools["legacy"].measured_size_calls == 0
    assert report.tools["legacy"].unmeasured_size_calls == 1
    # 没量到的那些要单独报个数，不能只在行里写「量不到」就完了。
    assert "另有 1 次调用没量到返回体字节数" in "\n".join(tool_cost_lines(report))


async def test_被挡下的调用算调用_但不算返回体(db: AsyncSession) -> None:
    """被沙箱自己的策略挡下的那些：调用次数里要算，体量里不算。

    `blocked_by_proxy` 与 `status="blocked"` 不是一回事（未声明的工具也是
    blocked，但那是被测对象做错了事）——两个都留在数里，报告不说谁对谁错。
    """
    await _run_row(db, "r1")
    await _tool(
        db, run_id="r1", seq=1, session_id="s1", name="http_post",
        result_bytes=None, status="blocked", blocked=True, is_write=True,
    )
    await db.commit()

    tool = (await by_tool(run_id="r1")).tools["http_post"]
    assert tool.calls == 1
    assert tool.blocked_by_proxy_calls == 1
    assert tool.by_status == {"blocked": 1}
    assert tool.measured_size_calls == 0
    assert tool.rent_bytes == 0


# --- 两个出口是同一串字符 --------------------------------------------------------


def test_正文只有一份_标题也是() -> None:
    """脚本与场景报告都从 `formatting` 取正文和标题。

    这条是**静态**的，只防一件事：有人把那段话复制一份出去改。真正证明"接线了"
    的是下面那条把脚本跑起来的测试。
    """
    import scripts.cost_report as report_script  # noqa: PLC0415

    assert report_script.TOOL_COST_TITLE is TOOL_COST_TITLE
    assert report_script.tool_cost_lines is tool_cost_lines


async def test_跨运行报告把这一段真的印出来了(temp_db: str, capsys) -> None:
    """把脚本跑起来读 stdout。

    只断言 import 是对的没用——`import` 了却不调用是"装上了但没接线"的标准写法，
    而那正是这一片要防的。
    """
    import scripts.cost_report as report_script  # noqa: PLC0415
    from sandbox.db.engine import session_scope

    async with session_scope() as db:
        await _run_row(db, "r1")
        await _llm(db, run_id="r1", seq=1, session_id="s1", request_bytes=1000, input_tokens=200)
        await _tool(db, run_id="r1", seq=2, session_id="s1", result_bytes=5000)
        await db.commit()

    rc = await report_script._report("day", 30, False)
    assert rc == 0
    out = capsys.readouterr().out

    assert f"=== {TOOL_COST_TITLE} ===" in out
    assert "list_orders" in out
    assert "5.0 KB" in out
    assert "租金是下界，不是账单" in out


async def test_跨运行报告的_json_里也带上这一段(temp_db: str, capsys) -> None:
    """键名是 `by_tool_context`，不是 `by_tool`：后者读起来像"按工具的账单"。"""
    import json  # noqa: PLC0415

    import scripts.cost_report as report_script  # noqa: PLC0415
    from sandbox.db.engine import session_scope

    async with session_scope() as db:
        await _run_row(db, "r1")
        await _llm(db, run_id="r1", seq=1, session_id="s1", request_bytes=1000, input_tokens=200)
        await _tool(db, run_id="r1", seq=2, session_id="s1", result_bytes=5000)
        # 这一次请求在工具调用**之后**：没有它，租金是 0（"之后一次请求都没有"），
        # 而这条测的是折算，得先把租金造出来。
        await _llm(db, run_id="r1", seq=3, session_id="s1", request_bytes=2000, input_tokens=400)
        await db.commit()

    rc = await report_script._report("day", 30, True)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)

    assert "by_tool" not in payload
    section = payload["by_tool_context"]
    assert section["tools"][0]["tool_name"] == "list_orders"
    assert section["tools"][0]["rent_tokens"] == 1000  # 5000 B ÷ 5.0 字节/token
    assert section["bytes_per_token"] == 5.0


# --- 折 token -----------------------------------------------------------------


async def test_每个工具的token折算用的是同一份密度(seeded: AsyncSession) -> None:
    """密度是整份报告一个数。逐行各算一遍就等于承认它们可以用不同的密度。"""
    report = await by_tool(run_id="r1")
    payload = report.to_dict()
    for item in payload["tools"]:
        assert item["rent_tokens"] == round(item["rent_bytes"] / report.bytes_per_token)


# --- 宽度 ---------------------------------------------------------------------


def test_中文占两列() -> None:
    """`f"{x:>9}"` 对中文是错的：3 个汉字是 6 列，不是 3 列。

    「量不到」正好是这一节里唯一的中文占位符，所以对齐不是理论问题。
    """
    assert display_width("量不到") == 6
    assert display_width("ok") == 2
    assert pad_left("量不到", 9) == "   " + "量不到"
    assert len(pad_left("量不到", 9)) == 6  # 3 个空格 + 3 个字
    assert pad_left("abc", 3) == "abc"
    assert pad_left("abcd", 3) == "abcd"  # 超了就原样，不截断


def test_表格里中文占两列_中文那一格也真对齐() -> None:
    """直接量渲染出来的那两行的列位置。

    这一局的两个工具**各占住一种中文**：

    * 名字是中文（「长名字工具」）——名字那一列按 `len()` 补宽会整列歪；
    * 一个是「实测增量」量得到（数字）、另一个量不到（中文占位符）。

    第二条是探针找出来的：原来这一局两个工具的增量**都**量不到，于是那一格
    两行都是「量不到」、歪得一样多，看起来还是齐的——`{growth:>9}` 那种写法
    照样绿。量到一个、量不到一个，才逼得出那一格的真实宽度。
    """
    report = ToolCostReport(
        tools={
            "a": ToolCost(
                tool_name="a",
                calls=1,
                total_result_bytes=1,
                measured_size_calls=1,
                context_growth_bytes=5,
                growth_measured_calls=1,
            ),
            "长名字工具": ToolCost(
                tool_name="长名字工具", calls=1, total_result_bytes=1, measured_size_calls=1
            ),
        },
        llm_calls=1,
        llm_calls_measured=1,
        bytes_per_token=1.0,
    )
    # 只取表格里的那两行。按 `次` 或 `返回体` 过滤都不行：说明句里两个词都有
    # （"1/1 次调用"、"每轮增量比返回体本身大"）。数据行独有的是「次」两边
    # 各两个空格——那是列分隔，说明句里的「次」后面直接跟字。
    rows = [line for line in tool_cost_lines(report) if " 次  " in line]
    assert len(rows) == 2

    def start_of(line: str, name: str) -> int:
        return display_width(line[: line.index(name)])

    # 名字列：「次」要从同一列开始。名字本身是中文，按 len() 补宽会歪。
    assert len({start_of(line, "次") for line in rows}) == 1
    # 数值列：「租金」要从同一列开始。这一行里一个是「5 B」、一个是「量不到」。
    assert len({start_of(line, "租金") for line in rows}) == 1
    assert "量不到" in rows[1] and "量不到" not in rows[0]


def test_负数照印() -> None:
    """工具调用之后上下文变小是真实发生过的事（Agent 自己裁过）。

    把它显示成 0 会把"它裁了"这件事抹掉，显示成正数则是反过来撒谎。
    """
    assert byte_size(-1500) == "-1.5 KB"
    assert byte_size(0) == "0 B"


# --- 场景报告那个出口 -------------------------------------------------------------


def test_场景报告里印出这一段_且排在成本之后() -> None:
    """`render()` 要真的调它。

    上面那条跨运行报告的测试证明的是**脚本**接线了；这一条证明**运行器**接线了。
    两处各一条，因为它们是两个调用点，只接一个的话另一个删掉不会有任何东西变红。
    """
    from sandbox.scenario.runner import ScenarioResult, render

    result = ScenarioResult(
        scenario_id="demo",
        run_id="r1",
        cost=CostRollup(llm_calls=6, input_tokens=13_600, output_tokens=115, cost_usd=0.0),
        tool_costs=ToolCostReport(
            tools={
                "list_orders": ToolCost(
                    tool_name="list_orders",
                    calls=5,
                    total_result_bytes=7502,
                    measured_size_calls=5,
                    context_growth_bytes=10_937,
                    growth_measured_calls=5,
                    rent_bytes=24_081,
                    calls_with_followup=5,
                )
            },
            llm_calls=6,
            llm_calls_measured=6,
            bytes_per_token=3.288,
        ),
    )

    text = render(result)
    assert f"  {TOOL_COST_TITLE}:" in text
    assert "list_orders" in text
    # 排在成本那一段里，不是跑到报告尾巴上去。
    assert text.index(TOOL_COST_TITLE) > text.index("成本")
    # 只有一个工具时不写「合计」——一个东西的合计就是它自己。
    assert "这一次的租金" in text
    assert "各工具合计" not in text


def test_没有工具成本时这一段整个不出现() -> None:
    """老运行、或者一次没调过工具的运行：那一段要整个消失，不留一个空标题。"""
    from sandbox.scenario.runner import ScenarioResult, render

    assert TOOL_COST_TITLE not in render(ScenarioResult(scenario_id="demo", run_id="r1"))
    empty = ToolCostReport(tools={})
    # 空报告也一样：标题下面一行都没有的话，"没调过工具"和"这一节坏了"长得一样。
    assert TOOL_COST_TITLE not in render(
        ScenarioResult(scenario_id="demo", run_id="r1", tool_costs=empty)
    )
