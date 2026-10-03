"""报告页顶上那句「所以呢、我该干什么」。

## 这一层唯一会犯的错，是**替结论多说一句话**

`explain.next_step` 不新增判定，它只把三个已经算好的结论（跑完了没有、钱对不对得上、
检测结论过没过）读成一句人话加一个下一步。所以它有两种失效方式，各自配一条断言：

**一、和真正的结论对不上。** 页面上写着「这次没查出问题」，而底下那张卡片是红的
——两句话都来自同一个报告，读的人却不知道信哪个。这类漂移不会有任何报错。
所以这里**拿真的 `ScenarioResult.ok` 去对**：语气是 `ok` 当且仅当它真的判通过。
不是手抄一张期望表，是让两边现算（同一条思路见 `test_explain_coverage.py`）。

**二、把「没查出问题」说成「没问题」。** 这是通俗化最容易丢掉的东西，也是
这个产品唯一值钱的地方。`pass` 那条自己必须写出边界（「不等于 Agent 没问题」），
而且**「没问题」每次出现都要紧挨着一个否定词**——写成「不等于『Agent 没问题』」是对的，
写成「这次没问题」是把「没测到」说成了「没风险」。

**三、话有了，但没接到页面上。** 表写对了、函数写对了，而模板没调它——页面上
一个字都不会变，**所有测试照样绿**（这个项目管这叫「装上了但没接线」）。所以
最后一条测试**真的起一次界面**，把每一个情形各渲染一遍，连底色一起核。

## 还有一条：不许有够不着的文案

八条里有任何一条永远走不到，它就是一段死文案——而**死文案和活文案在源码里长得
一模一样**。所以最后拿构造出来的报告把每一条都走一遍。

## 造这份快照时踩到的：`premises_held` 不是一个能传的参数

它和 `charges_ok` 一样是**算出来的**（`ScenarioResult` 上的 property），不是字段。
第一版测试直接写 `ScenarioResult(premises_held=True)`，当场 TypeError——但真正值得
记下来的是**它为什么会那样**：`premises_held` 要求 `sessions` 非空且每段前提成立。
于是"没给 sessions"和"会话没跑完"落在**同一档**上，都走 `premises` 那条文案。
这是对的（一次没有会话的运行确实什么都没测到），但它意味着**默认那份快照是判不通过的**
——拿它当"一切正常"用，那条「语气必须和真判定一致」的断言会在错的方向上变严。
"""

from __future__ import annotations

import itertools
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio

from sandbox.api import explain
from sandbox.api.app import create_app
from sandbox.db.engine import init_db, session_scope
from sandbox.db.models import DetectorVerdict, TestRun
from sandbox.scenario.expect import ExpectResult
from sandbox.scenario.runner import ScenarioResult, SessionOutcome
from sandbox.scenario.spec import Expectation

#: 页面上允许的四种语气，对应 `.panel` 的四个类（`ok`/`bad`/`warn` 和没有修饰类的
#: 中性底色）。多出一个值的话，模板上那个 `if step.tone in (...)` 会**静默地**
#: 把它渲染成中性色——一句「有问题」的话配一张灰卡片，看起来像没事。
_TONES = {"ok", "bad", "warn", "none"}


def _failed_expectation() -> ExpectResult:
    """一份「预期没达成」的比对结果。`ok` 是 False。"""
    return ExpectResult(
        missing=[Expectation(detector="loop", verdict=DetectorVerdict.FAIL)],
    )


def _session(*, premise_held: bool = True) -> SessionOutcome:
    """一段会话。只填决定「前提成不成立」的那几个字段。

    `premise_held` **不是能直接传进去的参数**，它是从脚本消费量和时钟对账算出来
    的（`SessionOutcome.premise_held`）。这正是要的形状：这里造不出一个「字段写着
    成立、而它的判据其实不成立」的会话。不成立的那一档照着它真实的含义来造——
    **脚本没被消费完**（`declared=2, consumed=1`），也就是"有会话没按场景写的走完"，
    而那正是这条文案存在的原因。
    """
    return SessionOutcome(
        name="s0",
        ordinal=0,
        agent_index=0,
        memory_horizon_ordinal=-1,
        restored_from=None,
        exit_code=0,
        elapsed_s=0.1,
        timed_out=False,
        script_steps_declared=2 if not premise_held else 1,
        script_steps_consumed=1,
        tool_calls=0,
    )


def _result(**overrides: object) -> ScenarioResult:
    """一个真的 `ScenarioResult`，用来拿它自己的 `ok` 当判据。

    默认那份是**能通过的最小一份**：一段前提成立的会话、一份空的预期。空预期
    的 `ok` 是 True（没有未命中、没有意外），而运行级的 `ok` 还要求前提成立与
    扣款对得上——`expected_charges` 不写就是"没验过钱"，按 True 算（见
    `ScenarioResult.charges_ok`）。少了一样，下面那条"语气和真判定必须一致"
    的断言就会在**错的方向上**变严。
    """
    base: dict[str, object] = {
        "scenario_id": "summary-gate",
        "run_id": "summary-gate",
        "sessions": [_session()],
        "expectation": ExpectResult(),
    }
    base.update(overrides)
    return ScenarioResult(**base)  # type: ignore[arg-type]


#: 每一种情形：构造参数、期望的语气、期望命中的条目名。
#: 期望的语气**不是**手抄的判定——真正的判据是下面 `_result(...).ok`。
_CASES = [
    ("一切正常", {}, "ok", "pass"),
    ("执行失败", {"error": "boom"}, "bad", "error"),
    ("前提不成立", {"sessions": [_session(premise_held=False)]}, "warn", "premises"),
    (
        "钱扣多了",
        {"expected_charges": 1, "egress_charges": 2},
        "bad",
        "charges",
    ),
    (
        "钱查不到",
        {"expected_charges": 1, "egress_charges": -1},
        "warn",
        "charges_unknown",
    ),
    ("没有预期", {"expectation": None}, "none", "no_expectation"),
    (
        "预期没达成",
        {"expectation": _failed_expectation()},
        "bad",
        "expectation_failed",
    ),
]


@pytest_asyncio.fixture
async def render_run_page(
    temp_db: str, tmp_path: Path
) -> AsyncIterator[Callable[[dict[str, Any]], Awaitable[str]]]:
    """把一份快照塞进库，**真起一次界面**，取回 `/runs/{id}` 的 HTML。

    为什么不自己拼一个 Jinja 环境：过滤器、`explain` 这个全局、`base.html` 的
    继承全都是在 `create_app` 里挂上去的。自己搭一份就是**另一条渲染路径**——
    测的是那条路，而对真正跑着的那条一无所知。`tests/test_api.py` 里那个
    `client` 夹具也不行：它是那个文件的局部夹具，而且带一个可控执行器（测并发
    用的替身），这里多一个替身就多一处和真实路径不一样的地方。

    起的是真 app、真的从库里读、真的走 `run.html`。唯一的假东西是这份快照本身。
    """
    await init_db()
    scenarios_root = tmp_path / "scenarios"
    scenarios_root.mkdir()
    app = create_app(scenarios_root=scenarios_root)
    counter = itertools.count()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://ui") as client:

        async def _render(report: dict[str, Any]) -> str:
            run_id = f"summary-page-{next(counter)}"
            async with session_scope() as db:
                db.add(
                    TestRun(
                        id=run_id,
                        scenario_id="summary-gate",
                        agent_version="react_demo@0.1",
                        config_hash="c" * 64,
                        session_sequence_id=f"seq-{run_id}",
                        status="completed",
                        started_at=datetime.now(UTC),
                        completed_at=datetime.now(UTC),
                        # 和快照里那个 `ok` 一致：真跑的时候就是这么落库的。
                        # 写成恒 True 的话，页面上会同时出现「不符合预期」的卡片和
                        # 一个绿的运行标签——那正是这整件事要防的自相矛盾。
                        verdict_ok=bool(report.get("ok")),
                        report_json=json.dumps(report, ensure_ascii=False),
                    )
                )
            response = await client.get(f"/runs/{run_id}")
            assert response.status_code == 200, (
                f"报告页没渲染出来：HTTP {response.status_code}。"
                f"渲染失败也是「没接线」的一种——只是它会响，比静默的那种好。"
            )
            return response.text

        yield _render


def test_the_table_is_not_empty_and_every_entry_is_well_formed() -> None:
    """相机：表得有内容，每条都要有语气、一句话、一个下一步。"""
    assert len(explain.NEXT_STEP) >= 8, f"NEXT_STEP 只剩 {len(explain.NEXT_STEP)} 条？"
    for name, entry in explain.NEXT_STEP.items():
        assert set(entry) == {"tone", "headline", "action"}, f"{name} 的键不对：{sorted(entry)}"
        assert entry["tone"] in _TONES, f"{name} 的语气 {entry['tone']!r} 不在 {sorted(_TONES)}"
        assert entry["headline"].strip(), f"{name} 没有一句话结论"
        assert len(entry["action"]) >= 20, f"{name} 的下一步太短，等于没说"


@pytest.mark.parametrize(
    ("label", "overrides", "tone", "key"), _CASES, ids=[c[0] for c in _CASES]
)
def test_each_situation_gets_its_own_words(
    label: str, overrides: dict[str, object], tone: str, key: str
) -> None:
    """七种情形各说各的话。"""
    step = explain.next_step(_result(**overrides).to_dict())
    assert step == explain.NEXT_STEP[key], f"{label}：应给 {key}，实际给了 {step['headline']}"
    assert step["tone"] == tone


@pytest.mark.parametrize(("label", "overrides", "tone", "key"), _CASES, ids=[c[0] for c in _CASES])
def test_the_summary_never_contradicts_the_real_verdict(
    label: str, overrides: dict[str, object], tone: str, key: str
) -> None:
    """**语气是 `ok` 当且仅当运行真的判通过。**

    判据是 `ScenarioResult.ok` 现算的，不是抄的。这条盯的是"页面顶上那句话和底下
    那张卡片各说各的"——两边都是同一份报告的产物，读的人却没理由知道该信哪个。
    """
    result = _result(**overrides)
    step = explain.next_step(result.to_dict())
    assert (step["tone"] == "ok") is result.ok, (
        f"{label}：运行级判定 ok={result.ok}，而顶上那句话的语气是 {step['tone']!r}"
        f"（{step['headline']}）——两边对不上"
    )


def test_the_old_snapshot_says_it_did_not_record_rather_than_failing() -> None:
    """旧快照（没有 `ok` 键）既不是绿也不是红。

    这是这一页最容易被写错的一格：`report.expectation` 里没有 `ok` 时，Jinja 取到的是
    Undefined——**恒假**，于是原先每一份报告的第一张卡片都写着「不符合预期」。
    顶上这句话同理：不许把「没记录」说成「不通过」，也不许说成「没问题」。
    """
    report = _result().to_dict()
    report["expectation"] = {k: v for k, v in report["expectation"].items() if k != "ok"}
    step = explain.next_step(report)
    assert step["tone"] == "none"
    assert step is explain.NEXT_STEP["expectation_unrecorded"]


def test_the_pass_wording_never_claims_the_agent_is_fine() -> None:
    """「没问题」不许不加否定地说出口。见文件头第二节。

    这不是修辞检查。这个工具全部的价值就建立在「判不了」和「通过」严格分开这件事上，
    而顶上这句话是**读的人唯一一定看得见的那句**——它要是写成「没问题」，
    底下所有的边界说明都救不回来。
    """
    step = explain.NEXT_STEP["pass"]
    text = step["headline"] + step["action"]
    negations = ("不等于", "不算", "不是", "别读成", "不能读", "不要读")
    for word in ("没问题", "正常", "全绿"):
        start = 0
        while (at := text.find(word, start)) != -1:
            window = text[max(0, at - 10) : at]
            assert any(neg in window for neg in negations), (
                f"{word!r} 出现在「通过」那句话里，前面却没有否定词：\n"
                f"    …{text[max(0, at - 18) : at + len(word) + 4]}…\n"
                "没查出问题就是没查出问题，不能读成没问题。"
            )
            start = at + 1


def test_no_entry_is_unreachable() -> None:
    """八条都要走得通。够不着的文案就是死文案。"""
    reached = {explain.next_step(_result(**o).to_dict())["headline"] for _, o, _, _ in _CASES}
    # `expectation_unrecorded` 那一档要一份去掉 `ok` 的旧快照才走得到，单独补上。
    old = _result().to_dict()
    old["expectation"] = {k: v for k, v in old["expectation"].items() if k != "ok"}
    reached.add(explain.next_step(old)["headline"])
    unreachable = [n for n, e in explain.NEXT_STEP.items() if e["headline"] not in reached]
    assert not unreachable, f"这几条文案没有任何情形走得到，等于死文案：{unreachable}"


async def test_the_report_page_actually_renders_it(
    render_run_page: Callable[[dict[str, Any]], Awaitable[str]],
) -> None:
    """真渲染一遍：这段话要真的出现在 `/runs/{id}` 上。

    光有函数和表不够——没接进模板的话，页面一个字都不会变，而所有测试照样绿
    （这个项目管这叫「装上了但没接线」）。

    **连底色一起核**，因为 `run.html` 那句 `if step.tone in ('ok','bad','warn')`
    会**静默地**把不认识的语气降级成中性色：一句「有问题」配一张灰卡片，看起来
    像没事。这不是理论风险——这行判语气的话和上面那张表是两处，改一处漏一处
    不会报错。
    """
    for label, overrides, tone, _key in _CASES:
        report = _result(**overrides).to_dict()
        step = explain.next_step(report)
        html = await render_run_page(report)
        assert step["headline"] in html, (
            f"{label}：报告页上没渲染出「{step['headline']}」——"
            f"`run.html` 里那一块八成没接上 `explain.next_step`"
        )
        assert step["action"] in html, f"{label}：只渲了那句话，「下一步」没渲出来"
        want = f'class="panel {tone}"' if tone in ("ok", "bad", "warn") else 'class="panel "'
        assert want in html, (
            f"{label}：语气 {tone!r} 没落到页面上（页面上找不到 {want}）——"
            f"`run.html` 那句 `if step.tone in (...)` 是不是把这一档漏了？"
        )
