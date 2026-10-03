"""一键体检：菜单的读取、结论的合成、以及那一页的渲染。

分三段，和被测的三层对应：

1. `catalog` —— 菜单是数据，校验它的形状和引用。
2. `report` —— 纯函数。**这一段最要紧**：体检页的风险不在于算错数，在于
   换个读法的时候把某个结论说成了另一个意思（"没判过"说成"没问题"）。
   下面每一条测试盯着一种说法。
3. 路由与渲染 —— 菜单页、跑起来、结论页。

这里刻意**不真跑场景**：真跑一趟六支要一分多钟，而起六个服务不是这一层
要测的东西（引擎有自己的端到端测试）。替身执行器 + 手写的报告快照，
测的是"结论怎么排"，那正是这一层新增的全部逻辑。
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio

from sandbox.api import explain, paths
from sandbox.api.app import create_app
from sandbox.api.jobs import JobRegistry
from sandbox.checkup import catalog
from sandbox.checkup import report as creport
from sandbox.db.engine import init_db, session_scope
from sandbox.db.models import TestRun

# --- 造菜单与场景 -----------------------------------------------------------


def _scenario_yaml(scenario_id: str, *, detector: str, verdict: str, write: bool = False) -> str:
    """一支最小场景。`expect` 是**判卷标准的唯一真相源**，体检不另写一份。"""
    return f"""
id: {scenario_id}
description: 测试用
agent:
  version: react_demo@0.1
  task: 查一下订单 A1。
tools:
  - name: get_order
    description: 查订单
    is_write: false
  - name: pay
    description: 付款
    is_write: true
    idempotency_param: key
    enforces_idempotency: true
sessions:
  - name: 第一次
    script:
      - tool_calls: [{{name: get_order, arguments: {{order_id: A1}}}}]
expect:
  - detector: {detector}
    verdict: {verdict}
"""


#: 一份能跑的小菜单。**必须有一支阴性对照**（`clean`：没有任何一条预期是 fail），
#: 见下面 `test_the_standard_menu_has_a_negative_control` 的理由。
MENU_YAML = """
name: 测试用菜单
summary: 三项，跑一下。
subject_note: 这一页测的是仓库里自带的演示 Agent。
items:
  - id: bad_money
    question: 同一笔钱，它会不会付两次？
    scenario: dup
    why: 重复付款是真金白银的损失。
  - id: false_alarm
    question: 它什么都没做错的时候，工具会不会冤枉它？
    scenario: clean
    why: 假警报一多，真警报就没人看了。
"""


@pytest.fixture
def checkup_roots(tmp_path: Path) -> tuple[Path, Path]:
    """(场景目录, 菜单目录)。两边都是临时目录。"""
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    (scenarios / "dup.yaml").write_text(
        _scenario_yaml("dup", detector="idempotency", verdict="fail"), encoding="utf-8"
    )
    (scenarios / "clean.yaml").write_text(
        _scenario_yaml("clean", detector="idempotency", verdict="pass"), encoding="utf-8"
    )
    menus = tmp_path / "checkups"
    menus.mkdir()
    (menus / "mini.yaml").write_text(MENU_YAML, encoding="utf-8")
    return scenarios, menus


# --- 造报告快照 -------------------------------------------------------------

#: `expectation` 的默认值哨兵。用 `None`（= 报告里那一节真的没有）当默认值的话，
#: 就没法表达"我没指定"，而这两种在报告里是两件事。
_UNSET = object()


def _snapshot(
    *,
    matched: list[tuple[str, str]] | None = None,
    missing: list[tuple[str, str]] | None = None,
    unexpected: list[tuple[str, str]] | None = None,
    uncovered: list[tuple[str, str]] | None = None,
    ok: bool = True,
    premises: bool = True,
    charges: bool = True,
    error: str | None = None,
    expectation: Any = _UNSET,
) -> dict[str, Any]:
    """一份 `report_json` 形状的字典。

    形状照着 `ScenarioResult.to_dict()` 与 `ExpectResult.to_dict()` 写——
    **特别是 `matched` 那个不对称的形状**（检测器名在 `expect` 里、结论原文
    在 `got` 里）。这里如果图省事把它拍平，被测代码里读错键也测不出来。
    """
    exp = expectation
    if expectation is _UNSET:
        exp = {
            "matched": [
                {
                    "expect": {"detector": d, "verdict": v, "reason_code": None},
                    "got": f"{d} 说了点什么",
                }
                for d, v in (matched or [])
            ],
            "missing": [
                {"detector": d, "verdict": v, "reason_code": None} for d, v in (missing or [])
            ],
            "unexpected": [
                {"detector": d, "verdict": v, "summary": f"{d} 多报了"}
                for d, v in (unexpected or [])
            ],
            "uncovered": [
                {"detector": d, "verdict": v, "summary": f"{d} 判不了"}
                for d, v in (uncovered or [])
            ],
        }
    return {
        "ok": ok,
        "premises_held": premises,
        "charges_ok": charges,
        "error": error,
        "expectation": exp,
    }


def _item(item_id: str, scenario: str = "dup") -> catalog.CheckupItem:
    return catalog.CheckupItem(
        id=item_id, question=f"{item_id} 会不会？", scenario=scenario, why="因为要查。"
    )


@pytest.fixture
def menu() -> catalog.Checkup:
    return catalog.Checkup(
        name="测试用菜单",
        summary="跑一下。",
        subject_note="这一页测的是仓库里自带的演示 Agent。",
        items=[_item("bad_money", "dup"), _item("false_alarm", "clean")],
    )


# --- 1. 菜单 -----------------------------------------------------------------


def test_the_standard_menu_loads_and_points_at_real_scenarios() -> None:
    """真正在用的那份菜单要能加载，而且点名的场景都真的在。"""
    checkup = catalog.load_checkup("standard")
    assert len(checkup.items) >= 4
    for item in checkup.items:
        assert (catalog.SCENARIOS_DIR / f"{item.scenario}.yaml").is_file(), item.scenario


def test_the_standard_menu_has_a_negative_control() -> None:
    """菜单里必须有一支**阴性对照**，而且其余各项不能都是对照。

    少了阴性对照，"这套工具一次都没报错"和"这套工具就是不会报错"在结论页上
    长得一模一样——而那正是这一页最不该有的失效。反过来，如果菜单里全是
    对照组（一支都不该抓到东西），那这一页也测不出任何东西。两个方向一起盯。

    判据从**场景自己的 `expect`** 里读，不从菜单里读：判卷标准只有一个真相源。
    """
    from sandbox.scenario.loader import load_scenario

    checkup = catalog.load_checkup("standard")
    negatives, positives = [], []
    for item in checkup.items:
        scenario = load_scenario(catalog.SCENARIOS_DIR / f"{item.scenario}.yaml")
        clean_run = all(e.verdict.value != "fail" for e in scenario.expect)
        (negatives if clean_run else positives).append(item.id)
    assert negatives, "菜单里一支阴性对照都没有：这样「一次都没报错」读不出任何意思。"
    assert positives, "菜单里每一项都是阴性对照，那这份体检永远不会有结论。"


def test_every_menu_scenario_declares_at_least_one_expectation() -> None:
    """菜单点名的场景必须声明了预期。

    没有预期的话，跑完一条都不命中也不落空，整项会在结论页上落到"没事"那一档，
    而它**一条都没有比过**。`report.py` 里另有一道同样的防线（那一层挡住的是
    运行出来的数据），这里挡的是菜单本身——两层都要，因为这一条在菜单改了之后
    才是第一现场。
    """
    from sandbox.scenario.loader import load_scenario

    checkup = catalog.load_checkup("standard")
    empty = [
        item.scenario
        for item in checkup.items
        if not load_scenario(catalog.SCENARIOS_DIR / f"{item.scenario}.yaml").expect
    ]
    assert empty == [], f"这几支场景一条预期都没写：{empty}"


def test_default_scenarios_dir_is_the_same_place_as_the_web_layer() -> None:
    """`catalog.SCENARIOS_DIR` 与 `api/paths.scenarios_dir()` 必须是同一个地方。

    两处各写一遍 `REPO_ROOT / "scenarios"`，是刻意的（一层管菜单点名的东西在
    哪，一层管 Web 的路径校验），但**漂了会很难发现**：菜单校验会去查一个
    不存在的目录，然后把每一份菜单都报成"场景不存在"。所以拿一道闸门盯着。
    """
    assert paths.scenarios_dir() == catalog.SCENARIOS_DIR


def test_a_menu_pointing_at_a_missing_scenario_refuses_to_load(tmp_path: Path) -> None:
    """菜单点名的场景不存在 → 加载失败，**而且一次报全**。

    报一个改一个要来回好几趟，而菜单是手工维护的，改完文件名往往一次错好几个。
    """
    menus = tmp_path / "checkups"
    menus.mkdir()
    (menus / "broken.yaml").write_text(
        MENU_YAML.replace("scenario: dup", "scenario: nope").replace(
            "scenario: clean", "scenario: also_nope"
        ),
        encoding="utf-8",
    )
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()

    with pytest.raises(catalog.CheckupError) as exc:
        catalog.load_checkup("broken", root=menus, scenarios_root=scenarios)
    assert "nope" in str(exc.value)
    assert "also_nope" in str(exc.value), "两条缺失都要报出来，不能只报第一条"


def test_duplicate_item_ids_are_rejected(tmp_path: Path) -> None:
    """重名的项会让两项共用一个结论——结果页按标识对回菜单。"""
    menus = tmp_path / "checkups"
    menus.mkdir()
    (menus / "dup.yaml").write_text(
        MENU_YAML.replace("id: false_alarm", "id: bad_money"), encoding="utf-8"
    )
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    (scenarios / "dup.yaml").write_text("id: dup\n", encoding="utf-8")
    (scenarios / "clean.yaml").write_text("id: clean\n", encoding="utf-8")

    with pytest.raises(catalog.CheckupError) as exc:
        catalog.load_checkup("dup", root=menus, scenarios_root=scenarios)
    assert "重复" in str(exc.value)


def test_a_menu_without_a_subject_note_is_rejected(tmp_path: Path) -> None:
    """`subject_note` 必填：缺了它，读者会默认这一页在诊断他自己的系统。"""
    menus = tmp_path / "checkups"
    menus.mkdir()
    (menus / "bare.yaml").write_text(
        "\n".join(
            line for line in MENU_YAML.splitlines() if not line.startswith("subject_note")
        ),
        encoding="utf-8",
    )
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    (scenarios / "dup.yaml").write_text("id: dup\n", encoding="utf-8")
    (scenarios / "clean.yaml").write_text("id: clean\n", encoding="utf-8")

    with pytest.raises(catalog.CheckupError) as exc:
        catalog.load_checkup("bare", root=menus, scenarios_root=scenarios)
    assert "subject_note" in str(exc.value)


def test_an_unknown_key_in_the_menu_is_rejected(tmp_path: Path) -> None:
    """写了没人认识的键就是"装上了但没接线"——加载期报错，不安静地忽略。"""
    menus = tmp_path / "checkups"
    menus.mkdir()
    (menus / "extra.yaml").write_text(MENU_YAML + "\nseverity: high\n", encoding="utf-8")
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    for name in ("dup", "clean"):
        (scenarios / f"{name}.yaml").write_text(f"id: {name}\n", encoding="utf-8")

    with pytest.raises(catalog.CheckupError):
        catalog.load_checkup("extra", root=menus, scenarios_root=scenarios)


# --- 2. 结论合成 -------------------------------------------------------------

#: 这八条是**换读法时的八种说错话**，一条一条盯着。


def test_a_missing_expectation_never_reads_as_clear(menu: catalog.Checkup) -> None:
    """该抓到的没抓到 → 「没抓到」，不是「没事」。

    这是整页最容易犯的错：一项该报的没报，折成"这一项做完了，没发现问题"的话，
    用户会以为这一格是绿的，而它其实是空的。两种原因（检查漏了 / 被测对象这次
    没照剧本做）数据上分不清，所以结论只能是"这一项没结论"。
    """
    report = creport.build(
        menu,
        [
            creport.ItemRun(
                item=menu.items[0],
                run_id="r1",
                report=_snapshot(matched=[("idempotency", "pass")], missing=[("loop", "fail")]),
            )
        ],
    )
    outcome = report.items[0]
    assert outcome.kind == "missed"
    assert "没抓到" in outcome.note
    assert report.clear == ()
    assert report.findings == ()


def test_an_unexpected_finding_never_reads_as_a_clean_pass(menu: catalog.Checkup) -> None:
    report = creport.build(
        menu,
        [
            creport.ItemRun(
                item=menu.items[0],
                run_id="r1",
                report=_snapshot(
                    matched=[("idempotency", "pass")], unexpected=[("loop", "fail")]
                ),
            )
        ],
    )
    assert report.items[0].kind == "false_alarm"
    assert report.clear == ()
    assert report.instrument_problems != ()


def test_a_report_with_no_conclusions_is_not_clear(menu: catalog.Checkup) -> None:
    """**读得出来，但一条都没判过** —— 不能落到"没事"。

    真会发生：菜单里挂一支 `expect: []` 的场景。它跑完什么都不命中也不落空，
    落进"没事"那一档的话，页面上写着「这项做完了，没发现问题」，而它一条都
    没有比过。所以 `clear` 要求至少有一条 pass 命中。
    """
    report = creport.build(
        menu, [creport.ItemRun(item=menu.items[0], run_id="r1", report=_snapshot())]
    )
    assert report.items[0].kind == "no_verdict"
    assert report.clear == ()
    assert report.items[0].is_incomplete, "没判过结论属于「没得出可用结论」，不是「没事」"


def test_a_run_with_only_inconclusive_matches_is_not_clear(menu: catalog.Checkup) -> None:
    """命中的全是"判不了"同样不算"没事"。"""
    report = creport.build(
        menu,
        [
            creport.ItemRun(
                item=menu.items[0],
                run_id="r1",
                report=_snapshot(matched=[("idempotency", "inconclusive")]),
            )
        ],
    )
    assert report.items[0].kind == "no_verdict"
    assert report.clear == ()
    assert report.items[0].unknown  # 那一条判不了的仍然被列出来


def test_a_failed_run_is_not_a_verdict(menu: catalog.Checkup) -> None:
    """执行失败 ≠ 不通过。前者是"这次没跑成"，结论不指向被测对象。"""
    report = creport.build(
        menu,
        [
            creport.ItemRun(
                item=menu.items[0], run_id="r1", report=_snapshot(error="boom", ok=False)
            )
        ],
    )
    assert report.items[0].kind == "failed"
    assert report.clear == ()
    assert report.incomplete != ()


def test_a_run_that_never_started_has_no_report(menu: catalog.Checkup) -> None:
    """作业层就失败了：连运行记录都没有，所以一个字都不该说。"""
    report = creport.build(menu, [creport.ItemRun(item=menu.items[0], error="端口没起来")])
    assert report.items[0].kind == "not_run"
    assert "端口没起来" in report.items[0].error
    assert report.items[0].run_ok is None


def test_broken_premises_outrank_the_finding_but_keep_it(menu: catalog.Checkup) -> None:
    """前提不成立时，**档是"前提不成立"，但抓到的那条不能被丢掉**。

    丢掉它等于把证据藏了；只说抓到它又等于忽略"这次结论不指向被测对象"。
    两样都留着，档只决定这一项的头一句话。
    """
    report = creport.build(
        menu,
        [
            creport.ItemRun(
                item=menu.items[0],
                run_id="r1",
                report=_snapshot(matched=[("idempotency", "fail")], premises=False),
            )
        ],
    )
    outcome = report.items[0]
    assert outcome.kind == "premises"
    assert [f.detector for f in outcome.caught] == ["idempotency"]
    assert report.findings == (), "档是「前提不成立」时它不进「抓到的那几项」"


def test_ground_truth_mismatch_is_its_own_thing(menu: catalog.Checkup) -> None:
    """钱扣了几次对不上，和"抓到没抓到"是两件事，档要分开。"""
    report = creport.build(
        menu,
        [
            creport.ItemRun(
                item=menu.items[0], run_id="r1", report=_snapshot(charges=False, ok=False)
            )
        ],
    )
    assert report.items[0].kind == "charges"


def test_the_caught_bucket_holds_the_fail_and_the_note_holds_the_warning(
    menu: catalog.Checkup,
) -> None:
    report = creport.build(
        menu,
        [
            creport.ItemRun(
                item=menu.items[0],
                run_id="r1",
                report=_snapshot(
                    matched=[("idempotency", "fail"), ("loop", "warning"), ("memory", "pass")]
                ),
            )
        ],
    )
    outcome = report.items[0]
    assert outcome.kind == "caught"
    assert [f.detector for f in outcome.caught] == ["idempotency"]
    assert [f.detector for f in outcome.noted] == ["loop"]
    assert outcome.fine == 1, "顺带命中的那条 pass 只数个数"
    assert report.notes == (), "这一项已经进了「抓到的」，不再进「值得看一眼」"


def test_clear_requires_a_pass_and_says_how_many(menu: catalog.Checkup) -> None:
    report = creport.build(
        menu,
        [
            creport.ItemRun(
                item=menu.items[1],
                run_id="r1",
                report=_snapshot(matched=[("idempotency", "pass")]),
            )
        ],
    )
    assert report.items[0].kind == "clear"
    assert "1 条预期全部命中" in report.items[0].note


def test_inconclusive_is_rolled_up_by_detector_with_counts(menu: catalog.Checkup) -> None:
    """判不了的按检测器汇总计数，不逐条铺开。

    实测里 `uncovered` 每次运行都非空（一支没有跨会话的场景，跨会话那一格
    本来就判不了），六项各铺三行会把整页淹掉。
    """
    runs = [
        creport.ItemRun(
            item=menu.items[0],
            run_id="r1",
            report=_snapshot(
                matched=[("idempotency", "pass")],
                uncovered=[("loop.cross_session", "inconclusive"), ("memory", "inconclusive")],
            ),
        ),
        creport.ItemRun(
            item=menu.items[1],
            run_id="r2",
            report=_snapshot(
                matched=[("idempotency", "pass")],
                uncovered=[("loop.cross_session", "inconclusive")],
            ),
        ),
    ]
    report = creport.build(menu, runs)
    rollups = {r.detector: r.count for r in report.unknown}
    assert rollups == {"loop.cross_session": 2, "memory": 1}
    assert report.unknown[0].detector == "loop.cross_session", "条数多的排前面"


def test_the_rollup_order_does_not_depend_on_item_order(menu: catalog.Checkup) -> None:
    """汇总顺序由条数定，不由字典插入顺序定。

    插入顺序跟着项的顺序走，而那是可以变的——同一批数据排成两种顺序，读的人
    会以为结论变了。这条规矩在本项目里是有前科的（`PYTHONHASHSEED` 那次）。
    """
    a = creport.ItemRun(
        item=menu.items[0],
        run_id="r1",
        report=_snapshot(
            matched=[("idempotency", "pass")], uncovered=[("memory", "inconclusive")]
        ),
    )
    b = creport.ItemRun(
        item=menu.items[1],
        run_id="r2",
        report=_snapshot(
            matched=[("idempotency", "pass")], uncovered=[("autonomy", "inconclusive")]
        ),
    )
    first = creport.build(menu, [a, b])
    second = creport.build(menu, [b, a])
    assert [r.detector for r in first.unknown] == [r.detector for r in second.unknown]


def test_identical_findings_are_merged_but_not_near_identical_ones(
    menu: catalog.Checkup,
) -> None:
    """一字不差的几条结论并成一条带次数；**差一个字的就不许并**。

    实测里 `schema_rejected` 那一项会印出三行一模一样的「Agent 调用 'list_orders'
    的参数不符合场景声明的 schema」——检测器对每一次被拒的调用各报一条，而这几条
    恰好一个字都不差。三行同样的字不是信息，是噪音，而**噪音一多，真结论就没人
    看了**。

    两边都要钉：只钉"会并"的话，把两条**不同**的结论并成一条也照样绿——那正是
    这份报告最怕的错（替结论说它没说过的话）。所以下面第二段里那两条只差一个
    句号，必须仍然是两行。
    """
    same = creport.build(
        menu,
        [
            creport.ItemRun(
                item=menu.items[0],
                run_id="r1",
                report=_snapshot(matched=[("idempotency", "fail")] * 3),
            )
        ],
    )
    caught = same.items[0].caught
    assert len(caught) == 1, f"一字不差的三条应当并成一条，实际 {len(caught)} 条"
    assert caught[0].count == 3, "次数要跟着合并一起记下来——不记的话，报了三次和报了一次长得一样"

    def _matched(detector: str, got: str) -> dict[str, Any]:
        return {
            "expect": {"detector": detector, "verdict": "fail", "reason_code": None},
            "got": got,
        }

    different = creport.build(
        menu,
        [
            creport.ItemRun(
                item=menu.items[0],
                run_id="r1",
                report=_snapshot(
                    expectation={
                        "matched": [
                            _matched("loop", "它又调了一次。"),
                            _matched("loop", "它又调了一次"),
                        ],
                        "missing": [],
                        "unexpected": [],
                        "uncovered": [],
                    }
                ),
            )
        ],
    )
    two = different.items[0].caught
    assert len(two) == 2, "只差一个句号的两条是不同的结论，不许并成一条"
    assert [f.count for f in two] == [1, 1]


async def test_the_merged_count_is_visible_on_the_page(
    client: httpx.AsyncClient, runner_state: dict[str, Any]
) -> None:
    """次数要印在页面上。

    合并了却不印次数，等于把"报了三次"读成"报了一次"——那是**报告说了比观测更少的
    话**，和另一头那个毛病一样坏。这一条与上一条分开：上一条管数据，这一条管页面；
    只钉数据的话，把模板里那个 `（N 次）` 删掉不会红。
    """
    saved = _FAKE_REPORTS["dup"]
    _FAKE_REPORTS["dup"] = _snapshot(matched=[("idempotency", "fail")] * 3)
    try:
        url = await _run_checkup(client, runner_state)
        page = await client.get(url)
    finally:
        _FAKE_REPORTS["dup"] = saved
    assert "（3 次）" in page.text, "合并之后次数没印出来"


def test_a_malformed_snapshot_does_not_take_the_page_down(menu: catalog.Checkup) -> None:
    """一份读不出来的快照只让自己那一项变成"读不出来"，其余各项照常。"""
    report = creport.build(
        menu,
        [
            creport.ItemRun(
                item=menu.items[0],
                run_id="bad",
                report={"expectation": {"matched": "不是一串"}},
            ),
            creport.ItemRun(
                item=menu.items[1],
                run_id="good",
                report=_snapshot(matched=[("idempotency", "pass")]),
            ),
        ],
    )
    assert report.items[0].kind == "unreadable"
    assert report.items[1].kind == "clear"


def test_a_snapshot_without_the_expectation_section_is_not_clear(
    menu: catalog.Checkup,
) -> None:
    """老报告（那一节存在之前跑的）没有预期比对 → 不能读成"没事"。"""
    report = creport.build(
        menu,
        [
            creport.ItemRun(
                item=menu.items[0], run_id="old", report=_snapshot(expectation=None)
            )
        ],
    )
    assert report.items[0].kind == "unreadable"
    assert report.clear == ()


def test_the_no_finding_caveat_says_what_it_does_not_know() -> None:
    """「一项都没抓到」时必须跟着那句"不等于它不会犯"。"""
    assert "不等于" in creport.NO_FINDING_CAVEAT
    assert "剧本" in creport.NO_FINDING_CAVEAT


def test_the_headline_only_states_counts(menu: catalog.Checkup) -> None:
    """顶栏那句话只数数，不替结论下判断。"""
    report = creport.build(
        menu,
        [
            creport.ItemRun(
                item=menu.items[0],
                run_id="r1",
                report=_snapshot(matched=[("idempotency", "fail")]),
            ),
            creport.ItemRun(item=menu.items[1], error="没起来"),
        ],
    )
    assert "2 项里跑成了 1 项" in report.headline
    assert "抓到 1 类问题" in report.headline
    assert "1 项没得出可用结论" in report.headline


# --- 3. 路由与渲染 -----------------------------------------------------------

CLEAN_YAML = _scenario_yaml("clean", detector="idempotency", verdict="pass")
DUP_YAML = _scenario_yaml("dup", detector="idempotency", verdict="fail")

#: 替身执行器按场景名给一份不同的报告快照。**这是"结论怎么排"那一段的输入**，
#: 真跑出来的形状由引擎的端到端测试保证。
_FAKE_REPORTS: dict[str, dict[str, Any]] = {
    # 演"重复扣款"的那一支：预期命中一条 fail。运行级的 `ok` 是 True——
    # **"通过"在体检里不是好话**，它说的是"预期达成了"，而这一项的预期正是
    # "应当抓到重复扣款"。这一条特意写出来，免得哪天有人拿 ok 去定档。
    "dup": _snapshot(matched=[("idempotency", "fail")], ok=True),
    "clean": _snapshot(matched=[("idempotency", "pass")], ok=True),
}


class _FakeRunner:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state

    async def run(self, scenario: Any) -> Any:  # type: ignore[no-untyped-def]
        from sandbox.scenario.runner import ScenarioResult

        while not self.state["release"].is_set():
            await asyncio.sleep(0.01)
        self.state["ran"].append(scenario.id)
        run_id = f"run-{scenario.id}"
        async with session_scope() as db:
            db.add(
                TestRun(
                    id=run_id,
                    scenario_id=scenario.id,
                    agent_version="react_demo@0.1",
                    config_hash="c" * 64,
                    status="completed",
                    started_at=datetime.now(UTC),
                    completed_at=datetime.now(UTC),
                    verdict_ok=True,
                    report_json=json.dumps(_FAKE_REPORTS[scenario.id], ensure_ascii=False),
                )
            )
        # `ok` 是 `ScenarioResult` 上的**属性**，不是字段（它由 error /
        # premises_held / charges_ok / expectation 算出来）——这里传不进去，
        # 也不该传：这一层要的是运行记录里的报告快照，不是这个对象。
        return ScenarioResult(scenario_id=scenario.id, run_id=run_id)


@pytest.fixture
def runner_state() -> dict[str, Any]:
    return {"release": threading.Event(), "ran": []}


@pytest_asyncio.fixture
async def client(
    temp_db: str, checkup_roots: tuple[Path, Path], runner_state: dict[str, Any]
) -> AsyncIterator[httpx.AsyncClient]:
    scenarios, menus = checkup_roots
    await init_db()
    app = create_app(
        scenarios_root=scenarios,
        checkups_root=menus,
        registry=JobRegistry(runner_factory=_FakeRunner),
    )
    app.state.jobs._runner_factory = lambda: _FakeRunner(runner_state)  # noqa: SLF001
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://ui") as c:
        yield c


async def _run_checkup(client: httpx.AsyncClient, runner_state: dict[str, Any]) -> str:
    """点一次体检，等它跑完，返回作业页地址。"""
    started = await client.post("/checkup/mini/start", follow_redirects=False)
    assert started.status_code == 303, started.text
    url = started.headers["location"]
    runner_state["release"].set()
    for _ in range(200):
        page = await client.get(url)
        if "正在体检" not in page.text:
            return url
        await asyncio.sleep(0.02)
    raise AssertionError("体检一直没跑完")


async def test_the_menu_page_lists_the_question_not_just_the_scenario_name(
    client: httpx.AsyncClient,
) -> None:
    page = await client.get("/checkup")
    assert page.status_code == 200
    assert "同一笔钱，它会不会付两次？" in page.text
    assert "这一页测的是仓库里自带的演示 Agent" in page.text


async def test_a_broken_menu_is_listed_with_the_reason(tmp_path: Path) -> None:
    """打不开的菜单也列出来，并说清为什么——和场景列表同一条规矩。"""
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    menus = tmp_path / "checkups"
    menus.mkdir()
    (menus / "broken.yaml").write_text(
        MENU_YAML.replace("scenario: dup", "scenario: nowhere"), encoding="utf-8"
    )
    # 把能跑的那两项对应的场景建出来，只留一个不存在
    (scenarios / "clean.yaml").write_text(CLEAN_YAML, encoding="utf-8")

    await init_db()
    app = create_app(scenarios_root=scenarios, checkups_root=menus)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://ui") as c:
        page = await c.get("/checkup")
    assert page.status_code == 200
    assert "这份菜单打不开" in page.text
    assert "nowhere" in page.text


async def test_starting_a_checkup_runs_every_item_in_order(
    client: httpx.AsyncClient, runner_state: dict[str, Any]
) -> None:
    """一趟体检把每一项都跑一遍，顺序照菜单。"""
    url = await _run_checkup(client, runner_state)
    assert runner_state["ran"] == ["dup", "clean"]
    page = await client.get(url)
    assert page.status_code == 200


async def test_the_result_page_shows_the_question_and_the_plain_language(
    client: httpx.AsyncClient, runner_state: dict[str, Any]
) -> None:
    url = await _run_checkup(client, runner_state)
    page = await client.get(url)
    assert "同一笔钱，它会不会付两次？" in page.text
    assert "抓到了" in page.text
    # 说人话那一层要跟着上：检测器名旁边必须有一句人话解释。**断言的是
    # `explain` 里那一份原文**，不在测试里手抄——手抄的话，那句话改了这边
    # 还是绿的，而它是这一页存在的理由。
    assert explain.DETECTOR_HELP["idempotency"] in page.text
    # 每一项都能点回各自的运行记录
    assert "/runs/run-dup" in page.text
    assert "/runs/run-clean" in page.text


async def test_the_result_page_keeps_inconclusive_out_of_the_good_news(
    client: httpx.AsyncClient, runner_state: dict[str, Any]
) -> None:
    """判不了的东西要单独列出来，而且不能读成"没事"。"""
    _FAKE_REPORTS["clean"] = _snapshot(
        matched=[("idempotency", "pass")], uncovered=[("memory", "inconclusive")]
    )
    try:
        url = await _run_checkup(client, runner_state)
        page = await client.get(url)
    finally:
        _FAKE_REPORTS["clean"] = _snapshot(matched=[("idempotency", "pass")])
    assert "这次判不了的" in page.text
    assert "判不了的意思是" in page.text
    assert "memory" in page.text


async def test_the_result_page_says_when_nothing_was_caught(
    client: httpx.AsyncClient, runner_state: dict[str, Any]
) -> None:
    """一项都没抓到时要跟着那句"不等于它不会犯"。"""
    saved_dup = _FAKE_REPORTS["dup"]
    _FAKE_REPORTS["dup"] = _snapshot(matched=[("idempotency", "pass")])
    try:
        url = await _run_checkup(client, runner_state)
        page = await client.get(url)
    finally:
        _FAKE_REPORTS["dup"] = saved_dup
    assert "这类毛病" in page.text or "这一类毛病" in page.text
    assert "不等于它不会犯" in page.text


async def test_the_question_survives_the_menu_being_edited_mid_run(
    client: httpx.AsyncClient, runner_state: dict[str, Any], checkup_roots: tuple[Path, Path]
) -> None:
    """跑到一半菜单被改过 → 对不上的项**原样列出来**，不按下标硬对。

    按下标硬对的话，A 项的结论会出现在 B 项的问题底下，而那一页读起来完全正常。
    """
    _, menus = checkup_roots
    started = await client.post("/checkup/mini/start", follow_redirects=False)
    url = started.headers["location"]
    # 趁它还在跑，把菜单换掉：标识全改
    (menus / "mini.yaml").write_text(
        MENU_YAML.replace("id: bad_money", "id: renamed_money").replace(
            "id: false_alarm", "id: renamed_alarm"
        ),
        encoding="utf-8",
    )
    runner_state["release"].set()
    for _ in range(200):
        page = await client.get(url)
        if "正在体检" not in page.text:
            break
        await asyncio.sleep(0.02)

    assert "菜单被改过" in page.text
    assert page.text.count("对不回菜单") >= 1


async def test_a_checkup_refuses_to_start_while_another_run_is_going(
    client: httpx.AsyncClient, runner_state: dict[str, Any]
) -> None:
    """执行位只有一格：体检也走同一条 409，而且要说清是谁占着。"""
    first = await client.post("/checkup/mini/start", follow_redirects=False)
    assert first.status_code == 303
    second = await client.post("/checkup/mini/start", follow_redirects=False)
    assert second.status_code == 409
    assert "已经有一次运行在跑" in second.text
    assert "体检" in second.text, "体检占着执行位时要说「体检」，不能只说某个场景名"
    runner_state["release"].set()
    for _ in range(200):
        page = await client.get(first.headers["location"])
        if "正在体检" not in page.text:
            break
        await asyncio.sleep(0.02)


async def test_the_reason_a_checkup_cannot_take_a_missing_scenario(
    client: httpx.AsyncClient, checkup_roots: tuple[Path, Path]
) -> None:
    """菜单里的场景文件在开跑之前就没了 → 400，说清是哪个文件。

    开跑之后才发现的话，前几项已经花掉了时间，而那份结论会缺一块。
    """
    scenarios, _ = checkup_roots
    (scenarios / "dup.yaml").unlink()
    resp = await client.post("/checkup/mini/start", follow_redirects=False)
    assert resp.status_code == 400
    assert "dup" in resp.text
