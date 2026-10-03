"""报告页上「这条我认可是误报」这条出口。

豁免机制引擎侧一直是完整的（`ExemptionDecl`、加载期校验、`EXEMPTION_AWARE`），
而界面上一个入口都没有——用户看到一条他认为冤枉的红，只能忍着。这个文件守的是
那个入口，以及它**最要紧的那条性质：不是每一条结论旁边都能点**。

19 个检测器名里能豁免的只有 6 个，记忆族 3 个另有出口（场景里的 `must_hold` /
`forbidden`），剩下 10 个那条路上压根没有查豁免那一步。三者渲染成一个样子的话，
用户点到后两种上就会撞见一个装上了但不生效的开关——而豁免整个机制存在的意义
就是"让结论消失"，一个静音不了的静音开关比没有更坏。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio

from sandbox.api import paths
from sandbox.api.app import create_app
from sandbox.db.engine import init_db
from sandbox.db.models import TestRun
from sandbox.detectors import NOT_EXEMPTIBLE, SCENARIO_DETECTOR_NAMES
from sandbox.detectors import base as detectors_base
from sandbox.detectors.base import DetectorVerdict, Finding
from sandbox.scenario import loader
from sandbox.scenario.runner import ScenarioResult

#: 场景里有**一个会写世界的工具和一个不会的**。两件事各需要一个工具：
#: 写属性是"要不要多勾一次确认"的唯一判据，而它按场景声明算，不按调用算。
SCENARIO_YAML = """
id: refund
description: 一张用来试豁免入口的场景。
agent:
  version: react_demo@0.1
  task: 帮我把 A1 退了。
  model: stub-model
tools:
  - name: get_order
    description: 查订单
    is_write: false
  - name: create_payment
    description: 发起一笔付款
    is_write: true
sessions:
  - name: 第一次
    script:
      - tool_calls: [{name: get_order, arguments: {order_id: A1}}]
expect:
  - detector: idempotency
    verdict: pass
"""

def _finding(
    detector: str, *, tool: str | None = None, verdict: DetectorVerdict = DetectorVerdict.FAIL
) -> Finding:
    detail: dict[str, Any] = {"reason_code": "missing_key"}
    if tool is not None:
        detail["tool_name"] = tool
    return Finding(
        detector=detector,
        verdict=verdict,
        summary=f"{detector} 说这里有问题。",
        detail=detail,
    )


@dataclass
class Ui:
    client: httpx.AsyncClient
    root: Path

    def text(self, scenario_id: str = "refund") -> str:
        return (self.root / f"{scenario_id}.yaml").read_text(encoding="utf-8")


@pytest_asyncio.fixture
async def ui(temp_db: str, tmp_path: Path) -> AsyncIterator[Ui]:
    await init_db()
    root = tmp_path / "scenarios"
    root.mkdir()
    (root / "refund.yaml").write_text(SCENARIO_YAML, encoding="utf-8")
    app = create_app(scenarios_root=root)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://ui") as client:
        yield Ui(client=client, root=root)


async def _seed(
    run_id: str, findings: list[Finding], *, scenario_id: str = "refund"
) -> None:
    """塞一次带报告快照的运行，够运行页把每条结论和它的出口渲染出来。

    快照**问运行器要**（`ScenarioResult.to_dict()`），不手写一个更瘦的字典：
    这一页还要读 `premises_held` / `charges_ok` / `egress_charges` 这些键，
    手写的那个会缺，而缺的表现是渲染到一半抛 `UndefinedError`——那种夹具
    和被测代码用的是同一份"我以为的形状"，谁也没量过真的那一份。
    """
    from sandbox.db.engine import SessionFactory

    report = ScenarioResult(
        scenario_id=scenario_id,
        run_id=run_id,
        findings=list(findings),
        egress_charges=0,
        expected_charges=0,
    ).to_dict()
    async with SessionFactory() as db:
        db.add(
            TestRun(
                id=run_id,
                scenario_id=scenario_id,
                agent_version="react_demo@0.1",
                config_hash="E" * 64,
                session_sequence_id=f"seq-{run_id}",
                status="completed",
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                verdict_ok=False,
                report_json=json.dumps(report),
            )
        )
        await db.commit()


async def _page(ui: Ui, run_id: str, findings: list[dict], **kw: Any) -> str:
    await _seed(run_id, findings, **kw)
    response = await ui.client.get(f"/runs/{run_id}")
    assert response.status_code == 200, response.text
    return response.text


def _cell(html: str, detector: str) -> str:
    """页面上那一条结论所在的 `<tr>`。

    按渲染出来的样子切，不按内部数据结构——这一页是人读的，一条结论做成什么样
    就是它给用户看到的东西。
    """
    rows = html.split("<tr>")
    for row in rows:
        if f"<code>{detector}</code>" in row:
            return row
    raise AssertionError(f"页面上找不到 {detector} 那条结论")


# --- 三种情形，各一条 --------------------------------------------------------


@pytest.mark.asyncio
async def test_a_finding_that_can_be_exempted_gets_the_form(ui: Ui) -> None:
    """能豁免的那一类：摆表单，而且表单里的检测器名就是这一条的。"""
    html = await _page(ui, "run-can", [_finding("idempotency", tool="create_payment")])
    cell = _cell(html, "idempotency")

    assert 'action="/runs/run-can/exempt"' in cell
    assert 'name="detector" value="idempotency"' in cell
    assert 'name="reason"' in cell, "理由栏必须在——空理由在加载期就会被拒"
    assert "写进场景文件" in cell


@pytest.mark.asyncio
async def test_a_memory_finding_points_at_the_scenario_instead(ui: Ui) -> None:
    """记忆族：**有**例外机制，但不是豁免——指去 `must_hold` / `forbidden`。

    给它摆一个豁免按钮的话，用户点下去会看到"这一族不吃豁免"的 400，而他本来
    要办的事（放过这一条）在场景文件里其实是有地方办的。
    """
    html = await _page(ui, "run-mem", [_finding("memory.drift")])
    cell = _cell(html, "memory.drift")

    assert 'action="/runs/run-mem/exempt"' not in cell
    assert "must_hold" in cell and "forbidden" in cell
    assert "记忆族不吃豁免" in cell


@pytest.mark.asyncio
async def test_a_detector_with_no_exit_says_so(ui: Ui) -> None:
    """没有出口的那一类：**如实说**，不摆一个按不动的按钮。"""
    html = await _page(ui, "run-none", [_finding("loop")])
    cell = _cell(html, "loop")

    assert 'action="/runs/run-none/exempt"' not in cell
    assert "今天没有出口" in cell


@pytest.mark.asyncio
async def test_two_findings_from_one_detector_each_get_their_own_row(ui: Ui) -> None:
    """同一个检测器在一次运行里可以出现好几条，各说各的。

    键是 `(检测器, 报告里的序号)`，序号用的是**原始顺序**——另排一遍的话，
    页面上的第 3 条和报告里的第 3 条会是两条不同的结论。
    """
    html = await _page(
        ui,
        "run-twice",
        [
            _finding("idempotency", tool="create_payment"),
            _finding("loop.strict", tool="get_order"),
        ],
    )
    first = _cell(html, "idempotency")
    second = _cell(html, "loop.strict")

    assert 'name="tool" value="create_payment"' in first
    assert 'name="tool" value="get_order"' in second


# --- 那条承重的：名字集合现算，不手抄 ----------------------------------------


@pytest.mark.asyncio
async def test_the_form_appears_for_exactly_the_detectors_that_look_for_exemptions(
    ui: Ui,
) -> None:
    """页面上有表单的检测器，恰好等于**真的会去查豁免**的那些。**集合现算。**

    手抄那 6 个名字的话，哪天检测器的豁免能力变了（有人给 `loop.cross_session`
    接上豁免，或者把 `autonomy` 从 `EXEMPTION_AWARE` 里拿掉），页面会继续对着
    一条永远不生效的路给按钮——而它看起来和"这条路是通的"一模一样。

    所以这里不写死任何一个名字：真值从 `base.EXEMPTION_AWARE` 的键现算，
    被测的那一边是模板渲染出来的东西。两边同时漂才可能骗过这条断言。
    """
    names = sorted(SCENARIO_DETECTOR_NAMES)
    # 一个运行里塞 19 条结论太重；一条结论一个运行，页面就只渲染那一条。
    for name in names:
        html = await _page(ui, f"run-{name}", [_finding(name)])
        cell = _cell(html, name)
        has_form = f'action="/runs/run-{name}/exempt"' in cell
        assert has_form == (name in detectors_base.EXEMPTION_AWARE), (
            f"{name!r} 在页面上的出口和它真实的豁免能力对不上："
            f"页面给了表单={has_form}，"
            f"而它在 EXEMPTION_AWARE 里={name in detectors_base.EXEMPTION_AWARE}。"
        )


def test_that_assertion_is_not_vacuous() -> None:
    """上一条只有在两个集合都非空、且不相等时才说明得了问题。

    全都不可豁免（或全都可以）的话，它一路绿灯，而它想钉的那件事一个字都没验到。
    这条不碰界面，所以不起应用——它管的是"上面那条断言还有没有内容"。
    """
    aware = set(detectors_base.EXEMPTION_AWARE)
    assert aware, "一个能豁免的检测器都没有？那这个入口整个是死的"
    assert NOT_EXEMPTIBLE, "记忆族是空的？"
    assert not (aware & NOT_EXEMPTIBLE), "一个检测器不可能两个表里都有"
    assert aware < set(SCENARIO_DETECTOR_NAMES), (
        "能豁免的集合等于全部检测器——那上一条断言就退化成「处处都是表单」了"
    )


# --- 真写进去 ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_clicking_it_writes_a_rule_into_the_scenario_file(ui: Ui) -> None:
    """点一次，场景文件里真的多一条豁免，而且这份文件还读得出来。"""
    await _seed("run-write", [_finding("idempotency", tool="create_payment")])
    before = ui.text()

    response = await ui.client.post(
        "/runs/run-write/exempt",
        data={
            "detector": "idempotency",
            "tool": "create_payment",
            "reason": "这笔是客户自己点的，不是重复扣款",
            "acknowledge_write": "1",
        },
    )

    assert response.status_code == 303, response.text
    assert response.headers["location"] == "/runs/run-write?exempted=1"

    after = ui.text()
    assert after != before
    # 原有内容一个字都不能被这次手术吃掉（这份文件里本来没有 `exemptions:`，
    # 所以规则是接在末尾的）。
    assert after.startswith(before)
    scenario = loader.parse_scenario(after, source="<写完之后>")
    assert len(scenario.exemptions) == 1
    decl = scenario.exemptions[0]
    assert decl.detector == "idempotency"
    assert decl.tool == "create_payment"
    assert decl.reason == "这笔是客户自己点的，不是重复扣款"
    assert decl.allow_write is True, "写工具的结论上，那次确认要落到文件里"


@pytest.mark.asyncio
async def test_the_write_says_so_on_the_next_page(ui: Ui) -> None:
    """写成功之后有回执。

    没有回执的话，「写成功了」和「点了个寂寞」在屏幕上长得一模一样——这一页上
    的报告一个字都不会变（快照是当时存下来的），用户会再点一次，于是场景文件里
    多一条重复的规则。静默的成功和静默的失败分不开。
    """
    await _seed("run-flash", [_finding("idempotency", tool="create_payment")])
    response = await ui.client.post(
        "/runs/run-flash/exempt",
        data={
            "detector": "idempotency",
            "tool": "create_payment",
            "reason": "认了",
            "acknowledge_write": "1",
        },
    )
    followed = await ui.client.get(response.headers["location"])
    assert followed.status_code == 200

    assert "写下去了" in followed.text
    # 回执必须说清"改的是以后的运行"——不然用户盯着没变的红会以为自己没点成功。
    assert "一个字都不会变" in followed.text
    # 以及代价。
    assert "配置指纹" in followed.text


@pytest.mark.asyncio
async def test_the_page_does_not_claim_a_write_that_did_not_happen(ui: Ui) -> None:
    """没写过就不许有回执。假回执比没有回执更坏：它会让人以为规则已经在了。"""
    html = await _page(ui, "run-noflash", [_finding("idempotency")])
    assert "写下去了" not in html


@pytest.mark.asyncio
async def test_a_read_only_tool_does_not_ask_for_the_extra_confirmation(ui: Ui) -> None:
    """只会读的工具不该多问一次。

    多问的那一次要**贵**才有意义——处处都问的话，用户会养成闭眼勾的习惯，
    于是真正会改世界的那一次也被顺手勾掉了。
    """
    html = await _page(ui, "run-read", [_finding("idempotency", tool="get_order")])
    cell = _cell(html, "idempotency")
    assert 'name="acknowledge_write"' not in cell


@pytest.mark.asyncio
async def test_a_write_tool_asks_for_the_extra_confirmation(ui: Ui) -> None:
    html = await _page(ui, "run-ack", [_finding("idempotency", tool="create_payment")])
    assert 'name="acknowledge_write"' in _cell(html, "idempotency")


@pytest.mark.asyncio
async def test_a_finding_that_names_no_tool_covers_the_write_side_too(ui: Ui) -> None:
    """没指名工具的结论，覆盖面按"这条规则能命中哪些调用"算。

    没指名就是所有工具，其中只要有一个会改世界，这条规则就可能落在写调用上——
    所以那一次确认照样要问。判据用的是**场景声明的**写属性，不是调用。
    """
    html = await _page(ui, "run-all", [_finding("idempotency")])
    cell = _cell(html, "idempotency")
    assert "<select name=\"tool\"" in cell, "没指名工具时给的是下拉框（含『所有工具』）"
    assert 'name="acknowledge_write"' in cell


# --- 拒绝路径：每条都要说清该改什么 ------------------------------------------


@pytest.mark.asyncio
async def test_exempting_a_memory_finding_is_refused_with_a_way_out(ui: Ui) -> None:
    """界面不给按钮，但地址栏是用户改得了的——服务端也得拒，并且指路。"""
    await _seed("run-memory-post", [_finding("memory")])
    response = await ui.client.post(
        "/runs/run-memory-post/exempt", data={"detector": "memory", "reason": "认了"}
    )
    assert response.status_code == 400
    assert "memory" in response.text
    assert "must_hold" in response.text


@pytest.mark.asyncio
async def test_exempting_a_no_exit_detector_is_refused(ui: Ui) -> None:
    await _seed("run-none-post", [_finding("loop")])
    response = await ui.client.post(
        "/runs/run-none-post/exempt", data={"detector": "loop", "reason": "认了"}
    )
    assert response.status_code == 400
    assert "没有出口" in response.text


@pytest.mark.asyncio
async def test_a_misspelled_detector_name_is_not_quietly_filed_as_no_exit(ui: Ui) -> None:
    """拼错的名字**报错**，不归进「没有出口」。

    归进去的话，一个拼错的名字会看起来像一条真的办不到的结论：用户看到的是
    「这一类今天没有出口」，而真相是他打错了一个字母。
    """
    await _seed("run-typo", [_finding("idempotency")])
    with pytest.raises(ValueError, match="不认识的检测器名"):
        await ui.client.post(
            "/runs/run-typo/exempt", data={"detector": "idempotencey", "reason": "认了"}
        )


@pytest.mark.asyncio
async def test_an_empty_reason_is_refused(ui: Ui) -> None:
    """理由必填。没有理由的豁免在报告里无法复核，等于永久静音。"""
    await _seed("run-empty", [_finding("idempotency", tool="get_order")])
    response = await ui.client.post(
        "/runs/run-empty/exempt", data={"detector": "idempotency", "tool": "get_order"}
    )
    assert response.status_code == 400
    assert "理由" in response.text
    assert ui.text() == SCENARIO_YAML, "拒掉了就不许动文件"


@pytest.mark.asyncio
async def test_a_write_finding_without_the_confirmation_is_refused(ui: Ui) -> None:
    """会改世界的调用要多一道确认，**服务端也要拦**——表单上那个勾是提示，不是闸门。"""
    await _seed("run-unack", [_finding("idempotency", tool="create_payment")])
    response = await ui.client.post(
        "/runs/run-unack/exempt",
        data={"detector": "idempotency", "tool": "create_payment", "reason": "认了"},
    )
    assert response.status_code == 400
    assert "会改世界" in response.text
    assert ui.text() == SCENARIO_YAML


@pytest.mark.asyncio
async def test_a_tool_that_is_not_declared_is_refused(ui: Ui) -> None:
    """指名一个场景里没声明过的工具，规则永远不会命中。

    这和"给没有出口的检测器写规则"是同一类错：文件里多了一条看着像有用的规则，
    而它一辈子不会生效。往可见的一侧错——当场拒。
    """
    await _seed("run-unknown-tool", [_finding("idempotency")])
    response = await ui.client.post(
        "/runs/run-unknown-tool/exempt",
        data={"detector": "idempotency", "tool": "send_email", "reason": "认了"},
    )
    assert response.status_code == 400
    assert "send_email" in response.text
    assert ui.text() == SCENARIO_YAML


@pytest.mark.asyncio
async def test_a_run_whose_scenario_file_is_gone_says_so(ui: Ui) -> None:
    """场景文件被删了或改坏了：**没有"以后的运行"可以改**，一个按钮都不摆。

    照实说，而不是摆一个点下去报错的按钮。
    """
    paths.scenario_file("refund", root=ui.root).unlink()
    html = await _page(ui, "run-gone", [_finding("idempotency", tool="create_payment")])
    cell = _cell(html, "idempotency")

    assert 'action="/runs/run-gone/exempt"' not in cell
    assert "打不开" in cell


@pytest.mark.asyncio
async def test_a_run_without_a_report_snapshot_still_opens(ui: Ui) -> None:
    """没有结论快照的老运行，这一页照样打得开。

    这一条是 mypy 抓出来的，不是有人先想到的：`_finding_actions` 的签名原先写死了
    `dict`，而 `parse_report` 真的会返回 `None`——于是"结论落库"这个字段存在之前
    跑的每一次运行，点进去就是一个 500。**收窄类型的地方正是"某个分支到底会不会
    走到"最容易想当然的地方**，而这里的想当然有一个具体的代价。

    所以这条测试钉的不是豁免，是那个 `None`：把签名改回 `dict` 它不会红（运行期
    才炸），但把 `if report is None: return {}` 那两行删掉它会红。
    """
    from sandbox.db.engine import SessionFactory

    async with SessionFactory() as db:
        db.add(
            TestRun(
                id="run-nosnapshot",
                scenario_id="refund",
                agent_version="react_demo@0.1",
                config_hash="E" * 64,
                session_sequence_id="seq-run-nosnapshot",
                status="completed",
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                verdict_ok=None,
                report_json=None,
            )
        )
        await db.commit()

    response = await ui.client.get("/runs/run-nosnapshot")
    assert response.status_code == 200, response.text
    assert "没有留下结论快照" in response.text


@pytest.mark.asyncio
async def test_a_second_click_appends_a_second_rule_rather_than_rewriting(ui: Ui) -> None:
    """第二次点是在**已有豁免**的基础上追加，不是把上一个换掉。

    手写的场景里可能已经有别的原因写下的规则；这一次手术要保住它们。这也是
    `append_decl` 存在的理由：不重新序列化整份 YAML（那会删掉所有注释）。
    """
    await _seed("run-second", [_finding("idempotency", tool="get_order")])
    data = {"detector": "idempotency", "tool": "get_order", "reason": "第一次"}
    assert (await ui.client.post("/runs/run-second/exempt", data=data)).status_code == 303
    data["reason"] = "第二次，另一个理由"
    assert (await ui.client.post("/runs/run-second/exempt", data=data)).status_code == 303

    scenario = loader.parse_scenario(ui.text(), source="<两次之后>")
    assert [d.reason for d in scenario.exemptions] == ["第一次", "第二次，另一个理由"]


# --- 页面上印着的那句代价，得真有东西在核 ------------------------------------


def test_the_cost_the_page_prints_is_real() -> None:
    """页面上写着「它会进配置指纹，回归对比会变成『改的是测试本身』」。

    那句话是**产品对用户的一句承诺**，所以它必须有东西在核——不然哪天
    `exemptions` 从 `SECTION_SOURCES["expect"]` 掉出去（或者 `expect` 从
    `TEST_SECTIONS` 掉出去），页面会继续信誓旦旦地把这个代价说给用户听，
    而它已经不成立了。用户按这句话做的决定（"先不豁免，我重跑一次看看"）
    就是白做的。

    这里走的就是页面上那条路：`exempt.append_decl` 把一条规则接进场景文本，
    两边各自解析，然后问**回归对比自己**这两份是不是同一套测试。
    """
    from sandbox.regression import diff
    from sandbox.scenario import identity

    after_text = append_exemption(SCENARIO_YAML, reason="这笔是客户自己点的")
    before = loader.parse_scenario(SCENARIO_YAML, source="<加豁免之前>")
    after = loader.parse_scenario(after_text, source="<加豁免之后>")

    changed = identity.changed_sections(identity.sections(before), identity.sections(after))
    assert changed == ["expect"], (
        f"加一条豁免只该动 `expect` 那一节，实际动了 {changed}——"
        f"多动的那几节会让指纹报出别的名目，页面上那句代价就说错了地方"
    )
    assert set(changed) <= identity.TEST_SECTIONS, (
        "`expect` 必须在 TEST_SECTIONS 里，否则回归对比报的是「配置变了」"
        "（像是在说被测对象换了），而不是「改的是测试本身」"
    )

    # 页面上写的是「进配置指纹」，所以**指纹本身**得真的变——只有节的名单变了
    # 而哈希没变的话，落库的那个 `config_hash` 还是老的，回归对比据此认出来的
    # 是"同一次配置"，那句话就只对了一半。
    assert _snapshot(before).config_hash != _snapshot(after).config_hash, (
        "加了一条豁免而配置指纹没变——那回归对比会把它当成同一套测试"
    )

    verdict, sections, notes = diff.comparability(
        _snapshot(before), _snapshot(after)
    )
    assert verdict == diff.TEST_CHANGED, f"回归对比说 {verdict!r}：{notes}"
    assert sections == ["expect"]


def _snapshot(scenario: object):
    from sandbox.regression.diff import Snapshot
    from sandbox.scenario import identity
    from sandbox.trace.recorder import CONFIG_SCHEME, compute_config_hash

    parts = identity.sections(scenario)  # type: ignore[arg-type]
    return Snapshot(
        run_id="",
        verdict_ok=True,
        report=None,
        config_scheme=CONFIG_SCHEME,
        config_hash=compute_config_hash(parts),
        config_parts=parts,
        env=None,
    )


def append_exemption(text: str, *, reason: str) -> str:
    """走页面上那条路（同两个函数），把一条豁免接进场景文本。"""
    from sandbox.api import exempt

    return exempt.append_decl(
        text, exempt.render_decl(detector="idempotency", tool="create_payment", reason=reason)
    )
