"""新建向导：人话问句 → 确认页 → 一份能跑的场景。

向导自己不判任何东西，所以它出错的方式和引擎不一样——**它错在一份看起来完全
正常的场景上**。这个文件盯的就是那三种：

**一、极性写反。** `expect` 的语义是「我预期会发生什么」，不是「我要查什么」。
勾「我预期会被抓到」写进去的是 `verdict: fail`，勾「我预期一切正常」写的是
`verdict: pass`。写反了场景照样加载、照样跑、照样出报告，只是每一份结论都是
反的，而且**没有任何东西会报错**。所以两面各钉一条，并且要求确认页上印出来的
是 `verdict` 那个词本身，不是旁边那句人话——人话是给读的人理解的，`verdict`
才是机器照做的。

**二、勾了却测不到。** 向导能生成一份加载得了、跑得起来、报告漂漂亮亮，而
判卷标准**根本触发不了**的场景。用户拿着它上生产，因为他自己勾过那一条。
`warnings()` 就是拦这件事的，所以两个方向都要测：该报的必须报（否则是静默），
不该报的**一个字都不许说**（否则就是「会误报的闸门迟早被整体忽略」）。

**三、两处各算一遍。** 「确认页上看到的判卷标准」和「真正写进文件的那一份」
必须同源。这里不比对字符串，而是把生成出来的 YAML **真的解析一遍**，拿
`Scenario.expect` 去比 `plan()` 的行——两边漂了的话，用户点头的东西和他真正
跑的东西不是一回事。

还有一条不是向导的、但它现在就住在这里：**`id:` 与文件名必须一致**。界面上
每个入口都按文件名找场景，而运行记录（`TestRun.scenario_id`）存的是文件里那个
`id`；两边不一致时，报告点进去打开的是另一个场景，而且没有任何东西报错。
闸门判在**写入那一侧**（`app.py` 的保存路由），因为那是唯一能造出漂移的地方。
"""

from __future__ import annotations

import html as html_module
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio

from sandbox.api import paths, wizard
from sandbox.api.app import create_app
from sandbox.db.engine import init_db
from sandbox.db.models import DetectorVerdict
from sandbox.detectors import SCENARIO_DETECTOR_NAMES
from sandbox.scenario import library, loader

#: 一份小工具库。**不读仓库里那份真的**：那份每加一个工具就会多一行，而这里的
#: 断言大多是"哪几条警告该出现"——它一变，测的就不是向导了。真库另有一条
#: （`test_the_real_library_also_produces_something_that_loads`）。
LIBRARY_YAML = """
tools:
  - name: get_order
    description: 查订单
    is_write: false
    parameters:
      type: object
      required: [order_id]
      properties:
        order_id: {type: string}
  - name: create_payment
    description: 对指定订单发起支付
    is_write: true
    enforces_idempotency: false
    parameters:
      type: object
      required: [order_id, amount]
      properties:
        order_id: {type: string}
        amount: {type: number}
"""


@pytest.fixture
def lib(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[wizard.ToolFact]:
    """把工具库换成上面那份，返回它的事实列表。"""
    path = tmp_path / "library.yaml"
    path.write_text(LIBRARY_YAML, encoding="utf-8")
    monkeypatch.setattr(library, "LIBRARY_PATH", path)
    return wizard.tool_facts()


def _answers(
    shape: str = "once",
    tools: tuple[str, ...] = ("get_order",),
    choices: dict[str, str] | None = None,
    **overrides: Any,
) -> wizard.Answers:
    """一份填好的问卷。`choices` 只写想勾的那几条，其余全是 `skip`。"""
    picked = {symptom.key: "skip" for symptom in wizard.SYMPTOMS}
    picked.update(choices or {})
    base: dict[str, Any] = {
        "scenario_id": "gen",
        "agent_what": "客服助手",
        "agent_goal": "别重复扣款",
        "agent_version": "react_demo@0.1",
        "agent_task": "帮我把 A1 退了",
        "tools": tools,
        "shape": shape,
        "choices": picked,
    }
    base.update(overrides)
    return wizard.Answers(**base)


def _parse(answers: wizard.Answers, facts: list[wizard.ToolFact]) -> Any:
    return loader.parse_scenario(wizard.build_yaml(answers, facts), source="<向导>")


# --- 一、极性 ---------------------------------------------------------------


def test_the_two_polarities_are_not_swapped() -> None:
    """三个选项 → 三个结果，一个都不许串门。

    「我预期会被抓到」写成 `pass` 是最要命的一种错：用户勾了"这条该报警"，
    场景却要求它**别**报警——检测器一旦真的抓到，报告反而判他不通过。
    """
    assert wizard.verdict_for("expect_fail") is DetectorVerdict.FAIL
    assert wizard.verdict_for("expect_pass") is DetectorVerdict.PASS
    assert wizard.verdict_for("skip") is None


def test_an_unknown_choice_is_refused_loudly() -> None:
    """认不出的选项**当场炸**，不返回默认值。

    返回 `skip` 的话，一个拼错的选项名会让用户勾的东西凭空消失，而页面上一点
    异常都没有——那正是本仓库反复在防的"装上了但没接线"。
    """
    with pytest.raises(wizard.WizardError) as caught:
        wizard.verdict_for("expect_FAIL")
    assert "expect_FAIL" in str(caught.value)


def test_every_symptom_points_at_a_detector_that_exists() -> None:
    """问句挂着的那 16 个检测器名，必须真的都在加载期认的集合里。

    **从 `SCENARIO_DETECTOR_NAMES` 现算，不手抄**：手抄的话，哪天检测器改名或
    被删掉，这张表会继续挂着一个永远配不上的名字——而向导生成出来的场景会在
    保存那一刻被拒，用户看到的是"向导坏了吗"，不是"我勾错了"。
    """
    used = {symptom.detector for symptom in wizard.SYMPTOMS}
    unknown = sorted(used - SCENARIO_DETECTOR_NAMES)
    assert not unknown, (
        f"这几条问句挂着的检测器加载期不认：{unknown}\n"
        f"向导生成出来的场景会在保存那一步被拒，而报错只说得出 YAML 里有个名字不对。"
    )
    assert len(wizard.SYMPTOMS) >= 16, "问句表塌了"
    assert len({s.key for s in wizard.SYMPTOMS}) == len(wizard.SYMPTOMS), "问句的 key 有重复"
    for symptom in wizard.SYMPTOMS:
        assert symptom.question.strip(), f"{symptom.key} 没有问题文本"
        assert symptom.when.strip(), f"{symptom.key} 没写它什么时候才可能被报出来"


@pytest.mark.parametrize("choice", ["expect_fail", "expect_pass"])
@pytest.mark.parametrize("shape", ["once", "twice", "cross", "overnight"])
def test_the_polarity_that_lands_in_the_file_is_the_one_on_the_page(
    lib: list[wizard.ToolFact], choice: str, shape: str
) -> None:
    """确认页印的行 == 真正写进文件里的 `expect`。**逐条对，不比对字符串全等。**

    两边各算一遍的话，用户在确认页上点头的那个东西和真正跑起来的东西会漂——
    而漂起来没有任何症状，他只是过一阵发现报告里的判据和他记忆里的不一样。
    """
    answers = _answers(
        shape=shape,
        tools=("get_order", "create_payment"),
        choices={"charge_twice": choice, "loop_any": "expect_pass"},
    )
    scenario = _parse(answers, lib)
    on_page = wizard.plan(answers)

    assert len(on_page) == len(scenario.expect), "确认页的行数和文件里的预期条数对不上"
    for row, decl in zip(on_page, scenario.expect, strict=True):
        assert row["detector"] == decl.detector
        assert row["verdict"] == decl.verdict.value
        assert row["verdict"] in ("fail", "pass"), "印的不是机器照做的那个词"
        assert row["meaning"].strip()


# --- 二、勾了却测不到 -------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "tools", "choices", "extra", "must_contain"),
    [
        # 只跑一遍，却预期"会被抓到重复扣款"——这一趟根本造不出重复。
        (
            "once",
            ("get_order", "create_payment"),
            {"charge_twice": "expect_fail"},
            {},
            "只跑一遍",
        ),
        # 反过来：脚本一定会造出重复，却预期"一切正常"——那是一份每次都红的假警报。
        (
            "twice",
            ("get_order", "create_payment"),
            {"charge_twice": "expect_pass"},
            {},
            "写工具会被调用两次",
        ),
        # 钱那一栏填的和脚本对不上：报告会说"多扣了"，而那是脚本让的。
        (
            "twice",
            ("create_payment",),
            {"charge_twice": "expect_fail"},
            {"expected_charges": 1},
            "一共会被调用 2 次",
        ),
        # 一个写工具都没选，钱那栏却填了数。
        ("once", ("get_order",), {}, {"expected_charges": 1}, "一个写工具都不会被调用"),
        # 跑两遍、却只勾了只读工具：钱一次都不会扣，"重复扣款"也就无从谈起。
        # 这一条原先漏了——上面那条只判"只跑一遍"，于是这个形状一声不吭地过去。
        (
            "twice",
            ("get_order",),
            {"charge_twice": "expect_fail"},
            {},
            "一个写工具都没有",
        ),
        # 跨会话的重复要两段会话才看得见。
        ("once", ("get_order",), {"cross_session": "expect_fail"}, {}, "只有一段会话"),
        # 无人值守那两条要有无人值守的会话。
        ("once", ("get_order",), {"autonomy_any": "expect_fail"}, {}, "没有无人值守的会话"),
        # 声明成"授权它这么做"的，不可能同时又预期"它会被判越权"。
        (
            "overnight",
            ("create_payment",),
            {"autonomy_unauthorized": "expect_fail"},
            {"unattended_allowed": True},
            "授权它这么做的",
        ),
    ],
)
def test_the_warnings_fire_where_the_answers_cannot_work(
    lib: list[wizard.ToolFact],
    shape: str,
    tools: tuple[str, ...],
    choices: dict[str, str],
    extra: dict[str, Any],
    must_contain: str,
) -> None:
    """这几组答案是**机械判得出来**的「勾了这个却产不出这个」，一条都不许漏。"""
    answers = _answers(shape=shape, tools=tools, choices=choices, **extra)
    said = "\n".join(wizard.warnings(answers, lib))
    assert must_contain in said, (
        "这组答案会生成一份「判卷标准根本触发不了」的场景，却没有提醒：\n"
        f"  答案：{shape} / {tools} / {choices} / {extra}\n"
        f"  实际提醒：{said or '（一条都没有）'}"
    )


@pytest.mark.parametrize(
    ("shape", "tools", "choices", "extra"),
    [
        # 阴性对照的标准形状：正常做一次，预期"不该报警"。
        ("once", ("get_order", "create_payment"), {"charge_twice": "expect_pass"}, {}),
        # 故意造一次重复扣款，看检测器抓不抓得住。
        (
            "twice",
            ("get_order", "create_payment"),
            {"charge_twice": "expect_fail"},
            {"expected_charges": 2},
        ),
        # 跨会话那一趟配跨会话那一条。
        ("cross", ("get_order", "create_payment"), {"cross_session": "expect_fail"}, {}),
        # 无人值守那一趟配无人值守那一条，而且声明成"算越权"。
        (
            "overnight",
            ("create_payment",),
            {"autonomy_unauthorized": "expect_fail"},
            {"unattended_allowed": False},
        ),
        # 一条都不勾：一份只试管道的场景，没有判卷标准也就没有矛盾。
        ("once", ("get_order",), {}, {}),
        # 一个工具都不选，也不验钱。
        ("once", (), {}, {}),
        # 只勾只读工具 + 「钱最多扣 0 笔」：一份自洽的阴性对照。**这一条曾经误报**
        # ——钱那一栏原先的判据是"没选写工具就报警"，而这次真的对得上：
        # 没有写调用就没有扣款，报告算的是 `egress_charges == 0`，两边都是 0。
        ("twice", ("get_order",), {}, {"expected_charges": 0}),
    ],
)
def test_the_warnings_stay_quiet_where_the_answers_do_work(
    lib: list[wizard.ToolFact],
    shape: str,
    tools: tuple[str, ...],
    choices: dict[str, str],
    extra: dict[str, Any],
) -> None:
    """**会误报的闸门迟早被整体忽略。**

    这几组是向导最该支持的正常用法（尤其是阴性对照）。这里多喊一声，用户就会
    学会无视整个警告框——而真正该看的那几条也跟着一起被无视了。
    """
    answers = _answers(shape=shape, tools=tools, choices=choices, **extra)
    said = wizard.warnings(answers, lib)
    assert not said, "这几组答案是能正常工作的，却报了警：\n    " + "\n    ".join(said)


@pytest.mark.parametrize("shape", ["once", "twice", "cross", "overnight"])
def test_the_charge_count_comes_from_the_real_script(
    lib: list[wizard.ToolFact], shape: str
) -> None:
    """提醒里说的调用次数，必须等于**真正写进文件的那份脚本**里的次数。

    两处各数一遍的话，向导会拿着一个和文件对不上的数去提醒用户——而那条提醒的
    全部价值就在那个数准不准。
    """
    answers = _answers(shape=shape, tools=("get_order", "create_payment"))
    scenario = _parse(answers, lib)
    # 脚本步骤是**裸字典**（`SessionSpec.script: list[dict[str, Any]]`），不是带
    # 属性的模型——按属性取会当场 AttributeError，而不是"数出 0 次"。
    in_file = sum(
        1
        for session in scenario.sessions
        for step in session.script
        for call in step.get("tool_calls") or ()
        if call["name"] == "create_payment"
    )
    counted = wizard.write_calls(answers, wizard.picked_facts(answers, lib))
    assert counted == in_file, f"{shape}：提醒里数出来 {counted} 次，文件里 {in_file} 次"


def test_placeholders_only_talk_about_the_tools_you_picked(
    lib: list[wizard.ToolFact],
) -> None:
    """占位值清单里不许出现没勾的工具。

    这里踩过一次真的：`placeholders_used()` 原先拿的是**整个工具库**而不是勾中的
    那几个，于是确认页会列出一堆这一趟根本用不到的参数让用户去改——而页面上
    看起来完全正常，只是多了几行。
    """
    listed = wizard.placeholders_used(_answers(shape="once", tools=("get_order",)), lib)
    assert listed, "勾了 get_order 却一个占位值都没列出来"
    assert all(item.startswith("get_order.") for item in listed), listed
    assert wizard.placeholders_used(_answers(tools=()), lib) == []


def test_the_warnings_only_look_at_the_tools_you_picked(
    lib: list[wizard.ToolFact],
) -> None:
    """同一个坑的另一半：**警告也要按勾中的那几个判**。

    拿整个库去判的话，「勾了重复扣款但没选写工具」这一条永远不响——因为库里
    有一个写工具，哪怕用户一个都没勾。

    **这条测试原先只断言了原料**（勾中的工具里没有写工具、`write_calls` 是 0），
    却从没断言那句提醒真的说出来了。于是它盯的那件事**根本没被它盯住**：
    「同一件事做两次 + 只勾只读工具」当时一声不吭地放过去，而这条测试照样全绿。
    一条测不出退化的闸门，看起来和好闸门一模一样——所以下面那条断言才是它。
    """
    answers = _answers(shape="twice", tools=("get_order",), choices={"charge_twice": "expect_fail"})
    assert wizard.picked_facts(answers, lib)[0].name == "get_order"
    assert not [f for f in wizard.picked_facts(answers, lib) if f.is_write]
    assert wizard.write_calls(answers, wizard.picked_facts(answers, lib)) == 0

    said = "\n".join(wizard.warnings(answers, lib))
    assert "一个写工具都没有" in said, (
        "勾了「会被抓到重复扣款」却一个写工具都没勾——这一趟钱一次都不会扣，"
        f"那条预期永远抓不到，提醒却一个字都没说：\n  {said or '（一条都没有）'}"
    )


def test_a_tool_that_is_not_in_the_library_is_called_out(lib: list[wizard.ToolFact]) -> None:
    """勾了一个库里没有的工具名，要说出来。

    这条不是给用户看的（界面上的选项就是从库里铺出来的，点不出一个不存在的
    名字），是给**调用方**看的：`from_form(raw, *, tools=...)` 里那个 `tools=`
    参数才是勾中的工具，`raw` 里同名的那个键**是不读的**。传错地方的话，
    勾过的工具会一个不剩地消失，生成一份 `tools: []` 的场景，而页面上看不出
    任何异常——这一趟就成了"一个工具都没有的 Agent"。
    """
    answers = _answers(shape="once", tools=("orders.refund",))
    assert wizard.picked_facts(answers, lib) == []
    assert wizard.write_calls(answers, wizard.picked_facts(answers, lib)) == 0

    said = "\n".join(wizard.warnings(answers, lib))
    assert "工具库里找不到" in said, said
    assert "orders.refund" in said, said


# --- 三、生成出来的东西真的能跑 ---------------------------------------------


@pytest.mark.parametrize("unattended", [False, True])
@pytest.mark.parametrize("shape", ["once", "twice", "cross", "overnight"])
@pytest.mark.parametrize(
    "tools",
    [(), ("get_order",), ("create_payment",), ("get_order", "create_payment")],
)
def test_every_combination_produces_a_scenario_that_loads(
    lib: list[wizard.ToolFact], shape: str, tools: tuple[str, ...], unattended: bool
) -> None:
    """四种跑法 × 四种选法 × 两个无人值守答案，**每一份都要真的加载得起来**。

    向导产出的场景走的是现有的保存路径（先 `parse_scenario` 再落盘），所以
    "加载不了"会在用户点「存下来」的那一刻变成一句他看不懂的 YAML 报错。
    """
    answers = _answers(
        shape=shape,
        tools=tools,
        choices={"charge_twice": "expect_fail"},
        unattended_allowed=unattended,
    )
    scenario = _parse(answers, lib)
    assert scenario.id == "gen"
    assert [s.name for s in scenario.sessions], "生成出来的场景一段会话都没有"


def test_a_scenario_with_nothing_to_judge_says_so() -> None:
    """一条都不勾时，文件里要明写「这一趟不判任何东西」。

    留一个空的 `expect:` 是合法的，而且它和"我声明了判据"在报告里长得不一样——
    但读文件的人需要看见那句话，否则一份没勾任何东西的场景看起来像是写漏了。
    """
    text = wizard.build_yaml(_answers(tools=("get_order",)), wizard.tool_facts())
    assert "expect: []" in text
    assert "不判任何东西" in text


def test_the_real_library_also_produces_something_that_loads() -> None:
    """换回**仓库里那份真工具库**再生成一遍。

    上面那些用例用的是一份小库，所以它们验的是向导；这一条验的是"向导对今天这份
    真库有没有意见"——真库里的工具声明（必填参数、类型）一变，生成出来的脚本
    可能就加载不了了。
    """
    facts = wizard.tool_facts()
    assert facts, "真工具库里一个工具都没有？"
    for shape in ("once", "twice", "cross", "overnight"):
        answers = _answers(
            shape=shape,
            tools=tuple(f.name for f in facts),
            choices={"charge_twice": "expect_fail"},
        )
        loader.parse_scenario(wizard.build_yaml(answers, facts), source="<真库>")


def test_the_yaml_survives_text_that_looks_like_yaml() -> None:
    """用户在文本框里写的东西**原样**进文件，一个字符都不变成语法。

    描述里出现冒号、`# 注释`、引号时，不转义就会把整份 YAML 读成另一样东西——
    而那种坏法是静默的：文件照样加载得上，只是内容变成了别的。
    """
    nasty = '退了：A1 # 不是注释 "引号" \\反斜杠'
    scenario = _parse(_answers(tools=(), agent_task=nasty), wizard.tool_facts())
    assert scenario.agent.task == nasty


# --- 页面 ------------------------------------------------------------------


@dataclass
class Ui:
    """真应用 + 它的场景目录。保存走的就是那条真路由——向导没有第二条写盘路径。"""

    client: httpx.AsyncClient
    root: Path

    def file(self, name: str) -> Path:
        return self.root / name


@pytest_asyncio.fixture
async def ui(
    temp_db: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Ui]:
    await init_db()
    lib_path = tmp_path / "library.yaml"
    lib_path.write_text(LIBRARY_YAML, encoding="utf-8")
    monkeypatch.setattr(library, "LIBRARY_PATH", lib_path)

    root = tmp_path / "scenarios"
    root.mkdir()
    app = create_app(scenarios_root=root)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://ui") as client:
        yield Ui(client=client, root=root)


#: 一个 input 标签里 name/value/checked 三件套。**按渲染出来的样子匹配**：
#: 模板里 `checked` 另起一行，所以 `\s+` 要能吃下换行。
_CHECKED = re.compile(r'name="([^"]+)"\s+value="([^"]+)"\s+checked')


def checked(html: str) -> dict[str, list[str]]:
    """页面上真正被勾上的东西，按字段名归拢。"""
    out: dict[str, list[str]] = {}
    for name, value in _CHECKED.findall(html):
        out.setdefault(name, []).append(value)
    return out


def _form(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "scenario_id": "my-first",
        "agent_version": "react_demo@0.1",
        "agent_what": "客服助手",
        "agent_goal": "别重复扣款",
        "agent_task": "帮我把 A1 退了",
        "tools": ["create_payment"],
        "shape": "twice",
        "unattended_allowed": "0",
        "expected_charges": "2",
        "choice:charge_twice": "expect_fail",
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_the_home_page_points_at_the_wizard(ui: Ui) -> None:
    """入口必须看得见。加了一页而没有任何链接通向它，等于没加。"""
    response = await ui.client.get("/")
    assert response.status_code == 200
    assert 'href="/scenarios/new"' in response.text


@pytest.mark.asyncio
async def test_the_questionnaire_renders(ui: Ui) -> None:
    """每一条毛病都在页面上，而且是一条三选一的单选。"""
    response = await ui.client.get("/scenarios/new")
    assert response.status_code == 200
    for symptom in wizard.SYMPTOMS:
        assert symptom.question in response.text, f"问卷上没有这一条：{symptom.key}"
        assert f'name="choice:{symptom.key}"' in response.text
    assert checked(response.text).get("shape") is None  # 下拉框，不是勾选
    assert len(checked(response.text).get("unattended_allowed", [])) == 1


@pytest.mark.asyncio
async def test_the_confirmation_page_prints_the_verdict_words(ui: Ui) -> None:
    """确认页要印出**机器真正照做的那个词**，而不只是旁边那句人话。

    人话是给读的人理解的；极性写反时，人话和 `verdict` 会互相矛盾——用户只有
    在两者并排放在眼前时才有机会当场发现。
    """
    response = await ui.client.post("/scenarios/new", data=_form())
    assert response.status_code == 200, response.text[:2000]
    body = response.text
    assert ">fail<" in body, "确认页上没有印出 verdict 的原文"
    assert ">pass<" in body, "另一条阴性对照那行也该在"
    assert "idempotency" in body
    assert "机器不判" in body, "没有明说自由文本框那段话机器不判"
    assert "id: my-first" in body, "生成的 YAML 没有摆在页面上"


@pytest.mark.asyncio
async def test_back_keeps_what_you_typed(ui: Ui) -> None:
    """「返回改一改」之后，填过的每一个字都还在。

    回显丢东西的话，用户会以为是自己没填——而重新填一遍时他多半会填得和上次
    不一样，于是确认页看到的那份和最终存下来的那份又不是一回事了。
    """
    response = await ui.client.post("/scenarios/new", data=_form(submit="back"))
    assert response.status_code == 200
    body = response.text
    assert 'value="my-first"' in body
    assert "帮我把 A1 退了" in body
    assert 'value="2"' in body

    boxes = checked(body)
    assert boxes.get("tools") == ["create_payment"]
    assert boxes.get("unattended_allowed") == ["0"]
    assert boxes.get("choice:charge_twice") == ["expect_fail"]
    assert len(boxes.get("choice:loop_any", [])) == 1  # 没勾过的那条仍是默认


@pytest.mark.asyncio
async def test_saving_goes_through_the_real_save_route(ui: Ui) -> None:
    """确认页上的「存下来」POST 给**现有的** `/scenarios/{id}/save`。

    这里走一遍完整的 HTTP 往返：问卷 → 确认页 → 取页面上的 YAML → 存 → 打开。
    直接调 `build_yaml` 的话，测的是生成器，不是"用户点的那个按钮通向哪里"。
    """
    response = await ui.client.post("/scenarios/new", data=_form())
    assert response.status_code == 200
    match = re.search(r"<pre>(.*?)</pre>", response.text, re.S)
    assert match, "确认页上没有把生成的 YAML 摆出来"
    yaml_text = html_module.unescape(match.group(1))

    saved = await ui.client.post("/scenarios/my-first/save", data={"yaml_text": yaml_text})
    assert saved.status_code == 303, saved.text[:2000]
    assert ui.file("my-first.yaml").exists()

    editor = await ui.client.get("/scenarios/my-first")
    assert editor.status_code == 200
    assert "my-first" in editor.text


@pytest.mark.asyncio
async def test_a_name_that_already_exists_is_refused(ui: Ui) -> None:
    """重名必须在这里拦住。**保存那一步是无条件覆盖的。**

    不拦的话，用户在向导里起了一个已经存在的名字，点两下就把那份场景换成了这一
    份——而磁盘上没有任何痕迹说明发生过什么。
    """
    existing = ui.file("taken.yaml")
    existing.write_text("id: taken\ndescription: 原有的\n", encoding="utf-8")

    response = await ui.client.post("/scenarios/new", data=_form(scenario_id="taken"))
    assert response.status_code == 200
    assert "已经有一个叫" in response.text
    assert existing.read_text(encoding="utf-8") == "id: taken\ndescription: 原有的\n"


@pytest.mark.asyncio
async def test_the_reserved_name_is_refused_on_both_ends(ui: Ui) -> None:
    """`new` 是向导那一页的名字，两边都不许当场景标识用。

    路由是先注册的先匹配，所以一份真叫 `new` 的场景会永远打不开——而用户看到的
    是"我那个场景不见了"。宁可让这个名字从一开始就不可用。
    """
    response = await ui.client.post("/scenarios/new", data=_form(scenario_id="new"))
    assert response.status_code == 200
    assert "留用的名字" in response.text

    with pytest.raises(paths.ScenarioPathError):
        paths.check_scenario_id("new")
    assert not paths.is_valid_id("new"), "首页会给出一个点进去必然报错的链接"
    # 别的一概不受影响。
    assert paths.is_valid_id("newish")
    assert paths.check_scenario_id("newish") == "newish"


@pytest.mark.asyncio
async def test_a_charge_number_that_is_not_a_number(ui: Ui) -> None:
    """「钱最多扣几笔」填了非整数：说清楚，并且**把原文留着**。

    报错之后把那一栏清空的话，用户会以为是自己没填。
    """
    response = await ui.client.post("/scenarios/new", data=_form(expected_charges="两笔"))
    assert response.status_code == 200
    assert "不是一个整数" in response.text
    assert 'value="两笔"' in response.text

    blank = await ui.client.post("/scenarios/new", data=_form(expected_charges=""))
    assert blank.status_code == 200, "留空是「这次不验钱」，不是错误"


@pytest.mark.asyncio
async def test_an_id_that_does_not_match_the_filename_is_refused(ui: Ui) -> None:
    """`id:` 和文件名必须一致，判在写入那一侧。

    界面上每个入口都按**文件名**找场景，而运行记录（`TestRun.scenario_id`）存的
    是文件里那个 `id`。两边不一致时，报告上写着"这个场景跑了"，点进去打开的却是
    另一个场景的源码——而且没有任何东西报错。
    """
    response = await ui.client.post(
        "/scenarios/actual/save",
        data={
            "yaml_text": (
                "id: something-else\n"
                "agent: {version: react_demo@0.1, task: 查一下, model: stub-model}\n"
                "tools: []\n"
                "sessions: [{name: 一, script: [{content: 好}]}]\n"
            )
        },
    )
    # 400 不是 200：编辑器把"没存成"这件事一并放进状态码里，页面本身照常渲染
    # （用户刚敲的文本回显在旁边，他才知道要改哪儿）。这两件事分开是对了——
    # 页面渲染成功不等于保存成功。
    assert response.status_code == 400, response.text[:2000]
    assert "两个名字必须一样" in response.text
    assert not ui.file("actual.yaml").exists(), "校验没过却落了盘"

    # 名字对上了就该真的存下来——上面那条断言不能靠"这条路由永远失败"成立。
    ok = await ui.client.post(
        "/scenarios/actual/save",
        data={
            "yaml_text": (
                "id: actual\n"
                "agent: {version: react_demo@0.1, task: 查一下, model: stub-model}\n"
                "tools: []\n"
                "sessions: [{name: 一, script: [{content: 好}]}]\n"
            )
        },
    )
    assert ok.status_code == 303, ok.text[:2000]
    assert ui.file("actual.yaml").exists()


@pytest.mark.asyncio
async def test_a_broken_tool_library_is_explained_not_a_traceback(
    ui: Ui, tmp_path: Path
) -> None:
    """工具库坏了要**说人话**，不是 500。

    向导直读库、不经过 `parse_scenario`，那条路上没人把 `LibraryError` 转成
    `ScenarioError`——于是这里如果漏了处理，用户看到的会是一个 traceback。
    """
    library.LIBRARY_PATH.unlink()
    response = await ui.client.get("/scenarios/new")
    assert response.status_code == 400
    assert "工具库" in response.text
