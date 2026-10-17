"""分页：近似层的两个分支各有一个活样本，以及"它们靠什么分开"。

**这个文件要回答的问题只有一个**：近似层凭什么把"翻了很多页"判成正常遍历、
把"翻得更多"判成要人看一眼？

最初的答案只有一条：**场景声明的那张字段权重表**——`page` 是光标，不是这个
操作的身份。不声明它，检测器只能拿数值的相对差去猜，而那个猜法有个实测出来的
后果：页码从 1 涨到 10，第 1 次和第 6 次的相似度只剩 0.72，一串连续翻页被切成
好几个簇，每个簇都够不着告警条数（8）——**于是那条"翻了很多页也不停"的 warning
永远不成立，而报告上看起来只是"检测器没报警"。**

**2026-10-02 加上了第二条路**（`LoopDetector._advancing_field`）：认得出"整组就是
一串往前走的游标"时整组判，不再靠簇长去够阈值。加它的理由是量出来的
（`../_probes/measure_warn_threshold.py`）：那句"8 页"的实际值是不给权重时
13~36 页，而声明权重也救不了"唯一的字段就是光标"的形状（权重是相对的，
只给一个字段声明等于没声明）。**方向是危险的那一头**——阈值说 8、实际要 30 多，
中间那些次报的是 `pass`「判为正常遍历」。

这件事改变了下面第 2 层的**对照方向**：原先"带着声明 → warning、拿掉声明 →
一串 pass"这组对比就是那条缺陷本身，它被修掉之后，两次的结论**必须一样**
（都是 warning），而"分簇仍然只由权重决定"改由 `_cluster` 直接钉。**对照组
不能因为结论变一致就删掉**——那两张表仍然是场景作者能改的唯一旋钮。

所以这里有三层，缺一层就有一条路能蒙混过去：

  1. **受控实验**：两份场景只差页数。世界、工具、Agent、判卷标准都得一样，
     否则"结论不同"就有第二种解释。
  2. **接线**：那张表真的走到了检测器手里，而且**两条路都对同一段轨迹给出
     同一个结论**。
  3. **看门狗**：运行**当中**读的是注册表，不是场景对象。表没过去的话，
     熔断与事后报告会拿着两把尺子量同一段轨迹。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from sandbox.config import REPO_ROOT
from sandbox.db.models import DetectorVerdict
from sandbox.detectors.base import CallView
from sandbox.detectors.loop import LoopDetector, checked_field_weights
from sandbox.gateway import app as gateway_app
from sandbox.runtime import runs
from sandbox.scenario import identity
from sandbox.scenario.loader import parse_scenario
from sandbox.scenario.runner import _run_detectors
from sandbox.scenario.spec import Scenario
from sandbox.trace.recorder import record_tool_call, start_run
from sandbox.watchdog import detector_for, enforce, inspect_run
from tests.control_pairs import assert_inline_only_differs, assert_only_differs, entry_for

SCENARIOS = REPO_ROOT / "scenarios"

TOOL = "list_orders"
STATUS = "unpaid"

#: 世界里面未支付的订单有 5 页。这是**场景那一侧的事实**，检测器看不见它
#: （ADR 002 不落工具返回体），所以它是"5 页是正常遍历"这句话的证据来源。
WORLD_PAGES = 5

#: 告警条数。与 `LoopDetector.monotonic_warn_after` 的默认值一致——这里写出来
#: 是为了让下面"越过了几条"的断言有个可读的基准，**不是**另一处阈值定义。
WARN_AFTER = 8

_PAGING_WORLD_HEAD = """\
id: loop_paging
agent:
  version: order_agent@0.1
  directory: order_agent
  task: 看看未支付订单有哪些。
  model: stub-model
tools:
  - name: list_orders
    description: 订单目录
    is_write: false
    parameters:
      type: object
      properties:
        status: {type: string}
        page: {type: integer}
        page_size: {type: integer}
      required: [status]
    field_weights: {page: 0.1}
sessions:
  - name: 翻页
    script:
"""


def _paging_text(pages: int) -> str:
    """按页数生成一份翻页场景：`bounded` 5 页、`unbounded` 10 页。"""
    calls = "".join(
        "      - tool_calls:\n"
        "          - name: list_orders\n"
        f"            arguments: {{status: unpaid, page: {page}, page_size: 20}}\n"
        for page in range(1, pages + 1)
    )
    return _PAGING_WORLD_HEAD + calls + f"      - content: 未支付订单共 {pages} 页\n"


BOUNDED = parse_scenario(_paging_text(WORLD_PAGES), source="<内存中的场景>")
UNBOUNDED = parse_scenario(_paging_text(10), source="<内存中的场景>")


def _paging_calls(pages: int) -> list[CallView]:
    """一串翻页调用，形状与场景脚本里那一段完全一致。"""
    return [
        CallView(
            id=f"call-{page}",
            seq=page,
            tool_name=TOOL,
            args={"status": STATUS, "page": page, "page_size": 20},
            # 指纹逐次不同：完全相同的调用归严格层管，近似层只看"参数在变"的。
            fingerprint=f"fp-{page}",
            is_write=False,
            session_ordinal=0,
        )
        for page in range(1, pages + 1)
    ]


def _without_weights(scenario: Scenario) -> Scenario:
    """同一份场景，把声明的那张权重表拿掉。

    **对照组的构造必须只改这一处**：模型、脚本、判卷标准都照旧，否则下面
    "结论不同"就有了第二种解释（比如换了个场景，而不是少了那句声明）。
    """
    return scenario.model_copy(
        update={
            "tools": [t.model_copy(update={"field_weights": {}}) for t in scenario.tools]
        }
    )


def _approximate(scenario: Scenario, pages: int) -> list:
    return [
        f
        for f in _run_detectors(scenario, _paging_calls(pages))
        if f.detector == "loop.approximate"
    ]


# --- 第一层：受控实验 ---------------------------------------------------------


def test_the_pair_differs_by_exactly_the_page_count() -> None:
    """两份场景之间只有**一个**变量：脚本往后多翻 4 页。

    比对用的是账本（`tests/control_pairs.py`），口径是生产上判"配置变没变"的那份
    `identity.sections`。`tools` 一节必须逐字相同——**权重表就在这一节里**，
    两支写的是同一张表。
    """
    bounded = identity.sections(BOUNDED)
    unbounded = identity.sections(UNBOUNDED)

    assert_inline_only_differs(
        BOUNDED,
        UNBOUNDED,
        claim="两支的差别只有脚本往后多翻几页（5 页对 10 页）。",
        differs_in=(
            "script[n].script[n].tool_calls[n].name",
            "script[n].script[n].tool_calls[n].arguments.page",
            "script[n].script[n].tool_calls[n].arguments.page_size",
            "script[n].script[n].tool_calls[n].arguments.status",
            "script[n].script[n].content",
        ),
    )

    assert bounded["script"] != unbounded["script"], "脚本必须不同，否则两支跑的是同一件事"
    assert len(bounded["script"]) == len(unbounded["script"]) == 1, "两支都是单会话"


def test_that_one_variable_is_the_page_count() -> None:
    """把"只差页数"从一句断言变成两个数：5 页 vs 10 页，别的都相同。"""
    bounded = BOUNDED
    unbounded = UNBOUNDED

    def pages(scenario: Scenario) -> list[int]:
        return [
            call["arguments"]["page"]
            for step in scenario.sessions[0].script
            for call in step.get("tool_calls", [])
        ]

    assert pages(bounded) == list(range(1, WORLD_PAGES + 1))
    assert pages(unbounded) == list(range(1, 11))
    assert pages(unbounded)[:WORLD_PAGES] == pages(bounded), (
        "多了的那一段必须接在后面——差别是'又翻了几页'，不是'换了别的页'"
    )
    # 场景里的页数与会话结束时那句汇报是同一个数：对齐了才谈得上"5 页到头了"。
    assert f"{WORLD_PAGES} 页" in bounded.sessions[0].script[-1]["content"]


def test_both_scenarios_declare_the_same_weights_and_say_why() -> None:
    """两支都声明了 `page: 0.1`，而且**是同一张表**。

    判卷标准不同的话，比的就是两把尺子：一支报 pass、一支报 warning 这件事
    就说明不了任何东西。
    """
    for scenario in (BOUNDED, UNBOUNDED):
        assert scenario.loop_field_weights() == {"page": 0.1}, f"{scenario.id} 没声明权重"
        # 权重必须落在真的会出现的字段上（加载器也查，这里再钉一次：这条声明
        # 是这一族结论的全部依据，写错了下面每条测试都会安静地变成"另一回事"）。
        assert "page" in scenario.tools[0].parameters["properties"]


# --- 第二层：接线 -------------------------------------------------------------


def test_the_declared_weights_are_what_turns_the_cluster_into_one() -> None:
    """**这个文件的核心。** 同一串 10 次调用，跑两遍，只有声明在不在的差别。

    带着声明的：10 次落进同一个簇，越过告警条数 → warning。
    拿掉声明的：**簇是散的**（`counts(None) != [10]`，`_cluster` 一个字没改），
    但**结论也是同一条 warning**——`_advancing_field` 认得出"整组就是一串往前
    走的游标"，不必再靠簇长去够阈值。

    **这条测试原先的对照方向是反的，而且是 2026-10-02 修掉的那个缺陷本身。**
    它当年断言"没有权重就一条 warning 都没有"，把 13~36 页才报（而不是 8 页）
    当成"权重在起作用"的证据。这个缺陷的代价在
    `LoopDetector._advancing_field` 的文档字符串里写着：阈值说 8、实际要 30 多，
    中间那些次报出来的是 `pass`「判为正常遍历」——一句做过的判定。

    修完之后的诚实版本是**把两件事分开钉**：

      * 分簇仍然只由权重决定（下面直接打 `_cluster`，不经过判定）；
      * 结论不再由它决定，两条路必须给出同一个结论。

    "一个看到翻页就报 warning 的检测器也能全绿"那条警戒线**没有被丢掉**，
    它在 `test_the_two_branches_are_separated_by_the_page_count_alone` 里：
    7 页仍然 PASS、9 页才 WARNING。
    """
    calls = _paging_calls(10)

    with_weights = _approximate(UNBOUNDED, 10)
    assert [f.verdict for f in with_weights] == [DetectorVerdict.WARNING]
    assert with_weights[0].detail["reason_code"] == "monotonic_unbounded"
    assert with_weights[0].detail["count"] == 10, "整串必须在同一个簇里"
    assert with_weights[0].detail["values_monotonic"] is True

    without = _approximate(_without_weights(UNBOUNDED), 10)
    assert [f.verdict for f in without] == [DetectorVerdict.WARNING], (
        "拿掉权重表之后结论必须照旧——这条 warning 的判据是'整组就是一串往前走的"
        "游标'，不依赖那张表。它曾经依赖过，那正是缺陷。"
    )
    assert without[0].detail["count"] == 10, "整组判：次数是整串的长度，不是某个簇的长度"
    assert without[0].detail["differing_fields"] == ["page"]
    assert without[0].detail["reason_code"] == "monotonic_unbounded"

    # 分簇这一半**没有变**：权重表仍然是唯一能改变它的东西。打的是私有方法——
    # 刻意的：这条要钉的就是"聚成几簇"，而判定层已经不再复述这个数了。
    def cluster_sizes(scenario: Scenario) -> list[int]:
        detector = LoopDetector(key_weights=scenario.loop_field_weights())
        return sorted((len(c) for c in detector._cluster(calls)), reverse=True)

    assert cluster_sizes(UNBOUNDED) == [10], "带着声明：整串一个簇"
    scattered = cluster_sizes(_without_weights(UNBOUNDED))
    assert len(scattered) >= 2, "拿掉声明：被切成好几个簇"
    assert min(scattered) < 3, (
        "有一簇小到不足 approximate_min_repeats（3）——**如果只走聚类这条路**，"
        "那一截调用连结论都不会出，报告上不留任何痕迹。这就是为什么"
        "`_advancing_field` 要整组判，而不是接着数簇。"
    )

    # 与上面那串**同一段构造**，只把页数换成世界里的真实页数。
    bounded = _approximate(BOUNDED, WORLD_PAGES)
    assert [f.verdict for f in bounded] == [DetectorVerdict.PASS]
    assert bounded[0].detail["count"] == WORLD_PAGES
    assert bounded[0].detail.get("reason_code") is None, "通过的那一支没有码可报"

    assert calls[0].args["page"] == 1  # 构造没被上面几次调用改坏（frozen dataclass）


def test_the_two_branches_are_separated_by_the_page_count_alone() -> None:
    """5 页 → pass，10 页 → warning。同一个工具、同一张权重表、同一串形状。

    这条把"阈值真的在阈值那个位置"钉住：8 次仍是正常遍历，9 次开始要人看一眼。
    少了它，把阈值改成 1 也能让上面那条全绿——而那样每一个翻两页的场景都会
    报警。
    """
    scenario = UNBOUNDED

    at_threshold = _approximate(scenario, WARN_AFTER - 1)
    assert [f.verdict for f in at_threshold] == [DetectorVerdict.PASS]

    over_threshold = _approximate(scenario, WARN_AFTER + 1)
    assert [f.verdict for f in over_threshold] == [DetectorVerdict.WARNING]
    assert over_threshold[0].detail["count"] == WARN_AFTER + 1


def test_a_weight_on_the_only_field_changes_nothing() -> None:
    """**权重是相对的**，所以只有一个参数时它压不动任何东西。

    分母上只有它自己：`0.1 × s / 0.1` 还是 `s`。声明造不出不存在的稳定性。

    这条划的是**分簇**这个机制的边界，**不是**判定层的边界——原先它下面写着
    "想把这类工具也判准，得改聚类方式本身（链式而非首成员），那是检测器行为变更，
    还没做"。那个说法两头都不对了（2026-10-02）：

      * 链式那条路早就被量翻过（`../_probes/measure_cluster_linkage.py`：形状依赖
        确实没了，十二个真不同的订单号从"不成簇"变成"一簇十二个"，漏报误报一起
        变多）——"还没做"应该改成"这条路不通";
      * 而这类工具**已经判得准了**，只是**没有走聚类**：`_advancing_field` 认得出
        "整组就是一串往前走的游标"，直接整组判。所以这张权重表对
        `list_orders` 这种"只有光标"的工具不再是那条 warning 的必要条件。

    写在这里，是因为"加了权重却没变化"最容易被读成接线断了——现在还得再补一句
    "而且它本来也不需要这张表"。
    """
    only_page = [
        CallView(
            id=f"call-{page}",
            seq=page,
            tool_name=TOOL,
            args={"page": page},
            fingerprint=f"fp-{page}",
            is_write=False,
        )
        for page in range(1, 11)
    ]

    def counts(weights: dict[str, float] | None) -> list[int]:
        detector = LoopDetector(key_weights=weights or {})
        return sorted((len(c) for c in detector._cluster(only_page)), reverse=True)

    assert counts(None) == counts({"page": 0.1}), "只有一个字段时，权重不该改变分簇"
    # 而它确实分不出"这是同一串翻页"：10 次调用被切成 6 个簇。
    assert counts(None) != [10]


def test_a_weight_the_registry_cannot_use_is_refused_over_the_wire() -> None:
    """过线那一侧：`"0.1"` 是文本，不是 0.1。

    这条只在**不过模型的那条路**上成立：YAML 那边 pydantic 会先转掉（见
    `tests/test_scenario_dsl.py`），而运行注册的 JSON 直接进校验函数。不拦的话
    `isfinite` 抛 `TypeError`，用户看到的是 500，而不是一句看得懂的话。
    """
    with pytest.raises(ValueError, match="必须是数字"):
        checked_field_weights({"page": "0.1"})


@pytest.mark.asyncio
async def test_the_gateway_refuses_a_bad_weight_before_the_run_starts(
    temp_db: str,
) -> None:
    """同上，走真的 HTTP 那一段：400 + 一个说得清的 err_type。"""
    gateway = AsyncClient(
        transport=ASGITransport(app=gateway_app.app), base_url="http://gateway.test"
    )
    resp = await gateway.post(
        "/_sandbox/runs",
        json={"scenario_id": "bad", "loop_field_weights": {"page": "0.1"}},
    )

    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "bad_loop_field_weights"
    assert "0.1" in resp.json()["error"]["message"]


@pytest.mark.asyncio
async def test_a_good_weight_reaches_the_registry_over_the_wire(temp_db: str) -> None:
    """正面的一半：声明真的落到了注册表上——看门狗读的就是那里。"""
    gateway = AsyncClient(
        transport=ASGITransport(app=gateway_app.app), base_url="http://gateway.test"
    )
    resp = await gateway.post(
        "/_sandbox/runs",
        json={
            "scenario_id": "good",
            "tools": {TOOL: f"/tools/{TOOL}"},
            "loop_field_weights": {"page": 0.1},
        },
    )

    assert resp.status_code == 200, resp.text
    active = runs.get(resp.json()["run_id"])

    assert active is not None
    assert active.loop_field_weights == {"page": 0.1}


# --- 第三层：看门狗 -----------------------------------------------------------


async def _run_with_pages(pages: int, *, weights: dict[str, float] | None = None) -> str:
    """起一个运行、落下这串翻页调用、注册进注册表。返回 run_id。"""
    ctx = await start_run(scenario_id="wd-paging", agent_version="v1", declared_writes=set())
    for page in range(1, pages + 1):
        await record_tool_call(
            ctx,
            tool_name=TOOL,
            args={"status": STATUS, "page": page, "page_size": 20},
        )
    runs.register(
        ctx,
        tools={TOOL: f"/tools/{TOOL}"},
        loop_field_weights=weights,
    )
    return ctx.run_id


def test_the_detector_is_built_from_the_run_not_from_a_module_global() -> None:
    """每个运行现造一个检测器：一个进程里会跑好几个场景。

    模块级单例的话，第二个场景要么沿用第一个场景的权重、要么得有人去改它——
    而"改它"在多进程与并发下就是串味，症状是同一段轨迹在不同运行里被判成
    不同的东西。
    """
    assert detector_for(None).key_weights == {}
    assert detector_for(None).monotonic_warn_after == WARN_AFTER

    class _Fake:
        loop_field_weights = {"page": 0.1}

    assert detector_for(_Fake()).key_weights == {"page": 0.1}


@pytest.mark.asyncio
async def test_the_watchdog_judges_the_trace_by_the_declared_weights(temp_db: str) -> None:
    """**熔断与事后报告必须拿同一把尺子。** 同一串 10 次调用，跑两遍。

    报告那一侧读的是场景对象，看门狗这一侧只能读注册表。所以这条测试真正要问的是：
    注册表里那份声明**有没有走过去**，以及过去之后**两边说的是不是同一句话**。

    原先它是拿"带着声明 → warning、没带 → 一串 pass"当证据的。那条对照 2026-10-02
    被修掉了：没带声明的那个运行现在**也**是 warning（`_advancing_field` 认得出
    整组游标，不再靠簇长够阈值）。**对照消失不等于这条测试可以删**——判据改成
    更直接的那个：把看门狗看到的结论和事后报告算出来的结论**逐字比一比**。
    两把尺子有一把没接上，这里就红了，而原先那种"一边 warning、一边 pass"的
    比法恰恰是拿缺陷当证据。
    """
    declared = await _run_with_pages(10, weights={"page": 0.1})
    plain = await _run_with_pages(10)

    # 事后报告那一侧：场景对象 → 检测器。
    expected = _approximate(UNBOUNDED, 10)

    for run_id, note in ((declared, "带声明"), (plain, "没带声明")):
        seen = [
            f for f in await inspect_run(run_id) if f.detector == "loop.approximate"
        ]
        assert [f.verdict for f in seen] == [DetectorVerdict.WARNING], (
            f"{note}的那次运行：看门狗必须认出这是一串没收敛的推进"
        )
        assert seen[0].detail["count"] == 10
        assert seen[0].detail["reason_code"] == "monotonic_unbounded"
        assert seen[0].detail["differing_fields"] == expected[0].detail["differing_fields"]
        assert seen[0].summary == expected[0].summary, (
            "熔断和事后报告对同一段轨迹说了两句话——而两边看起来都对"
        )

    # 注册表真的拿到了那张表（不是"反正结论一样所以无所谓"）。
    active = runs.get(declared)
    assert active is not None and active.loop_field_weights == {"page": 0.1}
    bare = runs.get(plain)
    assert bare is not None and bare.loop_field_weights == {}, (
        "没声明的那次运行注册表里是空表，不是 None——两种都表示'没有权重'，"
        "但钉住实际值，免得哪天它变成 None 而这里安静地过"
    )


# --- 端到端：真的拉起 Agent 子进程 ---------------------------------------------

AGENT_PYTHON = REPO_ROOT / "agents" / "react_demo" / ".venv"


@pytest.mark.asyncio
async def test_an_unbounded_paging_run_is_not_killed(temp_db: str) -> None:
    """翻页翻了 10 次**不杀进程**——这是刻意的，不是漏了。

    单看边界上的这一串调用分不清"在遍历一个很大的结果集"和"停不下来"：杀错
    一次正常的长遍历，代价是这次运行后面的东西一样都看不到；而多看一眼的代价
    只是看一眼。熔断只留给"一直烧钱且判得准"的那两层（`KILL_ON_DETECTORS`），
    近似层从来不在里面。
    """
    run_id = await _run_with_pages(10, weights={"page": 0.1})

    assert await enforce(run_id) is None
    assert runs.get(run_id).killed is False


@pytest.mark.asyncio
async def test_the_declaration_survives_the_trip_over_the_wire(temp_db: str) -> None:
    """**这条盯的是过线，不是逻辑。**

    场景文件里那张表要经过 `runner` 拼的那份载荷、`gateway` 的校验，才落到注册表
    上——而看门狗读的只有注册表。中间任何一环把**键名**写错都不会有报错：网关读
    不到那个键就当没声明（它必须允许，手工起运行的那条路本来就只给一半），注册表
    拿到空表，看门狗于是拿另一把尺子量同一段轨迹。这正是"装上了但没接线"的教科书
    样子，上面那些直接递给检测器的测试全绿也发现不了它。

    所以这里两边都用真的：载荷来自 `registration_payload`（运行时会发出去的那一份），
    校验来自真的网关端点。跑完注册表里那张表必须还是场景声明的那张。
    """
    from sandbox.scenario.runner import registration_payload

    payload = registration_payload(UNBOUNDED)
    assert payload["loop_field_weights"] == {"page": 0.1}, "载荷本身就不对"

    gateway = AsyncClient(
        transport=ASGITransport(app=gateway_app.app), base_url="http://gateway.test"
    )
    resp = await gateway.post("/_sandbox/runs", json=payload)

    assert resp.status_code == 200, resp.text
    active = runs.get(resp.json()["run_id"])
    assert active is not None
    assert active.loop_field_weights == {"page": 0.1}, (
        "声明没过线——看门狗读的就是这里，它会拿空表去量同一段轨迹"
    )


@pytest.mark.asyncio
async def test_the_pair_really_runs_and_lands_on_opposite_verdicts(temp_db: str) -> None:
    """真拉起被测 Agent 跑两遍：5 页判 pass，10 页判 warning，两次都不扣款。

    **两条断言缺一不可**，和记忆中毒那一对同一个道理：只看 warning 那一支的话，
    一个"看到翻页就报警"的检测器也能全绿（而那种检测器在真实使用里的下场是被
    关掉）；只看 pass 那一支的话，一个永远不报的检测器也能全绿。
    """
    if not AGENT_PYTHON.exists():
        pytest.skip(f"被测 Agent 的 venv 不在（{AGENT_PYTHON}），跳过端到端分页对照")
    from sandbox.scenario.runner import ScenarioRunner

    runner = ScenarioRunner(session_timeout_s=120.0)

    fine = await runner.run(BOUNDED)
    assert fine.error is None, fine.error
    assert fine.ok, fine.expectation and fine.expectation.to_dict()
    assert fine.egress_charges == 0, "这一对都只读，一笔款都不该动"
    assert [s.killed for s in fine.sessions] == [False]
    ok_finding = next(f for f in fine.findings if f.detector == "loop.approximate")
    assert ok_finding.verdict == DetectorVerdict.PASS
    assert ok_finding.detail["count"] == WORLD_PAGES
    assert ok_finding.detail.get("reason_code") is None

    runaway = await runner.run(UNBOUNDED)
    assert runaway.error is None, runaway.error
    assert runaway.ok, runaway.expectation and runaway.expectation.to_dict()
    assert runaway.egress_charges == 0
    assert [s.killed for s in runaway.sessions] == [False], "warning 不该熔断"
    bad_finding = next(f for f in runaway.findings if f.detector == "loop.approximate")
    assert bad_finding.verdict == DetectorVerdict.WARNING
    assert bad_finding.detail["reason_code"] == "monotonic_unbounded"
    assert bad_finding.detail["count"] == 10, "整串必须在同一个簇里"
