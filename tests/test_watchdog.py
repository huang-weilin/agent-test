"""熔断看门狗：运行当中就止损，以及**它认不认场景声明的那条豁免**。

看门狗是这个工具里唯一一处"我们替用户做决定"的地方——杀进程。所以它读进去的
东西必须是完整的，尤其是豁免。

豁免机制的全部意义是"我知道这看起来像缺陷，但它不是"（`docs/detectors.md`
第二节），而**那种场景里最需要它生效的时刻恰恰是熔断那一刻**。看门狗只吃注册表、
不看场景声明的话，一条被豁免的循环会得到两个相反的结论：进程在第 3 次调用时被
杀掉，报告却写着"loop.strict：pass（已被豁免规则静音）"。声明写了，运行还是没
跑完——这正是"装上了但没接线"那一类缺陷，只不过它的表现不是没生效，而是**只有
事后报告那一半生效了**。

这个文件盯三层：

  1. **注册表**：豁免跟着运行走，检测器、看门狗读的是同一份；
  2. **过线**：规则能序列化过去再还原回来，写错了在网关那一侧就报 400；
  3. **真的不杀**：同一串调用，有豁免跑完、没豁免被杀，而且被静音的那条带着
     理由留在结论里。
"""

from __future__ import annotations

import pytest
import yaml
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from sandbox.config import REPO_ROOT
from sandbox.db.models import DetectorFinding, DetectorVerdict
from sandbox.detectors.base import Exemption, ExemptionSet
from sandbox.gateway import app as gateway_app
from sandbox.runtime import runs
from sandbox.scenario.loader import load_scenario, parse_scenario
from sandbox.trace.reader import load_call_views
from sandbox.trace.recorder import record_tool_call, start_run
from sandbox.watchdog import KILL_ON_DETECTORS, enforce
from tests.control_pairs import assert_inline_only_differs, assert_only_differs, entry_for

SCENARIOS = REPO_ROOT / "scenarios"

_LOOP_TEXT = """\
id: loop
agent:
  version: order_agent@0.1
  directory: order_agent
  task: 查一下订单 A1。
  model: stub-model
tools:
  - name: get_order
    description: 查询
    is_write: false
sessions:
  - name: 一段
    script:
      - tool_calls:
          - name: get_order
            arguments: {order_id: A1}
      - content: 完成。
"""

_LOOP_EXEMPTED_TEXT = _LOOP_TEXT.replace(
    "id: loop\n",
    "id: loop_exempted\n\nexemptions:\n  - detector: loop.strict\n"
    "    tool: get_order\n    reason: 这个重复是场景刻意构造的样本\n",
)

LOOP = parse_scenario(_LOOP_TEXT, source="<内存中的场景>")
LOOP_EXEMPTED = parse_scenario(_LOOP_EXEMPTED_TEXT, source="<内存中的场景>")

LOOP_RULE = {
    "detector": "loop.strict",
    "tool": "get_order",
    "reason": "这个重复是场景刻意构造的样本",
}


async def _run_with_three_identical_calls(*, exemptions: ExemptionSet | None = None) -> str:
    """起一个运行，落三次完全相同的调用，注册进注册表。返回 run_id。"""
    ctx = await start_run(scenario_id="wd", agent_version="v1", declared_writes=set())
    for _ in range(3):
        await record_tool_call(ctx, tool_name="get_order", args={"order_id": "A1"})
    runs.register(ctx, tools={"get_order": "/tools/get_order"}, exemptions=exemptions)
    return ctx.run_id


async def _findings_of(run_id: str) -> list[DetectorFinding]:
    from sandbox.db.engine import session_scope

    async with session_scope() as db:
        rows = await db.execute(
            select(DetectorFinding).where(DetectorFinding.test_run_id == run_id)
        )
        return list(rows.scalars())


# --- 第一层：注册表 -----------------------------------------------------------


@pytest.mark.asyncio
async def test_a_loop_without_a_declared_exemption_is_killed(temp_db: str) -> None:
    """基线：没有豁免，三次相同的调用就该被杀。"""
    run_id = await _run_with_three_identical_calls()

    reason = await enforce(run_id)

    assert reason is not None and reason.startswith("loop.strict")
    assert runs.get(run_id).killed is True


@pytest.mark.asyncio
async def test_the_same_loop_with_a_declared_exemption_is_left_alone(temp_db: str) -> None:
    """**这条是这个文件存在的理由。**

    同样的轨迹、同样的检测器，只多了场景声明的那一条豁免——不杀。
    看门狗看不见豁免的话，这条会红，而其余所有测试都还是绿的：检测器那侧的
    豁免有测试、加载那侧的豁免有测试，唯独"运行当中那个读豁免的地方"没有。
    """
    run_id = await _run_with_three_identical_calls(
        exemptions=ExemptionSet.of([LOOP_RULE])
    )

    reason = await enforce(run_id)

    assert reason is None, "声明了豁免，看门狗还是把它杀了"
    assert runs.get(run_id).killed is False


@pytest.mark.asyncio
async def test_the_suppressed_kill_is_still_written_down_with_its_reason(temp_db: str) -> None:
    """不杀不等于没看见。结论照落库，带着理由和"本来会判 fail"。

    豁免不是静音：一条消失的结论读起来和"没发生过"一模一样，而这里的语义是
    "发生过、我看过了、允许了"。
    """
    run_id = await _run_with_three_identical_calls(
        exemptions=ExemptionSet.of([LOOP_RULE])
    )
    await enforce(run_id)

    strict = [f for f in await _findings_of(run_id) if f.detector == "loop.strict"]

    assert len(strict) == 1
    assert strict[0].verdict == DetectorVerdict.PASS.value
    assert strict[0].exempted_by == "exemption:loop.strict/get_order"
    assert LOOP_RULE["reason"] in strict[0].summary


@pytest.mark.asyncio
async def test_an_exemption_for_another_tool_does_not_stop_the_kill(temp_db: str) -> None:
    """规则写的是别的工具 → 匹配不上 → 和没写一样，照杀。

    豁免是"点名放行"，不是"开了个静音开关"。放宽这一条，用户写错一个工具名就
    把熔断关掉了，而报告上看起来他只是豁免了另一个工具。
    """
    other = {**LOOP_RULE, "tool": "create_payment"}
    run_id = await _run_with_three_identical_calls(exemptions=ExemptionSet.of([other]))

    assert await enforce(run_id) is not None


@pytest.mark.asyncio
async def test_an_exemption_for_another_layer_does_not_stop_the_kill(temp_db: str) -> None:
    """同理，豁免近似层不会拦住严格层的熔断。"""
    other = {**LOOP_RULE, "detector": "loop.approximate"}
    run_id = await _run_with_three_identical_calls(exemptions=ExemptionSet.of([other]))

    assert await enforce(run_id) is not None


# --- 账本：一个结论一行，不是扫一次一行 ---------------------------------------


@pytest.mark.asyncio
async def test_a_repeated_scan_does_not_record_the_same_conclusion_again(
    temp_db: str,
) -> None:
    """扫三遍，库里那两条结论各只有一行。

    看门狗一次运行里要重扫很多遍（`proxy/app.py` 的 `_settle`），而轨迹在两次
    调用之间不会平白多出一个结论。照单全收的话，一次 n 次调用的运行会往 `detector_finding`
    里写 O(n) 行——量于 2026-10-01，那是 **2 行/次调用**（开发库里当时 4,320 行，
    全部是副本）。那张表的名字是"检测器结论"，一行就该是一个结论。

    代价也不只是难看：落一行是常数 10 ms（量法见
    `../_probes/measure_watchdog_breakdown.py`），一次扫描落两条，于是**在小运行
    里它就是看门狗的全部开销**。
    """
    ctx = await start_run(scenario_id="wd-dedup", agent_version="v1", declared_writes=set())
    for i in range(5):
        await record_tool_call(ctx, tool_name="get_order", args={"order_id": f"A{i}"})
    runs.register(ctx, tools={"get_order": "/tools/get_order"}, exemptions=None)

    for _ in range(3):
        await enforce(ctx.run_id)

    rows = await _findings_of(ctx.run_id)
    seen = [(f.detector, f.verdict) for f in rows]

    assert len(seen) == len(set(seen)), f"同一结论落了多行：{seen}"
    assert len(seen) == 2, f"五次不同的调用应当只有两条结论，实际 {seen}"
    assert runs.get(ctx.run_id) is not None


@pytest.mark.asyncio
async def test_a_conclusion_that_changes_gets_its_own_row(temp_db: str) -> None:
    """去重去的是**重复**，不是**变化**：同一个检测器改判时，两行都在。

    这一条守的是去重键里那个**四态**。只用检测器做键的话，先落的
    `loop.approximate/pass` 会把后来的 `loop.approximate/warning` 吃掉——
    而后者正是"翻了很多页还没停"那条要人来看一眼的结论。报告那边照旧会印出
    warning（它自己重跑一遍检测器），于是库里和报告里说的不是一件事，
    且没有任何地方会红。

    形状是这么挑的：`monotonic_warn_after = 8` 判的是**簇有多长**，而页码
    1000 起步时数字相似度衰减得很慢（1000 与 1008 差 0.8%），整段会落进同一个
    簇——于是同一段轨迹在 7 次调用时是 pass、9 次时是 warning。
    """
    ctx = await start_run(
        scenario_id="wd-flip", agent_version="v1", declared_writes=set()
    )
    runs.register(ctx, tools={"list_orders": "/tools/list_orders"}, exemptions=None)

    # 7 页：簇 7 < 8，判为正常遍历 → pass。
    for i in range(7):
        await record_tool_call(ctx, tool_name="list_orders", args={"page": 1000 + i})
    await enforce(ctx.run_id)
    first = sorted(
        (f.detector, f.verdict)
        for f in await _findings_of(ctx.run_id)
        if f.detector == "loop.approximate"
    )
    assert first == [("loop.approximate", DetectorVerdict.PASS.value)], first

    # 再翻 2 页：簇 9 ≥ 8，同一条路改判 warning。
    for i in range(7, 9):
        await record_tool_call(ctx, tool_name="list_orders", args={"page": 1000 + i})
    await enforce(ctx.run_id)

    rows = sorted(
        (f.detector, f.verdict)
        for f in await _findings_of(ctx.run_id)
        if f.detector == "loop.approximate"
    )
    assert rows == [
        ("loop.approximate", DetectorVerdict.PASS.value),
        ("loop.approximate", DetectorVerdict.WARNING.value),
    ], f"先前的 pass 被后来的 warning 吃掉了，或反过来：{rows}"


def test_the_killable_set_is_what_the_docs_say() -> None:
    """只有会一直烧钱的那两层杀进程。越权会让人赔钱，但它不会自己一直发生。"""
    assert {"loop.strict", "loop.ping_pong"} == KILL_ON_DETECTORS


# --- 第二层：过线 -------------------------------------------------------------


def test_the_rules_survive_the_trip_over_the_wire() -> None:
    """序列化过去再还原回来，匹配行为必须一模一样。

    过线的时候字段少一个、`match` 被丢掉，表现都是"规则到了那边就不生效了"——
    而两边各自的测试都还是绿的。
    """
    original = ExemptionSet.of([LOOP_RULE, {**LOOP_RULE, "allow_write": True}])

    restored = ExemptionSet.of(original.to_dicts())

    assert restored.to_dicts() == original.to_dicts()
    for rules in (original, restored):
        found = rules.find("loop.strict", "get_order", is_write=False)
        assert found is not None and found.reason == LOOP_RULE["reason"]


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ({"tool": "get_order", "reason": "缺 detector"}, "detector"),
        ({"detector": "loop.strict", "reason": ""}, "reason"),
        ({"detector": "loop.strict", "reason": "多余字段", "extra": 1}, "extra"),
        ("loop.strict", "对象"),
        ({"detector": "memory.drift", "reason": "记忆族不吃豁免"}, "记忆族"),
    ],
)
def test_a_malformed_rule_says_what_is_wrong_with_it(rule, expected) -> None:
    """规则写错要在**过线那一刻**说清楚错在哪。

    直接 `Exemption(**rule)` 的话，多一个键是个 `TypeError`，读起来像程序崩了；
    少一个键要到熔断那一刻才炸——而那时候运行已经被杀了。
    """
    with pytest.raises(ValueError) as excinfo:
        ExemptionSet.of([rule])

    assert expected in str(excinfo.value)


@pytest.mark.asyncio
async def test_the_gateway_refuses_a_bad_rule_before_the_run_starts(temp_db: str) -> None:
    """手工 curl 也走这条路：规则写错就在开跑前被退回，而不是跑到一半才炸。"""
    gateway = AsyncClient(
        transport=ASGITransport(app=gateway_app.app), base_url="http://gateway.test"
    )
    resp = await gateway.post(
        "/_sandbox/runs",
        json={"scenario_id": "bad", "exemptions": [{"detector": "loop.strict"}]},
    )

    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "bad_exemptions"
    assert "reason" in resp.json()["error"]["message"]


@pytest.mark.asyncio
async def test_a_good_rule_reaches_the_registry_over_the_wire(temp_db: str) -> None:
    """正面的一半：规则真的落到了注册表上——看门狗读的就是那里。"""
    gateway = AsyncClient(
        transport=ASGITransport(app=gateway_app.app), base_url="http://gateway.test"
    )
    resp = await gateway.post(
        "/_sandbox/runs",
        json={
            "scenario_id": "good",
            "tools": {"get_order": "/tools/get_order"},
            "exemptions": [LOOP_RULE],
        },
    )

    assert resp.status_code == 200, resp.text
    active = runs.get(resp.json()["run_id"])

    assert active is not None
    assert active.exemptions.to_dicts() == [Exemption.of(LOOP_RULE).to_dict()]


def test_a_rule_without_a_tool_covers_every_tool() -> None:
    """`tool` 不写就是"所有工具"——那是文档写下的语义，不是这里新加的。"""
    rules = ExemptionSet.of([{"detector": "loop.strict", "reason": "整层静音"}])

    assert rules.find("loop.strict", "anything", is_write=False) is not None


# --- 第三层：两个场景文件的差别必须恰好是那一条豁免 ---------------------------


def test_the_pair_differs_by_exactly_the_exemption() -> None:
    """`loop.yaml` 与 `loop_exempted.yaml`：同一串调用，只差声明。

    端到端那一条（在下面）跑的是真进程、慢；这一条把"结论不同只可能是那一行
    造成的"钉死，于是它红了不必先怀疑 Agent 那边抖了。刻意**不比 `id`、
    散文与 `expect`**：两边的尺子本来就必须不同——"一边被杀、一边跑完"正是
    这组对照的全部内容。
    """
    plain = LOOP
    exempted = LOOP_EXEMPTED

    # 被测侧（prompt/tools/script/agent/world）一处都不差：豁免落在 `expect`
    # 那一节，而那是尺子——"一边被杀、一边跑完"正是这组对照的全部内容。
    assert_inline_only_differs(
        LOOP,
        LOOP_EXEMPTED,
        claim="同一串调用，只差声明：一边没有豁免、一边对 loop.strict 声明了一条。",
        differs_in=(),
    )

    assert plain.exemption_set().rules == []
    assert [r.detector for r in exempted.exemption_set().rules] == ["loop.strict"]


def test_the_exempted_scenario_entry_has_not_drifted_from_its_original() -> None:
    """文档引用的豁免是真实存在的——保留场景里那条 `loop.strict` 豁免。

    改 id 或删文件会让文档指向不存在的东西——那种失效没有任何一处会报错。
    保留场景（`order_delivery_chain_clean`）是 README/手册引用的示例，它的豁免
    声明就是文档承诺的那一条。
    """
    payload = yaml.safe_load(
        (SCENARIOS / "order_delivery_chain_clean.yaml").read_text(encoding="utf-8")
    )
    exemptions = [
        e for e in payload.get("exemptions", []) if e.get("detector") == "loop.strict"
    ]
    assert exemptions, "保留场景里必须有一条 loop.strict 豁免撑着文档里的说法"


# --- 端到端：真的拉起 Agent 子进程 ---------------------------------------------

AGENT_PYTHON = REPO_ROOT / "agents" / "react_demo" / ".venv"


@pytest.mark.asyncio
async def test_a_declared_exemption_lets_the_run_finish_and_keeps_the_observations(
    temp_db: str,
) -> None:
    """**这个特性的验收标准。** 8 次完全相同的调用，声明了豁免 → 不杀、跑完。

    三组断言防的是三种不同的失效，缺一组就有一条路能蒙混过去：

    - 只断言"没被杀"：检测器根本没报也能通过——那等于豁免把结论抹了；
    - 只断言"结论还在"：进程被杀掉、脚本没跑完也能通过——那等于豁免没有生效；
    - 只断言"脚本跑完了"：豁免把观测一起抹掉（报告显示它规规矩矩地查了一次）
      也能通过，所以链上那 8 行要一起看。
    """
    from sandbox.scenario.runner import ScenarioRunner

    if not AGENT_PYTHON.exists():
        pytest.skip(f"被测 Agent 的 venv 不在（{AGENT_PYTHON}），跳过端到端熔断豁免")

    result = await ScenarioRunner(session_timeout_s=120.0).run(LOOP_EXEMPTED)

    assert result.ok, result.error
    assert [s.killed for s in result.sessions] == [False], "声明了豁免，进程还是被熔断了"
    assert [s.script_steps_consumed for s in result.sessions] == [9], "脚本没跑完"

    strict = [f for f in result.findings if f.detector == "loop.strict"]
    assert [f.verdict for f in strict] == [DetectorVerdict.PASS], "豁免没生效或者没留痕"
    assert strict[0].exempted_by == "exemption:loop.strict/get_order"
    assert strict[0].detail["would_have_been"] == "fail", (
        "静音之后报告该说得出「本来会判 fail」——否则读的人以为这里查过了、没问题"
    )

    assert result.call_chain is not None
    chain = result.call_chain.chains[0]
    assert chain.total == 8, "被豁免的是结论，不是调用：8 次调用照样发生"
    assert chain.failed == 0 and chain.ours == 0
    assert len(await load_call_views(result.run_id)) == 8
