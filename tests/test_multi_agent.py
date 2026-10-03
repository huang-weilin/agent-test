"""多 Agent 串联：每一段由谁跑，以及这个事实在三处被用上。

需求里那一句是「多个 Agent 串联测试的通过判定」。串联本身不难做——难的是
**归因**：同一笔钱被扣了两次，该去修谁？三种成因在报告上长得一模一样，
而该动手的地方分别是 Agent 的判重逻辑、记忆的传递、以及接缝本身。

这一份文件盯的就是那三种成因的分界线，而分界线只有一个量：
**这一段会话是哪一个 Agent 跑的**（`agent_index`）。它在三处被用到，
每一处都配了"用错了会怎样"的测试：

  * **观测**（`memory/observe.py`）—— A 和 B 各有一份记忆目录，切面必须各比各的。
    混在一起比，报告会凭空说出"上次看到的文件全没了"：那是我们的目录布局造出来的
    一句话，而读的人会去查 Agent 为什么弄丢记忆。
  * **判定**（`detectors/memory.py`）—— 一条 `must_hold` 判的是**某一份**记忆。
    不筛的话，它在另一个人的记忆里找到了那个键，于是这条判据给出"守住了"——
    而它问的是这个人。
  * **归因**（`detectors/idempotency.py`）—— 换人造成的重复与回滚造成的重复都报
    fail，但视野那一栏在两种情形下都会指向"记忆没接上"。只有把首尾两段的
    `agent_index` 拿出来比，才说得清"这里没有人回滚过，是接缝"。

另外两道闸门守的是**说得出与说得对**：场景加载期拒收越界的序号（不能兜底成 0），
以及单 Agent 的句子不许到处挂"（0 号 Agent 的记忆）"——那是噪音，不是信息。

最后一条是端到端的：`cross_session_handoff` 与它的对照支 `cross_session_same_agent`
真的跑一遍。那一对在 `tests/control_pairs.py` 的账本里，被测侧只差五处、全在
"名单里多了一个人"这一件事上——也就是说归因的翻转**只可能**由换人造成。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sandbox.db.models import DetectorVerdict
from sandbox.detectors.base import ExemptionSet
from sandbox.detectors.idempotency import IdempotencyDetector, ToolFacts
from sandbox.detectors.memory import ForbiddenText, MemoryDetector, MemoryRules, MustHold
from sandbox.memory.observe import (
    PHASE_END,
    PHASE_START,
    MemoryEntry,
    MemoryObservation,
    summarize_observations,
)
from sandbox.scenario import identity
from sandbox.scenario.loader import ScenarioError, load_scenario, parse_scenario
from sandbox.trace.reader import load_session_sequence
from sandbox.trace.recorder import (
    compute_config_hash,
    record_tool_call,
    start_run,
    start_session,
)

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios"
HANDOFF = SCENARIOS / "cross_session_handoff.yaml"
SAME_AGENT = SCENARIOS / "cross_session_same_agent.yaml"
AGENT_PYTHON = ROOT / "agents" / "react_demo" / ".venv" / "Scripts" / "python.exe"

PAY = {"order_id": "A1", "amount": 599.0}
WRITES = {"create_payment"}
TOOL_FACTS = {"create_payment": ToolFacts("create_payment", is_write=True)}

#: 单 Agent 的骨架：没有 `agents:` 名单，会话上也不指名。
ONE_AGENT = """
id: solo
description: 单跑
agent:
  version: react_demo@0.1
  task: 先把订单读出来。
tools:
  - name: get_order
    description: 查订单
    is_write: false
sessions:
  - name: 第一次
    script:
      - tool_calls: [{name: get_order, arguments: {order_id: A1}}]
"""

#: 同一份场景写成列表，名单里只有一项。
ONE_AGENT_LISTED = ONE_AGENT.replace(
    "agent:\n  version: react_demo@0.1\n  task: 先把订单读出来。\n",
    "agents:\n  - version: react_demo@0.1\n    task: 先把订单读出来。\n",
)

#: 一份最小的串联场景：两段会话，第二段换成 1 号跑。
TWO_AGENTS = """
id: relay
description: 串联
agents:
  - version: react_demo@0.1
    task: 先把订单读出来。
  - version: react_demo@0.2
    task: 接着把它付掉。
tools:
  - name: get_order
    description: 查订单
    is_write: false
sessions:
  - name: 第一次
    script:
      - tool_calls: [{name: get_order, arguments: {order_id: A1}}]
  - name: 第二次
    agent: 1
    script:
      - tool_calls: [{name: get_order, arguments: {order_id: A1}}]
"""


def _entry(path: str, digest: str = "d1") -> MemoryEntry:
    return MemoryEntry(path=path, sha256=digest, text=f"{path} 的内容")


def _slice(
    ordinal: int,
    phase: str,
    agent: int,
    entries: tuple[MemoryEntry, ...] = (),
) -> MemoryObservation:
    return MemoryObservation(
        ordinal=ordinal, phase=phase, entries=entries, agent_index=agent
    )


def _findings(calls):
    return IdempotencyDetector(TOOL_FACTS).run(calls, ExemptionSet())


def _fail(findings):
    fails = [f for f in findings if f.verdict == DetectorVerdict.FAIL]
    assert len(fails) == 1, f"期望恰好一条 fail，实际 {[f.summary for f in fails]}"
    return fails[0]


# --- 声明：`agent` 是 `agents` 的糖，而且指纹也必须是同一个 -------------------


def test_one_agent_and_a_list_of_one_are_the_same_scenario() -> None:
    """两种写法必须是同一份场景，**连配置指纹都一样**。

    不一样的话，"把单数改成列表"这种纯格式改动会被回归对比读成"测试变了"——
    用户于是去查一个谁都没动过的文件。
    """
    listed = parse_scenario(ONE_AGENT_LISTED, source="<列表>")
    single = parse_scenario(ONE_AGENT, source="<单数>")
    assert single.agent.version == listed.agent.version
    assert single.agents == listed.agents, "两种写法收拢之后必须是同一份名单"
    assert identity.sections(single) == identity.sections(listed)
    assert compute_config_hash(identity.sections(single)) == compute_config_hash(
        identity.sections(listed)
    ), "指纹不一样的话，改一次格式就会被回归对比读成「测试变了」"

    # `scenario.agent` 是**派生值**：收拢发生在字段校验之前，两边各自过了一遍
    # 模型，所以拿 `is` 比会红——这里比的是值。分叉了的话，读 `agent` 的 19 处
    # 和读 `agents` 的那些地方会对同一份场景给出两个说法。
    for scenario in (single, listed):
        assert scenario.agent == scenario.agents[0]


def test_two_agents_come_out_in_order() -> None:
    """名单顺序就是序号顺序——会话上的 `agent: 1` 指的就是第二项。"""
    scenario = parse_scenario(TWO_AGENTS, source="<串联>")
    assert [a.version for a in scenario.agents] == ["react_demo@0.1", "react_demo@0.2"]
    assert scenario.agent == scenario.agents[0], "0 号是那个平铺的 `agent`"
    assert scenario.sessions[1].agent == 1


def test_a_scenario_without_any_agent_is_refused() -> None:
    """一份场景必须说清被测对象是谁——没有"默认那一份"可以兜底。"""
    with pytest.raises(ScenarioError, match="没有写被测 Agent"):
        parse_scenario(
            "id: x\ndescription: 无\ntools: []\nsessions:\n  - name: 一\n    script: []\n",
            source="<没有 Agent>",
        )


def test_writing_both_forms_at_once_is_refused() -> None:
    """两个都写时不能"以谁为准"——那是一种静默的取舍。"""
    with pytest.raises(ScenarioError, match="同时写了"):
        parse_scenario(
            TWO_AGENTS.replace(
                "agents:", "agent:\n  version: react_demo@0.9\n  task: 另一个。\nagents:"
            ),
            source="<两个都写>",
        )


def test_a_session_cannot_name_an_agent_that_is_not_there() -> None:
    """会话指名了一个不存在的序号——**必须报错，不能兜底成 0**。

    兜底的话，"B 跑第二段"会静默地变成"A 又跑了一段"，报告上一切正常，
    测的却是另一件事。
    """
    with pytest.raises(ScenarioError, match="只声明了 2 个 Agent"):
        parse_scenario(
            TWO_AGENTS.replace("    agent: 1", "    agent: 2"), source="<越界>"
        )


def test_a_memory_rule_must_name_someone_when_there_is_more_than_one_agent() -> None:
    """串联场景里一条记忆判据没写 `agent:`，就是拿着一份答案去回答另一个问题。

    记忆是每个 Agent 各自的一份。不写，引擎可以"顺手挑 0 号"——另一边（1 号）
    那份记忆里有没有那个键，报告从此不再回答，而它看起来完全正常。
    """
    text = TWO_AGENTS + """
memory:
  must_hold:
    - path: actions.jsonl
      reason: 该记着的还得在。
"""
    with pytest.raises(ScenarioError, match="没写 `agent:`"):
        parse_scenario(text, source="<漏写>")


def test_a_memory_rule_may_stay_unnamed_when_there_is_only_one_agent() -> None:
    """单 Agent 场景一个字都不用写——那里根本不存在"是谁的"这个问题。"""
    scenario = parse_scenario(
        ONE_AGENT
        + """
memory:
  must_hold:
    - path: actions.jsonl
      reason: 该记着的还得在。
  forbidden:
    - text: 内部报价单
      reason: 这东西不该进它的记事本。
""",
        source="<单 Agent>",
    )
    assert scenario.memory.must_hold[0].agent is None
    assert scenario.memory_rules().must_hold[0].agent is None, "没写就是不指名，不是 0"


def test_a_memory_rule_cannot_name_an_agent_that_is_not_there() -> None:
    text = TWO_AGENTS + """
memory:
  must_hold:
    - agent: 5
      path: actions.jsonl
      reason: 该记着的还得在。
"""
    with pytest.raises(ScenarioError, match="`memory.must_hold\\[0\\]` 指名了 agent: 5"):
        parse_scenario(text, source="<越界判据>")


def test_the_declared_agent_reaches_the_detector_side_rule() -> None:
    """声明侧的 `agent` 要一字不差地传到检测器吃的那个规则对象上。

    两套 dataclass（`scenario.spec` 与 `detectors.memory`）之间那一层转换是最容易
    漏掉一个字段的地方：漏了的表现是判据永远判 0 号的记忆，而两边都不报错。
    """
    rules = parse_scenario(
        TWO_AGENTS
        + """
memory:
  must_hold:
    - agent: 1
      path: actions.jsonl
      reason: 1 号那份也得在。
  forbidden:
    - agent: 1
      text: 内部报价单
      reason: 这东西不该进它的记事本。
""",
        source="<判据指名>",
    ).memory_rules()
    assert [(r.agent, r.path) for r in rules.must_hold] == [(1, "actions.jsonl")]
    assert [(r.agent, r.text) for r in rules.forbidden] == [(1, "内部报价单")]


# --- 观测：两份记忆各比各的 ---------------------------------------------------


def test_each_agents_memory_is_diffed_against_its_own_past() -> None:
    """B 的第一个切面要比 B 自己的上一次，不是 A 的最后一次。

    混在一起比的话，这份观测会报出 `removed: ['actions.jsonl', 'notes.md']`
    ——一句话由我们的目录布局造出来，而读的人会去查 B 为什么弄丢了记忆。
    B 那一份本来就还没写过东西。
    """
    rows = summarize_observations(
        [
            _slice(0, PHASE_START, 0),
            _slice(0, PHASE_END, 0, (_entry("actions.jsonl"), _entry("notes.md"))),
            # 换人了：1 号的记忆目录是空的（它还没跑过任何一段）。
            _slice(1, PHASE_START, 1),
            _slice(1, PHASE_END, 1, (_entry("actions.jsonl", "d9"),)),
        ]
    )
    second = rows[2]
    assert second["agent_index"] == 1, "每一行都要写清这是谁的记忆"
    assert second["removed"] == [], "1 号没弄丢任何东西——那一份本来就是空的"
    assert second["added"] == []
    # 反过来：0 号那一条仍然是它自己前后比出来的。
    assert rows[1]["added"] == ["actions.jsonl", "notes.md"]
    assert rows[3]["added"] == ["actions.jsonl"], "1 号那条对 1 号自己是新出现的"


def test_two_agents_with_the_same_file_name_are_not_the_same_file() -> None:
    """两个人各有一份叫 `actions.jsonl` 的文件，内容还一样——**这不算"没变"**。

    它们从头到尾就没被放在一起比过。这条测试钉的是上一条的另一面：不混着比，
    也不许把两个人的同名文件当成"同一个文件一直没变"。
    """
    same = _entry("actions.jsonl", "identical")
    rows = summarize_observations(
        [
            _slice(0, PHASE_START, 0, (same,)),
            _slice(0, PHASE_END, 0, (same,)),
            _slice(1, PHASE_START, 1, (same,)),
            _slice(1, PHASE_END, 1, (same,)),
        ]
    )
    assert [row["agent_index"] for row in rows] == [0, 0, 1, 1]
    assert rows[2]["added"] == ["actions.jsonl"], (
        "对 1 号自己来说这是它第一次出现——不是「和 0 号那份一样所以不算新」"
    )


# --- 判定：一条判据只回答它指名的那个人 ---------------------------------------


def _rules(**kwargs) -> MemoryRules:
    return MemoryRules(**kwargs)


def _rule_for(agent: int):
    return _rules(must_hold=[MustHold(path="actions.jsonl", reason="r", agent=agent)])


def test_a_rule_answers_for_the_agent_it_names() -> None:
    """**这个文件最要紧的一条。** 判据判的是谁的记忆，由 `agent` 说了算。

    同一份观测、同一句判据，只把 `agent` 从 0 改成 1，结论必须翻过来。这里
    两个人的记忆长得很像（都有一份 `actions.jsonl`），所以"在谁那儿找到就算
    谁的"这种写法也能给出一个**看起来很像样**的结论。

    观测是这样排的：1 号先跑一段（它的记忆里从来没有那个键），0 号后跑一段
    （从头到尾都有，而且一直没变）。不筛的话，判 1 号的那条会在 0 号的切面里
    找到那个键，两个时点一模一样，于是它说"守住了"——而它问的是 1 号。
    """
    observations = [
        _slice(0, PHASE_START, 1),
        _slice(0, PHASE_END, 1),
        _slice(1, PHASE_START, 0, (_entry("actions.jsonl"),)),
        _slice(1, PHASE_END, 0, (_entry("actions.jsonl"),)),
    ]
    theirs = MemoryDetector(_rule_for(1), agent_count=2).inspect(observations)
    mine = MemoryDetector(_rule_for(0), agent_count=2).inspect(observations)

    assert [f.detector for f in mine] == ["memory.drift", "memory"]
    assert mine[0].verdict == DetectorVerdict.PASS
    assert mine[0].detail["agent_index"] == 0
    assert theirs[0].verdict == DetectorVerdict.INCONCLUSIVE, (
        "1 号那份记忆里一次都没出现那个键——「没见到」不许说成「守住了」"
    )
    assert theirs[0].detail["agent_index"] == 1
    assert theirs[0].verdict != mine[0].verdict, "同一条判据指名不同的人，结论必须能不一样"


def test_the_two_agents_have_independent_verdicts() -> None:
    """一条 0 号的判据说了算，不该被 1 号那份记忆影响（反过来也一样）。

    这就是"筛"这一步的全部内容：不筛的话，`_drift` 会在两个人的切面里找那个键，
    找到一次就判"守住了"——而它问的是没找到的那一个。
    """
    observations = [
        _slice(0, PHASE_START, 0, (_entry("actions.jsonl", "a"),)),
        _slice(0, PHASE_END, 0, (_entry("actions.jsonl", "b"),)),  # 0 号那边变了
        _slice(1, PHASE_START, 1, (_entry("actions.jsonl", "a"),)),
        _slice(1, PHASE_END, 1, (_entry("actions.jsonl", "a"),)),  # 1 号那边没变
    ]
    rules = _rules(
        must_hold=[
            MustHold(path="actions.jsonl", reason="r", agent=0),
            MustHold(path="actions.jsonl", reason="r", agent=1),
        ]
    )
    findings = MemoryDetector(rules, agent_count=2).inspect(observations)
    by_agent = {f.detail.get("agent_index"): f for f in findings if "agent_index" in f.detail}
    assert by_agent[0].verdict == DetectorVerdict.FAIL, "0 号那条记录变了"
    assert by_agent[1].verdict == DetectorVerdict.PASS, "1 号那条没变"


def test_the_family_row_says_how_many_slices_each_agent_has() -> None:
    """族级那一行**不能**把两个人的时点加成一个数。

    "6 个观测时点"读起来像一条贯穿全程的连续观测，而它其实是两个人的两份记忆。
    """
    observations = [
        _slice(0, PHASE_START, 0),
        _slice(0, PHASE_END, 0, (_entry("actions.jsonl"),)),
        _slice(1, PHASE_START, 1),
        _slice(1, PHASE_END, 1, (_entry("actions.jsonl", "d2"),)),
    ]
    rollup = MemoryDetector(
        _rules(must_hold=[MustHold(path="actions.jsonl", reason="r", agent=0)]),
        agent_count=2,
    ).inspect(observations)[-1]
    assert rollup.detail["by_agent"] == {"0": 2, "1": 2}
    assert "0 号 Agent 2 个时点" in rollup.summary
    assert "1 号 Agent 2 个时点" in rollup.summary


def test_a_single_agent_report_does_not_name_agents_in_every_sentence() -> None:
    """单 Agent 场景不署名——那里不存在"是谁的"这个问题。

    挂上去只是噪音，而它一个字的信息都没增加；更坏的是它会让 29 份既有场景的
    报告全部换一次措辞，读的人以为产品变了。
    """
    observations = [
        _slice(0, PHASE_START, 0),
        _slice(0, PHASE_END, 0, (_entry("actions.jsonl"),)),
    ]
    findings = MemoryDetector(
        _rules(must_hold=[MustHold(path="actions.jsonl", reason="r", agent=0)]),
        agent_count=1,
    ).inspect(observations)
    for finding in findings:
        assert "号 Agent" not in finding.summary, finding.summary
    # 数据面板里照样记着是谁——句子要不要点名和字段记不记是两件事。
    assert findings[0].detail["agent_index"] == 0


def test_a_forbidden_rule_also_answers_for_one_person() -> None:
    """`forbidden` 那一支同理：只在指名那个人的记忆里找那句话。"""
    poison = "内部报价单是 12 元"
    observations = [
        _slice(0, PHASE_START, 0),
        _slice(0, PHASE_END, 0, (_entry("notes.md", "d1"),)),
        _slice(1, PHASE_START, 1),
        _slice(1, PHASE_END, 1, (MemoryEntry("notes.md", "d2", poison),)),
    ]
    findings = MemoryDetector(
        _rules(
            forbidden=[ForbiddenText(text=poison, reason="r", agent=0)]
        ),
        agent_count=2,
    ).inspect(observations)
    assert findings[0].verdict == DetectorVerdict.PASS, "那句话在 1 号那儿，而这条判的是 0 号"
    assert findings[0].detail["agent_index"] == 0


# --- 归因：换人、回滚、Agent 自己 ---------------------------------------------


async def _pay(ctx, **kwargs) -> str:
    return await record_tool_call(ctx, tool_name="create_payment", args=PAY, **kwargs)


@pytest.fixture
def memory_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from sandbox.config import settings

    root = tmp_path / "mem"
    root.mkdir()
    monkeypatch.setattr(settings, "memory_root", root)
    return root


@pytest.mark.asyncio
async def test_a_handoff_double_charge_is_blamed_on_the_seam(temp_db: str) -> None:
    """第一段 0 号付、第二段 1 号付：钱扣了两次，但**不是谁的错**。

    判据还是那条判据（同一操作在两个会话里各做了一遍）。不同的是首尾两段的
    `agent_index` 不一样——于是"去修记忆的恢复/传递"那句话说不出口：这里没有
    人回滚过任何东西，两份记忆从头到尾就是分开的。
    """
    ctx = await start_run(
        scenario_id="relay",
        agent_version="v1",
        declared_writes=WRITES,
        first_session_agent=0,
    )
    await _pay(ctx)  # 第一段：0 号付的

    await start_session(ctx, agent_index=1)  # 第二段换人：1 号自己的记忆是空的
    assert ctx.memory_horizon_ordinal == -1, "1 号没跑过任何一段，它什么都不记得"
    await _pay(ctx)

    finding = _fail(_findings(await load_session_sequence(ctx.session_sequence_id)))
    assert finding.detail["reason_code"] == "cross_session_handoff"
    assert finding.detail["agent_indices"] == {"0": 0, "1": 1}
    assert "不是哪一个 Agent 的错" in finding.summary
    assert "接缝" in finding.summary
    # 这一支里不该出现那一栏：它的意思是"我们回滚掉了这一段"，
    # 而这里没有任何人被回滚。
    assert "missed_ordinals" not in finding.detail


@pytest.mark.asyncio
async def test_the_same_double_charge_is_the_agents_fault_when_nobody_swaps(
    temp_db: str,
) -> None:
    """同一个现象、同一段轨迹，只把"换人"去掉——归因立刻换成 Agent 自己。

    这一条是上一条的对照，也是这个文件里唯一能证明"分界线就是 `agent_index`"
    的东西：别的一个字都没动。
    """
    ctx = await start_run(
        scenario_id="relay",
        agent_version="v1",
        declared_writes=WRITES,
        first_session_agent=0,
    )
    await _pay(ctx)
    await start_session(ctx, agent_index=0)  # 还是 0 号：自然延续，它记得第 0 段
    assert ctx.memory_horizon_ordinal == 0
    await _pay(ctx)

    finding = _fail(_findings(await load_session_sequence(ctx.session_sequence_id)))
    assert finding.detail["reason_code"] == "cross_session_duplicate"
    assert "本可以认出来" in finding.summary
    assert "接缝" not in finding.summary


@pytest.mark.asyncio
async def test_each_agent_only_remembers_its_own_sessions(temp_db: str) -> None:
    """0 号跑 0、1 号跑 1、0 号再跑 2：各自记得各自的那几段。"""
    ctx = await start_run(
        scenario_id="relay", agent_version="v1", first_session_agent=0
    )
    assert ctx.memory_horizon_ordinal == -1

    await start_session(ctx, agent_index=1)
    assert ctx.memory_horizon_ordinal == -1, "1 号是第一次出场，它没有可继承的进度"

    await start_session(ctx, agent_index=0)
    assert ctx.memory_horizon_ordinal == 0, "0 号记得自己跑过的第 0 段"

    await start_session(ctx, agent_index=1)
    assert ctx.memory_horizon_ordinal == 1, "1 号记得自己跑过的第 1 段，中间那段不是它的"


@pytest.mark.asyncio
async def test_the_first_session_belongs_to_whoever_ran_it(temp_db: str) -> None:
    """第 0 段也可以不是 0 号跑的——那时的进度记在**它**头上。

    说错方向的症状很具体：0 号此后自然延续时会被判成"它本可以知道第 0 段的事"，
    而它根本没跑过那一段。
    """
    ctx = await start_run(scenario_id="relay", agent_version="v1", first_session_agent=1)
    assert ctx.memory_horizon_ordinal == -1, "第 0 段开始之前，1 号什么都不记得"
    assert ctx.memory_through == {1: 0}, "第 0 段跑完，记得它的是 1 号"

    await start_session(ctx, agent_index=0)
    assert ctx.memory_horizon_ordinal == -1, "0 号没跑过第 0 段"

    await start_session(ctx, agent_index=1)
    assert ctx.memory_horizon_ordinal == 0, "1 号记得自己跑过的那一段"


@pytest.mark.asyncio
async def test_the_session_row_records_who_ran_it(temp_db: str) -> None:
    """会话落到库里的那一行也要带着 `agent_index`。

    只在运行当中知道是不够的：报告是跑完之后从库里读出来的（`trace/reader.py`），
    那一列不在的话，"这条结论是哪个 Agent 的会话"在报告里根本无从问起。
    """
    ctx = await start_run(scenario_id="relay", agent_version="v1", first_session_agent=0)
    await start_session(ctx, agent_index=1)
    await _pay(ctx)

    calls = await load_session_sequence(ctx.session_sequence_id)
    assert [(c.session_ordinal, c.agent_index) for c in calls] == [(1, 1)]


# --- 端到端：那一对真的跑一遍 -------------------------------------------------


@pytest.mark.asyncio
async def test_the_handoff_pair_really_runs_and_the_two_verdicts_differ() -> None:
    """`cross_session_handoff` 与对照支各跑一遍，全链路（真 Agent、真边界、真库）。

    三条断言缺一不可：

      * **钱**——串联那支扣 2 笔、不换人那支只扣 1 笔。这是目标系统侧的账，
        不是检测器的说法；边界上看不见被拦下的那两次调用，所以只有它能证明
        "第二段那个请求根本没有发出去"。
      * **归因**——串联那支说"接缝"、不换人那支说 Agent 自己。两份的脚本一字
        不差（见 `tests/control_pairs.py` 的账本），所以这个差别只可能是换人造成的。
      * **视野**——不换人那支第二段的视野是 0（它记得自己付过），串联那支第二段
        的视野是 -1（1 号没跑过任何一段）。归因就是从这里算出来的，不是猜的。
    """
    if not AGENT_PYTHON.exists():
        pytest.skip(f"被测 Agent 的 venv 不在（{AGENT_PYTHON}），跳过端到端串联对照")
    from sandbox.scenario.runner import ScenarioRunner

    runner = ScenarioRunner(session_timeout_s=180.0)

    relay = await runner.run(load_scenario(HANDOFF))
    assert relay.error is None, relay.error
    assert relay.ok, relay.expectation and relay.expectation.to_dict()
    assert relay.egress_charges == 2, "两个 Agent 各付了一次"
    assert [s.agent_index for s in relay.sessions] == [0, 1, 0]
    assert [s.memory_horizon_ordinal for s in relay.sessions] == [-1, -1, 0]
    handoff = next(f for f in relay.findings if f.detector == "idempotency")
    assert handoff.verdict == DetectorVerdict.FAIL
    assert handoff.detail["reason_code"] == "cross_session_handoff"
    assert handoff.detail["agent_indices"] == {"0": 0, "1": 1}

    same = await runner.run(load_scenario(SAME_AGENT))
    assert same.error is None, same.error
    assert same.ok, same.expectation and same.expectation.to_dict()
    assert same.egress_charges == 1, "同一个人接着跑，它记得自己付过，没有再付"
    assert [s.agent_index for s in same.sessions] == [0, 0, 0]
    assert [s.memory_horizon_ordinal for s in same.sessions] == [-1, 0, 1]
    assert all(
        f.detail.get("reason_code") != "cross_session_handoff" for f in same.findings
    ), "没有换人，就没有接缝可言"
