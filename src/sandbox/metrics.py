"""指标看板（需求 3.7）要的三个数：结论分布、不通过的原因、延迟。

需求那一行的原文是「指标看板 | 展示通过率、成本、延迟等核心指标」。成本那一半
在 `cost/aggregator.py`（它已经带一页 `/cost`），这里管另外两个，以及**这两个
各自的口径**。

这个模块的全部内容就是口径，所以它只做三件事，每一件后面都有一条"不做就会说错
话"：

1. **结论只按 `verdict_ok` 分，不按 `status` 分。** 这两件事在本项目里是分开
   的，而它们分开的方式恰恰最容易被误读：`loop` 场景里被熔断杀掉的运行
   `status="killed"`，而它是**通过**的（检测器报出来了、看门狗动手了——那正是
   场景想看到的东西）。按状态分类的话，一条通过会被算成一个失败。

2. **"不通过"要拆开报。** `ok` 要四个条件同时成立（执行没出错、前提成立、
   地面真值对得上、预期达成），而它们在运行列表里长得一模一样：一个红叉。
   把四种不同的毛病压成一个数，读者只能猜。

3. **延迟有三个，口径不同的两个不合并。** 模型调用、工具调用、运行墙钟各有各
   的边界，其中流式与非流式的模型调用连"一次"的含义都不同。

## 那个数不叫「通过率」

`ScenarioResult.ok` 回答的是"这次运行的结果和场景里写下的预期一致吗"，不是
"这个 Agent 合格吗"。一次未达成有两种相反的成因：

* 出现了没被任何预期覆盖的 fail 结论——多半是被测对象做了一件我们没料到的事；
* 预期没命中——**多半是我们预料它做的那个错没出现**。那往往说明用例要改
  （或者检测器漏报了），而不是产品有缺陷。

从运行记录里分不出"Agent 真的没犯错"和"检测器没报出来"（`scenario/expect.py`
的模块注释讲了为什么：那需要脚本消费量这个额外信息）。所以这个数在本项目里
叫**预期达成率**，页面上也这么写，并附上这句说明。叫它「通过率」的话，读的人
会把它当成"Agent 的合格率"，而那是它证明不了的东西。
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sandbox.cost.aggregator import CostRollup, for_runs
from sandbox.db.engine import session_scope
from sandbox.db.models import LlmCall, TestRun, ToolCall

# --- 不通过的四类原因 ---------------------------------------------------------

EXECUTION = "execution"
PREMISE = "premise"
CHARGES = "charges"
EXPECTATION = "expectation"

#: 每一类的一句话解释。**写在这里而不是模板里**：判据与措辞分开维护的话，
#: 改了一处忘另一处，页面就会说出一个和代码不符的理由。
_CAUSE_TEXT: dict[str, tuple[str, str]] = {
    EXECUTION: (
        "没跑到能判定的地方",
        "执行中途出了错，或者一段会话都没跑完，于是没有可比的预期。"
        "这一次的结论说的是引擎自己，不是被测对象。",
    ),
    PREMISE: (
        "前提不成立",
        "脚本没被走完，或者世界的时间没走对。检测器是在一段不成立的轨迹上下的"
        "结论，所以那句「没发现问题」指的可能是别的事情。",
    ),
    CHARGES: (
        "目标系统侧的扣款对不上",
        "这是地面真值：钱到底扣了几次，与场景声明的预期不符。"
        "检测器完全可能报出一条漂亮的 pass，而钱其实扣了两次。",
    ),
    EXPECTATION: (
        "预期未达成",
        "场景写下的预期与实际结论对不上。这一行不等于「Agent 做错了事」——"
        "预期没命中同样可能意味着我们预料的那个错根本没出现。",
    ),
}

#: `ScenarioResult.to_dict()` 里判 `ok` 用到的那几项。少任何一项就整条读不出来，
#: 因为缺的那一项既可能是"真没问题"，也可能是"当时还没这一项"——不猜。
_REPORT_KEYS = ("error", "premises_held", "charges_ok", "expectation")


@dataclass(frozen=True)
class RunSample:
    """一条运行记录里与指标有关的那几列。

    单独成形是为了让 `summarize` 不碰数据库：判断的规则要能脱离库测。
    """

    run_id: str
    scenario_id: str
    status: str
    verdict_ok: bool | None
    started_at: datetime
    completed_at: datetime | None
    #: 运行当时写下的结论快照。None = 没留存（在本字段存在之前跑的），
    #: **不是"没通过"**。
    report: dict[str, Any] | None


# --- 结论分布 -----------------------------------------------------------------


@dataclass(frozen=True)
class OutcomeTally:
    """一批运行的结论分布。

    `no_verdict_by_status` 把"没留结论"的那些按 `status` 原值拆开——**原值，
    不重新归类**：多出来的状态原样出现，而不是掉进一个「其他」里消失。
    """

    considered: int
    passed: int
    failed: int
    no_verdict: int
    no_verdict_by_status: tuple[tuple[str, int], ...] = ()

    @property
    def decided(self) -> int:
        """留了结论的运行数。**这才是达成率的分母。**"""
        return self.passed + self.failed

    @property
    def rate(self) -> float | None:
        """预期达成率。没有一条留下结论时是 None——**不是 0，也不是 1**。

        0 会被读成"全都不达标"，1 会被读成"全绿"，而这两种读法都在替一批
        什么都没判的运行编结论。界面上显示"没有样本"。
        """
        if not self.decided:
            return None
        return self.passed / self.decided


@dataclass(frozen=True)
class ScenarioTally:
    scenario_id: str
    tally: OutcomeTally


@dataclass(frozen=True)
class Cause:
    """不通过的一类原因，以及它在这一批里出现了几次。"""

    code: str
    label: str
    why: str
    runs: int


@dataclass(frozen=True)
class FailureLedger:
    """不通过的运行，按原因拆开。

    **一次运行可能同时占好几行**（比如执行崩了，预期自然也没达成），所以各行
    相加会大于 `failed`。这句话要写在页面上：那几行数字看起来像是可以相加的。

    `unreadable` 是不通过、但原因读不出来的条数：报告没留存，或者留存的那份
    缺了判 `ok` 要用的字段。**它们不进任何一行原因**——按缺失的字段反推原因，
    就是替一条没有证据的运行编一个理由。
    """

    failed: int
    causes: tuple[Cause, ...]
    unreadable: int
    #: 预期未达成那两类各出现在**几条运行**上（可重叠：同一条运行可能两类都有）。
    #: 单位是运行条数，和上面每一行一样——一张表里混两种单位，读的人会把它们
    #: 加在一起。
    missing_runs: int = 0
    unexpected_runs: int = 0


# --- 延迟 ---------------------------------------------------------------------


@dataclass(frozen=True)
class LatencyStats:
    """一组耗时。**报中位数和最大值，不报平均。**

    平均在这里特别有害：一次被超时强杀的工具调用（默认 180 秒）能把平均拉到
    一个任何一次真实调用都没见过的数，而它恰恰是读者最该看见的那个点——被
    平均掉了。中位数对长尾不敏感，最大值把长尾本身摆出来，两个一起看才知道
    这批里"通常多快"和"最坏多慢"。

    取值方法写死：偶数个取靠上的那一个（不取两个的中间值）。中间值会造出一个
    没有任何一次观测取过的数，而这里报的每个数都该是量到过的。
    """

    count: int
    p50_ms: int
    max_ms: int

    @classmethod
    def of(cls, values: Sequence[int]) -> LatencyStats:
        if not values:
            return cls(count=0, p50_ms=0, max_ms=0)
        ordered = sorted(values)
        return cls(count=len(ordered), p50_ms=ordered[len(ordered) // 2], max_ms=ordered[-1])


@dataclass(frozen=True)
class LatencyReport:
    """三个口径的延迟。**不合并**，理由见模块注释与各字段。"""

    #: 非流式模型调用：从请求发出到收到完整响应。
    model_calls: LatencyStats
    #: 流式模型调用：**到最后一帧为止**。含义与非流式那一栏不同，所以各算各的
    #: （`LlmCall.streaming` 就是为这个区分留的）。
    model_calls_streaming: LatencyStats
    #: 工具调用：出口代理到目标系统的往返。**不含看门狗的时间**——
    #: `latency_ms` 在 `enforce` 之前就量完了，而看门狗是在请求里 `await` 的。
    #: 所以它不是"Agent 等这个工具等了多久"。
    tool_calls: LatencyStats
    #: 运行墙钟：从运行登记到执行收尾。真实时间（虚拟时钟只拨世界时间，不拨它），
    #: 但**不含检测器与报告**——收尾在检测之前。
    run_wall: LatencyStats
    #: 没有 `completed_at`、因而算不出墙钟的运行数（还在跑，或者没走到收尾）。
    runs_without_wall: int = 0


@dataclass(frozen=True)
class OutcomeSummary:
    outcomes: OutcomeTally
    failures: FailureLedger
    by_scenario: tuple[ScenarioTally, ...]


@dataclass(frozen=True)
class MetricsReport:
    limit: int
    scenario_id: str | None
    outcomes: OutcomeTally
    failures: FailureLedger
    by_scenario: tuple[ScenarioTally, ...]
    latency: LatencyReport
    #: 这批运行（按运行 id）的总花费。与其余几栏**同一批运行**，理由见 `report`。
    cost: CostRollup

    @property
    def scope_text(self) -> str:
        """这批指标是哪些运行上的。**写在页面上**：不说的话，读者会以为
        "预期达成率"是历史全量，而用例集是一直在变的。"""
        where = f"场景 {self.scenario_id} 的" if self.scenario_id else ""
        return f"{where}最近 {self.limit} 次运行（实际取到 {self.outcomes.considered} 次）"


# --- 纯逻辑（不碰库，可以直接喂样本测） ---------------------------------------


def _tally(samples: Sequence[RunSample]) -> OutcomeTally:
    passed = sum(1 for s in samples if s.verdict_ok is True)
    failed = sum(1 for s in samples if s.verdict_ok is False)
    blank = [s for s in samples if s.verdict_ok is None]
    return OutcomeTally(
        considered=len(samples),
        passed=passed,
        failed=failed,
        no_verdict=len(blank),
        no_verdict_by_status=tuple(sorted(Counter(s.status for s in blank).items())),
    )


def _causes_of(report: dict[str, Any] | None) -> tuple[str, ...] | None:
    """这次运行为什么不通过。None = 读不出来。

    返回的是**全部**成立的原因，不是最像的那一个：判断"哪个更要紧"需要业务
    语义，而那是场景声明的事。
    """
    if report is None:
        return None
    if any(key not in report for key in _REPORT_KEYS):
        return None
    out: list[str] = []
    expectation = report["expectation"]
    if report["error"] is not None or expectation is None:
        out.append(EXECUTION)
    if not report["premises_held"]:
        out.append(PREMISE)
    if not report["charges_ok"]:
        out.append(CHARGES)
    if isinstance(expectation, dict) and (
        expectation.get("missing") or expectation.get("unexpected")
    ):
        out.append(EXPECTATION)
    return tuple(out)


def _ledger(samples: Sequence[RunSample]) -> FailureLedger:
    failed = [s for s in samples if s.verdict_ok is False]
    counts: Counter[str] = Counter()
    unreadable = 0
    missing = 0
    unexpected = 0
    for sample in failed:
        causes = _causes_of(sample.report)
        if causes is None:
            unreadable += 1
            continue
        counts.update(causes)
        expectation = (sample.report or {}).get("expectation")
        if isinstance(expectation, dict):
            missing += 1 if expectation.get("missing") else 0
            unexpected += 1 if expectation.get("unexpected") else 0
    return FailureLedger(
        failed=len(failed),
        causes=tuple(
            Cause(code=code, label=label, why=why, runs=counts.get(code, 0))
            for code, (label, why) in _CAUSE_TEXT.items()
        ),
        unreadable=unreadable,
        missing_runs=missing,
        unexpected_runs=unexpected,
    )


def summarize(samples: Sequence[RunSample]) -> OutcomeSummary:
    """结论分布 + 不通过的原因 + 按场景拆。纯函数，样本从外面给。"""
    by_scenario: dict[str, list[RunSample]] = {}
    for sample in samples:
        by_scenario.setdefault(sample.scenario_id, []).append(sample)
    return OutcomeSummary(
        outcomes=_tally(samples),
        failures=_ledger(samples),
        by_scenario=tuple(
            ScenarioTally(scenario_id=sid, tally=_tally(rows))
            # 场景之间按 id 排序，不按出现顺序：本函数拿到的样本顺序取决于调用方，
            # 界面上的表不该因此换一个排列。
            for sid, rows in sorted(by_scenario.items())
        ),
    )


def run_wall_stats(samples: Sequence[RunSample]) -> tuple[LatencyStats, int]:
    """运行墙钟那一栏。与其余三栏分开，是为了让这一半也能纯测。

    返回 (统计, 算不出来的条数)。**两种"算不出来"都不是 0**：没有
    `completed_at` 的运行（还在跑、或者没走到收尾）根本不进统计，它们被数出来
    单独说——把它们当 0 秒塞进去会把中位数拉低。
    """
    deltas = [
        int((s.completed_at - s.started_at).total_seconds() * 1000)
        for s in samples
        if s.completed_at is not None
    ]
    return LatencyStats.of(deltas), sum(1 for s in samples if s.completed_at is None)


# --- 读库 ---------------------------------------------------------------------


def _report_of(raw: str | None) -> dict[str, Any] | None:
    """与 `api.queries.parse_report` 同一件事，但这里只要一个 dict。

    重复的这两行是刻意的：本模块不 import `api`（那是 HTTP 层），而 `api` 那边
    import 本模块。读法必须与那边一致——两边都只做 `json.loads`，没有兜底、
    没有默认值：解析不了就是没留存，不编一份空报告出来。
    """
    if not raw:
        return None
    parsed: dict[str, Any] = json.loads(raw)
    return parsed


async def _sample_runs(
    db: AsyncSession, *, limit: int, scenario_id: str | None
) -> list[RunSample]:
    stmt = select(TestRun).order_by(TestRun.started_at.desc()).limit(limit)
    if scenario_id is not None:
        stmt = stmt.where(TestRun.scenario_id == scenario_id)
    runs = list((await db.execute(stmt)).scalars())
    return [
        RunSample(
            run_id=r.id,
            scenario_id=r.scenario_id,
            status=r.status,
            verdict_ok=r.verdict_ok,
            started_at=r.started_at,
            completed_at=r.completed_at,
            report=_report_of(r.report_json),
        )
        for r in runs
    ]


async def report(
    *,
    limit: int = 50,
    scenario_id: str | None = None,
    db: AsyncSession | None = None,
) -> MetricsReport:
    """一次取齐看板要的全部数字。

    **所有栏位取的是同一批运行。** 一个按"最近 50 次"算、另一个按全量算的话，
    页面上两栏说的就不是一回事，而它们看起来同样正常——两个口径不一致，比没有
    口径危险得多。成本那一格也是这一批的（`cost_of_runs`），**不是**
    `cost.by_scenario()` 那种全量口径：成本页有自己的入口，这里只回答"我眼下
    看的这批花了多少"。
    """

    async def _go(session: AsyncSession) -> MetricsReport:
        samples = await _sample_runs(session, limit=limit, scenario_id=scenario_id)
        run_ids = [s.run_id for s in samples]
        summary = summarize(samples)
        return MetricsReport(
            limit=limit,
            scenario_id=scenario_id,
            outcomes=summary.outcomes,
            failures=summary.failures,
            by_scenario=summary.by_scenario,
            cost=await cost_of_runs(run_ids, db=session),
            latency=await _latency(session, run_ids, samples),
        )

    if db is not None:
        return await _go(db)
    async with session_scope() as session:
        return await _go(session)


async def _latency(
    db: AsyncSession, run_ids: list[str], samples: Sequence[RunSample]
) -> LatencyReport:
    wall, without_wall = run_wall_stats(samples)
    empty = LatencyStats.of([])
    if not run_ids:
        return LatencyReport(
            model_calls=empty,
            model_calls_streaming=empty,
            tool_calls=empty,
            run_wall=wall,
            runs_without_wall=without_wall,
        )
    llm_rows = (
        await db.execute(
            select(LlmCall.streaming, LlmCall.latency_ms).where(LlmCall.test_run_id.in_(run_ids))
        )
    ).all()
    tool_rows = (
        await db.execute(select(ToolCall.latency_ms).where(ToolCall.test_run_id.in_(run_ids)))
    ).scalars()
    return LatencyReport(
        model_calls=LatencyStats.of([int(r[1]) for r in llm_rows if not r[0]]),
        model_calls_streaming=LatencyStats.of([int(r[1]) for r in llm_rows if r[0]]),
        tool_calls=LatencyStats.of([int(ms) for ms in tool_rows]),
        run_wall=wall,
        runs_without_wall=without_wall,
    )


async def cost_of_runs(run_ids: list[str], *, db: AsyncSession | None = None) -> CostRollup:
    """这批运行一共花了多少。**转手给 `cost/aggregator.py`**，本模块不重算。

    存在的理由是口径：看板上这一格必须和上面几栏是同一批运行。数怎么算仍然
    只有成本那边一个来源。
    """
    if not run_ids:
        return CostRollup()
    rollups = await for_runs(run_ids, db=db)
    total = CostRollup()
    for rollup in rollups.values():
        total = total + rollup
    return total
