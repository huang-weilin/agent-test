"""回归用例的存取。

`diff.py` 是纯函数——只认 `Snapshot`，不碰数据库。这一层负责把 ORM 行翻译成
`Snapshot`，以及建用例、列用例。

分开的理由不只是"分层好看"：基线侧和当前侧走的是**同一条比较代码路径**。
如果两边各写一份取数和比较，它们迟早会漂移成两套判据，于是"用哪种方式看"
会改变结论——那是最难被发现的一类错误。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select

from sandbox.db.engine import session_scope
from sandbox.db.models import RegressionCase, TestRun
from sandbox.regression.diff import Comparison, Snapshot, compare


def _loads(value: str | None) -> dict[str, Any] | None:
    """把 JSON 列解析成字典。

    **解析失败返回 None 而不是抛异常。** 这些列是历史记录，一条读不出来的
    记录不该让整个页面打不开——它应该表现为"这条没有记录"，然后在界面上
    如实显示成缺失。抛异常的话，用户看到的是一片 500，而问题只是一行旧数据。
    """
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def snapshot_of_run(run: TestRun) -> Snapshot:
    return Snapshot(
        run_id=run.id,
        verdict_ok=run.verdict_ok,
        report=_loads(run.report_json),
        config_scheme=run.config_scheme,
        config_hash=run.config_hash,
        config_parts=_loads(run.config_parts_json),
        env=_loads(run.env_json),
    )


def snapshot_of_case(case: RegressionCase) -> Snapshot:
    """用例冻结下来的那一侧。

    `run_id` 用的是出处运行的 id，好让差异报告里指得出"基线是哪一次"。
    出处运行不在了也不影响——这里用的是抄下来的列，不是外键。
    """
    return Snapshot(
        run_id=case.baseline_run_id,
        verdict_ok=case.baseline_verdict_ok,
        report=_loads(case.baseline_report_json),
        config_scheme=case.baseline_config_scheme,
        config_hash=case.baseline_config_hash,
        config_parts=_loads(case.baseline_config_parts_json),
        env=_loads(case.baseline_env_json),
    )


@dataclass(frozen=True)
class CaseRow:
    """用例列表里的一行。"""

    id: str
    name: str
    scenario_id: str
    note: str
    baseline_run_id: str
    baseline_verdict_ok: bool | None
    baseline_started_at: datetime | None
    created_at: datetime

    @property
    def baseline_ok(self) -> bool | None:
        return self.baseline_verdict_ok


class NoConclusionError(ValueError):
    """这次运行没有留下结论，当不了基线。"""


async def create_case(*, run_id: str, name: str, note: str = "") -> str:
    """把一次运行钉成基线。返回用例 id。

    从**运行记录**取数，不是从页面上传来的结论——传进来的话，一条被改过的
    请求就能造出一条基线，而它是所有后续比对的判据。
    """
    async with session_scope() as db:
        run = await db.get(TestRun, run_id)
        if run is None:
            raise KeyError(f"运行 {run_id} 不存在")
        if run.verdict_ok is None and run.report_json is None:
            # **拒绝没有结论的运行。** 规则放在这里而不是路由里：将来多一个
            # 调用方（命令行、导入脚本）就会绕过路由上的检查，而这条规则
            # 一旦破了，坏掉的是一条用例里再也改不掉的判据。
            #
            # 为什么必须拒：拿"未留存"当基线，之后每一次比对都会报出
            # "未留存 → 不通过"这样一个假的差异，而它看起来像是真的抓到了
            # 一次退化。没有结论的运行（跑崩了、在这个字段存在之前跑的）
            # 对它自己都还没有结论，更当不了判据。
            raise NoConclusionError(
                f"运行 {run_id} 没有留下结论（它可能跑崩了，或者是在结论字段"
                f"存在之前跑的），不能当基线。"
            )
        case = RegressionCase(
            id=str(uuid.uuid4()),
            name=name.strip() or f"{run.scenario_id} 的基线",
            scenario_id=run.scenario_id,
            note=note.strip(),
            baseline_run_id=run.id,
            baseline_verdict_ok=run.verdict_ok,
            baseline_report_json=run.report_json,
            baseline_config_scheme=run.config_scheme,
            baseline_config_hash=run.config_hash,
            baseline_config_parts_json=run.config_parts_json,
            baseline_env_json=run.env_json,
        )
        db.add(case)
        return case.id


async def get_case(case_id: str) -> RegressionCase | None:
    async with session_scope() as db:
        return await db.get(RegressionCase, case_id)


async def find_run(run_id: str) -> TestRun | None:
    """按 id 取一次运行。找不到返回 None，由调用方决定退回哪儿。

    **不抛异常**：这个 id 从查询串里来，用户手里的旧链接会指向一次已经不
    存在的运行，而那不是错误——退回"最近一次"的默认行为才对。
    """
    async with session_scope() as db:
        return await db.get(TestRun, run_id)


async def list_cases(limit: int = 200) -> list[CaseRow]:
    async with session_scope() as db:
        cases = list(
            (
                await db.execute(
                    select(RegressionCase).order_by(RegressionCase.created_at.desc()).limit(limit)
                )
            ).scalars()
        )
        # 基线的开始时间要回运行表查——它是"这条基线钉在哪一次运行上"的一部分，
        # 而用例自己只冻结结论，不冻结运行的时间线。**查不到就是 None，不报错**：
        # 那次运行可能已经被清理掉，而用例本身仍然是完整的（见模型 docstring）。
        started: dict[str, datetime] = {}
        run_ids = [c.baseline_run_id for c in cases]
        if run_ids:
            probes = await db.execute(
                select(TestRun.id, TestRun.started_at).where(TestRun.id.in_(run_ids))
            )
            started = {row[0]: row[1] for row in probes}
        return [
            CaseRow(
                id=c.id,
                name=c.name,
                scenario_id=c.scenario_id,
                note=c.note,
                baseline_run_id=c.baseline_run_id,
                baseline_verdict_ok=c.baseline_verdict_ok,
                baseline_started_at=started.get(c.baseline_run_id),
                created_at=c.created_at,
            )
            for c in cases
        ]


async def compare_with_run(case: RegressionCase, run_id: str) -> Comparison | None:
    """拿这条用例的基线，和某次运行比。运行不在就返回 None。"""
    async with session_scope() as db:
        run = await db.get(TestRun, run_id)
        if run is None:
            return None
        return compare(snapshot_of_case(case), snapshot_of_run(run))


async def latest_run_for(case: RegressionCase) -> TestRun | None:
    """这个场景在**这条用例建立之后**跑的那一次，最近的一回。

    页面打开时默认比它——"我上次跑完是什么样"是最常问的问题，比让用户先去
    运行列表里挑一次要顺手得多。比的是最近一次而不是"刚跑的那次"：如果用户
    用别的方式跑过（命令行、另一个标签页），最近一次才是最新的结论。

    **两个排除条件都是必须的**，而且它们各自挡掉一句骗人的话：

    * 排除基线那次运行。基线往往就是那个场景当时最近的一次运行（人总是在
      一次跑得对的运行上点"存为用例"），不排除的话页面会拿基线和它自己比，
      渲染出一句"配置一致，结论未变"——什么都没说，却读起来像"我确认过，
      没退化"。
    * 排除**建立用例之前**跑的那些。回归问的是"我认定这个结论是对的之后，
      它还好不好"，而一条早于这个认定的运行回答不了这个问题。不排除的话，
      用户刚建完用例、还没重跑，页面就会拿一次旧运行比出一句"结论未变"，
      而那句话读起来像是回归已经验过了。

    用 `created_at` 而不是基线运行的 `started_at` 当分界：前者是用户**做出
    断言**的时刻，后者只是那次执行开始的时刻。断言之后才是回归。
    """
    async with session_scope() as db:
        return (
            await db.execute(
                select(TestRun)
                .where(
                    TestRun.scenario_id == case.scenario_id,
                    TestRun.id != case.baseline_run_id,
                    TestRun.started_at > case.created_at,
                )
                .order_by(TestRun.started_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
