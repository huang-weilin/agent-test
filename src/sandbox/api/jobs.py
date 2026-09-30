"""跑场景的作业登记表。**一次只跑一个。**

并发之所以必须禁止，不是保守，是执行器现在**物理上不能并发**：它在运行期间
改进程级全局 `settings.tool_upstream_base_url`，把出口代理指向本次运行自己
那个 mock 的端口，跑完才复原（见 `scenario/runner.py`）。两个运行重叠的话，
后启动的那个会把前一个的上游地址改掉——前一个的代理于是去敲**后一个的 mock**，
两个运行的数据互相写进对方的目标系统，而**两边的报告都看起来正常**。
`runs.clear()` 也会互相清掉对方的路由。

这正是本项目一直在防的那类失效：报告读起来没问题，结论是错的。

所以这里**不做队列**。第二个请求直接拒绝，并且明说"上一个还没跑完"。排队
看起来更友好，但用户点了"运行"之后对着一个没有任何解释的转圈等一分钟，
和收到一句"`clean` 正在跑，等它结束"相比，后者才是可理解的。
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog

from sandbox.scenario.runner import ScenarioResult, ScenarioRunner
from sandbox.scenario.spec import Scenario

log = structlog.get_logger(__name__)

# 保留多少个已完成的作业供轮询。运行一旦结束，它的**结论**就在库里
# （`TestRun.report_json`），内存里这份只是为了"刚点完那一下"的重定向。
# 留几个是为了宽容一点：用户开了两个标签页点同一个场景。
_HISTORY = 20


@dataclass
class Job:
    """一次"点运行"的执行过程。

    `id` 与 `run_id` 是**两个东西**，不要合并：作业 id 在点下去的那一刻就
    有了（界面靠它轮询），而 `run_id` 要等执行器把三个服务和运行记录都建起来
    才有。执行器在非常早的阶段就失败时，`run_id` 会一直是 None——那时也
    确实没有任何运行记录可指。
    """

    id: str
    scenario_id: str
    started_at: datetime
    state: str = "running"  # running | done
    finished_at: datetime | None = None
    run_id: str | None = None
    ok: bool | None = None
    error: str = ""
    result: ScenarioResult | None = field(default=None, repr=False)
    #: 跑完之后跳到哪儿。None = 跳这次运行的详情页。
    #:
    #: 从回归用例点"一键回归测试"进来时放的是对比页——跑完直接看差异，
    #: 不用再点一次。**不由请求里的参数决定**（那会变成一个开放重定向），
    #: 调用方在服务端拼好，见 `_safe_return_to`。
    return_to: str | None = None

    @property
    def is_running(self) -> bool:
        return self.state == "running"


class JobRegistry:
    """当前在跑的作业 + 最近完成的几个。"""

    def __init__(self, *, runner_factory: type[ScenarioRunner] = ScenarioRunner) -> None:
        self._active: Job | None = None
        self._done: list[Job] = []
        self._runner_factory = runner_factory
        self._tasks: set[asyncio.Task[None]] = set()

    # --- 查询 -------------------------------------------------------------

    @property
    def active(self) -> Job | None:
        return self._active

    def get(self, job_id: str) -> Job | None:
        if self._active is not None and self._active.id == job_id:
            return self._active
        return next((j for j in self._done if j.id == job_id), None)

    # --- 启动 -------------------------------------------------------------

    def start(self, scenario: Scenario, *, return_to: str | None = None) -> Job | None:
        """占住执行位并起一个后台任务。已经有运行在跑时返回 None。

        `_active` 的"检查后置位"没有加锁，这是**对的**：asyncio 是单线程的，
        而这中间没有 `await`，所以从事件循环的视角看它是一步完成的原子操作。
        加个 `asyncio.Lock` 反而会让人以为需要跨线程同步。
        """
        if self._active is not None:
            return None

        job = Job(
            id=str(uuid.uuid4()),
            scenario_id=scenario.id,
            started_at=datetime.now(UTC),
            return_to=return_to,
        )
        self._active = job
        task = asyncio.create_task(self._execute(job, scenario))
        # 持住引用：asyncio 只对 task 持弱引用，不存的话它可能被 GC 掉，
        # 表现为"任务跑到一半静默消失"。
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    async def _execute(self, job: Job, scenario: Scenario) -> None:
        try:
            result = await self._runner_factory().run(scenario)
        except asyncio.CancelledError:
            job.state = "done"
            job.error = "服务关闭，这次运行被中断。"
            job.finished_at = datetime.now(UTC)
            self._retire(job)
            raise
        except Exception as exc:  # noqa: BLE001 — 作业的异常要进页面，不是崩掉服务
            log.exception("api.job_failed", scenario=job.scenario_id)
            job.error = f"{type(exc).__name__}: {exc}"
        else:
            job.run_id = result.run_id or None
            job.ok = result.ok
            job.error = result.error or ""
            job.result = result

        job.state = "done"
        job.finished_at = datetime.now(UTC)
        self._retire(job)

    def _retire(self, job: Job) -> None:
        if self._active is job:
            self._active = None
        self._done.append(job)
        del self._done[:-_HISTORY]
