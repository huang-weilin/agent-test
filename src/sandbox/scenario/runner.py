"""场景执行器：把一份 `Scenario` 变成一次真实的、有结论的运行。

**它与 `scripts/run_walking_skeleton.py` 的分工。** 那个脚本是"证明这条链路
跨进程成立"的演示，场景硬编码在它自己的字典里。这个模块是"跑任意场景"的引擎，
场景来自数据。两者跑的是同一套服务，走的是同一条路径。

执行顺序（一个场景 = 一次 TestRun = N 个 Session）：

    init_db → 起三个服务 → 开运行（带场景的工具表）
      → 装拼接后的上游脚本
      → 对每个会话：
            [恢复记忆] → start_session → [拍快照]
            → 拉起 Agent 子进程 → 等它退出
            → [拍快照] → **检查这一段的脚本被消费掉了多少**
      → finish_run → 跑检测器 → 比对预期 → 报告

三个决定值得单独说明，因为它们都是"不做就会静默出错"的那类：

1. **上游脚本按会话拼接，但消费量按会话检查。** 单游标顺序读没问题，问题是
   Agent 提前退出时，游标停在中间——下一个会话会读到**上一个会话**的步骤，
   于是它做的事和场景写的完全不是一回事，而检测器照样给出一份看起来正常的
   报告。所以每个会话结束都要断言"这一段被消费完了"，不满足就是前提没成立，
   不是通过。

2. **端口由操作系统分配。** 固定端口在并发跑两个场景时会抢，抢输的那个起不来，
   报错却指向"服务启动失败"这种和场景毫无关系的地方。

3. **Agent 的子进程环境里主动剔除已知的上游凭据变量名。** ADR 003 写着凭据
   绝不进被测 Agent 的进程环境。`**os.environ` 会把开发机上碰巧 export 过的
   真实密钥一起带进去——这条不是防 Agent，是防我们自己手滑。
"""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import structlog
import uvicorn

from sandbox.clock import clock
from sandbox.config import REPO_ROOT, settings
from sandbox.cost import aggregator as cost
from sandbox.db.engine import init_db
from sandbox.detectors.base import ExemptionSet, Finding
from sandbox.detectors.idempotency import IdempotencyDetector, ToolFacts
from sandbox.detectors.loop import LoopDetector
from sandbox.formatting import money as _money
from sandbox.formatting import span_human
from sandbox.formatting import tokens as _tokens
from sandbox.gateway.stub import scripted_upstream
from sandbox.memory.adapters.file_memory import FileMemoryAdapter
from sandbox.memory.snapshot import SnapshotRef, SnapshotStore
from sandbox.scenario import identity
from sandbox.scenario.expect import ExpectResult, evaluate
from sandbox.scenario.spec import Scenario
from sandbox.trace.reader import load_session_sequence
from sandbox.watchdog import summarize_verdicts

log = structlog.get_logger(__name__)

# 从 Agent 子进程环境里剔除的变量名。见模块 docstring 第 3 点。
_STRIP_FROM_AGENT_ENV = frozenset(
    {
        "SANDBOX_UPSTREAM_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
    }
)


@dataclass
class SessionOutcome:
    """一个会话实际发生了什么。**这些是事实，不是判断。**"""

    name: str
    ordinal: int
    memory_horizon_ordinal: int
    restored_from: str | None
    exit_code: int | None
    elapsed_s: float
    timed_out: bool
    # 本会话声明的脚本步数 vs 上游游标实际前进的步数。
    script_steps_declared: int
    script_steps_consumed: int
    tool_calls: int
    # 这一次会话**自己**把运行触发了熔断（而不是进来时就已经是熔断状态）。
    killed: bool = False
    kill_reason: str = ""
    stdout_tail: str = ""

    # --- 长时运行（功能需求 3.6 / 场景 6）-----------------------------------
    # 一段会话开始时的世界时刻。**不是** `started_at` 那种墙上时间——那个是
    # 记账用的，这个描述的是被模拟的那个世界。
    virtual_started_at: str | None = None
    #: 场景声明的"距上一段过去了多久"。
    after_gap_s: int = 0
    #: 场景声明：本会话开始时世界一共应该走了多少秒。
    declared_offset_s: int = 0
    #: 时钟实际观察值。与上一项不等 = 加速器没按场景说的做。
    observed_offset_s: int = 0

    @property
    def clock_matched(self) -> bool:
        """世界真的按场景声明的那样往前走了吗。

        这是**加速器自己的自检**，不是对被测对象的判断。它挡的是一种特别难查的
        失效：场景写着"隔了一天再看"，而引擎根本没拨时钟——于是第二段会话看到的
        世界和第一段一模一样，检测器照样给出一份看起来正常的报告，读的人以为
        "Agent 隔了一天还是这么干"，实际上什么都没测到。

        非长时运行的场景两边都是 0，天然成立。**上个场景残留的偏移也会在这里
        暴露**（声明 0、观察到非 0），这正是要的。
        """
        return self.observed_offset_s == self.declared_offset_s

    @property
    def premise_held(self) -> bool:
        """这次测试的前提成立吗。

        **不满足意味着"什么都没测到"，而不是"Agent 表现得很好"。** 分不清
        这两者的报告会把空结果读成全绿——这正是本工具最严重的失效方式，
        所以它是一个独立于检测结论的字段。

        三个成立条件，后两个容易被漏掉：

        1. 脚本被消费完了 —— Agent 确实做完了场景写的那些动作。
        2. **本会话自己把运行跑到了熔断。** 被熔断时脚本必然没消费完，但那
           恰恰是循环类场景的预期结果：检测器判出来了、看门狗动手了，说明
           场景想制造的行为确实发生了。按条件 1 判的话，一个完美触发的循环
           场景会被报成"没测到"——判据把成功的信号读成了失败。
        3. **世界的时间真的走了。** 见 `clock_matched`。
        """
        ran_out = self.script_steps_consumed >= self.script_steps_declared or self.killed
        return ran_out and self.clock_matched

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ordinal": self.ordinal,
            "memory_horizon_ordinal": self.memory_horizon_ordinal,
            "restored_from": self.restored_from,
            "exit_code": self.exit_code,
            "elapsed_s": round(self.elapsed_s, 2),
            "timed_out": self.timed_out,
            "script_steps_declared": self.script_steps_declared,
            "script_steps_consumed": self.script_steps_consumed,
            "premise_held": self.premise_held,
            "tool_calls": self.tool_calls,
            "killed": self.killed,
            "kill_reason": self.kill_reason,
            "virtual_started_at": self.virtual_started_at,
            "after_gap_s": self.after_gap_s,
            "declared_offset_s": self.declared_offset_s,
            "observed_offset_s": self.observed_offset_s,
            "clock_matched": self.clock_matched,
        }


@dataclass
class ClockReport:
    """这次运行里的"世界时间"是什么样。

    和 `ScenarioResult` 里那些真实耗时并列给出，**两个数都要有**：只给虚拟时间，
    读的人会以为跑一次要三天；只给真实耗时，就看不出这次模拟了多久。这也是
    "速度倍率"唯一诚实的算法——它是两个已知量的比值，不是我们许下的承诺。
    """

    virtual: bool
    started_at: str
    ended_at: str
    virtual_elapsed_s: int
    real_elapsed_s: float

    @property
    def speed(self) -> float:
        """虚拟时间比真实时间快多少倍。真实耗时按十分之一秒算，避免除零。"""
        if not self.virtual_elapsed_s:
            return 1.0
        return self.virtual_elapsed_s / max(self.real_elapsed_s, 0.1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "virtual": self.virtual,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "virtual_elapsed_s": self.virtual_elapsed_s,
            "real_elapsed_s": round(self.real_elapsed_s, 2),
            "speed": round(self.speed, 1),
        }


@dataclass
class ScenarioResult:
    scenario_id: str
    run_id: str
    sessions: list[SessionOutcome] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    expectation: ExpectResult | None = None
    summary: dict[str, float] = field(default_factory=dict)
    # 目标系统侧真实扣款笔数。负数表示查不到（mock 没起来）。
    egress_charges: int = -1
    expected_charges: int | None = None
    # 成本（需求场景 4）。**不进 `ok` 的判定**——理由和预算一样：花了多少钱是
    # 事实陈述，不是"这次运行的行为有没有缺陷"的结论。CostRollup 为 None 表示
    # 没查（执行失败提前返回），不是"花了 0"。
    cost: cost.CostRollup | None = None
    session_costs: dict[int, cost.CostRollup] = field(default_factory=dict)
    budget: list[cost.BudgetStatus] = field(default_factory=list)
    # 长时运行的世界时间。None = 没跑到那一步，或者这不是一次长时运行。
    clock: ClockReport | None = None
    error: str | None = None

    @property
    def premises_held(self) -> bool:
        return bool(self.sessions) and all(s.premise_held for s in self.sessions)

    @property
    def charges_ok(self) -> bool:
        """目标系统侧真实扣款笔数符合预期吗。

        `egress_charges < 0` 表示查不到（mock 没起来）。**查不到不等于没问题**，
        所以判为不通过——把"没测到"读成"没问题"是这个工具最严重的失效方式。
        """
        if self.expected_charges is None:
            return True
        return self.egress_charges == self.expected_charges

    @property
    def ok(self) -> bool:
        """整体是否通过。

        三个条件，任何一个不成立都不通过：

        1. 没有执行失败。
        2. **前提成立**——检测器是在一段符合场景描述的轨迹上得出结论的。
        3. **地面真值对得上**——钱到底扣了几次。检测器可能报出漂亮的 pass
           而钱其实扣了两次；这两件事必须分开验。
        """
        if self.error is not None or not self.premises_held or not self.charges_ok:
            return False
        return self.expectation is not None and self.expectation.ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "run_id": self.run_id,
            "ok": self.ok,
            "premises_held": self.premises_held,
            "charges_ok": self.charges_ok,
            "error": self.error,
            "sessions": [s.to_dict() for s in self.sessions],
            "summary": self.summary,
            "egress_charges": self.egress_charges,
            "expected_charges": self.expected_charges,
            "clock": self.clock.to_dict() if self.clock else None,
            "cost": self.cost.to_dict() if self.cost else None,
            "session_costs": {str(k): v.to_dict() for k, v in sorted(self.session_costs.items())},
            "budget": [
                {
                    "period": b.period,
                    "spent_usd": round(b.spent_usd, 6),
                    "ceiling_usd": b.ceiling_usd,
                    "exceeded": b.exceeded,
                }
                for b in self.budget
            ],
            "findings": [
                {
                    "detector": f.detector,
                    "verdict": f.verdict.value,
                    "summary": f.summary,
                    "detail": f.detail,
                }
                for f in self.findings
            ],
            "expectation": self.expectation.to_dict() if self.expectation else None,
        }


async def _serve(app: Any, sock: socket.socket) -> tuple[uvicorn.Server, asyncio.Task[None]]:
    """在一个已经 bind+listen 的套接字上起一个 uvicorn。

    套接字由调用方准备好，端口就由内核分配——见模块 docstring 第 2 点。

    返回 server 和它的 task 两个东西：关服务时只置 `should_exit` 是不够的，
    必须等 task 真的跑完再关套接字。不然 lifespan 还挂在 receive 上，套接字
    一关就抛一串 CancelledError——看着像崩溃，其实什么都没坏，白让人查一轮。
    """
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        if task.done():
            task.result()  # 把真实异常抛出来，而不是报一句"启动失败"
        await asyncio.sleep(0.02)
    return server, task


def _bind(port: int = 0) -> tuple[socket.socket, int]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen(128)
    return sock, sock.getsockname()[1]


class ScenarioRunner:
    """跑场景。一个实例可以被复用跑多个场景（每个场景内部自建服务）。"""

    def __init__(
        self,
        *,
        repo_root: Path = REPO_ROOT,
        session_timeout_s: float = 180.0,
        keep_servers: bool = False,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.session_timeout_s = session_timeout_s
        self.keep_servers = keep_servers

    # --- 对外入口 ---------------------------------------------------------

    async def run(self, scenario: Scenario) -> ScenarioResult:
        result = ScenarioResult(scenario_id=scenario.id, run_id="")
        try:
            await self._run(scenario, result)
        except Exception as exc:  # noqa: BLE001 — 引擎的异常要进报告，不是崩掉 CLI
            log.exception("scenario.run_failed", scenario=scenario.id)
            result.error = f"{type(exc).__name__}: {exc}"
            await self._record_abort(result)
        return result

    async def _record_abort(self, result: ScenarioResult) -> None:
        """执行中途炸了：把运行标成失败，并留下结论。

        **不做的话那条 `TestRun` 会永远停在 `running`。** 收尾走的是网关的
        `/finish` 端点，而异常路径根本到不了那里。CLI 上看不出问题（它直接把
        `result` 打进终端，那个对象是对的），但界面要列**历史**运行，于是一次
        崩掉的运行会一直显示"进行中"——用户以为它还在跑，等一个永远不会来的结果。

        结论照常写，`ok=False` 是**真的**：`ok` 的定义里第一条就是"没有执行
        失败"。这和"没记录"是两回事——后者是加这个字段之前的历史运行，界面
        必须显示"结论未留存"，而不是替它编一个红叉。

        整个动作包在 try 里：收尾失败不该盖掉真正的异常，那是唯一能说明
        出了什么事的线索。
        """
        if not result.run_id:
            return  # 连运行都还没建起来，没有可标记的对象
        try:
            from sandbox.trace.recorder import mark_run_failed, record_verdict

            await record_verdict(result.run_id, ok=False, report=result.to_dict())
            await mark_run_failed(result.run_id)
        except Exception:  # noqa: BLE001
            log.exception("scenario.record_abort_failed", run=result.run_id)

    # --- 主体 -------------------------------------------------------------

    async def _run(self, scenario: Scenario, result: ScenarioResult) -> None:
        # 延迟导入：把 app 模块的导入推迟到真的要起服务时，这样只做加载/校验的
        # 路径（CLI 的 --list、加载测试）不必拉起整个 web 栈。
        from sandbox.gateway import app as gateway_app
        from sandbox.mock import server as mock_server
        from sandbox.proxy import app as proxy_app
        from sandbox.runtime import runs

        await init_db()
        state = mock_server.get_state()
        # 长时运行才进虚拟模式。不进的话 `clock.now()` 就是真实时间，整条既有
        # 路径一行都不用改——这是刻意的：没在模拟时间的运行，报告里就不该出现
        # 任何"虚拟"字样。
        if scenario.is_long_run():
            clock.start()
        # **锚点必须在时钟就位之后再钉，而且要和会话读的是同一个对象。**
        # 订单的到期时间是相对"世界起点"算出来的，起点来自这次调用；换成读
        # `datetime.now()` 也能跑，但那意味着世界起点和世界当前时刻出自两个
        # 口径——今天恰好相等，改天 `start()` 换了语义就不等了。
        state.reset()
        world_started = clock.now()
        real_started = clock.real_now()
        # mock 目标系统侧是否真的按幂等键去重。
        #
        # 场景是按**工具**声明这个属性的，而 mock 只有一个全局开关——因为
        # mock 里目前只有一个写工具，两者等价。加第二个写工具时这里必须改成
        # 按工具分别控制，否则"工具 A 幂等、工具 B 不幂等"这种场景根本表达不出来。
        state.enforce_idempotency = any(
            t.enforces_idempotency for t in scenario.tools if t.is_write
        )
        proxy_app.set_policy(None)
        proxy_app.set_transport(None)
        # clear() 会连带外处理器一起清掉（见 runtime.RunRegistry.clear 的说明），
        # 所以处理器必须在 clear 之后注册。
        runs.clear()

        gw_sock, gw_port = _bind()
        px_sock, px_port = _bind()
        mk_sock, mk_port = _bind()
        servers = [
            await _serve(gateway_app.app, gw_sock),
            await _serve(proxy_app.app, px_sock),
            await _serve(mock_server.app, mk_sock),
        ]
        tasks = [task for _, task in servers]

        # 代理转发到"目标系统"的地址来自配置，而 mock 现在绑在随机端口上。
        # 不覆盖的话，代理会去敲配置里那个端口——可能没人监听（连接被拒），
        # 更糟的是**可能有另一个场景的 mock 在听**，于是两个场景互相往对方的
        # 目标系统里写数据，而两边报告都正常。所以必须把实际端口告诉它。
        #
        # 跑完恢复原值：这是进程级配置，泄漏到下一个场景会让它指向一个已经
        # 关掉的端口。
        original_upstream = settings.tool_upstream_base_url
        settings.tool_upstream_base_url = f"http://127.0.0.1:{mk_port}"

        try:
            await self._run_sessions(scenario, result, gw_port, px_port, mk_port, runs)
        finally:
            settings.tool_upstream_base_url = original_upstream
            # 世界时间也在这里收尾。**必须恢复**：它是进程级状态，泄漏给下一个
            # 场景的话，下一次运行会带着上一次的时间偏移开局，而它的报告里每一
            # 处都显示"世界已经是三天后了"——全都正常，全都错。
            if scenario.is_long_run():
                result.clock = ClockReport(
                    virtual=True,
                    started_at=world_started.isoformat(),
                    ended_at=clock.now().isoformat(),
                    virtual_elapsed_s=clock.offset_s(),
                    real_elapsed_s=(clock.real_now() - real_started).total_seconds(),
                )
                clock.reset()
            if not self.keep_servers:
                for server, _ in servers:
                    server.should_exit = True
                # 等三个服务真的收完再关套接字，顺序反了会刷一屏 CancelledError。
                await asyncio.gather(*tasks, return_exceptions=True)
                for sock in (gw_sock, px_sock, mk_sock):
                    sock.close()

    async def _run_sessions(
        self,
        scenario: Scenario,
        result: ScenarioResult,
        gw_port: int,
        px_port: int,
        mk_port: int,
        runs: Any,
    ) -> None:
        # 不导入 finish_run：收尾走 HTTP 端点，那条路径顺带验证了端点本身能用。
        from sandbox.trace.recorder import (
            capture_snapshot,
            horizon_of,
            record_verdict,
            restore_snapshot,
            start_session,
        )

        # 记忆目录由**引擎**开出来，再通过环境变量交给 Agent。让 Agent 自己
        # 决定记忆放哪儿的话，测试框架的回滚就可能回滚到一处它根本不读的地方——
        # 而那时报告会显示"回滚过了，还是重复扣款"，把 Agent 冤枉成不读记忆。
        #
        # **每次运行前清空。** 不清的话第二次跑同一个场景时，记忆里还留着上一次
        # 的记录，Agent 一上来就认为"做过了"——同一个场景跑两次结果不同，这对
        # 一个测试工具是很糟的性质，而且症状会伪装成"Agent 行为不稳定"。
        # 目录名带 scenario.id 只解决场景之间的隔离，解决不了场景自身跨运行的残留。
        memory_root = settings.memory_root / scenario.id
        if memory_root.exists():
            shutil.rmtree(memory_root)
        memory_root.mkdir(parents=True, exist_ok=True)
        adapter = FileMemoryAdapter(root=memory_root)
        store = SnapshotStore()
        # 名字 → 快照引用。场景用名字引用快照，引擎记住这个名字对应哪一次。
        named: dict[str, SnapshotRef] = {}

        # 上游脚本按会话顺序拼接。单游标顺序读，所以每个会话的步骤必须
        # 紧跟在它前面那个会话的步骤之后——见下面消费量检查。
        full_script: list[dict[str, Any]] = []
        for session in scenario.sessions:
            full_script.extend(session.script)

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"http://127.0.0.1:{gw_port}/_sandbox/runs",
                json={
                    "scenario_id": scenario.id,
                    "agent_version": scenario.agent.version,
                    # 分节的配置指纹 + 环境事实。**别退回成
                    # `{"scenario": id, "model": model}`**：那个口径下改了脚本
                    # 或 Prompt，哈希都不变，于是"配置没变但结果变了"这个最该
                    # 被抓到的信号，恰恰是唯一抓不到的。见 scenario/identity.py。
                    "config": identity.sections(scenario),
                    "env": identity.environment(scenario),
                    "declared_writes": sorted(scenario.declared_writes()),
                    "idempotency_key_params": scenario.idempotency_params(),
                    "tools": {t.name: f"/tools/{t.name}" for t in scenario.tools},
                },
            )
            resp.raise_for_status()
            run = resp.json()
            run_id = run["run_id"]
            result.run_id = run_id

            scripted_upstream.register(run_id, full_script)

            # 带外熔断处理器。Agent 进程每个会话换一个，所以用可变容器持有
            # "当前那个"，而不是在闭包里捕获第一个。
            current: dict[str, subprocess.Popen[str] | None] = {"proc": None}

            def kill_handler(_rid: str, reason: str) -> None:
                proc = current["proc"]
                if proc is not None and proc.poll() is None:
                    log.warning("scenario.kill_agent_process", reason=reason)
                    proc.terminate()

            runs.on_kill(kill_handler)

            active = runs.get(run_id)
            assert active is not None  # 刚注册的，不可能不在

            for index, spec in enumerate(scenario.sessions):
                ordinal = index  # 会话 0 由 start_run 建出，从会话 1 起要显式开

                # **把世界往前拨，然后什么也不等。** 这就是时间加速的全部：一段
                # 声明的间隔立刻过去，目标系统上该变的都变了，真实耗时是零。
                # 被测 Agent 这段时间本来就不在跑——云端常驻 Agent 也是被定时器
                # 叫醒的，这一点没有被跳过，是如实的。
                if spec.after_gap_s:
                    clock.advance(spec.after_gap_s)
                observed_offset = clock.offset_s()
                declared_offset = scenario.offset_before(index)
                # 会话**开始时**的世界时刻，不是结束时的——结束时要再取一次，
                # 取错的话报告里每一段会话都会显示成"和下一段同一时刻"。
                # 不在虚拟模式里就是 None：没在模拟时间的运行，不该出现任何
                # 一个叫"virtual"的字段带着一个真实时间。
                world_at_start = clock.now().isoformat() if clock.is_virtual else None

                restored: SnapshotRef | None = None
                if spec.restore_from is not None:
                    restored = named[spec.restore_from]
                    # allow_create=True：这个目录是引擎自己开的，见上面。
                    await restore_snapshot(restored, adapter, store, allow_create=True)

                if index > 0:
                    await start_session(active.ctx, restored_from=restored)

                if spec.snapshot_before:
                    named[spec.snapshot_before] = await capture_snapshot(
                        active.ctx, adapter, store
                    )

                cursor_before = scripted_upstream.calls_made(run_id)
                # 熔断状态是运行级的、会一直留着，所以要分"进来时就已经熔断"
                # 和"这一次会话把它跑熔断了"。前者说明这个会话根本没机会跑，
                # 后者才是"循环场景如期触发"。
                killed_before = active.killed
                exit_code, timed_out, stdout, elapsed = await self._launch_agent(
                    scenario,
                    run["api_key"],
                    gw_port,
                    px_port,
                    memory_root,
                    current,
                    # 这一段用户说的是什么。没写就沿用场景级的 task。
                    spec.task or scenario.agent.task,
                )
                cursor_after = scripted_upstream.calls_made(run_id)

                if spec.snapshot_after:
                    named[spec.snapshot_after] = await capture_snapshot(
                        active.ctx, adapter, store
                    )

                result.sessions.append(
                    SessionOutcome(
                        name=spec.name,
                        ordinal=ordinal,
                        memory_horizon_ordinal=active.ctx.memory_horizon_ordinal,
                        restored_from=spec.restore_from,
                        exit_code=exit_code,
                        elapsed_s=elapsed,
                        timed_out=timed_out,
                        script_steps_declared=len(spec.script),
                        script_steps_consumed=cursor_after - cursor_before,
                        tool_calls=0,  # 稍后按会话回填
                        killed=active.killed and not killed_before,
                        kill_reason=active.kill_reason,
                        stdout_tail=stdout[-2000:],
                        virtual_started_at=world_at_start,
                        after_gap_s=spec.after_gap_s,
                        declared_offset_s=declared_offset,
                        observed_offset_s=observed_offset,
                    )
                )

            # 收尾：注销路由（Agent 已经全部退出，不会再产生调用了）。
            await client.post(f"http://127.0.0.1:{gw_port}/_sandbox/runs/{run_id}/finish")

        # 检测跑在**会话序列**上，而不是单个运行上——跨会话归因需要
        # (ordinal, seq) 这个顺序，而它是会话序列的属性。
        calls = await load_session_sequence(active.ctx.session_sequence_id)
        per_ordinal: dict[int, int] = {}
        for call in calls:
            per_ordinal[call.session_ordinal] = per_ordinal.get(call.session_ordinal, 0) + 1
        for outcome in result.sessions:
            outcome.tool_calls = per_ordinal.get(outcome.ordinal, 0)

        result.findings = _run_detectors(scenario, calls)
        result.expectation = evaluate(scenario, result.findings)
        result.summary = summarize_verdicts(result.findings)
        result.egress_charges = await self._charges(mk_port)
        result.expected_charges = scenario.expected_charges

        # 成本。放在 finish 之后——LlmCall 明细是网关结算时才落的，提前查会
        # 得到一份少了几次调用的账单，而它看起来同样是个正常的数字。
        result.cost = await cost.for_run(run_id)
        result.session_costs = await cost.by_session(run_id)
        result.budget = await cost.budget_status()

        # 判定与完整报告落库。放在最后——到这一步 `result` 才完整，之前任何
        # 一个字段都还在变。CLI 直接把 result 打进终端，但 Web 界面要列出
        # **历史**运行，那时内存里的 result 早没了，只能靠这条记录。
        await record_verdict(run_id, ok=result.ok, report=result.to_dict())

        # 快照位置随报告一起给出：跨会话结论的归因完全建立在"回滚到了哪儿"上，
        # 不给出来就没法复核。
        refs = await store.list_for_sequence(active.ctx.session_sequence_id)
        log.info(
            "scenario.snapshots",
            scenario=scenario.id,
            positions=[
                {
                    "name": _name_of(ref, named),
                    "position": ref.position,
                    "horizon": horizon_of(ref),
                    "entries": ref.entry_count,
                }
                for ref in refs
            ],
        )

    async def _launch_agent(
        self,
        scenario: Scenario,
        token: str,
        gw_port: int,
        px_port: int,
        memory_root: Path,
        current: dict[str, subprocess.Popen[str] | None],
        task: str,
    ) -> tuple[int | None, bool, str, float]:
        """拉起被测 Agent 进程，等它退出。返回 (退出码, 是否超时, 输出, 耗时)。

        `task` 是**这一段**会话里用户说的话（见 `SessionSpec.task`）。
        """
        agent_dir = self.repo_root / "agents" / scenario.agent.directory
        python = agent_dir / ".venv" / "Scripts" / "python.exe"
        if not python.exists():
            # Windows 的 venv 布局。非 Windows 上退回 posix 布局，让这个
            # 运行器至少能在别处跑起来，而不是只在开发机上能跑。
            alt = agent_dir / ".venv" / "bin" / "python"
            python = alt if alt.exists() else python
        if not python.exists():
            raise FileNotFoundError(
                f"Agent 解释器不存在：{python}。请先在 {agent_dir} 下建好 venv 并装依赖"
                f"（见 agents/{scenario.agent.directory}/README.md）。"
            )

        env = {
            # 场景自带的环境变量放在前面，让沙箱注入的**覆盖**它。
            # 反过来的话，场景可以把 Agent 指向真实上游，那观测就断了。
            **scenario.agent.env,
            **{k: v for k, v in os.environ.items() if k not in _STRIP_FROM_AGENT_ENV},
            "SANDBOX_GATEWAY_BASE_URL": f"http://127.0.0.1:{gw_port}/v1",
            "SANDBOX_AGENT_API_KEY": token,
            "SANDBOX_TOOL_BASE_URL": f"http://127.0.0.1:{px_port}",
            "SANDBOX_AGENT_MODEL": scenario.agent.model,
            "SANDBOX_AGENT_TASK": task,
            # 记忆目录由引擎创建并告知 Agent。Agent 不该自己猜路径，
            # 否则测试框架的回滚就回滚到了一处它不读的地方。
            "SANDBOX_AGENT_MEMORY_DIR": str(memory_root),
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        }

        started = time.perf_counter()
        proc = subprocess.Popen(
            [str(python), scenario.agent.entry],
            cwd=str(agent_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        current["proc"] = proc

        timed_out = False
        try:
            stdout, _ = await asyncio.to_thread(proc.communicate, self.session_timeout_s)
        except (TimeoutError, subprocess.TimeoutExpired):
            # 超时**不能**当成"Agent 结束了"。它还在跑、还在烧钱，只是我们
            # 不等了；结论必须标出来，否则报告会把它读成一次正常结束。
            timed_out = True
            proc.kill()
            await asyncio.to_thread(proc.communicate)
            stdout = "(Agent 超时未退出，已强杀)"
        finally:
            current["proc"] = None

        return proc.returncode, timed_out, stdout or "", time.perf_counter() - started

    async def _charges(self, mock_port: int) -> int:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"http://127.0.0.1:{mock_port}/_mock/charges")
                return int(resp.json().get("count", 0))
        except Exception:  # noqa: BLE001 — mock 没起来不该让整份报告丢掉
            return -1


def _name_of(ref: SnapshotRef, named: dict[str, SnapshotRef]) -> str | None:
    return next((k for k, v in named.items() if v.id == ref.id), None)


def _run_detectors(scenario: Scenario, calls: list[Any]) -> list[Finding]:
    """跑全套检测器。工具事实来自场景声明，是写/读属性的单一来源。

    顺序固定：输出顺序稳定，比对预期时不会因为字典序漂移而时红时绿。
    """
    facts = {
        t.name: ToolFacts(t.name, t.is_write, t.enforces_idempotency) for t in scenario.tools
    }
    findings: list[Finding] = []
    findings.extend(IdempotencyDetector(facts).run(calls, ExemptionSet()))
    findings.extend(LoopDetector().run(calls, ExemptionSet()))
    return findings


def render(result: ScenarioResult, scenario: Scenario | None = None) -> str:
    """人类可读报告。"""
    lines: list[str] = []
    add = lines.append

    add(f"=== 场景 {result.scenario_id} ===")
    if scenario and scenario.description:
        add(f"    {scenario.description}")
    add(f"    运行 {result.run_id or '(未创建)'}")

    if result.error:
        add(f"\n!! 执行失败：{result.error}")
        return "\n".join(lines)

    add("\n--- 会话 ---")
    for s in result.sessions:
        mark = "ok" if s.premise_held else "!!"
        restored = f" ← 恢复到 {s.restored_from}" if s.restored_from else ""
        timeout = " [超时强杀]" if s.timed_out else ""
        killed = " [已被熔断]" if s.killed else ""
        add(
            f"  {mark} #{s.ordinal} {s.name}{restored}{timeout}{killed}"
            f"  退出码 {s.exit_code}  {s.elapsed_s:.1f}s"
        )
        add(
            f"       记忆视野 ≤ 会话 {s.memory_horizon_ordinal}"
            f" / 工具调用 {s.tool_calls} 次"
        )
        if s.virtual_started_at:
            gap = f"，距上一段过了 {span_human(s.after_gap_s)}" if s.after_gap_s else ""
            add(f"       世界时间 {s.virtual_started_at}{gap}")
            if not s.clock_matched:
                add(
                    f"       ** 时钟对不上：场景声明世界走了 {span_human(s.declared_offset_s)}，"
                    f"实际走了 {span_human(s.observed_offset_s)} **"
                )
        consumed = f"       脚本消费 {s.script_steps_consumed}/{s.script_steps_declared}"
        if s.killed:
            consumed += f"（熔断所致，前提成立）\n       熔断原因：{s.kill_reason}"
        elif not s.premise_held:
            consumed += "  ** 这一段没跑完，前提未成立 **"
        add(consumed)

    if result.clock is not None:
        # 两个时间并列给。只给虚拟时间，读的人会以为跑一次要三天；只给真实
        # 耗时，就看不出这次到底模拟了多久。
        c = result.clock
        add("\n--- 世界时间（长时运行，需求场景 6）---")
        add(f"  世界从 {c.started_at} 走到 {c.ended_at}")
        add(
            f"  模拟跨度 {span_human(c.virtual_elapsed_s)}，"
            f"真实耗时 {c.real_elapsed_s:.1f}s（约 {c.speed:.0f} 倍）"
        )
        add(
            "  **这段时间不是等出来的**：引擎把目标系统的时钟往前拨，世界该变的都\n"
            "     变了，而被测 Agent 在两次会话之间本来就不在跑。"
        )

    add("\n--- 检测结论 ---")
    if not result.findings:
        add("  （无）")
    for f in result.findings:
        flag = {"pass": "  ", "fail": "!!", "warning": " ?", "inconclusive": ".."}[
            f.verdict.value
        ]
        add(f"  {flag} [{f.verdict.value:<12}] {f.detector}")
        add(f"       {f.summary}")

    add("\n--- 汇总（四态，inconclusive 不并入 pass）---")
    s = result.summary
    add(
        f"  通过 {s.get('pass', 0):.0f} / 失败 {s.get('fail', 0):.0f} / "
        f"警告 {s.get('warning', 0):.0f} / **未覆盖 {s.get('inconclusive', 0):.0f}**"
    )
    add(
        f"  覆盖率 {s.get('coverage_ratio', 0):.0%}"
        f"（已覆盖部分的失败率 {s.get('failure_rate_among_covered', 0):.0%}）"
    )
    # 地面真值单独列出来，且和检测结论分开——检测器说 pass 而钱扣了两次，
    # 正是这个工具要抓的失效方式，两者放在同一行会看不出区别。
    if result.expected_charges is not None:
        mark = "ok" if result.charges_ok else "!!"
        got = result.egress_charges if result.egress_charges >= 0 else "查不到"
        add(f"  {mark} 目标系统侧真实扣款 {got} 笔（预期 {result.expected_charges} 笔）")
    elif result.egress_charges >= 0:
        add(f"  目标系统侧真实扣款 {result.egress_charges} 笔")

    if result.cost is not None:
        add("\n--- 成本（陈述，不参与通过与否）---")
        c = result.cost
        add(
            f"  {c.llm_calls} 次 LLM 调用，"
            f"{_tokens(c.input_tokens)} in / {_tokens(c.output_tokens)} out，"
            f"${_money(c.cost_usd)}"
        )
        if c.is_lower_bound:
            # 金额不全的时候必须说出来。只印 `$0.0000` 的话，读报告的人会以为
            # 这次运行不要钱——而实际上是我们没有这个模型的单价，一个字都没算。
            add(
                f"  **注意：其中 {c.unpriced_calls} 次调用的模型不在定价表中，"
                f"成本按 0 计。上面的金额是下界，不是账单。**"
            )
        if c.cache_read_tokens or c.cache_write_tokens:
            add(
                f"  缓存读 {_tokens(c.cache_read_tokens)}"
                f" / 写 {_tokens(c.cache_write_tokens)}"
            )
        # 只有多会话或者有"没归到会话"的调用时才逐会话列——单会话时上面那行
        # 就是它自己的数，再列一遍是重复。
        if len(result.session_costs) > 1 or -1 in result.session_costs:
            for ordinal, rollup in sorted(result.session_costs.items()):
                who = "（未归到会话）" if ordinal < 0 else f"会话 #{ordinal}"
                add(
                    f"    {who}  {rollup.llm_calls} 次调用  "
                    f"${_money(rollup.cost_usd)}"
                )

    if result.budget:
        add("\n--- 预算 ---")
        for b in result.budget:
            period = {"day": "本日", "week": "本周", "month": "本月"}[b.period]
            mark = "!!" if b.exceeded else "ok"
            add(
                f"  {mark} {period}已花 ${_money(b.spent_usd)}"
                f" / 预算 ${_money(b.ceiling_usd)}（{b.ratio:.0%}）"
            )
        if any(b.exceeded for b in result.budget):
            add(
                "     超预算**不会**让本次运行算作不通过——那是累计花费，和被测\n"
                "     Agent 这一次做没做错事无关。两者压成同一个结论的话，一次\n"
                "     完全干净的运行会因为上个月跑太多而报 fail。"
            )

    if result.expectation is not None:
        e = result.expectation
        add("\n--- 预期 ---")
        for expectation, finding in e.matched:
            add(f"  命中  {expectation.detector}/{expectation.verdict.value}：{finding.summary}")
        for expectation in e.missing:
            code = f"/{expectation.reason_code}" if expectation.reason_code else ""
            add(
                f"  未命中 {expectation.detector}/{expectation.verdict.value}{code}"
                f"{'：' + expectation.note if expectation.note else ''}"
            )
        for finding in e.unexpected:
            add(f"  意外  [{finding.verdict.value}] {finding.detector}：{finding.summary}")
        for finding in e.uncovered:
            add(f"  未覆盖 [{finding.verdict.value}] {finding.detector}：{finding.summary}")

    if not result.premises_held:
        add(
            "\n  ** 有会话没把脚本跑完，检测结论不指向被测对象，本次不算通过。**\n"
            "     常见原因：Agent 提前退出（工具调用报错、导入失败、环境变量没配好）。"
        )

    add(f"\n  结论：{'通过' if result.ok else '不通过'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """`scripts/run_scenario.py` 的实现，放这里是为了让脚本只做参数转发。"""
    import argparse
    import json

    from sandbox.scenario.loader import ScenarioError, load_all, load_scenario

    parser = argparse.ArgumentParser(description="跑一个场景（数据驱动）")
    parser.add_argument("scenarios", nargs="*", help="场景文件或目录")
    parser.add_argument("--dir", default=str(REPO_ROOT / "scenarios"), help="场景目录")
    parser.add_argument("--list", action="store_true", help="只列出场景，不跑")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告")
    parser.add_argument("--timeout", type=float, default=180.0, help="单会话超时（秒）")
    parser.add_argument("--keep-servers", action="store_true", help="跑完不关服务")
    args = parser.parse_args(argv)

    targets = args.scenarios or [args.dir]
    try:
        scenarios = []
        for target in targets:
            path = Path(target)
            scenarios.extend(load_all(path) if path.is_dir() else [load_scenario(path)])
    except ScenarioError as exc:
        print(f"场景不合法：\n{exc}", file=sys.stderr)
        return 2

    if args.list:
        for scenario in scenarios:
            print(f"{scenario.id:<24} {scenario.description}")
            print(
                f"    会话 {len(scenario.sessions)} 个 / 工具 {len(scenario.tools)} 个 "
                f"/ 预期 {len(scenario.expect)} 条"
            )
        return 0

    async def _all() -> int:
        runner = ScenarioRunner(
            session_timeout_s=args.timeout, keep_servers=args.keep_servers
        )
        failed = 0
        for scenario in scenarios:
            result = await runner.run(scenario)
            if args.json:
                print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            else:
                print(render(result, scenario))
                print()
            failed += 0 if result.ok else 1
        if len(scenarios) > 1:
            print(f"===== {len(scenarios) - failed}/{len(scenarios)} 个场景通过 =====")
        return 1 if failed else 0

    return asyncio.run(_all())
