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
from collections.abc import Sequence
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
from sandbox.detectors import run_all
from sandbox.detectors.base import Finding
from sandbox.detectors.idempotency import ToolFacts
from sandbox.formatting import money as _money
from sandbox.formatting import span_human
from sandbox.formatting import tokens as _tokens
from sandbox.gateway.stub import scripted_upstream
from sandbox.memory.adapters.file_memory import FileMemoryAdapter
from sandbox.memory.observe import (
    PHASE_END,
    PHASE_START,
    MemoryObservation,
    observe_memory,
    phase_label,
    summarize_observations,
)
from sandbox.memory.snapshot import SnapshotRef, SnapshotStore
from sandbox.scenario import call_chain, identity, session_diff
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

    # --- 主动行为审计（功能需求 3.6 / 场景 6）-------------------------------
    #: 这一段会话是无人值守的吗。
    #:
    #: 报告里要显示它。同一个 `create_payment`，旁边写着"无人值守"和没写，读出来
    #: 的行为完全不同——前者是它自己决定的，后者是用户要求它做的。少了这个标记，
    #: 一份长时运行的报告会把"夜里它自己动的手"读成"用户让它做的事"。
    autonomous: bool = False

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

    @property
    def ended_normally(self) -> bool:
        """这一段是不是正常结束的。**不是判定**，只决定要不要把 Agent 的输出印出来。

        判据是"有没有出岔子"：被强杀、被熔断、退出码非 0、脚本没跑完，四种都算。
        正常结束的会话照样有几千字输出，全印出来会把报告淹掉——而真正需要看输出的
        恰恰是出了岔子的那几次：崩了要看 traceback，卡住了要看它最后停在哪一行。
        """
        return (
            not self.timed_out
            and not self.killed
            and self.exit_code == 0
            and self.premise_held
        )

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
            "autonomous": self.autonomous,
            # 输出**照原样入库**，不因为"这一段很正常"就清空。清空是一种静默省略：
            # 读 JSON 的人分不出"它什么都没打印"和"我们没放进来"。要筛的是**展示**
            # （命令行与界面只在出岔子的那几段印它），不是记录。
            "stdout_tail": self.stdout_tail,
            "ended_normally": self.ended_normally,
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
class AutonomyReport:
    """这次运行里"谁被允许自己动手"是什么样。

    **它是报告的一部分，不是渲染时现查场景。** 场景文件事后会被编辑（这正是
    `config_parts_json` 存原值的理由），渲染历史报告时去读今天的场景，会让一份
    上周的报告显示成今天的授权——"未经授权"那半句话于是悄悄换了对象，而报告
    读起来完全正常。

    `None` 表示这次运行没有无人值守的会话，整块不该出现。和 `ClockReport` 一个
    道理：普通运行里挂一个空的自主行为面板，读的人分不出"这次没做自主测试"和
    "这次做了但没测到东西"。
    """

    #: 场景声明为无人值守的会话序号。
    autonomous_ordinals: list[int]
    #: 被授权在无人值守时运行的工具。**没有它，"未经授权"这句话就没法复核**。
    unattended_allowlist: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "autonomous_ordinals": self.autonomous_ordinals,
            "unattended_allowlist": self.unattended_allowlist,
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
    # 主动行为审计的授权对照。None = 这次运行没有无人值守的会话。
    autonomy: AutonomyReport | None = None
    # 会话边界上的记忆观测摘要（见 memory/observe.py）。**只有摘要，没有内容**：
    # 报告是长期保留的，而记忆里可能有用户真实数据。空列表 = 一个会话都没跑到。
    #
    # 它是观测不是判定：判定在 `memory.*` 那几条结论里，而这里列的是"看见了什么"
    # ——包括场景没有声明判据时那些不会被判的改动。两者要分得开。
    memory: list[dict[str, Any]] = field(default_factory=list)
    # 跨会话差异（需求 3.1 P1，见 scenario/session_diff.py）。None = 这个场景
    # 只有一段会话，没有可对照的对象。**同样是陈述不是判定**：它不进 findings、
    # 不影响 ok——同一个场景的两段会话本来就允许不一样。
    session_diff: session_diff.SessionDiff | None = None
    # 调用链（需求 3.2 P1，见 scenario/call_chain.py）。None = 这次运行一段会话
    # 都没有。同样是**陈述不是判定**：目标系统回了个错不等于被测对象做错了事，
    # 所以它也不进 findings、不影响 ok。
    call_chain: call_chain.CallChain | None = None
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
            "autonomy": self.autonomy.to_dict() if self.autonomy else None,
            "memory": self.memory,
            "session_diff": self.session_diff.to_dict() if self.session_diff else None,
            "call_chain": self.call_chain.to_dict() if self.call_chain else None,
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


# 每段会话最多留多少 Agent 输出。**这个数只有一个来源**：印出来的那行抬头和
# 真正截断的地方读的是同一个常量，各写各的迟早会说成两个数。
_STDOUT_TAIL_CHARS = 2000

# 超时之后盖在输出上的那句话。写在**末尾**：会话把输出截成最后 2000 字符
# （`SessionOutcome.stdout_tail`），盖在前面的话，输出一长就被截掉了——而那句话
# 正是"这一次不是正常结束"的唯一说明。
TIMEOUT_MARK = "(Agent 超时未退出，已强杀)"


async def _wait_for_exit(proc: subprocess.Popen[str], timeout_s: float) -> tuple[bool, str]:
    """等 Agent 退出，超时就不再等。返回 (是否超时, 它吐出来的输出)。

    **超时参数必须写成关键字 `timeout=`，这是这个函数存在的理由。**
    `Popen.communicate` 的签名是 `communicate(input=None, timeout=None)`——第一个
    位置参数是 `input`，不是 `timeout`。而这里的 `stdin` 根本不是管道，于是传进去的
    那个数被**静默丢掉**，`communicate` 退化成一句没有超时的 `wait()`：Agent 挂住多久，
    框架就等多久。它不报错、不留痕，只是那个 `--timeout` 从来没有生效过。

    （实测过：子进程睡 5 秒，`communicate(1.0)` 等满 5.09 秒正常返回；
    `communicate(timeout=1.0)` 在 1.00 秒抛 `TimeoutExpired`。）

    超时**不能**当成"Agent 结束了"。它还在跑、还在烧钱，只是我们不等了；结论必须
    标出来，否则报告会把它读成一次正常结束。它是独立于脚本消费量的第二道依据——
    消费量数的是**网关那边**被取走了几步（`scripted_upstream.calls_made`），
    一个卡在推理里还没发出请求的 Agent 在那条线上看起来和"刚开始跑"一样。

    已经吐出来的输出**留着**，盖在那句话前面：一个挂住的 Agent 最后打印的那一行
    往往就是唯一的线索，丢掉它等于把"它在干什么"换成一句"我们不等了"。

    输出要从**第二次** `communicate` 的返回值里拿，不能指望异常上那份：
    `TimeoutExpired.output` 在这条路上是 `None`（实测过）。标准库的承诺是
    "抓住超时之后再 communicate 一次不会丢输出"，所以那次调用的返回值才是它。
    **别把那个返回值写丢了**——写丢了就只剩一句"我们不等了"。

    做不到的事也要说清：输出走的是块缓冲的管道，Agent 没 flush 就被杀掉的话，
    它最后打印的东西是真的没了。引擎**不会**替它加 `-u`——那是在改被测进程里的
    行为，而 ADR 001 只允许换 base_url 和 api_key。
    """
    try:
        stdout, _ = await asyncio.to_thread(proc.communicate, timeout=timeout_s)
        return False, stdout or ""
    except (TimeoutError, subprocess.TimeoutExpired):
        proc.kill()
        drained, _ = await asyncio.to_thread(proc.communicate)
        return True, f"{drained or ''}\n{TIMEOUT_MARK}"


def _bind(port: int = 0) -> tuple[socket.socket, int]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen(128)
    return sock, sock.getsockname()[1]


def registration_payload(scenario: Scenario) -> dict[str, Any]:
    """开一个运行时要交给网关的那份声明。

    **单独成函数，因为它是一份要过线的契约。** 网关按**键名**读这张表，读不到
    就当没声明——不报错（也报不了：手工 curl 起运行的那条路本来就允许只给一半）。
    于是场景侧把键名改一个字母，症状是"声明写了、运行当中却没生效"，而这条链上
    没有任何一处会红。把这份载荷提出来，测试就能拿它去过一遍真的网关
    （见 `tests/test_loop_paging.py` 与 `tests/test_watchdog.py`）。

    每条声明的理由写在原处不搬——它们说的是"为什么这个东西必须过线"，
    而不是"这个函数怎么用"。
    """
    return {
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
        # 每个工具的调用从哪条路进来。**目标由 Agent 定的工具走通用出口**
        # （`/fetch/{名字}`），其余走 `/tools/{名字}`——见 `ToolDecl.proxy_path`。
        # 这张表就是"哪次调用该出现在哪条路上"的唯一依据，代理不猜。
        "tools": {t.name: t.proxy_path() for t in scenario.tools},
        # 参数 schema 交给边界。**这是它唯一的去处**——在此之前
        # 场景里那份声明只被列给 Agent 看过，从来没人拿它核对过
        # 一次真实调用，于是沙箱里的目标系统比真实的那一个宽松。
        "tool_schemas": {t.name: t.parameters for t in scenario.tools},
        # 豁免规则也要过线。**看门狗在运行当中读的就是它**——不传的
        # 话，一条"这个重复是我允许的"在事后把结论降级成 pass，而
        # 进程早在第 3 次调用时就被杀了。见 runtime.ActiveRun.exemptions。
        "exemptions": scenario.exemption_set().to_dicts(),
        # 字段权重也要过线，理由和豁免一模一样：看门狗在运行当中
        # 拿它判"这串调用是不是在原地打转"。不过线的话，事后报告
        # 用的是场景声明的那张表，看门狗用的是空的——同一段轨迹，
        # 一边判成正常遍历、一边判成循环，而两边看起来都对。
        "loop_field_weights": scenario.loop_field_weights(),
    }


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
        # 世界里的那几句话。**整体赋值，不是逐条 update**：逐条合并的话，上一条
        # 场景注入的备注会留在这一条场景的世界里，而两条场景的报告都正常——
        # 那正是"世界被上局污染"最难看出来的一种。reset() 也清空它，两处都是
        # "这一局的世界"这个语义。
        state.order_notes = dict(scenario.world.order_notes)
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

        # 这次运行的出口放行名单（ADR 003）。**必须在上面那行之后**：场景声明的
        # 策略是在沙箱默认策略上追加的，而默认策略里那条"沙箱自己的工具后端自动
        # 放行"是照配置现读的——早一步构造，它记下的就是上一个场景的 mock 端口，
        # 于是这一场的**每一次**工具调用都会被自己的出口策略拦下。而报告上写着
        # "Agent 访问了未授权主机"，指向的是被测对象。
        #
        # 没声明的场景拿到的是 None，代理那时按默认策略走（懒构造，同样在端口
        # 定下来之后）。
        proxy_app.set_policy(scenario.egress_policy())

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
            # 授权对照随报告一起落库。**不是渲染时现查场景**——场景文件事后会被
            # 编辑，而"未经授权"必须指的是这次运行当时的那份授权。
            if scenario.autonomous_ordinals():
                result.autonomy = AutonomyReport(
                    autonomous_ordinals=sorted(scenario.autonomous_ordinals()),
                    unattended_allowlist=sorted(scenario.unattended_allowlist()),
                )
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
        # 会话边界上的记忆切面，按时间顺序。**不进快照库**：快照库里的东西都是
        # 能恢复回去的时点，而观测不做恢复；混进去会让报告里那份"回滚到了哪儿"
        # 的清单多出几条永远用不上的位置，归因线索就被稀释了。
        observations: list[MemoryObservation] = []

        # 上游脚本按会话顺序拼接。单游标顺序读，所以每个会话的步骤必须
        # 紧跟在它前面那个会话的步骤之后——见下面消费量检查。
        full_script: list[dict[str, Any]] = []
        for session in scenario.sessions:
            full_script.extend(session.script)

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"http://127.0.0.1:{gw_port}/_sandbox/runs",
                json=registration_payload(scenario),
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

                # 记忆切面：会话**开始时**是什么样。
                #
                # 位置在恢复之后——恢复改的就是记忆，在它之前看等于看上一个会话
                # 的残留，而报告会说这是"这段会话开始时"的样子。这一条错了，
                # 漂移判定的基准就错了，而报告看起来仍然是正常的。
                #
                # 观测一律做、不按声明开关：**声明只决定判不判，不决定看不看。**
                # 让作者声明了才观测的话，"忘了声明"的表现就是这一族永远没有观测，
                # 与"Agent 的记忆一直是空的"在报告上长得一样。
                observations.append(
                    observe_memory(
                        adapter,
                        ordinal=ordinal,
                        phase=PHASE_START,
                        # 这段会话的记忆视野。判定"键不见了"时用它分清算谁的账：
                        # 视野比写下它的时刻还早，那就是我们自己回滚掉了。
                        horizon=active.ctx.memory_horizon_ordinal,
                    )
                )

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

                # 记忆切面：Agent 退出时是什么样。它在下一段会话恢复**之前**——
                # 顺序反了的话，"这段会话写了什么"会被下一段的回滚盖掉，
                # 而那种报告看起来只是"这段会话什么都没写"。
                observations.append(
                    observe_memory(
                        adapter,
                        ordinal=ordinal,
                        phase=PHASE_END,
                        horizon=active.ctx.memory_horizon_ordinal,
                    )
                )

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
                        stdout_tail=stdout[-_STDOUT_TAIL_CHARS:],
                        virtual_started_at=world_at_start,
                        after_gap_s=spec.after_gap_s,
                        declared_offset_s=declared_offset,
                        observed_offset_s=observed_offset,
                        autonomous=spec.autonomous,
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

        result.memory = summarize_observations(observations)
        # 调用链与跨会话差异。放在这里而不是更早：两者要的都是**会话序列**上的
        # 全部调用，以及每段会话的前提是否成立——到这一步才齐。两者都是陈述
        # 不是判定，所以既不进 `findings` 也不进 `ok`。会话侧视图构造一次、
        # 两个模块共用，免得同一段会话在两个地方被读成两套样子。
        sides = [session_diff.SessionSide.of(s) for s in result.sessions]
        result.call_chain = call_chain.build_chain(calls, sides)
        result.session_diff = session_diff.compare_sessions(calls, sides)
        result.findings = _run_detectors(scenario, calls, observations)
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

        try:
            timed_out, stdout = await _wait_for_exit(proc, self.session_timeout_s)
        finally:
            current["proc"] = None

        return proc.returncode, timed_out, stdout, time.perf_counter() - started

    async def _charges(self, mock_port: int) -> int:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"http://127.0.0.1:{mock_port}/_mock/charges")
                return int(resp.json().get("count", 0))
        except Exception:  # noqa: BLE001 — mock 没起来不该让整份报告丢掉
            return -1


def _name_of(ref: SnapshotRef, named: dict[str, SnapshotRef]) -> str | None:
    return next((k for k, v in named.items() if v.id == ref.id), None)


def _run_detectors(
    scenario: Scenario,
    calls: list[Any],
    memories: Sequence[MemoryObservation] = (),
) -> list[Finding]:
    """跑全套检测器。工具事实来自场景声明，是写/读属性的单一来源。

    装配在 `detectors.run_all` 里，**和 `tests/corpus/` 共用同一份**：语料算出来的
    精度只有在"语料跑的就是线上跑的那一套"时才有意义。

    六份声明全部从场景进来（工具事实、无人值守白名单、自主会话、豁免规则、
    记忆判据、字段权重）——观测分两条通道进来：边界上的 `calls` 和会话边界上的
    `memories`。这条分工是全项目的判据，不是这里的实现细节。

    `memories` 有默认值，是为了让只想验边界检测的调用方（比如豁免那一批测试）
    照旧只传两个参数——**不是**"记忆可以不看"：真跑一次场景时 runner 一定传。
    """
    facts = {
        t.name: ToolFacts(t.name, t.is_write, t.enforces_idempotency) for t in scenario.tools
    }
    return run_all(
        calls,
        tool_facts=facts,
        unattended_allowlist=frozenset(scenario.unattended_allowlist()),
        autonomous_ordinals=frozenset(scenario.autonomous_ordinals()),
        exemptions=scenario.exemption_set(),
        memories=memories,
        memory_rules=scenario.memory_rules(),
        loop_field_weights=scenario.loop_field_weights(),
    )


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
        # 无人值守单独标出来。少了它，一份长时运行的报告会把"夜里它自己动的手"
        # 读成"用户让它做的事"——同一串调用，两种完全不同的行为。
        unattended = " [无人值守]" if s.autonomous else ""
        add(
            f"  {mark} #{s.ordinal} {s.name}{restored}{timeout}{killed}{unattended}"
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
                    f"       !! 时钟对不上：场景声明世界走了 {span_human(s.declared_offset_s)}，"
                    f"实际走了 {span_human(s.observed_offset_s)}"
                )
        consumed = f"       脚本消费 {s.script_steps_consumed}/{s.script_steps_declared}"
        if s.killed:
            consumed += f"（熔断所致，前提成立）\n       熔断原因：{s.kill_reason}"
        elif not s.premise_held:
            consumed += "   这一段没跑完，前提未成立 "
        add(consumed)
        # 出岔子的那几段把 Agent 输出印出来——崩了要看 traceback，卡住了要看它
        # 停在哪一行，而在此之前它抓了 2000 字符**从来没给任何人看过**。
        # 正常结束的会话不印：那几千字会把报告淹掉，而它们什么都没解释。
        if s.stdout_tail and not s.ended_normally:
            add(f"       --- Agent 输出（末尾 {_STDOUT_TAIL_CHARS} 字符）---")
            for line in s.stdout_tail.strip().splitlines():
                add(f"       {line}")

    # 调用链紧跟在会话表后面：会话表说"这一段跑了多久、调了几次"，链说"那几次
    # 是按什么顺序发生的、哪几次没成功"。两者都是**看**的结果，判定在下面。
    if result.call_chain is not None:
        ch = result.call_chain
        add("\n--- 调用链（每段会话里，调用按发生顺序）---")
        for chain in ch.chains:
            add(f"  {chain.headline}")
            for node in chain.nodes:
                add(f"     {node.line()}")
            if not chain.premise_held:
                add(
                    "     注意：这一段的前提没成立（理由见上面「会话」一节），"
                    "下面这个顺序不指向被测对象的行为。"
                )
            for step in chain.steps:
                add(f"     > {step.sentence()}")
            if chain.omitted_steps:
                add(
                    f"     还有 {chain.omitted_steps} 种没成功的调用没列出来"
                    f"（每段会话最多列 {ch.step_limit} 种）。"
                )
        add(
            "  这一节是陈述，不是判定：目标系统回了个错不等于被测对象做错了事，所以\n"
            "     它既不进检测结论，也不影响通过与否。写的是顺序、不是因果——「紧接着又\n"
            "     是一次同样的调用」是边界上看见的，「它在重试」是读的人下的结论。"
        )

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
            "  这段时间不是等出来的：引擎把目标系统的时钟往前拨，世界该变的都\n"
            "     变了，而被测 Agent 在两次会话之间本来就不在跑。"
        )

    if result.autonomy is not None:
        # 授权对照表随报告一起给出。"未经授权"这四个字没有它就没法复核：读的人
        # 得自己知道当时允许了什么，才能判断哪一条算越权。
        a = result.autonomy
        allowed = "、".join(a.unattended_allowlist) or "（无）"
        sessions = "、".join(f"#{o}" for o in a.autonomous_ordinals)
        add("\n--- 自主行为（需求场景 6）---")
        add(f"  无人值守的会话：{sessions}")
        add(f"  被授权无人值守运行的工具：{allowed}")
        add(
            "  声明不拦截：没被授权的工具被调用时照样会执行，扣款照样发生。\n"
            "     审计要回答的是「它到底干了什么」；要拦，用的是出口代理的放行名单。"
        )

    # 记忆观测排在检测结论前面：先看"边界上看见的是什么"，再看"据此判了什么"。
    # 两段分开写，是因为判定只发生在场景**声明**过的判据上，而观测是全部——
    # 混在一起的话，"记忆里动过一个文件"会被读成"记忆出了问题"。
    add("\n--- 记忆观测（会话边界上看到的，不是判定）---")
    if result.memory:
        for o in result.memory:
            # 时点的说法从 `phase_label` 来（它也是检测结论和界面用的那一个）：
            # 各拼一份的话，报告里会出现"会话 0开始时"和"会话 0 开始时"两种写法。
            when = phase_label(o["ordinal"], o["phase"])
            horizon = "不知道" if o["horizon"] is None else f"≤ 会话 {o['horizon']}"
            add(f"  {when}  记忆视野 {horizon}  条目 {o['entries']}")
            for label, key in (("新出现", "added"), ("消失", "removed"), ("内容变了", "changed")):
                if o[key]:
                    add(f"       {label}：{'、'.join(o[key])}")
    else:
        add("  一次会话边界都没走到，没有观测。多半是执行失败提前退出了。")
    add(
        "  观测点是会话边界：中途改了又改回来的看不见。这上面列的是「这个时点上\n"
        "     它是什么」，不是「它在会话里没变过」；「内容变了」也不是判定。"
    )

    # 跨会话差异紧接着记忆观测：两节都是"看"的结果，判定在下面。
    if result.session_diff is not None:
        d = result.session_diff
        add("\n--- 跨会话差异（同一场景的两段会话，逐项对齐）---")
        for note in d.notes:
            add(f"  {note}")
        for pair in d.pairs:
            add(f"  {pair.headline}")
            for line in pair.background:
                add(f"     · {line}")
            for line in pair.reasons:
                add(f"     !! {line}")
            for item in pair.differences:
                add(f"     ? [{item.label}] {item.sentence}")
            if pair.omitted:
                add(
                    f"     还有 {pair.omitted} 条差异没列出来"
                    f"（每对会话最多列 {d.limit} 条）。"
                )
        add(
            "  这一节是陈述，不是判定：它不进检测结论，也不影响通过与否。同一个场景\n"
            "     的两段会话本来就允许不一样（长时运行的场景里，第二段看到的世界都不一样\n"
            "     了）。要判「该不该不一样」，看上面那几条结论。"
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
        f"警告 {s.get('warning', 0):.0f} / 未覆盖 {s.get('inconclusive', 0):.0f}"
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
                f"  !! 注意：其中 {c.unpriced_calls} 次调用的模型不在定价表中，"
                f"成本按 0 计。上面的金额是下界，不是账单。"
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
                "     超预算不会让本次运行算作不通过——那是累计花费，和被测\n"
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
            "\n   有会话没把脚本跑完，检测结论不指向被测对象，本次不算通过。\n"
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
