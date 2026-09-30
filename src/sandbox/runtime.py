"""运行注册表：把边界上进来的请求路由到具体的 TestRun。

**为什么需要它，以及"只改 base_url"这个说法要修正到什么程度。**

ADR 001 的验收标准写的是"被测 Agent 只改 `base_url` 就能被完整观测"。严格说
这不成立——任何 LLM 客户端都还要一个 api_key 才能发请求。所以要老实讲清楚：
**不能只改 base_url，还要改 api_key；但这是配置变更，不是代码变更，也不需要
注入 SDK。** 这才是 ADR 001 真正要主张的东西。

于是让 api_key 顺带承担运行标识：

    api_key = "sk-sandbox-<run_id>"

这样 Agent 手里的凭据从真实密钥换成了沙箱令牌——**真实密钥只存在于网关进程内**，
永远不进被测 Agent 的环境（ADR 003）。既解决了路由，又顺手完成了凭据隔离。

对于确实只能改 base_url、连 api_key 都改不了的情况（某些封装死的 SDK），
提供 `resolve_ambient()`：当且仅当**只有一个**运行处于活跃状态时，无标识的
请求路由到它。多个运行同时活跃时必须拒绝，绝不能猜——猜错会把 A 运行的工具
调用记到 B 运行头上，而这种串台极难发现。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sandbox.config import settings
from sandbox.gateway.budget import BudgetRegistry, TokenBudget
from sandbox.trace.recorder import RunContext

RUN_TOKEN_PREFIX = "sk-sandbox-"


def make_run_token(run_id: str) -> str:
    return f"{RUN_TOKEN_PREFIX}{run_id}"


def parse_run_token(token: str | None) -> str | None:
    """从 Authorization 头里取出 run_id。认不出就返回 None（不猜）。"""
    if not token:
        return None
    value = token.strip()
    if value.lower().startswith("bearer "):
        value = value[7:].strip()
    if value.startswith(RUN_TOKEN_PREFIX):
        candidate = value[len(RUN_TOKEN_PREFIX) :]
        return candidate or None
    return None


@dataclass
class ActiveRun:
    ctx: RunContext
    budget: TokenBudget
    scenario_id: str = ""

    # 本次运行声明的工具 → 它们在目标系统上的路径。
    #
    # 挂在**运行**上而不是做成模块级全局，是因为工具表来自场景文件。全局的话，
    # 一个进程里跑第二个场景时会串味，而串味的表现是"某个工具莫名其妙放行了"
    # 或"明明声明了却被判未声明"——两者都很难往配置上想。
    tools: dict[str, str] = field(default_factory=dict)

    # 本次运行声明的工具 → 参数的 JSON Schema。出口代理拿它核对每一次调用的参数。
    #
    # 与上面那张表分开是**有意的**：它们回答两个不同的问题（"这个工具往哪儿发"
    # 和"它的参数该长什么样"），而且路径表有一条"没传就回落到内置工具表"的便利
    # 路径——那条路径下没有 schema 表，正好表达"手工起的运行不做参数校验"，
    # 不需要再编一个"校验已关闭"的开关出来。
    #
    # 空表 = 不校验。这与"这个工具没声明参数"是同一件事，没有第二种解释。
    tool_schemas: dict[str, dict[str, Any]] = field(default_factory=dict)

    # 带外熔断状态。一旦置位，该运行的所有后续请求都会被网关拒绝——
    # 这是"杀掉"在带内那一侧的表现；带外那一侧是真正终止 Agent 进程。
    killed: bool = False
    kill_reason: str = ""

    @property
    def run_id(self) -> str:
        return self.ctx.run_id

    @property
    def token(self) -> str:
        return make_run_token(self.ctx.run_id)


class RunRegistry:
    """活跃运行的进程内注册表。

    刻意不做持久化：它是**运行时路由状态**，不是记录。进程重启后运行本就
    不该继续（沙箱里的运行不跨重启），重启还在跑的运行属于状态丢失，
    应该显式报错而不是让路由悄悄指向一个死掉的运行。
    """

    def __init__(self, default_ceiling: int | None = None) -> None:
        self._runs: dict[str, ActiveRun] = {}
        self._budgets = BudgetRegistry(default_ceiling or settings.run_token_ceiling)
        self._kill_handlers: list[Callable[[str, str], None]] = []

    def register(
        self,
        ctx: RunContext,
        *,
        scenario_id: str = "",
        ceiling: int | None = None,
        tools: dict[str, str] | None = None,
        tool_schemas: dict[str, dict[str, Any]] | None = None,
    ) -> ActiveRun:
        active = ActiveRun(
            ctx=ctx,
            budget=self._budgets.for_run(ctx.run_id, ceiling),
            scenario_id=scenario_id,
            tools=dict(tools or {}),
            tool_schemas=dict(tool_schemas or {}),
        )
        self._runs[ctx.run_id] = active
        return active

    def get(self, run_id: str) -> ActiveRun | None:
        return self._runs.get(run_id)

    def resolve(
        self, authorization: str | None = None, run_id_header: str | None = None
    ) -> ActiveRun | None:
        """按优先级解析：显式 run-id 头 > 沙箱令牌 > 唯一活跃运行兜底。"""
        if run_id_header:
            return self._runs.get(run_id_header)
        if (rid := parse_run_token(authorization)) is not None:
            return self._runs.get(rid)
        return self.resolve_ambient()

    def resolve_ambient(self) -> ActiveRun | None:
        """无标识请求的兜底。只有在唯一活跃运行时才敢这么干。"""
        if len(self._runs) == 1:
            return next(iter(self._runs.values()))
        return None

    def kill(self, run_id: str, reason: str) -> ActiveRun | None:
        """标记一个运行已被熔断，并通知带外处理器。

        返回 None 表示该运行已不在活跃表中（可能已自行结束）——这**不是**错误，
        但调用方应当知道"没杀到东西"和"杀掉了"是两回事。
        """
        active = self._runs.get(run_id)
        if active is None:
            return None
        active.killed = True
        active.kill_reason = reason
        for handler in self._kill_handlers:
            handler(run_id, reason)
        return active

    def on_kill(self, handler: Callable[[str, str], None]) -> None:
        """注册带外处理器（通常是"终止 Agent 进程"）。

        带外是关键词：网关拒绝后续请求只能让 Agent 的**下一次**调用失败，
        如果它正在一次长时间的推理里，钱还在烧。真正止损要杀掉进程。
        """
        self._kill_handlers.append(handler)

    def unregister(self, run_id: str) -> None:
        self._runs.pop(run_id, None)
        self._budgets.reset(run_id)

    def clear(self) -> None:
        """重置全部状态，**包括带外处理器**。

        处理器也要清，否则测试之间会串：上一个测试注册的 lambda 还挂着，
        下一个测试熔断时会去写一个已经结束的测试的局部变量。这类串扰不会
        让断言失败，但会让"熔断通知了几次"这类计数悄悄失真。
        """
        self._runs.clear()
        self._kill_handlers.clear()

    def active_ids(self) -> list[str]:
        return sorted(self._runs)

    def snapshot(self) -> list[dict[str, object]]:
        return [
            {
                "run_id": r.run_id,
                "scenario_id": r.scenario_id,
                **self._budgets.snapshot(r.run_id),
            }
            for r in self._runs.values()
        ]


# 进程级单例。网关与出口代理同进程时共用它，跨进程时各自持有并靠 run_id 对齐。
runs = RunRegistry()
