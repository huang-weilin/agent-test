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
from sandbox.detectors.base import ExemptionSet
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

    # 本次运行声明的豁免规则。与上面两张表同一个理由挂在运行上——它来自场景。
    #
    # **看门狗也要用它。** 熔断是运行**当中**发生的，那时检测器唯一的输入就是
    # 这份注册表；不看豁免的话，一条"这个重复是我允许的"声明在事后把结论降级成
    # pass，而进程早就在第 3 次调用时被杀掉了——声明写了，运行还是没跑完。那正是
    # 豁免机制存在的理由的反面：它本来是给"我知道这看起来像缺陷，但它不是"用的，
    # 而那种场景里最需要它生效的时刻恰恰是熔断那一刻。
    exemptions: ExemptionSet = field(default_factory=ExemptionSet)

    # 场景声明的字段权重（字段名 → 权重）。和上面那张豁免表同一个理由挂在这里：
    # 它来自场景，而**看门狗在运行当中要用它**判"这串调用算不算在原地打转"。
    #
    # 不过线的话，两边会用不同的判卷标准算出不同的结论：事后报告拿场景声明的那张
    # 表判成"正常遍历"，看门狗拿空表把同一串调用切成好几个簇——它未必会因此误杀，
    # 但"两处结论必须一致"是看门狗存在的全部前提。空表是合法的：它说的是"这份
    # 场景没有声明权重"，那时检测器照旧按参数形状猜（见 LoopDetector.key_weights）。
    loop_field_weights: dict[str, float] = field(default_factory=dict)

    # 已经就哪几条豁免警告过一次了。**只是日志去重**，不参与判定。
    #
    # 看门狗一次运行里要重扫很多遍，一条持续 100 次调用的被豁免循环会刷 100 行
    # 一模一样的 warning——而那条日志的全部意义是"事后查得到"，被刷屏之后它就
    # 查不到了。跟着运行走，运行结束自然消失，不需要谁去清理。
    warned_exemptions: set[str] = field(default_factory=set)

    # 已经落过库的结论，键是 `(检测器, 四态)`。**结论一行，不是扫描一次一行。**
    #
    # 与上面那条同一个来路：看门狗一次运行里要重扫很多遍，而同一份轨迹在
    # 下一次调用到来之前不会改变结论——不认这一点的话，一次 320 次调用的运行会
    # 往 `detector_finding` 里写 640 行，而它们**只是同两条结论的副本**（量于
    # 2026-10-01，开发库里当时 4,320 行）。那张表的名字是"检测器结论"，一行
    # 就该是一个结论。
    #
    # 键里只有检测器和四态，**不含 summary**：摘要里带着"已在 N 次调用中比对"
    # 这样的过程数字，每扫一次都会变。变的是过程，不是判定——过程在日志里，
    # 判定在库里。四态**必须**在键里：pass 变成 fail 是一个新结论，那正是
    # 事后最需要查到的一行（比如熔断那一刻），拿旧的那行盖过去是不行的。
    recorded_conclusions: set[tuple[str, str]] = field(default_factory=set)

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
        exemptions: ExemptionSet | None = None,
        loop_field_weights: dict[str, float] | None = None,
    ) -> ActiveRun:
        active = ActiveRun(
            ctx=ctx,
            budget=self._budgets.for_run(ctx.run_id, ceiling),
            scenario_id=scenario_id,
            tools=dict(tools or {}),
            tool_schemas=dict(tool_schemas or {}),
            exemptions=exemptions or ExemptionSet(),
            loop_field_weights=dict(loop_field_weights or {}),
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
