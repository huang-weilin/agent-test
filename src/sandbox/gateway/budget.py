"""运行级硬预算，由网关强制执行。

为什么必须在网关而不是在 Agent 侧：
- 网关是**唯一权威的 Token 计数点**。Agent 侧的 usage_metadata 可能缺失、被缓存
  影响、或被（有意无意地）错报。
- 熔断要能真正止损，就必须设在请求的必经之路上，且**在放行前判定**。

这是需求文档"循环识别与超时熔断"里唯一能真正省下钱的一环：进程内的步数计数
只能"发现"，网关的上限才能"阻止"。
"""

from __future__ import annotations

from dataclasses import dataclass, field


class BudgetExceededError(Exception):
    """超出运行级 Token 上限。网关应据此拒绝请求（而不是放行后告警）。"""

    def __init__(self, *, used: int, ceiling: int, requested: int) -> None:
        self.used = used
        self.ceiling = ceiling
        self.requested = requested
        super().__init__(
            f"运行级 Token 预算已耗尽：已用 {used}，上限 {ceiling}，"
            f"本次请求预估再需 {requested}"
        )


@dataclass
class TokenBudget:
    """单个运行的 Token 预算。

    预扣（reserve）→ 结算（settle）两步：请求放行前先按**预估**上限预扣，
    响应回来后再按**实际**用量修正。只做事后统计的话，超限的那一次已经花出去了。
    """

    ceiling: int
    used: int = 0
    _reserved: int = 0
    calls: int = 0
    rejected: int = 0
    history: list[tuple[int, int]] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return max(0, self.ceiling - self.used - self._reserved)

    def reserve(self, estimated: int) -> None:
        """放行前调用。预估超限则直接拒绝。"""
        if self.used + self._reserved + estimated > self.ceiling:
            self.rejected += 1
            raise BudgetExceededError(
                used=self.used, ceiling=self.ceiling, requested=estimated
            )
        self._reserved += estimated

    def settle(self, reserved: int, actual_input: int, actual_output: int) -> None:
        """响应回来后调用，用实际用量替换预扣。"""
        self._reserved = max(0, self._reserved - reserved)
        actual = actual_input + actual_output
        self.used += actual
        self.calls += 1
        self.history.append((actual_input, actual_output))

    def release(self, reserved: int) -> None:
        """请求失败时归还预扣。"""
        self._reserved = max(0, self._reserved - reserved)


class BudgetRegistry:
    """按 run_id 管理预算。网关是多运行的，必须隔离。"""

    def __init__(self, default_ceiling: int) -> None:
        self._default_ceiling = default_ceiling
        self._budgets: dict[str, TokenBudget] = {}

    def for_run(self, run_id: str, ceiling: int | None = None) -> TokenBudget:
        if run_id not in self._budgets:
            self._budgets[run_id] = TokenBudget(ceiling=ceiling or self._default_ceiling)
        return self._budgets[run_id]

    def snapshot(self, run_id: str) -> dict[str, int]:
        b = self.for_run(run_id)
        return {
            "used": b.used,
            "ceiling": b.ceiling,
            "remaining": b.remaining,
            "calls": b.calls,
            "rejected": b.rejected,
        }

    def reset(self, run_id: str) -> None:
        self._budgets.pop(run_id, None)
