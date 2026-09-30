"""出口策略：默认拒绝（ADR 003）。

这是一次**刻意的默认值反转**。直觉上沙箱应该"默认放行、拦住已知的坏目标"，
但那样做的失效模式很糟：新的坏目标你没列到，测试跑绿了，而 Agent 已经把钱转出去了。

反过来默认拒绝，失效模式变成"合法目标没列到，测试报错"。报错是可见的、可修的；
静默放行不是。**对"验证型"工具来说，错误的代价必须落在可见的一侧。**

三条附带规则，都来自同一套推理：

- **严格主机白名单**（精确匹配，不支持通配）。`*.example.com` 这种通配等于把
  判断权交给 DNS——而 DNS 恰恰是攻击面。
- **不使用路径正则**。正则匹配路径看着灵活，实际上 `re.match` 与 `re.fullmatch`
  的差别、`.*` 的回溯、编码绕过，每一条都是一个可被利用的缺口。工具级白名单
  已经能表达同样的意图，且没有这些坑。
- **不跟随重定向、不做 TLS 中间人**。跟随重定向等于让目标主机自己扩展白名单；
  做 MITM 就要在沙箱里安装根证书，那等于把被测 Agent 的整个信任链交出去。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class AllowRule:
    """一条放行规则。host 是精确匹配的主机名（不含端口，不含通配符）。"""

    host: str
    tool: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if "*" in self.host:
            raise ValueError(
                f"主机白名单不支持通配（收到 {self.host!r}）："
                f"通配等于把放行判断交给 DNS 解析结果。请写精确主机名。"
            )
        if not self.reason:
            raise ValueError(
                f"放行规则 {self.host!r} 缺少 reason。"
                f"没有理由的放行在审计时无法复核，等于永久豁免。"
            )


@dataclass(frozen=True)
class EgressDecision:
    allowed: bool
    reason: str
    matched_rule: AllowRule | None = None
    # 是否要注入真实凭据。默认 False——见 ADR 003。
    inject_credential: bool = False
    credential_reason: str = ""


class EgressPolicy:
    """出口放行判定。

    `default` 直接取自 settings.egress_default_policy，默认 "deny"。
    设为 "allow" 只应出现在临时排查场景，且必须显式设置——这样它会出现在
    运行配置里，也就出现在了 config_hash 里，报告能追溯。
    """

    def __init__(
        self,
        rules: list[AllowRule] | None = None,
        *,
        default: Literal["deny", "allow"] = "deny",
        credential_rules: list[dict[str, Any]] | None = None,
    ) -> None:
        self.rules = list(rules or [])
        self.default = default
        self.credential_rules = list(credential_rules or [])

    def check(self, host: str, tool: str | None = None) -> EgressDecision:
        host = (host or "").split(":")[0].lower()

        for rule in self.rules:
            if rule.host.lower() != host:
                continue
            if rule.tool is not None and rule.tool != tool:
                continue
            cred = self._credential_for(host, tool)
            return EgressDecision(
                allowed=True,
                reason=f"命中放行规则：{rule.reason}",
                matched_rule=rule,
                inject_credential=cred is not None,
                credential_reason=cred.get("reason", "") if cred else "",
            )

        if self.default == "allow":
            return EgressDecision(
                allowed=True,
                reason="出口默认策略为 allow（非默认配置，应仅用于临时排查）",
            )

        return EgressDecision(
            allowed=False,
            reason=(
                f"出口默认拒绝：主机 {host!r}"
                + (f"（工具 {tool!r}）" if tool else "")
                + " 不在白名单中。要放行需显式添加精确主机规则并写明 reason。"
            ),
        )

    def _credential_for(self, host: str, tool: str | None) -> dict[str, Any] | None:
        """查是否有为该目标注入真实凭据的规则。

        默认没有任何规则 → 不注入。凭据绝不进入被测 Agent 的进程环境或文件系统，
        只在出口代理这一跳被附加。
        """
        for rule in self.credential_rules:
            if str(rule.get("host", "")).lower() != host:
                continue
            if rule.get("tool") is not None and rule.get("tool") != tool:
                continue
            if not rule.get("reason"):
                raise ValueError(
                    f"凭据注入规则 {rule!r} 缺少 reason。"
                    f"注入真实凭据是高危操作，必须留下可复核的理由。"
                )
            return rule
        return None


def default_policy() -> EgressPolicy:
    """按 settings 构造默认策略。工具代理启动时用它。

    **这里有一处必须讲清楚的例外。** 纯"默认全拒"会让沙箱连自己的 mock 后端都
    连不上，所以要把"沙箱内部组件"和"外部出口"分开：

    - 沙箱自己的工具后端（`tool_upstream_base_url`）是**内部组件**，自动放行，
      但仅限精确主机、且仅限已声明的工具。
    - 其余一切出口走 `egress_default_policy`，默认拒绝。

    这个例外之所以安全，是因为它由**运行配置**决定：如果谁把 tool_upstream_base_url
    指到 `api.stripe.com`，那是一次显式的配置行为，会进入 config_hash，报告里
    看得到。它不是隐藏在代码里的后门。
    """
    from sandbox.config import settings

    internal_host = host_of(settings.tool_upstream_base_url)
    rules = [
        AllowRule(
            host=internal_host,
            tool=None,
            reason=(
                f"沙箱内部工具后端（tool_upstream_base_url 配为 {settings.tool_upstream_base_url}）"
            ),
        )
    ] if internal_host else []
    rules.extend(
        AllowRule(host=h, tool=None, reason="由 SANDBOX_EGRESS_ALLOW_HOSTS 显式放行")
        for h in settings.egress_allow_hosts
    )

    if not rules:
        raise RuntimeError(
            "无法从 tool_upstream_base_url 解析出主机，也没有显式配置的白名单主机。"
            "拒绝以全拒配置启动——那会让每次工具调用都失败，而失败原因看起来"
            "像 Agent 出了问题，会把人引到完全错误的方向。"
        )

    return EgressPolicy(
        rules=rules,
        default=settings.egress_default_policy,
        credential_rules=settings.credential_injection_rules,
    )


def host_of(url: str) -> str:
    """取 URL 的主机名（小写，不含端口）。"""
    from urllib.parse import urlparse

    return (urlparse(url).hostname or "").lower()
