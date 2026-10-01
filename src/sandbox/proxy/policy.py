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

#: `egress_host` 的**保留值**：这个工具的目标由被测对象在调用时指定。
#:
#: 它是唯一一类"目标不是说好的那台"的出口。所以它不是一个主机名，而是一句
#: 关于工具的声明——这个工具本身**没有**可声明的目标，判定只能按它实际要连的
#: 那台做。保留给这个意思，不做别用：真有一台主机叫这个名字的话，它写不进来，
#: 这是刻意的——一个值一旦既是主机名又是保留字，读的人没法从字面上分辨一条
#: 声明是哪个意思。
AGENT_CHOOSES_HOST = "agent"

#: 通用出口的路径前缀。目标由 Agent 定的工具，调用从这里进来。
GENERIC_EGRESS_PREFIX = "/fetch"

#: 通用出口信封里的两个字段：`url` 是目的地，`body` 是转发到那里的请求体。
#:
#: **只有这两个，而且是加载期校验过的**（见 `ToolDecl`）：目标由 Agent 定的工具
#: 在场景里声明参数时，声明只能是这个形状。不这么钉死的话，场景可以给这种工具
#: 声明一个目标系统的 schema——而那条路上没人执行它（沙箱不知道目标系统接受
#: 什么），于是"写了却没人核对"的那类失效又回来了。
GENERIC_EGRESS_URL_FIELD = "url"
GENERIC_EGRESS_BODY_FIELD = "body"


def generic_egress_path(tool_name: str) -> str:
    """目标由 Agent 定的工具，它的调用走哪条路进来。

    **带上工具名**，和 `/tools/{名字}` 一个道理：沙箱要能把这次调用记在哪个
    工具名下。名字是被测对象给的（它写 URL），所以它只是一句**声称**——代理会
    拿它去对本次运行的工具表，对不上照样按"未声明的工具"处理。它影响的是
    这次调用**记在谁名下**，不影响判它什么（判的是 URL 里那台主机）。
    """
    return f"{GENERIC_EGRESS_PREFIX}/{tool_name}"


def checked_mock_path(value: str, *, what: str) -> str:
    """模拟世界里"这台主机由谁扮演"的那条路径写对了没有。

    拦住两种写法：不写路径（那是 URL 的写法，会让代理拼出一个指向别处的东西）
    和带协议（`http://evil.test/x`——它看起来像"改投到 mock"，实际是把判定过
    的那台主机换成另一台）。
    """
    path = (value or "").strip()
    if not path:
        raise ValueError(f"{what}不能为空。")
    if "://" in path:
        raise ValueError(
            f"{what} {path!r} 里带了协议。这里要的是 mock 上的路径（例如 "
            f"/tools/create_payment），不是一条 URL——写 URL 等于把判定过的主机"
            f"换成另一台。"
        )
    if not path.startswith("/"):
        raise ValueError(
            f"{what} {path!r} 不是一条路径（要以 / 开头）。"
            f"要的是 mock 上的路径，例如 /tools/create_payment。"
        )
    return path


@dataclass(frozen=True)
class AllowRule:
    """一条放行规则。host 是精确匹配的主机名（不含端口，不含通配符）。"""

    host: str
    tool: str | None = None
    reason: str = ""
    #: 这个世界里，这台主机由 mock 的哪条路径扮演。`None` = 不翻译，真的发过去。
    #:
    #: 模拟出来的外部系统（合作方的回调地址、第三方接口）并不存在，出去的请求
    #: 得有个落点。写一条 mock 上的路径，代理就把这次调用改投到那儿，而判定
    #: 照旧按**写下来的那台主机**做——这正是"沙箱里每个工具连的都是自己的
    #: mock，真实部署里它们各是一台机器"在通用出口上的同一件事。
    #:
    #: 它**不是一条授权**：写不写、写成什么，都不影响放不放行，只影响这次调用
    #: 落在哪。不写它的放行是真出口——在测试环境里通常连不出去，那也是一种
    #: 如实的失败，不是静默放行。
    mock_path: str | None = None

    def __post_init__(self) -> None:
        if self.mock_path is not None:
            object.__setattr__(
                self,
                "mock_path",
                checked_mock_path(self.mock_path, what=f"放行规则 {self.host!r} 的 mock_path"),
            )
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


def checked_host(value: str, *, what: str) -> str:
    """一个主机名写对了没有。**放行名单按主机名逐字比对，所以它必须真的是主机名。**

    拦住三种写法，它们的共同点是"看起来像一条放行规则、实际上永远不会命中"：

    - 带协议（`https://payments.example.com`）：比对的是字符串，`https://` 那一段
      会让它和任何真实主机都对不上；
    - 带端口或路径（`payments.example.com:443`、`payments.example.com/v1`）：
      `EgressPolicy.check` 会先把端口切掉，于是这条规则**偶尔**命中偶尔不命中，
      取决于写的人当时怎么想的；
    - 通配（`*.example.com`）：见 `AllowRule` 的说明，那等于把判断权交给 DNS。

    报错而不是"帮你规整一下"：规整过的规则会让人以为自己写了什么就生效了什么。
    """
    host = (value or "").strip()
    if not host:
        raise ValueError(f"{what}的主机名不能为空。")
    if host.lower() == AGENT_CHOOSES_HOST:
        raise ValueError(
            f"{what}的主机名是保留值 {AGENT_CHOOSES_HOST!r}。它表示「这台主机由被测"
            f"对象在调用时指定」，是一句关于工具的声明，不是一台真的主机——"
            f"名单里写它只会得到一条永远不命中的规则。要声明开放式目标，"
            f"写在那个工具的 egress_host 上。"
        )
    if "://" in host:
        raise ValueError(
            f"{what}的主机名 {host!r} 里带了协议。这里要的是主机名本身"
            f"（例如 payments.example.com），放行名单是按主机名逐字比对的。"
        )
    if "*" in host:
        raise ValueError(
            f"{what}的主机名 {host!r} 里带通配。放行名单只支持精确主机名——"
            f"通配等于把放行判断交给 DNS 解析结果（见 AllowRule）。"
        )
    if "/" in host or ":" in host:
        raise ValueError(
            f"{what}的主机名 {host!r} 里带了路径或端口。这里只要主机名本身："
            f"端口会被切掉、路径不参与比对，写进去只会让这条声明的意思变得含糊。"
        )
    return host.lower()


class EgressPolicy:
    """出口放行判定。

    `default` 直接取自 settings.egress_default_policy，默认 "deny"。
    设为 "allow" 只应出现在临时排查场景，且必须显式设置——这样它会出现在
    运行配置里，也就出现在了 config_hash 里，报告能追溯。

    `tool_hosts`（工具名 → 主机名）回答的是"这个工具的出口**目标**是谁"。
    沙箱里每一次工具调用实际连的都是自己的 mock（传输层只有一个地址），但
    真实环境里每个工具出到的是各自那台机器——`payments.example.com` 与
    `orders.example.com` 就是两条不同的出口。没有这张表的话，放行名单只能按
    mock 的地址判，于是"允许读订单、不允许扣款"这种再普通不过的部署策略在这
    根本表达不出来：两个工具在名单上永远是同一个主机。

    它和规则表**必须一起来**（场景侧有加载期校验）：一张只认识 mock 地址的
    规则表碰上按真实主机名判定的调用，结果是把所有出口都拒掉，而报告上写着
    "Agent 访问了未授权主机"——归因完全反了。
    """

    def __init__(
        self,
        rules: list[AllowRule] | None = None,
        *,
        default: Literal["deny", "allow"] = "deny",
        credential_rules: list[dict[str, Any]] | None = None,
        tool_hosts: dict[str, str] | None = None,
    ) -> None:
        self.rules = list(rules or [])
        self.default = default
        self.credential_rules = list(credential_rules or [])
        self.tool_hosts = dict(tool_hosts or {})
        # 保留值不许进这张表。放进去的话 `declared_host` 会返回字面量 "agent"，
        # 于是这个工具的每一次调用都按一台名叫 agent 的主机判——一律拒绝，
        # 而报告上写着「Agent 访问了未授权主机」，归因完全反了。
        reserved = sorted(n for n, h in self.tool_hosts.items() if h == AGENT_CHOOSES_HOST)
        if reserved:
            raise ValueError(
                f"工具 {reserved} 的目标由被测对象指定（egress_host 写了保留值 "
                f"{AGENT_CHOOSES_HOST!r}），它们不该出现在 tool_hosts 里"
                f"——那张表回答的是「这个工具出到哪台主机」，而这几个工具没有"
                f"可写下来的目标。"
            )

    def declared_host(self, tool_name: str, fallback: str) -> str:
        """这次调用该按哪台主机判。

        声明过就用声明的那台（真实环境里的目标），没声明就用它实际连的那台
        ——后者是沙箱的内部组件，由 `default_policy()` 自动放行。

        **目标由被测对象指定的工具不在 `tool_hosts` 里**（见 `__init__` 的检查），
        所以它们走的正是回落那一支：按这次调用实际要连的那台主机判。调用方
        传进来的 `fallback` 就是那条 URL 里的主机。
        """
        return self.tool_hosts.get(tool_name) or fallback

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


def policy_with_rules(
    rules: list[AllowRule], *, tool_hosts: dict[str, str]
) -> EgressPolicy:
    """沙箱默认策略 + 场景声明的放行名单。

    **是在默认策略上追加，不是替换。** 默认策略里那条"沙箱自己的工具后端自动
    放行"是内部组件例外（见 `default_policy()` 的说明），它和"Agent 的出站
    目的地"是两件事：把整张表换掉的话，声明了出口名单的场景连自己的 mock 都
    连不上，而报告会显示"Agent 访问了未授权主机"——把沙箱自己的地址算到了
    被测对象头上。

    追加也不构成后门：声明过 `egress_host` 的工具**不按**它实际连的那台判，
    而是按真实目标主机判（见 `declared_host`），所以内部例外碰不到它们。
    目标由被测对象指定的那些（保留值，见 `AGENT_CHOOSES_HOST`）反过来：
    它们的 `tool_hosts` 项在场景侧就被摘掉了（`Scenario.egress_policy`），
    于是内部例外同样碰不到它们——判的永远是 URL 里那台主机。
    """
    base = default_policy()
    return EgressPolicy(
        rules=[*base.rules, *rules],
        default=base.default,
        credential_rules=base.credential_rules,
        tool_hosts=tool_hosts,
    )


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
