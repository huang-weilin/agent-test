"""全局配置。

Lite profile（本机开发）默认值：SQLite、无 Docker、mock 与网关为本地进程。
远程重型机（W3 之后）通过环境变量或 .env 覆盖，例如：

    SANDBOX_DATABASE_URL=postgresql+asyncpg://... SANDBOX_LANGFUSE_HOST=http://...
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

#: `.env` 的位置。**只有一个来源**：`model_config` 和下面那个校验器读的是同一个
#: 名字，各写各的就会有一天一个按绝对路径读、一个按当前目录读。
#:
#: 绝对路径是刻意的：相对路径按**当前工作目录**解析，从别处启动就读不到——
#: key 配了、什么都没报、`upstream_ready` 却是 False。这种"换个目录就没了"
#: 和写错变量名一样查不出来。
ENV_FILE = REPO_ROOT / ".env"

#: `SANDBOX_` 开头、但**不由本沙箱读**的环境变量：它们由沙箱注入给被测 Agent，
#: 读它们的是 Agent 自己（见 `agents/order_agent/README.md`）。
#: 这份名单是白名单，不是"认识的变量表"——沙箱自己的字段名由 `Settings` 现算。
AGENT_ENV_NAMES: frozenset[str] = frozenset(
    {
        "SANDBOX_AGENT_API_KEY",
        "SANDBOX_AGENT_MEMORY_DIR",
        "SANDBOX_AGENT_MODEL",
        "SANDBOX_AGENT_TASK",
        "SANDBOX_GATEWAY_BASE_URL",
        "SANDBOX_TOOL_BASE_URL",
    }
)


def _known_names() -> set[str]:
    """本沙箱会读的 `SANDBOX_*` 名字。从字段表现算，不手抄。"""
    return {f"SANDBOX_{name}".upper() for name in Settings.model_fields} | set(AGENT_ENV_NAMES)


def _names_in_env_file(path: Path) -> set[str]:
    """`.env` 里的键名（不是值）。**这份文件里写错的键同样会被静默丢掉**，
    而它正是绝大多数人配 key 的地方，所以它必须一起查。"""
    if not path.is_file():
        return set()
    names = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        names.add(line.removeprefix("export ").split("=", 1)[0].strip().upper())
    return names


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SANDBOX_",
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        # **校验失败时的报错里不许带上字段值。** 默认行为是把整个输入原样印出来，
        # 而 `upstream_api_key` 就在里面——一次配置写错（比如变量名拼错）会把
        # 真实密钥打进日志和 traceback。ADR 003 的整条主张是凭据不出沙箱，
        # 而从 stderr 漏出去和从网络漏出去没有区别。
        hide_input_in_errors=True,
    )

    @model_validator(mode="after")
    def _no_unread_sandbox_vars(self) -> Settings:
        """环境里那些**没有任何代码读**的 `SANDBOX_*` 变量，必须在启动时炸出来。

        `extra="ignore"` 是这里的关键：一个写错名字的变量不会报错、不会进日志，
        它会**静默消失**。表现是 key 明明配在 `.env` 里，`settings.upstream_api_key`
        却是 `None`，于是 `upstream_mode=real` 说"没有配置 SANDBOX_UPSTREAM_API_KEY"
        ——而那个人刚刚照着 `.env.example` 把它填了进去。整个真实上游那条路就此
        不通，且没有任何一处说得清为什么。

        （这不是假设：`.env.example` 里写的曾经就是 `SANDBOX_DEEPSEEK_API_KEY`，
        而代码读的是 `SANDBOX_UPSTREAM_API_KEY`。）

        只记一行日志是不够的——没人读的日志和静默是同一件事。
        """
        known = _known_names()
        found = {k.upper() for k in os.environ if k.upper().startswith("SANDBOX_")}
        found |= _names_in_env_file(ENV_FILE)
        unknown = sorted(found - known)
        if not unknown:
            return self

        raise ValueError(
            "环境里有以 SANDBOX_ 开头、但没有任何代码读的变量：\n    "
            + "\n    ".join(unknown)
            + "\n\n本沙箱读的是这些：\n    "
            + "\n    ".join(sorted(known))
            + '\n\n这一类写错名字的变量不会报错也不会进日志——extra="ignore" 会让它'
            "静默消失，于是那个值从来没生效过，而报告上完全看不出来。"
            "\n如果某个变量确实是配给被测 Agent 的（不是配给沙箱的），把它加进"
            " sandbox/config.py 的 AGENT_ENV_NAMES。"
        )

    # --- 数据层 -------------------------------------------------------------
    # ADR 002：自研库是 system of record。开发期 SQLite，生产期 PostgreSQL。
    database_url: str = f"sqlite+aiosqlite:///{(REPO_ROOT / 'data' / 'sandbox.db').as_posix()}"

    # Langfuse 仅作人的可视化前端，不是程序化查询的数据源（ADR 002）。
    langfuse_host: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None

    # --- LLM 网关（ADR 001：黑盒边界拦截的主路径之一）----------------------
    gateway_host: str = "127.0.0.1"
    gateway_port: int = 4100
    # 单次运行的硬 Token 上限。网关是唯一权威计数点，此上限由网关强制执行，
    # 不信任 Agent 侧的任何计数（这是"熔断"能真正止损的前提）。
    run_token_ceiling: int = 2_000_000

    # 上游模式。stub = 脚本化假上游，行为完全确定，用于验证管道本身；
    # deepseek/openai 类 = 真实上游。默认 stub 是刻意的：先证明管道正确，
    # 再引入真实模型的不确定性。
    upstream_mode: Literal["stub", "real"] = "stub"
    upstream_base_url: str = "https://api.deepseek.com"
    upstream_api_key: str | None = None

    @property
    def upstream_ready(self) -> bool:
        return bool(self.upstream_api_key)

    # --- 工具出口代理（ADR 003：默认拒绝）----------------------------------
    proxy_host: str = "127.0.0.1"
    proxy_port: int = 4101
    # 默认拒绝所有未显式声明的出口目标。
    egress_default_policy: Literal["deny", "allow"] = "deny"
    # 真实凭据注入规则。默认空列表 = 不注入任何真实凭据。
    # 每条规则需含 host（精确匹配）、reason（审计用），且禁止路径正则。
    credential_injection_rules: list[dict[str, Any]] = Field(default_factory=list)
    # 工具的真实后端（mock 目标系统）。代理是唯一知道这个地址的地方——
    # 被测 Agent 只应知道代理的地址，否则它可以绕过代理直连后端。
    tool_upstream_base_url: str = "http://127.0.0.1:4102"
    # 工具后端的精确主机白名单，代理据此放行。空 = 全部拒绝。
    # 形如 ["127.0.0.1"]，不支持通配（见 proxy/policy.py）。
    egress_allow_hosts: list[str] = Field(default_factory=list)

    # --- Trace 归一化 -------------------------------------------------------
    # 注意：OTel 的 gen_ai.* 语义约定仍在演进。此版本号必须在 W1–W2 实现
    # trace 层时按当时实际使用的版本核实并钉死，不要让属性名散落在业务代码里。
    otel_semconv_version: str = "TO_BE_PINNED"

    # --- 记忆快照（跨会话测试的操纵面）--------------------------------------
    # 快照与 Agent 记忆目录的根。**必须落在 data/ 之下**——那里已被 gitignore，
    # 而记忆快照里往往含对话历史与用户数据，绝不能进版本库。
    memory_root: Path = REPO_ROOT / "data" / "memory"
    # 恢复快照是破坏性操作（要覆盖 Agent 的记忆目录）。开启后，只允许恢复到
    # memory_root 之下的路径；指向别处会直接拒绝。
    # 关掉它只有一个正当理由：测试真实 Agent 时它的记忆目录不在我们管辖内。
    # 那种情况下请显式关闭，而不是让它默认就敢往任意路径写。
    memory_restore_guard: bool = True

    # --- 长期预算预警（需求场景 4）------------------------------------------
    # 单次运行的 token 上限是网关的事（`gateway/budget.py`，超了就熔断）。
    # 这三个是**跨运行的累计预算**：云端常驻 Agent 的成本失控不会发生在一次
    # 运行里，而是"这个月已经烧掉多少"。
    #
    # 单位美元，0 = 不设限（默认）。默认关掉是刻意的：一个没配过的阈值一旦
    # 生效，就会在别人第一次跑的时候弹出一堆他不认识的告警。
    #
    # 超预算**不会让任何运行失败**，只出现在报告的一节里。理由见
    # `cost/aggregator.py` 的 `budget_status()`。
    cost_budget_daily_usd: float = 0.0
    cost_budget_weekly_usd: float = 0.0
    cost_budget_monthly_usd: float = 0.0

    @property
    def is_lite_profile(self) -> bool:
        """Lite profile：SQLite + 进程内 sidecar，无需 Docker。"""
        return self.database_url.startswith("sqlite")

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_host and self.langfuse_public_key and self.langfuse_secret_key)


settings = Settings()
