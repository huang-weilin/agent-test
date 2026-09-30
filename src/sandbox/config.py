"""全局配置。

Lite profile（本机开发）默认值：SQLite、无 Docker、mock 与网关为本地进程。
远程重型机（W3 之后）通过环境变量或 .env 覆盖，例如：

    SANDBOX_DATABASE_URL=postgresql+asyncpg://... SANDBOX_LANGFUSE_HOST=http://...
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SANDBOX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
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
    credential_injection_rules: list[dict] = Field(default_factory=list)
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

    @property
    def is_lite_profile(self) -> bool:
        """Lite profile：SQLite + 进程内 sidecar，无需 Docker。"""
        return self.database_url.startswith("sqlite")

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_host and self.langfuse_public_key and self.langfuse_secret_key)


settings = Settings()
