"""锁定架构 ADR 里安全相关的默认值。

这些断言存在的意义不是"测配置能不能读"，而是**当有人把安全默认值改回去时，
测试会失败**。ADR 003 的默认值反转是刻意的设计决策，不是随手写的初值。
"""

from __future__ import annotations

from sandbox.config import Settings


def test_egress_defaults_to_deny() -> None:
    """ADR 003：出口默认拒绝，不是默认放行。

    若此测试失败，说明有人把 egress_default_policy 改成了 "allow" 或改了默认值——
    那是把"能打通就注入真实凭据"的老方案重新引入了。
    """
    assert Settings().egress_default_policy == "deny"


def test_no_credentials_injected_by_default() -> None:
    """ADR 003：默认不注入任何真实凭据。

    默认开启凭据注入会让被提示注入污染的 Agent 有 SSRF / 外泄通道。
    """
    assert Settings().credential_injection_rules == []


def test_default_profile_is_lite() -> None:
    """本机（7.9GB 内存 / exFAT）只能跑 lite profile：SQLite + 进程内 sidecar。

    生产环境通过 SANDBOX_DATABASE_URL 覆盖为 PostgreSQL。
    """
    s = Settings()
    assert s.is_lite_profile is True
    assert s.database_url.startswith("sqlite")


def test_langfuse_is_optional_not_required() -> None:
    """ADR 002：Langfuse 只是可视化前端，不是 system of record。

    缺省不配置时必须能正常启动——检测器的正确性不依赖 Langfuse 可用。
    """
    assert Settings().langfuse_enabled is False


def test_env_override_switches_to_postgres() -> None:
    """远程重型机通过环境变量切到 PostgreSQL，验证配置覆盖链路可用。"""
    s = Settings(database_url="postgresql+asyncpg://user:pw@host/db")
    assert s.is_lite_profile is False
