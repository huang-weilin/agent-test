"""锁定架构 ADR 里安全相关的默认值。

这些断言存在的意义不是"测配置能不能读"，而是**当有人把安全默认值改回去时，
测试会失败**。ADR 003 的默认值反转是刻意的设计决策，不是随手写的初值。

最后一段是另一件事：**写错名字的配置变量不许静默消失**。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from sandbox.config import AGENT_ENV_NAMES, REPO_ROOT, Settings, _names_in_env_file


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


# --- 写错名字的变量不许静默消失 -------------------------------------------------


def test_a_misspelled_env_var_refuses_to_start(monkeypatch) -> None:
    """**这个文件里最该红的一条。**

    `.env.example` 里写的曾经是 `SANDBOX_DEEPSEEK_API_KEY`，而代码读的是
    `SANDBOX_UPSTREAM_API_KEY`——`extra="ignore"` 让前者静默消失，于是 key 明明
    配着、`upstream_ready` 却是 False，`upstream_mode=real` 报"没有配置
    SANDBOX_UPSTREAM_API_KEY"，而那个人刚刚照着示例把它填了进去。真实上游那条路
    整条不通，且没有任何一处说得清为什么。

    只记一行日志不够：没人读的日志和静默是同一件事。
    """
    monkeypatch.setenv("SANDBOX_DEEPSEEK_API_KEY", "sk-whatever")

    with pytest.raises(ValidationError, match="没有任何代码读"):
        Settings()


def test_a_configuration_error_does_not_print_the_key(monkeypatch) -> None:
    """配置报错里**不许带上字段值**：那会把真实密钥打进日志和 traceback。

    pydantic 的默认行为是把整个输入原样印在错误里，而 `upstream_api_key` 就在
    里面——一次变量名拼错会顺带把凭据漏出去。从 stderr 漏和从网络漏没有区别，
    而 ADR 003 的整条主张是凭据不出沙箱。

    这条故意用一条**必然会失败**的配置来触发：要验的是"报错这条路"，不是某一条
    具体的校验规则。
    """
    monkeypatch.setenv("SANDBOX_DEEPSEEK_API_KEY", "sk-x")  # 触发失败
    monkeypatch.setenv("SANDBOX_UPSTREAM_API_KEY", "sk-this-must-not-be-printed")

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    assert "sk-this-must-not-be-printed" not in str(exc_info.value)


def test_the_agent_allowlist_matches_what_the_agent_actually_reads() -> None:
    """白名单必须和被测 Agent 真正读的那几个**恰好**对得上。

    多了：沙箱替 Agent 吃掉一个写错的变量名，那个错就再也报不出来——正是这份
    白名单想防的事，从另一边绕了回来。少了：Agent 的某个变量会让沙箱拒绝启动。

    所以它不手抄，直接扫 Agent 的源码。
    """
    pattern = re.compile(r"SANDBOX_[A-Z0-9_]+")
    found: set[str] = set()
    for path in (REPO_ROOT / "agents").rglob("*.py"):
        if ".venv" in path.parts:
            continue
        found |= set(pattern.findall(path.read_text(encoding="utf-8", errors="replace")))

    assert found == set(AGENT_ENV_NAMES)


def test_the_agent_s_own_env_vars_are_not_mistaken_for_typos(monkeypatch) -> None:
    """反面对照：配给**被测 Agent** 的那几个不是写错，不许因此拦住启动。

    少了这一条，上面那条可以靠"凡是 SANDBOX_ 开头就报错"通过——而那会让
    walking skeleton 的每一个子进程环境都炸掉。这份白名单是**唯一**的例外，
    加名字要动代码，不能靠命名习惯蒙混过关。
    """
    for name in sorted(AGENT_ENV_NAMES):
        monkeypatch.setenv(name, "x")

    Settings()  # 不抛异常就是通过


def test_a_misspelled_key_inside_the_env_file_refuses_to_start(monkeypatch, tmp_path) -> None:
    """**这个 bug 真正发生的地方是文件，不是环境变量。**

    `os.environ` 那一条测不到它：`.env` 里的变量从来没进过 `os.environ`。所以这条
    把 `ENV_FILE` 指到一份写着错名字的文件上，走完整条路。

    两处一起钉住：`ENV_FILE` 是 `model_config` 和校验器共用的**同一个**名字
    （下一条测试管另一半），所以这里换掉它就等于换掉了真正被读的那份。
    """
    from sandbox import config as config_module

    env = tmp_path / ".env"
    env.write_text("SANDBOX_DEEPSEEK_API_KEY=sk-whatever\n", encoding="utf-8")
    monkeypatch.setattr(config_module, "ENV_FILE", env)

    with pytest.raises(ValidationError, match="SANDBOX_DEEPSEEK_API_KEY"):
        Settings()


def test_the_env_file_is_looked_up_by_absolute_path() -> None:
    """从别的目录启动也要读得到 `.env`。

    相对路径按**当前工作目录**解析，从别处启动就读不到——key 配了、什么都没报、
    `upstream_ready` 是 False。这种"换个目录就没了"和写错变量名一样查不出来。
    """
    configured = Path(Settings.model_config["env_file"])

    assert configured.is_absolute()
    assert configured == REPO_ROOT / ".env"
    # 校验器读的必须是**同一个**名字。两边各写各的，总有一天会出现"按绝对路径
    # 读配置、按当前目录查错别字"——那一条查的是另一份文件。
    from sandbox import config as config_module

    assert configured == config_module.ENV_FILE


def test_env_file_keys_are_scanned_too(tmp_path) -> None:
    """`.env` 里的键名也要查——那正是绝大多数人配 key 的地方。

    只查 `os.environ` 的话，这个 bug 从它真正发生的地方（文件）旁边走过去了：
    文件里的变量从来没进过 `os.environ`。
    """
    env = tmp_path / ".env"
    env.write_text(
        "# 注释行\n"
        "\n"
        "SANDBOX_UPSTREAM_API_KEY=sk-x\n"
        "export SANDBOX_TYPO_NAME=1\n"
        "没有等号的一行\n",
        encoding="utf-8",
    )

    assert _names_in_env_file(env) == {"SANDBOX_UPSTREAM_API_KEY", "SANDBOX_TYPO_NAME"}


def test_an_absent_env_file_is_not_an_error(tmp_path) -> None:
    """没有 `.env` 是正常状态（CI、容器里都可能是），不是配置错误。"""
    assert _names_in_env_file(tmp_path / "nope") == set()
