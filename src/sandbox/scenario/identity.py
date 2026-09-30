"""把一次运行的**配置**压成分节的指纹。

需求要求"差异项可定位到具体的 Prompt 或工具改动"——改完 Prompt 重跑，结论
变了，得能说出**变的是配置还是行为**。没有这个指纹，两次运行的差异就只能靠
人回忆上次改了什么。

**为什么不是把场景文件的字节哈希一遍。** 那样改一个缩进、调一下 YAML 键的
顺序，指纹就变了，于是"配置没变但结果变了"会被误报成"配置变了"——而后者
恰恰是我们要抓的那个信号。所以按 pydantic 模型**语义化**地取字段
（`model_dump` 已经把键序固定了），而不是看文件长什么样。

**为什么分节而不是一个总数。** 一个总数只能回答"变了没有"。真正要回答的是
"改了哪一块"：Prompt 改了、工具声明改了、还是用例的期望改了？这三者指向的
修复动作完全不同。所以每一节单独算，差异报告里就能直说"改的是 tools"。

分节的口径是**用户会改的东西**，不是代码结构：

| 节 | 覆盖 | 改了意味着 |
|---|---|---|
| `prompt` | `agent.task`、`agent.model` | 改了被测对象的输入 |
| `tools` | 工具声明（含 `is_write`、幂等参数） | 改了它手里的工具 |
| `script` | 每个会话的脚本、快照点、记忆回滚位置 | 改了**测试本身** |
| `agent` | 版本号、目录、入口、额外环境变量 | 换了被测对象的构建 |
| `expect` | 期望列表、`expected_charges`、意外失败的口径 | 改了判卷标准 |

`script` 与 `expect` 变动尤其要认出来：它们是**测试定义**的一部分。测试改了
而结论变了，什么都不说明——把这种情况报成"行为回归"是本工具最不该犯的错。
"""

from __future__ import annotations

from typing import Any

from sandbox.scenario.spec import Scenario

#: 分节的名字与顺序。顺序固定，于是差异报告里各节的出现顺序也是固定的。
SECTIONS: tuple[str, ...] = ("prompt", "tools", "script", "agent", "expect")

#: 分节给人看的名字。界面上不能直接印 `prompt` —— 用户改的东西叫"任务描述"，
#: 不叫"prompt 节"。
SECTION_LABELS: dict[str, str] = {
    "prompt": "任务描述与模型",
    "tools": "工具声明",
    "script": "会话脚本",
    "agent": "Agent 构建",
    "expect": "期望与判卷标准",
}

#: 改动这几节意味着**测试本身**变了，而不是被测对象变了。见模块 docstring。
TEST_SECTIONS: frozenset[str] = frozenset({"script", "expect"})


def sections(scenario: Scenario) -> dict[str, Any]:
    """把场景切成"用户会改的那几块"。

    返回的是**原始值**而不是哈希：差异报告要说清改了什么，只有哈希就只能说
    "变了"。这些值随运行一起落库，所以场景文件事后被编辑也不影响追溯——
    和 `report_json` 是同一个理由。
    """
    return {
        # 被测对象拿到的东西：任务描述（通常就是"Prompt"）和模型。
        "prompt": {
            "task": scenario.agent.task,
            "model": scenario.agent.model,
        },
        # 它手里的工具。`is_write` 和幂等参数都在里面——它们直接决定
        # 幂等/重复扣款检测的触发范围，改一个等于换了检测的覆盖面。
        "tools": [
            t.model_dump(mode="json") for t in scenario.tools
        ],
        # 测试本身的步骤：脚本、快照点、记忆回滚位置。
        "script": [s.model_dump(mode="json") for s in scenario.sessions],
        # 被测对象的构建标识与启动方式。
        "agent": {
            "version": scenario.agent.version,
            "directory": scenario.agent.directory,
            "entry": scenario.agent.entry,
            "env": scenario.agent.env,
        },
        # 判卷标准。
        "expect": {
            "expect": [e.model_dump(mode="json") for e in scenario.expect],
            "unexpected_failures": scenario.unexpected_failures,
            "expected_charges": scenario.expected_charges,
        },
    }


def changed_sections(
    before: dict[str, Any] | None, after: dict[str, Any] | None
) -> list[str]:
    """两节配置之间，**哪几块**变了。

    `None` 和**空字典**都表示没记录（本字段存在之前跑的运行，或者一个没带
    配置就开出来的运行）。那时候**不能假设没变**，也不能报"每一节都变了"——
    后者读起来像是"有人把五块全改了"。返回全部节名，调用方据此判定
    "配置可比性未知"，那是第三种答案。
    """
    if not before or not after:
        return list(SECTIONS)
    return [name for name in SECTIONS if before.get(name) != after.get(name)]


def environment(scenario: Scenario) -> dict[str, Any]:
    """钉住不确定性的那些环境事实。

    `TestRun.env_json` 的注释说它是"回归可比性的前提"，但在这之前它一直是
    `{}`——而空的环境记录会让"两次运行环境相同"变成一句没有依据的话。

    这里只记**确实知道**的东西，不编造。比如 `llm_upstream: scripted` 是真的：
    演示跑的是脚本化上游（`gateway/stub.py`），不是真实模型，所以模型自身的
    采样不确定性根本不存在。真实模型接入后要在这里补 seed / temperature。
    """
    return {
        "model": scenario.agent.model,
        # Agent 进程自己的额外环境变量——它会影响行为，属于环境的一部分。
        "agent_env": scenario.agent.env,
        # 这次运行的上游是脚本化还是真实模型。**这句话必须准确**：
        # 脚本化上游意味着"模型行为"是被我们写死的，结论不能外推到真实模型。
        "llm_upstream": "scripted" if scenario.agent.model == "stub-model" else "live",
    }
