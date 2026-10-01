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
| `script` | 每个会话的脚本、任务、快照点、记忆回滚位置、时间间隔 | 改了**测试本身** |
| `agent` | 版本号、目录、入口、额外环境变量 | 换了被测对象的构建 |
| `world` | 世界初始状态（订单备注、出口放行名单） | 换了这次测试面对的**世界** |
| `expect` | 期望列表、豁免规则、记忆判据、`expected_charges`、意外失败的口径 | 改了判卷标准 |

`script`、`world` 与 `expect` 变动尤其要认出来：它们是**测试定义**的一部分。测试改了
而结论变了，什么都不说明——把这种情况报成"行为回归"是本工具最不该犯的错。
"""

from __future__ import annotations

from typing import Any

from sandbox.scenario.spec import Scenario

#: 分节的名字与顺序。顺序固定，于是差异报告里各节的出现顺序也是固定的。
SECTIONS: tuple[str, ...] = ("prompt", "tools", "script", "agent", "world", "expect")

#: 分节给人看的名字。界面上不能直接印 `prompt` —— 用户改的东西叫"任务描述"，
#: 不叫"prompt 节"。
SECTION_LABELS: dict[str, str] = {
    "prompt": "任务描述与模型",
    "tools": "工具声明",
    "script": "会话脚本",
    "agent": "Agent 构建",
    "world": "世界初始状态",
    "expect": "期望与判卷标准",
}

#: 改动这几节意味着**测试本身**变了，而不是被测对象变了。见模块 docstring。
#:
#: `world` 在这里：换掉世界里那句话（订单备注）换的是**这次测试面对的世界**，
#: 前后两次的结论不可比。它和 `script` 是同一类东西——都是作者摆出来的布景。
TEST_SECTIONS: frozenset[str] = frozenset({"script", "world", "expect"})

#: 每一节的字段来源：场景模型里哪些顶层字段进这一节。
#:
#: 这是一张**声明**，不是从代码里推出来的——所以 `tests/test_identity.py` 里
#: 有一条断言盯着它：每个顶层字段要么在某个节里，要么在 `PROSE_FIELDS` 里。
#: 往 `Scenario` 上加一个字段而忘了接线，红的会是那一条。少了它，"配置没变但
#: 结论变了"这条最该被看到的信号会直接消失，而用户会去查一个没动过的文件。
SECTION_SOURCES: dict[str, tuple[str, ...]] = {
    "prompt": ("agent",),
    "tools": ("tools",),
    "script": ("sessions",),
    "agent": ("agent",),
    "world": ("world",),
    "expect": (
        "expect",
        "exemptions",
        "memory",
        "unexpected_failures",
        "expected_charges",
    ),
}

#: 不进指纹的顶层字段，以及为什么：**它们是散文**。
#:
#: `id` 是这份场景在报告里的名字，`description` 是写给人看的那段话。改掉它们，
#: 被测对象拿到的东西一个字节都没变，这次运行也一模一样——所以报"配置变了"
#: 是误报，而误报的代价是真正的行为差异被归到配置头上。
PROSE_FIELDS: frozenset[str] = frozenset({"id", "description"})

#: 会话这一节里不进取指纹的字段。
#:
#: 今天只有一个：**会话名**。它和 `id`、`description` 是同一类东西——报告里
#: 那句话的标签，不是这次运行的输入。会话之间按序号对齐，名字从来不参与任何
#: 判定（它只出现在错误信息里）。
#:
#: 留着它的代价是量得出来的：把某一份场景里的一行 `name:` 改个叫法，两次运行
#: 一个字都没变，而 `changed_sections` 会报出 `script`——于是回归报告说"改的是
#: 测试本身，两次运行比的不是同一套测试"。那是一句**盖住真信号的话**：用户正好
#: 在那次重跑里改过 Prompt 的话，他会先去看自己的改动，而报告已经在说"比不了"。
SESSION_FIELDS_NOT_IN_FINGERPRINT: tuple[str, ...] = ("name",)


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
        # 测试本身的步骤：脚本、快照点、记忆回滚位置、时间间隔。
        #
        # 会话名抹掉：它是报告里的标签，不是这次运行的输入。见
        # `SESSION_FIELDS_NOT_IN_FINGERPRINT`。
        "script": [
            {
                k: v
                for k, v in s.model_dump(mode="json").items()
                if k not in SESSION_FIELDS_NOT_IN_FINGERPRINT
            }
            for s in scenario.sessions
        ],
        # 被测对象的构建标识与启动方式。
        "agent": {
            "version": scenario.agent.version,
            "directory": scenario.agent.directory,
            "entry": scenario.agent.entry,
            "env": scenario.agent.env,
        },
        # Agent 醒来时世界里有什么：订单上的自由文本备注（它是个注入面），以及
        # 这次运行的出口放行名单。
        #
        # 名单必须在这里，理由和备注同一条，但更硬：换了它等于换了 Agent 面对
        # 的网络。两份只差名单的场景会得出相反的结论（钱扣了 / 钱被挡在边界外），
        # 而名单不进指纹的话，差异报告会把这读成"配置没变，但行为变了"——最该
        # 被看见的那个信号，会把人引去查一个谁都没动过的文件。
        #
        # `None`（没声明出口策略）与"声明了空名单"在这里也是两个值，和场景侧
        # 那条规矩一致：空表和「没给」是两件事。
        "world": {
            "order_notes": dict(scenario.world.order_notes),
            "egress": (
                None
                if scenario.world.egress is None
                else scenario.world.egress.model_dump(mode="json")
            ),
        },
        # 判卷标准。豁免规则**也在这一节**：一条豁免改的是"什么算失败"，
        # 不是被测对象的行为。它落在这里，加规则的那次运行就会被判成
        # "测试变了、比不了"，而不是"失败少了几个"——后者是把静音读成修复。
        "expect": {
            "expect": [e.model_dump(mode="json") for e in scenario.expect],
            "exemptions": [e.model_dump(mode="json") for e in scenario.exemptions],
            # 记忆判据也是判卷标准。**声明本身就是观测的过滤器**：去掉一条
            # must_hold，原本报 fail 的那条会变成 inconclusive——差异报告里
            # 看起来就像"失败少了"。放进这一节，两次运行才被判成不可比。
            "memory": scenario.memory.model_dump(mode="json"),
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
