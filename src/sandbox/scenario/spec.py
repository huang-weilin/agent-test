"""场景规格：把"一次测试"写成数据，而不是写成 Python。

**为什么这件事重要。** 在此之前，加一个用例要改 `scripts/run_walking_skeleton.py`
里的字典，跨会话用例更是只能写成 pytest。那意味着两件事：用例的**作者**必须是
能改框架代码的人；用例的**生命周期**和框架代码绑在一起。

场景文件把这个边界挪开了：场景是数据，跑场景是引擎。作者写 YAML，引擎负责起服务、
拉起 Agent、拍快照、跑检测、比对预期。

设计上有一条是刻意从 `mock/toolspec.py` 继承来的：**`is_write` 不给默认值。**
写属性决定了幂等与重复扣款检测的触发范围，而"忘了声明"如果有个 false 默认值，
表现就是某些写操作悄悄不设防——从报告上完全看不出来。宁可让场景加载失败。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from sandbox.db.models import DetectorVerdict


class ToolDecl(BaseModel):
    """场景里声明的工具。这是写/读属性的**单一来源**。"""

    model_config = {"extra": "forbid"}

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    # 无默认值，必填。见模块 docstring。
    is_write: bool
    # 携带幂等键的参数名。None = 该工具没有幂等机制。
    idempotency_param: str | None = None
    # 工具**在目标系统侧**是否真的实现了幂等。与上一项分开：
    # 声明了参数却忘了实现，正是要测出来的缺陷。
    enforces_idempotency: bool = False

    @field_validator("name")
    @classmethod
    def _name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("工具名不能为空")
        return v


class AgentSpec(BaseModel):
    """被测 Agent 怎么启动。

    注意这里没有"接入方式"之类的字段——接入就是换 base_url 和 api_key，
    由引擎注入环境变量，场景不需要也不应该知道沙箱的内部地址。
    """

    model_config = {"extra": "forbid"}

    version: str
    task: str
    model: str = "stub-model"
    # agents/ 下的目录名与入口文件
    directory: str = "react_demo"
    entry: str = "agent.py"
    # 额外环境变量。用于给不同 Agent 传各自的配置项。
    env: dict[str, str] = Field(default_factory=dict)


class SessionSpec(BaseModel):
    """会话序列里的一段。

    **跨会话测试的全部表达力都在这里**，三个字段分别对应三件事：

      `restore_from`     —— 把记忆回滚到哪张快照。这是"制造遗忘"的旋钮，
                            也是跨会话归因的依据：回滚到哪儿决定了 Agent
                            本可以知道什么。
      `snapshot_before`  —— 会话开始前拍一张，命名以便后续回滚到这个时刻。
      `snapshot_after`   —— 会话结束后拍一张。

    不写 `restore_from` 就是自然延续，即记得此前一切。
    """

    model_config = {"extra": "forbid"}

    name: str
    # 脚本化上游的步骤。见 gateway/stub.py。
    script: list[dict[str, Any]] = Field(default_factory=list)
    restore_from: str | None = None
    snapshot_before: str | None = None
    snapshot_after: str | None = None


class Expectation(BaseModel):
    """一条预期。

    刻意用"检测器 + 结论 + 可选的 reason_code"，而不是"期望输出全文"：
    后者一改措辞就红，于是没人愿意维护它。reason_code 是稳定的机器可读标识，
    文案则可以随便改。
    """

    model_config = {"extra": "forbid"}

    detector: str
    verdict: DetectorVerdict
    reason_code: str | None = None
    note: str = ""


class Scenario(BaseModel):
    model_config = {"extra": "forbid"}

    id: str
    description: str = ""
    agent: AgentSpec
    tools: list[ToolDecl]
    sessions: list[SessionSpec]
    expect: list[Expectation] = Field(default_factory=list)

    # 出现了没被 expect 匹配上的 fail 结论时怎么办。
    # 默认 fail：这是测试工具，"没预料到的失败"必须让人知道。
    # 允许改成 warn 的唯一理由是场景本身还在探索期。
    unexpected_failures: Literal["fail", "warn", "allow"] = "fail"

    @field_validator("sessions")
    @classmethod
    def _at_least_one_session(cls, v: list[SessionSpec]) -> list[SessionSpec]:
        if not v:
            raise ValueError("场景至少要有一个会话")
        return v

    @field_validator("tools")
    @classmethod
    def _tool_names_unique(cls, v: list[ToolDecl]) -> list[ToolDecl]:
        seen = [t.name for t in v]
        dupes = {n for n in seen if seen.count(n) > 1}
        if dupes:
            raise ValueError(f"工具名重复：{sorted(dupes)}")
        return v

    def tool_index(self) -> dict[str, ToolDecl]:
        return {t.name: t for t in self.tools}

    def declared_writes(self) -> set[str]:
        return {t.name for t in self.tools if t.is_write}

    def idempotency_params(self) -> dict[str, str]:
        return {t.name: t.idempotency_param for t in self.tools if t.idempotency_param}

    def snapshot_names(self) -> set[str]:
        """场景里会被拍下来的快照名。

        `restore_from` 引用的名字必须在这张表里，否则场景就是错的——
        引用了不存在的快照会静默地变成"没有回滚"，于是归因整个反过来。
        加载时校验（见 loader.py）。
        """
        names: set[str] = set()
        for session in self.sessions:
            names.update(n for n in (session.snapshot_before, session.snapshot_after) if n)
        return names
