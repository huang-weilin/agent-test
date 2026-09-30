"""场景规格：把"一次测试"写成数据，而不是写成 Python。

**为什么这件事重要。** 在此之前，加一个用例要改 `scripts/run_walking_skeleton.py`
里的字典，跨会话用例更是只能写成 pytest。那意味着两件事：用例的**作者**必须是
能改框架代码的人；用例的**生命周期**和框架代码绑在一起。

场景文件把这个边界挪开了：场景是数据，跑场景是引擎。作者写 YAML，引擎负责起服务、
拉起 Agent、拍快照、跑检测、比对预期。

设计上有一条是刻意从 `mock/toolspec.py` 继承来的：**`is_write` 不给默认值。**
写属性决定了幂等与重复扣款检测的触发范围，而"忘了声明"如果有个 false 默认值，
表现就是某些写操作悄悄不设防——从报告上完全看不出来。宁可让场景加载失败。

同一条理由现在也适用于写操作的 `may_run_unattended`：它是"主动行为审计"
（功能需求 3.6 / 场景 6）里"未经授权"那半句话的全部依据，而漏声明的后果同样
是安静地不设防。差别只在于**它在什么时候必填**——条件是"场景里真的有一段
无人值守的会话"，见 `Scenario` 上那条跨字段校验。没有自主会话时它一次都不会
被读到，逼人写一个不起作用的声明只会制造噪音。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from sandbox.db.models import DetectorVerdict
from sandbox.detectors.base import Exemption, ExemptionSet
from sandbox.detectors.memory import ForbiddenText, MemoryRules, MustHold
from sandbox.toolschema import check_schema


class ToolDecl(BaseModel):
    """场景里声明的工具。这是写/读属性的**单一来源**。"""

    model_config = {"extra": "forbid"}

    name: str
    description: str
    #: 参数的 JSON Schema。**边界上会拿它逐条核对每一次调用**（见 toolschema.py）。
    #:
    #: 不只是一份给 Agent 看的说明书：真实目标系统会因为少一个必填字段回 400，
    #: 而一个不核对参数的 mock 会照单全收、照单扣款——那笔钱在报告上和一笔正常
    #: 的扣款长得一模一样，`expected_charges` 这个地面真值本身就被污染了。
    #:
    #: 空 `{}` = 这个工具没声明参数 = 边界上不校验它。支持的写法是**白名单**，
    #: 不认识的写法在加载场景时直接报错（不这么做的话，写了也没人执行）。
    parameters: dict[str, Any] = Field(default_factory=dict)
    # 无默认值，必填。见模块 docstring。
    is_write: bool
    # 携带幂等键的参数名。None = 该工具没有幂等机制。
    idempotency_param: str | None = None
    # 工具**在目标系统侧**是否真的实现了幂等。与上一项分开：
    # 声明了参数却忘了实现，正是要测出来的缺陷。
    enforces_idempotency: bool = False

    #: **无人值守时，允许它自己动手吗。有一个自主会话时必填。**
    #:
    #: 这是主动行为审计里"未经授权"四个字的唯一依据。边界上看到的每一次调用
    #: 都只是"它调了个工具"——没有一份事先写下的授权，"越权"就没有对照物，
    #: 一次半夜自动发出的消息看起来和一次正常响应一模一样。
    #:
    #: **不给默认值**，而且两个方向都有害：默认 `True` 的话，忘声明的后果是
    #: 这个工具的越权行为永远不会被发现；默认 `False` 的话，只想跑个普通场景
    #: 的人会被迫面对一条假的越权告警，于是这条告警很快就变成"反正它老报"。
    #:
    #: "必填"的条件是**场景里真的有一段无人值守的会话**，见 `Scenario` 上的
    #: 校验器。没有自主会话时这个字段一次都不会被读到，逼人写一个不起作用的
    #: 声明只会制造噪音；有自主会话而没写，那才是真的漏——审计会静默地失效。
    #:
    #: 只读工具不允许声明它：读不改变世界，不是"主动行为"。见下面的校验器。
    may_run_unattended: bool | None = None

    @field_validator("name")
    @classmethod
    def _name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("工具名不能为空")
        return v

    @model_validator(mode="after")
    def _parameters_are_checkable(self) -> ToolDecl:
        """声明的参数 schema 必须落在我们**真的会逐条核对**的那个子集里。

        支持的写法见 `sandbox/toolschema.py`。不认识的写法在这里让场景加载失败，
        而不是在运行时被默默忽略——"作者写了 `oneOf`、边界不认识、于是这一类参数
        从来没被检查过"是那种报告上完全看不出来的失效。

        **空 `parameters` 是合法的**，它表示"这个工具没声明参数"，边界上也就
        没什么可对照的。这不是漏检，是如实的：没有声明就没有标准。
        """
        check_schema(self.parameters, tool=self.name)
        return self

    @model_validator(mode="after")
    def _unattended_only_on_writes(self) -> ToolDecl:
        if not self.is_write and self.may_run_unattended is not None:
            # 允许它只会让人以为自己限制了什么。读操作不改变世界，主动行为审计
            # 根本不看它——一个不起作用的字段比没有这个字段更坏。
            raise ValueError(
                f"只读工具 {self.name!r} 不该声明 may_run_unattended："
                f"读操作不改变世界，不构成主动行为，审计不判定它。"
                f"要限制它，请用出口代理的放行名单（ADR 003）。"
            )
        return self


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

    **跨会话测试的全部表达力都在这里**，字段分别对应四件事：

      `restore_from`     —— 把记忆回滚到哪张快照。这是"制造遗忘"的旋钮，
                            也是跨会话归因的依据：回滚到哪儿决定了 Agent
                            本可以知道什么。
      `snapshot_before`  —— 会话开始前拍一张，命名以便后续回滚到这个时刻。
      `snapshot_after`   —— 会话结束后拍一张。
      `after_gap_s`      —— 距上一段会话过去了多久（虚拟秒）。见下面。
      `autonomous`       —— 这一段没人看着（还是有人看着）。见下面。

    不写 `restore_from` 就是自然延续，即记得此前一切。
    """

    model_config = {"extra": "forbid"}

    name: str
    # 脚本化上游的步骤。见 gateway/stub.py。
    script: list[dict[str, Any]] = Field(default_factory=list)
    restore_from: str | None = None
    snapshot_before: str | None = None
    snapshot_after: str | None = None

    #: 距上一段会话过去了多久——**长时运行的唯一旋钮**。
    #:
    #: 单位是秒，但语义是"世界过了这么久"，不是"我们等了这么久"。引擎不会真的
    #: 睡：它把进程里的虚拟时钟往前拨（`sandbox/clock.py`），于是这一段间隔里
    #: 目标系统上该变的都变了，而真实耗时是零。
    #:
    #: 会话 0 写它也有意义：那表示"世界在我们开口之前就已经过了这么久"。
    after_gap_s: int = 0

    #: 这一段会话里用户说的话。None = 沿用 `agent.task`。
    #:
    #: 单次运行里两者没有区别，长时运行里区别就是全部：**隔了一天，用户回来说的
    #: 是另一件事**。每一段都重复同一句"帮我把订单 A1 处理掉"的话，读报告的人
    #: 会以为 Agent 是在被反复要求做同一件事——那和"它隔了一晚自己又动了一次"
    #: 是完全不同的两种行为，也是最要命的一种误读。
    task: str | None = None

    #: 这一段会话是**无人值守**的自主运行吗。
    #:
    #: 默认 False —— 用户在。这是绝大多数会话的形态，也是安全的一侧。
    #:
    #: 它决定的不是"测什么"，而是**同样一次写操作该怎么读**：用户在时，Agent
    #: 动手是**响应**（他要求做的）；无人值守时，同样的调用是**主动行为**
    #: （它自己决定做的）。同一个 `create_payment`，前者叫正常，后者要拿授权
    #: 去对照。指标完全相同、结论相反，所以只能由场景显式说出来，不能猜。
    #:
    #: 典型写法是配合 `after_gap_s` 造出"夜里没人，它被定时器叫醒"的那一段。
    #: 反过来也成立：一个声明了自主却不带间隔的会话是在测"没人说话时它做什么"，
    #: 那是另一件事，同样合法。
    autonomous: bool = False

    @field_validator("after_gap_s")
    @classmethod
    def _gap_not_negative(cls, v: int) -> int:
        # 负数不是"回到过去"，是场景写错了。放行的话时钟会往回拨，于是上一段
        # 会话里已经逾期的东西重新变得不逾期——检测器看到的是一条物理上不可能
        # 的轨迹，而它照样会给出结论。
        if v < 0:
            raise ValueError(f"after_gap_s 不能为负（收到 {v}）：时间只往前走")
        return v


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


class ExemptionDecl(BaseModel):
    """一条豁免规则。把某一类结论降级成 pass，同时在报告里留痕。

    形状和 `sandbox.detectors.base.Exemption` 一一对应，但**故意分成两个类**：
    这一层管"用户写下来的东西长什么样"，那一层管"怎么匹配"。合成一个的话，
    检测器内部想加个字段就得动用户能写的 schema。

    理由（`reason`）是必填的，而且是唯一必填的——没有理由的豁免在报告里无法复核，
    等于永久静音。写操作的豁免还要额外写 `allow_write: true`，那是刻意设的第二道门：
    让"我在静音一次重复扣款告警"变成一个必须写下来的动作。

    `match` 的键在加载时校验（见 `loader._check_references`），写错的键会让规则
    永远匹配不上。
    """

    model_config = {"extra": "forbid"}

    detector: str
    tool: str | None = None
    reason: str
    match: dict[str, Any] = Field(default_factory=dict)
    allow_write: bool = False


class MustHoldDecl(BaseModel):
    """一条"这条记忆必须保持"的判据。

    `path` 以 `/` 结尾表示整个目录前缀。判的是**内容摘要**，不是"文件在不在"——
    空文件也算存在，但它的内容为空，两者在报告里要说得出区别。
    """

    model_config = {"extra": "forbid"}

    path: str
    reason: str


class ForbiddenTextDecl(BaseModel):
    """一段不允许出现在记忆里的文本。

    这是记忆中毒检测**现在能做的那一半**：受控实验里注入进去的那句话是我们自己
    写的，所以写得出来。它抓不到没声明的注入——那不是这个类的实现问题，是这条路
    的边界（见 docs/detectors.md 第九节）。
    """

    model_config = {"extra": "forbid"}

    text: str
    reason: str


class MemoryDecl(BaseModel):
    """场景对记忆的判据。**空 = 不判，只观测。**

    和 `exemptions` 一样属于"判卷标准"（进配置指纹的 `expect` 节），不属于被测对象：
    改了它，前后两次运行就不再可比"失败少了几个"。
    """

    model_config = {"extra": "forbid"}

    must_hold: list[MustHoldDecl] = Field(default_factory=list)
    forbidden: list[ForbiddenTextDecl] = Field(default_factory=list)


class WorldDecl(BaseModel):
    """Agent 醒来时，目标系统里长什么样。

    今天只有一样东西：订单上的自由文本备注。它是**注入面**——上游系统里那些
    第三方写得了的字段（订单备注、工单正文、邮件内容）。记忆中毒的受控实验要
    的就是这句话：世界里有一句话，问题是 Agent 会不会把它带进长效记忆。

    **它是世界，不是判据**：备注里写了什么与"什么算失败"无关（那是 `memory`
    段的事）。两条场景把同一句话一个写成注入、一个写成正常备注，跑出来的差别
    必须来自这里——只改判据不叫受控实验，那只是换了把尺子。
    """

    model_config = {"extra": "forbid"}

    #: 订单号 → 那句话。订单号必须是目标系统里真实存在的样例订单，加载时校验。
    order_notes: dict[str, str] = Field(default_factory=dict)


class Scenario(BaseModel):
    model_config = {"extra": "forbid"}

    id: str
    description: str = ""
    agent: AgentSpec
    tools: list[ToolDecl]
    sessions: list[SessionSpec]
    expect: list[Expectation] = Field(default_factory=list)

    # 误报控制：把某类结论降级成 pass 并留痕。
    #
    # 和"期望"分开是刻意的：期望是"我要求它报这个"，豁免是"我知道它报了但它没事"。
    # 前者是判据，后者是判据的例外，读报告的人需要分得开这两句话。
    #
    # 这一节进配置指纹的"判卷标准"里（见 `identity.sections`）：加一条豁免之后，
    # 前后两次运行就不该再比"失败少了几个"——少掉的那些是被规则静音的，不是修好的。
    exemptions: list[ExemptionDecl] = Field(default_factory=list)

    # 记忆族的判据（must_hold / forbidden）。**不写就是"这一族不判"**，不是
    # "这一族没问题"——检测器在没判据时给的是 inconclusive，见 detectors/memory.py。
    #
    # 观测本身（会话边界上记忆长什么样）不需要声明，引擎每段会话都自动看一眼：
    # 声明与否只决定**判不判**，不决定看不看。
    memory: MemoryDecl = Field(default_factory=MemoryDecl)

    # Agent 醒来时，目标系统里长什么样（今天只有订单备注）。
    #
    # 它和 `memory` 是一对：**这里写下的是世界里的那句话，那里写下的是"这句话
    # 不许出现在记忆里"**。分开是刻意的——把注入文本写进判据段的话，"世界里有
    # 这句话"和"我们要求它不在记忆里"就成了一件事，而对照实验要的正是两者的
    # 差别：同一句话，一个场景里它进了记忆（fail），另一个场景里它压根不在世界里
    # （pass）。
    #
    # 进配置指纹的 `world` 节：改了它等于换了这次测试面对的世界，前后两次
    # 运行不该拿来比"失败少了几个"。
    world: WorldDecl = Field(default_factory=WorldDecl)

    # 出现了没被 expect 匹配上的 fail 结论时怎么办。
    # 默认 fail：这是测试工具，"没预料到的失败"必须让人知道。
    # 允许改成 warn 的唯一理由是场景本身还在探索期。
    unexpected_failures: Literal["fail", "warn", "allow"] = "fail"

    # 目标系统侧**真实扣款的笔数**。None = 不检查。
    #
    # 这是地面真值，和检测结论是两回事，所以单独断言。检测器完全可能报出
    # 漂亮的 pass，而钱其实被扣了两次——那正是这个工具要抓的失效方式。
    # 反过来，"记忆守卫拦住了重复支付"这种正样本也只能靠它证明：边界上
    # 什么都看不到，没有扣款流水就没有证据。
    expected_charges: int | None = None

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

    @model_validator(mode="after")
    def _autonomous_needs_authorization(self) -> Scenario:
        """有无人值守的会话时，每个写操作都必须交代自己许不许无人值守运行。

        **这是一条跨字段的校验，单看任何一个工具都发现不了。** 它是主动行为
        审计的**防呆**：审计靠 `may_run_unattended` 区分"它自己动手"和"它被允许
        自己动手"，而漏声明的表现是某个工具的越权行为**永远不会被报出来**——
        报告上一切正常。宁可让场景加载失败。

        没有自主会话时不做这个要求：那个字段一次都不会被读到，逼人写一个不起作用
        的声明只会制造噪音，而噪音会让真正的告警被忽略。
        """
        if not any(s.autonomous for s in self.sessions):
            return self
        missing = [t.name for t in self.tools if t.is_write and t.may_run_unattended is None]
        if missing:
            raise ValueError(
                f"场景里有无人值守的会话，但写操作 {sorted(missing)} 没有声明 "
                f"may_run_unattended（true / false）。\n"
                f"它回答的是「Agent 半夜自己调它，算不算越权」。不声明的话，"
                f"这些工具的越权行为不会被主动行为审计发现，而报告上完全看不出来。"
            )
        return self

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

    # --- 长时运行（功能需求 3.6 / 场景 6）-----------------------------------

    def is_long_run(self) -> bool:
        """这个场景是不是在模拟一段长时间。

        判据是"声明了间隔"，不是"会话多"。三个会话也可以全都挤在同一秒里——
        那测的是跨会话，不是长时运行，两者要的东西不一样：前者问"它记不记得"
        （靠记忆快照），后者问"这个世界过了几天之后它还做了什么"（靠虚拟时钟）。
        报告里分开报，因为读的人要采取的行动不同。
        """
        return any(s.after_gap_s > 0 for s in self.sessions)

    def virtual_span_s(self) -> int:
        """从世界起点到最后一个会话开始，一共过去了多少虚拟秒。"""
        return sum(s.after_gap_s for s in self.sessions)

    def offset_before(self, index: int) -> int:
        """第 `index` 段会话开始时，世界应该已经走了多少秒。

        这是**声明值**，引擎启动会话时拿它和时钟实际观察到的偏移对账。两者不等
        说明加速器没按场景说的做，而那种情况下这份报告里的每一句"过了三天之后"
        都是假的——所以它进前提判定，见 `runner.SessionOutcome.premise_held`。
        """
        return sum(s.after_gap_s for s in self.sessions[: index + 1])

    # --- 主动行为审计（功能需求 3.6 / 场景 6）-------------------------------

    def autonomous_ordinals(self) -> set[int]:
        """被声明为无人值守的会话序号。

        **声明而不是观测**，所以它来自场景，而检测器只吃 `list[CallView]`。
        这个区别是必须的：一个自主会话如果**一次调用都没产生**，边界上就没有
        它的任何痕迹，只吃调用的检测器看不见它存在过。而"夜里它什么都没做"
        和"夜里它做了但都是被允许的"是两件事，报告必须分得开——前者是没测到。
        """
        return {i for i, s in enumerate(self.sessions) if s.autonomous}

    def unattended_allowlist(self) -> set[str]:
        """被授权在无人值守时运行的工具。审计的另一半对照物。"""
        return {t.name for t in self.tools if t.may_run_unattended}

    # --- 误报控制（见 docs/detectors.md 第二节）----------------------------

    def exemption_set(self) -> ExemptionSet:
        """把场景里的豁免规则转成检测器吃的形状。

        豁免和 `unattended_allowlist` 是同一类东西：**声明**。声明留在场景里，
        检测器只看到接口——所以这里是一次转换，不是一次判定。

        转换放在 spec 上而不是 runner 里，是为了让**校验**有个统一的落点：
        `loader._check_references` 会先调一遍它，用户 YAML 里的错在加载时就报出来，
        而不是等跑到一半才炸。
        """
        return ExemptionSet(
            [
                Exemption(
                    detector=e.detector,
                    tool=e.tool,
                    reason=e.reason,
                    match=dict(e.match),
                    allow_write=e.allow_write,
                )
                for e in self.exemptions
            ]
        )

    # --- 记忆族的判据（见 docs/detectors.md 第九节）-------------------------

    def memory_rules(self) -> MemoryRules:
        """场景里的记忆判据 → 检测器吃的形状。

        和 `exemption_set` 同一个理由放在这里：**校验要有统一的落点**，
        `loader._check_references` 先调一遍，用户 YAML 里的错在加载时就报出来。
        """
        return MemoryRules(
            must_hold=tuple(
                MustHold(path=r.path, reason=r.reason) for r in self.memory.must_hold
            ),
            forbidden=tuple(
                ForbiddenText(text=r.text, reason=r.reason) for r in self.memory.forbidden
            ),
        )
