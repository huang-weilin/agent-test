"""场景规格：把"一次测试"写成数据，而不是写成 Python。

**为什么这件事重要。** 在此之前，加一个用例要改一个演示脚本里的字典（那个脚本
已经删了，但这是当初的动机），跨会话用例更是只能写成 pytest。那意味着两件事：
用例的**作者**必须是能改框架代码的人；用例的**生命周期**和框架代码绑在一起。

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
from sandbox.detectors.idempotency import checked_identity_fields
from sandbox.detectors.loop import checked_field_weights
from sandbox.detectors.memory import ForbiddenText, MemoryRules, MustHold
from sandbox.proxy.policy import (
    AGENT_CHOOSES_HOST,
    GENERIC_EGRESS_BODY_FIELD,
    GENERIC_EGRESS_URL_FIELD,
    AllowRule,
    EgressPolicy,
    checked_host,
    checked_mock_path,
    generic_egress_path,
    policy_with_rules,
)
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

    #: **这个工具的"主键"是哪几个字段。** 声明了它，参数逐字不同但主键相同的
    #: 几次写操作就会被当成"同一个操作"报出来（`IdempotencyDetector`）。
    #:
    #: 为什么需要它：指纹是按**整个参数表**逐字算的，于是"同一件事、换个付款方式
    #: 再订一次"永远配不上对。量过（642 条真实轨迹、916 次写操作，
    #: `../_traces/measure_same_op.py`）：按"同工具"放宽会多报 145 组、按"只差
    #: 一个字段"多报 90 组，逐条看大多是误报；按主键字段放宽只剩 56 组，而
    #: **那张主键表只有懂业务的人给得出来**——`payment_methods` 变了不算另一件事、
    #: `order_id` 变了就算，这条区别产品推不出来。
    #:
    #: **不声明就一个字都不变**：那些量出来的误报形状照样不会被报出来。
    #: 声明了主键、调用却没带那几个字段的，会单独报一条 `inconclusive`——
    #: 不折进"判过了、没问题"里，见 `detectors/idempotency.py` 的 `Ungroupable`。
    #:
    #: 与 `field_weights` 不是同一个旋钮：那个是**相似度**里的相对权重（模糊，
    #: 给循环检测用），这个是**硬分组**的键（全等，给幂等检测用）。同一件事在
    #: 两处的判法不同，所以不是一个字段。
    #:
    #: **它不过线**（不进运行注册载荷）：运行当中没有消费者——看门狗只跑循环
    #: 检测那一族。过线的话就成了一条"装上了但没接线"的声明，而那种声明比
    #: 没有更坏（它会让人以为运行当中也按它判了）。
    identity_fields: list[str] = Field(default_factory=list)

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

    #: **这个工具的参数里，哪个字段不是它的"身份"。** 字段名 → 权重（越小越不重要）。
    #:
    #: 近似层要回答的是"同一个操作被反复做"，而"同一个操作"是参数里**稳定的那部分**
    #: 说了算。翻页是最好的例子：`list_orders(status=unpaid, page=N)` 里 page 每一页
    #: 都不一样，但它不是这个操作的身份——`status` 才是。
    #:
    #: **不声明的话，检测器只能拿数值的相对差去猜**，而这个猜法有个实测出来的
    #: 后果：页码从 1 涨到 10，第 1 次和第 6 次的相似度只剩 0.72，于是**一串
    #: 连续翻页被切成好几个簇**，每个簇都够不着告警条数（8）——报告里于是出现
    #: 两条"判为正常遍历"，而它其实一次都没停。字段越多越容易成簇，这也解释了
    #: `tests/corpus/loop.yaml` 里那条"刻意带五个字段"的注释。
    #:
    #: 权重是**相对**的（默认 1.0），所以只有降低它才有意义：把 page 压到 0.1，
    #: 相似度就由那些稳定的字段撑着，"这一串是同一个操作"才立得住。
    #:
    #: 它进配置指纹的 `tools` 节：改了权重等于改了"什么算重复"——那是判卷标准
    #: 的一部分，两次运行不该拿来比"失败少了几个"。见 `scenario/identity.py`。
    field_weights: dict[str, float] = Field(default_factory=dict)

    #: **这个工具在真实环境里出到哪台主机。** 声明它 = 它的出口要过放行名单。
    #:
    #: 沙箱里每一次调用实际连的都是自己的 mock（传输层只有一个地址），所以
    #: **不声明的话，出口策略根本没有分辨力**：所有工具在名单上都是同一台主机，
    #: "允许读订单、不允许扣款"这种再普通不过的部署策略表达不出来。
    #:
    #: 写的是**真实目标**（`payments.example.com`），不是沙箱的地址——和
    #: `is_write`、`enforces_idempotency` 同一种东西：关于真实目标系统的声明，
    #: 沙箱拿它做判断，Mock 拿它当替身。策略按它判，连接照旧发给 Mock。
    #:
    #: **要不要写，由场景有没有声明出口名单决定**（见下面那条跨字段校验）：
    #: 不声明名单时这个字段一次都不会被读到，写了也只会让人以为限制了什么。
    #:
    #: 它还可以写**保留值**（`agent`）：这个工具的目标由被测对象在调用时指定。
    #: 那是唯一一类"目标不是说好的那台"的工具（通用 HTTP 工具、抓取类 MCP 工具
    #: 都是这样），它的出口只能按**这次调用实际要连的那台主机**判，声明固定主机
    #: 就成了编造。它和写主机的工具一样要交代自己，只是交代的内容不同——
    #: 一个说"我出到哪"，一个说"我出到哪由它定"。
    egress_host: str | None = None

    @model_validator(mode="after")
    def _egress_host_is_a_host(self) -> ToolDecl:
        """写了就得是个主机名——或者是那个保留值。

        主机名的规则本身在 `checked_host` 里，与场景那边的名单共用。两边只留
        一份实现：一份写错了的规则**不会报错，只会永远不命中**——报告上看起来
        像"这次运行没放行它"，而真相是那行字压根没参与过判定。

        保留值在这里先接住（`checked_host` 见到它会报错，那是给名单用的：
        名单里写它只会得到一条永不命中的规则）。存回去的是规整过的小写值，
        下面所有判断都按字面比。
        """
        if self.egress_host is not None:
            if self.egress_host.strip().lower() == AGENT_CHOOSES_HOST:
                self.egress_host = AGENT_CHOOSES_HOST
                return self
            # 规整过的值**存回去**：`checked_host` 校验的是去过空白、小写过的
            # 版本，而放行名单是按字符串逐字比对的。不存回去的话，
            # `" Payments.example.com"` 能通过校验，然后在判定时永远不命中
            # ——一条写了却不起作用的规则，正是这里最该拦下的东西。
            self.egress_host = checked_host(
                self.egress_host, what=f"工具 {self.name!r} 的 egress_host"
            )
        return self

    @model_validator(mode="after")
    def _an_open_target_declares_the_route_protocol(self) -> ToolDecl:
        """目标由 Agent 定的工具，参数声明只能是通用出口的那两个字段。

        通用出口只做一件事：把 `body` 原样 POST 到 `url`。所以沙箱**能核对**的
        就只有这两样，能执行的也只有这两样。声明成别的形状会有两种坏法，都是
        这份文件一直在防的那种：

        - 声明了目标系统的 schema（"这个接口要 order_id 和 amount"）：那条路上
          没人执行它——沙箱不知道目标系统接受什么，那个形状只有目标系统自己
          说了算。写了却没人核对，正是我们最不想要的失效。
        - 声明了工具自己都不用的参数（`method`、`headers` 之类）：路由永远发
          POST、头由 `_passthrough_headers` 决定，于是那些字段名存实亡。

        两种都在加载时拦下，而不是让它们安静地待在声明里。

        `body` 于是不许对键表态——它里面每一条关于键的约束都是关于某个目标系统的
        断言，而这条路的目标由调用方定。两种写法都表示键上不核：什么都不写
        （`{type: object}`），或者写 `additionalProperties: true` 把这件事写下来。

        **还有一条：它必须是写操作。** 通用出口转发的是 POST，而目标由调用方
        定——所以"这个工具只读"是一句沙箱没法核实、也不该采信的话：一次往任意
        地址发的 POST 会不会改状态，只有目标系统知道。声明成读的话，它会被记成
        读操作，幂等检测和主动行为审计**同时**看不见它，而报告上一切正常。
        """
        if not self.egress_is_agent:
            return self

        if not self.is_write:
            raise ValueError(
                f"工具 {self.name!r} 的 egress_host 是保留值 {AGENT_CHOOSES_HOST!r}"
                f"（目标由调用方指定），它必须是写操作（is_write: true）。\n"
                f"通用出口转发的是一次 POST 到它自己指定的地址——会不会改目标"
                f"系统的状态只有那边知道，而记成读操作会让幂等与主动行为审计"
                f"都看不见它。"
            )

        properties = self.parameters.get("properties")
        if not isinstance(properties, dict) or not properties:
            raise ValueError(
                f"工具 {self.name!r} 的 egress_host 是保留值 {AGENT_CHOOSES_HOST!r}"
                f"（目标由调用方指定），它必须声明通用出口的参数形状："
                f"必填 {GENERIC_EGRESS_URL_FIELD!r}（字符串），可选 "
                f"{GENERIC_EGRESS_BODY_FIELD!r}（对象）。\n"
                f"没有这份声明，边界上一次调用都核不了——而"
                f"「声明空着」和「没声明」在报告上是一个样子。"
            )
        extra = sorted(set(properties) - {GENERIC_EGRESS_URL_FIELD, GENERIC_EGRESS_BODY_FIELD})
        if extra:
            raise ValueError(
                f"工具 {self.name!r} 声明了 {extra}，而通用出口只认 "
                f"{GENERIC_EGRESS_URL_FIELD!r} 和 {GENERIC_EGRESS_BODY_FIELD!r} 两个字段"
                f"（把 body 原样 POST 到 url）。声明别的字段等于写一句没人执行的约束。"
            )
        if GENERIC_EGRESS_URL_FIELD not in (self.parameters.get("required") or []):
            raise ValueError(
                f"工具 {self.name!r} 的 {GENERIC_EGRESS_URL_FIELD!r} 必须是必填字段："
                f"通用出口没有它就没有目的地——这是一次连不上任何地方的调用，"
                f"真实工具会直接拒绝。"
            )
        if properties.get(GENERIC_EGRESS_URL_FIELD, {}).get("type") != "string":
            raise ValueError(
                f"工具 {self.name!r} 的 {GENERIC_EGRESS_URL_FIELD!r} 必须是字符串类型的 URL。"
            )

        # `body` 不许对键表态。
        #
        # 上面那条"别声明目标系统的 schema"在这里露了原形：声明里对 `body` 的
        # 每一个约束都是**关于某个目标系统**的断言，而这条路的目标由调用方定
        # ——同一个工具今天发给支付网关、明天发给工单系统。所以"body 里只许有
        # order_id 和 amount"是一句沙箱核实不了、而且随目的地而变的话；写下去
        # 的后果不是"多核了一层"，而是每一次转发都被判成 invalid_tool_arguments
        # ——报告上写着"Agent 参数送错了"，而真相是这份声明让每一次调用都过不去。
        #
        # **两种写法都表示"键上不核"**：什么都不写（`{type: object}`），或者写
        # `additionalProperties: true` 把这件事故意写下来。表态了就拒——空的
        # properties 表是一句写下来的"一个字段都不许有"，一样会冤枉 Agent。
        # 这条规则于是和运行时的判法严丝合缝：**拒掉的，恰好是那些会让每一次调用
        # 都过不去的写法**（见 toolschema 的模块 docstring）。
        body = properties.get(GENERIC_EGRESS_BODY_FIELD)
        if body is None:
            return self
        if body.get("type") != "object":
            raise ValueError(
                f"工具 {self.name!r} 的 {GENERIC_EGRESS_BODY_FIELD!r} 必须是对象"
                f"（它是原样转发给目标系统的那份请求体）。"
            )
        if "properties" in body or body.get("additionalProperties") is False:
            raise ValueError(
                f"工具 {self.name!r} 的 {GENERIC_EGRESS_BODY_FIELD!r} 不许对键表态："
                f"不写 properties（或写 additionalProperties: true）都表示键上不核，"
                f"写了键的名单就会被当真的约束。\n"
                f"它对 body 的每一条约束都是关于某个「目标系统」的断言，而这条路"
                f"的目标由调用方定——同一个工具可以去不同的主机。写下去的后果不是"
                f"少核了一层，而是每一次带业务字段的转发都会被判成参数错误（一份"
                f"空的 properties 表在这里的意思是「一个字段都不许有」），报告上却写着"
                f"是 Agent 送错了参数。"
            )
        return self

    @property
    def egress_is_agent(self) -> bool:
        """这个工具的目标由被测对象在调用时指定（`egress_host` 写了保留值）。"""
        return self.egress_host == AGENT_CHOOSES_HOST

    def proxy_path(self) -> str:
        """它的调用从代理的哪条路进来。

        固定目标的工具走 `/tools/{名字}`（目的地由沙箱配置定），目标由 Agent
        定的那些只能走通用出口 `/fetch/{名字}`——那条路按 URL 里的主机判，
        因为它没有可声明的目标。

        **这张表随运行注册一起过线**（见 `runner.registration_payload`），网关
        按键名读。所以"这个工具的调用该走哪条路"在运行当中是有据可查的，
        不是代理自己猜的。
        """
        if self.egress_is_agent:
            return generic_egress_path(self.name)
        return f"/tools/{self.name}"

    @field_validator("field_weights")
    @classmethod
    def _weights_are_usable(cls, v: dict[str, float]) -> dict[str, float]:
        """权重必须是正的有限数。规则本身在 `checked_field_weights` 里。

        **两边只留一份实现**：同一张表还从 HTTP 载荷那条路进来（运行注册，
        `gateway/app.py`）。各自写一份的话，总有一天一条路收下了 `nan` 而另一条
        拦住了——收下它的那条会让近似层静默失效，而报告上看起来只是"这一段没
        发现循环"。
        """
        return checked_field_weights(v)

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
    def _identity_fields_are_usable(self) -> ToolDecl:
        """主键声明必须是能用的字段名，而且只许写在写工具上。

        规则本身在 `checked_identity_fields` 里——**和语料那边共用一份**：这份
        声明有两个前门（这里的场景 YAML，和 `corpus._parse_tools`），两处各写一份
        校验的话，总有一天一个前门收下了一个不存在的字段名，而收下它的那条会让
        这一档判据**一次都不生效**，报告上看起来和"判过了、没问题"一样。

        "字段名在参数里真的存在"是**另一条**，在 `scenario/loader.py`——那需要
        参数 schema，而且只有场景有 schema（语料只写读写属性），所以它放在
        有 schema 可对的地方，不在这儿假装两处都管。
        """
        checked_identity_fields(
            self.identity_fields, tool=self.name, is_write=self.is_write
        )
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

    #: 这一段由**哪一个** Agent 跑——`agents:` 里的序号，从 0 开始。
    #:
    #: `None` = 不指名，用 0 号。所以单 Agent 的场景一个字都不用写，而
    #: `agents:` 只写一个的场景也一样。
    #:
    #: **它是串联测试的全部表达力**：同一个场景里 A 跑第一段、B 跑第二段，
    #: 于是引擎在每一段拉起的是不同的进程、给的是不同的记忆目录。
    #: 不指名就等于"还是那一个"——这是安全的一侧，也是绝大多数场景的形态。
    agent: int | None = None

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

    **`count`：这条预期该命中几条结论，默认 1。** 加它是因为"报了几条"原先
    根本没地方写：标注说"loop.strict 该报 fail"，而实际报出三条，比对只消费掉
    一条，**另外两条谁也不看**——报告于是说了比标注更多的话，而统计上完全看不
    出来（精确率按族算，族一旦"该张口"，里面多报几条不算错）。一条轨迹上真能
    合理地报出多条的情况是有的（两个工具各自在原地打转，严格的循环检测各报
    一条），所以这不是"默认不许"的问题，是"多报得说出来"的问题。

    `count` 写小了的后果由消费方兜着：语料那边多出来的 fail/warning 会进
    `over_reported`（见 `sandbox/corpus.py`），场景那边本来就按 unexpected 算失败。
    """

    model_config = {"extra": "forbid"}

    detector: str
    verdict: DetectorVerdict
    reason_code: str | None = None
    note: str = ""
    #: 该命中几条。`None` = 不写 = 1 条——**不写的和写成 1 的是同一件事**，
    #: 所以指纹那边把空值排除掉（见 `identity.sections`），别让一次"加了个可选
    #: 字段"变成所有场景的"判卷标准变了"。
    count: int | None = None

    @field_validator("count")
    @classmethod
    def _count_is_positive(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError(f"count 至少是 1（收到 {v}）：'该报零条'要写成族级的 `none`/`absent`")
        return v


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

    #: 判的是**哪一个** Agent 的记忆——`agents:` 里的序号，从 0 开始。
    #:
    #: `None` = 不指名，0 号。单 Agent 场景一个字都不用写。
    #:
    #: **串联场景里它是必填的**（见 `_one_agent_list`）。理由：记忆是每个 Agent
    #: 各自的一份目录，同一份记忆判据落到不同 Agent 上会得到相反的结论，
    #: 而不写就没人知道作者想判谁——引擎替他挑一个的话，报告会拿着一份答案
    #: 去回答另一个问题，且看上去完全正常。
    agent: int | None = None


class ForbiddenTextDecl(BaseModel):
    """一段不允许出现在记忆里的文本。

    这是记忆中毒检测**现在能做的那一半**：受控实验里注入进去的那句话是我们自己
    写的，所以写得出来。它抓不到没声明的注入——那不是这个类的实现问题，是这条路
    的边界（见 docs/detectors.md 第九节）。
    """

    model_config = {"extra": "forbid"}

    text: str
    reason: str

    #: 同上：这一段文本不许出现在**哪一个** Agent 的记忆里。
    agent: int | None = None


class MemoryDecl(BaseModel):
    """场景对记忆的判据。**空 = 不判，只观测。**

    和 `exemptions` 一样属于"判卷标准"（进配置指纹的 `expect` 节），不属于被测对象：
    改了它，前后两次运行就不再可比"失败少了几个"。
    """

    model_config = {"extra": "forbid"}

    must_hold: list[MustHoldDecl] = Field(default_factory=list)
    forbidden: list[ForbiddenTextDecl] = Field(default_factory=list)


class EgressRuleDecl(BaseModel):
    """一条放行规则：哪台主机、以及凭什么放它出去。"""

    model_config = {"extra": "forbid"}

    host: str
    #: 必填。没有理由的放行在审计时无法复核，等于永久豁免（与 `AllowRule` 同）。
    reason: str

    #: 这个世界里，这台主机由 mock 的哪条路径扮演。`None` = 不翻译，真发过去。
    #:
    #: 模拟出来的外部系统并不存在，出去的请求得有个落点。**它是"布景"而不是
    #: "授权"**：写不写、写成什么，都不影响放不放行，只影响这次调用落在哪。
    #: 声明过 `egress_host` 的工具靠那条声明翻译（沙箱里每个工具连的都是自己的
    #: mock），而没有固定目标的那些（保留值，见 `ToolDecl.egress_host`）没有
    #: 可声明的目标——所以只能由名单来说"这台主机在这个世界里是谁"。
    mock_path: str | None = None

    @model_validator(mode="after")
    def _usable(self) -> EgressRuleDecl:
        self.host = checked_host(self.host, what="放行规则")
        if self.mock_path is not None:
            self.mock_path = checked_mock_path(
                self.mock_path, what=f"放行规则 {self.host!r} 的 mock_path"
            )
        if not self.reason.strip():
            raise ValueError(
                f"放行规则 {self.host!r} 缺少 reason。"
                f"没有理由的放行在审计时无法复核，等于永久豁免。"
            )
        return self


class EgressDecl(BaseModel):
    """这次运行的出口放行名单（ADR 003）。

    出口策略是**部署方的决定**，不是 Agent 的——所以它写在"世界"这一节里，
    和订单备注并列：两者都是 Agent 醒来时那个环境的既定事实。它进配置指纹的
    `world` 节，所以两次运行之间改了名单就**不可比**：换了网，不是换了行为。

    **默认拒绝是默认值，这里写的是例外。** 名单上有的才出得去。

    **空名单和没写是两件事**（这个仓库里反复出现的那条规矩）：没写 `egress`
    段 = 这次运行不声明出口策略，按沙箱默认走（沙箱自己的工具后端自动放行）；
    写了 `egress` 而 `allow: []` = 声明了"这次不放行任何出口"。前者是不声明，
    后者是一句很重的声明——所以后者要求每个工具都交代自己的真实目标（见
    `Scenario._egress_declaration_is_wired`），不能悄悄生效。
    """

    model_config = {"extra": "forbid"}

    allow: list[EgressRuleDecl] = Field(default_factory=list)

    def to_allow_rules(self) -> list[AllowRule]:
        """转成策略层那条规则的形状。**只此一处**：加载期校验与运行时判定
        必须看同一份数据，各转一遍的话，总有一天校验放过的和判定的不是同一条。"""
        return [
            AllowRule(
                host=r.host,
                tool=None,
                reason=f"场景声明：{r.reason}",
                mock_path=r.mock_path,
            )
            for r in self.allow
        ]


class WorldDecl(BaseModel):
    """Agent 醒来时，目标系统里长什么样。

    两样东西：订单上的自由文本备注（**注入面**——上游系统里那些第三方写得了的
    字段）与这次运行的出口放行名单（**部署方的网络策略**）。记忆中毒的受控实验
    要的是前者的那句话；出口被拦下的那对实验要的是后者。

    **它是世界，不是判据**：备注里写了什么与"什么算失败"无关（那是 `memory`
    段的事）；名单上放了谁也与"什么算失败"无关（那是 `expect` 段的事）。两条
    场景把同一句话一个写成注入、一个写成正常备注，跑出来的差别必须来自这里
    ——只改判据不叫受控实验，那只是换了把尺子。
    """

    model_config = {"extra": "forbid"}

    #: 订单号 → 那句话。订单号必须是目标系统里真实存在的样例订单，加载时校验。
    order_notes: dict[str, str] = Field(default_factory=dict)

    #: 这次运行的出口放行名单。`None` = 没声明（沙箱默认策略）。
    egress: EgressDecl | None = None


class Scenario(BaseModel):
    model_config = {"extra": "forbid"}

    id: str
    description: str = ""

    #: 被测 Agent。**两种写法都认，但只能写一种**：
    #:
    #:   `agent:`   —— 单数，只有一个 Agent 时的糖（今天 29 份场景全是这一种）；
    #:   `agents:`  —— 复数，**串联测试**用它：A 跑第一段、B 跑第二段。
    #:
    #: 两个字段都在模型上（而不是拿一个 `AgentSpec | list[AgentSpec]` 去接），
    #: 是为了让"写错位置"变成一句好话：pydantic 对联合类型报的是
    #: `Input should be a valid dictionary or instance of AgentSpec`，而这里
    #: 报的是"两种写法同时出现，选一种"。
    #:
    #: `agent` 在经过校验之后**总是**等于 `agents[0]`（见 `_one_agent_list`）。
    #: 于是所有读 `scenario.agent.X` 的地方（引擎起进程、配置指纹、界面、
    #: 体检菜单，19 处）在单 Agent 场景下一字不用改，也不会各自判断一次
    #: "这场景有几个 Agent"。**它是派生值，不是第二份真相**——唯一的写者
    #: 是那个校验器，`tests/test_multi_agent.py` 里有一条盯着它们不许分叉。
    #:
    #: 两个字段都**不带 `| None`**，尽管输入时可以只写一个：收拢发生在
    #: `mode="before"` 的校验器里（早于字段校验），所以到字段这一层两者都已经
    #: 有值了。这一点是**类型检查器替我们守着的**——写成 `AgentSpec | None` 的话，
    #: 那 19 处 `scenario.agent.X` 会各自变成"可能没有 Agent"，myy 会逐处报错，
    #: 而每一处的"修法"都是加一个没人能证明成立的断言。让收拢发生在上一层，
    #: `scenario.agent` 就是一个全函数，和从前一样。
    agent: AgentSpec
    agents: list[AgentSpec]

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
    #
    # 另一个值是 allow：这次不判它们。那条结论仍然列在报告的「检测结论」一节里
    # ——那一节列的是**看见了什么**，和判不判无关——只是不进预期比对。
    #
    # **刻意没有第三个值 warn。** 它一度写在类型里，但从加进来那天起就没接上：
    # 判定只看意外那一栏空不空，warn 落进和 fail 完全相同的分支。于是写了它的
    # 场景跑出来和 fail 一模一样——配置指纹里记着它、报告上什么都没有，而写的
    # 人以为"报了但不拦"。那不只是少一块功能：一个列出来又不影响结论的失败，
    # 和 `expect` 里写一条 fail 的区别，只剩**你没写理由**，而理由正是 note 存在
    # 的意义。知道会失败就写进 expect，不想被拦就写 allow，中间那一档没有诚实
    # 的产出方。`tests/test_scenario_dsl.py` 里有一条测试盯着它别再回来。
    unexpected_failures: Literal["fail", "allow"] = "fail"

    # 目标系统侧**真实扣款的笔数**。None = 不检查。
    #
    # 这是地面真值，和检测结论是两回事，所以单独断言。检测器完全可能报出
    # 漂亮的 pass，而钱其实被扣了两次——那正是这个工具要抓的失效方式。
    # 反过来，"记忆守卫拦住了重复支付"这种正样本也只能靠它证明：边界上
    # 什么都看不到，没有扣款流水就没有证据。
    expected_charges: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _one_agent_list(cls, data: Any) -> Any:
        """`agent` 与 `agents` **在字段校验之前**就收成一份名单。

        三条规矩：

        1. 一个都没写 —— 这份场景测谁？报错，不猜。
        2. 两种写法同时出现 —— 报错。两个说法同时在场时，读的人永远不知道
           哪个算数；而"两个都写、引擎挑一个"正是最坏的那种：文件里躺着
           另一份看起来同样有效的配置。（豁免那边同一个道理：宁可拒收。）
        3. 只写 `agent` —— 收成 `agents: [agent]`，并把 0 号回填给 `agent`。

        写成 `mode="before"` 而不是 `mode="after"`，是为了让两个字段都**不带
        `| None`**：到字段那一层两者都已经有值，`scenario.agent` 于是是全函数。
        放在 `after` 里的话字段类型只能写成 `AgentSpec | None`，而那 19 处
        `scenario.agent.X` 会各自变成"可能没有 Agent"——每一处的"修法"都是加一个
        没人能证明成立的断言。**能靠结构排除的，不要靠断言。**
        """
        if not isinstance(data, dict):
            # 输入不是字典（比如拿一个已有的场景实例再 `model_validate` 一次）：
            # 这一层只管"输入里两种写法怎么收"，收不动就原样交下去。
            return data
        payload = dict(data)
        single = payload.get("agent")
        many = payload.get("agents")
        if not many and single is None:
            raise ValueError(
                "场景没有写被测 Agent。写 `agent:`（只有一个）或者 `agents:`（一队）。"
            )
        if many and single is not None:
            raise ValueError(
                "`agent:` 和 `agents:` 同时写了。留一个：\n"
                "  只有一个 Agent 就写 `agent:`；\n"
                "  串联测试（A 跑一段、B 跑一段）写 `agents:`，"
                "再用每个会话的 `agent: <序号>` 指明这一段由谁跑。\n"
                "两个都写的话，文件里躺着的那一份到底算不算数没人说得清。"
            )
        if not many:
            many = [single]
        payload["agents"] = many
        payload["agent"] = many[0]
        return payload

    @model_validator(mode="after")
    def _agent_indices_resolve(self) -> Scenario:
        """会话与记忆判据指名的那个 Agent 必须真的存在。

        放在 `after` 里：这一步要**同时**看会话/判据上的序号和收拢之后的名单，
        而名单是上一步才收到的。

        两句都是**必须报错、不能兜底成 0**：兜底的话，"B 跑第二段"会静默地变成
        "A 又跑了一段"，报告上一切正常，测的却是另一件事。
        """
        for index, session in enumerate(self.sessions):
            if session.agent is not None and not 0 <= session.agent < len(self.agents):
                raise ValueError(
                    f"会话 #{index}（{session.name}）指名了 agent: {session.agent}，"
                    f"但这份场景只声明了 {len(self.agents)} 个 Agent"
                    f"（合法的序号是 0 到 {len(self.agents) - 1}）。\n"
                    f"序号是从 0 开始的，没有 `agents:` 时唯一那个就是 0。"
                )

        # 记忆判据也必须指到人。
        #
        # 两件事在这里一起管：序号越界（写错了），以及**多 Agent 场景里没写**
        # （写漏了）。后者是这一段的重点：一条 `must_hold` 判的是某一份记忆，
        # 而串联场景里有好几份。不写 `agent:` 时引擎可以"顺手挑 0 号"，
        # 报告看起来一模一样——它拿着一份答案去回答另一个问题。宁可拒收。
        # 两张表先拼成"哪一节、第几条、指名了谁"的三元组，再统一检查。
        # 分别写两遍循环的话，"越界"和"没写"这两种错会在两处各判一次，
        # 而两处的判据迟早会不一致——那时候哪一种错被漏掉取决于它是哪一节。
        named: list[tuple[str, int, int | None]] = [
            ("must_hold", i, r.agent) for i, r in enumerate(self.memory.must_hold)
        ] + [("forbidden", i, r.agent) for i, r in enumerate(self.memory.forbidden)]
        for where, index, target in named:
            if target is None:
                if len(self.agents) > 1:
                    raise ValueError(
                        f"`memory.{where}[{index}]` 没写 `agent:`，"
                        f"而这份场景声明了 {len(self.agents)} 个 Agent。\n"
                        f"记忆是每个 Agent 各自的一份，这条判据落在谁头上"
                        f"不定下来，报告就会拿着一份答案去回答另一个问题。\n"
                        f"补上 `agent: 0` 到 `agent: {len(self.agents) - 1}` "
                        f"里的一个（序号见 `agents:` 里的顺序）。"
                    )
                continue
            if not 0 <= target < len(self.agents):
                raise ValueError(
                    f"`memory.{where}[{index}]` 指名了 agent: {target}，"
                    f"但这份场景只声明了 {len(self.agents)} 个 Agent"
                    f"（合法的序号是 0 到 {len(self.agents) - 1}）。\n"
                    f"序号是从 0 开始的，没有 `agents:` 时唯一那个就是 0。"
                )
        return self

    @field_validator("unexpected_failures", mode="before")
    @classmethod
    def _the_policy_has_no_middle_value(cls, v: Any) -> Any:
        """把"没有 warn 这个值"说人话。

        类型上的 Literal 已经能拦住它，但 pydantic 的原话是
        `Input should be 'fail' or 'allow'`——一个写了 warn 的人读到的是"这工具不认
        我的写法"，而事实是那个写法**有过的意思**没有诚实的产出方。理由要说出来，
        因为这是最容易被谁顺手加回来的一个值：看起来只差十行实现。
        """
        if v == "warn":
            raise ValueError(
                "unexpected_failures 没有 warn 这个值。它一度写在类型里，但从加进来"
                "那天起就和 fail 走同一条分支——写它的场景跑出来和 fail 一模一样，"
                "而读报告的人以为这次运行已经判过了。要拦就写 fail，不拦就写 allow；"
                "如果是「我知道会有这条失败」，把它写进 expect 并把理由写在 note 里。"
            )
        return v

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

    @model_validator(mode="after")
    def _egress_declaration_is_wired(self) -> Scenario:
        """出口名单与工具的 `egress_host` 必须**成对出现**，而且互相对得上。

        三条都是"装上了但没接线"的写法，在报告上全都看不出来：

        1. 声明了名单，某个工具却没写 `egress_host` —— 那个工具的出口不受
           名单管（它按 mock 的地址判，一律放行）。读报告的人会以为这次运行
           限制了它，而它一次都没被限制过。
        2. 写了 `egress_host`，场景却没有 `world.egress` —— 那条声明一次都
           不会被读到。**不报错的话它就是一句纯粹的装饰**，而写它的人以为
           自己关上了一扇门。
        3. 名单上一条规则的主机，没有任何工具指向它 —— 那条规则永远不会命中。
           它多半是打错了字（`payments.example.com` 写成 `payment.example.com`），
           而症状是"那次调用被拒了"，读起来像 Agent 去了不该去的地方。

        第 3 条**不反过来要求**"每个工具的主机都在名单上"——不在名单上正是
        "拒绝"的表达方式，那是这套东西存在的理由。

        **目标由 Agent 定的工具（`egress_host` 写了保留值）破这三条里的两条**，
        而且是必然的：它没有可写下来的目标主机，所以第 1 条只要求它交代"目标由
        调用方定"（那就是它的交代），第 3 条则整个不成立——它的目标在运行当中
        才定，名单上任何一条都可能是它可以去的地方，"哪条没工具指向"这个问题
        在这里没有答案。第 2 条对它照旧：写了一句没人读的声明仍然要报错。
        """
        egress = self.world.egress
        declared = {t.name: t.egress_host for t in self.tools if t.egress_host}
        open_targets = sorted(t.name for t in self.tools if t.egress_is_agent)

        if egress is None:
            # 保留值不算"读不到"：它的路由作用与名单无关（见 proxy_path）。写死的
            # 主机名才算——那条声明要靠名单才判得动。
            fixed = sorted(n for n, h in declared.items() if h != AGENT_CHOOSES_HOST)
            if fixed:
                raise ValueError(
                    f"工具 {fixed} 声明了 egress_host，但场景没有 "
                    f"`world.egress` 段——这条声明一次都不会被读到。\n"
                    f"它说的是「这个工具在真实环境里出到哪台主机」，"
                    f"而按它判定的那份放行名单没写，于是它不影响任何一次调用。"
                )
            return self

        missing = [t.name for t in self.tools if not t.egress_host]
        if missing:
            raise ValueError(
                f"场景声明了出口放行名单（world.egress），但工具 {sorted(missing)} "
                f"没交代自己的出口。\n"
                f"名单是按真实目标主机判的；没写这个字段的工具仍然按沙箱"
                f"mock 的地址判，于是它不受名单管——报告上却看不出这个区别。\n"
                f"固定目标的写那台主机（payments.example.com），目标由调用方定的"
                f"写保留值 {AGENT_CHOOSES_HOST!r}：两种都要写，写出来的意思不同，"
                f"但都得写。"
            )

        if open_targets:
            return self

        known = set(declared.values())
        unwired = [r.host for r in egress.allow if r.host not in known]
        if unwired:
            raise ValueError(
                f"出口名单里的 {sorted(unwired)} 没有任何工具指向它——"
                f"这些规则永远不会命中。\n"
                f"工具声明的真实目标主机是 {sorted(known)}。"
                f"想拒绝某个工具，就不要在名单上写它的主机。"
            )
        return self

    def egress_policy(self) -> EgressPolicy | None:
        """这次运行的出口策略。`None` = 场景没声明，按沙箱默认走。

        **它必须在 mock 的端口定下来之后再构造**（`default_policy()` 会去读
        `settings.tool_upstream_base_url`），见 `ScenarioRunner`。
        """
        egress = self.world.egress
        if egress is None:
            return None
        tool_hosts: dict[str, str] = {}
        for t in self.tools:
            # 目标由 Agent 定的那些**不进这张表**：放进去的话它们会被按字面量
            # "agent" 判（`EgressPolicy` 也会在构造时直接报错），而正确的判法
            # 是走"没声明"那一支——按这次调用实际要连的那台主机判。
            if t.egress_host and not t.egress_is_agent:
                tool_hosts[t.name] = t.egress_host
        return policy_with_rules(egress.to_allow_rules(), tool_hosts=tool_hosts)

    def tool_index(self) -> dict[str, ToolDecl]:
        return {t.name: t for t in self.tools}

    def declared_writes(self) -> set[str]:
        return {t.name for t in self.tools if t.is_write}

    def idempotency_params(self) -> dict[str, str]:
        return {t.name: t.idempotency_param for t in self.tools if t.idempotency_param}

    def loop_field_weights(self) -> dict[str, float]:
        """各工具声明的字段权重合成一张表，给循环检测器的近似层用。

        **形状是扁的**（字段名 → 权重），因为检测器的入参就是扁的：它拿到的是
        一串调用，没打算按工具分别调参。两个工具给同一个字段名声明不同的权重
        时，合出来的是哪一份就成了掷骰子——那条冲突在加载时就报错
        （见 `loader._check_references`），不在这里替谁选。

        和 `exemption_set` / `memory_rules` 同一个理由放在这里：**校验要有统一
        的落点**，加载器先调一遍，用户 YAML 里的错在加载时就报出来。
        """
        merged: dict[str, float] = {}
        for tool in self.tools:
            merged.update(tool.field_weights)
        return merged

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
            # `agent` 原样带过去：检测器靠它把观测筛成"这个人的那一份"。
            # 到这里为止单 Agent 场景里它一直是 None，检测器再折成 0 号。
            must_hold=tuple(
                MustHold(path=r.path, reason=r.reason, agent=r.agent)
                for r in self.memory.must_hold
            ),
            forbidden=tuple(
                ForbiddenText(text=r.text, reason=r.reason, agent=r.agent)
                for r in self.memory.forbidden
            ),
        )
