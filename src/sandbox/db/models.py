"""System of record 的数据模型（ADR 002）。

设计取向：**原始 span 流不入库，结构化实体入库。** 检测器的查询对象是
"某个会话序列的 tool_call 序列"，不是全量 span——所以 tool_call 是一等实体，
带索引，而不是从 span JSON 里临时解析。
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


# --- 枚举 ----------------------------------------------------------------


class RunStatus(enum.StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"  # 被带外熔断终止


class DetectorVerdict(enum.StrEnum):
    """四态结果语义，见 docs/detectors.md。

    INCONCLUSIVE 绝不可折叠为 PASS —— 那会把"啥也没测"读成"全绿"。
    """

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    INCONCLUSIVE = "inconclusive"


class Confidence(enum.StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# --- 实体 ----------------------------------------------------------------


class TestRun(Base):
    """一次测试运行。

    config_hash 是回归归因的关键：需求要求"差异项可定位到具体的 Prompt 或
    工具改动"，没有内容寻址的配置哈希就永远定位不了。
    """

    __tablename__ = "test_run"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    scenario_id: Mapped[str] = mapped_column(String(128), index=True)
    agent_version: Mapped[str] = mapped_column(String(128))
    config_hash: Mapped[str] = mapped_column(String(64), index=True)
    #: 上面那个哈希的**口径**。NULL = 本字段存在之前跑的运行，它的
    #: `config_hash` 是按旧口径算的（只覆盖场景 id 与模型名，改了脚本或
    #: Prompt 它都不变），**不能拿来判"配置有没有变"**。
    #:
    #: 有这一列是因为"两种口径的哈希不相等"必须和"配置真的变了"分开。
    #: 少了它，一次旧运行和一次新运行放一起比，会得出"配置变了"——
    #: 那是一句基于不可比数据的结论。
    config_scheme: Mapped[str | None] = mapped_column(String(16))
    #: 分节的配置原始值（`scenario/identity.py` 的 `sections()`）。
    #:
    #: 存**原值**而不是哈希：差异报告要说清改了什么，只有哈希就只能说"变了"。
    #: 和 `report_json` 同一个理由——场景文件事后被编辑，这份快照仍是当时的。
    config_parts_json: Mapped[str | None] = mapped_column(Text)
    session_sequence_id: Mapped[str | None] = mapped_column(String(36), index=True)

    status: Mapped[str] = mapped_column(String(16), default=RunStatus.RUNNING.value)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    total_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    # --- 判定结果 --------------------------------------------------------
    # `status` 说的是**执行**（跑完了 / 崩了 / 被熔断），`verdict_ok` 说的是
    # **判定**（通过 / 不通过）。这两件事会分开，而且分开的方式正是本工具
    # 反复强调的那类：一次正常跑完的运行完全可能不通过；而 `loop` 场景里
    # 被熔断杀掉的那次运行——进程是被我们亲手杀的——恰恰是**通过**的。
    # 压成一个字段就再也分不出"没跑起来"和"跑起来了但没通过"。
    #
    # 可为空：本字段之前的历史运行没存过结论，NULL 表示**没记录**。
    # **NULL 不是 False。** 把没记录读成"不通过"，等于替一条没有结论的运行
    # 编一个结论出来——界面必须显示"结论未留存"，而不是显示一个红叉。
    verdict_ok: Mapped[bool | None] = mapped_column(Boolean)

    # 这次运行的**完整结论快照**（`ScenarioResult.to_dict()`）。
    #
    # 为什么不从结构化表重算：结构化的表记的是**发生了什么**（调用、会话、
    # 检测发现），快照记的是**当时报出去的结论**。后者含前者没有的事实——
    # 每段脚本被消费了多少、Agent 的退出码、熔断该归到哪个会话——而且依赖
    # 场景文件当时的样子；场景文件事后被编辑过，重算出来的是**另一份**结论。
    #
    # 两者冲突时以结构化表为准：它们是明细，快照只是投影。
    report_json: Mapped[str | None] = mapped_column(Text)

    # 环境钉版本：回归可比性的前提（LLM 不确定，必须记录 seed/温度/模型版本）
    env_json: Mapped[str | None] = mapped_column(Text)

    sessions: Mapped[list[Session]] = relationship(back_populates="test_run")
    tool_calls: Mapped[list[ToolCall]] = relationship(back_populates="test_run")
    findings: Mapped[list[DetectorFinding]] = relationship(back_populates="test_run")


class RegressionCase(Base):
    """一条回归用例：把某次运行的结论钉成基线，以后拿它比对。

    **基线是冻结的，不是"指向某次运行"。** 这一点值得解释，因为它看起来像
    冗余：把运行的列抄一份到用例里，为什么不直接存 `baseline_run_id`？

    因为用例记的是**有人在某个时刻认定"这个结论是对的"**。那是一个断言，
    断言必须自足。只存 id 的话，运行记录一旦被清理（保留期策略、手工删库），
    用例就静默退化成"没有基线"——而它在列表里看起来仍然是一条正常的用例，
    点进去才发现什么都没得比。**静默退化比报错难查得多。**

    `baseline_run_id` 因此降级为**出处**：它告诉你这个基线是哪次运行来的，
    好让人点回去看当时的调用明细。它不参与比较。

    抄过来的列与 `TestRun` 同名同义（加 `baseline_` 前缀），这不是巧合：
    两侧用同一套字段名，比较逻辑才能只写一遍。
    """

    __tablename__ = "regression_case"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    scenario_id: Mapped[str] = mapped_column(String(128), index=True)
    note: Mapped[str] = mapped_column(Text, default="")

    #: 出处的运行 id。**不参与比较**，只在界面上给一个"回看当时的明细"的入口。
    #: 那次运行被清掉之后这一列会指向一个不存在的运行——这是可接受的，因为
    #: 基线本身没丢。见类 docstring。
    baseline_run_id: Mapped[str] = mapped_column(String(36), index=True)

    # --- 冻结的基线（与 TestRun 的对应列同义） ---------------------------
    baseline_verdict_ok: Mapped[bool | None] = mapped_column(Boolean)
    baseline_report_json: Mapped[str | None] = mapped_column(Text)
    baseline_config_scheme: Mapped[str | None] = mapped_column(String(16))
    baseline_config_hash: Mapped[str | None] = mapped_column(String(64))
    baseline_config_parts_json: Mapped[str | None] = mapped_column(Text)
    baseline_env_json: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Session(Base):
    """一次会话。跨会话检测的最小单元。"""

    __tablename__ = "session"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    test_run_id: Mapped[str] = mapped_column(ForeignKey("test_run.id"), index=True)
    session_sequence_id: Mapped[str] = mapped_column(String(36), index=True)
    # 会话在序列中的序号——跨会话循环检测要按此排序
    ordinal: Mapped[int] = mapped_column(Integer, default=0)

    # --- 记忆视野：跨会话归因的**唯一依据** -----------------------------
    # 这个会话开始时，Agent 的记忆覆盖到哪个序号为止。
    #
    # 自然延续时 = ordinal - 1（记得此前发生的一切）。从旧快照恢复时，这个值
    # 会比 ordinal - 1 小——**缺的那一段就是 Agent 不可能知道的部分**。
    #
    # 归因规则因此不需要猜：某次调用发生在序号 a、又在序号 b 重复，
    # 若 a <= memory_horizon_ordinal(b)，Agent 本可以知道 → Agent 的锅；
    # 否则它不可能知道 → 是恢复机制的锅。
    #
    # 默认 -1 而不是 0：序号 0 的会话不存在"之前的记忆"，写成 0 会让
    # "本会话内的重复"被误判成"Agent 本可以知道"，把同会话的判定搅乱。
    memory_horizon_ordinal: Mapped[int] = mapped_column(Integer, default=-1)
    restored_from_snapshot_id: Mapped[str | None] = mapped_column(String(36))

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), default="active")

    test_run: Mapped[TestRun] = relationship(back_populates="sessions")


class MemorySnapshot(Base):
    """记忆快照。

    **它不是观测记录，是操纵记录。** 记忆在 Agent 进程内，不在 ADR 001 的
    网络边界上——边界拦不到它。所以这里做的不是"看见 Agent 的记忆"，而是
    "把 Agent 的记忆回滚到某个时刻，看它会不会因此重复扣款"。

    这一点决定了归因：重复如果是我们自己回滚造成的，报告就必须说清楚，
    否则用户会去改一个没问题的 Prompt。
    """

    __tablename__ = "memory_snapshot"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_sequence_id: Mapped[str] = mapped_column(String(36), index=True)

    # 快照取自哪个会话边界。ordinal=k 且 seq=s 表示"序号 k 的会话进行到 s
    # 之后"。边界快照的 s 为 0 时表示"序号 k 的会话开始之前"。
    taken_at_ordinal: Mapped[int] = mapped_column(Integer, default=0)
    taken_at_seq: Mapped[int] = mapped_column(Integer, default=0)

    # 内容寻址：摘要相同即内容相同。用于判断"回滚到的是不是同一个状态"，
    # 以及识别"恢复了快照但其实什么都没变"这种空操作。
    digest: Mapped[str] = mapped_column(String(64), index=True)
    # 适配器标识（hermes_file / langgraph_checkpointer ...）。恢复时必须用
    # 同一个适配器，否则等于拿 A 格式的字节去覆盖 B 格式的文件。
    adapter: Mapped[str] = mapped_column(String(64))

    # lite profile 落本地文件；生产期换 PostgreSQL 元数据 + S3 大对象
    payload_path: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    entry_count: Mapped[int] = mapped_column(Integer, default=0)

    taken_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ToolCall(Base):
    """工具调用。**场景 1/2/4 的主力表**——幂等、循环、重复计费全查它。

    刻意不放在 span JSON 里：跨会话查询需要索引，JSON 解析撑不住。
    """

    __tablename__ = "tool_call"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    test_run_id: Mapped[str] = mapped_column(ForeignKey("test_run.id"), index=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("session.id"), index=True)
    session_sequence_id: Mapped[str] = mapped_column(String(36), index=True)

    # 运行内序号：循环检测的窗口切片依赖它，不能只靠时间戳
    seq: Mapped[int] = mapped_column(Integer, index=True)
    tool_name: Mapped[str] = mapped_column(String(128), index=True)

    # 规范化后的参数（剔除 timestamp/request_id 等噪声），近似层比较用它
    args_json: Mapped[str] = mapped_column(Text)
    # 结构哈希：严格层判定"完全相同"
    arg_hash: Mapped[str] = mapped_column(String(64), index=True)
    # 跨会话指纹 = hash(tool_name + arg_hash)，幂等检测的核心
    fingerprint: Mapped[str] = mapped_column(String(64), index=True)

    # 工具返回体的**字节数**。正文不落库——这是刻意的，也是这里从 `result_json`
    # 改过来的原因：
    #
    # * **没有人读它。** 从第一次提交（4b290d6）起它就一直在写，而全仓的读取方
    #   只有两处：`detectors/base.py` 里一句"CallView 刻意不含它"的注释，和一处
    #   测试把它设成 None。**写进去、没人看**的那些字节，代价是实打实的：返回体
    #   是被测对象自己引出来的内容（可能很大、可能有 PII），而它在库里待多久
    #   由保留期决定，不由当初那次调用决定。
    # * **文档一直在说它不落库。** 一共**十三处**写着"返回体不落库（只留参数
    #   指纹）"，其中几处还挂着"按 ADR 002"。数法：在改正之前那次提交上
    #   `git grep -n "不落库\|不留档\|不留正文\|不入库\|留参数指纹"`，再去掉
    #   与工具返回值无关的那一条（`mock/toolspec.py` 里订单视图那句）。分布是
    #   README 三处、call-chain 两处、detectors 两处、scenarios 一处、
    #   session-diff 一处（README 与四份 docs 共九处），场景注释、测试注释各
    #   一处，源码注释两处。ADR 002 里没有这句话（它说的是 span 级数据不重复
    #   落库）。十三处说法与代码相反，而代码这一侧的实际情况是：
    #   **用户被告诉了一件不成立的事**。
    # * **报告需要的是体量，不是正文**：按工具拆成本要算"这个工具的返回体在
    #   上下文里被重发了多少"，那是个乘法，正文不参与。
    #
    # 于是两边对齐到中间：留下体量、不留正文，十三处说法改成"只留参数指纹与
    # 返回体字节数"。**NULL 与 0 是两件事**：0 表示返回了一个空体；NULL 表示
    # 没有体量记录——这次调用没出去（被策略挡下、参数不合法、还没发就失败了），
    # 或者这一行来自正文被清掉之前的老库（那时没量过体量，也就补不出来）。
    result_bytes: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="ok")
    error: Mapped[str | None] = mapped_column(Text)

    # 写操作（有副作用）默认不允许豁免告警，见 docs/detectors.md
    is_write: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    # 工具声明的幂等键（请求头/参数名）。None = 未声明幂等保护
    idempotency_key: Mapped[str | None] = mapped_column(String(255))

    # 区分"Agent 自发重试"与"快照恢复导致的重放"——这个区分本身就是卖点
    replay_reason: Mapped[str | None] = mapped_column(String(32))
    # 被出口代理拒绝（ADR 003）。**不计入 Agent 异常行为告警**，否则污染检测结果
    blocked_by_proxy: Mapped[bool] = mapped_column(Boolean, default=False)

    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    test_run: Mapped[TestRun] = relationship(back_populates="tool_calls")

    __table_args__ = (
        # 跨会话幂等检测的主查询路径
        Index("ix_tool_call_seq_fp", "session_sequence_id", "fingerprint"),
        # 循环检测的窗口扫描路径
        Index("ix_tool_call_run_seq", "test_run_id", "seq"),
    )


class LlmCall(Base):
    """LLM 调用。Token 计数以网关为准——**不信任 Agent 侧的 usage_metadata**。"""

    __tablename__ = "llm_call"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    test_run_id: Mapped[str] = mapped_column(ForeignKey("test_run.id"), index=True)
    session_id: Mapped[str | None] = mapped_column(String(36), index=True)

    seq: Mapped[int] = mapped_column(Integer, default=0)
    model: Mapped[str] = mapped_column(String(128), index=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_write_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    # 这一次是流式请求落的账。**读它的时候要知道 latency_ms 的含义不一样**：
    # 非流式是整个响应往返，流式是到最后一帧为止。由网关转发时写（`stream.py`），
    # 场景 `stream_clean` / `clean` 那一对是它的活样本。
    streaming: Mapped[bool] = mapped_column(Boolean, default=False)
    # 这几个 token 数是**从哪来的**。今天有两个值：
    #
    #   ok              —— 上游给的 usage（权威）；
    #   usage_estimated —— 没拿到（流式没带 usage 帧、或被中途断开），
    #                      数字是网关按看到的内容估的。
    #
    # 写它的是 `gateway/stream.py` 的 `numbers_for`，读它的是成本聚合
    # （`CostRollup.estimated_calls` → 报告里那句"这几行不是账单"）。
    # **读到别的值时不要当成 ok**：多出来的值是别人加的，含义未知。
    status: Mapped[str] = mapped_column(String(16), default="ok")
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (Index("ix_llm_call_run_seq", "test_run_id", "seq"),)


class DetectorFinding(Base):
    """检测器结论。四态语义 + 置信度，见 docs/detectors.md。"""

    __tablename__ = "detector_finding"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    test_run_id: Mapped[str] = mapped_column(ForeignKey("test_run.id"), index=True)
    detector: Mapped[str] = mapped_column(String(64), index=True)

    verdict: Mapped[str] = mapped_column(String(16), index=True)
    confidence: Mapped[str] = mapped_column(String(16), default=Confidence.HIGH.value)

    # 人类可读的一句话结论
    summary: Mapped[str] = mapped_column(Text)
    # 结构化细节：命中的窗口、涉及的工具调用 id、豁免规则等
    detail_json: Mapped[str | None] = mapped_column(Text)

    # 被豁免规则静音时必须记录，且**豁免不是静默的**——报告要显示"因 N 条豁免未告警"
    exempted_by: Mapped[str | None] = mapped_column(String(255))

    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    test_run: Mapped[TestRun] = relationship(back_populates="findings")
