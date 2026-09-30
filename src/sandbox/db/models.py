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
    session_sequence_id: Mapped[str | None] = mapped_column(String(36), index=True)

    status: Mapped[str] = mapped_column(String(16), default=RunStatus.RUNNING.value)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    total_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    # 环境钉版本：回归可比性的前提（LLM 不确定，必须记录 seed/温度/模型版本）
    env_json: Mapped[str | None] = mapped_column(Text)

    sessions: Mapped[list[Session]] = relationship(back_populates="test_run")
    tool_calls: Mapped[list[ToolCall]] = relationship(back_populates="test_run")
    findings: Mapped[list[DetectorFinding]] = relationship(back_populates="test_run")


class Session(Base):
    """一次会话。跨会话检测的最小单元。"""

    __tablename__ = "session"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    test_run_id: Mapped[str] = mapped_column(ForeignKey("test_run.id"), index=True)
    session_sequence_id: Mapped[str] = mapped_column(String(36), index=True)
    # 会话在序列中的序号——跨会话循环检测要按此排序
    ordinal: Mapped[int] = mapped_column(Integer, default=0)

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), default="active")

    test_run: Mapped[TestRun] = relationship(back_populates="sessions")


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

    result_json: Mapped[str | None] = mapped_column(Text)
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
    streaming: Mapped[bool] = mapped_column(Boolean, default=False)
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
