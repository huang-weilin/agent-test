# ADR 002：自研数据库是 system of record，Langfuse 仅作可视化前端

- **状态**：已接受
- **日期**：2026-09-30
- **决策者**：项目组

## 背景

需求文档二.二节的架构同时要求：

- 数据层：`PostgreSQL（事务）+ ClickHouse（Trace）`
- 可观测性后端：`Langfuse（自托管）`

而 Langfuse 自托管（v3）**本身依赖 ClickHouse + Redis + S3**。这意味着同一份 span 级数据会被存两遍：一遍进自研的 ClickHouse，一遍进 Langfuse 的 ClickHouse。

同时，文档 3.4 节把 Trace 采集定位为"为循环检测、成本统计、回归对比提供数据基础"，即自研代码要直接查询 Trace。而风险表里又写"LLM-as-Judge"和成本统计要自建。

结果就是：**两个真相源、双份存储成本、双份写入路径，且无法保证一致。**

## 决策驱动因素

- 本机（开发环境）为 7.9GB 内存，ClickHouse 常驻 1.5–4GB，是整套栈里最重的组件。
- 文档的 MVP 取舍已确定：**成本统计、Trace 可视化、LLM-as-Judge 复用 Langfuse 已有能力，不自研**（2026-09-30 决策）。既然是复用，就不该再自建一份 span 存储。
- 真正需要自研查询的是**行为级检测**：工具调用指纹、循环模式、记忆漂移。这些查询的对象是**结构化的实体**（run / session / tool_call），不是原始 span 流。

## 决策

1. **自研数据库（开发期 SQLite，生产期 PostgreSQL）是 system of record**，保存必须被查询和断言的实体：
   `TestRun`、`Session`、`SessionSequence`、`ToolCall`、`MemorySnapshot`、`DetectorFinding`、`RegressionCase`。
2. **Langfuse 只作为人的可视化前端**，不作为程序化查询的数据源。人看轨迹用 Langfuse，程序做检测读自研库。
3. **span 级原始数据不重复落自研库**，通过 `trace_ref`（Langfuse trace id）关联。若未来确需本地留存原始 span，再单独评估，不在 MVP 范围。
4. **ClickHouse 从 MVP 移除。** 开发期用 SQLite，生产期先用 PostgreSQL 按月分区。只有当真实验证到单表量级撑不住时（建议阈值：单表 > 5000 万行或查询 P95 > 2s）才引入 ClickHouse，且届时 Langfuse 与自研共用同一个 ClickHouse 实例，不各起一个。

## 后果

**正面**
- 单一真相源，检测逻辑与可视化不会出现数据不一致。
- 本机 lite profile 内存占用从 ~6GB 降到 ~1GB，直接可用。
- 省掉 ClickHouse 的运维、备份、schema 迁移成本。

**负面 / 需要接受的代价**
- 放弃 ClickHouse 的列存聚合性能。缓解：检测器的查询模式是"按 session/sequence 取一小段 tool_call 序列"，不是全表扫描聚合，行存足够；且 `tool_call` 表会按 `(session_sequence_id, ts)` 建索引。
- 与 Langfuse 的关联依赖 `trace_ref` 正确写入，需要在端到端测试里覆盖。
- 引入 Langfuse 作为外部依赖（其核心为 MIT，部分企业特性另授权，**需在 W3 接入前核实 license 边界并锁版本**），若其 API 变动会影响可视化层。

**触发重新评估的条件**
- 需要跨项目全局的 span 级聚合查询时；
- 单表行数超过 5000 万，或 P95 查询延迟超过 2 秒时。
