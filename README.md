# AI Agent 行为级测试沙盒

独立、对抗性的 Agent 行为验证工具。定位：**不能用平台自带的护栏去证明平台自带的护栏有效**，所以我们对被测 Agent 做黑盒边界拦截，而不是把 SDK 注入它的进程。

需求源文档：`D:\360MoveData\Users\Administrator\Desktop\agent-test-require.txt`（v1.1）

## 当前状态

**W1–W2 walking skeleton 已跑通，三条验收标准全部成立。**

**W3：跨会话 + 记忆快照已跑通。** 需求里的用户故事是"用户周一让 Agent 订票，
周三再次要求时，Agent 没有识别出这是重复操作"。这一刀切下去，重点不是"能不能
测出重复"——那个不难——而是**同一次重复，两种归因**：

| 情形 | 结论 | 该去修哪儿 |
|---|---|---|
| 重复时那件事在 Agent 记忆里 | `cross_session_duplicate` | Agent 的判重逻辑 |
| 记忆没覆盖到（我们回滚过） | `cross_session_memory_loss` | 记忆的恢复/传递，**别动 Agent** |

两者都报 `fail`（钱确实扣了两次），但指向完全不同的地方去修。分不清的话，
报告就是在指挥用户去改一个没问题的东西——比不报还糟。

判据是**记忆视野**，不是启发式。记忆在 Agent 进程内，网络边界拦不到它，所以我们
把记忆做成**操纵面**：主动回滚到某个快照，看它会不会因此重复扣款。回滚到哪儿是
我们自己决定的，于是"它本可以知道吗"就有了确定答案，不用猜。

**场景是数据，不是 Python。** 用例写在 `scenarios/*.yaml` 里，跑用例的是引擎：

```bash
.venv/Scripts/python.exe scripts/run_scenario.py --list      # 有哪些场景
.venv/Scripts/python.exe scripts/run_scenario.py             # 跑全部
.venv/Scripts/python.exe scripts/run_scenario.py scenarios/cross_session_memory_loss.yaml
.venv/Scripts/python.exe scripts/run_scenario.py --json       # 机器可读
```

| 场景 | 预期结论 |
|---|---|
| `clean` | 全绿（负样本：证明检测器不是"凡跑必红"） |
| `duplicate` | `idempotency` / fail —— 无幂等保护的写操作被执行两次 |
| `missing_key` | `idempotency` / fail · `reason_code=missing_key` —— 工具声明了保护，调用侧不带键 |
| `loop` | `loop.strict` / fail + 带外熔断终止进程 |
| `cross_session_duplicate` | `reason_code=cross_session_duplicate` —— 记忆里有，Agent 该认出来 |
| `cross_session_memory_loss` | `reason_code=cross_session_memory_loss` —— 我们回滚过，**别动 Agent** |

每个场景输出四态统计、逐条预期的命中/未命中、以及"前提是否成立"。**前提没成立
时一律不通过**，无论检测器说了什么——那些结论是在一段不符合场景描述的轨迹上跑
出来的，不指向被测对象。`inconclusive` 计入覆盖率但不让运行失败。

162 个单测覆盖网关、代理、记录器、规范化、两个检测器、记忆快照、跨会话归因，
以及场景 DSL（加载校验 / 预期比对 / 前提判定）。

**已知缺口（不是疏忽，是排期）：**

| 缺口 | 影响 | 为什么先欠着 |
|---|---|---|
| 演示 Agent 是无状态的 | 跨会话演示里它必然重复，因为**它真的没有记忆**；`cross_session_duplicate` 证明的是归因逻辑，不是这个 Agent 会犯这个错 | 要跑通"记得所以不重复"的正样本，得给它接一个真实记忆（LangGraph checkpointer 适配器只差一个实现） |
| 场景文件有重复样板 | 每个场景都要重抄一遍工具声明 | 先有 N 份能跑的场景、看清哪些字段真的总是重复，再决定用 YAML 锚点还是 include。现在抽象是猜 |
| mock 的幂等开关是全局的 | "工具 A 幂等、工具 B 不幂等"表达不出来 | mock 里目前只有一个写工具。加第二个时必须改成按工具控制 |
| 网关不支持流式响应 | `stream=true` 返回 501 | 流式要边转发边累计 usage，结算正确性另有一套逻辑，不走通不发布 |
| 熔断发生在副作用**之后** | 能阻止下一次调用，阻止不了当前这次 | 要"连这次都不发生"得做预测式放行。先证明熔断通路真能终止进程，再决定预测值不值得做 |
| 看门狗每次工具调用全量重扫 | O(n²) | walking skeleton 量级（几十次调用）无感；真实场景改滑动窗口即可 |
| 记忆漂移 / 记忆中毒检测 | 未实现 | P1，且依赖语料（`tests/corpus/`）才能校准精度 |

`scripts/run_walking_skeleton.py` 还在，它是**跨进程**的证明（真 uvicorn、真端口、
真独立 venv 的 Agent 进程），场景硬编码在它自己的字典里。跑场景走
`run_scenario.py`——那个是引擎，场景来自数据。两个都留着：一个证明链路跨进程成立，
一个是日常跑用例的入口。

**覆盖率是刻意显示的。** 报告同时给出"覆盖率"和"已覆盖部分的失败率"——单看通过率的话，
一个什么都没测到的运行会显示全绿。

## 先读这几份，再动手

| 文件 | 为什么重要 |
|---|---|
| `docs/decisions/001-integration-surface.md` | **最重要的架构决策**。集成面走网络层黑盒拦截。原文档的"进程内注入 TimeProvider"对目标用户（Muse/LightVela 型云端 Agent）根本不可执行 |
| `docs/decisions/002-system-of-record.md` | 自研库是唯一真相源；Langfuse 只做可视化；ClickHouse 从 MVP 移除 |
| `docs/decisions/003-credential-default.md` | 出口默认拒绝、默认不注入真实凭据。默认值反转，别改回去 |
| `docs/detectors.md` | 检测器的精度目标、豁免机制、**四态结果语义**（`inconclusive` 不可折叠为 `pass`） |
| `docs/scenarios.md` | 怎么写一个场景；跨会话归因的判据；为什么 `is_write` 没有默认值 |

## 环境约束（本机）

| 项 | 值 | 影响 |
|---|---|---|
| 内存 | 7.9 GB（空 1 插槽，最大 32GB） | **跑不动完整栈**。本机用 lite profile，重型组件上远程 Linux |
| E: 盘 | **exFAT** | 不能承载 WSL2 的 vhdx；不支持权限位；文件监听不可靠（HMR 可能失效） |
| C: 盘 | 仅 ~35 GB 空闲 | Docker 镜像不要落 C:（若将来启用） |
| D: 盘 | NTFS，~29 GB 空闲 | **NTFS** —— 需要 NTFS 语义的东西放这里 |

venv 在 exFAT 上实测可用（建环境 26 秒），但**未来启用 WSL2 / Docker Desktop 时，数据盘必须放 NTFS（C: 或 D:）**，E: 只能放源码和大体积产物。

## 本机 lite profile

不需要 Docker。mock 服务器、LLM 网关、被测 Agent 都是本地 Python 进程。

```bash
# 框架自身环境
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[dev]"

# 被测 Agent 用独立环境（关键：隔离，不是省内存）
# 否则你要测任意版本的 LangChain/CrewAI 时，永远分不清是 bug 还是环境被污染
python -m venv agents/react_demo/.venv
```

**两个 venv 是刻意的设计**，不是权宜之计。进程级隔离替代容器隔离，上层接口按"可替换为容器"来设计。

## 目录

```
src/sandbox/
├─ config.py       配置（lite profile 默认值）
├─ runtime.py      运行注册表 + 沙箱令牌（run_id 的路由与凭据隔离都在这里）
├─ watchdog.py     熔断：跑检测器 → 落库 → **然后**才杀进程
├─ gateway/        LLM 网关：Token 计数权威点 + 硬预算熔断
│   ├─ app.py         /v1/chat/completions：预留 → 放行 → 结算
│   ├─ budget.py      预留/结算（先扣后放，失败要还回去）
│   ├─ pricing.py     单价表，成本在网关内部算，不信调用方
│   └─ stub.py        脚本化上游，演示与测试用（不是降级路径）
├─ proxy/          工具出口代理：默认拒绝 + 凭据注入
│   ├─ app.py         /tools/{name} 走白名单；/fetch 走不可信 URL 的默认拒绝
│   └─ policy.py      allowlist 规则，构造时就拒绝通配符主机
├─ mock/           Mock 工具服务器（Python 工具规格，非 YAML）
│   ├─ toolspec.py    工具声明：是否写、幂等参数、是否真的强制幂等
│   └─ server.py      扣款账本（记每一笔尝试，不只记最终状态）
├─ trace/          轨迹记录
│   ├─ recorder.py    网关与代理唯一的写入路径；统一 seq 时间线；会话切换
│   ├─ normalize.py   严格层（结构哈希）+ 近似层（相似度）两层参数比较
│   └─ reader.py      出库成 CallView；单次运行 / 整个会话序列两种读法
├─ detectors/      规格见 docs/detectors.md
│   ├─ base.py        CallView / Finding / 豁免（写操作默认不可豁免）
│   ├─ idempotency.py 重复扣款，含"声明了保护但没生效"与跨会话归因
│   └─ loop.py        严格 / 近似 / 乒乓 / 跨会话
├─ memory/         记忆快照——**操纵面，不是观测面**
│   ├─ snapshot.py    内容寻址的快照存储
│   └─ adapters/      记忆载体适配器（Hermes 文件型；checkpointer 待做）
├─ db/             模型 + 引擎（SQLite；system of record，ADR 002）
│
├─ scenario/       场景 DSL：用例是数据，不是代码
│   ├─ spec.py        pydantic 模型；`is_write` 刻意不给默认值
│   ├─ loader.py      YAML 加载 + 两条跨字段校验（记忆回到未来 / 调未声明的工具）
│   ├─ expect.py      预期比对：matched / missing / unexpected / uncovered
│   └─ runner.py      执行器：起服务 → 逐会话跑 Agent → 检测 → 比对 → 报告
├─ session/        Session 序列追踪器             ← 占位，能力已在 trace/ 与 memory/
└─ api/            面向用户的 FastAPI              ← 占位，未实现

agents/react_demo/    被测 Agent（独立 venv，装不上 sandbox 包）
scenarios/            场景文件（YAML）。加用例 = 加文件，不改代码
scripts/              跑真实进程的入口脚本
tests/                162 个单测；tests/corpus/ 是留给精度验收的标注语料目录
docs/decisions/       ADR 001~003
```

`trace/` 里的 OTel 归一化层**还没写**——目前只有自研的规范化。
`session/`/`api/` 是目录占位，不要按它们已存在去排期。

**port 由操作系统分配，不是固定的 4100/4101/4102。** 场景执行器自己 bind 一个
空闲端口再交给 uvicorn，跑完把 `settings.tool_upstream_base_url` 指回原值。固定
端口在并发跑两个场景时会抢，而抢输的那个报的是"服务启动失败"——一个和场景
毫无关系的错。

`scripts/run_walking_skeleton.py` 仍用固定端口，因为那是给人手工排查用的。

**改过模型之后记得删掉 `data/sandbox.db`。** 建表走的是 `create_all`，它只建
缺的表、不会给已存在的表加列（Alembic 已在依赖里但尚未启用）。dev 库是可丢弃的。

## W1–W2 walking skeleton：三条验收标准的结论

原计划是"只改 `base_url` 就能被完整观测"。**实测下来这个说法要修正**：任何 LLM 客户端都
还需要一个 `api_key` 才能发请求。所以准确的说法是——

> **不能只改 base_url，还要改 api_key；但这是配置变更，不是代码变更，也不需要注入 SDK。**

于是让 api_key 顺带承担运行标识（`sk-sandbox-<run_id>`），既解决路由，又顺手完成了凭据隔离：
**真实上游密钥只存在于网关进程内，永远不进被测 Agent 的环境。** 详见 `src/sandbox/runtime.py`。

| 标准 | 结论 | 证据 |
|---|---|---|
| 1. 边界拦截能完整观测 | ✅ | 真 LangGraph Agent，独立进程 + 独立 venv，只拿到网关地址和沙箱令牌；它的每次模型调用、每次工具调用都在库里，一条不少 |
| 2. 构造的重复扣款被捕获 | ✅ | mock 侧真的产生 2 笔扣款，幂等检测器报 `fail`；负样本（两个不同订单）不误报 |
| 3. 构造的循环被捕获且能终止 | ✅ | 脚本给 8 步，Agent 跑到第 3 次工具调用即被熔断；带外处理器真的终止了 Agent 进程（退出码 1） |

标准三的关键证据是**次数**：3 而不是 8。报警不等于止损，终止进程才是。

> 原文档的路线图是 Phase 1a→1d 自底向上堆层，第 10 周才有端到端的东西，集成风险全压在最后。这里刻意反过来：先竖切一条。

## 明确不做（MVP）

- **不自研** Trace 可视化、Token 成本统计、LLM-as-Judge —— Langfuse 已有，只做对接适配层。
- 不引入 ClickHouse、MinIO、本地 K8s（理由见 ADR 002 与本机约束）。
- 不做进程内 SDK 注入作为主线（ADR 001）。
