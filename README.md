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

**W4：场景 DSL + 三方受控实验已跑通。** 7 个场景全部通过，其中三个跨会话场景构成
一个受控实验——脚本、任务、模型、工具声明一字不差，只差记忆回滚到哪儿、Agent 查不查
记忆，三种不同的结论（见下）。演示 Agent 现在有了真实记忆，"记得所以不重复"的正样本
第一次被证明，而不再只是"归因逻辑看起来对"。

**W5：成本统计与长期预算预警。** 至此需求里的 **MVP 四个模块全部跑通**（工具调用审计、
循环检测、跨会话记忆检查、成本统计）。成本这块的重点在**跨运行**：单次运行的账单随
场景报告出，长期预算看 `scripts/cost_report.py`。

```bash
.venv/Scripts/python.exe scripts/cost_report.py --period month   # 按天/周/月 + 按场景/Agent/模型
```

两条容易搞错的地方，都在 `docs/cost.md` 里写清了：**未定价的模型成本记 0，报告必须
标成"下界"而不是`$0.00`**（两者看起来一模一样，而后者会被读成"这次不要钱"）；**预算
超了不算这次运行失败**（那是累计花费，和被测 Agent 这次做没做错事无关，压成同一个
结论会让一次干净的运行因为上个月跑太多而报 fail）。

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
| `cross_session_remembered` | 全绿 —— **正样本**：记忆完整，Agent 认出来了没再付 |
| `cross_session_duplicate` | `reason_code=cross_session_duplicate` —— 记忆里有，Agent 该认出来 |
| `cross_session_memory_loss` | `reason_code=cross_session_memory_loss` —— 我们回滚过，**别动 Agent** |

每个场景输出四态统计、逐条预期的命中/未命中、以及"前提是否成立"。**前提没成立
时一律不通过**，无论检测器说了什么——那些结论是在一段不符合场景描述的轨迹上跑
出来的，不指向被测对象。`inconclusive` 计入覆盖率但不让运行失败。

还有一条判定独立于检测结论：**`expected_charges`**——目标系统侧到底扣了几笔。检测器
只看得见网络边界，而钱是在目标系统里扣的；边界干净、检测全绿、钱却扣了两次的情况
是存在的。两者分开验，对不上就一律不通过。

### 三方受控实验

三个跨会话场景的脚本、任务、模型、工具声明**一字不差**，只差两处：记忆回滚到哪儿、
Agent 查不查记忆。三个分支必须给出三种不同的结论，否则归因就是假的。

| 场景 | 记忆回滚到 | 查记忆 | 扣款 | 结论 |
|---|---|---|---|---|
| `cross_session_remembered` | 不回滚 | 是 | 1 | `pass` |
| `cross_session_memory_loss` | 付款**之前** | 是 | 2 | `fail` → 修记忆的恢复/传递 |
| `cross_session_duplicate` | 付款**之后** | **否** | 2 | `fail` → 修 Agent |

**回滚这个动作本身就决定了归因**，不需要事后猜：`memory_loss` 抹掉了付款那段，记忆
视野是 `-1`，Agent 查了但**不可能知道**；`duplicate` 保住了那段，视野是 `0`，记忆就
在那儿**它本可以知道**。演示 Agent 的 stdout 会打印读到了几条记录作为直接证据。

256 个单测覆盖网关、代理、记录器、规范化、两个检测器、记忆快照、跨会话归因、
被测 Agent 的记忆模块、成本聚合与预算、场景 DSL（加载校验 / 预期比对 / 前提判定 /
地面真值），以及 Web 界面（路径白名单 / 先校验后落盘 / 并发拒绝 / 结论标签）。

## 界面

```bash
./.venv/Scripts/python.exe scripts/serve_ui.py     # → http://127.0.0.1:8765
```

首页列场景与最近运行，能改场景（YAML 直接编辑）、能发起运行、能看报告（检测结论 /
工具调用审计 / 记忆快照 / 成本）与成本看板。**零 JavaScript**——轮询用
`<meta http-equiv="refresh">`，保存用 form POST，所以没有构建步骤。

| 约束 | 为什么 |
|---|---|
| **只监听回环地址**，不给远程开关 | 这个服务没有任何认证，而它能写场景文件、能拉起子进程。要远程用 `ssh -L 8765:127.0.0.1:8765 <主机>`，那条路上至少有认证 |
| **一次只跑一个场景**，第二个请求返回 409 | 执行器会改进程级的 `settings.tool_upstream_base_url`（`runner.py:342`）。两次运行重叠会互相把对方的代理指向错误的 mock，**两边都报出看起来正常的报告，而数据是串的**。拒绝比排队好：排队意味着用户等了十分钟才发现排的是错的 |
| **保存前先校验 YAML** | 全部用例都在这些文件里，写坏一个等于弄丢一个用例。校验不过就回显用户刚敲的内容并拒绝落盘 |
| **场景标识走白名单**（`api/paths.py`） | 从 URL 取来的字符串会被拼进文件路径，不校验就是一个任意文件写 |

报告的结论标签有**三种**，不是两种：`通过` / `不通过` / `未留存`。
`TestRun.verdict_ok` 可空，**NULL 不是 False**——它是加这个字段之前跑的历史运行，
没有留下结论。显示成红叉等于替一条没有结论的运行编一个结论出来。详见 `docs/ui.md`。

**已知缺口（不是疏忽，是排期）：**

| 缺口 | 影响 | 为什么先欠着 |
|---|---|---|
| 演示 Agent 的记忆不是真实方案 | 它用的是一个文件型 JSONL（`agents/react_demo/memory.py`），不是 LangGraph checkpointer / 向量库 | 沙箱只需要记忆落在 `SANDBOX_AGENT_MEMORY_DIR` 里，不关心是哪种。换真实方案是"写一个适配器"的量级，但它会同时把这条演示从"可复现"推向"依赖模型行为"——先把手上的链路验完 |
| 记忆守卫是加出来的开关 | `DEMO_AGENT_MEMORY_GUARD` 在真实 Agent 里不存在，是为造对照加的 | 上游被脚本化之后，记忆驱动的决策没法由 LLM 演出来，只能落到 Agent 代码里。这是受控实验的代价，不是想假装它真实 |
| 场景文件有重复样板 | 每个场景都要重抄一遍工具声明 | 先有 N 份能跑的场景、看清哪些字段真的总是重复，再决定用 YAML 锚点还是 include。现在抽象是猜 |
| mock 的幂等开关是全局的 | "工具 A 幂等、工具 B 不幂等"表达不出来 | mock 里目前只有一个写工具。加第二个时必须改成按工具控制 |
| 网关不支持流式响应 | `stream=true` 返回 501 | 流式要边转发边累计 usage，结算正确性另有一套逻辑，不走通不发布 |
| 熔断发生在副作用**之后** | 能阻止下一次调用，阻止不了当前这次 | 要"连这次都不发生"得做预测式放行。先证明熔断通路真能终止进程，再决定预测值不值得做 |
| 看门狗每次工具调用全量重扫 | O(n²) | walking skeleton 量级（几十次调用）无感；真实场景改滑动窗口即可 |
| 记忆漂移 / 记忆中毒检测 | 未实现 | P1，且依赖语料（`tests/corpus/`）才能校准精度 |
| 成本不能按**工具**拆解 | 需求 3.5 要求按 Agent/工具/场景三个维度拆解，只做了 Agent 和场景 | 工具调用本身不花 token，token 花在 LLM 上。沿时间线配对是个错的近似：工具**返回的内容**会留在上下文里，让后续每次调用继续为它付钱。硬报一个数，用户会拿它去砍工具，而那个数是错的。真要做方向是"量上下文占用 × 剩余轮次"，是另一个模块 |
| `stub-model` 没单价，演示跑出来的成本全是下界 | 场景报告里每次都带一句"金额是下界" | 演示走的是脚本化上游，没有真实计费。这恰恰是**故意留着**的：它证明未定价这条路径会被如实报出来，而不是静默显示 `$0.00` |

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
| `docs/scenarios.md` | 怎么写一个场景；跨会话归因的判据；`expected_charges`；三方受控实验；为什么 `is_write` 没有默认值 |
| `docs/cost.md` | 长期预算 vs 单次上限的分工；为什么预算不进 findings；未定价为什么必须标成"下界" |
| `docs/ui.md` | 界面为什么只绑回环、为什么一次只跑一个、`verdict_ok` 为什么可空、先校验后落盘 |

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
├─ mock/           Mock 工具服务器
│   ├─ toolspec.py    工具声明：是否写、幂等参数、是否真的强制幂等
│   │                 **跑场景时这不是真相源**——声明在场景 YAML 的 `tools:` 里，
│   │                 这里只是没声明工具表时的内建默认（walking skeleton 走这条）
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
├─ cost/           成本聚合（需求场景 4）——跨运行，不是单次
│   └─ aggregator.py  按运行/会话/场景/Agent/模型/周期拆解；**一律从明细重算**
├─ db/             模型 + 引擎（SQLite；system of record，ADR 002）
│
├─ scenario/       场景 DSL：用例是数据，不是代码
│   ├─ spec.py        pydantic 模型；`is_write` 刻意不给默认值
│   ├─ loader.py      YAML 加载 + 两条跨字段校验（记忆回到未来 / 调未声明的工具）
│   ├─ expect.py      预期比对：matched / missing / unexpected / uncovered
│   └─ runner.py      执行器：起服务 → 逐会话跑 Agent → 检测 → 比对 → 报告
│                     另含 `charges_ok`：独立于检测结论的地面真值
├─ session/        Session 序列追踪器             ← 占位，能力已在 trace/ 与 memory/
└─ api/            面向用户的 Web 界面（Phase 1d）—— **服务端渲染，不是 React**
    ├─ app.py        路由：页面 + 一组只读 JSON
    ├─ paths.py      场景标识校验与文件定位。**整个 Web 层唯一碰磁盘的地方**
    ├─ jobs.py       一次只跑一个（并发会让两个运行互相改掉对方的上游地址）
    ├─ queries.py    界面要读的东西。与 HTTP 无关，可单独测
    ├─ templates/    Jinja2 模板。自动转义是**引它的理由**，不是排版
    └─ static/       手写 CSS，无 CDN、无框架

agents/react_demo/    被测 Agent（独立 venv，装不上 sandbox 包）
├─ agent.py           两个工具 + 记忆守卫；**不 import 任何 sandbox.***
└─ memory.py          文件型记忆，只用标准库。目录由 SANDBOX_AGENT_MEMORY_DIR 指定
scenarios/            场景文件（YAML）。加用例 = 加文件，不改代码
scripts/              入口脚本：run_scenario.py（跑用例）、serve_ui.py（起界面）、
                      cost_report.py（看长期花费）、run_walking_skeleton.py
                      （跨进程证明，给人手工排查用）
tests/                256 个单测；tests/corpus/ 是留给精度验收的标注语料目录
docs/decisions/       ADR 001~003
```

`trace/` 里的 OTel 归一化层**还没写**——目前只有自研的规范化。
`session/` 是目录占位，不要按它已存在去排期。

**需求第 250 行写的前端栈是 React + TS + Vite，这里没有照做。** 那是个合理的长期
选择，但它要求目标机器上有一条 Node 构建链——而本工具的 lite profile 恰恰是
"7.9GB 内存、不装 Docker、一条命令跑起来"。为了一个本地测试工具拉一整条前端工具链，
换来的东西用户看不见。数据层（`api/queries.py`）与 HTTP 无关，将来真要换 React，
**被替换掉的是模板，不是接口**。

`tests/test_agent_memory.py` 用 `importlib` **按路径**加载 `agents/react_demo/memory.py`，
不是当包导入——这个别扭的加载方式是故意的。那个文件属于被测 Agent，跑在另一个 venv 里，
物理上看不到沙箱；测试不为了自己方便去动它的这一性质。哪天有人把它挪进 `src/sandbox/`，
这些测试会一起红。

**port 由操作系统分配，不是固定的 4100/4101/4102。** 场景执行器自己 bind 一个
空闲端口再交给 uvicorn，跑完把 `settings.tool_upstream_base_url` 指回原值。固定
端口在并发跑两个场景时会抢，而抢输的那个报的是"服务启动失败"——一个和场景
毫无关系的错。

`scripts/run_walking_skeleton.py` 仍用固定端口，因为那是给人手工排查用的。

**改过模型之后不必再删 `data/sandbox.db`。** `init_db()` 会在建表之后补上缺失的列
（`db/engine.py` 的 `_add_missing_columns`），实测在 44 条历史运行的库上直接加列成功，
老行的新列为 NULL。

**但那不是迁移系统**：只加列，不改类型、不改名、不删列、不回填、不记版本。新加的可空列
是安全的；**NOT NULL 且无 server_default 的列加不上去**——给历史行编一个值等于伪造事实，
所以那种改动要走真的迁移（Alembic 已在依赖里但尚未启用）。dev 库仍可丢弃。

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
