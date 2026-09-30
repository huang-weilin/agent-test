# react_demo —— 被测 Agent

一个标准的 LangGraph ReAct Agent，两个工具（`get_order` 只读、`create_payment` 有副作用）。
它存在的唯一目的是回答一个问题：

> **一个自己不配合的 Agent，能不能在完全不知情的情况下被完整观测？**

答案必须是"能"，而且这个"能"不能靠它自觉。所以这个目录里有两条性质是硬约束，
改代码时不要破坏：

| 约束 | 怎么保证的 | 破坏了会怎样 |
|---|---|---|
| 不 import 任何 `sandbox.*` | 独立 venv，**里面没装 sandbox 包**，物理上 import 不到 | 观测变成"Agent 自己上报的"，ADR 001 的前提作废 |
| 不知道自己在被观测 | 只读 4 个环境变量：模型端点、凭据、工具端点、任务文本 | 它会开始"表现良好"，测到的就不是真实行为 |

第二条尤其容易在不经意间破坏。所以这个文件里**没有**主机白名单、没有重试策略、
没有幂等键管理——那些都是 **Agent 自己的决策，也正是被测对象**。框架替它做了，
测的就是框架自己写的那套策略了。

## 装

```bash
# 在仓库根目录。注意是独立 venv，不是框架的 .venv
python -m venv agents/react_demo/.venv
PYTHONUTF8=1 agents/react_demo/.venv/Scripts/python.exe -m pip install \
    -r agents/react_demo/requirements.txt
```

**这个 venv 是刻意的设计，不是省内存。** 拿到一个任意版本的 LangChain/CrewAI 时，
得能分清是 Agent 的问题还是环境被污染了。

## 跑

单独跑它意义不大——它需要一个网关和一套 mock 工具在后面。日常用演示脚本：

```bash
# 在仓库根目录，脚本会起服务、申请运行、签发令牌、拉起本 Agent、再出报告
.venv/Scripts/python.exe scripts/run_walking_skeleton.py --scenario clean
.venv/Scripts/python.exe scripts/run_walking_skeleton.py --scenario duplicate
.venv/Scripts/python.exe scripts/run_walking_skeleton.py --scenario loop
```

要手工跑，就自己起三个服务（`uvicorn sandbox.gateway.app:app --port 4100` 等），
申请一次运行拿令牌，然后：

```bash
cd agents/react_demo
SANDBOX_GATEWAY_BASE_URL=http://127.0.0.1:4100/v1 \
SANDBOX_AGENT_API_KEY=<申请运行返回的 api_key> \
SANDBOX_TOOL_BASE_URL=http://127.0.0.1:4101 \
SANDBOX_AGENT_MODEL=stub-model \
SANDBOX_AGENT_TASK="帮我把订单 A1 处理掉。" \
.venv/Scripts/python.exe agent.py
```

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `SANDBOX_GATEWAY_BASE_URL` | `http://127.0.0.1:4100/v1` | 模型端点。**这就是接入沙箱要改的那个值** |
| `SANDBOX_AGENT_API_KEY` | 空 | 沙箱令牌 `sk-sandbox-<run_id>`，**不是**上游真实密钥 |
| `SANDBOX_TOOL_BASE_URL` | `http://127.0.0.1:4101` | 工具后端，指向出口代理而不是业务后端 |
| `SANDBOX_AGENT_MODEL` | `stub-model` | 模型名。走真实上游时填真实模型名 |
| `SANDBOX_AGENT_TASK` | `帮我把订单 A1 处理掉。` | 任务文本 |
| `SANDBOX_AGENT_MEMORY_DIR` | `./.agent-memory` | 记忆目录。**由沙箱创建并注入**，见下 |
| `DEMO_AGENT_MEMORY_GUARD` | `1` | `0` = 关掉记忆守卫。**只为造受控对照而存在** |
| `DEMO_AGENT_MEMORY_TIDY` | 空 | 醒来先整理记忆：`archive` = 挪进 `archive/`，`prune` = 直接删掉。**同上，只为造受控对照** |

`API_KEY` 为空时直接退出并打印申请运行的 curl，不静默降级。

## 记忆

`memory.py` 是文件型记忆：一行一条 JSONL，记"我做成了哪些有副作用的操作"。
它**只用标准库**——这个目录里的东西都不能依赖框架侧的包。

**目录由沙箱指定，Agent 不自己猜。** 猜错了，测试框架的回滚就会回滚到一处它
根本不读的地方，而报告会显示"回滚过了它还是重复扣款"——把 Agent 冤枉成不读记忆。
真实 Agent 用自己的记忆方案（checkpointer、向量库、业务库）都行，沙箱不关心是哪
种，只要求它落在 `SANDBOX_AGENT_MEMORY_DIR` 里。

两个刻意的设计，都是为了让跨会话实验成立：

- **不写时间戳。** 同一段经历每次产出同样的字节，快照才能内容寻址；否则"这次恢复
  其实什么都没变"就判断不出来了。
- **守卫查的是会话开始时加载的那份长效记忆，会话内不更新。** 会话内重复靠对话上下文
  管，是另一个机制。混在一起就分不清哪个在起作用——而 `duplicate.yaml`（同一会话里
  连付两次）正是靠"守卫不拦会话内重复"才演得出来。

`DEMO_AGENT_MEMORY_GUARD=0` 用来造对照：同一个 Agent、同一段脚本，记忆内容一模一样，
一个查一个不查。`cross_session_remembered`（查、认出来了）和 `cross_session_duplicate`
（不查、又付一次）之间只差这一个变量。**真实的被测 Agent 不该有这个东西**，它是
为了把"该不该由 Agent 负责"变成可复现的实验才加进来的。

`memory.tidy()` 是**记忆维护**：醒来时整理一遍。`DEMO_AGENT_MEMORY_TIDY=archive` 把
根目录下的文件挪进 `archive/`（内容一个字节都不改），`=prune` 直接删掉。它排在回忆
之前——**先收拾再干活**，于是"整理掉了、等会儿就想不起来"这件事才会发生。加它的
理由和守卫一样：记忆检测里"声明的路径不在了"那两条分支（内容还在 = warning / 内容
也没了 = fail）需要有活样本，而没有删除/改名这个动作就没人能把一份记忆弄没。
`memory_tidy_archived` 与 `memory_tidy_pruned` 是那一对。同样地：**真实的被测 Agent
不该有这个开关**，它对应的是两份不同的记忆维护实现。

记录时机是"副作用返回成功之后"，这中间有个窗口（调用成功了但还没写就崩了 → 那笔被
忘掉 → 下次重做）。先写后调用则会在失败时误记成做过了。这是 dual-write 问题，没有
免费的解，演示 Agent 选"后写"因为它对"重复扣款"这个缺陷偏保守。真实系统要用事务、
幂等键或 outbox 来解——**而那正是被测对象的一部分，沙箱不替它决定。**

## 换成真实环境

三个值分别换成 OpenAI/DeepSeek 的地址、真实密钥、真实业务后端，**这个目录的代码
一行都不用改**。真实密钥能进得来，是因为真实场景下这个 Agent 是你自己的；
在沙箱里它拿到的必须是令牌，真实上游密钥永远留在网关进程里（ADR 003）。

`agent.py` 里 `streaming=False` 是给沙箱网关用的——网关目前不支持流式，`stream=true`
会返回 501。走真实上游时删掉这行即可（见根 README 的已知缺口表）。

## 怎么写新场景

复制这个目录，改工具和任务，**保持那两条硬约束**。工具调用只走 `_call_tool_raw`
这一个出口，它带上工具名 POST 到 `{TOOL_BASE_URL}/tools/{name}`。

新增工具时**不用改沙箱代码**：在场景 YAML 的 `tools:` 块里声明它是不是写操作、
幂等参数叫什么、目标系统侧有没有真的实现幂等——检测器靠这份声明分辨"重复读"和
"重复扣款"。工具声明的单一来源是场景，不是 Agent，也不是 mock
（见 `docs/scenarios.md`）。
