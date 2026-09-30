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

`API_KEY` 为空时直接退出并打印申请运行的 curl，不静默降级。

## 换成真实环境

三个值分别换成 OpenAI/DeepSeek 的地址、真实密钥、真实业务后端，**这个目录的代码
一行都不用改**。真实密钥能进得来，是因为真实场景下这个 Agent 是你自己的；
在沙箱里它拿到的必须是令牌，真实上游密钥永远留在网关进程里（ADR 003）。

`agent.py` 里 `streaming=False` 是给沙箱网关用的——网关目前不支持流式，`stream=true`
会返回 501。走真实上游时删掉这行即可（见根 README 的已知缺口表）。

## 怎么写新场景

复制这个目录，改工具和任务，**保持那两条硬约束**。工具调用只走 `_call_tool` 这一个
出口，它会带上工具名 POST 到 `{TOOL_BASE_URL}/tools/{name}`；新增工具时在
`src/sandbox/mock/toolspec.py` 里同时声明它是不是写操作、幂等参数叫什么——
检测器靠这份声明分辨"重复读"和"重复扣款"。
