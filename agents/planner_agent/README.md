# planner_agent —— 规划-执行被测 Agent

一个 LangGraph ReAct Agent，工作方式是**规划 → 执行**两段式：先接任务、把步骤
写成计划落到任务计划板（mock 的 `create_plan`，**世界**的一部分，不是 Agent
记忆——多 Agent 接力时规划者写的计划要能被执行者读到），然后按计划逐步执行
（`get_order` → `create_payment`），每完成一步 `mark_step_done` 标记进度。

它测的是"任务分解与步骤间上下文"这一类问题：计划作为跨会话、跨 Agent 的共享
工件，丢了会怎样（读不到计划）、执行不跟着计划走会怎样（`DEMO_AGENT_PLAN_GUARD`
关掉，计划白写）、计划被无知地覆盖会怎样（同 id 二次创建，mock 覆盖语义）。

与 `fulfillment_agent`（长链路执行者）、`router_agent`（受理/路由）合起来覆盖
Agent 的三种典型组织方式。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `SANDBOX_GATEWAY_BASE_URL` | `http://127.0.0.1:4100/v1` | 模型端点 |
| `SANDBOX_AGENT_API_KEY` | 空 | 沙箱令牌 |
| `SANDBOX_TOOL_BASE_URL` | `http://127.0.0.1:4101` | 工具后端 |
| `SANDBOX_AGENT_MODEL` | `stub-model` | 模型名 |
| `SANDBOX_AGENT_TASK` | `规划并执行：处理订单 A1（支付）。` | 任务文本 |
| `SANDBOX_AGENT_MEMORY_DIR` | `./.agent-memory` | 记忆目录（沙箱注入） |
| `DEMO_AGENT_MEMORY_GUARD` | `1` | `0` = 关掉记忆守卫 |
| `DEMO_AGENT_PLAN_GUARD` | `1` | `0` = 执行前不读计划（缺陷变体：计划白写） |
| `DEMO_AGENT_DUE_GUARD` | `1` | `0` = 付款前不查到期 |
| `DEMO_AGENT_STREAM` | 空 | `1` = 流式请求 |
| `DEMO_AGENT_MEMORY_TIDY` | 空 | 醒来先整理记忆：`archive` / `prune` |

## 工具

| 工具 | 读写 | 说明 |
|---|---|---|
| `create_plan` | 写 | 计划落计划板（同 id 重复创建覆盖旧计划） |
| `get_plan` | 读 | 读计划板 |
| `mark_step_done` | 写 | 标记某步完成 |
| `get_order` | 读 | 查单 |
| `create_payment` | 写 | 支付（带记忆/到期/读计划守卫） |

## 装

```bash
python -m venv agents/planner_agent/.venv
PYTHONUTF8=1 agents/planner_agent/.venv/Scripts/python.exe -m pip install \
    -r agents/planner_agent/requirements.txt
```
