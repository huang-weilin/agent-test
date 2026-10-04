# router_agent —— 路由转接被测 Agent（受理台 + 路由 + 售后）

一个 LangGraph ReAct Agent，角色是**订单受理台**：接任务先判断归属，自己能做的
（查状态、退款+通知）自己做，属于别的环节的（支付、履约、催发货）调用 `handoff`
转交出去。

它是"多 Agent 接力"场景里的**分发者**：0 号（router）受理并转交，1 号
（fulfillment_agent）接手执行——转交动作在边界上可见（`/_mock/handoffs` 交接
流水），接手者的执行也在边界上可见（工具调用），报告能同时看到两边。

与 `fulfillment_agent`（长链路执行者）、`planner_agent`（规划-执行）合起来覆盖
Agent 的三种典型组织方式。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `SANDBOX_GATEWAY_BASE_URL` | `http://127.0.0.1:4100/v1` | 模型端点 |
| `SANDBOX_AGENT_API_KEY` | 空 | 沙箱令牌 |
| `SANDBOX_TOOL_BASE_URL` | `http://127.0.0.1:4101` | 工具后端 |
| `SANDBOX_AGENT_MODEL` | `stub-model` | 模型名 |
| `SANDBOX_AGENT_TASK` | `用户要求退款订单 A1，请处理。` | 任务文本 |
| `SANDBOX_AGENT_MEMORY_DIR` | `./.agent-memory` | 记忆目录（沙箱注入） |
| `DEMO_AGENT_MEMORY_GUARD` | `1` | `0` = 关掉记忆守卫 |
| `DEMO_AGENT_ROUTE_GUARD` | `1` | `0` = 转交前不确认（缺陷变体：乱转） |
| `DEMO_AGENT_STREAM` | 空 | `1` = 流式请求 |
| `DEMO_AGENT_MEMORY_TIDY` | 空 | 醒来先整理记忆：`archive` / `prune` |

## 工具

| 工具 | 读写 | 说明 |
|---|---|---|
| `get_order` / `list_orders` | 读 | 受理查单 |
| `create_refund` | 写 | 售后直接退款（带记忆守卫） |
| `send_notice` | 写 | 退款/处理后的通知（带记忆守卫） |
| `handoff` | 写 | 转交其它环节（产生交接流水，多 Agent 接力的交接证据） |

## 装

```bash
python -m venv agents/router_agent/.venv
PYTHONUTF8=1 agents/router_agent/.venv/Scripts/python.exe -m pip install \
    -r agents/router_agent/requirements.txt
```

## 场景声明示例

```yaml
agents:
  - version: router_agent@0.1
    directory: router_agent
    task: 用户要求处理订单 A1，请先受理并判断归属。
    model: stub-model
  - version: fulfillment_agent@0.1
    directory: fulfillment_agent
    task: 你接手继续履约订单 A1（支付并发货并通知）。
    model: stub-model
sessions:
  - name: 受理与转交
    script:
      - tool_calls: {name: handoff, arguments: {target_role: fulfillment, task: 订单 A1 履约}}
  - name: 接手履约
    agent: 1
    script: [...]
```
