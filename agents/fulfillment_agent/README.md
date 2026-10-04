# fulfillment_agent —— 履约长链路被测 Agent

一个 LangGraph ReAct Agent，业务是一条完整的订单履约链：

**查单（get_order）→ 查库存（check_stock）→ 支付（create_payment）→ 发货（ship_order）→ 通知（send_notice）**

它和 `react_demo` 的关系是刻意设计的：后者只测"查单 + 支付"两步链，这里把链拉长到
五步，让这些在两步链上看不出来的东西变成可测的形状：

- **长链路工具链**：五步全走完且每步都有副作用，`expected_charges` 之外还有发货
  流水、通知流水两张账（`/_mock/shipments`、`/_mock/notices`）。
- **中途失败怎么传播**：例如在支付之后、发货之前崩掉，下一次会话从记忆里能不能
  看出"已付未发"。
- **步骤依赖**：发货依赖支付的**世界状态**（`DEMO_AGENT_SHIP_GUARD`），库存依赖
  静态库存表（`DEMO_AGENT_STOCK_GUARD`）。

它保持和 `react_demo` 相同的两条硬约束（不 import `sandbox.*`、不知道自己在被观测），
所以 `agents/react_demo/README.md` 里的约束表在这里原样成立。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `SANDBOX_GATEWAY_BASE_URL` | `http://127.0.0.1:4100/v1` | 模型端点 |
| `SANDBOX_AGENT_API_KEY` | 空 | 沙箱令牌，不是上游真实密钥 |
| `SANDBOX_TOOL_BASE_URL` | `http://127.0.0.1:4101` | 工具后端 |
| `SANDBOX_AGENT_MODEL` | `stub-model` | 模型名 |
| `SANDBOX_AGENT_TASK` | `帮我把订单 A1 履约掉（支付并发货并通知）。` | 任务文本 |
| `SANDBOX_AGENT_MEMORY_DIR` | `./.agent-memory` | 记忆目录（沙箱注入） |
| `DEMO_AGENT_MEMORY_GUARD` | `1` | `0` = 关掉记忆守卫（防重复）。**只为造受控对照** |
| `DEMO_AGENT_DUE_GUARD` | `1` | `0` = 付款前不查到期 |
| `DEMO_AGENT_SHIP_GUARD` | `1` | `0` = 发货前不确认订单已支付 |
| `DEMO_AGENT_STOCK_GUARD` | `1` | `0` = 发货前不查库存 |
| `DEMO_AGENT_STREAM` | 空 | `1` = 流式请求 |
| `DEMO_AGENT_MEMORY_TIDY` | 空 | 醒来先整理记忆：`archive` / `prune` |

守卫开关与 `react_demo` 同一哲学：真实 Agent 里这些决策由模型/业务逻辑做出，上游被
脚本化之后只能落到 Agent 代码里。默认全开 = "做之前先回忆/先确认"是正确行为；关掉
才是缺陷变体。

## 装

```bash
python -m venv agents/fulfillment_agent/.venv
PYTHONUTF8=1 agents/fulfillment_agent/.venv/Scripts/python.exe -m pip install \
    -r agents/fulfillment_agent/requirements.txt
```

## 记忆

复用 `agents/react_demo/memory.py`（文件型 JSONL，只用标准库）。记忆键的命名空间按工具分：

- `create_payment:{order_id}` —— 付过这笔
- `ship_order:{order_id}` —— 发过这批
- `send_notice:{order_id}:{notice_type}` —— 发过这条通知
- `order_note:{order_id}` —— 订单备注（记忆中毒的注入面，与 react_demo 相同）

记录时机是"副作用返回成功之后"（dual-write 的"后写"，与 react_demo 相同）。

## 怎么写新场景

场景 YAML 的 `agents:` 列表里声明：

```yaml
agents:
  - version: fulfillment_agent@0.1
    directory: fulfillment_agent
    task: 帮我把订单 A1 履约掉（支付并发货并通知）。
    model: stub-model
```

工具声明用 `scenarios/_shared/tools.yaml` 里的 `use:` 引用（`get_order` /
`check_stock` / `create_payment` / `ship_order` / `send_notice` 都已入库）。

场景判据除了 `expected_charges`（扣款笔数），还可以用发货/通知流水对账——
`expected_shipments` / `expected_notices` 目前由场景作者在 `note` 里写明，引擎侧的
地面真值仍以 `/_mock/charges` 为准；发货/通知流水端点已提供（`/_mock/shipments`、
`/_mock/notices`），需要时可在验收脚本里读取。
