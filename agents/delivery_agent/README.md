# delivery_agent — 履约发货 Agent

角色：**履约执行者**。处理"继续履约订单 X"的任务：查订单确认已支付 → 发货 →
发通知，结果写入长效记忆。

## 配套 Agent

与 `order_agent` 是一对，可组成串联流程：`order_agent` 卖出并转交，
`delivery_agent` 接手发货通知。参考场景 `scenarios/order_delivery_chain_clean.yaml`。

## 工具

| 工具 | 说明 | 副作用 |
|---|---|---|
| `get_order(order_id)` | 查订单详情 | 无 |
| `ship_order(order_id, carrier)` | 发货 | 有（发货流水） |
| `send_notice(order_id, notice_type, message)` | 发订单通知 | 有（通知流水） |

## 运行

```bash
.venv/Scripts/python.exe agent.py
```

由沙箱注入环境变量（网关地址 / 令牌 / 工具地址 / 模型 / 任务 / 记忆目录），
自己不需要配置。Agent 用纯标准库实现（urllib），不依赖第三方包。

## 守卫开关（受控实验用）

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `DEMO_AGENT_PAID_GUARD` | 1 | 发货前确认订单已支付 |
| `DEMO_AGENT_MEMORY_GUARD` | 1 | 记忆防重复发货/通知 |

关掉对应开关（设为 0）可制造缺陷变体，用于回归测试。
