# order_agent — 售前下单受理 Agent

角色：**售前下单台**。处理"用户要买 X"的任务：查库存 → 创建支付（下单）→
调用 `handoff` 转交履约环节。

## 配套 Agent

与 `delivery_agent` 是一对，可组成串联流程：`order_agent` 卖出并转交，
`delivery_agent` 接手发货通知。参考场景 `scenarios/order_delivery_chain_clean.yaml`。

## 工具

| 工具 | 说明 | 副作用 |
|---|---|---|
| `check_stock(item)` | 查商品库存 | 无 |
| `create_payment(order_id, amount)` | 创建支付（下单） | 有（扣款流水） |
| `handoff(target_role, task)` | 转交履约环节 | 有（交接流水） |

## 运行

```bash
.venv/Scripts/python.exe agent.py
```

由沙箱注入环境变量（网关地址 / 令牌 / 工具地址 / 模型 / 任务 / 记忆目录），
自己不需要配置。Agent 用纯标准库实现（urllib），不依赖第三方包。

## 守卫开关（受控实验用）

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `DEMO_AGENT_STOCK_GUARD` | 1 | 下单前查一次库存 |
| `DEMO_AGENT_MEMORY_GUARD` | 1 | 记忆防重复下单 |

关掉对应开关（设为 0）可制造缺陷变体，用于回归测试。
