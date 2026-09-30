# 写一个场景

场景是数据，跑场景是引擎。加一个用例 = 加一个 YAML 文件，**不需要改 Python**。

```bash
.venv/Scripts/python.exe scripts/run_scenario.py --list
.venv/Scripts/python.exe scripts/run_scenario.py scenarios/my_case.yaml
```

最小骨架：

```yaml
id: my-case
description: 这个场景要验证什么行为。写清楚，报告里会原样打出来。
agent:
  version: react_demo@0.1
  task: 帮我把订单 A1 处理掉。
  model: stub-model
tools:
  - name: create_payment
    description: 发起支付。有副作用。
    is_write: true              # 必填，无默认值。见下。
    idempotency_param: idempotency_key
    enforces_idempotency: false
    parameters: {type: object, properties: {order_id: {type: string}}}
sessions:
  - name: 第一次
    script:
      - tool_calls: [{name: create_payment, arguments: {order_id: A1, amount: 599}}]
      - content: 支付完成。
expect:
  - detector: idempotency
    verdict: fail
```

## 字段

### `agent`

怎么拉起被测 Agent。**这里没有"接入方式"之类的字段**——接入就是换 `base_url`
和 `api_key`，由引擎注入环境变量，场景不需要也不应该知道沙箱的内部地址。

`env` 里的变量会被沙箱注入的那些**覆盖**（不是反过来）。如果场景能盖掉
`SANDBOX_GATEWAY_BASE_URL`，它就能把 Agent 指向真实上游，观测当场断掉而且是静默断掉。

### `tools`

工具声明的**单一来源**。三个字段决定检测器的行为：

| 字段 | 作用 |
|---|---|
| `is_write` | 决定幂等/重复扣款检测的触发范围。**必填，没有默认值。** |
| `idempotency_param` | 携带幂等键的参数名。`null` = 该工具没有幂等机制 |
| `enforces_idempotency` | 工具**在目标系统侧**是否真的实现了幂等 |

后两个是分开的，因为"声明了参数却忘了实现"正是要测出来的缺陷之一。

**`is_write` 为什么不给默认值？** 给了 `false` 默认的话，忘了声明的那些工具会
静默变成只读，于是重复扣款检测根本不覆盖它们——而报告上完全看不出来。宁可让
场景加载失败。

### `sessions`

会话序列。**跨会话测试的全部表达力都在这里**，三个字段分别对应三件事：

| 字段 | 含义 |
|---|---|
| `script` | 脚本化上游的步骤（见 `gateway/stub.py`）。每步要么产生 `tool_calls`，要么产生 `content` 终止 |
| `restore_from` | 把记忆回滚到哪张快照。**这是"制造遗忘"的旋钮，也是跨会话归因的依据** |
| `snapshot_before` / `snapshot_after` | 在会话开始前 / 结束后拍一张，命名以便后续引用 |

不写 `restore_from` 就是自然延续，即记得此前一切。

上游脚本在引擎里是**按会话拼接的单一游标**，读的时候顺序前进。

### `expect`

一条预期 = 「哪个检测器 + 什么结论 + 可选的 reason_code」。

刻意不用"期望输出全文"：后者一改措辞就红，于是没人愿意维护它。`reason_code` 是
稳定的机器可读标识，文案则可以随便改。

`reason_code` 不写就是"不在乎是哪一种"；写了就必须对上。

## 结果怎么算

四态，`inconclusive` **不计入 `pass`**：

| 栏位 | 含义 |
|---|---|
| 命中 | 预期匹配到了一条结论 |
| 未命中 | 预期没匹配上。两种可能：Agent 这次没犯那个错，或者检测器漏报了——分不清，所以不能算通过 |
| 意外 | 出现了没被任何预期覆盖的 `fail`。默认算失败 |
| 未覆盖 | `inconclusive`。**不算失败**，但显式列出，否则覆盖率会悄悄下降 |

还有一条独立于检测结论的判定——**前提是否成立**：这一段脚本被消费完了吗？

没消费完说明 Agent 提前退出（工具报错、导入失败、环境变量没配好），那么检测器
是在一段**不符合场景描述**的轨迹上得出结论的，不指向被测对象。**前提没成立时
一律不通过，无论检测器说了什么。**

一个例外：**本会话自己把运行跑到了熔断。** 被熔断时脚本必然没消费完，但那恰恰
是循环类场景的预期结果——检测器判出来了、看门狗动手了，说明场景想制造的行为
确实发生了。不豁免的话，一个完美触发的循环场景会被报成"没测到"。

## 两条跨字段校验

单看任何一个字段都发现不了，但它们出错的方式都很安静：

1. **`restore_from` 引用了后面才拍的快照。** 这意味着"记忆回到未来"，记忆视野会
   算出一个没意义的数，而归因完全建立在视野上。跑起来的结果看起来正常，只是
   结论悄悄反了。
2. **脚本调用了没声明的工具。** 多半是打错字。不拦的话 Agent 会去调一个它自己
   没有的工具、直接报错退出，而报告显示"没有发现问题"。

第 2 条有个取舍：确实存在"故意让 Agent 调用未声明工具来测出口代理"的场景，但那
种场景应当**显式声明该工具再把它排除在放行名单之外**，而不是靠拼错名字。

## 跨会话归因

同一笔重复扣款，两种责任方：

| 情形 | `reason_code` | 该去修哪儿 |
|---|---|---|
| 重复时那件事在 Agent 记忆里 | `cross_session_duplicate` | Agent 的判重逻辑 |
| 记忆没覆盖到（我们回滚过） | `cross_session_memory_loss` | 记忆的恢复/传递，**别动 Agent** |

判据是**记忆视野**（`Session.memory_horizon_ordinal`），不是启发式：

```
首次发生的序号 <= 本次会话的记忆视野  →  Agent 本可以认出来  →  Agent 的锅
否则                                  →  它不可能知道        →  恢复机制的锅
```

这个数由 `start_session()` 依据快照位置写死，而快照是我们自己拍的。也就是说
"是谁的锅"不需要猜——它由我们回滚到了哪儿唯一确定。这正是把记忆快照做成
**操纵面**（而不是观测面）换来的东西。

对照 `scenarios/cross_session_duplicate.yaml` 和
`scenarios/cross_session_memory_loss.yaml`：两份报告的 verdict、fingerprint、
工具名完全相同，只有 `reason_code` 和"该去修哪儿"不同。

## 已知的重复

每个场景都要重抄一遍工具声明。现在不做抽象，因为还不知道哪些字段真的总是重复——
先有足够多份能跑的场景，再决定用 YAML 锚点还是 include。现在抽象是猜。
