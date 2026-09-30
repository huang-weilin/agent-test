# 标注语料（Phase 0 交付物）

检测器的精度目标写在 `docs/detectors.md` 第七节。**没有这个目录，那些数字就只是
占位**——既无法验收，也无法调优。这里是人手写下的轨迹和标注，`scripts/calibrate.py`
用它把"判得准不准"算出来。

```
.venv/Scripts/python.exe scripts/calibrate.py            # 逐条看分歧 + 精度/召回率
.venv/Scripts/python.exe scripts/calibrate.py --verbose  # 连命中的也逐条列出结论
.venv/Scripts/python.exe -m pytest tests/test_corpus.py  # 同样的事，进 CI
```

## 一条语料长什么样

```yaml
- id: loop-strict-repeat-read
  source: 自造                    # 白名单：自造 / 公开-PERSIST / 公开-GhostWriter / 访谈
  why: >                          # 标注依据。没有它，标注只是一句无法复核的断言
    同一个只读调用，参数一字不差地发了三次。这是确定性最高的循环……
  tools:                          # 工具的**声明**属性（不是观测）
    get_order: {is_write: false}
    create_payment: {is_write: true, enforces_idempotency: true, may_run_unattended: false}
  sessions:                       # 哪几段会话，谁在看着
    - {ordinal: 0}
    - {ordinal: 1, autonomous: true, memory_horizon: 0}
  calls:                          # 边界上看到的那串调用，按时间顺序（seq 自动编号）
    - {tool: get_order, args: {order_id: A1}}
    - {tool: get_order, args: {order_id: A1}, session: 1, status: error}
    - {tool: create_payment, args: {order_id: A1}, blocked_by_proxy: true}
    # 可写字段：status / blocked_by_proxy / replay_reason / idempotency_key / session
  exemptions:                     # 可选。写操作要静音必须 allow_write: true
    - {detector: idempotency, tool: create_payment, reason: 已人工确认, allow_write: true}
    # 和场景 YAML 里的 exemptions 是同一个形状（docs/detectors.md 第二节）：
    # 语料这边多写一份，是为了让"有豁免/没豁免"的对照样本自洽。
  expect:                         # 每一族都要表态
    loop.strict: fail
    idempotency: {verdict: fail, reason_code: cross_session_duplicate}
    autonomy: none                # 这一族不许报 fail/warning
  absent:                         # 可选：某个具体结论明令不许出现
    - {detector: autonomy, verdict: pass}
```

`calls` 里的 `seq`、指纹都由加载器生成：**指纹用的是生产那条路径**
（`sandbox.trace.normalize.fingerprint`），检测器也是生产那一套
（`sandbox.detectors.run_all`）。语料里判"同一操作"和生产里判"同一操作"必须是
同一个判据，否则精确率是在说另一套东西。

## 三条规矩

**一、每一族都要表态，少一族就加载失败。** 没写不等于不关心，等于没标——而没标的
数据混进精确率里，那个数字就变成一句关于没标的空话。不想表态的写法是 `none`。

**二、轨迹是手写的，不是从某次运行里录回来的。** 录回来的话，标注只能照着检测器
当时给的结论写，那是同义反复：检测器说什么，"期望"就是什么，精确率永远是 1.00。
手写才可能写下"我认为这是循环"这个**独立判断**，然后拿它去撞检测器。

**三、负样本才是重点。** 一个只会报警的检测器等于是没有检测器：天天误报的下一步
是被整体忽略，包括那些真正要紧的。所以 `loop-monotonic-paging-is-not-a-loop`、
`loop-blocked-calls-do-not-count`、`auto-user-present-is-not-autonomy` 这些"什么
都不该报"的条目和正样本同等重要。

## 现在有什么

31 条：`loop.yaml` 12 条、`idempotency.yaml` 11 条、`autonomy.yaml` 8 条。
来源分布：自造 28、公开-PERSIST 2、公开-GhostWriter 1。

标着公开来源的那几条，复现的是**威胁类型**（比如"被边界拦下的反复尝试不该被读成
循环"来自 PERSIST 那类工具投毒的目标），不是论文或案例库里的原始轨迹。真实脱敏
轨迹还没收进来——那需要用户签署授权，见下面"还没做的"。

**写这批语料当场抓到两个缺陷**（都属同一类：报告说了比观测更多的话）：

1. 一段轨迹的调用全部是快照恢复重放时，循环检测器报 `pass`，说"已在 0 次工具调用
   中比对，未发现循环"——把没查说成查过。改成 `inconclusive`。
2. 幂等检测器遇到不认识的重放标记（`resume` 之类）时，代码注释写着"应当能被看见"，
   而那条信息只出现在一个永远不会被返回的 detail 里。改成报 `inconclusive`。

## 待决清单（已知的、故意先记下来的）

写语料时反复撞到的事，按"要不要动代码"分开记着。**没有一个是语料能自己解决的**，
所以它们留在这里而不是变成红的测试。

1. **写操作失败重试的文案说得比观测多。** `loop-write-retry-after-error` 那三次调用
   `status` 全是 `error`，幂等检测器报 fail 是对的（重试失败的写操作是支付里最危险的
   模式），但文案写"这是重复扣款"——观测支持的是"重复尝试"。判定不动，需要改的是措辞。
2. **快照重放的重复扣款写不出豁免来。** 幂等检测器在"归因给恢复机制"那条分支上
   提前返回，压根没查豁免规则——所以就算语料（或场景）里写了规则，它也静音不了。
   方向上偏保守（真实的重复扣款仍然看得见），但看不出是刻意的还是漏的。
3. **同一个 `replay_reason` 在两个检测器里待遇不同。** 幂等检测器把未知重放标记的
   调用排除在"Agent 自发"之外，循环检测器却把它们算作 Agent 的调用（只排除
   `snapshot_restore`）。同一份轨迹，两个检测器对"这是它干的吗"回答不一致。
4. **近似层的告警阈值依赖参数形状。** 聚类是拿**簇的第一个成员**当代表比相似度，
   而数字相似度按相对差衰减：三个字段时页码 1 和 2 的相似度只有 0.83、1 和 3 就已经
   不成簇了。实测：`{query, status, page}` 三个字段翻到第 9 页，成簇的只有 5 个成员，
   报的是 pass 而不是那条"翻了很多页还没停"的 warning。要让阈值真正生效，得改聚类
   方式（链式而非首成员），那是检测器行为变更，要单独做、单独测。
5. **语料表达不了"报几条"。** `expect` 说的是结论，不是数量。归组规则（同一工具在
   同一段会话里做五次是一个行为，不是五个问题）只在 `tests/test_corpus.py` 里有
   一条计数断言。
6. **真实脱敏轨迹还没有。** 需要用户签授权才能收。在那之前，这份语料的结论边界是
   "人手挑出来的 31 条轨迹"。

## 这些数字不算什么

`calibrate.py` 现在会打印三族各 100.0%。**那不是"检测器没有缺陷"**，那是"这批样本上
没找到分歧"。31 条里一条误报就能让精确率掉几个百分点，而样本是人手挑的（挑的时候
已经知道检测器怎么工作）。要更硬的结论，只有两条路：把语料做大，或者收真实轨迹。
两条都需要人，不是脚本能自动化的。
