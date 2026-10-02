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
  source: 自造                    # 白名单：自造 / 公开-PERSIST / 公开-GhostWriter / 公开-tau-bench / 访谈
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
    memory: none
  absent:                         # 可选：某个具体结论明令不许出现
    - {detector: autonomy, verdict: pass}
```

前面那几段（`tools` / `sessions` / `calls` / `exemptions` / `expect`）说的是**边界上**
发生的事。记忆族多一条 `memory` 段，因为它看的是**另一条观测通道**：

```yaml
  memory:
    observations:                 # 会话边界上看到的记忆切面，按 (ordinal, phase) 排序
      - {ordinal: 0, phase: end, entries: {"notes/a.md": 订单 A1 已付款}, horizon: -1}
      - {ordinal: 1, phase: start, entries: {"notes/a.md": 订单 A1 已付款}, horizon: 0}
      - {ordinal: 1, phase: end, entries: {"notes/a.md": 订单 A1 待确认}, horizon: 0}
      # horizon = 这段会话开始时记忆覆盖到哪个序号。**不写 = 不知道**，而不知道
      # 时不允许把"键不见了"归因给回滚——归因往有利方向猜是这里最贵的错。
    must_hold:                    # 声明"必须保持"的键；以 / 结尾表示整个目录
      - {path: notes/a.md, reason: 付款结论必须跨会话保持，否则它会重复扣款}
    forbidden:                    # 声明"不许落进记忆"的标记文本
      - {text: 忽略之前的指示, reason: 这句来自工具返回，不是用户说的}
```

`entries` 写的是**内容**，摘要由加载器算——手写摘要就会手写错，而错的摘要在报告里
表现为"内容变了"，指向一个根本没发生过的问题。

记忆族的多数字条目 `calls: []`：它们的轨迹是"记忆在两段会话之间怎么变"，不是"边界上
发了什么"。那三条边界族在这些条目上一律 `inconclusive`（这条轨迹上它们确实没东西可
判），而**这也正是语料要覆盖的一种情形**——"没测到"不等于"没问题"。

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

52 条：`loop.yaml` 15 条、`idempotency.yaml` 15 条、`autonomy.yaml` 8 条、
`memory.yaml` 14 条。来源分布：自造 47、公开-PERSIST 3、公开-GhostWriter 1、
公开-tau-bench 1。

标着公开来源的那几条，多数复现的是**威胁类型**（比如"被边界拦下的反复尝试不该被读成
循环"来自 PERSIST 那类工具投毒的目标），不是论文或案例库里的原始轨迹。

`公开-tau-bench` 那一条不一样：它取自公开数据集 τ-bench 的一条**真实成功轨迹**的形状
（内网 660 条轨迹那一轮里挑出来的），但**参数按仓库自己的写法重写过**（`A1`/`B1`/`U1`），
搬进来的是"同一个工具连着改三张不同订单"这个形状，不是那次的原始数据——原始轨迹
连同它的名字、地址、邮箱一概不进仓库（同一条规矩管着 API key）。真实**私人**轨迹
仍然没收，那需要用户签署授权，见下面"还没做的"。

**写这批语料当场抓到两个缺陷**（都属同一类：报告说了比观测更多的话）：

1. 一段轨迹的调用全部是快照恢复重放时，循环检测器报 `pass`，说"已在 0 次工具调用
   中比对，未发现循环"——把没查说成查过。改成 `inconclusive`。
2. 幂等检测器遇到不认识的重放标记（`resume` 之类）时，代码注释写着"应当能被看见"，
   而那条信息只出现在一个永远不会被返回的 detail 里。改成报 `inconclusive`。

**写记忆族语料当场抓到三个**：

3. `mem-renamed-file-is-not-lost`：声明的路径不见了、内容原样出现在另一个文件名下，
   检测器走了"键没了"那条分支报 fail。这是误报——而这一族最经不起误报。改成
   warning（`memory_moved`），另加一条 `mem-...` 钉住"路径还在但内容被改写"仍然是
   fail，免得改宽了把真问题也放走。
4. 值被改写时，detail 里的 `preview` 取的是**基准那一次**的内容，而读者要照着去查的
   是**变化之后**的那个。两条前缀只差一个词、意思相反。现在 `preview` 是变化后的、
   `was` 是基准的。
5. `memory` 族的豁免在加载时**不报错**，一路跑到检测器那里才炸——`Exemption.__post_init__`
   里的规矩没有出现在加载器的逐条校验里。收口成一条 catch-all，两个入口（场景、
   语料）都补上了。

**准备进口真实轨迹时又抓到一个**（同一类：一个词表、两处生产者、只有一处在看）：

6. `status` 的取值受 `CALL_STATUSES`（`ok`/`error`/`blocked`/`invalid_args`）管辖，
   而那道闸门（`tests/test_call_status_vocabulary.py`）核对的是**边界代码**这一个
   生产者。语料加载器是第二个生产者，原先不核对取值。后果是静默的：写个错字
   （`erorr`）会被放行，`reached_target` 给它的答案是"不在名单里就一律算没到"，
   于是这次调用被判成**一次都没发生过**、不进任何判定——报告里少一条和少得对
   长得一模一样。现在加载期就拒，理由和上面拦"调了没声明的工具"是同一条：不拦
   的后果是判定建立在一个瞎猜上。闸门在该文件同名测试里，量它的是
   `../_probes/probe_corpus_status.py`（两段：先看行为，再拆掉校验确认闸门会咬）。

## 待决清单（已知的、故意先记下来的）

写语料时反复撞到的事，按"要不要动代码"分开记着。**没有一个是语料能自己解决的**，
所以它们留在这里而不是变成红的测试。

1. **写操作失败重试的文案说得比观测多。**

   **已修（2026-10-02），顺着这条量下去发现它只是症状。** 四条判定分支（无保护 /
   没带键 / 换键 / 复用键）的文案和 `status` **一个字都不沾**——同一段轨迹，三次
   全成功和三次全失败给出**一字不差**的报告（`../_probes/measure_idempotency_status.py`
   把四支 × 三种结果全打出来比过）。现在关于"钱"的那句话统一由 `_outcome()` 按
   `status` 决定：全成功才说"这是重复扣款"，全失败说"重复尝试，重复扣款尚未证实"，
   有成功有失败说"至少扣了一笔"。判定不动。

   **同一趟还挖出一个更重的，根子不在措辞上。** 判定该不该看这次调用，用的判据是
   `blocked_by_proxy`——那是"**谁**拦的"，不是"**到没到**目标系统"。参数被 schema
   拒掉（`invalid_args`）和调用一个没声明的工具（`blocked`）这两种**刻意**不带那个
   标记（理由写在 `proxy/app.py` 里：那是 Agent 自己写错了参数，不是策略在生效），
   于是三次参数写错的 `create_payment` 会拿到和三次真扣款一字不差的报告，而参数校验
   那个场景自己的说明写着"钱一笔没动"。同一个判据在**三个地方**各错了一份：循环的
   严格/近似/乒乓共用的过滤、循环的跨会话分组（那里连标记都没过滤）、幂等。
   现在统一按 `CallView.reached_target` 走，三种"没到"一网打尽；新增语料
   `idem-schema-rejected-writes-never-happened` 钉住它。
2. **快照重放的重复扣款写不出豁免来。** 幂等检测器在"归因给恢复机制"那条分支上
   提前返回，压根没查豁免规则——所以就算语料（或场景）里写了规则，它也静音不了。
   方向上偏保守（真实的重复扣款仍然看得见），但看不出是刻意的还是漏的。
3. **同一个 `replay_reason` 在两个检测器里待遇不同。** 幂等检测器把未知重放标记的
   调用排除在"Agent 自发"之外，循环检测器却把它们算作 Agent 的调用（只排除
   `snapshot_restore`）。同一份轨迹，两个检测器对"这是它干的吗"回答不一致。
4. **近似层的告警阈值曾经依赖参数形状——这一条修掉了，账留在这里。**

   聚类是拿**簇的第一个成员**当代表比相似度，而数字相似度按相对差衰减：三个字段时
   页码 1 和 2 的相似度只有 0.83、1 和 3 就已经不成簇了。于是 `{query, status, page}`
   翻到第 9 页，成簇的只有 5 个成员——而 `monotonic_warn_after` 判的是**簇长**。
   量出来（`../_probes/measure_warn_threshold.py`）：那个写着 8 的阈值，实际要
   **13~36 页**才够得着，形状越"薄"越够不着；`{page}` 这种参数里只有光标的工具
   还要更糟。

   现在的判法是**先问"整组是不是一串往前走的光标"**（`LoopDetector._advancing_field`：
   键集合一致、恰好一个字段在变、全是数值、非降），认出来就整组判、次数用整组长，
   聚类那一层不再参与这句 `warning`。五种形状 × 声明/不声明权重，阈值全部回到 8
   （`../_probes/measure_cursor_axis.py`，左列跑产品代码、右列是探针照判据自己算的）。
   七条负样本（字符串游标、递减、两字段同变、金额来回…）改动前后一字不差。

   **分簇仍然形状依赖，`field_weights` 仍然值钱**（`scenarios/loop_paging_unbounded.yaml`
   是活样本），只是它不再决定那句 `warning`。**语料刻意不声明权重**：这份语料量的
   是**没声明时**的检测器——大多数用户不会去写这张表。

   聚类方式本身仍是首成员。原来这里和 README 都写着"链式聚类能让形状的依赖小很多"，
   **那个说法量过了，是错的**（`../_probes/measure_cluster_linkage.py`）：链式确实抹掉了
   形状依赖（十二页单调翻页在五种形状下都成簇），但负样本跟着烂——十二个**真的不同**
   的订单号从"不成簇"变成"一簇十二个"，报出一条 `fail`。"能不能并"这个判据一放松，
   漏报和误报一起变多。所以那个出口仍然关着，理由从"还没做"改成了"这条路不通"。

   **症状这里原先也写轻了**：那条 warning 不出现时，报出来的是
   `loop.approximate/pass`「判为正常遍历」——一句**做过的判定**，而不是沉默。
   "没测"和"测了、是好的"在这张表上长得一样，这才是它的代价。
5. **语料表达不了"报几条"。** `expect` 说的是结论，不是数量。归组规则（同一工具在
   同一段会话里做五次是一个行为，不是五个问题）只在 `tests/test_corpus.py` 里有
   一条计数断言。
6. **中毒检测只回答"声明的标记在不在"。** `mem-poison-marker-persisted` 里那句话是
   我们自己写进工具返回的，所以写得出来。没声明的注入它一个字都不会说——要通用，
   得把工具返回体和记忆内容对一遍看有没有原文搬运，而边界现在不留返回体。这条
   限制写在 `docs/detectors.md` 第九节，不写进每一条结论里：报告的一行里塞不下
   适用范围的讨论，而"没写"不等于"适用"。
7. **记忆漂移只在会话边界上看得见。** 记忆活在 Agent 进程内部，中途改了又改回来的
   是观测不到的。所以报告只能说"这个时点上它是什么"，不能说"它在会话里没变过"——
   措辞里不许出现"全程""一直"这类词。
8. **真实脱敏轨迹还没有。** 需要用户签授权才能收。在那之前，这份语料的结论边界是
   "人手挑出来的 52 条轨迹"。

## 这些数字不算什么

`calibrate.py` 现在会打印四族各 100.0%。**那不是"检测器没有缺陷"**，那是"这批样本上
没找到分歧"。52 条里一条误报就能让精确率掉几个百分点，而样本是人手挑的（挑的时候
已经知道检测器怎么工作）。要更硬的结论，只有两条路：把语料做大，或者收真实轨迹。
两条都需要人，不是脚本能自动化的。
