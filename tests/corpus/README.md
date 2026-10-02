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

`tools` 里的声明一共四个键：`is_write`（必填）、`enforces_idempotency`、
`may_run_unattended`、`identity_fields`。最后一个是"这个工具的**主键**是哪几个
字段"，写给幂等检测器看的：

```yaml
  tools:
    book_reservation: {is_write: true, identity_fields: [user_id, flights, passengers]}
```

**它是判据的开关，不是修饰。** 不声明时，"同一个操作"只有一个判据：规范化参数
逐字相同。声明之后，参数不同、但那几个字段相同的几次写操作也会被并成一组报出来
（判词里会写明是"按声明的主键"那一档，并列出哪几个字段不同）。这条放宽量过：
按"同工具"放宽在 642 条真实轨迹上多报 145 组、按"只差一个字段"多报 90 组，逐条看
大多是误报；按人手挑的主键放宽仍有 56 组，而那张表只有懂业务的人给得出来
（数字与反例在 `docs/detectors.md` 第四节末）。所以**默认关闭，一个字都不变**。

声明里写了一个调用根本没带的字段时，那几次调用单独报 `inconclusive`——判据对它们
一次都没跑过，不许折进"判过了、没问题"里。写错的字段名（场景那边）在加载期就报错。

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

56 条：`loop.yaml` 16 条、`idempotency.yaml` 18 条、`autonomy.yaml` 8 条、
`memory.yaml` 14 条。来源分布：自造 48、公开-PERSIST 3、公开-GhostWriter 1、
公开-tau-bench 4。

标着公开来源的那几条，多数复现的是**威胁类型**（比如"被边界拦下的反复尝试不该被读成
循环"来自 PERSIST 那类工具投毒的目标），不是论文或案例库里的原始轨迹。

`公开-tau-bench` 那四条不一样：它们取自公开数据集 τ-bench 的**真实成功轨迹**的形状
（内网 660 条轨迹那一轮里挑出来的），但**参数按仓库自己的写法重写过**（`A1`/`B1`/`U1`），
搬进来的是形状，不是那次的原始数据——原始轨迹连同它的名字、地址、邮箱一概不进仓库
（同一条规矩管着 API key）。四条里一条是"同一个工具连着改三张不同订单"（负样本），
另外三条是一对只差 `identity_fields` 那一行声明的对照加一条声明了主键的负样本，
它们演的那一对（同一趟行程、两种付款拆法）是那一轮实测里**重**的那一处漏报。
真实**私人**轨迹仍然没收，那需要用户签署授权，见下面"还没做的"。

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

## 隔壁那张表：τ-bench 的工具是读还是写

`tests/corpus/tau-tools-is-write.json` 不是轨迹语料，是**一张推出来的属性表**：
τ-bench（MIT）airline 15 个 + retail 15 个工具，逐行标着"改不改任务数据"。
推它的是 `../_traces/derive_is_write.py`（脚本和它读的第三方源码都在仓库外，
同一条规矩管着原始轨迹和 API key）。

**它为什么值得进仓库。** 这个仓库自己的 DSL 里 `is_write` 是**必填、没有默认值**，
而"哪些工具算写"这件事光看名字判不出来：`airline__search_onestop_flight` 名字里没有
一个写的意思，里面的 `.update()` 又足以让机械规则把它标出来"沾过 data"——可它改的是
自己用字典推导式造的新 dict，机械规则到这一步就答不了了，得人读代码。写操作漏报的
后果是漏掉一次重复扣款，所以宁可让人显式声明，也不替谁猜。这份表是那个判断在一套
真实工具集上的一次落地。

**值钱的不是那 30 个布尔值，是每一行的判据**（表里逐行写着，`how`）：

- `判据甲（AST）`——**最终判据**：`invoke` 里有没有改动落在指向 `data` 的名字上
  （别名闭包迭代到不动点，`reservations, users = data["reservations"], data["users"]`
  之后 `user = users[user_id]` 这种两次跳转要跟得上）。
- `人工裁决`——机械规则答不了，人读过代码之后的裁决，理由在 `why` 里。
- `判据乙（接口描述）`——**交叉核对，不是判据**：`get_info()` 里 description 的动词。
  它只负责把该有人看一眼的行挑出来，逐行说过什么记在 `by_description` 里。

30 行里，两条判据一致的 24 行；不一致的 4 行；机械规则答不了、人读了代码裁的 2 行。
写 13 行、读 17 行。

**那 4 行不一致，四次全是判据乙的假阳性**，成因只有两种：否定句里的动词（两个 `think`
的描述写着 "will not obtain new information or change the database"，乙匹配上了那个
change），和一词多义（两个 `find_user_id_*` 的描述里 "the function will return an
error message"，return 是"返回一个值"，不是"退货"）。甲在这四行给的都是最强的那一档
——`invoke` 里一处改动写法都没有——所以逐行看过之后采信甲，理由写在 `why` 里。
那 2 行人工裁决同理：`search_direct_flight` / `search_onestop_flight` 里确实有
`.append()` / `.update()`，但作用在字典推导式造出来的新 dict 上，改的不是 `data`。

**这些工具是被读过的，不是被跑过的。** 这批工具在我们这儿是**数据集**，不是可运行的
代码：推导只做静态 AST，一行第三方代码都没执行（"真跑一遍看它写不写"那条路要执行
第三方代码，没有走）。所以这张表说的是"照这段源码看，它改不改 `data`"，不是"跑起来
观测到它改了"。范围也只有 `data`：两个 `think` 都往日志里追加一句话，在这张表里仍然
是读——这里问的是任务数据，不是"这个工具有没有副作用"。

这几个数由 `tests/test_doc_counts.py` 从这份 JSON 现算，不在文档里手写。

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
2. **快照重放的重复扣款写不出豁免来。** ⬅ **已修（2026-10-03）。**

   原因很具体：那条分支**提前 return**，而"查豁免、命中就降级并留痕"这一小段在另外
   两条分支（Agent 自发重复、跨会话重复）上各抄了一份。于是**同一条规则管不管用，
   取决于检测器内部走了哪个 `if`**——而规则写的是检测器名，写规则的人看不见分支，
   只会看到"我写了规则，那条 fail 还在"。现在三个返回点走同一个
   `_silence_if_exempted`。写操作那道门照旧关着：不写 `allow_write: true` 的规则
   在哪一支上都不生效（`tests/test_detector_idempotency.py` 两条新用例分别钉住
   "接上了"与"没开新口子"）。
3. **同一个 `replay_reason` 在两个检测器里待遇不同。** ⬅ **已修（2026-10-03）。**

   幂等那边只把 `replay_reason is None` 的调用算作 Agent 自发；循环这边原先只排除
   `snapshot_restore` **那一个字符串**，剩下的（`resume` 之类）照单全收——一段可能
   根本不是 Agent 制造的重复，于是被报告指名道姓地算到它头上，还落进"原地打转"。
   现在循环同样只认 `replay_reason is None`，未知标记另走一路报
   `loop.unknown_replay / inconclusive`（"分不清是不是 Agent 自己发起的"），
   **不折进任何一个具体判定**。新增语料 `loop-all-calls-are-unknown-replays`：
   与 `loop-all-calls-are-snapshot-replays` **逐字同一条轨迹**，只把标记换成一个
   检测器不认识的，用来钉住"同一份轨迹、同一个问题，两个检测器答得一样"。
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
   "人手挑出来的 56 条轨迹"。

## 这些数字不算什么

`calibrate.py` 现在会打印四族各 100.0%。**那不是"检测器没有缺陷"**，那是"这批样本上
没找到分歧"。56 条里一条误报就能让精确率掉几个百分点，而样本是人手挑的（挑的时候
已经知道检测器怎么工作）。要更硬的结论，只有两条路：把语料做大，或者收真实轨迹。
两条都需要人，不是脚本能自动化的。
