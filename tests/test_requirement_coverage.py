"""需求覆盖表：每条需求都表态，且"已交付"点名的测试真的存在。

`docs/requirements.md` 是那份对外的答案——"需求文档里的每一条，仓库里有没有东西在撑"。
它和 README 的缺口表是两种东西：缺口表按**知道的时间**排（先发现的先记），覆盖表按
**需求原文**排。前者回答"还欠着什么"，后者回答"合同履行到哪了"。

这张表最容易烂的方式不是写错，是**写得太顺**：把"已交付"填上去，而后面什么都没有。

## 这道闸门核什么

1. **每条都表态**，且只在那四个值里。写个「-」或者「大概做了」混过去是不行的。
2. **`已交付` / `部分交付` 必须点名至少一个真的存在的测试函数**（`tests/xxx.py::test_yyy`
   那种写法）。名字不存在就红，并印出这个文件里最接近的几个名字——**手抄一个测试名
   而它不存在**，是这张表最常见的坏法，而读的人不会去核。
3. **`部分交付` / `未做` / `不适用` 的「边界」栏不许空**。这三种状态的价值全在理由里：
   没有理由的"没做"，读起来和"忘了"一模一样。
4. **每个测试文件都要被认领**：要么被某条需求的证据栏点名，要么在 `_NOT_A_DELIVERABLE`
   里带一条写着理由的说明。这是**反向**的那一半——正着查只查得出"表里写了假的"，
   查不出"表里漏了真的"。少了这一条，新加一整块能力而不进表，可以一路绿下去。
   （同一条理由见 `tests/test_reason_codes.py`：每一个产出方要么被场景走到，要么在一张
   写明理由的待办清单里。）
5. **别的文档里"某个级别做完了"这种话，必须和末尾那张汇总表对得上**。这条是表立起来
   当天加的：README 在目录树下面写着"需求里的 P1 已经全部做完"，而 3.1 的「回归集管理」
   也是 P1（需求原文第 120 行）、且没做——那句话是照 3.2 那张表数出来的，漏掉了 3.1
   那一栏。**手写的话配算出来的表，错的永远是话**；现在这句话有东西在核，判据就是
   汇总表本身（那个级别只要还有一行不在 `已交付`，说它做完了就是假的）。
6. **末尾那三堆"还欠着什么"，必须和第六栏对得上**。第六栏（`卡在哪`）是逐行表态
   "这条还欠不欠活"，三堆是按它分的组，堆名与那一栏**字面一致**：闸门拿那一栏现算
   出每一堆的成员和条数，再去核散文里手写的那两份。这一条是补的——三堆原先是一段
   一段手写的散文，写下来就没人核过，而**它对不上**：全表 38 行里有 12 行不是
   `已交付`，三堆只提到了 10 行（漏了 4 行），另外还点名了 2 行**已经交付**的当欠账。
   两个方向都错，而两种错法读起来都像一份认真的总结。

## 这道闸门**不**核什么

**需求文档本身没有闸门。** 它在仓库外（路径写在 `docs/requirements.md` 的文件头），
普通 CI 上根本没有那个路径，所以闸门读不到它。表里的「需求（原文）」一栏是手工从 v1.1
抄的；那份文档改了、加了一节，
或者哪条被删了，**这道闸门不会知道**。这不是可以自动化的一步——能自动化的那一半
（证据真的存在、文件都被认领）做掉了，剩下的靠下一次人工核对。

## 这道闸门被量过

`_probes/probe_requirement_coverage.py`（住在仓库外，`tests/test_doc_paths.py`
的白名单里记着为什么）把上面每一条各改坏一次：抄错一个测试名、状态写成第五个值、
理由清空、汇总表错一格、新加一个测试文件而不进表、清单里留一条不存在的文件——
每一条都红在预期的位置，一个绿灯对照一条都不红。

**其中最该看的是相机那一条**：把行判据改坏之后，上面两条逐行检查**一个用例都
收不到**，pytest 把它们记成 `skipped` 而不是红的——量到的是 `4 failed, 4 passed,
2 skipped`，那 2 skipped 正是它们，而"跳"在这个仓库里和"过"一样不拦人（全套本来
就常年带着 1 skipped）。换句话：这张表格式一烂，逐行那两条覆盖表的闸门就和没写
一样，剩下的全靠 `test_the_table_has_rows_at_all` 那几条不逐行的。
"""

from __future__ import annotations

import ast
import difflib
import re
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "requirements.md"
TESTS = ROOT / "tests"

#: 「现状」只许这四个值。`不适用` 与 `未做` 必须分开：一条需求"在这个形态下没有对象
#: 可测"和"我们没做"是两件事，混成一个值之后，读者没法判断哪一条还值得催。
STATUSES = frozenset({"已交付", "部分交付", "未做", "不适用"})

#: 要证据栏点名的两种状态。
_NEEDS_EVIDENCE = frozenset({"已交付", "部分交付"})
#: 这三种状态必须写理由。
_NEEDS_REASON = frozenset({"部分交付", "未做", "不适用"})

#: 覆盖表的表头。六列；「怎么读这张表」那几张图例不在此列。
_HEADER = ("需求（原文）", "级别", "现状", "证据", "边界 / 缺口", "卡在哪")

#: 覆盖表的「级别」栏。「—」是需求原文没标级别的非功能需求（见 `_NO_LEVEL`）。
_ROW_LEVELS = ("P0", "P1", "P2", "—")

#: 第六栏「卡在哪」的值。**堆名就是这里的三个值，字面一致**：末尾那段散文按堆名分组
#: 列成员，闸门拿这一栏现算一遍去对（`test_the_piles_account_for_every_row_that_owes_something`）。
#: 改一个词就会让那一堆静默地少掉一个成员，所以改动要两边一起改。
NOTHING_OWED = "没欠着"
PILE_KEYS = ("卡在人或者钱", "卡在没真实用例", "卡在部署形态")
KEYS = (NOTHING_OWED, *PILE_KEYS)

#: `tests/xxx.py` 或 `tests/xxx.py::test_yyy`。
#:
#: 名字那一组收 CJK：本仓库里有几个测试的函数名是中文的
#: （`test_storage_minimization.py` 整份都是），不收的话它会掉进"没人认领"里，
#: 而报错会指着一份明明被引过的文件。
_CITE = re.compile(r"^`(tests/[A-Za-z0-9_/]+\.py)(?:::([A-Za-z0-9_一-鿿]+))?`$")

#: 认领不满的测试文件。**每一条都要写理由**，而且理由要被核两处：条目对应的文件
#: 必须真的在（清单不许留陈旧的条目），理由本身不许短到只剩一个词。
#:
#: 为什么要这张清单而不是"全都算证据"：证据栏点名的那些是**直接**在量某一条需求的
#: 文件；剩下的多半是闸门自身、边界不变式、或者受控实验的账本——它们服务于整个仓库
#: 而不是某一条需求。写清楚每一条为什么不算，比让读者自己去猜强。
_NOT_A_DELIVERABLE: dict[str, str] = {
    "test_agent_memory.py": (
        "演示 Agent 自己的记忆实现（追加写、去重键、整理）。它服务于 3.4，"
        "但表里那条证据取的是检测侧更靠生产路径的一条。"
    ),
    "test_boundary_findings.py": (
        "边界当场判的那两条结论回到运行里的闸门，横跨 3.2 的出口拦截与参数校验两条需求。"
    ),
    "test_call_status_vocabulary.py": (
        "调用状态词的词汇表闸门——「到没到目标系统」不许拿「谁拦的」当替身。"
        "它守的是所有读状态的结论，不服务单条需求。"
    ),
    "test_checkup.py": (
        "一键体检的闸门。体检**不新增任何判定**——判卷标准仍归场景的 `expect`，"
        "它只是一层读法：把已有的场景按「毛病」重排成一面。所以它不服务单条需求，"
        "守的是贯穿全仓的那条规矩「报告不许替结论说它没说过的话」（`clear` 要求至少"
        "一条 `pass` 命中、运行级的 `ok` 不给任何一项定档、菜单里必须有反面样本）。"
    ),
    "test_config_defaults.py": (
        "配置缺省与拒绝启动的闸门，服务于非功能需求整段（安全性、部署形态），不指向某一条功能。"
    ),
    "test_corpus.py": (
        "55 条标注语料的闸门。语料服务 3.7 的用例集那半边，"
        "而那条需求本身是「部分交付」，证据取的是受控对照账本。"
    ),
    "test_cost_by_tool.py": (
        "按工具拆的那三个口径（返回体 / 实测增量 / 租金）的闸门，"
        "是 3.5 跨会话成本报告里那个「不是这个工具花了多少钱」的边界本体。"
    ),
    "test_db_schema.py": "库表结构与自动迁移的闸门，服务于所有落库的结论。",
    "test_doc_counts.py": (
        "文档计数闸门自身：README 里那个测试条数由 `--collect-only` 数出来，手写的数字会红。"
    ),
    "test_doc_paths.py": (
        "文档路径闸门自身：文档里点名的文件必须真的在，白名单里每条豁免还要带不会过期的理由。"
    ),
    "test_egress.py": (
        "出口放行名单的闸门，是非功能「安全性」那一条的地基（默认拒绝、主机名精确匹配）。"
    ),
    "test_exempt_ui.py": (
        "报告页上「这条我认可是误报」这个入口的闸门。豁免引擎侧一直是完整的，"
        "缺的是用户够得着的那一步。它守的是三件事：只有真的会去查豁免的那几个检测器"
        "才给按钮（记分表现算，不手抄）、写下去的东西先解析再落盘、以及页面上印着的"
        "那句代价（会进配置指纹）当场由回归对比自己验一遍。"
        "它服务的是所有出口的可操作性，不新增判定，也不指向某一条需求。"
    ),
    "test_explain_coverage.py": (
        "报告里那张大白话翻译表的闸门：每个检测器名、每种结论取值、每个理由码都要有一句"
        "解释，多一条少一条都红。它服务的是「不写代码的人也能读懂报告」——需求原文没有"
        "这一条（是第一个真实用户反馈加上来的），所以不指向某一条功能，指向所有出口的读法。"
    ),
    "test_gateway_stream.py": (
        "流式转发与结算的闸门，服务 3.5 的 Token 统计"
        "（取消时明细可能少一条这条边界就长在这里）。"
    ),
    "test_generic_egress.py": (
        "目标由调用方指定的通用出口，服务非功能「安全性」的默认拒绝，以及 3.2 的调用拦截。"
    ),
    "test_identity.py": (
        "配置指纹的闸门：改动能定位到节、改格式不算改动。"
        "它是 3.1 一键回归「先报算不算数」的判据来源。"
    ),
    "test_launcher.py": (
        "双击启动这条路自己的闸门：那个 .bat 必须是纯 ASCII（实测：非 ASCII 字节会让 "
        "cmd.exe 按字节偏移把批处理文件读错位，报出来的是「不是内部或外部命令」——"
        "看起来像程序本来就是坏的），以及「起来了」判的是能不能应答、不是端口通不通，"
        "正反两个方向各有一条对照。它服务的是「不写代码的人也能把界面打开」，"
        "需求原文没有这一条（是第一个真实用户用不上界面之后加上来的）。"
    ),
    "test_loop_cluster.py": (
        "近似层聚类的独立朴素实现对齐，是 3.3 循环识别那条边界（阈值依赖参数形状）的尺子。"
    ),
    "test_loop_paging.py": "翻页遍历与字段权重，同上一条：服务 3.3，但量的是那一条边界本身。",
    "test_normalize.py": "参数规范化与指纹。3.2 幂等检测和 3.1 差异配对共用这一个判据。",
    "test_open_route_parity.py": (
        "两条出口路由的受控对照（看门狗与熔断各接一遍），属于 3.3 超时熔断那一条的边界。"
    ),
    "test_pricing_cache.py": "单价表与缓存价。3.5 的 Token 统计靠它把数算成钱。",
    "test_reason_codes.py": "结论码覆盖率闸门：每个产出方要么被场景走到，要么在待办清单里。",
    "test_recorder.py": "轨迹落库的细节（体量、时间戳、NULL 与 0 分开），服务 3.2 的调用拦截。",
    "test_report_references.py": (
        "报告模板读的快照字段必须真的存在（`report.expectation.ok` 曾经不存在，"
        "于是每一份报告的「检测结论」卡片恒红）。它守的是「报告不许说它没观测到的话」"
        "这条贯穿全仓的规矩——`run.html` 那一页的每一个字段，不服务单条需求。"
    ),
    "test_report_text.py": (
        "报告文案闸门：终端、场景自由文本、页面模板里都不许出现 Markdown 星号。"
        "它守的是所有用户可见的文案。"
    ),
    "test_requirement_coverage.py": (
        "这张表自己的闸门（证据真的存在、每个文件都被认领、理由不许短到一个词）。"
        "它守的是本文档，不对应任何一条需求。"
    ),
    "test_run_summary.py": (
        "报告页最上面那句「所以呢、我该干什么」的闸门。它守的是两件事：那句话的语气"
        "必须和真正算出来的判定一致（不许页面顶上和底下那张卡片各说各的），"
        "以及「通过」那句话里每次出现「没问题」都要紧挨着一个否定词——"
        "通俗化最容易把「判不了」丢掉，而那是这个产品唯一值钱的地方。"
        "它服务的是所有出口的读法，不指向某一条功能。"
    ),
    "test_scenario_exemptions.py": (
        "场景豁免的加载期校验（六种「装上了但没接线」的写法都要让加载失败），横跨 3.2 与 3.3。"
    ),
    "test_sse.py": "流式响应的透传与取消，服务 3.2 调用拦截与 3.5 成本统计两条。",
    "test_token_estimator.py": (
        "重量字符到 token 的密度估计，是 3.5 里「没量到就不许猜」那条边界的实现。"
    ),
    "test_tool_library.py": "`_shared/` 工具库与 `use:` 引用，服务 3.2 那条「支持 YAML 配置」。",
    "test_tool_schema_liveness.py": (
        "schema 白名单关键字的活样本闸门——每个关键字要么有场景用过，要么有一行不过期的理由。"
    ),
    "test_types.py": "类型检查面的闸门（守 `strict = true` 不是件装饰品）。",
    "test_undeclared_keys.py": (
        "参数声明的键按「表过态没有」判（空表 vs 没给），是 3.2 参数校验那一条的边界本体。"
    ),
    "test_unwired_claims.py": (
        "文档里写「这东西没接上」的地方必须真的没接上。"
        "它守的是 README 那句话，不对应任何一条需求。"
    ),
    "test_watchdog_accounting.py": (
        "看门狗自己的耗时记在谁名下（谁的错分开写），"
        "是非功能「性能」那一条把延迟拆开的依据。"
    ),
    "test_wizard.py": (
        "新建测试向导（人话问句 → 确认页 → 场景文件）。它替的是「手写 YAML」"
        "这一步，不新增任何判定——生成出来的场景走现有的保存路径、跑的是同一个"
        "引擎，所以它是需求的入口而不是某一节的证据。"
    ),
    "test_world_notes.py": (
        "世界初始状态的备注（**不许是空的**），服务 3.4 记忆中毒那条受控实验的场景侧。"
    ),
}

#: 理由短于这个长度就退化成"一个词"。一个词的清单会变成抹布：谁都往里丢东西，
#: 而后来的人说不清每一条当初为什么写。这条阈值是刻意的、也是可以被质疑的——
#: 它挡不住有心写废话的人，只挡住顺手写一个词的人。
_MIN_REASON = 20

#: 枚举测试函数时算作"一个测试"的节点类型。
_FUNC_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _test_names(path: Path) -> set[str]:
    """一个测试文件里所有能被执行的名字。

    **`AsyncFunctionDef` 也要收**：本仓库里异步测试不少（`test_watchdog_accounting.py`
    整个文件都是），第一版只收同步 def，于是那些文件在反向检查里"一个名字都没有"
    ——而报错会指向别处。类里的方法同理。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, _FUNC_NODES) and node.name.startswith("test_")
    }


def _ids(value: object) -> str:
    """parametrize 的 id：行号用它自己，五个格子那一份留空。

    留空是故意的——每一行的 id 都是整张表的内容，印出来会把失败信息淹掉，
    而真正要认的是行号。
    """
    return str(value) if isinstance(value, int) else ""


def _rows() -> list[tuple[int, list[str]]]:
    """覆盖表的每一行：`(行号, 六个格子)`。

    **用列数加内容分辨，不用行号**：以后往文档里插一段就会让行号全错，而列数是这份
    表格的格式本身。列数现在不够了——末尾那张汇总表也是六列——所以再加一条：这一行的
    「级别」格子得是 `P0` / `P1` / `P2` / `—`。汇总表的第二格是个数字，图例表是别的列数，
    两张表因此分得开。

    代价是**「级别」栏写坏（写成 `p0`、或者多一个空格之外的字）会让那一行整行消失**，
    而不是报错。这一条由汇总表兜着：那一行没了，手写的汇总数就和现算的对不上，
    `test_the_tally_matches_the_rows` 会红。
    """
    out: list[tuple[int, list[str]]] = []
    for lineno, line in enumerate(DOC.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) != 6 or tuple(cells) == _HEADER or cells[1] not in _ROW_LEVELS:
            continue
        if all(set(c) <= set("-: ") for c in cells):
            continue
        out.append((lineno, cells))
    return out


def test_the_table_has_rows_at_all() -> None:
    """空表也必须红。**格式对了但一行都没有**，上面那些逐行检查会全部通过。"""
    assert len(_rows()) >= 30, f"只解析出 {len(_rows())} 行需求；表格格式变了，或者内容丢了"


@pytest.mark.parametrize("lineno,cells", _rows(), ids=_ids)
def test_every_row_takes_a_position(lineno: int, cells: list[str]) -> None:
    requirement, _level, status, evidence, reason, waiting = cells
    assert status in STATUSES, (
        f"docs/requirements.md:{lineno}（{requirement[:30]}）的「现状」是「{status}」，"
        f"不在 {sorted(STATUSES)} 里。**写别的东西等于没表态**——读者没法判断这条是"
        f"做了、没做、还是根本没法测"
    )
    assert waiting in KEYS, (
        f"docs/requirements.md:{lineno}（{requirement[:30]}）的「卡在哪」是「{waiting}」，"
        f"不在 {list(KEYS)} 里。**这一栏是末尾那三堆的分组依据**，写别的东西等于不表态"
    )
    if status != "已交付":
        assert waiting != NOTHING_OWED, (
            f"docs/requirements.md:{lineno}（{requirement[:30]}）的「现状」是「{status}」，"
            f"而「卡在哪」写着「{NOTHING_OWED}」——**这两个说法互相矛盾**："
            f"现状不是「已交付」，就是这条需求还欠着。要么把它归进某一堆并写清卡在哪，"
            f"要么把它做完"
        )
    if status in _NEEDS_EVIDENCE:
        assert evidence, f"docs/requirements.md:{lineno} 标着「{status}」，但证据栏是空的"
    if status in _NEEDS_REASON:
        assert reason, (
            f"docs/requirements.md:{lineno} 标着「{status}」，但「边界 / 缺口」栏是空的。"
            f"**没有理由的「{status}」读起来和「忘了」一模一样**"
        )


@pytest.mark.parametrize("lineno,cells", _rows(), ids=_ids)
def test_claimed_evidence_really_exists(lineno: int, cells: list[str]) -> None:
    """`tests/xxx.py::test_yyy` 里那个名字，必须真的在那个文件里。

    这是这道闸门的主判据：**表里的每一个"已交付"后面都站着一个能跑的东西**。
    名字对不上时印出最接近的几个，因为实际发生的情况多半是手抄时抄错了一个词。
    """
    requirement, _level, status, evidence, _reason, _waiting = cells
    if status not in _NEEDS_EVIDENCE:
        return
    citations = [m for span in re.findall(r"`[^`]+`", evidence) if (m := _CITE.match(span))]
    assert citations, (
        f"docs/requirements.md:{lineno}（{requirement[:30]}）标着「{status}」，"
        f"但证据栏里没有一个 `tests/xxx.py::test_yyy` 形式的引用（证据栏是：{evidence}）"
    )
    for match in citations:
        rel, name = match.group(1), match.group(2)
        path = ROOT / rel
        assert path.is_file(), f"docs/requirements.md:{lineno} 引了 {rel}，这个文件不在"
        if name is None:
            continue
        names = _test_names(path)
        if name in names:
            continue
        near = difflib.get_close_matches(name, sorted(names), n=3, cutoff=0.4)
        raise AssertionError(
            f"docs/requirements.md:{lineno} 说 {rel} 里有 `{name}`，而它不在。"
            f"这个文件里最接近的是：{near or '（一个都不像）'}"
        )


def test_every_test_file_is_claimed_by_something() -> None:
    """反着查的那一半：**每个测试文件要么是某条需求的证据，要么有一句写下来的理由**。

    少了这一条，正着查只会说"表里写的都是真的"——而"表里漏了整整一块"它一个字都不会说。
    这正是同类的账在别处翻过的车：`reason_code` 覆盖率闸门的第一版只查"场景里写的码
    对不对"，没查"源码里产出的码有没有场景走到"，于是漏掉一整个检测器也全绿。

    **证据栏和边界栏都算认领**。一处引用落在哪一栏取决于它在句子里扮演什么角色
    （"这就是证据"还是"这条边界的闸门在这里"），而这一条问的是更粗的那个问题：
    这个文件有没有在表里被认出来过。
    """
    claimed: set[str] = set()
    for _lineno, cells in _rows():
        for span in re.findall(r"`[^`]+`", cells[3] + cells[4]):
            match = _CITE.match(span)
            if match:
                claimed.add(match.group(1))

    on_disk = {p.relative_to(ROOT).as_posix() for p in sorted(TESTS.glob("test_*.py"))}
    unclaimed = on_disk - claimed - {f"tests/{k}" for k in _NOT_A_DELIVERABLE}
    assert not unclaimed, (
        "这些测试文件既不是任何一条需求的证据，也不在 `_NOT_A_DELIVERABLE` 里：\n  "
        + "\n  ".join(sorted(unclaimed))
        + "\n\n要么它在量某一条需求（那就把引用填进那一行的证据栏），"
        "要么它服务的不是单条需求（那就进清单，并写清楚为什么）"
    )

    stale = {k for k in _NOT_A_DELIVERABLE if f"tests/{k}" not in on_disk}
    assert not stale, (
        f"`_NOT_A_DELIVERABLE` 里的这些文件已经不在盘上了：{sorted(stale)}。"
        "**清单不许留陈旧的条目**——留着的每一条都在告诉后来的人「这里有个文件」，"
        "而那个文件早就没了（同一条规矩见 `tests/test_doc_paths.py` 的 `_ABSENT`）"
    )


def test_every_reason_says_something() -> None:
    """理由不许短到只剩一个词。**一个词的清单迟早变成抹布**（见常量处那段说明）。"""
    for filename, reason in _NOT_A_DELIVERABLE.items():
        assert len(reason) >= _MIN_REASON, (
            f"`_NOT_A_DELIVERABLE` 里 `{filename}` 的理由只有 {len(reason)} 个字："
            f"「{reason}」。一条理由要说清它服务的是什么、为什么不进证据栏"
        )


#: 「现在还剩什么」那张汇总表里的级别名。**原文没标级别的非功能需求单独一格**，
#: 不并进 P1——需求文档确实没给它们定级，替它定一个级就是替原文说话。
_NO_LEVEL = "非功能（原文没标级别）"
_TALLY_HEADER = ("级别", "条数", "已交付", "部分交付", "未做", "不适用")


def _tally_rows() -> dict[str, list[int]]:
    """汇总表：`级别 -> [条数, 已交付, 部分交付, 未做, 不适用]`。

    分辨办法与 `_rows()` 对称：也是六列，但**第一格是级别名**，而覆盖表的第一格是需求
    原文、第二格才是级别。两张表因此分得开（见 `_rows()` 的说明）。
    """
    out: dict[str, list[int]] = {}
    for line in DOC.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) != 6 or tuple(cells) == _TALLY_HEADER:
            continue
        if cells[0] not in (*_ROW_LEVELS, _NO_LEVEL):
            continue
        if all(set(c) <= set("-: ") for c in cells):
            continue
        out[cells[0]] = [int(c) for c in cells[1:]]
    return out


def test_the_tally_matches_the_rows() -> None:
    """汇总表里那几个数是**算出来**的，不是写下来的。

    一句手写的"P0 一条不剩"在这个仓库里已经错过一次（写之前数的是 3.x 那几张表，
    漏掉了非功能那一段）。数字必须有东西在核——这道闸门就是那个东西，理由和 README
    的测试计数一样：**手写的数会漂，而漂了之后读起来照样像真的**。
    """
    counted: Counter[tuple[str, str]] = Counter()
    for _lineno, cells in _rows():
        level = _NO_LEVEL if cells[1] == "—" else cells[1]
        counted[(level, cells[2])] += 1

    tally = _tally_rows()
    assert set(tally) == {*["P0", "P1", "P2"], _NO_LEVEL}, (
        f"汇总表的级别是 {sorted(tally)}，和覆盖表里的对不上"
    )
    for level, row in tally.items():
        expected = [counted[(level, s)] for s in ("已交付", "部分交付", "未做", "不适用")]
        assert row[0] == sum(expected), (
            f"汇总表里 {level} 的条数是 {row[0]}，覆盖表里数出来是 {sum(expected)}"
        )
        assert row[1:] == expected, (
            f"汇总表里 {level} 那一行是 {row[1:]}（已交付/部分交付/未做/不适用），"
            f"覆盖表里数出来是 {expected}。要改的是 docs/requirements.md"
        )


#: 「某个级别已经做完了」这种话，可能写在任何一份面向读者的文档里。它必须和覆盖表
#: 对得上：覆盖表里的数是**算出来**的，这句话是手写的。
#:
#: 这句话真的漂过。README 原先在目录树下面写着"需求里的 P1 已经全部做完"，而 3.1 的
#: 「回归集管理」也是 P1（需求原文第 120 行）、且没做——那句话是照 3.2 那张表数出来的，
#: 漏掉了 3.1 那一栏。**手写的话配算出来的表，错的永远是话**，所以让话有东西在核。
#:
#: 引号里的话不算"我们在声明"——那是引用或转述，不是我们在说这件事做完了。
#: 四对：中文的「」、两种弯引号、ASCII 直引号（这一条是被真实需求逼出来的：覆盖表自己
#: 要在正文里记下"README 原先那句话是错的"，而记下它就得提到那句话——不许提，
#: 改正的记录就只能写一半）。
#:
#: **代价写在明处**：一句假话只要加上引号就能溜过去。这道闸门是绊线不是证明，
#: 它挡的是"顺手写下的进展汇报"，不是"存心绕过去"（同一条见 `test_unwired_claims.py`）。
_QUOTE_PAIRS = (("「", "」"), ("“", "”"), ("\x22", "\x22"))
_COMPLETION_CLAIM = re.compile(r"P([012])[^。\n]{0,8}(?:全部做完|做完了|清空)")


def _unquoted(text: str) -> str:
    """把成对的引号连同里面的内容换成等长空白（行号与位置感都留着）。"""
    for opening, closing in _QUOTE_PAIRS:
        text = re.sub(
            f"{re.escape(opening)}[^{re.escape(closing)}\n]*{re.escape(closing)}",
            lambda m: " " * len(m.group(0)),
            text,
        )
    return text


def _completion_claims(text: str) -> set[str]:
    """这段文字在说哪几个级别已经做完了。引号里的不算（见常量处的代价说明）。"""
    return {f"P{m.group(1)}" for m in _COMPLETION_CLAIM.finditer(_unquoted(text))}


def _claiming_docs() -> list[Path]:
    """扫哪几份文档：README 与 `docs/` 下所有的 Markdown。

    `docs/requirements.md` 自己也在里面——它当然可以谈级别，但它也**只能说实话**，
    而它那句实话的判据就是它自己的汇总表。
    """
    return sorted({ROOT / "README.md", *(ROOT / "docs").rglob("*.md")})


def test_no_document_claims_a_level_is_finished_while_the_table_says_otherwise() -> None:
    """手写的"做完了"必须跟着算出来的数走。

    判据是**汇总表本身**：那个级别只要还有一行不在 `已交付`，说它"做完了"就是假的。
    `不适用` 也算没交付——它说的正是"这条我们没做，理由是这套形态下没有对象可测"，
    把那一条读成"做完"等于替它改口。
    """
    incomplete = {level for level, row in _tally_rows().items() if row[0] != row[1]}
    hits = {
        # `as_posix`：本仓库的报错里路径一律是正斜杠，这样它跟别处的指路长得一样，
        # 也能直接拿去 grep（Windows 上 `str()` 会给反斜杠）。
        path.relative_to(ROOT).as_posix(): sorted(found)
        for path in _claiming_docs()
        if (found := _completion_claims(path.read_text(encoding="utf-8")) & incomplete)
    }
    assert not hits, (
        "这些文档在说某个级别已经做完了，而 docs/requirements.md 数出来它不是：\n  "
        + "\n  ".join(f"{f}：{'、'.join(v)}" for f, v in sorted(hits.items()))
        + f"\n\n剩着的是：{'、'.join(sorted(incomplete))}。覆盖表里的数是算出来的，"
        "这句话是手写的——要么把话改准（说清还剩哪一条），要么去把那一类做完，"
        "**别改这道闸门**"
    )


def test_the_completion_scanner_finds_a_claim_when_there_is_one() -> None:
    """自证：喂它三句真的声明、两句不是声明的、一句引号里的转述。

    **没有这一条，"扫描器写错了"和"文档里确实没有这种话"长得一模一样。**
    """
    assert _completion_claims("需求里的 P1 已经全部做完") == {"P1"}
    assert _completion_claims("（需求 3.2 的 P1，P1 由此清空）") == {"P1"}
    assert _completion_claims("P2 那三条都做完了") == {"P2"}
    # 不带级别的话不是这类声明：界面模块那一句说的是六个页面，不是某个优先级。
    assert _completion_claims("需求第 229 行的六个界面模块全部做完") == set()
    # 引号里的是转述，不是声明——三种引号都算（`_QUOTE_PAIRS`），代价写在常量处。
    assert _completion_claims("上面那句「P1 全部做完」从前是错的") == set()
    assert _completion_claims("它原先写着“P1 已经全部做完”") == set()
    # 引号**没配对**时按没引号算：宁可咬，不可漏。
    assert _completion_claims("它原先写着“P1 已经全部做完") == {"P1"}


#: 末尾那三堆的写法。**格式是死的**：`N. **堆名**（条数 条）：成员、成员。`
#:
#: 成员那一串写在**同一行**并且以句号收尾，是为了让闸门能一句话切出来——散成两行、
#: 或者把句号写成逗号，都会让它悄悄少收一个成员。堆名必须与 `PILE_KEYS` 字面一致。
_PILE_BULLET = re.compile(
    r"^\d+\.\s+\*\*(?P<key>[^*]+)\*\*（(?P<count>\d+) 条）：(?P<members>[^。\n]+)。$",
    re.MULTILINE,
)


def _name(requirement: str) -> str:
    """一行需求的短名：「：」之前那段。三堆里就是这么指名的。

    需求原文那一栏一律是「短名：展开」的形状（38 行都是），所以这个切法是稳定的——
    它不是"从自由文本里猜名字"，是这份表格的一条格式。
    """
    return requirement.split("：")[0].strip()


def _piles(text: str | None = None) -> dict[str, tuple[int, list[str]]]:
    """末尾那三堆：`堆名 -> (写着的条数, 写着的成员)`。

    **把写着的两样都取出来，一个都不信**：它们要跟第六栏现算的结果对（见下一条）。
    """
    source = DOC.read_text(encoding="utf-8") if text is None else text
    return {
        m.group("key"): (int(m.group("count")), m.group("members").split("、"))
        for m in _PILE_BULLET.finditer(source)
    }


def test_the_piles_account_for_every_row_that_owes_something() -> None:
    """那三堆的条数与成员，都是按第六栏现算出来的。

    这一条补上之前，三堆是一段手写的散文，没有任何东西核过它，而**它当时是错的**：
    全表 38 行里有 12 行不是 `已交付`，三堆只提到 10 行——`记忆同步验证`、
    `状态快照 / 恢复`、`评测用例集管理`、`兼容性` 四条谁都没提；同时又把 `幂等检测`
    与 `参数 Schema 校验` 当成欠账列了进去（那两条是 `已交付`）。少了 4 条、多了 2 条，
    **而两种错法读起来都像一份认真的总结**——这正是"手写的话配算出来的表"那个毛病的
    又一笔，也是这个项目最贵的那种错：报告说了比观测更多的话。

    「已交付」的行**可以**进堆（幂等检测那条的漏报就是这么进去的：需求做完了，同行
    的边界栏里点名了一件还没做的事），所以这里不按「现状」栏分，只按第六栏。
    """
    by_key: dict[str, list[str]] = {}
    for _lineno, cells in _rows():
        if cells[5] != NOTHING_OWED:
            by_key.setdefault(cells[5], []).append(_name(cells[0]))

    piles = _piles()
    assert set(piles) == set(by_key), (
        f"散文里的堆是 {sorted(piles)}，覆盖表第六栏里数出来的是 {sorted(by_key)}。"
        f"堆名要与那一栏**字面一致**（见 `PILE_KEYS`），少一堆或者多一堆都会让某几条"
        f"需求无声地掉出这份「还剩什么」"
    )
    for key, (count, members) in sorted(piles.items()):
        expected = sorted(by_key[key])
        assert count == len(members), (
            f"「{key}」那一堆写着（{count} 条），可后面列了 {len(members)} 个：{members}"
        )
        assert sorted(members) == expected, (
            f"「{key}」那一堆的成员和覆盖表第六栏对不上。\n"
            f"  散文里写的：{'、'.join(members)}\n"
            f"  现算出来的：{'、'.join(expected)}\n"
            f"要改的是 docs/requirements.md 的那一段（成员用「、」分开、同一行、以句号收尾）"
        )


def test_the_pile_scanner_finds_a_pile_when_there_is_one() -> None:
    """自证：这套切法真的切得出东西。**没有这一条，"格式写歪了"和"确实没这一堆"
    长得一模一样**——`_piles()` 会安安静静地返回一个空字典。

    （同一条教训见 `_rows()` 那个相机：正则会退化、格式会漂，而退化之后**一条都不会红**。
    这里靠上面那条 `set(piles) == set(by_key)` 兜底：真切不出东西，两边就对不上。）
    """
    sample = "1. **卡在人或者钱**（2 条）：甲、乙。\n2. **卡在没真实用例**（1 条）：丙。"
    assert _piles(sample) == {"卡在人或者钱": (2, ["甲", "乙"]), "卡在没真实用例": (1, ["丙"])}
    # 成员串跨了行、或者没用句号收尾，就**什么都切不出来**——宁可缺，不可错收。
    assert _piles("1. **卡在人或者钱**（2 条）：甲、\n   乙。") == {}
    assert _piles("1. **卡在人或者钱**（2 条）：甲、乙") == {}
