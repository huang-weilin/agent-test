"""文档里写下的语料数字，必须是从 yaml 数出来的那个。

语料的条数、各族正负样本数、来源分布，在三个文档里各写了一遍；`docs/detectors.md`
第七节还手抄了一整张标定表（四族 × TP/FP/FN/TN/精确率/召回率，十六个数）。

**这些数原先没有一个地方在核。** 2026-10-01 给近似层加两条负样本时量到：
`docs/detectors.md` 里那句"已交付 45 条"已经陈旧了两轮——语料早就 47 条了。
改语料的人不会想起正文里那句数字，读的人也不会去数，于是它就那么放着，
**看起来和一句真话一模一样**。这是这个项目里反复出现的那一类：写下来的一句话，
没有东西在它后面。

所以这道闸门把两边接上：左边是 yaml（真值），右边是文档里的原句。比的是**句子**，
不是随便哪里的数字——认宽了立刻变噪音，而一个会误报的闸门迟早被整体忽略
（同一条理由见 `test_doc_paths.py` 与 `test_report_text.py`）。

## 句子改了也会红，这是有意的

模板和文案是绑死的：谁把"49 条轨迹"改写成"共四十九条轨迹"，谁就得来改这里。
代价是多改一处，换来的是**改文案的人一定会再看一眼那个数**——而这正是原先缺的
那一步。找不着那句话时给出的报错会说明这一点，别把它当成闸门坏了。

## 空白不敏感

模板里换行和缩进怎么写都行：两边都先把连续空白压掉再比。不然一句跨行的行内文案
就会因为折行位置变了而红，那种红量不到任何东西。

## 测试计数也接上了

README 里那两处测试计数原先**没有东西在核**（同一种毛病：写下来的一句话，没有东西在它
后面），只是真值要跑一次收集才数得出来。现在接上了：起一个 `--collect-only` 子进程数
真值，和那两句比。量过：收集本身 1.5 秒，连解释器启动 7 秒。

代价是**每加一条测试都要来改一次那两句**——和上面"句子改了也会红"是同一笔交易，同样
是有意的。

子进程那段输出是 pytest 的界面文字，不是稳定接口。所以认不出来时必须**报错**，不许
当成"没意见"：一个什么都没数到的检查，看起来和数过了一模一样。

## τ-bench 那张工具属性表也接上了（第三个真值来源）

`tests/corpus/tau-tools-is-write.json` 是"每个工具改不改任务数据"的推导结果，
`tests/corpus/README.md` 里讲了它一行有几种结论、两条判据在几行上一致。那几个数原先
只活在一份仓库外脚本的终端输出里，搬进仓库时**全靠人抄**——抄对的和抄错的长得一模一样。

接上之后多出来两条只属于这张表的规矩：

  * **图例和逐行的 `how` 是两份名单，两边都要核。** 只核一边的话，"每行都标着判据甲"
    和"图例里躺着一条没人用的判据"各自看着都没问题；而判据乙是**交叉核对不是判据**，
    它一旦出现在行里，那几行的假阳性就变成了判定。
  * **`why` 空着的行，只允许是"两条判据都说是读"的行。** 写、人工裁决、两条判据打架，
    这三种都要有人话。它把"机械规则直接给的"和"有人看过的"分开——一张理由栏全空的表，
    看起来和一张每行都核对过的表一模一样。

第三个来源也逼出了 `_check` 的一个改动：它报错里那句"真值是从 `tests/corpus/*.yaml`
数出来的"原先写死在函数里，对另外两条路（`--collect-only`、这张 JSON）都是假话。
现在出处由调用方传进来——**报错正是在人最需要指路的时候出现的东西**，指错路比不指路贵。

## 量它会咬的是哪只探针

`../_probes/probe_doc_counts.py`：九条方向（正文条数、标定表格子、那张表的数字 /
`how` / `why`，外加一条**绿灯对照**——改一句不带数字的话必须还是绿的）。

2026-10-03 补上这条指针的时候，那只探针已经**烂了两轮**：它住在仓库外，仓库里
没有任何一处引它，而它的锚点是**文档里的原句**——语料从 49 条长到 55 条之后，
九条方向里有五条锚点落空，探针自己却照常打出一份"跑了 N 条"的报告。所以：改这里的
模板、或者改那几份文档里的句子，都要把探针重跑一遍；它报「锚点 0 次」时**别去查
闸门**，那是这条探针过期了。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

from sandbox.corpus import FAMILIES, evaluate_all, load_corpus, score

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "corpus"

#: 语料族以外的来源标签。分布里的每一个都要在文档里写出来。
SOURCES = ("自造", "公开-PERSIST", "公开-GhostWriter", "公开-tau-bench")

#: τ-bench 工具属性表。**它不是轨迹语料**（所以上面那些 `*.yaml` 的 glob 看不见它），
#: 是"每个工具改不改任务数据"这份推导结果，由 `tests/corpus/README.md` 讲给人听。
IS_WRITE = CORPUS / "tau-tools-is-write.json"

#: 表里可能出现的三个 `how` 值。**图例和代码两边都要对得上**（见
#: `test_every_row_says_which_method_judged_it`）：只对一边的话，把名字改一改就能让
#: "标着甲的行"变成标着别的什么的行，而闸门照样绿。
METHOD_AST = "判据甲（AST）"
METHOD_DESC = "判据乙（接口描述）"
METHOD_HAND = "人工裁决"
METHODS = (METHOD_AST, METHOD_DESC, METHOD_HAND)


def _flat(text: str) -> str:
    """压掉所有连续空白，让模板不受折行影响。见文件头。"""
    return re.sub(r"\s+", "", text)


def _counts() -> dict[str, int]:
    """从 yaml 数出来：总数、每个文件的条数。**不读文档**。"""
    counts: dict[str, int] = {}
    for path in sorted(CORPUS.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        counts[path.stem] = len(payload.get("items") or [])
    counts["总"] = sum(v for k, v in counts.items() if k != "总")
    return counts


def _source_distribution() -> dict[str, int]:
    items = load_corpus(CORPUS)
    return {name: sum(1 for item in items if item.source == name) for name in SOURCES}


def _check(where: str, template: str, values: dict[str, int], truth: str) -> str:
    """模板必须在那个文件里（压掉空白后）出现，且把数字代进去之后恰好命中一次。

    返回空串表示通过，否则是一句给人看的报错。

    `truth` 是报错里那句"真值是从哪来的"。**它必须由调用方说清楚**：这道闸门现在
    有三个真值来源（yaml、`--collect-only`、τ-bench 那张 JSON），一句写死的出处
    在另外两条路上就是假话——而报错正是在人最需要指路的时候出现的东西。
    """
    try:
        filled = template.format(**values)
    except KeyError as exc:
        return f"{where}：模板用了一个没数出来的占位符 {exc}（模板写错了，不是文档写错了）"
    text = _flat((ROOT / where).read_text(encoding="utf-8"))
    hits = text.count(_flat(filled))
    if hits == 1:
        return ""
    if hits == 0:
        return (
            f"{where}：找不到这句（或者数字对不上）：\n    {filled}\n"
            f"    真值{truth}：{values}\n"
            f"    改了文案就来改 tests/test_doc_counts.py 里对应的模板——"
            f"**这是这道闸门唯一想要的那一步**：改文案的人再看一眼那个数。"
        )
    return f"{where}：这句出现了 {hits} 次，闸门认不出该核哪一处：\n    {filled}"


def _collected_count() -> int:
    """起一个 `--collect-only` 子进程，问出这次会话一共收集多少条。

    **不递归**：子进程带 `--collect-only`，一条都不会执行。认不出那句汇总就报错，
    见文件头最后一段。
    """
    proc = subprocess.run(
        # `-p no:cacheprovider`：这条闸门是在 pytest 里起 pytest，两个进程会同时写
        # 同一份 `.pytest_cache`（Windows 上那是个可能被锁住的文件）。收集数不需要缓存。
        [sys.executable, "-m", "pytest", "--collect-only", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    match = re.search(r"(\d+) tests? collected", proc.stdout or "")
    assert match, (
        "子进程的输出里找不到那句「N tests collected」，这道闸门什么也没数到——"
        "**一条什么都没数到的检查，看起来和数过了一模一样**。子进程的输出：\n"
        + (proc.stdout or "")[-1500:]
        + "\n--- stderr ---\n"
        + (proc.stderr or "")[-1500:]
    )
    return int(match.group(1))


def test_the_test_count_in_readme_is_real() -> None:
    """README 里两处「N 个单测」是真的，不是抄剩下的。

    真值只有一个来源：当前这一份代码收集出来是多少条。**不能从别处抄**——抄来的数
    和真的数长得一模一样，直到有人加了一条测试。
    """
    real = _collected_count()
    flat = _flat((ROOT / "README.md").read_text(encoding="utf-8"))
    templates = ("{n} 个单测覆盖网关", "{n} 个单测；tests/corpus/ 是标注语料")
    missing = [t.format(n=real) for t in templates if _flat(t.format(n=real)) not in flat]

    # 这里**刻意不复用 `_check`**：它那句报错写死了"真值从 yaml 数出来""去改模板"，
    # 对这条闸门两句都是假的——真值是收集出来的，要改的是 README。第一版复用了它，
    # 探针跑出来的报错就在指一条错的路。
    assert not missing, (
        f"README 里的测试计数和真值（{real} 条，`--collect-only` 数出来的）对不上。"
        f"要改的是 README，不是这里；这两句该长这样：\n  " + "\n  ".join(missing)
    )


def test_every_source_in_use_is_named_in_the_distribution() -> None:
    """语料里真用到的来源，每一个都必须在上面那份名单里。

    上面那份 `SOURCES` 是**第二份**名单：真值在 `sandbox.corpus.SOURCES`（加载期按它
    拦来源），文档那句分布句是照它拼的。两份名单原先没有东西对上，于是加一个新来源
    时坏的那一边是静默的：分布里数不到它，文档那句也就不必提它——**少写一个来源和
    全都数过长得一模一样**，而"来源"正是读报告的人判断可信度的依据。

    比的是"语料里真用到的"而不是"白名单里的"：白名单里有还没用上的词（`访谈`），
    那不该逼着文档写一条 `访谈 0`。
    """
    used = {item.source for item in load_corpus(CORPUS)}
    missing = sorted(used - set(SOURCES))
    assert not missing, (
        f"语料里用到了这些来源 {missing}，但这份名单和文档里的分布句都没跟上。"
        f"补进 `SOURCES`，再改三份文档里那句分布句（模板在本文件 `build_places`）。"
    )


def build_places(values: dict[str, int]) -> list[tuple[str, str]]:
    """各处要核的原句。占位符和 `values` 的键一一对应。"""
    return [
        # --- 语料自己的 README：逐文件的条数 + 来源分布 + 总数 ---
        (
            "tests/corpus/README.md",
            "{总} 条：`loop.yaml` {loop} 条、`idempotency.yaml` {idempotency} 条、"
            "`autonomy.yaml` {autonomy} 条、`memory.yaml` {memory} 条。"
            "来源分布：自造 {自造}、公开-PERSIST {公开-PERSIST}、"
            "公开-GhostWriter {公开-GhostWriter}、公开-tau-bench {公开-tau-bench}。",
        ),
        ("tests/corpus/README.md", '人手挑出来的 {总} 条轨迹"'),
        ("tests/corpus/README.md", "{总} 条里一条误报就能让精确率掉几个百分点"),
        # --- 检测器文档：标定表的小标题 + 结论里重复的那个数 ---
        ("docs/detectors.md", "### 实测（2026-10-02，{总} 条语料）"),
        ("docs/detectors.md", "{总} 条里一条误报就能让精确率掉几个百分点"),
        ("docs/detectors.md", "已交付 {总} 条手写并逐条标注的轨迹"),
        # --- 顶层 README：四处正文 ---
        ("README.md", "现在 `tests/corpus/` 有 {总} 条"),
        ("README.md", "{总} 条里一条误报就能让数字掉几个百分点"),
        ("README.md", "**标注语料**（{总} 条手写轨迹逐条命中"),
        ("README.md", '人手挑出来的 {总} 条"'),
        ("README.md", "tests/corpus/ 是标注语料（{总} 条，见它的 README）"),
    ]


def test_corpus_counts_in_docs_match_the_yaml() -> None:
    """三个文档里写下的语料规模，等于从 yaml 数出来的那个。"""
    values = {**_counts(), **_source_distribution()}
    truth = "是从 tests/corpus/*.yaml 数出来的"
    problems = [
        msg for where, tpl in build_places(values) if (msg := _check(where, tpl, values, truth))
    ]
    assert not problems, "文档里的语料数字对不上 yaml：\n  " + "\n  ".join(problems)


# --- τ-bench 那张工具属性表：数字从 JSON 来，不从手写来 -----------------------


def _is_write_rows() -> dict[str, dict[str, object]]:
    payload = json.loads(IS_WRITE.read_text(encoding="utf-8"))
    return payload["tools"]


def _agrees(row: dict[str, object]) -> bool:
    """两条判据都表了态、而且说的是同一件事。"""
    return row["how"] == METHOD_AST and row["by_description"] == row["is_write"]


def _disagrees(row: dict[str, object]) -> bool:
    return row["how"] == METHOD_AST and row["by_description"] != row["is_write"]


def _is_write_tally() -> dict[str, int]:
    """从 JSON 数出来：行数、写/读、一致、不一致、人工裁决。**不读文档**。"""
    rows = _is_write_rows()
    writes = sum(1 for row in rows.values() if row["is_write"])
    return {
        "行": len(rows),
        "写": writes,
        "读": len(rows) - writes,
        "一致": sum(1 for row in rows.values() if _agrees(row)),
        "不一致": sum(1 for row in rows.values() if _disagrees(row)),
        "人工": sum(1 for row in rows.values() if row["how"] == METHOD_HAND),
    }


def build_is_write_places(values: dict[str, int]) -> list[tuple[str, str]]:
    """`tests/corpus/README.md` 里讲那张表的那句原话。"""
    return [
        (
            "tests/corpus/README.md",
            "{行} 行里，两条判据一致的 {一致} 行；不一致的 {不一致} 行；"
            "机械规则答不了、人读了代码裁的 {人工} 行。写 {写} 行、读 {读} 行。",
        ),
    ]


def test_the_is_write_numbers_in_the_readme_are_computed() -> None:
    """那张表给了几个数，文档里就得是那几个数——**现算的，不是抄的**。

    这几个数原先只活在那份脚本的终端输出里，搬进仓库时全靠人抄；而抄错的和抄对的
    长得一模一样，直到有人去数。现在左边是 JSON，右边是那句原话。
    """
    values = _is_write_tally()
    truth = "是从 tests/corpus/tau-tools-is-write.json 数出来的"
    problems = [
        msg
        for where, tpl in build_is_write_places(values)
        if (msg := _check(where, tpl, values, truth))
    ]
    assert not problems, "那张 τ-bench 工具表的数字对不上 JSON：\n  " + "\n  ".join(problems)


def test_every_row_says_which_method_judged_it() -> None:
    """每行的 `how` 是图例里的一个，而且**图例里每一条都得有行在用**。

    图例（JSON 的 `how`）和逐行的 `how` 是两份名单：只核一边的话，把某一行的
    `判据甲（AST）` 改成图例里没有的名字，或者把图例悄悄加一条没人用的，
    两边都会各自看着没问题。**判据乙尤其不许出现在行里**——它是交叉核对，
    不是判据；谁把它升成判据，那些假阳性就成了判定。
    """
    payload = json.loads(IS_WRITE.read_text(encoding="utf-8"))
    legend = set(payload["how"])
    assert legend == set(METHODS), (
        f"JSON 的图例是 {sorted(legend)}，本文件认的是 {sorted(METHODS)}——"
        f"两边要一起改，改完还要来改 tests/corpus/README.md 里那段话"
    )
    rows = _is_write_rows()
    used = {row["how"] for row in rows.values()}
    unknown = sorted(used - legend)
    assert not unknown, f"这些行标了一个图例里没有的判据：{unknown}"
    assert METHOD_DESC not in used, (
        "有行把判据乙当成了判据。它只是交叉核对——不一致的行由人看过之后采信甲，"
        "理由写在 why 里，那个理由才是这一行的判据"
    )
    unused = sorted(legend - used)
    assert unused == [METHOD_DESC], f"图例里这些判据没有任何一行在用：{unused}"


def test_only_rows_that_both_methods_call_reads_go_unexplained() -> None:
    """`why` 空着的行，只允许是**两条判据都说读**的行。

    写、人工裁决、以及两条判据打架的行，都要有一句人话说明它为什么是这个答案。
    这条规则不是编出来好看的：它把"这一行是机械规则直接给的"和"这一行有人看过"
    分开——**一张全是空理由的表，看起来和一张每行都核对过的表一模一样**。
    """
    rows = _is_write_rows()
    silent = sorted(
        name
        for name, row in rows.items()
        if not row["why"] and not (row["how"] == METHOD_AST and not row["is_write"]
                                   and not row["by_description"])
    )
    assert not silent, (
        f"这些行的理由栏是空的，而它们不是「两条判据都说是读」的那种行：{silent}"
    )
    noisy = sorted(
        name
        for name, row in rows.items()
        if row["why"] and row["how"] == METHOD_AST and not row["is_write"]
        and not row["by_description"]
    )
    assert not noisy, f"这些行两条判据都说是读，却写了理由——理由栏不是许愿池：{noisy}"


def test_the_is_write_guard_is_not_vacuous() -> None:
    """相机：行数、两类的规模、三类的和都要对得上，且四类都得真出现过。

    上面几条的判据都是"没找到不一致"，而**一个只读到 0 行的检查**同样找不到不一致。
    """
    tally = _is_write_tally()
    rows = len(_is_write_rows())
    assert tally["行"] >= 25, f"只读到 {tally['行']} 行，这张表八成没读完"
    assert tally["一致"] + tally["不一致"] + tally["人工"] == rows, (
        f"三类加起来是 {tally['一致'] + tally['不一致'] + tally['人工']}，行数是 {rows}"
        "——有行两头都不算"
    )
    assert tally["不一致"] >= 1 and tally["人工"] >= 1, (
        f"一致 {tally['一致']} / 不一致 {tally['不一致']} / 人工 {tally['人工']}："
        "剩下的都一种，那几种理由（REVIEWED / ADJUDICATED）就没有一行在核了"
    )
    assert tally["写"] >= 8 and tally["读"] >= 8, f"写读分布塌了：{tally}"


#: 标定表那一段的小标题。表格在这句之后、下一个空行之前。
_TABLE_HEADING = "| 族 | TP | FP | FN | TN | 精确率 | 召回率 |"


def test_the_calibration_table_in_docs_matches_the_real_run() -> None:
    """`docs/detectors.md` 第七节那张表，是**跑出来的**，不是抄下来的。

    十六个数（四族 × 四格）加上八个比率，原先全靠手抄——而手抄的表和跑出来的表
    长得一模一样，直到语料变了。这里把两边接上：跑一遍语料，比每一行。
    """
    results = evaluate_all(load_corpus(CORPUS))
    real = score(results)

    text = (ROOT / "docs" / "detectors.md").read_text(encoding="utf-8")
    start = text.find(_TABLE_HEADING)
    assert start != -1, f"docs/detectors.md 里找不到标定表（'{_TABLE_HEADING}'）"
    tail = text[start:].split("\n\n", 1)[0]

    rows: dict[str, str] = {}
    for line in tail.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 7 and cells[0] in FAMILIES:
            rows[cells[0]] = line.strip()

    assert sorted(rows) == sorted(FAMILIES), (
        f"标定表里只有这些族：{sorted(rows)}，应当是 {sorted(FAMILIES)}"
    )

    problems = []
    for family in FAMILIES:
        s = real[family]
        want = (
            f"| {family} | {s.tp} | {s.fp} | {s.fn} | {s.tn} "
            f"| {_pct(s.precision)} | {_pct(s.recall)} |"
        )
        if rows[family] != want:
            problems.append(f"  {family}\n    文档里：{rows[family]}\n    跑出来：{want}")
    assert not problems, (
        "docs/detectors.md 的标定表和实跑对不上（改语料之后忘了重跑？）：\n"
        + "\n".join(problems)
    )


def _pct(value: float | None) -> str:
    """和 `scripts/calibrate.py` 的 `_pct` 同一个写法：一位小数的百分比。"""
    return "—" if value is None else f"{value * 100:.1f}%"


def test_the_guard_is_not_vacuous() -> None:
    """闸门自己也要有相机：句子数和族数都不能塌。

    上面两条的判据都是"没找到不一致"，而**一个什么都没扫到的检查**同样没找到
    不一致——两句报错都是空的，看起来一模一样。所以把规模钉住。
    """
    values = {**_counts(), **_source_distribution()}
    places = build_places(values) + build_is_write_places(_is_write_tally())
    assert len(places) >= 10, f"只盯了 {len(places)} 处，闸门八成被改窄了"
    files = {where for where, _ in places}
    assert files == {"README.md", "docs/detectors.md", "tests/corpus/README.md"}, (
        f"盯的文件变成了 {sorted(files)}"
    )
    assert values["总"] >= 40, f"数出来的语料只有 {values['总']} 条，八成没读到文件"
