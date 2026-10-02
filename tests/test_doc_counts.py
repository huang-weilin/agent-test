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

## 这道闸门**没有**盖住的东西

README 里那两处"932 个单测"不在这里，而且**仍然没有东西在核**。它是同一种毛病
（写下来的一句话，没有东西在它后面），只是真值要跑一次收集才数得出来，而
"在测试里起一个 `--collect-only` 子进程"既慢又绕。写在这里是为了让人知道这个
边界在哪儿——**"没有红"和"核过了"不是一回事**，这一点在这个仓库里反复出现。
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from sandbox.corpus import FAMILIES, evaluate_all, load_corpus, score

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "corpus"

#: 语料族以外的来源标签。分布里的每一个都要在文档里写出来。
SOURCES = ("自造", "公开-PERSIST", "公开-GhostWriter")


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


def _check(where: str, template: str, values: dict[str, int]) -> str:
    """模板必须在那个文件里（压掉空白后）出现，且把数字代进去之后恰好命中一次。

    返回空串表示通过，否则是一句给人看的报错。
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
            f"    真值是从 tests/corpus/*.yaml 数出来的：{values}\n"
            f"    改了文案就来改 tests/test_doc_counts.py 里对应的模板——"
            f"**这是这道闸门唯一想要的那一步**：改文案的人再看一眼那个数。"
        )
    return f"{where}：这句出现了 {hits} 次，闸门认不出该核哪一处：\n    {filled}"


def build_places(values: dict[str, int]) -> list[tuple[str, str]]:
    """各处要核的原句。占位符和 `values` 的键一一对应。"""
    return [
        # --- 语料自己的 README：逐文件的条数 + 来源分布 + 总数 ---
        (
            "tests/corpus/README.md",
            "{总} 条：`loop.yaml` {loop} 条、`idempotency.yaml` {idempotency} 条、"
            "`autonomy.yaml` {autonomy} 条、`memory.yaml` {memory} 条。"
            "来源分布：自造 {自造}、公开-PERSIST {公开-PERSIST}、"
            "公开-GhostWriter {公开-GhostWriter}。",
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
    problems = [msg for where, tpl in build_places(values) if (msg := _check(where, tpl, values))]
    assert not problems, "文档里的语料数字对不上 yaml：\n  " + "\n  ".join(problems)


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
    places = build_places(values)
    assert len(places) >= 10, f"只盯了 {len(places)} 处，闸门八成被改窄了"
    files = {where for where, _ in places}
    assert files == {"README.md", "docs/detectors.md", "tests/corpus/README.md"}, (
        f"盯的文件变成了 {sorted(files)}"
    )
    assert values["总"] >= 40, f"数出来的语料只有 {values['总']} 条，八成没读到文件"
