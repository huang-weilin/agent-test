"""文档里写着"这东西没接上"的地方，它必须真的没接上。

## 为什么单开一道

这个仓库已经有两道看文档的闸门，各自管一类**机械可查**的东西：路径在不在
（`tests/test_doc_paths.py`）、数字对不对（`tests/test_doc_counts.py`）。它们能成立
是因为那两类都有一条客观判据——文件在磁盘上、计数由 `--collect-only` 数出来。

"文档说某功能没做"没有这种判据。绝大多数这一类句子只能靠人读，而**读的人不会
去核**：他看到一句"适配层没写"，默认它是当真的。这句谎话的成本在于它让读者**
不去找**——一个以为已经有人做了的功能，不会有人去想"那我去看看它怎么用"。

所以这道闸门只收**当场能查的**那一类：句子说的是"仓库里没有 X"，而 X 是一种可以
按标识符扫出来的东西。收进来的第一条也是目前唯一一条：README 的"明确不做"那节
写着 **Langfuse 的适配层没写**、三个 `SANDBOX_LANGFUSE_*` 配置项没有任何代码读它们。

这句话是 2026-10-02 手工核出来的——原文写的是"Langfuse 已有，只做对接适配层"，
而适配层一行都没有，`trace_ref`（ADR 002 第 3 条点名要写的）全仓零处。**手核一次
不产生任何持续的保证**，所以核完把判据留在这里。

## 它红了要改什么

红了只有两种可能，两种都要人来看：

* **真的有人接上了适配层**（扫到了读者）——那 README 那句、缺口表那一行、ADR 002
  的那节现状说明，三处都该改掉。**闸门不是要阻止这件事**，是要让那句旧话跟着动。
* **README 里那句话被改写或删掉了**（锚点找不到了）——那是锚点，不是原件：改措辞
  就得顺手改锚点，否则这道闸门会空转成一条永远绿的测试。

## 这道闸门自己也会被验

`test_the_scanner_finds_a_reader_when_there_is_one` 往临时目录里放两份样例：一份
真的读配置、一份只在文档字符串里提到 Langfuse。前者必须被扫出来，后者必须不被扫
出来。**没有这一条，"扫描器写错了"和"仓库里确实没有读者"长得一模一样**——这正是
本项目反复撞见的那件事（一条测不出退化的闸门看起来和好闸门一模一样）。
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 扫哪些树找读者。`scripts/` 必须在内：报告的另一个出口在那儿（同一条理由见
#: `tests/test_report_text.py` 的文件头），跨运行报告与冒烟脚本都会 import 配置。
CODE_TREES = (ROOT / "src" / "sandbox", ROOT / "scripts")

#: 被看着的四个名字。前三个是字段，第四个是那个便利属性——有人接适配层的话，
#: 最容易先落在它上面。
_WATCHED = frozenset(
    {"langfuse_host", "langfuse_public_key", "langfuse_secret_key", "langfuse_enabled"}
)

#: 唯一的豁免：字段的定义处，以及 `langfuse_enabled` 自己（它读的就是那三个字段）。
#:
#: **代价写在明处**：这道闸门整个跳过这个文件。往它里面塞一个真的会出网的函数，
#: 这里看不见。之所以接受：`config.py` 是配置声明的地方，一个"只声明不出网"的
#: 约定比一个每次加字段都要重扫一遍的规则好核。真接了适配层，代码不会住在这里。
_EXEMPT = frozenset({ROOT / "src" / "sandbox" / "config.py"})

#: 豁免名单**允许**的样子。单独写一份、而不是把字面量直接摆进断言：这样"改豁免"
#: 在 diff 里是两处改动，第二次动手之前总得过一眼上面那段理由。
_EXPECTED_EXEMPT = {ROOT / "src" / "sandbox" / "config.py"}

#: README 里那句话的锚点。**改措辞就要改这里**（见文件头第二条）。
README_ANCHOR = "Langfuse 的适配层没写"
README = ROOT / "README.md"


def _readers(tree: Path, exempt: frozenset[Path] = frozenset()) -> dict[str, list[str]]:
    """扫一棵树，返回 `文件 -> ["行号: 命中的名字", ...]`。

    走 AST 而不是 `grep langfuse`：注释和文档字符串里提到 Langfuse 的地方有好几处
    （`src/sandbox/__init__.py` 的 ADR 清单、`config.py` 的注释、本文件），
    按字面扫会把它们全算成读者，而**一道会误报的闸门迟早被整体忽略**。
    只认真的在读值的那种写法：属性名、裸名字、关键字参数名。

    认不出来的写法（`getattr(settings, "langfuse_host")`、把名字拼出来）它看不见。
    这是绊线不是证明——它挡的是"顺手接上了却没改文档"，不是"故意绕过去"。
    """
    hits: dict[str, list[str]] = {}
    for path in sorted(tree.rglob("*.py")):
        if path in exempt or "__pycache__" in path.parts:
            continue
        found: list[str] = []
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            name: str | None = None
            if isinstance(node, ast.Attribute) and node.attr in _WATCHED:
                name = node.attr
            elif isinstance(node, ast.Name) and node.id in _WATCHED:
                name = node.id
            elif isinstance(node, ast.keyword) and node.arg in _WATCHED:
                name = node.arg
            if name is not None:
                found.append(f"{node.lineno}: {name}")
        if found:
            hits[path.relative_to(tree).as_posix()] = found
    return hits


def test_nothing_reads_the_langfuse_settings() -> None:
    """README 说"No one reads them"。为真的话，这一条绿。

    红了先别急着改代码：先去看 README 那句、缺口表那一行、ADR 002 的现状那节——
    三处说的都是"没接上"。
    """
    hits = {f"src/sandbox/{k}": v for k, v in _readers(CODE_TREES[0], _EXEMPT).items()}
    hits.update({f"scripts/{k}": v for k, v in _readers(CODE_TREES[1]).items()})

    assert hits == {}, (
        "README 写着 Langfuse 的适配层没写、三个 SANDBOX_LANGFUSE_* 没有任何读者，"
        "但扫描扫到了这些：\n  "
        + "\n  ".join(f"{f}  {', '.join(v)}" for f, v in sorted(hits.items()))
        + "\n\n要么是真的接上了（那就把 README 的"
        + f"「{README_ANCHOR}」那句、缺口表里那一行、ADR 002 的现状那节一起改掉），"
        "要么是这次改动其实没有接上任何东西、只是把名字搬了个地方。"
    )


def test_the_readme_still_says_the_adapter_is_not_written() -> None:
    """锚点还在不在。

    上面那条判的是"确实没人读"，这一条判的是"那句话还在"。少掉这一条的话，
    把 README 那句删掉、上面那条就变成了一条空转的绿灯——**它照样报通过，
    但已经什么都不在看了**。
    """
    text = README.read_text(encoding="utf-8")
    assert README_ANCHOR in text, (
        f"README 里找不到锚点「{README_ANCHOR}」。这道闸门盯的就是那句话；"
        "改了措辞就把本文件顶上的 README_ANCHOR 一起改，别把闸门留成一条空转的绿灯"
    )
    # 那句话得点名被看着的配置项，否则它约束不到上面那条扫描。
    assert "SANDBOX_LANGFUSE_HOST" in text, (
        "README 那句话没点名 `SANDBOX_LANGFUSE_HOST`——它得说清是哪些东西没人读，"
        "否则读者（和这道闸门）都不知道那句话覆盖到哪里"
    )


def test_the_scanner_finds_a_reader_when_there_is_one(tmp_path: Path) -> None:
    """自证：同一个扫描器，喂一个真的读者必须扫得出来，喂一段纯提及必须扫不出来。

    两份样例只差**一行代码**——一处 `settings.langfuse_host`，一处是把 Langfuse
    写进文档字符串。第二份是这道扫描为什么要走 AST 的那条理由本身。
    """
    tree = tmp_path / "src" / "sandbox"
    tree.mkdir(parents=True)
    reads_it = (
        '"""一个真的读者。"""\n'
        "from sandbox.config import settings\n"
        "URL = settings.langfuse_host\n"
    )
    only_says_it = (
        '"""只是提到：langfuse_host 这里没人读。"""\n'
        "# langfuse_enabled 也没人读\n"
        "X = 1\n"
    )
    # `write_bytes` 而不是 `write_text`：本机 `write_text` 落的是 CRLF，
    # 而这一对样例是拿来喂 `ast.parse` 的，内容得和写在源码里的一模一样。
    (tree / "reads_it.py").write_bytes(reads_it.encode("utf-8"))
    (tree / "only_says_it.py").write_bytes(only_says_it.encode("utf-8"))

    assert _readers(tree) == {"reads_it.py": ["3: langfuse_host"]}, (
        "扫描器没认出真的读者（那上面那条就永远绿），或者把注释里的提及也算成了读者"
        "（那上面那条会一直红）——两种都让这道闸门失效"
    )


def test_the_exempt_file_is_the_only_hole() -> None:
    """豁免名单就是"这道闸门看不见的地方"，所以它得有边界、且不能悄悄长大。

    `config.py` 今天是豁免的唯一一项：字段定义处 + `langfuse_enabled` 自己。
    """
    assert _EXEMPT == _EXPECTED_EXEMPT, (
        "豁免名单变了。这道闸门整个跳过被豁免的文件，所以每加一条都是在它身上开个洞"
        "——加之前先想清楚为什么那条不能改代码、只能改闸门"
    )
