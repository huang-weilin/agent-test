"""用户能看到的文案里不许出现 Markdown 的星号。

同一个字符串有两条出路：命令行报告（原样打印）和报告页面（`<strong>` 之外的地方
按纯文本转义）。Markdown 的 `**粗体**` 在这两条路上**都不生效**——终端里显示成
两个星号，页面上被转义后也显示成两个星号。所以强调只有一个写法：页面里写
`<strong>`，终端和纯文本里就靠措辞和 `!!` 这类前缀。

这个缺陷在本项目里复发了三次（页面模板、场景 description、Python 里的报告文案），
所以各留一道静态闸门：

  * 模板 —— `tests/test_api.py` 的 `test_templates_do_not_render_markdown_asterisks`
  * 场景 YAML 的自由文本 —— 本文件
  * Python 源码里的非文档字符串 —— 本文件

**为什么是"非文档字符串"而不是"能到报告的字符串"**：靠肉眼判断哪个字面量会进报告，
迟早会漏，而漏掉的正是没人想到的那条路径。反过来，文档字符串和注释是给读代码的
人看的，那里的 `**` 是有用的强调（本项目的注释一直这么写），必须放行。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "sandbox"
SCENARIOS = ROOT / "scenarios"
CORPUS = ROOT / "tests" / "corpus"


def _docstring_node_ids(tree: ast.Module) -> set[int]:
    """模块/类/函数的首条语句如果是字符串，那是文档字符串——放行。"""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            ids.add(id(first.value))
    return ids


def _string_literals(path: Path):
    """产出 `(行号, 文本)`，跳过文档字符串。f-string 的各段也在这里面。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = _docstring_node_ids(tree)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            yield node.lineno, node.value


def test_python_report_strings_have_no_markdown_asterisks() -> None:
    offenders = [
        f"{path.relative_to(ROOT)}:{lineno}: {text.strip()[:100]}"
        for path in sorted(SRC.rglob("*.py"))
        for lineno, text in _string_literals(path)
        if "**" in text
    ]
    assert not offenders, (
        "这些字符串里有 Markdown 星号（终端和页面都不会渲染它）：\n  "
        + "\n  ".join(offenders)
    )


def _descriptions(node, path: str = "") -> list[tuple[str, str]]:
    """把 YAML 里所有 `description` 字段都找出来——不只有顶层那一个。

    工具声明上的 description 也会被渲染到页面上，只查顶层就漏了一半。
    """
    found: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else str(key)
            if key == "description" and isinstance(value, str):
                found.append((here, value))
            else:
                found.extend(_descriptions(value, here))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_descriptions(value, f"{path}[{index}]"))
    return found


@pytest.mark.parametrize(
    "directory",
    [SCENARIOS, SCENARIOS / "_shared", CORPUS],
    ids=["scenarios", "shared", "corpus"],
)
def test_free_text_fields_have_no_markdown_asterisks(directory: Path) -> None:
    """场景、工具库和语料里由人写、又被原样打印出来的那些字段。

    这三处都不认 Markdown：页面和命令行都原样打印，写 `**` 只会让读的人看见星号。

    工具库要在这一列里：工具声明的 description 会渲染到页面的工具表上，而它现在
    住在库文件里——不列进来的话，那一段文字**恰好**从这道闸门底下漏过去。
    """
    offenders: list[str] = []
    for path in sorted(directory.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for where, text in _descriptions(payload):
            if "**" in text:
                offenders.append(f"{path.name}: {where}: {text.strip()[:100]}")
        # 语料还多一个自由文本字段：标注依据。
        items = payload.get("items") if isinstance(payload, dict) else None
        for index, item in enumerate(items or []):
            why = item.get("why") if isinstance(item, dict) else None
            if isinstance(why, str) and "**" in why:
                offenders.append(f"{path.name}: items[{index}].why")
    assert not offenders, "自由文本里有 Markdown 星号：\n  " + "\n  ".join(offenders)
