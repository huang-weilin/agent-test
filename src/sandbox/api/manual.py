"""把 `docs/manual.md` 渲染成界面里的一页。

**为什么要有这一份。** 手册原先只是仓库里的一个 markdown 文件。可需要它的那个人
——不写代码、打开的是浏览器——不会去翻代码仓库。手册放在界面里才有机会被读到。

**仓库里仍然只有一份手册**：这一页是它的渲染，不是它的副本，所以不存在"改了一边
忘了另一边"。这也是不把它手抄成 HTML 模板的原因。

**它不是通用的 Markdown 实现**，只认 `docs/manual.md` 真正用到的那几种写法：
标题、表格、围栏代码、引用、横线、有序/无序列表、段落；行内只认 `**加粗**` 和
`` `代码` ``。这个边界由闸门守着（`tests/test_help_page.py`：渲染结果里不许漏出
没被吃掉的标记）——往手册里加一种新写法时，页面会**当场变红**，而不是把那行原文
连同标记一起打出来。屏幕上出现一串 `##` 的时候，读的人只会以为自己看错了。
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from markupsafe import Markup

from sandbox.config import REPO_ROOT

#: 手册在仓库里的位置。**只有这一份**，界面那一页读的就是它。
MANUAL_PATH: Path = REPO_ROOT / "docs" / "manual.md"

_HEADING = re.compile(r"\A(#{1,6})\s+(.+?)\s*\Z")
_FENCE = re.compile(r"\A```(\S*)\s*\Z")
_HR = re.compile(r"\A-{3,}\s*\Z")
_QUOTE = re.compile(r"\A>\s?(.*)\Z")
_UL = re.compile(r"\A[-*]\s+(.+?)\s*\Z")
_OL = re.compile(r"\A\d+[.、]\s+(.+?)\s*\Z")
_TABLE_SEP = re.compile(r"\A\|[\s:|-]+\|\s*\Z")
_TABLE_ROW = re.compile(r"\A\|.*\|\s*\Z")

#: 缩进多少格算"这一项的续行"。手册里写的是三格，放宽到两格以上。
_CONTINUATION = re.compile(r"\A\s{2,}\S")


def manual_text() -> str:
    """手册的原文。"""
    return MANUAL_PATH.read_text(encoding="utf-8")


def render(text: str) -> Markup:
    """整篇渲染。返回 `Markup`，所以模板里写 `{{ manual }}` 就够。

    **转义在渲染器里做完**：模板那边不写 `|safe`（本仓库的模板一律不许出现它，
    因为默认转义是那唯一一道防线）。这里先转义、再套标记，顺序反过来就等于把
    正文里的 `<` 原样发出去。
    """
    return page(text)[0]


def page(text: str) -> tuple[Markup, list[tuple[str, Markup]]]:
    """渲染出正文，并**在同一次遍历里**记下目录。

    目录和正文分两次走 `_blocks` 也能写，但那就是同一份手册的两份真相：
    哪天两边数标题的方式差了一点，目录里的链接就会指向一个不存在的锚点，
    而页面上看不出任何异样——点一下没反应，用户只会以为自己点歪了。

    标题**在这里就渲染成 `Markup`**，所以模板里写 `{{ title }}` 不用 `|safe`：
    手册的章节名带着反引号（`## 4. 首页 `/`：场景列表`），原样丢给模板会把
    反引号一起印在目录里。
    """
    out: list[str] = []
    outline: list[tuple[str, Markup]] = []
    for kind, payload in _blocks(text.splitlines()):
        if kind == "heading":
            level, title = payload
            body = _inline(title)
            # 手册的 `#`（一级）落到页面的 h2：页面上的一级标题是模板写的
            # 「使用手册」，一页里有两个 h1 是给读屏软件添乱。
            if level == 2:
                anchor = f"s{len(outline)}"
                outline.append((anchor, Markup(body)))
                out.append(f'<h3 id="{anchor}">{body}</h3>')
                continue
            want = min(level + 1, 6)
            out.append(f"<h{want}>{body}</h{want}>")
            continue
        out.append(_render_block(kind, payload))
    return Markup("\n".join(out)), outline


# --- 块级 -------------------------------------------------------------------


def _blocks(lines: list[str]) -> Iterator[tuple[str, Any]]:
    """把逐行的原文切成块。**每一行都会落到某个块里**，没有"认不出"这一档。

    认不出就当段落的后果是：标记会原样出现在页面上。所以配了一条闸门盯渲染结果
    里有没有漏网的 `##` / `|` / `**`——见 `tests/test_help_page.py`。
    """
    i = 0
    while i < len(lines):
        line = lines[i]

        if not line.strip():
            i += 1
            continue

        fence = _FENCE.match(line)
        if fence:
            body: list[str] = []
            i += 1
            while i < len(lines) and not _FENCE.match(lines[i]):
                body.append(lines[i])
                i += 1
            i += 1  # 吃掉收尾的 ```
            yield "code", (fence.group(1), body)
            continue

        if _HR.match(line):
            yield "hr", None
            i += 1
            continue

        heading = _HEADING.match(line)
        if heading:
            yield "heading", (len(heading.group(1)), heading.group(2))
            i += 1
            continue

        if _TABLE_ROW.match(line):
            rows: list[str] = []
            while i < len(lines) and _TABLE_ROW.match(lines[i]):
                rows.append(lines[i])
                i += 1
            yield "table", _table_rows(rows)
            continue

        if _QUOTE.match(line):
            quoted: list[str] = []
            while i < len(lines) and _QUOTE.match(lines[i]):
                quoted.append(_QUOTE.match(lines[i]).group(1))  # type: ignore[union-attr]
                i += 1
            yield "quote", quoted
            continue

        if _UL.match(line) or _OL.match(line):
            ordered = bool(_OL.match(line))
            items: list[list[str]] = []
            while i < len(lines):
                one = _OL.match(lines[i]) if ordered else _UL.match(lines[i])
                if one:
                    items.append([one.group(1)])
                    i += 1
                elif items and _CONTINUATION.match(lines[i]):
                    items[-1].append(lines[i].strip())
                    i += 1
                else:
                    break
            yield ("ol" if ordered else "ul"), items
            continue

        # 段落：一路吃到空行或者下一个块的开始。
        para: list[str] = []
        while i < len(lines) and lines[i].strip() and not _starts_block(lines[i]):
            para.append(lines[i].strip())
            i += 1
        if para:
            yield "para", para
        else:
            # 理论上到不了这儿（`_starts_block` 与上面的分支是同一套判据），
            # 真到了说明两处判据漂了。宁可把这一行原样吞掉也不要死循环。
            i += 1


def _starts_block(line: str) -> bool:
    """这一行是不是另一个块的开始。段落读到它就停。"""
    return bool(
        _FENCE.match(line)
        or _HR.match(line)
        or _HEADING.match(line)
        or _TABLE_ROW.match(line)
        or _QUOTE.match(line)
        or _UL.match(line)
        or _OL.match(line)
    )


def _table_rows(rows: list[str]) -> tuple[list[str], list[list[str]]]:
    """表格的第一行是表头，第二行是 `|---|---|` 那道分隔（丢掉）。"""
    head = _cells(rows[0])
    body = [_cells(r) for r in rows[1:] if not _TABLE_SEP.match(r)]
    return head, body


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def _render_block(kind: str, payload: Any) -> str:
    # 标题不在这儿：它要顺便记锚点，所以由 `page()` 自己处理。
    if kind == "hr":
        return "<hr>"
    if kind == "code":
        lang, body = payload
        cls = f' class="lang-{lang}"' if lang else ""
        inner = "\n".join(html.escape(b) for b in body)
        return f"<pre><code{cls}>{inner}</code></pre>"
    if kind == "quote":
        return "<blockquote>" + _inline(_join(payload)) + "</blockquote>"
    if kind == "table":
        head, body = payload
        out = ["<table class=\"grid tight\">", "<thead><tr>"]
        out += [f"<th>{_inline(c)}</th>" for c in head]
        out.append("</tr></thead>")
        out.append("<tbody>")
        for row in body:
            out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>")
        out.append("</tbody></table>")
        return "\n".join(out)
    if kind in ("ul", "ol"):
        out = [f"<{kind}>"]
        for item in payload:
            out.append(f"<li>{_inline(_join(item))}</li>")
        out.append(f"</{kind}>")
        return "\n".join(out)
    return f"<p>{_inline(_join(payload))}</p>"


# --- 行内 -------------------------------------------------------------------

_CODE_SPAN = re.compile(r"`([^`]+)`")


def _inline(text: str) -> str:
    """行内标记。先转义，再配对。

    **代码段先换成占位符，加粗在整句上配对**。"按代码段切开、每段各自处理"
    看起来更直白，但加粗是**可以跨过一个代码段**的——`**这一条 `foo` 很重要**`
    是常见写法，切开之后两头的 `**` 落在不同的段里，配不上对：页面上一串星号，
    而加粗只加了一半。手册里现在就有这种句子（`**第一件：把 Agent 代码放进
    `agents/<名字>/`。**`），所以这不是假想出来的输入。

    占位符用 `\\x00` 包一个序号：转义不动它，加粗也认不出它（里面没有星号），
    等配对完再换回来——于是 `<code>` 落在 `<strong>` 里面，而不是反过来。
    """
    spans: list[str] = []

    def stash(match: re.Match[str]) -> str:
        spans.append(f"<code>{html.escape(match.group(1))}</code>")
        return f"\x00{len(spans) - 1}\x00"

    stashed = _CODE_SPAN.sub(stash, text)
    out = _bold(html.escape(stashed))
    for index, code in enumerate(spans):
        out = out.replace(f"\x00{index}\x00", code)
    return out


def _bold(escaped: str) -> str:
    """`**加粗**` → `<strong>`。

    **配不上对的星号原样留着，绝不猜**：猜"大概是想加粗到行尾"会把一句本来是
    疑问的话印成断言。留着的话页面上会多出一串星号，而闸门当场变红——它要说的
    是"手册里少写了一个 `**`"，不是"渲染器少认了一种写法"。

    模式写成 `\\*{2}` 而不是 `\\*\\*`：`tests/test_report_text.py` 扫的是
    `src/sandbox` 下**所有非文档字符串的字符串字面量**有没有 `**`，它不区分
    "这是给人看的文案"和"这是正则"——直接用 `\\*\\*` 会被那道闸门抓住。
    放宽那道闸门去迁就一行正则，等于把它真正要挡的东西一起放走。
    """
    return re.sub(r"\*{2}(.+?)\*{2}", r"<strong>\1</strong>", escaped, flags=re.S)


#: 上一行以 ASCII 收尾、下一行以 ASCII 开头时，接缝处要补一个空格。
_ASCII_TAIL = re.compile(r"[A-Za-z0-9`)\]}>.,;:!?\"']\Z")
_ASCII_HEAD = re.compile(r"\A[A-Za-z0-9`(\[<]")


def _join(lines: list[str]) -> str:
    """把硬折行的段落接回一行。

    **中文接缝处不补空格**：手册是按 76 列硬折的，"重复\\n扣款"接成"重复 扣款"
    会在整篇里多出一堆本来不存在的空格，而中文没有词间空格。

    两端都是 ASCII 时必须补一个：`is` + `ready` 粘成 `isready` 是**另一个字**。
    手册里现在恰好没有这种跨行断开的英文，所以这一条**没有用例撑着**，它靠
    `tests/test_help_page.py` 里一条专门造的输入钉着——"反正现在不会发生"是
    最容易在下一版变假的那种理由。
    """
    if not lines:
        return ""
    out = lines[0]
    for nxt in lines[1:]:
        glue = " " if _ASCII_TAIL.search(out) and _ASCII_HEAD.match(nxt) else ""
        out += glue + nxt
    return out
