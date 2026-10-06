"""使用手册那一页（`/help`）以及把它渲染出来的那个渲染器。

手册原先只是仓库里的一个 markdown 文件，而需要它的那个人不翻代码仓库。所以
`docs/manual.md` 现在多了一条出口：界面里的一页。**仓库里仍然只有一份手册**，
这一页是它的渲染——这条约束由下面第一组用例守着（页面上的正文必须来自那个文件）。

这个渲染器**不是通用的 Markdown 实现**，只认手册真正用到的那几种写法。它的失败
方式和别处不一样，所以盯的是这四种：

**一、"看起来渲染了"。** 认不出的一行不会报错，它会带着标记原样印在页面上——
读者看见 `## 7. 运行报告页`，只会以为自己看错了。这道闸门（`_visible` 那两条）
把渲染结果里已经是结构的部分挖掉，剩下的字符里不许再出现任何标记。往手册里
加一种新写法时，页面**当场变红**。

**二、"整行不见了"。** 比多出几个星号更糟：一段话静默消失，页面上没有任何
异常，而手册里写着的东西用户看不到。所以另有一条**逐行核对**（手册里每个
非空行去掉标记之后，必须能在渲染结果里找到）。

**三、加粗跨过一个代码段。** `**这一条 `foo` 很重要**` 是常见写法，而"按代码
段切开、每段各自配对"会让两头的星号落进不同的段里——页面上出现一串星号，
而加粗只加了一半。这条不是假想的：第一版渲染器就是这么写的，实测从真手册里
漏出 8 个 `**`（那 8 个全在代码段旁边），是渲染器的单元用例抓到的。

**四、手册和界面各说各的。** 手册第 2 节列的是顶栏那排入口。顶栏加了入口而
手册没加，"九个入口"这句话就是假的——而它**原本已经是假的**：标题写着九个，
表里只列了八个（数于 2026-10-03，加 `/help` 这一页之前）。所以这条是**现算**的：
从渲染出来的页面里取顶栏链接，从手册原文里取第 2 节的表，两个列表必须逐项相等，
标题里那个中文数字还得等于它们的条数。

渲染器只读文件、不碰库，所以下面那组页面用例不需要数据库——但 `create_app`
会被别的东西带着建库，走 `temp_db` 那条夹具最省事。
"""

from __future__ import annotations

import html as html_module
import re
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

from jinja2 import Environment, FileSystemLoader

import httpx
import pytest
import pytest_asyncio

from sandbox.api import manual
from sandbox.api.app import create_app

ROOT = Path(__file__).resolve().parents[1]
BASE_TEMPLATE = ROOT / "src" / "sandbox" / "api" / "templates" / "base.html"


# --- 渲染器：认得出、不漏标记 -----------------------------------------------

#: 挖掉"已经是结构"的部分，剩下的才是会原样落到读者眼里的字符。
#: 代码段和围栏代码要一起挖：`**` 在 `pre` 里是**正文**（手册讲的就是这些标记本身），
#: 不挖掉的话这条闸门会被自己的说明书绊倒。
#:
#: **顺序要紧**：先挖代码、再剥标签、最后还原实体。剥标签这一步是必需的——
#: 认不出的一行会掉进段落，渲染成 `<p>| 甲 | 乙</p>`，而按渲染结果的行首去
#: 找竖线是找不到的（那一行以 `<` 开头）。第一版就是这么写的，于是闸门对
#: **最可能发生的那种泄漏**视而不见，而它看起来完全正常。
_PRE = re.compile(r"<pre>.*?</pre>", re.S)
_CODE = re.compile(r"<code>.*?</code>", re.S)
_TAG = re.compile(r"<[^>]+>")


def _visible(markup: object) -> str:
    text = _CODE.sub("", _PRE.sub("", str(markup)))
    return html_module.unescape(_TAG.sub("", text))


#: 没被吃掉的标记。剥完标签之后**判在行首**（每块渲染结果占一行）：
#: 这样既抓得住掉进段落的那一行，又不会把正文里合法的符号算进来——
#: 手册里合法地出现过 `*`（`scenarios/*.yaml`）和 `<`/`>`（`agents/<名字>/`，
#: 写的时候在代码段里，剥标签之后仍是它自己）。今天这些符号一个都不在行首。
_LEAKS: dict[str, re.Pattern[str]] = {
    "加粗星号": re.compile(r"\*{2}"),
    "围栏": re.compile(r"```"),
    "标题井号": re.compile(r"(?m)^\s*#{1,6}\s"),
    "表格竖线": re.compile(r"(?m)^\s*\|"),
    "引用箭头": re.compile(r"(?m)^\s*>\s"),
    "横线": re.compile(r"(?m)^\s*-{3,}\s*$"),
}


def test_the_real_manual_leaks_no_markers() -> None:
    """真手册渲染一遍，六种标记一个都不许漏出来。"""
    visible = _visible(manual.render(manual.manual_text()))
    leaked = {name: len(pattern.findall(visible)) for name, pattern in _LEAKS.items()}
    assert not any(leaked.values()), (
        "渲染结果里漏出了标记（读者会原样看见它们）："
        f"{ {k: v for k, v in leaked.items() if v} }"
    )


def test_the_leak_gate_can_fail() -> None:
    """闸门自己得有牙：喂两段它**认不出**的写法，必须真的报出来。

    没有这一条，上面那条断言在"渲染器其实把什么都吐出去了"的情况下**照样绿**
    ——只要 `_visible` 挖错了东西。一条测不出退化的闸门看起来和好闸门一模一样。

    第一版这里是拿 `## 标题` 和一张两行的表当坏输入的，两条都没报——因为
    **它们本来就是渲染器认得的写法**：`##` 是标题、`|…|` 是表格。那种"坏输入"
    测的是"闸门会不会误报"，不是"闸门有没有牙"。真正越界的是下面这两种：
    配不上对的星号（`_bold` 刻意不猜），和不收尾的表格行（`_TABLE_ROW` 要求
    行尾也有竖线，缺了它这一行就掉进段落）。
    """
    for name, stranger in {
        "加粗星号": "这句少了一个星号**后面没有配对的\n",
        "表格竖线": "| 甲 | 乙\n",
    }.items():
        assert _LEAKS[name].search(_visible(manual.render(stranger))), (
            f"{name}：伪造的坏输入没被抓住，说明上面那条闸门是空的"
        )


def test_no_line_of_the_manual_disappears() -> None:
    """**每一行都要落到页面上**。

    比漏标记更隐蔽的一档：整行静默消失。页面上看不出异常——只是少了一段话，
    而写着它的那个人以为用户读到了。所以逐行核对：把原文那行的标记去掉、空白
    抹平，必须能在渲染结果（同样抹平空白、剥掉标签、还原实体）里找到。
    """
    source = manual.manual_text()
    rendered = str(manual.render(source))
    flat = re.sub(r"\s+", "", html_module.unescape(re.sub(r"<[^>]+>", "", rendered)))

    missing: list[str] = []
    for number, line in enumerate(source.splitlines(), 1):
        # 只跳过围栏那三行本身。**围栏里的正文要一起核**：那段 YAML 是手册里
        # 唯一给用户照抄的东西，它掉了一行比别处掉一行更容易出事。
        if line.strip().startswith("```"):
            continue
        plain = _bare(line)
        # 纯符号行（表格分隔、括号）没有可找的字，跳过。
        if not plain or not re.search(r"[\w一-鿿]", plain):
            continue
        if plain not in flat:
            missing.append(f"{number}: {line.strip()[:60]}")
    assert not missing, "这些行没进渲染结果：\n  " + "\n  ".join(missing)


def _bare(line: str) -> str:
    """原文一行去掉各种块标记与行内标记，只留字。"""
    out = line.strip()
    for pattern in (r"^#{1,6}\s+", r"^[-*]\s+", r"^\d+[.、]\s+", r"^>\s?"):
        out = re.sub(pattern, "", out)
    out = out.strip("|").replace("|", "")
    return re.sub(r"\s+", "", out.replace("**", "").replace("`", ""))


def test_the_line_survival_gate_can_fail() -> None:
    """同上：这一条也得有牙。渲染器把整段吞掉时，逐行核对必须报出来。"""
    assert _bare("**第一件**") not in re.sub(r"\s+", "", str(manual.render("另一段话\n")))


# --- 渲染器：逐个写法 -------------------------------------------------------


def test_headings_shift_down_one_level() -> None:
    """手册的 `#` 落到 h2。

    页面上的一级标题是模板写的「使用手册」。手册自己再来一个 h1，一页里就有
    两个——对读屏软件和浏览器的文档大纲都是添乱，而屏幕上看着完全正常。
    """
    body, outline = manual.page("# 大标题\n\n## 一节\n\n### 一小节\n")
    assert "<h2>大标题</h2>" in body
    assert '<h3 id="s0">一节</h3>' in body
    assert "<h4>一小节</h4>" in body
    assert outline == [("s0", "一节")], "只有二级标题进目录"


def test_the_manual_has_exactly_one_top_heading_and_ten_sections() -> None:
    """现算：手册里有几个一级/二级标题，页面上就该有几个 h2、几个锚点。

    **数标题不能拿正则扫原文**：第 11 节那段 YAML 里有一行 `# 不写 = 不检查`，
    `(?m)^#\\s` 会把它数进去——那样这条断言的一半在数一个**注释**，而它红了
    之后要花很久才看得明白。所以数的是 `_blocks` 认出来的标题，跟渲染器同一套读法。
    （第一版就是扫的原文，实测当场红。）
    """
    source = manual.manual_text()
    body, outline = manual.page(source)
    levels = [
        payload[0]
        for kind, payload in manual._blocks(source.splitlines())
        if kind == "heading"
    ]
    assert levels.count(1) == 1, f"一级标题有 {levels.count(1)} 个"
    assert levels.count(2) == 10, f"二级标题有 {levels.count(2)} 个"
    assert str(body).count("<h2>") == 1
    assert len(outline) == levels.count(2)


def test_a_fence_swallows_heading_markers() -> None:
    """围栏里的 `#` 是**正文**，不是标题。

    手册第 11 节那段 YAML 里就有一行 `# 不写 = 不检查`。没被围栏吃掉的话，
    它会变成页面上的一个真标题，而那段注释是讲"这一项省略意味着什么"的——
    升成标题之后，它看起来像手册在主张一件它没主张的事。
    """
    body, outline = manual.page("```yaml\ncharges: 2  # 不写 = 不检查\n```\n")
    # 断言"这一行在 `<pre>` 里"，而不是"页面上没有 `<h`"——**第一版就是后者，
    # 它在围栏彻底坏掉时照样绿**：围栏行落进 `_blocks` 最后那个兜底分支被丢掉，
    # 正文变成段落，于是既没有 `<h`、也还能找到那行字。测出来的是"这一段没崩"，
    # 不是"围栏生效了"。
    assert "<pre><code" in body, f"围栏没被认出来：{body}"
    assert "不写 = 不检查" in body.split("<pre><code", 1)[1].split("</code></pre>", 1)[0], (
        f"那行注释没进代码块：{body}"
    )
    assert "<h" not in body, f"围栏里的井号被当成标题了：{body}"
    assert outline == []


def test_a_table_drops_its_separator_row_and_keeps_the_cells() -> None:
    # 结构看**原样**的渲染结果（`_visible` 会把标签一起剥掉），
    # "分隔行有没有漏出来"才看剥完的文本——它剥完正好就是 `---`。
    raw = str(manual.render("| 口 | 答 |\n|---|---|\n| 甲 | 乙 |\n"))
    assert raw.count("<th>") == 2
    assert raw.count("<td>") == 2
    assert "<td>甲</td>" in raw and "<td>乙</td>" in raw
    assert "---" not in _visible(raw), "分隔行没被丢掉"


def test_a_fence_keeps_its_language_and_escapes_its_body() -> None:
    body = str(manual.render("```python\nx = '<b>' & 1\n```\n"))
    assert '<code class="lang-python">' in body
    assert "&lt;b&gt;" in body and "&amp;" in body
    assert "<b>" not in body


def test_a_quote_and_a_rule_and_lists() -> None:
    body = str(
        manual.render(
            "> 第一行\n> 第二行\n\n---\n\n- 甲\n- 乙\n  续行贴上来\n\n1. 一\n2. 二\n"
        )
    )
    assert "<blockquote>第一行第二行</blockquote>" in body, "引用的两行要接成一段"
    assert "<hr>" in body
    assert "<li>甲</li>" in body and "<li>乙续行贴上来</li>" in body
    assert "<ol>" in body and "<li>一</li>" in body


def test_bold_that_straddles_a_code_span() -> None:
    """**这一条是回归**：第一版渲染器按代码段切开来配对，于是这种句子的两半
    各自落在一个段里，星号配不上——真手册里当场漏出 8 个。"""
    body = str(manual.render("**第一件：把它放进 `agents/me/`。**后面还有话。\n"))
    assert body.count("<strong>") == 1 and body.count("</strong>") == 1
    assert "<strong>第一件：把它放进 <code>agents/me/</code>。</strong>" in body
    assert "**" not in body, f"星号没配对：{body}"


def test_an_unpaired_bold_marker_is_left_alone() -> None:
    """**配不上对的星号原样留着，绝不猜**。

    猜"大概是想加粗到行尾"会把一句本来是疑问的话印成断言。留着的话页面上会多出
    一串星号，而上面那条闸门当场变红——它要说的是"手册里少写了一个 `**`"。
    """
    body = str(manual.render("这句少了一个星号**后面没有配对的\n"))
    assert "**" in body
    assert "<strong>" not in body


def test_join_puts_a_space_between_two_ascii_ends() -> None:
    """手册里现在**没有**这种跨行断开的英文，所以这条是专门造的输入。

    "反正现在不会发生"是最容易在下一版变假的那种理由：底层的折行一改，
    `is` 和 `ready` 就会粘成 `isready`——那是**另一个字**，而两行分开读都对。
    """
    assert manual._join(["the tool is", "ready to run"]) == "the tool is ready to run"


def test_join_does_not_put_a_space_at_a_chinese_seam() -> None:
    """中文接缝处补空格会在整篇里多出一堆本来不存在的空格。"""
    assert manual._join(["它会不会", "重复扣款"]) == "它会不会重复扣款"
    assert manual._join(["看 `runs`", "这一页"]) == "看 `runs`这一页"


@pytest.mark.parametrize(
    "hostile",
    [
        "<script>alert(1)</script>",
        "<img src=x onerror=alert(1)>",
        "a & b < c > d \" e ' f",
    ],
)
def test_hostile_text_comes_out_escaped(hostile: str) -> None:
    """渲染器是**唯一**一处先转义再套标记的地方，所以它自己在几个位置上都要转：
    段落、标题、表格单元格、列表项、引用。少一处就是一个 XSS。

    （模板那边不写 `|safe`，所以这里的 `Markup` 是**唯一**被信任的字符串。）
    """
    for where, text in {
        "段落": f"{hostile}\n",
        "标题": f"# {hostile}\n",
        "表格": f"| 头 |\n|---|\n| {hostile} |\n",
        "列表": f"- {hostile}\n",
        "引用": f"> {hostile}\n",
    }.items():
        body = str(manual.render(text))
        assert hostile not in body, f"{where}没转义：{body}"
        assert "&lt;" in body or "&amp;" in body, f"{where}看起来根本没渲染"


# --- 手册原文的结构 ---------------------------------------------------------


def test_list_continuations_are_indented() -> None:
    """列表项的续行必须缩进。

    渲染器不认"惰性续行"（列 0 的续行），而 GitHub 认。也就是说：这样写的话，
    同一份 `.md` 在仓库里看着是对的、在这一页上是断成两段的。**两种读法不一致**
    比"不一样好看"严重，所以宁可当场变红，让写的人缩进两格。
    """
    lines = manual.manual_text().splitlines()
    offenders: list[str] = []
    after_item = False
    inside_fence = False
    for number, line in enumerate(lines, 1):
        if line.strip().startswith("```"):
            inside_fence = not inside_fence
            after_item = False
            continue
        if inside_fence or not line.strip():
            after_item = False
            continue
        if manual._starts_block(line) and not (
            manual._UL.match(line) or manual._OL.match(line)
        ):
            after_item = False
            continue
        is_item = bool(manual._UL.match(line) or manual._OL.match(line))
        # 列表项自己不算续行——否则每两条相邻的 `2. …` `3. …` 都会被报成一条。
        # 只有"上一行是列表项、这一行既不是新项也不缩进"的那种才算。
        if (
            after_item
            and not is_item
            and not manual._CONTINUATION.match(line)
            and not line.startswith(" ")
        ):
            offenders.append(f"{number}: {line.strip()[:60]}")
        after_item = is_item or (after_item and bool(manual._CONTINUATION.match(line)))
    assert not offenders, (
        "这些行紧跟在列表项后面、又没缩进——在这页上会断成另一段：\n  "
        + "\n  ".join(offenders)
    )


# --- 页面 -------------------------------------------------------------------


@pytest_asyncio.fixture
async def ui(temp_db: str) -> AsyncIterator[httpx.AsyncClient]:
    """真应用。手册那一页只读文件，走真路由是为了让模板和转义一起被验。"""
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://ui") as client:
        yield client


async def test_the_help_page_shows_the_manual(ui: httpx.AsyncClient) -> None:
    """页面上的正文**来自那个文件**：改一个字，页面上就多一个字。

    比对长度而不是逐字比，是因为中间隔着一层渲染（转义、加标签）。所以还要
    钉住"渲染确实发生了"——否则渲染器坏成"什么都不输出"时，这条也绿。
    """
    source = manual.manual_text()
    resp = await ui.get("/help")
    assert resp.status_code == 200
    assert len(resp.text) > len(source), "渲染后的页面比原文还短，不像是渲染过"
    assert "<article class=\"doc\">" in resp.text
    assert resp.text.count("**") == 0


async def test_every_toc_link_points_at_a_real_anchor(ui: httpx.AsyncClient) -> None:
    """目录里的每个链接都要有落点。

    目录和正文是同一次遍历产出的（`manual.page`），所以这条**应当**永远成立；
    它守的是以后有人把两边拆成两次遍历——那时链接指向一个不存在的锚点，
    点一下没反应，用户只会以为自己点歪了。
    """
    body, outline = manual.page(manual.manual_text())
    page = (await ui.get("/help")).text
    anchors = re.findall(r'<li><a href="#([^"]+)">', page)
    assert len(anchors) == len(outline) == 10
    for anchor in anchors:
        assert f'id="{anchor}"' in page, f"目录里的 {anchor} 没有落点"


async def test_the_page_has_exactly_one_h1(ui: httpx.AsyncClient) -> None:
    page = (await ui.get("/help")).text
    assert len(re.findall(r"<h1[ >]", page)) == 1


async def test_the_nav_and_the_manual_section_two_agree() -> None:
    """**手册第 2 节描述的顶栏，必须就是真的顶栏。**

    这条是现算的：从 `base.html` 里取顶栏链接，从手册原文里取第 2 节的表格。
    原先它对不上——标题写着「九个入口」，表里只有八行，而**没有任何东西报错**。
    加 `/help` 这一页时表补上了第九行，从此这条闸门盯着两边。
    """
    nav = _nav_entries(_render_base_template())
    rows = _section_two_rows(manual.manual_text())

    assert [text for text, _href in nav] == [label for label, _when in rows], (
        "顶栏和手册第 2 节的入口对不上（顺序也在这条断言里）"
    )
    # 标题里那个中文数字也得对：它是**最容易被漏改的一处**——表加了一行，
    # 「九个」没人动，于是手册里的那句话又变成假的。
    assert _chinese_count(manual.manual_text()) == len(nav)


def _render_base_template() -> str:
    """渲染 base.html 再解析——导航是 Jinja 循环生成的，源文件里只有 `{{ label }}`。"""
    env = Environment(loader=FileSystemLoader(str(ROOT / "src" / "sandbox" / "api" / "templates")))
    return env.get_template("base.html").render(
        request=SimpleNamespace(url=SimpleNamespace(path="/")), active=None,
    )


#: 侧边导航栏的入口：`class="nav-item"` 的链接，跳过图标 span，取末尾文字。
_NAV_ITEM = re.compile(
    r'<a class="nav-item[^"]*"\s+href="([^"]*)">.*?<span class="nav-ico">.*?</span>(?P<label>[^<]+)</a>',
    re.S,
)


def _nav_entries(source: str) -> list[tuple[str, str]]:
    """导航栏里那些入口：`(显示名, 链接)`。

    **品牌名不算入口**——它回首页，而且它的 class 是 `brand`、不是 `nav-item`，
    所以下面的正则天然不会把它当成一个入口：排除靠类名，不靠位置（假警报一多，
    真警报就没人看了）。入口按 `nav-group` 的渲染顺序排列，手册第 2 节跟着它走。
    """
    out: list[tuple[str, str]] = []
    for match in _NAV_ITEM.finditer(source):
        href = match.group(1)
        label = match.group("label").strip()
        assert href, f"导航栏这个入口没有 href：{label}"
        out.append((label, href))
    return out


def _section_two_rows(source: str) -> list[tuple[str, str]]:
    """手册第 2 节那张表的每一行：`(入口名, 什么时候点它)`。

    入口名在原文里是 `**加粗**`——去掉星号就是顶栏上那个词。
    """
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("## 2. "))
    rows: list[tuple[str, str]] = []
    for line in lines[start + 1 :]:
        if line.startswith("---") or line.startswith("## "):
            break
        if not line.startswith("|") or _is_separator(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == "入口":  # 表头
            continue
        rows.append((cells[0].strip("*"), cells[2]))
    return rows


def _is_separator(line: str) -> bool:
    return not set(line) - set("|-: \t")


#: 标题里的中文数字。只认一到十——手册的顶栏不会比这更长，多一档就该有人
#: 亲自去看一眼，而不是让闸门猜。
_CHINESE = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _chinese_count(source: str) -> int:
    heading = next(line for line in source.splitlines() if line.startswith("## 2. "))
    words = [c for c in heading if c in _CHINESE]
    assert len(words) == 1, f"第 2 节的标题里找不到那个数：{heading}"
    return _CHINESE[words[0]]


async def test_a_missing_manual_file_gives_a_readable_page(
    ui: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """手册文件不在时给一页能读懂的话，不是 500 的空白页。

    这一页恰恰是用户在别处撞了墙才会点开的——它自己塌了，用户就没有下一步。
    """
    monkeypatch.setattr(manual, "MANUAL_PATH", tmp_path / "没有这个文件.md")
    resp = await ui.get("/help")
    assert resp.status_code == 500
    assert "手册文件读不到" in resp.text
    assert str(tmp_path / "没有这个文件.md") in resp.text


async def test_the_page_uses_no_safe_filter_and_no_script(ui: httpx.AsyncClient) -> None:
    """模板里不许出现 `|safe`、不许出现脚本。

    渲染器交出来的是 `Markup`，模板因此不需要 `|safe`——而 `|safe` 一旦出现在
    这一页上，就等于把"手册正文"整段标成可信，将来手册里引用一段用户输入
    （比如某个 Agent 的名字）时就地变成 XSS。
    """
    template = (ROOT / "src" / "sandbox" / "api" / "templates" / "help.html").read_text(
        encoding="utf-8"
    )
    assert "|safe" not in template
    assert "<script" not in template.lower()
    page = (await ui.get("/help")).text
    assert "<script" not in page.lower()
