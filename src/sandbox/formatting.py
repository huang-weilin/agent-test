"""给人看的数字与参数格式。

成本、token、时长要出现在三个地方：场景报告（`scenario/runner.py`）、跨运行报告
（`scripts/cost_report.py`）、Web 界面（`api/`）。三处各写一份的后果不是难看，
是**三份会慢慢不一致**——同一个 $0.0001，一处显示 `0.0001`、另一处显示 `0.00`，
而用户分不清哪个是"真的花了这么多"、哪个是"显示精度不够"。在一个专门用来
抓"看起来正常但其实不对"的工具里，这种歧义不能留。

调用参数的紧凑写法（`args_text` / `value_text`）同理：跨会话差异报告与调用链
两处都要印参数，两处各截各的会让同一份报告里同一次调用看起来是两个样子。
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # 只为标注。运行期不导入 cost——它要连数据库，这儿只用它的属性。
    from sandbox.cost.aggregator import ToolCostReport

#: 单个参数值印出来的上限。超过就截断并带上省略号——**截断必须自己说出来**，
#: 否则"参数里少了个字段"和"参数太长没印全"在报告上是同一个样子。
MAX_VALUE_CHARS = 40


def money(usd: float) -> str:
    """按量级选精度，且不留无意义的零。

    固定两位小数会把 $0.00003 显示成 `0.00`——和"一分钱没花"长得一模一样，
    而演示用的预算恰好就是这么大的量级（`$0.0001`）。固定六位又会让 $12.5
    显示成 `12.500000`，读起来像有六位有效精度。
    """
    decimals = 6
    for candidate in (2, 4, 6):
        decimals = candidate
        if abs(usd) >= 10**-candidate:
            break
    # rstrip 只削掉**末尾**的零：`"10.00"` → `"10."` → `"10"`，
    # 不会把 `1000.00` 削成 `1`。
    return f"{usd:.{decimals}f}".rstrip("0").rstrip(".") or "0"


def tokens(n: int) -> str:
    """千分位。`1234567` 和 `1,234,567` 读起来不是一回事。"""
    return f"{n:,}"


def byte_size(n: int) -> str:
    """字节数，按量级换单位。

    和 `span_human` 同一个道理：量级不能印丢。这里的数用在"哪坨内容在吃上下文"
    上，差一位就是十倍。**负数照印**——工具调用之后上下文变小是真实发生过的事
    （Agent 自己裁过），把它显示成 0 会把"它裁了"这件事抹掉。
    """
    sign = "-" if n < 0 else ""
    size = abs(n)
    if size < 1000:
        return f"{sign}{size} B"
    if size < 1_000_000:
        return f"{sign}{size / 1000:.1f} KB"
    return f"{sign}{size / 1_000_000:.1f} MB"


#: 按工具那一段的标题。终端上两个出口（场景报告、跨运行成本报告）共用同一句：
#: 标题换一种说法，读的人会以为两处报的不是同一件事。
TOOL_COST_TITLE = "按工具（上下文占用，不是「这个工具花了多少钱」）"


def tool_cost_lines(report: ToolCostReport) -> list[str]:
    """按工具那一段的正文行（不含标题）。**终端上两个出口共用这一份。**

    为什么这十几行要单独抽出来：同一条「租金是条件数字、不是账单」的话，写在
    场景报告里一份、写在跨运行成本报告里又一份，两边迟早会有一边忘了跟着改。
    而这个项目的读者只有一个办法发现那种漂移——把两份报告并排看，而他不会。

    三个数各自带着名字，因为它们是三种口径：

    * 「返回体」是边界量到的那份字节（`ToolCall.result_bytes` 的和），数出来的；
    * 「实测增量」是这次调用之后第一次请求比之前那次大了多少——**观测**，
      量不到就写「量不到」，不写 0；
    * 「租金」是**条件数字**：返回体在此后每次请求里原样重发的话，它一共占多少
      字节。它只按返回体本身的字节数算，而实测的每轮增量比返回体还大（JSON
      转义与消息外壳），所以租金是下界——这句话必须印出来，否则读的人会拿它当
      账单。

    **总和印在说明句里，不印成一列。** 各工具的租金加起来是说得通的：在"原样
    重发"这个假设下，每一次请求里属于工具返回体的那部分字节**只会被算到产生它的
    那个工具头上**，所以合计是"输入账单里可归到工具返回体的那一部分"的下界，而且
    它不会超过整段范围的输入字节总量（量于 2026-10-01 的那次
    `loop_paging_bounded`：Σ请求体 44,720 B，租金 24,081 B，约一半）。它不该占
    一列，是因为它不是某一行的事——放在任意一行旁边都会被读成"这一行的"。
    """
    rows = sorted(report.tools.values(), key=lambda t: (-t.rent_bytes, t.tool_name))
    # 宽度按这一节里最长的名字算：写死宽度的话，一个长名字就会把整列顶歪，
    # 而歪掉的表格比没有表格更难读。名字按**列宽**算：工具名是被测对象自己
    # 声明的，可以是中文。
    width = max(display_width(t.tool_name) for t in rows)
    lines: list[str] = []
    for tool in rows:
        # **一次都没量到的时候印「量不到」，不印 0。** 0 B 是一句更强的话——它说的是
        # "这个工具返回了空体"，而作者知道的是"这一列存在之前跑的老行没有体量"。
        # 这两种情况在报告上都出现（老库的行、被挡下的调用），而它们的区别恰好是
        # 读的人要的那个：前者说明历史数据不全，后者说明策略在生效。
        measured = bool(tool.measured_size_calls)
        size_text = byte_size(tool.total_result_bytes) if measured else "量不到"
        rent = byte_size(tool.rent_bytes) if measured else "量不到"
        if measured:
            rent_tokens = report.tokens_for(tool.rent_bytes)
            if rent_tokens is not None:
                rent += f"（≈{tokens(rent_tokens)} tok）"
        growth = (
            byte_size(tool.context_growth_bytes) if tool.growth_measured_calls else "量不到"
        )
        lines.append(
            f"    {pad_left(tool.tool_name, width)}  {tool.calls:>3} 次  "
            f"返回体 {pad_left(size_text, 9)}  "
            f"实测增量 {pad_left(growth, 9)}  "
            f"租金 {rent}"
        )

    lines.append("    「实测增量」是这次调用之后第一次请求比之前那次大了多少字节——观测。")
    if report.bytes_per_token is None:
        lines.append(
            "    「租金」只报了字节：没有一条调用记下请求体字节数"
            "（在这一列存在之前跑的），折算 token 的密度无从量起。"
        )
    else:
        total = report.tokens_for(report.total_rent_bytes)
        total_text = byte_size(report.total_rent_bytes)
        if total is not None:
            total_text += f"（≈{tokens(total)} tok）"
        # 只有一个工具时不写「合计」：一个东西的合计就是它自己，那两个字只会
        # 让读的人去找另一个数。
        head = "各工具合计" if len(rows) > 1 else "这一次的租金"
        lines.append(
            f"    「租金」是条件数字，不是观测：假设某个工具的返回体在此后每次请求里"
            f"原样重发，它一共占多少字节。折算用这段范围实测的密度 "
            f"{report.bytes_per_token:.2f} 字节/token（Σ请求体 ÷ Σ输入 token，只算"
            f"量到的那 {report.llm_calls_measured}/{report.llm_calls} 次调用）；"
            f"{head} {total_text}——在同一个假设下，这就是输入账单里可归到工具返回体"
            f"的那一部分。"
        )
    lines.append(
        "    实测的每轮增量比返回体本身大（JSON 转义与消息外壳都要算进去），"
        "所以租金是下界，不是账单。"
    )
    unmeasured = sum(t.unmeasured_size_calls for t in report.tools.values())
    if unmeasured:
        lines.append(
            f"    另有 {unmeasured} 次调用没量到返回体字节数（在这一列存在之前跑的老行，"
            f"或者根本没出去的调用）。它们不计进「返回体」与「租金」——那两列是"
            f"量到的那部分，不是全部。"
        )
    without = sum(t.calls_without_followup for t in report.tools.values())
    if without:
        lines.append(
            f"    其中 {without} 次调用之后同会话里再没有请求，租金记 0 并单独计数："
            f"0 是「看不见」，不是「免费」。"
        )
    return lines


def display_width(text: str) -> int:
    """一串字在终端里占几格。

    Python 的 `len()` 数的是**字符**，终端数的是**列**：一个汉字占两列。所有
    `f"{x:>9}"` 对中文都是错的——「量不到」3 个字符会被补到 9 个字符，屏幕上
    却是 6 + 6 = 12 列，于是它右边那几列整列歪掉。而这一节里**恰好**有一个
    中文的占位符（量不到），所以这不是理论问题。

    只把 `W`（宽）和 `F`（全角）算两列。`A`（模糊）按一列算——它在中文终端里
    多数占两列，但它出现在这儿的字符串里的机会是零，为此加一张表不值。
    """
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def pad_left(text: str, width: int) -> str:
    """左补空格到 `width` **列**。见 `display_width`。"""
    return " " * max(0, width - display_width(text)) + text


def span_human(seconds: float) -> str:
    """秒数 → 人读的时长。

    **不四舍五入成"0 天"。** 90000 秒印成"1 天"没错，但 3600 秒印成"0 天"就是
    在说"没过去多久"——而它是一小时。长时运行报告里最不能出的错就是这个：
    数字没错，读出来的意思反了。

    所以按量级换单位，而不是固定一个单位硬除。
    """
    if seconds < 60:
        return f"{seconds:.0f} 秒"
    if seconds < 3600:
        return f"{seconds / 60:.0f} 分钟"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} 小时"
    return f"{seconds / 86400:.1f} 天"


def millis(ms: int) -> str:
    """毫秒 → 人读的耗时。

    **不把小于一秒的耗时交给 `span_human`。** 它按秒取整，几十毫秒的一次
    模型往返会被印成「0 秒」——读的人看到的是"没花时间"，而在一张讲延迟的
    表上，那一格恰恰是最要紧的。所以这个量级直接按毫秒印。
    """
    if abs(ms) < 1000:
        return f"{ms} ms"
    return f"{ms / 1000:.1f} s"


def value_text(value: Any) -> str:
    """一个参数值的写法。超长就截断，并带上省略号。"""
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= MAX_VALUE_CHARS:
        return text
    return text[:MAX_VALUE_CHARS] + "…"


def args_text(args: Mapping[str, Any]) -> str:
    """参数的紧凑写法：字段名排序，最多印 4 个，多出来的报个数。

    **排序**：参数在字典里的顺序取决于调用方怎么构造的，直接印会让同一次调用
    在两条时间线上看起来不一样。**少印的要报个数**：只印 4 个而不说还有几个，
    读者会把"没印"当成"没传"。
    """
    keys = sorted(args)
    shown = keys[:4]
    body = "，".join(f"{k}={value_text(args[k])}" for k in shown)
    if len(keys) > len(shown):
        body += f"，还有 {len(keys) - len(shown)} 个字段"
    return body or "（无参数）"
