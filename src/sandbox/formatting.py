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
from collections.abc import Mapping
from typing import Any

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
