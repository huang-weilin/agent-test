"""给人看的数字格式。

成本与 token 要出现在三个地方：场景报告（`scenario/runner.py`）、跨运行报告
（`scripts/cost_report.py`）、Web 界面（`api/`）。三处各写一份的后果不是难看，
是**三份会慢慢不一致**——同一个 $0.0001，一处显示 `0.0001`、另一处显示 `0.00`，
而用户分不清哪个是"真的花了这么多"、哪个是"显示精度不够"。在一个专门用来
抓"看起来正常但其实不对"的工具里，这种歧义不能留。
"""

from __future__ import annotations


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
