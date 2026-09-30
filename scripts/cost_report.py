#!/usr/bin/env python
"""跨运行成本报告：按天/周/月、按场景、按 Agent、按模型拆解 Token 与花费。

    .venv/Scripts/python.exe scripts/cost_report.py
    .venv/Scripts/python.exe scripts/cost_report.py --period week --limit 12
    .venv/Scripts/python.exe scripts/cost_report.py --json

**这是"长期预算管理"那个面。** 单次运行的成本随场景报告一起出；这个脚本回答
的是另一个问题——"这个月烧掉多少了、烧在哪儿"。需求场景 4 的用户故事是云端
Agent 夜间持续运行、成本是预期的 3 倍，那种失控不会发生在一次运行里。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sandbox.cost import aggregator as cost  # noqa: E402
from sandbox.db.engine import init_db  # noqa: E402

_PERIOD_LABEL = {"day": "本日", "week": "本周", "month": "本月"}


def _money(usd: float) -> str:
    for decimals in (2, 4, 6):
        if abs(usd) >= 10**-decimals:
            break
    return f"{usd:.{decimals}f}".rstrip("0").rstrip(".") or "0"


def _tokens(n: int) -> str:
    return f"{n:,}"


def _rollup_line(label: str, r: cost.CostRollup, width: int) -> str:
    caveat = "  ← 含未定价调用，金额是下界" if r.is_lower_bound else ""
    return (
        f"  {label:<{width}} {_tokens(r.total_tokens):>12} tok  "
        f"{r.llm_calls:>4} 次  ${_money(r.cost_usd):>10}{caveat}"
    )


def _section(
    title: str,
    rows: dict[str, cost.CostRollup] | list[tuple[str, cost.CostRollup]],
) -> None:
    """打一节。列宽按这一节里最长的名字算——写死宽度的话，一个长名字就会把
    整列顶歪，而歪掉的表格比没有表格更难读。"""
    items = list(rows.items()) if isinstance(rows, dict) else rows
    if not items:
        return
    width = max(len(label) for label, _ in items)
    print(f"\n=== {title} ===")
    for label, rollup in items:
        print(_rollup_line(label, rollup, width))


async def _report(period: str, limit: int, as_json: bool) -> int:
    await init_db()

    history = await cost.by_period(period, limit=limit)  # type: ignore[arg-type]
    by_scenario = await cost.by_scenario()
    by_agent = await cost.by_agent()
    by_model = await cost.by_model()
    statuses = await cost.budget_status()

    if as_json:
        print(
            json.dumps(
                {
                    "history": [
                        {"period": label, **rollup.to_dict()} for label, rollup in history
                    ],
                    "by_scenario": {k: v.to_dict() for k, v in by_scenario.items()},
                    "by_agent": {k: v.to_dict() for k, v in by_agent.items()},
                    "by_model": {k: v.to_dict() for k, v in by_model.items()},
                    "budget": [
                        {
                            "period": b.period,
                            "spent_usd": round(b.spent_usd, 6),
                            "ceiling_usd": b.ceiling_usd,
                            "exceeded": b.exceeded,
                        }
                        for b in statuses
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    unit = {"day": "天", "week": "周", "month": "月"}[period]
    if history:
        _section(f"按{unit}（近 {limit} 个）", history)
    else:
        print(f"=== 按{unit}（近 {limit} 个）===\n  （没有数据）")

    _section("按场景", by_scenario)
    _section("按 Agent 版本", by_agent)
    _section("按模型", by_model)

    print("\n=== 预算 ===")
    if not statuses:
        print(
            "  （未设预算。在 .env 里配 SANDBOX_COST_BUDGET_DAILY_USD / _WEEKLY_USD\n"
            "   / _MONTHLY_USD 开启。**默认不设限是刻意的**：一个没配过的阈值\n"
            "   会在别人第一次跑的时候弹出一堆他不认识的告警，而告警一多就会被\n"
            "   整体忽略，包括真正要紧的那些。）"
        )
    for b in statuses:
        mark = "!!" if b.exceeded else "ok"
        print(
            f"  {mark} {_PERIOD_LABEL[b.period]}已花 ${_money(b.spent_usd)}"
            f" / 预算 ${_money(b.ceiling_usd)}（{b.ratio:.0%}）"
        )

    # 未定价的调用单独汇总一次。分散在各行里会看不见，而它的后果是"总账
    # 少算了一部分"——这必须在报告的末尾再说一遍，而不是只标注在明细行上。
    unpriced = sum(r.unpriced_calls for r in by_model.values())
    if unpriced:
        print(
            f"\n**注意：有 {unpriced} 次调用的模型不在定价表中，成本按 0 计。**\n"
            f"   这些数字是**下界**，不是账单。定价表见 src/sandbox/gateway/pricing.py。"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="跨运行成本报告")
    parser.add_argument(
        "--period", choices=["day", "week", "month"], default="day", help="时间粒度"
    )
    parser.add_argument("--limit", type=int, default=30, help="看最近几个周期")
    parser.add_argument("--json", action="store_true", help="机器可读")
    args = parser.parse_args(argv)

    return asyncio.run(_report(args.period, args.limit, args.json))


if __name__ == "__main__":
    raise SystemExit(main())
