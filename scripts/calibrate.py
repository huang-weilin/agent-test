#!/usr/bin/env python
"""用标注语料校准检测器：算出每一族的精确率与召回率，并把分歧列出来。

    .venv/Scripts/python.exe scripts/calibrate.py
    .venv/Scripts/python.exe scripts/calibrate.py --corpus tests/corpus --verbose
    .venv/Scripts/python.exe scripts/calibrate.py --json

**这个脚本回答的是"检测器判得准不准"，不是"这次运行有没有问题"。** 后者看报告
页面（`scripts/serve_ui.py`）。这里跑的是 `tests/corpus/` 里手写并标注好的轨迹，
标注是人写的独立判断——所以这里算出来的数字才有意义。

没有语料的话，`docs/detectors.md` 第七节那些精度目标就只是占位：既无法验收，
也无法调优。这个脚本就是让它们可以被算出来的地方。

退出码：全部命中标注意为 0，任何一条不符为 1（可以直接进 CI）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sandbox.corpus import FAMILIES, ItemResult, evaluate_all, load_corpus, score  # noqa: E402

DEFAULT_CORPUS = Path(__file__).resolve().parents[1] / "tests" / "corpus"


def _finding_line(finding) -> str:
    code = finding.detail.get("reason_code")
    suffix = f" [{code}]" if code else ""
    return f"{finding.detector}{suffix} {finding.verdict.value}"


def _print_item(result: ItemResult, *, verbose: bool) -> None:
    item = result.item
    if result.ok and not verbose:
        return
    mark = "ok  " if result.ok else "不符"
    print(f"[{mark}] {item.id}  （{item.source}）")
    if result.missing:
        print("        没有报出标注里该报的：")
        for expectation in result.missing:
            code = f" [{expectation.reason_code}]" if expectation.reason_code else ""
            print(f"          - {expectation.detector}{code} {expectation.verdict.value}")
    if result.forbidden:
        print("        报了标注里说这一族不该报的：")
        for finding in result.forbidden:
            print(f"          - {_finding_line(finding)}：{finding.summary}")
    if result.over_reported:
        print("        这一族该报，但这一条标注里没有它（多报要么改 `count`，要么改标注）：")
        for finding in result.over_reported:
            print(f"          - {_finding_line(finding)}：{finding.summary}")
    if result.violated_absent:
        print("        报了 `absent` 里明令不许出现的：")
        for finding in result.violated_absent:
            print(f"          - {_finding_line(finding)}：{finding.summary}")
    if verbose:
        print("        实际结论：")
        for finding in result.findings:
            print(f"          {_finding_line(finding):<44} {finding.summary}")
    print(f"        标注依据：{item.why}")


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _print_scores(results: list[ItemResult]) -> None:
    scores = score(results)
    print("\n各族精确率/召回率（TP = 该报且报了）")
    print(f"  {'族':<14}{'TP':>4}{'FP':>4}{'FN':>4}{'TN':>4}  {'精确率':>8}{'召回率':>9}")
    for family in FAMILIES:
        s = scores[family]
        print(
            f"  {family:<14}{s.tp:>4}{s.fp:>4}{s.fn:>4}{s.tn:>4}  "
            f"{_pct(s.precision):>8}{_pct(s.recall):>9}"
        )


def _print_caveats(results: list[ItemResult]) -> None:
    """**数字必须带着它的边界一起读。** 一段 100% 印在报告里，长得很像"检测器
    没有缺陷"；实际它说的是"这批样本上没找到分歧"，而样本是人手挑的。"""
    sources = Counter(r.item.source for r in results)
    positives = {f: sum(1 for r in results if r.item.family_labels.get(f)) for f in FAMILIES}
    negatives = {f: sum(1 for r in results if not r.item.family_labels.get(f)) for f in FAMILIES}
    print("\n样本构成")
    print(f"  来源：{'、'.join(f'{k} {v}' for k, v in sorted(sources.items()))}")
    for family in FAMILIES:
        print(f"  {family:<14}正样本 {positives[family]:>2} 条，负样本 {negatives[family]:>2} 条")

    print("\n这些数字说的是")
    total = len(results)
    print(f"  · 语料共 {total} 条，全部手写并逐条标注。")
    print("  · 精确率/召回率是「关于这份语料」的陈述，不是关于世界的。")
    print("  · 标着公开来源的那几条，复现的是威胁类型，不是论文/案例库里的原始轨迹。")
    print("  · 真实脱敏轨迹（需签授权）尚未收进语料。")
    print(f"  · 一条误报就能让 n={total} 上的精确率掉几个百分点——结论要更硬，得先把语料做大。")
    print("  · 条数也是标注的一部分：报得比 `count`（不写 = 1 条）多的进 `over_reported`。")


def main() -> int:
    parser = argparse.ArgumentParser(description="用标注语料校准检测器精度")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="语料目录")
    parser.add_argument("--verbose", action="store_true", help="连命中的条目也逐条列出结论")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出（给别的工具吃）")
    args = parser.parse_args()

    items = load_corpus(args.corpus)
    results = evaluate_all(items)
    scores = score(results)
    failed = [r for r in results if not r.ok]

    if args.json:
        payload = {
            "total": len(results),
            "failed": [r.item.id for r in failed],
            "families": {
                f: {
                    "tp": s.tp,
                    "fp": s.fp,
                    "fn": s.fn,
                    "tn": s.tn,
                    "precision": s.precision,
                    "recall": s.recall,
                }
                for f, s in scores.items()
            },
            "items": [
                {
                    "id": r.item.id,
                    "source": r.item.source,
                    "ok": r.ok,
                    "missing": [e.model_dump(mode="json") for e in r.missing],
                    "forbidden": [_finding_line(f) for f in r.forbidden],
                    "over_reported": [_finding_line(f) for f in r.over_reported],
                    "violated_absent": [_finding_line(f) for f in r.violated_absent],
                }
                for r in results
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1 if failed else 0

    print(f"语料目录：{args.corpus}")
    print(f"共 {len(results)} 条\n")
    for result in results:
        _print_item(result, verbose=args.verbose)
    _print_scores(results)
    _print_caveats(results)

    if failed:
        print(f"\n{len(failed)} 条与标注不符：{'、'.join(r.item.id for r in failed)}")
        return 1
    print("\n全部命中标注。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
