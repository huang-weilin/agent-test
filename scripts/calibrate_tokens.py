#!/usr/bin/env python
"""拿真上游重量一遍"字符→token"的密度，把结果和估算器的偏差打出来。

    .venv/Scripts/python.exe scripts/calibrate_tokens.py
    .venv/Scripts/python.exe scripts/calibrate_tokens.py --json

**这个脚本是给 `gateway/stream.py` 那个 `_CHARS_PER_TOKEN` 用的相机。**
那个常数带着四行实测数字（中文 1.67、中文 JSON 2.19、英文 JSON 3.04、
纯英文 5.48 字符/token），`tests/test_token_estimator.py` 拿这四行当绊线。
但**绊线钉的是数字，不是数字是否还成立**——模型换了 tokenizer、上游换了
计费口径，钉住的数字会静静地过期。这个脚本负责重新量。

没配 `SANDBOX_UPSTREAM_API_KEY` 时直接跳过（退出码 0，并说明跳过原因）：
它不该在没 key 的机器上把 CI 弄红。四次小请求，成本可忽略。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
# 样本只写一遍：它们同时是 `tests/test_token_estimator.py` 的夹具。
# 两个文件各抄一份的话，脚本"验证"的就是测试没在盯的数字。
sys.path.insert(0, str(REPO_ROOT))

import httpx  # noqa: E402

from sandbox.config import settings  # noqa: E402
from sandbox.gateway.stream import estimate_input_tokens  # noqa: E402
from tests.test_token_estimator import SAMPLES  # noqa: E402

#: 基线的 prompt（真上游实测 9 个 token）。每个样本都扣掉它。
BASELINE_TEXT = "请只回复 ok。"
BASELINE_TOKENS = 9


async def _prompt_tokens(client: httpx.AsyncClient, text: str) -> int:
    resp = await client.post(
        "/chat/completions",
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 1,
        },
    )
    if resp.status_code != 200:
        raise SystemExit(f"上游拒绝：HTTP {resp.status_code} {resp.text[:300]}")
    return int((resp.json().get("usage") or {}).get("prompt_tokens") or 0)


async def _run(as_json: bool) -> int:
    key = settings.upstream_api_key
    if not key:
        print(
            "没有配 SANDBOX_UPSTREAM_API_KEY——跳过。\n"
            "（把 key 写进 .env 再跑；本脚本只读环境，不打印它的任何片段。）"
        )
        return 0

    base = settings.upstream_base_url.rstrip("/")
    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient(
        base_url=base, timeout=60.0, headers={"Authorization": f"Bearer {key}"}
    ) as client:
        baseline = await _prompt_tokens(client, BASELINE_TEXT)
        if baseline != BASELINE_TOKENS:
            print(
                f"注意：基线 prompt 这次是 {baseline} token，"
                f"而绊线里记的是 {BASELINE_TOKENS}——扣减口径可能变了。"
            )
        for name, (text, pinned) in SAMPLES.items():
            measured = await _prompt_tokens(client, BASELINE_TEXT + "\n" + text) - baseline
            estimate = estimate_input_tokens(
                {"messages": [{"role": "user", "content": text}]}
            )
            rows.append(
                {
                    "样本": name,
                    "字符": len(text),
                    "字节": len(text.encode("utf-8")),
                    "实测token": measured,
                    "字符每token": round(len(text) / measured, 2),
                    "估算token": estimate,
                    "估/实": round(estimate / measured, 3),
                    "绊线里钉的": pinned,
                    "还成立": measured == pinned,
                }
            )

    if as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(f"上游：{base}")
        print(
            f"{'样本':<10}{'字符':>7}{'字节':>7}{'实测tok':>9}{'字符/tok':>10}"
            f"{'估算tok':>9}{'估/实':>8}  绊线里的数还成立吗"
        )
        for row in rows:
            mark = "是" if row["还成立"] else f"!! 不是（现在是 {row['实测token']}）"
            print(
                f"{row['样本']:<10}{row['字符']:>7}{row['字节']:>7}"
                f"{row['实测token']:>9}{row['字符每token']:>10}"
                f"{row['估算token']:>9}{row['估/实']:>8}  {mark}"
            )
        print(
            "\n中文那两行的「估/实」小于 1 是已知且写在注释里的（取 3 会低估中文），"
            "\n不是回归；它们掉到 0.5 以下、或者任意一行的数对不上，才要动。"
        )

    stale = [row["样本"] for row in rows if not row["还成立"]]
    if stale:
        print(
            f"\n这几个样本的实测值已经和绊线里钉的不一样了：{'、'.join(stale)}。"
            f"\n改 `tests/test_token_estimator.py` 里的数、以及 `gateway/stream.py`"
            f"\n那段注释里的四行——两处一起改，否则注释里的「实测」就成了历史。"
        )
        return 1
    print("\n四个样本都与绊线一致。")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="重量字符→token 密度")
    parser.add_argument("--json", action="store_true", help="机器可读")
    args = parser.parse_args(argv)
    return asyncio.run(_run(args.json))


if __name__ == "__main__":
    raise SystemExit(main())
