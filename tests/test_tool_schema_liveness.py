"""白名单里的每个 schema 关键字，要么有活样本，要么有一行写下的理由。

`toolschema.SUPPORTED_KEYWORDS` 的立身之本是"支持的写法恰好等于我们能校验的写法"。
但"能校验"其实有两层：

  1. **校验器认得它**——`tests/test_tool_schema.py` 直接调 `validate_arguments`
     证这一层；
  2. **场景里写下的那一条真的被边界核对过一次**——这一层只有跑起来才证得了。

第二层塌了不会有任何一条测试变红：报告上一切正常，只是这个关键字从来没被真正
用过。判据和 `tests/test_reason_codes.py` 一样——**去场景文件里找它**（用产品
自己的加载器读，不是拿正则扫文本），找不到就必须在下面那张表里有一行，并写清楚
为什么。

反转过来也查：表里的每一行**必须仍然是缺的**。哪天有人拿一个带数组的工具补上了
活样本，那一行就得删掉——否则这张表会变成一句关于过去的实话。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sandbox.scenario.loader import load_all
from sandbox.toolschema import SUPPORTED_KEYWORDS

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = REPO_ROOT / "scenarios"

#: 白名单里**没有活样本**的关键字，以及为什么。加一行之前先问：真的找不到一个
#: 场景来用它吗？找不到的话，理由要写到别人读了不会再问一遍的程度。
NO_LIVE_SAMPLE: dict[str, str] = {
    "items": (
        "演示 Agent 手里没有收数组的工具：工具表是按它的 Python 签名过线的，"
        "而四个工具的签名里没有一个是列表。校验器认得它（单测里 items 的逐条"
        "递归核对是有样本的），缺的只是「场景里写一条、边界上拦一次」。要补上，"
        "得先给被测 Agent 加一个收数组的工具——那是改被测对象，不该为了覆盖一个"
        "关键字去做。"
    ),
}


def _keywords_in(node: Any) -> set[str]:
    """递归收集一份 JSON Schema 里用到的关键字。"""
    found: set[str] = set()
    if isinstance(node, dict):
        found |= {k for k in node if k in SUPPORTED_KEYWORDS}
        for value in node.values():
            found |= _keywords_in(value)
    elif isinstance(node, list):
        for item in node:
            found |= _keywords_in(item)
    return found


def _keywords_used_by_scenarios() -> set[str]:
    """把 `scenarios/` 下所有场景声明过的 schema 关键字并起来。

    走的是产品自己的加载器：这些声明就是边界上真的拿去核对的那一份
    （`runner.py` 的 `registration_payload` 逐字过线）。自己解析 YAML 的话，
    这条测试证明的是"文本里有这个词"，不是"边界上真的核对得到"。
    """
    used: set[str] = set()
    for scenario in load_all(SCENARIO_DIR):
        for tool in scenario.tools:
            used |= _keywords_in(tool.parameters)
    return used


def test_every_supported_keyword_has_a_live_sample_or_a_written_reason() -> None:
    missing = sorted(SUPPORTED_KEYWORDS - _keywords_used_by_scenarios() - set(NO_LIVE_SAMPLE))
    assert not missing, (
        f"这些关键字在场景里一个都没被用过：{missing}。\n"
        f"要么写一个用到它的场景（那才是「边界上真的核对过」），要么在 "
        f"NO_LIVE_SAMPLE 里加一行并写清楚为什么。"
    )


def test_a_row_in_the_ledger_is_not_stale() -> None:
    used = _keywords_used_by_scenarios()
    stale = sorted(k for k in NO_LIVE_SAMPLE if k in used)
    assert not stale, (
        f"这些关键字已经有场景在用了：{stale}——把 NO_LIVE_SAMPLE 里那一行删掉。"
        f"留着的后果是这张表慢慢变成一句关于过去的实话。"
    )


def test_the_ledger_only_names_keywords_that_exist() -> None:
    unknown = sorted(set(NO_LIVE_SAMPLE) - SUPPORTED_KEYWORDS)
    assert not unknown, f"NO_LIVE_SAMPLE 里写了不在白名单里的关键字：{unknown}"
