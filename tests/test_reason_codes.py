"""每一个 reason_code 要么有场景声明它，要么在一张写明理由的清单里。

**这条绊线的由来。** `key_rotated` 与 `protected` 两条分支曾经在单测里绿着、
在生产路径上永远到不了：没有任何场景传过 `idempotency_key`，于是每一份报告里
只可能出 `missing_key`。发现它靠的是人肉 grep 一遍 `reason_code`。

这件事的代价不对称：**报告里"没有这条结论"和"这条分支不成立"是同一个样子。**
检测器少说一句话，读的人会当成"没问题"。所以"分支写着但没人走得到"必须被发现，
而下一次不该再靠运气。

清单里的每一行同时是**待办**：补上活样本就得删掉那一行，留着会红（见最后一条
测试）。它也可能永远留着——那就得一直有一句说得过去的理由在那儿。
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "sandbox"
SCENARIOS = ROOT / "scenarios"

#: 到得了、但今天没有任何场景声明它的 reason_code。**这是待办清单，不是批准。**
#:
#: 每一条都要说清"为什么今天没有活样本"，而且理由要能被复核——"暂时没空"不是
#: 理由，"演示对象没有这个能力"是，因为那意味着要造先得改演示 Agent。
NO_LIVE_SAMPLE: dict[str, str] = {
    "memory_key_vanished": (
        "声明的路径不在了、内容也没在别处找到——要造得让记忆文件被删掉或清空，"
        "而演示 Agent 的记忆守卫只会追加，没有删除/改名这个动作（那是被测对象侧的能力）"
    ),
    "memory_moved": (
        "内容换个路径还在。同一个动作的另一半：守卫不会改名，所以走不到"
    ),
    "memory_rolled_back": (
        "可达，但**刻意不声明**：它只在'场景声明了 must_hold、同时又回滚到那条键写入之前'时出现，"
        "也就是两条声明自相矛盾时。这一族要留一支真的不声明判据的场景"
        "（cross_session_memory_loss），声明了就自相矛盾——没有活样本正是对的"
    ),
    "monotonic_unbounded": (
        "要一个**数值**字段在重复调用之间单调推进（page 1..8 那种），"
        "而 mock 的工具表里没有这样的参数：get_order 只有字符串 order_id，"
        "非数值字段一律保守判为不单调（loop.py 的 _all_monotonic）。要造得先加一个分页读工具"
    ),
    "poison_marker_present": (
        "要一个场景声明 memory.forbidden 且标记真的落进记忆——今天没有任何场景声明它。"
        "演示 Agent 的记忆里目前只写它自己的动作记录，标记要进去得让它把工具返回的"
        "内容也写进记忆，是 Agent 侧要加的能力"
    ),
    "unauthorized_attempt_blocked": (
        "要有一次带着 blocked_by_proxy 的调用，而今天 15 个场景一次都没有："
        "场景路径不配出口策略（runner 把它重置成默认策略），"
        "而未声明的工具调用走的是 tool_scope 那条路，不算写操作。"
        "要造得先想清楚这道拦截该由谁下、以及在报告里归给谁"
    ),
}


def _module_strings(tree: ast.Module) -> dict[str, str]:
    """模块级的字符串常量。

    `reason_code` 有两种写法：直接写字面量，和提出来当常量再引用。绊线要都认，
    否则把值提出来当常量就成了绕过它的最短路径——而那种改动看起来是"清理代码"。
    """
    found: dict[str, str] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found[target.id] = node.value.value
    return found


def _emitted_reason_codes() -> dict[str, list[str]]:
    """源码里所有会被写进 `detail["reason_code"]` 的值 → 它出现在哪儿。

    扫的是 `src/sandbox/**` 里每一个**字典字面量**的 `reason_code` 键。读它的地方
    （`detail.get("reason_code")`、`reason_code=` 这种关键字实参）不会被算进来，
    因为它们不是产出方。

    **读不出值就报错，不是跳过。** 认不出来的写法（f-string、三目、拼出来的字符串）
    会让这条绊线静默地少看一个码——那正是它要防的那种失效，所以宁可在这里红。
    """
    emitted: dict[str, list[str]] = defaultdict(list)
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants = _module_strings(tree)
        rel = path.relative_to(ROOT)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values, strict=False):
                if not (isinstance(key, ast.Constant) and key.value == "reason_code"):
                    continue
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    code = value.value
                elif isinstance(value, ast.Name):
                    code = constants.get(value.id)
                else:
                    code = None
                assert code is not None, (
                    f"{rel}:{node.lineno} 的 reason_code 读不出常量值——"
                    "写成字面量或模块级字符串常量，这条绊线才看得见它"
                )
                emitted[code].append(f"{rel}:{node.lineno}")
    return emitted


def _declared_reason_codes() -> dict[str, list[str]]:
    """场景的 `expect` 里声明过的 reason_code → 声明它的地方。"""
    declared: dict[str, list[str]] = defaultdict(list)
    for path in sorted(SCENARIOS.glob("*.yaml")):
        payload: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for index, item in enumerate(payload.get("expect") or []):
            code = item.get("reason_code") if isinstance(item, dict) else None
            if code:
                declared[str(code)].append(f"{path.name}:expect[{index}]")
    return declared


def test_every_reason_code_has_a_scenario_or_a_written_reason() -> None:
    """产出方在 `src/`，声明方在 `scenarios/`，两边必须对得上。

    对不上只有两种可能：这个码还没有任何场景走到过（要么去补一个场景，
    要么把理由写进 `NO_LIVE_SAMPLE`），或者场景里写了个打错的码。
    """
    emitted = _emitted_reason_codes()
    declared = _declared_reason_codes()

    assert len(emitted) > 5, f"只扫到 {len(emitted)} 个 reason_code，抽取本身可能坏了"

    unaccounted = {
        code: sorted(where)
        for code, where in emitted.items()
        if code not in declared and code not in NO_LIVE_SAMPLE
    }
    assert not unaccounted, (
        "这些 reason_code 在源码里写下了，却既没有场景声明它、也没有在 NO_LIVE_SAMPLE 里"
        "写明理由——**分支在、而没有任何场景会走到它**：\n    "
        + "\n    ".join(f"{code}（{where[0]}）" for code, where in sorted(unaccounted.items()))
        + "\n两者选一个：补一个真会走到它的场景（最好是对照组里的一支），"
        "或者把'为什么今天没有活样本'写进本文件的 NO_LIVE_SAMPLE。"
    )


def test_the_no_live_sample_list_has_not_rotted() -> None:
    """清单不许陈旧，也不许和场景打架。

    补上活样本之后忘了删那一行，就是同一个仓库里两处说法不一致；而删掉一条分支
    之后忘了删那一行，会让人以为那儿还有个待办。
    """
    emitted = _emitted_reason_codes()
    declared = _declared_reason_codes()

    covered_now = sorted(set(NO_LIVE_SAMPLE) & set(declared))
    assert not covered_now, (
        f"这些码已经有场景声明了，NO_LIVE_SAMPLE 里那几行该删掉：{covered_now}"
    )

    gone = sorted(set(NO_LIVE_SAMPLE) - set(emitted))
    assert not gone, f"这些码源码里已经不再产出了（改名或删分支了？）：{gone}"

    # 反向也看一眼：场景里写的码必须真的有人产出，否则那是一条永远不会命中的预期
    # ——它不会报错，只会让运行以"预期未达成"收场，而那是运行期的事，这里是测试期。
    unknown = sorted(set(declared) - set(emitted))
    assert not unknown, (
        f"这些码只在场景里出现、源码里没有产出方（拼错了？）："
        f"{ {code: declared[code] for code in unknown} }"
    )


def test_the_list_is_actually_consulted() -> None:
    """绊线的绊线：上面两条断言比的是两张真实的表，不是空对空。

    如果有人把 `_emitted_reason_codes` 改坏了（返回空字典），上一条会变成永真。
    这条钉住"表里有东西、而且清单里的码确实在里面"。
    """
    emitted = _emitted_reason_codes()
    assert set(NO_LIVE_SAMPLE) <= set(emitted), (
        "NO_LIVE_SAMPLE 里有源码中扫不到的码——抽取坏了，还是清单写错了？"
    )
    assert len(_declared_reason_codes()) > 3, "场景侧一个码都没读到，声明抽取坏了"


if __name__ == "__main__":  # pragma: no cover - 手工排查用
    print("产出方：")
    for code, where in sorted(_emitted_reason_codes().items()):
        mark = "有场景" if code in _declared_reason_codes() else "待办  "
        print(f"  {mark} {code:<32} {where[0]}")
    pytest.main([__file__, "-q"])
