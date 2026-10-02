"""边界上 `status` 的取值名单，以及"到没到目标系统"这根轴。

`CallView.reached_target` 判的是"这次调用到没到目标系统"，而它依据的
`REACHED_TARGET` / `REJECTED_AT_BOUNDARY` 是**手抄的名单**。名单和真正产出这些
取值的地方（`proxy/app.py` 里那十处 `record_tool_call(...)`）之间，原先没有
任何东西在核对——这就是本文件存在的全部理由。

**为什么这件事值得一个文件。** 漏掉一种 status 的症状是一条**静默**的路径：
新加一种拒绝方式，检测器不认识它，`reached_target` 给它的答案就取决于名单里
有没有它——而两种答案都"能跑"，都不会有任何东西变红：

  * 往 False 那边歪是漏报：真的到了目标系统的调用被判成没发生过，一次重复扣款
    消失在一句 inconclusive 里；
  * 往 True 那边歪是误报：没发生的被判成发生了——而这正是这次修掉的那个缺陷
    （三次参数写错的 `create_payment` 拿到和三次真扣款一字不差的报告）。

**核对分三层，缺一层都留一条静默路径：**

1. 名单**两头**都核。生产里写过的每一种都在名单上（漏了就没分类）；名单上的
   每一种生产里都写过（没写过的那是一句没接线的话——`reached_target` 声称它
   能判，而边界上产生不出这种取值）；
2. 两个集合**不重不漏**——重叠的话同一次调用的答案是"看哪边先问"，而
   `reached_target` 只问一边；
3. **跨字段**：真实边界上 `blocked_by_proxy=True` 永远和 `status="blocked"`
   成对出现（十处写法一致）。这两个字段一旦打架，记录里就出现一条边界上产生
   不出来的调用，而检测器只认得出其中一种说法——`tests/test_cross_session.py`
   里那条造出过 `status="ok"` + `blocked_by_proxy=True` 的假记录，就是这么来的。

第 1 层靠不上一份**手抄的计数**是不够的：AST 走空、或者 `status` 换了种写法
（不再传字面量），下面那些集合比对会**空着通过**。所以第一条测试是计数闸门。
"""

from __future__ import annotations

import ast
from pathlib import Path

from sandbox.detectors.base import CALL_STATUSES, REACHED_TARGET, REJECTED_AT_BOUNDARY

REPO = Path(__file__).resolve().parents[1]
PROXY = REPO / "src" / "sandbox" / "proxy" / "app.py"

#: 边界上写 `status` 的地方有多少处。**这个数字不是装饰**：AST 走空、或者函数
#: 改了名，下面三条会一条不落地全部通过。加一处 `record_tool_call` 就把这个数
#: 改掉，同时想清楚它的 status 属于哪一档——想不清楚，那正是要在这里停下的信号。
SITES = 10


def _strings(node: ast.expr | None) -> set[str]:
    """这个实参可能取到的字面量字符串。

    条件表达式（`"ok" if ok else "error"`）两个分支都要收——只看 `body` 的话，
    `error` 会在生产里出现而名单上没有，而这条测试会报「名单上多了一个」，
    指向一个完全错误的方向。
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.IfExp):
        return _strings(node.body) | _strings(node.orelse)
    return set()


def _sites() -> list[tuple[int, dict[str, ast.expr]]]:
    """`proxy/app.py` 里每一处 `record_tool_call(...)` 的关键字实参。

    **计数闸门长在这里，不长在一条单独的测试里。** 走空、或者函数改了名，
    下面几条集合比对都会**空着通过**——而"另一条测试会红"救不了它们：红的
    那一条说的是 AST 走法坏了，这个文件里其余几条照样显示绿色。所以每个用
    它的人都在这里撞上同一堵墙。
    """
    tree = ast.parse(PROXY.read_text(encoding="utf-8"))
    found: list[tuple[int, dict[str, ast.expr]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name != "record_tool_call":
            continue
        found.append((node.lineno, {kw.arg: kw.value for kw in node.keywords if kw.arg}))
    assert len(found) == SITES, (
        f"`proxy/app.py` 里数出 {len(found)} 处 record_tool_call（文件头声明的是 "
        f"{SITES} 处）。加了一处就改 `SITES` 并想清楚它的 status 属于哪一档；"
        f"少了一处说明这个 AST 走法已经读不到它们了——那样下面几条全是摆设。"
    )
    return found


def test_the_walk_finds_the_real_call_sites() -> None:
    """计数闸门本身也有一条，好让它红的时候读得出来是"走法坏了"。"""
    missing = [line for line, kw in _sites() if "status" not in kw]
    assert not missing, f"这几处没传 status，边界上的结果就没被记下来：{missing}"


def test_every_status_the_boundary_writes_is_on_the_list() -> None:
    """生产里写过的每一种取值，都有人给它分过类。"""
    written: set[str] = set()
    for line, kw in _sites():
        values = _strings(kw["status"])
        assert values, (
            f"`proxy/app.py:{line}` 的 status 不是字面量，这里读不出它可能取什么值。"
            f"读不出来就等于没核对——换个写法，或者把这条测试教会。"
        )
        written |= values

    unknown = written - CALL_STATUSES
    assert not unknown, (
        f"边界上会写出这些 status，而 `base.py` 的名单里没有它们：{sorted(unknown)}。"
        f"`reached_target` 现在给它们的答案是**碰巧**的：不在 `REACHED_TARGET` 里"
        f"就一律算没到。到没到是个判断，不该碰巧——挑一边写进名单。"
    )


def test_every_status_on_the_list_is_written_somewhere() -> None:
    """名单上的每一种，边界上都得真的写得出来。写不出来的那是一句没接线的话。"""
    written: set[str] = set()
    for _, kw in _sites():
        written |= _strings(kw["status"])

    unproduced = CALL_STATUSES - written
    assert not unproduced, (
        f"`base.py` 的名单上有这些 status，而 `proxy/app.py` 一处都不写它们："
        f"{sorted(unproduced)}。这是「装上了但没接线」：`reached_target` 声称能判"
        f"这种情况，而边界上产生不出这种记录。"
    )


def test_the_two_lists_do_not_overlap() -> None:
    """不重：重叠的取值两边都收，`reached_target` 只问一边，答案是看哪边先问。"""
    both = REACHED_TARGET & REJECTED_AT_BOUNDARY
    assert not both, (
        f"这些 status 同时在「到了」和「没到」两份名单里：{sorted(both)}。"
        f"同一次调用不可能既到了又没到——这两份名单是 `CALL_STATUSES` 的一个划分。"
    )
    assert CALL_STATUSES == REACHED_TARGET | REJECTED_AT_BOUNDARY


def test_a_proxy_block_is_always_recorded_as_blocked() -> None:
    """`blocked_by_proxy=True` 的那几处，status 必须是 `blocked`。

    这两根轴在真实边界上从不打架：「谁拦的」和「到没到」是两个问题，而策略拦下的
    调用一定没到。打架的记录是边界上产生不出来的——检测器只认得出其中一种说法，
    于是同一份轨迹在「按标记过滤」和「按状态过滤」两种读法下给出两个结论。
    """
    offenders = [
        line
        for line, kw in _sites()
        if isinstance(kw.get("blocked_by_proxy"), ast.Constant)
        and kw["blocked_by_proxy"].value is True
        and _strings(kw.get("status")) != {"blocked"}
    ]
    assert not offenders, (
        f"这几处把 blocked_by_proxy 写成 True，status 却不是 blocked：{offenders}。"
        f"记录里会出现一条边界上产生不出来的调用（标记说被拦、状态说发出去了）。"
    )
