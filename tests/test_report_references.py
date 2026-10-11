"""模板读的快照字段，必须真的存在于快照里。

## 这道闸门是被一个真事故逼出来的

`run.html` 最上面那张「检测结论」卡片读的是 `report.expectation.ok`，而
`ExpectResult.to_dict()` 当时**不吐 `ok`**（它只是一个 property）。`report` 是
**冻结下来的 JSON 快照**——一个普通 dict，不是那个对象——所以 Jinja 取不到，
给的是 Undefined，恒假。于是那张卡片在**每一份**报告上都是红的：库里 396 份判为
"通过"的运行，全都写着「不符合预期」。

这类失效的三个特点，正好决定了这道闸门要长什么样：

1. **它不报错。** 取一个不存在的键，Jinja 默认不抛异常，渲染出来只是"假"——
   而"假"在一张红/绿卡片上和一个真实的否定结论长得一模一样。
2. **测试全绿。** 没有任何一条测试渲染过这一页并核对那三个字，所以全部测试
   一起放它过去。
3. **报告说了比观测更多的话。** 它断言了一件它当时并不知道的事，还坐在每页最上面。

## 检查的范围是**从源码推出来的**，不是手写的模板清单

只有 `report` 装的是**运行快照**的那些模板才归这里管。别的页面也把变量叫
`report`，但装的是另一样东西：`metrics.html` 装的是指标报告。拿运行快照的
键去要求它们，会当场报出一堆并不存在的缺陷——
**一个会误报的闸门迟早被整体忽略**，那时它连真的缺陷也一起放过去了。

所以"哪些模板归这里管"由 `app.py` 里 `TemplateResponse` 的实参现推：`report`
的值来自 `parse_report(...)` 的就是运行快照（直接写在实参里，或者先落到一个局部
变量再进实参，两种都认）。这条推法本身也有一条相机盯着
（`test_the_run_snapshot_template_set_is_what_we_think`），免得哪天有人加了一页
`run2.html` 而闸门**静默地**没去看它。

**它已经咬过一次了。** `run_detail` 后来把那份快照提成了一个局部变量（同一个
快照还要交给 `_finding_actions`），而推法当时只认"直接写在实参里"这一种——
`run.html` 于是掉出范围，扫描结果变成空集，五条断言一起红。**那次红是相机在
干活，不是误报**：闸门真的什么都没查了。修的方向是让推法跟上这种写法（见
`_run_snapshot_templates` 的说明），不是把代码改回去迁就正则。

范围之外的是什么，写在这儿而不是假装不存在：`metrics.html` 与
它们各有各的产出函数，要不要照样配一道闸门是另一件事。

## 比的是**两张真表**，不是手抄的清单

左边：上面那些模板里真的写着 `report.X` / `report.X.Y` 的路径。
右边：把 `ScenarioResult` **真的构造出来、真的调 `to_dict()`** 拿到的键。

右边绝不手抄。手抄一份键名表的话，`to_dict` 改了名而表没改，这道闸门就会开始
**保护一个已经不存在的东西**——那比没有闸门更坏，因为它会挡住正确的修改。

## 哪些判得了、哪些判不了

* `report.<k>` —— 一律判得了：`<k>` 在不在快照里。
* `report.<k>.<j>` 且 `<k>` 的子键来自一个 `to_dict()` —— **判得了**，
  这正是出过事的那一类。子键从**类型注解**推：`expectation: ExpectResult | None`
  就构造一个 `ExpectResult()` 调它的 `to_dict()`。
* `report.<k>.<j>` 且 `<k>` 序列化出来是个**普通字典**（`session_costs` 就是）——
  它的键是什么要跑一次真实运行才知道，**静态判不了**，跳过。模板里那种写法今天
  只有 `report.session_costs.items`，那是字典的方法不是快照的键。
* 三层及以上 —— **判红**。今天一条都没有，所以现在加这个约束是零代价的；
  而"留一条不判的深路径"正是缺陷能藏进去的地方。
* 推不出子键（那个类构造不出默认实例）—— 跳过，但**用相机钉住"判了几条"不许塌**，
  免得覆盖面悄悄退化成零。

下面这些跳过都是**记了账的**，不是"没看见"。
"""

from __future__ import annotations

import re
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sandbox.scenario.expect import ExpectResult
from sandbox.scenario.runner import ScenarioResult

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "src" / "sandbox" / "api" / "templates"
APP_PY = ROOT / "src" / "sandbox" / "api" / "app.py"

#: `report` 后面跟着的点号路径。
#:
#: `report.py`（`\b` 在 `/` 和 `r` 之间是成立的），于是这道闸门第一版对着
#: `_macros.html` 里一句注释报了个"快照里没有 `report.py` 这个键"。
_PATH = re.compile(r"(?<![\w/.])report((?:\.[A-Za-z_][A-Za-z0-9_]*)+)")

#: `{% set exp = report.expectation %}` —— 给快照里的某一栏起个短名字。
#: 之后写 `exp.ok` 读的还是 `report.expectation.ok`，所以别名要跟着走。
_ALIAS = re.compile(r"\{%-?\s*set\s+([A-Za-z_]\w*)\s*=\s*report((?:\.[A-Za-z_]\w*)+)")

#: `TemplateResponse` 里模板名之后多远算"这个模板的实参"。
_CALL_WINDOW = 600

#: 字典自带的属性。`report.session_costs` 序列化出来是个普通 dict，
#: 模板里写 `report.session_costs.items` 是在调字典的方法，不是读一个快照键。
_DICT_ATTRS = frozenset({"items", "keys", "values", "get"})


#: `report = queries.parse_report(run)` 这种**先落到局部变量、再进实参**的写法。
#: 命名分组 `name` 是那个变量名。
_REPORT_LOCAL = re.compile(r"\b(\w+)\s*=\s*[^=\n]*\bparse_report\(")


def _run_snapshot_templates() -> set[str]:
    """哪些模板的 `report` 装的是运行快照——**从 `app.py` 现推**。

    写成清单手抄的话，新加一页就漏一页，而漏掉的表现是"这一页没被查过"
    ——和"这一页查过了没问题"长得一模一样。

    推法两步：

    1. `"report"` 的值直接在 `TemplateResponse` 的实参里写着 `parse_report(...)`；
    2. 或者**先落到一个局部变量**再进实参（`report = queries.parse_report(run)`
       写在上面，实参里是 `"report": report`）。

    第二步不是锦上添花：原先只判第一步，而 `run_detail` 因为要把同一个快照交给
    `_finding_actions` 而把它提了出去，于是 `run.html` **静默地掉出了这道闸门的
    范围**——扫描结果变成空表，五条断言一起红。红是好事（`test_the_run_snapshot_
    template_set_is_what_we_think` 那条相机正是为这一刻准备的），但闸门的正确反应
    是**跟上这种写法**，不是把代码改回去迁就它：哪一种写法更好读，不该由扫描器的
    正则决定。
    """
    text = APP_PY.read_text(encoding="utf-8")
    locals_from_parse = {m.group(1) for m in _REPORT_LOCAL.finditer(text)}
    found: set[str] = set()
    for match in re.finditer(r'"([\w_]+\.html)"', text):
        window = text[match.start() : match.start() + _CALL_WINDOW]
        if "parse_report(" in window:
            found.add(match.group(1))
            continue
        bound = re.search(r'"report"\s*:\s*(\w+)', window)
        if bound and bound.group(1) in locals_from_parse:
            found.add(match.group(1))
    return found


def _template_paths() -> dict[tuple[str, ...], set[str]]:
    """归这道闸门管的模板里出现过的 `report.` 路径 → 哪些模板用了它。

    **别名要跟着走**：`{% set exp = report.expectation %}` 之后写 `exp.ok`，
    读的还是快照里的 `expectation.ok`。不跟这条别名，闸门就会瞎——第一版正是
    这样：模板改成用中间变量之后它再也看不见 `expectation.ok` 了，而六条断言
    照样全绿。**一条测不出退化的闸门和一个好闸门长得一模一样。**
    """
    found: dict[tuple[str, ...], set[str]] = {}
    for name in sorted(_run_snapshot_templates()):
        path = TEMPLATES / name
        assert path.exists(), f"{name} 在 app.py 里被引用，模板目录里却没有"
        text = path.read_text(encoding="utf-8")
        for match in _PATH.finditer(text):
            parts = tuple(match.group(1).lstrip(".").split("."))
            found.setdefault(parts, set()).add(name)
        for alias_match in _ALIAS.finditer(text):
            alias = alias_match.group(1)
            base = tuple(alias_match.group(2).lstrip(".").split("."))
            # `\b` 在这里不够：`exp_state` 也会匹配上别名 `exp`。要求别名后面
            # 紧跟一个点，`exp_state` 就落不进来了。
            pattern = re.compile(rf"(?<![\w/.]){re.escape(alias)}((?:\.[A-Za-z_]\w*)+)")
            for match in pattern.finditer(text):
                extra = tuple(match.group(1).lstrip(".").split("."))
                found.setdefault(base + extra, set()).add(name)
    return found


def _sample_report() -> dict[str, Any]:
    """一份真的快照——**真的构造、真的序列化**，不是手抄的键名表。"""
    return ScenarioResult(scenario_id="gate", run_id="gate").to_dict()


def _sub_keys(name: str) -> set[str] | None:
    """`report.<name>` 序列化出来有哪些子键；推不出来返回 None。"""
    annotation = typing.get_type_hints(ScenarioResult).get(name)
    if annotation is None:
        return None
    # `ExpectResult | None` 要先剥掉 Optional 那层壳：`X | None` 的 args 是 (X, NoneType)。
    for candidate in (annotation, *typing.get_args(annotation)):
        if isinstance(candidate, type) and hasattr(candidate, "to_dict"):
            try:
                return set(candidate().to_dict())
            except Exception:
                # 构造不出默认实例（`ClockReport` 那种要参数的）——这一栏今天判不了。
                return None
    return None


def _unjudged(paths: dict[tuple[str, ...], set[str]], emitted: dict[str, Any]) -> list[str]:
    """这份数据上**判不了**的路径。相机拿它钉住覆盖面不许悄悄塌掉。"""
    out: list[str] = []
    for parts in paths:
        if len(parts) == 1:
            continue
        head = parts[0]
        # 两种"判不了"，理由不同但都不该悄悄发生：
        #   一是这一栏序列化出来是普通字典，键要跑一次真实运行才知道；
        #   二是这一栏的类型构造不出默认实例，子键推不出来。
        dynamic_dict = isinstance(emitted.get(head), dict) and parts[1] in _DICT_ATTRS
        if dynamic_dict or _sub_keys(head) is None:
            out.append(".".join(parts))
    return sorted(out)


def _missing(
    paths: dict[tuple[str, ...], set[str]],
    *,
    top: set[str],
    sub: Callable[[str], set[str] | None],
    emitted: dict[str, Any],
) -> list[str]:
    """把读不到的路径列出来。

    抽成纯函数是为了能被**反向**测：`test_the_gate_bites_when_a_key_goes_missing`
    喂一份故意少一个键的 `sub`，这里必须报出来。只测"真数据上全绿"的话，
    一个永远返回空清单的实现也能过——而那个实现什么也没查。
    """
    bad: list[str] = []
    for parts, used_by in sorted(paths.items()):
        where = "、".join(sorted(used_by))
        head = parts[0]
        if head not in top:
            bad.append(f"report.{head}（{where}）：快照里没有这个键")
            continue
        if len(parts) == 1:
            continue
        if len(parts) > 2:
            bad.append(
                f"report.{'.'.join(parts)}（{where}）：三层及以上的路径这道闸门没有实现。"
                f"要么把它补成判得了的，要么别在模板里读这么深——"
                f"**不判的深路径正是缺陷能藏进去的地方**。"
            )
            continue
        if isinstance(emitted.get(head), dict):
            # 普通字典，键是每次运行现长的：静态判不了。唯一要拦的是把字典方法
            # 当成快照键（`report.session_costs.items` 是合法的，`session_costs`
            # 序列化出来就是 dict）。其余一概跳过——**宁可说不判，不可误报。**
            if parts[1] not in _DICT_ATTRS:
                bad.append(
                    f"report.{'.'.join(parts)}（{where}）：`{head}` 序列化出来是个普通字典，"
                    f"它的键要跑一次真实运行才知道。这道闸门判不了它——"
                    f"请把它写进 `_unjudged()` 的解释里，别让它静默地没被查过。"
                )
            continue
        keys = sub(head)
        if keys is None:
            continue  # 记在 _unjudged 里，由相机钉住条数
        if parts[1] not in keys:
            bad.append(
                f"report.{'.'.join(parts)}（{where}）：`{head}` 序列化出来的键是 "
                f"{sorted(keys)}，没有 {parts[1]!r}"
            )
    return bad


# --- 断言 ----------------------------------------------------------------


def test_the_run_snapshot_template_set_is_what_we_think() -> None:
    """范围本身也要有相机：拿运行快照的模板**只有 `run.html` 一个**。

    这条盯的是"闸门静默地少看了一页"：哪天有人照 `run.html` 加一页 `run2.html`
    也吃 `parse_report`，那上面所有 `report.X` 就自动归这道闸门管；反过来，
    要是推法坏了（正则退化、`app.py` 改了写法），`_run_snapshot_templates()`
    会返回空集，下面几条会在**空集**上全绿——一个什么都没查的闸门。
    """
    found = _run_snapshot_templates()
    assert found == {"run.html"}, (
        f"从 app.py 推出来的「运行快照模板」是 {sorted(found) or '（空集）'}，而预期只有 run.html。"
        f"空集意味着下面的检查一条都没跑；多出来意味着有新页面归这道闸门管了——"
        f"那是好事，但要把这条预期一起改掉。"
    )


def test_the_scan_is_not_vacuous() -> None:
    """相机：比的那两张表必须都是真的、且有内容。"""
    paths = _template_paths()
    assert len(paths) >= 30, f"只从模板里扫到 {len(paths)} 条 report 路径，抽取八成塌了"
    deep = [p for p in paths if len(p) > 1]
    assert len(deep) >= 5, f"只扫到 {len(deep)} 条两级以上的路径，正则八成塌了"
    sample = _sample_report()
    assert len(sample) >= 15, f"快照只有 {len(sample)} 个键？to_dict 八成塌了"


def test_the_judged_coverage_has_not_collapsed() -> None:
    """判得了的条数不许悄悄变成 0。

    子键是从类型注解推的。哪天 `ScenarioResult` 的字段全改成构造不出默认实例的
    类型，`_sub_keys` 会一律返回 None，上面那条测试照样绿——**覆盖面塌了而闸门
    一声不吭**，这正是这个项目一直在修的那类失效。
    """
    paths = _template_paths()
    judged = [p for p in paths if len(p) == 2 and _sub_keys(p[0]) is not None]
    assert len(judged) >= 5, (
        f"两个字段的路径里只有 {len(judged)} 条判得了子键，覆盖面塌了。"
        f"判不了的：{_unjudged(paths, _sample_report())}"
    )


def test_every_field_a_template_reads_exists_in_the_snapshot() -> None:
    """模板读到的每一个 `report.` 路径都要真的存在。见文件头的事故。"""
    sample = _sample_report()
    bad = _missing(_template_paths(), top=set(sample), sub=_sub_keys, emitted=sample)
    assert not bad, (
        "模板读了快照里没有的东西。Jinja 取不到不会报错，只会渲染成假——\n"
        "而一张恒红的卡片和一个真实的否定结论长得一模一样：\n    "
        + "\n    ".join(bad)
        + "\n\n要么把它加进 to_dict()，要么改模板——**不要**在模板里现算一遍判定，"
        "那就是第二份会漂的说法。"
    )


def test_the_gate_bites_when_a_key_goes_missing() -> None:
    """这条闸门在**真出过事的那一处**必须真的咬得住。

    喂它一个少了 `ok` 的 `ExpectResult`，它必须报出 `report.expectation.ok`。
    少了这条，`_missing` 里任何一处提前 return 都能让上面那条测试继续绿——
    而上面那条绿着，这个缺陷就能第二次溜进来。
    """
    real_sub = _sub_keys

    def without_ok(name: str) -> set[str] | None:
        keys = real_sub(name)
        return None if keys is None else (keys - {"ok"})

    sample = _sample_report()
    bad = _missing(_template_paths(), top=set(sample), sub=without_ok, emitted=sample)
    assert any("report.expectation.ok" in line for line in bad), (
        "把 `ok` 从快照里拿掉之后，这道闸门没有报出 `report.expectation.ok`——"
        f"它咬不住它本该咬住的那个缺陷。它报的是：{bad or '（空）'}"
    )


def test_the_fixed_key_really_is_in_the_snapshot_and_read_by_the_template() -> None:
    """正面：`ok` 真的在快照里，而模板真的在读它。

    这是上面那条反向测试的对照。只有反向的话，一个"总是报错"的实现也能过——
    而那种实现会让所有人学会忽略这道闸门。
    """
    assert "ok" in ExpectResult().to_dict(), (
        "`ExpectResult.to_dict()` 又不吐 `ok` 了——`run.html` 那张卡片会重新恒红"
    )
    assert ("expectation", "ok") in _template_paths(), (
        "模板不再读 `report.expectation.ok` 了？那这条测试和上面那条要跟着一起改，"
        "别留一条保护着不存在的东西的闸门。"
    )
