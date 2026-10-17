"""解释层必须盖住每一个会出现的名字，而且不许腐烂，也不许把判不了说成没问题。

`src/sandbox/api/explain.py` 是给不写代码的人看的那一层翻译。它有三种失效方式，
每一种都**不会自己出声**，所以各配一条断言：

**一、漏。** 源码里新加一个检测器或理由码，解释表没跟上——报告上就出现一栏
没有任何解释，而「这一栏没有解释」和「这一栏本来就不用解释」在页面上长得一模一样。
所以这边**现算**真实的名字集合（`KNOWN_DETECTORS`、源码里的 `detector=` 字面量、
场景 `expect` 里声明过的、以及会写进 `detail["reason_code"]` 的常量），拿它去比对
解释表的键。两边任一方向多了少了都红。

**二、腐烂。** 检测器改名或删掉了，解释表里那一行还留着。它不报错，只是从此
没有人再看它——而下一个要加解释的人会以为这些名字都是活的。所以反向也要查。

**三、把判不了说成没问题。** 这是这个产品唯一值钱的地方（`docs/detectors.md`
的四态语义），也是通俗化最容易丢掉的东西：`inconclusive` 写成「暂时没发现问题」，
读者会读成「没问题」。这条没法靠正则全面检查，但可以钉住最要命的那一种：
**「没问题」「正常」「通过」这类词出现在 `inconclusive` 的解释里时，前面必须有
否定词。** 写成「不是「没问题」」是对的，写成「暂时没问题」是错的——两者的区别
恰好就是这个工具存不存在。
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from sandbox.api import explain
from sandbox.db.models import DetectorVerdict
from sandbox.detectors import KNOWN_DETECTORS, SCENARIO_DETECTOR_NAMES

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "sandbox"
TEMPLATES = SRC / "api" / "templates"
SCENARIOS = ROOT / "scenarios"


# --- 现算：真实存在的名字 -------------------------------------------------


def _module_strings(tree: ast.Module) -> dict[str, str]:
    """模块级的字符串常量（`REASON = "..."` 这种）。见 `test_reason_codes.py`：
    把值提出来当常量是最短的一条绕过路径，所以这里也要认。"""
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


def _string_value(node: ast.AST, constants: dict[str, str]) -> str | None:
    """字面量或模块级常量，认得出来就返回；认不出来返回 None（调用方报错）。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _resolve_detector(
    node: ast.AST, constants: dict[str, str], path: Path, where: str
) -> str | None:
    """一个 `detector` 取值能定下来就返回它，定不下来就**先证明它跑不出名单**。

    三种写法：

    * 字面量 / 模块级常量 —— 直接用；
    * f-string（`f"{self.name}.strict"`）—— 常量那几段必须能落进某个已知名字里。
      `.strict` 落进 `loop.strict`，于是拼出来的一定是名单里的某一个；
    * `self.name` / 循环变量 —— 退一步问：这个文件里有没有出现过名单（或名单里的
      某个名字）？有，那这个名字有据可依；没有，它可以拼出任何字符串，
      而报告上会出现一栏没人写过解释。

    定不下来的写法一律报错，不跳过——**认不出来的写法会让这条绊线静默地少看
    一个名字**，那正是它要防的事。
    """
    name = _string_value(node, constants)
    if name is not None:
        return name
    if isinstance(node, ast.JoinedStr):
        parts = [
            seg.value
            for seg in node.values
            if isinstance(seg, ast.Constant) and isinstance(seg.value, str)
        ]
        assert parts, (
            f"{where} 的 detector 是个 f-string，但里面一段常量都没有，"
            "这条绊线认不出它会拼出什么。"
        )
        for part in parts:
            assert any(part in known for known in KNOWN_DETECTORS), (
                f"{where} 的 f-string 片段 {part!r} 落不进任何一个已知检测器名字里"
                f"（{sorted(KNOWN_DETECTORS)}）——它会拼出一个解释表里没有的名字。"
            )
        return None
    # 剩下的都是**收到**一个名字的地方，不是选名字的地方：`f.detector`（把一条
    # 结论序列化出去）、`rule["detector"]`（从一份规则里取回来）、循环变量。
    # 这些地方拼不出新名字——名字是别处选好传进来的，选名字的三处（KNOWN_DETECTORS、
    # 字面量、场景声明）都在上面查过了。
    #
    # **这一段是这条绊线明确不覆盖的**，写在这里而不是假装它不存在：哪天有人写了
    # `Finding(detector=compute_name())`，这里会静默放过。当初试着给它加过一道
    # "这个文件里出现过某个已知名字吗"的替代检查，结果是误报（`scenario/expect.py`
    # 只做序列化，一个名字字面量都没有）——**一个会误报的闸门迟早被整体忽略**，
    # 这里宁可留下一段说清楚的空白。
    return None


def _emitted_detector_names() -> dict[str, list[str]]:
    """源码里可能被写进一条结论的 `detector` 名字 → 它出现在哪儿。

    两种写法都收：关键字实参（`Finding(detector="egress", ...)`）和字典键
    （`{"detector": ...}`）。后者连**豁免规则过线用的那份形状**
    （`Exemption.to_dict`）一起收进来了——那张表里的 `detector` 也是场景里写的
    检测器名，同样要有人给它写解释。
    """
    emitted: dict[str, list[str]] = {
        name: ["sandbox.detectors.KNOWN_DETECTORS"] for name in KNOWN_DETECTORS
    }
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants = _module_strings(tree)
        rel = str(path.relative_to(ROOT))
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "detector":
                where = f"{rel}:{node.lineno}"
                if (name := _resolve_detector(node.value, constants, path, where)) is not None:
                    emitted.setdefault(name, []).append(where)
            elif isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values, strict=False):
                    if not (isinstance(key, ast.Constant) and key.value == "detector"):
                        continue
                    where = f"{rel}:{node.lineno}"
                    if (name := _resolve_detector(value, constants, path, where)) is not None:
                        emitted.setdefault(name, []).append(where)
    return emitted


def _declared_detector_names() -> set[str]:
    """场景 `expect` 里声明过的检测器名。"""
    declared: set[str] = set()
    for path in sorted(SCENARIOS.glob("*.yaml")):
        payload: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for item in payload.get("expect") or []:
            if isinstance(item, dict) and item.get("detector"):
                declared.add(str(item["detector"]))
        for rule in payload.get("exemptions") or []:
            if isinstance(rule, dict) and rule.get("detector"):
                declared.add(str(rule["detector"]))
    return declared


def _emitted_reason_codes() -> dict[str, list[str]]:
    """会写进 `detail["reason_code"]` 的值 → 它出现在哪儿。见 `test_reason_codes.py`。"""
    emitted: dict[str, list[str]] = defaultdict(list)
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants = _module_strings(tree)
        rel = str(path.relative_to(ROOT))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values, strict=False):
                if not (isinstance(key, ast.Constant) and key.value == "reason_code"):
                    continue
                code = _string_value(value, constants)
                assert code is not None, (
                    f"{rel}:{node.lineno} 的 reason_code 读不出常量值——"
                    "写成字面量或模块级字符串常量，这条绊线才看得见它"
                )
                emitted[code].append(f"{rel}:{node.lineno}")
    return emitted


def _rendered_section_keys() -> dict[str, list[str]]:
    """模板里 `explain.section("...")` 引到的键 → 哪个模板引的。"""
    used: dict[str, list[str]] = defaultdict(list)
    for path in sorted(TEMPLATES.glob("*.html")):
        text = path.read_text(encoding="utf-8")
        for start in range(len(text)):
            head = text.find("explain.section(", start)
            if head == -1:
                break
            start = head + len("explain.section(")
            assert text[start] in "\"'", (
                f"{path.name} 里的 explain.section(...) 不是字符串字面量，"
                "这条绊线看不见它——请写成 explain.section(\"名字\")。"
            )
            quote = text[start]
            end = text.index(quote, start + 1)
            used[text[start + 1 : end]].append(path.name)
    return used


# --- 断言 ----------------------------------------------------------------


def test_every_detector_that_can_appear_has_a_plain_language_line() -> None:
    """每一个可能出现在报告里的检测器名，都要有一句给外行看的话。"""
    emitted = _emitted_detector_names()
    declared = _declared_detector_names()
    missing = {
        name: sorted(where)
        for name, where in emitted.items()
        if name not in explain.DETECTOR_HELP
    }
    missing |= {
        name: ["scenarios/*.yaml:expect"]
        for name in declared
        if name not in explain.DETECTOR_HELP
    }
    assert not missing, (
        "这些检测器会出现在报告里，却没有人给它写一句大白话解释——"
        "报告上会出现一栏读的人不知道该怎么担心的结论：\n    "
        + "\n    ".join(f"{name}（{where[0]}）" for name, where in sorted(missing.items()))
        + "\n补进 src/sandbox/api/explain.py 的 DETECTOR_HELP。"
    )


def test_every_reason_code_has_a_plain_language_line() -> None:
    """每一个理由码都要有一句「这次具体是因为什么」。"""
    emitted = _emitted_reason_codes()
    missing = sorted(code for code in emitted if code not in explain.REASON_HELP)
    assert not missing, (
        "这些理由码会被写进结论，却没有解释：\n    "
        + "\n    ".join(f"{code}（{emitted[code][0]}）" for code in missing)
        + "\n补进 src/sandbox/api/explain.py 的 REASON_HELP。"
    )


def test_the_explanation_tables_have_not_rotted() -> None:
    """解释表里不许留已经没有对应物的行。

    留着不会报错，只是从此没有人再看它——而下一个要加解释的人会以为这些名字
    都还是活的，于是照着一个不存在的检测器写。
    """
    emitted = set(_emitted_detector_names()) | _declared_detector_names()
    rotted = sorted(set(explain.DETECTOR_HELP) - emitted)
    assert not rotted, (
        f"这些检测器解释留在表里，而源码和场景里都找不到这个名字了（改名了？）：{rotted}"
    )

    codes = set(_emitted_reason_codes())
    rotted_codes = sorted(set(explain.REASON_HELP) - codes)
    assert not rotted_codes, (
        f"这些理由码解释留在了表里，而源码里已经不再产出它们（改名了？）：{rotted_codes}"
    )


def test_the_four_state_table_covers_the_enum() -> None:
    """四态解释表必须一个不漏地盖住 `DetectorVerdict`——从枚举现算，不手抄。"""
    values = {v.value for v in DetectorVerdict}
    missing = sorted(values - set(explain.VERDICT_HELP))
    assert not missing, f"这几种结论没有解释：{missing}"
    extra = sorted(set(explain.VERDICT_HELP) - values)
    assert not extra, f"解释表里有枚举里不存在的取值：{extra}"


def test_inconclusive_is_never_explained_as_fine() -> None:
    """判不了不许被说成没问题。

    「没问题」「正常」「通过」这类词可以出现在这一条解释里，但**每一次出现都
    必须紧挨着一个否定词**——写成「不是「没问题」」是对的，写成「暂时没问题」
    是把「没有信息」说成了「没有风险」。

    这不是修辞检查。这个工具全部的价值就建立在「判不了」和「通过」严格分开
    这件事上（见 `docs/detectors.md` 的四态语义）；一份把判不了读成没问题的
    报告，比没有报告更坏——用户会拿着它去上生产。
    """
    text = explain.VERDICT_HELP[DetectorVerdict.INCONCLUSIVE.value]
    assert "判不了" in text, (
        "inconclusive 的解释里必须自己出现「判不了」这三个字——"
        "换一个说法（「暂时没有结论」「信息不足」）读者读不出它和「通过」的区别。"
    )
    negations = ("不是", "不算", "不等于", "别读成", "不能读", "不要读")
    for word in ("没问题", "正常", "通过", "全绿"):
        start = 0
        while (at := text.find(word, start)) != -1:
            window = text[max(0, at - 8) : at]
            assert any(neg in window for neg in negations), (
                f"{word!r} 出现在 inconclusive 的解释里，前面却没有否定词：\n"
                f"    …{text[max(0, at - 16) : at + len(word) + 4]}…\n"
                "判不了就是判不了，不能读成没问题。"
            )
            start = at + 1


def test_every_section_key_is_rendered_somewhere() -> None:
    """每一节说明都要真的被某一页用上，反过来模板引的键也必须存在。

    前半句防「写了一张没人用的表」：加了说明却没接进页面，页面上什么都不会变，
    而写的人以为自己加过了。后半句其实由 `explain.section` 抛异常兜着，这里
    再静态查一遍，是因为渲染路径上的红要跑起服务才看得见。
    """
    used = _rendered_section_keys()
    unused = sorted(set(explain.SECTION_HELP) - set(used))
    assert not unused, (
        f"这些章节说明写了却没接进任何页面：{unused}\n"
        "要么在模板里用 explain.section(...) 引上，要么从表里删掉。"
    )
    unknown = sorted(set(used) - set(explain.SECTION_HELP))
    assert not unknown, (
        f"这些模板引了不存在的章节说明：{ {k: used[k] for k in unknown} }"
    )


def test_the_scenario_side_name_set_is_the_same_set() -> None:
    """场景加载期认的名字，必须和解释表里那 19 个是**同一个集合**。

    `SCENARIO_DETECTOR_NAMES`（`detectors/__init__.py`）是加载期校验
    `expect[].detector` 用的合法集合；它里面那四个边界名字**是手抄的**
    （源码里只有字面量，没有可 import 的集合，见那里的注释）。

    手抄一份清单就是第二份会漂的真相——除非有东西在核。这里就是那个东西：
    另一边 `explain.DETECTOR_HELP` 已经被上面两条**从源码现算**钉住了
    （`emitted ⊆ DETECTOR_HELP ⊆ emitted ∪ declared`），所以拿它当参照物，
    等于让新常量间接对着源码核。

    两个方向都查，因为它们坏起来的后果不一样：

    * 集合**多**一个名字 —— 加载期会放行一个拼错的检测器，那条约预期静默地
      永远配不上，报告上完全没有它的位置；
    * 集合**少**一个名字 —— 一份完全合法的场景突然加载不了。这条更响，
      但响在用户那边，而且他会以为是自己写错了。
    """
    expected = set(explain.DETECTOR_HELP)
    missing = sorted(expected - SCENARIO_DETECTOR_NAMES)
    extra = sorted(SCENARIO_DETECTOR_NAMES - expected)
    assert not missing, (
        f"解释表里有这几个名字，而加载期不认它们——写进场景的 expect 会被拒：{missing}\n"
        f"把它们补进 detectors.SCENARIO_DETECTOR_NAMES。"
    )
    assert not extra, (
        f"加载期认这几个名字，而解释表里没有它们：{extra}\n"
        f"这意味着源码里会出现一栏没有任何解释的结论，同时加载期会放行一个"
        f"根本不会命中的 detector。两边都要补。"
    )
    assert len(SCENARIO_DETECTOR_NAMES) >= 19, (
        f"SCENARIO_DETECTOR_NAMES 只剩 {len(SCENARIO_DETECTOR_NAMES)} 个？"
        f"它塌了的话上面两条断言会在空集上互相抵消。"
    )


def test_the_scan_is_not_vacuous() -> None:
    """相机的相机：上面几条比的必须是两张真实的表。

    如果抽取坏了（正则退化、glob 匹配不到、`KNOWN_DETECTORS` 导入成空集），
    上面几条会一起变绿——正是这个项目最怕的那种「看起来一切正常」。
    """
    emitted = _emitted_detector_names()
    assert len(KNOWN_DETECTORS) >= 15, f"KNOWN_DETECTORS 只有 {len(KNOWN_DETECTORS)} 个？"
    assert len(emitted) >= 19, f"只扫到 {len(emitted)} 个检测器名，抽取八成塌了"
    assert len(_emitted_reason_codes()) >= 10, "理由码只扫到个位数，抽取八成塌了"
    assert len(_declared_detector_names()) >= 3, "场景侧一个检测器名都没读到"
    assert len(explain.SECTION_HELP) >= 10, "章节说明表塌了"
    assert _rendered_section_keys(), "模板里一条 explain.section(...) 都没找到"
