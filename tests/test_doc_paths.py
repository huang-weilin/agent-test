"""文档里指名道姓的文件，必须真的在。

这个仓库里指一个文件的标准写法是反引号包起来的路径（``` `docs/cost.md` ```）。
一旦那个文件被删掉或改名，那句话就从"引一份文件"变成"引一个不存在的东西"——
而**读的人不会去核**：他看到一个像是路径的字符串，默认它是对的。

这个缺陷在 2026-10-01 真的发生过一次：`scripts/run_walking_skeleton.py` 被删了
（理由见 README），而 README 的三处、`scenario/runner.py` 的模块文档各留着一句
指向它的话。删之前没人拦，删之后也没有——**因为没有任何东西在核对这件事**。
这道闸门就是那个"东西"。

## 只认反引号，而且只认带斜杠的

不带斜杠的裸文件名（`runner.py`）在散文里到处都是泛指，收进来会立刻变成噪音，
而**一个会误报的闸门迟早被整体忽略**（同一条理由见 `test_report_text.py` 里
"为什么 `tests/` 不在名单上"）。带斜杠的才是在指一个具体的文件。

## 路径按几个根去试，不是只按仓库根

文档里的路径混着两种写法：`src/sandbox/api/jobs.py`（全路径）和 `api/jobs.py`
（相对 `src/sandbox/`）。两种都是这个仓库里的事实写法，所以解析时逐个根试一遍。
只按仓库根试的话，这一道闸门会一次报出三十几个"不存在"，而那全是它自己认错了
——闸门第一天就被忽略，等于没写。

## 白名单里每一条都要写理由，而且要能失效

`_ABSENT` 是"允许引一个不存在的路径"的清单。它必须有理由，且**理由本身也被
核对**：清单里某一条哪天真的存在了，这道闸门也会红——不然清单会变成一块
越积越厚的抹布，最后谁也说不清里面每一条当初是为什么写下的。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 会被扫的两类文本载体。
MD_TREES = ((ROOT, "*.md"), (ROOT / "docs", "**/*.md"), (ROOT / "agents", "**/*.md"))
PY_TREES = (ROOT / "src" / "sandbox", ROOT / "scripts", ROOT / "tests")

#: 反引号里、带斜杠、且以这些后缀结尾的，才算"在指一个文件"。
_REF = re.compile(r"`([A-Za-z0-9_./\-]+\.(?:py|md|yaml|toml))`")

#: 解析路径时依次试的根。见文件头。
ROOTS = (
    ROOT,
    ROOT / "src" / "sandbox",
    ROOT / "src",
    ROOT / "tests",
    ROOT / "scripts",
    ROOT / "docs",
    ROOT / "scenarios",
)

#: 允许引、但仓库里找不到的路径。**每一条都要写理由。**
_ABSENT = {
    "scripts/run_walking_skeleton.py": (
        "2026-10-01 删掉的旧演示脚本。README 与 `scenario/runner.py` 的文档"
        "是**故意**留着它的名字的：那几段讲的正是「它为什么没了」，把名字抹掉"
        "那句话就不成立了。"
    ),
    "_probes/probe_cost_by_tool.py": (
        "探针住在仓库外面（`E:/agent-test/_probes/`）。**这是刻意的**：探针会临时"
        "改源码再还原，放在仓库里就有被提交进去的风险。从仓库里永远找不到它，"
        "而 `docs/cost.md` 指它是对的——那是给改这段代码的人看的路标。"
    ),
    "_probes/probe_monotonic_state.py": (
        "同上，住在仓库外面。`docs/detectors.md` 第三节那几条「我量过」的说法"
        "（三态判定、两趟写法、十二个哈希种子）都由它核，指它是对的。"
    ),
    "_probes/measure_paging_shapes.py": (
        "同上，住在仓库外面。量的是「参数形状决定近似层结论」那张表，"
        "`docs/detectors.md` 第三节引它。"
    ),
    "_probes/measure_cluster_linkage.py": (
        "同上，住在仓库外面。README 的缺口表和 `docs/detectors.md` 都写着"
        "「链式聚类」这个出口**量过了、不通**，量的就是它。"
    ),
    "_probes/measure_rejected_calls.py": (
        "同上，住在仓库外面。量的是「被拒的调用各检测器拿它当什么」——"
        "`docs/detectors.md` 第一节那条「三次参数写错会拿到和三次真扣款一字不差"
        "的报告」就是它量出来的。"
    ),
    "_probes/measure_idempotency_status.py": (
        "同上，住在仓库外面。量的是「幂等检测器每一句话跟调用成功没成功有没有"
        "关系」（四支 × 三种结果），`docs/detectors.md` 第四节引它。"
    ),
    "notes/a.md": (
        "`docs/detectors.md` 举的例子：讲「目录判据的粒度是整块」时，举的是**记忆"
        "目录里**的布局（`notes/` 下逐条声明各写一行），不是仓库里的文件。"
    ),
}

#: 闸门自己也要有相机：把扫描规模的下限钉住。
#: 2026-10-01 量到的是 126 个文件、200 条能解析的引用。写下限而不是写等号：
#: 加文件不该让这条红，而**扫描塌成 0 个文件**必须红。
_MIN_FILES = 100
_MIN_RESOLVED = 150


def _md_files() -> list[Path]:
    found: set[Path] = set()
    for tree, pattern in MD_TREES:
        for path in sorted(tree.glob(pattern)):
            if ".venv" in path.parts or ".agent-memory" in path.parts:
                continue
            found.add(path)
    return sorted(found)


def _py_docstrings() -> list[tuple[Path, str]]:
    """每个源码文件的模块/类/函数文档字符串。

    只扫文档字符串，不扫普通字符串与注释：**能进报告的字面量另有闸门**
    （`test_report_text.py`），而注释里写路径是给自己人看的，宽容一点没坏处。
    """
    out: list[tuple[Path, str]] = []
    for tree in PY_TREES:
        for path in sorted(tree.rglob("*.py")):
            tree_ast = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree_ast):
                if not isinstance(
                    node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    continue
                doc = ast.get_docstring(node)
                if doc:
                    out.append((path, doc))
    return out


def _resolve(target: str) -> bool:
    return any((root / target).exists() for root in ROOTS)


def _references() -> dict[str, set[str]]:
    """`路径 → 引它的那些文件`，只收解析不到的。"""
    missing: dict[str, set[str]] = {}
    carriers: list[tuple[Path, str]] = [
        (path, path.read_text(encoding="utf-8")) for path in _md_files()
    ]
    carriers.extend(_py_docstrings())
    for path, text in carriers:
        where = str(path.relative_to(ROOT))
        for match in _REF.finditer(text):
            target = match.group(1)
            # 裸文件名到处都是泛指，不收。见文件头。
            if "/" not in target or _resolve(target):
                continue
            missing.setdefault(target, set()).add(where)
    return missing


def _scanned() -> tuple[int, int]:
    """(扫了几个文件, 解析到了几条引用)。给相机用。"""
    files = _md_files()
    refs = 0
    for path in files:
        refs += sum(
            1
            for m in _REF.finditer(path.read_text(encoding="utf-8"))
            if "/" in m.group(1) and _resolve(m.group(1))
        )
    for path, doc in _py_docstrings():
        files.append(path)
        refs += sum(1 for m in _REF.finditer(doc) if "/" in m.group(1) and _resolve(m.group(1)))
    return len(set(files)), refs


def test_paths_named_in_docs_exist() -> None:
    """文档与文档字符串里点名的文件，仓库里找得到——除非它在白名单上。"""
    missing = _references()
    unlisted = {t: w for t, w in missing.items() if t not in _ABSENT}
    detail = "\n  ".join(
        f"{target}（{', '.join(sorted(where))}）" for target, where in sorted(unlisted.items())
    )
    assert not unlisted, "这些路径在文档里被点名，仓库里却没有（也不在白名单上）：\n  " + detail


def test_the_allowlist_has_not_rotted() -> None:
    """白名单里每一条都要**仍然**不存在。

    它哪天真的被加回来了，这条会红：那时要么把白名单那一条删掉，要么改掉
    引用它的那段话——但**不能什么都不做**，因为一条已经不成立的豁免会让人
    以为那个路径还是坏的。
    """
    missing = _references()
    revived = sorted(t for t in _ABSENT if t not in missing)
    assert not revived, (
        "白名单里这些路径现在能找到文件了，那一条豁免该删了：\n  " + "\n  ".join(revived)
    )


def test_the_scan_is_not_vacuous() -> None:
    """扫描规模的下限。

    找一个不存在的路径只需要上面两条；但**如果扫描本身塌了**（根写错、正则退化成
    什么都匹配不上、glob 匹配不到文件），上面两条会一起变绿——正是这个项目最怕的
    那种"看起来一切正常"。所以把量到的规模钉在下限上。
    """
    files, refs = _scanned()
    assert files >= _MIN_FILES, f"只扫到 {files} 个文件（下限 {_MIN_FILES}），扫描八成塌了"
    assert refs >= _MIN_RESOLVED, (
        f"只解析到 {refs} 条引用（下限 {_MIN_RESOLVED}），正则或根八成退化了"
    )
