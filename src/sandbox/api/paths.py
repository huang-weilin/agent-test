"""场景文件的定位与写回。

**这是整个 Web 层唯一能碰磁盘的地方，所以路径检查集中在这里，不散在路由里。**

一个能写 YAML 的 HTTP 服务天然带一个危险：从 URL 里取来的字符串会被拼进文件
路径，于是"改一个场景"就变成了"往任意位置写文件"。这类漏洞的经典成因是把
危险字符挑出来禁掉（`../`、`..\\`、绝对路径……），而黑名单永远漏一个。

所以这里的做法是**白名单**：场景标识只允许 `[a-z0-9_-]`，别的字符根本进不来，
路径拼接处就没有可乘之机。加一道包含性检查是第二层——万一哪天有人放宽了
正则（比如想支持子目录分类），越界仍然会被拦住。
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from sandbox.config import REPO_ROOT

# 严格白名单。第一个字符限定为字母或数字，顺带挡掉 `.` 开头这种隐藏文件。
_ID_RE = re.compile(r"\A[a-z0-9][a-z0-9_-]{0,63}\Z")

# 场景文件的扩展名。只认 `.yaml`：允许 `.yml` 的话同一个场景会有两个合法
# 文件名，而"这个场景在哪儿"是编辑器和运行入口共用的一个判断，不该有歧义。
SUFFIX = ".yaml"


class ScenarioPathError(ValueError):
    """场景标识或路径不合法。消息要能直接给用户看。"""


def scenarios_dir() -> Path:
    """场景目录。Web 界面与 `scripts/run_scenario.py` 看的是同一个目录。"""
    return REPO_ROOT / "scenarios"


def is_valid_id(scenario_id: str) -> bool:
    return bool(_ID_RE.match(scenario_id))


def check_scenario_id(scenario_id: str) -> str:
    """校验场景标识，返回它本身。

    拒绝时把规则说出来——用户看到的是"为什么不让我用这个名字"，不是
    "400 Bad Request"。
    """
    if not _ID_RE.match(scenario_id):
        raise ScenarioPathError(
            f"场景标识 {scenario_id!r} 不合法。"
            f"只允许小写字母、数字、连字符和下划线，以字母或数字开头，最长 64 个字符。"
        )
    return scenario_id


def scenario_file(scenario_id: str, *, root: Path | None = None) -> Path:
    """场景标识 → 文件路径。**越界的标识在这里被拦住，调用方不需要再检查。**"""
    check_scenario_id(scenario_id)
    base = Path(root or scenarios_dir()).resolve()
    path = (base / f"{scenario_id}{SUFFIX}").resolve()
    # 第二道防线。正则已经挡掉了分隔符和 `.`，但包含性检查不依赖正则的细节；
    # `.resolve()` 会跟随符号链接，所以指向目录外的链接同样落在这里。
    if path.parent != base:
        raise ScenarioPathError(
            f"场景 {scenario_id!r} 解析到了场景目录之外（{path}），已拒绝。"
        )
    return path


def list_scenarios(*, root: Path | None = None) -> list[tuple[str, Path]]:
    """场景目录里全部 `.yaml`，按名字排序。

    **不合法的文件名也列出来**，交给界面显示成"打不开"。跳过去的话，用户
    在文件管理器里明明看得见一个 `My Scenario.yaml`，界面上却怎么都找不到，
    而没有任何东西告诉他为什么。
    """
    base = Path(root or scenarios_dir())
    if not base.is_dir():
        return []
    return [
        (p.stem, p)
        for p in sorted(base.iterdir(), key=lambda p: p.name.lower())
        if p.is_file() and p.suffix.lower() == SUFFIX
    ]


def write_scenario_text(scenario_id: str, text: str, *, root: Path | None = None) -> Path:
    """把场景文本写进文件。

    **调用方必须已经校验过这段文本能被加载**（`loader.parse_scenario`）。
    这里不做内容校验，是因为校验失败的**报错信息**要显示在编辑器里、和用户
    刚敲的内容并排，那个上下文只有路由层有。

    写入是原子的：先写同目录下的临时文件再 `os.replace`。直接
    `write_text` 的话，进程在写到一半时被杀（或被 Ctrl-C）会留下一个截断的
    YAML——而它看起来是个正常文件，只是加载不出来了。
    """
    path = scenario_file(scenario_id, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 临时文件必须和目标是同一个目录：`os.replace` 的原子性只在同一文件系统
    # 内成立，跨盘会退化成"先删后拷"。
    handle, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{scenario_id}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        # 失败时别把临时文件留在场景目录里——它以 `.` 开头，不会被列出，
        # 于是会永远躺在那儿没人知道。
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path


def read_scenario_text(scenario_id: str, *, root: Path | None = None) -> str:
    path = scenario_file(scenario_id, root=root)
    if not path.exists():
        raise ScenarioPathError(f"场景 {scenario_id!r} 不存在（{path}）")
    return path.read_text(encoding="utf-8")
