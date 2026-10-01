"""mypy 真的在跑吗，跑的是不是整棵树，以及它会不会被摁住。

**为什么这条绊线值得存在。** `pyproject.toml` 里配着 `strict = true`，从第一次
提交（`4b290d6`）起就配在那儿，而它一次都没跑起来过：`packages = ["sandbox"]`
让 mypy 去 site-packages 里找
"已安装的那个包"，而包里没有 `py.typed`，于是它一条都不检查、打一句提示就退出。
没有任何东西会因此变红——**那行配置是件装饰品**，而它看起来像"我们有类型检查"。

真正跑起来之后，它当场抓到几条单测抓不到的东西。最值钱的一条：
`MemoryAdapter` 协议上写着 `restore(payload)`，而唯一的实现和唯一的调用点
（`trace/recorder.py` 的快照恢复）都在传 `allow_create`。照着协议写第二个适配器
（LangGraph checkpointer）的人，会在**回滚记忆的那一刻**炸一个 `TypeError`。
这类"某处写着的话和代码对不上"的错误，喂构造好输入的单测永远碰不到。

所以这条测试盯三件事，每件对应一种"看起来在检查、其实没检查"：

1. 跑得起来，而且零错误；
2. 检查的是**整棵源码树**（不是被谁顺手缩到一个子目录，也不是反过来把没写的
   文件当成检查过了——所以数量是拿文件数现算的，不是写死的）；
3. 项目自己的配置没有把错误摁掉（往临时目录放一个**故意写错**的文件，用同一份
   配置去检查它，必须红）。这一条防的是 `ignore_errors = true` 那种"全绿是因为
   全哑"的改法——它能让上面两条一起通过。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src" / "sandbox"
CONFIG_PATH = REPO_ROOT / "pyproject.toml"

_FULL_RUN: subprocess.CompletedProcess[str] | None = None


def _mypy(*args: str) -> subprocess.CompletedProcess[str]:
    """在仓库根目录跑 mypy。带上 PYTHONIOENCODING，否则这台机器上会炸编码。

    整棵树的检查（不带参数的调用）会缓存一次：这一条测试跑三遍没有意义，
    而这个套件里已经有两个端到端用例在这台机器上各要二十秒。
    """
    if not args and _FULL_RUN is not None:
        return _FULL_RUN
    return _run(*args)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mypy", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=False,
    )


def _fail_helpfully(proc: subprocess.CompletedProcess[str]) -> None:
    if "No module named mypy" in proc.stderr:
        raise AssertionError(
            "这个解释器里没有 mypy。装开发依赖：pip install -e .[dev]"
        )


def test_mypy_跑得起来_而且一条意见都没有() -> None:
    global _FULL_RUN
    proc = _mypy()
    _FULL_RUN = proc
    _fail_helpfully(proc)
    assert proc.returncode == 0, (
        "mypy 报错了。它检查的是 src/sandbox 下的每一个文件，"
        "所以红的时候先看它自己打出来的那几行：\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


def test_mypy_看的是整棵源码树() -> None:
    """数量是现算的，不是写死的——写死了就只是"当年是 59 个文件"。

    往 `src/sandbox` 加一个文件而它没被检查到（配置被缩窄成某个子目录、
    或者新目录没被 glob 进去），红的就是这一条。
    """
    proc = _mypy()
    _fail_helpfully(proc)
    on_disk = len(list(SOURCE_ROOT.rglob("*.py")))
    # 两种汇总行都要认：绿的时候是 "Success: no issues found in 59 source files"，
    # 红的时候是 "Found 2 errors in 2 files (checked 59 source files)"。只认前一种
    # 的话，"mypy 有意见"会顺带把这一条也弄红，而它红的原因是「认不出汇总行」
    # ——两种完全不同的错压成同一个红，读的人会跑去查检查面。（实测：把协议上
    # 的 `allow_create` 拿掉，这一条就跟着那条一起红了。）
    matched = re.search(r"(\d+) source files", proc.stdout)
    assert matched is not None, f"认不出 mypy 的汇总行：{proc.stdout!r}"
    assert int(matched.group(1)) == on_disk, (
        f"mypy 只检查了 {matched.group(1)} 个文件，而 src/sandbox 下有 {on_disk} 个。"
        f"检查面被缩小了——缩小的那一半从此没人管，而它看起来还是绿的。"
    )


def test_mypy_的配置没有把错误摁掉(tmp_path: Path) -> None:
    """绊线的绊线：往临时目录放一个**故意写错**的文件，用同一份配置检查它。

    这一条防的是把 `ignore_errors = true` 之类的开关打开——那种改法能让
    上面两条一起变绿，而它们绿的理由就完全反了。
    """
    probe = tmp_path / "probe.py"
    probe.write_text(
        "def f(x: int) -> int:\n"
        "    return x\n"
        "\n"
        'y: int = f("这不是一个整数")\n',
        encoding="utf-8",
        newline="\n",
    )
    # `--no-incremental` 是必须的：mypy 的增量缓存按模块名存诊断，而这个探针
    # 每次跑都叫 probe.py，上一次的报错会带着**上一次那个路径**被原样端出来
    # （实测：断言里比的是本次的临时目录，输出里印的是上一轮的）。一条测试
    # 读到的若是上一轮的话，它证明的东西就没了。
    proc = _mypy("--no-incremental", "--config-file", str(CONFIG_PATH), str(probe))
    _fail_helpfully(proc)
    assert proc.returncode != 0, (
        "一个类型明显对不上的文件被放行了。mypy 的配置里多半开了"
        "ignore_errors 之类的开关——那会让整套检查全绿，而全绿的理由是"
        f"全哑。它的输出：\n{proc.stdout}"
    )
    assert "[arg-type]" in proc.stdout and str(probe) in proc.stdout, (
        f"报的错不是那一处故意的类型不符：{proc.stdout!r}"
    )


def test_配置仍然指着源码树() -> None:
    """`packages` 那个写法本身要留着理由，别改回去。

    它当年就是这里出的错：`packages = ["sandbox"]` 让 mypy 把 `sandbox` 当成
    "一个装好的第三方包"，去 site-packages 里找一个带 `py.typed` 的版本来检查。
    本仓库是 src 布局 + 可编辑安装，包里没有那个标记，于是 mypy 直接拒绝检查。
    """
    config = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    mypy_config = config["tool"]["mypy"]
    assert "packages" not in mypy_config, (
        "`packages = [...]` 会让 mypy 去 site-packages 里找「已安装的那个包」"
        "来检查，而那需要 py.typed 标记。本仓库用的是 files。"
    )
    assert mypy_config["files"] == ["src/sandbox"], (
        f"检查面被改了：{mypy_config['files']}。"
        f"改它可以，但要连着这条测试的理由一起看——它当年就是被一行配置"
        f"悄悄关掉的。"
    )
    assert mypy_config["strict"] is True
