# -*- coding: utf-8 -*-
"""首次安装：建虚拟环境、装框架依赖。

`start_ui.bat` 双击时发现没有 `.venv`，会提示先跑这个脚本（或直接双击
`install.bat`）。它做三件事：

1. 找一个能用的系统 Python（>= 3.12），用它建 `.venv`；
2. 在 `.venv` 里装框架本体（`pip install -e .`）；
3. 打印下一步：怎么启动界面、怎么跑第一个测试。

**这个脚本只装框架，不装内置示例 Agent 的依赖**——示例 Agent 有自己独立的
venv（`agents/*/.venv`，刻意和框架隔离，见 `agents/order_agent/README.md`）。
想跑示例 Agent 时，按那份 README 装它自己的依赖即可。
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = "python"
MIN_MAJOR, MIN_MINOR = 3, 12


def _say(text: str) -> None:
    # Windows 控制台可能是 GBK；宁可退化成问号也不要 print 抛异常。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    print(text)


def _find_python() -> str | None:
    """找一个能用的 Python：>= 3.12。优先 `py` 启动器，其次 PATH 里的 python。"""
    candidates = []
    py_launcher = shutil.which("py")
    if py_launcher:
        candidates.append((py_launcher, ["-3.12"]))
        candidates.append((py_launcher, ["-3"]))
    if shutil.which("python"):
        candidates.append(("python", []))
    # 常见安装目录探测：自动安装刚结束时 PATH 可能还没刷新，按路径找更可靠。
    for pattern in (
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Python", "Python3*", "python.exe"),
        "C:\\Python3*\\python.exe",
        os.path.join(os.environ.get("ProgramFiles", ""), "Python3*", "python.exe"),
    ):
        for found in sorted(glob.glob(pattern), reverse=True):
            candidates.append((found, []))
    for exe, args in candidates:
        try:
            proc = subprocess.run(
                [exe, *args, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode != 0:
            continue
        try:
            major, minor = (int(x) for x in proc.stdout.strip().split(".")[:2])
        except ValueError:
            continue
        if (major, minor) >= (MIN_MAJOR, MIN_MINOR):
            return exe if exe == "python" else f"{exe} {' '.join(args)}".strip()
    return None


def _venv_python() -> Path:
    return REPO / ".venv" / "Scripts" / "python.exe"


def main() -> int:
    _say("=" * 60)
    _say("Agent 测试沙盒 · 首次安装")
    _say("=" * 60)

    venv_py = _venv_python()
    if venv_py.exists():
        _say(f"已经装好了（{venv_py}）。")
        _say("直接双击 start_ui.bat 启动界面。")
        return 0

    system_py = _find_python()
    if not system_py:
        _say("没找到 Python。请先安装 Python 3.12 或更新版本：")
        _say("  https://www.python.org/downloads/")
        _say("装完（勾选“Add to PATH”）再双击 install.bat。")
        return 1

    _say(f"用 {system_py} 创建虚拟环境 .venv …")
    proc = subprocess.run([*system_py.split(), "-m", "venv", str(REPO / ".venv")])
    if proc.returncode != 0:
        _say("创建 .venv 失败。看上面的报错，修好后重新双击 install.bat。")
        return 1

    _say("安装框架依赖（首次要下载，耐心等）…")
    proc = subprocess.run([str(venv_py), "-m", "pip", "install", "--upgrade", "pip"])
    if proc.returncode != 0:
        _say("升级 pip 失败，继续尝试装依赖…")
    proc = subprocess.run([str(venv_py), "-m", "pip", "install", "-e", str(REPO)])
    if proc.returncode != 0:
        _say("装依赖失败。看上面的报错；常见原因是网络不通，可重试。")
        return 1

    _say("")
    _say("安装完成。接下来：")
    _say("  1. 双击 start_ui.bat 启动界面（浏览器会自动打开 http://127.0.0.1:8765/）")
    _say("  2. 打开「使用手册」看怎么建第一个测试")
    _say("  3. 想跑仓库内置的示例 Agent，按 agents/<名字>/README.md 装它自己的依赖")
    _say("")
    _say("配置界面里「连接设置」的模型密钥后，就能用一句话建测试。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
