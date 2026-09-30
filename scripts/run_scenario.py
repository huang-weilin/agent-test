"""跑场景：`scripts/run_scenario.py [场景文件或目录 ...]`

这个脚本只做一件事——把仓库根塞进 sys.path，然后把参数交给
`sandbox.scenario.runner.main`。执行逻辑全在 src 里，因为它是被测代码的一部分，
脚本目录不是。脚本本身没法被 import，放逻辑在这里等于放进了不可测试的地方。

用法：
    .venv/Scripts/python.exe scripts/run_scenario.py --list
    .venv/Scripts/python.exe scripts/run_scenario.py scenarios/clean.yaml
    .venv/Scripts/python.exe scripts/run_scenario.py            # 跑 scenarios/ 下全部
    .venv/Scripts/python.exe scripts/run_scenario.py --json      # 机器可读报告
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sandbox.scenario.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
