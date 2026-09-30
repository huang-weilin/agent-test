"""起 Web 界面：`scripts/serve_ui.py [--port 8765]`

和 `run_scenario.py` 一样，这个脚本只做一件事——把仓库根塞进 sys.path，
然后把参数交给 `sandbox.api.app`。逻辑全在 src 里，因为它是被测代码的一部分。

只绑 127.0.0.1，没有远程开关。要远程访问用端口转发：

    ssh -L 8765:127.0.0.1:8765 <主机>

这个服务没有认证，而它能写场景文件、能拉起 Agent 子进程——绑到局域网上等于
把这些开放给所有人。端口转发的路上至少有认证。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sandbox.api.app import DEFAULT_HOST, DEFAULT_PORT, serve  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI Agent 行为级测试沙盒 · Web 界面")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="只接受回环地址。要远程访问请用端口转发，不要绑到 0.0.0.0",
    )
    args = parser.parse_args(argv)
    serve(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
