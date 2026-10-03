"""双击 `start_ui.bat` 之后真正跑起来的那一段。

批处理文件本身只做一件事：找到这个仓库自带的 Python，然后调这里。**它是纯 ASCII 的**，
不是偷懒——cmd.exe 按字节偏移一行行读批处理文件，非 ASCII 字节会让那个偏移错位。
实测一份 2KB、正文是 GBK 的 .bat（开头还写了 `chcp 65001`），从第 6 行起被切得七零八落，
报出来的是「'xx' 不是内部或外部命令」：**报错内容指错方向**，看起来像这个程序本来就是坏的。
把中文挪进 .py，这个失效模式就不存在了——这里的话由 Python 自己按控制台编码写出去。

这个脚本只做一件事：起服务。逻辑在 `sandbox.api.app` 里，因为它是被测代码的一部分。
"""

from __future__ import annotations

import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sandbox.api.app import DEFAULT_HOST, DEFAULT_PORT, serve  # noqa: E402

URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/"

# 服务第一次启动要十几秒（要 import 一大堆东西）。这个上限给得很宽：
# 宁可多等一会儿，也不要在它慢一点点的时候就宣布失败——那种失败和真的起不来
# 长得一模一样，而它会把用户支去查一个根本不存在的问题。
READY_TIMEOUT = 90.0


def _probe(url: str, timeout: float = 3.0) -> bool:
    """这个地址是不是已经在应答了。

    **问的是「能不能应答」，不是「端口通不通」**：端口能连上只说明有东西在监听，
    那时候把浏览器打开，用户看到的还是「无法连接」——一个查不出所以然的失败。
    这个区别是真的：一个只 accept、不说话的服务端，浏览器照样打不开。

    `url` 带参数是为了能被测：`tests/test_launcher.py` 拿一个"只监听不应答"的
    桩端口和一个真的 HTTP 端口分别打给它，两边都必须给出正确的那一半。
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except (urllib.error.URLError, OSError):
        return False


def _answering() -> bool:
    return _probe(URL)


def _open_browser_when_ready() -> None:
    deadline = time.monotonic() + READY_TIMEOUT
    while time.monotonic() < deadline:
        if _answering():
            webbrowser.open(URL)
            return
        time.sleep(1.0)
    print(f"\n等了 {READY_TIMEOUT:.0f} 秒界面还没起来。请把这个窗口里的内容截图给我。")


def main() -> int:
    # 非 ASCII 的输出宁可退化成问号，也不要让 print 自己抛异常——
    # 那会把「说一句话」变成「整个程序崩掉」。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    # 已经在跑就别再起第二个：同一个端口起不了两个，第二个会报错退出，
    # 而用户看到的会是「浏览器打开了但界面是坏的」。
    if _answering():
        print(f"界面已经在运行，直接打开浏览器：{URL}")
        webbrowser.open(URL)
        return 0

    print(f"正在启动界面，第一次要等十几秒……起来之后浏览器会自己打开：{URL}")
    print("关掉这个窗口，界面就停了。\n")
    threading.Thread(target=_open_browser_when_ready, daemon=True).start()
    serve(host=DEFAULT_HOST, port=DEFAULT_PORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
