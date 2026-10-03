"""双击 `start_ui.bat` 这条路上，两个不会自己出声的失效。

## 一、批处理文件里不能有非 ASCII 字节（这是量出来的，不是推理）

cmd.exe 按**字节偏移**一行行读 .bat。文件里有非 ASCII 字节时那个偏移会漂：
2026-10-03 实测一份 2KB、正文是 GBK 的 .bat（开头还写着 `chcp 65001`），从第 6 行起
被切得七零八落，报出来的是

    'xx' 不是内部或外部命令，也不是可运行的程序或批处理文件。

**报错内容指错方向**——它看起来像"这个程序本来就是坏的"，而文件内容一个字都没错，
只是读它的人把位置算错了。所以 `start_ui.bat` 是纯 ASCII 的，中文全在
`scripts/start_ui.py` 里（那里由 Python 自己控制编码）。这道闸门钉的就是这一点：
**往那个 .bat 里加一个汉字，这条会红。**

当时排查的代价记在这儿：第一版 .bat 是 UTF-8 + `chcp 65001`。一个 120 字节的探针
（四行 echo）跑得好好的，于是"UTF-8 也能行"看上去像一个结论了；换成 2KB 的真文件才炸。
**小样本上的成功不是结论**，它只是还没越过 cmd 那个 512 字节的缓冲区边界。

## 二、"起来了"判的是能不能应答，不是端口通不通

一个只 accept、不说话的服务端，端口是通的，浏览器照样打不开。所以
`scripts/start_ui.py` 的探测发的是一次真的 HTTP 请求。

这一条**必须有正反两个方向**（同一条规矩见 `tests/control_pairs.py`）：
只测"不应答的端口要判 False"的话，一个永远返回 False 的探测也能过——
而那个探测的后果是界面永远打不开。
"""

from __future__ import annotations

import importlib.util
import re
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
BAT = ROOT / "start_ui.bat"
START_PY = ROOT / "scripts" / "start_ui.py"


def _load_start_ui() -> ModuleType:
    """按路径加载 `scripts/start_ui.py`。

    它不是一个包里的模块（`scripts/` 没有 `__init__.py`），只能这么装。
    加载会连带 import 整个 app——那是这个入口自己的行为，不绕开它才对。
    """
    spec = importlib.util.spec_from_file_location("_launcher_start_ui", START_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bat_text() -> str:
    """`start_ui.bat` 的文本，**遇到坏字节也不抛异常**。

    刻意不用 `read_text(encoding="ascii")`：那样一来，谁往 .bat 里加了一个汉字，
    除了编码那条闸门之外，另外三条也会各自抛一个 UnicodeDecodeError——
    **四条红指着四个方向，而根因只有一个**。这是这个项目一直在修的那类错
    （报错内容指错方向）。所以这里退化成乱码字符：只有
    `test_the_batch_file_is_pure_ascii` 该为编码问题负责，其余几条照常判它们自己的事。
    同一帖药见 `tests/test_memory_snapshot.py` 起 `mklink` 时带的 `errors="replace"`。
    """
    return BAT.read_bytes().decode("ascii", errors="replace")


def test_the_batch_file_is_pure_ascii() -> None:
    """`start_ui.bat` 里一个非 ASCII 字节都不许有。见文件头第一节。"""
    raw = BAT.read_bytes()
    offending = [i for i, byte in enumerate(raw) if byte > 0x7F]
    if offending:
        first = offending[0]
        line_no = raw[:first].count(b"\n") + 1
        snippet = raw.split(b"\n")[line_no - 1][:60]
        raise AssertionError(
            f"start_ui.bat 第 {line_no} 行有 {len(offending)} 个非 ASCII 字节（第一处在 "
            f"偏移 {first}）：{snippet!r}\n"
            f"cmd.exe 按字节偏移读 .bat，这些字节会让它从某一行起把文件读错位，"
            f"报出来的是「'xx' 不是内部或外部命令」——看起来像程序坏了。\n"
            f"中文请写进 scripts/start_ui.py。"
        )


def test_the_camera_sees_a_real_batch_file() -> None:
    """相机：上面那条在**空文件**上也会绿。

    一个被清空、或者被换成别的东西的 .bat 没有非 ASCII 字节，于是那道闸门一声不吭。
    所以这里钉住"它还是个真文件"：有内容，而且确实在调那个 Python 入口。
    """
    text = _bat_text()
    assert len(text) > 200, f"start_ui.bat 只有 {len(text)} 字节，像是被清空或截断了"
    assert "start_ui.py" in text, "start_ui.bat 没有在调 scripts/start_ui.py"


def test_the_scripts_the_batch_file_calls_exist() -> None:
    """那个 .bat 点名的 .py，仓库里要真的在。

    这道是防腐烂：`scripts/start_ui.py` 哪天改了名字，双击之后 cmd 会报
    「系统找不到指定的文件」——而那句话同样指不到根因上。
    """
    # `%~dp0` 是批处理里"这个 .bat 自己所在的目录"，**它就是仓库根**。
    # 不先把它抹掉的话，正则会把 `dp0scripts/start_ui.py` 当成路径——
    # 第一版就是这么写的，于是这道闸门报了一个并不存在的文件（它自己抓到的）。
    text = _bat_text().replace("%~dp0", "")
    referenced = sorted({m.group(0).replace("\\", "/") for m in re.finditer(r"[\w\\/]+\.py", text)})
    assert referenced, "start_ui.bat 里一个 .py 都没点名，这条闸门失去意义了"
    missing = [p for p in referenced if not (ROOT / p).exists()]
    assert not missing, f"start_ui.bat 点名了这些文件，仓库里却没有：{missing}"


def test_the_port_has_one_source_of_truth() -> None:
    """端口只由 `sandbox.api.app.DEFAULT_PORT` 决定，不写进那个 .bat。

    写死一份的话，哪天默认端口改了，双击开出来的还是老端口——而两边的报告都不出声。
    """
    pinned = re.findall(r"\b\d{4,5}\b", _bat_text())
    assert not pinned, (
        f"start_ui.bat 里写死了数字 {pinned}；端口应该由 scripts/start_ui.py 从 "
        f"DEFAULT_PORT 取，不然它会和 src/sandbox/api/app.py 里的默认值各走各的。"
    )


def test_nothing_listening_is_not_ready() -> None:
    """没人监听的端口：不算起来了。这正是服务启动中那十几秒的状态。"""
    module = _load_start_ui()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    # 端口已经关掉了，此刻没人监听它。
    assert module._probe(f"http://127.0.0.1:{port}/", timeout=0.5) is False


def test_a_port_that_only_listens_is_not_ready() -> None:
    """只 accept、不说话的服务端：**不算起来了**。

    这就是"端口通"和"能应答"的区别本身。少了这一条，把探测写成
    `socket.connect_ex` 也能过——而那样浏览器打开的还是「无法连接」。
    """
    module = _load_start_ui()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = int(listener.getsockname()[1])
    try:
        assert module._probe(f"http://127.0.0.1:{port}/", timeout=0.5) is False
    finally:
        listener.close()


def test_a_port_serving_http_is_ready() -> None:
    """真的在应答 HTTP 的端口：算起来了。**这是上面那条的对照。**

    没有它，一个永远返回 False 的探测也能让"只监听不算数"那条过绿。
    """

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # 这个名字是 stdlib 定的，`select` 里没开 pep8-naming
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args: object) -> None:
            """测试里不要把访问日志写到 stderr。"""

    module = _load_start_ui()
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = int(server.server_address[1])
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert module._probe(f"http://127.0.0.1:{port}/", timeout=3) is True
    finally:
        server.shutdown()
        server.server_close()
