"""单会话超时：它得真的生效，而且出了岔子时报告里得看得见 Agent 说了什么。

**这个特性从来没有生效过。** `--timeout` 这个命令行参数一直都在、界面上的
"超时强杀"徽章也一直都在，而那个值从来没有到达它该到的地方：

    await asyncio.to_thread(proc.communicate, self.session_timeout_s)

`Popen.communicate` 的签名是 `communicate(input=None, timeout=None)`——第一个位置
参数是 `input`，不是 `timeout`。而这里的 stdin 根本不是管道，于是传进去的那个数
被**静默丢掉**，`communicate` 退化成一句没有超时的 `wait()`。不报错、不留痕，
只是 Agent 挂住多久，框架就等多久。这一整条分支（`timed_out`、`[超时强杀]`、
界面上的徽章）此前是**死代码**：没有任何一次运行走到过它。

这个文件的第二件事是配套的：`stdout_tail` 抓了 2000 字符却**从没给任何人看过**
——命令行不印、JSON 里没有、界面上没有。于是 Agent 崩了只能看到"退出码 1"，
而那段能解释原因的 traceback 被存进对象然后扔掉。超时那条路尤其需要它：一个挂住的
Agent 最后打印的那一行往往就是唯一的线索。

分三层：

  1. **等待本身**：超时真的会放弃，正常退出真的会等，两种情况各自留下什么；
  2. **真的接上了**：`ScenarioRunner` 自己的那个超时值真的传到了进程上；
  3. **报告那一侧**：超时和崩溃的会话，输出得摆到读报告的人眼前。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sandbox.config import REPO_ROOT
from sandbox.scenario.runner import (
    _STDOUT_TAIL_CHARS,
    TIMEOUT_MARK,
    SessionOutcome,
    _wait_for_exit,
)
from sandbox.scenario.spec import AgentSpec, Scenario, SessionSpec, ToolDecl

AGENT_PYTHON = REPO_ROOT / "agents" / "react_demo" / ".venv"
AGENT_DIR = "react_demo"


def _spawn(body: str) -> subprocess.Popen[str]:
    """起一个真的子进程。和被测 Agent 一样：有 stdout 管道、**没有** stdin 管道。

    `-u` 是必须的，也是这里唯一的"作弊"：管道是块缓冲的，不 unflush 的话，
    一个被强杀的进程连一个字都留不下来——那样测的就不是"我们留不留输出"，
    而是"Python 缓不缓冲"。实测过那条差别，见下面 `test_..._mark_is_the_last_thing`。
    """
    return subprocess.Popen(
        [sys.executable, "-u", "-c", body],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )


# --- 第一层：等待本身 ---------------------------------------------------------


@pytest.mark.asyncio
async def test_a_process_that_will_not_exit_is_given_up_on() -> None:
    """**这个文件存在的理由。** 挂住 30 秒的进程，配 0.5 秒的超时，必须放得下。

    修之前这里会等满 30 秒然后报 `timed_out=False`——而"等满 30 秒"在测试里只能
    靠超时断言表达，所以下面两个断言缺一不可：`timed_out` 是结论，耗时是**证据**
    （只断言结论的话，一个"立刻返回 True"的坏实现也能过）。
    """
    proc = _spawn("import time; time.sleep(30)")
    started = time.perf_counter()

    timed_out, _ = await _wait_for_exit(proc, 0.5)
    elapsed = time.perf_counter() - started

    assert timed_out is True
    assert elapsed < 10, f"说是超时了，却等了 {elapsed:.1f} 秒——那个超时没生效"
    assert proc.poll() is not None, "报了超时却没把进程杀掉，它还在烧钱"


@pytest.mark.asyncio
async def test_a_process_that_exits_normally_is_waited_for_and_keeps_its_output() -> None:
    """反面：正常的会话不能被当成超时，而且它打印的东西要原样留下。

    少了这一条，"永远返回超时"的实现也能让上面那条通过——而那会让每一次会话
    都被报成"超时强杀"。
    """
    proc = _spawn("print('订单 A1 已支付')")

    timed_out, stdout = await _wait_for_exit(proc, 30.0)

    assert timed_out is False
    assert "订单 A1 已支付" in stdout


@pytest.mark.asyncio
async def test_the_timeout_mark_is_the_last_thing_in_the_output() -> None:
    """超时说明写在**末尾**：报告只留最后 2000 字符，盖在前面会被截掉。

    截掉之后，一个被强杀的会话读起来和一次正常结束**一模一样**——而"这一次不是
    正常结束"正是那句话唯一的用途。

    这一条同时钉住一个**做不到的事**：输出是管道、块缓冲的，Agent 没 flush 就被
    杀掉的话，它最后打印的东西就真的没了。所以这里显式 `-u`（见 `_spawn`）——
    测的是"拿到了就留得住"，不是"一定拿得到"。报告那边因此不能承诺"这里有它的
    最后遗言"，只能说"有就摆出来"。
    """
    proc = _spawn("import time; print('它卡在这一行'); time.sleep(30)")

    timed_out, stdout = await _wait_for_exit(proc, 0.5)

    assert timed_out is True
    assert stdout.rstrip().endswith(TIMEOUT_MARK)
    assert "它卡在这一行" in stdout, "挂住之前打印的那一行没了——那是唯一的线索"


@pytest.mark.asyncio
async def test_the_mark_survives_a_long_output() -> None:
    """输出长到超过上限时，那句话还得在——这正是"写在末尾"要防的那件事。"""
    proc = _spawn("import time; print('x' * 5000); time.sleep(30)")

    timed_out, stdout = await _wait_for_exit(proc, 0.5)
    kept = stdout[-_STDOUT_TAIL_CHARS:]

    assert timed_out is True
    assert TIMEOUT_MARK in kept, "截断之后那句话没了，这一节读起来像正常结束"


# --- 第二层：runner 自己的那个超时值真的传到了 ---------------------------------


def _scenario(entry: Path) -> Scenario:
    """一个只为拉起进程用的最小场景：跑哪个脚本由 `entry` 说了算。"""
    return Scenario(
        id="timeout",
        description="超时用",
        agent=AgentSpec(version="react_demo@0.1", task="随便", entry=str(entry)),
        tools=[ToolDecl(name="get_order", description="查订单", is_write=False)],
        sessions=[SessionSpec(name="会话")],
        expected_charges=0,
    )


async def _launch(tmp_path: Path, body: str, *, timeout_s: float):
    from sandbox.scenario.runner import ScenarioRunner

    if not AGENT_PYTHON.exists():
        pytest.skip(f"被测 Agent 的 venv 不在（{AGENT_PYTHON}），跳过超时端到端")

    entry = tmp_path / "agent_under_test.py"
    entry.write_text(body, encoding="utf-8")
    # 借演示 Agent 的解释器：这里要验的是**超时那条路**，不是 Agent 本身，
    # 而解释器只要是个能跑 Python 的真实独立进程就够了。
    runner = ScenarioRunner(session_timeout_s=timeout_s)
    # 传 `.agent` 而不是整个场景：串联之后这个参数是"**这一段**由谁跑"，
    # 收窄成 AgentSpec 就是为了让"每一段都拉起 0 号"写不出来
    # （见 `runner._launch_agent` 的 docstring）。
    return await runner._launch_agent(
        _scenario(entry).agent, "sk-sandbox-x", 1, 2, tmp_path, {"proc": None}, "随便"
    )


@pytest.mark.asyncio
async def test_the_runners_own_timeout_is_the_one_that_applies(tmp_path: Path) -> None:
    """`ScenarioRunner(session_timeout_s=...)` 那个数必须真的到进程上。

    上一条测的是"等待函数讲不讲道理"，这一条测的是**它有没有被用上**——
    那个 bug 恰恰长在这两者中间的一行调用上。
    """
    started = time.perf_counter()
    # flush=True：管道是块缓冲的，不 flush 的话被强杀时一个字都留不下来。
    # 引擎**不会**替 Agent 加 `-u`——那是改它进程里的行为，ADR 001 只允许换
    # base_url 和 api_key。所以这里测的是"Agent 吐出来了，我们就留得住"。
    exit_code, timed_out, stdout, elapsed = await _launch(
        tmp_path, "import time; print('卡住了', flush=True); time.sleep(60)", timeout_s=0.6
    )

    assert timed_out is True, "runner 自己的超时值没传到进程上"
    assert elapsed < 20, f"等了 {elapsed:.1f} 秒"
    assert time.perf_counter() - started < 40
    assert exit_code is not None, "强杀之后要有一个退出码，不能是 None"
    assert "卡住了" in stdout, "挂住之前打印的那一行没留下——那是唯一的线索"


@pytest.mark.asyncio
async def test_a_quick_agent_is_not_reported_as_timed_out(tmp_path: Path) -> None:
    """反面：跑得完的 Agent 不许被报成超时，它的输出要能读到。"""
    _, timed_out, stdout, elapsed = await _launch(
        tmp_path, "print('我跑完了')", timeout_s=30.0
    )

    assert timed_out is False
    assert elapsed < 30
    assert "我跑完了" in stdout


# --- 第三层：报告那一侧 -------------------------------------------------------


def _outcome(**kwargs) -> SessionOutcome:
    base = {
        "name": "会话",
        "ordinal": 0,
        # 这些用例测的是超时和输出留不留，跟"谁跑的"无关——但它是必填的：
        # 少写一次，串联场景里 1 号的会话就会被读成 0 号的（`runner.py:785`）。
        "agent_index": 0,
        "memory_horizon_ordinal": -1,
        "restored_from": None,
        "exit_code": 0,
        "elapsed_s": 1.0,
        "timed_out": False,
        "script_steps_declared": 3,
        "script_steps_consumed": 3,
        "tool_calls": 3,
    }
    return SessionOutcome(**{**base, **kwargs})


@pytest.mark.parametrize(
    ("how", "kwargs"),
    [
        ("被强杀", {"timed_out": True, "exit_code": -9, "script_steps_consumed": 1}),
        ("崩了", {"exit_code": 1, "script_steps_consumed": 1}),
        ("跑了一半就退出", {"script_steps_consumed": 1}),
        ("被熔断", {"killed": True, "script_steps_consumed": 1}),
    ],
)
def test_a_session_that_went_wrong_shows_its_output(how: str, kwargs: dict) -> None:
    """四种出岔子的方式，每一种都要把输出摆出来。

    参数化是刻意的：这四种在代码里是四个**不同**的字段，漏掉哪一个都会表现成
    "这一半的报告里没有输出"——而那正是这个文件在修的那个毛病。
    """
    s = _outcome(stdout_tail="它最后说了这一句", **kwargs)

    assert not s.ended_normally, f"{how}的会话被当成正常结束了"
    assert _rendered(s).count("它最后说了这一句") == 1


def test_a_clean_session_does_not_dump_its_output() -> None:
    """正常结束的会话不印输出：那几千字会把报告淹掉，而它什么都没解释。

    数据本身照旧入库（`to_dict` 里有），筛的是**展示**——清空记录是静默省略，
    读 JSON 的人分不出"没打印"和"我们没放"。
    """
    s = _outcome(stdout_tail="一长串正常的对话")

    assert s.ended_normally
    assert "一长串正常的对话" not in _rendered(s)
    assert s.to_dict()["stdout_tail"] == "一长串正常的对话"


def test_a_timed_out_session_says_so_on_its_own_line() -> None:
    """命令行的会话行上要有 [超时强杀]——它比输出更快被看到。"""
    s = _outcome(timed_out=True, exit_code=-9, script_steps_consumed=1)

    assert "[超时强杀]" in _rendered(s)


def test_the_output_header_says_how_much_was_kept() -> None:
    """抬头写清这是末尾一段。不写的话，"它一共就打印了这些"是读出来的意思。"""
    s = _outcome(exit_code=1, script_steps_consumed=1, stdout_tail="boom")

    assert f"末尾 {_STDOUT_TAIL_CHARS} 字符" in _rendered(s)


def _rendered(s: SessionOutcome) -> str:
    """把这一段会话放进一份最小的报告里渲染出来。

    只驱动 `render`：要看的就是**报告上印了什么**，不是结论对不对。`ok` 是算出来的
    属性，不给它传值。
    """
    from sandbox.scenario.runner import ScenarioResult, render

    return render(ScenarioResult(run_id="r", scenario_id="s", sessions=[s]))
