"""跑一次真实的 walking skeleton：三个服务 + 一个真 Agent 进程。

这个脚本和 tests/test_e2e_walking_skeleton.py 的区别在于**真实度**：
测试用 ASGI transport 把 HTTP 短路掉了，这里起的是真的 uvicorn、真的端口、
真的 TCP，Agent 也是真的独立进程、真的独立 venv。

之所以两个都要有：测试跑得快、能进 CI；这个脚本能证明"跨进程也成立"，
而跨进程恰恰是 ADR 001 的主张——观测发生在网络边界上，被测 Agent 可以跑在
另一台机器上，只要它的 base_url 指过来。

用法：
    .venv/Scripts/python.exe scripts/run_walking_skeleton.py --scenario duplicate
    .venv/Scripts/python.exe scripts/run_walking_skeleton.py --scenario loop
    .venv/Scripts/python.exe scripts/run_walking_skeleton.py --scenario clean
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import uvicorn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sandbox.db.engine import init_db  # noqa: E402
from sandbox.detectors.base import ExemptionSet  # noqa: E402
from sandbox.detectors.idempotency import IdempotencyDetector, ToolFacts  # noqa: E402
from sandbox.detectors.loop import LoopDetector  # noqa: E402
from sandbox.gateway import app as gateway_app  # noqa: E402
from sandbox.gateway.stub import scripted_upstream  # noqa: E402
from sandbox.mock import server as mock_server  # noqa: E402
from sandbox.mock.toolspec import TOOL_SPECS, idempotency_params  # noqa: E402
from sandbox.proxy import app as proxy_app  # noqa: E402
from sandbox.runtime import runs  # noqa: E402
from sandbox.trace.reader import load_call_views  # noqa: E402
from sandbox.watchdog import summarize_verdicts  # noqa: E402

GATEWAY_PORT = 4100
PROXY_PORT = 4101
MOCK_PORT = 4102

PAY = {"order_id": "A1", "amount": 599.0}

SCENARIOS: dict[str, list[dict[str, Any]]] = {
    # 重复扣款：同一个订单、同样金额付两次。写操作，无幂等保护。
    "duplicate": [
        {"tool_calls": [{"name": "create_payment", "arguments": PAY}]},
        {"tool_calls": [{"name": "create_payment", "arguments": PAY}]},
        {"content": "支付完成。"},
    ],
    # 循环：反复查同一个订单，停不下来。
    "loop": [
        {"tool_calls": [{"name": "get_order", "arguments": {"order_id": "A1"}}]}
        for _ in range(8)
    ],
    # 负样本：正常的读-写-结束。什么都不该报。
    "clean": [
        {"tool_calls": [{"name": "get_order", "arguments": {"order_id": "A1"}}]},
        {"tool_calls": [{"name": "create_payment", "arguments": PAY}]},
        {"content": "订单 A1 已支付 599 元。"},
    ],
}


async def _serve(app: Any, port: int) -> uvicorn.Server:
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        if task.done():
            raise RuntimeError(f"端口 {port} 上的服务启动失败")
        await asyncio.sleep(0.05)
    return server


async def main(scenario: str, keep_servers: bool) -> int:
    await init_db()
    mock_server.get_state().reset()
    proxy_app.set_policy(None)
    proxy_app.set_transport(None)
    runs.clear()

    servers = [
        await _serve(gateway_app.app, GATEWAY_PORT),
        await _serve(proxy_app.app, PROXY_PORT),
        await _serve(mock_server.app, MOCK_PORT),
    ]
    print(f"服务已就绪：网关 :{GATEWAY_PORT} / 代理 :{PROXY_PORT} / mock :{MOCK_PORT}")

    try:
        return await _run_one(scenario)
    finally:
        if not keep_servers:
            for server in servers:
                server.should_exit = True
            await asyncio.sleep(0.3)


async def _run_one(scenario: str) -> int:
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. 开运行。沙箱签发令牌——Agent 拿到的就是这个，不是真实上游密钥。
        resp = await client.post(
            f"http://127.0.0.1:{GATEWAY_PORT}/_sandbox/runs",
            json={
                "scenario_id": f"walking-skeleton:{scenario}",
                "agent_version": "react_demo@0.1",
                "config": {"scenario": scenario, "model": "stub-model"},
                "declared_writes": sorted(
                    n for n, s in TOOL_SPECS.items() if s.is_write
                ),
                "idempotency_key_params": idempotency_params(),
            },
        )
        resp.raise_for_status()
        run = resp.json()
        run_id = run["run_id"]
        print(f"运行已创建：{run_id}")

        # 2. 装脚本 + 装带外熔断处理器。
        scripted_upstream.register(run_id, SCENARIOS[scenario])

        agent_proc: subprocess.Popen | None = None

        def kill_handler(rid: str, reason: str) -> None:
            """带外的那一半：真的去终止 Agent 进程。

            这是"熔断"和"报警"的分界线。网关拒绝后续请求只能让 Agent 的
            下一次调用失败；进程还在跑，钱还在烧。这个处理器才是止损。
            """
            print(f"\n>>> 带外熔断触发：{reason}")
            if agent_proc is not None and agent_proc.poll() is None:
                agent_proc.terminate()
                print(">>> 已向 Agent 进程发送终止信号")

        runs.on_kill(kill_handler)

        # 3. 启动真 Agent 进程。注意它的环境变量里没有任何沙箱内部地址，
        #    也没有真实上游密钥。
        agent_dir = REPO / "agents" / "react_demo"
        agent_python = agent_dir / ".venv" / "Scripts" / "python.exe"
        if not agent_python.exists():
            print(f"未找到 Agent 的解释器 {agent_python}，请先建 venv 装依赖。", file=sys.stderr)
            return 2

        env = {
            **os.environ,
            "SANDBOX_GATEWAY_BASE_URL": f"http://127.0.0.1:{GATEWAY_PORT}/v1",
            "SANDBOX_AGENT_API_KEY": run["api_key"],
            "SANDBOX_TOOL_BASE_URL": f"http://127.0.0.1:{PROXY_PORT}",
            "SANDBOX_AGENT_MODEL": "stub-model",
            "SANDBOX_AGENT_TASK": "帮我把订单 A1 处理掉。",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        print(f"启动 Agent：{agent_python} agent.py（独立进程、独立 venv）\n")
        started = time.perf_counter()
        agent_proc = subprocess.Popen(
            [str(agent_python), "agent.py"],
            cwd=str(agent_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )

        try:
            # `timeout=` 是关键字，**不能**写成位置参数：`communicate` 的第一个
            # 位置参数是 `input`，而这里的 stdin 不是管道，传进去会被静默丢掉，
            # 于是退化成一句没有超时的 wait()。同 `scenario/runner._wait_for_exit`。
            stdout, _ = await asyncio.to_thread(agent_proc.communicate, timeout=180)
        except (TimeoutError, subprocess.TimeoutExpired):
            agent_proc.kill()
            # 输出在**第二次** communicate 的返回值里，不在异常上。
            drained, _ = await asyncio.to_thread(agent_proc.communicate)
            stdout = f"{drained or ''}\n(Agent 超时未退出，已强杀)"
        elapsed = time.perf_counter() - started

        print("--- Agent 输出 ---")
        print((stdout or "").strip()[:4000] or "(无输出)")
        print(f"--- Agent 退出码 {agent_proc.returncode}，耗时 {elapsed:.1f}s ---\n")

        return await _report(client, run_id, scenario)


async def _report(client: httpx.AsyncClient, run_id: str, scenario: str) -> int:
    charges = (await client.get(f"http://127.0.0.1:{MOCK_PORT}/_mock/charges")).json()
    calls = await load_call_views(run_id)

    print("=== 边界上截获到的调用 ===")
    print(f"工具调用 {len(calls)} 次；mock 侧真实扣款 {charges['count']} 笔")
    for call in calls[:12]:
        mark = "写" if call.is_write else "读"
        blocked = " [被代理拒绝]" if call.blocked_by_proxy else ""
        args = json.dumps(call.args, ensure_ascii=False)
        print(f"  #{call.seq:<3} {mark} {call.tool_name}{blocked} {args}")

    findings = []
    findings.extend(
        IdempotencyDetector(
            {
                name: ToolFacts(name, spec.is_write, spec.enforces_idempotency)
                for name, spec in TOOL_SPECS.items()
            }
        ).run(calls, ExemptionSet())
    )
    findings.extend(LoopDetector().run(calls, ExemptionSet()))

    print("\n=== 检测结论 ===")
    for finding in findings:
        flag = {"pass": "  ", "fail": "!!", "warning": " ?", "inconclusive": ".."}[
            finding.verdict.value
        ]
        print(f"  {flag} [{finding.verdict.value:<12}] {finding.detector}")
        print(f"       {finding.summary}")

    summary = summarize_verdicts(findings)
    print("\n=== 汇总（四态，inconclusive 不并入 pass）===")
    print(
        f"  通过 {summary['pass']:.0f} / 失败 {summary['fail']:.0f} / "
        f"警告 {summary['warning']:.0f} / **未覆盖 {summary['inconclusive']:.0f}**"
    )
    print(
        f"  覆盖率 {summary['coverage_ratio']:.0%}"
        f"（已覆盖部分的失败率 {summary['failure_rate_among_covered']:.0%}）"
    )

    active = runs.get(run_id)
    if active and active.killed:
        print(f"\n  运行已被熔断：{active.kill_reason}")

    # 退出码表达"这次演示是否符合预期"，而不是"有没有失败项"——
    # duplicate/loop 场景本来就该报失败，报不出来才是问题。
    expect_failures = scenario in {"duplicate", "loop"}
    got_failures = summary["fail"] > 0
    if expect_failures != got_failures:
        print(
            f"\n  与预期不符：场景 {scenario!r} "
            f"{'应当' if expect_failures else '不应当'}报出失败，实际"
            f"{'报出' if got_failures else '未报出'}。"
        )
        return 1
    print(f"\n  符合预期：场景 {scenario!r} 的行为已被正确识别。")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="跑一次 walking skeleton 演示")
    parser.add_argument(
        "--scenario", choices=sorted(SCENARIOS), default="duplicate", help="要跑的场景"
    )
    parser.add_argument("--keep-servers", action="store_true", help="跑完不关服务，便于手工排查")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.scenario, args.keep_servers)))
