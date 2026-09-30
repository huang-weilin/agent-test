"""面向用户的 Web 界面。需求 Phase 1d 的交付物（场景编辑器 + 报告）。

**为什么是服务端渲染，而不是需求里写的 React + Vite。** 需求第 250 行给的前端
技术栈是 React + TS + Vite，那是个合理的长期选择，但它要求目标机器上有一条
Node 构建链——而本工具的 lite profile 恰恰是"7.9GB 内存、不装 Docker、一条命令
跑起来"。为了一个本地测试工具去拉一整条前端工具链，换来的东西用户看不见。

服务端渲染 + `<meta http-equiv="refresh">` 轮询，配合零 JavaScript，任何人
`python scripts/serve_ui.py` 之后打开浏览器就有界面。数据层（`queries.py`）
跟 HTTP 无关，将来真要换 React，那层不用重写——**被替换掉的是模板，不是接口**。

三个安全边界，都是"不做就会静默出事"的那类：

1. **只绑回环地址。** 这个服务能写场景文件、能拉起子进程。绑 `0.0.0.0`
   等于把"往这台机器上写文件"开放给整个局域网，而它**没有任何认证**。
   这里不给远程开关——要远程就用 SSH 端口转发，那条路上至少有认证。

2. **场景标识走白名单**（`paths.py`）。从 URL 取来的字符串会被拼进文件路径，
   不校验就是一个任意文件写。

3. **保存之前先校验。** 写进磁盘的文本必须先能被 `load_scenario` 读出来，
   否则一次笔误就能把场景文件写成加载不出来的样子，而全部用例都在这些文件里。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sandbox.api import paths, queries
from sandbox.api.jobs import JobRegistry
from sandbox.cost import aggregator as cost
from sandbox.db.engine import init_db
from sandbox.formatting import money, tokens
from sandbox.scenario import loader
from sandbox.scenario.loader import ScenarioError
from sandbox.scenario.spec import Scenario

log = structlog.get_logger(__name__)

#: 界面只监听回环地址。**不要改成 0.0.0.0。** 见模块 docstring 第 1 点。
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

_HERE = Path(__file__).parent
_TEMPLATES = _HERE / "templates"
_STATIC = _HERE / "static"


def _localdt(value: datetime | None, fmt: str = "%m-%d %H:%M:%S") -> str:
    """UTC 存储 → 本地时间显示。

    库里一律存 UTC（见 `db/models.py` 的 `utcnow`），但**给人看的时间必须是
    他所在时区的时间**。直接印 UTC 的话，一个下午三点跑的运行会显示成早上七点，
    而用户会以为记录错了。
    """
    if value is None:
        return "—"
    return value.astimezone().strftime(fmt)


def _verdict_label(verdict_ok: bool | None) -> tuple[str, str]:
    """结论 → (样式类, 文字)。

    **`None` 不是失败。** 它是"这条运行没有留下结论"——加这个字段之前跑的
    历史运行。显示成红叉等于替一条没有结论的运行编一个结论出来，而用户会去
    查一个根本不存在的缺陷。
    """
    if verdict_ok is None:
        return ("unknown", "未留存")
    return ("pass", "通过") if verdict_ok else ("fail", "不通过")


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    # 建表 + 补齐新增的列。界面上每一个页面都要读库，所以这件事必须在开始
    # 接受请求之前完成，而不是等第一次查询时才发现少一张表。
    await init_db()
    yield


def create_app(
    *,
    scenarios_root: Path | None = None,
    registry: JobRegistry | None = None,
) -> FastAPI:
    app = FastAPI(
        title="AI Agent 行为级测试沙盒",
        lifespan=_lifespan,
        # 关掉自带的 /docs 与 /redoc：它们没有任何认证，而这个服务能拉起进程。
        # 接口不复杂，`/api/*` 的路径本身就是文档。
        docs_url=None,
        redoc_url=None,
    )
    app.state.scenarios_root = scenarios_root
    app.state.jobs = registry or JobRegistry()

    templates = Jinja2Templates(directory=str(_TEMPLATES))
    templates.env.filters["money"] = money
    templates.env.filters["tokens"] = tokens
    templates.env.filters["localdt"] = _localdt

    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

    # --- 错误处理 ---------------------------------------------------------
    # 场景文件的问题（路径越界、YAML 不合法）是**用户能自己修**的，所以页面
    # 上要原样显示那条消息，而不是一个 500。JSON 请求走 JSON，页面请求走页面。

    def _wants_json(request: Request) -> bool:
        return request.url.path.startswith("/api/")

    @app.exception_handler(paths.ScenarioPathError)
    async def _bad_path(request: Request, exc: paths.ScenarioPathError) -> Response:
        return _problem(request, 400, "场景标识不合法", str(exc))

    @app.exception_handler(ScenarioError)
    async def _bad_scenario(request: Request, exc: ScenarioError) -> Response:
        return _problem(request, 400, "场景文件有问题", str(exc))

    def _problem(request: Request, status: int, title: str, detail: str) -> Response:
        log.info("api.problem", path=request.url.path, status=status, detail=detail)
        if _wants_json(request):
            return JSONResponse(
                {"error": title, "detail": detail}, status_code=status
            )
        return templates.TemplateResponse(
            request,
            "problem.html",
            {"title": title, "detail": detail},
            status_code=status,
        )

    # --- 页面 -------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> Response:
        entries = []
        for scenario_id, path in paths.list_scenarios(root=app.state.scenarios_root):
            entry: dict[str, Any] = {"id": scenario_id, "usable": paths.is_valid_id(scenario_id)}
            if entry["usable"]:
                try:
                    scenario = loader.load_scenario(path)
                    entry |= {
                        "description": scenario.description,
                        "sessions": len(scenario.sessions),
                        "tools": len(scenario.tools),
                        "expects": len(scenario.expect),
                        "error": None,
                    }
                except ScenarioError as exc:
                    # 加载不出来的场景仍然列出来，并说清它为什么打不开。
                    # 从列表里藏掉的话，用户看见文件在磁盘上却怎么都找不到它。
                    entry |= {"description": "", "error": str(exc)}
            else:
                entry |= {
                    "description": "",
                    "error": "文件名不合法，界面无法打开或运行它。"
                    "请改成小写字母、数字、连字符或下划线。",
                }
            entries.append(entry)

        return templates.TemplateResponse(
            request,
            "home.html",
            {
                "scenarios": entries,
                "runs": await queries.recent_runs(limit=8),
                "budget": await cost.budget_status(),
                "active": app.state.jobs.active,
                "verdict_label": _verdict_label,
            },
        )

    @app.get("/scenarios/{scenario_id}", response_class=HTMLResponse)
    async def scenario_editor(request: Request, scenario_id: str, saved: int = 0) -> Response:
        path = paths.scenario_file(scenario_id, root=app.state.scenarios_root)
        if not path.exists():
            # 不存在就是不存在，**不给一个空编辑器**。空编辑器点保存就凭空
            # 建出一个场景来，于是"新建"有两条路径，其中一条用户根本不知道
            # 自己走了。新建只有一个入口：列表页的那个表单。
            return _problem(
                request, 404, "场景不存在", f"scenarios/{scenario_id}.yaml 不在场景目录里。"
            )

        text = path.read_text(encoding="utf-8")
        scenario = None
        error = None
        try:
            scenario = loader.load_scenario(path)
        except ScenarioError as exc:
            # 文件坏了也要让人打开它——不然就只能去命令行修，
            # 而编辑器存在的意义正是"不用去命令行"。
            error = str(exc)

        return templates.TemplateResponse(
            request,
            "editor.html",
            {
                "scenario_id": scenario_id,
                "yaml_text": text,
                "scenario": scenario,
                "error": error,
                "saved": bool(saved),
                "active": app.state.jobs.active,
            },
        )

    @app.post("/scenarios", response_class=HTMLResponse)
    async def create_scenario(new_id: str = Form(...)) -> Response:
        scenario_id = paths.check_scenario_id(new_id.strip())
        path = paths.scenario_file(scenario_id, root=app.state.scenarios_root)
        if path.exists():
            return RedirectResponse(f"/scenarios/{scenario_id}", status_code=303)
        # 骨架经由解析器过一遍再写：**连我们自己的模板都要通过那道校验**，
        # 不然就会有一个"编辑器一保存就报错"的默认场景。
        text = loader.dump_scenario(
            Scenario.model_validate(loader.dump_skeleton())
        )
        loader.parse_scenario(text, source="<新建场景骨架>")
        paths.write_scenario_text(scenario_id, text, root=app.state.scenarios_root)
        return RedirectResponse(f"/scenarios/{scenario_id}", status_code=303)

    @app.post("/scenarios/{scenario_id}/save", response_class=HTMLResponse)
    async def save_scenario(
        request: Request, scenario_id: str, yaml_text: str = Form(...)
    ) -> Response:
        # 标识先过一遍：不合法就没必要去解析 YAML 了，而且这条检查必须在
        # 任何一次文件操作之前。真正的路径解析在 `write_scenario_text` 里，
        # 那里同样会校验一次——两道都在，因为这是唯一会写盘的地方。
        paths.check_scenario_id(scenario_id)
        try:
            # **先校验，再落盘。** 顺序反过来的话，一次笔误就把场景文件写坏了，
            # 而全部用例都从这些文件来——写坏一个等于弄丢一个用例。
            loader.parse_scenario(yaml_text, source=f"scenarios/{scenario_id}.yaml")
        except ScenarioError as exc:
            return templates.TemplateResponse(
                request,
                "editor.html",
                {
                    "scenario_id": scenario_id,
                    "yaml_text": yaml_text,  # 回显**用户刚敲的**，不是磁盘上的
                    "scenario": None,
                    "error": str(exc),
                    "saved": False,
                    "active": app.state.jobs.active,
                },
                status_code=400,
            )
        paths.write_scenario_text(scenario_id, yaml_text, root=app.state.scenarios_root)
        return RedirectResponse(f"/scenarios/{scenario_id}?saved=1", status_code=303)

    @app.post("/scenarios/{scenario_id}/run", response_class=HTMLResponse)
    async def run_scenario(request: Request, scenario_id: str) -> Response:
        path = paths.scenario_file(scenario_id, root=app.state.scenarios_root)
        # **跑的是磁盘上的那份，不是编辑框里的。** 未保存的改动不参与运行，
        # 页面上也是这么写的——不然"我改了怎么结果没变"会变成一个
        # 查不出所以然的问题。
        scenario = loader.load_scenario(path)
        blocker = app.state.jobs.active
        if blocker is not None:
            return _problem(
                request,
                409,
                "已经有一次运行在跑",
                f"上一次运行（{blocker.scenario_id}）还没结束。"
                f"沙箱一次只能跑一个场景——两次运行会互相改掉对方的上游地址，"
                f"两边都会给出看起来正常的报告，而数据是串的。请等它跑完。",
            )
        job = app.state.jobs.start(scenario)
        assert job is not None  # 上面刚确认过没有活跃作业，且这里没有 await
        return RedirectResponse(f"/jobs/{job.id}", status_code=303)

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    async def job_status(request: Request, job_id: str) -> Response:
        job = app.state.jobs.get(job_id)
        if job is None:
            # 服务重启过。这不是"运行失败"，是"那次运行的作业记录不在内存里了"——
            # 已经落库的那些运行仍然在 /runs 里查得到。
            return _problem(
                request,
                404,
                "找不到这次运行的作业",
                "服务可能重启过，内存里的作业记录不在了。"
                "如果它当时已经跑完，结论在「运行记录」里能查到。",
            )
        if job.state == "done" and job.run_id:
            return RedirectResponse(f"/runs/{job.run_id}", status_code=303)
        return templates.TemplateResponse(
            request,
            "waiting.html",
            {"job": job, "active": app.state.jobs.active},
        )

    @app.get("/runs", response_class=HTMLResponse)
    async def run_list(request: Request, scenario_id: str | None = None) -> Response:
        return templates.TemplateResponse(
            request,
            "runs.html",
            {
                "runs": await queries.recent_runs(limit=100, scenario_id=scenario_id),
                "scenario_id": scenario_id,
                "active": app.state.jobs.active,
                "verdict_label": _verdict_label,
            },
        )

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    async def run_detail(request: Request, run_id: str) -> Response:
        run = await queries.get_run(run_id)
        if run is None:
            return _problem(request, 404, "找不到这次运行", f"运行 {run_id} 不在库里。")

        path = paths.scenario_file(run.scenario_id, root=app.state.scenarios_root)
        scenario = None
        if path.exists():
            try:
                scenario = loader.load_scenario(path)
            except ScenarioError:
                # 场景文件后来被改坏或删掉了，不影响这次运行的历史结论——
                # 报告是当时存下来的快照，本来就不依赖今天的场景文件。
                scenario = None

        return templates.TemplateResponse(
            request,
            "run.html",
            {
                "run": run,
                "report": queries.parse_report(run),
                "scenario": scenario,
                "tool_calls": await queries.tool_calls_for_run(run_id),
                "snapshots": await queries.snapshots_for_sequence(run.session_sequence_id),
                "label": _verdict_label(run.verdict_ok),
            },
        )

    @app.get("/cost", response_class=HTMLResponse)
    async def cost_dashboard(request: Request, period: str = "day") -> Response:
        if period not in ("day", "week", "month"):
            period = "day"
        return templates.TemplateResponse(
            request,
            "cost.html",
            {
                "period": period,
                "history": await cost.by_period(period, limit=30),  # type: ignore[arg-type]
                "by_scenario": await cost.by_scenario(),
                "by_agent": await cost.by_agent(),
                "by_model": await cost.by_model(),
                "budget": await cost.budget_status(),
                "active": app.state.jobs.active,
            },
        )

    # --- 记忆视图（需求 3.4） ---------------------------------------------
    #
    # 这一页的难点不是查询，是**别把话说错**。它显示的是记忆的**视野边界**，
    # 不是记忆的内容——内容在 Agent 进程内，边界拦不到（见
    # `memory/snapshot.py` 的模块注释）。措辞上把"我们回滚了"和"Agent 忘了"
    # 分开，是这一页存在的全部理由：说反了，用户会去改一个没问题的 Prompt。

    @app.get("/memory", response_class=HTMLResponse)
    async def memory_view(request: Request) -> Response:
        return templates.TemplateResponse(
            request,
            "memory.html",
            {
                "sequences": await queries.recent_sequences(limit=50),
                "active": app.state.jobs.active,
            },
        )

    @app.get("/memory/{session_sequence_id}", response_class=HTMLResponse)
    async def memory_detail(request: Request, session_sequence_id: str) -> Response:
        scenario_id = await queries.scenario_for_sequence(session_sequence_id)
        if scenario_id is None:
            # 没有运行引用这个序列 = 这个 id 不存在。**不给一个空页面**，
            # 空页面看起来和"这个序列还没有会话"一模一样，而那是两回事。
            return _problem(
                request, 404, "找不到这个会话序列", f"{session_sequence_id} 不在库里。"
            )
        return templates.TemplateResponse(
            request,
            "memory_detail.html",
            {
                "scenario_id": scenario_id,
                "session_sequence_id": session_sequence_id,
                "sessions": await queries.sessions_for_sequence(session_sequence_id),
                "snapshots": await queries.snapshots_for_sequence(session_sequence_id),
                "active": app.state.jobs.active,
            },
        )

    # --- 审计日志（需求 3.2） ---------------------------------------------

    @app.get("/audit", response_class=HTMLResponse)
    async def audit_view(
        request: Request,
        scenario_id: str = "",
        tool_name: str = "",
        writes_only: str = "",
        blocked_only: str = "",
    ) -> Response:
        # 复选框用**原始查询串**读（有值即为真），而不是 `bool` 类型参数：
        # 未勾选时浏览器根本不发这个字段，`bool` 会把"没发"和"发了空值"
        # 都当成 False——现在两者确实等价，但改天加一个三态筛选就会踩到。
        rows, total = await queries.audit_log(
            scenario_id=scenario_id or None,
            tool_name=tool_name or None,
            writes_only=bool(writes_only),
            blocked_only=bool(blocked_only),
        )
        return templates.TemplateResponse(
            request,
            "audit.html",
            {
                "rows": rows,
                "total": total,
                "scenario_id": scenario_id,
                "tool_name": tool_name,
                "writes_only": bool(writes_only),
                "blocked_only": bool(blocked_only),
                "scenarios": await queries.scenarios_seen(),
                "tools": await queries.tool_names_seen(),
                "active": app.state.jobs.active,
            },
        )

    # --- JSON 接口 --------------------------------------------------------
    #
    # 只读的那几个 + 一个"发起运行"。**没有 JSON 写接口**：改场景文件的路径
    # 只有 HTML 表单一条，于是路径校验、先校验后落盘这些只在一个地方落地，
    # 不用维护两套写入口的等价性。将来真需要，加的时候把表单处理器抽出来即可。

    @app.get("/api/scenarios")
    async def api_scenarios() -> list[dict[str, Any]]:
        out = []
        for scenario_id, path in paths.list_scenarios(root=app.state.scenarios_root):
            item: dict[str, Any] = {"id": scenario_id, "usable": paths.is_valid_id(scenario_id)}
            try:
                scenario = loader.load_scenario(path)
            except ScenarioError as exc:
                item |= {"valid": False, "error": str(exc)}
            else:
                item |= {
                    "valid": True,
                    "error": None,
                    "description": scenario.description,
                    "agent_version": scenario.agent.version,
                    "sessions": len(scenario.sessions),
                    "tools": len(scenario.tools),
                    "expected_charges": scenario.expected_charges,
                }
            out.append(item)
        return out

    @app.get("/api/runs")
    async def api_runs(limit: int = 50, scenario_id: str | None = None) -> list[dict[str, Any]]:
        return [
            {
                "run_id": r.id,
                "scenario_id": r.scenario_id,
                "agent_version": r.agent_version,
                "status": r.status,
                "ok": r.verdict_ok,  # None = 结论未留存，不是"不通过"
                "started_at": r.started_at.isoformat(),
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "cost": r.cost.to_dict(),
            }
            for r in await queries.recent_runs(limit=limit, scenario_id=scenario_id)
        ]

    @app.get("/api/runs/{run_id}")
    async def api_run(run_id: str) -> Response:
        run = await queries.get_run(run_id)
        if run is None:
            return JSONResponse({"error": "找不到这次运行", "run_id": run_id}, status_code=404)
        report = queries.parse_report(run)
        if report is None:
            return JSONResponse(
                {
                    "run_id": run.id,
                    "scenario_id": run.scenario_id,
                    "status": run.status,
                    "report": None,
                    "note": "这次运行没有留下结论快照（在本字段存在之前跑的）。不是失败。",
                }
            )
        return JSONResponse(report)

    @app.post("/api/runs")
    async def api_start_run(payload: dict[str, Any]) -> Response:
        scenario_id = str(payload.get("scenario_id", ""))
        path = paths.scenario_file(scenario_id, root=app.state.scenarios_root)
        scenario = loader.load_scenario(path)
        blocker = app.state.jobs.active
        if blocker is not None:
            return JSONResponse(
                {
                    "error": "已经有一次运行在跑",
                    "active_job": blocker.id,
                    "active_scenario": blocker.scenario_id,
                },
                status_code=409,
            )
        job = app.state.jobs.start(scenario)
        assert job is not None  # 上面刚确认过没有活跃作业，且这里没有 await
        return JSONResponse({"job_id": job.id, "scenario_id": scenario.id}, status_code=202)

    @app.get("/api/jobs/{job_id}")
    async def api_job(job_id: str) -> Response:
        job = app.state.jobs.get(job_id)
        if job is None:
            return JSONResponse({"error": "找不到作业", "job_id": job_id}, status_code=404)
        return JSONResponse(
            {
                "job_id": job.id,
                "scenario_id": job.scenario_id,
                "state": job.state,
                "run_id": job.run_id,
                "ok": job.ok,
                "error": job.error,
            }
        )

    @app.get("/api/memory")
    async def api_memory() -> list[dict[str, Any]]:
        return [
            {
                "session_sequence_id": s.session_sequence_id,
                "scenario_id": s.scenario_id,
                "run_id": s.run_id,
                "started_at": s.started_at.isoformat(),
                "sessions": s.session_count,
                "snapshots": s.snapshot_count,
                "rolled_back_sessions": s.rolled_back_sessions,
                # >0 表示有会话不可能知道某些历史序号——那一段里的重复**不怪 Agent**。
                "missing_ordinals": s.missing_ordinals,
            }
            for s in await queries.recent_sequences(limit=50)
        ]

    @app.get("/api/memory/{session_sequence_id}")
    async def api_memory_detail(session_sequence_id: str) -> Response:
        scenario_id = await queries.scenario_for_sequence(session_sequence_id)
        if scenario_id is None:
            return JSONResponse(
                {"error": "找不到这个会话序列", "session_sequence_id": session_sequence_id},
                status_code=404,
            )
        return JSONResponse(
            {
                "session_sequence_id": session_sequence_id,
                "scenario_id": scenario_id,
                "sessions": [
                    {
                        "ordinal": s.ordinal,
                        "memory_horizon_ordinal": s.memory_horizon_ordinal,
                        "expected_horizon": s.expected_horizon,
                        # 空列表 = 自然延续；非空 = 这些序号 Agent 不可能知道。
                        "missing_ordinals": s.missing_ordinals,
                        "restored_from_snapshot_id": s.restored_from_snapshot_id,
                        "status": s.status,
                    }
                    for s in await queries.sessions_for_sequence(session_sequence_id)
                ],
                # 只有元数据：内容按 adapter 私有格式存，视图层不解析它。
                "snapshots": [
                    {
                        "snapshot_id": s.id,
                        "position": [s.taken_at_ordinal, s.taken_at_seq],
                        "adapter": s.adapter,
                        "digest": s.digest,
                        "entry_count": s.entry_count,
                        "size_bytes": s.size_bytes,
                    }
                    for s in await queries.snapshots_for_sequence(session_sequence_id)
                ],
            }
        )

    @app.get("/api/audit")
    async def api_audit(
        scenario_id: str = "",
        tool_name: str = "",
        writes_only: str = "",
        blocked_only: str = "",
        limit: int = queries.AUDIT_PAGE,
    ) -> dict[str, Any]:
        rows, total = await queries.audit_log(
            scenario_id=scenario_id or None,
            tool_name=tool_name or None,
            writes_only=bool(writes_only),
            blocked_only=bool(blocked_only),
            limit=limit,
        )
        return {
            "count": len(rows),
            # `total` 是符合筛选条件的**全部**行数。不返回它的话，调用方
            # 会把"这一页只有 200 条"读成"一共就 200 条"。
            "total": total,
            "truncated": total > len(rows),
            "calls": [
                {
                    "seq": r.seq,
                    "run_id": r.run_id,
                    "scenario_id": r.scenario_id,
                    "agent_version": r.agent_version,
                    "session_ordinal": r.session_ordinal,
                    "tool_name": r.tool_name,
                    "is_write": r.is_write,
                    "idempotency_key": r.idempotency_key,
                    "status": r.status,
                    "blocked_by_proxy": r.blocked_by_proxy,
                    "replay_reason": r.replay_reason,
                    "args": r.args_json,
                    "error": r.error,
                    "latency_ms": r.latency_ms,
                    "ts": r.ts.isoformat(),
                }
                for r in rows
            ],
        }

    return app


def serve(*, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """起界面服务。

    **只允许回环地址。** 这个服务没有任何认证，而它能写场景文件、能拉起
    Agent 子进程。绑到 `0.0.0.0` 等于把这些开放给整个局域网。这里刻意
    **不留远程开关**——用 `ssh -L 8765:127.0.0.1:8765` 转发出去，那条路上
    至少有认证。

    `reload=False` 是刻意的：热重载会在文件变动时重启进程，而一次场景运行
    正在跑的时候重启，会留下一批"永远进行中"的运行记录。
    """
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise ValueError(
            f"界面只允许绑回环地址，收到 {host!r}。\n"
            f"这个服务没有认证，而它能写场景文件、能拉起子进程。"
            f"要远程访问请用端口转发：ssh -L {port}:127.0.0.1:{port} <主机>"
        )
    print(f"界面地址： http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
