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

from sandbox import metrics
from sandbox.api import explain, paths, queries, wizard
from sandbox.api.jobs import Job, JobRegistry, JobStep
from sandbox.checkup import catalog
from sandbox.checkup import report as checkup_report
from sandbox.cost import aggregator as cost
from sandbox.db.engine import init_db
from sandbox.formatting import byte_size, millis, money, span_human, tokens
from sandbox.memory.observe import phase_label
from sandbox.regression import cases, diff
from sandbox.scenario import identity, library, loader
from sandbox.scenario.loader import ScenarioError
from sandbox.scenario.spec import Scenario

log = structlog.get_logger(__name__)

#: 界面只监听回环地址。**不要改成 0.0.0.0。** 见模块 docstring 第 1 点。
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

#: 指标看板默认看多少次运行。50 是"够回头看一眼最近的变化、又不至于把上个月的
#: 场景版本混进来"的量级；页面上有 20 / 50 / 200 三个档，都可以点。
DEFAULT_METRICS_LIMIT = 50

#: 上限。地址栏里那个数是用户能改的，而这一页每次都要把 `report_json` 拉回来
#: 解析——不封顶的话 `?limit=1000000` 会把整个库读进内存再逐个 `json.loads`。
#: 截断在**入口**做，不在查询里做：读的人和写的人都看得到同一个数。
MAX_METRICS_LIMIT = 200


def _clamp_limit(value: int) -> int:
    """把地址栏里的 limit 收进 [1, MAX_METRICS_LIMIT]。

    **0 与负数按默认值处理，不按 1。** `?limit=0` 多半是参数没填上，把它当成
    "只看最近 1 次"会让页面显示一个看起来正常、实际被截到只有一条的结论分布——
    而"最近一条通过"读起来和"最近 50 条全通过"完全一样。宁可给默认值。
    """
    if value <= 0:
        return DEFAULT_METRICS_LIMIT
    return min(value, MAX_METRICS_LIMIT)

_HERE = Path(__file__).parent
_TEMPLATES = _HERE / "templates"
_STATIC = _HERE / "static"


def _localdt(value: datetime | str | None, fmt: str = "%m-%d %H:%M:%S") -> str:
    """UTC 存储 → 本地时间显示。

    库里一律存 UTC（见 `db/models.py` 的 `utcnow`），但**给人看的时间必须是
    他所在时区的时间**。直接印 UTC 的话，一个下午三点跑的运行会显示成早上七点，
    而用户会以为记录错了。

    也接受 ISO 字符串：长时运行的世界时刻是**报告快照**里的值（`report_json`），
    它以字符串原样存着，不是从库里读出来的 datetime。同一个页面上两种来源的
    时间必须长得一样，否则读的人会以为其中一个没转时区。
    """
    if value is None:
        return "—"
    if isinstance(value, str):
        try:
            # 存进另一个名字而不是就地覆盖 `value`：就地覆盖时它的类型是
            # "进来那个联合类型"，后面 `value.astimezone()` 在类型检查里就
            # 说不清了（运行时没事，因为认不出来那条路已经 return 了）。
            moment = datetime.fromisoformat(value)
        except ValueError:
            # 认不出来就原样印，不猜。猜错的时间比不显示的时间更坏。
            return value
        return moment.astimezone().strftime(fmt)
    return value.astimezone().strftime(fmt)


def _phase_label_of(observation: dict[str, Any]) -> str:
    """模板过滤器：把一条记忆观测的时点说成"会话 0 结束时"。

    收的是整条观测而不是两个参数，因为模板里拿到的是报告快照里那个字典，写法
    得跟着它走。拼法本身在 `memory/observe.py` 的 `phase_label` 里——命令行报告
    和检测结论用的也是它。
    """
    return phase_label(observation["ordinal"], observation["phase"])


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


def _job_return_to(job: Any) -> str | None:
    """作业跑完之后要跳回的那个内部地址。

    `{run_id}` 占位符在这里替换——**跑完之后才知道运行 id**，所以它不能
    在起作业时就被拼死。

    这一层的存在是为了那句"宁可不跳也不乱跳"：`return_to` 目前只由服务端
    自己拼，但它是被重定向出去的值，所以在这里挡一道——不以单个 `/` 开头
    的地址（`//evil.example`、`https://…`）一律不跳，退回运行详情页。
    少了这道检查，哪天有人接一个查询参数进来就是一个开放重定向。
    """
    target = getattr(job, "return_to", None)
    if not target or not target.startswith("/") or target.startswith("//"):
        return None
    run_id = job.run_id
    return target.replace("{run_id}", run_id) if run_id else None


def create_app(
    *,
    scenarios_root: Path | None = None,
    checkups_root: Path | None = None,
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
    # 菜单目录同样可覆盖，理由和场景目录一样：测试要在一个临时目录里造一份
    # 小菜单（真菜单跑一趟一分多钟，而起六个服务不是这层测试要测的东西）。
    app.state.checkups_root = checkups_root
    app.state.jobs = registry or JobRegistry()

    templates = Jinja2Templates(directory=str(_TEMPLATES))
    templates.env.filters["money"] = money
    templates.env.filters["tokens"] = tokens
    # 字节数 → 人话，同上：`byte_size` 里有两条规矩（负数照印、按量级换单位），
    # 在模板里重写一遍就是两份会漂的说法。
    templates.env.filters["byte_size"] = byte_size
    templates.env.filters["localdt"] = _localdt
    # 秒数 → 人话。与命令行报告共用 `formatting.py` 里那一份，不在模板里另写：
    # 它有一条极易搞错的规矩（不四舍五入成"0 天"——3600 秒印成"0 天"等于说
    # "没过去多久"，而它是一小时）。两份实现意味着网页和命令行可能对同一个
    # 数给出不同说法，读的人第一个念头是"哪个坏了"。
    templates.env.filters["span"] = span_human
    # 毫秒 → 人话。**不能用上面那个顶替**：`span_human` 按秒取整，几十毫秒的
    # 一次往返会被印成「0 秒」，而指标看板整张表讲的就是这个量级。
    templates.env.filters["ms"] = millis
    # 会话边界的时点说法。和上面同一个理由：`会话 0 结束时`这个说法在检测结论、
    # 命令行报告、界面上必须是同一串字符——模板里手拼的话，少一个空格就是三种
    # 写法，而读的人只会觉得报告马虎。（过滤器收的是整条观测，不是两个参数——
    # 模板里拿到的是 `summarize_observations` 写下的那个字典。）
    templates.env.filters["phase_label"] = _phase_label_of
    # 给不写代码的人看的那一层解释，走 Jinja 全局而不是每个路由的 context：
    # 前者漏了会**当场报错**（模板里 `explain` 未定义），后者漏了是渲染成空白
    # ——而空白和"这一节没什么好说的"在页面上一模一样。见 `api/explain.py`。
    templates.env.globals["explain"] = explain

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

    @app.exception_handler(catalog.CheckupError)
    async def _bad_checkup(request: Request, exc: catalog.CheckupError) -> Response:
        # 菜单是人手编辑的，报错要说清是哪一行该改什么——和场景文件同一条
        # 规矩。这里不是 500：用户改得了，改完再点一次就行。
        return _problem(request, 400, "体检菜单有问题", str(exc))

    @app.exception_handler(library.LibraryError)
    async def _bad_library(request: Request, exc: library.LibraryError) -> Response:
        # 工具库同样是手写的。**单独立一条**是因为新建向导直读库、不经过
        # `parse_scenario`——那条路上没人把 `LibraryError` 转成 `ScenarioError`
        # （`loader.py:105` 才做那件事），于是库坏掉时用户看到的会是一个 500，
        # 而消息里只有一句 traceback。库坏了同样是用户改得了的事。
        return _problem(request, 400, "工具库有问题", str(exc))

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

    def _busy_problem(request: Request, blocker: Job) -> Response:
        """执行位被占着。**三处入口共用这一句**（单项运行、一键回归、体检）。

        共用不是为了少写几个字：这段话要解释的是"为什么不能同时跑两个"，
        而三处各写一份的话，迟早有一处漏掉那半句解释，用户看到的就是一个
        没有任何理由的 409。
        """
        return _problem(
            request,
            409,
            # 与 `/api/runs` 那个 JSON 分支里的措辞保持一致：同一个拒绝理由
            # 在两个入口上说成两句话，用户会以为是两种不同的情况。
            "已经有一次运行在跑",
            f"上一个还没结束：{blocker.label}。"
            f"沙箱一次只能跑一个场景——两次运行会互相改掉对方的上游地址，"
            f"两边都会给出看起来正常的报告，而数据是串的。请等它跑完。",
        )

    def _scenario_exists(application: FastAPI, scenario_id: str) -> bool:
        """这个场景文件还在不在。

        用例存的是场景 id，不是文件。文件被删掉之后用例本身仍然完整（基线是
        冻结的），但重跑不了——这个区别要在页面上说清，不能等用户按了按钮
        才报错。
        """
        root = application.state.scenarios_root
        try:
            return paths.scenario_file(scenario_id, root=root).exists()
        except paths.ScenarioPathError:
            return False

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

    def _wizard_context(data: wizard.FormData, error: str | None = None) -> dict[str, Any]:
        """问卷页和确认页共用的上下文。

        **两页从同一个 `FormData` 渲染**，所以「返回改一改」之后再提交，
        生成的 YAML 和刚才看到的那一份不会有差别——中间没有任何东西被重新
        解读过一遍。
        """
        return {
            "form": data,
            "tools": wizard.tool_facts(),
            "symptoms": wizard.SYMPTOMS,
            "choices": wizard.CHOICES,
            "shapes": wizard.SHAPES,
            "error": error,
        }

    def _wizard_page(request: Request, data: wizard.FormData, error: str | None = None) -> Response:
        return templates.TemplateResponse(
            request, "wizard.html", _wizard_context(data, error)
        )

    @app.get("/scenarios/new", response_class=HTMLResponse)
    async def wizard_start(request: Request) -> Response:
        """新建测试向导。**必须注册在 `/scenarios/{scenario_id}` 之前。**

        路由是先注册的先匹配，反过来的话 `new` 会被当成一个场景标识，于是这一页
        永远打不开，而报出来的错是「场景不存在」——它指着的是别的问题。
        `paths.RESERVED_IDS` 是这件事的另一半：一个真叫 `new` 的场景从两边都
        不可达，所以那个名字一开始就不许用。
        """
        return _wizard_page(request, wizard.FormData())

    @app.post("/scenarios/new", response_class=HTMLResponse)
    async def wizard_preview(request: Request) -> Response:
        """收问卷 → 回显「机器打算这样判」。

        **这一步不写盘。** 页面上的确认按钮把生成好的 YAML 文本 POST 给现有的
        `/scenarios/{id}/save`（先 `parse_scenario` 再落盘），所以向导产出的场景
        和手写的走的是同一条保存路径、同一道校验，没有第二条真相。
        """
        raw = await request.form()
        data = wizard.FormData.from_form(
            raw, tools=[str(name) for name in raw.getlist("tools")]
        )

        # 「返回改一改」：把原始表单原样填回问卷，不重新解读。
        if str(raw.get("submit") or "") == "back":
            return _wizard_page(request, data)

        try:
            answers = data.to_answers()
            paths.check_scenario_id(answers.scenario_id)
        except (wizard.WizardError, paths.ScenarioPathError) as exc:
            return _wizard_page(request, data, error=str(exc))

        # 重名必须在这里拦住。保存那一步是**无条件覆盖**的，不拦的话，用户在
        # 向导里起了一个已经存在的名字，点两下就把那份场景换成了这一份——
        # 而磁盘上没有任何痕迹说明发生过什么。
        path = paths.scenario_file(answers.scenario_id, root=app.state.scenarios_root)
        if path.exists():
            return _wizard_page(
                request,
                data,
                error=(
                    f"已经有一个叫 {answers.scenario_id!r} 的场景了"
                    f"（{path.name}）。换一个名字，或者去"
                    f"「场景」列表里打开它改——从这里存下去会把它整份覆盖掉。"
                ),
            )

        facts = wizard.tool_facts()
        context = _wizard_context(data) | {
            "answers": answers,
            "facts": facts,
            "yaml_text": wizard.build_yaml(answers, facts),
            "plan": wizard.plan(answers),
            "warnings": wizard.warnings(answers, facts),
            "placeholders": wizard.placeholders_used(answers, facts),
        }
        return templates.TemplateResponse(request, "wizard_confirm.html", context)

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
        #
        # 原文照写，不再经由模型绕一圈序列化：骨架里的工具是 `use:` 引用，而
        # 序列化出来的是**展开之后**的声明——于是新建出来的第一个场景当场违了
        # 「库里的工具要写 use:」那条规矩，红在一份用户一个字都没改的文件上。
        text = loader.SKELETON_TEXT
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
            scenario = loader.parse_scenario(
                yaml_text, source=f"scenarios/{scenario_id}.yaml"
            )

            # `id:` 和文件名必须一致。
            #
            # **这是唯一一处能造出漂移的地方**，所以判在写的这一侧：读的一侧
            # （首页、运行、回归用例）全都按文件名找人，而**记进库里的
            # `TestRun.scenario_id` 取的是文件里那个 `id`**。两边不一致时，
            # 报告上写着"这个场景跑了"，点进去打开的是**另一个场景**的源码——
            # 报告说了比观测更多的话，而且没有任何东西报错。
            #
            # 不放在 `load_scenario` 里：那里只拿得到一个路径，`parse_scenario`
            # 又只拿得到文本（骨架、编辑器里的草稿都没有文件名，硬判会误伤）。
            if scenario.id != scenario_id:
                raise ScenarioError(
                    f"文件里写的 id 是 {scenario.id!r}，而这份文件叫 "
                    f"{scenario_id}.yaml——两个名字必须一样。\n"
                    f"界面上每一个入口都按「文件名」找场景，而运行记录里存的是"
                    f"文件里那个 id：两边不一致的话，报告点进去打开的是另一个场景，"
                    f"而且不会有任何东西报错。\n"
                    f"改哪一边都行，改完再存一次。"
                )
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
            return _busy_problem(request, blocker)
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
        # 体检：跑的时候是进度页，跑完就地变成结论页。**不跳走**——一趟体检
        # 有六次运行记录，跳去哪一次都是错的，而结论是这六次合起来的那一份。
        if job.checkup is not None:
            if job.state == "running":
                return templates.TemplateResponse(
                    request,
                    "waiting.html",
                    {"job": job, "active": app.state.jobs.active},
                )
            return await _checkup_result(request, job)
        if job.state == "done" and job.run_id:
            target = _job_return_to(job)
            return RedirectResponse(target or f"/runs/{job.run_id}", status_code=303)
        return templates.TemplateResponse(
            request,
            "waiting.html",
            {"job": job, "active": app.state.jobs.active},
        )

    async def _checkup_result(request: Request, job: Job) -> Response:
        """把一趟体检的六次运行读回来，合成一页结论。

        **结论页是现算的，不是跑的时候存下来的。** 每次运行各自的结论都快照在
        自己的运行记录里（那是永久的那一份），这里只做"换个读法"。存一份合成
        结果的话，两处会漂——而漂了之后没有任何东西会让它变红。
        """
        assert job.checkup is not None
        checkup = catalog.load_checkup(
            job.checkup,
            root=app.state.checkups_root,
            scenarios_root=app.state.scenarios_root,
        )
        by_item = {item.id: item for item in checkup.items}

        runs: list[checkup_report.ItemRun] = []
        unmatched: list[JobStep] = []
        for step in job.steps:
            item = by_item.get(step.item_id or "")
            if item is None:
                # 菜单在跑的过程中被改过（这一项被删了，或者标识改了）。**不
                # 按下标硬对**：把 A 项的结论安到 B 项的问题底下，那一页读起来
                # 完全正常，而它是错的。对不上的步骤原样列出来。
                unmatched.append(step)
                continue
            run = await queries.get_run(step.run_id) if step.run_id else None
            runs.append(
                checkup_report.ItemRun(
                    item=item,
                    run_id=step.run_id,
                    report=queries.parse_report(run) if run is not None else None,
                    error=step.error,
                )
            )

        return templates.TemplateResponse(
            request,
            "checkup_result.html",
            {
                "job": job,
                "report": checkup_report.build(checkup, runs),
                "checkup_name": checkup.name,
                "unmatched": unmatched,
                "caveat": checkup_report.NO_FINDING_CAVEAT,
                "active": app.state.jobs.active,
            },
        )

    # --- 一键体检 ---------------------------------------------------------
    #
    # 这一页和别处的差别只有一条：**读者是谁**。它假定读者不知道检测器叫什么、
    # 不知道"预期比对"是什么，只想知道"我这个 Agent 有没有毛病"。所以菜单上
    # 每一项是一个问句（「同一笔钱，它会不会付两次？」），而不是一支场景名。
    #
    # 有一件事必须在页面上说死：现在菜单里挂的场景测的是**仓库自带的演示
    # Agent**，它被写成一定会犯这些错。所以这一页的结论是"这套工具能抓住什么"，
    # 不是"你的系统有多少毛病"。那句话由菜单的 `subject_note` 提供，见
    # `checkup/catalog.py`——不由路由临场拼，因为它得跟着菜单的实际情况变。

    @app.get("/checkup", response_class=HTMLResponse)
    async def checkup_list(request: Request) -> Response:
        entries = []
        for name, _path in catalog.list_checkups(root=app.state.checkups_root):
            entry: dict[str, Any] = {"name": name, "usable": catalog.is_valid_id(name)}
            if entry["usable"]:
                try:
                    entry["checkup"] = catalog.load_checkup(
                        name,
                        root=app.state.checkups_root,
                        scenarios_root=app.state.scenarios_root,
                    )
                    entry["error"] = None
                except catalog.CheckupError as exc:
                    # 打不开的菜单也列出来，并说清为什么——和场景列表同一条
                    # 规矩：磁盘上看得见、界面上找不到，用户没有任何线索。
                    entry |= {"checkup": None, "error": str(exc)}
            else:
                entry |= {"checkup": None, "error": "文件名不合法，界面无法打开或运行它。"}
            entries.append(entry)

        return templates.TemplateResponse(
            request,
            "checkup.html",
            {
                "checkups": entries,
                "active": app.state.jobs.active,
            },
        )

    @app.post("/checkup/{name}/start", response_class=HTMLResponse)
    async def checkup_start(request: Request, name: str) -> Response:
        checkup = catalog.load_checkup(
            name,
            root=app.state.checkups_root,
            scenarios_root=app.state.scenarios_root,
        )
        blocker = app.state.jobs.active
        if blocker is not None:
            return _busy_problem(request, blocker)

        # **菜单点名的场景先全部加载出来，再开始跑。** 加载失败就在这里报错，
        # 不放到跑的时候：跑到第三项才发现文件坏了，前两项已经花掉了一分钟，
        # 而且那份结论会缺一块。
        plans: list[tuple[str, Scenario]] = []
        for item in checkup.items:
            path = paths.scenario_file(item.scenario, root=app.state.scenarios_root)
            # 加载失败会把 `ScenarioError` 抛给上面那个处理器，页面显示文件名
            # 和原因——这正是我们想要的，不吞掉它。
            plans.append((item.id, loader.load_scenario(path)))

        job = app.state.jobs.start_many(plans, checkup=name)
        assert job is not None  # 上面刚确认过没有活跃作业和空菜单
        return RedirectResponse(f"/jobs/{job.id}", status_code=303)

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

    # --- 指标看板（需求 3.7） ---------------------------------------------
    #
    # 这一页的难点不在查询（就三次），在**每个数的口径**：结论按 `verdict_ok`
    # 分不按 `status` 分、没结论的不进分母、延迟有五个口径不合并。这些规矩
    # 全在 `sandbox/metrics.py` 里，这里只负责把范围参数收干净。

    @app.get("/metrics", response_class=HTMLResponse)
    async def metrics_dashboard(
        request: Request, limit: int = DEFAULT_METRICS_LIMIT, scenario: str | None = None
    ) -> Response:
        return templates.TemplateResponse(
            request,
            "metrics.html",
            {
                "report": await metrics.report(
                    limit=_clamp_limit(limit),
                    # 空串按"没给"处理：表单/链接拼出来一个 `scenario=` 是常见
                    # 写法，而它当过滤条件用的话会筛出**零条**——页面上写着
                    # "最近 50 次运行（实际取到 0 次）"，用户以为自己没跑过东西。
                    scenario_id=scenario or None,
                ),
                "active": app.state.jobs.active,
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
                # 按工具这一格与前四格**不是一个口径**（它是上下文占用，没有
                # "花了多少"），所以它在页面上也单独一张表，见 cost.html。
                #
                # 传 `.to_dict()` 不是传对象：`ToolCostReport.tools` 是个 dict，
                # 模板里 `{% for t in ... %}` 遍历的是**键**（一串工具名）。
                # 量过（jinja 3.1.6，见 `_probes/probe_cost_by_tool.py`）：
                # 那样渲染出来是 `[name=|calls=|size=量不到|rent=yes]`——
                # 每一格静默空白，不报错，而且 `t.rent_tokens is not none` 对
                # Undefined 居然是**真**，于是还会多印一个没数的 `≈ tok`。
                # `.to_dict()` 把它摊成按租金降序的列表（"谁在吃上下文"就得从
                # 最大的读起），并顺手把每个工具的 `rent_tokens` 算好——那是要
                # 密度才折得出来的，单个 `ToolCost` 手里没有密度。
                "by_tool": (await cost.by_tool()).to_dict(),
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

    # --- 回归集（需求 3.6） ------------------------------------------------
    #
    # 这一页的重点全在**别把话说错**上，而且比别的页更难：这里同时有"两次运行
    # 的结论"和"这两次能不能放在一起比"两件事。混成一句"发现 3 处差异"，
    # 用户只能自己猜哪几处算数。所以 `Comparison` 里的可比性与差异清单是
    # 并列的，模板也必须把它们分开渲染——见 `regression/diff.py` 的模块注释。

    @app.get("/regression", response_class=HTMLResponse)
    async def regression_list(request: Request) -> Response:
        return templates.TemplateResponse(
            request,
            "regression.html",
            {
                "cases": await cases.list_cases(),
                "active": app.state.jobs.active,
            },
        )

    @app.get("/regression/{case_id}", response_class=HTMLResponse)
    async def regression_detail(
        request: Request, case_id: str, run_id: str = ""
    ) -> Response:
        case = await cases.get_case(case_id)
        if case is None:
            return _problem(
                request, 404, "找不到这条回归用例", f"{case_id} 不在库里。"
            )

        # 默认比"最近跑的那一次"，而不是"我刚点的那一次"：如果用户用别的方式
        # 跑过（命令行、另一个标签页），最近一次才是最新的结论。`run_id` 只在
        # 从别处带过来指定某一次时才用。
        target = None
        if run_id:
            target = await cases.find_run(run_id)
        if target is None:
            target = await cases.latest_run_for(case)

        comparison = (
            diff.compare(cases.snapshot_of_case(case), cases.snapshot_of_run(target))
            if target is not None
            else None
        )
        return templates.TemplateResponse(
            request,
            "regression_detail.html",
            {
                "case": case,
                "run": target,
                "comparison": comparison,
                "section_labels": identity.SECTION_LABELS,
                "kinds": diff,
                "active": app.state.jobs.active,
                # 场景文件可能已经被删了。这时"一键回归测试"按下去会得到一句
                # 说得清的错，而不是一个 500——所以先在这里判一次。
                "scenario_exists": _scenario_exists(app, case.scenario_id),
            },
        )

    @app.post("/regression", response_class=HTMLResponse)
    async def regression_create(
        request: Request,
        run_id: str = Form(...),
        name: str = Form(""),
        note: str = Form(""),
    ) -> Response:
        # **用例从运行记录里建，不从表单里建。** 表单只提供名字和备注；
        # 基线的那份结论是服务端从库里读的——让请求能把结论传进来，等于
        # 允许一条手写的请求伪造一条判据，而它是之后所有比对的标准。
        try:
            case_id = await cases.create_case(run_id=run_id, name=name, note=note)
        except KeyError:
            return _problem(
                request, 404, "找不到这次运行", f"{run_id} 不在库里，没法拿它当基线。"
            )
        except cases.NoConclusionError as exc:
            return _problem(
                request,
                400,
                "这次运行没有结论，不能当基线",
                f"{exc} 拿一个没有结论的运行当基线，之后每一次比对都会报出一个"
                f"假的差异（「未留存 → 不通过」），而它看起来像是真的抓到了退化。",
            )
        return RedirectResponse(f"/regression/{case_id}", status_code=303)

    @app.post("/regression/{case_id}/run", response_class=HTMLResponse)
    async def regression_run(request: Request, case_id: str) -> Response:
        case = await cases.get_case(case_id)
        if case is None:
            return _problem(request, 404, "找不到这条回归用例", f"{case_id} 不在库里。")

        path = paths.scenario_file(case.scenario_id, root=app.state.scenarios_root)
        if not path.exists():
            return _problem(
                request,
                404,
                "场景文件不在了",
                f"这条用例测的是 {case.scenario_id}，但 scenarios/ 下已经没有这个文件了。"
                f"用例的基线还在（结论是冻结的），但没法重跑。",
            )
        scenario = loader.load_scenario(path)
        blocker = app.state.jobs.active
        if blocker is not None:
            return _busy_problem(request, blocker)
        # 跑完直接回到对比页，带上这次的 run_id。**这里是服务端拼的**，
        # 没有一处用到请求里的字符串——所以不存在开放重定向。
        job = app.state.jobs.start(
            scenario, return_to=f"/regression/{case_id}?run_id={{run_id}}"
        )
        assert job is not None
        return RedirectResponse(f"/jobs/{job.id}", status_code=303)

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

    @app.get("/api/regression")
    async def api_regression_list() -> list[dict[str, Any]]:
        return [
            {
                "id": c.id,
                "name": c.name,
                "scenario_id": c.scenario_id,
                "note": c.note,
                "baseline_run_id": c.baseline_run_id,
                # NULL 原样给 null，不折成 false："没留存"和"不通过"是两回事，
                # 折进 JSON 之后调用方再也分不出来。
                "baseline_verdict_ok": c.baseline_verdict_ok,
                "baseline_started_at": (
                    c.baseline_started_at.isoformat() if c.baseline_started_at else None
                ),
                "created_at": c.created_at.isoformat(),
            }
            for c in await cases.list_cases()
        ]

    @app.get("/api/regression/{case_id}")
    async def api_regression_detail(case_id: str, run_id: str = "") -> Response:
        case = await cases.get_case(case_id)
        if case is None:
            return JSONResponse({"error": "找不到这条回归用例", "id": case_id}, status_code=404)
        target = await cases.find_run(run_id) if run_id else None
        target = target or await cases.latest_run_for(case)
        if target is None:
            return JSONResponse(
                {
                    "id": case.id,
                    "name": case.name,
                    "scenario_id": case.scenario_id,
                    "run_id": None,
                    "comparison": None,
                    "note": "这个场景还没有跑过第二次，没有可比的东西。",
                }
            )
        comparison = diff.compare(cases.snapshot_of_case(case), cases.snapshot_of_run(target))
        return JSONResponse(
            {
                "id": case.id,
                "name": case.name,
                "scenario_id": case.scenario_id,
                "baseline_run_id": case.baseline_run_id,
                "run_id": target.id,
                "comparison": {
                    "comparability": comparison.comparability,
                    "outcome": comparison.outcome,
                    "headline": comparison.headline,
                    "counts": comparison.counts,
                    "changed_sections": comparison.changed_sections,
                    "notes": comparison.notes,
                    "differences": [
                        {
                            "kind": d.kind,
                            "subject": d.subject,
                            "before": d.before,
                            "after": d.after,
                        }
                        for d in comparison.differences
                    ],
                },
            }
        )

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
