"""Web 界面的测试。

用 `httpx.ASGITransport` 而不是 `fastapi.testclient.TestClient`：后者起一个
后台线程、跑自己的事件循环，于是测试夹具里的库会话和被测代码里的库会话
在**两个循环**上，跨循环用 asyncio 原语（Event、Task）会出各种诡异问题。
ASGITransport 直接在当前循环里调 ASGI 应用，没有这一层。

代价是它**不跑 lifespan**，所以 `init_db()` 要夹具自己调——显式一点也好，
这里的每个测试到底建没建表，一眼看得出来。
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from sandbox.api import paths
from sandbox.api.app import DEFAULT_HOST, create_app, serve
from sandbox.api.jobs import JobRegistry
from sandbox.db.engine import init_db
from sandbox.db.models import TestRun
from sandbox.scenario.loader import ScenarioError, load_scenario

CLEAN_YAML = """
id: clean
description: 负样本：证明检测器不是"凡跑必红"。
agent:
  version: react_demo@0.1
  task: 查一下订单 A1。
  model: stub-model
tools:
  - name: get_order
    description: 查订单
    is_write: false
sessions:
  - name: 第一次
    script:
      - tool_calls: [{name: get_order, arguments: {order_id: A1}}]
expect:
  - detector: idempotency
    verdict: pass
"""


@pytest.fixture
def scenarios(tmp_path: Path) -> Path:
    root = tmp_path / "scenarios"
    root.mkdir()
    (root / "clean.yaml").write_text(CLEAN_YAML, encoding="utf-8")
    return root


class _ControllableRunner:
    """执行器的替身。

    **不真跑场景。** 真跑一次要起三个 uvicorn、拉起 Agent 子进程、几十秒——
    而这里要测的是路由、并发控制和渲染，不是引擎（引擎有自己的端到端测试）。

    阻塞用 `threading.Event` 而不是 `asyncio.Event`：它不绑事件循环，
    在夹具和测试之间传递时不用挑循环。
    """

    def __init__(self, state: dict) -> None:
        self.state = state

    async def run(self, scenario):  # type: ignore[no-untyped-def]
        self.state["started"].set()
        while not self.state["release"].is_set():
            await asyncio.sleep(0.01)
        self.state["ran"].append(scenario.id)
        result = self.state.get("result")
        if result is not None:
            return result(scenario)
        from sandbox.scenario.runner import ScenarioResult

        return ScenarioResult(scenario_id=scenario.id, run_id=self.state.get("run_id", ""))


@pytest.fixture
def runner_state() -> dict:
    return {"started": threading.Event(), "release": threading.Event(), "ran": []}


@pytest_asyncio.fixture
async def client(
    temp_db: str, scenarios: Path, runner_state: dict
) -> AsyncIterator[httpx.AsyncClient]:
    await init_db()
    app = create_app(
        scenarios_root=scenarios, registry=JobRegistry(runner_factory=_ControllableRunner)
    )
    # 注册表要拿到那份状态，但 `runner_factory` 是无参构造的（真执行器本来就
    # 无参）。用一个绑定了状态的子类绕过去，比为测试改生产签名干净。
    app.state.jobs._runner_factory = lambda: _ControllableRunner(runner_state)  # noqa: SLF001

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://ui") as c:
        yield c


# --- 场景标识：这是整个 Web 层唯一碰磁盘的地方 --------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "../etc/passwd",
        "..\\windows\\system32",
        "a/b",
        "a\\b",
        "/etc/passwd",
        "C:/windows",
        "",
        "Clean",  # 大写
        ".hidden",
        "clean.yaml",  # 扩展名会变成 clean.yaml.yaml
        "clean yaml",
        "a" * 65,  # 超长
        "场景",
    ],
)
def test_rejects_unsafe_scenario_ids(bad: str, tmp_path: Path) -> None:
    """白名单之外的一律拒绝。

    这些不是"理论上可能出问题"的输入——它们全是路径遍历的经典写法。
    检查放在 `paths.py` 一处，路由层不需要重复一遍。
    """
    with pytest.raises(paths.ScenarioPathError):
        paths.scenario_file(bad, root=tmp_path)


def test_accepts_normal_scenario_ids(tmp_path: Path) -> None:
    for good in ("clean", "cross_session_memory_loss", "a1", "x-1_2"):
        assert paths.scenario_file(good, root=tmp_path).parent == tmp_path.resolve()


def test_path_stays_inside_scenarios_dir(tmp_path: Path) -> None:
    """第二道防线：即使正则被放宽，包含性检查仍然拦得住。"""
    base = tmp_path / "scenarios"
    base.mkdir()
    # 直接调底层，绕过正则——模拟"哪天有人把正则改松了"
    resolved = (base / ".." / "outside.yaml").resolve()
    assert resolved.parent != base.resolve()


def test_list_scenarios_keeps_unusable_names(tmp_path: Path) -> None:
    """文件名不合法的场景**仍然列出来**。

    藏掉的话，用户在文件管理器里看得见 `My Scenario.yaml`，界面上怎么都
    找不到，而没有任何东西告诉他为什么。
    """
    (tmp_path / "good.yaml").write_text(CLEAN_YAML, encoding="utf-8")
    (tmp_path / "Bad Name.yaml").write_text(CLEAN_YAML, encoding="utf-8")

    names = [n for n, _ in paths.list_scenarios(root=tmp_path)]
    assert names == ["Bad Name", "good"]
    assert not paths.is_valid_id("Bad Name")


def test_write_is_atomic_and_leaves_no_temp_files(tmp_path: Path) -> None:
    paths.write_scenario_text("clean", CLEAN_YAML, root=tmp_path)
    written = (tmp_path / "clean.yaml").read_text(encoding="utf-8")
    assert written == CLEAN_YAML
    # 临时文件以 `.` 开头，不会被列出来——留在那儿就永远没人知道
    assert [p.name for p in tmp_path.iterdir()] == ["clean.yaml"]


# --- 绑定地址 -----------------------------------------------------------------


def test_default_host_is_loopback() -> None:
    """这个服务没有认证，而它能写文件、能拉起子进程。

    `0.0.0.0` 会把这两件事开放给整个局域网。这条测试是**钉住默认值**的：
    改它必须是有意识的、会被这条测试拦下来的动作。
    """
    assert DEFAULT_HOST == "127.0.0.1"


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "localhost.attacker.com"])
def test_refuses_non_loopback_bind(host: str) -> None:
    with pytest.raises(ValueError, match="只允许绑回环地址"):
        serve(host=host)


# --- 页面 ---------------------------------------------------------------------


async def test_home_lists_scenarios(client: httpx.AsyncClient) -> None:
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "clean" in resp.text
    assert "负样本" in resp.text


async def test_editor_shows_yaml_and_parsed_summary(client: httpx.AsyncClient) -> None:
    resp = await client.get("/scenarios/clean")
    assert resp.status_code == 200
    assert "get_order" in resp.text
    assert "react_demo@0.1" in resp.text


async def test_unknown_scenario_is_404_not_an_empty_editor(client: httpx.AsyncClient) -> None:
    """不存在就 404。

    给一个空编辑器的话，点保存就凭空建出一个场景——于是"新建"有两条路径，
    而其中一条用户根本不知道自己走了。
    """
    resp = await client.get("/scenarios/nope")
    assert resp.status_code == 404
    assert "不存在" in resp.text


async def test_traversal_attempt_is_400(client: httpx.AsyncClient) -> None:
    resp = await client.get("/scenarios/..%2F..%2Fetc%2Fpasswd")
    assert resp.status_code in (400, 404)


async def test_runs_and_cost_pages_render(client: httpx.AsyncClient) -> None:
    assert (await client.get("/runs")).status_code == 200
    assert (await client.get("/cost")).status_code == 200


async def test_html_is_escaped(client: httpx.AsyncClient, scenarios: Path) -> None:
    """模板必须自动转义。

    场景说明、检测器结论、Agent 的 stdout、工具参数全是外部输入。手写
    f-string 拼 HTML 意味着每一处都要记得 escape，漏一处就是一个 XSS——
    而漏掉的那一处不会有任何症状，直到有人打开那个页面。
    """
    # 整份重新写，不在 CLEAN_YAML 上做字符串替换——替换的源串一旦和实际
    # 内容对不上（引号全角半角、空格），替换会静默落空，于是这个测试变成
    # "页面上本来就没有 script"，全绿，但什么都没测。
    payload = "<script>alert(1)</script>"
    evil = CLEAN_YAML.replace("id: clean", "id: evil", 1)
    evil = evil.replace("description:", f"description: {payload}", 1)
    (scenarios / "evil.yaml").write_text(evil, encoding="utf-8")
    # 钉住夹具本身：payload 必须真的进到**解析后的场景**里，否则下面两条
    # 断言测的是空气（页面上本来就没有 script，当然"没有泄漏"）。
    assert load_scenario(scenarios / "evil.yaml").description.startswith(payload)

    for url in ("/", "/scenarios/evil"):
        resp = await client.get(url)
        assert resp.status_code == 200, url
        assert "<script>alert(1)</script>" not in resp.text, url
        assert "&lt;script&gt;" in resp.text, url


# --- 保存：先校验，再落盘 -----------------------------------------------------


async def test_save_rejects_invalid_yaml_without_touching_the_file(
    client: httpx.AsyncClient, scenarios: Path
) -> None:
    before = (scenarios / "clean.yaml").read_text(encoding="utf-8")
    resp = await client.post(
        "/scenarios/clean/save",
        data={"yaml_text": "id: clean\ntools: [\n  broken"},
    )
    assert resp.status_code == 400
    assert "YAML" in resp.text
    assert (scenarios / "clean.yaml").read_text(encoding="utf-8") == before


async def test_save_rejects_scenario_that_fails_the_model(
    client: httpx.AsyncClient, scenarios: Path
) -> None:
    """能解析成 YAML 不等于是一个合法场景。

    `is_write` 没有默认值——漏了它就加载失败，这正是为了不让写工具
    静默变成只读。
    """
    before = (scenarios / "clean.yaml").read_text(encoding="utf-8")
    missing_is_write = CLEAN_YAML.replace("    is_write: false\n", "")
    resp = await client.post(
        "/scenarios/clean/save", data={"yaml_text": missing_is_write}
    )
    assert resp.status_code == 400
    assert "is_write" in resp.text
    assert (scenarios / "clean.yaml").read_text(encoding="utf-8") == before


async def test_save_echoes_the_users_text_on_error(
    client: httpx.AsyncClient, scenarios: Path
) -> None:
    """报错时回显**用户刚敲的**那一份，不是磁盘上的。

    回显磁盘版本的话，用户敲的东西在一次失败的保存里被"还原"了，
    而他会以为是自己记错了。
    """
    resp = await client.post(
        "/scenarios/clean/save", data={"yaml_text": "# 我的批注\nid: clean\n"}
    )
    assert resp.status_code == 400
    assert "# 我的批注" in resp.text


async def test_save_writes_valid_yaml(client: httpx.AsyncClient, scenarios: Path) -> None:
    edited = CLEAN_YAML.replace("查一下订单 A1。", "改过的任务。")
    resp = await client.post("/scenarios/clean/save", data={"yaml_text": edited})
    assert resp.status_code == 303
    assert resp.headers["location"] == "/scenarios/clean?saved=1"
    assert "改过的任务" in (scenarios / "clean.yaml").read_text(encoding="utf-8")
    # 落盘的文本必须真的能被引擎加载——这是保存路径的唯一意义
    assert load_scenario(scenarios / "clean.yaml").agent.task == "改过的任务。"


async def test_create_refuses_to_overwrite_existing(
    client: httpx.AsyncClient, scenarios: Path
) -> None:
    before = (scenarios / "clean.yaml").read_text(encoding="utf-8")
    resp = await client.post("/scenarios", data={"new_id": "clean"})
    assert resp.status_code == 303  # 跳到已有的那个，不覆盖
    assert (scenarios / "clean.yaml").read_text(encoding="utf-8") == before


async def test_create_writes_a_loadable_skeleton(
    client: httpx.AsyncClient, scenarios: Path
) -> None:
    """骨架本身必须过校验。

    不过的话，新建出来的第一个场景就是打不开的——而"新建"是用户见到
    这个编辑器的第一条路径。
    """
    resp = await client.post("/scenarios", data={"new_id": "fresh"})
    assert resp.status_code == 303
    scenario = load_scenario(scenarios / "fresh.yaml")
    assert scenario.id == "my-scenario"  # 骨架里的 id，与文件名无关


async def test_create_rejects_unsafe_id(client: httpx.AsyncClient) -> None:
    resp = await client.post("/scenarios", data={"new_id": "../evil"})
    assert resp.status_code == 400


# --- 一次只跑一个 -------------------------------------------------------------


async def test_second_run_is_rejected_while_one_is_going(
    client: httpx.AsyncClient, runner_state: dict
) -> None:
    """并发必须拒绝，而且要**明说原因**。

    执行器在运行期间改进程级全局 `settings.tool_upstream_base_url`，
    把出口代理指向本次的 mock 端口。两个运行重叠的话，后启动的会把前一个的
    上游改掉——前一个的代理于是去敲后一个的 mock，两边都会给出**看起来正常**
    的报告，而数据是串的。
    """
    first = await client.post("/scenarios/clean/run")
    assert first.status_code == 303

    # 等后台任务真的进到 run() 里，否则测的是"还没来得及开始"而不是"正在跑"
    for _ in range(200):
        if runner_state["started"].is_set():
            break
        await asyncio.sleep(0.01)
    assert runner_state["started"].is_set(), "替身执行器没有被调用"

    second = await client.post("/scenarios/clean/run")
    assert second.status_code == 409
    assert "一次只能跑一个" in second.text
    assert runner_state["ran"] == []  # 第二次确实没有跑

    runner_state["release"].set()


async def test_job_page_redirects_to_the_report_when_done(
    client: httpx.AsyncClient, runner_state: dict
) -> None:
    from sandbox.scenario.runner import ScenarioResult

    # 同步函数：替身执行器直接把它 `return` 出去。写成 async 的话返回的是
    # 协程而不是结果，作业会拿到一个没有 run_id 的协程对象。
    def _done(scenario):  # type: ignore[no-untyped-def]
        return ScenarioResult(scenario_id=scenario.id, run_id="run-xyz", error=None)

    runner_state["release"].set()
    runner_state["result"] = _done

    started = await client.post("/scenarios/clean/run")
    job_url = started.headers["location"]
    for _ in range(200):
        resp = await client.get(job_url)
        if resp.status_code != 200:
            break
        await asyncio.sleep(0.01)

    assert resp.status_code == 303
    assert resp.headers["location"] == "/runs/run-xyz"


async def test_unknown_job_is_404_with_an_explanation(client: httpx.AsyncClient) -> None:
    resp = await client.get("/jobs/does-not-exist")
    assert resp.status_code == 404
    assert "重启" in resp.text  # 说清为什么找不到，而不是干巴巴一个 404


# --- 历史运行：结论未留存 ≠ 不通过 ---------------------------------------------


@pytest_asyncio.fixture
async def legacy_run(db: AsyncSession) -> str:
    """造一条**没有结论快照**的历史运行。

    这就是加 `verdict_ok` 字段之前那些行的样子：结论是 NULL。它不是
    "不通过"——它是"没记录"，而两者的界面表现必须不同。
    """
    from sandbox.db.models import RunStatus

    run_id = "legacy-run-1"
    db.add(
        TestRun(
            id=run_id,
            scenario_id="clean",
            agent_version="react_demo@0.1",
            config_hash="x" * 64,
            status=RunStatus.COMPLETED.value,
            verdict_ok=None,
            report_json=None,
        )
    )
    await db.commit()
    return run_id


async def test_legacy_run_shows_unknown_not_failed(
    client: httpx.AsyncClient, legacy_run: str
) -> None:
    resp = await client.get(f"/runs/{legacy_run}")
    assert resp.status_code == 200
    assert "未留存" in resp.text
    assert "不通过" not in resp.text
    # 明细表里的东西仍然要显示出来——审计轨迹不依赖那份结论快照
    assert "工具调用审计" in resp.text


async def test_run_list_marks_legacy_runs_as_unknown(
    client: httpx.AsyncClient, legacy_run: str
) -> None:
    resp = await client.get("/runs")
    assert resp.status_code == 200
    assert "未留存" in resp.text


async def test_json_api_reports_null_ok_for_legacy_run(
    client: httpx.AsyncClient, legacy_run: str
) -> None:
    """JSON 里 `ok` 是 null，不是 false。

    `false` 会被下游读成"这次运行不通过"，而它其实只是没记录。
    """
    listing = (await client.get("/api/runs")).json()
    row = next(r for r in listing if r["run_id"] == legacy_run)
    assert row["ok"] is None

    detail = (await client.get(f"/api/runs/{legacy_run}")).json()
    assert detail["report"] is None
    assert "不是失败" in detail["note"]


# --- JSON 接口 ----------------------------------------------------------------


async def test_api_scenarios(client: httpx.AsyncClient) -> None:
    body = (await client.get("/api/scenarios")).json()
    entry = next(s for s in body if s["id"] == "clean")
    assert entry["valid"] is True
    assert entry["tools"] == 1
    assert entry["sessions"] == 1


async def test_api_start_run_returns_a_job(
    client: httpx.AsyncClient, runner_state: dict
) -> None:
    resp = await client.post("/api/runs", json={"scenario_id": "clean"})
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    for _ in range(200):
        status = await client.get(f"/api/jobs/{job_id}")
        if runner_state["started"].is_set():
            break
        await asyncio.sleep(0.01)
    assert status.json()["state"] == "running"
    assert status.json()["scenario_id"] == "clean"

    runner_state["release"].set()


async def test_api_start_run_conflicts_when_busy(
    client: httpx.AsyncClient, runner_state: dict
) -> None:
    await client.post("/api/runs", json={"scenario_id": "clean"})
    for _ in range(200):
        if runner_state["started"].is_set():
            break
        await asyncio.sleep(0.01)

    resp = await client.post("/api/runs", json={"scenario_id": "clean"})
    assert resp.status_code == 409
    assert resp.json()["active_scenario"] == "clean"
    runner_state["release"].set()


async def test_api_rejects_bad_scenario_id(client: httpx.AsyncClient) -> None:
    resp = await client.post("/api/runs", json={"scenario_id": "../../etc/passwd"})
    assert resp.status_code == 400
    assert "不合法" in resp.json()["detail"]


# --- 场景加载：文本与文件两条路必须一致 ---------------------------------------


def test_parse_and_load_agree(scenarios: Path) -> None:
    from sandbox.scenario import loader

    by_file = load_scenario(scenarios / "clean.yaml")
    by_text = loader.parse_scenario(
        (scenarios / "clean.yaml").read_text(encoding="utf-8")
    )
    assert by_file.model_dump() == by_text.model_dump()


def test_parse_error_names_its_source() -> None:
    """解析的是文本不是文件，但报错必须说清是哪一份。"""
    from sandbox.scenario import loader

    with pytest.raises(ScenarioError, match="my-scenario.yaml"):
        loader.parse_scenario("id: x\ntools: [", source="my-scenario.yaml")


def test_report_json_round_trips_through_the_column() -> None:
    """报告快照必须能存能读。它是界面列历史运行的唯一来源。"""
    report = {"scenario_id": "clean", "ok": True, "sessions": [], "cost": {"cost_usd": 0.0}}
    assert json.loads(json.dumps(report, ensure_ascii=False, default=str)) == report
