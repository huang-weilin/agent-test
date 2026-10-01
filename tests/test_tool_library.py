"""工具库：引用它的规矩，以及那些规矩真的会被执行。

背景是量出来的：29 份场景 2239 行里，660 行是工具声明；55 条声明去重后只有 14 种，
其中 `get_order` 被一字不差地抄了 21 遍，`create_payment` 13 遍。**去样板做完了
不算完**——没有东西盯着的话，下一份场景会把样板抄回来，而抄回来的文件和现在这份
长得一模一样，谁也不会发现。所以这里有两道闸门：

  * 名字在库里的工具，场景里必须写 `use:` 引用；
  * 同一份完整声明在两份以上场景里内联出现，就该进库（新长出来的样板也拦）。

两道都读**原始 YAML**，不读加载器解析之后的声明：解析之后每一条都带着名字、字段
齐全，从那里看不出它到底是引用来的还是抄回来的——而"抄回来的"正是要拦的那件事。
这跟 `tests/test_reason_codes.py`、`tests/test_tool_schema_liveness.py` 是同一个
套路：去文件里找，找不到就必须有一行写下的理由。

剩下的几条是给库本身的：改库里一个字段，用到它的场景要跟着变（含指纹）。那条是
"装上了但没接线"的探针——库要是没接进加载链路，只有这条会红，其余测试一条都不会
红，因为每份场景自己都把声明写全了。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from sandbox.scenario import identity, library
from sandbox.scenario.loader import SKELETON_TEXT, ScenarioError, parse_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = REPO_ROOT / "scenarios"

#: 探针用的最小工具声明。规矩（合并、引用）都不依赖某一份具体的库，所以这里
#: 自己造一份——真实库里的内容会变，而这些规矩不该跟着变。
GET_ORDER: dict[str, Any] = {
    "name": "get_order",
    "description": "按订单号查询订单详情。只读，无副作用。",
    "is_write": False,
    "parameters": {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
}


def _raw_tools(path: Path) -> list[dict[str, Any]]:
    """场景文件里**写下来的** tools，每一条原样。"""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [entry for entry in (payload.get("tools") or []) if isinstance(entry, dict)]


def _write_library(path: Path, tools: list[dict[str, Any]]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(yaml.safe_dump({"tools": tools}, allow_unicode=True, sort_keys=False))
    return path


def _point_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tools: list[dict[str, Any]]
) -> None:
    """把库指到这一份临时的。

    探针不读真库里的内容：那些内容会变，而这里要证的规矩（引用、覆盖、名字）
    不该跟着变。
    """
    monkeypatch.setattr(library, "LIBRARY_PATH", _write_library(tmp_path / "tools.yaml", tools))


def _scenario(tools: str) -> str:
    """一份最小的、能被加载的场景；`tools:` 那一段由调用方给。"""
    return (
        "id: probe\n"
        "description: 探针场景。\n"
        "agent:\n"
        "  version: react_demo@0.1\n"
        "  task: 把订单 A1 处理掉。\n"
        "  model: stub-model\n"
        "tools:\n"
        f"{tools}\n"
        "sessions:\n"
        "  - name: 第一次\n"
        "    script:\n"
        "      - content: 完成了。\n"
    )


# --- 两道闸门 ---------------------------------------------------------------


def test_库里的名字在场景里必须写成引用() -> None:
    known = set(library.load_library())
    offenders = [
        f"{path.name} 第 {index + 1} 条：{tool.get('name')!r}"
        for path in sorted(SCENARIO_DIR.glob("*.yaml"))
        for index, tool in enumerate(_raw_tools(path))
        if "use" not in tool and tool.get("name") in known
    ]
    assert not offenders, (
        "这些工具的名字已经在库里了，场景里却整段抄了一遍：\n  "
        + "\n  ".join(offenders)
        + f"\n\n改成 `- use: <名字>`，要改哪个字段就在它下面补哪一行"
        f"（库：{library.LIBRARY_PATH.relative_to(REPO_ROOT).as_posix()}）。\n"
        "抄一遍的代价不只是啰嗦：库里改了，它不会跟着变，而读场景的人会以为"
        "它就是库里那个工具。"
    )


def test_两处以上的同一份声明该进库了() -> None:
    """这条拦的是**新长出来的**样板：刚写的两个场景各自抄了一份同样的声明。"""
    seen: dict[str, list[str]] = {}
    for path in sorted(SCENARIO_DIR.glob("*.yaml")):
        for tool in _raw_tools(path):
            if "use" in tool:
                continue
            blob = json.dumps(tool, sort_keys=True, ensure_ascii=False)
            seen.setdefault(blob, []).append(f"{path.name}:{tool.get('name')}")
    repeated = {blob: users for blob, users in seen.items() if len(users) > 1}
    detail = [
        f"{json.loads(blob)['name']}：{', '.join(users)}"
        for blob, users in sorted(repeated.items())
    ]
    assert not detail, (
        "同一份工具声明在多份场景里各写了一遍：\n  "
        + "\n  ".join(detail)
        + f"\n\n把它挪进 {library.LIBRARY_PATH.relative_to(REPO_ROOT).as_posix()}"
        "（名字已在那儿的话就删掉场景里这一份，换成 `- use: <名字>` 加覆盖）。\n"
        "整段抄的声明是漂移的温床：写这份库的时候就量到过——同一个 "
        "create_payment 的说明在仓库里有两个版本，哪一版取决于当初从哪个文件复制的。"
    )


# --- 库真的接在链路上 -------------------------------------------------------


def test_改库里的一个字段_用到它的场景跟着变(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """库改了，场景跟着变，指纹也跟着变。

    **这是"装上了但没接线"的探针。** 库要是没接进加载链路，只有这条会红：
    其余测试一条都不会红，因为每份场景自己都把声明写全了——它们看起来完全正常。
    """
    _point_at(tmp_path, monkeypatch, [GET_ORDER])
    stock = parse_scenario(_scenario("  - use: get_order"))
    assert stock.tools[0].description == GET_ORDER["description"], "没展开：读到的不是库里的那一份"

    bumped = dict(GET_ORDER, description="改过的说明，只可能来自库的这一版。")
    _point_at(tmp_path, monkeypatch, [bumped])
    changed = parse_scenario(_scenario("  - use: get_order"))

    assert changed.tools[0].description == bumped["description"]
    assert identity.sections(changed)["tools"] != identity.sections(stock)["tools"], (
        "库换了而配置指纹没动。指纹必须取**解析之后**的工具声明（identity.sections）——"
        "取不到的话，改库这件事在两次运行的对比里完全看不出来。"
    )


def test_没写到的字段跟着库走_写了的字段整块替换(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """覆盖是整块字段的替换，不做深合并。

    深合并会让"这个工具到底是什么"取决于库那边长什么样：库里给 properties 加一个
    键，所有覆盖过 parameters 的场景跟着变，而场景文件上一个字都没动。
    """
    base = {
        "name": "list_orders",
        "description": "分页列出订单。只读，无副作用。",
        "is_write": False,
        "parameters": {
            "type": "object",
            "properties": {"status": {"type": "string"}, "page": {"type": "integer"}},
            "required": ["status"],
        },
    }
    _point_at(tmp_path, monkeypatch, [base])
    scenario = parse_scenario(
        _scenario(
            "  - use: list_orders\n"
            "    parameters:\n"
            "      type: object\n"
            "      properties: {page: {type: integer}}\n"
            "      required: [page]"
        )
    )
    tool = scenario.tools[0]
    assert set(tool.parameters["properties"]) == {"page"}, "深合并了：库里那个 status 不该留下"
    assert tool.parameters["required"] == ["page"]
    assert tool.description == base["description"], "没覆盖的字段该跟着库走"


def test_新建出来的骨架本身就守着这些规矩() -> None:
    """界面"新建"写出来的第一份文件，不许是当场违了规矩的那一份。

    骨架早先是 `model_validate` → `dump_scenario` 出来的，落盘的是**展开之后**
    的声明——用户一个字都没改，`test_库里的名字在场景里必须写成引用` 就红了。
    """
    payload = yaml.safe_load(SKELETON_TEXT)
    assert payload["tools"] == [{"use": "get_order"}]
    scenario = parse_scenario(SKELETON_TEXT, source="<新建场景骨架>")
    assert scenario.tools[0].name == "get_order"
    assert (
        scenario.tools[0].description
        == library.load_library()["get_order"]["description"]
    )


# --- 引用断了的时候说什么 ---------------------------------------------------


def test_引用库里没有的名字时_报的是那个名字(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _point_at(tmp_path, monkeypatch, [GET_ORDER])
    with pytest.raises(ScenarioError) as excinfo:
        parse_scenario(_scenario("  - use: get_ordr"), source="probe.yaml")

    message = str(excinfo.value)
    assert "get_ordr" in message, "报的必须是被引用错的那个名字"
    assert "get_order" in message, "要顺带说库里有的是什么"
    assert "probe.yaml" in message, "要带上是哪一份场景"


def test_库不在场时_没用引用的场景照跑(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """把一份不带 `use:` 的场景单独拷出去跑，旁边没有库也该跑得起来。"""
    monkeypatch.setattr(library, "LIBRARY_PATH", tmp_path / "不在这儿.yaml")
    scenario = parse_scenario(
        _scenario("  - name: get_order\n    description: 查单。\n    is_write: false")
    )
    assert scenario.tools[0].name == "get_order"


def test_库不在场时_用了引用的场景要说清是哪个文件不在(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "不在这儿.yaml"
    monkeypatch.setattr(library, "LIBRARY_PATH", missing)
    with pytest.raises(ScenarioError, match=re.escape(str(missing))):
        parse_scenario(_scenario("  - use: get_order"), source="probe.yaml")


def test_库里两条同名_加载就拒(tmp_path: Path) -> None:
    """字典里同名后一条顶掉前一条，静默。所以名字要自己数一遍。"""
    path = tmp_path / "tools.yaml"
    path.write_text(
        "tools:\n"
        "  - name: get_order\n"
        "    description: 第一次声明。\n"
        "    is_write: false\n"
        "  - name: get_order\n"
        "    description: 第二次声明，字段还不一样。\n"
        "    is_write: true\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(library.LibraryError, match="get_order"):
        library.load_library(path)


def test_库里的条目本身也要是合法声明(tmp_path: Path) -> None:
    """没有场景用它的那条也要拦：错字躺在库里，谁也不会发现。"""
    path = _write_library(tmp_path / "tools.yaml", [{"name": "get_order", "description": "查单。"}])
    with pytest.raises(library.LibraryError, match="is_write"):
        library.load_library(path)


def test_库文件形状不对时说得清要什么(tmp_path: Path) -> None:
    path = tmp_path / "tools.yaml"
    path.write_text("- get_order\n", encoding="utf-8", newline="\n")
    with pytest.raises(library.LibraryError, match="tools"):
        library.load_library(path)


def test_不许拿name覆盖引用的名字(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`use: get_order` 的意思就是"这个工具是库里的 get_order"，改名让这句话不成立。"""
    _point_at(tmp_path, monkeypatch, [GET_ORDER])
    with pytest.raises(ScenarioError) as excinfo:
        parse_scenario(_scenario("  - use: get_order\n    name: get_orders"))

    message = str(excinfo.value)
    assert "get_order" in message and "get_orders" in message
