"""工具库：同一份工具声明，29 份场景里只用写一遍。

**为什么要有它。** 加到二十九份能跑的场景之后，量了一遍 `scenarios/`：2239 行里
有 660 行（29%）是工具声明，而 55 条声明去重之后只有 14 种——`get_order` 被一字
不差地抄了 21 遍，`create_payment` 13 遍。加一个场景，作者要先从别的文件里复制
二十来行，一个字都不能改错。

复制出来的东西不只是啰嗦，它会**漂**。写这份库的时候量到一处真漂了的：
`create_payment` 的 description 在仓库里有两个版本，短的那一版（少一句
「有副作用：」）出现在六份场景里，长的在其余地方——哪一版取决于当初是从哪个
文件复制来的。而 `mock/toolspec.py` 和演示 Agent 里写的都是长的那一版，于是
**同一个工具在仓库里长得不一样**。这件事没有任何人声明过，也没有任何东西在核，
它只是两个文件各自的字面量。所以库里收的是长的那一版：统一的方向是**朝着仓库里
已经存在的那一份**，不是数哪一版抄得多。

**为什么不是 YAML 锚点，也不是 include 整份文件。** 锚点跨不了文件，而"一份场景
一个文件"是这个项目的形状（加用例 = 加文件）。整份 include 又太粗：量出来的差异
几乎全是一两个顶层字段——出口那一族只改 `egress_host`，自主那一族只改
`may_run_unattended`，幂等那一族只改 `enforces_idempotency`。整份引用的话，这些
场景仍要把整段声明抄一遍才能改掉一个字段，去样板就去了个零头。

所以这里给的是**按名字取一份，再覆盖字段**：

    tools:
      - use: get_order
      - use: create_payment
        enforces_idempotency: true

**覆盖是整块字段的替换，不做深合并。** 深合并会让"这个工具到底是什么"取决于库里
那段长什么样：库那边给 `parameters.properties` 加一个键，所有覆盖了 `parameters`
的场景语义跟着变，而场景文件上一个字都没动。浅合并的规则一句话说得完，读场景的
人也能自己算出结果。`name` 不许覆盖——`use: get_order` 的意思就是"这个工具是库里
的 get_order"，允许改名的话这句话就不成立了。

**它不在场景文件的指纹里，但它的效果在。** 指纹取的是**解析之后**的工具声明
（见 `identity.sections`），所以改库里的一行，所有用到它的场景指纹一起变——那句
"配置变了"是真的，而且是对所有受影响的场景一起说的。这也是为什么库要和场景一起
进版本控制、一起评审。

**为什么放在子目录里。** `scenarios/*.yaml` 这个形状在四处都有（界面列表、
受控对照的账本、两道静态闸门），一个不是场景的 `.yaml` 混在目录里，这四处会各自
出问题（列成一个打不开的场景 / 记成没有对照组的独苗 / 被要求交代 reason_code）。
子目录对这四处 glob 都是不可见的。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from sandbox.config import REPO_ROOT
from sandbox.scenario.spec import ToolDecl

#: 库文件的位置。**测试可以改它**（`monkeypatch.setattr(library, "LIBRARY_PATH", …)`），
#: 所以读它的地方一律在调用时取这个模块属性，不写进默认参数。
LIBRARY_PATH = REPO_ROOT / "scenarios" / "_shared" / "tools.yaml"

#: 场景里引用库里某一行的键。
USE_KEY = "use"


class LibraryError(ValueError):
    """工具库不合法，或者场景引用了库里没有的东西。

    与 `loader.ScenarioError` 分开：这个模块不认识"场景文件"这个概念，
    报出来的错由调用方（解析器）加上出处再转述。测试要单独用库时也不必
    先构造一份场景。
    """


def load_library(path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """读库，按工具名索引。

    **每一条都当场过一遍 `ToolDecl`**，不管有没有场景在用它。理由和别处的加载期
    校验一样：库里的条目出错的方式是安静的——它不会报错，只会在某个场景用到它
    的时候让那个场景的加载以一句无关的话失败。没有场景用它的时候更糟：错字躺在
    那儿，谁也不知道。
    """
    where = Path(path) if path is not None else LIBRARY_PATH
    if not where.exists():
        raise LibraryError(
            f"工具库不在：{where}\n"
            f"有场景写了 use: 引用它。要么把库放回去，要么把那一处写成完整的工具声明。"
        )
    try:
        raw = yaml.safe_load(where.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise LibraryError(f"{where} 不是合法的 YAML：{exc}") from exc

    if not isinstance(raw, dict) or not isinstance(raw.get("tools"), list):
        raise LibraryError(
            f"{where} 的形状不对：顶层要有一个 tools:，"
            f"它是一个列表，每一项是一个工具声明。"
        )

    library: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(raw["tools"]):
        at = f"{where} 的第 {index + 1} 条工具"
        name = entry.get("name") if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name.strip():
            raise LibraryError(
                f"{at} 缺 name。库里的每一条都要有名字，场景靠名字引用它。"
            )
        if name in library:
            raise LibraryError(
                f"{at} 的名字 {name!r} 在它前面已经出现过了。\n"
                f"同名两条时，按名字查只会拿到其中一条，而读文件的人看到的是另一条——"
                f"场景到底引用了哪一份，光读库看不出来。"
            )
        try:
            ToolDecl.model_validate(entry)
        except ValidationError as exc:
            raise LibraryError(f"{at}（{name}）不合法：\n{exc}") from exc
        library[name] = entry
    return library


def uses_library(raw_tools: Any) -> bool:
    """这段 `tools` 里有没有 `use:` 引用。

    **没有引用就不读库。** 于是把一份场景文件单独拷出去跑，只要它没有引用，旁边
    没有库文件也照样跑得起来——库不在场时报错的，只有真正用到它的那些场景。
    """
    return isinstance(raw_tools, list) and any(
        isinstance(entry, dict) and USE_KEY in entry for entry in raw_tools
    )


def resolve_tools(
    raw_tools: Any,
    library: Mapping[str, dict[str, Any]],
    *,
    library_path: Path | str,
) -> Any:
    """把带 `use:` 的条目展开成完整的工具声明。

    **纯函数**：库由调用方传进来，这里不碰磁盘——于是"合并的规矩是什么"可以单独
    测，不必先造一个库文件。`library_path` 只用于报错：引用断了的时候，读的人第一
    个想知道的就是该去哪儿找那份库。
    """
    if not isinstance(raw_tools, list):
        # 形状不对不是这一步该报的错。放过去，pydantic 会说得更准（它指得出是哪个
        # 字段、期望什么类型），而这里只能报一句"不是列表"。
        return raw_tools

    resolved: list[Any] = []
    for index, entry in enumerate(raw_tools):
        where = f"tools[{index}]"
        if not isinstance(entry, dict) or USE_KEY not in entry:
            resolved.append(entry)
            continue

        name = entry[USE_KEY]
        if not isinstance(name, str) or not name.strip():
            raise LibraryError(
                f"{where} 的 use 必须是库里的工具名（字符串），"
                f"实际是 {type(name).__name__}。"
            )
        base = library.get(name)
        if base is None:
            raise LibraryError(
                f"{where} 引用了工具库里没有的 {name!r}。\n"
                f"库（{library_path}）里有：{sorted(library) or '（空）'}\n"
                f"名字对不上的引用在这里不拦的话，报出来的是「这个工具缺 is_write」——"
                f"那句话指着别处，而真正的原因是这个拼错的名字。"
            )

        overrides = {key: value for key, value in entry.items() if key != USE_KEY}
        if "name" in overrides:
            raise LibraryError(
                f"{where} 引用了 {name!r}，又把 name 覆盖成了 {overrides['name']!r}。\n"
                f"use: {name} 的意思就是「这个工具是库里的 {name}」，改名会让这句话"
                f"不成立。要的是另一个工具的话，把声明整段写出来（那它也不该叫 "
                f"{name!r} 了）。"
            )

        # 整块字段替换，不做深合并。见模块 docstring。
        merged = {key: value for key, value in base.items() if key != "name"}
        merged.update(overrides)
        merged["name"] = name
        resolved.append(merged)
    return resolved
