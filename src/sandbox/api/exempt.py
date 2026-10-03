"""把「这条我认可是误报」写进用户的场景文件。

这个模块只做两件事：把一条 `ExemptionDecl` 渲染成几行 YAML，以及把它**追加**进
那份场景文本里——**其余部分一字不动**。

两件事都是刻意的：

**不重新序列化整份 YAML。** 场景文件是人写的，注释里写着"为什么这么写"，那是
这份文件一半的信息量；`yaml.safe_dump` 往返一遍就是把它们悄悄删掉。同样的理由
写在 `paths.write_scenario_text` 的文档字符串里，这里是它在"改一个字符"这条路上
的兑现。所以插入是**按行**做的，行号由 `yaml.compose` 的标记给出——它只解析出
节点树，不碰原文。

**渲染出来的那几行是纯文本，不经过模型。** 理由里的引号、冒号、井号都靠
YAML 单引号标量兜住（单引号里唯一要转义的字符是单引号自己，写两遍即可），
所以不需要 `safe_dump`，也就不需要担心它给整份文档加的那个 `...` 结尾标记。

**写完必须重新解析一遍再落盘。** 这是全仓统一的那条顺序（`app.save_scenario`
的注释里写着为什么）：插入是一次文本手术，手术做歪了的表现是"文件还在、只是
加载不出来了"，而全部用例都在这些文件里。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import yaml

#: 缩进。和仓库里所有场景文件、以及 `loader.SKELETON_TEXT` 一致。
INDENT = "  "


class ExemptError(ValueError):
    """这条豁免写不进去。**说得清是哪种写不进去**，别让用户去猜。"""


def quote_reason(reason: str) -> str:
    """把理由渲染成一个 YAML 单引号标量。

    单引号标量里唯一需要转义的字符就是单引号本身（写两遍）。这也是为什么
    理由里的冒号、井号、`-` 开头、`true` 这种看着像别的类型的词全都不用管。
    """
    text = reason.strip()
    if not text:
        raise ExemptError(
            "理由不能空着。没有理由的豁免在报告里无法复核，等于永久静音——"
            "下一个人（也可能是三个月后的你）只会看到一条结论不见了，"
            "而没有任何东西说明它为什么不见了。"
        )
    if "\n" in text or "\r" in text:
        # 单引号标量跨行要写成折叠形式，那会把用户的换行读成空格——用户写的
        # 和他以为写进去的就悄悄不一样了。与其替他决定，不如让他写成一行。
        raise ExemptError("理由请写成一行。换行的话，写进去的和你看到的会不一样。")
    return "'" + text.replace("'", "''") + "'"


def render_decl(
    *,
    detector: str,
    reason: str,
    tool: str | None = None,
    match: dict[str, Any] | None = None,
    allow_write: bool = False,
) -> list[str]:
    """一条豁免规则渲染成 YAML 的几行（**不含**文档缩进，调用方自己加）。

    `allow_write` 是刻意设的第二道门：它让"我在静音一次会改世界的告警"变成一个
    必须写下来的动作。所以这里只是照写，**不由模块自己推断**——谁来填这个参数，
    谁就得先让用户明确点头（见 `app.exempt_finding`）。
    """
    lines = [f"- detector: {detector}"]
    if tool is not None:
        lines.append(f"  tool: {tool}")
    lines.append(f"  reason: {quote_reason(reason)}")
    if match:
        rendered = ", ".join(f"{k}: {_plain(v)}" for k, v in sorted(match.items()))
        lines.append(f"  match: {{{rendered}}}")
    if allow_write:
        lines.append("  allow_write: true")
    return lines


def _plain(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return quote_reason(value)
    raise ExemptError(f"match 里放不下 {value!r} 这种值（只认字符串、数字和真假）。")


def append_decl(text: str, lines: Sequence[str]) -> str:
    """把 `lines` 追加进场景文本的 `exemptions:` 底下。**其余部分一字不动。**

    两种情形，实测的分布是 28:1：

    * **没有 `exemptions:` 这个键**（29 份场景里 28 份，骨架也是）——追加到文件
      末尾。放在 `expect:` 后面不影响解析：YAML 映射的键顺序不参与语义。
    * **有一个块状列表**（`loop_exempted.yaml`）——插在它最后一条之后。

    剩下两种写法（`exemptions:` 空着、`exemptions: []`）**明说办不到**而不是猜：
    往一个流式空列表后面接块状条目是坏 YAML，而"把它替换掉"要动用户那一行。
    实测 29 份场景里一份都没有，所以这条路上没有用户会撞见；真撞见了，报错里
    会告诉他改成 `exemptions:` 再来。
    """
    if not lines:
        raise ExemptError("没有要写的东西。")

    block = _find_exemptions(text)
    if block is None:
        body = text if text.endswith("\n") else text + "\n"
        if body.strip():
            body += "\n"
        return body + "exemptions:\n" + "".join(f"{INDENT}{line}\n" for line in lines)

    insert_at, indent = block
    rendered = [f"{indent}{line}" for line in lines]
    src = text.splitlines()
    # 块尾原有的空行由这次插入**接管**：往上吃掉、往下吃掉，然后自己铺一次。
    #
    # 两个方向都要吃。只往上吃的话，新条目下面会跟着原来那行空行**再加**自己
    # 补的那一行，于是文件里出现连续两个空行——解析得出来，但读起来像被谁随手
    # 改过（实测第一版就是这样）。
    while insert_at > 0 and not src[insert_at - 1].strip():
        insert_at -= 1
    tail = insert_at
    while tail < len(src) and not src[tail].strip():
        tail += 1
    merged = src[:insert_at] + rendered + [""] + src[tail:]
    return "\n".join(merged).rstrip("\n") + "\n"


def _find_exemptions(text: str) -> tuple[int, str] | None:
    """找到 `exemptions:` 块末尾该插在哪一行，以及条目该缩进几个空格。

    返回 `None` 表示这份文件里没有这个键。
    """
    try:
        root = yaml.compose(text)
    except yaml.YAMLError as exc:
        raise ExemptError(
            f"这份场景读不出来（{exc}），先把它修回能加载的状态再来加豁免。"
        ) from exc
    if root is None or not isinstance(root, yaml.MappingNode):
        raise ExemptError("这份场景的顶层不是一个映射，看不出该把豁免写在哪。")

    for key, value in root.value:
        if key.value != "exemptions":
            continue
        indent = " " * (key.start_mark.column + len(INDENT))
        if isinstance(value, yaml.SequenceNode) and not value.flow_style:
            # `end_mark` 指的是**下一个键那一行**（实测），所以直接插在那儿。
            return value.end_mark.line, indent
        raise ExemptError(
            f"这份场景里写着 `exemptions:` 但底下不是一份列表"
            f"（第 {key.start_mark.line + 1} 行）。\n"
            f"把它改成单独一行 `exemptions:`（去掉后面的 `[]` 或那一行的其他内容），"
            f"再从报告页点一次就行——我不想替你改那一行。"
        )
    return None
