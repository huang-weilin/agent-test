"""工具参数的 Schema 校验（功能需求：参数 Schema 校验，P0）。

**为什么这件事必须做。** 场景里 `ToolDecl.parameters` 是一份 JSON Schema，它
描述"目标系统接受什么样的参数"。在这之前，那份 schema 只有一个去处：mock 的
`/ _mock/tools` 列表，供 Agent 自己取用。**它从来没有被用来检查任何一次真实的
调用。**

于是沙箱里的目标系统比真实的那一个宽松：真实 API 会因为少一个必填字段回 400，
我们的 mock 照单全收，甚至照单扣款。那是一种很难发现的失效——`expected_charges`
数的是"扣了几笔"，而**一笔用非法参数扣出来的钱，在报告上和一笔正常的扣款长得
一模一样**。地面真值本身被污染了。

## 为什么是自写的子集校验器，而不是装一个 `jsonschema`

1. **支持的写法必须恰好等于我们能校验的写法。** 第三方库会接受 `oneOf`、`$ref`、
   `patternProperties` 之类我们不会逐条核对的东西，而"场景作者写了 `oneOf`、
   校验器默默忽略、于是这一整类约束从来没被检查过"正是这个项目一直在防的失效：
   报告上一切正常，只是有一块地方是空的。所以这里是**白名单**——
   `SUPPORTED_KEYWORDS` 之外的写法在**加载场景时**就让场景失败，不留到运行时。

2. **报错要是人话。** `'amount' is not of type 'number'` 对写场景的人没有帮助；
   "参数 amount 应该是数字，实际是字符串 '599'" 才有。

代价说清楚：这是一个**子集**，不是 JSON Schema 实现。它不做 `$ref`、不做组合子、
不做字符串正则。要写这些的场景会加载失败，而不是被静默放行。

## 一条刻意偏离 JSON Schema 默认值的地方

`additionalProperties` 的 JSON Schema 默认值是"允许"，这里反过来：**没写就是
不许**。理由是这个字段在目标系统上几乎从来不是"随便传"——多出来的参数要么被
静默丢掉（我们的 mock 就是这样：`create_payment` 只看 `order_id` / `amount` /
`idempotency_key`），要么让接口报错。两种都不该算通过，而"模型编了一个不存在的
参数名、值被丢掉、报告全绿"正是最该被抓到的一类缺陷。要放宽，写
`additionalProperties: true`——**让"我知道我在放宽什么"成为一个写下来的动作**。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: 我们真的会逐条核对的 JSON Schema 关键字。**白名单。**
#:
#: `description` 在列表里只是因为它无害且常用——它是注解，不产生约束。
SUPPORTED_KEYWORDS: frozenset[str] = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "enum",
        "minimum",
        "maximum",
        "items",
        "description",
    }
)

#: JSON Schema 的类型名里我们认识的那些。
TYPE_NAMES: frozenset[str] = frozenset(
    {"object", "string", "number", "integer", "boolean", "array", "null"}
)

#: 一个值的类型用中文怎么说。报告和报错都用它，只有一处实现。
_KIND_LABELS: tuple[tuple[str, str], ...] = (
    ("null", "空值"),
    ("boolean", "布尔值"),
    ("integer", "整数"),
    ("number", "数字"),
    ("string", "字符串"),
    ("array", "数组"),
    ("object", "对象"),
)


class SchemaError(ValueError):
    """场景声明的参数 schema 落在我们能校验的子集之外。加载场景时就该炸。"""


@dataclass(frozen=True)
class ArgumentProblem:
    """一次调用里的一处参数问题。"""

    code: str
    path: str
    message: str

    #: 报告里按 code 分类，文案可以随便改。与 `Expectation.reason_code` 同一个道理。
    def __str__(self) -> str:
        return self.message


def kind_of(value: Any) -> str:
    """一个值是什么类型，用中文说。`True` 是布尔值不是整数——Python 里它是 int 的子类。"""
    for name, label in _KIND_LABELS:
        if _matches(name, value):
            return label
    return type(value).__name__


def _matches(type_name: str, value: Any) -> bool:
    """`value` 是不是 `type_name`。

    **`bool` 必须先于 `int` 判。** Python 里 `True` 是 `int` 的实例，不特判的话
    `{"amount": true}` 会被当成一个合法的数字——而真实 API 会拒绝它。
    """
    if type_name == "null":
        return value is None
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "integer":
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        # JSON Schema 里 1.0 是一个合法的 integer。JSON 解析成 int 还是 float
        # 取决于写法（`599` vs `599.0`），把后者判成类型错误是纯粹的噪音。
        return isinstance(value, float) and value.is_integer()
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, dict)
    return False


def _declared_types(schema: dict[str, Any]) -> tuple[str, ...]:
    declared = schema.get("type")
    if declared is None:
        return ()
    if isinstance(declared, str):
        return (declared,)
    return tuple(declared)


def kind_of_name(type_name: str) -> str:
    """类型名 → 中文。与 `kind_of` 用同一张表，免得两处说法不一致。"""
    for name, label in _KIND_LABELS:
        if name == type_name:
            return label
    return type_name


def _type_phrase(schema: dict[str, Any]) -> str:
    names = _declared_types(schema)
    if not names:
        return "任意类型"
    return " 或 ".join(kind_of_name(n) for n in names)


# --- 第一部分：这份 schema 我们能校验吗（加载场景时）------------------------


def check_schema(schema: dict[str, Any], *, tool: str, path: str = "") -> None:
    """校验一份**声明**是否落在支持子集里。不合法就抛 `SchemaError`。

    调用点是场景加载。放在那里而不是运行时，是因为运行时才发现"这个写法我们不
    认识"时，唯一能做的就是忽略它——而忽略正是要防的事。
    """
    where = f"工具 {tool!r} 的参数 {path or '(根)'}"
    if not isinstance(schema, dict):
        raise SchemaError(f"{where} 必须是一个对象（JSON Schema 的 mapping）")

    unknown = sorted(set(schema) - SUPPORTED_KEYWORDS)
    if unknown:
        raise SchemaError(
            f"{where} 用了不支持的写法：{unknown}。\n"
            f"支持的只有：{sorted(SUPPORTED_KEYWORDS)}。\n"
            f"**这不是实现遗漏，是刻意的**：不认识的写法会被静默忽略，"
            f"而「这一类参数从来没被检查过」在报告上完全看不出来。"
            f"要么改用支持的写法，要么把这条约束从声明里去掉——"
            f"留在那里假装有效是最坏的一种。"
        )

    for name in _declared_types(schema):
        if name not in TYPE_NAMES:
            raise SchemaError(
                f"{where} 的 type 里出现了不认识的类型 {name!r}。"
                f"认识的有：{sorted(TYPE_NAMES)}"
            )

    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, dict):
            raise SchemaError(f"{where} 的 properties 必须是一个对象")
        for prop, sub in properties.items():
            check_schema(sub, tool=tool, path=f"{path}.{prop}" if path else str(prop))

    required = schema.get("required")
    if required is not None and (
        not isinstance(required, list) or not all(isinstance(r, str) for r in required)
    ):
        raise SchemaError(f"{where} 的 required 必须是一个字符串列表")

    additional = schema.get("additionalProperties")
    if additional is not None and not isinstance(additional, bool):
        raise SchemaError(
            f"{where} 的 additionalProperties 只支持 true / false。"
            f"用 schema 描述「多出来的参数长什么样」这一层我们不做——"
            f"与其支持一半，不如让人写不下去。"
        )

    enum = schema.get("enum")
    if enum is not None and (not isinstance(enum, list) or not enum):
        raise SchemaError(f"{where} 的 enum 必须是一个非空列表")

    for bound in ("minimum", "maximum"):
        value = schema.get(bound)
        if value is not None and not isinstance(value, (int, float)):
            raise SchemaError(f"{where} 的 {bound} 必须是一个数")

    items = schema.get("items")
    if items is not None:
        if not isinstance(items, dict):
            raise SchemaError(f"{where} 的 items 必须是一个对象（只支持单一 schema，不支持元组）")
        check_schema(items, tool=tool, path=f"{path}[]")


# --- 第二部分：这次调用合不合规（运行时，边界上）----------------------------


def validate_arguments(
    schema: dict[str, Any], args: Any, *, tool: str = ""
) -> list[ArgumentProblem]:
    """把一次调用的参数和声明的 schema 对一遍，返回全部问题。

    **空 schema = 不校验**，这是显式的：没声明参数就没什么可对照的，而且
    `parameters` 本来就有 `{}` 默认值（手工起一个运行来排查时不会被迫先写 schema）。

    `tool` 只进报错文案，让读的人不必自己回想这是哪个工具。
    """
    if not schema:
        return []
    return _check(schema, args, path="")


def _check(schema: dict[str, Any], value: Any, *, path: str) -> list[ArgumentProblem]:
    problems: list[ArgumentProblem] = []
    # 根上不叫「参数」而叫「调用参数」：报错句子里"参数里多了一个…"读起来像缺了主语。
    where = f"参数 {path}" if path else "调用参数"

    declared = _declared_types(schema)
    if declared and not any(_matches(name, value) for name in declared):
        # 类型不对就不再往下走：对着一个错的类型继续查它的字段，只会生成一串
        # 衍生出来的噪音，而真正的问题只有第一句。
        return [
            ArgumentProblem(
                code="type_mismatch",
                path=path,
                message=(
                    f"{where} 应该是{_type_phrase(schema)}，"
                    f"实际是{kind_of(value)}：{_short(value)}"
                ),
            )
        ]

    if isinstance(value, dict):
        problems.extend(_check_object(schema, value, path=path))
    elif isinstance(value, list):
        items = schema.get("items")
        if items:
            for index, element in enumerate(value):
                problems.extend(
                    _check(items, element, path=f"{path}[{index}]" if path else f"[{index}]")
                )

    enum = schema.get("enum")
    if enum is not None and value not in enum:
        problems.append(
            ArgumentProblem(
                code="enum_mismatch",
                path=path,
                message=f"{where} 只能取 {'、'.join(repr(e) for e in enum)} 之一，"
                f"实际是 {_short(value)}",
            )
        )

    for bound, phrase in (("minimum", "不小于"), ("maximum", "不大于")):
        limit = schema.get(bound)
        # 布尔值不算数（见 `_matches`），非数值也无从比较——两种都直接跳过，
        # 类型不对那件事上面已经报过了，这里再报一次只是同一个问题的回声。
        if limit is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if (bound == "minimum" and value < limit) or (bound == "maximum" and value > limit):
            problems.append(
                ArgumentProblem(
                    code="range_mismatch",
                    path=path,
                    message=f"{where} 应{phrase} {limit}，实际是 {value}",
                )
            )

    return problems


def _check_object(
    schema: dict[str, Any], value: dict[str, Any], *, path: str
) -> list[ArgumentProblem]:
    problems: list[ArgumentProblem] = []
    properties = schema.get("properties") or {}
    where = f"参数 {path}" if path else "调用参数"

    for name in schema.get("required") or []:
        if name not in value:
            sub = properties.get(name)
            wanted = f"（声明要求：{_type_phrase(sub)}）" if sub else ""
            problems.append(
                ArgumentProblem(
                    code="missing_required",
                    path=f"{path}.{name}" if path else name,
                    message=(
                        f"缺少必填参数 {name}{wanted}，已给的参数有 {sorted(value)}"
                    ),
                )
            )

    for name, element in value.items():
        if name in properties:
            problems.extend(
                _check(properties[name], element, path=f"{path}.{name}" if path else name)
            )
        elif schema.get("additionalProperties") is not True:
            # 默认不许（见模块 docstring）。多出来的参数在真实系统上要么被丢掉、
            # 要么让接口报错，两种都不该算通过。
            problems.append(
                ArgumentProblem(
                    code="unexpected_argument",
                    path=f"{path}.{name}" if path else name,
                    message=(
                        f"{where}里多了一个声明中没有的 {name}={_short(element)}。"
                        f"声明里只有 {sorted(properties) or '（无）'}。"
                        f"若目标系统确实接受它，写进 properties，"
                        f"或对整份声明写 additionalProperties: true"
                    ),
                )
            )

    return problems


def _short(value: Any, limit: int = 60) -> str:
    """把值缩到一行里。报告里的一行比完整 repr 有用。"""
    if isinstance(value, str):
        return f"{value!r}"
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def describe(problems: list[ArgumentProblem]) -> str:
    """把若干问题拼成一句给 Agent 和报告看的错误。"""
    return "；".join(p.message for p in problems)
