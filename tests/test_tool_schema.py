"""工具参数的 Schema 校验：声明的是一份合同，边界上要真的拿它对。

这个文件覆盖两件事，它们的失败方式完全不同，所以要分开测：

1. **这份声明我们能校验吗**（加载场景时）。不认识的写法必须让场景加载失败——
   默默忽略的话，"这一类参数从来没被检查过"在报告上完全看不出来。
2. **这次调用合不合规**（运行时，边界上）。代理侧的行为在 `test_proxy.py`，
   这里只测判定本身。
"""

from __future__ import annotations

import pytest

from sandbox.scenario.loader import ScenarioError, parse_scenario
from sandbox.toolschema import (
    SchemaError,
    check_schema,
    describe,
    kind_of,
    validate_arguments,
)

_PAYMENT = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string"},
        "amount": {"type": "number"},
        "idempotency_key": {"type": "string"},
    },
    "required": ["order_id", "amount"],
}


def _codes(schema, args) -> list[str]:
    return [p.code for p in validate_arguments(schema, args)]


# --- 第二部分：这次调用合不合规 -------------------------------------------


def test_a_call_that_matches_the_declaration_has_nothing_to_report() -> None:
    assert validate_arguments(_PAYMENT, {"order_id": "A1", "amount": 599.0}) == []
    assert validate_arguments(_PAYMENT, {"order_id": "A1", "amount": 599}) == []


def test_a_missing_required_argument_is_reported_with_what_was_given() -> None:
    """报错要能让人当场改对，所以要说清缺什么、以及它要求的是什么类型。

    "少了参数"和"少了参数 amount（数字）"对写场景的人是完全不同的两句话：
    后者不用回去翻声明。
    """
    problems = validate_arguments(_PAYMENT, {"order_id": "A1"})
    assert [p.code for p in problems] == ["missing_required"]
    assert "amount" in problems[0].message
    assert "数字" in problems[0].message
    assert "'order_id'" in problems[0].message, "已给的参数要列出来"


def test_a_type_mismatch_is_reported_with_the_value_in_it() -> None:
    """`amount: "599"` 是这类缺陷里最常见的一种：模型把数字写成了字符串。

    真实 API 会因为它回 400，而一个不核对参数的 mock 会照单扣款——那笔钱在
    报告上和一笔正常的扣款长得一模一样。
    """
    problems = validate_arguments(_PAYMENT, {"order_id": "A1", "amount": "599"})
    assert [p.code for p in problems] == ["type_mismatch"]
    assert "'599'" in problems[0].message
    assert "字符串" in problems[0].message


def test_a_boolean_is_not_a_number() -> None:
    """Python 里 `True` 是 `int` 的实例。

    不特判的话 `{"amount": true}` 会被当成一个合法的数字放行——而它会一路走到
    目标系统上变成一笔金额为 True 的扣款。这类"语言自带的坑"是必须钉住的。
    """
    assert kind_of(True) == "布尔值"
    assert _codes({"type": "number"}, True) == ["type_mismatch"]
    assert _codes({"type": "integer"}, True) == ["type_mismatch"]
    assert _codes({"type": "boolean"}, True) == []


def test_an_integral_float_is_a_valid_integer() -> None:
    """`599.0` 是合法的整数。

    同一个数在 JSON 里写成 `599` 还是 `599.0` 取决于上游怎么序列化，把后者判成
    类型错误是纯粹的噪音——而噪音会让真正的告警被忽略。
    """
    assert validate_arguments({"type": "integer"}, 599.0) == []
    assert _codes({"type": "integer"}, 599.5) == ["type_mismatch"]


def test_an_extra_argument_is_reported_by_default() -> None:
    """**这里刻意和 JSON Schema 的默认值相反。**

    规范说没写 `additionalProperties` 就是允许。但工具参数不是这样工作的：
    多出来的参数在真实系统上要么被静默丢掉、要么让接口报错。而"模型编了一个
    不存在的参数名、值被丢掉、报告全绿"正是最该被抓到的一类缺陷。
    """
    problems = validate_arguments(
        _PAYMENT, {"order_id": "A1", "amount": 599, "currency": "CNY"}
    )
    assert [p.code for p in problems] == ["unexpected_argument"]
    assert "currency" in problems[0].message
    assert "additionalProperties" in problems[0].message, "要说清怎么放宽"


def test_extra_arguments_can_be_allowed_explicitly() -> None:
    """放宽要写下来。不留默认，也不留"算了这次放过"的暗门。"""
    lenient = {**_PAYMENT, "additionalProperties": True}
    assert validate_arguments(lenient, {"order_id": "A1", "amount": 599, "x": 1}) == []


def test_a_wrong_type_does_not_also_report_its_fields() -> None:
    """类型不对就不再往里查。

    对着一个错的类型继续查它的字段，会生成一串衍生出来的噪音，而真正的问题
    只有第一句。报告里堆五行，读的人会以为是五个问题。
    """
    schema = {
        "type": "object",
        "properties": {"order": {"type": "object", "properties": {"id": {"type": "string"}}}},
        "required": ["order"],
    }
    problems = validate_arguments(schema, {"order": "A1"})
    assert [p.code for p in problems] == ["type_mismatch"]


def test_nested_problems_carry_a_path() -> None:
    """路径要能定位到出问题的那一层，否则读的人只知道"参数不对"。"""
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"sku": {"type": "string"}},
                    "required": ["sku"],
                },
            }
        },
        "required": ["items"],
    }
    problems = validate_arguments(schema, {"items": [{"sku": "x"}, {"qty": 2}]})
    # 第二个元素上两处都不对：少了 sku，还多了一个 qty。两处都要报——
    # 合成一句"这个元素的参数不对"就把定位丢了。
    assert [p.path for p in problems] == ["items[1].sku", "items[1].qty"]
    assert "缺少必填参数 sku" in problems[0].message
    assert problems[1].code == "unexpected_argument"


def test_enum_and_range_are_enforced() -> None:
    assert _codes({"type": "string", "enum": ["CNY", "USD"]}, "EUR") == ["enum_mismatch"]
    assert _codes({"type": "number", "minimum": 1}, 0.5) == ["range_mismatch"]
    assert _codes({"type": "number", "maximum": 10}, 11) == ["range_mismatch"]
    assert validate_arguments({"type": "number", "minimum": 1, "maximum": 10}, 5) == []


def test_an_empty_schema_checks_nothing() -> None:
    """空声明 = 没什么可对照的。**这不是漏检，是如实的。**

    `parameters` 有 `{}` 默认值，所以这是"作者没写参数"的正常形态，而不是
    一个需要报警的异常。
    """
    assert validate_arguments({}, {"whatever": object()}) == []


def test_the_message_reads_as_a_sentence_for_a_human() -> None:
    """`'amount' is not of type 'number'` 对写场景的人没有帮助。"""
    text = describe(validate_arguments(_PAYMENT, {"order_id": "A1", "amount": "599"}))
    assert text.startswith("参数 amount")
    assert "应该是数字" in text


# --- 第一部分：这份 schema 我们能校验吗 -----------------------------------


def test_an_unsupported_keyword_fails_loudly_instead_of_being_ignored() -> None:
    """**这是整个模块最要紧的一条测试。**

    `oneOf` 之类我们不会逐条核对的写法，如果被接受，场景作者会以为自己声明了
    一条约束——而边界上从来没检查过它。报告上一切正常，只是有一块地方是空的。
    所以宁可让场景加载失败。
    """
    with pytest.raises(SchemaError, match="oneOf"):
        check_schema({"oneOf": [{"type": "string"}, {"type": "number"}]}, tool="pay")


def test_an_unknown_type_name_fails_at_load_time() -> None:
    """打错类型的名字（`str` / `strng`）是最容易发生的一种笔误。

    放行的话这个字段就永远不被检查，和上面同一种失效。
    """
    with pytest.raises(SchemaError, match="strng"):
        check_schema({"type": "object", "properties": {"a": {"type": "strng"}}}, tool="pay")


def test_a_schema_valued_additional_properties_is_refused() -> None:
    """与其支持一半，不如让人写不下去。"""
    with pytest.raises(SchemaError, match="additionalProperties"):
        check_schema(
            {"type": "object", "additionalProperties": {"type": "string"}}, tool="pay"
        )


def test_the_error_names_the_tool_and_the_path() -> None:
    with pytest.raises(SchemaError) as excinfo:
        check_schema(
            {"type": "object", "properties": {"order": {"type": "object", "$ref": "#/x"}}},
            tool="create_payment",
        )
    message = str(excinfo.value)
    assert "create_payment" in message
    assert "order" in message, "要说清是哪一个参数上出的问题"


# --- 加载场景时就要拦住 ----------------------------------------------------


def _scenario_text(*, parameters: str) -> str:
    return f"""
id: bad-schema
description: 声明了一份我们校验不了的参数 schema。
agent:
  version: react_demo@0.1
  task: 帮我把订单 A1 处理掉。
tools:
  - name: create_payment
    description: 发起支付。
    is_write: true
    parameters:
{parameters}
sessions:
  - name: 第一次
    script:
      - tool_calls: [{{name: create_payment, arguments: {{order_id: A1, amount: 599}}}}]
      - content: 完成了。
"""


def test_a_scenario_with_an_uncheckable_schema_does_not_load() -> None:
    """**失败要发生在加载场景的时候，不是运行的时候。**

    运行时才发现"这个写法我们不认识"，那时唯一能做的就是忽略它——而忽略正是
    要防的事。加载时失败则逼作者当场做一个决定：改用支持的写法，或者把这条
    约束从声明里删掉。
    """
    with pytest.raises(ScenarioError, match="oneOf"):
        parse_scenario(
            _scenario_text(
                parameters=(
                    "      type: object\n"
                    "      oneOf:\n"
                    "        - required: [order_id]\n"
                    "        - required: [amount]\n"
                )
            ),
            source="bad-schema.yaml",
        )


def test_the_loader_error_names_the_file() -> None:
    with pytest.raises(ScenarioError, match=r"bad-schema\.yaml"):
        parse_scenario(
            _scenario_text(parameters="      type: object\n      patternProperties: {}\n"),
            source="bad-schema.yaml",
        )


def test_a_plain_schema_loads_fine() -> None:
    """别把好场景一起拦下。跨字段校验最容易的翻车方式就是误报。"""
    scenario = parse_scenario(
        _scenario_text(
            parameters=(
                "      type: object\n"
                "      properties:\n"
                "        order_id: {type: string}\n"
                "        amount: {type: number}\n"
                "      required: [order_id, amount]\n"
            )
        ),
        source="good.yaml",
    )
    assert scenario.tools[0].parameters["required"] == ["order_id", "amount"]
