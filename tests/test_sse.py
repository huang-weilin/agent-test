"""累计器：把一串帧读成一次调用。

这个文件的重点不是"函数能跑"，而是**哪些错法是安静的**。累计器错起来全都不
指向自己：漏拼一段工具参数，Agent 那边报"参数不是合法 JSON"，看起来像被测
对象的缺陷；漏读最后一帧 usage，成本记成 0，报告显示这次运行不要钱。两种都
不会有人回来查这里，所以每条都得有一句盯着它的断言。
"""

from __future__ import annotations

import json

from sandbox.gateway.sse import (
    DONE,
    StreamAccumulator,
    chunks_of,
    done_frame,
    frame_of,
    parse_frame,
)

RESPONSE = {
    "id": "chatcmpl-abc",
    "object": "chat.completion",
    "created": 1700,
    "model": "stub-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "create_payment",
                            "arguments": '{"order_id": "A1", "amount": 199}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49},
}


def feed(chunks: list[dict]) -> StreamAccumulator:
    acc = StreamAccumulator()
    for chunk in chunks:
        acc.feed(chunk)
    return acc


def test_a_stream_round_trips_back_to_the_response_it_came_from() -> None:
    """拆帧 → 累计 → 还原，必须和原来那份响应一模一样。

    这是整个模块的主性质。它一句话盯住了三件事：正文按顺序拼、工具参数按
    **分片**拼、用量取到手里。少了这一条，上面每一项都要单独写测试，而且
    写的人只会写自己想到的那种错法。
    """
    acc = feed(chunks_of(RESPONSE))
    back = acc.to_response()

    assert back["choices"][0]["message"] == RESPONSE["choices"][0]["message"], (
        "拼回来的消息和原响应不一致——要么正文顺序错了，要么工具调用的参数"
        "分片没有按顺序拼上"
    )
    assert back["choices"][0]["finish_reason"] == "tool_calls"
    assert back["usage"] == RESPONSE["usage"], (
        "用量没读到。它是结算唯一信得过的数，读不到就会退回估算（估算本身有"
        "另一条测试盯着，但这条盯的是「明明发了却没读」）"
    )
    assert back["model"] == "stub-model"


def test_a_reply_with_no_tool_calls_keeps_its_text() -> None:
    """纯正文的响应：拼出来的正文要一字不差，且**不**凭空长出一个 tool_calls。"""
    response = {
        "id": "c",
        "created": 1,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "订单已处理。"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }
    message = feed(chunks_of(response)).to_response()["choices"][0]["message"]

    assert message["content"] == "订单已处理。"
    assert "tool_calls" not in message, (
        "没有工具调用时不该出现这个键：下游按「有没有 tool_calls」判断这一轮"
        "是要调工具还是收工，凭空多一个空表会让它走进工具分支"
    )


def test_the_frames_are_cut_into_pieces_that_need_joining() -> None:
    """帧是**真的分片**，不是一次发完。

    拆帧的意义就在这里：只认第一帧的实现（或者把 `delta` 当成完整消息的实现）
    必须在这个测试里露出来，而不是等到某个真实上游按 8 个字符一片发的时候。
    """
    chunks = chunks_of(RESPONSE, piece=5)
    argument_frames = [
        chunk
        for chunk in chunks
        if chunk["choices"] and chunk["choices"][0]["delta"].get("tool_calls")
    ]
    assert len(argument_frames) > 2, "参数应当被切成好几帧"

    pieces = [
        fragment
        for chunk in argument_frames
        for call in chunk["choices"][0]["delta"]["tool_calls"]
        for fragment in [call.get("function", {}).get("arguments")]
        if fragment
    ]
    assert "".join(pieces) == '{"order_id": "A1", "amount": 199}', (
        "分片拼起来必须等于完整的参数——拼不上就说明拆的时候丢了东西"
    )
    assert all(len(piece) <= 5 for piece in pieces), "每一片都不该超过声明的长度"


def test_without_the_usage_frame_the_accumulator_says_so() -> None:
    """上游没发 usage 帧时，累计器必须**说没拿到**，而不是给一个 0。

    这一条是结算那一边的开关：`saw_usage` 为假时网关会退回估算并给调用打上
    `usage_estimated`。要是这里静静地返回 0 而不是"没有"，缺失就会被当成
    "这次调用不花 token"——那正是最危险的那种静默。
    """
    without = feed(chunks_of(RESPONSE, include_usage=False))
    assert without.saw_usage is False
    assert without.usage is None

    with_usage = feed(chunks_of(RESPONSE, include_usage=True))
    assert with_usage.saw_usage is True


def test_a_tool_fragment_without_an_index_is_counted_not_guessed() -> None:
    """没有下标的工具分片：宁可少记一条，也不要拼出一条错的。

    拼错的函数名看起来像个正常的函数名（`get_` + `order` 覆盖成 `order`），
    拼错的参数看起来是一段正常的 JSON——它们都会让报告给出一个假的结论。
    """
    acc = StreamAccumulator()
    acc.feed(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"function": {"name": "get_order", "arguments": "{}"}}
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
    )
    assert acc.tool_calls == {}
    assert acc.unmerged_tool_fragments == 1, (
        "拼不起来的碎片必须被数出来：它是「这次累计不完整」唯一的外部信号"
    )


def test_a_tool_name_split_across_frames_is_joined_not_overwritten() -> None:
    """函数名也有可能分片。覆盖着写的话，名字会变成后一半。"""
    acc = StreamAccumulator()
    for piece in ("create_", "payment"):
        acc.feed(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"name": piece}}]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        )
    calls = acc.to_response()["choices"][0]["message"]["tool_calls"]
    assert calls[0]["function"]["name"] == "create_payment"


def test_parse_frame_takes_only_json_data_lines() -> None:
    """SSE 里不是每一行都是数据：心跳、注释、`[DONE]` 都不该进累计器。"""
    assert parse_frame(": keep-alive") is None
    assert parse_frame("") is None
    assert parse_frame("event: ping") is None
    assert parse_frame(f"data: {DONE}") is None
    assert parse_frame("data: {不是 JSON}") is None
    assert parse_frame('data: {"id": "x"}') == {"id": "x"}


def test_a_frame_is_one_line_and_a_record_ends_with_a_blank_line() -> None:
    """编码出来的一帧要能被按行读回来——转发的两端靠这个约定对齐。

    帧里含换行的话，客户端按行解析会把它劈成两半；不带结尾空行的话，两条
    记录会粘成一条。两个错都会让对端解析失败，而错误指向对端的解析器。
    """
    text = frame_of({"id": "x", "choices": []})
    assert text.endswith("\n\n")
    assert len(text.strip().splitlines()) == 1
    assert parse_frame(text) == {"id": "x", "choices": []}
    assert done_frame() == f"data: {DONE}\n\n"


def test_chunks_of_uses_ascii_false_so_chinese_survives() -> None:
    """中文不转义：转义过的帧仍然合法，但人读日志时看到的是 `\\u8ba2\\u5355`。"""
    chunk = chunks_of({"choices": [{"message": {"content": "订单"}, "index": 0}]})[1]
    assert "订单" in json.dumps(chunk, ensure_ascii=False)
    assert "订单" in frame_of(chunk)
