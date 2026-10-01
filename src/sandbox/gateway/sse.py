"""流式响应：把一串 SSE 帧累计成一次调用。

网关要同时干两件事：**把上游的帧原样转给被测 Agent**（它那边的客户端要边收边
渲染，我们不能攒完再发），以及**自己把这一串帧读成一次调用**（模型名、产出、
用量、结束的理由）。这个模块只做后半件——不碰网络，输入是解好的帧。

**为什么单独成模块。** 累计错的方式全都是安静的：漏掉一段 `arguments` 分片，
Agent 那边收到的是半个 JSON、报一句"工具参数解析失败"，看起来像被测对象的缺陷；
漏掉最后那一帧 usage，成本记成 0，报告显示这次运行不要钱。两种都不指向这里。
所以它得是一个能拿帧序列直接喂、直接断言的纯函数式东西。

**帧的形状**（OpenAI 兼容，`stream: true`）：每一帧是一个 JSON，`choices[0].delta`
里带这一帧新增的内容；工具调用的参数是**字符串分片**，要按下标拼回去；用量不在
正文里，只在**最后一帧**（`choices` 为空）——而且上游只有收到
`stream_options: {"include_usage": true}` 才会发那一帧，所以网关发请求时要替调用方
补上这个开关（见 `app.py`：没有那个开关就永远拿不到权威用量）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

#: SSE 里表示"流结束"的那一帧。它不是 JSON，转发给客户端，但不进累计器。
DONE = "[DONE]"


def parse_frame(line: str) -> dict[str, Any] | None:
    """一帧 SSE 行 → 一个 chunk。解析不了的返回 None（调用方负责照转）。

    看不懂的帧**照转不误**：上游将来加了新字段、发了个我们没见过的帧，客户端
    该收到的还是它。我们只是不把它记进这次调用——少记一条不会让报告说谎，
    丢一帧会让被测 Agent 那边解析失败，而那个失败会指向它。
    """
    text = line.strip()
    if not text or text.startswith(":"):
        return None  # 心跳/注释
    if text.startswith("event:") or text.startswith("id:") or text.startswith("retry:"):
        return None
    if not text.startswith("data:"):
        return None
    body = text[len("data:") :].strip()
    if not body or body == DONE:
        return None
    try:
        chunk = json.loads(body)
    except json.JSONDecodeError:
        return None
    return chunk if isinstance(chunk, dict) else None


def frame_of(chunk: dict[str, Any]) -> str:
    """一个 chunk → 一帧 SSE。`ensure_ascii=False`，中文不转义。"""
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def done_frame() -> str:
    return f"data: {DONE}\n\n"


@dataclass
class StreamAccumulator:
    """把 chunk 序列累计成一份**非流式**响应。

    累计出来的东西和同一次调用的非流式响应对得上（同一个 `to_response()` 形状），
    于是落库、成本、报告那一整条链路不需要知道这次是流式的——除了
    `LlmCall.streaming` 那一个布尔值。
    """

    model: str = "unknown"
    id: str = ""
    created: int = 0
    content: list[str] = field(default_factory=list)
    #: 下标 → 工具调用的碎片。名字与参数都是**分片**，要拼。
    tool_calls: dict[int, dict[str, Any]] = field(default_factory=dict)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    chunks: int = 0
    #: 拼不起来的工具调用分片数（没有下标，不知道该拼到哪一路上去）。
    #: **不为零要说出来**：它意味着这次累计出来的内容是不完整的，而缺的那部分
    #: 在被测 Agent 那边会表现为"参数解析失败"——看起来像它的缺陷。
    unmerged_tool_fragments: int = 0

    def feed(self, chunk: dict[str, Any]) -> None:
        self.chunks += 1
        if chunk.get("model"):
            self.model = chunk["model"]
        if chunk.get("id"):
            self.id = chunk["id"]
        if chunk.get("created"):
            self.created = chunk["created"]
        # 用量只在最后一帧出现（`choices` 为空）。**见到就覆盖**：上游重发一次
        # 用量帧时以后到的为准，而不是把两个数加起来。
        if isinstance(chunk.get("usage"), dict):
            self.usage = chunk["usage"]

        choices = chunk.get("choices") or []
        if not choices:
            return
        choice = choices[0]
        if choice.get("finish_reason"):
            self.finish_reason = choice["finish_reason"]

        delta = choice.get("delta") or {}
        piece = delta.get("content")
        if isinstance(piece, str):
            self.content.append(piece)
        # 非字符串的 content（多模态分片那种写法）有意忽略：今天没有客户端这么
        # 发，硬猜一个拼法会把一段不属于文本的东西拼进正文里。

        for frag in delta.get("tool_calls") or []:
            self._feed_tool_call(frag)

    def _feed_tool_call(self, frag: dict[str, Any]) -> None:
        index = frag.get("index")
        if not isinstance(index, int):
            # 没有下标就没法拼：两路工具调用的分片会被拼到一起，拼出来的是
            # 一段两个函数各一半的 JSON。宁可少记一条，也不要记一条错的。
            self.unmerged_tool_fragments += 1
            return
        slot = self.tool_calls.setdefault(
            index, {"id": "", "type": "function", "name": "", "arguments": []}
        )
        if frag.get("id"):
            slot["id"] = frag["id"]
        if frag.get("type"):
            slot["type"] = frag["type"]
        function = frag.get("function") or {}
        if function.get("name"):
            # 名字也分片的情况极罕见，但真发两次（`"get_"` + `"order"`）时
            # 覆盖会让函数名变成后一半，而它看起来像个正常的名字。
            slot["name"] += function["name"]
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            slot["arguments"].append(arguments)

    @property
    def saw_usage(self) -> bool:
        """上游到底给没给权威用量。

        **这一条决定了这次调用的数字能不能信**：没给的时候 token 数是网关自己
        估的（见 `app.py` 的结算分支），成本那几行必须如实标出来。
        """
        return self.usage is not None

    @property
    def text(self) -> str:
        return "".join(self.content)

    def to_response(self) -> dict[str, Any]:
        """累计结果 → 与同一次调用的非流式响应同形状的一份 dict。"""
        message: dict[str, Any] = {"role": "assistant", "content": self.text or None}
        if self.tool_calls:
            message["content"] = self.text or None
            message["tool_calls"] = [
                {
                    "id": slot["id"],
                    "type": slot["type"],
                    "function": {
                        "name": slot["name"],
                        "arguments": "".join(slot["arguments"]),
                    },
                }
                for _, slot in sorted(self.tool_calls.items())
            ]
        return {
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": self.finish_reason,
                }
            ],
            "usage": self.usage or {},
        }


def chunks_of(
    response: dict[str, Any], *, include_usage: bool = True, piece: int = 8
) -> list[dict[str, Any]]:
    """把一份非流式响应**拆成帧序列**。脚本化上游用它演"真在流"。

    拆法刻意笨：正文按固定长度切、工具参数逐段切。这样被测客户端必须真的做
    拼接才能拿到完整的参数——只认第一帧的实现会在测试里露出来。

    `include_usage` 对应真实上游的 `stream_options`：不发那一帧时，客户端（和
    网关）都拿不到用量。这不是可以忽略的细节，它正是"结算正确性"那一半。
    """
    head = {
        "id": response.get("id", ""),
        "object": "chat.completion.chunk",
        "created": response.get("created", 0),
        "model": response.get("model", "unknown"),
    }
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    chunks: list[dict[str, Any]] = []

    def push(delta: dict[str, Any], finish: str | None = None) -> None:
        chunks.append(
            {
                **head,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
        )

    push({"role": "assistant", "content": ""})
    text = message.get("content") or ""
    for start in range(0, len(text), piece):
        push({"content": text[start : start + piece]})

    for index, call in enumerate(message.get("tool_calls") or []):
        function = call.get("function") or {}
        push(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": call.get("id", ""),
                        "type": call.get("type", "function"),
                        "function": {
                            "name": function.get("name", ""),
                            "arguments": "",
                        },
                    }
                ]
            }
        )
        arguments = function.get("arguments") or ""
        for start in range(0, len(arguments), piece):
            push(
                {
                    "tool_calls": [
                        {
                            "index": index,
                            "function": {"arguments": arguments[start : start + piece]},
                        }
                    ]
                }
            )

    push({}, finish=choice.get("finish_reason") or "stop")
    if include_usage:
        chunks.append({**head, "choices": [], "usage": response.get("usage") or {}})
    return chunks
