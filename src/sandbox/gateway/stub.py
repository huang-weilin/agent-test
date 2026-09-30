"""脚本化的假上游 LLM。

**为什么需要它：** LLM 是不确定的。如果一开始就接真实模型，那么"检测器没报出问题"
就有两种可能——检测器坏了，或者模型这次没触发那条路径。分不清。

用脚本化上游，每个场景的行为完全确定：脚本说第 3 步重复支付，那就一定在第 3 步
重复支付。于是"检测器报没报出来"就成了唯一变量。

这也顺带解决了成本问题：验证管道不需要烧钱、不需要网络。

脚本格式（场景 JSON 的 `script` 字段）：

    [
      {"tool_calls": [{"name": "create_payment", "arguments": {...}}]},
      {"content": "已完成。"}
    ]

每一步要么产生 tool_calls（驱动 Agent 继续），要么产生 content（终止 ReAct 循环）。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScriptStep:
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    content: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScriptStep:
        return cls(tool_calls=d.get("tool_calls", []), content=d.get("content"))


class ScriptedUpstream:
    """按 run_id 维护脚本游标，逐次返回下一步。"""

    def __init__(self, default_model: str = "stub-model") -> None:
        self.default_model = default_model
        self._scripts: dict[str, list[ScriptStep]] = {}
        self._cursor: dict[str, int] = {}

    def register(self, run_id: str, steps: list[dict[str, Any]]) -> None:
        self._scripts[run_id] = [ScriptStep.from_dict(s) for s in steps]
        self._cursor[run_id] = 0

    def unregister(self, run_id: str) -> None:
        self._scripts.pop(run_id, None)
        self._cursor.pop(run_id, None)

    def register_from_json_file(self, run_id: str, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            self.register(run_id, json.load(f))

    def calls_made(self, run_id: str) -> int:
        return self._cursor.get(run_id, 0)

    def next_response(
        self, run_id: str, model: str | None = None, approx_input_tokens: int = 0
    ) -> dict[str, Any]:
        """构造一个 OpenAI 兼容的 chat.completion 响应。"""
        model = model or self.default_model
        steps = self._scripts.get(run_id)
        idx = self._cursor.get(run_id, 0)

        if steps is None:
            # 未注册脚本 —— 明确报错而不是返回空响应。
            # 返回空响应会让 Agent 静默结束，看起来像"通过"，非常危险。
            raise KeyError(
                f"run {run_id!r} 没有注册脚本。若这是真实上游场景，"
                f"应把 upstream mode 切到真实模式而不是走 stub。"
            )

        if idx < len(steps):
            step = steps[idx]
        else:
            # 脚本耗尽：返回终止消息，避免无限循环。
            # 注意这不等于"通过"——测试应断言脚本被完整消费。
            step = ScriptStep(content="[stub] 脚本已耗尽，强制终止。")

        self._cursor[run_id] = idx + 1

        message: dict[str, Any] = {"role": "assistant", "content": step.content}
        finish_reason = "stop"

        if step.tool_calls:
            message["content"] = step.content  # 通常为 None
            message["tool_calls"] = [
                {
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        # OpenAI 要求 arguments 是 JSON 字符串，不是对象
                        "arguments": json.dumps(tc.get("arguments", {}), ensure_ascii=False),
                    },
                }
                for tc in step.tool_calls
            ]
            finish_reason = "tool_calls"

        # 造一个与真实上游同量级的 usage，让预算与成本链路被真实驱动
        prompt_tokens = max(10, approx_input_tokens or 100 + idx * 20)
        completion_tokens = 15 + 5 * len(step.tool_calls)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


# 进程级单例：网关与测试共用同一份脚本注册表。
scripted_upstream = ScriptedUpstream()
