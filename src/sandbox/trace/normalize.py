"""工具调用的规范化、指纹与相似度。

这是幂等检测与循环检测的共同地基。需求文档实现篇用 `_hash(tool_name, args)`
一处哈希同时承担"判重复"和"判相似"两个职责——**哈希只能判相同，判不了相似**。
而参数微调导致的循环（页码 1→2→3）恰恰是最常见的失控形态。

所以这里分两层（见 docs/detectors.md）：

    严格层：规范化 → 结构哈希 → 完全相同判定
    近似层：关键字段 → 编辑距离 / 数值邻近 → 加权得分

**两层各自负责什么，边界要划清楚：**

近似层回答的是"这是不是同一次调用"，不是"这是不是一个循环"。它要抓的是
*语义相同但字面不同*——`amount=100` 与 `amount=100.0`、模型重新生成时多出的
空白或近义字符串。这类差异严格层抓不到。

而"页码 1→2→3"是**语义上确实不同**的请求，近似层就该判它不同（得分 0.75，
低于 0.8 的等价阈值），它是不是循环要由**序列层**回答：看 `differing_fields`
是不是恰好只有页码、且页码单调递增、且会终止。所以 `differing_fields` 不是
调试信息，它是序列检测器的输入。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

# 这些字段每次调用都不同，但语义上不构成"不同的调用"。
# 不剔除的话，同一个操作每次都算出不同指纹，幂等检测直接失效。
DEFAULT_NOISE_FIELDS: frozenset[str] = frozenset(
    {
        "timestamp",
        "ts",
        "request_id",
        "requestid",
        "trace_id",
        "traceparent",
        "idempotency_key",  # 幂等键本身要单独看，不参与指纹
        "nonce",
        "correlation_id",
        "session_id",
    }
)


def canonical_json(value: Any) -> str:
    """稳定序列化：键排序、无多余空白。保证同值同串。"""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def normalize_args(
    args: dict[str, Any] | None,
    noise_fields: frozenset[str] = DEFAULT_NOISE_FIELDS,
    *,
    casefold_keys: bool = False,
) -> dict[str, Any]:
    """剔除噪声字段并递归规范化。"""
    if not args:
        return {}

    out: dict[str, Any] = {}
    for key, value in args.items():
        k = key.lower() if casefold_keys else key
        if k.lower() in noise_fields:
            continue
        if isinstance(value, dict):
            out[k] = normalize_args(value, noise_fields, casefold_keys=casefold_keys)
        elif isinstance(value, str):
            # 去首尾空白：模型经常生成 "  A1" 这种，语义上相同
            out[k] = value.strip()
        else:
            out[k] = value
    return out


def structural_hash(tool_name: str, normalized_args: dict[str, Any]) -> str:
    """严格层哈希。规范化后完全相同 → 哈希相同。"""
    payload = f"{tool_name}\x00{canonical_json(normalized_args)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprint(tool_name: str, normalized_args: dict[str, Any]) -> str:
    """跨会话幂等检测用的指纹。

    目前与 structural_hash 同构，但**刻意分开命名**：将来若要给指纹加入
    "忽略数量字段"之类的业务规则（比如同订单不同金额仍算重复下单），
    改这里即可，不必动摇严格层的语义。
    """
    return structural_hash(tool_name, normalized_args)


# --- 近似层 ---------------------------------------------------------------


def _levenshtein_ratio(a: str, b: str) -> float:
    """归一化编辑距离相似度，1.0 表示完全相同。"""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    if la > 200 or lb > 200:  # 超长串退化为前缀比较，避免 O(n*m) 爆炸
        return 1.0 if a[:200] == b[:200] else 0.0
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (a[i - 1] != b[j - 1]),
            )
        prev = cur
    return 1.0 - prev[lb] / max(la, lb)


def _numeric_similarity(a: float, b: float) -> float:
    """数值邻近度。0 与 0 视为相同；用一个小的绝对容差避免除零。"""
    if a == b:
        return 1.0
    denom = max(abs(a), abs(b), 1e-9)
    return max(0.0, 1.0 - abs(a - b) / denom)


@dataclass(frozen=True)
class SimilarityResult:
    score: float  # 0..1
    compared_fields: tuple[str, ...]
    differing_fields: tuple[str, ...]

    @property
    def is_similar(self) -> bool:
        return self.score >= 0.8


def _key_similarity(a: Any, b: Any) -> float:
    if isinstance(a, bool) or isinstance(b, bool):
        return 1.0 if a is b else 0.0
    if isinstance(a, int | float) and isinstance(b, int | float):
        return _numeric_similarity(float(a), float(b))
    if isinstance(a, str) and isinstance(b, str):
        return _levenshtein_ratio(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a) | set(b)
        if not keys:
            return 1.0
        return sum(_key_similarity(a.get(k), b.get(k)) for k in keys) / len(keys)
    return 1.0 if a == b else 0.0


def args_similarity(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    key_weights: dict[str, float] | None = None,
) -> SimilarityResult:
    """近似层：判断两组规范化参数有多相似。

    key_weights 允许场景声明哪些字段更重要（例如订单号比页码重要得多）。
    默认等权重。
    """
    keys = sorted(set(a) | set(b))
    if not keys:
        return SimilarityResult(score=1.0, compared_fields=(), differing_fields=())

    weights = key_weights or {}
    total_w = 0.0
    total_score = 0.0
    differing: list[str] = []

    for k in keys:
        w = weights.get(k, 1.0)
        s = _key_similarity(a.get(k), b.get(k))
        total_w += w
        total_score += w * s
        if s < 1.0:
            differing.append(k)

    return SimilarityResult(
        score=total_score / total_w if total_w else 1.0,
        compared_fields=tuple(keys),
        differing_fields=tuple(differing),
    )


def looks_like_write(tool_name: str, declared_writes: set[str] | None = None) -> bool:
    """推断是否为写操作。

    **优先使用场景显式声明**（declared_writes）。启发式只是兜底——
    猜错方向的代价不对称：把写当成读会漏掉重复扣款告警。
    所以默认偏保守：无法确定时按"写"处理。
    """
    if declared_writes is not None:
        return tool_name in declared_writes

    name = tool_name.lower()
    read_markers = ("get", "list", "search", "query", "fetch", "read", "find", "lookup", "check")
    is_read = any(re.match(rf"^{m}[_\W]", name) or name == m for m in read_markers)
    return not is_read
