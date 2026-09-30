"""规范化是幂等与循环检测的地基，它的错误会直接变成误报或漏报。"""

from __future__ import annotations

from sandbox.trace.normalize import (
    args_similarity,
    fingerprint,
    looks_like_write,
    normalize_args,
    structural_hash,
)


def test_noise_fields_do_not_change_fingerprint() -> None:
    """同一操作带上不同的 request_id/timestamp，必须仍是同一指纹。

    否则每次调用都算出新指纹，幂等检测彻底失效——这是最致命的失效模式。
    """
    a = {"order_id": "A1", "amount": 100, "request_id": "req-1", "timestamp": 111}
    b = {"order_id": "A1", "amount": 100, "request_id": "req-2", "timestamp": 999}
    assert fingerprint("create_payment", normalize_args(a)) == fingerprint(
        "create_payment", normalize_args(b)
    )


def test_whitespace_is_trimmed() -> None:
    """模型常生成 '  A1' 这种，语义上等于 'A1'。"""
    a = normalize_args({"order_id": "A1"})
    b = normalize_args({"order_id": "  A1  "})
    assert fingerprint("get_order", a) == fingerprint("get_order", b)


def test_genuinely_different_args_differ() -> None:
    """不能因为过度规范化把不同操作混为一谈。"""
    a = normalize_args({"order_id": "A1", "amount": 100})
    b = normalize_args({"order_id": "A1", "amount": 200})
    assert fingerprint("create_payment", a) != fingerprint("create_payment", b)


def test_structural_hash_is_key_order_independent() -> None:
    a = normalize_args({"order_id": "A1", "amount": 100})
    b = normalize_args({"amount": 100, "order_id": "A1"})
    assert structural_hash("create_payment", a) == structural_hash("create_payment", b)


def test_pagination_is_not_an_equivalence_but_exposes_what_changed() -> None:
    """页码递增：近似层判"不同"，但必须说清**是哪个字段**不同。

    两次翻页是语义上确实不同的请求，判成"同一次调用"是错的。但序列检测器要
    靠 differing_fields 认出"只有页码在动"，再结合单调性决定它是正常翻页
    （该放行）还是翻回去的循环（该报）。所以这条断言同时钉住两件事。
    """
    a = normalize_args({"query": "widget", "page": 1})
    b = normalize_args({"query": "widget", "page": 2})
    assert structural_hash("search", a) != structural_hash("search", b), "严格层应判为不同"

    sim = args_similarity(a, b)
    assert not sim.is_similar, "翻页不是同一次调用，不该判为等价"
    assert sim.score > 0.7, f"但应当明显高于不相关调用，实得 {sim.score}"
    assert sim.differing_fields == ("page",)


def test_fractional_amount_is_an_equivalence() -> None:
    """int/float 表示差异是近似层的正职：严格层抓不到，语义上却是同一次调用。

    模型重新生成参数时把 100 写成 100.0 很常见；漏掉就会让重复扣款逃过检测。
    """
    a = normalize_args({"order_id": "A1", "amount": 100})
    b = normalize_args({"order_id": "A1", "amount": 100.0})
    assert structural_hash("create_payment", a) != structural_hash("create_payment", b)

    sim = args_similarity(a, b)
    assert sim.is_similar, f"100 与 100.0 应判为等价，实得 {sim.score}"
    assert sim.differing_fields == ()


def test_similarity_separates_different_orders() -> None:
    """不同订单号的两次支付不能算相似，否则正常业务会被报成循环。"""
    a = normalize_args({"order_id": "A1", "amount": 100})
    b = normalize_args({"order_id": "Z9", "amount": 100})
    sim = args_similarity(a, b)
    assert not sim.is_similar
    assert "order_id" in sim.differing_fields


def test_key_weights_can_demote_a_field() -> None:
    """场景可声明字段重要性：订单号重要，页码不重要。"""
    a = normalize_args({"order_id": "A1", "page": 1})
    b = normalize_args({"order_id": "A1", "page": 99})
    low = args_similarity(a, b, key_weights={"page": 0.01, "order_id": 10.0})
    assert low.score > 0.99


def test_write_detection_prefers_declaration() -> None:
    """显式声明优先于启发式——猜错方向的代价不对称。"""
    assert looks_like_write("create_payment", declared_writes={"create_payment"})
    assert not looks_like_write("create_payment", declared_writes={"other"})
    assert not looks_like_write("get_order", declared_writes=set())


def test_write_detection_heuristic_is_conservative() -> None:
    """启发式兜底时默认按"写"处理：把写误判成读会漏掉重复扣款告警。"""
    assert not looks_like_write("get_order")
    assert not looks_like_write("list_orders")
    assert not looks_like_write("search_products")
    assert looks_like_write("create_payment")
    assert looks_like_write("send_email")
    assert looks_like_write("transfer_funds")
    # 无法确定的按写处理
    assert looks_like_write("frobnicate")
