"""网关的字符→token 估算，偏差到底有多大、朝哪边偏。

**为什么这条绊线值得存在。** `gateway/stream.py` 里那个 `_CHARS_PER_TOKEN = 3`
原来带一句注释说"取 3 是往安全的方向估——估多了只是提前熔断"。那句话**没有
量过**，而且对中文是反的：真上游实测下来，中文是 1.67–2.19 字符/token，取 3
意味着**低估**，而低估在止损上的方向是坏的那一边（预扣少于实际，熔断晚一步）。

这个估算器还决定"这次请求放不放行"（`Reserved.of` → `TokenBudget.reserve`），
而在此之前**没有任何测试碰过它**——`estimate_input_tokens` 全仓零引用。

所以这里钉住四个实测点，把它变成一条"改坏了会红"的线。四个数来自
2026-10-01 对真上游的实测，`scripts/calibrate_tokens.py` 可以在有 key 的
机器上重新量一遍（没有 key 就跳过）。

**断言的是方向，不是数值。** 估算器不需要准，需要的是"偏哪边、偏多少"
和注释里写的一致：中文可以低估（且不许低估到离谱），英文不许低估
（低估意味着英文内容的预算会提前被花穿）。
"""

from __future__ import annotations

import json

from sandbox.gateway.stream import estimate_input_tokens

#: 实测日期。数字过期了要重量（模型换了 tokenizer 就会变）。
MEASURED_ON = "2026-10-01"


def _prompt(text: str) -> int:
    """按网关真实调用形状估一次：它估的是 `json.dumps` 出来的那个 blob。

    注意这层 JSON 外壳本身约 40 字符（≈13 token），它让估算值比"只量正文"
    略高一点。实测数是在真上游上量的正文，所以英文 JSON 那一条的余量很薄
    （1.03）——那不是笔误，是"外壳把低估刚好推回了安全侧"。
    """
    return estimate_input_tokens({"messages": [{"role": "user", "content": text}]})


#: 四个样本 + 真上游给的输入 token 数（`prompt_tokens`，已扣掉基线那句
#: 「请只回复 ok。」的 9 个 token）。样本的构造方式与标定脚本逐字相同。
SAMPLES: dict[str, tuple[str, int]] = {
    "纯中文": ("订单已经处理完毕，请核对金额后确认收款。" * 40, 480),
    "中文JSON": (
        json.dumps(
            {
                "ok": True,
                "items": [
                    {
                        "order_id": f"A{1000 + i}",
                        "product": "机械键盘 87 键 无线 双模",
                        "amount": 199 + i,
                        "currency": "CNY",
                        "status": "unpaid",
                        "created_at": f"2026-09-{(i % 28) + 1:02d}T08:30:00+08:00",
                        "note": "客户备注：尽快发货，谢谢。",
                    }
                    for i in range(20)
                ],
                "page": 1,
                "page_size": 20,
                "total": 90,
                "has_more": True,
            },
            ensure_ascii=False,
        ),
        1636,
    ),
    "英文JSON": (
        json.dumps(
            {
                "ok": True,
                "items": [
                    {
                        "order_id": f"A{1000 + i}",
                        "product": "Mechanical Keyboard",
                        "amount": 199 + i,
                        "status": "unpaid",
                    }
                    for i in range(20)
                ],
                "page": 1,
                "has_more": True,
            },
            ensure_ascii=False,
        ),
        623,
    ),
    "纯英文": (
        "The order has been processed correctly; please verify the amount. " * 30,
        361,
    ),
}


def test_对中文不许高估_而且低估不超过一半() -> None:
    """中文是本项目的常态内容（场景、工具返回体、记忆全是中文）。

    实测这两个样本都会**被低估**（估算值小于真值）。这条线上限锁 1.0
    （不许高估——高估会让中文预算提前熔断），下限锁 0.5（不许估到一半以下
    ——那等于预扣形同虚设）。
    """
    for name in ("纯中文", "中文JSON"):
        text, actual = SAMPLES[name]
        estimate = _prompt(text)
        ratio = estimate / actual
        assert 0.5 <= ratio <= 1.0, (
            f"{name}：估 {estimate} / 实测 {actual} = {ratio:.2f}，"
            f"落在注释写明的窗口 [0.5, 1.0] 之外。"
            f"中文被高估意味着预算提前熔断，被低估超过一半意味着预扣没意义。"
            f"（实测于 {MEASURED_ON}；改了这个常数就要重跑 "
            f"scripts/calibrate_tokens.py 并把两处注释一起改。）"
        )


def test_对英文不许低估() -> None:
    """英文方向相反：这里**不许低估**。

    低估的后果不是"少扣点"，而是英文内容的请求会**先花穿预算再发现**——
    那正是预扣要防的那一次。上限 2.5 倍是防另一个极端：估得虚高会让一个
    纯英文的 Agent 在真实花费的三分之一处就被熔断。
    """
    for name in ("纯英文", "英文JSON"):
        text, actual = SAMPLES[name]
        estimate = _prompt(text)
        ratio = estimate / actual
        assert 1.0 <= ratio <= 2.5, (
            f"{name}：估 {estimate} / 实测 {actual} = {ratio:.2f}，"
            f"落在注释写明的窗口 [1.0, 2.5] 之外。英文被低估 = 预算会被花穿；"
            f"高估超过 2.5 倍 = 熔断得太早。"
        )


def test_估算器在放行判定里真的被用到() -> None:
    """防的是"这个估算器哪天被绕开、预扣改用一个常数"。

    `Reserved.of` 是预扣的唯一入口，它必须走估算器；直接写死一个数的话，
    上面两条测试测的就是一个没人用的函数。
    """
    from sandbox.gateway.stream import Reserved

    payload = {"messages": [{"role": "user", "content": "订单已经处理完毕。" * 50}]}
    assert Reserved.of(payload).input_tokens == estimate_input_tokens(payload)
    assert Reserved.of(payload).input_tokens > 1
