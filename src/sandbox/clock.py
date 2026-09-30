"""虚拟时钟：长时运行里"世界现在几点"的唯一来源。

**为什么这件事只能发生在边界上。** 需求实现篇给的方案是"往被测 Agent 进程里注入
TimeProvider，拦截 sleep、定时任务、cron"。ADR 001 已经把进程内注入整条路否掉了，
理由对时间是逐字成立的：Muse 跑在它自己的 Secure VM 里、LightVela 的 Hermes 跑在
腾讯轻云的独立实例里，测试者拿不到那个进程——`monkeypatch time.sleep` 一行都执行
不了。演示 Agent 连 `sandbox.*` 都不 import，这不是巧合，是环境保证。

但"长时运行"这件事本身还是成立的，只是要把话说准：

    时间加速 ≠ 让 Agent 觉得时间变快了
    时间加速 = 两次会话之间，**世界**过去了很久；而真实等待是零

被测 Agent 两次会话之间本来就不在跑——云端常驻 Agent 也是被定时器叫醒的。它醒来
时看到的世界（订单逾期没有、上一次扣款是什么时候）才是"过了多久"的全部证据，而
那些都在边界上，我们完全掌控。Agent 自己进程内的 `time.sleep()` 我们拦不到，**也
不假装拦得到**：一个内部按真实时间排程的 Agent，在长时运行里的表现和它在生产上
被定时器叫醒时并不相同，这一点写在 `docs/longrun.md` 的"不支持"清单里。

**一条纪律：一个时钟。** mock 的时间戳、报告里的时刻、会话之间的间隔，全部只读
这一个对象。边界两侧各读各的钟的话，Agent 会看到一个自相矛盾的世界（工具说这是
三天后，模型说这是刚才），而基于它得出的每一条结论都是垃圾——报告却照样长得正常。
分节指纹那件事教过一次：两个口径不一致，比没有口径危险得多。

**没人在跑长时运行时，`now()` 就是真实的 UTC 现在。** 默认路径一点没变，于是
既有的代码和测试一行都不用改——读这个时钟和读 `datetime.now(UTC)` 语义等价，
差别只在于前者可以被"拧"。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

#: 偏移量按秒存整数。间隔都是小时/天量级，浮点秒只会引入"两次计算差一微秒"
#: 这类无从解释的差异。
OffsetSeconds = int


class ClockNotStarted(RuntimeError):
    """时钟还没进入虚拟模式就想推进它。"""


class VirtualClock:
    """进程级的可拧时钟。

    刻意**不加锁**：跑长时运行的路径是 asyncio 单线程事件循环，与 `api/jobs.py`
    的作业登记表同一个前提。真要在多线程下用，得先解决执行器并发那个更大的问题
    （见 `docs/ui.md` 第 2 条：一次只跑一个）。

    刻意**不提供"往回调"**：`advance` 只接受非负值。往回拨一下就能让一件已经逾期
    的事重新变得不逾期，而报告上完全看不出来——那会让一条已经得出的结论在下一次
    读取时静默失效。
    """

    def __init__(self) -> None:
        # None = 没启用虚拟时间，读到的就是真实时间。
        self._offset: timedelta | None = None

    # --- 读 -----------------------------------------------------------------

    @property
    def is_virtual(self) -> bool:
        return self._offset is not None

    def real_now(self) -> datetime:
        """真实的现在。**不受虚拟时间影响**，用于记账、排序、耗时统计。

        真实时间和虚拟时间必须分开这两个用途，不能互相替代：

        * 真实时间决定"哪一次运行更晚"（回归集就靠它排序）、一次运行花了多久。
          让这些跟着虚拟时间走的话，一次模拟了三个月的运行会排到所有真实运行的
          后面去，而它其实只是刚才跑的。
        * 虚拟时间是**被模拟的世界**的属性，只用来描述那个世界。

        混用的症状是排序错乱和耗时看起来像天方夜谭，两者都不会报错。
        """
        return datetime.now(UTC)

    def now(self) -> datetime:
        """被模拟的世界里的现在。未启用虚拟时间时等于 `real_now()`。"""
        if self._offset is None:
            return self.real_now()
        return self.real_now() + self._offset

    def offset_s(self) -> OffsetSeconds:
        """虚拟时间比真实时间快多少秒。未启用时为 0。"""
        if self._offset is None:
            return 0
        return int(self._offset.total_seconds())

    # --- 拧 -----------------------------------------------------------------

    def start(self) -> None:
        """进入虚拟模式，从此刻起偏移为 0。

        只把开关打开、不跳时间：调用方随后自己决定第一次跳多少。这样"这一场到底
        有没有在用虚拟时间"就成了一个显式事实，而不是靠偏移量恰好为 0 来推断——
        两者在后一种写法里长得一模一样。
        """
        self._offset = timedelta(0)

    def advance(self, seconds: float) -> None:
        """把世界往后拨 `seconds` 秒。立即返回，不等待。

        **这就是"加速"的全部**：声明一段时间过去了，没有任何东西真的睡那么久。
        虚拟期间的真实耗时由调用方各自统计（`ScenarioResult` 里的 real/virtual
        两个数并列给出），报告上写清楚，免得读的人以为跑一次要三天。
        """
        if self._offset is None:
            raise ClockNotStarted(
                "时钟还没进入虚拟模式。先调 start()——静默地自动开始的话，"
                "「这个场景到底有没有在模拟时间」就没人说得清了。"
            )
        if seconds < 0:
            raise ValueError(
                f"虚拟时间不能往回拨（请求 {seconds} 秒）。往回拨会让一件已经逾期"
                f"的事重新变得不逾期，而报告上看不出来。"
            )
        self._offset += timedelta(seconds=seconds)

    def reset(self) -> None:
        """回到真实时间。

        跑完必须调：这是进程级状态，泄漏给下一个场景的话，下一次运行会带着上一次
        的时间偏移开始，而它的报告里每一处都会显示"世界是三天后的"——全都正常，
        全都错。
        """
        self._offset = None


#: 进程级单例。与 `gateway.stub.scripted_upstream`、`runtime.runs` 同一套路：
#: 被测 Agent 是独立进程，只能通过共享的目标系统与它对齐"现在几点"。
clock = VirtualClock()
