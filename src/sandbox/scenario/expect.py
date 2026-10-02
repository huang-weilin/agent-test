"""把检测结论与场景预期比对，得出这次运行"算不算通过"。

三种结果，分开统计——把它们合并成一个布尔值是这个模块最容易犯的错：

  `matched`    预期命中。正常。
  `missing`    **预期没命中**。两种可能：被测 Agent 这次没犯那个错，或者检测器坏了。
               分不清，所以不能算通过，也不能算失败，只能报"未达成"。
  `unexpected` 出现了没被任何预期覆盖的 fail 结论。默认按失败处理——这是测试工具，
               没预料到的问题必须让人知道。场景可以声明 `allow` 让这一批不参与判定
               （结论本身仍然列在报告的「检测结论」一节里，那一节是"看见了什么"）。

**`missing` 为什么不直接算失败？** 因为一次"检测器没报出循环"的运行，和一个
"Agent 这次真的没循环"的运行，在数据上无法区分——除非场景里写了脚本化的上游
（那就确定了 Agent 一定会循环）。所以 `missing` 要连同脚本消费情况一起看：
脚本被完整消费 = Agent 确实做了那些事 = 那检测器就是漏报了。

**这条推断目前没有任何一处实现它。** 这里原先写着"这条判断放在 runner 里做"，
而那是句没兑现的话：runner 只在**前提不成立**时印一句"结论不指向被测对象"
（`_render` 里那一段，方向恰好相反——它说的是"别怪 Agent"，而漏报推断说的是
"该怪检测器"）。脚本消费量（`SessionOutcome.script_steps_consumed`）照常进报告
快照，但把它变成一句结论需要新判据，而新判据要自己的一轮误报测量。在那之前，
报告与指标看板都只报"预期没命中"，"所以是谁的错"留给读的人。
（`docs/metrics.md` 的不支持清单里记着这一条。）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sandbox.db.models import DetectorVerdict
from sandbox.detectors.base import Finding
from sandbox.scenario.spec import Expectation, Scenario


@dataclass(frozen=True)
class ExpectResult:
    matched: list[tuple[Expectation, Finding]] = field(default_factory=list)
    missing: list[Expectation] = field(default_factory=list)
    unexpected: list[Finding] = field(default_factory=list)
    # 未被任何预期覆盖的 inconclusive。**不计为失败**——未覆盖是信息，
    # 不是缺陷。但要在报告里显式列出来，否则覆盖率会悄悄下降。
    uncovered: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing and not self.unexpected

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched": [
                {"expect": e.model_dump(mode="json"), "got": f.summary} for e, f in self.matched
            ],
            "missing": [e.model_dump(mode="json") for e in self.missing],
            "unexpected": [
                {"detector": f.detector, "verdict": f.verdict.value, "summary": f.summary}
                for f in self.unexpected
            ],
            "uncovered": [
                {"detector": f.detector, "verdict": f.verdict.value, "summary": f.summary}
                for f in self.uncovered
            ],
        }


def _reason_code(finding: Finding) -> str | None:
    code = finding.detail.get("reason_code")
    return str(code) if code is not None else None


def _matches(expectation: Expectation, finding: Finding) -> bool:
    if expectation.detector != finding.detector:
        return False
    if expectation.verdict != finding.verdict:
        return False
    # 不写 reason_code 就是"不在乎是哪一种"。写了就必须对上——跨会话那两条
    # 归因的事实完全相同、只有 reason_code 不同，不比对它就分不出该去修谁。
    return expectation.reason_code is None or expectation.reason_code == _reason_code(finding)


def match_expectations(
    expectations: list[Expectation], findings: list[Finding]
) -> tuple[list[tuple[Expectation, Finding]], list[Expectation], list[Finding]]:
    """一条预期**各自**去找一条没被占用的实际结论，而不是按顺序配对——
    同一批结论可能同时满足多条预期，按顺序配对会让结果依赖于字典序。

    **`count` 说的是这条预期该命中几条**（默认 1）。够不着就按缺的条数往
    `missing` 里放几份——`matched` 加上 `missing` 里那条预期的份数，正好等于
    它声明的条数。这样"要两条只来了一条"和"一条都没来"在数据上是两件事，
    而报告里那一栏本来就只列预期、不列条数。

    单独抽出来给 `tests/corpus` 用：语料的标注和场景的 `expect` 是同一套语义，
    共用这一份实现才不会出现"语料里判通过、场景里判不通过"这种两套标准。
    """
    remaining = list(findings)
    matched: list[tuple[Expectation, Finding]] = []
    missing: list[Expectation] = []
    for expectation in expectations:
        for _ in range(expectation.count or 1):
            hit = next((f for f in remaining if _matches(expectation, f)), None)
            if hit is None:
                missing.append(expectation)
            else:
                remaining.remove(hit)
                matched.append((expectation, hit))
    return matched, missing, remaining


def evaluate(scenario: Scenario, findings: list[Finding]) -> ExpectResult:
    """比对预期与实际结论。

    策略只有场景声明这一个来源。这里曾经有一个调用方覆盖的参数，但从加进来那天
    起就没有任何调用方——同一个旋钮两个来源，迟早出现"报告按场景判、代码按参数
    判"，而配置指纹里记的是场景那一份。
    """
    matched, missing, remaining = match_expectations(scenario.expect, findings)

    unexpected: list[Finding] = []
    if scenario.unexpected_failures != "allow":
        unexpected = [f for f in remaining if f.verdict == DetectorVerdict.FAIL]

    uncovered = [f for f in remaining if f.verdict == DetectorVerdict.INCONCLUSIVE]

    return ExpectResult(
        matched=matched, missing=missing, unexpected=unexpected, uncovered=uncovered
    )
