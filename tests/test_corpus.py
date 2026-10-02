"""标注语料：加载校验、标注比对、以及语料机制自己的自检。

测试分三层：

1. **语料内容** —— `tests/corpus/*.yaml` 里每一条都要命中它自己的标注。
   红了只有两种可能：检测器改了行为，或者改坏了。两种都要人看一眼。
2. **加载校验** —— 写错的语料必须**在加载时**报错。不报的话，一条写坏的语料会
   以"未命中"的形式出现在结果里，而那看起来和"检测器漏报"一模一样——
   一个格式笔误会被读成检测器的缺陷。
3. **机制自检** —— `forbidden` / `absent` / 族级标注这些东西本身要能抓到问题。
   一个永远为空的检查比没有检查更坏：它会让准确率看起来是 1.00。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from sandbox.corpus import (
    FAMILIES,
    NONE,
    CorpusError,
    evaluate_all,
    family_of,
    load_corpus,
    parse_item,
    run_item,
    score,
)
from sandbox.detectors import KNOWN_DETECTORS

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"


@pytest.fixture(scope="module")
def corpus() -> list:
    return load_corpus(CORPUS_DIR)


@pytest.fixture(scope="module")
def results(corpus) -> list:
    return evaluate_all(corpus)


# --- 一、语料内容 -----------------------------------------------------------


def test_every_item_matches_its_labels(results) -> None:
    """每一条语料都要命中标注。红了先看是哪一栏：

    `missing` = 该报的没报；`forbidden` = 不该报的报了；
    `violated_absent` = 明令不许出现的出现了。
    """
    failures = []
    for result in results:
        if result.ok:
            continue
        failures.append(
            f"\n[{result.item.id}] {result.item.path}\n"
            f"  标注依据：{result.item.why}\n"
            f"  该报未报：{[f'{e.detector} {e.verdict.value}' for e in result.missing]}\n"
            f"  不该报却报：{[f'{f.detector} {f.verdict.value}' for f in result.forbidden]}\n"
            f"  明令禁止：{[f'{f.detector} {f.verdict.value}' for f in result.violated_absent]}\n"
            f"  实际结论：{[f'{f.detector} {f.verdict.value}' for f in result.findings]}"
        )
    assert not failures, "语料与标注不符：" + "".join(failures)


def test_corpus_covers_every_detector_family(results) -> None:
    """每一族都要有正样本和负样本。

    只覆盖一族的话，另外几族的精确率是拿 0 个样本算出来的——而结果会打印成
    "100.0%"，看起来和真的测过一样。
    """
    for family in FAMILIES:
        positives = [r for r in results if r.item.family_labels.get(family)]
        negatives = [r for r in results if not r.item.family_labels.get(family)]
        assert positives, f"{family} 一族没有正样本，它的召回率是空的"
        assert negatives, f"{family} 一族没有负样本，它的精确率是空的"


def test_families_mirror_the_detectors_in_the_pipeline() -> None:
    """语料按族统计，而族是从检测器名的前缀切的。新增一族检测器时，
    这条会红——那正是要人做的决定：它算不算一"族"，语料要不要对它表态。
    """
    assert {family_of(d) for d in KNOWN_DETECTORS} == set(FAMILIES)


def test_corpus_ids_are_unique(corpus) -> None:
    ids = [item.id for item in corpus]
    assert len(ids) == len(set(ids))


def test_grouping_reports_one_finding_for_one_repeated_behavior(corpus) -> None:
    """同一个工具在一夜里试了五次，是**一个**行为重复了五次。

    这条断言写在这里而不是语料里：语料表达得了"这一族该报什么"，表达不了
    "报几条"。而数量正是归组规则的整个内容——按调用记五条的话，报告里会变成
    五个问题，读的人会去修五处。
    """
    item = next(i for i in corpus if i.id == "auto-grouping-by-tool")
    findings = run_item(item).findings
    unauthorized = [f for f in findings if f.detector == "autonomy.unauthorized"]
    assert len(unauthorized) == 1
    assert unauthorized[0].detail["count"] == 5


#: 语料里的受控对照：两组 id，按说法唯一的不同是**那一行主键声明**。
#:
#: 场景那边的账本在 `tests/control_pairs.py`（它比的是 `scenarios/*.yaml`）。
#: 语料这一层今天只有这一族对照，所以账直接写在这里——但规矩是同一条：
#: 一句「唯一的不同是 X」必须有东西在核，因为这句话**错了没有任何症状**。
IDENTITY_PAIRS: tuple[tuple[str, str], ...] = (
    (
        "idem-declared-key-catches-two-ways-to-pay",
        "idem-undeclared-key-leaves-the-same-calls-clean",
    ),
    (
        "idem-declared-key-does-not-merge-distinct-targets",
        "idem-same-tool-distinct-targets",
    ),
)


@pytest.mark.parametrize(("declared_id", "bare_id"), IDENTITY_PAIRS)
def test_the_identity_pairs_differ_only_in_the_declaration(
    corpus, declared_id: str, bare_id: str
) -> None:
    """这两对的说法是「同一段轨迹，唯一的不同是那一行主键声明」——**要有东西核**。

    那句话承重得很具体：第一对的结论从 pass 变成 fail，全靠那一行。
    核法与场景账本同一条——把声明那一格抹掉之后，两条必须逐字相等；而且两边
    真的一边有一边没有（否则那句声明是空的：写在说法里，两支之间根本没这个差别）。
    """
    declared = next(i for i in corpus if i.id == declared_id)
    bare = next(i for i in corpus if i.id == bare_id)

    def trace(item):
        # 调用的 id 里带语料 id，那是"这两行分别是谁"的标签，不是轨迹的一部分。
        return [
            replace(c, id="", session_id="", session_sequence_id="") for c in item.calls
        ]

    assert trace(declared) == trace(bare), "两条的轨迹不一样，那这一对不比声明"
    assert declared.tool_facts.keys() == bare.tool_facts.keys()
    wiped = {
        name: replace(facts, identity_fields=())
        for name, facts in declared.tool_facts.items()
    }
    assert wiped == dict(bare.tool_facts), "两条的工具声明除了主键那一行之外还有别的差别"

    fields = {n: f.identity_fields for n, f in declared.tool_facts.items() if f.identity_fields}
    assert fields, "声明那一侧一个字段都没声明——这一对什么都没在演"
    assert not any(f.identity_fields for f in bare.tool_facts.values()), (
        f"对照那一侧也声明了主键（{fields}）——那两边差的不止一个变量"
    )


# --- 二、加载校验 -----------------------------------------------------------

BASE_ITEM = {
    "id": "t-1",
    "source": "自造",
    "why": "这是测试用的最小语料，理由必须写出来。",
    "tools": {"get_order": {"is_write": False}},
    "sessions": [{"ordinal": 0}],
    "calls": [{"tool": "get_order", "args": {"order_id": "A1"}}],
    "expect": {
        "loop": "pass",
        "idempotency": "inconclusive",
        "autonomy": "none",
        "memory": "none",
    },
}


def _item(**overrides):
    """造一条最小语料，把要测的字段覆盖掉。"""
    item = {k: (dict(v) if isinstance(v, dict) else v) for k, v in BASE_ITEM.items()}
    item.update(overrides)
    return item


def _rejects(item, *keywords: str) -> None:
    with pytest.raises(CorpusError) as excinfo:
        parse_item(item, "t.yaml")
    message = str(excinfo.value)
    for keyword in keywords:
        assert keyword in message, f"报错信息里没提到 {keyword!r}：{message}"


def test_minimal_item_loads() -> None:
    """先证明"正确的写法能过"，否则下面那些报错断言可能只是因为全都过不了。"""
    item = parse_item(BASE_ITEM, "t.yaml")
    assert item.id == "t-1"
    assert item.calls[0].fingerprint, "指纹要从和生产同一处算出来"
    assert item.family_labels == {
        "loop": False,
        "idempotency": False,
        "autonomy": False,
        "memory": False,
    }


def test_unknown_field_is_rejected() -> None:
    _rejects(_item(expectt={"loop": "pass"}), "expectt")


def test_unknown_field_in_a_call_is_rejected() -> None:
    _rejects(_item(calls=[{"tool": "get_order", "arg": {}, "args": {"order_id": "A1"}}]), "arg")


def test_missing_family_label_is_rejected() -> None:
    """少一族就等于那族的样本没进统计，而结果看起来完好。"""
    _rejects(_item(expect={"loop": "pass", "idempotency": "pass"}), "autonomy")


def test_expect_must_reference_a_known_detector() -> None:
    _rejects(
        _item(
            expect={"loop.strickt": "fail", "idempotency": "pass", "autonomy": "none"}
        ),
        "loop.strickt",
    )


def test_source_must_be_in_the_whitelist() -> None:
    _rejects(_item(source="我从哪儿抄的忘了"), "白名单")


def test_call_status_must_be_in_the_vocabulary() -> None:
    """语料里的 `status` 和边界上写的是**同一份词表**，错一个字就是另一种判定。

    `CallView.reached_target` 只问一句：`status` 在不在 `REACHED_TARGET` 里。
    不在就一律算「没到目标系统」，也就是**这次调用一次都没发生过**，不进任何
    判定。所以 `erorr` 不是"填错了"，是"把这次调用删掉了"——而且删得很安静：
    报告里少一条和少得对长得一模一样。

    边界那边的取值一直有闸门（`test_call_status_vocabulary.py`，拿 `proxy/app.py`
    的十处真实写法核对），语料这边是 `status` 的**第二个生产者**，原先谁都没核。
    这条补的就是那半边：同一份名单，两个生产者，两边都要有人看着。
    """
    _rejects(
        _item(calls=[{"tool": "get_order", "args": {"order_id": "A1"}, "status": "erorr"}]),
        "词表",
    )


def test_missing_why_is_rejected() -> None:
    _rejects(_item(why="   "), "why")


def test_call_to_an_undeclared_tool_is_rejected() -> None:
    _rejects(_item(calls=[{"tool": "creat_payment", "args": {}}]), "creat_payment")


def test_call_in_an_undeclared_session_is_rejected() -> None:
    _rejects(_item(calls=[{"tool": "get_order", "args": {}, "session": 3}]), "#3")


def test_tool_declaration_must_state_is_write() -> None:
    """读写属性没有默认值：给了默认，忘记声明的写操作会静默变成只读，
    而重复扣款检测根本不覆盖它。"""
    _rejects(_item(tools={"get_order": {}}), "is_write")


def test_read_tool_cannot_declare_unattended_authorization() -> None:
    """一个不起作用的字段比没有这个字段更坏：它让人以为自己限制了什么。"""
    _rejects(
        _item(tools={"get_order": {"is_write": False, "may_run_unattended": True}}),
        "may_run_unattended",
    )


def test_exemption_needs_a_reason() -> None:
    _rejects(
        _item(
            exemptions=[
                {"detector": "idempotency", "tool": "get_order", "reason": ""},
            ]
        ),
        "reason",
    )


def test_exemption_pointing_at_an_unknown_detector_is_rejected() -> None:
    """指向不存在的检测器的豁免永远不会命中，而它看起来像"已经处理过了"。"""
    _rejects(
        _item(exemptions=[{"detector": "idempotence", "reason": "拼错了"}]),
        "idempotence",
    )


def test_absent_pointing_at_an_unknown_detector_is_rejected() -> None:
    _rejects(_item(absent=[{"detector": "loop.strickt"}]), "loop.strickt")


def test_absent_with_an_unknown_verdict_is_rejected() -> None:
    _rejects(_item(absent=[{"detector": "loop", "verdict": "failed"}]), "failed")


def test_duplicate_ids_across_files_are_rejected(tmp_path) -> None:
    """重名的两条语料会让统计里少一条，而结果看起来完好。"""
    body = (
        "items:\n"
        "  - id: same-id\n"
        "    source: 自造\n"
        "    why: 重复的 id。\n"
        "    tools: {get_order: {is_write: false}}\n"
        "    sessions: [{ordinal: 0}]\n"
        "    calls: [{tool: get_order, args: {order_id: A1}}]\n"
        "    expect: {loop: pass, idempotency: inconclusive, autonomy: none, memory: none}\n"
    )
    (tmp_path / "a.yaml").write_text(body, encoding="utf-8")
    (tmp_path / "b.yaml").write_text(body, encoding="utf-8")
    with pytest.raises(CorpusError, match="same-id"):
        load_corpus(tmp_path)


def test_missing_corpus_directory_is_rejected(tmp_path) -> None:
    with pytest.raises(CorpusError):
        load_corpus(tmp_path / "nope")


# --- 三、机制自检 -----------------------------------------------------------


WRITE_ITEM = {
    "id": "t-write",
    "source": "自造",
    "why": "两次一模一样的写操作，工具没有幂等保护。",
    "tools": {"create_payment": {"is_write": True}},
    "sessions": [{"ordinal": 0}],
    "calls": [
        {"tool": "create_payment", "args": {"order_id": "A1", "amount": 99}},
        {"tool": "create_payment", "args": {"order_id": "A1", "amount": 99}},
    ],
    "expect": {
        "loop": "pass",
        "idempotency": "fail",
        "autonomy": "none",
        "memory": "none",
    },
}


def test_a_wrong_label_shows_up_as_missing_and_forbidden() -> None:
    """自检：故意把标注写反，两个方向都要被抓到。

    这一条如果不红，前面那些绿的就全是废话——一个永远为空的检查会让
    精确率看起来是 100%。
    """
    # 该报 fail 却标了"这一族不许报" → 报出来了，进 forbidden
    wrong = dict(
        WRITE_ITEM,
        expect={"loop": "pass", "idempotency": NONE, "autonomy": NONE, "memory": NONE},
    )
    result = run_item(parse_item(wrong, "t.yaml"))
    assert not result.ok
    assert [f.detector for f in result.forbidden] == ["idempotency"]

    # 反过来：标了该报 fail，实际却不报 → 进 missing
    quiet = dict(
        WRITE_ITEM,
        tools={"create_payment": {"is_write": True, "enforces_idempotency": True}},
        calls=[
            {"tool": "create_payment", "args": {"order_id": "A1"}, "idempotency_key": "k"},
            {"tool": "create_payment", "args": {"order_id": "A1"}, "idempotency_key": "k"},
        ],
        expect={"loop": NONE, "idempotency": "fail", "autonomy": NONE, "memory": NONE},
    )
    result = run_item(parse_item(quiet, "t.yaml"))
    assert not result.ok
    assert [e.detector for e in result.missing] == ["idempotency"]


def test_absent_catches_a_finding_the_family_labels_cannot() -> None:
    """`absent` 拦的是族级标注拦不住的那种：报了一条 fail 有道理，
    同时又报了一条 pass 说它很规矩。族级标注只管 fail/warning。
    """
    item = parse_item(
        dict(
            WRITE_ITEM,
            expect={"loop": "pass", "idempotency": "fail", "autonomy": "none", "memory": NONE},
            absent=[{"detector": "idempotency", "verdict": "pass"}],
        ),
        "t.yaml",
    )
    # 这一条实际会报 fail 而不是 pass，所以 absent 这一栏应当是空的
    assert run_item(item).violated_absent == []

    sneaky = parse_item(
        dict(
            WRITE_ITEM,
            tools={"create_payment": {"is_write": True, "enforces_idempotency": True}},
            sessions=[{"ordinal": 0}],
            calls=[
                {
                    "tool": "create_payment",
                    "args": {"order_id": "A1"},
                    "idempotency_key": "k",
                },
                {
                    "tool": "create_payment",
                    "args": {"order_id": "A1"},
                    "idempotency_key": "k",
                },
            ],
            expect={"loop": "pass", "idempotency": "warning", "autonomy": "none", "memory": NONE},
            absent=[{"detector": "idempotency"}],
        ),
        "t.yaml",
    )
    result = run_item(sneaky)
    assert not result.ok, "absent 没能拦住一条被禁的结论"
    assert [f.detector for f in result.violated_absent] == ["idempotency"]


def test_family_labels_or_fold_sub_detectors() -> None:
    """同一族的不同层可以合理地一报一不报：跨会话层报 warning，严格层干净。
    族级标注取或——它问的是"这一族张口了吗"，不是"每一层都同意吗"。
    """
    item = parse_item(
        dict(
            WRITE_ITEM,
            expect={
                "loop.cross_session": "warning",
                "loop": "pass",
                "idempotency": "none",
                "autonomy": "none",
                "memory": "none",
            },
        ),
        "t.yaml",
    )
    assert item.family_labels["loop"] is True


def test_score_counts_precision_and_recall() -> None:
    """四格里每一格都算对。`inconclusive` 的轨迹不进任何一格——
    它既不是误报也不是漏报，是"没走到"。
    """
    positives = parse_item(WRITE_ITEM, "t.yaml")  # 该报且报了 → TP
    negatives = parse_item(
        dict(
            WRITE_ITEM,
            id="t-quiet",
            calls=[{"tool": "create_payment", "args": {"order_id": "A1", "amount": 99}}],
            expect={"loop": "pass", "idempotency": "pass", "autonomy": "none", "memory": NONE},
        ),
        "t.yaml",
    )  # 不该报也没报 → TN
    scores = score([run_item(positives), run_item(negatives)])
    assert (scores["idempotency"].tp, scores["idempotency"].tn) == (1, 1)
    assert (scores["idempotency"].fp, scores["idempotency"].fn) == (0, 0)
    assert scores["idempotency"].precision == 1.0
    assert scores["idempotency"].recall == 1.0


def test_score_has_no_verdict_when_the_grid_is_empty() -> None:
    """一条正样本都没有的那一族，精确率和召回率都要报 n/a。

    报 0 会被读成"很差"，报 1 会被读成"完美"，而实际是"这一格没测"。
    """
    item = parse_item(
        dict(
            WRITE_ITEM,
            calls=[{"tool": "create_payment", "args": {"order_id": "A1", "amount": 99}}],
            expect={"loop": "pass", "idempotency": "pass", "autonomy": "none", "memory": NONE},
        ),
        "t.yaml",
    )
    scores = score([run_item(item)])
    for family in FAMILIES:
        assert scores[family].precision is None, f"{family} 没有正样本，精确率该是 n/a"
        assert scores[family].recall is None, f"{family} 没有正样本，召回率该是 n/a"
        assert scores[family].tn == 1, f"{family} 在这条轨迹上是实打实的“没报也没该报”"
