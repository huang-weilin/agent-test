"""场景文件的加载与校验。

校验里有几条是**跨字段**的，单看任何一个字段都发现不了，但它们出错的方式都很安静
（完整清单是七条，见 `docs/scenarios.md`；第五条——无人值守会话里的写操作必须交代
`may_run_unattended`——在 `spec.py` 的模型校验里，不在这里）：

1. `restore_from` 引用了一张**后面**才拍的快照。这意味着"回到未来"，
   记忆视野会算出一个没意义的数，而归因完全建立在视野上。必须报错，
   不能让它跑起来——跑起来的结果看起来是正常的，只是结论悄悄反了。

2. 脚本里调用了**没声明**的工具。多半是打错字。不报错的话，Agent 会去调一个
   它自己没有的工具，ReAct 循环直接报错退出，而报告会显示"没有发现问题"。

第二条的取舍值得说明：确实存在"故意让 Agent 调用未声明工具来测出口代理"的场景，
但那种场景应当显式声明该工具再把它放到 allowlist 之外，而不是靠拼错名字。

3. 豁免规则指向未知检测器、未声明的工具，或者 `match` 里写了不认识的键。
   三种写法都让规则**永远匹配不上**，而检测器不会因此报错——它只是安静地不生效。
   用户于是以为自己已经处理过这个问题了，报告里那条 fail 却还在。这类错误在
   加载时必须说清楚，因为它是"装上了但没接线"的静音开关。

4. 记忆判据是空的（缺 path / 缺标记文本 / 缺理由）。同一个道理的反面：判据写得空
   不是"不生效"，而是**生效得过了头**——空的标记文本会命中每一个条目，报出一条
   满屏的中毒结论，把真正的问题淹掉。

6. 世界里的备注挂在一张不存在的订单上，或者是空的。备注读不到时，这一支的预期
   是 fail 就会明着失败（预期未达成），偏巧是 pass 就会安静地通过——而它证明的
   东西是零。见 `WorldDecl`。

7. `tools` 里的 `use:` 引用的工具名，库里没有（或者库文件本身不在）。这一条是
   **引用**的校验，和上面第 2、3 条同一类：名字对不上要在这里说清楚，否则报出来
   的是「这个工具缺 is_write」——那句话指着别处。库是什么、为什么要有它、覆盖
   的规矩是什么，见 `library.py` 的模块文档。
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from sandbox.detectors import KNOWN_DETECTORS
from sandbox.detectors.base import check_match
from sandbox.mock.toolspec import SAMPLE_ORDER_IDS
from sandbox.scenario import library
from sandbox.scenario.spec import Scenario

SCENARIO_SUFFIXES = (".yaml", ".yml", ".json")


class ScenarioError(ValueError):
    """场景文件不合法。消息里要带上文件名与位置，否则用户无从下手。"""


def load_scenario(path: str | Path) -> Scenario:
    """从文件加载一个场景。"""
    path = Path(path)
    if not path.exists():
        raise ScenarioError(f"场景文件不存在：{path}")
    return parse_scenario(path.read_text(encoding="utf-8"), source=str(path))


def parse_scenario(text: str, *, source: str = "<内存中的场景>") -> Scenario:
    """从 YAML 文本解析并校验一个场景。

    与 `load_scenario` 分开，是为了让 Web 编辑器能**先校验、再落盘**。编辑器
    的保存路径必须先解析成功才写文件，否则一次笔误就能把场景文件写成一个
    加载不出来的样子——而本工具的全部用例都从这些文件来，写坏一个就等于
    弄丢一个用例。写之前先问"这份文本能不能被加载"，是唯一能拦住这件事的地方。

    `source` 只用于报错信息：解析的是文本不是文件，但错误必须说清是哪一份。
    """
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ScenarioError(f"{source} 不是合法的 YAML：{exc}") from exc

    if not isinstance(raw, dict):
        raise ScenarioError(
            f"{source} 的顶层必须是一个映射（mapping），实际是 {type(raw).__name__}"
        )

    # `tools` 里的 `use:` 引用先展开，再进模型。
    #
    # **展开必须在解析这一层做**，理由有两个。一是"引用要能解析"本来就是解析
    # 器的职责（上面第 2、3 条同款）；二是展开之后 `Scenario` 才拿得到完整的
    # 工具声明——指纹、出口名单、写操作集合全都读解析后的那一份，在这里展开，
    # 它们看到的和场景作者写下的就还是同一件事（见 identity.sections）。
    #
    # 没有引用就不读库：一份不带 `use:` 的场景文件单独拷出去，旁边没有库也照跑。
    if library.uses_library(raw.get("tools")):
        try:
            raw["tools"] = library.resolve_tools(
                raw["tools"],
                library.load_library(),
                library_path=library.LIBRARY_PATH,
            )
        except library.LibraryError as exc:
            raise ScenarioError(f"{source}: {exc}") from exc

    try:
        scenario = Scenario.model_validate(raw)
    except ValidationError as exc:
        raise ScenarioError(f"{source} 场景定义不合法：\n{exc}") from exc

    _check_references(scenario, source)
    return scenario


def _check_references(scenario: Scenario, source: str) -> None:
    tool_names = set(scenario.tool_index())

    # --- 快照引用必须指向**之前**拍下的快照 ---
    defined: set[str] = set()
    for index, session in enumerate(scenario.sessions):
        if session.restore_from is not None and session.restore_from not in defined:
            raise ScenarioError(
                f"{source}: 会话 #{index}（{session.name}）要恢复到快照 "
                f"{session.restore_from!r}，但那个快照此时还不存在。\n"
                f"到这一步为止可用的快照：{sorted(defined) or '（无）'}\n"
                f"恢复到之后才拍的快照等于让记忆回到未来，"
                f"记忆视野会算错，而归因完全建立在视野上。"
            )
        # 本会话拍下的快照，供**后面的**会话引用。所以校验在循环开头、
        # 登记在循环末尾——一个会话只能恢复到它之前拍下的状态。
        for name in (session.snapshot_before, session.snapshot_after):
            if name:
                defined.add(name)

    # --- 豁免规则必须真的能命中 ---
    #
    # 三条都是"规则看着在，实际永远不生效"的写法。检测器那边一条规则匹配不上
    # 不会报错，只会安静地不静音——于是用户以为自己已经处理过这个问题了，
    # 而报告里那条 fail 还在。这类错误必须在加载时说清楚。
    for index, rule in enumerate(scenario.exemptions):
        where = f"{source}: 豁免规则 #{index}"
        if rule.detector not in KNOWN_DETECTORS:
            raise ScenarioError(
                f"{where} 指向未知检测器 {rule.detector!r}，它永远不会命中。\n"
                f"已知检测器：{sorted(KNOWN_DETECTORS)}"
            )
        if not rule.reason.strip():
            raise ScenarioError(
                f"{where}（{rule.detector}）缺 reason。"
                f"没有理由的豁免在报告里无法复核，等于永久静音。"
            )
        if rule.tool is not None and rule.tool not in tool_names:
            raise ScenarioError(
                f"{where}（{rule.detector}）指向未声明的工具 {rule.tool!r}，它永远不会命中。\n"
                f"已声明的工具：{sorted(tool_names) or '（无）'}"
            )
        try:
            check_match(rule.match)
        except ValueError as exc:
            raise ScenarioError(f"{where}（{rule.detector}）的 {exc}") from exc

    # 收口：把场景里的豁免整体转一遍（`spec.exemption_set` 会构造每一条 `Exemption`，
    # 于是 `Exemption.__post_init__` 里那些**不在这段循环里**的规矩也会在这里生效）。
    #
    # 上面那段逐条校验是为了给出带位置的好消息，但"校验只有这一处"是危险的——
    # `__post_init__` 后来加的规矩（比如"记忆族不吃豁免"）不会自动出现在上面，
    # 而漏掉的症状是规则在加载时通过、跑到检测器那里才炸。`spec.exemption_set`
    # 的文档字符串一直写着"加载时会先调一遍它"，这段就是那句话的兑现。
    try:
        scenario.exemption_set()
    except ValueError as exc:
        raise ScenarioError(f"{source}: {exc}") from exc

    # --- 字段权重必须真的用得上 ---
    #
    # 与上面几条同一类：写错了不会报错，只会安静地不生效。两种写法都在这里挡掉：
    #
    #   * 权重挂在一个不存在的字段上。检测器会拿着它去算相似度，而那个字段
    #     一次都没出现过——于是这个工具的参数看起来还是"每个字段都是身份"，
    #     翻页那一串照样被切成好几个簇（见 `ToolDecl.field_weights`）。
    #   * 两个工具给**同一个字段名**声明了不同的权重。检测器拿到的是一张扁表
    #     （字段名 → 权重），谁赢取决于工具在场景里的书写顺序——同一份场景挪
    #     一下工具顺序就换一种判法，而报告上完全看不出来。
    seen_weights: dict[str, tuple[str, float]] = {}
    for tool in scenario.tools:
        properties = (tool.parameters or {}).get("properties") or {}
        for field, weight in tool.field_weights.items():
            if field not in properties:
                raise ScenarioError(
                    f"{source}: 工具 {tool.name!r} 的 field_weights 里有 {field!r}，"
                    f"而它的参数里没有这个字段。\n"
                    f"这个工具声明的字段：{sorted(properties) or '（无）'}\n"
                    f"写错的权重不会报错，它只会让这个工具的参数看起来还是"
                    f"「每个字段都是它的身份」——这一条声明于是什么都没改。"
                )
            previous = seen_weights.get(field)
            if previous is not None and previous[1] != weight:
                raise ScenarioError(
                    f"{source}: 字段 {field!r} 被两个工具声明成了不同的权重："
                    f"{previous[0]} 说 {previous[1]}，{tool.name} 说 {weight}。\n"
                    f"检测器拿到的是一张扁的权重表（字段名 → 权重），两份声明"
                    f"合不出来——哪一份赢取决于工具在场景里的书写顺序。"
                )
            seen_weights[field] = (tool.name, weight)

    # --- 记忆判据必须真的在判东西 ---
    #
    # 两条都是"判据看着在，其实什么都没判"的写法。和豁免那三条同一类：检测器
    # 不会因为判据写得空而报错，它只会安静地给一个 pass——而那看起来和
    # "这条记忆守住了"一模一样。
    # 两条声明各判各的，变量名也分开（`must_hold` 看的是 path、`forbidden` 看的是
    # text）。共用一个 `rule` 的话，读的人得记住它此刻是哪一种，而漏掉其中一条
    # 校验的方式恰恰是"复制上面那段改一改"。
    for index, hold in enumerate(scenario.memory.must_hold):
        where = f"{source}: memory.must_hold[{index}]"
        if not hold.path.strip():
            raise ScenarioError(f"{where} 的 path 是空的，它匹配不到任何东西。")
        if not hold.reason.strip():
            raise ScenarioError(
                f"{where}（{hold.path}）缺 reason。"
                f"没有理由的判据在报告里无法复核——读者看不出为什么这条必须保持。"
            )
    for index, forbidden in enumerate(scenario.memory.forbidden):
        where = f"{source}: memory.forbidden[{index}]"
        if not forbidden.text.strip():
            # 空串是任何字符串的子串，会**命中每一个条目**——报一条满屏的
            # "中毒"，而真正的问题被淹在里面。
            raise ScenarioError(
                f"{where} 的 text 是空的。空文本是任何内容的子串，"
                f"它会命中记忆里的每一个条目。"
            )
        if not forbidden.reason.strip():
            raise ScenarioError(f"{where}（{forbidden.text!r}）缺 reason。")

    # --- 世界里的那句话必须真的能被读到 ---
    #
    # 备注挂在目标系统的一张订单上，而订单是 mock 里那两张样例单。写错订单号
    # （或者写了一张不存在的单）时，Agent 永远读不到那句话——如果这一支的预期
    # 恰好是 pass（对照支就是这样），它会安安静静地通过，而它证明的东西是零。
    for order_id, note in scenario.world.order_notes.items():
        where = f"{source}: world.order_notes[{order_id!r}]"
        if not note.strip():
            raise ScenarioError(
                f"{where} 的备注是空的。空备注在目标系统那边等同于没有备注，"
                f"而「没有备注」会让这一支无论怎么跑都通过——它看起来像"
                f"「Agent 没把世界里的东西记下来」。"
            )
        if order_id not in SAMPLE_ORDER_IDS:
            raise ScenarioError(
                f"{where} 指向一张不存在的订单。目标系统里存在的订单："
                f"{sorted(SAMPLE_ORDER_IDS)}。\n"
                f"备注挂在不存在的单上，Agent 永远读不到它——场景会以"
                f"「世界里的那句话没进记忆」收场，而那和「Agent 没把它记下来」"
                f"长得一模一样。"
            )

    # --- 脚本只能调用声明过的工具 ---
    for index, session in enumerate(scenario.sessions):
        for step_no, step in enumerate(session.script):
            for call in step.get("tool_calls", []) or []:
                name = call.get("name")
                if name not in tool_names:
                    raise ScenarioError(
                        f"{source}: 会话 #{index}（{session.name}）第 {step_no} 步调用了 "
                        f"未声明的工具 {name!r}。\n"
                        f"已声明的工具：{sorted(tool_names) or '（无）'}\n"
                        f"若确实要测「调用未声明工具被代理拦下」，"
                        f"请把该工具声明出来并排除在放行名单之外。"
                    )


def load_all(directory: str | Path) -> list[Scenario]:
    """加载一个目录下的全部场景，按文件名排序。"""
    directory = Path(directory)
    if not directory.is_dir():
        raise ScenarioError(f"场景目录不存在：{directory}")
    paths = sorted(
        p for p in directory.iterdir() if p.suffix.lower() in SCENARIO_SUFFIXES
    )
    return [load_scenario(p) for p in paths]


#: 新建场景用的骨架。界面上的"新建"直接把这段文字写进文件。
#:
#: **它是一段文字，不是"一个 dict 序列化出来的"**，因为注释要留得住：骨架是
#: 大多数用户读到的第一份场景，而工具那一段的规矩（库里的工具要写 `use:`）
#: 只有写在旁边才看得见。序列化出来的文件里塞不进注释。
#:
#: 它也不经由模型绕一圈（早先是 `model_validate` → `dump_scenario` 的写法）：
#: 那样写出来的是**展开之后**的工具声明，于是新建出来的第一个场景当场违了
#: 库里那条规矩——静态闸门会红在一份用户什么都没改的文件上。
SKELETON_TEXT = """\
# 新建场景。改完保存即可运行；完整的字段说明见 docs/scenarios.md。
id: my-scenario
description: 描述这个场景要验证什么行为。写清楚，报告里会原样打出来。

agent:
  version: react_demo@0.1
  task: 帮我把订单 A1 处理掉。
  model: stub-model

# 工具声明。库里已经有的（scenarios/_shared/tools.yaml）写 use: 引用，别再抄一遍——
# 名字在库里的工具必须这么写，要改哪个字段就在它下面补哪一行：
#
#   - use: create_payment
#     enforces_idempotency: true
#
# 覆盖是整块字段的替换，不做深合并。自己新造的工具就整段写在这儿：
#
#   - name: refund_order
#     description: 退款。有副作用。
#     is_write: true
#     parameters: {type: object, properties: {order_id: {type: string}}, required: [order_id]}
tools:
  - use: get_order

sessions:
  - name: 第一次
    script:
      - tool_calls:
          - name: get_order
            arguments: {order_id: A1}
      - content: 完成了。

expect:
  - detector: idempotency
    verdict: pass
    note: 这个场景不该报警
"""
