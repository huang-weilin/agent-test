"""体检菜单的读取与校验。

菜单是**数据**，不是 Python——理由和场景是一致的（见 `scenario/spec.py` 的
模块注释）：改菜单的人不该被迫去改框架代码。

一条菜单项只写三样东西：这个毛病用一句人话怎么问（`question`）、挂在哪个
场景上（`scenario`）、以及为什么要查它（`why`）。**这一项该抓到什么，不在
这里。** 判卷标准写在场景自己的 `expect` 里，那是唯一的真相源；抄一份到
菜单上就是两份会各自漂的说法，而"两处都在说同一件事"正是本项目最警惕的
那类结构。

校验分两层，和场景文件一样：

1. **形状**（pydantic）：必填、不许有陌生键、标识唯一且合法。
2. **引用**（`load_checkup` 末尾）：菜单点名的场景文件必须真的在。

第 2 条是这一层最要紧的一条。少了它，菜单上可以挂一个不存在的场景，而
用户按下去才会发现——更坏的是，如果哪天有人把"场景不存在"处理成"跳过
这一项"，页面读起来会是「这一项没发现问题」，而它其实一次都没跑。
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from sandbox.config import REPO_ROOT

#: 菜单目录。和 `scenarios/` 平级：菜单本身也是一份人手编辑的输入文件。
CHECKUP_SUFFIX = ".yaml"

#: 场景目录的默认位置。**和 `api/paths.scenarios_dir()` 必须是同一个地方**，
#: 写两遍是因为那一层是 Web 的路径策略（带越界校验），这一层是"菜单点名的东西
#: 在哪"。`tests/test_checkup.py` 里有一条盯着这两个默认值相等——不需要谁记得
#: 同步，漂了会红。
SCENARIOS_DIR = REPO_ROOT / "scenarios"

_ID_RE = re.compile(r"\A[a-z0-9][a-z0-9_-]{0,63}\Z")


class CheckupError(ValueError):
    """菜单不合法。消息要能直接给用户看——他会照着它去改 YAML。"""


class CheckupItem(BaseModel):
    """体检里的一项。"""

    model_config = {"extra": "forbid"}

    id: str
    #: 这个毛病用一句人话怎么问。**它会印在结论那一行上**，所以是问句：
    #: 「同一笔钱，它会不会付两次？」读起来是一个待回答的问题，而
    #: 「重复扣款检测」是一个已经答完的题目名——后者会让读者跳过答案。
    question: str
    scenario: str
    #: 为什么要查这个毛病。它是这一项存在的理由，所以必填：一项说不出为什么
    #: 要查，多半就是凑数的，而凑数的检查会让这一页越来越长、越来越没人读。
    why: str

    @model_validator(mode="after")
    def _check(self) -> CheckupItem:
        if not _ID_RE.match(self.id):
            raise ValueError(
                f"标识 {self.id!r} 不合法：只允许小写字母、数字、连字符和下划线，"
                f"以字母或数字开头，最长 64 个字符。"
            )
        if not _ID_RE.match(self.scenario):
            raise ValueError(
                f"场景标识 {self.scenario!r} 不合法：只允许小写字母、数字、"
                f"连字符和下划线，以字母或数字开头。"
            )
        for field_name in ("question", "why"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} 不能为空——这一项就没法读了。")
        return self


class Checkup(BaseModel):
    """一份菜单。"""

    model_config = {"extra": "forbid"}

    name: str
    summary: str
    #: 摆在结论页最上面的那句话：**这一页测的是谁**。
    #:
    #: 必填，因为体检最容易出的一种误读就是把它当成"对我的系统下的诊断"。
    #: 现在菜单里的场景测的都是仓库自带的演示 Agent（它被写成一定会犯这些
    #: 错），所以这一页的结论是"这套工具能抓到什么"，不是"你的东西有多少毛病"。
    #: 接了用户自己的 Agent 之后，同一页的意思才变成后者——而那句话要说得出
    #: 这个区别，只能由菜单写。
    subject_note: str
    items: list[CheckupItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> Checkup:
        if not self.name.strip():
            raise ValueError("name 不能为空。")
        if not self.summary.strip():
            raise ValueError("summary 不能为空。")
        if not self.subject_note.strip():
            raise ValueError(
                "subject_note 不能为空。体检结论页上必须有一句话说明"
                "「这一页测的是谁」——省掉它，读者会默认这一页是在诊断他自己的系统。"
            )
        if not self.items:
            raise ValueError("items 不能为空：一份没有项的体检菜单跑出来是一页空结论。")
        seen: set[str] = set()
        for item in self.items:
            if item.id in seen:
                raise ValueError(
                    f"标识 {item.id!r} 重复了。体检结果按标识对回菜单，重名会让"
                    f"两项共用一个结论。"
                )
            seen.add(item.id)
        return self


def is_valid_id(name: str) -> bool:
    """菜单文件名是不是一个合法的标识。

    和 `api/paths.is_valid_id` 是同一个正则的两个实例——那一份管场景文件名，
    这一份管菜单文件名，两边的名字不会互相拼进对方的路径。
    """
    return bool(_ID_RE.match(name))


def checkups_dir() -> Path:
    return REPO_ROOT / "checkups"


def list_checkups(*, root: Path | None = None) -> list[tuple[str, Path]]:
    """菜单目录里全部 `.yaml`，按名字排序。

    和 `api/paths.list_scenarios` 同一条规矩：**不合法的文件名也列出来**，
    交给界面显示成"打不开"。跳过去的话，文件在磁盘上看得见、界面上找不到，
    而没有任何东西告诉用户为什么。
    """
    base = Path(root or checkups_dir())
    if not base.is_dir():
        return []
    return [
        (p.stem, p)
        for p in sorted(base.iterdir(), key=lambda p: p.name.lower())
        if p.is_file() and p.suffix.lower() == CHECKUP_SUFFIX
    ]


def _checkup_file(name: str, *, root: Path | None = None) -> Path:
    if not _ID_RE.match(name):
        raise CheckupError(
            f"体检名 {name!r} 不合法：只允许小写字母、数字、连字符和下划线，"
            f"以字母或数字开头，最长 64 个字符。"
        )
    base = Path(root or checkups_dir()).resolve()
    path = (base / f"{name}{CHECKUP_SUFFIX}").resolve()
    # 第二道防线，和 `api/paths.scenario_file` 一样：正则已经挡掉了分隔符，
    # 但包含性检查不依赖正则的细节。
    if path.parent != base:
        raise CheckupError(f"体检 {name!r} 解析到了菜单目录之外（{path}），已拒绝。")
    return path


def load_checkup(
    name: str, *, root: Path | None = None, scenarios_root: Path | None = None
) -> Checkup:
    """按名字加载一份菜单并校验。

    `scenarios_root` 只为测试而存在：真的那份在 `SCENARIOS_DIR`。
    """
    path = _checkup_file(name, root=root)
    if not path.exists():
        raise CheckupError(f"体检菜单不存在：{path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CheckupError(f"{path} 不是合法的 YAML：{exc}") from exc

    if not isinstance(raw, dict):
        raise CheckupError(
            f"{path} 的顶层必须是一个映射（mapping），实际是 {type(raw).__name__}"
        )

    try:
        checkup = Checkup.model_validate(raw)
    except ValidationError as exc:
        raise CheckupError(f"{path} 不合法：\n{exc}") from exc

    _check_scenarios_exist(checkup, path, scenarios_root=scenarios_root)
    return checkup


def _check_scenarios_exist(
    checkup: Checkup, source: Path, *, scenarios_root: Path | None = None
) -> None:
    """菜单点名的场景文件必须真的在。

    **一次报全，不是报第一个。** 菜单是手工维护的，改完场景文件名之后往往
    一次错好几个；报一个改一个要来回好几趟。

    这一条挡住的失效很安静：菜单里挂着不存在的场景，跑的时候才知道。若调用方
    图省事把"加载不出来"处理成"这一项略过"，页面上那一项就会以"没有发现问题"
    的样子出现——而它一次都没跑。
    """
    base = Path(scenarios_root or SCENARIOS_DIR)
    missing = [
        item.scenario
        for item in checkup.items
        if not (base / f"{item.scenario}.yaml").is_file()
    ]
    if missing:
        raise CheckupError(
            f"{source}: 菜单里有 {len(missing)} 项点名的场景不存在："
            f"{'、'.join(missing)}。\n"
            f"场景目录：{base}\n"
            f"改菜单里那一行的 `scenario:`，或者把场景文件补上。"
        )
