"""文件型记忆适配器（Hermes 式：记忆就是 Agent 目录下的一堆文件）。

**这个文件里有一个破坏性操作，读之前先知道它是什么。**

`restore()` 会把整个记忆目录删掉再按快照重建。这是刻意的——"把 Agent 的记忆
回滚到周三之前"这件事本身就是破坏性的，做成"合并"反而更危险：合并会留下一半
新一半旧的混合状态，而那种状态在真实系统里根本不存在，测出来的结论也就没有意义。

三道闸：
  1. `memory_restore_guard` 开启时，只允许恢复到 `settings.memory_root` 之下。
     指向别处会直接拒绝——不能因为路径写错就把用户的东西删了。
  2. 目标目录必须已存在，或显式允许创建。手滑写一个不存在的路径不该凭空建目录。
  3. 恢复前校验快照摘要。内容对不上就拒绝恢复，宁可报错也不覆盖。

快照格式是**确定性 JSON**（键排序、固定分隔符），不是 tar。理由是内容寻址需要
`capture()` 对同一份记忆产出同样的字节，而 tar 会带上 mtime 和遍历顺序。
顺带的好处：出问题时可以直接打开快照看看到底存了什么。
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from sandbox.config import settings

FORMAT_VERSION = 1
ADAPTER_NAME = "hermes_file"


class RestoreRefused(RuntimeError):
    """恢复被闸门拦下。**不是内部错误，是设计如此**——别把它 catch 成通用异常。"""


@dataclass(frozen=True)
class FileMemoryAdapter:
    """把 `root` 目录整体当作 Agent 的记忆。

    `root` 应当由测试框架创建并交给被测 Agent（通过它自己的配置项），
    而不是指向真实用户的某个目录。
    """

    root: Path
    name: str = ADAPTER_NAME

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    # --- capture ---------------------------------------------------------

    def capture(self) -> bytes:
        """把整个目录序列化成确定性 JSON。目录不存在 → 空快照，不报错。

        "还没建过记忆目录"是合法状态（新 Agent 第一次跑），把它当错误会让
        会话边界的第一次快照失败，而那次快照恰恰是用来做基线的。
        """
        entries: list[dict[str, object]] = []
        if self.root.exists():
            for path in sorted(self._iter_files(), key=lambda p: p.as_posix()):
                blob = path.read_bytes()
                entries.append(
                    {
                        "path": path.relative_to(self.root).as_posix(),
                        "sha256": hashlib.sha256(blob).hexdigest(),
                        "content_b64": base64.b64encode(blob).decode("ascii"),
                    }
                )
        doc = {
            "adapter": ADAPTER_NAME,
            "version": FORMAT_VERSION,
            "root_name": self.root.name,
            "entries": entries,
        }
        # sort_keys + 固定分隔符：同一份内容必须产出同样的字节，否则内容寻址失效，
        # "这次恢复其实什么都没变"就判断不出来了。
        return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )

    def _iter_files(self) -> list[Path]:
        """列文件。**不跟任何链接出目录。**

        跟着符号链接走会在 capture 时把目录外的文件吸进快照（可能包含凭据），
        更糟的是 restore 时会往目录外写。

        这里用 `os.walk(followlinks=False)` 而不是 `rglob` + 逐个 `is_symlink()`：
        后者只挡住了链接**文件**，却挡不住链接**目录**——rglob 会先走进那个目录，
        再把它里面的普通文件列出来，而那些文件本身不是链接，逐个检查根本查不出来。
        把外面一整棵树吸进快照就是这么发生的。
        """
        files: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
            # 剔除链接目录。**必须显式做**：os.walk(followlinks=False) 挡不住
            # Windows 的目录联接，详见 _is_link_like 的说明。
            dirnames[:] = [
                d for d in dirnames if not _is_link_like(Path(dirpath) / d)
            ]
            for name in filenames:
                path = Path(dirpath) / name
                if _is_link_like(path):
                    continue
                files.append(path)
        return files

    def entry_count(self) -> int:
        return len(self._iter_files()) if self.root.exists() else 0

    # --- restore ---------------------------------------------------------

    def restore(self, payload: bytes, *, allow_create: bool = False) -> None:
        """把记忆目录覆盖成快照里的状态。**会删掉目录现有内容。**

        `allow_create=True` 时才允许目标目录不存在。
        """
        doc = _parse(payload)
        self._guard(doc, allow_create=allow_create)

        # rmtree 不跟随符号链接删除，所以目录里就算有人塞了链接也不会被带到外面。
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

        for entry in doc["entries"]:
            target = self.root / entry["path"]
            # 快照里的相对路径理论上不会逃逸，但恢复是覆盖写，值得再挡一道：
            # 一个 `..` 就是写到目录外去了。
            if not _is_within(target, self.root):
                raise RestoreRefused(
                    f"快照里的路径 {entry['path']!r} 指向记忆目录之外，拒绝恢复。"
                )
            blob = base64.b64decode(entry["content_b64"])
            actual = hashlib.sha256(blob).hexdigest()
            if actual != entry["sha256"]:
                raise RestoreRefused(
                    f"快照内 {entry['path']!r} 内容与摘要不符，拒绝写入。"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)

    def _guard(self, doc: dict, *, allow_create: bool) -> None:
        if doc.get("adapter") != ADAPTER_NAME:
            raise RestoreRefused(
                f"快照的适配器是 {doc.get('adapter')!r}，不能喂给 {ADAPTER_NAME!r}。"
                f"拿 A 格式的字节去覆盖 B 格式的存储会把记忆砸坏。"
            )
        if doc.get("version") != FORMAT_VERSION:
            raise RestoreRefused(
                f"快照格式版本 {doc.get('version')} 与本适配器 {FORMAT_VERSION} 不符，拒绝恢复。"
            )
        if not self.root.exists() and not allow_create:
            raise RestoreRefused(
                f"记忆目录 {self.root} 不存在。确认这是你要覆盖的目录，"
                f"再显式传 allow_create=True。"
            )
        if settings.memory_restore_guard and not _is_within(
            self.root, settings.memory_root
        ):
            raise RestoreRefused(
                f"记忆目录 {self.root} 不在 {settings.memory_root} 之下，"
                f"而 memory_restore_guard 是开着的。恢复快照会清空目标目录——"
                f"这条闸就是防止路径写错时删掉不该删的东西。"
                f"确实要恢复到别处，请显式设置 SANDBOX_MEMORY_RESTORE_GUARD=0。"
            )


def _parse(payload: bytes) -> dict:
    try:
        doc = json.loads(payload.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise RestoreRefused(f"快照内容不是合法的 {ADAPTER_NAME} 格式：{exc}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("entries"), list):
        raise RestoreRefused("快照缺少 entries 字段，无法恢复。")
    return doc


def _is_link_like(path: Path) -> bool:
    """这个目录项会不会把遍历带到别的地方去。

    **不能只用 `is_symlink()`。** Windows 上目录联接（junction，`mklink /J`）
    走的是另一套机制，实测下来：

        pathlib.Path.is_symlink()   → False
        os.path.islink()            → False
        stat.S_ISLNK(st_mode)       → False
        os.lstat().st_reparse_tag   → 0xa0000003  ← 只有这个认得出来

    而 `os.walk(followlinks=False)` 内部用的就是 `islink`，所以它对 junction
    照样往下走。只靠 is_symlink 的防护在 Windows 上是**空的**：捕获快照时会把
    联接目标那边的整棵树吸进来（可能含凭据），而目标在记忆目录之外。

    查重解析标记能一并覆盖符号链接（0xA000000C）、目录联接（0xA0000003），
    以及 OneDrive 占位文件之类的其它重解析点——那些同样不该被跟进。
    （`shutil.rmtree` 这边是安全的：实测它只删链接本身，不跟进目标。）
    """
    if path.is_symlink():
        return True
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return getattr(st, "st_reparse_tag", 0) not in (0, None)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def summarize_payload(payload: bytes) -> dict[str, object]:
    """不恢复、只看快照里有什么。用于报告与人工排查。"""
    doc = _parse(payload)
    return {
        "adapter": doc.get("adapter"),
        "version": doc.get("version"),
        "entries": [e.get("path") for e in doc["entries"]],
    }
