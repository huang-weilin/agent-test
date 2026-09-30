"""记忆载体适配器。每种记忆实现一个：Hermes 文件型、LangGraph checkpointer 等。

新增适配器时的两条要求：
  1. `restore(capture())` 必须是恒等操作（对称性）。
  2. 快照必须是**确定性**字节——同一份记忆两次 capture 要产出同样的内容，
     否则内容寻址失效，"这次恢复其实什么都没变"就判断不出来。
"""

from sandbox.memory.adapters.file_memory import FileMemoryAdapter, RestoreRefused

__all__ = ["FileMemoryAdapter", "RestoreRefused"]
