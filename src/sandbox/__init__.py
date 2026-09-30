"""AI Agent 行为级测试沙盒。

架构决策见 docs/decisions/：
  001 - 集成面采用黑盒边界拦截（LLM 网关 + 工具出口代理），而非进程内 SDK 注入
  002 - 自研数据库是 system of record，Langfuse 仅作可视化前端
  003 - 出口默认拒绝、默认不注入真实凭据

检测器规格（精度目标、白名单、结果语义）见 docs/detectors.md。
"""

__version__ = "0.0.1"
