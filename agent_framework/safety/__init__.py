"""Safety 子包 —— 输入/输出防护、限流、HITL 审批(阶段六)。

- :mod:`input_filter`:输入清洗 + 注入检测(标记不硬拦)+ prompt 加固条款 +
  工具返回边界标记(间接注入防御);
- :mod:`output_filter`:出口敏感信息脱敏(朴素正则,诚实边界见模块文档);
- :mod:`rate_limiter`:滑动窗口限流 + 单任务 token 预算(进程内,生产外置);
- :mod:`approval`:**HITL 核心** —— ApprovalGate 权限闸门(Registry 装饰器)+
  HandoffQueue 统一人工队列(审批/升级两入口,JSON 落盘)+ 审批后幂等执行。

防御纵深的排序:注入防不胜防,**最小权限(闸门)是最后一道墙**——
就算模型被骗,能造成的破坏也被权限锁死。
"""

from agent_framework.safety.approval import (
    ApprovalGate,
    ApprovalPolicy,
    HandoffItem,
    HandoffQueue,
    PendingAction,
)
from agent_framework.safety.input_filter import (
    HARDENING_CLAUSE,
    TOOL_DATA_PREFIX,
    TOOL_DATA_SUFFIX,
    BoundaryRegistry,
    InputCheck,
    inspect_input,
    wrap_tool_data,
)
from agent_framework.safety.output_filter import OutputCheck, filter_output
from agent_framework.safety.rate_limiter import RateLimiter, TokenBudget

__all__ = [
    "HARDENING_CLAUSE",
    "TOOL_DATA_PREFIX",
    "TOOL_DATA_SUFFIX",
    "ApprovalGate",
    "ApprovalPolicy",
    "BoundaryRegistry",
    "HandoffItem",
    "HandoffQueue",
    "InputCheck",
    "OutputCheck",
    "PendingAction",
    "RateLimiter",
    "TokenBudget",
    "filter_output",
    "inspect_input",
    "wrap_tool_data",
]
