"""Planning 子包 —— 任务规划与计划执行(阶段五)。

- :mod:`planner`:``Plan``/``PlanStep`` 数据结构 + ``Planner``(生成与动态重规划);
- :mod:`executor`:``PlanExecutor``(顺序调度、失败上报、重规划回路)+
  ``ScratchPad``(步骤间共享黑板)。

设计文档:docs/stage-5-design.md §5。本子包不 import ``multi_agent``
(依赖方向:multi_agent → planning,不反向)。
"""

from agent_framework.planning.executor import (
    ExecutionResult,
    PlanExecutor,
    ScratchPad,
    StepResult,
    StepRunner,
)
from agent_framework.planning.planner import Plan, Planner, PlanStep

__all__ = [
    "ExecutionResult",
    "Plan",
    "PlanExecutor",
    "PlanStep",
    "Planner",
    "ScratchPad",
    "StepResult",
    "StepRunner",
]
