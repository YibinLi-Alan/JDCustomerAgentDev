"""工具子包入口 —— 对外导出工具抽象层与内置工具。

阶段三新增(见 stage-3-design.md):``BaseTool`` / ``ToolResult`` / ``@tool`` /
``ToolRegistry`` 及配套错误类型;阶段二的极简 ``Tool`` 协议与 JD mock 工具继续保留。

用法::

    from agent_framework.tools import ToolRegistry, tool

    @tool
    def current_time() -> str:
        \"\"\"查当前时间。何时用:回答里需要“今天/现在”时。\"\"\"
        ...

    registry = ToolRegistry([current_time])
    registry.invoke("current_time", {})
"""

from __future__ import annotations

from agent_framework.tools.base import (
    BaseTool,
    Tool,
    ToolError,
    ToolResult,
    ToolTimeoutError,
    ToolValidationError,
)
from agent_framework.tools.function_tool import FunctionTool, tool
from agent_framework.tools.jd_mock import (
    JD_MOCK_TOOLS,
    QueryLogisticsTool,
    QueryOrderTool,
)
from agent_framework.tools.registry import (
    ToolRegistrationError,
    ToolRegistry,
    UnknownToolError,
)

__all__ = [
    # 抽象层
    "Tool",
    "BaseTool",
    "ToolResult",
    "FunctionTool",
    "tool",
    # 注册中心
    "ToolRegistry",
    # 错误类型
    "ToolError",
    "ToolValidationError",
    "ToolTimeoutError",
    "ToolRegistrationError",
    "UnknownToolError",
    # 阶段二 mock 工具
    "QueryOrderTool",
    "QueryLogisticsTool",
    "JD_MOCK_TOOLS",
]
