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
    Permission,
    Tool,
    ToolError,
    ToolResult,
    ToolTimeoutError,
    ToolValidationError,
)
from agent_framework.tools.common import (
    HttpRequestTool,
    calculator,
    create_common_tools,
    current_time,
)
from agent_framework.tools.function_tool import FunctionTool, tool
from agent_framework.tools.jd_mock import (
    JD_MOCK_TOOLS,
    ApplyRefundTool,
    CancelOrderTool,
    CreateTicketTool,
    QueryLogisticsTool,
    QueryOrderTool,
    QueryProductTool,
    QueryUserOrdersTool,
    SearchFAQTool,
    create_jd_tools,
)
from agent_framework.tools.jd_mock_data import DEFAULT_STORE, JDMockStore
from agent_framework.tools.presets import default_registry
from agent_framework.tools.registry import (
    ToolRegistrationError,
    ToolRegistry,
    UnknownToolError,
)

__all__ = [
    # 抽象层
    "Tool",
    "BaseTool",
    "Permission",
    "ToolResult",
    "FunctionTool",
    "tool",
    # 注册中心与装配
    "ToolRegistry",
    "default_registry",
    "create_jd_tools",
    "create_common_tools",
    # 错误类型
    "ToolError",
    "ToolValidationError",
    "ToolTimeoutError",
    "ToolRegistrationError",
    "UnknownToolError",
    # 业务工具(A 组)与数据层
    "QueryOrderTool",
    "QueryLogisticsTool",
    "QueryProductTool",
    "SearchFAQTool",
    "QueryUserOrdersTool",
    "ApplyRefundTool",
    "CancelOrderTool",
    "CreateTicketTool",
    "JDMockStore",
    "DEFAULT_STORE",
    "JD_MOCK_TOOLS",
    # 通用工具(B 组)
    "calculator",
    "current_time",
    "HttpRequestTool",
]
