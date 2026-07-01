"""工具子包入口 —— 对外导出阶段二的极简 ``Tool`` 协议与京东 mock 工具。

用法::

    from agent_framework.tools import JD_MOCK_TOOLS
    agent = ReActAgent(llm, tools=JD_MOCK_TOOLS)

阶段三会在此追加完整的 ``BaseTool`` / ``ToolRegistry`` 与更多真实工具。
"""

from __future__ import annotations

from agent_framework.tools.base import Tool
from agent_framework.tools.jd_mock import (
    JD_MOCK_TOOLS,
    QueryLogisticsTool,
    QueryOrderTool,
)

__all__ = [
    "Tool",
    "QueryOrderTool",
    "QueryLogisticsTool",
    "JD_MOCK_TOOLS",
]
