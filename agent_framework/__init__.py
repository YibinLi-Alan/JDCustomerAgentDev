"""agent_framework:从零搭建的可复用 Agent 框架。

对外导出各阶段的公共接口与类型:
- 阶段一:LLM 抽象 + 配置 + provider 工厂(``create_llm``)。
- 阶段二:ReAct 最小 Agent 循环(``ReActAgent``)+ 极简 ``Tool`` 接口 + mock 工具。
- 阶段三:Tool Use 系统 —— ``BaseTool``(strict mode)+ ``ToolRegistry`` + ``@tool``
  + 原生 Function Calling(``ToolCallingAgent`` / ``ToolCall``)+ 11 个内置工具
  (``default_registry`` 一行装配)。

上层代码(CLI、示例、未来的业务)应只从这里导入接口与类型,并通过 ``create_llm``
按配置拿到具体 LLM 实现,不直接 import ``anthropic`` / ``openai``。
"""

from agent_framework.core.agent import (
    AgentAction,
    AgentResult,
    AgentStep,
    ReActAgent,
    StepParseError,
    StepTrace,
    ToolCallingAgent,
    parse_step,
)
from agent_framework.core.config import Settings, get_settings
from agent_framework.core.llm import (
    LLM,
    ChatResponse,
    Message,
    ToolCall,
    Usage,
    create_llm,
)
from agent_framework.core.llm_claude import ClaudeLLM
from agent_framework.core.llm_openai import OpenAILLM
from agent_framework.tools import (
    JD_MOCK_TOOLS,
    BaseTool,
    QueryLogisticsTool,
    QueryOrderTool,
    Tool,
    ToolRegistry,
    ToolResult,
    default_registry,
    tool,
)

__all__ = [
    # 阶段一:LLM + 配置
    "LLM",
    "Message",
    "Usage",
    "ChatResponse",
    "create_llm",
    "ClaudeLLM",
    "OpenAILLM",
    "Settings",
    "get_settings",
    # 阶段二:ReAct Agent
    "ReActAgent",
    "AgentResult",
    "AgentStep",
    "AgentAction",
    "StepTrace",
    "StepParseError",
    "parse_step",
    # 阶段三:Function Calling + 工具系统
    "ToolCallingAgent",
    "ToolCall",
    "BaseTool",
    "ToolResult",
    "ToolRegistry",
    "tool",
    "default_registry",
    # 工具(阶段二遗留导出)
    "Tool",
    "QueryOrderTool",
    "QueryLogisticsTool",
    "JD_MOCK_TOOLS",
]

__version__ = "0.3.0"
