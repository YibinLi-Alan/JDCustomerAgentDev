"""agent_framework:从零搭建的可复用 Agent 框架。

对外导出各阶段的公共接口与类型:
- 阶段一:LLM 抽象 + 配置 + provider 工厂(``create_llm``)。
- 阶段二:ReAct 最小 Agent 循环(``ReActAgent``)+ 极简 ``Tool`` 接口 + mock 工具。

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
    parse_step,
)
from agent_framework.core.config import Settings, get_settings
from agent_framework.core.llm import LLM, ChatResponse, Message, Usage, create_llm
from agent_framework.core.llm_claude import ClaudeLLM
from agent_framework.core.llm_openai import OpenAILLM
from agent_framework.tools import (
    JD_MOCK_TOOLS,
    QueryLogisticsTool,
    QueryOrderTool,
    Tool,
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
    # 阶段二:工具
    "Tool",
    "QueryOrderTool",
    "QueryLogisticsTool",
    "JD_MOCK_TOOLS",
]

__version__ = "0.2.0"
