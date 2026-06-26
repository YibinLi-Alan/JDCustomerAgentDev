"""agent_framework:从零搭建的可复用 Agent 框架。

阶段一对外导出基础接口与类型(LLM 抽象 + 配置 + provider 工厂)。上层代码
(CLI、实验脚本、未来的 Agent)应只从这里导入接口与类型,并通过 ``create_llm``
按配置拿到具体实现,不直接 import ``anthropic`` / ``openai``。
"""

from agent_framework.core.config import Settings, get_settings
from agent_framework.core.llm import LLM, ChatResponse, Message, Usage, create_llm
from agent_framework.core.llm_claude import ClaudeLLM
from agent_framework.core.llm_openai import OpenAILLM

__all__ = [
    "LLM",
    "Message",
    "Usage",
    "ChatResponse",
    "create_llm",
    "ClaudeLLM",
    "OpenAILLM",
    "Settings",
    "get_settings",
]

__version__ = "0.1.0"
