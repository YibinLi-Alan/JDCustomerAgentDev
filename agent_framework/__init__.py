"""agent_framework:从零搭建的可复用 Agent 框架。

对外导出各阶段的公共接口与类型:
- 阶段一:LLM 抽象 + 配置 + provider 工厂(``create_llm``)。
- 阶段二:ReAct 最小 Agent 循环(``ReActAgent``)+ 极简 ``Tool`` 接口 + mock 工具。
- 阶段三:Tool Use 系统 —— ``BaseTool``(strict mode)+ ``ToolRegistry`` + ``@tool``
  + 原生 Function Calling(``ToolCallingAgent`` / ``ToolCall``)+ 11 个内置工具
  (``default_registry`` 一行装配)。
- 阶段四:Memory 与上下文管理 —— ``MemoryManager`` 统一门面(短期滑动窗口 +
  递归摘要 + 长期三因子检索,``create_memory_manager`` 一行装配)。
- 阶段五:Planning 与 Multi-Agent —— ``Planner``/``PlanExecutor``(先规划再执行 +
  动态重规划)+ ``Router``/``Supervisor`` 双编排模式 + 三业务专员
  (``create_specialists`` 一行装配)+ ``Critic`` 终稿质检。
- 阶段六:生产化 —— 可靠性(``ReliableLLM``/``FallbackLLM`` 重试降级)、可观测
  (``Tracer``/metrics)、安全(``ApprovalGate``/``HandoffQueue`` HITL + 过滤/限流)、
  评估(``Judge`` LLM-as-Judge)、``AgentService`` 整栈门面 + FastAPI 服务。

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
from agent_framework.core.llm_reliable import FallbackLLM, ReliableLLM
from agent_framework.evaluation.judge import Judge
from agent_framework.memory import (
    LongTermMemory,
    MemoryContext,
    MemoryManager,
    MemoryRecord,
    ScoredMemory,
    ShortTermMemory,
    SummaryCompressor,
    Turn,
    TurnReport,
    WriteOp,
    create_memory_manager,
)
from agent_framework.multi_agent import (
    Critic,
    Critique,
    RouteDecision,
    Router,
    Specialist,
    Supervisor,
    SupervisorResult,
    TaskAssignment,
    TaskOutcome,
    create_specialists,
    render_roster,
)
from agent_framework.observability import (
    TaskMetrics,
    TraceEvent,
    Tracer,
    aggregate,
    summarize_trace,
)
from agent_framework.planning import (
    ExecutionResult,
    Plan,
    PlanExecutor,
    Planner,
    PlanStep,
    ScratchPad,
    StepResult,
)
from agent_framework.safety import (
    ApprovalGate,
    ApprovalPolicy,
    BoundaryRegistry,
    HandoffItem,
    HandoffQueue,
    RateLimiter,
    TokenBudget,
    filter_output,
    inspect_input,
)
from agent_framework.service import AgentService, ServiceResult
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
    # 阶段四:Memory 与上下文管理
    "MemoryManager",
    "MemoryContext",
    "TurnReport",
    "ShortTermMemory",
    "Turn",
    "SummaryCompressor",
    "LongTermMemory",
    "MemoryRecord",
    "ScoredMemory",
    "WriteOp",
    "create_memory_manager",
    # 阶段五:Planning 与 Multi-Agent
    "Plan",
    "PlanStep",
    "Planner",
    "PlanExecutor",
    "ExecutionResult",
    "StepResult",
    "ScratchPad",
    "Specialist",
    "TaskAssignment",
    "TaskOutcome",
    "render_roster",
    "create_specialists",
    "Router",
    "RouteDecision",
    "Supervisor",
    "SupervisorResult",
    "Critic",
    "Critique",
    # 阶段六:生产化
    "ReliableLLM",
    "FallbackLLM",
    "Tracer",
    "TraceEvent",
    "TaskMetrics",
    "aggregate",
    "summarize_trace",
    "ApprovalGate",
    "ApprovalPolicy",
    "HandoffQueue",
    "HandoffItem",
    "BoundaryRegistry",
    "RateLimiter",
    "TokenBudget",
    "inspect_input",
    "filter_output",
    "Judge",
    "AgentService",
    "ServiceResult",
]

__version__ = "0.6.0"
