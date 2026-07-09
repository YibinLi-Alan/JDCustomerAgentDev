"""Multi-Agent 子包 —— 专员协作编排(阶段五)。

- :mod:`protocol`:通信协议(``Specialist`` 封装、任务/回报消息、失败前缀约定);
- :mod:`specialists`:三个业务专员的定义(订单物流/售后/商品导购);
- :mod:`router`:模式一,入口分诊直派(Swarm handoff 思想);
- :mod:`supervisor`:模式二,中心调度(按计划派工 + 重规划 + 汇总 + 质检);
- :mod:`critic`:终稿质检(Reflection/Critic,不合格回炉一次)。

依赖方向:multi_agent → planning(不反向);``core/agent.py`` 零改动。
设计文档:docs/stage-5-design.md §6/§7。
"""

from agent_framework.multi_agent.critic import CRITIC_SYSTEM, Critic, Critique
from agent_framework.multi_agent.protocol import (
    FAILURE_MARKER,
    Specialist,
    TaskAssignment,
    TaskOutcome,
    parse_json_object,
    render_roster,
)
from agent_framework.multi_agent.router import (
    DIRECT_TARGET,
    ROUTE_SYSTEM,
    SUPERVISOR_TARGET,
    RouteDecision,
    Router,
)
from agent_framework.multi_agent.specialists import create_specialists
from agent_framework.multi_agent.supervisor import (
    SYNTHESIZE_SYSTEM,
    Supervisor,
    SupervisorResult,
)

__all__ = [
    "CRITIC_SYSTEM",
    "DIRECT_TARGET",
    "FAILURE_MARKER",
    "ROUTE_SYSTEM",
    "SUPERVISOR_TARGET",
    "SYNTHESIZE_SYSTEM",
    "Critic",
    "Critique",
    "RouteDecision",
    "Router",
    "Specialist",
    "Supervisor",
    "SupervisorResult",
    "TaskAssignment",
    "TaskOutcome",
    "create_specialists",
    "parse_json_object",
    "render_roster",
]
