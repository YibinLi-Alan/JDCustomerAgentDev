"""Router 模式(模式一):入口分诊,一次决策直派专员或升级 Supervisor。

Swarm handoff 思想的入口版(stage-5-design.md §6.3):分诊只在入口发生一次,
派出后该专员全权负责本轮;不做运行中随时互相 handoff(链路不可控)。
降级原则:解析失败/未知目标 → 判 "supervisor"——宁走贵的全能路径,不派错专员。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from agent_framework.core.llm import LLM, Message
from agent_framework.multi_agent.protocol import Specialist, parse_json_object, render_roster

#: 「复杂问题,升级中心调度」的路由目标(不是专员机器名)。
SUPERVISOR_TARGET = "supervisor"

#: 「无需派工,客服直接回复」的路由目标(告知类/寒暄类输入的出口,
#: 2026-07-09 用户自测后评审新增:自报家门不该被派进带工具的专员流水线)。
DIRECT_TARGET = "direct"

ROUTE_SYSTEM = (
    "你是京东客服的分诊员。根据用户诉求,判定处理路径:直接回复、派给一位专员,"
    "或升级为需要多专员协作的复杂问题。\n"
    "判定规则:\n"
    "- 诉求不需要查询或操作任何数据——如自报姓名/地址等个人信息、寒暄、感谢、"
    f'询问你能做什么 → 输出 "{DIRECT_TARGET}"(客服直接回复,不派工);\n'
    "- 诉求只涉及单一专员的职责,且不需要“先查证再操作”的多步骤配合 → 输出该专员的机器名;\n"
    f'- 诉求涉及多个领域、包含多个动作、或需要先查证再操作 → 输出 "{SUPERVISOR_TARGET}"。\n'
    "只输出 JSON,不要多余文字/解释/代码围栏:\n"
    '{"target": "<direct或专员机器名或supervisor>", "reason": "<一句话理由>"}'
)


@dataclass(frozen=True)
class RouteDecision:
    """一次分诊决策。

    Attributes:
        target: 专员机器名、:data:`SUPERVISOR_TARGET`(升级中心调度),
            或 :data:`DIRECT_TARGET`(无需派工,客服直接回复)。
        reason: 一句话理由(trace 展示用;降级时说明降级原因)。
    """

    target: str
    reason: str


class Router:
    """入口分诊器(一次 LLM 调用,无循环)。"""

    def __init__(self, llm: LLM, specialists: Mapping[str, Specialist]) -> None:
        self._llm = llm
        self._specialists = dict(specialists)
        self._roster = render_roster(list(self._specialists.values()))

    def route(self, query: str, *, context: str = "") -> RouteDecision:
        """给用户诉求分诊。任何失败降级为 supervisor,永不抛。

        Args:
            query: 用户本轮诉求原文。
            context: 已知背景(记忆附加段等),帮助理解指代;可为空。

        Returns:
            :class:`RouteDecision`;``target`` 保证是合法专员名或 supervisor。
        """
        prompt = (
            f"【专员花名册】\n{self._roster}\n\n"
            f"【已知背景】\n{context or '(无)'}\n\n"
            f"【用户诉求】\n{query}"
        )
        try:
            response = self._llm.chat([Message("user", prompt)], system=ROUTE_SYSTEM)
            data = parse_json_object(response.content)
        except Exception:  # noqa: BLE001 — 分诊失败必须降级,不能拦住用户
            data = None
        if data is None:
            return RouteDecision(SUPERVISOR_TARGET, "分诊输出无法解析,降级走中心调度")
        target = str(data.get("target", "")).strip()
        reason = str(data.get("reason", "")).strip()
        if target not in (SUPERVISOR_TARGET, DIRECT_TARGET) and target not in self._specialists:
            return RouteDecision(SUPERVISOR_TARGET, f"分诊目标 {target!r} 不存在,降级走中心调度")
        return RouteDecision(target, reason)
