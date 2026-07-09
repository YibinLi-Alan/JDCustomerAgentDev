"""Agent 间通信协议 —— 专员封装、任务消息结构、花名册渲染(大纲产出④)。

设计要点(stage-5-design.md §6.1):

- 专员之间**不直接通话**,一切经编排器(Router/Supervisor)中转,星型拓扑;
- :class:`Specialist` 是纯数据 + 工厂方法:名字 + 职责 + 工具子集 + 专属 prompt,
  不持有会话状态;内核就是阶段三的 ``ToolCallingAgent``(核心零改动);
- **失败约定**:专员办不到时答复以 ``FAILURE_MARKER`` 开头(写进公共 prompt 底座),
  编排器据此判 ``ok``——这就是"通信协议"里的状态码;
- Router 直派和 Supervisor 派工共用同一批 ``Specialist`` 对象:定义一次、两种编排复用。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from agent_framework.core.agent import AgentResult, ToolCallingAgent
from agent_framework.core.llm import LLM
from agent_framework.tools.registry import ToolRegistry

#: 专员「明示办不到」的答复前缀(prompt 约定 + 编排器解析,协议的两端)。
FAILURE_MARKER = "无法完成"


# --------------------------------------------------------------------------- #
# 消息结构                                                                       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TaskAssignment:
    """编排器 → 专员的一次任务派发。

    Attributes:
        task: 任务描述(Router 直派 = 用户原话;Supervisor 派工 = 计划步骤描述)。
        context: 共享上下文渲染文本(ScratchPad 前序结论等),可为空串。
    """

    task: str
    context: str = ""

    def to_user_input(self) -> str:
        """拼成专员的 user 输入:前序结论在前、当前任务在后。"""
        if not self.context:
            return self.task
        return f"{self.context}\n\n【当前任务】\n{self.task}"


@dataclass(frozen=True)
class TaskOutcome:
    """专员 → 编排器的执行回报。

    Attributes:
        specialist: 执行专员的机器名。
        answer: 专员的最终答复。
        ok: 是否成功。撞步数上限或答复以 ``FAILURE_MARKER`` 开头均为 False。
        trace: 专员内环完整轨迹(/trace 与调试用)。
    """

    specialist: str
    answer: str
    ok: bool
    trace: AgentResult | None = None


# --------------------------------------------------------------------------- #
# 专员封装                                                                       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Specialist:
    """一名专员的完整定义(纯数据,无会话状态)。

    Attributes:
        name: 机器名(如 ``"order_agent"``),Router/Planner 指派时用。
        title: 中文头衔(如 ``"订单物流专员"``),进 prompt 与展示。
        description: 职责边界一句话——Router 分诊与 Planner 选人的**唯一依据**,
            要写清管什么、不管什么。
        registry: 工具子集(``ToolRegistry.subset()`` 切出,实例与全量库共享)。
        system_prompt: 专属 system prompt(公共底座 + 域块,见 specialists.py)。
    """

    name: str
    title: str
    description: str
    registry: ToolRegistry
    system_prompt: str

    def build(self, llm: LLM, *, extra_system: str = "", max_steps: int = 5) -> ToolCallingAgent:
        """按轮现造专员 Agent(无状态,用完即弃)。

        Args:
            llm: LLM 实例(全团队共享一个连接)。
            extra_system: 追加的 system 附加段(阶段四记忆的
                ``ctx.system_suffix()`` 从这里注入,专员共享同一份用户事实)。
            max_steps: 专员内环步数上限(循环护栏之一)。
        """
        system = self.system_prompt + extra_system
        return ToolCallingAgent(llm, self.registry, max_steps=max_steps, system_prompt=system)

    def handle(
        self,
        llm: LLM,
        assignment: TaskAssignment,
        *,
        extra_system: str = "",
        max_steps: int = 5,
    ) -> TaskOutcome:
        """执行一次任务派发,回报 :class:`TaskOutcome`。

        ``ok`` 判定(协议约定):撞步数上限(``stopped_reason == "max_steps"``)
        或答复以 :data:`FAILURE_MARKER` 开头 → False;其余 True。
        """
        agent = self.build(llm, extra_system=extra_system, max_steps=max_steps)
        result = agent.run(assignment.to_user_input())
        failed = result.stopped_reason == "max_steps" or result.final_answer.strip().startswith(
            FAILURE_MARKER
        )
        return TaskOutcome(
            specialist=self.name, answer=result.final_answer, ok=not failed, trace=result
        )


def render_roster(specialists: Sequence[Specialist]) -> str:
    """专员花名册渲染文本(Router 分诊与 Planner 选人的 prompt 素材)。"""
    return "\n".join(f"- {s.name}({s.title}):{s.description}" for s in specialists)


# --------------------------------------------------------------------------- #
# 协议层的 JSON 解析(Router 路由决策 / Critic 审查结论共用)                          #
# --------------------------------------------------------------------------- #


def parse_json_object(text: str) -> dict[str, object] | None:
    """从模型输出解析 JSON 对象;容忍 ``` 围栏;不是对象返回 None。"""
    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
