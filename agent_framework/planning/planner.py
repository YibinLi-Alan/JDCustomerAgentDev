"""Planner —— 把复杂诉求拆成线性步骤计划,并支持失败后的动态重规划。

设计要点(stage-5-design.md §5.1/§5.2,评审拍板):

- **线性步骤列表**,不做 DAG:客服场景步骤强依赖(先查证→再操作→再推荐),
  线性让重规划语义干净(截断剩余步骤重排);
- 拆解时就完成「步骤→专员」指派(prompt 里带专员花名册);产出校验:
  引用不存在的专员一律矫正为**兜底专员**(``specialists[0]``);
- **全链路降级**:计划 JSON 非法 → 单步计划(整个诉求打包给兜底专员,
  = 退化成阶段三的单 Agent 行为),永不炸;
- ``replan()`` 只重排**未完成部分**,已完成步骤不重跑(幂等意识,阶段六伏笔)。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent_framework.core.llm import LLM, Message

if TYPE_CHECKING:
    from agent_framework.planning.executor import StepResult

# --------------------------------------------------------------------------- #
# 数据结构                                                                       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PlanStep:
    """计划中的一步。

    Attributes:
        id: 1 起步的序号;重规划产生的新步骤接在原计划最大序号之后,
            保证 trace 里的步骤编号全局不重复。
        description: 这一步要做什么(给专员的任务描述,应具体可执行)。
        specialist: 执行专员的机器名(如 ``"order_agent"``)。
    """

    id: int
    description: str
    specialist: str


@dataclass(frozen=True)
class Plan:
    """一份线性步骤计划。

    Attributes:
        goal: 原始诉求的复述(重规划时锚定目标,防止越改越偏)。
        steps: 按执行顺序排列的步骤。
    """

    goal: str
    steps: tuple[PlanStep, ...]


# --------------------------------------------------------------------------- #
# Prompt(结构与要点见 stage-5-design.md 附录 A;定稿以此为准)                      #
# --------------------------------------------------------------------------- #

PLAN_SYSTEM = (
    "你是京东客服的方案规划师。把用户诉求拆解为按顺序执行的步骤,每一步指派给一位专员执行。\n"
    "规则:\n"
    "- 只拆必要的步骤,最多 {max_steps} 步;能一步办完就不要拆成多步。\n"
    "- 每步描述要具体可执行:写清要查什么/办什么,带上已知的订单号、商品名等关键信息。\n"
    "- 涉及退款、取消订单等操作的步骤之前,必须先安排查证步骤(核实订单状态与条件)。\n"
    "- 信息不足时,安排一步“向用户确认缺失信息”,不要凭空假设。\n"
    "- specialist 必须从【专员花名册】的机器名中选。\n"
    "只输出 JSON 数组,不要多余文字/解释/代码围栏:\n"
    '[{{"step": "步骤描述", "specialist": "专员机器名"}}, ...]'
)

REPLAN_SYSTEM = (
    "你是京东客服的方案规划师。原计划执行中有一步失败了,请重新规划**剩余未完成部分**。\n"
    "规则:\n"
    "- 已完成的步骤不要重复安排。\n"
    "- 失败的路径走不通就换替代方案(例如退款超时效 → 建工单转人工处理)。\n"
    "- 其余规则同首次规划:步骤具体可执行,最多 {max_steps} 步,"
    "specialist 必须从【专员花名册】的机器名中选。\n"
    "只输出 JSON 数组,不要多余文字/解释/代码围栏:\n"
    '[{{"step": "步骤描述", "specialist": "专员机器名"}}, ...]'
)


# --------------------------------------------------------------------------- #
# Planner                                                                       #
# --------------------------------------------------------------------------- #


class Planner:
    """计划的生成与重规划(各一次 LLM 调用,无循环)。"""

    def __init__(self, llm: LLM, *, max_steps: int = 6) -> None:
        """
        Args:
            llm: 满足 ``LLM`` 协议的实例。
            max_steps: 计划长度上限(prompt 约束 + 超长硬截断,防把简单事拆成八步)。
        """
        self._llm = llm
        self.max_steps = max_steps

    def plan(
        self,
        goal: str,
        *,
        roster: str,
        specialists: Sequence[str],
        context: str = "",
    ) -> Plan:
        """把诉求拆解为步骤计划。解析失败降级为单步计划,永不抛。

        Args:
            goal: 用户诉求原文。
            roster: 专员花名册的渲染文本(进 prompt,供模型选人)。
            specialists: 合法专员机器名集合;越界指派矫正为 ``specialists[0]``(兜底)。
            context: 已知背景(如记忆附加段),可为空。

        Returns:
            步骤 1 起编号的 :class:`Plan`;降级时为单步计划(整包给兜底专员)。

        Raises:
            ValueError: ``specialists`` 为空(没有兜底对象,属装配错误,启动即失败)。
        """
        fallback = self._require_fallback(specialists)
        prompt = (
            f"【专员花名册】\n{roster}\n\n"
            f"【已知背景】\n{context or '(无)'}\n\n"
            f"【用户诉求】\n{goal}"
        )
        steps = self._ask_for_steps(PLAN_SYSTEM, prompt, set(specialists), fallback, start_id=1)
        if not steps:
            steps = [PlanStep(id=1, description=goal, specialist=fallback)]
        return Plan(goal=goal, steps=tuple(steps))

    def replan(
        self,
        plan: Plan,
        *,
        completed: Sequence[StepResult],
        failure: StepResult,
        roster: str,
        specialists: Sequence[str],
    ) -> Plan:
        """某步失败后,重排剩余未完成部分。解析失败降级为单步兜底计划,永不抛。

        Args:
            plan: 失败发生时的现行计划(取 ``goal`` 与最大步骤号)。
            completed: 已成功完成的步骤结果(喂给模型,防重复安排;不会被重跑)。
            failure: 失败的那一步及原因。
            roster: 专员花名册渲染文本。
            specialists: 合法专员机器名集合(兜底同 :meth:`plan`)。

        Returns:
            只含剩余工作的新 :class:`Plan`;步骤编号接在原计划最大序号之后。
        """
        fallback = self._require_fallback(specialists)
        start_id = max((s.id for s in plan.steps), default=0) + 1
        done_lines = (
            "\n".join(
                f"- step-{r.step.id}({r.step.specialist}){r.step.description}"
                f" → {_clip(r.output)}"
                for r in completed
            )
            or "(无)"
        )
        prompt = (
            f"【专员花名册】\n{roster}\n\n"
            f"【原目标】\n{plan.goal}\n\n"
            f"【已完成步骤及结果】\n{done_lines}\n\n"
            f"【失败步骤及原因】\n"
            f"step-{failure.step.id}({failure.step.specialist}){failure.step.description}"
            f" → 失败:{_clip(failure.output)}"
        )
        steps = self._ask_for_steps(
            REPLAN_SYSTEM, prompt, set(specialists), fallback, start_id=start_id
        )
        if not steps:
            steps = [
                PlanStep(
                    id=start_id,
                    description=(
                        f"继续完成用户诉求(注意:前序步骤失败,原因:{_clip(failure.output)}):"
                        f"{plan.goal}"
                    ),
                    specialist=fallback,
                )
            ]
        return Plan(goal=plan.goal, steps=tuple(steps))

    # ------------------------------------------------------------------ #
    # 内部                                                                  #
    # ------------------------------------------------------------------ #

    def _ask_for_steps(
        self,
        system_template: str,
        prompt: str,
        valid: set[str],
        fallback: str,
        *,
        start_id: int,
    ) -> list[PlanStep]:
        """一次 LLM 调用 → 解析/校验/矫正/截断。任何失败返回空列表(由调用方兜底)。"""
        system = system_template.format(max_steps=self.max_steps)
        try:
            response = self._llm.chat([Message("user", prompt)], system=system)
            items = _parse_json_array(response.content)
        except Exception:  # noqa: BLE001 — 规划失败不炸,降级单步计划
            return []
        if not items:
            return []
        steps: list[PlanStep] = []
        for item in items[: self.max_steps]:
            if not isinstance(item, dict):
                continue
            description = str(item.get("step", "")).strip()
            if not description:
                continue
            specialist = str(item.get("specialist", "")).strip()
            if specialist not in valid:
                specialist = fallback  # 编造/越界的专员名 → 矫正为兜底(防派单落空)
            steps.append(
                PlanStep(id=start_id + len(steps), description=description, specialist=specialist)
            )
        return steps

    @staticmethod
    def _require_fallback(specialists: Sequence[str]) -> str:
        if not specialists:
            raise ValueError("specialists 不能为空:Planner 需要至少一个兜底专员")
        return specialists[0]


def _clip(text: str, limit: int = 200) -> str:
    """结果摘要截断(进重规划 prompt 用,防单步长输出撑爆上下文)。"""
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _parse_json_array(text: str) -> list[object] | None:
    """从模型输出解析 JSON 数组;容忍 ``` 围栏;不是数组返回 None。

    与 ``memory/long_term.py`` 的同名函数刻意保持两份小副本:planning 不依赖
    memory 子包(依赖方向纪律),15 行解析器不值得为共享而引入耦合。
    """
    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None
