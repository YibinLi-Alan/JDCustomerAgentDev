"""Supervisor 模式(模式二):中心调度 —— 按计划派工、失败重规划、汇总、质检。

与 LangGraph Supervisor 的关键差异(答辩点,stage-5-design.md §6.4):
不让 LLM 每步自由选人,而是**按 Planner 的计划派工**,LLM 自由度收敛在
plan/replan 两个口子——行为可预测、轨迹可断言可测试。

组合自由度(与 MemoryManager 三件套同哲学):``planner=None`` 退化为
整包直派兜底专员;``critic=None`` 跳过质检;都不影响主链路。
全链路降级:汇总调用失败 → 无 LLM 的步骤结果拼接,永不炸。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from agent_framework.core.llm import LLM, Message
from agent_framework.multi_agent.critic import Critic, Critique
from agent_framework.multi_agent.protocol import (
    Specialist,
    TaskAssignment,
    render_roster,
)
from agent_framework.planning.executor import PlanExecutor, ScratchPad, StepResult
from agent_framework.planning.planner import Plan, Planner, PlanStep

SYNTHESIZE_SYSTEM = (
    "你是京东客服主管。根据执行记录,给用户写最终答复:\n"
    "- 按用户诉求逐项交代结果,一项都不要漏;\n"
    "- 办成的说清凭据(订单号/退款单号/工单号);\n"
    "- 没办成的如实说明原因,并给出替代方案(如已转人工、建议联系渠道);\n"
    "- 不要提及内部分工、步骤编号或系统细节;语气专业友好。"
)


@dataclass(frozen=True)
class SupervisorResult:
    """一次中心调度的完整结果(CLI /trace 与测试断言的数据源)。

    Attributes:
        final_answer: 给用户的最终答复。
        plan: 最终生效的计划。
        step_results: 全部步骤结果(含失败步与重规划后的新步)。
        replanned: 是否发生过动态重规划。
        critiques: 各轮质检结论(0~2 条:未配 Critic 为空;回炉过则有两条)。
        resynthesized: 是否因质检不合格回炉重写过答复。
    """

    final_answer: str
    plan: Plan
    step_results: list[StepResult] = field(default_factory=list)
    replanned: bool = False
    critiques: list[Critique] = field(default_factory=list)
    resynthesized: bool = False


class _SpecialistRunner:
    """把「专员执行任务」适配成 planning 的 ``StepRunner`` 协议。

    planning 子包不认识 Specialist(依赖方向纪律),这个适配器就是两个子包
    之间的桥:计划步骤 → 任务派发 → 专员回报 → 步骤结果。
    """

    def __init__(
        self,
        llm: LLM,
        specialists: Mapping[str, Specialist],
        *,
        extra_system: str,
        max_steps: int,
    ) -> None:
        self._llm = llm
        self._specialists = dict(specialists)
        self._fallback = next(iter(self._specialists.values()))
        self._extra_system = extra_system
        self._max_steps = max_steps

    def run_step(self, step: PlanStep, context: str) -> StepResult:
        # Planner 已矫正越界指派;这里再兜一道底,防手写计划/未来新入口漏校验
        specialist = self._specialists.get(step.specialist, self._fallback)
        outcome = specialist.handle(
            self._llm,
            TaskAssignment(task=step.description, context=context),
            extra_system=self._extra_system,
            max_steps=self._max_steps,
        )
        return StepResult(step=step, output=outcome.answer, ok=outcome.ok, trace=outcome.trace)


class Supervisor:
    """中心调度器:plan → 逐步派工(失败重规划)→ 汇总 → 质检(不合格回炉)。"""

    def __init__(
        self,
        llm: LLM,
        specialists: Mapping[str, Specialist],
        *,
        planner: Planner | None = None,
        critic: Critic | None = None,
        max_steps_per_specialist: int = 5,
        max_replans: int = 1,
        critic_max_retries: int = 1,
    ) -> None:
        """
        Args:
            llm: LLM 实例(规划/派工/汇总/质检共享)。
            specialists: 专员集合;**第一个是兜底专员**(降级单步计划的接收人)。
            planner: 计划器;None 则整包直派兜底专员(退化为单 Agent 行为)。
            critic: 质检员;None 则跳过质检。
            max_steps_per_specialist: 专员内环步数上限(循环护栏)。
            max_replans: 整轮重规划次数上限(循环护栏)。
            critic_max_retries: 质检不合格的回炉次数上限(循环护栏)。

        Raises:
            ValueError: ``specialists`` 为空(装配错误,启动即失败)。
        """
        if not specialists:
            raise ValueError("specialists 不能为空:Supervisor 需要至少一个专员")
        self._llm = llm
        self._specialists = dict(specialists)
        self._roster = render_roster(list(self._specialists.values()))
        self._names = tuple(self._specialists)
        self._planner = planner
        self._critic = critic
        self._max_steps_per_specialist = max_steps_per_specialist
        self._max_replans = max_replans
        self._critic_max_retries = critic_max_retries

    def handle(self, query: str, *, memory_context: str = "") -> SupervisorResult:
        """处理一条(复杂)诉求,走完整调度链路。

        Args:
            query: 用户诉求原文。
            memory_context: 阶段四记忆的 ``ctx.system_suffix()``——既作规划的
                已知背景,也注入每个专员的 system(全员共享同一份用户事实)。

        Returns:
            带全轨迹的 :class:`SupervisorResult`。
        """
        # ① 计划(planner 缺省 → 整包直派兜底专员)
        if self._planner is not None:
            plan = self._planner.plan(
                query, roster=self._roster, specialists=self._names, context=memory_context
            )
        else:
            plan = Plan(
                goal=query, steps=(PlanStep(id=1, description=query, specialist=self._names[0]),)
            )

        # ② 逐步派工(失败触发重规划,护栏见构造参数)
        runner = _SpecialistRunner(
            self._llm,
            self._specialists,
            extra_system=memory_context,
            max_steps=self._max_steps_per_specialist,
        )
        executor = PlanExecutor(
            runner,
            replanner=self._planner,
            roster=self._roster,
            specialists=self._names,
            max_replans=self._max_replans,
        )
        execution = executor.execute(plan, notes=ScratchPad())

        # ③ 汇总 → ④ 质检回炉(≤ critic_max_retries 次)
        evidence = _render_evidence(execution.results)
        answer = self._synthesize(query, evidence)
        critiques: list[Critique] = []
        resynthesized = False
        if self._critic is not None:
            critique = self._critic.review(query, answer, evidence)
            critiques.append(critique)
            retries = 0
            while not critique.passed and retries < self._critic_max_retries:
                retries += 1
                resynthesized = True
                answer = self._synthesize(query, evidence, issues=critique.issues)
                critique = self._critic.review(query, answer, evidence)
                critiques.append(critique)
            # 二审仍不合格:放行但留痕(critiques 里可见),阶段六 metrics 的素材

        return SupervisorResult(
            final_answer=answer,
            plan=execution.plan or plan,
            step_results=execution.results,
            replanned=execution.replanned,
            critiques=critiques,
            resynthesized=resynthesized,
        )

    # ------------------------------------------------------------------ #
    # 内部                                                                  #
    # ------------------------------------------------------------------ #

    def _synthesize(self, query: str, evidence: str, *, issues: list[str] | None = None) -> str:
        """汇总调用(回炉时带质检意见重写)。失败降级为无 LLM 的结果拼接,永不炸。"""
        prompt = f"【用户诉求】\n{query}\n\n【执行记录】\n{evidence or '(无)'}"
        if issues:
            joined = "\n".join(f"- {i}" for i in issues)
            prompt += f"\n\n【质检意见(上一稿的问题,必须改正)】\n{joined}"
        try:
            return self._llm.chat([Message("user", prompt)], system=SYNTHESIZE_SYSTEM).content
        except Exception:  # noqa: BLE001 — 汇总失败降级拼接,答复不能开天窗
            return _fallback_answer(query, evidence)


def _render_evidence(results: list[StepResult]) -> str:
    """步骤结果 → 执行记录文本(汇总 prompt 与 Critic 证据共用)。"""
    lines = []
    for r in results:
        status = "成功" if r.ok else "失败"
        lines.append(
            f"step-{r.step.id}({r.step.specialist}){r.step.description} → {status}:{r.output}"
        )
    return "\n".join(lines)


def _fallback_answer(query: str, evidence: str) -> str:
    """汇总调用失败时的降级答复:直接罗列各步结果,信息不丢、格式从简。"""
    return (
        "已为您处理如下(系统繁忙,答复由执行记录直接生成):\n"
        f"{evidence or '(本次未能执行任何步骤,请稍后重试或联系人工客服)'}"
    )
