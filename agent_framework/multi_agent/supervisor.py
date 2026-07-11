"""Supervisor 模式(模式二):中心调度 —— 按计划派工、失败重规划、汇总、质检。

与 LangGraph Supervisor 的关键差异(答辩点,stage-5-design.md §6.4):
不让 LLM 每步自由选人,而是**按 Planner 的计划派工**,LLM 自由度收敛在
plan/replan 两个口子——行为可预测、轨迹可断言可测试。

组合自由度(与 MemoryManager 三件套同哲学):``planner=None`` 退化为
整包直派兜底专员;``critic=None`` 跳过质检;都不影响主链路。
全链路降级:汇总调用失败 → 无 LLM 的步骤结果拼接,永不炸。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from agent_framework.core.llm import LLM, Message
from agent_framework.multi_agent.critic import Critic, Critique
from agent_framework.multi_agent.protocol import (
    Specialist,
    TaskAssignment,
    render_roster,
)
from agent_framework.observability.tracer import Tracer
from agent_framework.planning.executor import (
    ExecutionResult,
    PlanExecutor,
    ScratchPad,
    StepResult,
)
from agent_framework.planning.planner import Plan, Planner, PlanStep
from agent_framework.safety.approval import HandoffItem, HandoffQueue
from agent_framework.safety.rate_limiter import TokenBudget

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
        escalations: 本次任务产生的人工介入单(阶段六:重规划耗尽仍失败/
            质检二审不合格/整任务超时或超预算,未配队列时恒为空)。
        interrupted: 是否被第④层保险(超时/超预算)提前刹停。
    """

    final_answer: str
    plan: Plan
    step_results: list[StepResult] = field(default_factory=list)
    replanned: bool = False
    critiques: list[Critique] = field(default_factory=list)
    resynthesized: bool = False
    escalations: list[HandoffItem] = field(default_factory=list)
    interrupted: bool = False


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
        on_event: Callable[[str, dict[str, object]], None] | None = None,
        emit: Callable[..., object] | None = None,
    ) -> None:
        self._llm = llm
        self._specialists = dict(specialists)
        self._fallback = next(iter(self._specialists.values()))
        self._extra_system = extra_system
        self._max_steps = max_steps
        self._on_event = on_event  # 透传给专员内环(llm_call/tool_call 级事件)
        self._emit = emit  # 步骤级事件(step_start/step_end)

    def run_step(self, step: PlanStep, context: str) -> StepResult:
        # Planner 已矫正越界指派;这里再兜一道底,防手写计划/未来新入口漏校验
        specialist = self._specialists.get(step.specialist, self._fallback)
        if self._emit is not None:
            self._emit(
                "step_start", step=step.id, specialist=specialist.name, task=step.description
            )
        outcome = specialist.handle(
            self._llm,
            TaskAssignment(task=step.description, context=context),
            extra_system=self._extra_system,
            max_steps=self._max_steps,
            on_event=self._on_event,
        )
        if self._emit is not None:
            self._emit("step_end", step=step.id, specialist=specialist.name, ok=outcome.ok)
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
        handoff: HandoffQueue | None = None,
        task_deadline_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
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
            handoff: 人工介入队列(阶段六);None 则兜底升级只留在结果里不入队。
            task_deadline_seconds: 整任务总超时(超时四层保险之④);None 不限。
                超时/超预算不是失败,是**转人工的信号**——任务终止入队。
            clock: 单调时钟注入(测试用)。

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
        self._handoff = handoff
        self._deadline_seconds = task_deadline_seconds
        self._clock = clock

    def handle(
        self,
        query: str,
        *,
        memory_context: str = "",
        user_id: str = "guest",
        tracer: Tracer | None = None,
        token_budget: TokenBudget | None = None,
    ) -> SupervisorResult:
        """处理一条(复杂)诉求,走完整调度链路。

        Args:
            query: 用户诉求原文。
            memory_context: 阶段四记忆的 ``ctx.system_suffix()``——既作规划的
                已知背景,也注入每个专员的 system(全员共享同一份用户事实)。
            user_id: 当前用户(人工介入单归属;框架注入,不进 prompt)。
            tracer: 任务轨迹记录器(阶段六);None 不记录。
            token_budget: 单任务 token 预算;需配合 ``tracer``(用量从
                llm_call 事件累计),超限在下一步边界刹车转人工。

        Returns:
            带全轨迹的 :class:`SupervisorResult`。
        """

        def emit(kind: str, **payload: object) -> None:
            if tracer is not None:
                tracer.emit(kind, **payload)

        if tracer is not None and token_budget is not None:
            tracer.add_listener(token_budget.on_trace_event)
        on_event = tracer.as_on_event() if tracer is not None else None

        # ① 计划 → ② 逐步派工(含重规划、超时/超预算刹车)
        plan = self._build_plan(query, memory_context, emit)
        execution = self._execute_plan(plan, memory_context, on_event, emit, token_budget)
        final_plan = execution.plan or plan  # executor 总会回填,None 时兜底原计划
        if execution.replanned:
            emit("replan", plan=[f"step-{s.id} {s.description}" for s in final_plan.steps])

        escalations: list[HandoffItem] = []
        evidence = _render_evidence(execution.results)

        # 第④层保险:超时/超预算 → 早退转人工,不再烧钱汇总
        if execution.interrupted:
            return self._interrupted_result(
                execution, final_plan, evidence or query, user_id, escalations, emit, token_budget
            )

        # 兜底升级入口 B-①:重规划耗尽仍有失败 → 转人工,证据里如实标注
        evidence = self._escalate_unresolved(execution, evidence, user_id, escalations, emit)

        # ③ 汇总 → ④ 质检回炉(不合格 → B-② 人工复核)
        answer, critiques, resynthesized = self._synthesize_and_review(
            query, evidence, user_id, escalations, emit
        )

        return SupervisorResult(
            final_answer=answer,
            plan=final_plan,
            step_results=execution.results,
            replanned=execution.replanned,
            critiques=critiques,
            resynthesized=resynthesized,
            escalations=escalations,
        )

    # ------------------------------------------------------------------ #
    # handle 的分阶段私有方法(阶段六代码审查:把 148 行的 handle 拆开)          #
    # ------------------------------------------------------------------ #

    def _build_plan(self, query: str, memory_context: str, emit: Callable[..., None]) -> Plan:
        """① 生成计划;planner 缺省则整包直派兜底专员(退化为单 Agent 行为)。"""
        if self._planner is not None:
            plan = self._planner.plan(
                query, roster=self._roster, specialists=self._names, context=memory_context
            )
        else:
            plan = Plan(
                goal=query, steps=(PlanStep(id=1, description=query, specialist=self._names[0]),)
            )
        emit("plan", steps=[f"step-{s.id}({s.specialist}){s.description}" for s in plan.steps])
        return plan

    def _execute_plan(
        self,
        plan: Plan,
        memory_context: str,
        on_event: Callable[[str, dict[str, object]], None] | None,
        emit: Callable[..., None],
        token_budget: TokenBudget | None,
    ) -> ExecutionResult:
        """② 逐步派工;失败触发重规划,第④层保险(超时/超预算)在步骤边界刹车。"""
        deadline = (
            self._clock() + self._deadline_seconds if self._deadline_seconds is not None else None
        )

        def stop_when() -> bool:
            if deadline is not None and self._clock() > deadline:
                return True
            return token_budget is not None and token_budget.exceeded

        runner = _SpecialistRunner(
            self._llm,
            self._specialists,
            extra_system=memory_context,
            max_steps=self._max_steps_per_specialist,
            on_event=on_event,
            emit=emit,
        )
        executor = PlanExecutor(
            runner,
            replanner=self._planner,
            roster=self._roster,
            specialists=self._names,
            max_replans=self._max_replans,
        )
        return executor.execute(plan, notes=ScratchPad(), stop_when=stop_when)

    def _interrupted_result(
        self,
        execution: ExecutionResult,
        final_plan: Plan,
        context: str,
        user_id: str,
        escalations: list[HandoffItem],
        emit: Callable[..., None],
        token_budget: TokenBudget | None,
    ) -> SupervisorResult:
        """第④层保险触发的早退结果:转人工话术 + 升级单,不做汇总。"""
        reason = (
            "单任务 token 预算耗尽"
            if token_budget is not None and token_budget.exceeded
            else f"整任务超时(上限 {self._deadline_seconds} 秒)"
        )
        item = self._escalate(user_id, reason, context, escalations, emit)
        suffix = f"(工单号 {item.id})" if item is not None else ""
        answer = (
            "非常抱歉,您的问题处理耗时超出预期,已为您转接人工客服跟进" f"{suffix},会尽快与您联系。"
        )
        return SupervisorResult(
            final_answer=answer,
            plan=final_plan,
            step_results=execution.results,
            replanned=execution.replanned,
            escalations=escalations,
            interrupted=True,
        )

    def _escalate_unresolved(
        self,
        execution: ExecutionResult,
        evidence: str,
        user_id: str,
        escalations: list[HandoffItem],
        emit: Callable[..., None],
    ) -> str:
        """入口 B-①:重规划耗尽仍有失败 → 转人工,并在证据里标注供汇总如实告知。"""
        if not execution.unresolved_failures:
            return evidence
        failed = ";".join(
            f"step-{r.step.id} {r.step.description}" for r in execution.unresolved_failures
        )
        item = self._escalate(user_id, f"自动处理未完成:{failed}", evidence, escalations, emit)
        if item is not None:
            evidence += (
                f"\n(系统提示:上述失败部分已自动转人工跟进,工单号 {item.id},"
                "24 小时内联系用户——请在答复中如实告知)"
            )
        return evidence

    def _synthesize_and_review(
        self,
        query: str,
        evidence: str,
        user_id: str,
        escalations: list[HandoffItem],
        emit: Callable[..., None],
    ) -> tuple[str, list[Critique], bool]:
        """③ 汇总 → ④ 质检回炉(≤ critic_max_retries 次);二审不过 → 入口 B-② 人工复核。"""
        answer = self._synthesize(query, evidence)
        emit("synthesize", chars=len(answer))
        critiques: list[Critique] = []
        resynthesized = False
        if self._critic is None:
            return answer, critiques, resynthesized

        critique = self._critic.review(query, answer, evidence)
        critiques.append(critique)
        emit("critic", round=1, passed=critique.passed, issues=critique.issues)
        retries = 0
        while not critique.passed and retries < self._critic_max_retries:
            retries += 1
            resynthesized = True
            answer = self._synthesize(query, evidence, issues=critique.issues)
            critique = self._critic.review(query, answer, evidence)
            critiques.append(critique)
            emit("critic", round=retries + 1, passed=critique.passed, issues=critique.issues)
        if not critique.passed:  # 二审仍不合格:放行留痕 + B-② 人工复核
            self._escalate(
                user_id,
                "质检二审仍不合格,答复已放行,需人工复核",
                f"【答复】{answer}\n【质检意见】{'; '.join(critique.issues)}",
                escalations,
                emit,
            )
        return answer, critiques, resynthesized

    # ------------------------------------------------------------------ #
    # 内部                                                                  #
    # ------------------------------------------------------------------ #

    def _escalate(
        self,
        user_id: str,
        reason: str,
        context: str,
        escalations: list[HandoffItem],
        emit: Callable[..., None],
    ) -> HandoffItem | None:
        """提交一张兜底升级单(未配队列返回 None,只发轨迹事件)。"""
        emit("escalation", reason=reason)
        if self._handoff is None:
            return None
        item = self._handoff.submit_escalation(user_id=user_id, reason=reason, context=context)
        escalations.append(item)
        return item

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
