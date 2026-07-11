"""AgentService —— 把六个阶段装配成一个可调用的服务门面(阶段六)。

这是**唯一的整栈装配点**:安全出入口 + 记忆 + 分诊 + 三专员(工具子集包边界标记 +
审批闸门)+ 中心调度 + 观测 + 人工队列。CLI、评测 pipeline、FastAPI 都调它,
不各自重复装配(与阶段五「定义一次两种编排复用」同精神)。

一次 :meth:`handle` 的完整链路(stage-6-design.md §3):
入口安全(限流→输入清洗/注入检测)→ 记忆 load → 分诊(direct/专员/supervisor)
→ 执行(专员工具经边界标记+审批闸门)→ 出口安全(脱敏)→ 记忆落账 → 全程 trace。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field

from agent_framework.core.config import Settings
from agent_framework.core.llm import LLM, Message
from agent_framework.memory.manager import MemoryContext, MemoryManager
from agent_framework.memory.short_term import Turn
from agent_framework.multi_agent.critic import Critic
from agent_framework.multi_agent.protocol import FAILURE_MARKER, Specialist, TaskAssignment
from agent_framework.multi_agent.router import DIRECT_TARGET, SUPERVISOR_TARGET, Router
from agent_framework.multi_agent.specialists import create_specialists
from agent_framework.multi_agent.supervisor import Supervisor
from agent_framework.observability.tracer import Tracer
from agent_framework.planning.planner import Planner
from agent_framework.safety.approval import ApprovalGate, ApprovalPolicy, HandoffItem, HandoffQueue
from agent_framework.safety.input_filter import BoundaryRegistry, inspect_input
from agent_framework.safety.output_filter import filter_output
from agent_framework.safety.rate_limiter import RateLimiter, TokenBudget
from agent_framework.tools.registry import ToolRegistry

_DIRECT_SYSTEM = (
    "你是京东客服。当前输入不需要查询或操作任何数据,直接给出简短、友好、专业的中文回复:"
    "用户告知的个人信息(姓名/地址等)礼貌确认收到即可,不要提出核实、修改或建工单;"
    "寒暄和感谢自然回应;问你能做什么就介绍订单物流查询、售后处理、商品导购三类服务。"
)


@dataclass
class ServiceResult:
    """一次服务处理的完整结果(API 响应 / 评测 / CLI 共用)。"""

    answer: str
    task_id: str
    route: str  # direct / 专员名 / supervisor
    handoffs: list[HandoffItem] = field(default_factory=list)
    redactions: list[str] = field(default_factory=list)
    suspicious_input: bool = False
    rate_limited: bool = False
    trace_events: int = 0


class AgentService:
    """整栈门面:装配一次,反复 :meth:`handle`。"""

    def __init__(
        self,
        llm: LLM,
        base_registry: ToolRegistry,
        settings: Settings,
        *,
        memory_factory: Callable[[str], MemoryManager] | None = None,
        handoff_queue: HandoffQueue | None = None,
        enable_trace: bool = True,
    ) -> None:
        """
        Args:
            llm: 已带可靠性包装的 LLM(create_llm 产出)。
            base_registry: 全量工具库(default_registry)。
            settings: 全局配置(护栏参数、审批规则、限流等)。
            memory_factory: 造 MemoryManager 的工厂(缺省 = 无长期记忆的空壳);
                多用户各自一个 manager,由服务缓存。
            handoff_queue: 人工介入队列;缺省按 settings 落盘路径新建。
            enable_trace: 是否 JSONL 落盘轨迹(评测时可关,只用内存)。
        """
        self._llm = llm
        self._settings = settings
        self._queue = handoff_queue or HandoffQueue(settings.handoff_store_path)
        self._policy = ApprovalPolicy.from_settings(settings)
        self._limiter = RateLimiter(max_per_minute=settings.rate_limit_per_minute)
        self._enable_trace = enable_trace
        self._trace_dir = settings.trace_dir if enable_trace else None
        self._memory_factory = memory_factory or (lambda _uid: MemoryManager())
        self._managers: dict[str, MemoryManager] = {}

        # 专员集合:每个专员工具子集包 边界标记(内) + 审批闸门(外)
        self._plain_specialists = create_specialists(base_registry)
        self._base_registry = base_registry
        self._current_user = "guest"  # 闸门取当前用户(单请求内串行,线程池每请求一个 service 或加锁)
        self._current_context = ""
        self._pending_this_task: list[HandoffItem] = []  # 本次请求内闸门挂起的审批单
        self._gated_specialists = {
            name: self._wrap_specialist(spec) for name, spec in self._plain_specialists.items()
        }
        self._router = Router(llm, self._plain_specialists)  # 分诊只看职责,用不包装的
        self._planner = Planner(llm, max_steps=settings.planner_max_steps)
        self._critic = Critic(llm)

    def _wrap_specialist(self, spec: Specialist) -> Specialist:
        gated = ApprovalGate(
            BoundaryRegistry(spec.registry),
            self._queue,
            self._policy,
            user_id_provider=lambda: self._current_user,
            context_provider=lambda: self._current_context,
            on_pending=lambda item: self._pending_this_task.append(
                item
            ),  # 延迟读属性,配合每请求清空
        )
        return dataclasses.replace(spec, registry=gated)  # type: ignore[arg-type]

    def _manager(self, user_id: str) -> MemoryManager:
        if user_id not in self._managers:
            self._managers[user_id] = self._memory_factory(user_id)
        return self._managers[user_id]

    def handle(
        self,
        user_id: str,
        message: str,
        *,
        mode: str = "auto",
        on_event: Callable[[str, dict[str, object]], None] | None = None,
    ) -> ServiceResult:
        """处理一条用户消息,走完整生产链路。永不抛(失败折叠进答复)。

        Args:
            user_id: 用户标识(记忆隔离 + 限流 + 审批单归属;框架注入,不进 prompt)。
            message: 用户原始输入。
            mode: auto(分诊)/ supervisor(强制中心调度)/ router(观察分诊)。
            on_event: 额外事件订阅者(SSE 推送用);与内部 tracer 叠加。
        """
        tracer = Tracer(trace_dir=self._trace_dir)
        if on_event is not None:
            tracer.add_listener(lambda e: on_event(e.kind, e.payload))
        tracer.emit("task_start", user_id=user_id, mode=mode)
        self._current_user = user_id
        self._pending_this_task.clear()  # 每请求原地清空(闸门 on_pending 往里登记)

        # ① 入口安全:限流 → 输入清洗/注入检测
        if not self._limiter.allow(user_id):
            tracer.emit("task_end", ok=False, reason="rate_limited")
            return ServiceResult(
                answer="您的操作过于频繁,请稍后再试。",
                task_id=tracer.task_id,
                route="blocked",
                rate_limited=True,
                trace_events=len(tracer.events),
            )
        check = inspect_input(message, max_chars=self._settings.max_input_chars)
        if check.suspicious:
            tracer.emit("input_flagged", reasons=check.reasons)
        clean = check.text
        warning = check.system_warning()

        # ② 记忆 load
        manager = self._manager(user_id)
        ctx = manager.load(user_id, clean)
        mem_suffix = ctx.system_suffix() + warning
        self._current_context = clean

        # ③ 分诊 → 执行
        try:
            answer, route, handoffs = self._route_and_run(
                clean, mode, ctx, mem_suffix, user_id, tracer, on_event
            )
        except Exception as exc:  # noqa: BLE001 — 兜底:任何未预期异常都转人工,不给用户报错栈
            item = self._queue.submit_escalation(
                user_id=user_id, reason=f"系统异常:{type(exc).__name__}", context=clean
            )
            tracer.emit("escalation", reason="unhandled_exception")
            answer, route, handoffs = (
                f"抱歉,系统处理时遇到问题,已为您转人工跟进(工单号 {item.id})。",
                "error",
                [item],
            )

        # ④ 出口安全:脱敏
        filtered = filter_output(answer)
        if filtered.redactions:
            tracer.emit("output_redacted", kinds=filtered.redactions)

        # ⑤ 记忆落账
        manager.on_turn_end(user_id, Turn(user_text=clean, assistant_text=filtered.text))
        tracer.emit("task_end", ok=True, route=route)

        return ServiceResult(
            answer=filtered.text,
            task_id=tracer.task_id,
            route=route,
            handoffs=handoffs,
            redactions=filtered.redactions,
            suspicious_input=check.suspicious,
            trace_events=len(tracer.events),
        )

    def _route_and_run(
        self,
        query: str,
        mode: str,
        ctx: MemoryContext,
        mem_suffix: str,
        user_id: str,
        tracer: Tracer,
        on_event: Callable[[str, dict[str, object]], None] | None,
    ) -> tuple[str, str, list[HandoffItem]]:
        # 分诊
        if mode == "supervisor":
            target = SUPERVISOR_TARGET
        else:
            decision = self._router.route(query, context=mem_suffix)
            target = decision.target
            tracer.emit("route", target=target, reason=decision.reason)

        # direct:轻量无工具直接回复
        if target == DIRECT_TARGET:
            resp = self._llm.chat(
                ctx.to_messages() + [Message("user", query)],
                system=_DIRECT_SYSTEM + mem_suffix,
            )
            tracer.emit("final_answer", route="direct")
            return resp.content, "direct", []

        # supervisor:中心调度(带审批闸门的专员 + 人工队列 + 超时/预算保险）
        if target == SUPERVISOR_TARGET:
            recent = "\n".join(t.render_text() for t in ctx.recent_turns)
            background = mem_suffix + (f"\n\n【近期对话】\n{recent}" if recent else "")
            supervisor = Supervisor(
                self._llm,
                self._gated_specialists,
                planner=self._planner,
                critic=self._critic,
                max_steps_per_specialist=self._settings.supervisor_specialist_max_steps,
                max_replans=self._settings.max_replans,
                critic_max_retries=self._settings.critic_max_retries,
                handoff=self._queue,
                task_deadline_seconds=self._settings.task_deadline_seconds,
            )
            budget = TokenBudget(self._settings.task_token_budget)
            result = supervisor.handle(
                query,
                memory_context=background,
                user_id=user_id,
                tracer=tracer,
                token_budget=budget,
            )
            return result.final_answer, "supervisor", result.escalations

        # 专员直派(快路径,带审批闸门 + 外层历史)
        specialist = self._gated_specialists[target]
        outcome = specialist.handle(
            self._llm,
            TaskAssignment(task=query),
            extra_system=mem_suffix,
            max_steps=self._settings.agent_max_steps,
            history=ctx.to_messages(),
            on_event=tracer.as_on_event(),
        )
        answer = outcome.answer
        if answer.strip().startswith(FAILURE_MARKER):  # 快路径剥离协议前缀
            answer = answer.strip()[len(FAILURE_MARKER) :].lstrip(":: \n") or outcome.answer
        tracer.emit("final_answer", route=target)
        return answer, target, list(self._pending_this_task)  # 闸门本次挂起的审批单

    @property
    def queue(self) -> HandoffQueue:
        """人工介入队列(API /approvals 与控制台共用)。"""
        return self._queue

    @property
    def base_registry(self) -> ToolRegistry:
        """全量工具库(审批放行时用它执行挂起动作,与业务共享同一数据源)。"""
        return self._base_registry
