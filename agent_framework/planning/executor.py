"""PlanExecutor —— 顺序执行计划、经 ScratchPad 传递中间结果、失败触发重规划。

设计要点(stage-5-design.md §5.3/§8):

- 执行器不认识具体专员:依赖 :class:`StepRunner` 协议(multi_agent 的 Supervisor
  提供适配),保持 planning → multi_agent **零依赖**(依赖方向纪律);
- 步骤间共享走 **ScratchPad 黑板**(显式传递优于隐式共享):每步结论以
  ``[step-N 专员] 结论`` 追加,渲染进下一步的 context;
- 失败(专员明示办不到 / 抛异常)→ 触发 ``replan()``,剩余步骤整体替换,
  **整轮全局最多 max_replans 次**;预算耗尽后失败如实进结果,不兜圈。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from agent_framework.planning.planner import Plan, Planner, PlanStep

if TYPE_CHECKING:
    from agent_framework.core.agent import AgentResult

# --------------------------------------------------------------------------- #
# 共享黑板                                                                       #
# --------------------------------------------------------------------------- #


class ScratchPad:
    """步骤间的共享黑板:append 追加结论,render 渲染成 context 文本。

    带字符预算截断保护(超预算时**丢最旧**的条目并留截断标记),防长计划把
    专员上下文撑爆。用字符数而非 token:黑板条目是我们自己写入的结论摘要,
    粗粒度保护足够,不值得为此耦合 memory 子包的 TokenCounter。
    """

    def __init__(self, *, max_chars: int = 6000) -> None:
        self._entries: list[tuple[str, str]] = []
        self._max_chars = max_chars

    @property
    def entries(self) -> list[tuple[str, str]]:
        """(label, content) 列表快照(测试/trace 用)。"""
        return list(self._entries)

    def append(self, label: str, content: str) -> None:
        """追加一条结论。label 形如 ``step-1 order_agent``。"""
        self._entries.append((label, content.strip()))

    def render(self) -> str:
        """渲染为进 prompt 的文本;空黑板返回空串。

        超出字符预算时保最新、丢最旧,并以「(更早的记录已截断)」占位——
        近期结论对下一步最有用,与短期记忆「最新轮永不弹出」同理。
        """
        if not self._entries:
            return ""
        lines: list[str] = []
        used = 0
        truncated = False
        for label, content in reversed(self._entries):
            line = f"[{label}] {content}"
            if lines and used + len(line) > self._max_chars:
                truncated = True
                break
            lines.append(line)
            used += len(line)
        lines.reverse()
        if truncated:
            lines.insert(0, "(更早的记录已截断)")
        return "【前序步骤结论】\n" + "\n".join(lines)


# --------------------------------------------------------------------------- #
# 结果结构与运行协议                                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StepResult:
    """一个计划步骤的执行结果。

    Attributes:
        step: 对应的计划步骤。
        output: 专员的最终答复(失败时为失败说明/异常文本)。
        ok: 是否成功。专员明示办不到、撞步数上限或抛异常均为 False。
        trace: 专员内环的完整轨迹(``AgentResult``),/trace 与调试用;可为 None。
    """

    step: PlanStep
    output: str
    ok: bool
    trace: AgentResult | None = None


class StepRunner(Protocol):
    """执行单个计划步骤的协议(由 multi_agent 的编排层实现)。"""

    def run_step(self, step: PlanStep, context: str) -> StepResult:
        """执行一步并返回结果。context 是 ScratchPad 渲染文本(可为空串)。"""
        ...


@dataclass(frozen=True)
class ExecutionResult:
    """一次完整计划执行的结果。

    Attributes:
        results: 全部步骤结果,按实际执行顺序(含失败步与重规划后的新步)。
        replanned: 是否发生过重规划。
        plan: 最终生效的计划(未重规划时即原计划)。
    """

    results: list[StepResult] = field(default_factory=list)
    replanned: bool = False
    plan: Plan | None = None


# --------------------------------------------------------------------------- #
# 执行器                                                                         #
# --------------------------------------------------------------------------- #


class PlanExecutor:
    """顺序执行计划;失败触发重规划(≤ ``max_replans`` 次)后继续。"""

    def __init__(
        self,
        runner: StepRunner,
        *,
        replanner: Planner | None = None,
        roster: str = "",
        specialists: tuple[str, ...] = (),
        max_replans: int = 1,
    ) -> None:
        """
        Args:
            runner: 步骤执行器(编排层适配专员;测试注入假 runner)。
            replanner: 重规划用的 Planner;None 则失败只记录、不重规划
                (组合自由度:与 MemoryManager 三件套可缺省同哲学)。
            roster: 专员花名册渲染文本(重规划 prompt 用)。
            specialists: 合法专员机器名(重规划的越界矫正/兜底用)。
            max_replans: 整轮执行的重规划次数上限(评审拍板默认 1)。
        """
        self._runner = runner
        self._replanner = replanner
        self._roster = roster
        self._specialists = specialists
        self._max_replans = max_replans

    def execute(self, plan: Plan, *, notes: ScratchPad | None = None) -> ExecutionResult:
        """执行整份计划。

        每步:黑板渲染进 context → runner 执行 → 结论回填黑板(失败也回填,
        带「(失败)」前缀,让后续步骤/重规划看得见)。失败且重规划预算未耗尽时,
        以新计划替换剩余步骤继续;否则带着失败继续跑完剩余步骤
        (各步独立,前一步失败不必然废掉后面的)。

        Args:
            plan: 待执行计划。
            notes: 共享黑板;缺省新建(Supervisor 传入以便最终汇总时复用)。

        Returns:
            :class:`ExecutionResult`(全步骤结果 + 是否重规划 + 最终计划)。
        """
        notes = notes if notes is not None else ScratchPad()
        results: list[StepResult] = []
        replans_used = 0
        current_plan = plan
        pending = deque(plan.steps)

        while pending:
            step = pending.popleft()
            result = self._run_one(step, notes.render())
            results.append(result)
            label = f"step-{step.id} {step.specialist}"
            notes.append(label, result.output if result.ok else f"(失败){result.output}")

            if result.ok:
                continue
            if self._replanner is None or replans_used >= self._max_replans:
                continue  # 预算耗尽/未配重规划:失败如实留在结果里,继续剩余步骤

            replans_used += 1
            completed = [r for r in results if r.ok]
            current_plan = self._replanner.replan(
                current_plan,
                completed=completed,
                failure=result,
                roster=self._roster,
                specialists=self._specialists,
            )
            pending = deque(current_plan.steps)  # 剩余步骤整体替换为新计划

        return ExecutionResult(results=results, replanned=replans_used > 0, plan=current_plan)

    def _run_one(self, step: PlanStep, context: str) -> StepResult:
        """跑一步;runner 抛异常折叠为失败结果(执行器永不炸)。"""
        try:
            return self._runner.run_step(step, context)
        except Exception as exc:  # noqa: BLE001 — 单步崩溃不拖垮整轮执行
            return StepResult(step=step, output=f"执行异常:{exc}", ok=False, trace=None)
