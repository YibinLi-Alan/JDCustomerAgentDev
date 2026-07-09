"""PlanExecutor 与 ScratchPad 单测(全离线,假 runner + 假 replanner)。

覆盖:黑板渲染/顺序/截断 · 顺序执行与中间结果传递 · 失败触发重规划恰一次 ·
重规划后再失败不再触发 · 未配重规划器时失败照记继续 · runner 异常折叠 ·
subset 切子集(registry 的阶段五新方法)。
"""

from __future__ import annotations

from agent_framework.planning import (
    ExecutionResult,
    Plan,
    PlanExecutor,
    PlanStep,
    ScratchPad,
    StepResult,
)
from agent_framework.tools.jd_mock_data import JDMockStore
from agent_framework.tools.presets import default_registry
from agent_framework.tools.registry import ToolRegistry

# --------------------------------------------------------------------------- #
# 测试基础设施:脚本化 runner / 计数 replanner                                     #
# --------------------------------------------------------------------------- #


class ScriptedRunner:
    """按步骤描述查表应答的假 runner;记录每步收到的 context 供断言。"""

    def __init__(self, outcomes: dict[str, tuple[str, bool]]) -> None:
        self._outcomes = outcomes
        self.seen: list[tuple[PlanStep, str]] = []

    def run_step(self, step: PlanStep, context: str) -> StepResult:
        self.seen.append((step, context))
        output, ok = self._outcomes.get(step.description, (f"完成:{step.description}", True))
        if output == "__raise__":
            raise RuntimeError("专员进程崩了")
        return StepResult(step=step, output=output, ok=ok)


class CountingReplanner:
    """记录调用次数、返回固定新计划的假 replanner(鸭子类型替代 Planner)。"""

    def __init__(self, new_steps: tuple[PlanStep, ...]) -> None:
        self._new_steps = new_steps
        self.calls: list[dict[str, object]] = []

    def replan(self, plan: Plan, *, completed, failure, roster, specialists) -> Plan:  # type: ignore[no-untyped-def]
        self.calls.append({"completed": list(completed), "failure": failure, "roster": roster})
        return Plan(goal=plan.goal, steps=self._new_steps)


def _plan(*descriptions: str) -> Plan:
    return Plan(
        goal="测试目标",
        steps=tuple(PlanStep(i + 1, desc, "order_agent") for i, desc in enumerate(descriptions)),
    )


# --------------------------------------------------------------------------- #
# ScratchPad                                                                    #
# --------------------------------------------------------------------------- #


def test_scratchpad_empty_renders_empty() -> None:
    assert ScratchPad().render() == ""


def test_scratchpad_renders_in_order_with_labels() -> None:
    pad = ScratchPad()
    pad.append("step-1 order_agent", "订单已签收")
    pad.append("step-2 aftersales_agent", "退款已受理")
    text = pad.render()
    assert text.startswith("【前序步骤结论】")
    assert text.index("[step-1 order_agent] 订单已签收") < text.index(
        "[step-2 aftersales_agent] 退款已受理"
    )


def test_scratchpad_truncates_oldest_first() -> None:
    pad = ScratchPad(max_chars=60)
    pad.append("step-1 a", "早" * 50)
    pad.append("step-2 b", "新结论")
    text = pad.render()
    assert "新结论" in text
    assert "早早" not in text
    assert "更早的记录已截断" in text


def test_scratchpad_keeps_single_oversized_entry() -> None:
    # 只有一条时即便超预算也保留(最新结论永不丢,与滑窗「最新轮永不弹出」同理)
    pad = ScratchPad(max_chars=10)
    pad.append("step-1 a", "很长的结论" * 10)
    assert "很长的结论" in pad.render()


# --------------------------------------------------------------------------- #
# PlanExecutor                                                                  #
# --------------------------------------------------------------------------- #


def test_execute_sequential_and_context_flows_between_steps() -> None:
    runner = ScriptedRunner({"查订单": ("订单已签收,单号SF123", True)})
    result = PlanExecutor(runner).execute(_plan("查订单", "申请退款"))
    assert isinstance(result, ExecutionResult)
    assert [r.ok for r in result.results] == [True, True]
    # 第 1 步 context 为空;第 2 步能看到第 1 步写进黑板的结论
    assert runner.seen[0][1] == ""
    assert "单号SF123" in runner.seen[1][1]
    assert "step-1 order_agent" in runner.seen[1][1]


def test_execute_failure_triggers_replan_exactly_once() -> None:
    new_steps = (PlanStep(4, "建工单转人工", "order_agent"),)
    replanner = CountingReplanner(new_steps)
    runner = ScriptedRunner({"申请退款": ("退款超时效", False)})
    executor = PlanExecutor(runner, replanner=replanner, roster="r", specialists=("order_agent",))  # type: ignore[arg-type]

    result = executor.execute(_plan("查订单", "申请退款", "推荐替代品"))

    assert len(replanner.calls) == 1
    assert result.replanned is True
    # 已完成的只有 step-1;失败信息传给了 replan
    call = replanner.calls[0]
    assert [r.step.description for r in call["completed"]] == ["查订单"]  # type: ignore[index]
    assert call["failure"].output == "退款超时效"  # type: ignore[union-attr]
    # 剩余步骤(推荐替代品)被新计划整体替换
    executed = [r.step.description for r in result.results]
    assert executed == ["查订单", "申请退款", "建工单转人工"]
    assert result.plan is not None and result.plan.steps == new_steps
    # 失败结论带「(失败)」前缀进了黑板,新步骤看得见
    assert "(失败)退款超时效" in runner.seen[-1][1]


def test_execute_replan_budget_exhausted_records_failure_and_continues() -> None:
    new_steps = (
        PlanStep(4, "建工单转人工", "order_agent"),
        PlanStep(5, "推荐替代品", "order_agent"),
    )
    replanner = CountingReplanner(new_steps)
    runner = ScriptedRunner(
        {"申请退款": ("退款超时效", False), "建工单转人工": ("工单系统也挂了", False)}
    )
    executor = PlanExecutor(runner, replanner=replanner, specialists=("order_agent",))  # type: ignore[arg-type]

    result = executor.execute(_plan("查订单", "申请退款"))

    assert len(replanner.calls) == 1  # 第二次失败不再重规划
    outcomes = [(r.step.description, r.ok) for r in result.results]
    assert outcomes == [
        ("查订单", True),
        ("申请退款", False),
        ("建工单转人工", False),
        ("推荐替代品", True),  # 预算耗尽后继续跑完剩余步骤
    ]


def test_execute_without_replanner_records_failure_and_continues() -> None:
    runner = ScriptedRunner({"申请退款": ("退款超时效", False)})
    result = PlanExecutor(runner).execute(_plan("查订单", "申请退款", "推荐替代品"))
    assert result.replanned is False
    assert [(r.step.description, r.ok) for r in result.results] == [
        ("查订单", True),
        ("申请退款", False),
        ("推荐替代品", True),
    ]


def test_execute_runner_exception_folded_to_failed_result() -> None:
    runner = ScriptedRunner({"查订单": ("__raise__", False)})
    result = PlanExecutor(runner).execute(_plan("查订单", "推荐替代品"))
    first = result.results[0]
    assert first.ok is False
    assert "专员进程崩了" in first.output
    assert result.results[1].ok is True  # 单步崩溃不拖垮整轮


def test_execute_reuses_caller_scratchpad() -> None:
    pad = ScratchPad()
    runner = ScriptedRunner({})
    PlanExecutor(runner).execute(_plan("查订单"), notes=pad)
    assert len(pad.entries) == 1  # Supervisor 汇总时可复用同一块黑板


# --------------------------------------------------------------------------- #
# ToolRegistry.subset(阶段五新方法)                                             #
# --------------------------------------------------------------------------- #


def test_registry_subset_picks_tools_and_shares_instances() -> None:
    registry = default_registry(JDMockStore())
    sub = registry.subset("query_order", "query_logistics")
    assert sub.names == ["query_order", "query_logistics"]
    assert sub.get("query_order") is registry.get("query_order")  # 实例共享,store 状态互通


def test_registry_subset_unknown_name_raises() -> None:
    from agent_framework.tools.registry import UnknownToolError

    registry = ToolRegistry([])
    try:
        registry.subset("ghost")
    except UnknownToolError:
        pass
    else:  # pragma: no cover
        raise AssertionError("未知工具名应当抛 UnknownToolError")
