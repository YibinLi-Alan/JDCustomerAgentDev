"""阶段六 P-B 编排层 HITL 集成测试:Supervisor 的三个升级入口 + 闸门端到端(全离线)。"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from agent_framework.multi_agent import (
    FAILURE_MARKER,
    Critic,
    Supervisor,
    create_specialists,
)
from agent_framework.observability import Tracer
from agent_framework.planning import Planner
from agent_framework.safety import ApprovalGate, ApprovalPolicy, BoundaryRegistry, HandoffQueue
from agent_framework.safety.rate_limiter import TokenBudget
from agent_framework.tools.jd_mock_data import JDMockStore
from agent_framework.tools.presets import default_registry
from tests.mock_llm import MockLLM

PLAN_2 = (
    '[{"step":"查订单 11111","specialist":"order_agent"},'
    '{"step":"为订单 11111 退款","specialist":"aftersales_agent"}]'
)


def _queue(tmp_path: Path) -> HandoffQueue:
    return HandoffQueue(tmp_path / "queue.json")


# --------------------------------------------------------------------------- #
# 入口 B-①:重规划耗尽仍失败 → 升级 + 汇总如实告知                                   #
# --------------------------------------------------------------------------- #


def test_unresolved_failure_escalates(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    llm = MockLLM(
        [
            PLAN_2,
            "订单 11111 已签收,已超 7 天。",  # step-1 成功
            f"{FAILURE_MARKER}:超退款时效。",  # step-2 失败 → 触发重规划
            '[{"step":"再次尝试特殊通道退款","specialist":"aftersales_agent"}]',  # replan
            f"{FAILURE_MARKER}:特殊通道也不行。",  # 新步骤也失败(预算耗尽,不再重规划)
            "已为您转人工跟进退款事宜。",  # 汇总
        ]
    )
    supervisor = Supervisor(
        llm,
        create_specialists(default_registry(JDMockStore())),
        planner=Planner(llm),
        handoff=queue,
    )
    result = supervisor.handle("给订单 11111 退款", user_id="alice")

    assert len(result.escalations) == 1
    item = result.escalations[0]
    assert item.kind == "escalation" and item.user_id == "alice"
    assert "自动处理未完成" in item.summary
    assert queue.get(item.id).status == "pending"
    # 汇总 prompt 里带了「已转人工 + 工单号」的系统提示
    synth_prompt = llm.seen_messages[-1][0].content
    assert item.id in synth_prompt and "转人工" in synth_prompt


# --------------------------------------------------------------------------- #
# 入口 B-②:质检二审仍不合格 → 放行 + 升级复核                                      #
# --------------------------------------------------------------------------- #


def test_critic_double_fail_escalates(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    llm = MockLLM(
        [
            "处理完成。",  # 兜底专员(无 planner)
            "一稿。",  # 汇总
            '{"passed": false, "issues": ["漏了诉求"], "suggestion": ""}',
            "二稿。",  # 回炉
            '{"passed": false, "issues": ["还是漏"], "suggestion": ""}',  # 二审仍不合格
        ]
    )
    supervisor = Supervisor(
        llm,
        create_specialists(default_registry(JDMockStore())),
        critic=Critic(llm),
        handoff=queue,
    )
    result = supervisor.handle("复合诉求", user_id="bob")
    assert result.final_answer == "二稿。"  # 放行
    assert len(result.escalations) == 1
    assert "人工复核" in result.escalations[0].summary
    assert "二稿" in result.escalations[0].context  # 上下文快照带答复与质检意见


# --------------------------------------------------------------------------- #
# 入口 B-③:第④层保险 —— 整任务超时 / token 预算耗尽                                #
# --------------------------------------------------------------------------- #


def test_task_deadline_interrupts_and_escalates(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    # 时钟脚本:算 deadline(0.0)→ 步1 前检查(10.0,未超)→ 步2 前检查(100.0,超时)
    clock = iter([0.0, 10.0, 100.0, 100.1, 100.2]).__next__
    llm = MockLLM([PLAN_2, "step-1 完成。"])
    supervisor = Supervisor(
        llm,
        create_specialists(default_registry(JDMockStore())),
        planner=Planner(llm),
        handoff=queue,
        task_deadline_seconds=50.0,
        clock=clock,
    )
    result = supervisor.handle("退款", user_id="alice")

    assert result.interrupted is True
    assert len(result.step_results) == 1  # 第二步没执行
    assert len(result.escalations) == 1 and "超时" in result.escalations[0].summary
    assert result.escalations[0].id in result.final_answer  # 答复带工单号
    assert llm.call_count == 2  # 超时后不再烧钱汇总


def test_token_budget_exhaustion_interrupts(tmp_path: Path) -> None:
    from agent_framework.core.llm import ChatResponse, Usage

    queue = _queue(tmp_path)
    big_usage = ChatResponse(content="step-1 完成。", usage=Usage(90, 20), model="mock")
    llm = MockLLM([PLAN_2, big_usage])
    tracer = Tracer("t-budget")
    supervisor = Supervisor(
        llm,
        create_specialists(default_registry(JDMockStore())),
        planner=Planner(llm),
        handoff=queue,
    )
    result = supervisor.handle(
        "退款", user_id="alice", tracer=tracer, token_budget=TokenBudget(limit=100)
    )
    assert result.interrupted is True
    assert "预算" in result.escalations[0].summary


# --------------------------------------------------------------------------- #
# 闸门 × 编排端到端:高权限动作被拦 → 专员如实告知 → 审批放行真执行                     #
# --------------------------------------------------------------------------- #


def test_gate_supervisor_end_to_end(tmp_path: Path) -> None:
    from agent_framework.core.llm import ChatResponse, ToolCall, Usage

    store = JDMockStore()
    queue = _queue(tmp_path)
    policy = ApprovalPolicy(("high",))
    specialists = create_specialists(default_registry(store))
    # 装配:给每个专员的工具子集包上 边界标记 + 审批闸门(dataclasses.replace,专员定义零改动)
    specialists = {
        name: dataclasses.replace(
            spec,
            registry=ApprovalGate(  # type: ignore[arg-type]
                BoundaryRegistry(spec.registry),
                queue,
                policy,
                user_id_provider=lambda: "alice",
            ),
        )
        for name, spec in specialists.items()
    }

    refund_call = ChatResponse(
        content="",
        usage=Usage(0, 0),
        model="mock",
        tool_calls=[
            ToolCall(id="c1", name="apply_refund", args={"order_id": "12345", "reason": "损坏"})
        ],
    )
    llm = MockLLM(
        [
            '[{"step":"为订单 12345 申请退款","specialist":"aftersales_agent"}]',  # 计划
            refund_call,  # 专员调 apply_refund → 被闸门拦下
            "退款申请已提交人工审批,审批单号见回执,预计 24 小时内处理。",  # 专员作答
            "您的退款已提交人工审批,请留意后续通知。",  # 汇总
        ]
    )
    tracer = Tracer("t-gate")
    supervisor = Supervisor(llm, specialists, planner=Planner(llm), handoff=queue)
    result = supervisor.handle("给订单 12345 退款", user_id="alice", tracer=tracer)

    # 拦截生效:没真退款,队列里有 approval 单;步骤视为成功(移交≠失败)
    assert store.refunds == []
    approvals = queue.list(kind="approval")
    assert len(approvals) == 1
    assert result.step_results[0].ok is True
    assert "人工审批" in result.final_answer

    # 人工放行 → 真执行(request_id 幂等,重复 approve 抛错防重放)
    queue.approve(approvals[0].id, default_registry(store))
    assert len(store.refunds) == 1

    # 轨迹里有步骤级与内环级事件
    kinds = [e.kind for e in tracer.events]
    assert "plan" in kinds and "step_start" in kinds and "tool_call" in kinds
    assert "synthesize" in kinds
