"""Multi-Agent 单测(全离线,MockLLM 脚本化)。

覆盖:protocol(专员封装/失败前缀/花名册/任务拼装)· specialists(装配/工具子集)·
Router(直派/升级/降级)· Critic(通过/不合格/降级)· Supervisor(全链路编排/
退化路径/汇总降级/质检回炉/记忆注入)。
"""

from __future__ import annotations

import pytest

from agent_framework.core.llm import ChatResponse, ToolCall, Usage
from agent_framework.multi_agent import (
    FAILURE_MARKER,
    SUPERVISOR_TARGET,
    Critic,
    Router,
    Supervisor,
    TaskAssignment,
    create_specialists,
    render_roster,
)
from agent_framework.planning import Planner
from agent_framework.tools.jd_mock_data import JDMockStore
from agent_framework.tools.presets import default_registry
from tests.mock_llm import MockLLM


def _registry():
    return default_registry(JDMockStore())


def _tool_call_response(name: str, args: dict[str, object] | None = None) -> ChatResponse:
    return ChatResponse(
        content="",
        usage=Usage(0, 0),
        model="mock",
        tool_calls=[ToolCall(id="tc-1", name=name, args=args or {})],
    )


# --------------------------------------------------------------------------- #
# protocol:Specialist / TaskAssignment / 花名册                                 #
# --------------------------------------------------------------------------- #


def test_task_assignment_composes_context_and_task() -> None:
    assert TaskAssignment("查订单").to_user_input() == "查订单"
    assignment = TaskAssignment("申请退款", context="【前序步骤结论】\n[step-1] 已签收")
    composed = assignment.to_user_input()
    assert composed.index("已签收") < composed.index("【当前任务】")
    assert composed.endswith("申请退款")


def test_specialist_build_appends_extra_system() -> None:
    spec = create_specialists(_registry())["order_agent"]
    agent = spec.build(MockLLM([]), extra_system="\n\n【关于该用户的已知信息】住上海")
    assert agent.system_prompt.startswith(spec.system_prompt)
    assert agent.system_prompt.endswith("住上海")


def test_specialist_handle_success_and_failure_marker() -> None:
    spec = create_specialists(_registry())["order_agent"]
    ok_outcome = spec.handle(MockLLM(["您的订单已签收。"]), TaskAssignment("查订单 12345"))
    assert ok_outcome.ok is True
    assert ok_outcome.specialist == "order_agent"

    fail = spec.handle(
        MockLLM([f"{FAILURE_MARKER}:该诉求属于售后操作,超出我的职责。"]),
        TaskAssignment("给订单 12345 退款"),
    )
    assert fail.ok is False
    assert FAILURE_MARKER in fail.answer


def test_specialist_handle_max_steps_reports_failure() -> None:
    spec = create_specialists(_registry())["order_agent"]
    llm = MockLLM([_tool_call_response("current_time"), "只能查到这里。"])
    outcome = spec.handle(llm, TaskAssignment("查订单"), max_steps=1)
    assert outcome.ok is False  # 撞步数上限 = 失败(协议约定)
    assert outcome.trace is not None and outcome.trace.stopped_reason == "max_steps"


def test_create_specialists_subsets_and_roster() -> None:
    specialists = create_specialists(_registry())
    assert list(specialists) == ["order_agent", "aftersales_agent", "product_agent"]
    assert specialists["aftersales_agent"].registry.names == [
        "query_order",
        "apply_refund",
        "cancel_order",
        "create_ticket",
        "search_faq",
    ]
    assert "apply_refund" not in specialists["order_agent"].registry
    roster = render_roster(list(specialists.values()))
    for name in specialists:
        assert name in roster
    # 失败前缀约定写进了每个专员的 prompt(协议的两端:prompt 约定 + 编排器解析)
    for spec in specialists.values():
        assert FAILURE_MARKER in spec.system_prompt


# --------------------------------------------------------------------------- #
# Router                                                                        #
# --------------------------------------------------------------------------- #


def _router(script: list[str]) -> Router:
    return Router(MockLLM(script), create_specialists(_registry()))


def test_router_dispatches_to_specialist() -> None:
    decision = _router(['{"target":"order_agent","reason":"单纯查订单"}']).route("订单12345到哪了")
    assert decision.target == "order_agent"
    assert decision.reason == "单纯查订单"


def test_router_escalates_complex_to_supervisor() -> None:
    decision = _router([f'{{"target":"{SUPERVISOR_TARGET}","reason":"退款+推荐跨域"}}']).route(
        "耳机坏了退款并推荐新的"
    )
    assert decision.target == SUPERVISOR_TARGET


@pytest.mark.parametrize(
    "bad_response",
    ["我觉得应该找订单专员。", '{"target":"ghost_agent","reason":"?"}', '["not","object"]'],
)
def test_router_degrades_to_supervisor(bad_response: str) -> None:
    decision = _router([bad_response]).route("随便问问")
    assert decision.target == SUPERVISOR_TARGET
    assert "降级" in decision.reason


# --------------------------------------------------------------------------- #
# Critic                                                                        #
# --------------------------------------------------------------------------- #


def test_critic_passes() -> None:
    critique = Critic(MockLLM(['{"passed": true, "issues": [], "suggestion": ""}'])).review(
        "查订单", "已签收。", "step-1 → 成功:已签收"
    )
    assert critique.passed is True
    assert critique.degraded is False


def test_critic_fails_with_issues() -> None:
    critique = Critic(
        MockLLM(['{"passed": false, "issues": ["没有回应推荐诉求"], "suggestion": "补上推荐"}'])
    ).review("退款并推荐", "退款办好了。", "…")
    assert critique.passed is False
    assert critique.issues == ["没有回应推荐诉求"]


@pytest.mark.parametrize("bad", ["这答复不错。", '{"no_passed": 1}', '{"passed": "yes"}'])
def test_critic_degrades_to_pass(bad: str) -> None:
    critique = Critic(MockLLM([bad])).review("q", "a", "e")
    assert critique.passed is True
    assert critique.degraded is True


# --------------------------------------------------------------------------- #
# Supervisor                                                                    #
# --------------------------------------------------------------------------- #

PLAN_2_STEPS = (
    '[{"step":"查订单 12345 的状态","specialist":"order_agent"},'
    '{"step":"为订单 12345 申请退款","specialist":"aftersales_agent"}]'
)


def test_supervisor_full_pipeline_order() -> None:
    llm = MockLLM(
        [
            PLAN_2_STEPS,  # ① Planner
            "订单 12345 已签收,在退款时效内。",  # ② step-1 订单专员直接作答
            "退款已受理,退款单号 RF-1。",  # ③ step-2 售后专员直接作答
            "您好,退款已受理(单号 RF-1),请留意到账。",  # ④ 汇总
        ]
    )
    specialists = create_specialists(_registry())
    supervisor = Supervisor(llm, specialists, planner=Planner(llm))
    result = supervisor.handle("耳机坏了,帮我退掉订单 12345")

    assert result.final_answer.startswith("您好,退款已受理")
    assert [r.step.specialist for r in result.step_results] == ["order_agent", "aftersales_agent"]
    assert all(r.ok for r in result.step_results)
    assert result.replanned is False
    assert result.critiques == []  # 未配 Critic
    # step-2 的输入里能看到 step-1 写进黑板的结论(ScratchPad 生效)
    step2_input = llm.seen_messages[2][0].content
    assert "已签收" in step2_input
    assert "【当前任务】" in step2_input
    # 汇总 prompt 里有执行记录
    synth_prompt = llm.seen_messages[3][0].content
    assert "RF-1" in synth_prompt and "执行记录" in synth_prompt


def test_supervisor_injects_memory_into_every_specialist() -> None:
    memory = "\n\n【关于该用户的已知信息(长期记忆,仅供参考)】\n- 常用地址:上海"
    llm = MockLLM([PLAN_2_STEPS, "步骤一完成。", "步骤二完成。", "汇总答复。"])
    supervisor = Supervisor(llm, create_specialists(_registry()), planner=Planner(llm))
    supervisor.handle("退款", memory_context=memory)
    # 调用序:0=plan,1=step1,2=step2,3=汇总;两个专员的 system 都带记忆附加段
    assert llm.seen_systems[1] is not None and llm.seen_systems[1].endswith("常用地址:上海")
    assert llm.seen_systems[2] is not None and llm.seen_systems[2].endswith("常用地址:上海")
    # 规划的已知背景里也有(帮助拆解时用上用户事实)
    assert "上海" in llm.seen_messages[0][0].content


def test_supervisor_without_planner_degrades_to_single_dispatch() -> None:
    llm = MockLLM(["全部处理完毕。", "汇总:处理完毕。"])
    supervisor = Supervisor(llm, create_specialists(_registry()))
    result = supervisor.handle("查订单 12345")
    assert len(result.step_results) == 1
    assert result.step_results[0].step.specialist == "order_agent"  # 兜底 = 第一个专员
    assert result.final_answer == "汇总:处理完毕。"


def test_supervisor_replans_on_step_failure() -> None:
    llm = MockLLM(
        [
            PLAN_2_STEPS,  # ① 计划
            "订单 12345 已签收,但已超 7 天退款时效。",  # ② step-1 成功
            f"{FAILURE_MARKER}:订单已超退款时效,无法自动退款。",  # ③ step-2 失败
            '[{"step":"建工单转人工跟进退款","specialist":"aftersales_agent"}]',  # ④ replan
            "工单已创建,编号 T-001。",  # ⑤ 新步骤成功
            "很抱歉超出自动退款时效,已为您转人工(工单 T-001)。",  # ⑥ 汇总
        ]
    )
    supervisor = Supervisor(llm, create_specialists(_registry()), planner=Planner(llm))
    result = supervisor.handle("给订单 12345 退款")

    assert result.replanned is True
    outcomes = [(r.step.description, r.ok) for r in result.step_results]
    assert outcomes[-1] == ("建工单转人工跟进退款", True)
    assert any(not ok for _, ok in outcomes)  # 失败步骤如实留在轨迹里
    assert "T-001" in result.final_answer
    # replan prompt 里带了失败原因
    replan_prompt = llm.seen_messages[3][0].content
    assert "超退款时效" in replan_prompt and "已完成步骤及结果" in replan_prompt


def test_supervisor_critic_retry_then_pass() -> None:
    llm = MockLLM(
        [
            "处理完成。",  # ① 兜底专员(无 planner)
            "退款办好了。",  # ② 汇总一稿
            # ③ 质检不合格
            '{"passed": false, "issues": ["没回应推荐诉求"], "suggestion": "补推荐"}',
            "退款办好了,另推荐 XX 耳机。",  # ④ 回炉重写
            '{"passed": true, "issues": [], "suggestion": ""}',  # ⑤ 二审通过
        ]
    )
    supervisor = Supervisor(llm, create_specialists(_registry()), critic=Critic(llm))
    result = supervisor.handle("退款并推荐替代品")

    assert result.resynthesized is True
    assert [c.passed for c in result.critiques] == [False, True]
    assert "推荐" in result.final_answer
    # 回炉的汇总 prompt 里带了质检意见
    retry_prompt = llm.seen_messages[3][0].content
    assert "质检意见" in retry_prompt and "没回应推荐诉求" in retry_prompt


def test_supervisor_critic_second_fail_releases_with_trace() -> None:
    llm = MockLLM(
        [
            "处理完成。",
            "一稿。",
            '{"passed": false, "issues": ["A"], "suggestion": ""}',
            "二稿。",
            '{"passed": false, "issues": ["B"], "suggestion": ""}',  # 二审仍不合格 → 放行留痕
        ]
    )
    supervisor = Supervisor(llm, create_specialists(_registry()), critic=Critic(llm))
    result = supervisor.handle("诉求")
    assert result.final_answer == "二稿。"
    assert [c.passed for c in result.critiques] == [False, False]


def test_supervisor_synthesis_failure_degrades_to_evidence_dump() -> None:
    # 脚本只给到专员应答,汇总调用时 MockLLM 脚本耗尽抛错 → 降级为执行记录拼接
    llm = MockLLM(["订单已签收,单号SF123。"])
    supervisor = Supervisor(llm, create_specialists(_registry()))
    result = supervisor.handle("查订单 12345")
    assert "单号SF123" in result.final_answer  # 信息不丢
    assert "系统繁忙" in result.final_answer


def test_supervisor_requires_specialists() -> None:
    with pytest.raises(ValueError):
        Supervisor(MockLLM([]), {})
