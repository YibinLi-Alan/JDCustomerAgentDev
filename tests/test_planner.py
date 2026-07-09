"""Planner 单测(全离线,MockLLM 脚本化)。

覆盖:正常拆解 / 围栏容忍 / 越界专员矫正 / JSON 非法降级单步 / 超长截断 /
空数组降级 / 缺字段条目跳过 / replan 编号续接与只排剩余 / replan 降级 / 空专员报错。
"""

from __future__ import annotations

import pytest

from agent_framework.planning import Plan, Planner, PlanStep, StepResult
from tests.mock_llm import MockLLM

ROSTER = "- order_agent:订单物流\n- aftersales_agent:售后\n- product_agent:导购"
SPECIALISTS = ("order_agent", "aftersales_agent", "product_agent")


def make_planner(script: list[str], **kwargs: int) -> Planner:
    return Planner(MockLLM(script), **kwargs)


def test_plan_parses_steps() -> None:
    planner = make_planner(
        [
            '[{"step":"查订单 12345","specialist":"order_agent"},'
            '{"step":"申请退款","specialist":"aftersales_agent"}]'
        ]
    )
    plan = planner.plan("耳机坏了要退款", roster=ROSTER, specialists=SPECIALISTS)
    assert plan.goal == "耳机坏了要退款"
    assert [s.id for s in plan.steps] == [1, 2]
    assert plan.steps[0].description == "查订单 12345"
    assert plan.steps[1].specialist == "aftersales_agent"


def test_plan_tolerates_code_fence() -> None:
    planner = make_planner(['```json\n[{"step":"查订单","specialist":"order_agent"}]\n```'])
    plan = planner.plan("查订单", roster=ROSTER, specialists=SPECIALISTS)
    assert len(plan.steps) == 1
    assert plan.steps[0].specialist == "order_agent"


def test_plan_unknown_specialist_corrected_to_fallback() -> None:
    planner = make_planner(['[{"step":"查订单","specialist":"ghost_agent"}]'])
    plan = planner.plan("查订单", roster=ROSTER, specialists=SPECIALISTS)
    assert plan.steps[0].specialist == "order_agent"  # specialists[0] 兜底


def test_plan_invalid_json_degrades_to_single_step() -> None:
    planner = make_planner(["抱歉,我认为应该先查订单再退款。"])
    plan = planner.plan("耳机坏了要退款", roster=ROSTER, specialists=SPECIALISTS)
    assert len(plan.steps) == 1
    assert plan.steps[0] == PlanStep(id=1, description="耳机坏了要退款", specialist="order_agent")


def test_plan_empty_array_degrades_to_single_step() -> None:
    planner = make_planner(["[]"])
    plan = planner.plan("查订单", roster=ROSTER, specialists=SPECIALISTS)
    assert len(plan.steps) == 1
    assert plan.steps[0].description == "查订单"


def test_plan_truncates_to_max_steps() -> None:
    items = ",".join(f'{{"step":"第{i}步","specialist":"order_agent"}}' for i in range(8))
    planner = make_planner([f"[{items}]"], max_steps=6)
    plan = planner.plan("复杂诉求", roster=ROSTER, specialists=SPECIALISTS)
    assert len(plan.steps) == 6


def test_plan_skips_items_without_description() -> None:
    planner = make_planner(
        [
            '[{"specialist":"order_agent"},{"step":"","specialist":"order_agent"},'
            '{"step":"查订单","specialist":"order_agent"},"不是字典"]'
        ]
    )
    plan = planner.plan("查订单", roster=ROSTER, specialists=SPECIALISTS)
    assert len(plan.steps) == 1
    assert plan.steps[0].description == "查订单"


def test_plan_requires_specialists() -> None:
    planner = make_planner([])
    with pytest.raises(ValueError):
        planner.plan("查订单", roster=ROSTER, specialists=())


def _three_step_plan() -> Plan:
    return Plan(
        goal="耳机坏了退款并推荐替代品",
        steps=(
            PlanStep(1, "查订单 12345", "order_agent"),
            PlanStep(2, "申请退款", "aftersales_agent"),
            PlanStep(3, "推荐替代品", "product_agent"),
        ),
    )


def test_replan_continues_numbering_and_feeds_context() -> None:
    plan = _three_step_plan()
    completed = [StepResult(plan.steps[0], "订单已签收,商品:降噪耳机", ok=True)]
    failure = StepResult(plan.steps[1], "退款超时效,无法自动退款", ok=False)
    llm = MockLLM(
        [
            '[{"step":"建工单转人工处理退款","specialist":"aftersales_agent"},'
            '{"step":"推荐替代品","specialist":"product_agent"}]'
        ]
    )
    new_plan = Planner(llm).replan(
        plan, completed=completed, failure=failure, roster=ROSTER, specialists=SPECIALISTS
    )
    assert new_plan.goal == plan.goal
    assert [s.id for s in new_plan.steps] == [4, 5]  # 接在原计划最大序号 3 之后
    # 重规划 prompt 里带了已完成结果与失败原因
    prompt = llm.seen_messages[0][0].content
    assert "订单已签收" in prompt
    assert "退款超时效" in prompt
    assert "已完成步骤及结果" in prompt


def test_replan_invalid_json_degrades_to_single_fallback_step() -> None:
    plan = _three_step_plan()
    failure = StepResult(plan.steps[1], "退款失败", ok=False)
    planner = make_planner(["办不了。"])
    new_plan = planner.replan(
        plan, completed=[], failure=failure, roster=ROSTER, specialists=SPECIALISTS
    )
    assert len(new_plan.steps) == 1
    step = new_plan.steps[0]
    assert step.id == 4
    assert step.specialist == "order_agent"
    assert "退款失败" in step.description  # 失败原因带进兜底步骤描述
    assert plan.goal in step.description
