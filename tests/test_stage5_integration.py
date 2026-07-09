"""阶段五端到端集成测试:复合客诉一条龙(全离线,MockLLM 脚本化)。

模拟 demo 主场景的完整链路:
Router 判复杂 → Planner 拆 3 步 → 订单专员查证(真调 mock 工具)→
售后专员退款被 7 天规则拒绝(真调 apply_refund,失败上报)→ 动态重规划 →
售后专员建工单(真调 create_ticket,状态落库)→ 导购专员推荐 →
汇总 → Critic 一审不合格 → 回炉 → 二审通过。

与 test_multi_agent 的分工:那边逐模块验证行为,这里验证**真实工具链 + 全模块
串联**——工具不是脚本,是真的 JDMockStore(退款拒绝与工单落库都是真实状态变化)。
"""

from __future__ import annotations

from agent_framework.core.llm import ChatResponse, ToolCall, Usage
from agent_framework.multi_agent import (
    SUPERVISOR_TARGET,
    Critic,
    Router,
    Supervisor,
    create_specialists,
)
from agent_framework.planning import Planner
from agent_framework.tools.jd_mock_data import JDMockStore
from agent_framework.tools.presets import default_registry
from tests.mock_llm import MockLLM


def _tool_call(name: str, args: dict[str, object], call_id: str = "tc-1") -> ChatResponse:
    return ChatResponse(
        content="",
        usage=Usage(0, 0),
        model="mock",
        tool_calls=[ToolCall(id=call_id, name=name, args=args)],
    )


def test_compound_complaint_end_to_end() -> None:
    store = JDMockStore()
    registry = default_registry(store)
    specialists = create_specialists(registry)

    script: list[str | ChatResponse] = [
        # ── Router:判复杂,升级中心调度
        f'{{"target":"{SUPERVISOR_TARGET}","reason":"查证+退款+推荐,跨域多动作"}}',
        # ── Planner:拆 3 步
        '[{"step":"查询订单 11111 的状态与签收时间","specialist":"order_agent"},'
        '{"step":"为订单 11111 申请退款","specialist":"aftersales_agent"},'
        '{"step":"推荐一款蓝牙耳机替代品","specialist":"product_agent"}]',
        # ── step-1 订单专员:真调 query_order → 作答
        _tool_call("query_order", {"order_id": "11111"}),
        "订单 11111(小米 蓝牙耳机 Pro)已签收。",
        # ── step-2 售后专员:真调 apply_refund → 被 7 天规则拒绝 → 明示失败
        _tool_call("apply_refund", {"order_id": "11111", "reason": "商品损坏"}),
        "无法完成:订单 11111 已超出 7 天无理由退货期,自动退款通道走不通。",
        # ── 重规划:改走建工单 + 保留推荐
        '[{"step":"为订单 11111 创建质量售后工单转人工","specialist":"aftersales_agent"},'
        '{"step":"推荐一款蓝牙耳机替代品","specialist":"product_agent"}]',
        # ── 新 step:售后专员真调 create_ticket → 作答
        _tool_call("create_ticket", {"summary": "订单 11111 耳机质量问题售后"}),
        "已创建人工工单 T0001,售后专员将跟进质量退款。",
        # ── 新 step:导购专员真调 query_product → 作答
        _tool_call("query_product", {"keyword": "耳机"}),
        "推荐京造机械键盘不合适;推荐同类蓝牙耳机:小米 蓝牙耳机 Pro 299 元。",
        # ── 汇总一稿
        "已为您转人工处理退款(工单 T0001)。",
        # ── Critic 一审:漏了推荐诉求
        '{"passed": false, "issues": ["未回应替代品推荐诉求"], "suggestion": "补上推荐"}',
        # ── 回炉二稿
        "您的耳机已超 7 天无理由期,已建工单 T0001 转人工跟进质量退款;替代品推荐:小米蓝牙耳机 Pro。",
        # ── Critic 二审:通过
        '{"passed": true, "issues": [], "suggestion": ""}',
    ]
    llm = MockLLM(script)
    router = Router(llm, specialists)
    supervisor = Supervisor(llm, specialists, planner=Planner(llm), critic=Critic(llm))

    query = "我买的蓝牙耳机(订单 11111)用了几天就坏了,查下能不能退,能退帮我退,再推荐个替代品"
    decision = router.route(query)
    assert decision.target == SUPERVISOR_TARGET

    result = supervisor.handle(query)

    # 链路轨迹:3 步原计划 → step-2 失败 → 重规划 2 步,共执行 4 步
    assert result.replanned is True
    executed = [(r.step.specialist, r.ok) for r in result.step_results]
    assert executed == [
        ("order_agent", True),
        ("aftersales_agent", False),
        ("aftersales_agent", True),
        ("product_agent", True),
    ]
    # 步骤编号全局不重复且续接(1,2 → 重规划 4,5)
    assert [r.step.id for r in result.step_results] == [1, 2, 4, 5]

    # 真实工具状态变化:退款没落库,工单真的建了
    assert store.refunds == []
    assert len(store.tickets) == 1

    # 质检回炉:一审不合格 → 二稿 → 二审通过
    assert [c.passed for c in result.critiques] == [False, True]
    assert result.resynthesized is True
    assert "T0001" in result.final_answer and "推荐" in result.final_answer

    # 全轮 LLM 调用数有界且符合预期:1 分诊 + 1 计划 + 4×2 专员 + 1 重规划 + 2 汇总 + 2 质检 = 15
    assert llm.call_count == 15


def test_simple_query_fast_path_end_to_end() -> None:
    """快路径:分诊直派订单专员,真调两个工具接力(查订单 → 查物流)。"""
    registry = default_registry(JDMockStore())
    specialists = create_specialists(registry)
    llm = MockLLM(
        [
            '{"target":"order_agent","reason":"单纯物流查询"}',
            _tool_call("query_order", {"order_id": "12345"}),
            _tool_call("query_logistics", {"tracking_no": "SF123"}, call_id="tc-2"),
            "您的订单 12345 正在运输中,预计明天送达。",
        ]
    )
    decision = Router(llm, specialists).route("订单 12345 到哪了")
    assert decision.target == "order_agent"

    from agent_framework.multi_agent import TaskAssignment

    outcome = specialists[decision.target].handle(llm, TaskAssignment("订单 12345 到哪了"))
    assert outcome.ok is True
    assert "预计明天送达" in outcome.answer
    assert llm.call_count == 4  # 快路径全程 4 次调用,对照复杂链路的 15 次
