"""阶段二 ReAct 内核关键路径单测（离线、确定性、零 API 成本）。

用 :class:`tests.mock_llm.MockLLM` 脚本化驱动 :class:`~agent_framework.core.agent.ReActAgent`，
覆盖 stage-2-design.md §9 全部用例：直接回答、单工具、两步工具链、解析失败恢复、
未知工具恢复、工具异常恢复、撞 max_steps 强制作答，以及 :func:`parse_step` 的单元测试。

约定：每条 MockLLM 脚本项都是「一步」的原始文本。凡触发工具执行的步，其下一步的
观察会被喂回，但 Mock 不消费上下文，故脚本纯按调用顺序供给即可。
"""

from __future__ import annotations

import json

import pytest

from agent_framework.core.agent import (
    AgentResult,
    ReActAgent,
    StepParseError,
    StepTrace,
    parse_step,
)
from agent_framework.tools.jd_mock import JD_MOCK_TOOLS
from tests.mock_llm import MockLLM


# --------------------------------------------------------------------------- #
# 脚本构造小工具:把「一步」的意图拼成模型该输出的 JSON 文本                       #
# --------------------------------------------------------------------------- #
def _action(thought: str, tool: str, **kwargs: object) -> str:
    """构造一条 action 步的 JSON 文本。"""
    return json.dumps(
        {"thought": thought, "action": {"tool": tool, "input": kwargs}},
        ensure_ascii=False,
    )


def _final(thought: str, answer: str) -> str:
    """构造一条 final_answer 步的 JSON 文本。"""
    return json.dumps(
        {"thought": thought, "final_answer": answer},
        ensure_ascii=False,
    )


# --------------------------------------------------------------------------- #
# 1. 直接回答:第一步就 final_answer                                            #
# --------------------------------------------------------------------------- #
def test_direct_answer() -> None:
    """第一步就给 final_answer,应立即收尾、只有一步。"""
    llm = MockLLM([_final("无需查询,直接回答", "您好,有什么可以帮您?")])
    agent = ReActAgent(llm, tools=JD_MOCK_TOOLS)

    result = agent.run("你好")

    assert isinstance(result, AgentResult)
    assert result.stopped_reason == "final_answer"
    assert result.final_answer == "您好,有什么可以帮您?"
    assert len(result.steps) == 1
    assert result.steps[0].final_answer == "您好,有什么可以帮您?"
    assert result.steps[0].action is None
    assert llm.call_count == 1


# --------------------------------------------------------------------------- #
# 2. 单工具:query_order → 作答                                                 #
# --------------------------------------------------------------------------- #
def test_single_tool() -> None:
    """一步查订单拿到观察,再一步作答。"""
    llm = MockLLM(
        [
            _action("先查订单", "query_order", order_id="12345"),
            _final("已知订单状态,可以回答", "您的订单 12345 已发货。"),
        ]
    )
    agent = ReActAgent(llm, tools=JD_MOCK_TOOLS)

    result = agent.run("我的订单 12345 状态?")

    assert result.stopped_reason == "final_answer"
    assert len(result.steps) == 2
    # 第一步是工具步,观察里应含真实工具输出
    assert result.steps[0].action is not None
    assert result.steps[0].action.tool == "query_order"
    assert result.steps[0].observation is not None
    assert "SF123" in result.steps[0].observation
    assert "已发货" in result.steps[0].observation
    # 第二步收尾
    assert result.steps[1].final_answer == "您的订单 12345 已发货。"


# --------------------------------------------------------------------------- #
# 3. 两步工具链(核心用例):query_order → query_logistics → 作答                 #
# --------------------------------------------------------------------------- #
def test_two_step_tool_chain() -> None:
    """核心链路:查订单拿单号 → 查物流拿进度 → 作答;逐步断言观察内容。"""
    llm = MockLLM(
        [
            _action("先查订单拿运单号", "query_order", order_id="12345"),
            _action("拿到 SF123,再查物流", "query_logistics", tracking_no="SF123"),
            _final("已知物流进度,可以回答", "您的订单顺丰运输中,预计明天送达。"),
        ]
    )
    agent = ReActAgent(llm, tools=JD_MOCK_TOOLS)

    result = agent.run("我的订单 12345 到哪了?")

    assert result.stopped_reason == "final_answer"
    assert len(result.steps) == 3

    # 第一步:查订单,观察含运单号 SF123
    assert result.steps[0].action is not None
    assert result.steps[0].action.tool == "query_order"
    assert result.steps[0].observation is not None
    assert "SF123" in result.steps[0].observation

    # 第二步:查物流,观察含物流进度
    assert result.steps[1].action is not None
    assert result.steps[1].action.tool == "query_logistics"
    assert result.steps[1].observation is not None
    assert "SF123" in result.steps[1].observation
    assert "预计明天送达" in result.steps[1].observation

    # 第三步:收尾
    assert result.steps[2].final_answer == "您的订单顺丰运输中,预计明天送达。"


# --------------------------------------------------------------------------- #
# 4. 解析失败恢复:先非法 JSON,再合法 final_answer                              #
# --------------------------------------------------------------------------- #
def test_parse_error_recovery() -> None:
    """一步非法 JSON 应被记为 error 步,下一步合法输出成功收尾。"""
    llm = MockLLM(
        [
            "这不是 JSON,只是一句话。",  # 非法 → 触发 StepParseError
            _final("这次乖乖输出 JSON", "已为您处理。"),
        ]
    )
    agent = ReActAgent(llm, tools=JD_MOCK_TOOLS)

    result = agent.run("帮我看看")

    assert result.stopped_reason == "final_answer"
    assert result.final_answer == "已为您处理。"
    assert len(result.steps) == 2
    # 第一步是解析失败步:error 被记录,raw 保留原始文本
    assert result.steps[0].error is not None
    assert result.steps[0].raw == "这不是 JSON,只是一句话。"
    assert result.steps[0].final_answer is None
    # 第二步成功收尾
    assert result.steps[1].final_answer == "已为您处理。"


# --------------------------------------------------------------------------- #
# 5. 未知工具恢复:调不存在的工具,再作答                                        #
# --------------------------------------------------------------------------- #
def test_unknown_tool_recovery() -> None:
    """调用不存在的工具应把「工具...不存在」当观察喂回,下一步成功收尾。"""
    llm = MockLLM(
        [
            _action("试试这个工具", "not_a_real_tool", foo="bar"),
            _final("那个工具不存在,直接回答", "抱歉,暂时无法查询。"),
        ]
    )
    agent = ReActAgent(llm, tools=JD_MOCK_TOOLS)

    result = agent.run("随便问问")

    assert result.stopped_reason == "final_answer"
    assert len(result.steps) == 2
    assert result.steps[0].observation is not None
    assert "不存在" in result.steps[0].observation
    assert "not_a_real_tool" in result.steps[0].observation
    assert result.steps[1].final_answer == "抱歉,暂时无法查询。"


# --------------------------------------------------------------------------- #
# 6. 工具异常恢复:内联一个总是抛错的工具                                        #
# --------------------------------------------------------------------------- #
class _BoomTool:
    """内联测试工具:调用即抛异常,用于验证工具异常的错误恢复。"""

    name: str = "boom"
    description: str = "总是抛错(测试用)。"

    def run(self, **kwargs: object) -> str:
        """无条件抛出运行时异常。"""
        raise RuntimeError("炸了")


def test_tool_exception_recovery() -> None:
    """工具执行抛异常应被捕获、把「工具执行失败」当观察喂回,下一步成功收尾。"""
    llm = MockLLM(
        [
            _action("先调那个会炸的工具", "boom"),
            _final("工具坏了,直接回答", "系统繁忙,请稍后再试。"),
        ]
    )
    # 在默认 JD 工具之外再挂一个内联的 boom 工具(可插拔:agent.py 不用改)
    agent = ReActAgent(llm, tools=[*JD_MOCK_TOOLS, _BoomTool()])

    result = agent.run("触发异常")

    assert result.stopped_reason == "final_answer"
    assert len(result.steps) == 2
    assert result.steps[0].observation is not None
    assert "工具执行失败" in result.steps[0].observation
    assert "炸了" in result.steps[0].observation
    assert result.steps[1].final_answer == "系统繁忙,请稍后再试。"


# --------------------------------------------------------------------------- #
# 7. 撞 max_steps:永远只输出 action,触发强制作答                               #
# --------------------------------------------------------------------------- #
def test_max_steps_force_final() -> None:
    """模型永不收尾时,撞 max_steps 应触发强制作答且 stopped_reason=max_steps。"""
    llm = MockLLM(
        [
            _action("第一步查订单", "query_order", order_id="12345"),
            _action("第二步再查一次", "query_order", order_id="12345"),
            # 第三次 chat 是 _force_final_answer 的强制作答调用
            "根据现有信息,您的订单已发货。",
        ]
    )
    agent = ReActAgent(llm, tools=JD_MOCK_TOOLS, max_steps=2)

    result = agent.run("我的订单到哪了?")

    assert result.stopped_reason == "max_steps"
    # 强制作答的文本非空(直接取自 chat 文本,不经过 JSON 解析)
    assert result.final_answer == "根据现有信息,您的订单已发货。"
    assert result.final_answer != ""
    # 循环内跑满 max_steps 步(都是 action 步),强制作答不再计入 steps 轨迹
    assert len(result.steps) == 2
    # 共发生 3 次 chat:2 步循环 + 1 次强制作答
    assert llm.call_count == 3


# --------------------------------------------------------------------------- #
# 8. parse_step 单元测试                                                       #
# --------------------------------------------------------------------------- #
def test_parse_step_valid_plain_json() -> None:
    """合法纯 JSON(final_answer)应解析成功。"""
    step = parse_step('{"thought":"好的","final_answer":"您好"}')
    assert step.thought == "好的"
    assert step.final_answer == "您好"
    assert step.action is None


def test_parse_step_valid_action() -> None:
    """合法 action JSON 应解析出工具名与参数。"""
    step = parse_step(
        '{"thought":"查订单","action":{"tool":"query_order","input":{"order_id":"12345"}}}'
    )
    assert step.action is not None
    assert step.action.tool == "query_order"
    assert step.action.input == {"order_id": "12345"}
    assert step.final_answer is None


def test_parse_step_json_fence() -> None:
    """带 ```json 代码围栏的合法 JSON 应能去围栏后解析。"""
    raw = (
        "```json\n"
        '{"thought":"查订单","action":{"tool":"query_order","input":{"order_id":"12345"}}}'
        "\n```"
    )
    step = parse_step(raw)
    assert step.action is not None
    assert step.action.tool == "query_order"


def test_parse_step_invalid_json_raises() -> None:
    """非法 JSON 应抛 StepParseError。"""
    with pytest.raises(StepParseError):
        parse_step("这根本不是 JSON")


def test_parse_step_both_fields_raises() -> None:
    """同时给 action 和 final_answer 应抛 StepParseError。"""
    raw = (
        '{"thought":"矛盾输出","action":{"tool":"query_order","input":{}},'
        '"final_answer":"也想直接答"}'
    )
    with pytest.raises(StepParseError):
        parse_step(raw)


def test_parse_step_neither_field_raises() -> None:
    """action 和 final_answer 都不给应抛 StepParseError。"""
    with pytest.raises(StepParseError):
        parse_step('{"thought":"啥也不给"}')


# --------------------------------------------------------------------------- #
# 补充:StepTrace 结构完整性(解析失败步的字段)                                 #
# --------------------------------------------------------------------------- #
def test_step_trace_error_fields_populated() -> None:
    """解析失败步的 StepTrace 应同时带 error 与 raw、其余业务字段为空。"""
    llm = MockLLM(["不是 JSON", _final("收尾", "好的")])
    agent = ReActAgent(llm, tools=JD_MOCK_TOOLS)

    result = agent.run("测试")

    err_step: StepTrace = result.steps[0]
    assert err_step.error is not None
    assert err_step.raw == "不是 JSON"
    assert err_step.thought is None
    assert err_step.action is None
    assert err_step.observation is None
