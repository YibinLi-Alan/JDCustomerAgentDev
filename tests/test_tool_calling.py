"""阶段三 P-B 单元测试:ToolCallingAgent 循环 + 厂商协议转换(全部离线)。

覆盖:原生 Function Calling 循环(单工具 / 一步并行多工具 / 未知工具折叠 /
strict 校验失败喂回 / 步数上限强制作答),以及 Claude / OpenAI 两家的消息与
工具 Schema 转换纯函数(不实例化客户端、不碰网络)。
"""

from __future__ import annotations

from agent_framework.core.agent import ToolCallingAgent
from agent_framework.core.llm import ChatResponse, Message, ToolCall, Usage
from agent_framework.core.llm_claude import _to_anthropic_messages, _to_anthropic_tools
from agent_framework.core.llm_openai import (
    _qualifies_for_openai_strict,
    _to_openai_messages,
    _to_openai_tools,
)
from agent_framework.tools import ToolRegistry, tool
from tests.mock_llm import MockLLM


# --------------------------------------------------------------------------- #
# 测试用工具与脚本工厂                                                            #
# --------------------------------------------------------------------------- #
@tool
def echo(text: str, times: int = 1) -> str:
    """重复输出文本。何时用:测试。"""
    return text * times


def _registry() -> ToolRegistry:
    return ToolRegistry([echo])


def _tool_call_resp(*calls: ToolCall, content: str = "") -> ChatResponse:
    """构造一条带 ``tool_calls`` 的 mock 应答。"""
    return ChatResponse(
        content=content,
        usage=Usage(input_tokens=0, output_tokens=0),
        model="mock",
        tool_calls=list(calls),
    )


# --------------------------------------------------------------------------- #
# ToolCallingAgent 循环                                                          #
# --------------------------------------------------------------------------- #
def test_single_tool_call_then_answer():
    llm = MockLLM(
        [
            _tool_call_resp(
                ToolCall(id="c1", name="echo", args={"text": "嗨", "times": 2}),
                content="我需要先调工具",
            ),
            "查到了:嗨嗨",
        ]
    )
    result = ToolCallingAgent(llm, _registry()).run("测试")

    assert result.final_answer == "查到了:嗨嗨"
    assert result.stopped_reason == "final_answer"
    assert len(result.steps) == 2
    assert result.steps[0].action is not None
    assert result.steps[0].action.tool == "echo"
    assert result.steps[0].observation == "嗨嗨"
    assert result.steps[0].thought == "我需要先调工具"
    assert result.steps[1].final_answer == "查到了:嗨嗨"

    # 每步都把工具 Schema 下发给了模型
    assert llm.seen_tools[0] is not None
    assert llm.seen_tools[0][0]["name"] == "echo"
    # 第二次调用的上下文:assistant(带 tool_calls)+ tool 结果消息,id 正确配对
    msgs = llm.seen_messages[1]
    assert msgs[-2].role == "assistant" and msgs[-2].tool_calls[0].id == "c1"
    assert msgs[-1].role == "tool"
    assert msgs[-1].tool_call_id == "c1"
    assert msgs[-1].content == "嗨嗨"


def test_parallel_tool_calls_in_one_step():
    llm = MockLLM(
        [
            _tool_call_resp(
                ToolCall(id="a", name="echo", args={"text": "1"}),
                ToolCall(id="b", name="echo", args={"text": "2"}),
            ),
            "两个都查到了",
        ]
    )
    result = ToolCallingAgent(llm, _registry()).run("并行")

    assert result.stopped_reason == "final_answer"
    # 一步两个并行调用 → 两条工具轨迹 + 一条收尾
    assert [s.observation for s in result.steps[:2]] == ["1", "2"]
    # 上下文里:一条 assistant + 两条 tool(按调用顺序、id 配对)
    msgs = llm.seen_messages[1]
    assert [m.role for m in msgs[-3:]] == ["assistant", "tool", "tool"]
    assert [m.tool_call_id for m in msgs[-2:]] == ["a", "b"]


def test_unknown_tool_folded_and_loop_continues():
    llm = MockLLM(
        [
            _tool_call_resp(ToolCall(id="x", name="nope", args={})),
            "抱歉,我换个方式回答",
        ]
    )
    result = ToolCallingAgent(llm, _registry()).run("未知工具")

    assert result.stopped_reason == "final_answer"
    assert "不存在" in (result.steps[0].observation or "")
    assert "echo" in (result.steps[0].observation or "")  # 报错里列出可用工具


def test_strict_validation_error_fed_back_as_observation():
    llm = MockLLM(
        [
            # strict 下 "2"(字符串)不允许强转成 int → 校验失败喂回
            _tool_call_resp(ToolCall(id="x", name="echo", args={"text": "a", "times": "2"})),
            "参数写错了,我修正后再答",
        ]
    )
    result = ToolCallingAgent(llm, _registry()).run("严格校验")

    observation = result.steps[0].observation or ""
    assert observation.startswith("[工具执行失败]")
    assert "times" in observation


def test_max_steps_forces_final_answer_without_tools():
    call = ToolCall(id="l", name="echo", args={"text": "x"})
    llm = MockLLM(
        [
            _tool_call_resp(call),
            _tool_call_resp(call),
            "根据已有结果,答案是 x",  # 强制作答那次
        ]
    )
    result = ToolCallingAgent(llm, _registry(), max_steps=2).run("死循环")

    assert result.stopped_reason == "max_steps"
    assert result.final_answer == "根据已有结果,答案是 x"
    assert llm.call_count == 3
    # 强制作答那次不再下发工具,杜绝继续调用
    assert llm.seen_tools[-1] is None
    # 且追加了「不要再调用工具」的用户指令
    assert "不要再调用工具" in llm.seen_messages[-1][-1].content


# --------------------------------------------------------------------------- #
# Claude(Anthropic)协议转换                                                    #
# --------------------------------------------------------------------------- #
def test_to_anthropic_tools_uses_input_schema():
    schemas = _registry().to_schemas()
    converted = _to_anthropic_tools(schemas)
    assert converted[0]["name"] == "echo"
    assert converted[0]["input_schema"] == schemas[0]["parameters"]
    assert "parameters" not in converted[0]


def test_to_anthropic_messages_blocks_and_merging():
    msgs = [
        Message("user", "你好"),
        Message("assistant", "想一下", tool_calls=(ToolCall("a", "echo", {"x": 1}),)),
        Message("tool", "结果1", tool_call_id="a"),
        Message("tool", "结果2", tool_call_id="b"),
        Message("assistant", "答案"),
    ]
    out = _to_anthropic_messages(msgs)

    assert out[0] == {"role": "user", "content": "你好"}
    # assistant 带 tool_calls → text block + tool_use block
    blocks = out[1]["content"]
    assert out[1]["role"] == "assistant"
    assert blocks[0] == {"type": "text", "text": "想一下"}
    assert blocks[1] == {"type": "tool_use", "id": "a", "name": "echo", "input": {"x": 1}}
    # 连续两条 tool 消息合并成一条 user 消息(两个 tool_result block)
    assert out[2]["role"] == "user"
    results = out[2]["content"]
    assert [b["type"] for b in results] == ["tool_result", "tool_result"]
    assert [b["tool_use_id"] for b in results] == ["a", "b"]
    # 之后的普通 assistant 不受影响,总条数正确(5 → 4)
    assert out[3] == {"role": "assistant", "content": "答案"}
    assert len(out) == 4


def test_to_anthropic_messages_assistant_without_text_has_no_empty_block():
    out = _to_anthropic_messages(
        [Message("assistant", "", tool_calls=(ToolCall("a", "echo", {}),))]
    )
    assert [b["type"] for b in out[0]["content"]] == ["tool_use"]


# --------------------------------------------------------------------------- #
# OpenAI 协议转换                                                                #
# --------------------------------------------------------------------------- #
def test_to_openai_messages_tool_calls_and_results():
    msgs = [
        Message("user", "你好"),
        Message("assistant", "", tool_calls=(ToolCall("a", "echo", {"text": "嗨"}),)),
        Message("tool", "嗨", tool_call_id="a"),
    ]
    out = _to_openai_messages(msgs, system="sys")

    assert out[0] == {"role": "system", "content": "sys"}
    assert out[1] == {"role": "user", "content": "你好"}
    assistant = out[2]
    assert assistant["content"] is None  # 空文本 → None(OpenAI 惯例)
    tc = assistant["tool_calls"][0]
    assert tc["id"] == "a" and tc["type"] == "function"
    assert tc["function"]["name"] == "echo"
    assert tc["function"]["arguments"] == '{"text": "嗨"}'  # 参数序列化为 JSON 字符串
    assert out[3] == {"role": "tool", "tool_call_id": "a", "content": "嗨"}


def test_openai_strict_flag_only_when_schema_qualifies():
    qualifying = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    assert _qualifies_for_openai_strict(qualifying)
    # 有可选参数(required ≠ properties)→ 不满足 OpenAI 严格模式前提
    optionalized = dict(qualifying, properties={"a": {}, "b": {}})
    assert not _qualifies_for_openai_strict(optionalized)
    # 宽松工具(无 additionalProperties: false)→ 不满足
    assert not _qualifies_for_openai_strict({"type": "object", "properties": {}})

    tools = _to_openai_tools(
        [
            {"name": "t1", "description": "d", "parameters": qualifying},
            {"name": "t2", "description": "d", "parameters": optionalized},
        ]
    )
    assert tools[0]["function"]["strict"] is True
    assert "strict" not in tools[1]["function"]
    assert tools[0]["type"] == "function"
