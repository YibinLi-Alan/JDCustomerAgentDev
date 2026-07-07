"""短期记忆(滑动窗口 + Token 预算)的单元测试。全部离线。"""

from __future__ import annotations

import pytest

from agent_framework.core.llm import Message, ToolCall
from agent_framework.memory.short_term import HeuristicTokenCounter, ShortTermMemory, Turn


class CharCounter:
    """测试用计数器:1 字符 = 1 token,预算完全可预测。"""

    def count(self, text: str) -> int:
        return len(text)


def make_turn(user: str, assistant: str, *, with_tools: bool = False) -> Turn:
    inner: tuple[Message, ...] = ()
    if with_tools:
        call = ToolCall(id="c1", name="query_order", args={"order_id": "12345"})
        inner = (
            Message(role="assistant", content="", tool_calls=(call,)),
            Message(role="tool", content="已发货", tool_call_id="c1"),
        )
    return Turn(user_text=user, assistant_text=assistant, inner_messages=inner)


# ---------------------------------------------------------------- 计数器


def test_heuristic_counter_cjk_and_ascii() -> None:
    counter = HeuristicTokenCounter()
    assert counter.count("") == 0
    assert counter.count("你好") == 2  # 中文 1 字 1 token
    assert counter.count("abcd") == 1  # 英文 4 字符 1 token
    assert counter.count("abc") == 1  # 不足 4 字符向上取整
    assert counter.count("你好ab") == 3  # 混合:2 + ceil(2/4)


# ---------------------------------------------------------------- 窗口与预算


def test_within_budget_no_eviction() -> None:
    mem = ShortTermMemory(max_tokens=100, counter=CharCounter())
    evicted = mem.add(make_turn("你好", "您好"))
    assert evicted == []
    assert len(mem) == 1


def test_over_budget_evicts_oldest_first() -> None:
    # 每轮 render_text = "用户:第一轮问题\n客服:第一轮回答" = 17 字符;预算 40 装两轮。
    mem = ShortTermMemory(max_tokens=40, counter=CharCounter())
    t1 = make_turn("第一轮问题", "第一轮回答")
    t2 = make_turn("第二轮问题", "第二轮回答")
    t3 = make_turn("第三轮问题", "第三轮回答")
    assert mem.add(t1) == []
    assert mem.add(t2) == []
    evicted = mem.add(t3)  # 第三轮进来(51 > 40)→ 最旧的 t1 被弹出
    assert evicted == [t1]
    assert mem.window() == [t2, t3]


def test_window_order_is_oldest_to_newest() -> None:
    mem = ShortTermMemory(max_tokens=10_000, counter=CharCounter())
    turns = [make_turn(f"问{i}", f"答{i}") for i in range(3)]
    for t in turns:
        mem.add(t)
    assert mem.window() == turns


def test_oversize_single_turn_is_kept() -> None:
    """单轮超预算也不弹出最新轮 —— 当前对话的直接上下文必须保住。"""
    mem = ShortTermMemory(max_tokens=5, counter=CharCounter())
    big = make_turn("这是一个特别长的问题" * 3, "特别长的回答" * 3)
    evicted = mem.add(big)
    assert evicted == []
    assert mem.window() == [big]
    # 下一轮进来时,超预算的旧大轮要被弹出。
    small = make_turn("短", "短")
    evicted = mem.add(small)
    assert evicted == [big]
    assert mem.window() == [small]


# ---------------------------------------------------------------- 轮的原子性


def test_turn_with_tool_messages_evicted_atomically() -> None:
    """含工具往返的轮要么整轮在窗口、要么整轮弹出,tool 消息不落单。"""
    mem = ShortTermMemory(max_tokens=40, counter=CharCounter())
    tool_turn = make_turn("查订单12345", "已发货", with_tools=True)
    mem.add(tool_turn)
    evicted: list[Turn] = []
    for i in range(5):
        evicted.extend(mem.add(make_turn(f"后续问题{i}", f"后续回答{i}")))
    if tool_turn in evicted:
        # 弹出的轮完整携带自己的工具往返。
        assert len(tool_turn.inner_messages) == 2
    # 窗口展开的消息里,不允许出现"孤儿 tool 消息"(前面没有带 tool_calls 的 assistant)。
    messages = mem.to_messages()
    for i, msg in enumerate(messages):
        if msg.role == "tool":
            assert i > 0 and messages[i - 1].role in ("assistant", "tool")
            j = i - 1
            while messages[j].role == "tool":
                j -= 1
            assert messages[j].tool_calls  # 一定能追溯到发起调用的 assistant 消息


def test_to_messages_expands_turn_in_order() -> None:
    mem = ShortTermMemory(max_tokens=10_000, counter=CharCounter())
    mem.add(make_turn("查订单", "已发货", with_tools=True))
    roles = [m.role for m in mem.to_messages()]
    assert roles == ["user", "assistant", "tool", "assistant"]


# ---------------------------------------------------------------- 其它


def test_clear_resets_window() -> None:
    mem = ShortTermMemory(max_tokens=100, counter=CharCounter())
    mem.add(make_turn("问", "答"))
    mem.clear()
    assert len(mem) == 0
    assert mem.total_tokens == 0
    assert mem.to_messages() == []


def test_invalid_budget_rejected() -> None:
    with pytest.raises(ValueError):
        ShortTermMemory(max_tokens=0)
