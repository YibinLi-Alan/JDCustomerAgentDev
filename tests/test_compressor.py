"""压缩器(递归摘要)的单元测试。全部离线(MockLLM)。"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from agent_framework.core.llm import ChatResponse, Message
from agent_framework.memory.compressor import SummaryCompressor
from agent_framework.memory.short_term import Turn
from tests.mock_llm import MockLLM


class ExplodingLLM:
    """一调用就抛异常的假 LLM,用来验证压缩失败的降级路径。"""

    model = "exploding"

    def chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: Sequence[dict[str, object]] | None = None,
    ) -> ChatResponse:
        raise RuntimeError("API 挂了")

    def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
    ) -> Iterator[str]:
        raise RuntimeError("API 挂了")


def make_turn(user: str, assistant: str) -> Turn:
    return Turn(user_text=user, assistant_text=assistant)


def test_compress_merges_old_summary_and_evicted_turns() -> None:
    llm = MockLLM(["用户王先生咨询订单 12345,已告知发货;地址在上海。"])
    compressor = SummaryCompressor(llm)
    summary = compressor.compress(
        "用户王先生,地址在上海。",
        [make_turn("订单12345到哪了", "已发货,运单 SF123")],
    )
    assert "12345" in summary
    # LLM 收到的 prompt 里必须同时有旧摘要和新弹出的轮。
    prompt = llm.seen_messages[0][0].content
    assert "王先生" in prompt
    assert "订单12345到哪了" in prompt


def test_first_compress_without_old_summary() -> None:
    llm = MockLLM(["用户询问过退货政策。"])
    compressor = SummaryCompressor(llm)
    summary = compressor.compress(None, [make_turn("怎么退货", "7 天无理由")])
    assert summary == "用户询问过退货政策。"
    assert "(无)" in llm.seen_messages[0][0].content  # 首次压缩,旧摘要标记为空


def test_empty_evicted_turns_is_noop_without_llm_call() -> None:
    llm = MockLLM([])  # 脚本为空:一旦被调用就会 AssertionError
    compressor = SummaryCompressor(llm)
    assert compressor.compress("旧摘要", []) == "旧摘要"
    assert compressor.compress(None, []) == ""


def test_llm_failure_falls_back_without_raising() -> None:
    compressor = SummaryCompressor(ExplodingLLM(), max_tokens=1000)
    summary = compressor.compress("旧摘要:用户是李女士", [make_turn("查订单", "已签收")])
    # 降级:不抛异常,旧摘要与新内容以拼接形式保留。
    assert "李女士" in summary
    assert "查订单" in summary


def test_fallback_truncates_to_budget() -> None:
    compressor = SummaryCompressor(ExplodingLLM(), max_tokens=20)
    long_turns = [make_turn("很长的问题" * 50, "很长的回答" * 50)]
    summary = compressor.compress(None, long_turns)
    # HeuristicTokenCounter 下,截断后应不超预算(留一次截断步长的余量)。
    from agent_framework.memory.short_term import HeuristicTokenCounter

    assert HeuristicTokenCounter().count(summary) <= 20


def test_blank_llm_output_falls_back() -> None:
    llm = MockLLM([""])  # 模型返回空串 → 视为无效摘要,走降级
    compressor = SummaryCompressor(llm, max_tokens=1000)
    summary = compressor.compress("旧摘要", [make_turn("问", "答")])
    assert "旧摘要" in summary
