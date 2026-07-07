"""MemoryManager(统一门面)的集成测试。全部离线。"""

from __future__ import annotations

import json
from datetime import datetime

from agent_framework.memory.compressor import SummaryCompressor
from agent_framework.memory.long_term import LongTermMemory
from agent_framework.memory.manager import MemoryContext, MemoryManager
from agent_framework.memory.short_term import ShortTermMemory, Turn
from agent_framework.memory.vector_store import InMemoryVectorStore
from tests.mock_embedder import MockEmbedder
from tests.mock_llm import MockLLM

NOW = datetime(2026, 7, 7, 12, 0, 0)


class CharCounter:
    def count(self, text: str) -> int:
        return len(text)


def fact(text: str, importance: int = 5) -> str:
    return json.dumps([{"fact": text, "importance": importance}], ensure_ascii=False)


def make_manager(
    *,
    window_tokens: int = 10_000,
    extract_responses: list[str] | None = None,
    compress_responses: list[str] | None = None,
    store: InMemoryVectorStore | None = None,
    embedder: MockEmbedder | None = None,
) -> MemoryManager:
    """短期/长期/压缩各用独立 MockLLM,脚本互不干扰。"""
    short_term = ShortTermMemory(max_tokens=window_tokens, counter=CharCounter())
    compressor = SummaryCompressor(MockLLM(compress_responses or []), max_tokens=300)
    long_term = LongTermMemory(
        MockLLM(extract_responses or []),
        embedder or MockEmbedder(),
        store if store is not None else InMemoryVectorStore(),
        now_fn=lambda: NOW,
    )
    return MemoryManager(short_term=short_term, long_term=long_term, compressor=compressor)


# ================================================================ load / 上下文组装


def test_load_assembles_summary_retrieval_and_window() -> None:
    store = InMemoryVectorStore()
    embedder = MockEmbedder(mapping={"用户地址在上海": [1.0, 0.0], "地址是哪": [1.0, 0.0]})
    manager = make_manager(
        extract_responses=[fact("用户地址在上海", 8), "[]"],
        compress_responses=[],
        store=store,
        embedder=embedder,
    )
    # 第一轮:写入一条长期记忆 + 进窗口
    manager.on_turn_end("alice", Turn(user_text="我地址是上海", assistant_text="已记录"))
    # 第二轮开始前 load
    ctx = manager.load("alice", "地址是哪")
    assert [s.record.text for s in ctx.retrieved] == ["用户地址在上海"]
    assert len(ctx.recent_turns) == 1
    suffix = ctx.system_suffix()
    assert "用户地址在上海" in suffix and "已知信息" in suffix
    assert "前情提要" not in suffix  # 还没有发生压缩
    roles = [m.role for m in ctx.to_messages()]
    assert roles == ["user", "assistant"]


def test_eviction_triggers_compressor_and_summary_appears_in_load() -> None:
    manager = make_manager(
        window_tokens=40,  # 每轮约 17 字符,第三轮触发弹出
        extract_responses=["[]", "[]", "[]"],
        compress_responses=["用户此前咨询过第一轮问题。"],
    )
    manager.on_turn_end("alice", Turn(user_text="第一轮问题", assistant_text="第一轮回答"))
    manager.on_turn_end("alice", Turn(user_text="第二轮问题", assistant_text="第二轮回答"))
    report = manager.on_turn_end("alice", Turn(user_text="第三轮问题", assistant_text="第三轮回答"))
    assert report.evicted_turns == 1
    assert report.summary_updated is True
    ctx = manager.load("alice", "随便问")
    assert ctx.summary == "用户此前咨询过第一轮问题。"
    assert "前情提要" in ctx.system_suffix()
    assert len(ctx.recent_turns) == 2  # 窗口里剩两轮


def test_turn_report_passes_through_write_ops() -> None:
    manager = make_manager(extract_responses=[fact("用户是学生", 6)])
    report = manager.on_turn_end("alice", Turn(user_text="我是学生", assistant_text="好的"))
    assert [op.action for op in report.write_ops] == ["ADD"]


# ================================================================ 缺省组合


def test_manager_with_nothing_configured_is_a_noop() -> None:
    manager = MemoryManager()
    report = manager.on_turn_end("alice", Turn(user_text="问", assistant_text="答"))
    assert report == (report.__class__())  # 全默认:没弹出、没摘要、没写入
    ctx = manager.load("alice", "问")
    assert ctx == MemoryContext()
    assert ctx.system_suffix() == ""
    manager.reset_session()  # 不炸即可


def test_manager_short_term_only() -> None:
    manager = MemoryManager(short_term=ShortTermMemory(max_tokens=10_000, counter=CharCounter()))
    manager.on_turn_end("alice", Turn(user_text="问", assistant_text="答"))
    ctx = manager.load("alice", "再问")
    assert len(ctx.recent_turns) == 1
    assert ctx.retrieved == [] and ctx.summary is None


def test_evicted_turns_dropped_silently_without_compressor() -> None:
    manager = MemoryManager(short_term=ShortTermMemory(max_tokens=40, counter=CharCounter()))
    for i in range(4):
        turn = Turn(user_text=f"第{i}轮问题", assistant_text=f"第{i}轮回答")
        report = manager.on_turn_end("alice", turn)
    assert report.evicted_turns >= 1
    assert report.summary_updated is False  # 没配压缩器:只弹出,不摘要


# ================================================================ 会话与跨会话


def test_reset_session_clears_window_and_summary_but_not_long_term() -> None:
    store = InMemoryVectorStore()
    embedder = MockEmbedder(mapping={"用户地址在上海": [1.0, 0.0], "地址是哪": [1.0, 0.0]})
    manager = make_manager(
        window_tokens=40,
        extract_responses=[fact("用户地址在上海", 8), "[]", "[]"],
        compress_responses=["旧摘要内容"],
        store=store,
        embedder=embedder,
    )
    manager.on_turn_end("alice", Turn(user_text="我地址是上海", assistant_text="已记录"))
    manager.on_turn_end("alice", Turn(user_text="第二轮问题", assistant_text="第二轮回答"))
    manager.on_turn_end("alice", Turn(user_text="第三轮问题", assistant_text="第三轮回答"))
    assert manager.summary is not None
    manager.reset_session()
    ctx = manager.load("alice", "地址是哪")
    assert ctx.recent_turns == [] and ctx.summary is None  # 会话态清空
    assert [s.record.text for s in ctx.retrieved] == ["用户地址在上海"]  # 长期记忆还在


def test_cross_session_retrieval_via_shared_store() -> None:
    """模拟「跨会话引用」:新的 manager 实例(同一 store)仍能检索到旧会话写入的事实。"""
    store = InMemoryVectorStore()
    embedder = MockEmbedder(mapping={"用户地址在上海": [1.0, 0.0], "寄到哪": [1.0, 0.0]})
    session1 = make_manager(
        extract_responses=[fact("用户地址在上海", 8)], store=store, embedder=embedder
    )
    session1.on_turn_end("alice", Turn(user_text="我地址是上海", assistant_text="已记录"))

    session2 = make_manager(extract_responses=[], store=store, embedder=embedder)  # 新会话
    ctx = session2.load("alice", "寄到哪")
    assert [s.record.text for s in ctx.retrieved] == ["用户地址在上海"]
    assert ctx.recent_turns == []  # 窗口是新的,只有长期记忆穿越了会话
