"""MemoryManager:三件套(短期/长期/压缩)的统一门面(阶段四 P-C)。

Agent 循环对记忆零感知 —— 记忆发生在「组装上下文」这一层(与阶段二「外层干净
历史」同一位置)。调用方每轮只跟两个入口打交道::

    manager = create_memory_manager(settings, llm)      # 或手工注入三件套
    ctx = manager.load(user_id, question)               # ① 检索长期 + 取摘要 + 取窗口
    system = agent_base_prompt + ctx.system_suffix()    # ② 记忆作为 system 附加段
    result = agent.run(question, history=ctx.to_messages())
    manager.on_turn_end(user_id, turn)                  # ③ 窗口滚动→压缩;提炼→长期库

设计边界(stage-4-design.md §8):记忆属于**会话**(谁在说话),Agent 属于**能力**
(会干什么)—— 所以 memory 不塞进 Agent 构造参数,Agent 保持单纯。
三件套都可为 ``None``,缺哪个就退化掉哪个能力(如只配滑窗 = 纯短期记忆)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agent_framework.core.llm import LLM, Message
from agent_framework.memory.compressor import SummaryCompressor
from agent_framework.memory.long_term import LongTermMemory, ScoredMemory, WriteOp
from agent_framework.memory.short_term import ShortTermMemory, Turn

if TYPE_CHECKING:
    from agent_framework.core.config import Settings


@dataclass
class MemoryContext:
    """一轮对话开始前组装好的记忆上下文。

    Attributes:
        summary: 前情提要(被窗口挤出的旧轮的递归摘要);无则 None。
        retrieved: 长期记忆的三因子检索结果(带分量,可解释)。
        recent_turns: 短期窗口内的近期轮次(从旧到新)。
    """

    summary: str | None = None
    retrieved: list[ScoredMemory] = field(default_factory=list)
    recent_turns: list[Turn] = field(default_factory=list)

    def to_messages(self) -> list[Message]:
        """把近期窗口展开成消息列表,作为 ``agent.run(history=...)``。"""
        messages: list[Message] = []
        for turn in self.recent_turns:
            messages.extend(turn.to_messages())
        return messages

    def system_suffix(self) -> str:
        """摘要 + 长期记忆的 system prompt 附加段;两者皆无时返回空串。

        只注入、不指令 —— 事实标注「仅供参考」,由模型自行决定用不用,
        避免过期记忆压过工具的实时查询结果。
        """
        parts: list[str] = []
        if self.summary:
            parts.append(f"【前情提要(早前对话的摘要)】\n{self.summary}")
        if self.retrieved:
            lines = "\n".join(f"- {s.record.text}" for s in self.retrieved)
            parts.append(f"【关于该用户的已知信息(长期记忆,仅供参考)】\n{lines}")
        if not parts:
            return ""
        return "\n\n" + "\n\n".join(parts)


@dataclass(frozen=True)
class TurnReport:
    """``on_turn_end`` 的执行报告(CLI /trace 展示与测试断言用)。

    Attributes:
        evicted_turns: 本轮触发滑动窗口弹出的旧轮数量。
        summary_updated: 本轮是否重算了前情提要。
        write_ops: 长期记忆的写入决策执行结果(ADD/UPDATE/DELETE/NOOP)。
    """

    evicted_turns: int = 0
    summary_updated: bool = False
    write_ops: tuple[WriteOp, ...] = ()


class MemoryManager:
    """短期窗口 + 前情摘要 + 长期记忆的统一编排。"""

    def __init__(
        self,
        short_term: ShortTermMemory | None = None,
        long_term: LongTermMemory | None = None,
        compressor: SummaryCompressor | None = None,
    ) -> None:
        """注入三件套,均可为 ``None``(缺哪个退化哪个,组合自由)。

        Args:
            short_term: 滑动窗口;None 时不保留近期原文。
            long_term: 长期记忆;None 时不做跨会话检索与写入。
            compressor: 摘要压缩器;None 时被弹出的轮直接丢弃(只依赖长期记忆兜底)。
        """
        self._short_term = short_term
        self._long_term = long_term
        self._compressor = compressor
        self._summary: str | None = None

    @property
    def summary(self) -> str | None:
        """当前会话的前情提要(无则 None)。"""
        return self._summary

    def load(self, user_id: str, query: str) -> MemoryContext:
        """一轮开始前组装记忆上下文(长期检索 + 摘要 + 近期窗口)。

        Args:
            user_id: 当前用户(框架注入,模型不可见)。
            query: 用户本轮输入(作为长期记忆的检索 query)。
        """
        retrieved = self._long_term.retrieve(user_id, query) if self._long_term else []
        recent = self._short_term.window() if self._short_term else []
        return MemoryContext(summary=self._summary, retrieved=retrieved, recent_turns=recent)

    def on_turn_end(self, user_id: str, turn: Turn) -> TurnReport:
        """一轮结束后落账:窗口滚动 → 溢出压缩;LLM 提炼事实 → 长期库。

        Args:
            user_id: 当前用户。
            turn: 刚结束的一轮对话。

        Returns:
            本轮的记忆变更报告。
        """
        evicted: list[Turn] = []
        summary_updated = False
        if self._short_term is not None:
            evicted = self._short_term.add(turn)
            if evicted and self._compressor is not None:
                self._summary = self._compressor.compress(self._summary, evicted)
                summary_updated = True
        ops: list[WriteOp] = []
        if self._long_term is not None:
            ops = self._long_term.remember(user_id, turn)
        return TurnReport(
            evicted_turns=len(evicted),
            summary_updated=summary_updated,
            write_ops=tuple(ops),
        )

    def reset_session(self) -> None:
        """清空会话态(窗口 + 摘要);长期记忆不动 —— 跨会话记住的东西还在。"""
        if self._short_term is not None:
            self._short_term.clear()
        self._summary = None

    # ------------------------------------------------------- 长期记忆直通(CLI 用)

    @property
    def long_term(self) -> LongTermMemory | None:
        """暴露长期记忆组件,供 CLI 的 /memories、/forget 等手动通道使用。"""
        return self._long_term


def create_memory_manager(settings: Settings, llm: LLM) -> MemoryManager:
    """按配置装配全套 MemoryManager(唯一装配点,与 ``create_llm`` 同一纪律)。

    短期/压缩零外部依赖;长期记忆用 OpenAI embedding + Chroma 持久化
    (落盘 ``settings.memory_persist_dir``,跨会话可检索)。

    Args:
        settings: 框架配置(memory_* 系列字段)。
        llm: 摘要与事实提炼共用的 LLM(与主循环同一实例即可)。
    """
    from agent_framework.memory.embedder import OpenAIEmbedder
    from agent_framework.memory.vector_store import ChromaVectorStore

    short_term = ShortTermMemory(max_tokens=settings.memory_window_tokens)
    compressor = SummaryCompressor(llm, max_tokens=settings.memory_summary_max_tokens)
    long_term = LongTermMemory(
        llm,
        OpenAIEmbedder(settings),
        ChromaVectorStore(settings.memory_persist_dir),
        top_k=settings.memory_top_k,
        weight_relevance=settings.memory_weight_relevance,
        weight_recency=settings.memory_weight_recency,
        weight_importance=settings.memory_weight_importance,
        half_life_hours=settings.memory_half_life_hours,
        dedup_threshold=settings.memory_dedup_threshold,
    )
    return MemoryManager(short_term=short_term, long_term=long_term, compressor=compressor)
