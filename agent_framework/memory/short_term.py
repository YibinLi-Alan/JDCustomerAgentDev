"""短期记忆:滑动窗口 + Token 预算(阶段四 P-A)。

设计要点(见 stage-4-design.md §5):

- **以「轮」(:class:`Turn`)为原子单位**,不是以消息为单位。阶段三之后,一轮对话
  内部可能包含 ``assistant(tool_calls)`` 与 ``tool(tool_call_id)`` 消息,它们成对
  出现、不可拆散 —— 若窗口从中间切断,留下没有对应 ``tool_use`` 的 ``tool_result``,
  Claude 端直接 400。所以裁剪要么整轮保留、要么整轮弹出。
- **Token 预算**::class:`TokenCounter` 是 Protocol,默认 :class:`HeuristicTokenCounter`
  启发式估算(中日韩 1 字 ≈ 1 token、其余 4 字符 ≈ 1 token)。故意不引入厂商
  tokenizer(tiktoken 只对 OpenAI 准,对 Claude 本来就是估)—— 精度换零依赖 +
  厂商无关;要精确,换一个 ``TokenCounter`` 实现即可。
- **弹出的轮不丢**::meth:`ShortTermMemory.add` 把被挤出的旧轮返回给调用方,
  由压缩器(:mod:`agent_framework.memory.compressor`)摘要进「前情提要」。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from agent_framework.core.llm import Message


class TokenCounter(Protocol):
    """token 计数接口:估算一段文本折合多少 token。

    实现可以是启发式(默认)、也可以换成真 tokenizer —— 短期记忆只依赖本接口。
    """

    def count(self, text: str) -> int:
        """返回 ``text`` 估算的 token 数。"""
        ...


class HeuristicTokenCounter:
    """零依赖的启发式 token 估算:中日韩字符 1 字 ≈ 1 token,其余 4 字符 ≈ 1 token。

    对预算控制来说,估算的**一致性**比绝对精度重要:窗口只需要「大约装下多少轮」,
    差个百分之十几不影响机制正确性。
    """

    def count(self, text: str) -> int:
        """估算 ``text`` 的 token 数(至少为 0,空串为 0)。"""
        cjk = sum(1 for ch in text if self._is_cjk(ch))
        other = len(text) - cjk
        return cjk + (other + 3) // 4

    @staticmethod
    def _is_cjk(ch: str) -> bool:
        code = ord(ch)
        return (
            0x4E00 <= code <= 0x9FFF  # CJK 统一表意文字
            or 0x3400 <= code <= 0x4DBF  # 扩展 A
            or 0x3000 <= code <= 0x303F  # CJK 标点
            or 0xFF00 <= code <= 0xFFEF  # 全角字符
        )


@dataclass
class Turn:
    """一轮完整对话:用户问题 → (可选的工具往返)→ 最终回答。

    这是滑动窗口的原子单位:裁剪时整轮去留,``inner_messages`` 里成对的
    tool_use/tool_result 永远不会被拆散。

    Attributes:
        user_text: 用户这轮说了什么。
        assistant_text: 模型的最终回答。
        inner_messages: 中间的工具往返消息(``assistant(tool_calls)`` / ``tool``),
            可为空;随整轮一起保留或弹出。
        created_at: 本轮发生时间(供长期记忆的时近性因子用)。
    """

    user_text: str
    assistant_text: str
    inner_messages: tuple[Message, ...] = ()
    created_at: datetime = field(default_factory=datetime.now)

    def to_messages(self) -> list[Message]:
        """展开成消息列表:user → 工具往返 → 最终 assistant 回答。"""
        return [
            Message(role="user", content=self.user_text),
            *self.inner_messages,
            Message(role="assistant", content=self.assistant_text),
        ]

    def render_text(self) -> str:
        """渲染成给「token 计数 / 摘要压缩」看的纯文本(工具往返只计 content)。"""
        parts = [f"用户:{self.user_text}"]
        for msg in self.inner_messages:
            if msg.content:
                parts.append(f"[{msg.role}] {msg.content}")
        parts.append(f"客服:{self.assistant_text}")
        return "\n".join(parts)


class ShortTermMemory:
    """滑动窗口短期记忆:按 token 预算保留最近的完整轮次。

    超预算时从**最旧的轮**开始整轮弹出;弹出的轮由 :meth:`add` 返回,调用方
    (通常是 ``MemoryManager``)交给压缩器摘要,信息不凭空丢失。

    特殊规则:**最新一轮永不弹出** —— 即使单轮就超预算,也要保住当前对话的
    直接上下文,否则窗口为空、模型连刚才说了什么都不知道。
    """

    def __init__(self, max_tokens: int, counter: TokenCounter | None = None) -> None:
        """构造滑动窗口。

        Args:
            max_tokens: 窗口的 token 预算(所有轮 ``render_text`` 计数之和的上限)。
            counter: token 计数实现;缺省用 :class:`HeuristicTokenCounter`。
        """
        if max_tokens <= 0:
            raise ValueError(f"max_tokens 必须为正数,收到 {max_tokens!r}。")
        self.max_tokens = max_tokens
        self._counter = counter or HeuristicTokenCounter()
        self._turns: list[tuple[Turn, int]] = []  # (轮, 该轮 token 数)

    def add(self, turn: Turn) -> list[Turn]:
        """追加一轮,并把因超预算被挤出的旧轮返回(可能为空列表)。

        Args:
            turn: 刚结束的一轮对话。

        Returns:
            被弹出的旧轮(从最旧到次旧的顺序),交给压缩器;窗口未超预算时为空。
        """
        self._turns.append((turn, self._counter.count(turn.render_text())))
        evicted: list[Turn] = []
        # 最新一轮(刚 append 的)永不弹出,所以只在长度 > 1 时收缩。
        while self.total_tokens > self.max_tokens and len(self._turns) > 1:
            oldest, _ = self._turns.pop(0)
            evicted.append(oldest)
        return evicted

    def window(self) -> list[Turn]:
        """当前窗口内的轮次(从旧到新)。"""
        return [turn for turn, _ in self._turns]

    def to_messages(self) -> list[Message]:
        """把窗口内所有轮展开成消息列表,供组装上下文。"""
        messages: list[Message] = []
        for turn, _ in self._turns:
            messages.extend(turn.to_messages())
        return messages

    @property
    def total_tokens(self) -> int:
        """窗口内所有轮的 token 估算之和。"""
        return sum(tokens for _, tokens in self._turns)

    def clear(self) -> None:
        """清空窗口(新会话)。"""
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)
