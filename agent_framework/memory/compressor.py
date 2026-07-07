"""上下文压缩:被挤出窗口的旧轮 → 递归合并进「前情提要」(阶段四 P-A)。

对应大纲 4.3「混合策略:近期保留原文 + 早期使用摘要」+「渐进式摘要」:

- 每次只摘「旧摘要 + 新被弹出的轮」,不重摘全史,成本 O(1);
- 产出的摘要作为 system prompt 的附加段(「前情提要:…」)注入,不占对话消息;
- **压缩失败不能炸对话**:LLM 调用失败时降级为「旧摘要 + 弹出轮文本」的截断拼接,
  信息略糙但流程照走。

用注入的 :class:`~agent_framework.core.llm.LLM` 接口(与主循环同一个抽象),
测试用 MockLLM 零成本。
"""

from __future__ import annotations

from agent_framework.core.llm import LLM, Message
from agent_framework.memory.short_term import HeuristicTokenCounter, TokenCounter, Turn

DEFAULT_COMPRESS_SYSTEM = (
    "你是客服对话的记录员。把「旧前情提要」与「新增对话」合并为一段新的前情提要。\n"
    "必须保留:用户身份信息、订单号/单号、已承诺事项、未解决的问题。\n"
    "可以丢弃:寒暄、重复内容、已完结且无后续的琐事。\n"
    "只输出提要正文,不要任何前缀、解释或列表符号。"
)


class SummaryCompressor:
    """递归摘要压缩器:``新摘要 = LLM(旧摘要 + 被弹出的轮)``。

    Attributes:
        llm: 用来做摘要的 LLM(注入,与主循环共用抽象;测试给 MockLLM)。
        max_tokens: 摘要长度上限(估算值,写进指令并用于降级截断)。
    """

    def __init__(
        self,
        llm: LLM,
        *,
        max_tokens: int = 300,
        counter: TokenCounter | None = None,
        system_prompt: str = DEFAULT_COMPRESS_SYSTEM,
    ) -> None:
        """构造压缩器。

        Args:
            llm: 摘要用的 LLM 实现。
            max_tokens: 摘要的 token 上限(默认 300,来自 stage-4-design §5.2)。
            counter: token 计数(降级截断用);缺省 :class:`HeuristicTokenCounter`。
            system_prompt: 摘要指令,可替换(如换业务域)。
        """
        self._llm = llm
        self.max_tokens = max_tokens
        self._counter = counter or HeuristicTokenCounter()
        self._system = system_prompt

    def compress(self, old_summary: str | None, evicted_turns: list[Turn]) -> str:
        """把旧摘要与新弹出的轮合并为新摘要。**永不抛异常**。

        Args:
            old_summary: 现有的前情提要;首次压缩时为 ``None``。
            evicted_turns: 刚被滑动窗口挤出的轮(从旧到新)。

        Returns:
            新的前情提要文本;LLM 失败时返回降级的截断拼接(信息不丢,只是糙)。
        """
        if not evicted_turns:
            return old_summary or ""
        prompt = self._build_prompt(old_summary, evicted_turns)
        try:
            response = self._llm.chat(
                [Message(role="user", content=prompt)],
                system=self._system,
            )
            summary = response.content.strip()
            if summary:
                return summary
        except Exception:  # noqa: BLE001 — 压缩失败必须降级,不能炸对话流程
            pass
        return self._fallback(old_summary, evicted_turns)

    def _build_prompt(self, old_summary: str | None, evicted_turns: list[Turn]) -> str:
        rendered = "\n\n".join(turn.render_text() for turn in evicted_turns)
        return (
            f"旧前情提要:\n{old_summary or '(无)'}\n\n"
            f"新增对话:\n{rendered}\n\n"
            f"请输出合并后的新前情提要(不超过 {self.max_tokens} 字)。"
        )

    def _fallback(self, old_summary: str | None, evicted_turns: list[Turn]) -> str:
        """降级策略:旧摘要 + 弹出轮文本直接拼接,按 token 上限从头截断。"""
        parts = [old_summary] if old_summary else []
        parts.extend(turn.render_text() for turn in evicted_turns)
        text = "\n".join(parts)
        if self._counter.count(text) <= self.max_tokens:
            return text
        # 逐步截短到预算内(粗粒度即可,降级路径不追求精确)。
        while text and self._counter.count(text) > self.max_tokens:
            text = text[: max(len(text) - 50, 0)]
        return text
