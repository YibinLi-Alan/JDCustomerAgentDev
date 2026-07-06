"""可复用的假 LLM(MockLLM)—— 单测的测试基础设施。

`MockLLM` 满足 :class:`agent_framework.core.llm.LLM` 协议,但**不接任何真实
API / 网络**:构造时传入一串预设「回复脚本」,每调用一次 :meth:`chat`
就按顺序返回下一条。

脚本条目两种写法:

- ``str``:包成纯文本 :class:`~agent_framework.core.llm.ChatResponse`
  (阶段二 ReActAgent 的 JSON 文本步);
- ``ChatResponse``:原样返回 —— 用于阶段三 ``ToolCallingAgent``,
  可携带 ``tool_calls`` 模拟原生 Function Calling 应答。

它的三个价值:

- **离线**:零网络依赖,不 import 任何厂商 SDK;
- **确定性**:脚本写死,循环走哪条分支完全可预测,断言稳定;
- **零成本**:不烧 token,CI 可无限跑。

用法::

    llm = MockLLM(["{...第1步JSON...}", "{...第2步JSON...}"])
    agent = ReActAgent(llm, tools=JD_MOCK_TOOLS)

    llm = MockLLM([
        ChatResponse(content="", usage=Usage(0, 0), model="mock",
                     tool_calls=[ToolCall(id="1", name="echo", args={"text": "hi"})]),
        "好的,已经查到了。",
    ])
    agent = ToolCallingAgent(llm, registry)
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from agent_framework.core.llm import ChatResponse, Message, Usage


class MockLLM:
    """按预设脚本依次应答的假 LLM,满足 ``LLM`` 协议。

    Attributes:
        model: 模型 id,固定为 ``"mock"``,供日志 / CLI 展示。
        seen_tools: 每次 ``chat`` 收到的 ``tools`` 参数(按调用顺序),
            供测试断言「工具 Schema 是否被下发 / 何时不下发」。
        seen_messages: 每次 ``chat`` 收到的消息列表快照(按调用顺序),
            供测试断言上下文的组装(如 tool 消息是否正确回传)。
    """

    model: str = "mock"

    def __init__(self, responses: list[str | ChatResponse]) -> None:
        """用一串预设回复脚本构造。

        Args:
            responses: 每条是一次 ``chat`` 应答 —— ``str`` 包成纯文本应答,
                ``ChatResponse`` 原样返回(可带 ``tool_calls``)。
                第 N 次 ``chat`` 调用返回第 N 条;脚本要给足够多条,
                否则耗尽后再调用会抛 :class:`AssertionError`(帮助暴露脚本写少了)。
        """
        self._responses = list(responses)
        self._cursor = 0
        self.seen_tools: list[Sequence[dict[str, object]] | None] = []
        self.seen_messages: list[list[Message]] = []

    @property
    def call_count(self) -> int:
        """已发生的 ``chat`` 调用次数(供测试断言循环步数用)。"""
        return self._cursor

    def chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: Sequence[dict[str, object]] | None = None,
    ) -> ChatResponse:
        """按顺序返回脚本里的下一条回复,并记录收到的入参供断言。

        Args:
            messages: 上下文消息列表(记录快照,不消费内容)。
            system: system prompt(Mock 忽略,仅为符合接口签名)。
            tools: 工具 Schema 列表(记录,不消费)。

        Returns:
            脚本下一条:``str`` 包成 ``usage`` 全 0 的纯文本应答;
            ``ChatResponse`` 原样返回。

        Raises:
            AssertionError: 脚本已耗尽仍被调用(说明预设回复条数不够)。
        """
        assert self._cursor < len(self._responses), (
            f"MockLLM 脚本已耗尽:已被调用 {self._cursor + 1} 次,"
            f"但只预设了 {len(self._responses)} 条回复。"
        )
        self.seen_tools.append(tools)
        self.seen_messages.append(list(messages))
        item = self._responses[self._cursor]
        self._cursor += 1
        if isinstance(item, ChatResponse):
            return item
        return ChatResponse(
            content=item,
            usage=Usage(input_tokens=0, output_tokens=0),
            model=self.model,
        )

    def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
    ) -> Iterator[str]:
        """流式版本:把 :meth:`chat` 的整段结果一次性 yield 出去。

        Args:
            messages: 上下文消息列表(Mock 不消费)。
            system: system prompt(Mock 忽略)。

        Yields:
            脚本下一条回复的完整文本(只 yield 一段,足够满足接口)。
        """
        yield self.chat(messages, system=system).content
