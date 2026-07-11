"""LLM 接口、通用数据结构与 provider 工厂。

本模块只定义框架对「大模型」的**厂商无关**抽象:

- ``Message`` / ``Usage`` / ``ChatResponse`` / ``ToolCall``：通用数据结构,不绑定任何厂商。
- ``LLM``：所有厂商实现都要满足的 :class:`typing.Protocol` 接口。
- ``create_llm``：按配置(``settings.provider``)挑选并构造具体实现的工厂。

设计立场(见 stage-1-design.md §4):核心代码只依赖 ``LLM`` 接口与本模块的通用
类型,**任何厂商 SDK 都不在此出现** —— ``anthropic`` 只在 ``llm_claude.py``、
``openai`` 只在 ``llm_openai.py``。换模型 = 换实现(甚至只改 ``.env``),核心不动。

阶段三 P-B 扩展(见 stage-3-design.md):原生 Function Calling。``chat()`` 可传
``tools=``(厂商无关 Schema,来自 ``ToolRegistry.to_schemas()``),应答里的
``tool_calls`` 是模型要求的工具调用;``Message`` 相应新增 ``assistant`` 消息携带
``tool_calls``、``tool`` 角色回传工具结果两种形态。厂商差异(Claude ``tool_use``
block / OpenAI ``tool_calls``)全部封装在各自实现里。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Literal, Protocol, Sequence

if TYPE_CHECKING:
    from agent_framework.core.config import Settings

Role = Literal["user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    """模型发起的一次工具调用请求(厂商无关)。

    Attributes:
        id: 本次调用的唯一 id(厂商生成);回传结果时用它配对。
        name: 要调用的工具名。
        args: 模型生成的参数字典(已从 JSON 解析)。
    """

    id: str
    name: str
    args: dict[str, object]


@dataclass(frozen=True)
class Message:
    """一条对话消息(通用,不绑定任何厂商)。

    三种形态:

    - ``user``:用户输入,只用 ``content``;
    - ``assistant``:模型输出;若模型要求调工具,``tool_calls`` 非空
      (此时 ``content`` 可为空字符串);
    - ``tool``:一条工具执行结果,``content`` 是 Observation 文本,
      ``tool_call_id`` 指回对应的 :class:`ToolCall`。

    system prompt 仍作为 :meth:`LLM.chat` / :meth:`LLM.stream` 的独立参数传入,
    不放进消息列表。
    """

    role: Role
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True)
class Usage:
    """token 用量,用于成本估算与日志。"""

    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        """输入与输出 token 之和。"""
        return self.input_tokens + self.output_tokens


@dataclass
class ChatResponse:
    """一次非流式应答的统一返回。"""

    content: str  # 拼接后的纯文本回复
    usage: Usage  # token 用量
    model: str  # 实际使用的模型 id
    stop_reason: str | None = None
    # 模型要求的工具调用(未传 tools 或模型直接作答时为空列表)。
    tool_calls: list[ToolCall] = field(default_factory=list)
    # 逃生舱:原始 SDK 响应,调试用(不参与 repr)。
    raw: object | None = field(default=None, repr=False)


class LLM(Protocol):
    """所有 LLM 厂商实现都要满足的接口。

    核心只依赖它,不依赖任何具体 SDK。方法签名里**不出现**
    ``temperature`` / ``max_tokens`` / ``model`` 等参数 —— 这些属于
    「实现 + 配置」职责,由实现的构造函数从 :class:`Settings` 注入。
    """

    model: str  # 实现解析后实际使用的模型 id(供 CLI / 日志展示)

    def chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: Sequence[dict[str, object]] | None = None,
    ) -> ChatResponse:
        """一次性返回完整应答。

        Args:
            messages: 对话消息(可含 ``tool_calls`` / ``tool`` 形态,见 :class:`Message`)。
            system: system prompt。
            tools: 厂商无关的工具 Schema 列表(``ToolRegistry.to_schemas()`` 的输出);
                传入后模型可在应答里返回 ``tool_calls`` 要求调用工具。
        """
        ...

    def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
    ) -> Iterator[str]:
        """流式返回文本增量(一段段 yield),用于 CLI 实时打印。

        流式暂不支持 ``tools``(阶段三约定:工具调用走非流式 ``chat``;
        流式工具事件留到阶段六生产化再议)。
        """
        ...


def create_llm(settings: Settings) -> LLM:
    """按 ``settings.provider`` 构造具体 LLM 实现,并套上可靠性包装(阶段六)。

    这是框架唯一的「装配点」:上层(CLI / 实验 / 未来的 Agent)只调用本函数拿到
    一个 ``LLM``,不关心背后是哪家厂商。具体 provider 的 SDK 在此**延迟导入**,
    因此只用其中一家的用户无需安装另一家的 SDK。

    装配结构(自内向外):裸 provider → ``ReliableLLM``(指数退避重试)→
    若配置了 ``fallback_provider`` 再套 ``FallbackLLM``(主挂切备)。
    上层拿到的永远是同一个 ``LLM`` 接口,可靠性对其不可见。

    Args:
        settings: 框架配置;``provider`` 决定厂商,其余字段注入具体实现。

    Returns:
        满足 :class:`LLM` 接口的实例(已带重试;可能带降级)。

    Raises:
        ValueError: ``provider`` 不在支持列表内。
    """
    from agent_framework.core.llm_reliable import FallbackLLM, ReliableLLM

    primary: LLM = ReliableLLM(
        _create_provider(settings, settings.provider), max_retries=settings.llm_max_retries
    )
    if settings.fallback_provider:
        secondary: LLM = ReliableLLM(
            _create_provider(settings, settings.fallback_provider),
            max_retries=settings.llm_max_retries,
        )
        return FallbackLLM(primary, secondary)
    return primary


def _create_provider(settings: Settings, provider_name: str) -> LLM:
    """构造某一家厂商的裸实现(SDK 延迟导入;可靠性包装在 ``create_llm`` 统一加)。"""
    provider = provider_name.lower()
    if provider in ("claude", "anthropic"):
        from agent_framework.core.llm_claude import ClaudeLLM

        return ClaudeLLM(settings)
    if provider in ("openai", "chatgpt", "gpt"):
        from agent_framework.core.llm_openai import OpenAILLM

        return OpenAILLM(settings)
    raise ValueError(f"未知的 LLM provider:{provider_name!r}。当前支持:claude / openai。")
