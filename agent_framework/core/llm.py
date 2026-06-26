"""LLM 接口、通用数据结构与 provider 工厂。

本模块只定义框架对「大模型」的**厂商无关**抽象:

- ``Message`` / ``Usage`` / ``ChatResponse``：通用数据结构,不绑定任何厂商。
- ``LLM``：所有厂商实现都要满足的 :class:`typing.Protocol` 接口。
- ``create_llm``：按配置(``settings.provider``)挑选并构造具体实现的工厂。

设计立场(见 stage-1-design.md §4):核心代码只依赖 ``LLM`` 接口与本模块的通用
类型,**任何厂商 SDK 都不在此出现** —— ``anthropic`` 只在 ``llm_claude.py``、
``openai`` 只在 ``llm_openai.py``。换模型 = 换实现(甚至只改 ``.env``),核心不动。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Literal, Protocol

if TYPE_CHECKING:
    from agent_framework.core.config import Settings

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class Message:
    """一条对话消息(通用,不绑定任何厂商)。

    仅承载 ``user`` / ``assistant`` 两种角色;system prompt 作为
    :meth:`LLM.chat` / :meth:`LLM.stream` 的独立参数传入,不放进消息列表。
    """

    role: Role
    content: str


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
    ) -> ChatResponse:
        """一次性返回完整应答。"""
        ...

    def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
    ) -> Iterator[str]:
        """流式返回文本增量(一段段 yield),用于 CLI 实时打印。"""
        ...


def create_llm(settings: Settings) -> LLM:
    """按 ``settings.provider`` 构造具体 LLM 实现。

    这是框架唯一的「装配点」:上层(CLI / 实验 / 未来的 Agent)只调用本函数拿到
    一个 ``LLM``,不关心背后是哪家厂商。具体 provider 的 SDK 在此**延迟导入**,
    因此只用其中一家的用户无需安装另一家的 SDK。

    Args:
        settings: 框架配置;``provider`` 决定厂商,其余字段注入具体实现。

    Returns:
        满足 :class:`LLM` 接口的具体实现实例。

    Raises:
        ValueError: ``provider`` 不在支持列表内。
    """
    provider = settings.provider.lower()
    if provider in ("claude", "anthropic"):
        from agent_framework.core.llm_claude import ClaudeLLM

        return ClaudeLLM(settings)
    if provider in ("openai", "chatgpt", "gpt"):
        from agent_framework.core.llm_openai import OpenAILLM

        return OpenAILLM(settings)
    raise ValueError(f"未知的 LLM provider:{settings.provider!r}。当前支持:claude / openai。")
