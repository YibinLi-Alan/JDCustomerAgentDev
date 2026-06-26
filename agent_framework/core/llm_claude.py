"""Claude(Anthropic)LLM 实现。

``anthropic`` SDK **只允许出现在本文件里**(见 stage-1-design.md §4)。且采用
延迟导入:只有真正构造 :class:`ClaudeLLM` 时才 ``import anthropic``,这样只用
其他厂商(如 OpenAI)的用户无需安装 anthropic。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from agent_framework.core.llm import ChatResponse, Message, Usage

if TYPE_CHECKING:
    from agent_framework.core.config import Settings

# 未显式配置 MODEL 时,Claude provider 使用的默认模型。
DEFAULT_MODEL = "claude-opus-4-8"

# 已知「不接受 temperature / top_p / top_k 采样参数」的模型前缀。
# 命中这些前缀时,ClaudeLLM 不会下发 temperature(否则 Anthropic 返回 400)。
# 详见 stage-1-design.md §4.4。
_NO_SAMPLING_PARAM_MODEL_PREFIXES: tuple[str, ...] = (
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-fable-5",
    "claude-mythos-5",
    "claude-mythos-preview",
)


def _model_supports_temperature(model: str) -> bool:
    """判断目标模型是否接受 ``temperature`` 采样参数。

    用于「只把目标模型接受的参数下发」的能力过滤,避免换模型时踩 400。
    """
    return not model.startswith(_NO_SAMPLING_PARAM_MODEL_PREFIXES)


class ClaudeLLM:
    """基于 Anthropic 官方 SDK 的 :class:`LLM` 实现。

    结构化满足 ``LLM`` Protocol,无需显式继承。所有 Anthropic 特有的细节
    (system 放顶层、采样参数能力过滤、token 字段名)都封装在这里。
    """

    def __init__(self, settings: Settings) -> None:
        """用配置构造客户端,一次性绑定模型与采样参数。

        Args:
            settings: 框架配置,提供 api key / 模型名 / max_tokens / temperature。

        Raises:
            ValueError: 未配置 ``ANTHROPIC_API_KEY``。
        """
        import anthropic

        if not settings.anthropic_api_key:
            raise ValueError("使用 Claude 需在 .env 配置 ANTHROPIC_API_KEY。")
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.model or DEFAULT_MODEL
        self._max_tokens = settings.max_tokens
        self._temperature = settings.temperature

    def _build_request_kwargs(
        self,
        messages: list[Message],
        system: str | None,
    ) -> dict[str, object]:
        """把通用入参映射成 Anthropic ``messages.create`` 的关键字参数。

        - 通用 ``Message`` → Anthropic 的 ``{"role", "content"}``。
        - 通用 ``system`` 参数 → Anthropic 顶层 ``system=``(None 时不下发)。
        - ``temperature`` 仅在目标模型支持时下发(能力过滤,避免 400)。
        """
        kwargs: dict[str, object] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if system is not None:
            kwargs["system"] = system
        if _model_supports_temperature(self.model):
            kwargs["temperature"] = self._temperature
        return kwargs

    def chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
    ) -> ChatResponse:
        """一次性请求并返回完整应答(含 token 用量)。"""
        resp = self._client.messages.create(**self._build_request_kwargs(messages, system))
        text = "".join(block.text for block in resp.content if block.type == "text")
        return ChatResponse(
            content=text,
            usage=Usage(
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
            ),
            model=resp.model,
            stop_reason=resp.stop_reason,
            raw=resp,
        )

    def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
    ) -> Iterator[str]:
        """流式请求,逐段 yield 文本增量。

        阶段一约定:流式只回传文本增量,不回传结构化事件 / ``Usage``
        (见 stage-1-design.md §4.3、§11)。
        """
        with self._client.messages.stream(**self._build_request_kwargs(messages, system)) as stream:
            yield from stream.text_stream
