"""ChatGPT(OpenAI)LLM 实现。

``openai`` SDK **只允许出现在本文件里**,且延迟导入:只有真正构造
:class:`OpenAILLM` 时才 ``import openai``,只用 Claude 的用户无需安装 openai。

与 Claude 的几处协议差异都封装在这里(见 stage-1-design.md §4),核心无感:

- **system prompt**:放进 messages 列表首条 ``{"role": "system"}``
  (Claude 是顶层独立参数)。
- **token 字段**:``prompt_tokens`` / ``completion_tokens``
  (Claude 是 ``input_tokens`` / ``output_tokens``)。
- **输出上限**:用 ``max_completion_tokens``(新版 Chat Completions 已用它取代
  ``max_tokens``,GPT‑5 系列要求用前者)。
- **temperature**:GPT‑5 系列推理模型只接受默认值(自定义会 400),故本实现
  **默认不下发** temperature,交由模型默认(=1.0)。阶段一实验本就不引入温度
  变量(见设计 §11),行为一致;需要时再按模型能力放开即可。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from agent_framework.core.llm import ChatResponse, Message, Usage

if TYPE_CHECKING:
    from agent_framework.core.config import Settings

# 未显式配置 MODEL 时,OpenAI provider 使用的默认模型(便宜档,适合开发/实验)。
DEFAULT_MODEL = "gpt-5.4-mini"


class OpenAILLM:
    """基于 OpenAI 官方 SDK(Chat Completions)的 :class:`LLM` 实现。

    结构化满足 ``LLM`` Protocol,无需显式继承。所有 OpenAI 特有的细节
    (system 进消息列表、``max_completion_tokens``、token 字段名)都封装在这里。
    """

    def __init__(self, settings: Settings) -> None:
        """用配置构造客户端,一次性绑定模型与输出上限。

        Args:
            settings: 框架配置,提供 api key / 模型名 / max_tokens。

        Raises:
            ValueError: 未配置 ``OPENAI_API_KEY``。
        """
        import openai

        if not settings.openai_api_key:
            raise ValueError("使用 OpenAI 需在 .env 配置 OPENAI_API_KEY。")
        self._client = openai.OpenAI(api_key=settings.openai_api_key)
        self.model = settings.model or DEFAULT_MODEL
        self._max_tokens = settings.max_tokens

    def _build_messages(
        self,
        messages: list[Message],
        system: str | None,
    ) -> list[dict[str, str]]:
        """把通用消息映射成 OpenAI 的消息列表。

        与 Claude 不同,system prompt 作为首条 ``{"role": "system"}`` 放进列表,
        而非顶层参数。
        """
        out: list[dict[str, str]] = []
        if system is not None:
            out.append({"role": "system", "content": system})
        out.extend({"role": m.role, "content": m.content} for m in messages)
        return out

    def chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
    ) -> ChatResponse:
        """一次性请求并返回完整应答(含 token 用量)。"""
        resp = self._client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self._max_tokens,
            messages=self._build_messages(messages, system),
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage
        return ChatResponse(
            content=text,
            usage=Usage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            ),
            model=resp.model,
            stop_reason=choice.finish_reason,
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
        stream = self._client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self._max_tokens,
            messages=self._build_messages(messages, system),
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
