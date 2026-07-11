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

阶段三 P-B:原生 Function Calling 的 OpenAI 侧协议差异也封装在这里
(见 stage-3-design.md §5.4):

- 工具描述:通用 Schema → ``{"type": "function", "function": {...}}``;若 Schema
  满足 OpenAI 严格模式的前提(``additionalProperties: false`` 且全部字段必填),
  自动附加 ``"strict": true``(服务端保证参数严格符合 Schema)。
- 模型要调工具:``message.tool_calls``(参数是 JSON **字符串**)→ 解析成通用
  :class:`ToolCall`;参数 JSON 非法时折叠成空参数,交由工具校验报错、模型重试。
- 回传结果:通用 ``tool`` 消息 → ``{"role": "tool", "tool_call_id", "content"}``。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Iterator, Sequence

from agent_framework.core.llm import ChatResponse, Message, ToolCall, Usage

if TYPE_CHECKING:
    from agent_framework.core.config import Settings

# 未显式配置 MODEL 时,OpenAI provider 使用的默认模型(便宜档,适合开发/实验)。
DEFAULT_MODEL = "gpt-5.4-mini"


def _qualifies_for_openai_strict(parameters: dict[str, object]) -> bool:
    """判断参数 Schema 是否满足 OpenAI ``"strict": true`` 的前提条件。

    OpenAI 严格模式要求 ``additionalProperties: false`` 且 **所有** 属性都在
    ``required`` 里;我们的 strict 工具(``BaseTool.strict=True``)天然满足前者,
    后者取决于工具是否有带默认值的可选参数。
    """
    if parameters.get("additionalProperties") is not False:
        return False
    properties = parameters.get("properties")
    required = parameters.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required, list):
        return False
    return set(required) == set(properties)


def _to_openai_tools(tools: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """通用工具 Schema → OpenAI ``tools`` 格式(满足前提时自动加 ``strict``)。"""
    out: list[dict[str, object]] = []
    for t in tools:
        function: dict[str, object] = {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["parameters"],
        }
        parameters = t["parameters"]
        if isinstance(parameters, dict) and _qualifies_for_openai_strict(parameters):
            function["strict"] = True
        out.append({"type": "function", "function": function})
    return out


def _to_openai_messages(
    messages: list[Message],
    system: str | None,
) -> list[dict[str, object]]:
    """通用 ``Message`` 列表 → OpenAI 消息列表。

    与 Claude 不同:system prompt 作为首条 ``{"role": "system"}`` 进列表;
    ``assistant`` 的 ``tool_calls`` 是独立字段(参数须序列化成 JSON 字符串);
    工具结果用独立的 ``{"role": "tool"}`` 消息逐条回传,无需合并。
    """
    out: list[dict[str, object]] = []
    if system is not None:
        out.append({"role": "system", "content": system})
    for m in messages:
        if m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        elif m.role == "assistant" and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.name,
                                "arguments": json.dumps(c.args, ensure_ascii=False),
                            },
                        }
                        for c in m.tool_calls
                    ],
                }
            )
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def _parse_tool_calls(message: object) -> list[ToolCall]:
    """从 OpenAI 应答消息解析 ``tool_calls``(通用 :class:`ToolCall` 列表)。

    OpenAI 的参数是 JSON **字符串**;解析失败时折叠成空参数字典,让工具的
    参数校验产生可读错误喂回模型重试,而不是在这里抛异常炸掉循环。
    """
    raw_calls = getattr(message, "tool_calls", None) or []
    calls: list[ToolCall] = []
    for tc in raw_calls:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args))
    return calls


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
        # 超时四层保险之②:LLM 单次调用超时(①工具超时 ③步数上限 ④整任务 deadline)
        self._client = openai.OpenAI(
            api_key=settings.openai_api_key, timeout=settings.llm_timeout_seconds
        )
        self.model = settings.model or DEFAULT_MODEL
        self._max_tokens = settings.max_tokens

    def chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: Sequence[dict[str, object]] | None = None,
    ) -> ChatResponse:
        """一次性请求并返回完整应答(含 token 用量与解析后的 ``tool_calls``)。"""
        kwargs: dict[str, object] = {
            "model": self.model,
            "max_completion_tokens": self._max_tokens,
            "messages": _to_openai_messages(messages, system),
        }
        if tools:
            kwargs["tools"] = _to_openai_tools(tools)
        resp = self._client.chat.completions.create(**kwargs)
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
            tool_calls=_parse_tool_calls(choice.message),
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
            messages=_to_openai_messages(messages, system),
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
