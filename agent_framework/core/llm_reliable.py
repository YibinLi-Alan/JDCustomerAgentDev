"""可靠性包装层 —— 重试(指数退避 + 抖动)与 provider 降级(阶段六 P-A)。

设计要点(stage-6-design.md §4):

- **装饰器模式**:``ReliableLLM`` / ``FallbackLLM`` 都实现 ``LLM`` 协议、包住任意
  LLM 实例——重试与降级装在传输层,上层(Agent/编排)零感知;
- **只重试可重试错误**(超时/限流/连接/5xx);参数非法、鉴权失败重试一万次也不会
  变对,直接上抛;分类不 import 任何厂商 SDK,靠异常类名与 ``status_code``
  鸭子识别(厂商无关);
- **工具层不在此重试**:工具可能有副作用,重试的前提是幂等(见 tools/base.py
  的 request_id 机制);LLM 调用是纯读,可安心重试;
- ``sleep_fn`` / ``rng`` 可注入 —— 测试不真睡、退避序列可断言。
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterator, Sequence

from agent_framework.core.llm import LLM, ChatResponse, Message

#: 异常类名中出现这些关键词 → 认定可重试(限流/超时/连接/服务端故障)。
_RETRYABLE_NAME_MARKERS = (
    "ratelimit",
    "timeout",
    "connection",
    "apiconnection",
    "internalserver",
    "serviceunavailable",
    "overloaded",
)

#: HTTP 状态码:可重试(408 超时 / 429 限流 / 5xx / 529 Anthropic overloaded)。
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504, 529}


def is_retryable(exc: BaseException) -> bool:
    """判断一个异常是否值得重试(厂商无关的鸭子识别)。

    优先看 ``status_code`` 属性(OpenAI/Anthropic SDK 的 APIStatusError 都带),
    其次看异常类名关键词。默认 **不可重试**——宁可快速失败上报,不做无谓等待。
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS
    name = type(exc).__name__.lower()
    return any(marker in name for marker in _RETRYABLE_NAME_MARKERS)


class ReliableLLM:
    """给任意 ``LLM`` 加上「指数退避 + 抖动」重试的包装(实现 ``LLM`` 协议)。"""

    def __init__(
        self,
        inner: LLM,
        *,
        max_retries: int = 3,
        base_delay: float = 1.0,
        jitter: float = 0.2,
        sleep_fn: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        """
        Args:
            inner: 被包装的 LLM 实现。
            max_retries: 最大重试次数(不含首次调用;「凡是循环必有刹车」第三次出现)。
            base_delay: 首次重试等待秒数;之后按 2 的幂递增(1s→2s→4s…)。
            jitter: 随机抖动幅度(±比例),避免多个客户端同时重试再次挤爆对方。
            sleep_fn: 睡眠函数,测试注入假睡眠。
            rng: 0~1 随机数源,测试注入固定值。
        """
        self._inner = inner
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._jitter = jitter
        self._sleep = sleep_fn
        self._rng = rng

    @property
    def model(self) -> str:
        """透传底层模型 id(供 CLI/日志展示)。"""
        return self._inner.model

    def chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: Sequence[dict[str, object]] | None = None,
    ) -> ChatResponse:
        """带重试的一次性应答。不可重试错误立刻上抛;重试耗尽抛最后一个异常。"""
        attempt = 0
        while True:
            try:
                return self._inner.chat(messages, system=system, tools=tools)
            except Exception as exc:  # noqa: BLE001 — 分类后决定重试或上抛
                if attempt >= self._max_retries or not is_retryable(exc):
                    raise
                delay = self._base_delay * (2**attempt)
                delay *= 1 + self._jitter * (2 * self._rng() - 1)  # ±jitter
                self._sleep(max(0.0, delay))
                attempt += 1

    def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
    ) -> Iterator[str]:
        """流式透传。**不做中途重试**:流已开始吐字后重试会重复输出,
        半途失败的正确姿势是上层整段重发;此处仅透传。"""
        return self._inner.stream(messages, system=system)


class FallbackLLM:
    """主 provider 失败(重试已在内层耗尽)后切备用的降级包装(实现 ``LLM`` 协议)。

    阶段一「可切换 provider」设计在此兑现:主备都是 ``LLM`` 接口,
    部分服务优于完全罢工。备用也失败则上抛主异常链。
    """

    def __init__(self, primary: LLM, secondary: LLM) -> None:
        self._primary = primary
        self._secondary = secondary
        self.last_used: str = "primary"  # 观测用:最近一次实际走了谁

    @property
    def model(self) -> str:
        """展示主模型 id(降级是异常态,不改常规展示)。"""
        return self._primary.model

    def chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: Sequence[dict[str, object]] | None = None,
    ) -> ChatResponse:
        """主失败切备。任何异常(含不可重试)都触发降级——降级层的职责是
        「尽量给出答复」,错误分类由内层 ReliableLLM 负责。"""
        try:
            resp = self._primary.chat(messages, system=system, tools=tools)
            self.last_used = "primary"
            return resp
        except Exception:  # noqa: BLE001 — 主挂了就切备,备也挂再上抛
            resp = self._secondary.chat(messages, system=system, tools=tools)
            self.last_used = "secondary"
            return resp

    def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
    ) -> Iterator[str]:
        """流式:只尝试主(流中途切备会重复吐字,不做)。"""
        return self._primary.stream(messages, system=system)
