"""阶段六 P-A 可靠性单测:重试/退避/降级/幂等(全离线)。"""

from __future__ import annotations

import pytest

from agent_framework.core.llm import ChatResponse, Message, Usage
from agent_framework.core.llm_reliable import FallbackLLM, ReliableLLM, is_retryable
from agent_framework.tools.jd_mock import ApplyRefundTool
from agent_framework.tools.jd_mock_data import JDMockStore

# --------------------------------------------------------------------------- #
# 测试基础设施:按脚本抛错/应答的假 LLM + 假异常                                    #
# --------------------------------------------------------------------------- #


class RateLimitError(Exception):
    """类名含 ratelimit → 可重试。"""


class BadRequestError(Exception):
    def __init__(self, msg: str = "参数非法") -> None:
        super().__init__(msg)
        self.status_code = 400


class OverloadedByStatus(Exception):
    def __init__(self) -> None:
        super().__init__("服务过载")
        self.status_code = 529


def _resp(text: str = "ok") -> ChatResponse:
    return ChatResponse(content=text, usage=Usage(1, 1), model="flaky")


class FlakyLLM:
    """前 N 次 chat 抛指定异常,之后正常应答。"""

    model = "flaky"

    def __init__(self, failures: list[Exception]) -> None:
        self._failures = list(failures)
        self.calls = 0

    def chat(self, messages, *, system=None, tools=None):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self._failures:
            raise self._failures.pop(0)
        return _resp()

    def stream(self, messages, *, system=None):  # type: ignore[no-untyped-def]
        yield "ok"


def _reliable(inner, **kwargs):  # type: ignore[no-untyped-def]
    """不真睡的 ReliableLLM,记录退避序列。"""
    sleeps: list[float] = []
    llm = ReliableLLM(inner, sleep_fn=sleeps.append, rng=lambda: 0.5, **kwargs)  # rng=0.5 → 零抖动
    return llm, sleeps


# --------------------------------------------------------------------------- #
# is_retryable 分类                                                              #
# --------------------------------------------------------------------------- #


def test_is_retryable_by_status_and_name() -> None:
    assert is_retryable(OverloadedByStatus()) is True  # 529
    assert is_retryable(RateLimitError()) is True  # 类名
    assert is_retryable(TimeoutError()) is True  # 类名含 timeout
    assert is_retryable(BadRequestError()) is False  # 400
    assert is_retryable(ValueError("随便")) is False  # 默认不可重试


# --------------------------------------------------------------------------- #
# ReliableLLM                                                                   #
# --------------------------------------------------------------------------- #


def test_retry_backoff_then_success() -> None:
    inner = FlakyLLM([RateLimitError(), RateLimitError()])
    llm, sleeps = _reliable(inner)
    resp = llm.chat([Message("user", "hi")])
    assert resp.content == "ok"
    assert inner.calls == 3  # 失败2次 + 成功1次
    assert sleeps == [1.0, 2.0]  # 指数退避(rng=0.5 → 抖动为 0)


def test_non_retryable_raises_immediately() -> None:
    inner = FlakyLLM([BadRequestError()])
    llm, sleeps = _reliable(inner)
    with pytest.raises(BadRequestError):
        llm.chat([Message("user", "hi")])
    assert inner.calls == 1 and sleeps == []  # 不睡不重试


def test_retries_exhausted_raises_last_error() -> None:
    inner = FlakyLLM([RateLimitError() for _ in range(5)])
    llm, sleeps = _reliable(inner, max_retries=3)
    with pytest.raises(RateLimitError):
        llm.chat([Message("user", "hi")])
    assert inner.calls == 4  # 首次 + 3 次重试(凡是循环必有刹车)
    assert sleeps == [1.0, 2.0, 4.0]


def test_model_property_passthrough() -> None:
    llm, _ = _reliable(FlakyLLM([]))
    assert llm.model == "flaky"


# --------------------------------------------------------------------------- #
# FallbackLLM                                                                   #
# --------------------------------------------------------------------------- #


def test_fallback_switches_on_primary_failure() -> None:
    primary = FlakyLLM([BadRequestError()])
    secondary = FlakyLLM([])
    llm = FallbackLLM(primary, secondary)
    resp = llm.chat([Message("user", "hi")])
    assert resp.content == "ok"
    assert llm.last_used == "secondary"
    assert secondary.calls == 1


def test_fallback_both_fail_raises() -> None:
    llm = FallbackLLM(FlakyLLM([RateLimitError()]), FlakyLLM([RateLimitError()]))
    with pytest.raises(RateLimitError):
        llm.chat([Message("user", "hi")])


def test_fallback_prefers_primary_when_healthy() -> None:
    primary, secondary = FlakyLLM([]), FlakyLLM([])
    llm = FallbackLLM(primary, secondary)
    llm.chat([Message("user", "hi")])
    assert llm.last_used == "primary" and secondary.calls == 0


# --------------------------------------------------------------------------- #
# 工具幂等(request_id)                                                          #
# --------------------------------------------------------------------------- #


def test_same_request_id_executes_once() -> None:
    store = JDMockStore()
    tool = ApplyRefundTool(store)
    args = {"order_id": "12345", "reason": "商品损坏"}
    first = tool.invoke(args, request_id="req-1")
    second = tool.invoke(args, request_id="req-1")  # 重放:直接返回首次结果
    assert first.ok and second.content == first.content
    assert len(store.refunds) == 1  # 副作用只发生一次


def test_different_request_ids_execute_separately() -> None:
    store = JDMockStore()
    tool = ApplyRefundTool(store)
    tool.invoke({"order_id": "12345", "reason": "a"}, request_id="req-1")
    tool.invoke({"order_id": "12345", "reason": "b"}, request_id="req-2")
    assert len(store.refunds) == 2


def test_no_request_id_no_dedup() -> None:
    store = JDMockStore()
    tool = ApplyRefundTool(store)
    tool.invoke({"order_id": "12345", "reason": "a"})
    tool.invoke({"order_id": "12345", "reason": "a"})
    assert len(store.refunds) == 2  # 模型路径不带 id,不去重(两次退款是模型的决定)


def test_failed_result_not_cached() -> None:
    store = JDMockStore()
    tool = ApplyRefundTool(store)
    bad = tool.invoke({"order_id": "99999", "reason": "x"}, request_id="req-1")
    assert "未找到" in bad.content  # 业务失败(ok=True 但没落库)
    # 参数校验失败的情况:缺 reason → 不缓存,修正后同 id 可正常执行
    invalid = tool.invoke({"order_id": "12345"}, request_id="req-2")
    assert invalid.ok is False
    fixed = tool.invoke({"order_id": "12345", "reason": "补上理由"}, request_id="req-2")
    assert fixed.ok is True
    assert len(store.refunds) == 1


def test_registry_passes_request_id_through() -> None:
    from agent_framework.tools.presets import default_registry

    store = JDMockStore()
    registry = default_registry(store)
    registry.invoke("apply_refund", {"order_id": "12345", "reason": "x"}, request_id="rq")
    registry.invoke("apply_refund", {"order_id": "12345", "reason": "x"}, request_id="rq")
    assert len(store.refunds) == 1
