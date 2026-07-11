"""成本与频率限制 —— 滑动窗口限流 + 单任务 token 预算(阶段六 P-B)。

防的不只是恶意刷量,也防自己的 bug(一个死循环一夜烧光预算)。
**诚实边界**:进程内实现,重启清零、多实例不共享;生产要外置到 Redis 等
共享存储 —— 接口不变,换实现即可(与 VectorStore 可替换同一思路)。
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Callable


class RateLimiter:
    """每 user_id 的滑动窗口限流(进程内)。"""

    def __init__(
        self,
        *,
        max_per_minute: int = 20,
        window_seconds: float = 60.0,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max = max_per_minute
        self._window = window_seconds
        self._now = now_fn
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, user_id: str) -> bool:
        """检查并记账:窗口内未超额返回 True 并计一次;超额返回 False 不计。"""
        now = self._now()
        hits = self._hits[user_id]
        while hits and now - hits[0] > self._window:
            hits.popleft()
        if len(hits) >= self._max:
            return False
        hits.append(now)
        return True

    def remaining(self, user_id: str) -> int:
        """当前窗口剩余额度(展示/429 响应头用)。"""
        now = self._now()
        hits = self._hits[user_id]
        while hits and now - hits[0] > self._window:
            hits.popleft()
        return max(0, self._max - len(hits))


class TokenBudget:
    """单任务 token 预算:超限即刹车(第 4 次出现的「凡是循环必有刹车」)。

    用法:挂在 tracer listener 上累计 ``llm_call`` 事件的用量,
    编排层每步之间查 ``exceeded`` 决定是否终止转人工。
    """

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self.used = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.used += int(input_tokens) + int(output_tokens)

    def on_trace_event(self, event) -> None:  # type: ignore[no-untyped-def]
        """Tracer listener 适配:自动从 llm_call 事件累计。"""
        if getattr(event, "kind", "") == "llm_call":
            payload = getattr(event, "payload", {})
            self.add(payload.get("input_tokens", 0) or 0, payload.get("output_tokens", 0) or 0)

    @property
    def exceeded(self) -> bool:
        return self.used > self._limit

    @property
    def limit(self) -> int:
        return self._limit
