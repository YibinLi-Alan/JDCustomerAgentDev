"""阶段六 P-A 可观测单测:Tracer/JSONL 回读/on_event 钩子/metrics 聚合(全离线)。"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from agent_framework.core.agent import ToolCallingAgent
from agent_framework.core.llm import ChatResponse, ToolCall, Usage
from agent_framework.observability import (
    Tracer,
    aggregate,
    load_trace,
    render_table,
    summarize_trace,
)
from agent_framework.tools.jd_mock_data import JDMockStore
from agent_framework.tools.presets import default_registry
from tests.mock_llm import MockLLM


class _Clock:
    """可推进的假时钟(duration 断言用)。"""

    def __init__(self) -> None:
        self._now = datetime(2026, 7, 12, 10, 0, 0)

    def tick(self, ms: int = 100) -> None:
        self._now += timedelta(milliseconds=ms)

    def __call__(self) -> datetime:
        return self._now


# --------------------------------------------------------------------------- #
# Tracer 基础                                                                    #
# --------------------------------------------------------------------------- #


def test_tracer_emits_ordered_events_and_notifies_listeners() -> None:
    seen: list[str] = []
    tracer = Tracer("t1", listeners=(lambda e: seen.append(e.kind),))
    tracer.emit("task_start", user_id="alice")
    tracer.emit("tool_call", tool="query_order")
    assert [e.seq for e in tracer.events] == [1, 2]
    assert seen == ["task_start", "tool_call"]
    assert tracer.events[0].payload == {"user_id": "alice"}


def test_tracer_listener_exception_swallowed() -> None:
    def bad_listener(event) -> None:  # type: ignore[no-untyped-def]
        raise RuntimeError("订阅者炸了")

    tracer = Tracer("t1", listeners=(bad_listener,))
    tracer.emit("task_start")  # 不抛 = 旁路失败不拖垮业务
    assert len(tracer.events) == 1


def test_tracer_jsonl_roundtrip(tmp_path: Path) -> None:
    tracer = Tracer("t42", trace_dir=tmp_path)
    tracer.emit("task_start", user_id="u1")
    tracer.emit("task_end", ok=True)
    events = load_trace(tmp_path / "t42.jsonl")
    assert [e.kind for e in events] == ["task_start", "task_end"]
    assert events[1].payload["ok"] is True
    assert events[0].task_id == "t42"


# --------------------------------------------------------------------------- #
# on_event 钩子(唯一 core 改动:只读、异常被吞、缺省零变化)                          #
# --------------------------------------------------------------------------- #


def _tool_call_resp(name: str, args: dict[str, object]) -> ChatResponse:
    return ChatResponse(
        content="",
        usage=Usage(10, 5),
        model="mock",
        tool_calls=[ToolCall(id="c1", name=name, args=args)],
    )


def test_agent_on_event_sequence() -> None:
    tracer = Tracer("t1")
    llm = MockLLM([_tool_call_resp("query_order", {"order_id": "12345"}), "订单已发货。"])
    agent = ToolCallingAgent(llm, default_registry(JDMockStore()), on_event=tracer.as_on_event())
    result = agent.run("订单 12345 状态?")
    assert result.final_answer == "订单已发货。"
    kinds = [e.kind for e in tracer.events]
    assert kinds == ["llm_call", "tool_call", "tool_result", "llm_call", "final_answer"]
    # llm_call 带 token 用量;tool_result 带 ok
    assert tracer.events[0].payload["input_tokens"] == 10
    assert tracer.events[2].payload["ok"] is True
    assert tracer.events[4].payload["stopped_reason"] == "final_answer"


def test_agent_on_event_exception_does_not_break_loop() -> None:
    def bomb(kind: str, payload: dict) -> None:  # type: ignore[type-arg]
        raise RuntimeError("观测炸了")

    llm = MockLLM(["直接作答。"])
    agent = ToolCallingAgent(llm, default_registry(JDMockStore()), on_event=bomb)
    assert agent.run("你好").final_answer == "直接作答。"


def test_agent_without_hook_behaves_identically() -> None:
    llm = MockLLM(["直接作答。"])
    agent = ToolCallingAgent(llm, default_registry(JDMockStore()))
    assert agent.run("你好").stopped_reason == "final_answer"


# --------------------------------------------------------------------------- #
# metrics 聚合                                                                   #
# --------------------------------------------------------------------------- #


def _make_trace(clock: _Clock, *, ok: bool, handoff: bool = False) -> list:
    tracer = Tracer(now_fn=clock)
    tracer.emit("task_start")
    clock.tick(100)
    tracer.emit("llm_call", input_tokens=100, output_tokens=50, tool_calls=1)
    tracer.emit("tool_call", tool="query_order")
    if handoff:
        tracer.emit("approval_pending", item_id="a1")
    clock.tick(100)
    tracer.emit("task_end", ok=ok)
    return tracer.events


def test_summarize_trace_counts_and_duration() -> None:
    clock = _Clock()
    m = summarize_trace(_make_trace(clock, ok=True, handoff=True))
    assert m.ok is True
    assert m.llm_calls == 1 and m.tool_calls == 1
    assert m.total_tokens == 150
    assert m.handoff is True
    assert m.duration_ms == 200


def test_aggregate_and_render() -> None:
    clock = _Clock()
    metrics = [
        summarize_trace(_make_trace(clock, ok=True)),
        summarize_trace(_make_trace(clock, ok=False, handoff=True)),
    ]
    agg = aggregate(metrics)
    assert agg["tasks"] == 2
    assert agg["success_rate"] == 0.5
    assert agg["handoff_rate"] == 0.5
    assert "成功率" in render_table(agg)


def test_aggregate_empty_no_division_error() -> None:
    assert aggregate([])["tasks"] == 0
