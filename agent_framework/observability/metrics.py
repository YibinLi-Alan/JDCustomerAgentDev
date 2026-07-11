"""指标聚合 —— Trace 是显微镜(看单次),指标是仪表盘(看整体)。

从轨迹事件聚合出:成功率、平均步数、平均耗时、平均 token 成本、HITL 触发率。
这些数字的价值在**趋势和对比**——改动前后各跑一遍评测集,数字说话
(评估报告 stage-6-eval-report.md 的数据源)。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from agent_framework.observability.tracer import TraceEvent, load_trace


@dataclass(frozen=True)
class TaskMetrics:
    """单次任务的量化画像(从一份 trace 提炼)。"""

    task_id: str
    ok: bool  # task_end 报告的成败(未见 task_end = False)
    llm_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    duration_ms: float | None  # 首尾事件时间差;事件不足则 None
    handoff: bool  # 本次任务是否触发了人工介入(审批挂起/兜底升级)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def summarize_trace(events: list[TraceEvent]) -> TaskMetrics:
    """把一次任务的事件流压成一行指标。"""
    llm_calls = tool_calls = input_tokens = output_tokens = 0
    ok = False
    handoff = False
    for e in events:
        if e.kind == "llm_call":
            llm_calls += 1
            input_tokens += int(e.payload.get("input_tokens", 0) or 0)
            output_tokens += int(e.payload.get("output_tokens", 0) or 0)
        elif e.kind == "tool_call":
            tool_calls += 1
        elif e.kind in ("approval_pending", "escalation"):
            handoff = True
        elif e.kind == "task_end":
            ok = bool(e.payload.get("ok", False))
    duration_ms: float | None = None
    if len(events) >= 2:
        try:
            start = datetime.fromisoformat(events[0].ts)
            end = datetime.fromisoformat(events[-1].ts)
            duration_ms = (end - start).total_seconds() * 1000
        except ValueError:
            pass
    task_id = events[0].task_id if events else "unknown"
    return TaskMetrics(
        task_id=task_id,
        ok=ok,
        llm_calls=llm_calls,
        tool_calls=tool_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_ms=duration_ms,
        handoff=handoff,
    )


def load_all_metrics(trace_dir: str | Path) -> list[TaskMetrics]:
    """读取目录下全部 trace 文件并逐个提炼(仪表盘的数据装载)。"""
    directory = Path(trace_dir)
    if not directory.exists():
        return []
    return [summarize_trace(load_trace(p)) for p in sorted(directory.glob("*.jsonl"))]


def aggregate(metrics: list[TaskMetrics]) -> dict[str, float]:
    """聚合成仪表盘数字。空集合返回全零(不除零)。"""
    n = len(metrics)
    if n == 0:
        return {
            "tasks": 0,
            "success_rate": 0.0,
            "avg_llm_calls": 0.0,
            "avg_tool_calls": 0.0,
            "avg_total_tokens": 0.0,
            "avg_duration_ms": 0.0,
            "handoff_rate": 0.0,
        }
    timed = [m.duration_ms for m in metrics if m.duration_ms is not None]
    return {
        "tasks": n,
        "success_rate": sum(1 for m in metrics if m.ok) / n,
        "avg_llm_calls": sum(m.llm_calls for m in metrics) / n,
        "avg_tool_calls": sum(m.tool_calls for m in metrics) / n,
        "avg_total_tokens": sum(m.total_tokens for m in metrics) / n,
        "avg_duration_ms": (sum(timed) / len(timed)) if timed else 0.0,
        "handoff_rate": sum(1 for m in metrics if m.handoff) / n,
    }


def render_table(agg: dict[str, float]) -> str:
    """把聚合结果渲染成终端可读的小表(评测报告同款数字)。"""
    lines = [
        "┌──────────────────┬──────────┐",
        f"│ 任务数            │ {int(agg['tasks']):>8} │",
        f"│ 成功率            │ {agg['success_rate']:>7.0%} │",
        f"│ 平均 LLM 调用     │ {agg['avg_llm_calls']:>8.1f} │",
        f"│ 平均工具调用       │ {agg['avg_tool_calls']:>8.1f} │",
        f"│ 平均 tokens       │ {agg['avg_total_tokens']:>8.0f} │",
        f"│ 平均耗时(ms)     │ {agg['avg_duration_ms']:>8.0f} │",
        f"│ 人工介入率         │ {agg['handoff_rate']:>7.0%} │",
        "└──────────────────┴──────────┘",
    ]
    return "\n".join(lines)
