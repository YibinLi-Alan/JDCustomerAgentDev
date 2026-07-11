"""Observability 子包 —— Trace / 结构化日志 / 指标(阶段六)。

- :mod:`tracer`:任务级轨迹记录(JSONL 落盘 + listener 广播,一份事件流三个消费者);
- :mod:`logger`:JSON 行结构化日志(标准库 logging,零新依赖);
- :mod:`metrics`:从 trace 聚合仪表盘数字(成功率/步数/耗时/token/人工介入率)。

全部是**旁路**:任何观测失败都不允许影响业务主链路。
不接 LangSmith/Langfuse(减法);对接口子 = 给 Tracer 加一个 listener。
"""

from agent_framework.observability.logger import JsonFormatter, get_logger, log_event
from agent_framework.observability.metrics import (
    TaskMetrics,
    aggregate,
    load_all_metrics,
    render_table,
    summarize_trace,
)
from agent_framework.observability.tracer import TraceEvent, Tracer, load_trace

__all__ = [
    "JsonFormatter",
    "TaskMetrics",
    "TraceEvent",
    "Tracer",
    "aggregate",
    "get_logger",
    "load_all_metrics",
    "load_trace",
    "log_event",
    "render_table",
    "summarize_trace",
]
