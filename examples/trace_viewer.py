"""阶段六交付物② Trace 可视化查看器(极简终端版)。

用法:
    python -m examples.trace_viewer                # 列出 data/traces/ 下全部任务 + 汇总表
    python -m examples.trace_viewer <task_id|path> # 渲染单任务完整轨迹时间线

大纲「可视化查看执行轨迹」的达标件;生产环境可对接 Langfuse 等平台
(接口 = 给 Tracer 加 listener),本项目按减法自建极简版。
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent_framework.core.config import get_settings
from agent_framework.observability import (
    aggregate,
    load_all_metrics,
    load_trace,
    render_table,
    summarize_trace,
)

_KIND_ICONS = {
    "task_start": "▶",
    "route": "⇒",
    "plan": "📋",
    "llm_call": "🧠",
    "tool_call": "🔧",
    "tool_result": "↩",
    "replan": "♻",
    "synthesize": "∑",
    "critic": "🔍",
    "approval_pending": "⏸",
    "escalation": "🙋",
    "final_answer": "💬",
    "task_end": "■",
}


def _render_one(path: Path) -> None:
    events = load_trace(path)
    if not events:
        print(f"[{path} 中没有事件]")
        return
    print(f"任务 {events[0].task_id} 轨迹({len(events)} 条事件):")
    for e in events:
        icon = _KIND_ICONS.get(e.kind, "·")
        detail = " ".join(f"{k}={v}" for k, v in e.payload.items())
        print(f"  {e.seq:>3}  {e.ts[11:23]}  {icon} {e.kind:<16} {detail}")
    m = summarize_trace(events)
    print(
        f"\n小结:LLM 调用 {m.llm_calls} 次 · 工具调用 {m.tool_calls} 次 · "
        f"tokens {m.total_tokens} · 人工介入 {'是' if m.handoff else '否'} · "
        f"成功 {'是' if m.ok else '否/未记录'}"
    )


def main() -> None:
    trace_dir = Path(get_settings().trace_dir)
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        path = Path(arg) if arg.endswith(".jsonl") else trace_dir / f"{arg}.jsonl"
        if not path.exists():
            print(f"[找不到轨迹文件:{path}]")
            sys.exit(1)
        _render_one(path)
        return

    metrics = load_all_metrics(trace_dir)
    if not metrics:
        print(f"[{trace_dir} 下还没有轨迹;跑一次 multi_agent_cli / API 后再来]")
        return
    print(f"共 {len(metrics)} 次任务(目录:{trace_dir}):")
    for m in metrics:
        print(
            f"  {m.task_id}  {'✓' if m.ok else '·'}  llm={m.llm_calls} tools={m.tool_calls} "
            f"tokens={m.total_tokens}  handoff={'Y' if m.handoff else '-'}"
        )
    print("\n" + render_table(aggregate(metrics)))
    print("\n查看单个任务:python -m examples.trace_viewer <task_id>")


if __name__ == "__main__":
    main()
