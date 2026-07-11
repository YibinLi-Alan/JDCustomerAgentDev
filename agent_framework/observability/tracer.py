"""Trace 系统 —— 一次任务从进到出的完整决策记录(阶段六 P-A)。

设计要点(stage-6-design.md §5.1):

- **Trace 对我们几乎是免费的**:ReAct/编排循环里这些信息本来就流经代码,
  只是记下来而不是用完即弃;
- 一份事件流,三个消费者:JSONL 落盘(离线分析)、终端 trace_viewer(调试)、
  SSE 推送(P-D 的中间步骤流式)——都通过 ``listeners`` 订阅;
- 事件结构化(kind + payload dict),不写自由文本——「面向机器的信息永远结构化」
  第五次出现;
- Tracer 是旁路:落盘失败、listener 抛异常都被吞掉,观测永不拖垮业务。
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

#: 事件 kind 约定(编排层与 Agent 钩子共同遵守;viewer/metrics 按此解读)。
#: task_start / route / plan / step_start / llm_call / tool_call / tool_result /
#: step_end / replan / synthesize / critic / approval_pending / escalation /
#: final_answer / task_end


@dataclass(frozen=True)
class TraceEvent:
    """一条结构化轨迹事件。

    Attributes:
        task_id: 所属任务 id。
        seq: 任务内单调递增序号(排序与断言用)。
        ts: ISO 时间戳。
        kind: 事件类型(见模块头部约定)。
        payload: 事件数据(必须可 JSON 序列化)。
    """

    task_id: str
    seq: int
    ts: str
    kind: str
    payload: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "task_id": self.task_id,
                "seq": self.seq,
                "ts": self.ts,
                "kind": self.kind,
                "payload": self.payload,
            },
            ensure_ascii=False,
            default=str,
        )


class Tracer:
    """单个任务的轨迹记录器:emit 收集事件,可选 JSONL 落盘 + listener 广播。"""

    def __init__(
        self,
        task_id: str | None = None,
        *,
        trace_dir: str | Path | None = None,
        listeners: tuple[Callable[[TraceEvent], None], ...] = (),
        now_fn: Callable[[], datetime] = datetime.now,
    ) -> None:
        """
        Args:
            task_id: 任务 id;缺省生成短 uuid。
            trace_dir: JSONL 落盘目录;None 只留内存(测试/评测用)。
            listeners: 事件订阅者(SSE 队列、日志器…);抛异常会被吞掉。
            now_fn: 时钟注入(测试固定时间)。
        """
        self.task_id = task_id or uuid.uuid4().hex[:12]
        self._events: list[TraceEvent] = []
        self._listeners = list(listeners)
        self._now = now_fn
        self._path: Path | None = None
        if trace_dir is not None:
            directory = Path(trace_dir)
            directory.mkdir(parents=True, exist_ok=True)
            self._path = directory / f"{self.task_id}.jsonl"

    @property
    def events(self) -> list[TraceEvent]:
        """已记录事件的快照(按 seq 有序)。"""
        return list(self._events)

    def add_listener(self, listener: Callable[[TraceEvent], None]) -> None:
        """追加订阅者(P-D 的 SSE 队列在请求期挂上来)。"""
        self._listeners.append(listener)

    def emit(self, kind: str, **payload: object) -> TraceEvent:
        """记录一条事件:内存 + 落盘 + 广播。旁路失败全部吞掉,永不抛。"""
        event = TraceEvent(
            task_id=self.task_id,
            seq=len(self._events) + 1,
            ts=self._now().isoformat(timespec="milliseconds"),
            kind=kind,
            payload=dict(payload),
        )
        self._events.append(event)
        if self._path is not None:
            try:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(event.to_json() + "\n")
            except OSError:
                pass  # 落盘是旁路,磁盘问题不拖垮业务
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:  # noqa: BLE001 — 订阅者的锅不甩给业务
                pass
        return event

    def as_on_event(self) -> Callable[[str, dict[str, object]], None]:
        """适配 ``ToolCallingAgent(on_event=...)`` 钩子签名的转发器。"""

        def _forward(kind: str, payload: dict[str, object]) -> None:
            self.emit(kind, **payload)

        return _forward


def load_trace(path: str | Path) -> list[TraceEvent]:
    """从 JSONL 文件回读一次任务的完整轨迹(viewer 与 metrics 用)。"""
    events: list[TraceEvent] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        events.append(
            TraceEvent(
                task_id=str(raw["task_id"]),
                seq=int(raw["seq"]),
                ts=str(raw["ts"]),
                kind=str(raw["kind"]),
                payload=dict(raw.get("payload", {})),
            )
        )
    return sorted(events, key=lambda e: e.seq)
