"""结构化日志 —— JSON 行格式,可查询可聚合(阶段六 P-A)。

自由文本日志只能人肉翻阅;结构化日志才能回答「过去一天所有失败任务的共同点」。
极简实现:标准库 ``logging`` + JSON Formatter,零新依赖。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime


class JsonFormatter(logging.Formatter):
    """把 LogRecord 渲染成一行 JSON(ts/level/logger/event + 附加字段)。"""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            entry.update(fields)
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def get_logger(name: str = "agent_framework") -> logging.Logger:
    """取一个已配置 JSON 输出的 logger(幂等:重复调用不叠 handler)。"""
    logger = logging.getLogger(name)
    if not any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_event(logger: logging.Logger, event: str, **fields: object) -> None:
    """记一条结构化事件:``log_event(log, "task_end", task_id=..., ok=True)``。"""
    logger.info(event, extra={"fields": fields})
