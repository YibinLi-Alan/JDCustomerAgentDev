"""ChromaVectorStore 的落盘往返测试(未安装 chromadb 时整文件跳过)。"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip("chromadb")

from agent_framework.memory.vector_store import ChromaVectorStore, MemoryRecord  # noqa: E402

NOW = datetime(2026, 7, 7, 12, 0, 0)


def rec(record_id: str, user_id: str, text: str, importance: int = 5) -> MemoryRecord:
    return MemoryRecord(
        id=record_id,
        user_id=user_id,
        text=text,
        importance=importance,
        created_at=NOW,
        last_accessed_at=NOW,
    )


def test_chroma_roundtrip_and_isolation(tmp_path: Path) -> None:
    store = ChromaVectorStore(str(tmp_path / "chroma"))
    store.add(rec("a1", "alice", "地址在上海", importance=8), [1.0, 0.0, 0.0])
    store.add(rec("a2", "alice", "爱喝咖啡"), [0.0, 1.0, 0.0])
    store.add(rec("b1", "bob", "爱喝茶"), [1.0, 0.0, 0.0])

    hits = store.search("alice", [1.0, 0.0, 0.0], k=5)
    assert [h.record.id for h in hits] == ["a1", "a2"]  # bob 不可见;按相关性降序
    assert hits[0].relevance > 0.99
    assert hits[0].record.importance == 8  # metadata 完整还原

    # update:同 id 覆盖文本与向量
    store.update(rec("a1", "alice", "地址在北京", importance=9), [0.0, 0.0, 1.0])
    updated = store.search("alice", [0.0, 0.0, 1.0], k=1)[0]
    assert updated.record.id == "a1" and updated.record.text == "地址在北京"

    # touch:刷新 last_accessed_at
    later = NOW + timedelta(hours=3)
    store.touch(["a2"], later)
    by_id = {r.id: r for r in store.list_user("alice")}
    assert by_id["a2"].last_accessed_at == later

    # 持久化:新实例(同目录)仍能读到
    reopened = ChromaVectorStore(str(tmp_path / "chroma"))
    assert len(reopened.list_user("alice")) == 2

    # delete / delete_user
    store.delete(["a2"])
    assert [r.id for r in store.list_user("alice")] == ["a1"]
    assert store.delete_user("bob") == 1
    assert store.list_user("bob") == []
