"""长期记忆的存储层:``MemoryRecord`` + ``VectorStore`` 接口 + 两个实现(阶段四 P-B)。

- :class:`InMemoryVectorStore`:纯 Python 余弦检索,零依赖 —— 单测全离线跑它,
  也是「存储可替换」的活证明;
- :class:`ChromaVectorStore`:Chroma ``PersistentClient`` 持久化到磁盘,
  跨会话记忆的真实落盘处(``chromadb`` 延迟导入,只在用到时才需要安装)。

**user_id 隔离在存储层强制执行**(检索/列表/删除都按 user_id 过滤),
不依赖上层自觉 —— 见 stage-4-design.md §7.5。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Protocol, Sequence


@dataclass(frozen=True)
class MemoryRecord:
    """一条长期记忆(提炼后的事实)。

    Attributes:
        id: 记录唯一 id。
        user_id: 归属用户;检索/删除的强制过滤键。
        text: 事实文本(如「用户的常用收货地址在上海浦东」)。
        importance: 写入时 LLM 打的重要性分(1–10)。
        created_at: 首次写入时间。
        last_accessed_at: 最近被检索命中的时间(「保鲜」,时近性因子的基准)。
    """

    id: str
    user_id: str
    text: str
    importance: int
    created_at: datetime
    last_accessed_at: datetime


@dataclass(frozen=True)
class SearchHit:
    """一次向量检索的单条命中:记录 + 相关性(0~1,越大越相关)。"""

    record: MemoryRecord
    relevance: float


class VectorStore(Protocol):
    """长期记忆存储接口。所有方法都以 user_id 为隔离边界(除按 id 直删)。"""

    def add(self, record: MemoryRecord, vector: Sequence[float]) -> None:
        """写入一条新记录及其向量。"""
        ...

    def update(self, record: MemoryRecord, vector: Sequence[float]) -> None:
        """按 ``record.id`` 整条覆盖(文本/向量/元数据)。"""
        ...

    def delete(self, ids: Sequence[str]) -> None:
        """按 id 删除(不存在的 id 静默忽略)。"""
        ...

    def delete_user(self, user_id: str) -> int:
        """删除该用户的全部记忆,返回删除条数(「删除我的个人信息」通道)。"""
        ...

    def search(self, user_id: str, vector: Sequence[float], k: int) -> list[SearchHit]:
        """在该用户的记忆中按余弦相似度召回 top-k(相关性归一化到 0~1)。"""
        ...

    def touch(self, ids: Sequence[str], when: datetime) -> None:
        """把这些记录的 ``last_accessed_at`` 刷新为 ``when``(检索命中后「保鲜」)。"""
        ...

    def list_user(self, user_id: str) -> list[MemoryRecord]:
        """列出该用户的全部记忆(按创建时间升序),供 CLI ``/memories``。"""
        ...


def _cosine_relevance(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度,截断到 0~1(文本 embedding 的负相似没有排序意义)。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if norm == 0:
        return 0.0
    return max(0.0, min(1.0, dot / norm))


class InMemoryVectorStore:
    """纯内存实现:dict + 暴力余弦。教学规模(几百条)绰绰有余,单测零依赖。"""

    def __init__(self) -> None:
        self._rows: dict[str, tuple[MemoryRecord, list[float]]] = {}

    def add(self, record: MemoryRecord, vector: Sequence[float]) -> None:
        self._rows[record.id] = (record, list(vector))

    def update(self, record: MemoryRecord, vector: Sequence[float]) -> None:
        self._rows[record.id] = (record, list(vector))

    def delete(self, ids: Sequence[str]) -> None:
        for record_id in ids:
            self._rows.pop(record_id, None)

    def delete_user(self, user_id: str) -> int:
        doomed = [rid for rid, (rec, _) in self._rows.items() if rec.user_id == user_id]
        for rid in doomed:
            del self._rows[rid]
        return len(doomed)

    def search(self, user_id: str, vector: Sequence[float], k: int) -> list[SearchHit]:
        hits = [
            SearchHit(record=rec, relevance=_cosine_relevance(vector, vec))
            for rec, vec in self._rows.values()
            if rec.user_id == user_id
        ]
        hits.sort(key=lambda h: h.relevance, reverse=True)
        return hits[:k]

    def touch(self, ids: Sequence[str], when: datetime) -> None:
        for record_id in ids:
            row = self._rows.get(record_id)
            if row is not None:
                record, vector = row
                self._rows[record_id] = (replace(record, last_accessed_at=when), vector)

    def list_user(self, user_id: str) -> list[MemoryRecord]:
        records = [rec for rec, _ in self._rows.values() if rec.user_id == user_id]
        records.sort(key=lambda r: r.created_at)
        return records

    def __len__(self) -> int:
        return len(self._rows)


class ChromaVectorStore:
    """Chroma 持久化实现:数据落磁盘,跨会话仍在(``chromadb`` 延迟导入)。

    向量由我们自己算(:class:`~agent_framework.memory.embedder.Embedder`),
    Chroma 只当「带 metadata 过滤的向量存储」用;collection 用 cosine 空间,
    ``relevance = 1 - distance``。三因子打分在框架层(``long_term.py``),不在存储层。
    """

    def __init__(self, persist_dir: str, *, collection_name: str = "agent_memory") -> None:
        """打开(或创建)持久化 collection。

        Args:
            persist_dir: 落盘目录(建议用 ``settings.memory_persist_dir``,已 gitignore)。
            collection_name: collection 名,默认一个库一张表。
        """
        import chromadb  # 延迟导入:只有选用 Chroma 后端才需要安装

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ---- MemoryRecord <-> Chroma 行 ----------------------------------------

    @staticmethod
    def _to_metadata(record: MemoryRecord) -> dict[str, object]:
        return {
            "user_id": record.user_id,
            "importance": record.importance,
            "created_at": record.created_at.isoformat(),
            "last_accessed_at": record.last_accessed_at.isoformat(),
        }

    @staticmethod
    def _to_record(record_id: str, text: str, meta: dict[str, object]) -> MemoryRecord:
        return MemoryRecord(
            id=record_id,
            user_id=str(meta["user_id"]),
            text=text,
            importance=int(meta["importance"]),  # type: ignore[arg-type]
            created_at=datetime.fromisoformat(str(meta["created_at"])),
            last_accessed_at=datetime.fromisoformat(str(meta["last_accessed_at"])),
        )

    # ---- VectorStore 接口 ---------------------------------------------------

    def add(self, record: MemoryRecord, vector: Sequence[float]) -> None:
        self._collection.add(
            ids=[record.id],
            documents=[record.text],
            embeddings=[list(vector)],
            metadatas=[self._to_metadata(record)],
        )

    def update(self, record: MemoryRecord, vector: Sequence[float]) -> None:
        self._collection.update(
            ids=[record.id],
            documents=[record.text],
            embeddings=[list(vector)],
            metadatas=[self._to_metadata(record)],
        )

    def delete(self, ids: Sequence[str]) -> None:
        if ids:
            self._collection.delete(ids=list(ids))

    def delete_user(self, user_id: str) -> int:
        existing = self._collection.get(where={"user_id": user_id})
        doomed = existing["ids"]
        if doomed:
            self._collection.delete(ids=doomed)
        return len(doomed)

    def search(self, user_id: str, vector: Sequence[float], k: int) -> list[SearchHit]:
        if k <= 0:
            return []
        result = self._collection.query(
            query_embeddings=[list(vector)],
            n_results=k,
            where={"user_id": user_id},
            include=["documents", "metadatas", "distances"],
        )
        hits: list[SearchHit] = []
        ids = result["ids"][0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for record_id, text, meta, distance in zip(ids, docs, metas, distances):
            relevance = max(0.0, min(1.0, 1.0 - float(distance)))  # cosine 距离 → 相似度
            hits.append(
                SearchHit(record=self._to_record(record_id, text, dict(meta)), relevance=relevance)
            )
        return hits

    def touch(self, ids: Sequence[str], when: datetime) -> None:
        if not ids:
            return
        existing = self._collection.get(ids=list(ids), include=["metadatas"])
        found_ids = existing["ids"]
        metas = existing.get("metadatas") or []
        if not found_ids:
            return
        new_metas = []
        for meta in metas:
            merged = dict(meta)
            merged["last_accessed_at"] = when.isoformat()
            new_metas.append(merged)
        self._collection.update(ids=found_ids, metadatas=new_metas)

    def list_user(self, user_id: str) -> list[MemoryRecord]:
        result = self._collection.get(
            where={"user_id": user_id}, include=["documents", "metadatas"]
        )
        records = [
            self._to_record(record_id, text, dict(meta))
            for record_id, text, meta in zip(
                result["ids"], result.get("documents") or [], result.get("metadatas") or []
            )
        ]
        records.sort(key=lambda r: r.created_at)
        return records
