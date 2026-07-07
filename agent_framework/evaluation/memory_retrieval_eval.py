"""阶段四交付物⑥:不同记忆策略下的检索准确率对比。

对比四种策略在同一标注集(``datasets/memory_retrieval.json``)上的 Hit@1 / Recall@3:

- **无长期记忆**:什么都检索不到的基线(短期窗口早已滚走,答案全靠长期记忆);
- **纯相关性**:只按余弦相似度排序(w = 1.0 / 0 / 0);
- **三因子·默认**:相关性主导(w = 1.0 / 0.5 / 0.5,评审拍板的默认值);
- **三因子·论文平权**:Generative Agents 原味(w = 1.0 / 1.0 / 1.0)。

embedding 用真实 OpenAI API(``text-embedding-3-small``,全集一次批量调用,
成本忽略不计);每条 query 用**新鲜 store** 评测,避免检索的「保鲜」副作用
(touch 更新 last_accessed_at)污染后续 query 的时近性。

运行(repo 根目录):``python -m agent_framework.evaluation.memory_retrieval_eval``
结果人工誊入报告:``docs/stage-4-memory-eval.md``。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Sequence

from agent_framework.core.config import get_settings
from agent_framework.core.llm import ChatResponse, Message
from agent_framework.memory.embedder import OpenAIEmbedder
from agent_framework.memory.long_term import LongTermMemory
from agent_framework.memory.vector_store import InMemoryVectorStore, MemoryRecord

DATASET = Path(__file__).parent / "datasets" / "memory_retrieval.json"

STRATEGIES: list[tuple[str, tuple[float, float, float] | None]] = [
    ("无长期记忆(基线)", None),
    ("纯相关性(1.0/0/0)", (1.0, 0.0, 0.0)),
    ("三因子·轻量(1.0/0.25/0.25)", (1.0, 0.25, 0.25)),
    ("三因子·默认(1.0/0.5/0.5)", (1.0, 0.5, 0.5)),
    ("三因子·论文平权(1.0/1.0/1.0)", (1.0, 1.0, 1.0)),
]


class _NullLLM:
    """检索不需要 LLM;满足协议、被调用即报错(帮助暴露误用)。"""

    model = "null"

    def chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: Sequence[dict[str, object]] | None = None,
    ) -> ChatResponse:
        raise RuntimeError("评测只做检索,不应触发 LLM 调用。")

    def stream(self, messages: list[Message], *, system: str | None = None) -> Iterator[str]:
        raise RuntimeError("评测只做检索,不应触发 LLM 调用。")


class PrecomputedEmbedder:
    """查表 Embedder:全部向量已一次性算好,评测过程零 API 调用。"""

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._table[t] for t in texts]


def build_store(data: dict, now: datetime) -> InMemoryVectorStore:
    """按数据集预置一个新鲜 store(每条 query 一个,互不污染)。"""
    store = InMemoryVectorStore()
    for m in data["memories"]:
        when = now - timedelta(hours=m["hours_ago"])
        record = MemoryRecord(
            id=m["id"],
            user_id=m["user_id"],
            text=m["text"],
            importance=m["importance"],
            created_at=when,
            last_accessed_at=when,
        )
        store.add(record, m["_vector"])
    return store


def evaluate(data: dict, embedder: PrecomputedEmbedder, now: datetime) -> list[dict]:
    """跑全部策略,返回每策略的指标与逐条命中明细。"""
    queries = data["queries"]
    results = []
    for name, weights in STRATEGIES:
        hit1 = 0
        recall3_sum = 0.0
        details = []
        for q in queries:
            expected = set(q["expected"])
            if weights is None:
                got: list[str] = []
            else:
                ltm = LongTermMemory(
                    _NullLLM(),
                    embedder,
                    build_store(data, now),
                    top_k=3,
                    weight_relevance=weights[0],
                    weight_recency=weights[1],
                    weight_importance=weights[2],
                    now_fn=lambda: now,
                )
                scored = ltm.retrieve(q["user_id"], q["query"])
                got = [s.record.id for s in scored]
                # 隔离红线:命中的记忆必须全部属于查询用户。
                owners = {s.record.user_id for s in scored}
                assert owners <= {q["user_id"]}, f"隔离被破坏:{q} 命中了 {owners}"
            if got and got[0] in expected:
                hit1 += 1
            recall3_sum += len(expected & set(got[:3])) / len(expected)
            details.append({"query": q["query"], "expected": q["expected"], "got": got})
        results.append(
            {
                "strategy": name,
                "hit@1": hit1 / len(queries),
                "recall@3": recall3_sum / len(queries),
                "details": details,
            }
        )
    return results


def main() -> None:
    """跑评测并打印 markdown 结果表(誊入 docs/stage-4-memory-eval.md)。"""
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    now = datetime.fromisoformat(data["now"])

    texts = [m["text"] for m in data["memories"]] + [q["query"] for q in data["queries"]]
    print(f"向量化 {len(texts)} 条文本(一次批量 API 调用)……")
    embedder_api = OpenAIEmbedder(get_settings())
    vectors = embedder_api.embed(texts)
    table = dict(zip(texts, vectors))
    for m in data["memories"]:
        m["_vector"] = table[m["text"]]

    results = evaluate(data, PrecomputedEmbedder(table), now)

    n = len(data["queries"])
    print(f"\n评测集:{n} 条查询 · embedding = {embedder_api.model} · top_k = 3\n")
    print("| 策略 | Hit@1 | Recall@3 |")
    print("|---|---|---|")
    for r in results:
        print(f"| {r['strategy']} | {r['hit@1']:.0%} | {r['recall@3']:.0%} |")

    # 逐条明细:只打印有策略答错的查询,方便分析差异来源。
    print("\n逐条差异(某策略 Hit@1 失败的查询):")
    for i, q in enumerate(data["queries"]):
        line = []
        for r in results[1:]:  # 跳过基线
            got = r["details"][i]["got"]
            top1 = got[0] if got else "-"
            mark = "✓" if top1 in q["expected"] else f"✗(得到 {top1})"
            line.append(f"{r['strategy'].split('(')[0]}:{mark}")
        if any("✗" in item for item in line):
            print(f"  「{q['query']}」期望 {q['expected']} → " + " · ".join(line))


if __name__ == "__main__":
    main()
