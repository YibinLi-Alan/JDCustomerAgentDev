"""长期记忆(提炼 / 写入决策 / 三因子检索 / 隔离)的单元测试。全部离线。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from agent_framework.memory.long_term import LongTermMemory
from agent_framework.memory.short_term import Turn
from agent_framework.memory.vector_store import InMemoryVectorStore, MemoryRecord
from tests.mock_embedder import MockEmbedder
from tests.mock_llm import MockLLM

NOW = datetime(2026, 7, 7, 12, 0, 0)


def fixed_now() -> datetime:
    return NOW


def make_ltm(
    llm: MockLLM,
    embedder: MockEmbedder,
    store: InMemoryVectorStore,
    **overrides: object,
) -> LongTermMemory:
    kwargs: dict[str, object] = dict(top_k=3, now_fn=fixed_now)
    kwargs.update(overrides)
    return LongTermMemory(llm, embedder, store, **kwargs)  # type: ignore[arg-type]


def seed(
    store: InMemoryVectorStore,
    record_id: str,
    user_id: str,
    text: str,
    vector: list[float],
    *,
    importance: int = 5,
    hours_ago: float = 0.0,
) -> MemoryRecord:
    when = NOW - timedelta(hours=hours_ago)
    record = MemoryRecord(
        id=record_id,
        user_id=user_id,
        text=text,
        importance=importance,
        created_at=when,
        last_accessed_at=when,
    )
    store.add(record, vector)
    return record


def turn(user: str = "随便问点什么", assistant: str = "好的") -> Turn:
    return Turn(user_text=user, assistant_text=assistant)


def facts_json(*facts: tuple[str, int]) -> str:
    return json.dumps([{"fact": f, "importance": i} for f, i in facts], ensure_ascii=False)


# ================================================================ 存储层


def test_inmemory_store_search_filters_by_user_and_sorts() -> None:
    store = InMemoryVectorStore()
    seed(store, "a1", "alice", "爱喝咖啡", [1.0, 0.0])
    seed(store, "a2", "alice", "住在上海", [0.7, 0.7])
    seed(store, "b1", "bob", "爱喝茶", [1.0, 0.0])
    hits = store.search("alice", [1.0, 0.0], k=5)
    assert [h.record.id for h in hits] == ["a1", "a2"]  # bob 的不出现;按相似度降序
    assert hits[0].relevance > hits[1].relevance


def test_inmemory_store_touch_and_delete_user() -> None:
    store = InMemoryVectorStore()
    seed(store, "a1", "alice", "爱喝咖啡", [1.0, 0.0])
    seed(store, "a2", "alice", "住在上海", [0.0, 1.0])
    later = NOW + timedelta(hours=1)
    store.touch(["a1", "不存在的id"], later)
    assert store.list_user("alice")[0].last_accessed_at == later
    assert store.delete_user("alice") == 2
    assert store.list_user("alice") == []


# ================================================================ 提炼与写入


def test_remember_adds_facts_to_empty_store_without_reconcile_call() -> None:
    llm = MockLLM([facts_json(("用户地址在上海", 8))])  # 只有提炼一次调用
    store = InMemoryVectorStore()
    ltm = make_ltm(llm, MockEmbedder(), store)
    ops = ltm.remember("alice", turn("我地址是上海", "好的已记录"))
    assert [op.action for op in ops] == ["ADD"]
    assert llm.call_count == 1  # 库为空 → 无候选 → 不做裁决调用
    assert store.list_user("alice")[0].text == "用户地址在上海"
    assert store.list_user("alice")[0].importance == 8


def test_remember_empty_facts_writes_nothing() -> None:
    llm = MockLLM(["[]"])
    embedder = MockEmbedder()
    store = InMemoryVectorStore()
    ops = make_ltm(llm, embedder, store).remember("alice", turn("你好", "您好"))
    assert ops == []
    assert embedder.seen_texts == []  # 没有事实,连 embedding 都不该调
    assert len(store) == 0


def test_remember_invalid_json_is_silent_noop() -> None:
    llm = MockLLM(["我觉得没什么好记的哦~"])  # 非 JSON
    store = InMemoryVectorStore()
    ops = make_ltm(llm, MockEmbedder(), store).remember("alice", turn())
    assert ops == []
    assert len(store) == 0


def test_remember_truncates_to_max_facts_and_clamps_importance() -> None:
    llm = MockLLM([facts_json(("事实一", 99), ("事实二", 0), ("事实三", 5), ("事实四", 5))])
    store = InMemoryVectorStore()
    ops = make_ltm(llm, MockEmbedder(), store).remember("alice", turn())
    assert len(ops) == 3  # 每轮最多 3 条
    assert ops[0].importance == 10 and ops[1].importance == 1  # 越界分数被夹回 1–10


def test_remember_parses_fenced_json() -> None:
    llm = MockLLM(['```json\n[{"fact": "用户是学生", "importance": 6}]\n```'])
    store = InMemoryVectorStore()
    ops = make_ltm(llm, MockEmbedder(), store).remember("alice", turn())
    assert [op.action for op in ops] == ["ADD"]


# ================================================================ 增删改裁决


def _similar_pair() -> MockEmbedder:
    """让「北京」与「上海」两条地址向量高度相似(同一件事),供裁决测试。"""
    return MockEmbedder(
        mapping={
            "用户地址在北京": [1.0, 0.0, 0.1],
            "用户地址在上海": [0.98, 0.0, 0.12],
        }
    )


def test_reconcile_update_overwrites_old_record() -> None:
    store = InMemoryVectorStore()
    seed(store, "old1", "alice", "用户地址在北京", [1.0, 0.0, 0.1], importance=8)
    llm = MockLLM(
        [
            facts_json(("用户地址在上海", 8)),
            json.dumps([{"index": 0, "action": "UPDATE", "target_id": "old1"}]),
        ]
    )
    ops = make_ltm(llm, _similar_pair(), store).remember("alice", turn("我搬到上海了", "好的"))
    assert ops[0].action == "UPDATE" and ops[0].target_id == "old1"
    records = store.list_user("alice")
    assert len(records) == 1  # 库里只剩一条
    assert records[0].id == "old1" and records[0].text == "用户地址在上海"  # id 不变,内容换新


def test_reconcile_delete_removes_old_without_adding_new() -> None:
    store = InMemoryVectorStore()
    seed(store, "old1", "alice", "订单 12345 的问题未解决", [1.0, 0.0, 0.0])
    llm = MockLLM(
        [
            facts_json(("订单 12345 的问题已解决", 5)),
            json.dumps([{"index": 0, "action": "DELETE", "target_id": "old1"}]),
        ]
    )
    embedder = MockEmbedder(mapping={"订单 12345 的问题已解决": [0.99, 0.0, 0.05]})
    ops = make_ltm(llm, embedder, store).remember("alice", turn("问题解决了", "太好了"))
    assert ops[0].action == "DELETE"
    assert store.list_user("alice") == []  # 旧的删了,新事实也不入库


def test_reconcile_noop_skips_duplicate() -> None:
    store = InMemoryVectorStore()
    seed(store, "old1", "alice", "用户地址在北京", [1.0, 0.0, 0.1])
    llm = MockLLM(
        [
            facts_json(("用户地址在北京", 8)),
            json.dumps([{"index": 0, "action": "NOOP"}]),
        ]
    )
    embedder = MockEmbedder(mapping={"用户地址在北京": [1.0, 0.0, 0.1]})
    ops = make_ltm(llm, embedder, store).remember("alice", turn())
    assert ops[0].action == "NOOP"
    assert len(store.list_user("alice")) == 1


def test_reconcile_fabricated_target_id_degrades_to_add() -> None:
    """模型编造不在候选里的 target_id → 矫正为 ADD,不误伤别的记忆。"""
    store = InMemoryVectorStore()
    seed(store, "old1", "alice", "用户地址在北京", [1.0, 0.0, 0.1])
    llm = MockLLM(
        [
            facts_json(("用户地址在上海", 8)),
            json.dumps([{"index": 0, "action": "UPDATE", "target_id": "编造的id"}]),
        ]
    )
    ops = make_ltm(llm, _similar_pair(), store).remember("alice", turn())
    assert ops[0].action == "ADD"
    assert len(store.list_user("alice")) == 2  # 旧的没被动


def test_reconcile_invalid_json_falls_back_noop_on_high_similarity() -> None:
    store = InMemoryVectorStore()
    seed(store, "old1", "alice", "用户地址在北京", [1.0, 0.0, 0.1])
    llm = MockLLM([facts_json(("用户地址在上海", 8)), "裁决不出来了"])  # 裁决非 JSON
    ops = make_ltm(llm, _similar_pair(), store).remember("alice", turn())
    # 「上海」与「北京」向量余弦 > 0.9 → 降级判重复,NOOP。
    assert ops[0].action == "NOOP"
    assert len(store.list_user("alice")) == 1


def test_reconcile_invalid_json_falls_back_add_on_low_similarity() -> None:
    store = InMemoryVectorStore()
    seed(store, "old1", "alice", "用户爱喝咖啡", [0.0, 1.0, 0.0])
    llm = MockLLM([facts_json(("用户地址在上海", 8)), "裁决不出来了"])
    ops = make_ltm(llm, _similar_pair(), store).remember("alice", turn())
    assert ops[0].action == "ADD"
    assert len(store.list_user("alice")) == 2


# ================================================================ 三因子检索


def _seed_rank_case(store: InMemoryVectorStore) -> None:
    # A:与 query 同向(rel=1.0),30 天没访问,importance=1
    seed(store, "a", "alice", "高相关旧记忆", [1.0, 0.0], importance=1, hours_ago=720)
    # B:相关性约 0.5,刚访问过,importance=10
    seed(store, "b", "alice", "低相关新记忆", [0.5, 0.866], importance=10, hours_ago=0)


def test_retrieve_weights_change_ranking() -> None:
    """构造「高相关但又旧又不重要」vs「低相关但新且重要」,验证权重决定排序。

    两个场景各用独立 store:检索会 touch 命中记录(保鲜),复用同一 store 会让
    第一次检索污染第二次的时近性。
    """
    embedder = MockEmbedder(mapping={"查询": [1.0, 0.0]})

    store1 = InMemoryVectorStore()
    _seed_rank_case(store1)
    only_relevance = make_ltm(
        MockLLM([]), embedder, store1, weight_recency=0.0, weight_importance=0.0
    )
    ids = [s.record.id for s in only_relevance.retrieve("alice", "查询")]
    assert ids[0] == "a"  # 纯相关性:A 赢

    store2 = InMemoryVectorStore()
    _seed_rank_case(store2)
    default_weights = make_ltm(MockLLM([]), embedder, store2)  # 1.0 / 0.5 / 0.5
    scored = default_weights.retrieve("alice", "查询")
    assert scored[0].record.id == "b"  # 时近+重要把 B 抬到第一
    # 分量可解释:各因子都在 0~1。
    for s in scored:
        assert 0.0 <= s.relevance <= 1.0 and 0.0 <= s.recency <= 1.0 and 0.0 <= s.importance <= 1.0


def test_retrieve_isolation_between_users() -> None:
    """红线:alice 的记忆,bob 检索必须 0 命中。"""
    store = InMemoryVectorStore()
    seed(store, "a1", "alice", "alice 的地址在上海", [1.0, 0.0])
    embedder = MockEmbedder(mapping={"地址在哪": [1.0, 0.0]})
    ltm = make_ltm(MockLLM([]), embedder, store)
    assert ltm.retrieve("bob", "地址在哪") == []
    assert [s.record.id for s in ltm.retrieve("alice", "地址在哪")] == ["a1"]


def test_retrieve_touches_last_accessed() -> None:
    store = InMemoryVectorStore()
    seed(store, "a1", "alice", "爱喝咖啡", [1.0, 0.0], hours_ago=48)
    embedder = MockEmbedder(mapping={"咖啡": [1.0, 0.0]})
    ltm = make_ltm(MockLLM([]), embedder, store)
    result = ltm.retrieve("alice", "咖啡")
    assert result[0].record.last_accessed_at == NOW  # 返回值已刷新
    assert store.list_user("alice")[0].last_accessed_at == NOW  # 存储里也刷新(保鲜)


def test_retrieve_returns_top_k_only() -> None:
    store = InMemoryVectorStore()
    for i in range(6):
        seed(store, f"m{i}", "alice", f"记忆{i}", [1.0, 0.01 * i])
    embedder = MockEmbedder(mapping={"查询": [1.0, 0.0]})
    ltm = make_ltm(MockLLM([]), embedder, store, top_k=2)
    assert len(ltm.retrieve("alice", "查询")) == 2


def test_retrieve_blank_query_returns_empty() -> None:
    ltm = make_ltm(MockLLM([]), MockEmbedder(), InMemoryVectorStore())
    assert ltm.retrieve("alice", "   ") == []


# ================================================================ 手动通道


def test_forget_and_delete_user() -> None:
    store = InMemoryVectorStore()
    seed(store, "a1", "alice", "记忆一", [1.0, 0.0])
    seed(store, "a2", "alice", "记忆二", [0.0, 1.0])
    ltm = make_ltm(MockLLM([]), MockEmbedder(), store)
    ltm.forget("a1")
    assert [r.id for r in ltm.list_memories("alice")] == ["a2"]
    assert ltm.delete_user("alice") == 1
    assert ltm.list_memories("alice") == []
