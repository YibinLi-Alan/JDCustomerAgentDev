"""长期记忆:LLM 提炼事实 → Mem0 式写入决策 → 三因子检索(阶段四 P-B)。

三条线(见 stage-4-design.md §7):

- **写入** :meth:`LongTermMemory.remember`:每轮结束后一次 LLM 调用提炼事实
  (四条规则:记身份/诉求/承诺,不记工具能查的易变状态;重要性 1–10;每轮 ≤3 条;
  严格 JSON,失败不写入)→ 对每条新事实召回相似旧记忆 → 一次 LLM 裁决
  ADD/UPDATE/DELETE/NOOP → 执行。裁决失败降级为「全 ADD + 高相似跳过」。
- **检索** :meth:`LongTermMemory.retrieve`:先按余弦召回 ``top_k × 4`` 候选
  (存储层已按 user_id 过滤),再三因子重排(Generative Agents 公式)::

      score = w_rel · relevance + w_rec · 0.5^(距上次访问小时数 / 半衰期) + w_imp · importance/10

  命中的记忆刷新 ``last_accessed_at``(「保鲜」)。
- **手动通道** :meth:`forget` / :meth:`delete_user` / :meth:`list_memories`。

一切失败都不炸对话流程:提炼失败 = 这轮不写入;检索依赖注入的组件,组件错误上抛
属编程错误(装配问题要炸在装配时)。
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Callable, Literal

from agent_framework.core.llm import LLM, Message
from agent_framework.memory.embedder import Embedder
from agent_framework.memory.short_term import Turn
from agent_framework.memory.vector_store import MemoryRecord, SearchHit, VectorStore

EXTRACT_SYSTEM = (
    "你是客服系统的记忆提炼器。从「本轮对话」中提取值得跨会话记住的用户事实。\n"
    "【记什么】用户身份/偏好/常用地址;对某订单或商品的诉求与情绪;客服做出的承诺;未解决的问题。\n"
    "【不记什么】闲聊寒暄;工具随时能查到的易变状态——记「用户关心订单 12345 的进度」,"
    "不记「订单 12345 已发货」(状态会过期)。\n"
    "【重要性】1-3 琐事;4-7 偏好与一般事实;8-10 身份地址、投诉、承诺。\n"
    "【输出】只输出 JSON 数组,最多 3 条,没有值得记的输出 []。格式:\n"
    '[{"fact": "用户的常用收货地址在上海浦东", "importance": 8}]'
)

RECONCILE_SYSTEM = (
    "你是客服系统的记忆管理器。对每条「新事实」,对照给出的「相似旧记忆」裁决一个动作:\n"
    "- ADD:全新事实,与旧记忆都不是一回事 → 入库;\n"
    "- UPDATE:与某条旧记忆是同一件事但内容更新或矛盾(如地址变了)→ 用新文本覆盖旧条,"
    "需给出 target_id;\n"
    "- DELETE:新信息表明某条旧记忆已失效且新事实本身无需保存(如问题已解决)→ 删旧条,"
    "需给出 target_id;\n"
    "- NOOP:与旧记忆重复,什么都不做。\n"
    "【输出】只输出 JSON 数组,每条新事实一项,按输入顺序。格式:\n"
    '[{"index": 0, "action": "UPDATE", "target_id": "abc123"}, {"index": 1, "action": "ADD"}]'
)

WriteAction = Literal["ADD", "UPDATE", "DELETE", "NOOP"]


@dataclass(frozen=True)
class WriteOp:
    """一次写入决策的执行结果(供 CLI /trace 展示与测试断言)。

    Attributes:
        action: 执行的动作。
        fact: 触发本动作的新事实文本。
        importance: 新事实的重要性分。
        target_id: UPDATE/DELETE 作用到的旧记录 id(ADD/NOOP 为 None)。
        record_id: 落库的记录 id(ADD 是新 id,UPDATE 是被覆盖的旧 id;DELETE/NOOP 为 None)。
    """

    action: WriteAction
    fact: str
    importance: int
    target_id: str | None = None
    record_id: str | None = None


@dataclass(frozen=True)
class ScoredMemory:
    """检索结果:记录 + 三因子总分与各分量(可解释,评测与 /memories 展示用)。"""

    record: MemoryRecord
    score: float
    relevance: float
    recency: float
    importance: float


@dataclass(frozen=True)
class _Fact:
    text: str
    importance: int


class LongTermMemory:
    """长期记忆的读写门面:提炼、裁决、三因子检索、手动增删。"""

    def __init__(
        self,
        llm: LLM,
        embedder: Embedder,
        store: VectorStore,
        *,
        top_k: int = 3,
        weight_relevance: float = 1.0,
        weight_recency: float = 0.5,
        weight_importance: float = 0.5,
        half_life_hours: float = 24.0,
        dedup_threshold: float = 0.9,
        max_facts_per_turn: int = 3,
        candidate_multiplier: int = 4,
        now_fn: Callable[[], datetime] = datetime.now,
    ) -> None:
        """构造长期记忆。所有策略参数来自 ``config.py``,不硬编码。

        Args:
            llm: 提炼与裁决用的 LLM(与主循环同一抽象;测试给 MockLLM)。
            embedder: 文本向量化实现。
            store: 向量存储实现(Chroma / 内存版均可)。
            top_k: 检索最终返回条数。
            weight_relevance: 三因子之相关性权重(主排序)。
            weight_recency: 三因子之时近性权重。
            weight_importance: 三因子之重要性权重。
            half_life_hours: 时近性指数衰减的半衰期(小时)。
            dedup_threshold: 裁决降级时的去重阈值(候选相关性 ≥ 此值视为重复)。
            max_facts_per_turn: 每轮最多写入的事实条数(防灌库)。
            candidate_multiplier: 检索召回倍数(召回 ``top_k × 此值`` 再重排)。
            now_fn: 取当前时间的函数(测试注入假时钟)。
        """
        self._llm = llm
        self._embedder = embedder
        self._store = store
        self.top_k = top_k
        self.weight_relevance = weight_relevance
        self.weight_recency = weight_recency
        self.weight_importance = weight_importance
        self.half_life_hours = half_life_hours
        self.dedup_threshold = dedup_threshold
        self.max_facts_per_turn = max_facts_per_turn
        self.candidate_multiplier = candidate_multiplier
        self._now = now_fn

    # ------------------------------------------------------------- 写入

    def remember(self, user_id: str, turn: Turn) -> list[WriteOp]:
        """从一轮对话提炼事实并写入(含增删改裁决)。**永不抛业务异常**。

        Args:
            user_id: 归属用户(由框架注入,不来自模型)。
            turn: 刚结束的一轮对话。

        Returns:
            实际执行的写入操作列表(闲聊轮 / 提炼失败时为空)。
        """
        facts = self._extract_facts(turn)
        if not facts:
            return []
        vectors = self._embedder.embed([f.text for f in facts])
        candidates = [self._store.search(user_id, vec, self.max_facts_per_turn) for vec in vectors]
        if any(cands for cands in candidates):
            decisions = self._reconcile(facts, candidates)
        else:
            # 库里没有任何相似旧记忆:全部 ADD,省一次 LLM 调用。
            decisions = [("ADD", None)] * len(facts)
        return self._execute(user_id, facts, vectors, candidates, decisions)

    def _extract_facts(self, turn: Turn) -> list[_Fact]:
        """提炼事实。JSON 非法 / LLM 失败 → 返回空(这轮不写入,宁缺勿滥)。"""
        prompt = f"本轮对话:\n{turn.render_text()}\n\n请输出 JSON 数组。"
        try:
            response = self._llm.chat([Message(role="user", content=prompt)], system=EXTRACT_SYSTEM)
            items = _parse_json_array(response.content)
        except Exception:  # noqa: BLE001 — 提炼失败不炸对话
            return []
        if items is None:
            return []
        facts: list[_Fact] = []
        for item in items[: self.max_facts_per_turn]:
            if not isinstance(item, dict):
                continue
            text = str(item.get("fact", "")).strip()
            if not text:
                continue
            try:
                importance = int(item.get("importance", 5))
            except (TypeError, ValueError):
                importance = 5
            facts.append(_Fact(text=text, importance=max(1, min(10, importance))))
        return facts

    def _reconcile(
        self, facts: list[_Fact], candidates: list[list[SearchHit]]
    ) -> list[tuple[WriteAction, str | None]]:
        """一次 LLM 调用批量裁决所有新事实。失败 → 降级「全 ADD + 高相似跳过」。"""
        lines: list[str] = []
        for i, (fact, cands) in enumerate(zip(facts, candidates)):
            lines.append(f"新事实[{i}]:{fact.text}")
            if cands:
                for hit in cands:
                    lines.append(f"  - 旧记忆(id={hit.record.id}):{hit.record.text}")
            else:
                lines.append("  - (无相似旧记忆)")
        prompt = "\n".join(lines) + "\n\n请对每条新事实输出裁决 JSON 数组。"
        try:
            response = self._llm.chat(
                [Message(role="user", content=prompt)], system=RECONCILE_SYSTEM
            )
            items = _parse_json_array(response.content)
            decisions = self._parse_decisions(items, facts, candidates)
            if decisions is not None:
                return decisions
        except Exception:  # noqa: BLE001 — 裁决失败必须降级,不能炸对话
            pass
        return self._fallback_decisions(candidates)

    def _parse_decisions(
        self,
        items: list[object] | None,
        facts: list[_Fact],
        candidates: list[list[SearchHit]],
    ) -> list[tuple[WriteAction, str | None]] | None:
        """校验裁决输出;整体不可解析返回 None(触发降级),单条不合法就地矫正为 ADD。"""
        if items is None:
            return None
        by_index: dict[int, tuple[WriteAction, str | None]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if not 0 <= index < len(facts):
                continue
            action = str(item.get("action", "")).upper()
            target_id = item.get("target_id")
            target = str(target_id) if target_id else None
            if action in ("UPDATE", "DELETE"):
                # target 必须真的在这条事实的候选里 —— 防模型编造 id 误伤别的记忆。
                valid_ids = {hit.record.id for hit in candidates[index]}
                if target not in valid_ids:
                    action, target = "ADD", None
            if action not in ("ADD", "UPDATE", "DELETE", "NOOP"):
                action, target = "ADD", None
            by_index[index] = (action, target)  # type: ignore[assignment]
        return [by_index.get(i, ("ADD", None)) for i in range(len(facts))]

    def _fallback_decisions(
        self, candidates: list[list[SearchHit]]
    ) -> list[tuple[WriteAction, str | None]]:
        """降级策略:候选里有高相似(≥ dedup_threshold)→ NOOP,否则 ADD。"""
        decisions: list[tuple[WriteAction, str | None]] = []
        for cands in candidates:
            top = max((hit.relevance for hit in cands), default=0.0)
            decisions.append(("NOOP", None) if top >= self.dedup_threshold else ("ADD", None))
        return decisions

    def _execute(
        self,
        user_id: str,
        facts: list[_Fact],
        vectors: list[list[float]],
        candidates: list[list[SearchHit]],
        decisions: list[tuple[WriteAction, str | None]],
    ) -> list[WriteOp]:
        now = self._now()
        ops: list[WriteOp] = []
        for fact, vector, _cands, (action, target) in zip(facts, vectors, candidates, decisions):
            if action == "ADD":
                record = MemoryRecord(
                    id=uuid.uuid4().hex[:12],
                    user_id=user_id,
                    text=fact.text,
                    importance=fact.importance,
                    created_at=now,
                    last_accessed_at=now,
                )
                self._store.add(record, vector)
                ops.append(
                    WriteOp(
                        action="ADD",
                        fact=fact.text,
                        importance=fact.importance,
                        record_id=record.id,
                    )
                )
            elif action == "UPDATE" and target:
                # id 不变、文本与重要性换新、时间刷新(评审拍板)。
                record = MemoryRecord(
                    id=target,
                    user_id=user_id,
                    text=fact.text,
                    importance=fact.importance,
                    created_at=now,
                    last_accessed_at=now,
                )
                self._store.update(record, vector)
                ops.append(
                    WriteOp(
                        action="UPDATE",
                        fact=fact.text,
                        importance=fact.importance,
                        target_id=target,
                        record_id=target,
                    )
                )
            elif action == "DELETE" and target:
                self._store.delete([target])
                ops.append(
                    WriteOp(
                        action="DELETE",
                        fact=fact.text,
                        importance=fact.importance,
                        target_id=target,
                    )
                )
            else:  # NOOP
                ops.append(WriteOp(action="NOOP", fact=fact.text, importance=fact.importance))
        return ops

    # ------------------------------------------------------------- 检索

    def retrieve(self, user_id: str, query: str) -> list[ScoredMemory]:
        """三因子检索该用户的相关记忆,并给命中记录「保鲜」。

        Args:
            user_id: 当前用户(框架注入,模型不可见)。
            query: 当前用户输入(检索 query)。

        Returns:
            按总分降序的 top-k 记忆(带各因子分量,可解释)。
        """
        if not query.strip():
            return []
        vector = self._embedder.embed([query])[0]
        hits = self._store.search(user_id, vector, self.top_k * self.candidate_multiplier)
        if not hits:
            return []
        now = self._now()
        scored = [self._score(hit, now) for hit in hits]
        scored.sort(key=lambda s: s.score, reverse=True)
        top = scored[: self.top_k]
        self._store.touch([s.record.id for s in top], now)
        return [replace(s, record=replace(s.record, last_accessed_at=now)) for s in top]

    def _score(self, hit: SearchHit, now: datetime) -> ScoredMemory:
        hours = max(0.0, (now - hit.record.last_accessed_at).total_seconds() / 3600.0)
        recency = 0.5 ** (hours / self.half_life_hours) if self.half_life_hours > 0 else 0.0
        importance = hit.record.importance / 10.0
        score = (
            self.weight_relevance * hit.relevance
            + self.weight_recency * recency
            + self.weight_importance * importance
        )
        return ScoredMemory(
            record=hit.record,
            score=score,
            relevance=hit.relevance,
            recency=recency,
            importance=importance,
        )

    # ------------------------------------------------------------- 手动通道

    def forget(self, record_id: str) -> None:
        """删除单条记忆(CLI ``/forget <id>``)。"""
        self._store.delete([record_id])

    def delete_user(self, user_id: str) -> int:
        """一键清空该用户全部记忆(「删除我的个人信息」),返回删除条数。"""
        return self._store.delete_user(user_id)

    def list_memories(self, user_id: str) -> list[MemoryRecord]:
        """列出该用户全部记忆(CLI ``/memories``)。"""
        return self._store.list_user(user_id)


def _parse_json_array(text: str) -> list[object] | None:
    """从模型输出解析 JSON 数组;容忍 ``` 围栏;不是数组返回 None。"""
    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None
