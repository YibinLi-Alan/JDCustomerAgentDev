"""Memory 子包:短期记忆(滑动窗口)+ 长期记忆(向量检索)+ 压缩 + 统一管理。

阶段四交付(见 stage-4-design.md)。P-A:短期记忆与压缩;P-B:长期记忆;
P-C:MemoryManager 统一门面。
"""

from agent_framework.memory.compressor import SummaryCompressor
from agent_framework.memory.embedder import Embedder, OpenAIEmbedder
from agent_framework.memory.long_term import LongTermMemory, ScoredMemory, WriteOp
from agent_framework.memory.manager import (
    MemoryContext,
    MemoryManager,
    TurnReport,
    create_memory_manager,
)
from agent_framework.memory.short_term import (
    HeuristicTokenCounter,
    ShortTermMemory,
    TokenCounter,
    Turn,
)
from agent_framework.memory.vector_store import (
    ChromaVectorStore,
    InMemoryVectorStore,
    MemoryRecord,
    SearchHit,
    VectorStore,
)

__all__ = [
    "ChromaVectorStore",
    "Embedder",
    "HeuristicTokenCounter",
    "InMemoryVectorStore",
    "LongTermMemory",
    "MemoryContext",
    "MemoryManager",
    "MemoryRecord",
    "OpenAIEmbedder",
    "ScoredMemory",
    "SearchHit",
    "ShortTermMemory",
    "SummaryCompressor",
    "TokenCounter",
    "Turn",
    "TurnReport",
    "VectorStore",
    "WriteOp",
    "create_memory_manager",
]
