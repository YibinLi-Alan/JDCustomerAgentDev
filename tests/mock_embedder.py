"""确定性假 Embedder —— 长期记忆单测的测试基础设施(离线、零成本)。

两种用法:

- **映射模式**:构造时给 ``mapping={文本: 向量}``,测试完全掌控谁和谁相似
  (比如让「地址在上海」与「地址在北京」余弦很高,验证 UPDATE 裁决路径);
- **兜底模式**:未映射的文本用「字符哈希桶」生成确定性向量 —— 同文本同向量,
  不同文本大概率不同,足够支撑「无相似旧记忆」的分支。
"""

from __future__ import annotations

from typing import Sequence


class MockEmbedder:
    """满足 ``Embedder`` 协议的确定性假实现。"""

    def __init__(self, mapping: dict[str, list[float]] | None = None, *, dim: int = 8) -> None:
        """构造。

        Args:
            mapping: 文本 → 向量的显式映射(优先命中)。
            dim: 兜底哈希向量的维数(需与 mapping 里向量同维)。
        """
        self._mapping = dict(mapping or {})
        self._dim = dim
        self.seen_texts: list[str] = []  # 供断言「谁被向量化了/调用了几次」

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.seen_texts.extend(texts)
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        if text in self._mapping:
            return list(self._mapping[text])
        vec = [0.0] * self._dim
        for ch in text:
            vec[ord(ch) % self._dim] += 1.0
        return vec
