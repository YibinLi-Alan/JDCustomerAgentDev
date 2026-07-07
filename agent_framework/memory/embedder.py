"""Embedding 接口与 OpenAI 实现(阶段四 P-B)。

与 ``core/llm.py`` 同一纪律:核心只依赖 :class:`Embedder` Protocol,
``openai`` SDK 只在具体实现里延迟导入。换 embedding 提供方 = 换实现类。
测试用 ``tests/mock_embedder.py`` 的确定性假向量,零网络零成本。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence

if TYPE_CHECKING:
    from agent_framework.core.config import Settings


class Embedder(Protocol):
    """文本向量化接口:一批文本 → 一批同维向量。"""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """把 ``texts`` 逐条向量化,返回与输入同序的向量列表。"""
        ...


class OpenAIEmbedder:
    """OpenAI Embeddings 实现(默认 ``text-embedding-3-small``,便宜)。

    Attributes:
        model: 实际使用的 embedding 模型 id。
    """

    def __init__(self, settings: Settings) -> None:
        """从配置构造;缺 API key 时启动即失败,给出明确提示。

        Args:
            settings: 框架配置;用 ``openai_api_key`` 与 ``embedding_model``。
        """
        if not settings.openai_api_key:
            raise ValueError("缺少 OPENAI_API_KEY:embedding 走 OpenAI 接口,请在 .env 中配置。")
        from openai import OpenAI  # 延迟导入:只有用到本实现才需要装 openai

        self._client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """批量向量化(一次 API 调用),按输入顺序返回。

        Args:
            texts: 待向量化的文本;空列表直接返回空。
        """
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.model, input=list(texts))
        # API 保证 data 带 index;按 index 排序以防乱序。
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]
