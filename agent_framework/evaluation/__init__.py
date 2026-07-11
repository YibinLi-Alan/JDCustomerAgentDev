"""Evaluation 子包:评测集与评测脚本。

- 阶段四:``memory_retrieval_eval``(不同记忆策略的检索准确率对比)。
- 阶段六:``judge``(LLM-as-Judge,四维打分 + A/B 对比防位置偏见)、
  ``agent_eval``(端到端评测 pipeline:跑集 → 裁判 → 聚合报告)。
"""

from agent_framework.evaluation.judge import (
    Judge,
    Judgement,
    PairwiseResult,
    compare_pairwise,
)

__all__ = ["Judge", "Judgement", "PairwiseResult", "compare_pairwise"]
