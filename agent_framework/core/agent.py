"""Agent 核心(阶段二实现,本阶段仅占位)。

阶段二将在此落地 **ReAct 循环**(``Thought -> Action -> Observation``):
最大步数保护、结构化输出解析、错误恢复等(见 frame/Agent框架实习培训大纲.md「阶段二」)。

设计要点(已在阶段一确立,见 stage-1-design.md §9):
- ``Agent`` 只依赖 :class:`agent_framework.core.llm.LLM` 接口,**不**直接依赖任何具体厂商 SDK;
  这样将来换模型 / 换厂商时,Agent 与 ReAct 循环零改动。
- system prompt 经 ``LLM.chat`` / ``LLM.stream`` 的 ``system`` 参数传入。

本文件当前不含实质逻辑,仅声明未来的依赖方向与接口形态,避免阶段二落地时重构目录。

# TODO(阶段二): 实现 Agent 基类 + ReAct Loop。预期形态大致如下:
#
#     from agent_framework.core.llm import LLM, Message
#
#     class Agent:
#         def __init__(self, llm: LLM, *, system: str | None = None,
#                      max_steps: int = 10) -> None:
#             self._llm = llm          # 仅依赖 LLM 接口
#             self._system = system
#             self._max_steps = max_steps
#
#         def run(self, task: str) -> str:
#             '''执行一次 ReAct 循环并返回最终答案(阶段二实现)。'''
#             raise NotImplementedError
"""

from __future__ import annotations

__all__: list[str] = []
