"""极简 ``Tool`` 协议 —— 阶段二的最小工具抽象。

本模块只定义 Agent 在 ReAct 循环里「知道有哪些工具、怎么调、调完拿到文字结果」
所需的**最小**契约(见 stage-2-design.md §5.1)。它刻意保持极简:

- 只声明 ``name`` / ``description`` 两个属性和一个 ``run`` 方法;
- **不做**参数 JSON-Schema 校验 —— 参数不对就让 ``run`` 自然抛错,交给上层
  ReAct 循环的错误恢复(把异常当 Observation 喂回模型)。

设计立场:核心循环只依赖这个 ``Tool`` 协议,**加一个新能力 = 新写一个满足该协议
的类并装配进去**,``core/agent.py`` 一行不用改。

> **这是阶段二的最小前身。** 阶段三会把它扩成完整的 ``BaseTool``
> (ABC + Pydantic/JSON-Schema 参数校验 + 统一错误规范),并新增 ``ToolRegistry``;
> 届时 MCP 适配器等「可插拔工具源」也从这个口子接入,核心循环保持不变。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    """一个工具的最小协议:模型可据此选择、调用工具并拿到文字结果。

    满足本协议只需提供两个属性和一个方法,无需继承任何基类
    (``Protocol`` 是结构化类型:长得像即可)。

    Attributes:
        name: 工具的唯一名字,模型在 ``action.tool`` 里用它指定要调哪个工具。
        description: 给模型看的说明 —— 这工具干嘛用、何时用、需要什么参数。
            会被拼进 system prompt 的工具清单,写清楚能显著提高模型的调用准确率。
    """

    name: str
    description: str

    def run(self, **kwargs: object) -> str:
        """执行工具,返回**喂回给模型**的文字结果。

        参数以关键字形式传入(对应模型输出的 ``action.input`` 字典)。本阶段
        **不做**参数校验:若缺少必需参数,Python 会自然抛 ``TypeError``,由上层
        ReAct 循环捕获并当作 Observation 喂回,让模型自我纠正(见 stage-2-design.md §4.6)。

        Args:
            **kwargs: 工具所需的参数,来自模型 ``action.input`` 的键值对。

        Returns:
            工具执行结果的文字表述,将作为 Observation 追加进上下文喂回模型。
        """
        ...
