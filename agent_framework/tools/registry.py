"""``ToolRegistry`` —— 工具的注册、发现、管理中枢(见 stage-3-design.md §6)。

Agent 不再拿一个工具 list,而是拿一个 Registry:

- **装配时**:``register()`` 重名默认报错(装配 bug 要炸在装配时,不要炸在运行时);
- **运行时**:``invoke(name, args)`` 是模型驱动的统一入口,未知工具/执行失败都
  折叠成 ``ToolResult(ok=False)``,循环永不崩;
- **对接 LLM**:``to_schemas()`` 批量导出厂商无关 Schema(P-B 传给 LLM 的 tools
  参数);``render_catalog()`` 渲染文本工具清单(阶段二文本版 system prompt 用)。
"""

from __future__ import annotations

from typing import Iterable, Iterator, Protocol, runtime_checkable

from agent_framework.tools.base import BaseTool, ToolError, ToolResult


@runtime_checkable
class ToolRegistryLike(Protocol):
    """工具库的**运行时接口** —— Agent 循环与装饰器共同遵守的最小契约。

    ``ToolRegistry`` 是它的规范实现;``safety`` 的装饰器(``ApprovalGate`` /
    ``BoundaryRegistry``)也实现它,从而能透明包装真实 Registry——上层(专员 /
    编排 / ToolCallingAgent)只依赖本协议,不关心手里的 registry 被包过几层。
    这把原先的「鸭子约定」升级为正式接口(阶段六代码审查补强)。
    """

    def invoke(
        self, name: str, args: dict[str, object] | None = None, *, request_id: str | None = None
    ) -> ToolResult: ...

    def to_schemas(self) -> list[dict[str, object]]: ...

    def get(self, name: str) -> BaseTool: ...

    def render_catalog(self) -> str: ...

    @property
    def names(self) -> list[str]: ...


class ToolRegistrationError(ToolError):
    """注册冲突:同名工具已存在且未指定 ``replace=True``。"""


class UnknownToolError(ToolError):
    """按名字查找的工具不存在。"""


class ToolRegistry:
    """工具注册中心:``name → BaseTool`` 的显式映射,像容器一样用。

    支持 ``in`` / ``len()`` / 迭代::

        registry = ToolRegistry([QueryOrderTool(), QueryLogisticsTool()])
        "query_order" in registry     # True
        for tool in registry: ...     # 迭代 BaseTool 实例
    """

    def __init__(self, tools: Iterable[BaseTool] = ()) -> None:
        """构造并批量注册初始工具。

        Args:
            tools: 初始工具序列,逐个走 :meth:`register`(重名同样报错)。
        """
        self._tools: dict[str, BaseTool] = {}
        for tool in tools:
            self.register(tool)

    # ------------------------------ 装配接口 ------------------------------ #
    def register(self, tool: BaseTool, *, replace: bool = False) -> None:
        """注册一个工具。

        Args:
            tool: 待注册的工具实例;``name`` 不能为空。
            replace: 为 ``True`` 时允许覆盖同名工具(默认不允许)。

        Raises:
            ToolRegistrationError: ``name`` 为空,或重名且未指定 ``replace``。
        """
        name = getattr(tool, "name", "")
        if not name:
            raise ToolRegistrationError(f"工具 {tool!r} 缺少非空的 name,无法注册。")
        if name in self._tools and not replace:
            raise ToolRegistrationError(f"工具名 {name!r} 已被注册;如确要覆盖请传 replace=True。")
        self._tools[name] = tool

    def unregister(self, name: str) -> None:
        """移除一个已注册的工具。

        Raises:
            UnknownToolError: 该名字未注册。
        """
        if name not in self._tools:
            raise UnknownToolError(f"工具 {name!r} 未注册,无法移除。可用工具:{self.names}")
        del self._tools[name]

    # ------------------------------ 查找接口 ------------------------------ #
    def get(self, name: str) -> BaseTool:
        """按名字取工具(**编程路径**,查不到直接抛错)。

        Raises:
            UnknownToolError: 该名字未注册,消息附上可用工具列表。
        """
        tool = self._tools.get(name)
        if tool is None:
            raise UnknownToolError(f"工具 {name!r} 不存在。可用工具:{self.names}")
        return tool

    def subset(self, *names: str) -> ToolRegistry:
        """按名字挑选工具,组成新的 Registry(阶段五:给专员配工具子集)。

        工具**实例共享**不复制——业务工具背后是同一个 ``JDMockStore``,
        跨专员的状态变化(如售后专员退了款)对其他专员的查询可见。

        Args:
            names: 要挑选的工具名(按给定顺序装配)。

        Returns:
            只含指定工具的新 :class:`ToolRegistry`。

        Raises:
            UnknownToolError: 任一名字未注册(装配期即失败,不留到运行时)。
        """
        return ToolRegistry([self.get(name) for name in names])

    @property
    def names(self) -> list[str]:
        """已注册的工具名列表(按注册顺序)。"""
        return list(self._tools)

    @property
    def tools(self) -> list[BaseTool]:
        """已注册的工具实例列表(按注册顺序)。"""
        return list(self._tools.values())

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[BaseTool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    # ------------------------------ 运行时接口 ------------------------------ #
    def invoke(
        self,
        name: str,
        args: dict[str, object] | None = None,
        *,
        request_id: str | None = None,
    ) -> ToolResult:
        """模型驱动的统一调用入口:未知工具与执行失败都折叠为 ``ok=False``。

        与 :meth:`BaseTool.invoke` 同构,Agent 循环只需要这一个方法。

        Args:
            name: 模型指定的工具名。
            args: 模型生成的参数字典。
            request_id: 幂等请求 ID,透传给 :meth:`BaseTool.invoke`
                (框架生成,HITL 审批执行等可重放路径用;模型路径不传)。

        Returns:
            标准化的 :class:`ToolResult`,**永不抛异常**。
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, error=f"工具 {name!r} 不存在。可用工具:{self.names}")
        return tool.invoke(args, request_id=request_id)

    # ------------------------------ 对接 LLM ------------------------------ #
    def to_schemas(self) -> list[dict[str, object]]:
        """批量导出厂商无关的工具 Schema(P-B 直接传给 LLM 的 tools 参数)。"""
        return [tool.to_schema() for tool in self._tools.values()]

    def render_catalog(self) -> str:
        """渲染文本版工具清单(``- name: description`` 每行一个)。

        供阶段二文本解析式 system prompt 使用;没有工具时返回提示句。
        """
        if not self._tools:
            return "(当前没有可用工具,只能直接作答)"
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())
