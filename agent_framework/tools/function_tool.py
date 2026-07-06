"""``@tool`` 装饰器 —— 普通函数一行变工具(借鉴 LangChain,见 stage-3-design.md §7)。

用法::

    @tool(timeout=3.0)
    def calculator(expression: str) -> str:
        \"\"\"数学计算器。何时用:需要精确计算金额、差价、折扣时。\"\"\"
        ...

- ``name`` ← 函数名(可用 ``name=`` 覆盖);
- ``description`` ← docstring(**没有 docstring 直接报错** —— 描述决定模型会不会用对工具);
- ``args_schema`` ← 从**类型注解**用 ``pydantic.create_model`` 自动推断
  (参数缺注解 → 报错;参数描述可用 ``Annotated[str, Field(description=...)]`` 补充)。

产物是 :class:`FunctionTool`(``BaseTool`` 子类)实例,与手写类完全同构,照常进 Registry。
"""

from __future__ import annotations

import inspect
from typing import Callable, get_type_hints, overload

from pydantic import BaseModel, create_model

from agent_framework.tools.base import BaseTool


def _schema_from_signature(func: Callable[..., object], tool_name: str) -> type[BaseModel] | None:
    """从函数签名的类型注解推断参数的 Pydantic 模型。

    Args:
        func: 被装饰的函数。
        tool_name: 工具名(用于报错信息与模型命名)。

    Returns:
        推断出的 Pydantic 模型;函数无参数时返回 ``None``。

    Raises:
        ValueError: 存在缺少类型注解的参数,或出现 ``*args`` / ``**kwargs``。
    """
    signature = inspect.signature(func)
    # include_extras=True 保留 Annotated[..., Field(description=...)] 里的元数据
    hints = get_type_hints(func, include_extras=True)

    fields: dict[str, object] = {}
    for param_name, param in signature.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise ValueError(
                f"@tool 工具 {tool_name!r} 不支持 *args/**kwargs 参数(模型无法生成不定参)。"
            )
        if param_name not in hints:
            raise ValueError(
                f"@tool 工具 {tool_name!r} 的参数 {param_name!r} 缺少类型注解;"
                "类型注解是生成 JSON Schema 的依据,必须写。"
            )
        default = ... if param.default is param.empty else param.default
        fields[param_name] = (hints[param_name], default)

    if not fields:
        return None
    return create_model(f"{tool_name}_args", **fields)  # type: ignore[call-overload]


class FunctionTool(BaseTool):
    """把一个普通函数包装成 ``BaseTool``(通常经由 :func:`tool` 构造)。"""

    def __init__(
        self,
        func: Callable[..., object],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool = True,
        timeout: float | None = None,
    ) -> None:
        """包装函数为工具。

        Args:
            func: 业务函数,参数必须全部带类型注解。
            name: 工具名;默认取函数名。
            description: 工具说明;默认取函数 docstring。
            strict: strict mode 开关(语义见 :class:`BaseTool`)。
            timeout: 执行超时秒数;``None`` 不限时。

        Raises:
            ValueError: 既没有 ``description`` 也没有 docstring,或签名无法推断 Schema。
        """
        self.name = name or func.__name__
        desc = description or inspect.getdoc(func) or ""
        if not desc.strip():
            raise ValueError(
                f"@tool 工具 {self.name!r} 缺少 description(写在 docstring 里):"
                "工具描述决定模型会不会用对它,不允许省略。"
            )
        self.description = desc.strip()
        self.strict = strict
        self.timeout = timeout
        self.args_schema = _schema_from_signature(func, self.name)
        self._func = func

    def _run(self, **kwargs: object) -> object:
        """直接把已校验的参数转发给被包装的函数。"""
        return self._func(**kwargs)


@overload
def tool(func: Callable[..., object]) -> FunctionTool: ...
@overload
def tool(
    *,
    name: str | None = ...,
    description: str | None = ...,
    strict: bool = ...,
    timeout: float | None = ...,
) -> Callable[[Callable[..., object]], FunctionTool]: ...


def tool(
    func: Callable[..., object] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    strict: bool = True,
    timeout: float | None = None,
) -> FunctionTool | Callable[[Callable[..., object]], FunctionTool]:
    """装饰器:把带类型注解 + docstring 的函数变成 :class:`FunctionTool`。

    支持两种写法::

        @tool                       # 无参形式
        def current_time() -> str: ...

        @tool(timeout=3.0)          # 带参形式
        def calculator(expression: str) -> str: ...

    Args:
        func: 无参形式下被装饰的函数(带参形式为 ``None``)。
        name: 覆盖工具名(默认函数名)。
        description: 覆盖工具说明(默认 docstring)。
        strict: strict mode 开关,默认开。
        timeout: 执行超时秒数。

    Returns:
        :class:`FunctionTool` 实例,或(带参形式)一个再接收函数的装饰器。
    """

    def wrap(f: Callable[..., object]) -> FunctionTool:
        return FunctionTool(f, name=name, description=description, strict=strict, timeout=timeout)

    return wrap(func) if func is not None else wrap
