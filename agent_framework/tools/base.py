"""工具抽象层:``Tool`` 协议(阶段二遗留)+ 完整 ``BaseTool``(阶段三)。

本模块是工具子系统的地基(见 stage-3-design.md §5):

- :class:`Tool`:阶段二的极简协议,ReAct 循环对工具的最小契约(``name`` /
  ``description`` / ``run() -> str``),继续保留,循环不改。
- :class:`BaseTool`:阶段三的正式抽象基类。子类只声明
  ``name`` / ``description`` / ``args_schema``(Pydantic)并实现 ``_run()``;
  参数校验(含 **strict mode**)、超时控制、异常捕获、结果标准化全部由基类的
  :meth:`BaseTool.invoke` 管线统一处理。
- :class:`ToolResult`:执行结果的标准化返回 —— 成功/失败都是数据,**永不向
  循环抛异常**,失败原因喂回模型让其自我纠正(与阶段二错误恢复一脉相承)。

设计立场:``BaseTool.run()`` 仍返回 ``str``,**结构性满足** :class:`Tool` 协议,
所以新工具可直接插进现有 ``ReActAgent``;本模块对 LLM 厂商零感知,厂商格式转换
(Claude ``input_schema`` / OpenAI ``function``)放在 llm 层(P-B)。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

#: 工具的权限级别(阶段六 safety/HITL 依据它决定「直接执行 / 触发人工审批」):
#: - ``low``:只读查询,Agent 直接执行;
#: - ``medium``:有外部副作用但可控(如调外部 HTTP);
#: - ``high``:写操作 / 资金相关(退款、取消订单、建工单),阶段六接人工审批。
Permission = Literal["low", "medium", "high"]


@runtime_checkable
class Tool(Protocol):
    """一个工具的最小协议:模型可据此选择、调用工具并拿到文字结果。

    阶段二的遗留契约,``ReActAgent`` 只依赖它。满足本协议只需提供两个属性和
    一个方法,无需继承任何基类(``Protocol`` 是结构化类型:长得像即可)。
    :class:`BaseTool` 的实例结构性满足本协议。

    Attributes:
        name: 工具的唯一名字,模型在 ``action.tool`` 里用它指定要调哪个工具。
        description: 给模型看的说明 —— 这工具干嘛用、何时用、需要什么参数。
    """

    name: str
    description: str

    def run(self, **kwargs: object) -> str:
        """执行工具,返回**喂回给模型**的文字结果。"""
        ...


# --------------------------------------------------------------------------- #
# 错误类型(见 stage-3-design.md §5.3)                                          #
# --------------------------------------------------------------------------- #
class ToolError(Exception):
    """工具子系统所有错误的基类。"""


class ToolValidationError(ToolError):
    """参数校验失败(缺必填 / 类型不符 / strict 下出现未知参数)。"""


class ToolTimeoutError(ToolError):
    """工具执行超过 ``timeout`` 秒未返回。"""


# --------------------------------------------------------------------------- #
# 标准化执行结果                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class ToolResult:
    """一次工具调用的标准化结果 —— 成功与失败都是数据,不是异常。

    Attributes:
        ok: 是否执行成功。
        content: 成功时给模型看的结果文本(str 原样,其它对象 JSON 序列化)。
        error: 失败时的可读原因(校验失败 / 超时 / ``_run`` 抛出的异常)。
        data: 成功时 ``_run`` 的原始返回值(逃生舱,程序侧取结构化数据用;不参与 repr)。
    """

    ok: bool
    content: str = ""
    error: str | None = None
    data: object | None = field(default=None, repr=False)

    def to_observation(self) -> str:
        """转成可直接喂回模型的 Observation 文本(失败时带统一前缀)。"""
        return self.content if self.ok else f"[工具执行失败] {self.error}"


# --------------------------------------------------------------------------- #
# BaseTool:抽象基类 + 标准执行管线                                               #
# --------------------------------------------------------------------------- #
class BaseTool(ABC):
    """所有工具的抽象基类:声明元数据,业务逻辑只写 ``_run()``。

    子类声明(类属性即可)::

        class QueryOrderTool(BaseTool):
            name = "query_order"
            description = "查订单状态。何时用:用户询问订单进度/状态时。"
            args_schema = QueryOrderArgs   # Pydantic 模型;None = 无参数工具
            timeout = 5.0                  # 可选;None = 不限时

            def _run(self, order_id: str) -> str:
                ...

    公开入口 :meth:`invoke` 走固定管线:校验 → 限时执行 → 结果标准化,任何一步
    失败都折叠成 ``ToolResult(ok=False)``,**永不抛异常**(编程错误除外)。

    Attributes:
        name: 工具唯一名字(注册与模型选择都用它)。
        description: 给模型看的说明,写清「干嘛用 + 何时用 + 参数含义」。
        args_schema: 参数的 Pydantic 模型;``None`` 表示无参数工具。
        strict: strict mode 开关(默认 **True**):禁止类型强转(``"5"`` 不会转成
            ``5``)、拒绝未知参数、导出的 Schema 带 ``additionalProperties: false``。
        timeout: 单次执行的秒数上限;``None`` 不限时。注意 Python 线程无法强杀,
            超时后工作线程可能仍在后台跑完,这里只保证**调用方**及时拿到超时结果。
        permission: 权限级别元数据(默认 ``"low"``)。本阶段只声明不拦截;
            阶段六 safety 读它实现分层执行与 HITL 审批,工具本身零改动。
    """

    name: str
    description: str
    args_schema: type[BaseModel] | None = None
    strict: bool = True
    timeout: float | None = None
    permission: Permission = "low"

    @property
    def _idempotency_cache(self) -> dict[str, ToolResult]:
        """幂等缓存(request_id → 首次成功结果)。惰性挂在**实例**上——
        BaseTool 不强制子类调 ``super().__init__()``,不能用类属性(会跨实例共享)。"""
        cache = getattr(self, "_idem_cache", None)
        if cache is None:
            cache = {}
            self._idem_cache: dict[str, ToolResult] = cache
        return cache

    # ------------------------------ 公开入口 ------------------------------ #
    def invoke(
        self, args: dict[str, object] | None = None, *, request_id: str | None = None
    ) -> ToolResult:
        """标准执行管线:校验 → 限时执行 → 标准化,失败折叠为 ``ok=False``。

        这是**模型驱动路径**的唯一入口(Agent 循环 / ``ToolRegistry.invoke``
        都走这里),保证循环永不因工具而崩。

        Args:
            args: 模型生成的参数字典(对应 JSON 里的 arguments);``None`` 当空 dict。
            request_id: 幂等请求 ID(阶段六可靠性):同一 ID 重复调用直接返回
                首次的**成功**结果,不重复执行——使「审批通过后执行挂起动作」
                这类可能被重放的路径可以放心重试。ID 由**框架**生成
                (HITL 队列等),不进模型可见的 Schema,不靠模型传参。
                失败结果不缓存(失败后允许换参重试)。缓存在工具实例内存中,
                进程重启即清(与 mock 数据同生命周期;生产应落外部存储)。

        Returns:
            标准化的 :class:`ToolResult`。
        """
        if request_id is not None and request_id in self._idempotency_cache:
            return self._idempotency_cache[request_id]

        try:
            validated = self._validate(args or {})
        except ToolValidationError as e:
            return ToolResult(ok=False, error=f"参数校验失败:{e}")

        try:
            output = self._execute_with_timeout(validated)
        except ToolTimeoutError as e:
            return ToolResult(ok=False, error=str(e))
        except Exception as e:  # noqa: BLE001 - 工具任何业务异常都折叠喂回,循环不崩
            return ToolResult(ok=False, error=f"{type(e).__name__}: {e}")

        result = ToolResult(ok=True, content=self._format_output(output), data=output)
        if request_id is not None:
            self._idempotency_cache[request_id] = result
        return result

    def run(self, **kwargs: object) -> str:
        """兼容阶段二 :class:`Tool` 协议的薄适配:``invoke`` 后取 Observation 文本。

        现有 ``ReActAgent`` 无需任何修改即可使用 ``BaseTool`` 工具。
        """
        return self.invoke(dict(kwargs)).to_observation()

    def to_schema(self) -> dict[str, object]:
        """导出厂商无关的工具描述:``{"name", "description", "parameters"}``。

        ``parameters`` 是标准 JSON Schema;strict 时附加
        ``additionalProperties: false`` 提示模型不要编造参数。
        厂商格式转换(Claude ``input_schema`` / OpenAI ``function``)在 llm 层做。
        """
        if self.args_schema is not None:
            parameters: dict[str, object] = self.args_schema.model_json_schema()
            parameters.pop("title", None)
        else:
            parameters = {"type": "object", "properties": {}}
        if self.strict:
            parameters["additionalProperties"] = False
        return {"name": self.name, "description": self.description, "parameters": parameters}

    # ------------------------------ 子类实现 ------------------------------ #
    @abstractmethod
    def _run(self, **kwargs: object) -> object:
        """业务逻辑本体。参数已通过校验,**不要**自己再 try/except 包一层。

        Returns:
            任意结果对象;str 原样作为 Observation,其它对象会被 JSON 序列化。
        """

    # ------------------------------ 管线内部 ------------------------------ #
    def _validate(self, args: dict[str, object]) -> dict[str, object]:
        """按 ``args_schema`` 校验参数;strict 下额外拒绝未知参数、禁止类型强转。

        Raises:
            ToolValidationError: 任何校验失败,携带给模型看的可读原因。
        """
        if self.args_schema is None:
            return dict(args)

        allowed = set(self.args_schema.model_fields)
        if self.strict:
            extra = set(args) - allowed
            if extra:
                raise ToolValidationError(
                    f"未知参数 {sorted(extra)};该工具只接受参数 {sorted(allowed)}"
                )
        try:
            model = self.args_schema.model_validate(args, strict=self.strict)
        except ValidationError as e:
            reasons = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or '(root)'}: {err['msg']}"
                for err in e.errors()
            )
            raise ToolValidationError(reasons) from e
        # 取校验后的真实字段值(不用 model_dump:它会把嵌套模型降级成 dict)
        return {name: getattr(model, name) for name in allowed}

    def _execute_with_timeout(self, kwargs: dict[str, object]) -> object:
        """执行 ``_run``;设置了 ``timeout`` 则在工作线程中限时等待。

        Raises:
            ToolTimeoutError: 超过 ``timeout`` 秒未返回。
        """
        if self.timeout is None:
            return self._run(**kwargs)

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(self._run, **kwargs)
            try:
                return future.result(timeout=self.timeout)
            except FuturesTimeoutError:
                raise ToolTimeoutError(
                    f"工具 {self.name} 执行超时(超过 {self.timeout} 秒)"
                ) from None
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _format_output(output: object) -> str:
        """把 ``_run`` 的返回值标准化成喂回模型的文本。"""
        if isinstance(output, str):
            return output
        if isinstance(output, BaseModel):
            return output.model_dump_json()
        try:
            return json.dumps(output, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(output)
