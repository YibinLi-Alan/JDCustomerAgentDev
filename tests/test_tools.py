"""阶段三 P-A 单元测试:BaseTool 执行管线(含 strict mode)、ToolRegistry、@tool 装饰器。

全部离线,不调用任何真实 LLM / 网络。覆盖点见 stage-3-design.md §11。
"""

from __future__ import annotations

import time
from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from agent_framework.tools import (
    BaseTool,
    Tool,
    ToolRegistrationError,
    ToolRegistry,
    UnknownToolError,
    tool,
)
from agent_framework.tools.jd_mock import JD_MOCK_TOOLS


# --------------------------------------------------------------------------- #
# 测试用工具                                                                     #
# --------------------------------------------------------------------------- #
class EchoArgs(BaseModel):
    text: str
    times: int = 1


class EchoTool(BaseTool):
    name = "echo"
    description = "重复输出文本。何时用:测试。参数:text(文本)、times(次数,默认 1)。"
    args_schema = EchoArgs

    def _run(self, text: str, times: int = 1) -> str:
        return text * times


class LaxEchoTool(EchoTool):
    """strict=False 的宽松版,用于对照。"""

    name = "lax_echo"
    strict = False


class BoomTool(BaseTool):
    name = "boom"
    description = "总是抛异常的工具,用于验证错误折叠。"

    def _run(self) -> str:
        raise RuntimeError("业务炸了")


class SlowTool(BaseTool):
    name = "slow"
    description = "睡 0.5 秒的慢工具,用于验证超时。"
    timeout = 0.1

    def _run(self) -> str:
        time.sleep(0.5)
        return "done"


# --------------------------------------------------------------------------- #
# BaseTool:正常路径与协议兼容                                                    #
# --------------------------------------------------------------------------- #
def test_invoke_ok():
    result = EchoTool().invoke({"text": "嗨", "times": 3})
    assert result.ok
    assert result.content == "嗨嗨嗨"
    assert result.error is None
    assert result.data == "嗨嗨嗨"


def test_run_returns_observation_str_and_satisfies_stage2_protocol():
    echo = EchoTool()
    assert isinstance(echo, Tool)  # 结构性满足阶段二协议 → 可直接插进现有 ReActAgent
    assert echo.run(text="a", times=2) == "aa"


def test_jd_mock_tools_still_satisfy_protocol():
    for mock_tool in JD_MOCK_TOOLS:
        assert isinstance(mock_tool, Tool)


# --------------------------------------------------------------------------- #
# 参数校验:strict mode                                                          #
# --------------------------------------------------------------------------- #
def test_missing_required_param():
    result = EchoTool().invoke({})
    assert not result.ok
    assert result.error is not None
    assert "参数校验失败" in result.error
    assert "text" in result.error


def test_strict_rejects_type_coercion():
    # strict 下 "2"(字符串)不允许强转成 int
    result = EchoTool().invoke({"text": "a", "times": "2"})
    assert not result.ok
    assert "times" in (result.error or "")


def test_strict_rejects_unknown_params():
    result = EchoTool().invoke({"text": "a", "made_up": 1})
    assert not result.ok
    assert "made_up" in (result.error or "")
    assert "text" in (result.error or "")  # 报错里列出合法参数,帮模型纠正


def test_lax_mode_coerces_and_ignores_extras():
    result = LaxEchoTool().invoke({"text": "a", "times": "2", "made_up": 1})
    assert result.ok
    assert result.content == "aa"


def test_tool_without_schema_accepts_empty_args():
    result = BoomTool().invoke(None)
    assert not result.ok  # 走到了 _run(说明校验放行),错误来自业务异常
    assert "业务炸了" in (result.error or "")


# --------------------------------------------------------------------------- #
# 执行:异常折叠与超时                                                            #
# --------------------------------------------------------------------------- #
def test_run_exception_folded_not_raised():
    result = BoomTool().invoke({})
    assert not result.ok
    assert "RuntimeError" in (result.error or "")
    # run() 的 Observation 文本带统一失败前缀
    assert BoomTool().run().startswith("[工具执行失败]")


def test_timeout():
    result = SlowTool().invoke({})
    assert not result.ok
    assert "超时" in (result.error or "")


# --------------------------------------------------------------------------- #
# Schema 导出                                                                    #
# --------------------------------------------------------------------------- #
def test_to_schema_strict():
    schema = EchoTool().to_schema()
    assert schema["name"] == "echo"
    assert schema["description"] == EchoTool.description
    params = schema["parameters"]
    assert params["type"] == "object"
    assert set(params["properties"]) == {"text", "times"}
    assert params["required"] == ["text"]
    assert params["additionalProperties"] is False  # strict 才有


def test_to_schema_lax_has_no_additional_properties_flag():
    assert "additionalProperties" not in LaxEchoTool().to_schema()["parameters"]


def test_to_schema_no_args():
    params = BoomTool().to_schema()["parameters"]
    assert params["properties"] == {}
    assert params["additionalProperties"] is False


# --------------------------------------------------------------------------- #
# ToolRegistry                                                                  #
# --------------------------------------------------------------------------- #
def test_registry_register_and_lookup():
    registry = ToolRegistry([EchoTool(), BoomTool()])
    assert len(registry) == 2
    assert "echo" in registry
    assert registry.names == ["echo", "boom"]
    assert registry.get("echo").name == "echo"
    assert [t.name for t in registry] == ["echo", "boom"]


def test_registry_duplicate_name_raises():
    registry = ToolRegistry([EchoTool()])
    with pytest.raises(ToolRegistrationError):
        registry.register(EchoTool())


def test_registry_replace_allows_override():
    registry = ToolRegistry([EchoTool()])
    replacement = EchoTool()
    registry.register(replacement, replace=True)
    assert registry.get("echo") is replacement
    assert len(registry) == 1


def test_registry_get_unknown_raises_with_available_list():
    registry = ToolRegistry([EchoTool()])
    with pytest.raises(UnknownToolError, match="echo"):
        registry.get("nope")
    with pytest.raises(UnknownToolError):
        registry.unregister("nope")


def test_registry_invoke_ok_and_unknown_folded():
    registry = ToolRegistry([EchoTool()])
    ok = registry.invoke("echo", {"text": "x"})
    assert ok.ok and ok.content == "x"
    bad = registry.invoke("nope", {})
    assert not bad.ok
    assert "echo" in (bad.error or "")  # 未知工具时列出可用工具,帮模型纠正


def test_registry_schemas_and_catalog():
    registry = ToolRegistry([EchoTool(), BoomTool()])
    schemas = registry.to_schemas()
    assert [s["name"] for s in schemas] == ["echo", "boom"]
    catalog = registry.render_catalog()
    assert "- echo:" in catalog and "- boom:" in catalog
    assert "没有可用工具" in ToolRegistry().render_catalog()


# --------------------------------------------------------------------------- #
# @tool 装饰器                                                                   #
# --------------------------------------------------------------------------- #
def test_tool_decorator_infers_schema_and_runs():
    @tool(timeout=1.0)
    def add(a: int, b: int = 0) -> int:
        """加法。何时用:需要精确求和时。"""
        return a + b

    assert add.name == "add"
    assert "加法" in add.description
    params = add.to_schema()["parameters"]
    assert set(params["properties"]) == {"a", "b"}
    assert params["required"] == ["a"]

    result = add.invoke({"a": 1, "b": 2})
    assert result.ok and result.data == 3 and result.content == "3"
    # 与手写类同构:可直接进 Registry、满足阶段二协议
    assert isinstance(add, Tool)
    assert ToolRegistry([add]).invoke("add", {"a": 2}).data == 2


def test_tool_decorator_bare_form_and_no_params():
    @tool
    def ping() -> str:
        """连通性测试。何时用:测试。"""
        return "pong"

    assert ping.args_schema is None
    assert ping.invoke({}).content == "pong"


def test_tool_decorator_annotated_field_description():
    @tool
    def greet(person: Annotated[str, Field(description="要问候的人名")]) -> str:
        """打招呼。何时用:测试。"""
        return f"你好,{person}"

    props = greet.to_schema()["parameters"]["properties"]
    assert props["person"]["description"] == "要问候的人名"


def test_tool_decorator_requires_docstring():
    with pytest.raises(ValueError, match="description"):

        @tool
        def nodoc(x: int) -> int:
            return x


def test_tool_decorator_requires_annotations():
    with pytest.raises(ValueError, match="类型注解"):

        @tool
        def bad(x) -> int:  # noqa: ANN001 - 故意缺注解触发报错
            """描述。"""
            return x


def test_tool_decorator_rejects_var_args():
    with pytest.raises(ValueError, match="args"):

        @tool
        def varargs(*items: str) -> str:
            """描述。"""
            return ",".join(items)


def test_tool_decorator_overrides():
    @tool(name="my_name", description="覆盖的描述。", strict=False)
    def whatever(x: int) -> int:
        """原始 docstring。"""
        return x

    assert whatever.name == "my_name"
    assert whatever.description == "覆盖的描述。"
    assert whatever.invoke({"x": "3"}).data == 3  # strict=False 允许强转
