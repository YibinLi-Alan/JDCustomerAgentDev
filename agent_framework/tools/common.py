"""通用工具(阶段三 P-C 的 B 组):calculator / current_time / http_request。

对齐大纲「实现常用 Tool」的示例;前两个用 ``@tool`` 装饰器实现(顺便验证装饰器
好不好用),``http_request`` 因带白名单配置做成类。

安全要点:

- ``calculator`` **不用 eval**:用 ``ast`` 解析后白名单求值,只允许数字与
  ``+ - * / // % **`` 及括号,指数大小设限防炸内存;
- ``http_request`` 只放行**白名单域名**(构造时注入,配置驱动而非硬编码),
  永远限时、限响应体大小。
"""

from __future__ import annotations

import ast
import datetime
import json
import operator
import urllib.error
import urllib.parse
import urllib.request

from pydantic import BaseModel, Field

from agent_framework.tools.base import BaseTool
from agent_framework.tools.function_tool import tool

# --------------------------------------------------------------------------- #
# calculator:AST 白名单求值(不用 eval)                                          #
# --------------------------------------------------------------------------- #
_BIN_OPS: dict[type[ast.operator], object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], object] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    """递归求值一棵只含数字与算术运算的 AST;遇到白名单外的节点即报错。

    Raises:
        ValueError: 表达式含不支持的语法(变量、函数调用、字符串等),
            或指数过大(防止 ``9**9**9`` 这类算式炸内存)。
    """
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"只支持数字,不支持 {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的运算符:{type(node.op).__name__}")
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("指数过大(|指数| 需 ≤ 100)")
        return op(left, right)  # type: ignore[operator]
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的一元运算符:{type(node.op).__name__}")
        return op(_safe_eval(node.operand))  # type: ignore[operator]
    raise ValueError(f"不支持的表达式成分:{type(node).__name__}(只允许数字与四则/幂/取余运算)")


@tool(timeout=3.0)
def calculator(expression: str) -> str:
    """数学计算器。何时用:需要精确计算金额、差价、折扣、退款分摊时(心算不可靠)。
    参数:expression(算式字符串,如 "399*0.85-50",支持 + - * / // % ** 与括号)。"""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"算式语法错误:{e.msg}") from e
    result = _safe_eval(tree)
    # 整数结果去掉小数尾巴(8.0 → 8),给模型更干净的数字
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{expression} = {result}"


# --------------------------------------------------------------------------- #
# current_time                                                                  #
# --------------------------------------------------------------------------- #
_WEEKDAYS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


@tool
def current_time() -> str:
    """查当前日期时间。何时用:回答里需要“今天/现在/还有几天”时(如预计送达还有几天、
    是否还在 7 天无理由退货期内)。无参数。返回:当前日期、时间与星期。"""
    now = datetime.datetime.now()
    return f"当前时间:{now.strftime('%Y-%m-%d %H:%M:%S')},{_WEEKDAYS[now.weekday()]}。"


# --------------------------------------------------------------------------- #
# http_request:白名单 + 限时 + 限响应体                                          #
# --------------------------------------------------------------------------- #
class HttpRequestArgs(BaseModel):
    url: str = Field(description="完整 URL(含 https://),域名必须在白名单内")
    method: str = Field(default="GET", description="HTTP 方法,支持 GET / POST")
    body: dict[str, object] | None = Field(
        default=None, description="POST 时的 JSON 请求体,GET 忽略"
    )


class HttpRequestTool(BaseTool):
    """调用外部 HTTP API(带域名白名单,中等权限)。

    白名单在构造时注入(配置驱动,不硬编码进逻辑);响应体截断到
    ``max_bytes``,避免超长响应挤爆上下文。
    """

    name = "http_request"
    description = (
        "调用外部 HTTP API 获取数据。何时用:需要的数据不在现有业务工具里、"
        "且目标域名在白名单内时。参数:url(完整地址)、method(GET/POST,默认 GET)、"
        "body(POST 的 JSON 体,可选)。返回:HTTP 状态码与响应体文本(超长截断)。"
    )
    args_schema = HttpRequestArgs
    permission = "medium"
    timeout = 15.0  # 整体兜底;单次请求另有 request_timeout

    def __init__(
        self,
        allowed_hosts: tuple[str, ...] = ("httpbin.org", "api.github.com"),
        *,
        request_timeout: float = 10.0,
        max_bytes: int = 4096,
    ) -> None:
        """配置白名单与限额。

        Args:
            allowed_hosts: 放行的域名(精确匹配或其子域);白名单外一律拒绝。
            request_timeout: 单次 HTTP 请求的超时秒数。
            max_bytes: 响应体最多读取的字节数,超出截断。
        """
        self._allowed_hosts = allowed_hosts
        self._request_timeout = request_timeout
        self._max_bytes = max_bytes

    def _host_allowed(self, host: str) -> bool:
        """判断域名是否在白名单内(允许子域,如 api.github.com ⊂ github.com 不成立,
        但 sub.httpbin.org ⊂ httpbin.org 成立)。"""
        return any(
            host == allowed or host.endswith("." + allowed) for allowed in self._allowed_hosts
        )

    def _run(self, url: str, method: str = "GET", body: dict[str, object] | None = None) -> str:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return f"仅支持 http/https 协议,收到:{parsed.scheme or '(无协议)'}。"
        host = parsed.hostname or ""
        if not self._host_allowed(host):
            return (
                f"域名 {host or '(空)'} 不在白名单内,已拒绝访问。"
                f"当前白名单:{list(self._allowed_hosts)}。"
            )
        method = method.upper()
        if method not in ("GET", "POST"):
            return f"仅支持 GET / POST,收到:{method}。"

        data = None
        headers = {"User-Agent": "agent-framework/0.3"}
        if method == "POST" and body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self._request_timeout) as resp:
                raw = resp.read(self._max_bytes + 1)
                status = resp.status
        except urllib.error.HTTPError as e:
            # 4xx/5xx 也是有效观察结果:让模型知道对方怎么说
            raw = e.read(self._max_bytes + 1)
            status = e.code
        except urllib.error.URLError as e:
            return f"请求失败:{e.reason}。请检查地址是否正确、网络是否可达。"

        text = raw[: self._max_bytes].decode("utf-8", errors="replace")
        truncated = "(已截断)" if len(raw) > self._max_bytes else ""
        return f"HTTP {status}{truncated}:\n{text}"


def create_common_tools() -> list[BaseTool]:
    """构造全套 3 个通用工具(calculator / current_time / http_request)。"""
    return [calculator, current_time, HttpRequestTool()]
