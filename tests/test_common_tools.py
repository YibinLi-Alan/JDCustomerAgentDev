"""阶段三 P-C 单元测试:3 个通用工具(calculator / current_time / http_request)。

全部离线:calculator/current_time 本地执行;http_request 只测白名单与协议拦截
(拦截发生在真正发请求**之前**,不碰网络)。
"""

from __future__ import annotations

import datetime

from agent_framework.tools import HttpRequestTool, calculator, current_time


# --------------------------------------------------------------------------- #
# calculator                                                                    #
# --------------------------------------------------------------------------- #
def test_calculator_basic_and_precedence():
    assert calculator.invoke({"expression": "399*0.85-50"}).content == "399*0.85-50 = 289.15"
    assert calculator.invoke({"expression": "(129+399)*2"}).content == "(129+399)*2 = 1056"
    # 整数化:8/2 → 4 而非 4.0
    assert calculator.invoke({"expression": "8/2"}).content == "8/2 = 4"


def test_calculator_rejects_non_math():
    # 变量/函数调用/字符串都不在白名单内 → ok=False(不是崩溃)
    for evil in ("__import__('os')", "open('x')", "'a'*9", "x+1"):
        result = calculator.invoke({"expression": evil})
        assert not result.ok, evil
    # 语法错误
    assert not calculator.invoke({"expression": "1+*2"}).ok
    # 指数炸弹
    boom = calculator.invoke({"expression": "9**9**9"})
    assert not boom.ok and "指数过大" in (boom.error or "")


def test_calculator_metadata():
    assert calculator.name == "calculator"
    assert calculator.permission == "low"
    assert calculator.to_schema()["parameters"]["required"] == ["expression"]


# --------------------------------------------------------------------------- #
# current_time                                                                  #
# --------------------------------------------------------------------------- #
def test_current_time():
    result = current_time.invoke({})
    assert result.ok
    # 内容里是可解析的今天日期
    today = datetime.date.today().strftime("%Y-%m-%d")
    assert today in result.content
    assert "周" in result.content
    # 无参数工具:多余参数在 strict 下被拒
    assert not current_time.invoke({"tz": "UTC"}).ok


# --------------------------------------------------------------------------- #
# http_request(只测不碰网络的拦截路径)                                           #
# --------------------------------------------------------------------------- #
def test_http_request_whitelist_rejection():
    tool = HttpRequestTool(allowed_hosts=("httpbin.org",))
    result = tool.invoke({"url": "https://evil.example.com/steal"})
    assert result.ok  # 拒绝是一条给模型看的正常观察,不是系统故障
    assert "不在白名单" in result.content
    assert "httpbin.org" in result.content  # 告诉模型哪些可用


def test_http_request_subdomain_allowed_but_lookalike_rejected():
    tool = HttpRequestTool(allowed_hosts=("httpbin.org",))
    assert tool._host_allowed("sub.httpbin.org")  # 子域放行
    assert not tool._host_allowed("evilhttpbin.org")  # 后缀伪装拒绝
    assert not tool._host_allowed("httpbin.org.evil.com")


def test_http_request_scheme_and_method_guards():
    tool = HttpRequestTool()
    ftp = tool.invoke({"url": "ftp://httpbin.org/file"})
    assert "仅支持 http/https" in ftp.content
    delete = tool.invoke({"url": "https://httpbin.org/x", "method": "DELETE"})
    assert "仅支持 GET / POST" in delete.content


def test_http_request_metadata():
    tool = HttpRequestTool()
    assert tool.permission == "medium"
    params = tool.to_schema()["parameters"]
    assert set(params["properties"]) == {"url", "method", "body"}
    assert params["required"] == ["url"]
