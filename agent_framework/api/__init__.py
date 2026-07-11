"""API 子包 —— FastAPI 服务(阶段六 P-D)。

- :mod:`schemas`:请求/响应 Pydantic 模型(自动校验 + 文档);
- :mod:`server`:``create_app()`` 把 ``AgentService`` 包成 HTTP 接口
  (``/chat`` 非流式、``/chat/stream`` SSE、``/approvals`` 人工审批、``/health``)。

FastAPI 是可选依赖:只在实际起服务时才需要;import 本子包会触发 fastapi import。
"""
