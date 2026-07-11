"""API 请求/响应模型 —— 全 Pydantic,FastAPI 自动校验 + 文档(阶段六 P-D)。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /chat 的请求体。"""

    user_id: str = Field(
        description="用户标识;记忆隔离 + 限流 + 审批单归属(阶段一预留参数正式接上)"
    )
    message: str = Field(description="用户本轮消息")
    mode: str = Field(default="auto", description="auto | supervisor | router")


class ChatResponse(BaseModel):
    """POST /chat 的非流式响应体(stream=false 时用)。"""

    answer: str
    task_id: str
    route: str
    handoff_ids: list[str] = Field(default_factory=list, description="本轮产生的人工介入单号")
    redactions: list[str] = Field(default_factory=list, description="出口脱敏命中的类型")
    rate_limited: bool = False


class HandoffView(BaseModel):
    """人工介入单的对外视图。"""

    id: str
    kind: str
    user_id: str
    status: str
    summary: str
    created_at: str
    resolution: str = ""


class ApprovalAction(BaseModel):
    """审批/驳回操作的请求体。"""

    note: str = Field(default="", description="人工备注")


class ApprovalResult(BaseModel):
    """审批放行后的执行结果。"""

    id: str
    status: str
    executed: str = Field(default="", description="放行后工具执行的返回文本(approve 时)")
