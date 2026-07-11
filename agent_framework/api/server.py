"""FastAPI 服务 —— 把 AgentService 包成 HTTP 接口(阶段六 P-D)。

大纲 6.5 落地:API 化封装 + 流式输出(SSE)+ 异步外壳 + 人工审批接口。

接口:
- ``POST /chat``:非流式,一次拿完整答复;
- ``POST /chat/stream``:**SSE 流式**——中间步骤(分诊/计划/派工/审批…)实时推给
  用户,长任务体验天差地别;中间事件来自 §5.2 的 Tracer(又一次复用);
- ``GET /approvals`` / ``POST /approvals/{id}/approve|reject``:人工控制台 API 化,
  与 HITL 闭环打通;
- ``GET /health``:存活探针。

架构(§8.2):Agent 同步循环跑在**线程池**,``on_event`` 钩子把事件塞进
``asyncio.Queue``,SSE 生成器从队列消费——**同步核心 + 异步外壳**,core 不改 async。
任务队列不做(减法);长任务体验由 SSE 中间步骤兜底。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from agent_framework.api.schemas import (
    ApprovalAction,
    ApprovalResult,
    ChatRequest,
    ChatResponse,
    HandoffView,
)
from agent_framework.core.config import get_settings
from agent_framework.core.llm import create_llm
from agent_framework.safety.approval import HandoffItem
from agent_framework.service import AgentService
from agent_framework.tools.presets import default_registry


def _to_view(item: HandoffItem) -> HandoffView:
    return HandoffView(
        id=item.id,
        kind=item.kind,
        user_id=item.user_id,
        status=item.status,
        summary=item.summary,
        created_at=item.created_at,
        resolution=item.resolution,
    )


def create_app(service: AgentService | None = None) -> FastAPI:
    """构造 FastAPI 应用。``service`` 可注入(测试传假的);缺省按配置装配整栈。"""
    app = FastAPI(
        title="JD 客服 Agent 框架",
        description="六阶段自建 Agent 框架的生产化 HTTP 服务(阶段六)",
        version="0.6.0",
    )

    if service is None:
        settings = get_settings()
        llm = create_llm(settings)
        service = AgentService(llm, default_registry(), settings)
    app.state.service = service

    @app.get("/health")
    def health() -> dict[str, str]:
        """存活探针。"""
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        """非流式:一次拿完整答复(含路由、审批单号、脱敏记录)。"""
        result = app.state.service.handle(req.user_id, req.message, mode=req.mode)
        return ChatResponse(
            answer=result.answer,
            task_id=result.task_id,
            route=result.route,
            handoff_ids=[h.id for h in result.handoffs],
            redactions=result.redactions,
            rate_limited=result.rate_limited,
        )

    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest) -> StreamingResponse:
        """SSE 流式:中间步骤事件 + 最终答复实时推送。"""
        return StreamingResponse(
            _stream_events(app.state.service, req),
            media_type="text/event-stream",
        )

    @app.get("/approvals", response_model=list[HandoffView])
    def list_approvals(status: str | None = None) -> list[HandoffView]:
        """列出人工介入单(可按 status 过滤)。"""
        items = app.state.service.queue.list(status=status)  # type: ignore[arg-type]
        return [_to_view(i) for i in items]

    @app.post("/approvals/{item_id}/approve", response_model=ApprovalResult)
    def approve(item_id: str, action: ApprovalAction) -> ApprovalResult:
        """放行一张审批单:真正执行挂起的工具调用(幂等)。"""
        svc = app.state.service
        try:
            result = svc.queue.approve(item_id, svc.base_registry, note=action.note)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ApprovalResult(id=item_id, status="done", executed=result.to_observation())

    @app.post("/approvals/{item_id}/reject", response_model=ApprovalResult)
    def reject(item_id: str, action: ApprovalAction) -> ApprovalResult:
        """驳回一张审批单(动作不执行)。"""
        try:
            item = app.state.service.queue.reject(item_id, note=action.note)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ApprovalResult(id=item.id, status=item.status)

    return app


async def _stream_events(service: AgentService, req: ChatRequest) -> AsyncIterator[str]:
    """SSE 事件生成器:线程池跑同步 handle,on_event 事件经队列桥接到异步流。"""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, dict] | None] = asyncio.Queue()

    def on_event(kind: str, payload: dict[str, object]) -> None:
        # 在 Agent 工作线程里被调,线程安全地投递到事件循环
        loop.call_soon_threadsafe(queue.put_nowait, (kind, dict(payload)))

    def run() -> object:
        result = service.handle(req.user_id, req.message, mode=req.mode, on_event=on_event)
        loop.call_soon_threadsafe(queue.put_nowait, None)  # 结束哨兵
        return result

    task = loop.run_in_executor(None, run)

    # 中间步骤事件流(只推对用户有意义的几种,避免刷屏)
    user_facing = {
        "route",
        "plan",
        "step_start",
        "step_end",
        "replan",
        "approval_pending",
        "escalation",
        "input_flagged",
    }
    while True:
        event = await queue.get()
        if event is None:
            break
        kind, payload = event
        if kind in user_facing:
            yield _sse("step", {"kind": kind, **payload})

    result = await task
    yield _sse(
        "answer",
        {
            "answer": result.answer,  # type: ignore[attr-defined]
            "task_id": result.task_id,  # type: ignore[attr-defined]
            "route": result.route,  # type: ignore[attr-defined]
            "handoff_ids": [h.id for h in result.handoffs],  # type: ignore[attr-defined]
        },
    )
    yield _sse("done", {"task_id": result.task_id})  # type: ignore[attr-defined]


def _sse(event: str, data: dict) -> str:
    """格式化一条 SSE 消息。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


# uvicorn agent_framework.api.server:app 的入口(生产用 create_app 装配)
app = None  # 延迟:import 时不装配(避免无 key 环境 import 即失败);uvicorn 用工厂


def _factory() -> FastAPI:  # pragma: no cover — uvicorn --factory 入口
    return create_app()
