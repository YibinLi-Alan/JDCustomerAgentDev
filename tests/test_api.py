"""阶段六 P-D API 单测:FastAPI TestClient + MockLLM 装的 AgentService(全离线)。"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")  # 未装 fastapi 时跳过,不拖垮全套
from fastapi.testclient import TestClient  # noqa: E402

from agent_framework.api.server import create_app  # noqa: E402
from agent_framework.core.config import get_settings  # noqa: E402
from agent_framework.core.llm import ChatResponse, ToolCall, Usage  # noqa: E402
from agent_framework.safety import HandoffQueue  # noqa: E402
from agent_framework.service import AgentService  # noqa: E402
from agent_framework.tools.jd_mock_data import JDMockStore  # noqa: E402
from agent_framework.tools.presets import default_registry  # noqa: E402
from tests.mock_llm import MockLLM  # noqa: E402


def _client(llm, store=None) -> TestClient:  # type: ignore[no-untyped-def]
    settings = get_settings()
    settings.provider = "openai"
    service = AgentService(
        llm,
        default_registry(store or JDMockStore()),
        settings,
        handoff_queue=HandoffQueue(),  # 内存队列,隔离测试(不读落盘的 data/handoff_queue.json)
        enable_trace=False,
    )
    return TestClient(create_app(service))


def test_health() -> None:
    client = _client(MockLLM([]))
    assert client.get("/health").json() == {"status": "ok"}


def test_index_lists_endpoints() -> None:
    resp = _client(MockLLM([])).get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["docs"] == "/docs"
    assert "POST /chat" in body["endpoints"]


def test_chat_direct() -> None:
    client = _client(MockLLM(['{"target":"direct","reason":"寒暄"}', "您好,很高兴为您服务!"]))
    resp = client.post("/chat", json={"user_id": "u1", "message": "你好"})
    assert resp.status_code == 200
    body = resp.json()
    assert "您好" in body["answer"]
    assert body["route"] == "direct" and body["rate_limited"] is False


def test_chat_request_validation() -> None:
    client = _client(MockLLM([]))
    resp = client.post("/chat", json={"message": "缺 user_id"})  # 缺必填字段
    assert resp.status_code == 422  # FastAPI 自动校验


def test_chat_stream_sse_events() -> None:
    llm = MockLLM(
        [
            '{"target":"order_agent","reason":"查订单"}',
            ChatResponse(
                content="",
                usage=Usage(5, 5),
                model="mock",
                tool_calls=[ToolCall(id="c1", name="query_order", args={"order_id": "12345"})],
            ),
            "您的订单已发货。",
        ]
    )
    client = _client(llm)
    with client.stream(
        "POST", "/chat/stream", json={"user_id": "u1", "message": "订单 12345"}
    ) as r:
        assert r.status_code == 200
        raw = "".join(chunk for chunk in r.iter_text())
    assert "event: step" in raw  # 中间步骤事件(route 等)
    assert "event: answer" in raw and "已发货" in raw
    assert "event: done" in raw


def test_approvals_flow() -> None:
    store = JDMockStore()
    llm = MockLLM(
        [
            '{"target":"aftersales_agent","reason":"退款"}',
            ChatResponse(
                content="",
                usage=Usage(5, 5),
                model="mock",
                tool_calls=[
                    ToolCall(
                        id="c1", name="apply_refund", args={"order_id": "12345", "reason": "坏了"}
                    )
                ],
            ),
            "已提交退款审批。",
        ]
    )
    client = _client(llm, store)
    # 触发高权限 → 审批单入队
    client.post("/chat", json={"user_id": "alice", "message": "订单 12345 退款"})
    assert store.refunds == []

    pending = client.get("/approvals", params={"status": "pending"}).json()
    assert len(pending) == 1
    item_id = pending[0]["id"]

    # 放行 → 真执行
    approve = client.post(f"/approvals/{item_id}/approve", json={"note": "核实属实"})
    assert approve.status_code == 200
    assert approve.json()["status"] == "done"
    assert len(store.refunds) == 1

    # 重复放行 → 400
    again = client.post(f"/approvals/{item_id}/approve", json={"note": ""})
    assert again.status_code == 400


def test_approve_unknown_id_returns_400() -> None:
    client = _client(MockLLM([]))
    resp = client.post("/approvals/nope/approve", json={"note": ""})
    assert resp.status_code == 400
