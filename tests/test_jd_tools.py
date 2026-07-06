"""阶段三 P-C 单元测试:8 个京东业务工具,每个覆盖正常 + 异常/查无路径。

规范(stage-3-design.md §8):「查无结果」是正常返回(``ok=True`` 的提示文本),
不是错误;测试全部注入独立 ``JDMockStore``,互不串味、全部离线。
"""

from __future__ import annotations

import pytest

from agent_framework.tools import (
    ApplyRefundTool,
    CancelOrderTool,
    CreateTicketTool,
    JDMockStore,
    QueryLogisticsTool,
    QueryOrderTool,
    QueryProductTool,
    QueryUserOrdersTool,
    SearchFAQTool,
    create_jd_tools,
    default_registry,
)


@pytest.fixture()
def store() -> JDMockStore:
    """每个测试一份独立数据,写操作互不影响。"""
    return JDMockStore()


# --------------------------------------------------------------------------- #
# 低权限查询工具                                                                  #
# --------------------------------------------------------------------------- #
def test_query_order(store):
    ok = QueryOrderTool(store).invoke({"order_id": "12345"})
    assert ok.ok
    assert "已发货" in ok.content and "SF123" in ok.content and "¥129.00" in ok.content

    miss = QueryOrderTool(store).invoke({"order_id": "99999"})
    assert miss.ok  # 查无 ≠ 错误
    assert "未找到" in miss.content


def test_query_order_unshipped_has_no_tracking(store):
    result = QueryOrderTool(store).invoke({"order_id": "67890"})
    assert result.ok
    assert "尚未发货" in result.content


def test_query_logistics(store):
    ok = QueryLogisticsTool(store).invoke({"tracking_no": "SF123"})
    assert ok.ok and "预计明天送达" in ok.content

    miss = QueryLogisticsTool(store).invoke({"tracking_no": "XX999"})
    assert miss.ok and "未查询到" in miss.content


def test_query_product(store):
    ok = QueryProductTool(store).invoke({"keyword": "充电器"})
    assert ok.ok and "¥129.00" in ok.content and "有货" in ok.content

    out_of_stock = QueryProductTool(store).invoke({"keyword": "蓝牙耳机"})
    assert "无货" in out_of_stock.content

    miss = QueryProductTool(store).invoke({"keyword": "冰箱"})
    assert miss.ok and "未找到" in miss.content


def test_search_faq(store):
    ok = SearchFAQTool(store).invoke({"query": "怎么退货?"})
    assert ok.ok and "7 天无理由" in ok.content

    miss = SearchFAQTool(store).invoke({"query": "怎么给差评"})
    assert miss.ok and "转人工" in miss.content  # 查无时引导 create_ticket


def test_query_user_orders(store):
    result = QueryUserOrdersTool(store).invoke({})
    assert result.ok
    # 按下单时间倒序:67890(07-04)在最前
    lines = result.content.splitlines()
    assert len(lines) == 3
    assert "67890" in lines[0]

    limited = QueryUserOrdersTool(store).invoke({"limit": 1})
    assert len(limited.content.splitlines()) == 1
    # limit 超出范围(Schema ge=1 le=20)→ 校验失败
    bad = QueryUserOrdersTool(store).invoke({"limit": 0})
    assert not bad.ok


# --------------------------------------------------------------------------- #
# 高权限写操作工具                                                                #
# --------------------------------------------------------------------------- #
def test_apply_refund_creates_record(store):
    result = ApplyRefundTool(store).invoke({"order_id": "12345", "reason": "商品与描述不符"})
    assert result.ok and "R0001" in result.content
    assert store.refunds[0]["status"] == "待人工审核"


def test_apply_refund_edge_cases(store):
    miss = ApplyRefundTool(store).invoke({"order_id": "99999", "reason": "x"})
    assert miss.ok and "未找到" in miss.content
    # 待发货 → 引导改用 cancel_order
    unshipped = ApplyRefundTool(store).invoke({"order_id": "67890", "reason": "不想要了"})
    assert "取消订单" in unshipped.content
    assert store.refunds == []  # 两次都没落库


def test_cancel_order_state_machine(store):
    tool = CancelOrderTool(store)
    # 待发货 → 可取消,且状态真的变了(跨工具可见)
    ok = tool.invoke({"order_id": "67890", "reason": "不想要了"})
    assert ok.ok and "已成功取消" in ok.content
    assert store.orders["67890"].status == "已取消"
    assert "已取消" in QueryOrderTool(store).invoke({"order_id": "67890"}).content
    # 重复取消 → 幂等提示
    again = tool.invoke({"order_id": "67890"})
    assert "无需重复" in again.content
    # 已发货 → 拒绝并引导退款
    shipped = tool.invoke({"order_id": "12345"})
    assert "无法直接取消" in shipped.content and "退款" in shipped.content
    # 查无
    assert "未找到" in tool.invoke({"order_id": "99999"}).content


def test_create_ticket(store):
    result = CreateTicketTool(store).invoke({"summary": "投诉快递员", "detail": "态度恶劣"})
    assert result.ok and "T0001" in result.content
    assert store.tickets[0]["summary"] == "投诉快递员"
    # 缺必填参数 → 校验失败
    bad = CreateTicketTool(store).invoke({})
    assert not bad.ok


# --------------------------------------------------------------------------- #
# 装配与元数据                                                                    #
# --------------------------------------------------------------------------- #
def test_create_jd_tools_and_permissions(store):
    tools = create_jd_tools(store)
    assert len(tools) == 8
    by_permission = {t.name: t.permission for t in tools}
    assert by_permission["query_order"] == "low"
    assert by_permission["apply_refund"] == "high"
    assert by_permission["cancel_order"] == "high"
    assert by_permission["create_ticket"] == "high"
    # 同一 store:cancel 后 query 能看到(数据一致性规范)
    assert all(t._store is tools[0]._store for t in tools)


def test_default_registry_assembles_all_11(store):
    registry = default_registry(store)
    assert len(registry) == 11
    assert {"query_order", "calculator", "current_time", "http_request"} <= set(registry.names)
    # 每个工具的 Schema 都能导出且带 name/description/parameters
    for schema in registry.to_schemas():
        assert schema["name"] and schema["description"] and "parameters" in schema
