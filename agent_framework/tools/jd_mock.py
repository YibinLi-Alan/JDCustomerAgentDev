"""京东客服业务工具(mock 数据版)—— 阶段三 P-C 的 A 组工具。

8 个满足 :class:`~agent_framework.tools.base.BaseTool` 的业务工具,数据一律来自
共享的 :class:`~agent_framework.tools.jd_mock_data.JDMockStore`(默认连进程单例
``DEFAULT_STORE``,测试可注入独立 store)。以后接真实京东 API = 换数据层,工具零改动。

统一遵守工具编写规范(stage-3-design.md §8/§10):

- description 三段式:功能。何时用。参数;
- **「查无结果」不算错误**:返回「未找到,请核对」的正常文本(``ok=True``),
  让模型转述;只有系统性故障才走 ``ok=False``;
- 高权限写操作(退款 / 取消 / 建工单)标注 ``permission="high"``,本阶段只声明,
  阶段六 safety 读它接 HITL 审批。

``JD_MOCK_TOOLS`` 保留为阶段二 ReAct CLI 的两件套;全量装配请用
:func:`create_jd_tools` 或 ``presets.default_registry()``。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent_framework.tools.base import BaseTool
from agent_framework.tools.jd_mock_data import DEFAULT_STORE, JDMockStore


class _JDTool(BaseTool):
    """业务工具的共同基座:注入共享数据源(默认进程单例)。"""

    def __init__(self, store: JDMockStore | None = None) -> None:
        """绑定数据源。

        Args:
            store: mock 数据源;缺省用进程共享的 ``DEFAULT_STORE``,
                让「取消订单后再查询」这类跨工具状态变化可见。测试请注入独立实例。
        """
        self._store = store or DEFAULT_STORE


# --------------------------------------------------------------------------- #
# 低权限:只读查询                                                                #
# --------------------------------------------------------------------------- #
class QueryOrderArgs(BaseModel):
    order_id: str = Field(description="京东订单号,如 12345")


class QueryOrderTool(_JDTool):
    """查单个订单的状态与物流单号。"""

    name = "query_order"
    description = (
        "查订单状态与物流单号。何时用:用户询问某个订单的进度/状态/是否发货时。"
        "参数:order_id(订单号)。返回:商品、金额、状态、快递公司与运单号、下单时间。"
    )
    args_schema = QueryOrderArgs
    permission = "low"

    def _run(self, order_id: str) -> str:
        order = self._store.orders.get(order_id)
        if order is None:
            return f"未找到订单号 {order_id} 对应的订单,请核对后重试。"
        shipping = (
            f"快递={order.carrier};运单号={order.tracking_no}"
            if order.tracking_no
            else "尚未发货,暂无运单号"
        )
        return (
            f"订单 {order.order_id}:商品={order.item};金额=¥{order.amount:.2f};"
            f"状态={order.status};{shipping};下单时间={order.placed_at}。"
        )


class QueryLogisticsArgs(BaseModel):
    tracking_no: str = Field(description="物流运单号,如 SF123")


class QueryLogisticsTool(_JDTool):
    """按运单号查物流进度。"""

    name = "query_logistics"
    description = (
        "查物流进度。何时用:已拿到运单号,需要知道包裹当前位置/预计送达时间时。"
        "参数:tracking_no(物流单号)。返回:运输状态、当前位置、预计送达。"
    )
    args_schema = QueryLogisticsArgs
    permission = "low"

    def _run(self, tracking_no: str) -> str:
        progress = self._store.logistics.get(tracking_no)
        if progress is None:
            return f"未查询到运单号 {tracking_no} 的物流信息,请核对后重试。"
        return f"运单 {tracking_no}:{progress}"


class QueryProductArgs(BaseModel):
    keyword: str = Field(description="商品名称关键词,如“充电器”")


class QueryProductTool(_JDTool):
    """按关键词查商品的价格、库存、保修。"""

    name = "query_product"
    description = (
        "查商品信息。何时用:用户询问商品的价格/库存/是否有货/保修政策时。"
        "参数:keyword(商品名称关键词)。返回:命中商品的价格、库存与保修。"
    )
    args_schema = QueryProductArgs
    permission = "low"

    def _run(self, keyword: str) -> str:
        hits = [p for p in self._store.products if keyword in p.name]
        if not hits:
            return f"未找到与“{keyword}”相关的商品,请换个关键词试试。"
        lines = [
            f"{p.name}:价格=¥{p.price:.2f};"
            f"库存={'有货(' + str(p.stock) + ' 件)' if p.stock else '无货'};{p.warranty}"
            for p in hits
        ]
        return "\n".join(lines)


class SearchFAQArgs(BaseModel):
    query: str = Field(description="用户问题的关键词,如“退货”“发票”")


class SearchFAQTool(_JDTool):
    """检索平台规则 FAQ(退换货、发票、优惠券等)。"""

    name = "search_faq"
    description = (
        "查平台规则 FAQ。何时用:用户问退换货政策、退款时效、发票、优惠券等平台规则类问题时。"
        "参数:query(问题关键词)。返回:命中的规则条目与答案。"
    )
    args_schema = SearchFAQArgs
    permission = "low"

    def _run(self, query: str) -> str:
        hits = [f for f in self._store.faqs if any(kw in query for kw in f.keywords)]
        if not hits:
            return f"未找到与“{query}”相关的平台规则,建议转人工工单处理。"
        return "\n".join(f"【{f.question}】{f.answer}" for f in hits)


class QueryUserOrdersArgs(BaseModel):
    limit: int = Field(default=5, ge=1, le=20, description="最多返回几条,默认 5")


class QueryUserOrdersTool(_JDTool):
    """列出当前用户最近的订单(不需要订单号)。"""

    name = "query_user_orders"
    description = (
        "查当前用户最近的订单列表。何时用:用户没报订单号、问“我最近买的东西/我的订单”时,"
        "先用它拿到订单号再查详情。参数:limit(最多返回几条,默认 5)。"
        "返回:按下单时间倒序的订单摘要。"
    )
    args_schema = QueryUserOrdersArgs
    permission = "low"

    def _run(self, limit: int = 5) -> str:
        orders = sorted(self._store.orders.values(), key=lambda o: o.placed_at, reverse=True)
        if not orders:
            return "当前用户没有任何订单。"
        lines = [
            f"订单 {o.order_id}:{o.item};¥{o.amount:.2f};状态={o.status};下单={o.placed_at}"
            for o in orders[:limit]
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 高权限:写操作(阶段六接 HITL 审批,本阶段只声明 permission)                       #
# --------------------------------------------------------------------------- #
class ApplyRefundArgs(BaseModel):
    order_id: str = Field(description="要退款的订单号")
    reason: str = Field(description="退款理由,如“商品与描述不符”")


class ApplyRefundTool(_JDTool):
    """为订单提交退款申请(高权限)。"""

    name = "apply_refund"
    description = (
        "提交退款申请。何时用:用户明确要求退款/退货,且已确认订单号与理由时;"
        "未发货的订单应改用 cancel_order。参数:order_id(订单号)、reason(退款理由)。"
        "返回:退款受理单号与后续流程说明。"
    )
    args_schema = ApplyRefundArgs
    permission = "high"

    def _run(self, order_id: str, reason: str) -> str:
        order = self._store.orders.get(order_id)
        if order is None:
            return f"未找到订单号 {order_id} 对应的订单,无法申请退款,请核对后重试。"
        if order.status == "已取消":
            return f"订单 {order_id} 已是取消状态,无需退款。"
        if order.status == "待发货":
            return f"订单 {order_id} 尚未发货,建议直接取消订单(cancel_order),退款更快。"
        refund_id = self._store.next_refund_id()
        self._store.refunds.append(
            {"refund_id": refund_id, "order_id": order_id, "reason": reason, "status": "待人工审核"}
        )
        return (
            f"退款申请已受理:受理单号 {refund_id},订单 {order_id},理由「{reason}」。"
            "人工审核通过后 1-3 个工作日原路退回。"
        )


class CancelOrderArgs(BaseModel):
    order_id: str = Field(description="要取消的订单号")
    reason: str = Field(default="", description="取消原因,可留空")


class CancelOrderTool(_JDTool):
    """取消未发货的订单(高权限)。"""

    name = "cancel_order"
    description = (
        "取消订单。何时用:用户要求取消**尚未发货**的订单时;已发货的订单无法取消,"
        "应改用 apply_refund。参数:order_id(订单号)、reason(取消原因,可选)。"
        "返回:取消结果。"
    )
    args_schema = CancelOrderArgs
    permission = "high"

    def _run(self, order_id: str, reason: str = "") -> str:
        order = self._store.orders.get(order_id)
        if order is None:
            return f"未找到订单号 {order_id} 对应的订单,请核对后重试。"
        if order.status == "已取消":
            return f"订单 {order_id} 已是取消状态,无需重复操作。"
        if order.status != "待发货":
            return (
                f"订单 {order_id} 当前状态为「{order.status}」,已无法直接取消;"
                "如不需要该商品,可申请退款(apply_refund)。"
            )
        order.status = "已取消"
        note = f"(原因:{reason})" if reason else ""
        return f"订单 {order_id} 已成功取消{note},款项将在 1-3 个工作日原路退回。"


class CreateTicketArgs(BaseModel):
    summary: str = Field(description="问题的一句话概述")
    detail: str = Field(default="", description="问题详情与已尝试的处理,可留空")


class CreateTicketTool(_JDTool):
    """创建人工客服工单(高权限)。"""

    name = "create_ticket"
    description = (
        "创建人工客服工单。何时用:问题超出现有工具能力(投诉、复杂纠纷、FAQ 查无)、"
        "或用户明确要求人工处理时。参数:summary(问题概述)、detail(详情,可选)。"
        "返回:工单号与人工跟进时效。"
    )
    args_schema = CreateTicketArgs
    permission = "high"

    def _run(self, summary: str, detail: str = "") -> str:
        ticket_id = self._store.next_ticket_id()
        self._store.tickets.append(
            {"ticket_id": ticket_id, "summary": summary, "detail": detail, "status": "待人工处理"}
        )
        return f"人工工单已创建:工单号 {ticket_id}(问题:{summary})。人工客服将在 24 小时内联系您。"


# --------------------------------------------------------------------------- #
# 装配便捷入口                                                                    #
# --------------------------------------------------------------------------- #
def create_jd_tools(store: JDMockStore | None = None) -> list[BaseTool]:
    """构造全套 8 个业务工具(共享同一个数据源)。

    Args:
        store: mock 数据源;缺省共用进程单例 ``DEFAULT_STORE``。

    Returns:
        8 个业务工具实例(5 低权限查询 + 3 高权限写操作)。
    """
    return [
        QueryOrderTool(store),
        QueryLogisticsTool(store),
        QueryProductTool(store),
        SearchFAQTool(store),
        QueryUserOrdersTool(store),
        ApplyRefundTool(store),
        CancelOrderTool(store),
        CreateTicketTool(store),
    ]


#: 阶段二 ReAct CLI 的两件套(保持既有 demo 语义不变);全量装配用 create_jd_tools()。
JD_MOCK_TOOLS: list[BaseTool] = [QueryOrderTool(), QueryLogisticsTool()]
