"""京东风格 mock 工具 —— 供阶段二演示两步 ReAct 链路。

本模块提供两个满足 :class:`agent_framework.tools.base.Tool` 协议的具体工具,内置
**固定的假数据**(不接任何真实 API / 网络),足以跑通一条真实的多步链路:

    用户问「我的订单 12345 到哪了?」
        → 模型调 ``query_order(order_id="12345")`` 拿到运单号 SF123
        → 再调 ``query_logistics(tracking_no="SF123")`` 拿到物流进度
        → 给出最终答案。

> **预期对齐**:mock 只对**内置的固定几条**数据作答,其它输入返回「未找到」提示。
> 「问任意京东问题都能答」需要接真实数据源 / MCP,那是**阶段三**的事。

两个工具都**优雅返回字符串、不主动抛异常**;参数缺失时由 Python 自然抛
``TypeError``,交给上层 ReAct 循环的错误恢复处理(见 stage-2-design.md §4.6)。
"""

from __future__ import annotations

from agent_framework.tools.base import Tool


class QueryOrderTool:
    """查订单状态与物流单号(mock)。

    满足 :class:`~agent_framework.tools.base.Tool` 协议。内置一条订单
    ``12345``,返回其状态与运单号 ``SF123``,好让模型下一步拿去查物流。
    """

    name: str = "query_order"
    description: str = (
        "查订单状态与物流单号。何时用:用户询问某个订单的进度 / 状态 / 是否发货时。"
        "参数:order_id(订单号,字符串)。返回:订单状态、快递公司、运单号、下单时间。"
    )

    def run(self, order_id: str) -> str:
        """按订单号返回订单信息。

        Args:
            order_id: 订单号(字符串)。

        Returns:
            命中内置订单时返回其状态与运单号;否则返回「未找到」提示。
        """
        if order_id == "12345":
            return "订单 12345:状态=已发货;快递=顺丰;运单号=SF123;下单时间=2026-06-28。"
        return f"未找到订单号 {order_id} 对应的订单,请核对后重试。"


class QueryLogisticsTool:
    """查物流进度(mock)。

    满足 :class:`~agent_framework.tools.base.Tool` 协议。内置一条运单
    ``SF123``,返回其当前物流进度。
    """

    name: str = "query_logistics"
    description: str = (
        "查物流进度。何时用:已拿到运单号,需要知道包裹当前位置 / 预计送达时间时。"
        "参数:tracking_no(物流单号,字符串)。返回:运输状态、当前位置、预计送达。"
    )

    def run(self, tracking_no: str) -> str:
        """按运单号返回物流进度。

        Args:
            tracking_no: 物流单号(字符串)。

        Returns:
            命中内置运单时返回其物流进度;否则返回「未查询到」提示。
        """
        if tracking_no == "SF123":
            return "运单 SF123:运输中,当前在【北京分拣中心】,预计明天送达。"
        return f"未查询到运单号 {tracking_no} 的物流信息。"


#: 便捷列表:一次性装配这两个 mock 工具,直接传给 ``ReActAgent(tools=...)``。
JD_MOCK_TOOLS: list[Tool] = [QueryOrderTool(), QueryLogisticsTool()]
