"""京东 mock 数据层 —— 所有业务工具共享的一致假数据(规范第 5 条)。

设计立场(见 stage-3-design.md §8):

- **数据互相一致**:订单 12345 ↔ 运单 SF123 ↔ 商品「快充充电器」……多工具连环
  调用的 demo 才真实;
- **收口在一处**:业务工具只跟 :class:`JDMockStore` 打交道,以后接真实京东 API =
  换一个同接口的数据层实现,所有工具零改动;
- **可写**:退款 / 取消 / 工单是会改状态的写操作,store 持有可变状态与自增 id,
  测试用 ``JDMockStore()`` 各建一份,互不串味。

模块底部的 ``DEFAULT_STORE`` 是进程内共享的默认单例:CLI / demo 里所有工具默认
连它,这样「取消订单后再查订单」能看到状态变化。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Order:
    """一条 mock 订单。``status`` 会被取消/退款等写操作修改。"""

    order_id: str
    item: str
    amount: float
    status: str  # 待发货 / 已发货 / 已签收 / 已取消
    placed_at: str
    carrier: str | None = None  # 已发货后才有
    tracking_no: str | None = None


@dataclass
class Product:
    """一条 mock 商品。"""

    product_id: str
    name: str
    price: float
    stock: int
    warranty: str


@dataclass
class FAQ:
    """一条 mock 平台规则问答;``keywords`` 用于朴素的关键词命中。"""

    keywords: tuple[str, ...]
    question: str
    answer: str


class JDMockStore:
    """业务工具共享的内存假数据源(单用户视角:当前用户就是提问的这位)。

    每个实例是一份独立数据:构造即载入标准数据集,写操作只改本实例。
    """

    def __init__(self) -> None:
        """载入标准 mock 数据集(订单/物流/商品/FAQ 互相一致)。"""
        self.orders: dict[str, Order] = {
            o.order_id: o
            for o in (
                Order(
                    order_id="12345",
                    item="Anker 快充充电器 65W",
                    amount=129.0,
                    status="已发货",
                    placed_at="2026-06-28",
                    carrier="顺丰",
                    tracking_no="SF123",
                ),
                Order(
                    order_id="67890",
                    item="京造 机械键盘 87 键",
                    amount=399.0,
                    status="待发货",
                    placed_at="2026-07-04",
                ),
                Order(
                    order_id="11111",
                    item="小米 蓝牙耳机 Pro",
                    amount=299.0,
                    status="已签收",
                    placed_at="2026-06-15",
                    carrier="京东物流",
                    tracking_no="JD456",
                ),
            )
        }
        self.logistics: dict[str, str] = {
            "SF123": "运输中,当前在【北京分拣中心】,预计明天送达。",
            "JD456": "已签收(2026-06-18 14:32,本人签收)。",
        }
        self.products: list[Product] = [
            Product("p001", "Anker 快充充电器 65W", 129.0, 500, "1 年质保"),
            Product("p002", "京造 机械键盘 87 键", 399.0, 32, "2 年质保"),
            Product("p003", "小米 蓝牙耳机 Pro", 299.0, 0, "1 年质保"),
        ]
        self.faqs: list[FAQ] = [
            FAQ(
                ("退货", "退换", "七天", "7天", "无理由"),
                "退换货政策",
                "自营商品支持 7 天无理由退货(签收次日起算);商品需保持完好、配件齐全。"
                "生鲜、定制类商品除外。",
            ),
            FAQ(
                ("退款", "到账", "多久"),
                "退款多久到账",
                "退款在审核通过后 1-3 个工作日原路退回;银行卡支付最长可能 7 个工作日。",
            ),
            FAQ(
                ("发票", "开票", "报销"),
                "如何开具发票",
                "在「订单详情 → 发票服务」可申请电子普票或增值税专票,电子发票开具后发送至邮箱。",
            ),
            FAQ(
                ("优惠券", "券", "满减"),
                "优惠券使用规则",
                "优惠券在结算页选择使用;不可叠加、不找零、过期作废,退款时按实付金额退回。",
            ),
        ]
        # 写操作的落库处与自增 id 计数器
        self.refunds: list[dict[str, str]] = []
        self.tickets: list[dict[str, str]] = []
        self._refund_seq = 0
        self._ticket_seq = 0

    # ------------------------------ 写操作 ------------------------------ #
    def next_refund_id(self) -> str:
        """生成下一个退款单号(R0001、R0002……)。"""
        self._refund_seq += 1
        return f"R{self._refund_seq:04d}"

    def next_ticket_id(self) -> str:
        """生成下一个人工工单号(T0001、T0002……)。"""
        self._ticket_seq += 1
        return f"T{self._ticket_seq:04d}"


#: 进程内共享的默认数据源:CLI / demo 里所有业务工具默认连同一份数据,
#: 「先取消再查询」这类跨工具状态变化才可见。测试请自建 ``JDMockStore()``。
DEFAULT_STORE = JDMockStore()
