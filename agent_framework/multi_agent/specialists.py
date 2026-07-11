"""三个业务专员的定义(评审拍板:按业务域切分)。

- **order_agent 订单物流专员**:query_order / query_logistics / query_user_orders /
  current_time —— 订单与物流的一切查询;
- **aftersales_agent 售后专员**:query_order / apply_refund / cancel_order /
  create_ticket / search_faq —— 退款/取消/工单(高权限工具全在他手里,阶段六闸门只卡他);
- **product_agent 商品导购专员**:query_product / search_faq / calculator ——
  商品信息、推荐与价格计算。

- ``query_order`` 在订单与售后专员间**有意重叠**:售后操作前需自行查证,
  不依赖跨专员喊话(星型拓扑,专员间不通话);
- 加专员 = 加一条定义,Router/Supervisor 一行不改(与 ``presets.default_registry``
  同一装配模式)。
"""

from __future__ import annotations

from agent_framework.multi_agent.protocol import FAILURE_MARKER, Specialist
from agent_framework.safety.input_filter import HARDENING_CLAUSE
from agent_framework.tools.registry import ToolRegistry

# --------------------------------------------------------------------------- #
# 公共底座:阶段三客服三规则 + 阶段五协作条款(见 stage-5-design.md 附录 A)          #
# --------------------------------------------------------------------------- #

_COMMON_RULES = (
    "\n规则:\n"
    "- 优先参考上文对话历史理解用户的指代(如“那个订单 / 它 / 刚才那个”通常指"
    "之前对话里出现过的订单号、物流单号等)。\n"
    "- 对话历史里已出现过的信息(订单号、物流单号等)不要再向用户索要,直接拿来用;"
    "只有历史中确实找不到时,才礼貌地请用户提供。\n"
    "- 工具返回“未找到 / 执行失败”时,先核对参数再重试或换工具;确实拿不到就如实告知。\n"
    "协作条款(你是客服团队的一员,任务可能来自调度系统):\n"
    "- 【前序步骤结论】里已有的信息(订单号、查询结果等)直接使用,不要重复查询。\n"
    "- 超出你职责范围的诉求不要勉强处理,如实说明该找哪类专员。\n"
    f"- 如果任务无法完成(条件不满足/工具走不通/超出职责),答复必须以“{FAILURE_MARKER}:”"
    "开头并说明原因——调度系统靠这个前缀识别失败。\n"
    "- 工具返回“已提交人工审批”时视为该操作已妥善移交,把审批单号告知用户即可,"
    "这不算失败,也不要重复提交。"
    f"{HARDENING_CLAUSE}"
)


def _prompt(title: str, domain_rules: str) -> str:
    return f"你是京东客服团队的{title}。{domain_rules}{_COMMON_RULES}"


# --------------------------------------------------------------------------- #
# 装配                                                                          #
# --------------------------------------------------------------------------- #


def create_specialists(registry: ToolRegistry) -> dict[str, Specialist]:
    """从全量工具库切子集,装配三个业务专员。

    Args:
        registry: 全量工具库(``presets.default_registry()``);子集与其共享工具
            实例,跨专员的数据状态互通(售后退了款,订单专员再查看得到)。

    Returns:
        ``{机器名: Specialist}``,**插入顺序即兜底顺序**(第一个 = 兜底专员,
        Planner 矫正越界指派、降级单步计划都派给他)。
    """
    order_agent = Specialist(
        name="order_agent",
        title="订单物流专员",
        description="负责订单状态、物流进度、用户历史订单的一切查询;只查询、不做任何修改操作",
        registry=registry.subset(
            "query_order", "query_logistics", "query_user_orders", "current_time"
        ),
        system_prompt=_prompt(
            "订单物流专员",
            "你负责订单状态、物流进度、历史订单查询。只查询、不做任何修改操作。\n"
            "关键规则:用户问“到哪了 / 到哪儿了 / 什么时候到 / 物流进度”时,他要的是"
            "**包裹现在到了哪个城市/哪一站、预计何时送达**。你必须先 query_order 拿到"
            "物流单号,再**接着 query_logistics 查实际位置**,然后开门见山地回答当前所在"
            "地点和预计到达时间——例如“您的包裹已到达【北京分拣中心】,预计明天送达”。\n"
            "不要只答“已发货”就停下让用户再问一次;也不要罗列商品名、金额、下单时间等"
            "用户没问的信息(除非用户确实在问这些)。需要时附上物流单号即可。",
        ),
    )
    aftersales_agent = Specialist(
        name="aftersales_agent",
        title="售后专员",
        description="负责退款、取消订单、创建人工工单等售后处理;涉及资金与订单变更的操作找他",
        registry=registry.subset(
            "query_order", "apply_refund", "cancel_order", "create_ticket", "search_faq"
        ),
        system_prompt=_prompt(
            "售后专员",
            "你负责退款、取消订单、创建人工工单。执行任何操作前必须先用 query_order "
            "核实订单当前状态与条件;不符合条件时如实说明原因,并评估是否建工单转人工跟进,"
            "绝不强行操作。办成的操作要回报凭据(退款单号/工单号)。",
        ),
    )
    product_agent = Specialist(
        name="product_agent",
        title="商品导购专员",
        description="负责商品信息查询、商品推荐与价格计算;买什么、怎么选的问题找他",
        registry=registry.subset("query_product", "search_faq", "calculator"),
        system_prompt=_prompt(
            "商品导购专员",
            "你负责商品信息与推荐。推荐要给出理由(参数/价格对比);"
            "不清楚的商品参数如实说不知道,绝不编造。",
        ),
    )
    return {s.name: s for s in (order_agent, aftersales_agent, product_agent)}
