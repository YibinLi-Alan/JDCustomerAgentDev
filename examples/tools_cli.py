"""阶段三交付物③ 交互式客服 CLI(原生 Function Calling 版)。

与阶段二 ``react_cli`` 同构的交互循环,但换成 :class:`ToolCallingAgent`:
工具 Schema 走厂商原生协议下发,模型返回结构化 ``tool_calls``(支持一步并行多个),
装配的是全套 11 个工具(``default_registry()``:8 业务 + 3 通用)。

试试这些问题,观察 /trace 里的工具选择:

- 我的订单 12345 到哪了?          (query_order → query_logistics 两连)
- 我最近买的东西都到哪了?          (query_user_orders → 逐单查询,可并行)
- 蓝牙耳机还有货吗?保修多久?      (query_product)
- 键盘不想要了,帮我取消            (query_user_orders → cancel_order,高权限)
- 这两个订单一共花了多少钱?        (calculator)

运行:``python -m examples.tools_cli``(需先在 ``.env`` 配好 ``PROVIDER`` 与对应 key)。

注意:本脚本只 import ``agent_framework`` 的公开接口与工厂,不直接 import 任何厂商 SDK。
"""

from __future__ import annotations

from agent_framework import (
    AgentResult,
    Message,
    StepTrace,
    ToolCallingAgent,
    create_llm,
    default_registry,
    get_settings,
)

_HELP_TEXT = """\
可用命令:
  /help              显示本帮助
  /exit, /quit       退出程序
  /reset             清空对话历史(外层干净问答对)
  /trace             切换是否显示中间步骤(默认关)
  /tools             列出已装配的工具(名称 · 权限)
  其它任意文本        作为一个问题交给 Agent 处理
"""


def _print_help() -> None:
    """打印命令帮助。"""
    print(_HELP_TEXT, end="")


def _print_trace(steps: list[StepTrace]) -> None:
    """逐步打印本轮的中间轨迹(``/trace`` 打开时)。

    Args:
        steps: 一次 ``run()`` 的完整轨迹,每步一条 :class:`StepTrace`。
    """
    for i, step in enumerate(steps, start=1):
        print(f"  第{i}步:")
        if step.thought is not None:
            print(f"    Thought: {step.thought}")
        if step.action is not None:
            print(f"    ToolCall: {step.action.tool} 参数={step.action.input}")
        if step.observation is not None:
            print(f"    Observation: {step.observation}")
        if step.final_answer is not None:
            print(f"    FinalAnswer: {step.final_answer}")


def main() -> None:
    """启动交互式客服 CLI(原生 Function Calling)主循环。"""
    settings = get_settings()
    llm = create_llm(settings)  # 唯一装配点;换厂商/模型只改 .env(PROVIDER/MODEL)
    registry = default_registry()
    agent = ToolCallingAgent(llm, registry, max_steps=settings.agent_max_steps)

    conversation: list[Message] = []  # 外层历史:只存干净的 (问题, 答案) 对
    show_trace = False

    print(f"交互式客服 CLI · 原生 Function Calling(模型:{llm.model},最大步数:{agent.max_steps})")
    print(f"已装配 {len(registry)} 个工具:{', '.join(registry.names)}")
    print("输入一个问题即可;输入 /help 查看命令,/exit 退出。")

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()  # 换行,保持终端整洁
            break

        if not line:
            continue

        # ---- 命令分发 ----
        if line.startswith("/"):
            command = line.split()[0].lower()

            if command in ("/exit", "/quit"):
                break
            if command == "/help":
                _print_help()
                continue
            if command == "/reset":
                conversation.clear()
                print("[已清空对话历史]")
                continue
            if command == "/trace":
                show_trace = not show_trace
                print(f"[中间步骤显示已{'开启' if show_trace else '关闭'}]")
                continue
            if command == "/tools":
                for t in registry:
                    print(f"  {t.name:<20} 权限={t.permission}")
                continue

            print(f"[未知命令:{command};输入 /help 查看可用命令]")
            continue

        # ---- 普通问题:跑一轮 Function Calling 循环 ----
        try:
            res: AgentResult = agent.run(line, history=conversation)
            if show_trace:
                _print_trace(res.steps)
            print(f"assistant> {res.final_answer}")
            print(f"[stopped_reason={res.stopped_reason}, steps={len(res.steps)}]")
            # 成功才写入外层历史,且只留干净问答对(不含中间轨迹)
            conversation.append(Message("user", line))
            conversation.append(Message("assistant", res.final_answer))
        except Exception as exc:  # noqa: BLE001 - demo:任何失败都友好提示,不崩溃
            # 失败这一轮不写进 conversation,保持外层历史干净。
            print(f"[处理失败:{exc}]")

    print("再见 👋")


if __name__ == "__main__":
    main()
