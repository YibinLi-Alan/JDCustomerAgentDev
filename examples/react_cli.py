"""交付物② 交互式多轮 ReAct 客服 CLI。

一个交互式的命令行循环:每一轮把用户问题交给 :class:`ReActAgent` 跑完整的
``Thought → Action → Observation`` 循环,再打印最终答案。详见 stage-2-design.md §6.2。

上下文分内外两层(见 §6.2):

- **内层** ``context``:单个问题的 Thought/Action/Observation 草稿,是 ``agent.run()``
  的局部变量,run 结束即丢弃。
- **外层** ``conversation``:本 CLI 持有的**干净 (用户问题, 最终答案) 对**,不留任何
  中间推理;进程存活期间有效,``/reset`` 或 ``/exit`` 即清空。

不做持久化:关掉即忘,磁盘不留痕(持久化 / 长期记忆留到阶段四)。

运行:``python -m examples.react_cli``(需先在 ``.env`` 配好 ``PROVIDER`` 与对应 key)。

注意:本脚本只 import ``agent_framework`` 的公开接口与工厂,不直接 import 任何厂商 SDK。
"""

from __future__ import annotations

from agent_framework import (
    JD_MOCK_TOOLS,
    AgentResult,
    Message,
    ReActAgent,
    StepTrace,
    create_llm,
    get_settings,
)

_HELP_TEXT = """\
可用命令:
  /help              显示本帮助
  /exit, /quit       退出程序
  /reset             清空对话历史(外层干净问答对)
  /trace             切换是否显示中间推理步骤(默认关)
  其它任意文本        作为一个问题交给 ReAct Agent 处理
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
            print(f"    Action: {step.action.tool} 参数={step.action.input}")
        if step.observation is not None:
            print(f"    Observation: {step.observation}")
        if step.final_answer is not None:
            print(f"    FinalAnswer: {step.final_answer}")
        if step.error is not None:
            print(f"    [解析失败] {step.error}")


def main() -> None:
    """启动交互式多轮 ReAct 客服 CLI 主循环。"""
    settings = get_settings()
    llm = create_llm(settings)  # 唯一装配点;换厂商/模型只改 .env(PROVIDER/MODEL)
    agent = ReActAgent(llm, JD_MOCK_TOOLS, max_steps=settings.agent_max_steps)

    conversation: list[Message] = []  # 外层历史:只存干净的 (问题, 答案) 对
    show_trace = False

    tool_names = ", ".join(tool.name for tool in JD_MOCK_TOOLS)
    print(f"交互式 ReAct 客服 CLI(模型:{llm.model},最大步数:{agent.max_steps})")
    print(f"可用工具:{tool_names}")
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

            print(f"[未知命令:{command};输入 /help 查看可用命令]")
            continue

        # ---- 普通问题:跑一轮 ReAct ----
        try:
            res: AgentResult = agent.run(line, history=conversation)
            if show_trace:
                _print_trace(res.steps)
            print(f"assistant> {res.final_answer}")
            print(f"[stopped_reason={res.stopped_reason}, steps={len(res.steps)}]")
            # 成功才写入外层历史,且只留干净问答对(不含中间推理)
            conversation.append(Message("user", line))
            conversation.append(Message("assistant", res.final_answer))
        except Exception as exc:  # noqa: BLE001 - 阶段二:任何失败都友好提示,不崩溃
            # 失败这一轮不写进 conversation,保持外层历史干净。
            print(f"[处理失败:{exc}]")

    print("再见 👋")


if __name__ == "__main__":
    main()
