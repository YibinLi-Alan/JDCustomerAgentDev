"""交付物① 多轮对话 CLI。

一个朴素但完整的命令行聊天循环:维护全量对话历史(多轮记忆最原始的形态),
支持几条管理命令、可选流式输出、token 统计与基础错误处理。
详见 stage-1-design.md §6。

运行:``python -m examples.chat_cli``(需先在 ``.env`` 配好 ``PROVIDER`` 与对应 key)。

注意:本脚本只 import ``agent_framework`` 的接口与工厂,不直接 import 任何厂商 SDK。
"""

from __future__ import annotations

from agent_framework import LLM, Message, create_llm, get_settings

_HELP_TEXT = """\
可用命令:
  /help              显示本帮助
  /exit, /quit       退出程序
  /reset             清空对话历史(保留当前 system prompt)
  /system <文本>     设定/覆盖 system prompt,并自动清空历史
  /stream            切换流式 / 非流式输出
  其它任意文本        作为一句话发给模型
"""


def _print_help() -> None:
    print(_HELP_TEXT, end="")


def main() -> None:
    """启动多轮对话 CLI 主循环。"""
    settings = get_settings()
    llm: LLM = create_llm(settings)  # 唯一装配点;换厂商/模型只改 .env(PROVIDER/MODEL)

    history: list[Message] = []
    system: str | None = None
    stream_on = settings.stream

    print(f"多轮对话 CLI(模型:{llm.model},流式:{'开' if stream_on else '关'})")
    print("输入 /help 查看命令,/exit 退出。")

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
            command, _, argument = line.partition(" ")
            command = command.lower()
            argument = argument.strip()

            if command in ("/exit", "/quit"):
                break
            if command == "/help":
                _print_help()
                continue
            if command == "/reset":
                history.clear()
                print("[已清空对话历史]")
                continue
            if command == "/system":
                if not argument:
                    print("[用法:/system <文本>]")
                    continue
                system = argument
                history.clear()  # system 变更后旧上下文语义可能错位,自动 reset
                print("[已设定 system prompt 并清空历史]")
                continue
            if command == "/stream":
                stream_on = not stream_on
                print(f"[流式输出已{'开启' if stream_on else '关闭'}]")
                continue

            print(f"[未知命令:{command};输入 /help 查看可用命令]")
            continue

        # ---- 普通对话 ----
        history.append(Message("user", line))
        try:
            if stream_on:
                print("assistant> ", end="", flush=True)
                chunks: list[str] = []
                for delta in llm.stream(history, system=system):
                    print(delta, end="", flush=True)
                    chunks.append(delta)
                reply = "".join(chunks)
                print()  # 收尾换行
            else:
                resp = llm.chat(history, system=system)
                reply = resp.content
                print("assistant>", reply)
                print(
                    f"[tokens in={resp.usage.input_tokens} "
                    f"out={resp.usage.output_tokens} "
                    f"total={resp.usage.total_tokens}]"
                )
            history.append(Message("assistant", reply))
        except Exception as exc:  # noqa: BLE001 - 阶段一:任何调用失败都友好提示,不崩溃
            # 回滚刚追加的 user 消息,避免脏历史污染后续请求。
            history.pop()
            print(f"\n[调用失败:{exc}]")

    print("再见 👋")


if __name__ == "__main__":
    main()
