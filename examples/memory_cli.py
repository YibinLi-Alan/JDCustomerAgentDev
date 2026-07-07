"""阶段四交付物⑤ 带记忆的交互式客服 CLI(多用户 + 跨会话)。

在阶段三 ``tools_cli``(原生 Function Calling + 11 工具)之上接入 ``MemoryManager``:

- **短期记忆**:滑动窗口(token 预算)+ 溢出递归摘要成「前情提要」;
- **长期记忆**:每轮 LLM 提炼事实 → Mem0 式增删改 → Chroma 持久化(重启程序还在);
- **多用户隔离**:``/user`` 切换身份,各自的记忆互不可见(存储层强制过滤)。

演示脚本(答辩可用):

1. ``/user alice`` → 「我叫王小明,常用地址是上海市浦东新区张江路 88 号」
2. 聊几轮别的(订单、商品……),然后问「把东西寄到我常用地址,是哪里来着?」
3. ``/user bob`` → 问同样的问题 —— bob 什么都不知道(隔离)
4. 退出程序重开,``/user alice`` 再问 —— 还记得(跨会话,Chroma 落盘)
5. ``/memories`` 看 alice 的记忆;「我搬家了,新地址在北京市朝阳区」→ 再看
   ``/memories``,旧地址被 UPDATE 而不是多出一条(打开 ``/trace`` 能看到裁决)

运行:``python -m examples.memory_cli``(``.env`` 需配 ``PROVIDER`` 与对应 key;
长期记忆的 embedding 走 OpenAI,还需 ``OPENAI_API_KEY``,缺了会自动降级为纯短期记忆)。
"""

from __future__ import annotations

from agent_framework import (
    AgentResult,
    MemoryManager,
    ShortTermMemory,
    SummaryCompressor,
    ToolCallingAgent,
    Turn,
    create_llm,
    create_memory_manager,
    default_registry,
    get_settings,
)

_HELP_TEXT = """\
可用命令:
  /help              显示本帮助
  /exit, /quit       退出程序
  /user <id>         切换当前用户(默认 guest);各用户的长期记忆互相隔离
  /memories          查看当前用户的全部长期记忆
  /forget <id>       删除当前用户的某条记忆(id 见 /memories)
  /wipe              清空当前用户的全部长期记忆(「删除我的个人信息」)
  /reset             清空本会话(窗口+前情提要);长期记忆保留
  /trace             切换是否显示记忆轨迹(检索命中/写入决策/窗口压缩,默认关)
  其它任意文本        作为一个问题交给 Agent 处理
"""


def _print_help() -> None:
    """打印命令帮助。"""
    print(_HELP_TEXT, end="")


def main() -> None:
    """启动带记忆的交互式客服 CLI 主循环。"""
    settings = get_settings()
    llm = create_llm(settings)
    registry = default_registry()
    base_prompt = ToolCallingAgent(llm, registry).system_prompt  # 复用阶段三默认客服提示词

    try:
        manager = create_memory_manager(settings, llm)  # 全套:窗口+摘要+Chroma 长期记忆
        memory_mode = f"全套(长期记忆落盘 {settings.memory_persist_dir})"
    except ValueError as exc:  # 缺 OPENAI_API_KEY 等:降级为纯短期,demo 照跑
        manager = MemoryManager(
            short_term=ShortTermMemory(max_tokens=settings.memory_window_tokens),
            compressor=SummaryCompressor(llm, max_tokens=settings.memory_summary_max_tokens),
        )
        memory_mode = f"仅短期(长期记忆不可用:{exc})"

    user_id = "guest"
    show_trace = False

    print(f"带记忆的客服 CLI(模型:{llm.model};记忆:{memory_mode})")
    print(f"当前用户:{user_id}(用 /user <id> 切换);/help 查看命令,/exit 退出。")

    while True:
        try:
            line = input(f"{user_id}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        # ---- 命令分发 ----
        if line.startswith("/"):
            parts = line.split()
            command = parts[0].lower()

            if command in ("/exit", "/quit"):
                break
            if command == "/help":
                _print_help()
                continue
            if command == "/user":
                if len(parts) < 2:
                    print("[用法:/user <id>,例如 /user alice]")
                    continue
                user_id = parts[1]
                manager.reset_session()  # 换人 = 新会话(窗口/摘要清空;长期记忆按 user 隔离)
                print(f"[已切换到用户 {user_id};会话已重置,长期记忆按用户隔离]")
                continue
            if command == "/memories":
                if manager.long_term is None:
                    print("[长期记忆未启用]")
                    continue
                records = manager.long_term.list_memories(user_id)
                if not records:
                    print(f"[{user_id} 还没有长期记忆]")
                    continue
                for r in records:
                    print(f"  {r.id}  重要性={r.importance}  {r.text}")
                continue
            if command == "/forget":
                if manager.long_term is None or len(parts) < 2:
                    print("[用法:/forget <id>(id 见 /memories)]")
                    continue
                manager.long_term.forget(parts[1])
                print(f"[已删除记忆 {parts[1]}]")
                continue
            if command == "/wipe":
                if manager.long_term is None:
                    print("[长期记忆未启用]")
                    continue
                count = manager.long_term.delete_user(user_id)
                print(f"[已清空 {user_id} 的全部长期记忆,共 {count} 条]")
                continue
            if command == "/reset":
                manager.reset_session()
                print("[会话已清空(窗口+前情提要);长期记忆保留]")
                continue
            if command == "/trace":
                show_trace = not show_trace
                print(f"[记忆轨迹显示已{'开启' if show_trace else '关闭'}]")
                continue

            print(f"[未知命令:{command};输入 /help 查看可用命令]")
            continue

        # ---- 普通问题:load 记忆 → 跑 Agent → 落账 ----
        try:
            ctx = manager.load(user_id, line)
            if show_trace and ctx.retrieved:
                print("  [检索命中]")
                for s in ctx.retrieved:
                    print(
                        f"    {s.record.id}  总分={s.score:.2f}"
                        f"(相关{s.relevance:.2f}/时近{s.recency:.2f}/重要{s.importance:.2f})"
                        f"  {s.record.text}"
                    )
            # 记忆注入 system 附加段;Agent 无状态,按轮重建零成本、核心零改动。
            agent = ToolCallingAgent(
                llm,
                registry,
                max_steps=settings.agent_max_steps,
                system_prompt=base_prompt + ctx.system_suffix(),
            )
            res: AgentResult = agent.run(line, history=ctx.to_messages())
            print(f"assistant> {res.final_answer}")

            report = manager.on_turn_end(
                user_id, Turn(user_text=line, assistant_text=res.final_answer)
            )
            if show_trace:
                for op in report.write_ops:
                    target = f" → {op.target_id}" if op.target_id else ""
                    print(f"  [记忆写入] {op.action}{target}  {op.fact}(重要性 {op.importance})")
                if report.evicted_turns:
                    print(
                        f"  [窗口压缩] 弹出 {report.evicted_turns} 轮"
                        f"{',前情提要已更新' if report.summary_updated else ''}"
                    )
        except Exception as exc:  # noqa: BLE001 - demo:任何失败都友好提示,不崩溃
            print(f"[处理失败:{exc}]")

    print("再见 👋")


if __name__ == "__main__":
    main()
