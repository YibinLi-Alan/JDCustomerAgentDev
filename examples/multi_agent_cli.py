"""阶段五交付物⑤ 多 Agent 协作客服 CLI(Router 分诊 + Supervisor 中心调度 + 记忆)。

在阶段四 ``memory_cli`` 之上换上多 Agent 编排:

- **Router 分诊**:简单问题直派单个专员(快路径,省钱省时);
- **Supervisor 调度**:复杂问题 → Planner 拆步骤 → 三专员接力(ScratchPad 传中间
  结果)→ 失败动态重规划 → 汇总 → Critic 质检(不合格回炉一次);
- **记忆沿用阶段四**:``/user`` 切换身份,前情提要 + 用户事实注入每个专员。

演示脚本(答辩可用,/trace 开轨迹):

1. 快路径:「订单 12345 到哪了」→ 分诊直派订单物流专员;
2. 复合客诉一条龙:「我买的蓝牙耳机(订单 11111)用了几天就坏了,查一下还能不能退,
   能退帮我退掉,再推荐一个靠谱的替代品」→ 升级 Supervisor →
   计划(查证→退款→推荐)→ 订单 11111 已超 7 天无理由期,退款步受阻 →
   重规划/售后专员建工单转人工 → 导购推荐 → Critic 审 → 输出完整方案;
3. ``/mode supervisor`` 强制中心调度,对比同一问题两种路径的轨迹。

运行:``python -m examples.multi_agent_cli``(``.env`` 配 ``PROVIDER`` 与对应 key;
长期记忆需 ``OPENAI_API_KEY``,缺了自动降级为纯短期记忆)。
"""

from __future__ import annotations

from agent_framework import (
    Critic,
    MemoryManager,
    Router,
    ShortTermMemory,
    SummaryCompressor,
    Supervisor,
    SupervisorResult,
    TaskAssignment,
    Turn,
    create_llm,
    create_memory_manager,
    create_specialists,
    default_registry,
    get_settings,
)
from agent_framework.multi_agent.protocol import FAILURE_MARKER
from agent_framework.multi_agent.router import SUPERVISOR_TARGET
from agent_framework.planning import Planner

_HELP_TEXT = """\
可用命令:
  /help              显示本帮助
  /exit, /quit       退出程序
  /user <id>         切换当前用户(默认 guest);各用户的长期记忆互相隔离
  /mode <m>          处理模式:auto(分诊决定,默认)/ supervisor(强制中心调度)
                     / router(观察分诊:判复杂时提示后仍升级处理)
  /trace             切换是否显示协作轨迹(分诊/计划/派工/重规划/质检,默认开)
  /reset             清空本会话(窗口+前情提要);长期记忆保留
  其它任意文本        作为一个诉求交给客服团队处理
"""


def _print_help() -> None:
    """打印命令帮助。"""
    print(_HELP_TEXT, end="")


def _clip(text: str, limit: int = 90) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _strip_failure_marker(answer: str) -> str:
    """剥掉「无法完成:」协议前缀再展示给用户。

    该前缀是专员↔编排器的内部状态码(protocol.FAILURE_MARKER);Supervisor 路径
    由汇总调用消化,快路径直出专员原话,须在显示层剥掉,内部协议不外泄。
    """
    text = answer.strip()
    if text.startswith(FAILURE_MARKER):
        text = text[len(FAILURE_MARKER) :].lstrip(":: \n")
    return text or answer


def _print_supervisor_trace(result: SupervisorResult) -> None:
    """打印中心调度的完整轨迹(计划/派工/重规划/质检)。"""
    print("  [计划]")
    for step in result.plan.steps if result.plan else ():
        print(f"    step-{step.id} → {step.specialist}:{step.description}")
    print("  [执行]")
    for r in result.step_results:
        mark = "✓" if r.ok else "✗"
        print(f"    {mark} step-{r.step.id}({r.step.specialist}){_clip(r.output)}")
    if result.replanned:
        print("  [重规划] 有步骤失败,剩余计划已重排(见上方新步骤)")
    for i, critique in enumerate(result.critiques, 1):
        verdict = "通过" if critique.passed else f"不合格:{'; '.join(critique.issues)}"
        degraded = "(降级放行)" if critique.degraded else ""
        print(f"  [质检 第{i}轮] {verdict}{degraded}")
    if result.resynthesized:
        print("  [回炉] 已按质检意见重写答复")


def main() -> None:
    """启动多 Agent 协作客服 CLI 主循环。"""
    settings = get_settings()
    llm = create_llm(settings)
    registry = default_registry()

    # —— 团队装配:三专员 + 分诊 + 规划 + 质检 + 中心调度(全部可独立替换)——
    specialists = create_specialists(registry)
    router = Router(llm, specialists)
    planner = Planner(llm, max_steps=settings.planner_max_steps)
    supervisor = Supervisor(
        llm,
        specialists,
        planner=planner,
        critic=Critic(llm),
        max_steps_per_specialist=settings.supervisor_specialist_max_steps,
        max_replans=settings.max_replans,
        critic_max_retries=settings.critic_max_retries,
    )

    # —— 记忆沿用阶段四;缺 embedding key 自动降级纯短期 ——
    try:
        manager = create_memory_manager(settings, llm)
        memory_mode = f"全套(长期记忆落盘 {settings.memory_persist_dir})"
    except ValueError as exc:
        manager = MemoryManager(
            short_term=ShortTermMemory(max_tokens=settings.memory_window_tokens),
            compressor=SummaryCompressor(llm, max_tokens=settings.memory_summary_max_tokens),
        )
        memory_mode = f"仅短期(长期记忆不可用:{exc})"

    user_id = "guest"
    mode = "auto"
    show_trace = True

    print(f"多 Agent 协作客服 CLI(模型:{llm.model};记忆:{memory_mode})")
    print(f"团队:{', '.join(s.title for s in specialists.values())} + 分诊/规划/质检")
    print(f"当前用户:{user_id};模式:{mode};/help 查看命令,/exit 退出。")

    while True:
        try:
            line = input(f"{user_id}({mode})> ").strip()
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
                manager.reset_session()
                print(f"[已切换到用户 {user_id};会话已重置,长期记忆按用户隔离]")
                continue
            if command == "/mode":
                if len(parts) < 2 or parts[1] not in ("auto", "supervisor", "router"):
                    print("[用法:/mode auto|supervisor|router]")
                    continue
                mode = parts[1]
                print(f"[处理模式已切换为 {mode}]")
                continue
            if command == "/trace":
                show_trace = not show_trace
                print(f"[协作轨迹显示已{'开启' if show_trace else '关闭'}]")
                continue
            if command == "/reset":
                manager.reset_session()
                print("[会话已清空(窗口+前情提要);长期记忆保留]")
                continue

            print(f"[未知命令:{command};输入 /help 查看可用命令]")
            continue

        # ---- 普通诉求:load 记忆 → 分诊/调度 → 落账 ----
        try:
            ctx = manager.load(user_id, line)
            mem_suffix = ctx.system_suffix()

            # 分诊(supervisor 模式跳过)
            if mode == "supervisor":
                target = SUPERVISOR_TARGET
            else:
                decision = router.route(line, context=mem_suffix)
                target = decision.target
                if show_trace:
                    print(f"  [分诊] → {target}({decision.reason})")
                if mode == "router" and target == SUPERVISOR_TARGET:
                    print("  [提示] 分诊判定为复杂问题,本轮升级中心调度处理")

            if target == SUPERVISOR_TARGET:
                # 近期对话拼进背景:Planner 拆解与专员执行都能解析跨轮指代
                recent = "\n".join(t.render_text() for t in ctx.recent_turns)
                background = mem_suffix + (f"\n\n【近期对话】\n{recent}" if recent else "")
                result = supervisor.handle(line, memory_context=background)
                answer = result.final_answer
                if show_trace:
                    _print_supervisor_trace(result)
            else:
                outcome = specialists[target].handle(
                    llm,
                    TaskAssignment(task=line),
                    extra_system=mem_suffix,
                    max_steps=settings.agent_max_steps,
                    history=ctx.to_messages(),  # 快路径带外层历史,跨轮指代接得上
                )
                answer = _strip_failure_marker(outcome.answer)
                if show_trace and not outcome.ok:
                    print(f"  [专员回报] {outcome.specialist} 未能完成任务(内部标记已剥离)")

            print(f"assistant> {answer}")

            report = manager.on_turn_end(user_id, Turn(user_text=line, assistant_text=answer))
            if show_trace:
                for op in report.write_ops:
                    op_target = f" → {op.target_id}" if op.target_id else ""
                    print(f"  [记忆写入] {op.action}{op_target}  {op.fact}(重要性 {op.importance})")
                if report.evicted_turns:
                    print(
                        f"  [窗口压缩] 弹出 {report.evicted_turns} 轮"
                        f"{',前情提要已更新' if report.summary_updated else ''}"
                    )
        except Exception as exc:  # noqa: BLE001 - demo:任何失败都友好提示,不崩溃
            print(f"[处理失败:{exc}]")


if __name__ == "__main__":
    main()
