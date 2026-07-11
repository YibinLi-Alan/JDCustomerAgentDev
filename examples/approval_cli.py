"""阶段六 人工控制台 —— 审批/驳回高权限动作,办结升级单(HITL 的人这一侧)。

用法:``python -m examples.approval_cli``,命令:
    list [pending|done|rejected|closed]   列出介入单(默认 pending)
    show <id>                             看单详情(含上下文快照)
    approve <id> [备注]                   放行:真正执行挂起的工具调用(幂等)
    reject <id> [理由]                    驳回:动作不执行
    close <id> <处理结果>                 办结升级单(人工处理完毕)
    help / exit

注意:mock 数据是进程内的——本控制台 approve 时在**自己进程**的 JDMockStore 上
执行(演示审批闭环的机制);与业务进程共享数据库后即为完整生产形态。
"""

from __future__ import annotations

from agent_framework.core.config import get_settings
from agent_framework.safety import HandoffItem, HandoffQueue
from agent_framework.tools.presets import default_registry

_STATUS_ICONS = {"pending": "⏳", "done": "✅", "rejected": "🚫", "closed": "📁"}


def _print_item(item: HandoffItem, *, detail: bool = False) -> None:
    icon = _STATUS_ICONS.get(item.status, "·")
    kind = "审批" if item.kind == "approval" else "升级"
    print(f"  {icon} [{item.id}] ({kind}/{item.status}) user={item.user_id}  {item.summary}")
    if detail:
        print(f"      创建:{item.created_at}")
        if item.action is not None:
            print(f"      挂起动作:{item.action.tool}  参数:{item.action.args}")
        if item.context:
            print("      上下文:\n        " + item.context.replace("\n", "\n        "))
        if item.resolution:
            print(f"      处理结果:{item.resolution}")


def main() -> None:
    settings = get_settings()
    queue = HandoffQueue(settings.handoff_store_path)
    registry = default_registry()  # 审批执行用的工具库(与业务共享数据源时效果完整)

    pending = queue.list(status="pending")
    print(f"人工控制台(队列:{settings.handoff_store_path};待处理 {len(pending)} 单)")
    print(
        "命令:list / show <id> / approve <id> [备注] / reject <id> [理由]"
        " / close <id> <结果> / exit"
    )

    while True:
        try:
            line = input("staff> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        parts = line.split(maxsplit=2)
        command = parts[0].lower()

        try:
            if command in ("exit", "quit"):
                break
            elif command == "help":
                print(__doc__)
            elif command == "list":
                status = parts[1] if len(parts) > 1 else "pending"
                items = queue.list(status=status)  # type: ignore[arg-type]
                if not items:
                    print(f"  [{status} 状态没有介入单]")
                for item in items:
                    _print_item(item)
            elif command == "show" and len(parts) > 1:
                _print_item(queue.get(parts[1]), detail=True)
            elif command == "approve" and len(parts) > 1:
                note = parts[2] if len(parts) > 2 else ""
                result = queue.approve(parts[1], registry, note=note)
                print(f"  [已放行并执行] {result.to_observation()}")
            elif command == "reject" and len(parts) > 1:
                note = parts[2] if len(parts) > 2 else ""
                item = queue.reject(parts[1], note=note)
                print(f"  [已驳回] {item.id}")
            elif command == "close" and len(parts) > 2:
                item = queue.close(parts[1], resolution=parts[2])
                print(f"  [已办结] {item.id}")
            else:
                print("  [无效命令;输入 help 查看用法]")
        except (KeyError, ValueError) as exc:
            print(f"  [操作失败:{exc}]")


if __name__ == "__main__":
    main()
