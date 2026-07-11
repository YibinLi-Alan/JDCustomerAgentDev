"""HITL 人工介入 —— 审批闸门 + 统一人工队列 + 审批后执行(阶段六 P-B 核心)。

业务闭环(stage-6-design.md §7,评审拍板「拦截-挂起-审批-执行」):

- **一个队列,两个入口**:
  入口 A(approval)= :class:`ApprovalGate` 在工具调用点拦下的高权限动作
  (挂起工具名+参数+幂等 request_id);
  入口 B(escalation)= 编排层的兜底升级(重规划耗尽仍失败/质检二审不合格/
  整任务超时),带轨迹摘要作为上下文快照;
- **拦截点在 Registry 层**:专员/编排一行不改,给谁装闸门 = 装配时包一下
  (闸门是插件不是改造);审批规则由 :class:`ApprovalPolicy` **配置驱动**;
- **审批后执行**:人工 approve → 真正执行挂起的工具调用(request_id 幂等,
  重放安全)→ 结果记入 resolution;
- 队列 JSON 落盘(重启不丢);生产换数据库,接口不变。
  注意:mock 数据(JDMockStore)是进程内的——跨进程审批执行落在**审批进程**
  自己的 store 上,demo 时在同一进程(CLI/API)内审批可见完整效果。
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Collection
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from agent_framework.tools.base import ToolResult
from agent_framework.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from agent_framework.core.config import Settings

HandoffKind = Literal["approval", "escalation"]
HandoffStatus = Literal["pending", "done", "rejected", "closed"]


@dataclass(frozen=True)
class PendingAction:
    """被闸门挂起、等待人工放行的工具调用。"""

    tool: str
    args: dict[str, object]
    request_id: str  # 框架生成的幂等 ID:审批执行可安心重放


@dataclass
class HandoffItem:
    """人工介入队列中的一项。

    Attributes:
        id: 单号(短 uuid,用户话术与控制台都用它)。
        kind: ``approval``(高权限审批)/ ``escalation``(兜底升级)。
        user_id: 归属用户。
        created_at: ISO 时间。
        status: pending → done(审批执行完)/ rejected(驳回)/ closed(人工办结)。
        summary: 一句话摘要(控制台列表用)。
        action: 挂起的工具调用(approval 才有)。
        context: 上下文快照(任务原文/轨迹摘要,人工判断的依据)。
        resolution: 处理结果/人工备注。
    """

    id: str
    kind: HandoffKind
    user_id: str
    created_at: str
    status: HandoffStatus
    summary: str
    action: PendingAction | None = None
    context: str = ""
    resolution: str = ""


class ApprovalPolicy:
    """审批规则:哪些权限级别需要人工放行(配置驱动,不写死 if/else)。"""

    def __init__(self, required: Collection[str] = ("high",)) -> None:
        self._required = frozenset(required)

    @classmethod
    def from_settings(cls, settings: Settings) -> ApprovalPolicy:
        return cls(settings.approval_required_permissions)

    def requires_approval(self, permission: str) -> bool:
        return permission in self._required


class HandoffQueue:
    """统一人工介入队列(JSON 文件落盘;path=None 纯内存,测试用)。"""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        now_fn: Callable[[], datetime] = datetime.now,
        id_fn: Callable[[], str] | None = None,
    ) -> None:
        self._path = Path(path) if path is not None else None
        self._now = now_fn
        self._id_fn = id_fn or (lambda: uuid.uuid4().hex[:8])
        self._items: dict[str, HandoffItem] = {}
        self._load()

    # ------------------------------ 两个入口 ------------------------------ #
    def submit_action(
        self, *, user_id: str, tool: str, args: dict[str, object], context: str = ""
    ) -> HandoffItem:
        """入口 A:高权限动作挂起待审批(由 ApprovalGate 调用)。"""
        item = HandoffItem(
            id=self._id_fn(),
            kind="approval",
            user_id=user_id,
            created_at=self._now().isoformat(timespec="seconds"),
            status="pending",
            summary=f"待审批:{tool}({json.dumps(args, ensure_ascii=False)})",
            action=PendingAction(tool=tool, args=dict(args), request_id=uuid.uuid4().hex),
            context=context,
        )
        self._items[item.id] = item
        self._save()
        return item

    def submit_escalation(self, *, user_id: str, reason: str, context: str = "") -> HandoffItem:
        """入口 B:Agent 办不了的升级人工(重规划耗尽/质检不过/超时)。"""
        item = HandoffItem(
            id=self._id_fn(),
            kind="escalation",
            user_id=user_id,
            created_at=self._now().isoformat(timespec="seconds"),
            status="pending",
            summary=f"转人工:{reason}",
            context=context,
        )
        self._items[item.id] = item
        self._save()
        return item

    # ------------------------------ 人工控制台 ------------------------------ #
    def list(
        self, *, status: HandoffStatus | None = None, kind: HandoffKind | None = None
    ) -> list[HandoffItem]:
        items = list(self._items.values())
        if status is not None:
            items = [i for i in items if i.status == status]
        if kind is not None:
            items = [i for i in items if i.kind == kind]
        return sorted(items, key=lambda i: i.created_at)

    def get(self, item_id: str) -> HandoffItem:
        if item_id not in self._items:
            raise KeyError(f"人工介入单 {item_id!r} 不存在")
        return self._items[item_id]

    def approve(self, item_id: str, registry: ToolRegistry, *, note: str = "") -> ToolResult:
        """放行一张审批单:**真正执行**挂起的工具调用(幂等重放安全)。

        Args:
            item_id: 审批单号。
            registry: 执行用的工具库(控制台侧装配,与业务共享数据源时效果完整)。
            note: 人工备注。

        Returns:
            执行结果(同时写入 ``resolution``,状态置 done)。

        Raises:
            KeyError: 单号不存在。
            ValueError: 不是 approval 单,或不在 pending 状态(防重复审批)。
        """
        item = self.get(item_id)
        if item.kind != "approval" or item.action is None:
            raise ValueError(f"{item_id} 不是待审批的动作单(kind={item.kind})")
        if item.status != "pending":
            raise ValueError(f"{item_id} 已处理过(status={item.status}),不可重复审批")
        result = registry.invoke(
            item.action.tool, item.action.args, request_id=item.action.request_id
        )
        item.status = "done"
        note_part = f"(备注:{note})" if note else ""
        item.resolution = f"已人工放行并执行{note_part}:{result.to_observation()}"
        self._save()
        return result

    def reject(self, item_id: str, *, note: str = "") -> HandoffItem:
        """驳回一张审批单(动作不执行)。"""
        item = self.get(item_id)
        if item.status != "pending":
            raise ValueError(f"{item_id} 已处理过(status={item.status})")
        item.status = "rejected"
        item.resolution = f"人工驳回{':' + note if note else ''}"
        self._save()
        return item

    def close(self, item_id: str, *, resolution: str) -> HandoffItem:
        """办结一张升级单(人工处理完毕,记录处理结果)。"""
        item = self.get(item_id)
        if item.status != "pending":
            raise ValueError(f"{item_id} 已处理过(status={item.status})")
        item.status = "closed"
        item.resolution = resolution
        self._save()
        return item

    # ------------------------------ 落盘 ------------------------------ #
    def _save(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = [asdict(item) for item in self._items.values()]
            self._path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
        except OSError:
            pass  # 落盘是旁路;真丢了控制台看不到,但业务答复已发出

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return  # 队列文件损坏:从空队列开始,不炸启动
        for entry in raw:
            action = None
            if entry.get("action"):
                action = PendingAction(
                    tool=str(entry["action"]["tool"]),
                    args=dict(entry["action"]["args"]),
                    request_id=str(entry["action"]["request_id"]),
                )
            item = HandoffItem(
                id=str(entry["id"]),
                kind=entry["kind"],
                user_id=str(entry["user_id"]),
                created_at=str(entry["created_at"]),
                status=entry["status"],
                summary=str(entry.get("summary", "")),
                action=action,
                context=str(entry.get("context", "")),
                resolution=str(entry.get("resolution", "")),
            )
            self._items[item.id] = item


# --------------------------------------------------------------------------- #
# 审批闸门(Registry 装饰器)                                                     #
# --------------------------------------------------------------------------- #


class ApprovalGate:
    """权限闸门:高权限工具调用不执行,挂起入队;低权限透传(registry 鸭子协议)。

    与 ``BoundaryRegistry`` 组合使用(边界在内层贴数据,闸门在外层管权限)::

        gated = ApprovalGate(BoundaryRegistry(registry), queue, policy,
                             user_id_provider=lambda: current_user)
    """

    def __init__(
        self,
        inner: ToolRegistry,
        queue: HandoffQueue,
        policy: ApprovalPolicy,
        *,
        user_id_provider: Callable[[], str] = lambda: "guest",
        context_provider: Callable[[], str] = lambda: "",
        on_pending: Callable[[HandoffItem], None] | None = None,
    ) -> None:
        """
        Args:
            inner: 被包装的工具库(或另一个包装层)。
            queue: 人工介入队列。
            policy: 审批规则(配置驱动)。
            user_id_provider: 取当前用户 id(框架注入,不进 prompt——老规矩)。
            context_provider: 取当前任务的上下文快照(人工判断依据)。
            on_pending: 挂起时回调(编排层挂 tracer.emit,SSE/轨迹可见)。
        """
        self._inner = inner
        self._queue = queue
        self._policy = policy
        self._user_id = user_id_provider
        self._context = context_provider
        self._on_pending = on_pending

    def invoke(
        self,
        name: str,
        args: dict[str, object] | None = None,
        *,
        request_id: str | None = None,
    ) -> ToolResult:
        try:
            tool = self._inner.get(name)
        except Exception:  # noqa: BLE001 — 未知工具交给内层折叠标准错误
            return self._inner.invoke(name, args, request_id=request_id)
        if self._policy.requires_approval(tool.permission):
            item = self._queue.submit_action(
                user_id=self._user_id(),
                tool=name,
                args=dict(args or {}),
                context=self._context(),
            )
            if self._on_pending is not None:
                try:
                    self._on_pending(item)
                except Exception:  # noqa: BLE001 — 观测旁路不拖垮业务
                    pass
            return ToolResult(
                ok=True,
                content=(
                    f"该操作涉及高权限,已提交人工审批,审批单号 {item.id},"
                    "预计 24 小时内处理。请把审批单号告知用户并说明等待审批结果,"
                    "不要重复提交同一操作。"
                ),
                data=item,
            )
        return self._inner.invoke(name, args, request_id=request_id)

    # ---- registry 鸭子协议透传 ---- #
    def to_schemas(self) -> list[dict[str, object]]:
        return self._inner.to_schemas()

    def get(self, name: str):  # type: ignore[no-untyped-def]
        return self._inner.get(name)

    @property
    def names(self) -> list[str]:
        return self._inner.names

    def render_catalog(self) -> str:
        return self._inner.render_catalog()
