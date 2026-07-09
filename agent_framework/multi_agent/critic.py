"""Critic —— 终稿质检(评审拍板:审最终答复,不合格带意见回炉一次)。

覆盖大纲 5.3 三项:Reflection(执行后检查)= :meth:`Critic.review`;
Critic Agent = 本类;自动重试/替代方案 = Supervisor 的回炉 + Planner 的重规划。

设计边界(stage-5-design.md §7):

- 只审**终稿**不审每步(多数步骤是简单工具查询,审了也白审,调用量翻倍);
- Critic 对照**执行证据**审,不是凭空审;
- 解析失败 → 视为通过:Critic 是增益件,不能成为新的故障点(降级哲学)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_framework.core.llm import LLM, Message
from agent_framework.multi_agent.protocol import parse_json_object

CRITIC_SYSTEM = (
    "你是京东客服质检员。对照执行证据,审查这份给用户的答复:\n"
    "① 用户的每一项诉求都被回应了吗(一条都不能漏)?\n"
    "② 答复内容与执行证据一致吗,有没有编造证据里没有的信息?\n"
    "③ 没办成的事,是否如实告知了原因、给了替代方案(如转人工)?\n"
    "④ 语气专业友好吗?\n"
    "只输出 JSON,不要多余文字/解释/代码围栏:\n"
    '{"passed": true/false, "issues": ["问题1", "问题2"], "suggestion": "一句话修改建议"}'
)


@dataclass(frozen=True)
class Critique:
    """一次审查结论。

    Attributes:
        passed: 是否合格。
        issues: 具体问题清单(回炉时拼进汇总 prompt)。
        suggestion: 一句话修改建议。
        degraded: True 表示这是解析失败的降级放行,不是模型真实结论(trace 用)。
    """

    passed: bool
    issues: list[str] = field(default_factory=list)
    suggestion: str = ""
    degraded: bool = False


class Critic:
    """终稿质检员(一次 LLM 调用,无循环;回炉循环由 Supervisor 掌握)。"""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def review(self, query: str, answer: str, evidence: str) -> Critique:
        """对照证据审查答复。任何失败降级为通过,永不抛。

        Args:
            query: 用户原始诉求。
            answer: 待审的最终答复。
            evidence: 执行证据(各步骤结果的渲染文本)。

        Returns:
            :class:`Critique`;解析失败时 ``passed=True, degraded=True``。
        """
        prompt = (
            f"【用户诉求】\n{query}\n\n"
            f"【执行证据】\n{evidence or '(无)'}\n\n"
            f"【待审答复】\n{answer}"
        )
        try:
            response = self._llm.chat([Message("user", prompt)], system=CRITIC_SYSTEM)
            data = parse_json_object(response.content)
        except Exception:  # noqa: BLE001 — 质检失败不能拦住答复
            data = None
        if data is None or not isinstance(data.get("passed"), bool):
            return Critique(passed=True, degraded=True, suggestion="质检输出无法解析,降级放行")
        raw_issues = data.get("issues")
        issues = (
            [str(i).strip() for i in raw_issues if str(i).strip()]
            if isinstance(raw_issues, list)
            else []
        )
        return Critique(
            passed=bool(data["passed"]),
            issues=issues,
            suggestion=str(data.get("suggestion", "")).strip(),
        )
