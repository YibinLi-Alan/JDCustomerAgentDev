"""Agent 端到端评测 pipeline —— 跑评测集 → LLM-as-Judge 打分 → 聚合报告(阶段六 P-C)。

一条命令的回归测试能力(= A/B 对比雏形):每次改动后跑一遍,数字说话。
用真实 LLM(评测本身要花钱,但用便宜模型 + 25 条集合,成本可控)。

用法:
    python -m agent_framework.evaluation.agent_eval               # 跑全部,打印报告
    python -m agent_framework.evaluation.agent_eval --category multi_agent
    python -m agent_framework.evaluation.agent_eval --limit 5     # 只跑前 5 条(冒烟)

裁判局限(报告如实声明,见 judge.py):裁判与被评同款模型有自偏;25 条小样本
有波动——数字给区间感觉,不当真理。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from agent_framework.core.config import get_settings
from agent_framework.core.llm import create_llm
from agent_framework.evaluation.judge import Judge, Judgement
from agent_framework.service import AgentService, ServiceResult
from agent_framework.tools.jd_mock_data import JDMockStore
from agent_framework.tools.presets import default_registry

_DATASET = Path(__file__).parent / "datasets" / "agent_cases.json"


@dataclass
class CaseResult:
    """单条用例的评测结果。"""

    case_id: str
    category: str
    answer: str
    route: str
    handoff: bool
    judgement: Judgement


def run_case(service: AgentService, case: dict) -> ServiceResult:
    """跑一条用例(支持多轮 turns:前几轮建立上下文,评最后一轮)。"""
    user_id = case["user_id"]
    turns = case.get("turns") or [case["query"]]
    result: ServiceResult | None = None
    for turn in turns:
        result = service.handle(user_id, turn)
    assert result is not None
    return result


def evaluate(cases: list[dict], *, limit: int | None = None) -> list[CaseResult]:
    """跑评测集并逐条评判。每条用例用独立 store(隔离,避免互相污染状态)。"""
    settings = get_settings()
    llm = create_llm(settings)
    judge = Judge(llm)
    results: list[CaseResult] = []

    for case in cases[:limit] if limit else cases:
        # 每条用例独立 service + store:mock 状态不跨用例(如 c01 取消订单不影响 s02)
        service = AgentService(llm, default_registry(JDMockStore()), settings, enable_trace=False)
        outcome = run_case(service, case)
        query = case.get("turns", [case.get("query", "")])[-1]
        judgement = judge.score(
            query=query,
            expected_points=case["expected"],
            answer=outcome.answer,
            trace_summary=(
                f"路由={outcome.route}; 人工介入={'是' if outcome.handoffs else '否'}; "
                f"审批预期={'是' if case.get('approval_expected') else '否'}"
            ),
            max_steps_hint=case.get("max_steps"),
        )
        results.append(
            CaseResult(
                case_id=case["id"],
                category=case["category"],
                answer=outcome.answer,
                route=outcome.route,
                handoff=bool(outcome.handoffs),
                judgement=judgement,
            )
        )
        print(
            f"  [{case['id']:<4}] {case['category']:<12} 均分 {judgement.average:.1f} "
            f"{'✓' if judgement.passed else '✗'}  route={outcome.route}"
        )
    return results


def render_report(results: list[CaseResult]) -> str:
    """把逐条结果聚合成 markdown 报告(evaluation/reports 存档 + 答辩用)。"""
    n = len(results)
    if n == 0:
        return "# 评测报告\n\n(无用例)"
    passed = sum(1 for r in results if r.judgement.passed)
    dims = ("accuracy", "completeness", "efficiency", "safety")
    avg = {d: sum(getattr(r.judgement, d) for r in results) / n for d in dims}
    degraded = sum(1 for r in results if r.judgement.degraded)

    lines = [
        "# 阶段六 Agent 端到端评测报告",
        "",
        f"- 用例数:**{n}**  ·  通过:**{passed}**  ·  通过率:**{passed / n:.0%}**",
        f"- 四维均分(1-5):准确 {avg['accuracy']:.2f} · 完整 {avg['completeness']:.2f} · "
        f"效率 {avg['efficiency']:.2f} · 安全 {avg['safety']:.2f}",
        f"- 裁判降级(解析失败判中性分)条数:{degraded}",
        "",
        "## 分类通过率",
        "",
        "| 类别 | 用例 | 通过 | 均分 |",
        "|---|---|---|---|",
    ]
    cats: dict[str, list[CaseResult]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r)
    for cat, rs in cats.items():
        p = sum(1 for r in rs if r.judgement.passed)
        a = sum(r.judgement.average for r in rs) / len(rs)
        lines.append(f"| {cat} | {len(rs)} | {p} | {a:.2f} |")

    lines += [
        "",
        "## 逐条明细",
        "",
        "| 用例 | 类别 | 均分 | 通过 | 路由 | 裁判依据 |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        reason = r.judgement.reason.replace("|", "/")[:50]
        lines.append(
            f"| {r.case_id} | {r.category} | {r.judgement.average:.1f} | "
            f"{'✓' if r.judgement.passed else '✗'} | {r.route} | {reason} |"
        )

    lines += [
        "",
        "## 局限性(如实声明)",
        "",
        "- **裁判自偏**:裁判与被评为同款模型(gpt-5.4-mini),可能偏爱同源文风;"
        "更严谨应换更强模型当裁判或双裁判(见 stage-6-design.md §9.2)。",
        "- **小样本波动**:25 条评测集不足以给出稳定百分比,数字用于**改动前后对比**"
        "(回归测试)而非绝对结论——延续阶段四 16 条评测集的同一教训。",
        "- **单跑随机性**:未固定温度多次取均值;生产评估应多跑取中位数。",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent 端到端评测")
    parser.add_argument("--category", help="只跑某一类别")
    parser.add_argument("--limit", type=int, help="只跑前 N 条")
    args = parser.parse_args()

    data = json.loads(_DATASET.read_text(encoding="utf-8"))
    cases = data["cases"]
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]

    print(f"评测集:{len(cases)} 条用例(真实 LLM,请耐心)…\n")
    results = evaluate(cases, limit=args.limit)
    report = render_report(results)
    print("\n" + report)

    out = Path(__file__).parent / "reports" / "agent_eval_latest.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\n报告已写入 {out}")


if __name__ == "__main__":
    main()
