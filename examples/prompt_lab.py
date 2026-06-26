"""交付物② Prompt Engineering 对比实验脚本。

用同一个 ``LLM`` 接口、同一个模型,跑同一批任务,对比三种提问策略对输出质量的影响:
**直接提问 / Chain-of-Thought / Few-shot**(见 stage-1-design.md §7)。

实验只比 prompt 策略,不引入温度变量(已锁定决策,见设计 §11)。

运行:``python -m examples.prompt_lab``(需在 ``.env`` 配好所选 provider 的真实 key)。
运行后会把结果汇总成 Markdown 写进 ``docs/stage-1-prompt-experiment.md``。

注意:本脚本只 import ``agent_framework`` 的接口与工厂,不直接 import 任何厂商 SDK。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from agent_framework import LLM, ChatResponse, Message, create_llm, get_settings

# 报告写入位置:与设计文档同级,放在仓库 docs/ 下。
_REPORT_PATH = Path(__file__).resolve().parent.parent / "docs" / "stage-1-prompt-experiment.md"

# CoT 引导语:附加到 user 题面后,引导模型显式分步推理。
_COT_SUFFIX = "\n\n请一步一步思考,展示推理过程,最后再给出最终答案。"


@dataclass(frozen=True)
class Task:
    """一道实验题。

    Attributes:
        name: 任务名(用于报告表头)。
        category: 任务类型(多步推理 / 结构化抽取 / 常识陷阱)。
        prompt: 题面(直接提问时原样发送)。
        few_shot_examples: Few-shot 范例,(输入, 输出)对,前置到题面之前。
    """

    name: str
    category: str
    prompt: str
    few_shot_examples: tuple[tuple[str, str], ...]


# ---- 任务清单:选「直接答易错、过程化能纠偏」的题型,让策略差异看得见 ----
TASKS: tuple[Task, ...] = (
    Task(
        name="多步算术",
        category="多步推理",
        prompt="一个数加 17 后再乘以 3,结果等于 60。求这个数。",
        few_shot_examples=(
            (
                "一个数加 5 后乘以 2 等于 18,求这个数。",
                "设这个数为 x:(x+5)*2=18 → x+5=9 → x=4。答案:4。",
            ),
            (
                "一个数乘以 4 再减 6 等于 14,求这个数。",
                "设这个数为 x:4x-6=14 → 4x=20 → x=5。答案:5。",
            ),
        ),
    ),
    Task(
        name="情绪与问题分类",
        category="结构化抽取",
        prompt=(
            "下面是一条京东用户反馈,请仅输出 JSON,字段为 情绪(positive/negative/neutral)"
            "与 问题类型(物流/质量/客服/价格/其他):\n"
            "「快递放门口也不打电话,东西到了我都不知道,差评!」"
        ),
        few_shot_examples=(
            (
                "「手机用了一周就死机,太失望了。」",
                '{"情绪": "negative", "问题类型": "质量"}',
            ),
            (
                "「客服小姐姐很耐心,问题很快解决了,点赞。」",
                '{"情绪": "positive", "问题类型": "客服"}',
            ),
        ),
    ),
    Task(
        name="单位换算陷阱",
        category="常识陷阱",
        prompt="1 公斤棉花和 1 公斤铁,哪个更重?请直接回答。",
        few_shot_examples=(
            ("1 千克羽毛和 1 千克石头哪个更重?", "一样重,都是 1 千克。"),
            ("100 克金子和 100 克纸哪个更重?", "一样重,都是 100 克。"),
        ),
    ),
)


@dataclass(frozen=True)
class Trial:
    """一次「任务 × 策略」的运行结果。"""

    task: Task
    strategy: str
    response: ChatResponse


def _build_messages(task: Task, strategy: str) -> tuple[list[Message], str | None]:
    """按策略构造 (messages, system)。

    三种策略共用 ``ClaudeLLM.chat()``,仅 prompt 不同,正好验证接口的复用性。
    """
    if strategy == "direct":
        return [Message("user", task.prompt)], None

    if strategy == "cot":
        return [Message("user", task.prompt + _COT_SUFFIX)], None

    if strategy == "few-shot":
        messages: list[Message] = []
        for example_input, example_output in task.few_shot_examples:
            messages.append(Message("user", example_input))
            messages.append(Message("assistant", example_output))
        messages.append(Message("user", task.prompt))
        return messages, None

    raise ValueError(f"未知策略:{strategy}")


STRATEGIES: tuple[tuple[str, str], ...] = (
    ("direct", "直接提问"),
    ("cot", "Chain-of-Thought"),
    ("few-shot", "Few-shot"),
)


def run_experiment(llm: LLM) -> list[Trial]:
    """遍历 任务 × 策略,逐个调用 ``chat()`` 并收集结果。"""
    trials: list[Trial] = []
    for task in TASKS:
        for strategy_key, strategy_label in STRATEGIES:
            messages, system = _build_messages(task, strategy_key)
            print(f"[运行] {task.name} / {strategy_label} ...", flush=True)
            response = llm.chat(messages, system=system)
            trials.append(Trial(task=task, strategy=strategy_label, response=response))
    return trials


def _md_escape(text: str) -> str:
    """把回复压成单行,转义竖线,便于放进 Markdown 表格。"""
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>").strip()


def render_report(trials: list[Trial], model: str) -> str:
    """把实验结果渲染成 Markdown 报告。"""
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    lines: list[str] = [
        "# 阶段一实验报告 · Prompt 三策略对比",
        "",
        f"- 运行时间:{now}",
        f"- 模型:`{model}`(实验只比 prompt 策略,不引入温度变量)",
        f"- 策略:{'、'.join(label for _, label in STRATEGIES)}",
        f"- 任务数:{len(TASKS)}",
        "",
        "> 本文件由 `examples/prompt_lab.py` 自动生成。",
        "",
        "## 逐题输出对比",
        "",
    ]
    for task in TASKS:
        lines.append(f"### {task.name}({task.category})")
        lines.append("")
        lines.append(f"题面:{task.prompt.splitlines()[0]} ……")
        lines.append("")
        lines.append("| 策略 | 模型回复 | in | out | total |")
        lines.append("| --- | --- | --- | --- | --- |")
        for trial in trials:
            if trial.task is not task:
                continue
            usage = trial.response.usage
            lines.append(
                f"| {trial.strategy} | {_md_escape(trial.response.content)} "
                f"| {usage.input_tokens} | {usage.output_tokens} | {usage.total_tokens} |"
            )
        lines.append("")

    lines.append("## 结论(运行后人工填写)")
    lines.append("")
    lines.append("- 哪种策略在「多步推理」类任务更稳?")
    lines.append("- 哪种策略最能稳定「结构化抽取」的输出格式?")
    lines.append("- 各策略的 token 成本权衡如何?")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """运行实验并写出报告。需在 .env 配好所选 provider 的真实 key。"""
    settings = get_settings()
    llm: LLM = create_llm(settings)

    trials = run_experiment(llm)
    report = render_report(trials, model=llm.model)

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\n[完成] 报告已写入:{_REPORT_PATH}")


if __name__ == "__main__":
    main()
