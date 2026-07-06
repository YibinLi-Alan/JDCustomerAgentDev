"""Agent 循环内核:文本解析版 ReAct(阶段二)+ 原生 Function Calling 版(阶段三 P-B)。

本模块实现两个同构的自主多步循环:

- :class:`ReActAgent`(阶段二):模型输出 JSON **文本**,框架自己解析
  (``parse_step``)。不依赖厂商的工具调用能力,任何能吐文本的模型都能跑,是兜底方案。
- :class:`ToolCallingAgent`(阶段三 P-B):走厂商**原生 Function Calling** 协议
  —— 工具 Schema 通过 ``chat(tools=...)`` 下发,模型返回结构化 ``tool_calls``
  (支持一步并行多个),无需文本解析,更稳、支持 strict mode。**优先用它。**

两者共享 :class:`AgentResult` / :class:`StepTrace` 轨迹结构。详见
stage-2-design.md 与 stage-3-design.md。

设计立场:

- **只依赖 ``LLM`` 接口**(见 :mod:`agent_framework.core.llm`),不关心背后是 Claude 还是 OpenAI。
- **工具可插拔**:对工具只做「鸭子类型」依赖(``.name`` / ``.description`` / ``.run``),
  运行时不 import :mod:`agent_framework.tools`。加一个新工具 = 新写一个满足 ``Tool`` 协议的类
  并加进装配列表,本循环一行不用改。
- **错误恢复优先**:解析失败 / 未知工具 / 工具异常都当作 Observation 喂回模型自我纠正,程序不崩;
  API 级重试 / 降级 / 超时留到阶段六。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

from agent_framework.core.llm import LLM, Message

if TYPE_CHECKING:
    from agent_framework.tools.base import Tool
    from agent_framework.tools.registry import ToolRegistry


# --------------------------------------------------------------------------- #
# 每步输出契约(Pydantic 校验)                                                  #
# --------------------------------------------------------------------------- #
class AgentAction(BaseModel):
    """模型决定调用的一个工具。

    Attributes:
        tool: 工具唯一名字,对应某个 ``Tool.name``。
        input: 传给 ``Tool.run`` 的关键字参数;默认空 dict。
    """

    tool: str
    input: dict[str, object] = Field(default_factory=dict)


class AgentStep(BaseModel):
    """模型每一步输出的结构化对象:``action`` 与 ``final_answer`` **恰有其一**。

    Attributes:
        thought: 本步的推理说明(为什么这么做)。
        action: 若还要调用工具,则给出动作;否则为 ``None``。
        final_answer: 若可以收尾,则给出给用户的最终回复;否则为 ``None``。
    """

    thought: str
    action: AgentAction | None = None
    final_answer: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> AgentStep:
        """强制「action / final_answer 二者恰有其一」,否则视为非法输出。"""
        if (self.action is None) == (self.final_answer is None):
            raise ValueError("必须且只能提供 action 或 final_answer 之一")
        return self


class StepParseError(Exception):
    """解析或 Pydantic 校验模型每步输出失败时统一抛出。

    携带可读原因,供错误恢复分支当作 Observation 喂回模型。
    """


def parse_step(raw: str) -> AgentStep:
    """把模型输出的原始文本鲁棒地解析为 :class:`AgentStep`。

    按顺序做(见 stage-2-design.md §4.4):
    1. 去掉 ``` ```json ``` / ``` ``` ``` 代码围栏;
    2. 截取第一个 ``{`` 到最后一个 ``}``(容忍前后杂字);
    3. ``json.loads``;
    4. ``AgentStep.model_validate``。

    任何一步失败都抛出 :class:`StepParseError`(附带可读原因)。

    Args:
        raw: 模型返回的原始文本。

    Returns:
        校验通过的 :class:`AgentStep`。

    Raises:
        StepParseError: 去围栏 / 定位 JSON / 反序列化 / 校验任一步失败。
    """
    text = raw.strip()

    # ① 去掉代码围栏 ```json ... ``` 或 ``` ... ```
    if text.startswith("```"):
        # 去掉首行围栏(可能带语言标注,如 ```json)
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        else:
            text = text[3:]
        # 去掉结尾围栏
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    text = text.strip()

    # ② 截取第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise StepParseError(f"未找到 JSON 对象:{raw!r}")
    candidate = text[start : end + 1]

    # ③ json.loads
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise StepParseError(f"JSON 解析失败:{e}") from e

    # ④ Pydantic 校验
    try:
        return AgentStep.model_validate(data)
    except Exception as e:  # noqa: BLE001 - 统一转成 StepParseError 供上层喂回恢复
        raise StepParseError(f"结构校验失败:{e}") from e


# --------------------------------------------------------------------------- #
# 轨迹数据结构                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class StepTrace:
    """单步的完整轨迹记录(供调试 / CLI 的 ``/trace`` 展示)。

    正常步填 ``thought`` +(``action`` / ``observation`` 或 ``final_answer``);
    解析失败步填 ``error`` + ``raw``。
    """

    thought: str | None = None
    action: AgentAction | None = None
    observation: str | None = None
    final_answer: str | None = None
    error: str | None = None
    raw: str | None = None


@dataclass
class AgentResult:
    """一次 :meth:`ReActAgent.run` 的结构化结果。

    Attributes:
        final_answer: 给用户的最终答案。
        steps: 本次 run 的完整轨迹(每步一条)。
        stopped_reason: 终止原因,取值 ``"final_answer"`` 或 ``"max_steps"``。
    """

    final_answer: str
    steps: list[StepTrace]
    stopped_reason: str


# --------------------------------------------------------------------------- #
# ReAct 主循环                                                                  #
# --------------------------------------------------------------------------- #
class ReActAgent:
    """ReAct 最小内核:``Thought → Action → Observation`` 的自主多步循环。

    只依赖 :class:`~agent_framework.core.llm.LLM` 接口与鸭子类型的 ``Tool``
    (需要 ``.name`` / ``.description`` / ``.run``),换 provider 或加工具都不改循环本身。
    """

    def __init__(
        self,
        llm: LLM,
        tools: Sequence[Tool],
        *,
        max_steps: int = 5,
        system_prompt: str | None = None,
    ) -> None:
        """构造一个 ReAct Agent。

        Args:
            llm: 满足 ``LLM`` 接口的实现,每步用其 ``chat()`` 拿完整 JSON。
            tools: 一组满足 ``Tool`` 协议的工具;内部按 ``name`` 建查找表。
            max_steps: 最大步数(防死循环);撞上限触发强制作答。默认 5。
            system_prompt: 自定义 system prompt;留空则据工具清单自动生成。
        """
        self._llm = llm
        self._tools: dict[str, Tool] = {tool.name: tool for tool in tools}
        self._max_steps = max_steps
        self._system_prompt = system_prompt or self._default_system_prompt(tools)

    @property
    def max_steps(self) -> int:
        """最大步数(只读)。"""
        return self._max_steps

    @property
    def system_prompt(self) -> str:
        """当前生效的 system prompt(只读)。"""
        return self._system_prompt

    def run(
        self,
        user_input: str,
        history: list[Message] | None = None,
    ) -> AgentResult:
        """围绕一个用户问题跑完整的 ReAct 循环,直到收尾或撞上步数上限。

        内层 ``context`` 是本轮的「草稿纸」:外层干净历史 + 本轮问题,再叠加每步的
        assistant 原始输出与 Observation。撞上限时不报错,而是强制模型用现有信息作答。

        Args:
            user_input: 用户本轮的问题 / 反馈。
            history: 外层干净对话历史(仅 (问题, 最终答案) 对),可为空。

        Returns:
            带完整轨迹的 :class:`AgentResult`。
        """
        context: list[Message] = list(history or []) + [Message("user", user_input)]
        steps: list[StepTrace] = []

        for _step_no in range(1, self._max_steps + 1):
            raw = self._llm.chat(context, system=self._system_prompt).content

            try:
                step = parse_step(raw)
            except StepParseError as e:
                # 解析失败:当 Observation 喂回让模型自我纠正,计入步数
                context += [
                    Message("assistant", raw),
                    Message("user", f"[解析失败] {e};请严格只输出 JSON"),
                ]
                steps.append(StepTrace(raw=raw, error=str(e)))
                continue

            if step.final_answer is not None:
                # 收尾
                steps.append(StepTrace(thought=step.thought, final_answer=step.final_answer))
                return AgentResult(
                    final_answer=step.final_answer,
                    steps=steps,
                    stopped_reason="final_answer",
                )

            # 否则是 action:执行工具,把结果当 Observation 喂回
            assert step.action is not None  # _exactly_one 已保证
            observation = self._execute(step.action)
            context += [
                Message("assistant", raw),
                Message("user", f"[Observation] {observation}"),
            ]
            steps.append(
                StepTrace(
                    thought=step.thought,
                    action=step.action,
                    observation=observation,
                )
            )

        # 撞到步数上限:强制模型用现有信息作答
        final = self._force_final_answer(context)
        return AgentResult(
            final_answer=final,
            steps=steps,
            stopped_reason="max_steps",
        )

    def _execute(self, action: AgentAction) -> str:
        """执行一个工具动作,把结果 / 错误统一转成可喂回的 Observation 文本。

        未知工具与工具异常都不抛出,而是返回一段文字让模型下一步自我纠正。

        Args:
            action: 模型给出的工具动作。

        Returns:
            工具的返回文本,或描述「未知工具 / 执行失败」的可读文本。
        """
        tool = self._tools.get(action.tool)
        if tool is None:
            return f"工具 {action.tool} 不存在。可用工具:{list(self._tools)}"
        try:
            return tool.run(**action.input)
        except Exception as e:  # noqa: BLE001 - 阶段二任何工具异常都喂回让模型自我纠正,不崩
            return f"工具执行失败:{e}"

    def _force_final_answer(self, context: list[Message]) -> str:
        """撞到步数上限时,强制模型用现有观察结果直接作答。

        Args:
            context: 已积累的内层上下文(草稿纸)。

        Returns:
            模型据现有信息给出的最终答案文本。
        """
        context = context + [
            Message(
                "user",
                "已达到最大步数,请根据已有的观察结果,直接给出你能给的最好的最终答案,"
                "不要再调用工具。",
            )
        ]
        return self._llm.chat(context, system=self._system_prompt).content

    def _default_system_prompt(self, tools: Sequence[Tool]) -> str:
        """据工具清单生成默认 system prompt(见 stage-2-design.md §8)。

        包含:角色说明 + 逐个列出的工具清单 + 输出 JSON 格式二选一约定 + few-shot 示例。

        Args:
            tools: 装配的工具序列。

        Returns:
            完整的 system prompt 文本。
        """
        if tools:
            tool_lines = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)
        else:
            tool_lines = "(当前没有可用工具,只能直接作答)"

        return (
            "你是京东客服 Agent。你可以通过“思考-行动-观察”的循环解决用户问题:\n"
            "先思考(thought)决定要做什么,若需要就调用工具(action)拿到观察结果,再继续思考,"
            "直到可以给出最终答案(final_answer)。\n"
            "\n"
            "可用工具:\n"
            f"{tool_lines}\n"
            "\n"
            "每一步只输出一个 JSON 对象,格式二选一:\n"
            '1) 需要调用工具:{"thought":"...", "action":{"tool":"工具名","input":{参数}}}\n'
            '2) 可以回答了:{"thought":"...", "final_answer":"给用户的回复"}\n'
            "\n"
            "规则:\n"
            "- 只输出 JSON 本身,不要多余文字 / 解释 / 代码围栏;"
            "action 与 final_answer 只能给其中一个。\n"
            "- 优先参考上文对话历史来理解用户的指代:如“那个订单 / 它 / 刚才那个 / 那”通常指"
            "之前对话里出现过的订单号、物流单号等。\n"
            "- 对话历史里已经出现过的信息(订单号、物流单号等)不要再向用户索要,直接拿来用;"
            "只有历史中确实找不到时,才礼貌地请用户提供。\n"
            "\n"
            "示例一(先查订单 → 再查物流 → 作答):\n"
            "用户:我的订单 12345 到哪了?\n"
            '第1步:{"thought":"用户问订单进度,我需要先查订单拿到物流单号",'
            '"action":{"tool":"query_order","input":{"order_id":"12345"}}}\n'
            "观察:已发货,物流单号 SF123\n"
            '第2步:{"thought":"拿到单号 SF123,再查物流进度",'
            '"action":{"tool":"query_logistics","input":{"tracking_no":"SF123"}}}\n'
            "观察:运输中,预计明天送达\n"
            '第3步:{"thought":"已拿到物流进度,可以回答了",'
            '"final_answer":"您的订单已发货,顺丰运输中,预计明天送达。"}\n'
            "\n"
            "示例二(利用对话历史解析指代,不重复索要已知信息):\n"
            "(对话历史里,用户此前问过“我的订单 12345 到哪了”,已答复该订单已发货、预计明天送达)\n"
            "用户:那大概几点能到?\n"
            '第1步:{"thought":"“那”指历史里提到的订单 12345,已知该订单,无需向用户再要订单号,'
            '我直接重新查一下它的物流拿最新进度","action":{"tool":"query_order",'
            '"input":{"order_id":"12345"}}}\n'
            "观察:已发货,物流单号 SF123\n"
            '第2步:{"thought":"用单号查物流最新进度",'
            '"action":{"tool":"query_logistics","input":{"tracking_no":"SF123"}}}\n'
            "观察:运输中,预计明天送达\n"
            '第3步:{"thought":"根据物流进度回答用户","final_answer":"您的订单预计明天送达哦。"}'
        )


# --------------------------------------------------------------------------- #
# 原生 Function Calling 循环(阶段三 P-B)                                        #
# --------------------------------------------------------------------------- #

#: ToolCallingAgent 的默认 system prompt。工具清单走原生 ``tools=`` 参数下发,
#: 这里只写角色与行为规则,不再需要 JSON 输出格式约定与 few-shot(协议层保证了结构)。
DEFAULT_TOOL_CALLING_SYSTEM_PROMPT = (
    "你是京东客服 Agent。需要订单、物流、商品等数据时,调用合适的工具获取;"
    "一次可以并行调用多个互不依赖的工具;信息足够后直接给出简洁、友好的中文答复。\n"
    "规则:\n"
    "- 优先参考上文对话历史理解用户的指代(如“那个订单 / 它 / 刚才那个”通常指"
    "之前对话里出现过的订单号、物流单号等)。\n"
    "- 对话历史里已出现过的信息(订单号、物流单号等)不要再向用户索要,直接拿来用;"
    "只有历史中确实找不到时,才礼貌地请用户提供。\n"
    "- 工具返回“未找到 / 执行失败”时,先核对参数再重试或换工具;确实拿不到就如实告知用户。"
)


class ToolCallingAgent:
    """原生 Function Calling 版的自主循环:模型返回结构化 ``tool_calls``。

    与 :class:`ReActAgent` 的分工(见 stage-3-design.md):本类走厂商原生协议,
    无文本解析环节(没有 ``StepParseError`` 分支),支持一步并行多个工具调用与
    strict mode;``ReActAgent`` 保留作为「任何文本模型都能跑」的兜底。

    只依赖 ``LLM`` 接口与鸭子类型的 Registry(需要 ``to_schemas()`` /
    ``invoke(name, args)`` / ``render_catalog()``),运行时不 import tools 子包。
    """

    def __init__(
        self,
        llm: LLM,
        registry: ToolRegistry,
        *,
        max_steps: int = 5,
        system_prompt: str | None = None,
    ) -> None:
        """构造一个原生 Function Calling Agent。

        Args:
            llm: 满足 ``LLM`` 接口的实现;其 ``chat(tools=...)`` 需支持工具下发。
            registry: 工具注册中心;每步把 ``to_schemas()`` 下发给模型,
                执行统一走 ``invoke()``(未知工具 / 失败都折叠,循环不崩)。
            max_steps: 最大步数(一步 = 一次模型调用,可含多个并行工具调用);
                撞上限触发强制作答。默认 5。
            system_prompt: 自定义 system prompt;留空用默认京东客服提示词
                (工具清单由协议层下发,无需写进提示词)。
        """
        self._llm = llm
        self._registry = registry
        self._max_steps = max_steps
        self._system_prompt = system_prompt or DEFAULT_TOOL_CALLING_SYSTEM_PROMPT

    @property
    def max_steps(self) -> int:
        """最大步数(只读)。"""
        return self._max_steps

    @property
    def system_prompt(self) -> str:
        """当前生效的 system prompt(只读)。"""
        return self._system_prompt

    def run(
        self,
        user_input: str,
        history: list[Message] | None = None,
    ) -> AgentResult:
        """围绕一个用户问题跑完整循环,直到模型直接作答或撞上步数上限。

        每步:带工具 Schema 调用模型 → 若返回 ``tool_calls`` 则逐个执行
        (支持并行调用:一步多个 call),结果以 ``tool`` 消息回传;若无
        ``tool_calls`` 则 ``content`` 即最终答案。

        Args:
            user_input: 用户本轮的问题 / 反馈。
            history: 外层干净对话历史(仅 (问题, 最终答案) 对),可为空。

        Returns:
            带完整轨迹的 :class:`AgentResult`。
        """
        context: list[Message] = list(history or []) + [Message("user", user_input)]
        steps: list[StepTrace] = []
        schemas = self._registry.to_schemas()

        for _step_no in range(1, self._max_steps + 1):
            resp = self._llm.chat(context, system=self._system_prompt, tools=schemas)

            if not resp.tool_calls:
                # 模型直接作答 → 收尾
                steps.append(StepTrace(final_answer=resp.content))
                return AgentResult(
                    final_answer=resp.content,
                    steps=steps,
                    stopped_reason="final_answer",
                )

            # 模型要求调工具(可能一步多个):先原样记下 assistant 消息,再逐个执行回传
            context.append(Message("assistant", resp.content, tool_calls=tuple(resp.tool_calls)))
            for call in resp.tool_calls:
                observation = self._registry.invoke(call.name, call.args).to_observation()
                context.append(Message("tool", observation, tool_call_id=call.id))
                steps.append(
                    StepTrace(
                        thought=resp.content or None,
                        action=AgentAction(tool=call.name, input=call.args),
                        observation=observation,
                    )
                )

        # 撞到步数上限:强制模型用现有信息作答(不再下发工具,杜绝继续调用)
        context.append(
            Message(
                "user",
                "已达到最大步数,请根据已有的工具结果,直接给出你能给的最好的最终答案,"
                "不要再调用工具。",
            )
        )
        final = self._llm.chat(context, system=self._system_prompt).content
        return AgentResult(final_answer=final, steps=steps, stopped_reason="max_steps")
