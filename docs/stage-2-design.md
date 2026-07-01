# 阶段二设计文档 · 最小 Agent 循环(ReAct)

> 对应大纲「阶段二:最小 Agent 循环」。先设计 → 评审 → 再编码。
> 本阶段把阶段一的「一问一答 Chatbot」升级为「自主多步循环 Agent」。

---

## 1. 目标与范围

**目标**:实现 ReAct(Reasoning + Acting)最小内核——一个 `Thought → Action → Observation`
的自主循环,让 Agent 能根据用户问题**自己决定**是直接回答,还是先调用工具、拿到结果再继续想。

**范围内(阶段二做)**
- ReAct 主循环控制流(`core/agent.py`)
- 最大步数限制(防死循环)+ 撞上限的兜底
- 结构化输出:模型每步输出 JSON,用 Pydantic 校验解析
- 错误处理与自我纠正(解析失败 / 未知工具 / 工具异常)
- 一个**极简 `Tool` 接口** + 2 个**京东风格 mock 工具**(只为跑通循环)
- 带完整轨迹的返回值 `AgentResult`
- 演示脚本 + 关键路径单测(用 MockLLM,离线、免费)

**范围外(明确留给后续阶段)**
- 完整 `BaseTool` / `ToolRegistry` / JSON-Schema 参数 / ≥5 真工具 / 原生 Function Calling → **阶段三**
- 真实数据源接入:京东订单数据库、RAG 检索、**MCP** 可插拔工具源 → **阶段三**
- 上下文压缩 / 长期记忆 → 阶段四
- 规划 / 多 Agent → 阶段五
- 重试 / 降级 / 超时 / 可观测 → 阶段六

> **关于 MCP**:MCP 不是「另一种工具」,而是「工具的可插拔来源」。阶段三它会作为
> `Tool` 接口背后的**一个实现(MCP 适配器)**接入,核心循环不动。阶段二不碰,但接口会预留口子(见 §5.3)。

---

## 2. 交付物清单(逐条对照大纲「阶段产出」)

| # | 大纲产出 | 本阶段交付物 | 落点 |
|---|---|---|---|
| ① | Agent Loop 骨架代码(`agent_loop.py`) | `ReActAgent`:ReAct 主循环 + 步数守卫 + 解析 + 错误恢复 | `agent_framework/core/agent.py` |
| ② | Agent 能自主思考,决定进一步行动或直接回答 | ReAct 循环 + 极简 Tool 接口 + 2 个 JD mock 工具,能跑通多步 | `core/agent.py` + `tools/` |
| ③ | 设计文档:Agent Loop 的状态机图 + 流程说明 | 本文档(§4 状态机图 + 流程) | `docs/stage-2-design.md` |

> 三项都依赖同一个 `core/agent.py`,这是本阶段的代码重心;它只依赖阶段一的 `LLM` 接口。

---

## 3. Agent vs Chatbot(为什么需要这个循环)

| | 阶段一 Chatbot | 阶段二 Agent |
|---|---|---|
| 交互 | 一问一答,一轮即止 | 感知 → 思考 → 行动的**自主循环**,可多步 |
| 能力 | 只能生成文字 | 能**调用工具**去做事 / 取数据,再基于结果继续 |
| 驱动 | 被动应答 | **目标驱动**:围绕"解决用户问题"自主决定下一步 |

ReAct 的精髓:让模型**交替**做两件事——**推理(Thought)**决定要干嘛,**行动(Action)**去调工具,
把工具结果作为**观察(Observation)**喂回,再推理……直到它认为可以给出**最终答案**。

---

## 4. 核心设计:ReAct 主循环

### 4.1 状态机 / 流程图(大纲要求)

```
            ┌──────────────────────────── 每轮 step += 1 ────────────────────────────┐
            │                                                                         │
[START] 组装上下文(system: ReAct指令+工具清单 ; user: 问题)                          │
            │                                                                         │
            ▼                                                                         │
        ┌───────┐   step > max_steps ?  ── 是 ──►  [FORCE] 追加"请用现有信息直接作答,勿再调工具"
        │ THINK │                                        │  再 chat 一次,取文本
        │ 调 LLM │◄───────────────────────────────────────┘         │
        └───┬───┘                                                    ▼
            │ 拿到原始文本                                        [DONE] 返回 AgentResult
            ▼                                                    (stopped_reason=max_steps)
     解析 JSON + Pydantic 校验
            │
   ┌────────┴─────────┐
   │解析失败           │解析成功
   ▼                  ▼
[RECOVER]         是 final_answer ? ── 是 ──► [DONE] 返回 AgentResult(stopped_reason=final_answer)
把"解析错误"           │ 否(是 action)
当 Observation        ▼
喂回上下文         工具存在 ? ── 否 ──► [RECOVER] 把"工具不存在,可用:[...]"当 Observation 喂回
   │                  │ 是                                   │
   │                  ▼                                      │
   │              [ACT] 执行 tool.run(**input)               │
   │                  │                                      │
   │          执行抛异常 ? ── 是 ──► [RECOVER] 把"工具执行失败:<msg>"喂回
   │                  │ 否                                   │
   │                  ▼                                      │
   │           [OBSERVE] 把工具结果当 Observation 追加进上下文  │
   │                  │                                      │
   └──────────────────┴──────────────────────────────────────┘
                      │
                      ▼  回到 THINK(进入下一步)
```

**要点**:所有分支(正常观察、解析失败、未知工具、工具异常)最终都**回到 THINK**,
把信息喂回让模型继续;而每一次 THINK 都**计入 max_steps**,所以即使模型持续输出坏内容也不会无限循环。

### 4.2 主循环控制流(伪代码)

```python
def run(self, user_input: str, history: list[Message] | None = None) -> AgentResult:
    # 外层对话历史(干净的问答对)+ 本轮问题 → 组成本轮的内层上下文(草稿纸)
    context = list(history or []) + [Message("user", user_input)]   # system prompt 单独传
    steps: list[StepTrace] = []

    for step_no in range(1, self.max_steps + 1):
        raw = self._llm.chat(context, system=self._system_prompt).content   # 非流式,要完整 JSON
        try:
            step = parse_step(raw)               # → AgentStep(Pydantic 校验)
        except StepParseError as e:
            context += [Message("assistant", raw), Message("user", f"[解析失败] {e};请严格输出 JSON")]
            steps.append(StepTrace(raw=raw, error=str(e)))
            continue                              # 计入步数,回到 THINK

        if step.final_answer is not None:         # 收尾
            steps.append(StepTrace(thought=step.thought, final_answer=step.final_answer))
            return AgentResult(step.final_answer, steps, stopped_reason="final_answer")

        # 否则是 action:执行工具
        observation = self._execute(step.action)  # 内部处理"未知工具/工具异常",都返回可喂回的文本
        context += [Message("assistant", raw), Message("user", f"[Observation] {observation}")]
        steps.append(StepTrace(thought=step.thought, action=step.action, observation=observation))

    # 撞到步数上限:强制模型用现有信息作答
    final = self._force_final_answer(context)
    return AgentResult(final, steps, stopped_reason="max_steps")
```

### 4.3 每步的输出契约(JSON schema)

模型**每一步只输出一个 JSON 对象**,二选一:

```jsonc
// (A) 还要调工具:
{ "thought": "用户问订单进度,我需要先查订单", "action": { "tool": "query_order", "input": { "order_id": "12345" } } }

// (B) 可以收尾了:
{ "thought": "已拿到物流信息,可以回答了", "final_answer": "您的订单已发货,顺丰运输中,预计明天送达。" }
```

用 Pydantic 表达并**强制"action / final_answer 二者恰有其一"**:

```python
class AgentAction(BaseModel):
    tool: str
    input: dict[str, object] = Field(default_factory=dict)

class AgentStep(BaseModel):
    thought: str
    action: AgentAction | None = None
    final_answer: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "AgentStep":
        if (self.action is None) == (self.final_answer is None):
            raise ValueError("必须且只能提供 action 或 final_answer 之一")
        return self
```

### 4.4 结构化解析与鲁棒性(`parse_step`)

模型偶尔会给 JSON 裹上 ` ```json ` 代码围栏、或前后带解释文字。解析按顺序做:
1. 去掉 ` ``` ` / ` ```json ` 代码围栏;
2. 截取第一个 `{` 到最后一个 `}` 之间的内容(容忍前后杂字);
3. `json.loads` → 失败抛 `StepParseError`;
4. `AgentStep.model_validate` → 校验失败(缺字段 / 两者都给)也抛 `StepParseError`。

任何一步失败都抛统一的 `StepParseError`,交给 §4.6 的错误恢复。

### 4.5 最大步数与兜底(`_force_final_answer`)

- `max_steps` 默认 **5**,来自配置(`Settings.agent_max_steps`,可在 `.env` 改)。
- 每次 THINK(含解析失败后的重试)都占一步 → 天然防住"无限坏输出"。
- 撞上限时**不冷冰冰报错**:追加一条指令
  `"已达到最大步数,请根据已有的观察结果,直接给出你能给的最好的最终答案,不要再调用工具。"`,
  再 `chat` 一次,把返回文本当 `final_answer`,`stopped_reason="max_steps"`。

### 4.6 错误处理与异常恢复

核心原则:**能让 Agent 自己纠正的错误,就当成 Observation 喂回,不要让程序崩**。

| 错误类型 | 处理方式 | 是否计入步数 |
|---|---|---|
| 解析失败(非法 JSON / schema 不符) | 喂回「解析失败原因 + 请严格按 JSON 格式重出」 | 是 |
| 未知工具(调了不存在的 tool) | 喂回「工具 X 不存在,可用工具:[...]」 | 是(在下一次 THINK) |
| 工具执行抛异常 | `try/except` 捕获,喂回「工具执行失败:\<msg>」 | 是(在下一次 THINK) |
| LLM API 错误(网络 / 限流) | **阶段二不处理**,向上抛出 | — |

> API 级的重试 / 降级 / 超时 / 幂等属于**可靠性**,统一留到**阶段六**做,避免阶段二职责膨胀。

---

## 5. 极简 Tool 接口(阶段二版,为阶段三留口子)

### 5.1 接口(`tools/base.py`)

阶段二只需要能让模型「知道有哪些工具、怎么调、调完拿到文字结果」,所以先定一个**最小**协议:

```python
class Tool(Protocol):
    name: str            # 唯一名字,模型用它指定要调哪个
    description: str      # 给模型看:这工具干嘛用、何时用
    def run(self, **kwargs) -> str: ...   # 真正执行;返回喂回给模型的文字
```

> 阶段三会把它扩成完整的 `BaseTool`(ABC + JSON-Schema 参数校验 + 错误规范),并新增 `ToolRegistry`。
> 阶段二的 `Tool` 是它的**最小前身**,故意不做参数 schema 校验——参数不对就让 `run` 抛错,走 §4.6 恢复。

### 5.2 京东风格 mock 工具(`tools/jd_mock.py`)

2 个假工具,内置几条**编好的假数据**,足以演示一条真实的多步 ReAct 链路:

- `query_order(order_id)` → 返回订单状态 + 物流单号(如订单 `12345` → 已发货 / 顺丰 `SF123`)。
- `query_logistics(tracking_no)` → 返回物流进度(如 `SF123` → 运输中,预计明天送达)。

**演示链路**:用户问「我的订单 12345 到哪了?」→ 模型 `query_order(12345)` 拿到单号 →
再 `query_logistics(SF123)` 拿到进度 → 给出最终答案。**两步工具链**,把 ReAct 的价值直观展示出来。

> ⚠️ 预期对齐:mock 只对**内置的固定几条**数据作答(其它订单号返回「订单不存在」)。
> 「问任意京东问题都能答」需要接**真实数据源**,那是**阶段三**接入真工具 / MCP 之后的事。

### 5.3 装配与可插拔(呼应「随时接入新工具」)

`ReActAgent(llm, tools=[...], ...)` 接收一组 `Tool`,内部按 `name` 建一张查找表。
**加一个新能力 = 新写一个满足 `Tool` 协议的类、在装配点加进列表**,`agent.py` 的循环**一行不用改**。
阶段三的 `ToolRegistry`、以及 **MCP 适配器工具**,都是从这个口子插进来的「更多 Tool」。

---

## 6. 与阶段一的衔接 / 复用

### 6.1 复用阶段一
- **只依赖 `LLM` 接口**:`ReActAgent` 通过 `create_llm` 拿到的 `LLM` 工作,不关心是 Claude 还是 OpenAI。
- **复用** `Message` / `ChatResponse`:上下文仍是 `list[Message]`。
- **每步用 `chat()` 非流式**:因为要拿到**完整 JSON** 才能解析(`stream()` 逐字吐,不适合)。
- **system prompt** 承载:ReAct 角色说明 + 工具清单(名字+描述)+ 输出格式约定 + 1~2 个 few-shot 例子。

### 6.2 多轮对话下的上下文:内外两层(交互式 CLI 用)

「上下文」和「循环」在 ReAct 里各有**两层**,分清才不乱:

```
CLI 会话(进程存活期间)
├─ 第1轮对话:用户问 Q1
│    └─ run(Q1):内层 context(草稿纸)
│         ├─ step1: THINK→ACT→OBSERVE      ← 一次"循环迭代 / 一步"
│         ├─ step2: THINK→ACT→OBSERVE
│         └─ step3: THINK→final_answer → 返回 A1
│    ★ run() 返回后,内层草稿(所有 thought/observation)丢弃,只把 (Q1, A1) 存进外层
├─ 第2轮对话:用户问 Q2(能看到 Q1/A1)
│    └─ run(Q2): 又开一张新草稿纸 …
└─ /exit / 关窗 → 进程结束 → 外层历史(内存 list)消失,磁盘不留痕
```

| 层 | 存什么 | 谁持有 | 生命周期 |
|---|---|---|---|
| **内层 `context`** | 单个问题的 Thought/Action/Observation 草稿 | `run()` 局部变量 | 每次 run 结束即丢弃 |
| **外层 `conversation`** | **干净的 (用户问题, 最终答案) 对** | CLI 持有 | 进程存活期间;`/exit` 或 `/reset` 清空 |

**策略**:外层**只留干净问答对,不留中间推理**——中间的"查了哪个单号/想了什么"是这一问的临时工作记忆,
答完即弃;下一轮需要会重新查(mock 很便宜)。这样上下文不膨胀。
**跨轮记住更多细节 / 关程序再开还记得(持久化)= 阶段四 Memory 的职责,阶段二不做**。

### 6.3 术语:什么算"一个循环"

| 说法 | 指什么 | 边界 |
|---|---|---|
| **一步(一次循环迭代)** | 一次 `THINK →(解析)→ ACT → OBSERVE` | `for step in range(max_steps)` 的一圈 |
| **一次 run** | 从一个问题到最终答案(内部可能好几步) | 一次 `run()` 调用 |
| **一轮对话** | 用户问一次、Agent 答一次(= 一次 run) | 交互式 CLI 里可有多轮 |

---

## 7. 文件规划

```
agent_framework/
├── core/
│   └── agent.py          # ★ ReActAgent + AgentResult + AgentStep/parse_step (本阶段重心)
├── tools/
│   ├── base.py           # 极简 Tool 协议(阶段三扩成 BaseTool)
│   └── jd_mock.py        # query_order / query_logistics 两个 mock 工具
├── core/config.py        # 追加 agent_max_steps 配置项
examples/
└── react_cli.py          # ★ 交互式多轮 CLI:外层维护干净对话历史,每轮调 agent.run();
                          #    命令 /exit /reset /trace(切换是否显示中间 Thought/Action/Observation)
tests/
├── conftest.py / mock_llm.py   # MockLLM:按脚本返回预设 JSON,离线、免费
└── test_agent.py         # 关键路径单测(见 §9)
```

---

## 8. system prompt 设计(草案)

```
你是京东客服 Agent。你可以通过"思考-行动-观察"的循环解决用户问题。

可用工具:
- query_order(order_id): 查订单状态与物流单号
- query_logistics(tracking_no): 查物流进度

每一步只输出一个 JSON 对象,格式二选一:
1) 需要调用工具:{"thought":"...", "action":{"tool":"工具名","input":{...}}}
2) 可以回答了:{"thought":"...", "final_answer":"给用户的回复"}

规则:只输出 JSON 本身,不要多余文字/解释/代码围栏。
```
(附 1~2 个 few-shot 示例,演示"先查订单→再查物流→作答"的链路。)

---

## 9. 测试计划(关键路径,用 MockLLM 离线跑,不烧钱)

`MockLLM` 满足阶段一的 `LLM` 接口,`chat()` 按**预设脚本**依次返回准备好的 JSON,
从而**确定性地**驱动循环,零 API 成本。覆盖:

- [ ] 直接回答:第一步就 `final_answer` → 一步返回
- [ ] 单工具:查订单 → 作答
- [ ] 多工具链:`query_order` → `query_logistics` → 作答(核心用例)
- [ ] 解析失败 → 喂回 → 下一步正确 → 成功恢复
- [ ] 未知工具 → 喂回可用工具 → 恢复
- [ ] 工具执行抛异常 → 喂回错误 → 恢复
- [ ] 撞 `max_steps` → 触发强制作答,`stopped_reason="max_steps"`

---

## 10. 验收标准(本阶段「做完」的判定)

- [ ] `ReActAgent.run(user_input, history)` 能跑通一条**≥2 步工具链**的 JD 客服例子。
- [ ] **交互式多轮 CLI** `examples/react_cli.py`:多轮对话、外层只存干净问答对、`/reset` 清空、`/trace` 可看中间步骤。
- [ ] `max_steps` 生效;撞上限触发**强制作答**而非崩溃。
- [ ] 模型输出走 **JSON + Pydantic 校验**;坏输出能**自我纠正**后继续。
- [ ] 未知工具 / 工具异常都能喂回并恢复,程序不崩。
- [ ] **加一个新 `Tool` 不需要改 `agent.py`**(可插拔验证)。
- [ ] `ReActAgent` **只依赖 `LLM` 接口**,换 provider 不影响。
- [ ] **MockLLM 单测**覆盖 §9 关键路径,离线可跑、无 API 费用。
- [ ] `ruff check .` 与 `black --check .` 通过;公开接口有类型标注与 docstring。

---

## 11. 未决问题 / 需用户确认

> **✅ 评审结论(2026-07-01 用户拍板,已定):**
> 1. **动作机制**:Prompt 式 ReAct + JSON,模型输出后由我们解析并执行;原生 Function Calling 留**阶段三**。
> 2. **工具深度**:极简 `Tool` 接口 + 2 个 JD 风格 **mock** 工具;完整 `BaseTool`/`ToolRegistry` 与真实接入(DB/RAG/**MCP**)留**阶段三**。
> 3. **输出格式**:每步单个 **JSON 对象**(`thought` + `action` | `final_answer`)+ **Pydantic** 校验。
> 4. **最大步数**:默认 **5**(可配);撞上限时**强制模型用现有信息作答**。
> 5. **错误恢复**:解析失败 / 未知工具 / 工具异常 → 当 **Observation 喂回**让模型自我纠正;API 级重试/降级留**阶段六**。
> 6. **返回值**:`run(user_input, history)` 返回结构化 **`AgentResult`**{`final_answer`, `steps`(完整轨迹), `stopped_reason`}。
> 7. **demo 主题**:**京东客服**(`query_order` / `query_logistics` mock)。
> 8. **交互形态**:做**交互式多轮 CLI** `examples/react_cli.py`(用户拍板)。上下文采「内外两层」策略(见 §6.2):
>    外层只存干净 (问题, 最终答案) 对、不持久化;`/reset` 清空、`/trace` 看中间步骤。
>    跨会话持久化记忆留**阶段四**。
