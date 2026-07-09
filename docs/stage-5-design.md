# 阶段五设计文档 · Planning 与 Multi-Agent

> 对应大纲「阶段五:Planning 与 Multi-Agent(5-7 天)」。
> 评审记录:2026-07-09 拍板 —— **线性步骤计划 / 专员按业务域切分 /
> Critic 审终稿回炉一次 / demo 用复合客诉一条龙**(四项均为推荐方案,用户确认)。
> 实施同样分三步走:P-A Planning → P-B Multi-Agent 双模式 → P-C Critic + demo + 评测,
> 每步测试全绿后提交。

## 1. 目标与范围

阶段五回答一个问题:**一条复杂客诉进来,框架怎么把它拆开、派给对的专员、
盯着执行完、再把结果审一遍交出去。**

- 新增 `planning/`(Planner + Executor)与 `multi_agent/`(protocol + Router + Supervisor)
  两个子包,外加 Critic(放 `multi_agent/critic.py`,属于协作编排的一环);
- **`core/agent.py` 一行不改**(延续阶段四纪律):专员内核就是现有 `ToolCallingAgent`,
  编排层只是"造 Agent、给工具子集、递上下文"的组装工;
- 记忆沿用阶段四 `MemoryManager`(user 维度隔离已就绪),编排层在最外圈 load / on_turn_end;
- 权限拦截(HITL)**不在本阶段**:售后专员会调用高权限工具,本阶段照常执行,
  阶段六在 executor/专员调用点接 `safety/approval.py` 闸门(接口这次留好)。

不做的(明确出界):AutoGen 式群聊(上下文膨胀不可控)、MetaGPT 式文档流水线(过重)、
Hierarchical/Collaborative 模式(大纲只要求 ≥2 种,做 Router + Supervisor)、
Reflexion 式多轮试错(只做 Critic 单轮回炉,理由见 §7)。

## 2. 交付物清单(逐条对照大纲「阶段产出」)

| # | 大纲产出 | 本设计的承接 |
|---|---|---|
| 1 | Planning Agent:复杂任务拆解为可执行子步骤 | `planning/planner.py`:`Planner.plan()` 产出结构化 `Plan`(线性步骤);`replan()` 动态重规划 |
| 2 | ≥2 种 Multi-Agent 架构模式 | `multi_agent/router.py`(Router 分诊直派)+ `multi_agent/supervisor.py`(Supervisor 中心调度) |
| 3 | 3 个专业 Agent | 订单物流专员 / 售后专员 / 商品导购专员(`multi_agent/specialists.py`,共用同一套定义,两种模式复用) |
| 4 | Agent 间通信协议和上下文管理 | `multi_agent/protocol.py`:`Specialist` 封装 + `TaskAssignment`/`TaskOutcome` 消息结构;上下文策略见 §8 |
| 5 | Demo:多 Agent 协作完成复杂业务场景 | `examples/multi_agent_cli.py`:复合客诉一条龙(查订单→退款→推荐替代品),/trace 展示计划、派工、重规划、Critic |
| 6 | 架构设计文档 | 本文档(含架构图 §4 与设计决策记录) |

## 3. 借鉴与对照(本阶段的 comparative reading 对象)

| 来源 | 借什么 | 改什么 / 不借什么 |
|---|---|---|
| Plan-and-Solve(arXiv 2305.04091)/ LangGraph plan-execute | 先规划再执行;重规划回路(原计划+已完成+失败原因→重排剩余) | 计划表示选线性列表而非 DAG(客服步骤强依赖,并行收益小;评审拍板) |
| OpenAI Swarm | Router = 一次分诊交接;专员 = prompt + 工具子集;user_id 不进 prompt(阶段四已用) | 不借"运行中随时互相 handoff"(链路不可控),分诊只在入口发生一次 |
| LangGraph Supervisor | 中心调度循环:Supervisor 决定下一步派谁,结果回中枢 | 我们的 Supervisor 不让 LLM 自由选下一步,而是**按 Planner 的计划派工**,LLM 自由度收敛到"重规划"一个口子(可测可预测) |
| AutoGen GroupChat | 只作对比阅读,理解"群聊"变体 | 不实现:上下文随参与者数膨胀,教学框架求可控 |
| Reflexion(arXiv 2303.11366) | 反思思想:失败信号转化为下一次尝试的语言反馈 | 简化为 Critic 单轮回炉:反思收益首轮最大,多轮试错成本/时长不可控 |
| MetaGPT(arXiv 2308.00352) | SOP 思想:客服 SOP(分诊→查证→方案→执行/转人工)编码进分工 | 不借文档驱动流水线;专员按业务域切而非软件工程角色(评审拍板) |

## 4. 架构图

```
                       用户反馈(user_id + query)
                                │
                        MemoryManager.load()          ← 阶段四,最外圈
                                │
                            Router 分诊
                     (一次 LLM 调用,输出结构化路由)
                      │                     │
              简单/单域问题             复杂/跨域问题
                      │                     │
                      ▼                     ▼
              对应专员直接处理          Supervisor
             (快路径,省钱省时)           │
                                 ┌──────┴──────┐
                                 │ Planner.plan() │ → Plan(线性步骤,每步标注执行专员)
                                 └──────┬──────┘
                                        │ 逐步执行
                          ┌─────────────┼─────────────┐
                          ▼             ▼             ▼
                    订单物流专员      售后专员       商品导购专员
                 (query_order/     (apply_refund/  (query_product/
                  logistics/        cancel_order/    search_faq/
                  user_orders/      create_ticket/   calculator)
                  current_time)     search_faq/
                                    query_order)
                          │             │             │
                          └──────┬──────┴─────────────┘
                          步骤失败│→ Planner.replan() 重排剩余步骤(最多 1 次)
                                 ▼
                            汇总答案
                                 │
                            Critic 审查(独立 LLM 调用,检查单)
                                 │ 不合格 → 带意见回炉重跑汇总(最多 1 次)
                                 ▼
                        MemoryManager.on_turn_end()
                                 │
                            解决方案输出
```

三个专员是**同一套 `Specialist` 定义**,Router 直派和 Supervisor 派工用的是同一批对象——
"定义一次、两种编排复用"是本阶段可拓展性的主要证据。

## 5. 核心设计一:Planning(`planning/`)

### 5.1 计划表示:线性步骤列表(评审拍板)

```python
@dataclass(frozen=True)
class PlanStep:
    id: int                  # 1 起步的序号
    description: str         # 这一步要做什么(给专员的任务描述)
    specialist: str          # 执行专员名(必须在编排器注册的专员集合内)

@dataclass(frozen=True)
class Plan:
    goal: str                # 原始诉求的一句话复述(重规划时锚定目标)
    steps: tuple[PlanStep, ...]
```

- 不做 DAG:客服场景步骤几乎串行强依赖(先查证→再操作→再推荐),并行收益小;
  线性让重规划语义干净(截断剩余步骤重排)。DAG 是扩展方向,答辩讲清楚即可。
- `specialist` 字段让 Planner 在拆解时就完成"步骤→专员"的指派
  (Planner 的 prompt 里带专员清单及职责描述,来自 `render_roster()`,见 §6.1);
  产出校验:引用不存在的专员 → 该步矫正为派给 Supervisor 配置的**兜底专员**(默认第一个)。

### 5.2 Planner:生成与重规划一体

```python
class Planner:
    def __init__(self, llm: LLM, *, max_steps: int = 6): ...
    def plan(self, goal: str, roster: str, context: str = "") -> Plan: ...
    def replan(self, plan: Plan, completed: Sequence[StepResult],
               failure: StepResult, roster: str) -> Plan: ...
```

- `plan()`:一次 LLM 调用,输出严格 JSON(数组每项 `{step, specialist}`);
  解析失败 → **降级为单步计划**(整个诉求打包成一步派给兜底专员 = 退化成阶段三的单 Agent
  行为,永不炸——延续全链路降级哲学);
- `replan()`:输入 = 目标 + 已完成步骤及其结果摘要 + 失败步骤及原因,输出**剩余部分的新计划**;
  已完成步骤不重跑(幂等意识,也是阶段六 reliability 的伏笔);
- `max_steps` 限制计划长度(prompt 约束 + 超长截断),防 Planner 把简单事拆成八步。

### 5.3 Executor:调度与失败上报

```python
@dataclass(frozen=True)
class StepResult:
    step: PlanStep
    output: str              # 专员的 final_answer
    ok: bool                 # 专员明示失败/异常 = False
    trace: AgentResult | None

class StepRunner(Protocol):   # 编码期定稿:executor 依赖此协议而非 Specialist
    def run_step(self, step: PlanStep, context: str) -> StepResult: ...

class PlanExecutor:
    def __init__(self, runner: StepRunner, *, replanner: Planner | None = None,
                 roster: str = "", specialists: tuple[str, ...] = (),
                 max_replans: int = 1): ...
    def execute(self, plan: Plan, *, notes: ScratchPad | None = None)
        -> ExecutionResult: ...  # results + replanned + 最终计划
```

> 编码期修订(架构决策):executor 不直接持有 ``Specialist`` 映射,改依赖
> ``StepRunner`` 协议,由 Supervisor 提供适配器 —— 保证 **planning 不 import
> multi_agent** 的依赖方向纪律;返回值升级为 ``ExecutionResult``
> (Supervisor 需要 ``replanned`` 标志)。

- 顺序执行;每步把 `ScratchPad`(§8)渲染进专员任务,专员产出回填 ScratchPad——
  这就是步骤间传递中间结果的通道(订单专员查到的单号,售后专员下一步能看见);
- 步骤失败(专员异常/超 max_steps/明示办不到)→ 触发 `replan()`,剩余步骤替换为新计划,
  **重规划全局最多 1 次**;再失败则如实进结果(Critic 与最终汇总会看到失败,如实告知用户
  或建议转人工),不无限兜圈。

## 6. 核心设计二:Multi-Agent(`multi_agent/`)

### 6.1 通信协议(`protocol.py`)——大纲产出④

```python
@dataclass(frozen=True)
class Specialist:
    name: str                # "order_agent" 等,机器名
    title: str               # "订单物流专员",给 prompt/展示用
    description: str         # 职责边界(Router/Planner 选人的依据)
    registry: ToolRegistry   # 工具子集
    system_prompt: str       # 专属人设与行为规则
    def build(self, llm: LLM, *, extra_system: str = "", max_steps: int = 5)
        -> ToolCallingAgent: ...   # 按轮现造,无状态

@dataclass(frozen=True)
class TaskAssignment:        # 编排器 → 专员
    task: str                # 任务描述(Router 直派 = 原 query;Supervisor = 步骤描述)
    context: str             # 共享上下文渲染文本(ScratchPad + 记忆附加段)

@dataclass(frozen=True)
class TaskOutcome:           # 专员 → 编排器
    specialist: str
    answer: str
    ok: bool
    trace: AgentResult | None

def render_roster(specialists: Sequence[Specialist]) -> str:
    """专员花名册(name + title + description),给 Router 和 Planner 的 prompt 用。"""
```

- 专员之间**不直接通话**,一切经编排器中转(星型拓扑)——链路可 trace、可测;
- `Specialist` 是纯数据 + 一个工厂方法,不持有会话状态;
  给 `ToolRegistry` 加 `subset(*names) -> ToolRegistry` 便利方法(从全量 registry 挑子集,
  工具实例共享同一 `JDMockStore`,跨专员状态变化可见——售后专员退了款,订单专员再查看得到)。

### 6.2 三个专员(`specialists.py`)——按业务域切(评审拍板)

| 专员 | 工具子集 | 职责一句话 |
|---|---|---|
| `order_agent` 订单物流专员 | query_order · query_logistics · query_user_orders · current_time | 订单状态、物流进度、历史订单的一切查询 |
| `aftersales_agent` 售后专员 | query_order · apply_refund · cancel_order · create_ticket · search_faq | 退款/取消/工单等售后处理(高权限工具全在这,阶段六闸门只需卡他) |
| `product_agent` 商品导购专员 | query_product · search_faq · calculator | 商品信息、推荐、价格计算 |

- `query_order` 在订单与售后专员间有意重叠:售后操作前需自行查证,不依赖跨专员喊话;
- 每个专员的 `system_prompt` = 阶段三客服基础规则 + 本域职责 + **边界条款**
  ("超出职责的诉求,说明办不到并建议找哪位专员"——Supervisor 重规划的信号源);
- 装配函数 `create_specialists(llm, registry) -> dict[str, Specialist]`,
  与 `presets.default_registry()` 同一模式:加专员 = 加一条定义,编排器不改。

### 6.3 Router 模式(`router.py`)——模式一:分诊直派

```python
class Router:
    def __init__(self, llm: LLM, specialists: Mapping[str, Specialist],
                 *, complex_marker: str = "supervisor"): ...
    def route(self, query: str, context: str = "") -> RouteDecision: ...
        # RouteDecision(target="order_agent"|...|"supervisor", reason=...)
```

- 一次 LLM 调用输出严格 JSON `{"target": ..., "reason": ...}`;花名册来自 `render_roster()`;
- 判据写进 prompt:**单域可答 → 直派对应专员;跨域/多动作/需先查证再操作 → "supervisor"**;
- 解析失败/未知目标 → 降级派 supervisor(宁可走贵的全能路径,不给用户错误专员);
- 这是 Swarm handoff 的入口版:分诊只发生一次,派出后该专员全权负责本轮。

### 6.4 Supervisor 模式(`supervisor.py`)——模式二:中心调度

```python
class Supervisor:
    def __init__(self, llm: LLM, specialists: Mapping[str, Specialist],
                 *, planner: Planner | None = None, critic: Critic | None = None,
                 max_steps_per_specialist: int = 5): ...
    def handle(self, query: str, *, context: str = "") -> SupervisorResult: ...
        # SupervisorResult(final_answer, plan, step_results, replanned, critique)
```

- 流程:`Planner.plan()` → `PlanExecutor.execute()`(失败重规划≤1 次)→
  **汇总调用**(一次 LLM:目标 + 各步结果 → 面向用户的解决方案)→ `Critic` 审(§7);
- 与 LangGraph Supervisor 的关键差异(答辩点):不让 LLM 每步自由选人,
  而是按计划派工,LLM 自由度收敛在 plan/replan 两个口子——行为可预测、轨迹可断言;
- `planner=None` 时退化为"整包派给兜底专员"(组合自由度,与 MemoryManager 三件套同哲学);
- `SupervisorResult` 携带全轨迹(计划、每步结果、是否重规划、Critic 意见),
  CLI 的 /trace 与测试断言都吃这个结构——也是阶段六 observability 的数据源伏笔。

## 7. 核心设计三:Critic 审终稿(`multi_agent/critic.py`,评审拍板)

```python
class Critic:
    def __init__(self, llm: LLM, *, max_retries: int = 1): ...
    def review(self, query: str, answer: str, evidence: str) -> Critique: ...
        # Critique(passed: bool, issues: list[str], suggestion: str)
```

- 检查单(写进 prompt):①用户的**每一项**诉求都被回应了吗(复合客诉最容易漏)
  ②答案与工具证据一致吗,有没有编造 ③该办没办成的事,是否如实告知/建议转人工
  ④语气与格式合格吗;
- `evidence` = 各步骤结果摘要——Critic 对照证据审,不是凭空审;
- 不合格 → 把 `issues` 拼进汇总 prompt 重跑**汇总调用**(不重跑工具步骤,便宜且幂等),
  最多 1 次;二审仍不合格 → 放行但在 trace 里记录(demo 可见,阶段六 metrics 的素材);
- 解析失败 → 视为通过(Critic 是增益件,不能成为新的故障点——降级哲学)。

覆盖大纲 5.3 三项:Reflection(执行后检查)= review;Critic Agent = 本类;
自动重试与替代方案 = 回炉一次 + 重规划里的替代路径。

## 8. 上下文管理:共享黑板 + 隔离内环(大纲 5.2「共享 vs 隔离」的回答)

| 层 | 策略 | 理由 |
|---|---|---|
| 专员内环(tool_use/tool_result 循环) | **隔离**:每次派工新建 `ToolCallingAgent`,内环消息不出专员 | 中间过程是噪音;阶段二"内外两层上下文"纪律的延续 |
| 步骤间中间结果 | **共享黑板 `ScratchPad`**:每步产出以 `[step-N 专员] 结论` 追加,渲染进下一步的 `TaskAssignment.context` | 订单专员查到的物流单号,售后/导购专员要用;显式传递优于隐式共享内存 |
| 用户维度记忆 | 沿用阶段四:编排器最外圈 `MemoryManager.load()` 注入 / `on_turn_end()` 落账,`ctx.system_suffix()` 拼进**每个**专员的 system(`Specialist.build(extra_system=...)`) | 记忆属于会话不属于能力,专员共享"这个用户是谁"的同一份事实 |
| 外层对话历史 | 只存干净问答对(user query + 最终方案),供跨轮指代 | 阶段二约定的延续;计划/派工轨迹不进历史,进 trace |

`ScratchPad` 是个纯数据小类(append / render,含 token 上限截断保护),放 `planning/executor.py`。

## 9. Demo 设计(`examples/multi_agent_cli.py`)

- 主场景(评审拍板"复合客诉一条龙"):
  `我买的降噪耳机用了三天就坏了,查一下我的订单还能不能退,能退帮我申请,再推荐一个靠谱的替代品`
  → Router 判复杂 → Planner 拆 ~4 步(查订单验证→申请退款→查替代品→汇总方案)
  → 订单/售后/导购三专员接力(ScratchPad 传单号)→ Critic 审 → 输出解决方案;
- 快路径对照:`订单 12345 到哪了` → Router 直派订单专员(演示分诊省钱);
- 重规划演示:mock 数据里放一单**超退款时效**的订单,退款步失败 → replan 改走
  `create_ticket` 转人工路径(顺带预演阶段六 HITL 叙事);
- 命令:`/trace`(计划/派工/重规划/Critic 全轨迹)、`/user <id>`(记忆沿用)、
  `/mode router|supervisor|auto`(强制模式,方便对比演示)、`/reset` `/help` `/exit`。

## 10. 配置与依赖变更

- `Settings` 新增:`planner_max_steps: int = 6`、`max_replans: int = 1`、
  `critic_max_retries: int = 1`、`supervisor_specialist_max_steps: int = 5`
  (全部环境变量可覆写,规则不写死——可拓展性硬要求);
- 无新第三方依赖(编排是纯 Python + 现有 LLM 接口);
- `agent_framework/__init__.py` 导出新公共类,`__version__ = "0.5.0"`。

## 11. 测试计划(全部离线:MockLLM 脚本化)

| 模块 | 关键用例 |
|---|---|
| Planner | 正常拆解(JSON→Plan);specialist 越界矫正兜底;JSON 非法降级单步计划;超长截断;replan 只重排剩余步骤、已完成不重跑 |
| Executor | 顺序执行与 ScratchPad 传递;步骤失败触发 replan(恰一次);再失败如实进结果;replanner=None 时失败直接记录 |
| protocol/Specialist | subset() 正确切子集且共享 store;build() 拼接 extra_system;render_roster 含全部专员 |
| Router | 正常路由;跨域判 supervisor;非法 JSON/未知目标降级 supervisor |
| Supervisor | 全链路编排顺序(plan→步→汇总→critic);planner=None 退化整包直派;SupervisorResult 轨迹完整 |
| Critic | 通过/不合格回炉一次/二审放行留痕;解析失败视为通过 |
| 集成 | 复合客诉端到端(MockLLM 脚本:路由→计划→3 专员→汇总→critic);重规划端到端;记忆注入到达每个专员的 system |

## 12. 实施顺序(P-A/B/C,每步测试全绿后 commit)

1. **P-A Planning**:`Plan`/`PlanStep`/`Planner`/`PlanExecutor`/`ScratchPad` +
   `ToolRegistry.subset()` + 测试;
2. **P-B Multi-Agent**:`protocol.py`/`specialists.py`/`router.py`/`supervisor.py` + 测试;
3. **P-C 收口**:`critic.py` + `examples/multi_agent_cli.py` + 集成测试 + 真实 API 冒烟
   (复合客诉一条龙)+ 文档与知识库。

---

## 附录 A:各角色 system prompt 初稿(评审 2026-07-09 补充)

> 定稿以代码为准;结构与要点如下,编码时允许措辞微调,不允许增删条款。

**Router(分诊员,单次调用)**:角色 + `render_roster()` 花名册 + 判定规则
(单域可答→直派;跨域/多动作/先查证后操作→"supervisor")+ 严格 JSON
`{"target", "reason"}`。

**Planner(规划师,单次调用)**:角色 + 花名册 + 已知背景(记忆附加段)+ 约束
(只拆必要步骤 ≤`planner_max_steps`;每步具体可执行、含已知单据号;操作类步骤前
必须有查证步骤;信息不足安排"向用户确认"步骤而非凭空假设)+ 严格 JSON 数组
`[{"step", "specialist"}]`。重规划变体追加:原目标 + 已完成步骤及结果 + 失败步骤
及原因,要求只重排未完成部分、失败路径换替代方案(如退款超时效→建工单转人工)。

**三专员 = 公共底座 + 域块**。公共底座 = 阶段三 `DEFAULT_TOOL_CALLING_SYSTEM_PROMPT`
三规则(指代解析/不重复索要/失败先核参数)+ 两条协作条款:①【前序步骤结论】里
已有的信息直接使用,不重复查询;②超出职责的诉求不勉强处理,明示需转交谁。域块:
- 订单物流:只查询不修改;回答带关键单据号;
- 售后:操作前必须先 `query_order` 核实状态与时效;不符合条件如实说明并评估建工单
  转人工,不强行操作;
- 导购:推荐给理由(参数/价格对比);不知道的参数如实说,不编造。

**Supervisor 汇总(单次调用)**:根据执行记录给用户写最终答复——按诉求逐项交代;
办成的给凭据(单号/工单号),没办成的如实说明原因与替代方案;不提内部分工与步骤编号。

**Critic(质检员,单次调用)**:四项检查单(①每项诉求都回应了吗 ②与执行证据一致吗、
有无编造 ③没办成的是否如实告知/给替代方案 ④语气专业友好)+ 严格 JSON
`{"passed", "issues", "suggestion"}`。

## 附录 B:循环护栏总表(评审问题"做几次强制结束")

| 层 | 上限(均入 Settings 可配) | 触发后行为 |
|---|---|---|
| 专员内环 | `max_steps=5`(阶段二护栏) | 强制作答,`ok=False` 上报 |
| 计划长度 | `planner_max_steps=6` | prompt 约束 + 超长硬截断 |
| 重规划 | `max_replans=1`(整轮全局) | 再失败如实进结果,不兜圈 |
| Critic 回炉 | `critic_max_retries=1` | 二审仍不过→放行 + trace 留痕 |
| Router/Planner/汇总/Critic 自身 | 单次调用,无循环 | — |

最坏情况单轮 LLM 调用 ≈ 43 次封顶(1 路由 + 1 计划 + 6 步 × 6 + 1 重规划 + 2 汇总 +
2 Critic),复合客诉 demo 实际预计 12~15 次;不存在无限循环路径。

## 附录 C:评审问题"做错了要不要存进记忆"的结论

- **轮内错误必须喂回**(已设计):工具报错喂回专员循环、失败原因喂 replan、
  Critic 意见喂回炉——即 Reflexion 思想的单轮内落地;
- **跨轮"教训库"不做**,三条理由:①错误多绑定具体单据,跨单据无参考价值且会误导
  (撞上"不记易变状态"规则);②系统性错误的正确修法是改 prompt/工具描述
  (阶段六评测负责抓),不是往单个用户的记忆里塞补丁;③LLM 自我归因常错,
  无验证机制的"教训"会长期污染行为。写入对比阅读的"不借"清单;
- **但"错误的后果"作为事实自然进记忆是对的**:阶段四提炼规则记承诺/诉求,
  "订单 12345 退款未成,已建工单 T-001 跟进"会被正常存下,下轮"我那事怎么样了"
  能接上——记**事情的状态**,不记**我犯过的错**;
- 带验证机制的教训记忆列入阶段六 backlog 候选(待用户定)。
