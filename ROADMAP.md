# ROADMAP — Agent 框架实习路线图

> 基于 `frame/Agent框架实习培训大纲.md` 提炼的可执行路线图,4–6 周。
> 目标:从零搭一个可复用、可落地业务的通用 Agent 框架(完整六阶段,目标架构见大纲与 `CLAUDE.md`)。
> 每个阶段:**先写设计文档 → mentor 评审 → 再编码**。

## 进度总览
- [x] 阶段一 基础认知与环境搭建 ✅(2026-06-28)
- [x] 阶段二 最小 Agent 循环(ReAct)✅(2026-07-01)
- [x] 阶段三 Tool Use 系统 ✅(2026-07-06)
- [x] 阶段四 Memory 与上下文管理 ✅(2026-07-07)
- [x] 阶段五 Planning 与 Multi-Agent ✅(2026-07-09)
- [x] 阶段六 生产化与业务落地 ✅(2026-07-12)—— **六阶段全部完成 🎉**

---

## 阶段一 · 基础认知与环境搭建(第 1 周前半,2–3 天)
**目标**:吃透 LLM API 调用、Prompt Engineering;搭好开发环境。
**核心点**:Chat Completion 结构(messages/role/temperature)· 流式输出 · Token 与成本 · System Prompt / Few-shot / CoT / 结构化 JSON 输出 · Python 3.10+ · 依赖管理 · `.env` 管 API Key。
**交付**
- [x] 能多轮对话的 CLI 程序(`examples/chat_cli.py`;并做成可切换 provider:Claude / OpenAI)
- [x] 实验报告:直接提问 vs CoT vs Few-shot 的输出质量对比(`docs/stage-1-prompt-experiment.md`)
- [x] 项目骨架搭好(`agent_framework/` + pyproject/ruff/black/.env.example)
**必读**:OpenAI API Reference · Prompt Engineering Guide · Lilian Weng《Prompt Engineering》

## 阶段二 · 最小 Agent 循环(第 1 周后半~第 2 周前半,3–4 天)
**目标**:理解 Agent vs Chatbot;实现最小可运行的 ReAct Loop。
**核心点**:感知→思考→行动自主循环 · 四要素(LLM/Tools/Memory/Planning)· Thought→Action→Observation · 最大步数防死循环 · 结构化输出解析 · 异常恢复。
**交付**
- [x] ReAct Loop(`core/agent.py` 的 `ReActAgent`),能自主决定继续行动还是直接回答
- [x] 设计文档:Agent Loop 状态机图(`docs/stage-2-design.md` §4)
- [x] 附加:极简 `Tool` 接口 + JD mock 工具、交互式多轮 CLI(`examples/react_cli.py`)、MockLLM + 14 单测
**必读**:ReAct 论文(2210.03629)· Lilian Weng《LLM Powered Autonomous Agents》· Simon Willison 极简实现

## 阶段三 · Tool Use 系统(第 2 周后半~第 3 周前半,4–5 天)
**目标**:掌握 Function Calling;设计可扩展 Tool 抽象层。
**核心点**:Function Calling 协议 · Tool 的 JSON Schema · 并行调用 · `BaseTool` 抽象 · `ToolRegistry` · Pydantic 参数校验 · 超时/异常/结果标准化。
**交付**
- [x] `BaseTool` + `ToolRegistry`(P-A:strict mode + 超时 + `ToolResult` 标准化 + `@tool` 装饰器 + permission 权限标记)
- [x] ≥5 个可用 Tool(P-C:11 个 = 8 业务 + 3 通用,共享 `JDMockStore` 假数据层,`default_registry()` 装配)
- [x] Agent 能自主选择并调用 Tool(P-B:原生 Function Calling,`ToolCallingAgent` + Claude/OpenAI 双协议适配 + 并行调用;demo:`examples/tools_cli.py`)
- [x] 每个 Tool 的单元测试(正常/异常)(tests 共 70 条全过)
- [x] 设计文档:Tool 系统类图 + 扩展指南(`docs/stage-3-design.md`,§8 含评审定稿的七条工具规范)
**必读**:OpenAI Function Calling Guide · Toolformer · OpenAI Swarm 源码

## 阶段四 · Memory 与上下文管理(第 3 周后半,3–4 天)
**目标**:实现短期记忆(对话管理)+ 长期记忆(向量检索)。
**核心点**:滑动窗口 · Token 预算 · Embedding · 向量库(Chroma/FAISS)· 余弦相似度 · 对话摘要 / 渐进式压缩 · `MemoryManager` 统一接口。
**交付**
- [x] `MemoryManager`(短期 + 长期)(`memory/manager.py` 统一门面,三件套可缺省组合,`create_memory_manager()` 一行装配)
- [x] 基于向量库的存取检索(`memory/long_term.py`:LLM 提炼事实 → Mem0 式增删改裁决 → 三因子检索;`vector_store.py`:Chroma 持久化 + 内存版双实现;user_id 隔离在存储层强制)
- [x] 自动摘要压缩(`memory/short_term.py` 滑动窗口按轮弹出 + `compressor.py` 递归摘要,失败降级不炸对话)
- [x] Demo:跨会话记住关键信息(`examples/memory_cli.py`:/user 隔离 + Chroma 落盘重启仍记得,真实 API 冒烟通过,写入裁决 NOOP 去重在真实运行中生效)
- [x] 不同记忆策略的检索准确率对比(`evaluation/memory_retrieval_eval.py` + 16 查询标注集;三因子轻量 1.0/0.25/0.25 Hit@1 94% 完胜,数据驱动改默认权重,报告 `docs/stage-4-memory-eval.md`)
**必读**:MemGPT · Generative Agents · ChromaDB 文档 · Mem0 源码

## 阶段五 · Planning 与 Multi-Agent(第 4 周~第 5 周前半,5–7 天)
**目标**:任务分解与规划;实现多 Agent 协作。
**核心点**:Plan-and-Execute · 动态重规划 · DAG/层次分解 · Router/Supervisor/Hierarchical/Collaborative 架构 · Agent 间通信 · Reflection/Critic 自我纠错 · 串行/并行/条件/循环工作流。
**交付**
- [x] Planning Agent(复杂任务拆子步骤)✅(2026-07-09:`planning/` —— Planner 线性计划 + 越界矫正/降级单步;PlanExecutor 顺序派工 + ScratchPad 黑板 + 动态重规划 ≤1 次)
- [x] ≥2 种 Multi-Agent 架构(Router + Supervisor)✅(2026-07-09:`multi_agent/router.py` 分诊直派(降级 supervisor)+ `supervisor.py` 中心调度(按计划派工,LLM 自由度收敛在 plan/replan))
- [x] 3 个专业 Agent ✅(2026-07-09:按业务域切 —— 订单物流/售后/商品导购专员,工具子集经 `registry.subset()`,定义一次两种编排复用;评审拍板替代大纲原例 Coder/Researcher/Writer)
- [x] Agent 间通信协议和上下文管理 ✅(`protocol.py`:Specialist/TaskAssignment/TaskOutcome + 「无法完成:」失败前缀约定;隔离内环 + ScratchPad 共享黑板 + 记忆注入全员)
- [x] Demo:多 Agent 协作完成复杂场景 ✅(2026-07-09:`examples/multi_agent_cli.py` 复合客诉一条龙 —— 查证→退款被 7 天规则拦→重规划建工单转人工→推荐→Critic 回炉;真实 API 冒烟通过)
- [x] 架构设计文档 ✅(`docs/stage-5-design.md`,含架构图/借鉴对照/循环护栏/评审附录)
**必读**:HuggingGPT · MetaGPT · AutoGen · Reflexion · LangGraph 源码
**加练**:Critic 终稿质检(大纲 5.3,不合格回炉 ≤1 次,解析失败降级放行);`apply_refund` 补 7 天无理由时效规则(重规划演示的真实触发点)

## 阶段六 · 生产化与业务落地(第 5 周后半~第 6 周,5–7 天)
**目标**:工程化 + 评估体系 + 可部署服务。
**核心点**:重试/降级/超时/幂等 · Trace 追踪 · 结构化日志与指标(成功率/步数/耗时/成本)· 评测集 + LLM-as-Judge + A/B · Prompt 注入防御 · 输出过滤 · 权限与成本限制 · FastAPI + SSE + Docker。
**交付**
- [x] 完整框架代码库(全模块)✅(2026-07-12:补齐 observability/ safety/ api/,evaluation/ 扩容;目标架构 10 子包全部落地)
- [x] Trace 可视化 ✅(`observability/tracer.py` JSONL 落盘 + `examples/trace_viewer.py` 终端时间线;core 加唯一 on_event 只读钩子)
- [x] 评估报告(准确率/成功率/耗时)✅(`agent_eval.py` 25 例端到端 + LLM-as-Judge 四维;真实跑 76% 通过,`docs/stage-6-eval-report.md`)
- [x] 可部署 HTTP 服务(支持流式)✅(`api/server.py` FastAPI:/chat + /chat/stream SSE + /approvals;真实冒烟全通过)
- [x] 安全测试报告(Prompt 注入防御)✅(`security_eval.py` 9 例:直接/间接注入+越权+泄漏+刷量,9/9 防住,`docs/stage-6-security-report.md`)
- [x] 部署文档 ✅(`docs/deployment.md` + `Dockerfile` + `.dockerignore`,本地/Docker 两路径)
- [ ] **最终答辩 PPT**(编码全部完成,PPT 待做——素材=六阶段知识库笔记 + 各评测报告)
**加练**:可靠性(ReliableLLM 指数退避 + FallbackLLM 降级 + 工具 request_id 幂等)· **HITL 人工介入闭环**(ApprovalGate 权限闸门 + HandoffQueue 审批/升级两入口 + 审批后幂等执行)· AgentService 整栈门面 · 四层超时保险
**必读**:Chip Huyen《Building LLM Applications for Production》· Eugene Yan《LLM Patterns》· Judging LLM-as-a-Judge · OWASP Top 10 for LLM

---

## 贯穿全程
- **代码**:Type Hints 完整 · 接口用 Protocol/ABC · 关键路径有测试 · ruff/black 统一风格
- **文档**:每阶段先设计文档再编码 · docstring · README 说明运行与扩展
- **过程**:每日站会 · 每周周报(完成/问题/下周计划)· PR 需 Code Review
- **对标阅读**:LangChain · LangGraph · OpenAI Swarm · AutoGen · CrewAI · Dify

## 评分(Mentor 参考)
代码质量 25% · 设计能力 25% · 学习深度 20% · 业务效果 20% · 沟通协作 10%
