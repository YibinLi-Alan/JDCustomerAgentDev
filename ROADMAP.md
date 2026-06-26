# ROADMAP — Agent 框架实习路线图

> 基于 `frame/Agent框架实习培训大纲.md` 提炼的可执行路线图,4–6 周。
> 目标:从零搭一个可复用、可落地业务的通用 Agent 框架(完整六阶段,目标架构见大纲与 `CLAUDE.md`)。
> 每个阶段:**先写设计文档 → mentor 评审 → 再编码**。

## 进度总览
- [ ] 阶段一 基础认知与环境搭建
- [ ] 阶段二 最小 Agent 循环(ReAct)
- [ ] 阶段三 Tool Use 系统
- [ ] 阶段四 Memory 与上下文管理
- [ ] 阶段五 Planning 与 Multi-Agent
- [ ] 阶段六 生产化与业务落地

---

## 阶段一 · 基础认知与环境搭建(第 1 周前半,2–3 天)
**目标**:吃透 LLM API 调用、Prompt Engineering;搭好开发环境。
**核心点**:Chat Completion 结构(messages/role/temperature)· 流式输出 · Token 与成本 · System Prompt / Few-shot / CoT / 结构化 JSON 输出 · Python 3.10+ · 依赖管理 · `.env` 管 API Key。
**交付**
- [ ] 能多轮对话的 CLI 程序
- [ ] 实验报告:直接提问 vs CoT vs Few-shot 的输出质量对比
- [ ] 项目骨架搭好
**必读**:OpenAI API Reference · Prompt Engineering Guide · Lilian Weng《Prompt Engineering》

## 阶段二 · 最小 Agent 循环(第 1 周后半~第 2 周前半,3–4 天)
**目标**:理解 Agent vs Chatbot;实现最小可运行的 ReAct Loop。
**核心点**:感知→思考→行动自主循环 · 四要素(LLM/Tools/Memory/Planning)· Thought→Action→Observation · 最大步数防死循环 · 结构化输出解析 · 异常恢复。
**交付**
- [ ] `agent_loop.py` 骨架,能自主决定继续行动还是直接回答
- [ ] 设计文档:Agent Loop 状态机图
**必读**:ReAct 论文(2210.03629)· Lilian Weng《LLM Powered Autonomous Agents》· Simon Willison 极简实现

## 阶段三 · Tool Use 系统(第 2 周后半~第 3 周前半,4–5 天)
**目标**:掌握 Function Calling;设计可扩展 Tool 抽象层。
**核心点**:Function Calling 协议 · Tool 的 JSON Schema · 并行调用 · `BaseTool` 抽象 · `ToolRegistry` · Pydantic 参数校验 · 超时/异常/结果标准化。
**交付**
- [ ] `BaseTool` + `ToolRegistry`
- [ ] ≥5 个可用 Tool(含完整 Schema 与错误处理)
- [ ] Agent 能自主选择并调用 Tool
- [ ] 每个 Tool 的单元测试(正常/异常)
- [ ] 设计文档:Tool 系统类图 + 扩展指南
**必读**:OpenAI Function Calling Guide · Toolformer · OpenAI Swarm 源码

## 阶段四 · Memory 与上下文管理(第 3 周后半,3–4 天)
**目标**:实现短期记忆(对话管理)+ 长期记忆(向量检索)。
**核心点**:滑动窗口 · Token 预算 · Embedding · 向量库(Chroma/FAISS)· 余弦相似度 · 对话摘要 / 渐进式压缩 · `MemoryManager` 统一接口。
**交付**
- [ ] `MemoryManager`(短期 + 长期)
- [ ] 基于向量库的存取检索
- [ ] 自动摘要压缩
- [ ] Demo:跨会话记住关键信息
- [ ] 不同记忆策略的检索准确率对比
**必读**:MemGPT · Generative Agents · ChromaDB 文档 · Mem0 源码

## 阶段五 · Planning 与 Multi-Agent(第 4 周~第 5 周前半,5–7 天)
**目标**:任务分解与规划;实现多 Agent 协作。
**核心点**:Plan-and-Execute · 动态重规划 · DAG/层次分解 · Router/Supervisor/Hierarchical/Collaborative 架构 · Agent 间通信 · Reflection/Critic 自我纠错 · 串行/并行/条件/循环工作流。
**交付**
- [ ] Planning Agent(复杂任务拆子步骤)
- [ ] ≥2 种 Multi-Agent 架构(如 Router + Supervisor)
- [ ] 3 个专业 Agent(Coder / Researcher / Writer)
- [ ] Demo:多 Agent 协作完成复杂场景(如调研竞品出报告)
- [ ] 架构设计文档
**必读**:HuggingGPT · MetaGPT · AutoGen · Reflexion · LangGraph 源码

## 阶段六 · 生产化与业务落地(第 5 周后半~第 6 周,5–7 天)
**目标**:工程化 + 评估体系 + 可部署服务。
**核心点**:重试/降级/超时/幂等 · Trace 追踪 · 结构化日志与指标(成功率/步数/耗时/成本)· 评测集 + LLM-as-Judge + A/B · Prompt 注入防御 · 输出过滤 · 权限与成本限制 · FastAPI + SSE + Docker。
**交付**
- [ ] 完整框架代码库(全模块)
- [ ] Trace 可视化
- [ ] 评估报告(准确率/成功率/耗时)
- [ ] 可部署 HTTP 服务(支持流式)
- [ ] 安全测试报告(Prompt 注入防御)
- [ ] 部署文档
- [ ] **最终答辩 PPT**
**必读**:Chip Huyen《Building LLM Applications for Production》· Eugene Yan《LLM Patterns》· Judging LLM-as-a-Judge · OWASP Top 10 for LLM

---

## 贯穿全程
- **代码**:Type Hints 完整 · 接口用 Protocol/ABC · 关键路径有测试 · ruff/black 统一风格
- **文档**:每阶段先设计文档再编码 · docstring · README 说明运行与扩展
- **过程**:每日站会 · 每周周报(完成/问题/下周计划)· PR 需 Code Review
- **对标阅读**:LangChain · LangGraph · OpenAI Swarm · AutoGen · CrewAI · Dify

## 评分(Mentor 参考)
代码质量 25% · 设计能力 25% · 学习深度 20% · 业务效果 20% · 沟通协作 10%
