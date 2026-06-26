# Agent 框架实习培训大纲

> 目标：从零搭建一个可用于业务的 Agent 框架
> 建议周期：4-6 周
> 适用对象：有基础编程能力的实习生

---

## 阶段一：基础认知与环境搭建（第 1 周前半，2-3 天）

### 学习目标

- 理解 LLM 的基本工作原理和 API 调用方式
- 掌握 Prompt Engineering 核心技巧
- 搭建好开发环境

### 学习内容

#### 1.1 LLM API 基础

- Chat Completion API 的请求结构（messages、role、temperature、max_tokens）
- 流式输出（streaming）的原理与使用
- Token 计算与成本概念

#### 1.2 Prompt Engineering

- System Prompt 设计原则
- Few-shot Prompting（少样本提示）
- Chain-of-Thought（思维链）提示
- 结构化输出引导（让模型输出 JSON）

#### 1.3 开发环境

- Python 项目初始化（推荐 Python 3.10+）
- 依赖管理（Poetry 或 pip + requirements.txt）
- API Key 安全管理（环境变量、.env 文件）
- 基础项目结构设计

### 学习材料

| 类型 | 材料 | 说明 |
|------|------|------|
| 官方文档 | [OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat) | 必读，理解 API 结构 |
| 官方文档 | [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) | 官方 Prompt 最佳实践 |
| 课程 | [DeepLearning.AI - ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) | 免费短课，1-2 小时完成 |
| 课程 | [DeepLearning.AI - Building Systems with ChatGPT API](https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/) | 理解如何用 API 构建系统 |
| 文章 | [Lilian Weng - Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) | 学术视角的 Prompt 技巧总结 |
| 实践 | [OpenAI Cookbook](https://cookbook.openai.com/) | 各种实用代码示例 |

### 阶段产出

- [ ] 一个能调用 LLM API 完成多轮对话的 CLI 程序
- [ ] 实验报告：对比不同 Prompt 策略（直接提问 vs CoT vs Few-shot）对输出质量的影响
- [ ] 项目基础骨架搭建完成

---

## 阶段二：最小 Agent 循环（第 1 周后半 ~ 第 2 周前半，3-4 天）

### 学习目标

- 理解 Agent 与普通 Chatbot 的本质区别
- 掌握 ReAct 循环模式
- 实现一个最小可运行的 Agent Loop

### 学习内容

#### 2.1 Agent 核心概念

- Agent 的定义：感知 → 思考 → 行动的自主循环
- Agent vs Chatbot：自主决策能力、工具使用、目标驱动
- Agent 的组成要素：LLM（大脑）、Tools（手脚）、Memory（记忆）、Planning（规划）

#### 2.2 ReAct 模式

- Reasoning + Acting 交替进行
- Thought → Action → Observation 循环
- 循环终止条件设计

#### 2.3 Agent Loop 工程实现

- 主循环控制流
- 最大步数限制（防止死循环）
- 结构化输出解析（让 LLM 输出可执行的指令）
- 错误处理与异常恢复

### 学习材料

| 类型 | 材料 | 说明 |
|------|------|------|
| 论文 | [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) | **核心必读论文**，理解 Agent 的理论基础 |
| 博客 | [Lilian Weng - LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) | **强烈推荐**，Agent 领域最好的综述文章 |
| 课程 | [DeepLearning.AI - AI Agents in LangGraph](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/) | 理解 Agent 构建模式 |
| 代码 | [LangChain ReAct Agent 源码](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/react/agent.py) | 参考工业级实现 |
| 视频 | [What are AI Agents? - IBM Technology](https://www.youtube.com/watch?v=F8NKVhkZZWI) | 快速理解概念，10 分钟 |
| 文章 | [Building a ReAct Agent from Scratch](https://til.simonwillison.net/llms/python-react-pattern) | Simon Willison 的极简实现参考 |

### 阶段产出

- [ ] 实现 Agent Loop 骨架代码（`agent_loop.py`）
- [ ] Agent 能根据用户输入自主思考，决定是否需要进一步行动或直接回答
- [ ] 设计文档：Agent Loop 的状态机图和流程说明

### 核心代码骨架参考

```python
class AgentLoop:
    def run(self, user_input: str) -> str:
        context = [{"role": "user", "content": user_input}]
        for step in range(self.max_steps):
            response = self.llm.chat(context)
            if response.is_final_answer:
                return response.content
            action = self.parse_action(response)
            observation = self.execute(action)
            context.append(observation)
        return "达到最大步数限制"
```

---

## 阶段三：Tool Use 系统（第 2 周后半 ~ 第 3 周前半，4-5 天）

### 学习目标

- 理解 Function Calling 的原理和协议
- 设计可扩展的 Tool 抽象层
- 实现多个实用 Tool 并集成到 Agent 中

### 学习内容

#### 3.1 Function Calling 机制

- OpenAI Function Calling 协议详解
- Tool 描述的 JSON Schema 规范
- 模型如何理解和选择 Tool
- 并行 Tool 调用（parallel function calling）

#### 3.2 Tool 系统设计

- `BaseTool` 抽象基类设计
- Tool 参数定义与验证（使用 Pydantic/JSON Schema）
- `ToolRegistry`：Tool 的注册、发现、管理
- Tool 描述的最佳实践（如何让模型更好地理解 Tool）

#### 3.3 Tool 执行引擎

- 参数解析与类型转换
- 执行超时控制
- 异常捕获与错误信息格式化
- 执行结果的标准化返回

#### 3.4 实现常用 Tool

- `WebSearch`：网络搜索
- `CodeExecutor`：代码沙箱执行
- `FileReader/Writer`：文件读写
- `HttpRequest`：HTTP API 调用
- `Calculator`：数学计算

### 学习材料

| 类型 | 材料 | 说明 |
|------|------|------|
| 官方文档 | [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling) | **必读**，理解 Function Calling 协议 |
| 论文 | [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) | 理解模型如何学会使用工具 |
| 论文 | [Gorilla: Large Language Model Connected with APIs](https://arxiv.org/abs/2305.15334) | API 调用的模型能力研究 |
| 代码 | [LangChain Tools 源码](https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/tools) | 参考 Tool 抽象设计 |
| 代码 | [OpenAI Swarm 框架](https://github.com/openai/swarm) | OpenAI 官方轻量 Agent 框架，Tool 设计参考 |
| 规范 | [JSON Schema 官方文档](https://json-schema.org/learn/getting-started-step-by-step) | Tool 参数定义的基础 |
| 课程 | [DeepLearning.AI - Functions, Tools and Agents with LangChain](https://www.deeplearning.ai/short-courses/functions-tools-agents-langchain/) | Function Calling 实践课 |

### 阶段产出

- [ ] `BaseTool` 抽象类 + `ToolRegistry` 注册中心
- [ ] 实现至少 5 个可用 Tool，每个 Tool 有完整的参数 Schema 和错误处理
- [ ] Agent 能根据用户问题自主选择并调用合适的 Tool
- [ ] 单元测试：每个 Tool 的正常/异常场景覆盖
- [ ] 设计文档：Tool 系统的类图和扩展指南

---

## 阶段四：Memory 与上下文管理（第 3 周后半，3-4 天）

### 学习目标

- 理解 Agent 记忆系统的分类和作用
- 实现短期记忆（对话管理）和长期记忆（向量检索）
- 掌握上下文窗口的高效管理策略

### 学习内容

#### 4.1 短期记忆（Working Memory）

- 对话历史管理
- 滑动窗口策略
- Token 预算计算与控制
- 消息优先级排序

#### 4.2 长期记忆（Long-term Memory）

- Embedding 原理与模型选择
- 向量数据库基础（ChromaDB / FAISS / Milvus）
- 相似度检索算法（余弦相似度、内积）
- 记忆的存储、检索、更新、删除

#### 4.3 上下文压缩策略

- 对话摘要（Summarization）
- 重要信息提取
- 渐进式摘要（递归压缩长对话）
- 混合策略：近期保留原文 + 早期使用摘要

#### 4.4 记忆架构设计

- MemoryManager 统一接口
- 记忆的生命周期管理
- 记忆与 Agent Loop 的集成

### 学习材料

| 类型 | 材料 | 说明 |
|------|------|------|
| 博客 | [Lilian Weng - LLM Powered Autonomous Agents (Memory 部分)](https://lilianweng.github.io/posts/2023-06-23-agent/#memory) | Memory 系统综述 |
| 论文 | [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) | 虚拟上下文管理，分页式记忆 |
| 论文 | [Generative Agents: Interactive Simulacra](https://arxiv.org/abs/2304.03442) | 斯坦福小镇，经典的 Agent 记忆架构 |
| 官方文档 | [ChromaDB 文档](https://docs.trychroma.com/) | 轻量级向量数据库，推荐入门使用 |
| 官方文档 | [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings) | 理解文本向量化 |
| 课程 | [DeepLearning.AI - LangChain Chat with Your Data](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/) | 向量检索实践 |
| 代码 | [Mem0 (原 EmbedChain)](https://github.com/mem0ai/mem0) | 开源 Agent 记忆层实现参考 |

### 阶段产出

- [ ] `MemoryManager` 类：统一管理短期/长期记忆
- [ ] 短期记忆：滑动窗口 + Token 预算控制
- [ ] 长期记忆：基于向量数据库的存储与检索
- [ ] 上下文压缩：自动摘要策略实现
- [ ] Demo：Agent 能记住历史对话中的关键信息，跨会话引用
- [ ] 性能测试：不同记忆策略下的检索准确率对比

---

## 阶段五：Planning 与 Multi-Agent（第 4 周 ~ 第 5 周前半，5-7 天）

### 学习目标

- 理解复杂任务分解与规划的方法
- 掌握 Multi-Agent 协作的架构模式
- 实现一个多 Agent 协作系统

### 学习内容

#### 5.1 Planning（任务规划）

- Plan-and-Execute 模式：先规划再执行
- 动态重规划：执行中发现问题时调整计划
- 任务分解策略：递归分解、并行/串行识别
- 计划表示方式：DAG、线性步骤、层次结构

#### 5.2 Multi-Agent 架构

- 单 Agent vs Multi-Agent 的选择时机
- 常见架构模式：
  - Router 模式（路由分发）
  - Supervisor 模式（中心调度）
  - Hierarchical 模式（层级管理）
  - Collaborative 模式（平等协作）
- Agent 间通信协议设计
- 共享上下文 vs 隔离上下文

#### 5.3 自我反思与纠错

- Reflection 机制：执行后检查结果
- Critic Agent：专门负责评估质量
- 自动重试与替代方案生成

#### 5.4 工作流编排

- 线性工作流（Sequential）
- 并行工作流（Parallel）
- 条件分支（Conditional）
- 循环工作流（Loop）

### 学习材料

| 类型 | 材料 | 说明 |
|------|------|------|
| 论文 | [HuggingGPT: Solving AI Tasks with ChatGPT and Friends](https://arxiv.org/abs/2303.17580) | 任务规划 + 模型调度 |
| 论文 | [MetaGPT: Meta Programming for Multi-Agent](https://arxiv.org/abs/2308.00352) | 多 Agent 协作框架 |
| 论文 | [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) | 微软多 Agent 对话框架 |
| 论文 | [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) | Agent 自我反思机制 |
| 论文 | [Plan-and-Solve Prompting](https://arxiv.org/abs/2305.04091) | 计划驱动的问题解决 |
| 课程 | [DeepLearning.AI - AI Agentic Design Patterns with AutoGen](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen/) | Multi-Agent 设计模式 |
| 课程 | [DeepLearning.AI - Multi AI Agent Systems with CrewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/) | CrewAI 多 Agent 实践 |
| 代码 | [LangGraph](https://github.com/langchain-ai/langgraph) | 图结构的 Agent 编排框架 |
| 代码 | [AutoGen](https://github.com/microsoft/autogen) | 微软多 Agent 框架源码 |
| 代码 | [CrewAI](https://github.com/crewAIInc/crewAI) | 角色扮演式多 Agent 框架 |
| 代码 | [MetaGPT](https://github.com/geekan/MetaGPT) | 多 Agent 软件公司模拟 |

### 阶段产出

- [ ] Planning Agent：能将复杂任务拆解为可执行子步骤
- [ ] 实现至少 2 种 Multi-Agent 架构模式（如 Router + Supervisor）
- [ ] 3 个专业 Agent：如 Coder Agent、Researcher Agent、Writer Agent
- [ ] Agent 间通信协议和上下文管理
- [ ] Demo：多 Agent 协作完成一个复杂业务场景（如"调研竞品并生成分析报告"）
- [ ] 架构设计文档：Multi-Agent 系统的架构图和设计决策

---

## 阶段六：生产化与业务落地（第 5 周后半 ~ 第 6 周，5-7 天）

### 学习目标

- 掌握 Agent 系统的工程化最佳实践
- 建立评估体系，量化 Agent 表现
- 完成一个可部署的 Agent 服务

### 学习内容

#### 6.1 可靠性工程

- 重试机制：指数退避、最大重试次数
- 降级策略：Tool 失败时的替代方案
- 超时控制：每一步的时间限制
- 幂等性：重复执行不产生副作用
- 输入校验与输出过滤

#### 6.2 可观测性（Observability）

- Trace 追踪：记录 Agent 每一步的决策过程
- 结构化日志：输入、输出、耗时、Token 用量
- 指标监控：成功率、平均步数、响应时间、成本
- Debug 工具：可视化 Agent 执行轨迹

#### 6.3 评估体系

- 评测集构建：人工标注 + 自动生成
- 评估维度：准确性、完整性、效率、安全性
- 自动化评估 Pipeline
- A/B 测试框架
- LLM-as-Judge（用 LLM 评估 LLM）

#### 6.4 安全与防护

- Prompt 注入攻击与防御
- 输出内容过滤（敏感信息、有害内容）
- 权限控制：Tool 的执行权限分级
- 成本限制：Token 用量上限、API 调用频率控制
- 用户输入清洗

#### 6.5 部署与服务化

- API 化封装（FastAPI/Flask）
- 流式输出（SSE/WebSocket）
- 异步执行与任务队列
- 容器化部署（Docker）
- 配置管理与环境隔离

### 学习材料

| 类型 | 材料 | 说明 |
|------|------|------|
| 工具 | [LangSmith](https://docs.smith.langchain.com/) | Agent 追踪和评估平台 |
| 工具 | [Phoenix by Arize](https://docs.arize.com/phoenix) | 开源 LLM 可观测性工具 |
| 工具 | [Langfuse](https://langfuse.com/docs) | 开源 LLM 工程平台，追踪+评估 |
| 文章 | [Building Reliable LLM Applications](https://huyenchip.com/2023/04/11/llm-engineering.html) | Chip Huyen 的 LLM 工程实践 |
| 文章 | [Patterns for Building LLM-based Systems](https://eugeneyan.com/writing/llm-patterns/) | LLM 系统设计模式总结 |
| 论文 | [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) | LLM 评估 LLM 的方法论 |
| 文章 | [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) | LLM 应用安全指南 |
| 官方文档 | [FastAPI 文档](https://fastapi.tiangolo.com/) | API 服务开发 |
| 代码 | [Dify](https://github.com/langgenius/dify) | 开源 LLM 应用开发平台，参考生产架构 |
| 课程 | [DeepLearning.AI - Quality and Safety for LLM Applications](https://www.deeplearning.ai/short-courses/quality-safety-llm-applications/) | LLM 应用质量与安全 |

### 阶段产出

- [ ] 完整的 Agent 框架代码库（含所有模块）
- [ ] Trace 系统：可视化查看每次 Agent 执行的完整轨迹
- [ ] 评估报告：在业务场景下的准确率、成功率、平均耗时
- [ ] API 服务：可部署的 HTTP 服务，支持流式输出
- [ ] 安全测试报告：Prompt 注入等攻击的防御验证
- [ ] 部署文档：从零部署的完整指南
- [ ] **最终答辩 PPT**：向团队展示框架设计、技术选型、业务效果

---

## 贯穿全程的要求

### 代码规范

- 类型标注（Type Hints）完整
- 每个模块有清晰的接口定义（Protocol/ABC）
- 关键路径有单元测试覆盖
- 遵循项目统一的代码风格（配置 ruff/black）

### 文档规范

- 每个阶段开始前先写设计文档，mentor 评审通过再编码
- 代码有必要的注释和 docstring
- README 说明如何运行和扩展

### 过程管理

- 每日站会简述进展和阻塞
- 每周输出周报：本周完成、遇到的问题、下周计划
- 每个 PR 需要 Code Review 通过后合并

### 对标阅读（贯穿全程）

建议在开发过程中持续对比阅读以下开源框架的源码：

| 框架 | 关注点 | 链接 |
|------|--------|------|
| LangChain | Tool 抽象、Chain 设计 | https://github.com/langchain-ai/langchain |
| LangGraph | 图编排、状态管理 | https://github.com/langchain-ai/langgraph |
| OpenAI Swarm | 极简 Agent 设计 | https://github.com/openai/swarm |
| AutoGen | 多 Agent 对话 | https://github.com/microsoft/autogen |
| CrewAI | 角色协作 | https://github.com/crewAIInc/crewAI |
| Dify | 生产级架构 | https://github.com/langgenius/dify |

---

## 推荐学习路线图（按优先级）

### 必读论文（按阅读顺序）

1. [ReAct](https://arxiv.org/abs/2210.03629) — Agent 的理论基础
2. [Toolformer](https://arxiv.org/abs/2302.04761) — 工具使用
3. [Generative Agents](https://arxiv.org/abs/2304.03442) — 记忆架构
4. [HuggingGPT](https://arxiv.org/abs/2303.17580) — 任务规划
5. [MetaGPT](https://arxiv.org/abs/2308.00352) — 多 Agent
6. [Reflexion](https://arxiv.org/abs/2303.11366) — 自我反思

### 必看博客

1. [Lilian Weng - LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)
2. [Chip Huyen - Building LLM Applications for Production](https://huyenchip.com/2023/04/11/llm-engineering.html)
3. [Eugene Yan - Patterns for Building LLM-based Systems](https://eugeneyan.com/writing/llm-patterns/)

### 推荐课程（DeepLearning.AI，均免费）

1. ChatGPT Prompt Engineering for Developers
2. Building Systems with ChatGPT API
3. Functions, Tools and Agents with LangChain
4. AI Agents in LangGraph
5. AI Agentic Design Patterns with AutoGen
6. Multi AI Agent Systems with CrewAI

---

## 最终框架目标架构

```
agent-framework/
├── core/
│   ├── agent.py           # Agent 基类和 Loop
│   ├── llm.py             # LLM 调用封装
│   └── config.py          # 配置管理
├── tools/
│   ├── base.py            # BaseTool 抽象
│   ├── registry.py        # ToolRegistry
│   ├── web_search.py      # 搜索工具
│   ├── code_executor.py   # 代码执行
│   └── ...
├── memory/
│   ├── manager.py         # MemoryManager
│   ├── short_term.py      # 短期记忆
│   ├── long_term.py       # 长期记忆（向量存储）
│   └── compressor.py      # 上下文压缩
├── planning/
│   ├── planner.py         # 任务规划器
│   └── executor.py        # 计划执行器
├── multi_agent/
│   ├── router.py          # 路由分发
│   ├── supervisor.py      # 中心调度
│   └── protocol.py        # Agent 间通信协议
├── observability/
│   ├── tracer.py          # 执行追踪
│   ├── logger.py          # 结构化日志
│   └── metrics.py         # 指标收集
├── safety/
│   ├── input_filter.py    # 输入过滤
│   ├── output_filter.py   # 输出过滤
│   └── rate_limiter.py    # 频率限制
├── api/
│   ├── server.py          # FastAPI 服务
│   └── schemas.py         # 请求/响应模型
├── evaluation/
│   ├── evaluator.py       # 评估器
│   ├── datasets/          # 评测数据集
│   └── reports/           # 评估报告
├── tests/                 # 单元测试
├── docs/                  # 设计文档
└── examples/              # 使用示例
```

---

## 评估标准（Mentor 评分参考）

| 维度 | 权重 | 说明 |
|------|------|------|
| 代码质量 | 25% | 结构清晰、可扩展、有测试 |
| 设计能力 | 25% | 架构合理、接口设计好、文档完整 |
| 学习深度 | 20% | 对论文和原理的理解程度 |
| 业务效果 | 20% | Agent 在实际场景中的表现 |
| 沟通协作 | 10% | 文档表达、Code Review、答辩表现 |
