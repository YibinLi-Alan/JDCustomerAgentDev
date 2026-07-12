# 京东客服 Agent 框架 · JDCustomerAgentDev

> 一个**从零手写、不依赖 LangChain/AutoGen** 的可复用 Agent 框架,以「京东客服」为落地业务,
> 完整覆盖 **LLM 接口 → ReAct 循环 → 工具系统 → 记忆 → 多 Agent 编排 → 生产化(可靠性 /
> 可观测 / 评估 / 安全 / HITL / 部署)** 六个阶段。核心理念一句话:
> **核心只认接口,加能力 = 加实现;全链路降级永不炸;凡是循环必有刹车。**

<p align="left">
  <img alt="python" src="https://img.shields.io/badge/python-3.11-blue">
  <img alt="tests" src="https://img.shields.io/badge/tests-225%20passing-brightgreen">
  <img alt="checks" src="https://img.shields.io/badge/ruff%20%7C%20black%20%7C%20mypy-passing-brightgreen">
  <img alt="version" src="https://img.shields.io/badge/version-0.6.0-informational">
  <img alt="stages" src="https://img.shields.io/badge/stages-6%2F6%20done-success">
</p>

---

## ✨ 这个项目是什么

一个 4–6 周的从零构建实习项目。目标不是"用现成框架搭个 demo",而是**逐层手写、真正理解
每个部件为什么这样设计**,最终产出一个模块化、可换业务换实现而不动核心的通用 Agent 框架。

落地业务是**京东客服 Agent**:用户提一个诉求,Agent 产出的不是单句回答,而是一套**解决方案**——
低权限动作(查订单/物流/商品)直接办并返回结果;高权限动作(退款/取消/建工单)**触发人工审批**;
Agent 彻底办不了的诉求带完整轨迹**升级人工队列**。这条「分级执行 + human-in-the-loop」的业务闭环,
正是本框架的招牌能力。

### 亮点

- 🧩 **九个正交子包,接口解耦**:LLM / 工具 / 记忆 / 向量库全部藏在 `Protocol`/`ABC` 后,换厂商、
  换工具、换存储只需加实现,核心一行不改。
- 🔁 **两种 Agent 循环**:`ToolCallingAgent`(原生 Function Calling,并行调用)+ `ReActAgent`
  (文本 JSON 解析,任何模型都能跑的兜底)。
- 🧠 **完整记忆系统**:短期滑窗 + 递归摘要 + 长期三因子检索(Generative Agents)+ Mem0 式增删改 +
  多用户存储层隔离 + Chroma 跨会话持久化。
- 🕹️ **多 Agent 控制系统**:三级分流(direct / 专员直派 / 中心调度)+ Planner 计划驱动派工 +
  动态重规划 + Critic 终稿质检回炉。
- 🛡️ **生产级外壳**:指数退避重试 + provider 降级 + 四层超时 + 工具幂等;结构化 Trace + 指标;
  六层注入纵深防御;FastAPI + SSE + Docker。
- 🙋 **HITL 业务闭环**:权限闸门拦截 → 挂起入队 → 人工审批 → **真正执行挂起动作(幂等重放安全)**。
- ✅ **质量可量化**:225 条离线单测(零 API 成本)+ LLM-as-Judge 端到端评测(76% 通过)+
  安全攻击集(9/9 防住);ruff / black / mypy 三项静态检查全过。

---

## 🏗️ 架构总览

```
   用户/前端 ─HTTP─► api/server.py (FastAPI:/chat · /chat/stream SSE · /approvals)
                              │
                    ╔═════════▼══════════════════════════════════════╗
                    ║  service.py · AgentService(整栈门面,唯一装配点) ║
                    ║  ① 入口安全   限流 + 注入检测                     ║
                    ║  ② 记忆装载   短期滑窗 + 前情摘要 + 三因子检索      ║──► Embedding / 向量库(Chroma)
                    ║  ③ 分诊       Router:direct / 专员 / supervisor  ║
                    ║  ④ 执行       专员(ToolCallingAgent)+ 计划驱动   ║──► 京东业务数据(JDMockStore→真 API)
                    ║               工具子集 = 审批闸门∘边界标记∘Registry║──► 人工审批队列(HandoffQueue)
                    ║  ⑤ 出口安全   敏感信息脱敏                        ║
                    ║  ⑥ 记忆落账   提炼 → 增删改裁决 → 写库             ║
                    ╚═════════╤══════════════════════════════════════╝
                              │ 全程旁路 on_event
                    observability/Tracer → JSONL / metrics / SSE 推送  ──► (可对接 Langfuse)
   所有 LLM 调用 ↓ 经 ReliableLLM 重试 → FallbackLLM 降级 ──────────────► Claude / OpenAI
```

> 完整版(含 8 个外部数据接入点标注、九子包拆解、三专员人设、控制系统、数据流时序)见
> **[`docs/架构框架图.md`](docs/架构框架图.md)**。

### 目录结构

```
agent_framework/
├── core/            # LLM 接口 + 两个 Agent 循环 + 可靠性包装 + 配置
├── tools/           # BaseTool(ABC) + ToolRegistry + @tool + 11 个工具 + JDMockStore
├── memory/          # 短期滑窗 / 递归摘要 / 长期三因子检索 / 向量库 / MemoryManager
├── planning/        # Planner(线性计划 + 重规划) + PlanExecutor + ScratchPad
├── multi_agent/     # 三业务专员 + Router + Supervisor + Critic + 通信协议
├── observability/   # Tracer(JSONL) + 结构化日志 + 指标聚合
├── safety/          # 输入/输出过滤 + 限流 + HITL(审批闸门 + 人工队列)
├── evaluation/      # LLM-as-Judge + 端到端评测 + 安全攻击集
├── api/             # FastAPI(/chat + SSE + /approvals)
└── service.py       # AgentService 整栈门面(CLI/评测/API 共用)
examples/            # 7 个 CLI 演示(见下)
tests/               # 225 条离线单测(MockLLM / MockEmbedder)
docs/                # 设计文档 / 架构图 / 设计论文 / 测试手册 / 评测报告 / 部署文档
```

---

## 🚀 快速开始

### 1. 环境与依赖(Python 3.11)

```bash
conda activate jingdong          # 或任意 Python 3.11 venv
pip install -r requirements.txt
pip install -r requirements-dev.txt   # 可选:ruff / black / mypy / pytest
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env:PROVIDER=openai|claude,填对应的 OPENAI_API_KEY / ANTHROPIC_API_KEY
```

`.env` 已 gitignore,代码任何地方不出现明文 key。换厂商只改 `PROVIDER` 一行。

### 3. 跑最能看全貌的 Demo:多 Agent 协作客服

```bash
python -m examples.multi_agent_cli
```

试三条(trace 默认开,能看到分诊/计划/派工/审批全轨迹):

- `订单 12345 到哪了` —— **快路径**:分诊直派订单专员;
- `我买的耳机订单 11111 坏了,查还能不能退,能退帮我退,再推荐个替代品` —— **复合客诉一条龙**:
  计划 → 查证 → 退款被 7 天时效拦下 → 重规划建工单转人工 → 推荐 → 质检;
- `你好呀` —— **direct 出口**:寒暄不进专员流水线。

### 4. 起 HTTP 服务(FastAPI + SSE + 审批接口)

```bash
uvicorn agent_framework.api.server:_factory --factory --port 8000
```

打开 http://localhost:8000/docs 看自动生成的交互式 API 文档。审批闭环示例:

```bash
# 高权限退款被拦,返回审批单号
curl -X POST localhost:8000/chat -H 'Content-Type: application/json' \
     -d '{"user_id":"me","message":"订单 12345 有质量问题帮我退款"}'
# 人工放行 → 真正执行退款(request_id 幂等)
curl "localhost:8000/approvals?status=pending"
curl -X POST localhost:8000/approvals/<id>/approve \
     -H 'Content-Type: application/json' -d '{"note":"核实属实"}'
```

---

## 🎬 全部演示入口

| 命令 | 阶段 | 演示什么 |
|---|---|---|
| `python -m examples.chat_cli` | 一 | 多轮对话 + 流式 + Prompt 技巧 |
| `python -m examples.react_cli` | 二 | ReAct 循环(`/trace` 看 Thought/Action/Observation) |
| `python -m examples.tools_cli` | 三 | 11 个工具 + 原生 Function Calling(`/tools` 看权限) |
| `python -m examples.memory_cli` | 四 | 带记忆多用户(`/user` 切换 · 跨会话 · 隔离) |
| `python -m examples.multi_agent_cli` | 五 | 多 Agent 协作(`/mode` · `/trace`) |
| `python -m examples.trace_viewer` | 六 | Trace 时间线 + 成功率/步数/token 指标仪表盘 |
| `python -m examples.approval_cli` | 六 | 人工审批控制台(list / approve / reject) |

---

## 🧩 核心设计(为什么这样设计)

| 原则 | 体现 |
|---|---|
| **核心只认接口,加能力 = 加实现** | LLM / Embedder / VectorStore / ToolRegistryLike 全是 Protocol;换厂商/工具/存储只加实现 |
| **全链路降级,永不炸对话** | 工具失败折叠为数据喂回、记忆压缩/裁决降级、分诊/汇总/裁判失败都有兜底 |
| **凡是循环必有刹车** | Agent 步数上限 / 重规划 ≤1 / Critic 回炉 ≤1 / LLM 重试 ≤3 / 整任务超时 / token 预算 |
| **结构化优于自由文本** | JSON 输出、标准化 ToolResult、通信协议、结构化 Trace/日志 |
| **核心零改动** | 记忆在装配层、专员复用 ToolCallingAgent、安全用装饰器;全程只加一个 `on_event` 只读钩子 |
| **最小权限是最后一道墙** | 注入防不胜防,但高权限操作被审批闸门锁死,被骗也难成灾 |

框架依赖接口而非厂商,当前用 mock/便宜实现,**生产替换只换实现**:

| 外部依赖 | 当前 | 生产替换 |
|---|---|---|
| LLM | Claude / OpenAI(gpt-5.4-mini) | 改 `.env` 的 `PROVIDER` |
| 业务数据 | `JDMockStore` 内存 mock | 京东真实订单/商品/物流 API(11 个工具零改动) |
| 向量库 | Chroma(持久化)/ 内存版 | Milvus / pgvector 等(实现 `VectorStore`) |
| 人工队列 | JSON 文件 | 数据库 / 工单系统 |
| 可观测 | JSONL + 终端 viewer | Langfuse / Phoenix(给 Tracer 加 listener) |

---

## ✅ 测试与质量

```bash
python -m pytest tests/ -q      # 225 条离线单测(MockLLM,零 API 成本)
ruff check .                    # 代码规范
black --check .                 # 格式
mypy agent_framework/           # 静态类型检查
```

四项全绿是「做完」的判定。真实 API 评测(会花钱):

```bash
python -m agent_framework.evaluation.agent_eval        # 端到端 25 例 → LLM-as-Judge 打分(76% 通过)
python -m agent_framework.evaluation.security_eval     # 9 例注入/越权攻击(9/9 防住)
python -m agent_framework.evaluation.memory_retrieval_eval   # 记忆检索策略对比(离线)
```

> 每个阶段测什么、覆盖点、精确复现命令见 **[`docs/测试复现手册.md`](docs/测试复现手册.md)**。

---

## 📦 部署(Docker)

```bash
docker build -t jd-agent .
docker run -p 8000:8000 --env-file .env -v "$(pwd)/data:/app/data" jd-agent
```

`.env` 外部注入(不进镜像)、`data/` 挂卷(记忆/轨迹/审批队列持久化)。完整指南见
**[`docs/deployment.md`](docs/deployment.md)**。

---

## 📚 文档地图

| 文档 | 内容 |
|---|---|
| [`docs/设计论文.md`](docs/设计论文.md) | 六阶段每个问题的思考/设计/依据论文,逐条回答培训大纲 |
| [`docs/架构框架图.md`](docs/架构框架图.md) | 系统全景图 + 九子包 + 三专员人设 + 外部接口 + 数据流 |
| [`docs/测试复现手册.md`](docs/测试复现手册.md) | 每阶段测什么 + 精确复现命令 + 速查表 |
| [`docs/stage-1~6-design.md`](docs/) | 六份阶段设计文档(编码前评审) |
| [`docs/stage-6-eval-report.md`](docs/stage-6-eval-report.md) · [`security-report.md`](docs/stage-6-security-report.md) | 评测报告 + 安全测试报告(含诚实局限声明) |
| [`ROADMAP.md`](ROADMAP.md) | 六阶段路线图与产出清单 |

---

## 🗺️ 六阶段一览

| 阶段 | 主题 | 核心产出 |
|---|---|---|
| 一 | 基础认知 | LLM `Protocol` + 可切换 provider + 配置 + Prompt 实验 |
| 二 | 最小 Agent 循环 | ReAct(Thought→Action→Observation)+ 步数守卫 + 错误恢复 |
| 三 | Tool Use 系统 | `BaseTool` + `ToolRegistry` + 原生 Function Calling + 11 工具 |
| 四 | Memory | 短期滑窗 + 递归摘要 + 长期三因子检索 + 多用户隔离(评测调优权重) |
| 五 | Planning & Multi-Agent | Router + Supervisor + Planner + 三专员 + Critic |
| 六 | 生产化 | 可靠性 / 可观测 / 评估 / 安全 / **HITL 闭环** / FastAPI + Docker |

---

## 🛠️ 技术栈

Python 3.11 · Pydantic / pydantic-settings · anthropic / openai SDK(接口后隔离) ·
ChromaDB(向量库,可替换)· FastAPI + uvicorn(服务化)· pytest / ruff / black / mypy(质量) ·
零重量级 Agent 框架依赖(LangChain/AutoGen 仅作对比阅读,不引入)。

## 📖 对比阅读(每一块设计都有出处)

LangChain(工具/记忆)· OpenAI Swarm(handoff/context_variables)· Generative Agents(三因子检索)·
MemGPT(上下文分区)· Mem0(增删改写入)· LangGraph / AutoGen / MetaGPT / Reflexion(多 Agent 编排)·
Chip Huyen / Eugene Yan / OWASP LLM / Judging LLM-as-a-Judge(生产化)。详见 `docs/设计论文.md`
与知识库对比阅读笔记。

---

## 📌 说明

- 仓库:https://github.com/YibinLi-Alan/JDCustomerAgentDev
- 这是学习性质的实习项目,准则是**功能完整 > 工业级打磨**——大纲要求的每一项都真实存在、能跑、
  能演示;评测报告如实记录局限,不假装无懈可击。
- `AgentKnowledge/`(Obsidian 知识库)、`data/`(运行时数据)、`.env`(密钥)均不进版本库。
