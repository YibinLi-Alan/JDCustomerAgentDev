# 部署文档 · 从零部署指南(阶段六产出⑥)

京东客服 Agent 框架的 HTTP 服务部署。两条路径:**本地直跑**(开发)与
**Docker**(交付/生产一致环境)。

## 0. 前置

- Python 3.11+(本地直跑)或 Docker(容器部署,推荐);
- 一个 LLM provider 的 API key:`OPENAI_API_KEY`(用 OpenAI)或 `ANTHROPIC_API_KEY`
  (用 Claude);长期记忆额外需要 `OPENAI_API_KEY`(embedding)。

## 1. 配置(环境隔离,从第一天沿用)

所有配置走环境变量 / `.env`,**代码零改动切环境**(dev/prod 给不同 env 即可)。
复制 `.env.example` 为 `.env`,填:

```bash
PROVIDER=openai            # openai | claude
OPENAI_API_KEY=sk-...      # 对应 provider 的 key
MODEL=gpt-5.4-mini         # 可选,留空用 provider 默认
# 生产化护栏(全部可选,默认见 core/config.py)
LLM_MAX_RETRIES=3
TASK_DEADLINE_SECONDS=300
RATE_LIMIT_PER_MINUTE=20
FALLBACK_PROVIDER=          # 填另一家可开启 provider 降级
```

> `.env` 已在 `.gitignore` 与 `.dockerignore` 中——**密钥绝不进仓库、绝不进镜像**。

## 2. 本地直跑(开发)

```bash
pip install -r requirements.txt
uvicorn agent_framework.api.server:_factory --factory --reload --port 8000
```

打开 http://localhost:8000/docs —— FastAPI 自动生成的交互式 API 文档
(基于类型标注,免写)。

## 3. Docker(三步,环境一致)

```bash
# ① 装 Docker(略,见 docker.com)
# ② 构建镜像
docker build -t jd-agent .
# ③ 运行(key 外部注入,data 挂卷持久化)
docker run -p 8000:8000 \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  jd-agent
```

要点:
- **`--env-file .env`**:密钥运行时注入,不在镜像里;
- **`-v .../data:/app/data`**:Chroma 记忆、Trace、审批队列落在宿主机 `data/`,
  **容器重启不丢**(阶段四 backlog 在此兑现);
- 镜像不含 `AgentKnowledge/`、`data/`、`.env`、测试(见 `.dockerignore`)。

## 4. 接口速查

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/health` | 存活探针 |
| POST | `/chat` | 一问一答(非流式) |
| POST | `/chat/stream` | **SSE 流式**:中间步骤(分诊/计划/派工/审批)实时推送 + 最终答复 |
| GET | `/approvals?status=pending` | 列出人工介入单 |
| POST | `/approvals/{id}/approve` | 放行:真正执行挂起的高权限动作(幂等) |
| POST | `/approvals/{id}/reject` | 驳回 |

示例:

```bash
curl -X POST localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"alice","message":"订单 12345 到哪了"}'

# 流式(看中间步骤):
curl -N -X POST localhost:8000/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"alice","message":"耳机坏了,订单 11111,查下能不能退,能退帮我退,再推荐个替代品"}'
```

人工审批闭环(高权限操作被拦后):

```bash
curl localhost:8000/approvals?status=pending          # 拿到 <id>
curl -X POST localhost:8000/approvals/<id>/approve -H 'Content-Type: application/json' -d '{"note":"核实属实"}'
# 或用命令行控制台:python -m examples.approval_cli
```

## 5. 可观测

- 每次请求生成一条 Trace(`data/traces/<task_id>.jsonl`);
- 查看:`python -m examples.trace_viewer`(汇总表)/ `... <task_id>`(单任务时间线);
- 指标(成功率/步数/token/耗时/人工介入率)从 traces 聚合,同一份数据进评估报告。

## 6. 生产化局限(如实,超出本项目范围)

- **限流/token 预算是进程内**:多副本部署需外置 Redis 等共享存储(接口不变,换实现);
- **审批队列是 JSON 文件**:高并发/多副本需换数据库;
- **无任务队列**:几十秒级任务靠 SSE 中间步骤兜底体验;分钟级长任务需接 Celery 等
  (大纲减法,说得出概念即可);
- **可观测未接平台**:自建极简 Trace 已满足;生产可对接 Langfuse/Phoenix
  (= 给 Tracer 加一个 listener,不改核心)。
