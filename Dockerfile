# 阶段六 · 容器化(解决「在我电脑上明明能跑」)。
# 三步部署:装 Docker → docker build → docker run(见 docs/deployment.md)。
FROM python:3.11-slim

WORKDIR /app

# 先装依赖(利用镜像层缓存:requirements 不变则不重装)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再拷代码(AgentKnowledge/ 与 data/ 由 .dockerignore 排除,不进镜像)
COPY agent_framework/ ./agent_framework/

# 有状态数据挂卷:Chroma 记忆 + Trace + 审批队列,重启不丢(docs/deployment.md)
VOLUME ["/app/data"]

EXPOSE 8000

# .env 不打进镜像(密钥外部注入:docker run --env-file .env ...)
# uvicorn --factory:延迟装配,避免 import 期就要求 key
CMD ["uvicorn", "agent_framework.api.server:_factory", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
