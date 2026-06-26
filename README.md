# Agent 框架(实习项目)

从零搭建一个可复用的 Agent 框架。学习与交付顺序见 `ROADMAP.md`,
权威大纲见 `frame/Agent框架实习培训大纲.md`,各阶段设计文档见 `docs/`。

代码包:`agent_framework/`(纯库)。使用示例:`examples/`。

## 阶段一:如何运行

阶段一交付:LLM 接口(`agent_framework/core/llm.py`)、配置
(`agent_framework/core/config.py`)、多轮对话 CLI、Prompt 三策略对比实验。
设计依据:`docs/stage-1-design.md`。

### 1. 准备环境(conda `jingdong`,Python 3.11)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jingdong
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
# 开发(ruff / black)可选:
pip install -r requirements-dev.txt
```

### 3. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env,把 ANTHROPIC_API_KEY 改成你的真实 key
```

`.env` 已在 `.gitignore` 中,不会被提交;代码任何地方都不出现明文 key。

### 4. 运行多轮对话 CLI

```bash
python -m examples.chat_cli
```

CLI 命令:`/help`、`/exit`(或 `/quit`)、`/reset`、`/system <文本>`、`/stream`。
默认是否流式由 `.env` 的 `STREAM` 决定;非流式会显示 token 用量。

### 5. 运行 Prompt 对比实验(需真实 key)

```bash
python -m examples.prompt_lab
```

会跑「直接提问 / CoT / Few-shot」三策略对比,并把结果写入
`docs/stage-1-prompt-experiment.md`。

## 设计要点

- **可拓展**:核心只依赖 `LLM`(`typing.Protocol`)接口,`ClaudeLLM` 是其背后的一个实现。
  换模型 = 改 `.env` 的 `MODEL`(同厂商)或新增一个 `LLM` 实现 + 改装配点一行(跨厂商)。
- **配置集中**:密钥/模型/采样参数统一经 `agent_framework/core/config.py` 从 `.env` 读取。
- **厂商隔离**:`anthropic` SDK 只出现在 `agent_framework/core/llm.py`,上层代码绝不直接 import 它。
