# 阶段一设计文档 · 基础认知与环境搭建

> 状态：**设计稿（待 mentor / Tech Lead 评审）** · 角色：Architect
> 范围：仅本阶段（`agent_framework/core/` 的 `llm.py`、`config.py` + 工程骨架）
> 配套源头：`frame/Agent框架实习培训大纲.md`「阶段一」、`CLAUDE.md`
> 本文档**只做设计**，不含可运行的业务实现（接口签名 / 类型定义 / 伪代码除外）。

---

## 0. 设计最高准则（贯穿全文）

1. **功能完整 > 工业级打磨**：大纲阶段一列出的三项产出都要真实落地、可演示，不追花哨。
2. **可拓展性是硬要求**：会变的东西（首当其冲是 LLM 厂商）一律藏在接口后。`core/llm.py` 先定义 `LLM`（`typing.Protocol`），Claude 只是它背后的一个实现 `ClaudeLLM`。换模型 = 换实现，核心不动。
3. **配置集中**：密钥、模型名、采样参数等统一从 `.env` 经 `config.py` 读取，绝不硬编码进代码。
4. **业务场景（京东客服 Agent）本阶段不展开**：阶段一只验证「能多轮对话 + Prompt 技巧对比」，点到为止。

---

## 1. 阶段一目标与范围

### 1.1 做什么（对照大纲「学习内容」）

| 大纲条目 | 本阶段如何覆盖 |
|---|---|
| 1.1 LLM API 基础（messages/role/temperature/max_tokens、streaming、token/成本） | `LLM` 接口的 `chat` / `stream` 方法，`ChatResponse` 暴露 token 用量，CLI 支持流式打印 |
| 1.2 Prompt Engineering（System Prompt / Few-shot / CoT / 结构化输出） | CLI 支持运行时设定 system prompt；独立实验脚本对比「直接提问 vs CoT vs Few-shot」 |
| 1.3 开发环境（Python 3.10+、依赖管理、`.env`、项目结构） | conda 环境 `jingdong`(3.11) + `pyproject.toml` + `requirements.txt` + `.env.example` + 骨架目录 |

### 1.2 明确不做（防止范围蔓延）

- **不做** ReAct 循环 / Tool Use / Memory / Planning / Multi-Agent —— 那是阶段二及以后。
- **不做** `agent.py` 的实质实现（本阶段仅占位，确立它将依赖 `LLM` 接口而非具体厂商）。
- **不做** FastAPI 服务、向量库、评估体系、可观测性。
- **不做** 京东客服业务工具与权限分级（阶段五/六落地）。
- **不做** 多厂商同时支持 —— 但**接口必须为之预留**，本阶段只实现 `ClaudeLLM` 一个。

---

## 2. 交付物清单（逐条对照大纲「阶段产出」）

| # | 大纲产出 | 本阶段交付物 | 落点 |
|---|---|---|---|
| ① | 能调用 LLM API 完成多轮对话的 CLI 程序 | 多轮对话 CLI，支持 `/exit` `/reset` `/system` 等命令、可选流式输出、token 统计 | `examples/chat_cli.py` |
| ② | 实验报告：直接提问 vs CoT vs Few-shot 对输出质量的影响 | 一个可复跑的实验脚本 + 一份 Markdown 报告 | `examples/prompt_lab.py` + `docs/stage-1-prompt-experiment.md` |
| ③ | 项目基础骨架搭建完成 | `agent_framework` 包 + 工程化配置（pyproject / requirements / .env.example）+ README 运行说明 | 仓库根 + `agent_framework/` |

> 三项产出全部依赖同一个 `core/llm.py` + `core/config.py`，这是本阶段的代码重心。

---

## 3. 目录与文件规划

阶段一新增 / 修改文件（骨架空目录已存在，只填本阶段需要的部分）：

```
JD/
├── agent_framework/
│   ├── __init__.py                # 包入口，导出 LLM / Message / ChatResponse / Settings
│   └── core/
│       ├── __init__.py
│       ├── llm.py                 # ★ Message / ChatResponse / Usage / LLM(Protocol) / ClaudeLLM
│       ├── config.py              # ★ Settings(pydantic-settings)，从 .env 读取
│       └── agent.py               # 占位：留 TODO，声明将依赖 LLM 接口（阶段二实现）
├── examples/
│   ├── chat_cli.py                # ★ 交付物① 多轮对话 CLI 入口
│   └── prompt_lab.py              # ★ 交付物② Prompt 对比实验脚本
├── docs/
│   ├── stage-1-design.md          # 本文档
│   └── stage-1-prompt-experiment.md  # ★ 交付物② 实验报告
├── .env.example                   # ★ 密钥/配置模板（进版本库）
├── .env                           # 实际密钥（.gitignore，不进库）
├── .gitignore                     # 确保含 .env
├── requirements.txt               # ★ 依赖清单（pip 安装）
├── pyproject.toml                 # ★ 项目元信息 + ruff/black 配置
└── README.md                      # 运行与扩展说明（追加阶段一章节）
```

**CLI 入口落点说明**：阶段一把 CLI 放在 `examples/`，作为框架的「使用示例」，符合大纲目标架构里 `examples/` 的定位；框架核心 `agent_framework/` 保持纯库、不含可执行脚本。运行方式 `python -m examples.chat_cli` 或 `python examples/chat_cli.py`。后续若要正式命令行工具，再在 `pyproject.toml` 配 `console_scripts`（本阶段不强制）。

---

## 4. LLM 接口设计（核心）

### 4.1 设计立场

Anthropic 的请求结构里 **system 是顶层参数**、**messages 仅在 user/assistant 间交替**。为了既贴合主流 Chat API、又不被某一家绑死，本框架的通用约定：

- `Message` 只承载 `user` / `assistant` 两种角色；
- system prompt 作为 `chat()` / `stream()` 的**独立可选参数**传入（而非塞进 messages），这样映射到 Anthropic 的 `system=` 顶层字段最自然，映射到 OpenAI 的 `{"role":"system"}` 也只是实现层一行转换。

### 4.2 数据结构

```python
# agent_framework/core/llm.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Protocol, Iterator

Role = Literal["user", "assistant"]

@dataclass(frozen=True)
class Message:
    """一条对话消息（通用，不绑定任何厂商）。"""
    role: Role
    content: str

@dataclass(frozen=True)
class Usage:
    """token 用量，用于成本估算与日志。"""
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

@dataclass
class ChatResponse:
    """一次非流式应答的统一返回。"""
    content: str                 # 拼接后的纯文本回复
    usage: Usage                 # token 用量
    model: str                   # 实际使用的模型 id
    stop_reason: str | None = None
    raw: object | None = field(default=None, repr=False)  # 逃生舱：原始 SDK 响应，调试用
```

### 4.3 `LLM` Protocol

```python
class LLM(Protocol):
    """所有 LLM 厂商实现都要满足的接口。核心只依赖它，不依赖任何具体 SDK。"""

    def chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
    ) -> ChatResponse:
        """一次性返回完整应答。"""
        ...

    def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
    ) -> Iterator[str]:
        """流式返回文本增量（一段段 yield）。用于 CLI 实时打印。"""
        ...
```

> 设计要点：方法签名里**不出现** `temperature`、`max_tokens`、`model` 等参数 —— 这些属于「实现 + 配置」职责，由 `ClaudeLLM.__init__` 从 `Settings` 注入。接口只描述「给消息、拿回复」这件事本身，保证不同厂商签名一致。流式接口本阶段约定为「只 yield 文本增量」，token 用量在流结束后可由实现侧补记（见 4.4）；阶段一不强制流式回传 `Usage`，保持最简。

### 4.4 `ClaudeLLM` 实现要点

```python
import anthropic
from agent_framework.core.config import Settings

class ClaudeLLM:                       # 结构化满足 LLM Protocol，无需显式继承
    def __init__(self, settings: Settings) -> None:
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.model              # 默认 "claude-opus-4-8"
        self._max_tokens = settings.max_tokens
        self._temperature = settings.temperature  # 见下方「温度参数」注意事项

    def chat(self, messages, *, system=None) -> ChatResponse:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system or anthropic.NOT_GIVEN,          # None 时不传
            messages=[{"role": m.role, "content": m.content} for m in messages],
            # temperature: 见注意事项 —— Opus 4.8 不接受，仅在模型支持时下发
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return ChatResponse(
            content=text,
            usage=Usage(resp.usage.input_tokens, resp.usage.output_tokens),
            model=resp.model,
            stop_reason=resp.stop_reason,
            raw=resp,
        )

    def stream(self, messages, *, system=None):
        with self._client.messages.stream(
            model=self._model, max_tokens=self._max_tokens,
            system=system or anthropic.NOT_GIVEN,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        ) as stream:
            yield from stream.text_stream
```

映射约定：
- 通用 `Message(role, content)` → Anthropic 的 `{"role", "content"}`（role 已限定为 user/assistant，天然合法交替由 CLI 维护）。
- 通用 `system` 参数 → Anthropic 顶层 `system=`。
- 通用 `ChatResponse` ← Anthropic 响应：文本取 `content` 里 `type=="text"` 的块拼接；token 取 `usage.input_tokens / output_tokens`。
- `model` / `max_tokens` / 默认温度 从 `Settings` 注入，构造时一次性绑定。

> **⚠️ 关键技术事实（务必在评审时确认）：所选模型 `claude-opus-4-8` 不接受 `temperature`/`top_p`/`top_k`，发送会返回 400**；它用「自适应思考 + effort」取代固定采样/思考预算。因此：
> - 若坚持用 Opus 4.8，`ClaudeLLM` 应**不下发 `temperature`**（`Settings.temperature` 仍保留为通用配置字段，供其他模型/厂商使用）；可选地下发 `output_config={"effort": ...}` 控制深浅，本阶段建议先不加，保持最简。
> - 若实验报告（交付物②）需要「调温度看输出差异」，则需改用支持温度的模型（如 `claude-sonnet-4-5` / `claude-opus-4-5`）。这是「用哪个具体模型」未决问题的核心，见 §11。
> - 实现层应做到：**只把目标模型接受的参数下发**（一个小的「按模型能力过滤参数」逻辑），避免换模型时踩 400。

### 4.5 为什么这样设计能「换模型不动核心」

- 核心代码（未来的 `agent.py`、CLI、实验脚本）**只 import `LLM` / `Message` / `ChatResponse`**，从不 import `anthropic`。
- 新增一家厂商 = 新写一个满足 `LLM` Protocol 的类（如 `OpenAILLM`），在它内部做自己的消息映射与参数过滤；核心一行都不改。
- 厂商差异（system 位置、采样参数是否支持、token 字段名）全部被关在各自实现里。`ChatResponse` 是统一出口，上层看到的永远是同一种返回。
- 选择哪个实现由「装配点」决定（CLI / 实验脚本启动时 `llm = ClaudeLLM(settings)`），后续可做成由 `settings.provider` 决定的简单工厂（本阶段不必，但接口已为它留好）。

---

## 5. 配置管理设计（`config.py`）

用 `pydantic-settings` 的 `BaseSettings` 从 `.env` 自动读取，带类型校验与默认值。

```python
# agent_framework/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str                     # 必填，无默认 —— 缺失即启动报错
    model: str = "claude-opus-4-8"             # 默认模型，可被 .env 覆盖
    max_tokens: int = 1024                     # 单次回复上限
    temperature: float = 1.0                   # 通用字段；Opus 4.8 实际不下发（见 §4.4）
    stream: bool = True                        # CLI 默认是否流式
    # 可选：provider: str = "claude"           # 为未来多厂商工厂预留，本阶段可不启用

def get_settings() -> Settings:
    """单例式获取配置，供 CLI / 实验脚本调用。"""
    return Settings()  # type: ignore[call-arg]
```

`.env.example`（进版本库，作模板）字段：

```dotenv
# 必填：Anthropic API Key（真实值写到 .env，勿提交）
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
# 可选：覆盖默认值
MODEL=claude-opus-4-8
MAX_TOKENS=1024
TEMPERATURE=1.0
STREAM=true
```

> 安全：`.env` 必须在 `.gitignore` 中；代码任何地方都不出现明文 key；`Settings` 缺 `ANTHROPIC_API_KEY` 时启动即抛错，避免「跑到一半才发现没配 key」。

---

## 6. 多轮对话 CLI 设计（交付物①）

### 6.1 职责

一个最朴素但完整的命令行聊天循环：维护 `messages` 历史（这就是「多轮记忆」最原始的形态 —— 全量历史拼进每次请求），支持几条管理命令，支持流式打印与基础错误处理。

### 6.2 支持的命令

| 命令 | 行为 |
|---|---|
| 普通文本 | 作为 user 消息追加进历史，请求模型，打印回复并把回复追加进历史 |
| `/exit`（或 `/quit`） | 退出程序 |
| `/reset` | 清空对话历史（system prompt 保留） |
| `/system <文本>` | 设定 / 覆盖当前 system prompt，并自动 `/reset` 历史（system 变更后旧上下文语义可能错位） |
| `/stream` | 切换流式 / 非流式打印 |
| `/help` | 打印命令列表 |

### 6.3 主循环（伪代码）

```
settings = get_settings()
llm: LLM = ClaudeLLM(settings)        # 唯一装配点；换模型只改这一行
history: list[Message] = []
system: str | None = None
stream_on = settings.stream

print 欢迎语 + /help 提示
loop:
    line = input("you> ").strip()
    if 空行: continue
    if line 以 "/" 开头:
        分发到命令处理（/exit /reset /system /stream /help）
        continue
    history.append(Message("user", line))
    try:
        if stream_on:
            print("assistant> ", end="")
            chunks = []
            for delta in llm.stream(history, system=system):
                print(delta, end="", flush=True); chunks.append(delta)
            reply = "".join(chunks); print()
            # 流式下 token 用量本阶段可不展示（接口未回传 Usage）
        else:
            resp = llm.chat(history, system=system)
            reply = resp.content
            print("assistant>", reply)
            print(f"[tokens in={resp.usage.input_tokens} out={resp.usage.output_tokens}]")
        history.append(Message("assistant", reply))
    except anthropic.APIError as e:        # 见 6.4
        print("⚠️ 调用失败：", e); history.pop()   # 回滚刚加的 user 消息，避免脏历史
```

### 6.4 最简错误处理（阶段一够用即可）

- 捕获 `anthropic.APIError` 家族（401 鉴权 / 429 限流 / 5xx 服务端）打印友好提示，**不崩溃**；请求失败时把刚追加的 user 消息回滚，保持历史干净。
- `KeyboardInterrupt` / `EOF`（Ctrl-C / Ctrl-D）视作退出。
- 不在阶段一做重试 / 退避（SDK 默认已对 429/5xx 重试 2 次）；不做并发。

---

## 7. Prompt Engineering 实验设计（交付物②）

### 7.1 目标

用同一个 `LLM` 接口、同一个模型，跑同一批任务，对比三种提问策略对输出质量的影响：**直接提问 / Chain-of-Thought / Few-shot**。

### 7.2 任务选择

选「直接答易错、过程化能纠偏」的题型，让差异看得见。建议各取 2–3 题：

| 任务类型 | 示例 | 为什么能体现差异 |
|---|---|---|
| 多步算术 / 逻辑推理 | 「一个数加 17 后乘 3 等于 60，求这个数」 | 直接问易跳步出错；CoT 显式分步更稳 |
| 分类 / 抽取（结构化输出） | 给一段京东风格的用户反馈，要求输出 `{情绪, 问题类型}` JSON | Few-shot 给范例能显著稳定输出格式 |
| 简单常识陷阱题 | 经典「单位换算 / 顺序」陷阱 | 对比三策略的鲁棒性 |

### 7.3 三种策略如何构造

- **直接提问**：`system=None`，messages 只有题面。
- **CoT**：在 system 或 user 里加「请一步一步思考再给最终答案」，引导显式推理。
- **Few-shot**：在 user 内容前置 2–3 个「输入→输出」范例，再给真正的题。

> 三种策略共用 `ClaudeLLM.chat()`，仅 prompt 不同 —— 正好验证接口的复用性。

### 7.4 脚本与报告产出

- `examples/prompt_lab.py`：遍历 `任务 × 策略`，逐个调用 `chat()`，把「策略 / 题面 / 模型回复 / token 用量」收集起来，打印成表并可写盘。
- `docs/stage-1-prompt-experiment.md`：实验报告，含 ①实验设置（模型、参数、任务清单）②三策略逐题输出对比表 ③简短结论（哪种策略在哪类任务更优、token 成本权衡）。报告放 `docs/` 与本设计文档同级。

> 注意：若用 Opus 4.8（不支持 temperature），实验固定不调温度，对比维度就是「prompt 策略」本身；若想引入「温度」变量，需换用支持温度的模型（见 §11 未决问题）。

---

## 8. 工程化配置

### 8.1 `pyproject.toml` 放什么

- **项目元信息**：`[project]` name=`agent-framework`、version、`requires-python = ">=3.10"`、描述、作者。
- **包发现**：声明 `agent_framework` 为包（setuptools 或 hatchling 均可，阶段一用 setuptools 最省事）。
- **ruff 配置**：`[tool.ruff]` 选 `line-length=100`、启用 `E,F,I`（pyflakes + isort）等基础规则；`target-version = "py311"`。
- **black 配置**：`[tool.black]` `line-length=100`、`target-version=["py311"]`（与 ruff 行宽一致，避免互相打架）。

### 8.2 依赖清单（`requirements.txt`，pip 管理）

大纲允许 Poetry 或 pip+requirements；本项目选 **pip + requirements.txt**（更轻、团队上手快）。

| 依赖 | 用途 |
|---|---|
| `anthropic` | 官方 SDK，`ClaudeLLM` 的底座 |
| `pydantic` | 数据模型基础（pydantic-settings 依赖） |
| `pydantic-settings` | `Settings` 从 `.env` 读配置 |
| `python-dotenv` | `.env` 加载（pydantic-settings 已集成，列出以示意/兜底） |
| `ruff`（dev） | lint |
| `black`（dev） | 格式化 |

> 版本策略：阶段一不锁死小版本，先用合理下限（如 `anthropic>=0.40`）；待全项目稳定后再考虑 lock。dev 依赖可单列或放 `requirements-dev.txt`。

---

## 9. 可拓展性如何体现（为后续阶段留的扩展点）

| 阶段一的设计 | 为后续哪一步留口子 |
|---|---|
| `LLM` Protocol + `ClaudeLLM` 实现分离 | 阶段二起 ReAct 循环只依赖 `LLM`，将来换模型/厂商零改动 |
| `system` 作为独立参数 | 阶段二 Agent 的系统提示、阶段五多 Agent 的角色设定都走这里 |
| `Message` 通用结构（role/content） | 阶段三 Tool Use 可在此基础上扩展 tool_use / tool_result 消息类型 |
| `ChatResponse` 带 `usage` + `raw` | 阶段六可观测性（token/成本/trace）直接接 `usage`；`raw` 作调试逃生舱 |
| `Settings` 集中配置 + `provider` 预留字段 | 阶段六环境隔离、多模型 A/B、配置驱动权限分级都从这里长 |
| `agent.py` 占位且声明依赖 `LLM` | 阶段二落地 ReAct 时无需重构目录 |

---

## 10. 验收标准（本阶段「做完」的判定）

- [ ] `conda activate jingdong` 后 `pip install -r requirements.txt` 可装齐依赖。
- [ ] 复制 `.env.example` 为 `.env` 填入真实 key 后，`python -m examples.chat_cli` 能跑通**多轮**对话（第二轮能引用第一轮上下文）。
- [ ] CLI 支持 `/exit`、`/reset`、`/system <文本>`（切换系统提示生效）、流式打印。
- [ ] API key 只存在于 `.env`，`grep -r sk-ant 仓库代码` 无命中；`.env` 在 `.gitignore`。
- [ ] `examples/prompt_lab.py` 能跑出三策略对比结果，并产出 `docs/stage-1-prompt-experiment.md` 报告。
- [ ] `ruff check .` 与 `black --check .` 通过；公开接口有类型标注与 docstring。
- [ ] 换模型只需改 `.env` 的 `MODEL`（同厂商），或新增一个 `LLM` 实现 + 改装配点一行（跨厂商）—— 核心不动。

---

## 11. 未决问题 / 需用户确认

> **✅ 评审结论(2026-06-27 用户拍板,已定):**
> 1. **模型:`claude-opus-4-8`**;实验②**只比 prompt 策略,不引入温度变量**。`ClaudeLLM` 对 Opus 4.8 **不下发 temperature**(`Settings.temperature` 仍保留作通用字段供他模型用)。
> 2. **token 显示**:非流式 CLI 显示 token 用量;成本估算留阶段六。
> 3. **对话存盘**:阶段一不做,留阶段四(Memory)。
> 4. **流式接口**:`stream()` 只 yield 文本增量,不返结构化事件。
> 5. **依赖版本**:用宽松下限,暂不锁 lockfile。
> 6. **CLI 入口**:仅 `examples/chat_cli.py` 脚本,不配 `console_scripts`。
>
> 下方为评审前的原始未决项,保留作记录。

1. **用哪个 Claude 具体模型？** —— 关键决策。
   - `claude-opus-4-8`（最新、最强）**不支持 `temperature`**（发送会 400），用自适应思考/effort。优点：能力强、贴合最新实践；代价：实验②无法用「调温度」作对比变量，只能比 prompt 策略。
   - 若实验②想保留「温度」维度，需选支持温度的模型（如 `claude-sonnet-4-5` / `claude-opus-4-5`）。
   - **请拍板**：阶段一默认模型定哪个？实验是否需要温度变量？（建议：默认 `claude-opus-4-8`，实验只比 prompt 策略，不引温度 —— 最贴合「功能完整、不花哨」准则。）
2. **是否要展示 token 计数 / 成本估算？** 非流式已能拿到 `usage`；流式下要不要也补回 `Usage` 并在 CLI 显示成本（需配单价）？建议阶段一：非流式显示 token，成本估算放阶段六。
3. **CLI 是否需要把对话保存到文件**（如 `--save chat.json`）？建议阶段一不做，留到 Memory 阶段（阶段四）。
4. **流式接口 `stream()` 要不要返回结构化事件**（而非纯文本增量），以便将来流式 Tool Use？建议阶段一保持「只 yield 文本」最简，阶段三再扩展。
5. **依赖是否需要立刻锁版本**（lockfile）？建议阶段一用宽松下限，稳定后再锁。
6. **CLI 入口形态**：仅 `examples/` 脚本，还是要顺手在 `pyproject.toml` 配 `console_scripts` 暴露 `jd-chat` 命令？建议阶段一先用脚本。

---

*本设计经评审通过后再进入编码。如对 §4.4 的温度/模型取舍或 §11 任一问题有不同意见，请在评审时指出，我据此修订后再开工。*
