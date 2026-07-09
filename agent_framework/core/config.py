"""框架配置管理。

用 ``pydantic-settings`` 从 ``.env`` 自动读取配置,带类型校验与默认值。
密钥等敏感信息只存在于 ``.env``(在 ``.gitignore`` 中),绝不硬编码进代码。
详见 stage-1-design.md §5。
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """框架运行配置,从 ``.env`` 读取(环境变量名大小写不敏感)。

    Attributes:
        provider: LLM 厂商,决定 ``create_llm`` 装配哪个实现。支持 ``claude`` /
            ``openai``。换厂商只改这一项(+ 对应的 key),核心与 CLI 不动。
        anthropic_api_key: Anthropic API Key。仅 ``provider=claude`` 时必填;
            缺失会在构造 ``ClaudeLLM`` 时报错。
        openai_api_key: OpenAI API Key。仅 ``provider=openai`` 时必填;
            缺失会在构造 ``OpenAILLM`` 时报错。
        model: 模型 id,可被 ``.env`` 的 ``MODEL`` 覆盖。留空(None)时由具体
            provider 用各自默认模型(Claude→opus-4-8,OpenAI→gpt-5.4-mini)。
        max_tokens: 单次回复的 token 上限。
        temperature: 通用采样温度字段。注意 Opus 4.8 / GPT‑5 系列实际不下发此参数
            (见各 provider 实现),保留它供支持的模型使用。
        stream: CLI 默认是否采用流式输出。
        agent_max_steps: ReAct 主循环的最大步数(防死循环);撞上限触发强制作答。
            可由 ``.env`` 的 ``AGENT_MAX_STEPS`` 覆盖。
        memory_window_tokens: 短期记忆滑动窗口的 token 预算(紧凑档,约 5–8 轮,
            让压缩机制在正常 demo 中真实触发)。
        memory_summary_max_tokens: 前情提要(递归摘要)的长度上限。
        memory_top_k: 长期记忆检索返回条数(召回候选为其 4 倍再三因子重排)。
        memory_weight_relevance: 三因子权重之相关性(主排序)。
        memory_weight_recency: 三因子权重之时近性(加分项)。
        memory_weight_importance: 三因子权重之重要性(加分项)。
        memory_half_life_hours: 时近性的指数衰减半衰期(小时)。
        memory_dedup_threshold: 写入决策降级时的去重阈值(余弦相似度)。
        embedding_model: embedding 模型 id(OpenAI)。
        memory_persist_dir: Chroma 持久化目录(gitignore,跨会话记忆落盘处)。
        planner_max_steps: 计划长度上限(prompt 约束 + 超长硬截断)。
        max_replans: 单轮执行的动态重规划次数上限(防兜圈)。
        critic_max_retries: Critic 审查不合格时的回炉次数上限。
        supervisor_specialist_max_steps: Supervisor 派工时,专员内环的步数上限。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: str = "claude"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    model: str | None = None
    max_tokens: int = 1024
    temperature: float = 1.0
    stream: bool = True
    agent_max_steps: int = 5

    # —— 阶段四 Memory(默认值为评审拍板的「紧凑档」,见 stage-4-design.md §5.2/§7.3)——
    memory_window_tokens: int = 2000
    memory_summary_max_tokens: int = 300
    memory_top_k: int = 3
    memory_weight_relevance: float = 1.0
    # 时近/重要权重初定 0.5/0.5(评审),后经评测集实测 0.25/0.25 更优
    # (Hit@1 81%→94%,见 docs/stage-4-memory-eval.md),按数据改为默认。
    memory_weight_recency: float = 0.25
    memory_weight_importance: float = 0.25
    memory_half_life_hours: float = 24.0
    memory_dedup_threshold: float = 0.9
    embedding_model: str = "text-embedding-3-small"
    memory_persist_dir: str = "data/memory"

    # —— 阶段五 Planning 与 Multi-Agent(循环护栏总表见 stage-5-design.md 附录 B)——
    planner_max_steps: int = 6
    max_replans: int = 1
    critic_max_retries: int = 1
    supervisor_specialist_max_steps: int = 5


def get_settings() -> Settings:
    """读取并返回一份配置实例。

    供 CLI / 实验脚本在启动时调用。所需的 API key 由具体 provider 在构造时校验
    (缺失则「启动即失败」并给出明确提示),而非跑到一半才发现没配 key。
    """
    return Settings()  # type: ignore[call-arg]
