# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A 4–6 week internship project to **build a reusable Agent framework from scratch** that can be used for real business scenarios. The work follows a fixed 6-stage curriculum and culminates in a complete, modular `agent_framework/` package plus a final presentation.

> **The codebase is in the skeleton phase — the `agent_framework/` package directory tree exists (empty dirs), but no module files or tests yet.** Build strictly according to the curriculum below; do not invent structure that contradicts it, and do not narrow the scope to a single hard-coded business script.

### The prime directive: completeness over polish

This project does **not** need to be industrial-grade. The single most important goal is that **the Agent's functionality is complete — every capability the outline requires must actually exist, run, and be demonstrable.** No required module may be faked, stubbed-and-forgotten, or silently dropped to save time. If you spot a gap in the outline or think something should be added, **ask the user before deciding.** Favor "modest but real and working" over "impressive but missing pieces."

### Business scenario (the stage 5/6 landing target)

The concrete business case is a **京东客服 Agent (JD customer-service agent)**, defined by `frame/给你一个直观的例子.docx`:

- **Input**: a user's feedback / problem (prompt + the feedback text).
- **Output**: not a single answer but a **solution plan** — what actions to take, plus supporting materials.
- **Tiered execution / human-in-the-loop**: low-permission actions the Agent performs directly and returns info to the user; high-permission actions **trigger a human** to handle them.

This scenario naturally exercises every module: ReAct loop (core), multi-step plans (planning), order/product/refund tools (tools), order & history context (memory), problem routing & collaboration (multi_agent), and **permission tiers + an approval (HITL) hook (safety)**.

## Authoritative document (the single source of truth)

`frame/Agent框架实习培训大纲.md` — the **Agent 框架实习培训大纲**. It defines the learning order, the required papers/courses, the per-stage deliverables, the target directory architecture, and the mentor grading rubric. **Everything in this project follows this outline.** When in doubt, defer to it.

`ROADMAP.md` is the condensed, checklist form of that outline — use it to track progress.

## The 6 stages (build in this order)

Each stage: **write a design doc first → mentor review → then code.**

1. **基础认知与环境搭建** — LLM API (Chat Completion, streaming, tokens/cost), Prompt Engineering (System Prompt, Few-shot, CoT, structured JSON), Python 3.10+ env, `.env` key management, project skeleton.
2. **最小 Agent 循环** — the **ReAct loop** (`Thought → Action → Observation`), max-step guard, structured output parsing, error recovery. This is the autonomous core; do not replace it with a hard-coded pipeline.
3. **Tool Use 系统** — Function Calling protocol, `BaseTool` abstraction, `ToolRegistry`, Pydantic/JSON-Schema params, ≥5 real tools with full schemas and error handling.
4. **Memory 与上下文管理** — short-term (sliding window + token budget) and long-term (embeddings + vector store) memory, context compression/summarization, unified `MemoryManager`.
5. **Planning 与 Multi-Agent** — Plan-and-Execute, dynamic re-planning, ≥2 Multi-Agent patterns (e.g. Router + Supervisor), Reflection/Critic self-correction, workflow orchestration.
6. **生产化与业务落地** — reliability (retry/fallback/timeout/idempotency), observability (trace/logs/metrics), evaluation (eval sets + LLM-as-Judge + A/B), safety (prompt-injection defense, output filtering, rate limits), deployment (FastAPI + SSE + Docker).

## Target architecture (from the outline)

Build toward this module layout — the core depends on interfaces (`Protocol`/ABC), not concrete vendors. The package is named `agent_framework` (underscore) so it is importable; the outline writes it `agent-framework`. The empty directory tree already exists; add files as each stage's design doc is approved.

```
agent_framework/
├── core/            # agent.py (Agent base + ReAct Loop), llm.py, config.py
├── tools/           # base.py (BaseTool), registry.py (ToolRegistry), + concrete tools
├── memory/          # manager.py, short_term.py, long_term.py (vector), compressor.py
├── planning/        # planner.py, executor.py
├── multi_agent/     # router.py, supervisor.py, protocol.py
├── observability/   # tracer.py, logger.py, metrics.py
├── safety/          # input_filter.py, output_filter.py, rate_limiter.py, approval.py (HITL)
├── api/             # server.py (FastAPI), schemas.py
├── evaluation/      # evaluator.py, datasets/, reports/
├── tests/  examples/
```

Design docs live at the repo-root `docs/` (one per stage, written before code).

### LLM provider

The concrete LLM is **Claude (Anthropic API)**. But `core/llm.py` must expose an `LLM` interface (`Protocol`/ABC) so the provider stays swappable — Claude is one implementation behind that interface, not a hard dependency of the core loop.

## How we work — the Agent Team operating model

This project is built by a **multi-agent team modeled on a big-tech Agent Platform org**. The main Claude Code loop acts as **Orchestrator / Tech Lead**: it owns the plan and **spawns specialized subagents** (via the Agent tool) when a stage needs them. Each subagent owns one module of the target architecture, so there is no ownerless code. Execution is **stage-by-stage** (not all-parallel): spawn the agents a stage needs, have Architect + QA review their output, then move on.

| Role | Big-tech counterpart | Owns | Stage |
|------|----------------------|------|-------|
| **Orchestrator** (main loop) | EM + Tech Lead | whole project, task routing, acceptance vs outline | all |
| **Architect** | Principal Engineer | interfaces (Protocol/ABC), directory discipline, design-doc review | all (esp. 1–2) |
| **Core Runtime** | Agent Runtime Eng | `core/` (agent · llm · config) | 1–2 |
| **Tools Engineer** | Integrations Eng | `tools/` | 3 |
| **Memory/RAG** | Memory/RAG Eng | `memory/` | 4 |
| **Orchestration** | Multi-Agent Eng | `planning/`, `multi_agent/` | 5 |
| **Platform/Infra** | Platform/MLOps | `observability/`, `api/` | 6 |
| **Safety** | Trust & Safety | `safety/` (incl. approval/HITL) | 6 |
| **QA/Eval** | Eval/QA Eng | `evaluation/`, `tests/` | 3 onward |
| **Tech Writer/DevRel** | Developer Advocate | `docs/`, `examples/`, README, defense deck | all |
| **PM/Acceptance** | Product Manager | align deliverables to outline, per-stage sign-off, weekly report | all |

The same discipline still applies to every subagent: **design doc → review → code**, complete type hints, tests on key paths.

## Commands

Targets **Python 3.10+** (uses `Protocol`, `X | None`). The dev environment is the **conda env `jingdong`** (Python 3.11; the user referred to it as "JD" but the actual env name is `jingdong`). Activate with `conda activate jingdong`. No build/lint/test tooling is configured yet — the outline requires type hints throughout, `ruff`/`black` for style, and unit tests on key paths. The Anthropic API key belongs in `.env` (gitignored), never in code.

## Cross-cutting requirements (apply to every stage)

- **Extensibility is a hard requirement (user-stressed)**: everything that can vary — LLM provider, tools, permission/HITL rules, memory backend — must sit behind an interface and be swappable as a plugin, never hard-coded. Concretely: customer-service tools start as **mocks behind `BaseTool`** (real JD APIs plug in later with no core change); permission tiers are **config-driven rules**, not inline `if/else`. Adding a new capability = adding an implementation, not editing the core loop.
- **Type hints complete**; module interfaces via `Protocol`/ABC; key paths unit-tested; consistent style (ruff/black).
- **Design doc before code**, reviewed by mentor; docstrings; README explaining how to run and extend.
- **Process**: daily standup, weekly report (done / blockers / next), PRs require code review.
- **Comparative reading throughout**: LangChain, LangGraph, OpenAI Swarm, AutoGen, CrewAI, Dify.
- **Grading rubric** (keep in mind): code quality 25% · design 25% · learning depth 20% · business effect 20% · communication 10%.

## AgentKnowledge/ — Obsidian knowledge base, not code

`AgentKnowledge/` is an **Obsidian vault** used as the personal knowledge base (project home, chat logs, decisions, prompts, research). It is not part of the framework and must not be imported by code. Entry point: `AgentKnowledge/01_Project/Agent 框架项目主页.md`; usage conventions in `AgentKnowledge/README 怎么用这个知识库.md`. Keep application code under `agent_framework/`.

## Git

Repo root is `~/Desktop/JD` (scoped here intentionally, not the home directory). Remote `origin` → https://github.com/YibinLi-Alan/JDCustomerAgentDev (public).
