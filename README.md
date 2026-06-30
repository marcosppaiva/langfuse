# Langfuse Knowledge Share

Demo repo for DS team knowledge-share session on LLM observability with Langfuse.

## What is Langfuse

[Langfuse](https://langfuse.com) is an open-source LLM observability platform. It provides tracing, evaluation, prompt management, and cost tracking for LLM applications. This repo covers its core features through hands-on Jupyter notebooks.

## Stack

| Tool | Version | Role |
|------|---------|------|
| Python | 3.12 | Runtime |
| Langfuse | 3+ | Observability |
| Groq (Llama-3.3-70b) | — | LLM inference |
| Docker | — | Self-hosted infra |
| Jupyter | — | Notebooks |

## Prerequisites

- [Docker](https://www.docker.com/) + Docker Compose
- [`uv`](https://docs.astral.sh/uv/) package manager
- Groq API key (free at [console.groq.com](https://console.groq.com))

## Quick Start

```bash
# 1. Configure env
cp .env.example .env
# Edit .env and fill in GROQ_API_KEY

# 2. Start Langfuse infrastructure
docker compose up -d

# 3. Install Python dependencies
uv sync

# 4. Launch notebooks
uv run jupyter lab
```

Access Langfuse UI at **http://localhost:3000**
- Email: `admin@demo.local`
- Password: `admin123`

## Notebooks

All notebooks are in the `notebooks/` directory. Follow them in order.

| # | Notebook | Topic |
|---|----------|-------|
| 01 | `01_setup_e_primeiro_trace.ipynb` | Connect to Langfuse, `@observe` decorator, traces, cost tracking |
| 02 | `02_rag_pipeline.ipynb` | RAG pipeline with retrieval spans and span hierarchy |
| 03 | `03_prompt_e_sessions.ipynb` | Prompt versioning, session management, conversation traces |
| 04 | `04_evaluation_em_escala.ipynb` | Batch scoring, LLM-as-judge, evaluation thresholds |
| 05 | `05_opentelemetry.ipynb` | Auto-instrumentation via OpenTelemetry + Groq |
| 06 | `06_Bonus_prompt_template.ipynb` | Jinja2 PromptManager: load, render, extract metadata |

## Project Structure

```
src/langfuse_demo/
├── config.py           # Settings loader (Langfuse + Groq keys from .env)
├── client.py           # Langfuse client singleton
├── llm.py              # Groq wrapper with token/cost tracking
├── rag.py              # In-memory RAG knowledge base (BM25 + reranking)
├── eval.py             # LLM-as-judge + heuristic keyword evaluation
└── prompt_template.py  # Jinja2 prompt manager with YAML frontmatter

prompts/                # Jinja2 template examples (.j2 files)
notebooks/              # Jupyter notebooks (01–06)
```

## Infrastructure

Docker services started by `docker compose up -d`:

| Service | Port | Role |
|---------|------|------|
| langfuse-web | 3000 | UI + API |
| langfuse-worker | 3030 | Async processing |
| postgres | 5434 | Transactional DB |
| clickhouse | 8123 | Analytics & trace logs |
| minio | 9090 | S3-compatible blob storage |
| redis | 6379 | Caching + job queue |

## Notes

This is a **demo repo** for educational purposes — not production-ready. Credentials in `.env.example` are for local development only.
