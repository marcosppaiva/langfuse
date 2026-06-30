"""Simulated RAG: in-memory knowledge base + BM25 keyword retrieval. No vector DB needed."""
import math
import re
from collections import Counter

DOCS: dict[str, dict] = {
    "onboarding": {
        "id": "onboarding",
        "title": "Onboarding Guide — Development Environment",
        "content": (
            "To set up the local development environment:\n\n"
            "1. Prerequisites: Python 3.12+, uv, Docker Desktop\n"
            "2. Clone the repo: git clone https://github.com/celfocus/projeto.git\n"
            "3. Install dependencies: uv sync\n"
            "4. Configure environment variables: copy .env.example to .env and fill in the keys\n"
            "5. Start local infra: docker compose up -d\n"
            "6. Verify: uv run python -m pytest tests/\n\n"
            "Common issues:\n"
            "- Port 5432 in use: change POSTGRES_PORT in docker-compose.yaml\n"
            "- Auth error: check that .env has LANGFUSE_PUBLIC_KEY and SECRET_KEY\n"
            "- Docker won't start: ensure Docker Desktop is running"
        ),
        "tags": ["setup", "environment", "dev", "onboarding", "install", "docker", "python", "configuration"],
    },
    "arquitetura": {
        "id": "arquitetura",
        "title": "System Architecture",
        "content": (
            "The system uses a microservices architecture with the following components:\n\n"
            "Frontend: Next.js 14 (App Router), TypeScript, Tailwind CSS\n"
            "Backend API: FastAPI (Python 3.12), async/await, Pydantic v2\n"
            "Databases:\n"
            "  - PostgreSQL 16 — transactional and relational data\n"
            "  - ClickHouse — analytics and high-volume logs (billions of events)\n"
            "  - Redis — L2 cache and background job queues\n"
            "Storage: MinIO (S3-compatible) for artifacts, models, and datasets\n"
            "Observability: Langfuse for LLM tracing, OpenTelemetry for general metrics\n\n"
            "Service communication: REST (synchronous) and Redis Streams (asynchronous).\n"
            "Deploy: Kubernetes with Helm charts. Environments: dev, staging, production."
        ),
        "tags": ["architecture", "microservices", "fastapi", "nextjs", "postgres", "redis", "system", "kubernetes"],
    },
    "api_llm": {
        "id": "api_llm",
        "title": "LLM Inference API",
        "content": (
            "The inference API exposes endpoints for LLM calls:\n\n"
            "POST /api/v1/chat/completions\n"
            "  Headers: Authorization: Bearer {API_KEY}\n"
            "  Body: {\"model\": \"llama-3.3-70b\", \"messages\": [...], \"temperature\": 0.7}\n"
            "  Response: {\"choices\": [{\"message\": {\"content\": \"...\"}}], \"usage\": {...}}\n\n"
            "Available models:\n"
            "  - llama-3.3-70b-versatile: general use, best cost/quality ratio\n"
            "  - llama-3.1-8b-instant: low latency (<500ms), simple tasks\n"
            "  - mixtral-8x7b-32768: long context (32k tokens)\n\n"
            "Rate limits: 30 req/min (free tier), 500 req/min (pro)\n"
            "Authentication: via GROQ_API_KEY in .env"
        ),
        "tags": ["api", "llm", "groq", "inference", "models", "rate limit", "authentication", "endpoint"],
    },
    "mlops": {
        "id": "mlops",
        "title": "MLOps Pipeline — Training and Deploy",
        "content": (
            "Team MLOps pipeline:\n\n"
            "Experimentation: MLflow for run tracking, hyperparameters, and metrics\n"
            "Data versioning: DVC with MinIO storage\n"
            "Training: Kubernetes jobs with GPU (A100), monitored via Grafana\n"
            "Evaluation: automated benchmark suite, minimum threshold of 85% accuracy\n\n"
            "Deploy:\n"
            "  - Staging: automatic on every merge to main\n"
            "  - Production: manual approval + canary deployment (5% → 20% → 100%)\n\n"
            "Monitoring: data drift via Evidently, alerts in PagerDuty\n"
            "SLA: P99 latency < 500ms, availability > 99.9%"
        ),
        "tags": ["mlops", "mlflow", "kubernetes", "deploy", "training", "production", "monitoring", "canary"],
    },
    "langfuse_guia": {
        "id": "langfuse_guia",
        "title": "Observability Guide with Langfuse",
        "content": (
            "Langfuse is the observability platform for LLM applications.\n\n"
            "Core concepts:\n"
            "  - Trace: represents a complete interaction (e.g. a user question)\n"
            "  - Span: sub-operation within the trace (e.g. retrieval, reranking)\n"
            "  - Generation: LLM call with token accounting\n"
            "  - Score: quality evaluation (manual, heuristic, or LLM-judge)\n"
            "  - Session: grouping of multiple traces from the same user/conversation\n\n"
            "Integration in 2 lines:\n"
            "  from langfuse.decorators import observe\n"
            "  @observe()\n"
            "  def my_function(input): ...\n\n"
            "Best practices:\n"
            "  - Always include user_id and session_id in traces\n"
            "  - Use tags to filter by environment (prod/staging/dev)\n"
            "  - Create automatic scores in CI/CD pipelines"
        ),
        "tags": ["langfuse", "observability", "trace", "span", "score", "integration", "llm", "session"],
    },
    "dados_governanca": {
        "id": "dados_governanca",
        "title": "Data Governance and Privacy",
        "content": (
            "Team data governance policies:\n\n"
            "PII (Personally Identifiable Information):\n"
            "  - Never log sensitive data in observability systems\n"
            "  - Anonymize user_id before sending to Langfuse (SHA-256 hash)\n"
            "  - Log retention: 90 days for training data, 30 days for inference\n\n"
            "Access control:\n"
            "  - Production data: access only via VPN + MFA\n"
            "  - Training data: ticket request with Data Owner approval\n"
            "  - Models: versioned and auditable via DVC\n\n"
            "Compliance: GDPR, ISO 27001\n"
            "Contact: data-governance@company.com"
        ),
        "tags": ["governance", "privacy", "gdpr", "pii", "compliance", "security", "data", "audit"],
    },
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _bm25_score(query_tokens: list[str], doc_tokens: list[str], corpus_size: int) -> float:
    """Simplified BM25 scoring."""
    k1, b = 1.5, 0.75
    avg_len = 100
    doc_freq = Counter(doc_tokens)
    doc_len = len(doc_tokens)
    score = 0.0
    for token in query_tokens:
        if token in doc_freq:
            tf = doc_freq[token]
            idf = math.log(1 + (corpus_size - 1 + 0.5) / 1.5)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / avg_len)
            score += idf * (numerator / denominator)
    return score


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """BM25 keyword retrieval over in-memory knowledge base."""
    query_tokens = _tokenize(query)
    scored = []
    for doc in DOCS.values():
        doc_text = doc["title"] + " " + doc["content"] + " " + " ".join(doc["tags"])
        doc_tokens = _tokenize(doc_text)
        score = _bm25_score(query_tokens, doc_tokens, len(DOCS))
        scored.append({
            "id": doc["id"],
            "title": doc["title"],
            "content": doc["content"],
            "retrieval_score": round(score, 4),
        })
    scored.sort(key=lambda x: x["retrieval_score"], reverse=True)
    return scored[:top_k]


def rerank(query: str, docs: list[dict]) -> list[dict]:
    """Boost docs where query terms appear in title (simulates cross-encoder reranking)."""
    query_tokens = set(_tokenize(query))
    for doc in docs:
        title_tokens = set(_tokenize(doc["title"]))
        title_overlap = len(query_tokens & title_tokens)
        doc["final_score"] = round(doc["retrieval_score"] + title_overlap * 0.3, 4)
    docs.sort(key=lambda x: x["final_score"], reverse=True)
    return docs


def format_context(docs: list[dict], max_chars: int = 1500) -> str:
    """Format retrieved docs into LLM context string."""
    parts = []
    total = 0
    for doc in docs:
        snippet = f"[{doc['title']}]\n{doc['content']}"
        if total + len(snippet) > max_chars:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n\n---\n\n".join(parts)
