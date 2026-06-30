"""Evaluation utilities: LLM-as-judge and heuristic scoring, shared across notebooks."""
import json
import re

from langfuse_demo.llm import call_llm


def evaluate_with_llm(
    question: str,
    answer: str,
    expected: str | None = None,
) -> tuple[float, str]:
    """LLM-as-judge evaluation. Returns (score 0.0–1.0, comment)."""
    reference = f"\n\nReference answer:\n{expected}" if expected else ""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI response quality evaluator. "
                "Rate considering: technical accuracy, clarity, completeness, and relevance to the question. "
                "Respond ONLY with JSON: {\"score\": 0.85, \"comment\": \"short reason\"}"
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nResponse to evaluate:\n{answer}{reference}",
        },
    ]
    raw, _ = call_llm(messages)
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group()) if match else {}
        score = float(data.get("score", 0.5))
        comment = str(data.get("comment", "automatic evaluation"))
        return min(max(score, 0.0), 1.0), comment
    except Exception:
        return 0.5, "evaluation parse failed"


def evaluate_heuristic(answer: str, keywords: list[str]) -> float:
    """Keyword coverage: fraction of expected keywords present in answer."""
    if not keywords:
        return 0.0
    answer_lower = answer.lower()
    found = sum(1 for kw in keywords if kw in answer_lower)
    return round(found / len(keywords), 2)
