from groq import Groq
from .config import settings

# Groq llama-3.3-70b-versatile pricing — verify at https://groq.com/pricing
GROQ_PRICE_INPUT = 0.59 / 1_000_000   # $0.59 per 1M input tokens
GROQ_PRICE_OUTPUT = 0.79 / 1_000_000  # $0.79 per 1M output tokens

_groq: Groq | None = None


def _get_groq() -> Groq:
    global _groq
    if _groq is None:
        _groq = Groq(api_key=settings.groq_api_key)
    return _groq


def call_llm(messages: list[dict], model: str | None = None) -> tuple[str, dict]:
    """Call model via Groq. Returns (text, usage) where usage has input/output tokens and costs."""
    response = _get_groq().chat.completions.create(
        model=model or settings.groq_model,
        messages=messages,
        max_tokens=512,
    )
    text = response.choices[0].message.content
    usage = {
        "input": response.usage.prompt_tokens,
        "output": response.usage.completion_tokens,
        "input_cost": round(response.usage.prompt_tokens * GROQ_PRICE_INPUT, 8),
        "output_cost": round(response.usage.completion_tokens * GROQ_PRICE_OUTPUT, 8),
    }
    return text, usage
