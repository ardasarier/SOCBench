"""
Composition step: prompt an LLM with retrieved OpenAPI chunks, get Python back.

This does not exist in socbench-d. That codebase stops at retrieval; socbenchsc
analyses code but does not produce it. The README describes combining them but
leaves generation to the experimenter.

Scoring lives in scoring.py so it can be used without an API key.
"""
import os
import re
import time

from openai import OpenAI, RateLimitError

# Matches the model socbench-d's own evaluate.py uses, so numbers are comparable.
CODEGEN_MODEL_NAME = "gpt-4o-2024-11-20"

_client = None

# The absolute-URL rule is not stylistic: socbenchsc's parse_argument() discards
# any URL without a scheme, so requests.get("/search") yields nothing at all.
COMPOSITION_PROMPT = """You are given documentation for several API endpoints and a user task.
Write Python code using the `requests` library that fulfils the task.

Rules:
- Use ONLY endpoints that appear in the documentation below.
- Always use complete absolute URLs including the scheme, e.g.
  requests.get("{base_url}/search"), never requests.get("/search").
- Output ONLY Python code. No explanation, no markdown fences.

API documentation:
{context}

Task: {query}

Python code:"""


def strip_code_fences(text: str) -> str:
    """Models wrap code in ```python ... ``` despite being told not to."""
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return fenced[0] if fenced else text


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key or key.startswith("not-a-real"):
            raise RuntimeError(
                "OPENAI_API_KEY is unset or still the dummy value. "
                "Set a real key in ~/.zprofile."
            )
        _client = OpenAI()
    return _client


def generate_composition(
    chunks: list, query: str, base_url: str, max_new_tokens: int = 512
) -> str:
    """chunks is a list of plain strings, not LlamaIndex nodes."""
    client = _get_client()
    prompt = COMPOSITION_PROMPT.format(
        context="\n\n".join(chunks), query=query, base_url=base_url
    )

    # Tier-0 API keys allow 3 requests/minute; back off rather than lose a run.
    for attempt in range(6):
        try:
            response = client.chat.completions.create(
                model=CODEGEN_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=0,
            )
            return strip_code_fences(response.choices[0].message.content).strip()
        except RateLimitError:
            wait = 25 * (attempt + 1)
            print(f"    [rate limited, waiting {wait}s]")
            time.sleep(wait)

    raise RuntimeError("rate limited after 6 attempts")
