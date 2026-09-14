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

from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError

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


def generate_composition(chunks: list, query: str, base_url: str, max_new_tokens: int = 1024) -> str:
    """chunks is a list of plain strings, not LlamaIndex nodes."""
    client = _get_client()
    prompt = COMPOSITION_PROMPT.format(context="\n\n".join(chunks), query=query, base_url=base_url)

    # Tier-0 API keys allow 3 requests/minute. Connection and timeout errors
    # are also transient and would otherwise abort a 550-query run.
    for attempt in range(6):
        try:
            response = client.chat.completions.create(
                model=CODEGEN_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=0,
            )
            choice = response.choices[0]

            # Truncated code fails to parse, so extract_endpoints_from_code
            # returns an empty set and the query scores 0. Without this warning
            # that is indistinguishable from the model getting it wrong.
            # SOCBench-D queries need up to 10 endpoints, so it is a real risk.
            if choice.finish_reason == "length":
                print(f"    [WARNING: output truncated at {max_new_tokens} tokens]")

            return strip_code_fences(choice.message.content).strip()

        except (RateLimitError, APIConnectionError, APITimeoutError) as e:
            wait = 25 * (attempt + 1)
            print(f"    [{type(e).__name__}: {e}]")
            print(f"    [waiting {wait}s]")
            time.sleep(wait)

    raise RuntimeError("API errors after 6 attempts")