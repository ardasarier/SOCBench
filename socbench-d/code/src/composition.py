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

# gpt-6-luna (released 2026-09-22): reasoning model, roughly 25x cheaper than
# gpt-4o. The API exposes only the alias -- no dated snapshot, and responses
# echo the alias -- so run dates must be recorded for reproducibility.
CODEGEN_MODEL_NAME = "gpt-6-luna"

# Reasoning effort for gpt-6 models: "none", "low", "medium" (API default),
# "high", "xhigh", "max". None sends nothing, i.e. the default -- and is the
# only valid value for gpt-4o, which rejects the parameter.
REASONING_EFFORT = "none"

# Temperature only takes effect with reasoning off; with any other effort the
# API rejects values other than 1. None sends nothing.
TEMPERATURE = 0

# Caches and output files are keyed by this label, not the bare model name:
# generations with and without reasoning must never share a baseline cache.
# The default effort keeps the bare name, so existing caches stay valid.
CODEGEN_LABEL = (CODEGEN_MODEL_NAME if REASONING_EFFORT is None
                 else f"{CODEGEN_MODEL_NAME}-{REASONING_EFFORT}")

# USD per 1M tokens: (input, cached input, output). Reasoning tokens are part
# of completion_tokens and billed as output.
PRICES = {
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-4o-2024-11-20": (2.50, 1.25, 10.00),
}

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


def cost_usd(usage: dict, model: str = CODEGEN_MODEL_NAME) -> float:
    price_in, price_cached, price_out = PRICES[model]
    uncached = usage["prompt_tokens"] - usage["cached_tokens"]
    return (uncached * price_in
            + usage["cached_tokens"] * price_cached
            + usage["completion_tokens"] * price_out) / 1e6


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


def generate_composition(chunks: list, query: str, base_url: str,
                         max_new_tokens: int = 4096) -> tuple:
    """chunks is a list of plain strings, not LlamaIndex nodes.

    Returns (code, usage). usage holds the token counts the API reports --
    the authoritative input size, unlike our tiktoken estimate -- and the
    cost in USD.
    """
    client = _get_client()
    prompt = COMPOSITION_PROMPT.format(context="\n\n".join(chunks), query=query, base_url=base_url)

    # Tier-0 API keys allow 3 requests/minute. Connection and timeout errors
    # are also transient and would otherwise abort a 550-query run.
    for attempt in range(6):
        try:
            # Reasoning models reject max_tokens. Internal reasoning tokens
            # count against this budget -- hence 4096 instead of the 1024 used
            # for gpt-4o.
            request = {
                "model": CODEGEN_MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": max_new_tokens,
            }
            if REASONING_EFFORT is not None:
                request["reasoning_effort"] = REASONING_EFFORT
            if TEMPERATURE is not None:
                request["temperature"] = TEMPERATURE
            response = client.chat.completions.create(**request)
            choice = response.choices[0]

            # Truncated code fails to parse and scores 0, indistinguishable
            # from a model error. With a reasoning model the budget can also be
            # spent entirely on reasoning, leaving no code at all.
            if choice.finish_reason == "length":
                print(f"    [WARNING: output truncated at {max_new_tokens} tokens]")

            u = response.usage
            details_out = getattr(u, "completion_tokens_details", None)
            details_in = getattr(u, "prompt_tokens_details", None)
            usage = {
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "reasoning_tokens": getattr(details_out, "reasoning_tokens", 0) or 0,
                "cached_tokens": getattr(details_in, "cached_tokens", 0) or 0,
                "finish_reason": choice.finish_reason,
            }
            usage["cost_usd"] = cost_usd(usage)

            content = choice.message.content or ""
            return strip_code_fences(content).strip(), usage

        except (RateLimitError, APIConnectionError, APITimeoutError) as e:
            wait = 25 * (attempt + 1)
            print(f"    [{type(e).__name__}: {e}]")
            print(f"    [waiting {wait}s]")
            time.sleep(wait)

    raise RuntimeError("API errors after 6 attempts")
