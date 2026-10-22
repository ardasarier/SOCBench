"""
Token counters for the local Qwen models.

hint_prefix and attention_features record what they process here instead of
returning it, so their signatures -- used by the demos, the postprocessor and
the diagnostics -- stay unchanged. prepare_contexts.py takes the counts after
every query, which also resets them.

Counts are in Qwen tokens, not tiktoken; the local models cost no API money,
so these measure local workload, not spend.
"""
from collections import Counter

_counts = Counter()


def record(name: str, value: int = 1) -> None:
    _counts[name] += value


def take() -> dict:
    """Counts since the last take(), then resets them."""
    snapshot = dict(_counts)
    _counts.clear()
    return snapshot