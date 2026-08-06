"""
Two models, split by what each step actually needs.

Hint prefix generation needs language capability: Qwen2.5-0.5B returns "None"
for past-tense wh-questions where 1.5B produces a correct prefix. Runs once
per query.

Anchor token + attention extraction runs TOP_K times per query. AttentionRAG's
Table 1 ablation reports that model size barely affects the attention step;
we replicated this at 0.5B vs 1.5B (identical top-5 token ranking, CR 1.55x
vs 1.51x on a real RestBench query), so the cheap model does the hot path.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[models] using device: {DEVICE}")

HINT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
COMPRESS_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

hint_tokenizer = AutoTokenizer.from_pretrained(HINT_MODEL_NAME)
hint_model = AutoModelForCausalLM.from_pretrained(HINT_MODEL_NAME).to(DEVICE)
hint_model.eval()

compress_tokenizer = AutoTokenizer.from_pretrained(COMPRESS_MODEL_NAME)
compress_model = AutoModelForCausalLM.from_pretrained(
    COMPRESS_MODEL_NAME,
    attn_implementation="eager",  # sdpa does not expose attention weights
).to(DEVICE)
compress_model.eval()


def empty_cache() -> None:
    """Release cached MPS blocks between queries. No-op on other backends."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
