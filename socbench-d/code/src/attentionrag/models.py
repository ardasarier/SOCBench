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
import warnings

from transformers import AutoModelForCausalLM, AutoTokenizer

# llama-index's own type definitions trigger this on every import; nothing we
# can fix and nothing we need to see.
warnings.filterwarnings(
    "ignore",
    message=".*validate_default.*",
    module="pydantic._internal._generate_schema",
)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# float16 halves memory on CUDA. MPS stays float32 -- float16 there can produce
# NaNs in attention, and every measurement so far was taken in float32.
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"[models] using device: {DEVICE} ({DTYPE})")

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
    """Release cached GPU/MPS blocks between queries. No-op on CPU."""
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()
