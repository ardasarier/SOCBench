"""
Calibrates fixed vs. relative attention thresholds on SOCBench-D.

The raw attention scores are sums over all layers; their scale on SOCBench-D is
unknown, so a fixed threshold cannot be chosen blind. This samples real chunks,
reports the score distribution, and estimates the compression ratio for
candidate thresholds -- WITHOUT the identity rule, the only setting where the
selection strategy measurably matters.

Target: attention top-k=10 without identity rule ran at 3.13x. Pick fixed and
relative values that land near it, so both are compared at equal budget.

Run from socbench-d/code/:
    python src/calibrate_threshold.py
"""
import json
import warnings

warnings.filterwarnings("ignore")

# torch first: importing faiss (pulled in via `benchmark`) before torch can
# segfault on macOS, as both ship their own OpenMP runtime. `benchmark` was only
# needed for the tokenizer, so it is replaced by tiktoken directly.
from attentionrag.attention_features import compute_attention_feature
from attentionrag.compression import compress_chunk
from attentionrag.hint_prefix import generate_answer_hint_prefix
from attentionrag.models import empty_cache

import tiktoken

CONTEXTS = "data/contexts/socbenchd_1_k10_all_noanchor_k10_n110.json"
SAMPLE_EVERY = 7          # every 7th query -> 16 queries across all sectors

enc = tiktoken.encoding_for_model("gpt-4o")   # same as benchmark.get_encoding()
records = json.load(open(CONTEXTS))[::SAMPLE_EVERY]

chunks = []
for i, r in enumerate(records):
    # Release MPS blocks between queries, as prepare_contexts.py does. Without
    # it, 160 chunks fragment MPS memory, which can surface as a segfault.
    empty_cache()
    # flush=True: output piped through tee is buffered, and a crash would
    # swallow it -- this way the log shows exactly how far the run got.
    print(f"[{i + 1}/{len(records)}] {r['queryset']}", flush=True)
    hint = generate_answer_hint_prefix(r["query"])
    for ch in r["raw_chunks"]:
        scores, _ = compute_attention_feature(ch, r["query"], hint, "")
        chunks.append((ch, scores))

all_scores = sorted(s["score"] for _, sc in chunks for s in sc)
maxima = sorted(max(s["score"] for s in sc) for _, sc in chunks if sc)


def pct(values, q):
    return values[min(len(values) - 1, int(q / 100 * len(values)))]


print(f"\n{len(chunks)} chunks, {len(all_scores)} token scores")
print("score distribution:")
for q in (50, 90, 95, 98, 99, 99.5):
    print(f"   p{q:<5} {pct(all_scores, q):.4f}")
print(f"per-chunk max:  min {maxima[0]:.4f}  median {pct(maxima, 50):.4f}  "
      f"max {maxima[-1]:.4f}")

raw = sum(len(enc.encode(ch)) for ch, _ in chunks)


def estimate(**kw):
    comp = empty = 0
    for ch, sc in chunks:
        text, _ = compress_chunk(ch, sc, keep_identity=False, **kw)
        if not text.strip():
            empty += 1          # would be dropped by the postprocessor
        comp += len(enc.encode(text))
    return (raw / comp if comp else float("inf")), empty


print(f"\nestimated CR without identity rule (target ~3.13x):")
cr, e = estimate(k=10)
print(f"   top-k=10 (reference)    CR {cr:5.2f}x   empty chunks {e}")
print("   fixed thresholds, from the score percentiles:")
for q in (90, 95, 97, 98, 99, 99.5):
    v = pct(all_scores, q)
    cr, e = estimate(threshold_abs=v)
    print(f"     abs {v:.4f} (p{q:<4})   CR {cr:5.2f}x   empty chunks {e}")
print("   relative thresholds:")
for ratio in (0.1, 0.25, 0.4, 0.5, 0.7):
    cr, e = estimate(threshold_ratio=ratio)
    print(f"     ratio {ratio:<4}           CR {cr:5.2f}x   empty chunks {e}")
