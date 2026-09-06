"""
STAGE 2: generate composition code from the prepared contexts and score it.

Generates twice per query -- once from the raw chunks, once from the compressed
ones -- so any difference is attributable to compression alone.

THE UNCOMPRESSED ARM IS CACHED. Retrieval is deterministic, so the raw chunks
are byte-identical across every condition sharing an API and k. Regenerating
them per condition wasted half the API budget and, worse, made comparisons
unreadable: two runs on identical input gave recall 0.633 and 0.733, because
gpt-4o is not deterministic even at temperature 0. Caching turns the baseline
into one fixed reference every condition is measured against.

The cache stores generated CODE, not scores, so improvements to the scoring
logic apply retroactively to cached generations.

Run from socbench-d/code/ after prepare_contexts.py:
    python src/run_codegen.py 2>&1 | tee data/logs/task1_run.txt
"""
import hashlib
import json
import os
import time

import benchmark
from composition import generate_composition
from scoring import build_templates, extract_endpoints_from_code, score

INPUT_PATH = "data/contexts_tmdb_k10_middeep_noanchor_k10.json"
# One cache per (API, k). Conditions differing only in compression settings
# share a baseline; different retrieval settings must not.
BASELINE_CACHE_PATH = "data/baseline_codegen_tmdb_k10.json"

restbench = benchmark.get_restbench()
templates = build_templates(restbench.queries[0].openapis)

with open(INPUT_PATH) as f:
    records = json.load(f)
print(f"loaded {len(records)} prepared contexts from {INPUT_PATH}")

if os.path.exists(BASELINE_CACHE_PATH):
    with open(BASELINE_CACHE_PATH) as f:
        baseline_cache = json.load(f)
    print(f"loaded {len(baseline_cache)} cached baseline generations "
          f"from {BASELINE_CACHE_PATH}\n")
else:
    baseline_cache = {}
    print(f"no baseline cache yet -- will create {BASELINE_CACHE_PATH}\n")


def chunks_fingerprint(chunks: list) -> str:
    """Guards against reusing a baseline whose raw chunks have changed (a
    different k, a different chunking strategy, a rebuilt index)."""
    joined = "\n".join(chunks).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:16]


class Accumulator:
    """Running totals for one condition."""

    def __init__(self, label: str):
        self.label = label
        self.recall = 0.0
        self.precision = 0.0
        self.hallucinated = 0
        self.exact = 0
        self.tokens = 0
        self.n = 0

    def add(self, result: dict, tokens: int) -> None:
        self.recall += result["recall"]
        self.precision += result["precision"]
        self.hallucinated += result["hallucinated"]
        self.exact += 1 if result["exact_match"] else 0
        self.tokens += tokens
        self.n += 1

    def report(self) -> None:
        n = max(self.n, 1)
        print(f"{self.label:<22} recall {self.recall / n:.3f}   "
              f"precision {self.precision / n:.3f}   "
              f"exact {self.exact}/{self.n}   "
              f"halluc {self.hallucinated}   "
              f"tokens {self.tokens}")


raw_acc = Accumulator("WITHOUT compression")
compressed_acc = Accumulator("WITH compression")

cache_hits = 0
cache_misses = 0
start_all = time.time()

for i, record in enumerate(records):
    query = record["query"]
    print(f"\n[{i + 1}/{len(records)}] {query!r}")
    print(f"    solution: {record['solution']}")
    started = time.time()

    # --- uncompressed arm: cached ---
    fingerprint = chunks_fingerprint(record["raw_chunks"])
    cached = baseline_cache.get(query)

    if cached and cached["fingerprint"] == fingerprint:
        code_raw = cached["code"]
        cache_hits += 1
    else:
        if cached:
            print("    [baseline cache stale -- raw chunks changed, regenerating]")
        code_raw = generate_composition(
            record["raw_chunks"], query, record["base_url"]
        )
        baseline_cache[query] = {"fingerprint": fingerprint, "code": code_raw}
        cache_misses += 1

    result_raw = score(
        extract_endpoints_from_code(code_raw), record["solution"],
        templates, record["base_path"],
    )
    raw_acc.add(result_raw, record["raw_tokens"])

    # --- compressed arm: always fresh ---
    code_compressed = generate_composition(
        record["compressed_chunks"], query, record["base_url"]
    )
    result_compressed = score(
        extract_endpoints_from_code(code_compressed), record["solution"],
        templates, record["base_path"],
    )
    compressed_acc.add(result_compressed, record["compressed_tokens"])

    print(f"    without: recall {result_raw['recall']:.2f}  "
          f"prec {result_raw['precision']:.2f}  "
          f"halluc {result_raw['hallucinated']}  found {sorted(result_raw['found'])}")
    print(f"    with:    recall {result_compressed['recall']:.2f}  "
          f"prec {result_compressed['precision']:.2f}  "
          f"halluc {result_compressed['hallucinated']}  "
          f"found {sorted(result_compressed['found'])}")
    print(f"    {time.time() - started:.1f}s")

with open(BASELINE_CACHE_PATH, "w") as f:
    json.dump(baseline_cache, f, indent=2)

print("\n" + "=" * 78)
print(f"condition:          {INPUT_PATH}")
print(f"baseline cache:     {cache_hits} hits, {cache_misses} generated")
print()
raw_acc.report()
compressed_acc.report()
print()
if compressed_acc.tokens:
    print(f"Compression ratio:  {raw_acc.tokens / compressed_acc.tokens:.2f}x")
delta = (compressed_acc.recall / max(compressed_acc.n, 1)
         - raw_acc.recall / max(raw_acc.n, 1))
print(f"Recall delta:       {delta:+.3f}")
print(f"Total time:         {(time.time() - start_all) / 60:.1f} min")
print("=" * 78)