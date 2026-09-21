"""
STAGE 2: generate composition code from the prepared contexts and score it.

Generates twice per query -- once from the raw chunks, once from the compressed
ones -- so any difference is attributable to compression alone.

THE UNCOMPRESSED ARM IS CACHED. Retrieval is deterministic, so the raw chunks
are byte-identical across every condition sharing a benchmark and k.
Regenerating them per condition wasted half the API budget and, worse, made
comparisons unreadable: two runs on identical input gave recall 0.633 and
0.733, because gpt-4o is not deterministic even at temperature 0. Caching turns
the baseline into one fixed reference every condition is measured against.

The cache stores generated CODE, not scores, so improvements to the scoring
logic apply retroactively to cached generations. Its path is derived from the
records and the model name, so it cannot be pointed at the wrong file by hand.

SOCBench-D queries span 11 GICS sectors, each with its own five OpenAPI specs.
Templates are therefore built PER QUERYSET -- a global set would let an
endpoint from one sector match a generated path from another.

BOTH ARMS' CODE IS SAVED to data/generated/. Scoring only checks WHICH
endpoints get called, not whether parameters, ordering or data flow are right.
Identity-only compression discards exactly the parameter definitions and
response schemas needed to build a call correctly, so equal endpoint scores do
not mean equal code. Reading the saved code side by side is the only way to see
that difference.

Run from socbench-d/code/ after prepare_contexts.py:
    python -u src/run_codegen.py 2>&1 | tee data/logs/<name>.txt
"""
import hashlib
import json
import os
import time
from collections import defaultdict

from benchmark_loader import load_benchmark
from composition import CODEGEN_MODEL_NAME, generate_composition
from scoring import build_templates, extract_endpoints_from_code, score

INPUT_PATH = "data/contexts/socbenchd_1_k10_all_noanchor_random-k5-noid_n110.json"

with open(INPUT_PATH) as f:
    records = json.load(f)
print(f"loaded {len(records)} prepared contexts from {INPUT_PATH}")

# Derived, not configured: a hand-set path was easy to leave pointing at the
# previous k, which silently overwrote the wrong cache.
BENCHMARK_NAME = records[0]["benchmark"]
TOP_K = records[0]["top_k"]
BASELINE_CACHE_PATH = (f"data/baselines/{BENCHMARK_NAME}_k{TOP_K}"
                       f"_{CODEGEN_MODEL_NAME}.json")
os.makedirs(os.path.dirname(BASELINE_CACHE_PATH), exist_ok=True)

# Named after the condition, so conditions can be diffed against each other.
CONDITION = os.path.splitext(os.path.basename(INPUT_PATH))[0]
GENERATED_PATH = f"data/generated/{CONDITION}_{CODEGEN_MODEL_NAME}.json"
os.makedirs(os.path.dirname(GENERATED_PATH), exist_ok=True)

bench = load_benchmark(BENCHMARK_NAME)
templates_by_queryset = {qs.name: build_templates(qs.openapis) for qs in bench.queries}
print(f"benchmark {BENCHMARK_NAME}, k={TOP_K}, "
      f"{len(templates_by_queryset)} queryset(s)")

if os.path.exists(BASELINE_CACHE_PATH):
    with open(BASELINE_CACHE_PATH) as f:
        baseline_cache = json.load(f)
    print(f"loaded {len(baseline_cache)} cached baseline generations "
          f"from {BASELINE_CACHE_PATH}")
else:
    baseline_cache = {}
    print(f"no baseline cache yet -- will create {BASELINE_CACHE_PATH}")

print(f"saving generated code to {GENERATED_PATH}\n")


def cache_key(record: dict) -> str:
    """Queryset-scoped: the same query text could recur across sectors."""
    return f"{record['queryset']}||{record['query']}"


def chunks_fingerprint(chunks: list) -> str:
    """Guards against reusing a baseline whose raw chunks have changed (a
    different chunking strategy, a rebuilt index)."""
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

    @property
    def mean_recall(self) -> float:
        return self.recall / max(self.n, 1)

    @property
    def mean_precision(self) -> float:
        return self.precision / max(self.n, 1)

    @property
    def f1(self) -> float:
        """Reported so the numbers line up with Pesl et al.'s results_*.json."""
        p, r = self.mean_precision, self.mean_recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def report(self) -> None:
        print(f"{self.label:<22} recall {self.mean_recall:.3f}   "
              f"precision {self.mean_precision:.3f}   "
              f"f1 {self.f1:.3f}   "
              f"exact {self.exact}/{self.n}   "
              f"halluc {self.hallucinated}   "
              f"tokens {self.tokens}")


raw_acc = Accumulator("WITHOUT compression")
compressed_acc = Accumulator("WITH compression")

# Per-sector totals, so results are comparable with Pesl's per-domain numbers.
per_queryset = defaultdict(lambda: (Accumulator("without"), Accumulator("with")))

generated = []
cache_hits = 0
cache_misses = 0
start_all = time.time()

for i, record in enumerate(records):
    query = record["query"]
    queryset = record["queryset"]
    templates = templates_by_queryset[queryset]

    print(f"\n[{i + 1}/{len(records)}] ({queryset}) {query[:70]!r}")
    print(f"    solution: {record['solution']}")
    started = time.time()

    # --- uncompressed arm: cached ---
    key = cache_key(record)
    cached = baseline_cache.get(key)
    fingerprint = chunks_fingerprint(record["raw_chunks"])

    if cached and cached["fingerprint"] == fingerprint:
        code_raw = cached["code"]
        cache_hits += 1
    else:
        if cached:
            print("    [baseline cache stale -- raw chunks changed, regenerating]")
        code_raw = generate_composition(
            record["raw_chunks"], query, record["base_url"]
        )
        baseline_cache[key] = {"fingerprint": fingerprint, "code": code_raw}
        cache_misses += 1
        with open(BASELINE_CACHE_PATH, "w") as f:  # persist instantly
            json.dump(baseline_cache, f, indent=2)

    result_raw = score(
        extract_endpoints_from_code(code_raw), record["solution"],
        templates, record["base_path"],
    )
    raw_acc.add(result_raw, record["raw_tokens"])
    per_queryset[queryset][0].add(result_raw, record["raw_tokens"])

    # --- compressed arm: always fresh ---
    code_compressed = generate_composition(
        record["compressed_chunks"], query, record["base_url"]
    )
    result_compressed = score(
        extract_endpoints_from_code(code_compressed), record["solution"],
        templates, record["base_path"],
    )
    compressed_acc.add(result_compressed, record["compressed_tokens"])
    per_queryset[queryset][1].add(result_compressed, record["compressed_tokens"])

    # Saved for qualitative inspection: endpoint scores cannot show whether
    # parameters, ordering or data flow survived compression.
    generated.append({
        "queryset": queryset,
        "query": query,
        "solution": record["solution"],
        "code_without": code_raw,
        "code_with": code_compressed,
        "found_without": sorted(result_raw["found"]),
        "found_with": sorted(result_compressed["found"]),
        "recall_without": result_raw["recall"],
        "recall_with": result_compressed["recall"],
    })
    with open(GENERATED_PATH, "w") as f:
        json.dump(generated, f, indent=2)

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
print(f"generated code:     {GENERATED_PATH}")

if len(per_queryset) > 1:
    print()
    print(f"{'queryset':<26}{'recall':>18}{'precision':>18}")
    print(f"{'':<26}{'without':>9}{'with':>9}{'without':>9}{'with':>9}")
    for name in sorted(per_queryset):
        without, with_ = per_queryset[name]
        print(f"{name:<26}{without.mean_recall:>9.3f}{with_.mean_recall:>9.3f}"
              f"{without.mean_precision:>9.3f}{with_.mean_precision:>9.3f}")

print()
raw_acc.report()
compressed_acc.report()
print()
if compressed_acc.tokens:
    print(f"Compression ratio:  {raw_acc.tokens / compressed_acc.tokens:.2f}x")
print(f"Recall delta:       {compressed_acc.mean_recall - raw_acc.mean_recall:+.3f}")
print(f"Total time:         {(time.time() - start_all) / 60:.1f} min")
print("=" * 78)