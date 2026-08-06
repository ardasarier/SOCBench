"""
STAGE 2: generate composition code from the prepared contexts and score it.

Generates twice per query -- once from the raw chunks, once from the compressed
ones. Everything else is identical, so any difference is attributable to
compression alone.

Run from socbench-d/code/ after prepare_contexts.py:
    python src/run_codegen.py 2>&1 | tee data/logs/task1_run.txt
"""
import json
import time

import benchmark
from composition import generate_composition
from scoring import build_templates, extract_endpoints_from_code, score

INPUT_PATH = "data/task1_contexts_tmdb_k28.json"

restbench = benchmark.get_restbench()
templates = build_templates(restbench.queries[0].openapis)

with open(INPUT_PATH) as f:
    records = json.load(f)
print(f"loaded {len(records)} prepared contexts from {INPUT_PATH}\n")


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

start_all = time.time()

for i, record in enumerate(records):
    print(f"\n[{i + 1}/{len(records)}] {record['query']!r}")
    print(f"    solution: {record['solution']}")
    started = time.time()

    code_raw = generate_composition(record["raw_chunks"], record["query"], record["base_url"])
    result_raw = score(
        extract_endpoints_from_code(code_raw), record["solution"], templates, record["base_path"]
    )
    raw_acc.add(result_raw, record["raw_tokens"])

    code_compressed = generate_composition(
        record["compressed_chunks"], record["query"], record["base_url"]
    )
    result_compressed = score(
        extract_endpoints_from_code(code_compressed),
        record["solution"], templates, record["base_path"],
    )
    compressed_acc.add(result_compressed, record["compressed_tokens"])

    print(f"    without: recall {result_raw['recall']:.2f}  prec {result_raw['precision']:.2f}  "
          f"halluc {result_raw['hallucinated']}  found {sorted(result_raw['found'])}")
    print(f"    with:    recall {result_compressed['recall']:.2f}  "
          f"prec {result_compressed['precision']:.2f}  "
          f"halluc {result_compressed['hallucinated']}  found {sorted(result_compressed['found'])}")
    print(f"    {time.time() - started:.1f}s")

print("\n" + "=" * 78)
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
