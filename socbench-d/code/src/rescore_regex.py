"""
Rescores saved generations with the endpoint regex from Pesl et al.'s
SOCBench-SC evaluation, so results are methodologically comparable with the
paper (~80% F1 for large models at composition level).

Free: works on the code stored in data/generated/, no API calls. Possible only
because run_codegen.py caches generated CODE rather than scores.

Differences from scoring.match_to_template, which this cross-checks:
  - direction: the SOLUTION is turned into a regex and matched against the
    extracted endpoints, not the other way round
  - version prefixes (/v1, /3) are optional instead of stripped
  - on SOCBench-D, path parameters match (.*?), which spans slashes:
    GET /users/{id} also matches GET /users/1/orders. RestBench uses
    ([^/]*?), which does not. This is the paper's choice, kept as-is.

Run from socbench-d/code/:
    python src/rescore_regex.py
"""
import glob
import json
import logging
import os
import re

from scoring import extract_endpoints_from_code

# socbenchsc logs a warning for every unsupported construct; per Pesl these
# concern other code parts, not endpoint extraction.
logging.disable(logging.WARNING)

GENERATED_DIR = "data/generated"


def convert_to_regex(endpoint: str, benchmark_name: str) -> str:
    """Verbatim from Pesl et al."""
    if benchmark_name == "restbench.json":
        endpoint_regex = re.sub(r"{(.*?)}", "([^/]*?)", endpoint)
        endpoint_regex = re.sub(r" /", " ((/3|/v1)?)/", endpoint_regex)
    else:
        endpoint_regex = re.sub(r"{(.*?)}", "(.*?)", endpoint)
        endpoint_regex = re.sub(r" /", " ((/v1)?)/", endpoint_regex)
    endpoint_regex = re.sub(r"(/)$", "(/?)", endpoint_regex)
    return "^" + endpoint_regex + "$"


def score_regex(found: set, solution: list, benchmark_name: str) -> dict:
    patterns = [re.compile(convert_to_regex(e, benchmark_name)) for e in solution]
    hit_solution = sum(1 for p in patterns if any(p.match(f) for f in found))
    hit_found = sum(1 for f in found if any(p.match(f) for p in patterns))
    return {
        "recall": hit_solution / len(solution) if solution else 0.0,
        "precision": hit_found / len(found) if found else 0.0,
    }


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0


def summarise(rows: list, arm: str) -> tuple:
    n = len(rows)
    r = sum(x[f"{arm}_recall"] for x in rows) / n
    p = sum(x[f"{arm}_precision"] for x in rows) / n
    old = sum(x[f"{arm}_old_recall"] for x in rows) / n
    return r, p, f1(p, r), old


files = sorted(glob.glob(os.path.join(GENERATED_DIR, "*.json")))
print(f"{len(files)} condition(s) in {GENERATED_DIR}\n")
print(f"{'condition':<44}{'arm':<9}{'recall':>8}{'prec':>8}{'f1':>8}"
      f"{'old rec':>9}{'diff':>6}")

for path in files:
    name = os.path.basename(path)
    benchmark = "restbench.json" if name.startswith("restbench") else "socbenchd"
    records = json.load(open(path))

    rows = []
    for rec in records:
        row = {}
        for arm, code_key, old_key in (("without", "code_without", "recall_without"),
                                       ("with", "code_with", "recall_with")):
            found = extract_endpoints_from_code(rec[code_key])
            s = score_regex(found, rec["solution"], benchmark)
            row[f"{arm}_recall"] = s["recall"]
            row[f"{arm}_precision"] = s["precision"]
            row[f"{arm}_old_recall"] = rec[old_key]
        rows.append(row)

    label = name.replace("_gpt-4o-2024-11-20.json", "")
    for arm in ("without", "with"):
        r, p, f, old = summarise(rows, arm)
        changed = sum(1 for x in rows if abs(x[f"{arm}_recall"] - x[f"{arm}_old_recall"]) > 1e-9)
        print(f"{label[:43]:<44}{arm:<9}{r:>8.3f}{p:>8.3f}{f:>8.3f}{old:>9.3f}{changed:>6}")
    print()
