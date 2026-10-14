"""
Paired comparison of precision and call counts between two conditions.

Question: path+verb keeps recall at baseline level but loses precision
(0.659 vs 0.745 for identity). Is that drop significant, and does it come from
the model calling MORE endpoints, or from calling DIFFERENT ones?

Scores with the regex from Pesl et al. (same as rescore_regex.py). Free -- works
on the saved code in data/generated/.

Run from socbench-d/code/:
    python src/compare_precision.py
"""
import json
import logging
import re

from scipy.stats import wilcoxon

from scoring import extract_endpoints_from_code

logging.disable(logging.WARNING)

D = "data/generated/"
S = "_n110_gpt-4o-2024-11-20.json"
CONDITIONS = {
    "path+verb": D + "socbenchd_1_k10_all_noanchor_pathverb" + S,
    "identity":  D + "socbenchd_1_k10_all_noanchor_identity" + S,
}


def convert_to_regex(endpoint: str) -> str:
    """Pesl et al., SOCBench-D branch."""
    r = re.sub(r"{(.*?)}", "(.*?)", endpoint)
    r = re.sub(r" /", " ((/v1)?)/", r)
    r = re.sub(r"(/)$", "(/?)", r)
    return "^" + r + "$"


def per_query(code: str, solution: list) -> dict:
    found = extract_endpoints_from_code(code)
    patterns = [re.compile(convert_to_regex(e)) for e in solution]
    correct = sum(1 for f in found if any(p.match(f) for p in patterns))
    return {
        "precision": correct / len(found) if found else 0.0,
        "calls": len(found),          # distinct endpoints called
        "correct": correct,
        "wrong": len(found) - correct,
    }


def load(path: str, arm: str) -> dict:
    """arm is "code_with" or "code_without"."""
    return {(r["queryset"], r["query"]): per_query(r[arm], r["solution"])
            for r in json.load(open(path))}


def compare(name_a, a, name_b, b):
    keys = sorted(set(a) & set(b))
    print(f"\n=== {name_a}  vs  {name_b}  ({len(keys)} gepaarte Queries) ===")
    print(f"{'':<22}{name_a:>12}{name_b:>12}")
    for metric in ("precision", "calls", "correct", "wrong"):
        ma = sum(a[k][metric] for k in keys) / len(keys)
        mb = sum(b[k][metric] for k in keys) / len(keys)
        print(f"{'mean ' + metric:<22}{ma:>12.3f}{mb:>12.3f}")

    xs = [a[k]["precision"] for k in keys]
    ys = [b[k]["precision"] for k in keys]
    d = [x - y for x, y in zip(xs, ys)]
    p = wilcoxon(xs, ys).pvalue if any(d) else 1.0
    print(f"precision: {name_a} besser {sum(v > 0 for v in d)}, "
          f"{name_b} besser {sum(v < 0 for v in d)}, gleich {sum(v == 0 for v in d)}, "
          f"p = {p:.2e}")


pv = load(CONDITIONS["path+verb"], "code_with")
idt = load(CONDITIONS["identity"], "code_with")
base = load(CONDITIONS["identity"], "code_without")   # cached, identical everywhere

compare("path+verb", pv, "identity", idt)
compare("path+verb", pv, "baseline", base)