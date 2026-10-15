"""
Paired significance for the schema metrics of check_schema.py.

Per query: required-field coverage (query parameters + body fields set /
required) and the unknown-key rate, both on calls to SOLUTION endpoints.
Queries where a metric is undefined (no required fields, no keys sent) are
left out of that comparison, so n varies between rows.

Note: these are per-query means, so they differ slightly from the pooled
ratios check_schema.py prints.

Run from socbench-d/code/:
    python src/schema_significance.py
"""
import json

from scipy.stats import wilcoxon

from benchmark_loader import load_benchmark
from check_schema import build_index, check, new_stats

D = "data/generated/socbenchd_1_k10_all_noanchor_"
GPT4O = "_n110_gpt-4o-2024-11-20.json"
LUNA = "_n110_gpt-6-luna.json"

bench = load_benchmark("socbenchd_1")
INDICES = {qs.name: build_index(qs.openapis) for qs in bench.queries}


def per_query(path: str, arm: str) -> dict:
    out = {}
    for rec in json.load(open(path)):
        st = new_stats()
        check(rec[arm], rec["solution"], INDICES[rec["queryset"]], st)
        s = st["solution"]
        required = s["query_required"] + s["body_required"]
        covered = s["query_required_set"] + s["body_required_set"]
        out[(rec["queryset"], rec["query"])] = {
            "coverage": covered / required if required else None,
            "unknown": s["keys_unknown"] / s["keys_sent"] if s["keys_sent"] else None,
        }
    return out


def compare(label: str, a: dict, b: dict) -> None:
    for metric in ("coverage", "unknown"):
        keys = [k for k in a if k in b
                and a[k][metric] is not None and b[k][metric] is not None]
        if not keys:
            continue
        xs = [a[k][metric] for k in keys]
        ys = [b[k][metric] for k in keys]
        d = [x - y for x, y in zip(xs, ys)]
        p = wilcoxon(xs, ys).pvalue if any(d) else 1.0
        print(f"{label:<34}{metric:<10}n={len(keys):>3}   "
              f"A {sum(xs) / len(xs):.3f}  B {sum(ys) / len(ys):.3f}   "
              f"A>B {sum(v > 0 for v in d):>3}  B>A {sum(v < 0 for v in d):>3}   "
              f"p = {p:.2e}")


def q(cond: str, model: str, arm: str = "code_with") -> dict:
    return per_query(D + cond + model, arm)


print("A vs B, Wilcoxon signed-rank, paired by query\n")
print("--- gpt-4o")
compare("baseline vs attention+rule", q("k10", GPT4O, "code_without"), q("k10", GPT4O))
compare("random+rule vs attention+rule", q("random", GPT4O), q("k10", GPT4O))
compare("attention+rule vs identity", q("k10", GPT4O), q("identity", GPT4O))
compare("identity vs path+verb", q("identity", GPT4O), q("pathverb", GPT4O))
print("\n--- gpt-6-luna")
compare("baseline vs attention+rule", q("k10", LUNA, "code_without"), q("k10", LUNA))
compare("attention+rule vs identity", q("k10", LUNA), q("identity", LUNA))
compare("identity vs path+verb", q("identity", LUNA), q("pathverb", LUNA))
