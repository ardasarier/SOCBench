"""
Equal-budget results over all SOCBench-D instances.

Per condition and instance: recall averaged over the repeated runs (read from
the run_codegen logs, which print each query's recall), and required-field
coverage summed over the runs' generated code (check_schema.py).

Pooled over all instances: paired Wilcoxon tests on per-query means across
runs, Holm-Bonferroni over all comparisons -- and, per comparison, in how
many instances the direction agrees, since a pooled result could otherwise be
carried by one instance.

Instance 1 predates run_e2e.py: its logs use the old names, and only its last
run's code was kept, so its schema numbers come from one run.

Run from socbench-d/code/:
    python src/e2e_stats.py
"""
import os
import re
import statistics as st
import sys
from collections import defaultdict

from scipy.stats import wilcoxon

INSTANCES = ["socbenchd_1", "socbenchd_2", "socbenchd_3", "socbenchd_4", "socbenchd_5"]
CONDITIONS = {                      # display name -> label in file names
    "Regel": "reqschema",
    "roh k=20": "raw20",
    "AttentionRAG": "attention40",
    "Attn. fester": "attention40-fixhint",
    "Identity": "identity",
}
COMPARISONS = [
    ("Regel", "roh k=10"), ("AttentionRAG", "roh k=10"), ("Attn. fester", "roh k=10"),
    ("Identity", "roh k=10"), ("roh k=20", "roh k=10"),
    ("AttentionRAG", "Regel"), ("Attn. fester", "Regel"), ("Identity", "Regel"),
    ("Regel", "roh k=20"),
    ("Attn. fester", "AttentionRAG"), ("Identity", "AttentionRAG"), ("Identity", "Attn. fester"),
]
RUNS = 3
MODEL_LABEL = "gpt-6-luna-none"


def log_paths(instance: str, label: str) -> list:
    if instance == "socbenchd_1":        # names from before run_e2e.py
        names = [f"codegen_budget_{label}.txt", f"codegen_budget_{label}_r2.txt",
                 f"codegen_budget_{label}_r3.txt"]
        return [os.path.join("data/logs", n) for n in names]
    return [f"data/logs/e2e/{instance}/codegen_{label}_r{r}.txt" for r in range(1, RUNS + 1)]


def generated_paths(instance: str, label: str) -> list:
    stem = f"data/generated/{instance}_k10_budget-{label}_n110_{MODEL_LABEL}"
    if instance == "socbenchd_1":
        return [stem + ".json"]
    return [f"{stem}_r{r}.json" for r in range(1, RUNS + 1)]


def parse_log(path: str):
    """{query index: recall} for the compressed and the uncompressed arm."""
    idx, with_, without = None, {}, {}
    for line in open(path):
        m = re.match(r"\[(\d+)/\d+\]", line)
        if m:
            idx = int(m.group(1))
            continue
        m = re.match(r"\s+without: recall ([\d.]+)", line)
        if m and idx:
            without[idx] = float(m.group(1))
            continue
        m = re.match(r"\s+with:\s+recall ([\d.]+)", line)
        if m and idx:
            with_[idx] = float(m.group(1))
    return with_, without


def holm(pvalues: list) -> list:
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def schema_coverage(instance: str, label: str, indices: dict, cs):
    """Pooled required body/query coverage and unknown-key rate over runs."""
    import json
    stats = cs.new_stats()
    found = 0
    for path in generated_paths(instance, label):
        if not os.path.exists(path):
            continue
        found += 1
        for rec in json.load(open(path)):
            cs.check(rec["code_with"], rec["solution"], indices[rec["queryset"]], stats)
    s = stats["solution"]
    ratio = lambda a, b: a / b if b else float("nan")
    return (ratio(s["body_required_set"], s["body_required"]),
            ratio(s["query_required_set"], s["query_required"]),
            ratio(s["keys_unknown"], s["keys_sent"]), found)


def main():
    # --- recall: per query, averaged over runs ---
    per_query = defaultdict(dict)       # condition -> {(instance, idx): mean recall}
    run_means = defaultdict(dict)       # condition -> {instance: [mean per run]}
    for inst in INSTANCES:
        for name, label in CONDITIONS.items():
            runs = []
            for path in log_paths(inst, label):
                if os.path.exists(path):
                    with_, without = parse_log(path)
                    runs.append(with_)
                    for i, r in without.items():           # the cached reference
                        per_query["roh k=10"][(inst, i)] = r
            if not runs:
                continue
            keys = sorted(runs[0])
            run_means[name][inst] = [sum(r[k] for k in keys if k in r) / len(keys) for r in runs]
            for k in keys:
                values = [r[k] for r in runs if k in r]
                per_query[name][(inst, k)] = sum(values) / len(values)

    names = ["roh k=10"] + list(CONDITIONS)
    print("Recall bei gleichem Budget, Mittel über Läufe (pro Query gemittelt)\n")
    print(f"{'Bedingung':<14}" + "".join(f"{i[-1]:>8}" for i in INSTANCES) + f"{'alle':>9}{'n':>6}")
    for name in names:
        row = f"{name:<14}"
        for inst in INSTANCES:
            vals = [v for (i, _), v in per_query[name].items() if i == inst]
            row += f"{sum(vals) / len(vals):>8.3f}" if vals else f"{'-':>8}"
        allv = list(per_query[name].values())
        row += f"{sum(allv) / len(allv):>9.3f}{len(allv):>6}" if allv else ""
        print(row)

    print("\nStreuung zwischen Läufen (SD der Lauf-Mittel, je Instanz)\n")
    print(f"{'Bedingung':<14}" + "".join(f"{i[-1]:>8}" for i in INSTANCES))
    for name in CONDITIONS:
        row = f"{name:<14}"
        for inst in INSTANCES:
            m = run_means[name].get(inst, [])
            row += f"{st.stdev(m):>8.3f}" if len(m) > 1 else f"{'-':>8}"
        print(row)

    # --- significance over all queries ---
    rows = []
    for a, b in COMPARISONS:
        keys = sorted(set(per_query[a]) & set(per_query[b]))
        if not keys:
            continue
        xs, ys = [per_query[a][k] for k in keys], [per_query[b][k] for k in keys]
        d = [x - y for x, y in zip(xs, ys)]
        p = wilcoxon(xs, ys).pvalue if any(d) else 1.0
        agree = 0
        insts = sorted({i for i, _ in keys})
        for inst in insts:
            di = [per_query[a][k] - per_query[b][k] for k in keys if k[0] == inst]
            agree += (sum(di) > 0)
        rows.append((f"{a} vs {b}", sum(v > 0 for v in d), sum(v < 0 for v in d), p,
                     f"{agree}/{len(insts)}"))
    adjusted = holm([r[3] for r in rows])
    print(f"\n{'Vergleich (alle Queries)':<30}{'besser':>8}{'schlechter':>12}{'p (Holm)':>11}{'höher in':>10}")
    for (name, better, worse, _, agree), p_adj in zip(rows, adjusted):
        mark = "  *" if p_adj < 0.05 else ""
        print(f"{name:<30}{better:>8}{worse:>12}{p_adj:>11.2e}{agree:>10}{mark}")
    print("\n* signifikant nach Holm (alpha = 0.05) | höher in: Instanzen, in denen A im Mittel über B liegt")

    # --- schema, pooled over runs ---
    try:
        sys.path.insert(0, "src")
        import check_schema as cs
        from benchmark_loader import load_benchmark
    except Exception as error:
        print(f"\n(Schema-Teil übersprungen: {error})")
        return
    print("\nPflichtfelder Body / Query und erfundene Schlüssel, über alle Läufe gepoolt\n")
    print(f"{'Bedingung':<14}" + "".join(f"{i[-1]:>17}" for i in INSTANCES))
    indices = {}
    for inst in INSTANCES:
        indices[inst] = {qs.name: cs.build_index(qs.openapis)
                         for qs in load_benchmark(inst).queries}
    for name, label in CONDITIONS.items():
        row = f"{name:<14}"
        for inst in INSTANCES:
            body, query, unknown, found = schema_coverage(inst, label, indices[inst], cs)
            row += f"   {body:.2f}/{query:.2f}/{unknown:.2f}" if found else f"{'-':>17}"
        print(row)
    print("\nje Zelle: Body-Pflichtfelder / Query-Pflichtparameter / erfundene Schlüssel"
          "\n(Instanz 1: nur der letzte Lauf, die anderen: drei Läufe)")


if __name__ == "__main__":
    main()
