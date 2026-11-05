"""
Equal-budget results over all SOCBench-D instances and RestBench.

Per condition and instance: recall averaged over the repeated runs (read from
the run_codegen logs, which print each query's recall), and required-field
coverage summed over the runs' generated code (check_schema.py).

Pooled over all instances: paired Wilcoxon tests on per-query means across
runs, Holm-Bonferroni over all comparisons -- and, per comparison, in how
many instances the direction agrees, since a pooled result could otherwise be
carried by one instance.

Instance 1 predates run_e2e.py: its logs use the old names, and only its last
run's code was kept, so its schema numbers come from one run.

Each benchmark is evaluated on its own -- recall, significance (Holm within
the benchmark) and correctness -- since synthetic and real-world APIs are
not pooled. For RestBench, the output side (keys read from responses) is
reported too: unlike SOCBench-D, its responses define schemas.

Run from socbench-d/code/:
    python src/e2e_stats.py                 # both
    python src/e2e_stats.py restbench       # one benchmark
"""
import os
import re
import statistics as st
import sys
from collections import defaultdict

from scipy.stats import wilcoxon

# benchmark -> its parts: (display name, log directory, generated-file stem
# prefix, queries, API base path). Parts of one benchmark are pooled for the
# significance tests; RestBench is also tested per API, since its two APIs
# behave differently (the composition model knows TMDB from training).
SUITES = {
    "SOCBench-D": [(f"{i}", f"socbenchd_{i}", f"socbenchd_{i}_k10_budget-", 110, "")
                   for i in range(1, 6)],
    "RestBench": [("spotify", "restbench-spotify", "restbench_k10_budget-spotify-", 57, "/v1"),
                  ("tmdb", "restbench-tmdb", "restbench_k10_budget-tmdb-", 100, "/3")],
}
CONDITIONS = {                      # display name -> label in file names
    "rule": "reqschema",
    "raw k=20": "raw20",
    "AttentionRAG": "attention40",
    "fixed hint": "attention40-fixhint",
    "identity": "identity",
}
COMPARISONS = [
    ("rule", "raw k=10"), ("AttentionRAG", "raw k=10"), ("fixed hint", "raw k=10"),
    ("identity", "raw k=10"), ("raw k=20", "raw k=10"),
    ("AttentionRAG", "rule"), ("fixed hint", "rule"), ("identity", "rule"),
    ("rule", "raw k=20"),
    ("fixed hint", "AttentionRAG"), ("identity", "AttentionRAG"), ("identity", "fixed hint"),
]
RUNS = 3

# Without any retrieved context: what the model knows from training. Reported
# beside the main table and tested on its own against uncompressed k=10, so
# the Holm family of the main comparisons stays unchanged.
NOCTX = ("no context", "noctx")
MODEL_LABEL = "gpt-6-luna-none"


def log_paths(part, label: str) -> list:
    logdir = part[1]
    e2e = [f"data/logs/e2e/{logdir}/codegen_{label}_r{r}.txt" for r in range(1, RUNS + 1)]
    if logdir == "socbenchd_1" and not any(os.path.exists(p) for p in e2e):
        # names from before run_e2e.py
        names = [f"codegen_budget_{label}.txt", f"codegen_budget_{label}_r2.txt",
                 f"codegen_budget_{label}_r3.txt"]
        return [os.path.join("data/logs", n) for n in names]
    return [f"data/logs/e2e/{logdir}/codegen_{label}_r{r}.txt" for r in range(1, RUNS + 1)]


def generated_paths(part, label: str) -> list:
    logdir, prefix, n = part[1], part[2], part[3]
    stem = f"data/generated/{prefix}{label}_n{n}_{MODEL_LABEL}"
    numbered = [f"{stem}_r{r}.json" for r in range(1, RUNS + 1)]
    if logdir == "socbenchd_1" and not any(os.path.exists(p) for p in numbered):
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


def schema_coverage(part, label: str, indices: dict, cs, arm: str = "code_with"):
    """Pooled required body/query coverage, unknown-key rate and output-side
    reads over runs."""
    import json
    stats = cs.new_stats()
    found = 0
    for path in generated_paths(part, label):
        if not os.path.exists(path):
            continue
        found += 1
        for rec in json.load(open(path)):
            cs.check(rec[arm], rec["solution"], indices[rec["queryset"]], stats,
                     base_path=part[4])
    s = stats["solution"]
    ratio = lambda a, b: a / b if b else float("nan")
    return (ratio(s["body_required_set"], s["body_required"]),
            ratio(s["query_required_set"], s["query_required"]),
            ratio(s["keys_unknown"], s["keys_sent"]),
            s["reads"], ratio(s["reads_unknown"], s["reads"]), found)


def significance(per_query, title: str, only=None):
    """Paired Wilcoxon on per-query means, Holm over the comparisons; with
    `only`, restricted to the queries of one part."""
    rows = []
    for a, b in COMPARISONS:
        keys = sorted(k for k in set(per_query[a]) & set(per_query[b])
                      if only is None or k[0] == only)
        if not keys:
            continue
        xs, ys = [per_query[a][k] for k in keys], [per_query[b][k] for k in keys]
        d = [x - y for x, y in zip(xs, ys)]
        p = wilcoxon(xs, ys).pvalue if any(d) else 1.0
        groups = sorted({g for g, _ in keys})
        agree = sum(1 for g in groups
                    if sum(per_query[a][k] - per_query[b][k] for k in keys if k[0] == g) > 0)
        rows.append((f"{a} vs {b}", sum(v > 0 for v in d), sum(v < 0 for v in d), p,
                     f"{agree}/{len(groups)}"))
    adjusted = holm([r[3] for r in rows])
    print(f"\n{'Comparison (' + title + ')':<34}{'better':>8}{'worse':>12}{'p (Holm)':>11}{'higher in':>10}")
    for (name, better, worse, _, agree), p_adj in zip(rows, adjusted):
        mark = "  *" if p_adj < 0.05 else ""
        print(f"{name:<34}{better:>8}{worse:>12}{p_adj:>11.2e}{agree:>10}{mark}")
    if only is None:
        print("* significant after Holm (alpha = 0.05) | higher in: parts in which A has the higher mean")


def load_indices(part, cs, load_benchmark):
    """{queryset name: index} for one part, or None if the loader cannot
    provide its specifications."""
    logdir = part[1]
    name = "restbench" if logdir.startswith("restbench") else logdir
    try:
        return {qs.name: cs.build_index(qs.openapis) for qs in load_benchmark(name).queries}
    except Exception as error:
        print(f"  ({logdir}: specifications not loadable -- {error})")
        return None


def evaluate(suite: str, parts: list):
    print("=" * 96)
    print(f"{suite}: recall at equal budget, mean over runs (per-query mean)\n")
    per_query = defaultdict(dict)       # condition -> {(part, idx): mean recall}
    run_means = defaultdict(dict)       # condition -> {part: [mean per run]}
    for part in parts:
        key = part[0]
        for name, label in list(CONDITIONS.items()) + [NOCTX]:
            runs = []
            for path in log_paths(part, label):
                if os.path.exists(path):
                    with_, without = parse_log(path)
                    runs.append(with_)
                    for i, r in without.items():           # the cached reference
                        per_query["raw k=10"][(key, i)] = r
            if not runs:
                continue
            keys = sorted(runs[0])
            run_means[name][key] = [sum(r[k] for k in keys if k in r) / len(keys) for r in runs]
            for k in keys:
                values = [r[k] for r in runs if k in r]
                per_query[name][(key, k)] = sum(values) / len(values)

    names = ["raw k=10"] + list(CONDITIONS) + ([NOCTX[0]] if per_query[NOCTX[0]] else [])
    print(f"{'Condition':<14}" + "".join(f"{p[0]:>9}" for p in parts) + f"{'all':>9}{'n':>6}")
    for name in names:
        row = f"{name:<14}"
        for part in parts:
            vals = [v for (i, _), v in per_query[name].items() if i == part[0]]
            row += f"{sum(vals) / len(vals):>9.3f}" if vals else f"{'-':>9}"
        allv = list(per_query[name].values())
        row += f"{sum(allv) / len(allv):>9.3f}{len(allv):>6}" if allv else ""
        print(row)

    print("\nRun-to-run spread (SD of the run means)\n")
    print(f"{'Condition':<14}" + "".join(f"{p[0]:>9}" for p in parts))
    for name in CONDITIONS:
        row = f"{name:<14}"
        for part in parts:
            m = run_means[name].get(part[0], [])
            row += f"{st.stdev(m):>9.3f}" if len(m) > 1 else f"{'-':>9}"
        print(row)

    significance(per_query, "all queries")
    if per_query[NOCTX[0]]:
        print(f"\nNo context vs raw k=10 (separate planned test, not Holm-adjusted)")
        for only in [None] + ([p[0] for p in parts] if len(parts) > 1 else []):
            keys = sorted(k for k in set(per_query[NOCTX[0]]) & set(per_query["raw k=10"])
                          if only is None or k[0] == only)
            if not keys:
                continue
            xs = [per_query[NOCTX[0]][k] for k in keys]
            ys = [per_query["raw k=10"][k] for k in keys]
            d = [x - y for x, y in zip(xs, ys)]
            p = wilcoxon(xs, ys).pvalue if any(d) else 1.0
            print(f"  {('all' if only is None else only):<10} n={len(keys):<4} "
                  f"no context {sum(xs) / len(xs):.3f}  raw k=10 {sum(ys) / len(ys):.3f}  "
                  f"better {sum(v > 0 for v in d):>3} worse {sum(v < 0 for v in d):>3}  p = {p:.2e}")
    if suite == "RestBench":
        for part in parts:
            significance(per_query, f"only {part[0]}", only=part[0])

    try:
        sys.path.insert(0, "src")
        import check_schema as cs
        from benchmark_loader import load_benchmark
    except Exception as error:
        print(f"\n(schema part skipped: {error})")
        return
    print(f"\n{suite}: required body fields / query parameters, unknown keys, pooled over all runs\n")
    output = suite == "RestBench"
    header = f"{'Condition':<14}" + "".join(f"{p[0]:>17}" for p in parts)
    print(header + (f"   output: reads / r-unk per part" if output else ""))
    indices = {p[0]: load_indices(p, cs, load_benchmark) for p in parts}
    # the uncompressed reference: cached k=10 code, identical in every file
    rows = ([("raw k=10", "reqschema", "code_without")]
            + [(n, l, "code_with") for n, l in CONDITIONS.items()]
            + [(NOCTX[0], NOCTX[1], "code_with")])
    for name, label, arm in rows:
        row, out = f"{name:<14}", ""
        for part in parts:
            if indices[part[0]] is None:
                row += f"{'-':>17}"
                continue
            body, query, unknown, reads, r_unk, found = schema_coverage(part, label, indices[part[0]], cs, arm)
            row += f"   {body:.2f}/{query:.2f}/{unknown:.2f}" if found else f"{'-':>17}"
            if output and found:
                out += f"   {reads:>4} / {r_unk:.2f}"
        print(row + out)
    print("\nper cell: required body fields set / required query parameters set / unknown keys"
          + ("\noutput: response keys read / share not in the response schema" if output else "")
          + ("\n(instance 1: last run only, otherwise three runs)" if suite == "SOCBench-D" else ""))


def main():
    wanted = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    for suite, parts in SUITES.items():
        if wanted and wanted not in suite.lower().replace("-", ""):
            continue
        evaluate(suite, parts)
        print()


if __name__ == "__main__":
    main()
