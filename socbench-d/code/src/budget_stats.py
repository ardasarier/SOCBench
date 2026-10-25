"""
Recall at equal token budget, averaged over repeated runs, with paired tests.

The generated/*.json files only keep the LAST run of a condition, but every
run_codegen log prints each query's recall. This reads all runs of each
condition from the logs, reports the per-run means and their spread, and
tests on per-query means -- a single run decided the attention-vs-rule
comparison the wrong way before (0.730 in run 1, 0.756 over three runs).

p-values are Holm-Bonferroni adjusted over all comparisons printed, since
many tests are run on the same 110 queries.

Recall here is the old scorer's, rounded to two decimals in the log; it
agrees with Pesl's regex within 0.004.

Run from socbench-d/code/:
    python src/budget_stats.py
"""
import os
import re
import statistics as st

from scipy.stats import wilcoxon

LOG = "data/logs/codegen_budget_{}.txt"
CONDITIONS = {
    "Regel":        ["reqschema", "reqschema_r2", "reqschema_r3"],
    "AttentionRAG": ["attention40", "attention40_r2", "attention40_r3"],
    "Identity":     ["identity", "identity_r2", "identity_r3"],
    "roh k=20":     ["raw20", "raw20_r2", "raw20_r3"],
}
COMPARISONS = [
    ("Regel", "roh k=10"), ("AttentionRAG", "roh k=10"),
    ("Identity", "roh k=10"), ("roh k=20", "roh k=10"),
    ("AttentionRAG", "Regel"), ("Identity", "Regel"), ("Identity", "AttentionRAG"),
    ("Regel", "roh k=20"), ("AttentionRAG", "roh k=20"), ("Identity", "roh k=20"),
]


def parse(path: str):
    """{query index: recall} for both arms of one run_codegen log."""
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
    """Holm-Bonferroni adjusted p-values, in the original order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def main():
    means, baseline = {}, None
    print(f"{'Bedingung':<14}{'Läufe':<28}{'Mittel':>8}{'± SD':>8}")
    for name, labels in CONDITIONS.items():
        paths = [LOG.format(l) for l in labels if os.path.exists(LOG.format(l))]
        if not paths:
            print(f"{name:<14}(keine Logs)")
            continue
        runs = []
        for p in paths:
            with_, without = parse(p)
            runs.append(with_)
            baseline = baseline or without
        keys = sorted(runs[0])
        run_means = [sum(r[k] for k in keys) / len(keys) for r in runs]
        sd = st.stdev(run_means) if len(run_means) > 1 else float("nan")
        print(f"{name:<14}{str([round(x, 3) for x in run_means]):<28}"
              f"{st.mean(run_means):>8.3f}{sd:>8.3f}")
        means[name] = {k: sum(r[k] for r in runs) / len(runs) for k in keys}
    means["roh k=10"] = baseline
    print(f"{'roh k=10':<14}{'(gecacht, 1 Lauf)':<28}"
          f"{sum(baseline.values()) / len(baseline):>8.3f}")

    rows = []
    for a, b in COMPARISONS:
        if a not in means or b not in means:
            continue
        ks = sorted(set(means[a]) & set(means[b]))
        xs, ys = [means[a][k] for k in ks], [means[b][k] for k in ks]
        d = [x - y for x, y in zip(xs, ys)]
        p = wilcoxon(xs, ys).pvalue if any(d) else 1.0
        rows.append((f"{a} vs {b}", sum(v > 0 for v in d), sum(v < 0 for v in d), p))
    adjusted = holm([r[3] for r in rows])

    print(f"\n{'Vergleich (Mittel je Query)':<32}{'besser':>8}{'schlechter':>12}{'p':>11}{'p (Holm)':>11}")
    for (name, better, worse, p), p_adj in zip(rows, adjusted):
        mark = "  *" if p_adj < 0.05 else ""
        print(f"{name:<32}{better:>8}{worse:>12}{p:>11.2e}{p_adj:>11.2e}{mark}")
    print("\n* signifikant nach Holm-Korrektur (alpha = 0.05)")


if __name__ == "__main__":
    main()