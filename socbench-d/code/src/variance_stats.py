"""
Variance and stability of the equal-budget results.

Question: pruning raises recall -- does it also make results less stable?

1. Run to run, per query. The composition model is not deterministic even
   at temperature 0. For every query and condition, the recall of the
   repeated runs is compared: mean per-query SD, share of queries whose
   recall changes at all between runs, and mean range. Whether conditions
   differ in this is tested with the Friedman test over the conditions
   (as Pesl et al. use it to compare chunking strategies), then pairwise
   against the rule with Wilcoxon and Holm.

2. Across domains (SOCBench-D). Following the stability analysis of Pesl
   et al. (TSC 2026): per condition, the mean recall of each of the 11 GICS
   sectors (all instances pooled), their SD and coefficient of variation
   CV = SD / mean, and in how many sectors the condition beats
   uncompressed k=10.

Reads the same run_codegen logs as e2e_stats.py.

Run from socbench-d/code/:
    python src/variance_stats.py
"""
import os
import re
import statistics as st
import sys
from collections import defaultdict

from scipy.stats import friedmanchisquare, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2e_stats import CONDITIONS, SUITES, holm, log_paths  # noqa: E402


def parse(path: str):
    """{query index: (queryset, recall with, recall without)}"""
    out, idx, sector, without = {}, None, None, None
    for line in open(path):
        m = re.match(r"\[(\d+)/\d+\] \(([^)]*)\)", line)
        if m:
            idx, sector = int(m.group(1)), m.group(2)
            continue
        m = re.match(r"\s+without: recall ([\d.]+)", line)
        if m and idx:
            without = float(m.group(1))
            continue
        m = re.match(r"\s+with:\s+recall ([\d.]+)", line)
        if m and idx:
            out[idx] = (sector, float(m.group(1)), without)
    return out


def budget_depth(part, label: str):
    """Mean depth from the last budget construction logged for this part."""
    path = f"data/logs/e2e/{part[1]}/budget_{label}.txt"
    if not os.path.exists(path):
        return None
    found = re.findall(r"mean depth \(chunks\):\s+([\d.]+)", open(path).read())
    return float(found[-1]) if found else None


def collect(parts):
    runs = defaultdict(dict)        # condition -> {(part, idx): [recall per run]}
    sector = {}                     # (part, idx) -> sector
    base = {}                       # (part, idx) -> uncompressed k=10 recall
    for part in parts:
        for name, label in CONDITIONS.items():
            for path in log_paths(part, label):
                if not os.path.exists(path):
                    continue
                for idx, (sec, with_, without) in parse(path).items():
                    key = (part[0], idx)
                    runs[name].setdefault(key, []).append(with_)
                    sector[key] = sec
                    base[key] = without
    return runs, sector, base


def run_to_run(suite, parts, runs):
    print(f"\n{suite}: run-to-run spread per query\n")
    print(f"{'Condition':<14}{'depth':>7}{'Recall':>8}{'SD/Query':>10}"
          f"{'unstable':>10}{'range':>9}{'n':>6}")
    sds = {}
    for name, label in CONDITIONS.items():
        per_query = {k: v for k, v in runs[name].items() if len(v) >= 2}
        if not per_query:
            continue
        sds[name] = {k: st.stdev(v) for k, v in per_query.items()}
        ranges = [max(v) - min(v) for v in per_query.values()]
        unstable = sum(r > 0 for r in ranges) / len(ranges)
        depths = [d for d in (budget_depth(p, label) for p in parts) if d is not None]
        depth = f"{sum(depths) / len(depths):.1f}" if depths else "-"
        mean_recall = st.mean(st.mean(v) for v in per_query.values())
        print(f"{name:<14}{depth:>7}{mean_recall:>8.3f}{st.mean(sds[name].values()):>10.3f}"
              f"{unstable:>9.1%}{st.mean(ranges):>9.3f}{len(per_query):>6}")
    print("\nSD/Query: mean SD of a query's recall over the runs | unstable: share"
          "\nof queries whose recall changes between runs | range: mean range"
          "\ndepth: mean number of chunks in the budget (from the budget logs, where available)")

    names = list(sds)
    common = sorted(set.intersection(*(set(sds[n]) for n in names))) if names else []
    if len(names) >= 3 and common:
        stat, p = friedmanchisquare(*[[sds[n][k] for k in common] for n in names])
        print(f"\nFriedman over {len(names)} conditions ({len(common)} queries): "
              f"chi2 = {stat:.1f}, p = {p:.2e}")
    if "rule" in sds:
        rows = []
        for name in names:
            if name == "rule":
                continue
            keys = sorted(set(sds[name]) & set(sds["rule"]))
            xs, ys = [sds[name][k] for k in keys], [sds["rule"][k] for k in keys]
            d = [x - y for x, y in zip(xs, ys)]
            p = wilcoxon(xs, ys).pvalue if any(d) else 1.0
            rows.append((name, sum(v > 0 for v in d), sum(v < 0 for v in d), p))
        adjusted = holm([r[3] for r in rows])
        print(f"\n{'SD/query vs rule':<28}{'larger':>8}{'smaller':>9}{'p (Holm)':>11}")
        for (name, more, less, _), p_adj in zip(rows, adjusted):
            print(f"{name:<28}{more:>8}{less:>9}{p_adj:>11.2e}{'  *' if p_adj < 0.05 else ''}")


def across_domains(suite, runs, sector, base):
    names = sorted({sector[k] for k in sector})
    print(f"\n{suite}: stability across domains ({len(names)} sectors, all instances pooled)\n")
    print(f"{'Condition':<14}{'mean':>8}{'SD':>8}{'CV':>8}{'min':>8}{'max':>8}{'> raw k=10':>12}")
    table = {"raw k=10": base}
    for name in CONDITIONS:
        table[name] = {k: st.mean(v) for k, v in runs[name].items()}
    base_by_sector = {}
    for name, values in table.items():
        by_sector = defaultdict(list)
        for k, v in values.items():
            by_sector[sector[k]].append(v)
        means = {s: st.mean(v) for s, v in by_sector.items()}
        if name == "raw k=10":
            base_by_sector = means
        overall = st.mean(means.values())
        sd = st.stdev(means.values()) if len(means) > 1 else float("nan")
        better = ("-" if name == "raw k=10" else
                  f"{sum(means[s] > base_by_sector[s] for s in means)}/{len(means)}")
        print(f"{name:<14}{overall:>8.3f}{sd:>8.3f}{sd / overall:>8.3f}"
              f"{min(means.values()):>8.3f}{max(means.values()):>8.3f}{better:>12}")
    print("\nmean/SD/CV over the sector means; CV = SD / mean (as in Pesl et al.)")


def main():
    for suite, parts in SUITES.items():
        runs, sector, base = collect(parts)
        if not any(runs.values()):
            continue
        print("=" * 80)
        run_to_run(suite, parts, runs)
        if suite == "SOCBench-D":
            across_domains(suite, runs, sector, base)
        print()


if __name__ == "__main__":
    main()
