"""
End-to-end: the equal-budget comparison on further benchmarks, unattended.

Per benchmark, in order:
  1. contexts  k=10 request-schema rule  -> reference (its raw chunks and
                                            per-query budgets)
               k=50 request-schema rule  -> source for the rule and raw k=20
               k=50 identity             -> source for identity
               k=40 AttentionRAG         -> generated hint
               k=40 AttentionRAG         -> fixed hint
  2. budget    one equal-budget context file per condition
  3. codegen   every condition --runs times, one file per run

Every step is idempotent, so the script can simply be started again after a
crash or an interrupted night: prepare_contexts skips complete outputs
(--skip-existing), make_budget_contexts is deterministic, and run_codegen
resumes numbered runs. A failing benchmark is logged and skipped; the others
still run.

File names are taken from what each step reports it wrote ("writing to",
"already complete:", "wrote"), never guessed from a pattern, so a pilot with
--max-queries cannot pick up the full-size files or vice versa.

Run from socbench-d/code/:
    python -u src/run_e2e.py --benchmarks socbenchd_2 socbenchd_3
    python -u src/run_e2e.py --benchmarks socbenchd_2 --max-queries 1 --runs 1   # pilot
    python -u src/run_e2e.py --benchmarks restbench --api tmdb
Logs: data/logs/e2e/<benchmark>/
"""
import argparse
import os
import re
import subprocess
import sys
import time

FIXED_HINT = "The API endpoint required for this task is"
PY = sys.executable


def run(cmd: list, log: str) -> str:
    """Runs cmd, appends its output to log, returns the output. Raises on failure."""
    os.makedirs(os.path.dirname(log), exist_ok=True)
    shown = " ".join(f"'{c}'" if " " in c else c for c in cmd[1:])
    print(f"  $ {shown}", flush=True)
    started = time.time()
    lines = []
    with open(log, "a") as f:
        f.write(f"\n### {shown}\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            f.write(line)
            f.flush()
            lines.append(line)
        proc.wait()
    print(f"    exit {proc.returncode}, {(time.time() - started) / 60:.1f} min  ({log})", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"step failed (exit {proc.returncode}): {shown} -- see {log}")
    return "".join(lines)


def reported_path(output: str, *markers: str) -> str:
    """The last .json path a step reported after one of the markers."""
    found = re.findall(r"(?:%s) (\S+\.json)" % "|".join(map(re.escape, markers)), output)
    if not found:
        raise RuntimeError(f"no output path reported after {markers}")
    return found[-1]


def one_benchmark(bench: str, api: str, runs: int, max_queries) -> None:
    scope = f"{bench}-{api}" if bench == "restbench" else bench
    logs = f"data/logs/e2e/{scope}"
    base = [PY, "-u", "src/prepare_contexts.py", "--benchmark", bench, "--skip-existing"]
    if bench == "restbench":
        base += ["--api", api]
    if max_queries:
        base += ["--max-queries", str(max_queries)]

    def prepare(name: str, extra: list) -> str:
        out = run(base + extra, f"{logs}/prepare_{name}.txt")
        return reported_path(out, "writing to", "already complete:")

    print(f"\n=== {scope}: contexts", flush=True)
    reference = prepare("k10_reqschema", ["--top-k", "10", "--keep-fields", "reqschema"])
    src_rule = prepare("k50_reqschema", ["--top-k", "50", "--keep-fields", "reqschema"])
    src_identity = prepare("k50_identity", ["--top-k", "50", "--identity-only"])
    src_attention = prepare("k40_attention", ["--top-k", "40"])
    src_fixed = prepare("k40_fixhint", ["--top-k", "40", "--fixed-hint", FIXED_HINT])

    print(f"\n=== {scope}: equal-budget contexts", flush=True)
    prefix = f"{api}-" if bench == "restbench" else ""
    conditions = [
        ("reqschema", src_rule, []),
        ("raw20", src_rule, ["--raw-k", "20"]),
        ("identity", src_identity, []),
        ("attention40", src_attention, []),
        ("attention40-fixhint", src_fixed, []),
    ]
    budget_files = []
    for label, source, extra in conditions:
        out = run([PY, "-u", "src/make_budget_contexts.py", "--reference", reference,
                   "--source", source, "--label", prefix + label] + extra,
                  f"{logs}/budget_{label}.txt")
        budget_files.append((label, reported_path(out, "wrote")))

    print(f"\n=== {scope}: codegen, {runs} run(s) per condition", flush=True)
    for label, path in budget_files:
        for r in range(1, runs + 1):
            run([PY, "-u", "src/run_codegen.py", path, "--run", str(r)],
                f"{logs}/codegen_{label}_r{r}.txt")


def main():
    ap = argparse.ArgumentParser(description="Unattended equal-budget runs.")
    ap.add_argument("--benchmarks", nargs="+", required=True,
                    help="e.g. socbenchd_2 socbenchd_3, or restbench (with --api)")
    ap.add_argument("--api", default="tmdb", help="RestBench only: spotify | tmdb")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-queries", type=int, default=None, help="pilot: queries per set")
    args = ap.parse_args()

    started = time.time()
    failed = []
    for bench in args.benchmarks:
        try:
            one_benchmark(bench, args.api, args.runs, args.max_queries)
        except Exception as error:          # keep going with the next benchmark
            print(f"\n!!! {bench} aborted: {error}", flush=True)
            failed.append(bench)

    print(f"\n=== done in {(time.time() - started) / 3600:.1f} h"
          + (f", FAILED: {' '.join(failed)}" if failed else ", all benchmarks complete"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
