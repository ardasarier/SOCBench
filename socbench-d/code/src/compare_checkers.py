"""
Two independent implementations of the same schema check, on the same code.

  check_schema.py         -- our shallow AST resolver (one assignment per
                             name, no control flow)
  socbenchsc.compliance   -- SOCBench-SC's reaching definition analysis,
                             extended to keep every call parameter

Both use the same definitions (required fields, $ref bodies excluded,
solution endpoints only), so differences come from how the code is
analysed, not from what is counted. Agreement validates the measurement;
disagreements show which analysis misses what.

SOCBench-SC is imported from ../../socbench-sc/src unless SOCBENCHSC_SRC
points elsewhere.

Run from socbench-d/code/:
    python src/compare_checkers.py                     # all generated files
    python src/compare_checkers.py budget-reqschema    # files containing this
"""
import glob
import json
import logging
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.environ.get("SOCBENCHSC_SRC", "../../socbench-sc/src"))
logging.disable(logging.WARNING)            # SOCBench-SC logs every unsupported construct

import check_schema as ours                  # noqa: E402
from benchmark_loader import load_benchmark  # noqa: E402
from socbenchsc.analysis import Analysis     # noqa: E402
from socbenchsc import compliance            # noqa: E402


def ratio(a, b):
    return f"{a / b:.3f}" if b else "  -  "


def sc_check(records, arm, indices):
    total = Counter()
    for rec in records:
        index, ambiguous = indices[rec["queryset"]]
        try:
            invocations = Analysis(rec[arm]).perform_invocation_analysis()
        except Exception:                    # NotImplementedError, syntax errors, ...
            total["failed"] += 1
            continue
        total += compliance.check(invocations, index, ambiguous,
                                  solution=set(rec["solution"]), base_path="/v1")
    return total


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    files = sorted(p for p in glob.glob("data/generated/socbenchd_*.json") if pattern in p)
    print(f"{len(files)} file(s)\n")
    print(f"{'condition':<54}{'arm':<9}"
          f"{'b-req ours':>11}{'b-req SC':>10}{'q-req ours':>11}{'q-req SC':>10}"
          f"{'unk ours':>9}{'unk SC':>8}{'SC fail':>8}{'SC untr':>8}{'SC amb':>7}")

    cache = {}
    for path in files:
        name = os.path.basename(path)[:-5]
        bench = re.match(r"socbenchd_\d", name).group(0)
        if bench not in cache:
            querysets = load_benchmark(bench).queries
            cache[bench] = (
                {qs.name: ours.build_index(qs.openapis) for qs in querysets},
                {qs.name: compliance.build_index(qs.openapis) for qs in querysets},
            )
        our_index, sc_index = cache[bench]
        records = json.load(open(path))

        for arm in ("code_without", "code_with"):
            stats = ours.new_stats()
            for rec in records:
                ours.check(rec[arm], rec["solution"], our_index[rec["queryset"]], stats)
            o = stats["solution"]
            s = sc_check(records, arm, sc_index)
            print(f"{name[:53]:<54}{arm[5:]:<9}"
                  f"{ratio(o['body_required_set'], o['body_required']):>11}"
                  f"{ratio(s['solution_body_required_set'], s['solution_body_required']):>10}"
                  f"{ratio(o['query_required_set'], o['query_required']):>11}"
                  f"{ratio(s['solution_query_required_set'], s['solution_query_required']):>10}"
                  f"{ratio(o['keys_unknown'], o['keys_sent']):>9}"
                  f"{ratio(s['solution_keys_unknown'], s['solution_keys_sent']):>8}"
                  f"{s['failed']:>8}"
                  f"{s['solution_body_untraceable'] + s['solution_query_untraceable']:>8}"
                  f"{s['solution_ambiguous']:>7}")
        print()

    print("b-req / q-req   required body fields / query parameters set (solution endpoints)")
    print("unk             sent keys not defined in the spec / sent keys")
    print("SC fail         programs SOCBench-SC could not analyse (unsupported syntax)")
    print("SC untr         solution calls whose params or body keys are not fully known")
    print("SC amb          solution calls to an endpoint two services define (not judged by SC)")


if __name__ == "__main__":
    main()
