"""
Measures how often the known blind spots of check_schema.py occur.

All four make the checker report a field as MISSING although the code sets it:
  (a) a body/params variable is assigned more than once -- the Resolver keeps
      one assignment per name, so both calls are checked against one dict
  (b) keys are added later: params["x"] = ... or params.update(...)
  (c) query parameters written into the URL (?x=...), which to_path discards
  (d) the same "METHOD /path" in two specs of one sector -- the index keeps
      only one operation

They would only distort the RESULTS if they were much more frequent in the
compressed conditions than in the baseline. This shows whether they are.

Run from socbench-d/code/:
    python src/audit_checker.py
"""
import ast
import json
import re
from collections import Counter

from benchmark_loader import load_benchmark

D = "data/generated/socbenchd_1_k10_all_noanchor_"
G4 = "_n110_gpt-4o-2024-11-20.json"
LU = "_n110_gpt-6-luna.json"
CASES = [
    ("baseline       gpt-4o", D + "k10" + G4, "code_without"),
    ("attention+rule gpt-4o", D + "k10" + G4, "code_with"),
    ("identity       gpt-4o", D + "identity" + G4, "code_with"),
    ("reqschema      gpt-4o", D + "reqschema" + G4, "code_with"),
    ("baseline       luna", D + "k10" + LU, "code_without"),
    ("identity       luna", D + "identity" + LU, "code_with"),
    ("reqschema      luna", D + "reqschema" + LU, "code_with"),
]
KWARGS = ("params", "json", "data")
QUERY_IN_URL = re.compile(r"""["'][^"'\n]*/[^"'\n]*\?[^"'\n]*=""")


def blind_spots(code: str):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    nodes = list(ast.walk(tree))
    passed = {kw.value.id for n in nodes if isinstance(n, ast.Call)
              for kw in n.keywords if kw.arg in KWARGS and isinstance(kw.value, ast.Name)}
    assigned = Counter(t.id for n in nodes if isinstance(n, ast.Assign)
                       for t in n.targets if isinstance(t, ast.Name))
    reused = any(assigned[name] > 1 for name in passed)
    mutated = any(
        (isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
            and t.value.id in passed for t in n.targets))
        or (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "update" and isinstance(n.func.value, ast.Name)
            and n.func.value.id in passed)
        for n in nodes)
    return reused, mutated, bool(QUERY_IN_URL.search(code))


print("Share of generated programs showing each blind spot\n")
print(f"{'':<24}{'n':>5}{'(a) reused':>12}{'(b) mutated':>13}{'(c) ?query':>12}")
for label, path, arm in CASES:
    try:
        records = json.load(open(path))
    except FileNotFoundError:
        print(f"{label:<24}  (file missing)")
        continue
    flags = [f for f in (blind_spots(r[arm]) for r in records) if f is not None]
    n = len(flags)
    share = lambda i: f"{100 * sum(f[i] for f in flags) / n:.0f}%"
    print(f"{label:<24}{n:>5}{share(0):>12}{share(1):>13}{share(2):>12}")

bench = load_benchmark("socbenchd_1")
total = colliding = 0
for qs in bench.queries:
    seen = Counter()
    for oas in qs.openapis:
        spec = json.loads(oas) if isinstance(oas, str) else oas
        for path, ops in spec.get("paths", {}).items():
            for verb in ops:
                if verb.lower() in {"get", "post", "put", "patch", "delete"}:
                    seen[f"{verb.upper()} {path}"] += 1
    total += len(seen)
    colliding += sum(1 for c in seen.values() if c > 1)
print(f"\n(d) endpoints colliding across the 5 specs of a sector: {colliding} of {total}")