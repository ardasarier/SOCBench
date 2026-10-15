"""
Static schema check of generated composition code.

SOCBench-SC extracts WHICH endpoints a composition calls; Pesl et al. list
schema compliance as future work. This checks HOW they are called: for each
HTTP call whose endpoint can be resolved, are the operation's required query
parameters and request-body fields set, and does the code send keys the spec
does not define?

Purely static (AST): no execution, no API, free on the code already saved in
data/generated/. Calls it cannot resolve -- URLs built inside helper
functions, bodies passed through untraceable variables -- are counted
separately, so the share of analysable calls is reported, not hidden.

Scope: query parameters and JSON bodies. Path and header parameters are not
checked.

Run from socbench-d/code/:
    python src/check_schema.py
"""
import ast
import glob
import json
import os
import re
from collections import defaultdict

from benchmark_loader import load_benchmark

GENERATED_DIR = "data/generated"
HTTP_VERBS = {"get", "post", "put", "patch", "delete"}
BODY_KWARGS = ("json", "data")


# --- spec side --------------------------------------------------------------

def build_index(openapis) -> dict:
    """{"METHOD /path": operation} over all specs of one queryset."""
    index = {}
    for oas in openapis:
        spec = json.loads(oas) if isinstance(oas, str) else oas
        for path, ops in spec.get("paths", {}).items():
            for verb, op in ops.items():
                if verb.lower() in HTTP_VERBS:
                    index[f"{verb.upper()} {path}"] = op
    return index


def match_template(method: str, path: str, index: dict):
    """Most specific template matching a concrete path, else None. Same rule
    as scoring.match_to_template: literal segments beat {parameters}."""
    segs = path.lstrip("/").split("/")
    best, best_score = None, -1
    for key in sorted(index):
        verb, _, tpath = key.partition(" ")
        if verb != method:
            continue
        tsegs = tpath.lstrip("/").split("/")
        if len(tsegs) != len(segs):
            continue
        score, ok = 0, True
        for t, s in zip(tsegs, segs):
            if t == s:
                score += 1
            elif not t.startswith("{"):
                ok = False
                break
        if ok and score > best_score:
            best, best_score = key, score
    return best


def requirements(op: dict) -> dict:
    query = {p["name"]: p for p in op.get("parameters", []) if p.get("in") == "query"}
    body = op.get("requestBody") or {}
    schema = body.get("content", {}).get("application/json", {}).get("schema", {})
    return {
        "query_known": set(query),
        "query_required": {n for n, p in query.items() if p.get("required")},
        "body_known": set(schema.get("properties", {})),
        "body_required": set(schema.get("required", [])),
        "body_mandatory": bool(body.get("required")),
    }


# --- code side --------------------------------------------------------------

class Resolver:
    """Collects simple assignments so URLs and dicts passed by name can be
    traced. Deliberately shallow: one assignment per name, no control flow."""

    def __init__(self, tree: ast.AST):
        self.values = {}
        self.clients = {"requests"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.values[target.id] = node.value
                        if self._is_session(node.value):
                            self.clients.add(target.id)
            elif isinstance(node, ast.With):
                for item in node.items:
                    if (isinstance(item.optional_vars, ast.Name)
                            and self._is_session(item.context_expr)):
                        self.clients.add(item.optional_vars.id)

    @staticmethod
    def _is_session(node) -> bool:
        """requests.Session() / requests.session()"""
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr.lower() == "session")

    def string(self, node, depth=0):
        if depth > 5 or node is None:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in self.values:
            return self.string(self.values[node.id], depth + 1)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = self.string(node.left, depth + 1), self.string(node.right, depth + 1)
            return left + right if left is not None and right is not None else None
        if isinstance(node, ast.JoinedStr):
            parts = []
            for v in node.values:
                if isinstance(v, ast.Constant):
                    parts.append(str(v.value))
                else:                       # FormattedValue
                    resolved = self.string(v.value, depth + 1)
                    parts.append(resolved if resolved is not None else "{}")
            return "".join(parts)
        return None

    def keys(self, node, depth=0):
        """Literal string keys of a dict, or None if not traceable."""
        if depth > 5 or node is None:
            return None
        if isinstance(node, ast.Dict):
            return {k.value for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        if isinstance(node, ast.Name) and node.id in self.values:
            return self.keys(self.values[node.id], depth + 1)
        return None


def http_calls(code: str):
    """Yields (method, url_or_None, kwargs) per HTTP call. None if unparseable."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    resolver = Resolver(tree)
    calls = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        receiver = node.func.value
        if not (isinstance(receiver, ast.Name) and receiver.id in resolver.clients):
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        attr = node.func.attr.lower()
        if attr in HTTP_VERBS:
            method = attr.upper()
            url_node = node.args[0] if node.args else kw.get("url")
        elif attr == "request":
            method_node = node.args[0] if node.args else kw.get("method")
            method = (resolver.string(method_node) or "").upper()
            url_node = node.args[1] if len(node.args) > 1 else kw.get("url")
            if method.lower() not in HTTP_VERBS:
                continue
        else:
            continue
        calls.append((method, resolver.string(url_node), kw, resolver))
    return calls


def to_path(url: str):
    path = re.sub(r"^https?://[^/]+", "", url).split("?")[0]
    return path if path.startswith("/") else None


# --- scoring ----------------------------------------------------------------

def check(code: str, solution: list, index: dict, stats: dict) -> None:
    calls = http_calls(code)
    if calls is None:
        stats["unparseable"] += 1
        return
    for method, url, kw, resolver in calls:
        stats["calls"] += 1
        path = to_path(url) if url else None
        if path is None:
            stats["unresolved"] += 1
            continue
        key = match_template(method, path, index)
        if key is None and path.startswith("/v1/"):
            key = match_template(method, path[3:], index)
        if key is None:
            stats["unknown_endpoint"] += 1
            continue

        scope = "solution" if key in solution else "other"
        s = stats[scope]
        s["calls"] += 1
        req = requirements(index[key])

        # query parameters
        if "params" in kw:
            keys = resolver.keys(kw["params"])
            if keys is None:
                s["query_untraceable"] += 1
                keys = None
        else:
            keys = set()
        if keys is not None:
            s["query_required"] += len(req["query_required"])
            s["query_required_set"] += len(req["query_required"] & keys)
            s["keys_sent"] += len(keys)
            s["keys_unknown"] += len(keys - req["query_known"])

        # body
        body_node = next((kw[k] for k in BODY_KWARGS if k in kw), None)
        if body_node is not None:
            keys = resolver.keys(body_node)
            if keys is None:
                s["body_untraceable"] += 1
                continue
        else:
            keys = set()
            if not req["body_mandatory"]:
                continue
        s["body_required"] += len(req["body_required"])
        s["body_required_set"] += len(req["body_required"] & keys)
        s["keys_sent"] += len(keys)
        s["keys_unknown"] += len(keys - req["body_known"])


def new_stats() -> dict:
    stats = defaultdict(int)
    stats["solution"] = defaultdict(int)
    stats["other"] = defaultdict(int)
    return stats


def ratio(a, b):
    return f"{a / b:.3f}" if b else "  -  "


if __name__ == "__main__":
    indices = {}
    files = sorted(glob.glob(os.path.join(GENERATED_DIR, "*.json")))
    print(f"{len(files)} condition(s) in {GENERATED_DIR}")
    print("Scope: calls to endpoints in the SOLUTION (correctly selected ones).\n")
    print(f"{'condition':<52}{'arm':<9}{'calls/q':>8}{'resolv':>8}"
          f"{'q-req':>7}{'b-req':>7}{'unknown':>8}")

    for path in files:
        name = os.path.basename(path)[:-5]
        bench_name = re.match(r"(socbenchd_\d|restbench)", name)
        if not bench_name or bench_name.group(1) == "restbench":
            continue
        bench_name = bench_name.group(1)
        if bench_name not in indices:
            bench = load_benchmark(bench_name)
            indices[bench_name] = {qs.name: build_index(qs.openapis) for qs in bench.queries}

        records = json.load(open(path))
        for arm in ("code_without", "code_with"):
            stats = new_stats()
            for rec in records:
                check(rec[arm], rec["solution"], indices[bench_name][rec["queryset"]], stats)
            s = stats["solution"]
            resolved = stats["calls"] - stats["unresolved"]
            print(f"{name[:51]:<52}{arm[5:]:<9}"
                  f"{stats['calls'] / len(records):>8.2f}"
                  f"{ratio(resolved, stats['calls']):>8}"
                  f"{ratio(s['query_required_set'], s['query_required']):>7}"
                  f"{ratio(s['body_required_set'], s['body_required']):>7}"
                  f"{ratio(s['keys_unknown'], s['keys_sent']):>8}")
        print()

    print("calls/q  HTTP calls found per query")
    print("resolv   share whose URL could be resolved statically")
    print("q-req    required query parameters set / required (solution endpoints)")
    print("b-req    required body fields set / required (solution endpoints)")
    print("unknown  sent keys not defined in the spec / all sent keys")