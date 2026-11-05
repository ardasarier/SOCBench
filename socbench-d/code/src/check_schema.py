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

Scope: query parameters and JSON bodies on the INPUT side. Path and header
parameters are not checked.

OUTPUT side: which keys the code READS from a response -- data["id"],
data.get("id"), r.json()["id"], and element reads data[0]["id"] or
`for item in data: item["id"]` -- and whether the response schema of the
first 2xx response defines them. A read the schema does not define is a
field the code expects but will not get. Reads are attributed to a call by
following simple assignments in line order; control flow is not followed.
Responses without a usable schema (none, no properties, or a $ref) are
counted separately.

Bodies whose schema is only a $ref into components/schemas are excluded from
both body measures and counted separately. The endpoint split does not carry
components, so the referenced fields are in no chunk -- not even uncompressed.
Counting them would measure a limit of chunking, not of pruning, and every key
sent to such a body would count as unknown (16 % of solution endpoints with a
body on socbenchd_1).

Run from socbench-d/code/:
    python src/check_schema.py
"""
import ast
import glob
import json
import os
import random
import re
import sys
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
        "body_ref": "$ref" in schema,
    }


def response_schema(op: dict):
    """(kind, properties) of the first 2xx JSON response, kind "object" or
    "array" (properties of its items); or (status, None) with status "none"
    (no schema or no declared properties) or "ref" (schema only a $ref)."""
    for code in sorted(op.get("responses", {}), key=str):
        if not str(code).startswith("2"):
            continue
        resp = op["responses"][code] or {}
        schema = resp.get("content", {}).get("application/json", {}).get("schema")
        if not schema:
            return "none", None
        if "$ref" in schema:
            return "ref", None
        if schema.get("type") == "array" or "items" in schema:
            items = schema.get("items", {})
            if "$ref" in items:
                return "ref", None
            props = set(items.get("properties", {}))
            return ("array", props) if props else ("none", None)
        props = set(schema.get("properties", {}))
        return ("object", props) if props else ("none", None)
    return "none", None


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


class ResponseFlow:
    """Which keys the code reads from the response of a given call.

    Follows simple assignments (one Name target) in line order: the value a
    name refers to at line L is its latest assignment before L. Control flow
    is not followed, matching the shallow Resolver."""

    def __init__(self, tree: ast.AST):
        self.assigns = defaultdict(list)            # name -> [(line, value)]
        self.loops = []                             # (var, iterable name, first, last line)
        self.reads = []                             # (line, base node, key)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                self.assigns[node.targets[0].id].append((node.lineno, node.value))
            elif (isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                    and isinstance(node.iter, ast.Name)):
                self.loops.append((node.target.id, node.iter.id, node.lineno,
                                   getattr(node, "end_lineno", node.lineno)))
            elif (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)):
                self.reads.append((node.lineno, node.value, node.slice.value))
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get" and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                self.reads.append((node.lineno, node.func.value, node.args[0].value))
        for entries in self.assigns.values():
            entries.sort(key=lambda e: e[0])

    def _owner(self, name: str, line: int):
        """(line, value) of the latest assignment to name before line."""
        found = None
        for assign_line, value in self.assigns.get(name, []):
            if assign_line < line:
                found = (assign_line, value)
        return found

    @staticmethod
    def _json_of(node):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "json"):
            return node.func.value
        return None

    def _is_response(self, node, line, call) -> bool:
        if node is call:
            return True
        if isinstance(node, ast.Name):
            owner = self._owner(node.id, line)
            return owner is not None and owner[1] is call
        return False

    def _is_data(self, node, line, call) -> bool:
        inner = self._json_of(node)
        if inner is not None:
            return self._is_response(inner, line, call)
        if isinstance(node, ast.Name):
            owner = self._owner(node.id, line)
            if owner is not None:
                inner = self._json_of(owner[1])
                return inner is not None and self._is_response(inner, owner[0], call)
        return False

    def keys_read(self, call):
        """(keys read on the response body, keys read on its elements)"""
        top, element = [], []
        for line, base, key in self.reads:
            if self._is_data(base, line, call):
                top.append((line, key))
            elif (isinstance(base, ast.Subscript) and isinstance(base.slice, ast.Constant)
                    and isinstance(base.slice.value, int)
                    and self._is_data(base.value, line, call)):
                element.append((line, key))
            elif isinstance(base, ast.Name):
                for var, iterable, first, last in self.loops:
                    if (var == base.id and first <= line <= last
                            and self._is_data(ast.Name(id=iterable), first, call)):
                        element.append((line, key))
                        break
        return top, element


def _http_calls(code: str):
    """(tree, [(call node, method, url_or_None, kwargs, resolver)]), or None."""
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
        calls.append((node, method, resolver.string(url_node), kw, resolver))
    return tree, calls


def http_calls(code: str):
    """Yields (method, url_or_None, kwargs, resolver) per HTTP call. None if
    unparseable. Kept for audit_checker.py; check() uses _http_calls."""
    parsed = _http_calls(code)
    if parsed is None:
        return None
    return [(m, u, kw, r) for _, m, u, kw, r in parsed[1]]


def to_path(url: str):
    path = re.sub(r"^https?://[^/]+", "", url).split("?")[0]
    return path if path.startswith("/") else None


# --- scoring ----------------------------------------------------------------

SAMPLES = []                    # filled when run with --sample


def check(code: str, solution: list, index: dict, stats: dict, label: str = "",
          base_path: str = "") -> None:
    """base_path: the API's path prefix (RestBench: /v1 for Spotify, /3 for
    TMDB), removed before matching against the specified paths."""
    parsed = _http_calls(code)
    if parsed is None:
        stats["unparseable"] += 1
        return
    tree, calls = parsed
    flow = ResponseFlow(tree)
    lines = code.splitlines()
    for node, method, url, kw, resolver in calls:
        stats["calls"] += 1
        path = to_path(url) if url else None
        if path is None:
            stats["unresolved"] += 1
            continue
        prefix = base_path.rstrip("/")
        if prefix and path.startswith(prefix + "/"):
            path = path[len(prefix):]
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

        # output side: keys read from this call's response
        kind, props = response_schema(index[key])
        if kind in ("none", "ref"):
            s[f"resp_{kind}"] += 1
        else:
            top, element = flow.keys_read(node)
            # reads on the body of an array response, or on elements of an
            # object response, address a structure the schema does not have
            checked = ([(l, k, k in props) for l, k in top] if kind == "object"
                       else [(l, k, k in props) for l, k in element] + [(l, k, False) for l, k in top])
            if kind == "object":
                checked += [(l, k, False) for l, k in element]
            s["resp_calls"] += 1
            s["reads"] += len(checked)
            s["reads_unknown"] += sum(1 for _, _, known in checked if not known)
            if scope == "solution":
                for line, k, known in checked:
                    SAMPLES.append((label, key, k, known, sorted(props),
                                    lines[line - 1].strip() if 0 < line <= len(lines) else ""))

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

        # body -- a schema that is only a $ref has no visible fields; see docstring
        if req["body_ref"]:
            s["body_ref_calls"] += 1
            continue
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
          f"{'q-req':>7}{'b-req':>7}{'unknown':>8}{'$ref':>6}"
          f"{'reads':>7}{'r-unk':>7}{'no-sch':>7}")

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
                check(rec[arm], rec["solution"], indices[bench_name][rec["queryset"]], stats,
                      label=f"{name} {arm[5:]}")
            s = stats["solution"]
            resolved = stats["calls"] - stats["unresolved"]
            print(f"{name[:51]:<52}{arm[5:]:<9}"
                  f"{stats['calls'] / len(records):>8.2f}"
                  f"{ratio(resolved, stats['calls']):>8}"
                  f"{ratio(s['query_required_set'], s['query_required']):>7}"
                  f"{ratio(s['body_required_set'], s['body_required']):>7}"
                  f"{ratio(s['keys_unknown'], s['keys_sent']):>8}"
                  f"{s['body_ref_calls']:>6}"
                  f"{s['reads']:>7}"
                  f"{ratio(s['reads_unknown'], s['reads']):>7}"
                  f"{s['resp_none'] + s['resp_ref']:>7}")
        print()

    print("calls/q  HTTP calls found per query")
    print("resolv   share whose URL could be resolved statically")
    print("q-req    required query parameters set / required (solution endpoints)")
    print("b-req    required body fields set / required (solution endpoints)")
    print("unknown  sent keys not defined in the spec / all sent keys")
    print("$ref     solution calls whose body schema is only a $ref (excluded from b-req and unknown)")
    print("reads    keys read from responses of solution calls that have a response schema")
    print("r-unk    reads the response schema does not define / reads (output side)")
    print("no-sch   solution calls whose response has no usable schema (none or $ref)")

    if "--sample" in sys.argv:
        n = int(sys.argv[sys.argv.index("--sample") + 1]) if len(sys.argv) > sys.argv.index("--sample") + 1 else 20
        random.seed(0)
        print(f"\n--- {n} random response reads (solution endpoints), for checking by hand ---")
        for label, key, k, known, props, line in random.sample(SAMPLES, min(n, len(SAMPLES))):
            print(f"\n[{label}] {key}\n  read:   {k!r}  -> {'defined' if known else 'NOT IN SCHEMA'}"
                  f"\n  schema: {props}\n  code:   {line}")
