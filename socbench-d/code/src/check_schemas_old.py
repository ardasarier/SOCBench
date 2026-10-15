"""
Static schema check of generated composition code against the OpenAPI specs.

SOCBench-SC extracts WHICH endpoints a composition calls; it does not check
whether the calls are built correctly -- the paper lists schema compliance as
future work. This checks, per call:

  - required query parameters and required request-body fields: are they set?
  - keys passed in params= / json=: do they exist in the spec at all?

Static and free: runs on the saved code in data/generated/, no API calls.

Scope, stated honestly:
  - calls on any object with .get/.post/.put/.patch/.delete/.request count
    (requests.*, session.*), provided the URL resolves to a path
  - URLs from string literals, f-strings, + concatenation, and names assigned
    once to such values; unresolved parts become path placeholders
  - params=/json=/data= read when they are dict literals or names assigned to
    one; anything else is counted as unresolvable rather than guessed
  - header parameters are not checked

Run from socbench-d/code/:
    python src/check_schemas_old.py
"""
import ast
import glob
import json
import os
import re
from urllib.parse import parse_qs, urlparse

from benchmark_loader import load_benchmark

GENERATED_DIR = "data/generated"
OUT_DIR = "data/schema_check"
HTTP_METHODS = {"get", "post", "put", "patch", "delete"}


# ---------------------------------------------------------------- spec side

def build_spec_index(openapis) -> dict:
    """{"VERB /path": {"query": {name: required}, "body_props": set,
                       "body_required": list, "body_mandatory": bool}}"""
    index = {}
    for oas in openapis:
        spec = json.loads(oas) if isinstance(oas, str) else oas
        for path, ops in spec.get("paths", {}).items():
            shared = ops.get("parameters", [])
            for verb, op in ops.items():
                if verb not in HTTP_METHODS:
                    continue
                query = {p["name"]: bool(p.get("required"))
                         for p in shared + op.get("parameters", [])
                         if p.get("in") == "query"}
                body = op.get("requestBody", {})
                schema = (body.get("content", {}).get("application/json", {})
                          .get("schema", {}))
                index[f"{verb.upper()} {path}"] = {
                    "query": query,
                    "body_props": set(schema.get("properties", {})),
                    "body_required": list(schema.get("required", [])),
                    "body_mandatory": bool(body.get("required")),
                }
    return index


def template_regex(template: str):
    verb, path = template.split(" ", 1)
    body = re.sub(r"\{[^}]*\}", "[^/]+", re.escape(path).replace(r"\{", "{").replace(r"\}", "}"))
    return verb, re.compile(r"^(/v1)?" + body + r"/?$")


def match_endpoint(method: str, path: str, compiled: list):
    """Most specific matching template (most literal characters), or None."""
    best, best_len = None, -1
    for template, verb, rx, literal_len in compiled:
        if verb == method and rx.match(path) and literal_len > best_len:
            best, best_len = template, literal_len
    return best


# ---------------------------------------------------------------- code side

def collect_assignments(tree) -> dict:
    env = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            env[node.targets[0].id] = node.value
    return env


def resolve_str(node, env, depth=0):
    if node is None or depth > 5:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in env:
        return resolve_str(env[node.id], env, depth + 1)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = resolve_str(node.left, env, depth + 1), resolve_str(node.right, env, depth + 1)
        return left + right if left is not None and right is not None else None
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                parts.append(str(v.value))
            else:
                inner = resolve_str(v.value, env, depth + 1)
                parts.append(inner if inner is not None else "{x}")
        return "".join(parts)
    return None


def url_to_path_and_query(url: str):
    if "://" in url:
        parsed = urlparse(url)
        return parsed.path or "/", set(parse_qs(parsed.query, keep_blank_values=True))
    if "/" in url:
        rest = url[url.index("/"):]
        path, _, qs = rest.partition("?")
        return path, set(parse_qs(qs, keep_blank_values=True))
    return None, set()


def dict_keys(node, env, depth=0):
    """Returns (keys, resolvable). keys may be partial if some keys are dynamic."""
    if node is None:
        return set(), True
    if isinstance(node, ast.Name) and node.id in env and depth < 5:
        return dict_keys(env[node.id], env, depth + 1)
    if isinstance(node, ast.Dict):
        keys = {k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        return keys, all(isinstance(k, ast.Constant) for k in node.keys)
    return set(), False


def extract_calls(code: str):
    """Yields (method, path, query_keys, body_keys, resolvable) per HTTP call."""
    tree = ast.parse(code)
    env = collect_assignments(tree)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        attr = node.func.attr
        args = list(node.args)
        if attr == "request" and args:
            method = resolve_str(args[0], env)
            if not method:
                continue
            method, args = method.lower(), args[1:]
        elif attr in HTTP_METHODS:
            method = attr
        else:
            continue

        kwargs = {k.arg: k.value for k in node.keywords if k.arg}
        url_node = args[0] if args else kwargs.get("url")
        url = resolve_str(url_node, env)
        if url is None:
            continue
        path, url_query = url_to_path_and_query(url)
        if path is None:          # e.g. dict.get("key") -- not an HTTP call
            continue

        q_keys, q_ok = dict_keys(kwargs.get("params"), env)
        b_node = kwargs.get("json", kwargs.get("data"))
        b_keys, b_ok = dict_keys(b_node, env)
        yield (method.upper(), path, q_keys | url_query, b_keys,
               b_node is not None, q_ok and b_ok)


# ---------------------------------------------------------------- scoring

def check_code(code: str, compiled: list, index: dict) -> dict:
    s = dict(calls=0, matched=0, resolvable=0, req_total=0, req_hit=0,
             keys_total=0, keys_unknown=0, parse_error=0)
    try:
        calls = list(extract_calls(code))
    except SyntaxError:
        s["parse_error"] = 1
        return s
    for method, path, q_keys, b_keys, has_body, ok in calls:
        s["calls"] += 1
        template = match_endpoint(method, path, compiled)
        if template is None:
            continue
        s["matched"] += 1
        if not ok:
            continue
        s["resolvable"] += 1
        spec = index[template]

        required_q = {n for n, req in spec["query"].items() if req}
        s["req_total"] += len(required_q)
        s["req_hit"] += len(required_q & q_keys)
        if has_body or spec["body_mandatory"]:
            s["req_total"] += len(spec["body_required"])
            s["req_hit"] += len(set(spec["body_required"]) & b_keys)

        s["keys_total"] += len(q_keys) + len(b_keys)
        s["keys_unknown"] += len(q_keys - set(spec["query"]))
        s["keys_unknown"] += len(b_keys - spec["body_props"])
    return s


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "    -"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cache = {}
    print(f"{'condition':<52}{'arm':<9}{'calls':>6}{'matched':>9}{'analys.':>9}"
          f"{'required set':>14}{'unknown keys':>14}")
    for path in sorted(glob.glob(os.path.join(GENERATED_DIR, "*.json"))):
        name = os.path.basename(path).replace(".json", "")
        bench_name = name.split("_k")[0]
        if bench_name not in cache:
            bench = load_benchmark(bench_name)
            per_qs = {}
            for qs in bench.queries:
                index = build_spec_index(qs.openapis)
                compiled = []
                for t in index:
                    verb, rx = template_regex(t)
                    compiled.append((t, verb, rx, len(re.sub(r"\{[^}]*\}", "", t))))
                per_qs[qs.name] = (compiled, index)
            cache[bench_name] = per_qs
        per_qs = cache[bench_name]

        records = json.load(open(path))
        out = []
        for arm in ("without", "with"):
            tot = dict(calls=0, matched=0, resolvable=0, req_total=0, req_hit=0,
                       keys_total=0, keys_unknown=0, parse_error=0)
            for r in records:
                compiled, index = per_qs[r["queryset"]]
                s = check_code(r[f"code_{arm}"], compiled, index)
                out.append({"queryset": r["queryset"], "query": r["query"], "arm": arm, **s})
                for k in tot:
                    tot[k] += s[k]
            print(f"{name[:51]:<52}{arm:<9}{tot['calls']:>6}"
                  f"{pct(tot['matched'], tot['calls']):>9}"
                  f"{pct(tot['resolvable'], tot['matched']):>9}"
                  f"{pct(tot['req_hit'], tot['req_total']):>14}"
                  f"{pct(tot['keys_unknown'], tot['keys_total']):>14}")
        print()
        json.dump(out, open(os.path.join(OUT_DIR, name + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()