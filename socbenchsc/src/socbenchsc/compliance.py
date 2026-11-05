"""
Schema compliance of the invocations found by the analysis.

perform_invocation_analysis() yields, per service invocation, the endpoint
and every parameter of the call. This module checks those parameters against
the OpenAPI specification of the matched endpoint:

  - required query parameters: which of them the call sets (params=)
  - required body fields: which of them the call sets (json=, else data=)
  - unknown keys: sent keys the specification does not define

Endpoints are matched with the regular expressions SOCBench-SC uses for the
benchmark solution: path parameters become ([^/]*?), the trailing slash is
optional, and the most literal template wins.

Two cases are counted separately instead of being judged:
  - a body schema that is only a $ref: the referenced fields are not part of
    the endpoint's specification text, so neither coverage nor unknown keys
    can be judged from it
  - an ambiguous endpoint: the same method and path defined by two services
    of the same domain -- the analysis cannot know which one is meant
Parameters whose keys are not completely known (an Unknown value, a computed
key) are counted as untraceable and not judged either.

Scope: query parameters and JSON bodies. Path and header parameters, value
types and response handling are not checked.
"""
from __future__ import annotations

import json
import re
from collections import Counter

HTTP_VERBS = {"get", "options", "head", "post", "put", "patch", "delete"}


def build_index(openapis) -> tuple[dict, set]:
    """({"METHOD /path": operation}, {ambiguous "METHOD /path"}) over the
    specifications of one domain."""
    index, ambiguous = {}, set()
    for oas in openapis:
        spec = json.loads(oas) if isinstance(oas, str) else oas
        for path, operations in spec.get("paths", {}).items():
            for verb, operation in operations.items():
                if verb.lower() not in HTTP_VERBS:
                    continue
                key = f"{verb.upper()} {path}"
                if key in index:
                    ambiguous.add(key)
                index[key] = operation
    return index, ambiguous


def endpoint_pattern(key: str) -> re.Pattern:
    """SOCBench-SC's matching: GET /a/{id}/ -> ^GET /a/([^/]*?)(/?)$"""
    method, _, path = key.partition(" ")
    parts = [re.escape(p) if not p.startswith("{") else "([^/]*?)"
             for p in path.strip("/").split("/")]
    return re.compile(f"^{method} /" + "/".join(parts) + "(/?)$")


def match_endpoint(method: str, path: str, index: dict, base_path: str = ""):
    """The specified endpoint an invocation calls, or None. A base path
    (e.g. /v1) in front of the specified paths is removed first."""
    if base_path and path.startswith(base_path.rstrip("/") + "/"):
        path = path[len(base_path.rstrip("/")):]
    best, best_literal = None, -1
    for key in sorted(index):
        if endpoint_pattern(key).match(f"{method} {path}"):
            literal = sum(1 for p in key.split("/") if p and not p.startswith("{"))
            if literal > best_literal:
                best, best_literal = key, literal
    return best


def requirements(operation: dict) -> dict:
    query = {p["name"]: p for p in operation.get("parameters", []) if p.get("in") == "query"}
    body = operation.get("requestBody") or {}
    schema = body.get("content", {}).get("application/json", {}).get("schema", {})
    return {
        "query_known": set(query),
        "query_required": {n for n, p in query.items() if p.get("required")},
        "body_known": set(schema.get("properties", {})),
        "body_required": set(schema.get("required", [])),
        "body_mandatory": bool(body.get("required")),
        "body_ref": "$ref" in schema,
    }


def check(invocations, index: dict, ambiguous: set = frozenset(),
          solution=None, base_path: str = "") -> Counter:
    """Counts for one program. With a solution, only invocations of
    solution endpoints are judged (keys prefixed "solution_"); the rest are
    counted as "other_calls"."""
    stats = Counter()
    for invocation in invocations:
        stats["calls"] += 1
        key = match_endpoint(invocation.method, invocation.path, index, base_path)
        if key is None:
            stats["unmatched"] += 1
            continue
        if solution is not None and key not in solution:
            stats["other_calls"] += 1
            continue
        s = "solution_" if solution is not None else ""
        stats[s + "calls"] += 1
        if key in ambiguous:
            stats[s + "ambiguous"] += 1
            continue
        req = requirements(index[key])

        query_keys, query_complete = invocation.keys("params")
        if not query_complete:
            stats[s + "query_untraceable"] += 1
        else:
            stats[s + "query_required"] += len(req["query_required"])
            stats[s + "query_required_set"] += len(req["query_required"] & query_keys)
            stats[s + "keys_sent"] += len(query_keys)
            stats[s + "keys_unknown"] += len(query_keys - req["query_known"])

        if req["body_ref"]:
            stats[s + "body_ref"] += 1
            continue
        body_name = "json" if "json" in invocation.arguments else "data"
        if body_name not in invocation.arguments and not req["body_mandatory"]:
            continue
        body_keys, body_complete = invocation.keys(body_name)
        if not body_complete:
            stats[s + "body_untraceable"] += 1
            continue
        stats[s + "body_required"] += len(req["body_required"])
        stats[s + "body_required_set"] += len(req["body_required"] & body_keys)
        stats[s + "keys_sent"] += len(body_keys)
        stats[s + "keys_unknown"] += len(body_keys - req["body_known"])
    return stats
