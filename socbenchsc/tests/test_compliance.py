from socbenchsc.analysis import Analysis
from socbenchsc.compliance import build_index, check, match_endpoint

SPEC_A = {"paths": {
    "/orders": {"post": {"requestBody": {"required": True, "content": {"application/json": {
        "schema": {"type": "object", "required": ["item", "qty"],
                   "properties": {"item": {}, "qty": {}, "note": {}}}}}}}},
    "/orders/{id}": {"get": {"parameters": [{"name": "id", "in": "path", "required": True}]}},
    "/orders/active": {"get": {"parameters": [
        {"name": "region", "in": "query", "required": True},
        {"name": "limit", "in": "query"}]}},
    "/software": {"post": {"requestBody": {"required": True, "content": {"application/json": {
        "schema": {"$ref": "#/components/schemas/Software"}}}}}},
    "/alerts": {"get": {}},
}}
SPEC_B = {"paths": {"/alerts": {"get": {}}}}
INDEX, AMBIGUOUS = build_index([SPEC_A, SPEC_B])


def run(code, solution=None, base_path=""):
    invocations = Analysis(code).perform_invocation_analysis()
    return check(invocations, INDEX, AMBIGUOUS, solution, base_path)


def test_build_index_finds_ambiguous_paths():
    assert "GET /alerts" in AMBIGUOUS
    assert "POST /orders" not in AMBIGUOUS


def test_matching_prefers_literal_segments():
    assert match_endpoint("GET", "/orders/7", INDEX) == "GET /orders/{id}"
    assert match_endpoint("GET", "/orders/active", INDEX) == "GET /orders/active"
    assert match_endpoint("GET", "/orders/active/", INDEX) == "GET /orders/active"
    assert match_endpoint("GET", "/v1/orders/7", INDEX, base_path="/v1") == "GET /orders/{id}"
    assert match_endpoint("DELETE", "/orders/7", INDEX) is None


def test_required_body_fields():
    s = run('requests.post("http://x/orders", json={"item": "a", "qty": 1})')
    assert (s["body_required"], s["body_required_set"], s["keys_unknown"]) == (2, 2, 0)
    s = run('requests.post("http://x/orders", json={"item": "a", "colour": "red"})')
    assert (s["body_required"], s["body_required_set"]) == (2, 1)
    assert (s["keys_sent"], s["keys_unknown"]) == (2, 1)


def test_mandatory_body_missing_counts_as_unset():
    s = run('requests.post("http://x/orders")')
    assert (s["body_required"], s["body_required_set"]) == (2, 0)


def test_required_query_parameters():
    s = run('requests.get("http://x/orders/active", params={"limit": 5})')
    assert (s["query_required"], s["query_required_set"]) == (1, 0)
    s = run('requests.get("http://x/orders/active", params={"region": "eu"})')
    assert (s["query_required"], s["query_required_set"]) == (1, 1)


def test_ref_ambiguous_and_untraceable_are_not_judged():
    s = run('requests.post("http://x/software", json={"name": "n"})')
    assert s["body_ref"] == 1 and s["body_required"] == 0 and s["keys_sent"] == 0
    s = run('requests.get("http://x/alerts", params={"a": 1})')
    assert s["ambiguous"] == 1 and s["keys_sent"] == 0
    s = run('requests.post("http://x/orders", json=build())')
    assert s["body_untraceable"] == 1 and s["body_required"] == 0


def test_solution_scope():
    code = """
requests.post("http://x/orders", json={"item": "a"})
requests.get("http://x/orders/active", params={})
"""
    s = run(code, solution={"POST /orders"})
    assert s["solution_calls"] == 1 and s["other_calls"] == 1
    assert (s["solution_body_required"], s["solution_body_required_set"]) == (2, 1)
    assert s["solution_query_required"] == 0
