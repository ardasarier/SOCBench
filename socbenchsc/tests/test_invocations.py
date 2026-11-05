from socbenchsc.analysis import Analysis


def invocations(code):
    return Analysis(code).perform_invocation_analysis()


def only(code):
    result = invocations(code)
    assert len(result) == 1
    return result[0]


def test_endpoint_matches_perform_analysis():
    code = """
requests.get("http://a/b1", params={"q": "x"})
requests.post(url="http://a/b2", json={"name": "n"})
requests.request("DELETE", "http://a/b3")
"""
    endpoints = {f"{i.method} {i.path}" for i in invocations(code)}
    assert endpoints == Analysis(code).perform_analysis()


def test_both_entry_points_on_one_analysis():
    analysis = Analysis('requests.get("http://a/b", params={"q": 1})')
    assert analysis.perform_analysis() == {"GET /b"}
    result = analysis.perform_invocation_analysis()
    assert len(result) == 1                           # visited once, not twice
    assert result[0].keys("params") == (frozenset({"q"}), True)


def test_params_and_json_literals():
    call = only('requests.get("http://a/items", params={"q": "x", "limit": 5})')
    assert (call.method, call.path) == ("GET", "/items")
    assert call.keys("params") == (frozenset({"q", "limit"}), True)
    assert call.keys("json") == (frozenset(), True)  # not passed
    call = only('requests.post("http://a/items", json={"name": "n", "price": 1})')
    assert call.keys("json") == (frozenset({"name", "price"}), True)


def test_dict_through_variable():
    call = only("""
body = {"reportId": "r1", "feedback": "ok"}
url = "https://api.x/report-feedback"
requests.post(url, json=body)
""")
    assert call.path == "/report-feedback"
    assert call.keys("json") == (frozenset({"reportId", "feedback"}), True)


def test_dict_built_by_subscript_assignment():
    call = only("""
payload = {}
payload["a"] = 1
payload["b"] = 2
requests.post("http://x/y", json=payload)
""")
    assert call.keys("json") == (frozenset({"a", "b"}), True)


def test_branches_unite_keys():
    call = only("""
if flag:
    payload = {"a": 1}
else:
    payload = {"b": 2}
requests.put("http://x/y", json=payload)
""")
    assert call.keys("json") == (frozenset({"a", "b"}), True)


def test_positional_arguments_are_named():
    call = only('requests.get("http://x/s", {"q": 1})')
    assert call.keys("params") == (frozenset({"q"}), True)
    call = only('requests.post("http://x/s", {"form": 1}, {"j": 2})')
    assert call.keys("data") == (frozenset({"form"}), True)
    assert call.keys("json") == (frozenset({"j"}), True)
    call = only('requests.request("POST", "http://x/s", json={"k": 1})')
    assert call.method == "POST"
    assert call.keys("json") == (frozenset({"k"}), True)


def test_through_helper_function():
    result = invocations("""
def call(endpoint, body):
    return requests.post("http://x" + endpoint, json=body)
call("/orders", {"item": 1, "qty": 2})
""")
    with_body = [i for i in result if i.keys("json")[0]]
    assert len(with_body) == 1
    assert with_body[0].path == "/orders"
    assert with_body[0].keys("json") == (frozenset({"item", "qty"}), True)


def test_unknown_body_is_incomplete():
    call = only('requests.post("http://x/y", json=make_body())')
    assert call.keys("json") == (frozenset(), False)
    call = only('requests.post("http://x/y", data="raw string")')
    assert call.keys("data") == (frozenset(), False)


def test_other_parameters_are_kept():
    call = only('requests.get("http://x/y", headers={"Authorization": "t"}, timeout=5)')
    assert call.keys("headers") == (frozenset({"Authorization"}), True)
    assert set(call.arguments) == {"url", "headers", "timeout"}
