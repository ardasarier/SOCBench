"""
Scoring generated composition code against RestBench ground truth.

    Python code -> socbenchsc.Analysis -> {"GET /v1/search", ...}
                -> strip version prefix -> map to OpenAPI templates
                -> compare with query.solution

No LLM dependency, so this runs and can be tested without an API key.

Sanity check: python src/scoring.py   (from socbench-d/code/)
"""
from socbenchsc.analysis import Analysis
from socrag.file import extract_endpoints


def extract_endpoints_from_code(code: str) -> set:
    """Unparseable code is a result, not an exception -- return an empty set."""
    try:
        return Analysis(code).perform_analysis()
    except SyntaxError as e:
        print(f"[scoring] generated code does not parse: {e}")
        return set()


def normalise(endpoint: str, base_path: str = "") -> str:
    """socbenchsc returns the full URL path ("GET /v1/search"); RestBench
    solutions are relative to the API root ("GET /search")."""
    if not base_path:
        return endpoint
    verb, _, path = endpoint.partition(" ")
    if path.startswith(base_path):
        path = path[len(base_path):] or "/"
    return f"{verb} {path}"


def build_templates(openapis) -> set:
    """All "VERB /path/{param}" templates from the specs."""
    return {e for oas in openapis for e in extract_endpoints(oas)}


def match_to_template(endpoint: str, templates: set):
    """
    Map a concrete generated path back to its OpenAPI template.

    A path can match several templates: "/movie/now_playing" satisfies both
    the literal "/movie/now_playing" and the parameterised "/movie/{movie_id}".
    Returning the first match found while iterating a set was nondeterministic
    -- Python randomises string hashes per process, so the same input scored
    differently between runs (measured: 13 of 100 queries).

    Prefer the most specific match, counting literal segment equalities, and
    sort for a deterministic tie-break.

        "POST /users/YOUR_USER_ID/playlists" -> "POST /users/{user_id}/playlists"
        "POST /playlists//tracks"            -> "POST /playlists/{playlist_id}/tracks"

    Returns None when nothing matches, i.e. a hallucinated endpoint.
    """
    verb, _, path = endpoint.partition(" ")
    segs = path.lstrip("/").split("/")

    best, best_score = None, -1
    for template in sorted(templates):
        t_verb, _, t_path = template.partition(" ")
        if t_verb != verb:
            continue
        t_segs = t_path.lstrip("/").split("/")
        if len(t_segs) != len(segs):
            continue

        score, matches = 0, True
        for t_seg, seg in zip(t_segs, segs):
            if t_seg == seg:
                score += 1                    # literal is more specific
            elif not t_seg.startswith("{"):
                matches = False
                break
        if matches and score > best_score:
            best, best_score = template, score
    return best


def score(generated_endpoints: set, solution: list, templates: set, base_path: str = "") -> dict:
    stripped = {normalise(e, base_path) for e in generated_endpoints}
    found = {m for m in (match_to_template(e, templates) for e in stripped) if m}
    expected = set(solution)
    true_positives = found & expected
    return {
        "found": found,
        "hallucinated": len(stripped) - len(found),
        "expected": expected,
        "recall": len(true_positives) / len(expected) if expected else 0.0,
        "precision": len(true_positives) / len(found) if found else 0.0,
        "exact_match": found == expected,
    }


if __name__ == "__main__":
    sample = '''
import requests

BASE = "https://api.spotify.com/v1"
r = requests.get(f"{BASE}/search", params={"q": "Mariah Carey", "type": "track"})
me = requests.get("https://api.spotify.com/v1/me")
requests.post("https://api.spotify.com/v1/users/abc/playlists", json={"name": "Love Mariah"})
'''
    endpoints = extract_endpoints_from_code(sample)
    print("extracted: ", endpoints)
    print("normalised:", {normalise(e, "/v1") for e in endpoints})
