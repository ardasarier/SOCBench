"""
Benchmark selection, shared by both pipeline stages.

RestBench and SOCBench-D differ structurally, and both scripts need the same
resolution logic:

  RestBench    one queryset holding Spotify's 57 queries followed by TMDB's
               100, over a single API each. Base URL is the real one.

  SOCBench-D   five instances (socbenchd_1..5), each with 11 GICS-sector
               querysets of 10 queries. Every queryset spans FIVE generated
               OpenAPI specs.
"""
import benchmark

RESTBENCH_SPOTIFY_QUERY_COUNT = 57

# SOCBench-D's generated specs have inconsistent `servers` fields -- some carry
# a path component, some are absent entirely -- and five services share one
# queryset. The chunks never contain `servers` anyway (OpenApiParser emits only
# the operation object), so the model learns the base URL solely from our
# prompt. One synthetic host without a path keeps generated paths identical to
# the solution format, at the cost of being functionally wrong if the code were
# actually executed. Endpoint-level scoring is unaffected.
SOCBENCHD_BASE_URL = "https://api.socbench.example"
SOCBENCHD_BASE_PATH = ""

RESTBENCH_APIS = {
    "spotify": ("https://api.spotify.com/v1", "/v1"),
    "tmdb": ("https://api.themoviedb.org/3", "/3"),
}


def load_benchmark(name: str):
    """name is "restbench" or "socbenchd_1".."socbenchd_5"."""
    if name == "restbench":
        return benchmark.get_restbench()
    for candidate in benchmark.get_socbenchd():
        if candidate.name == name:
            return candidate
    raise ValueError(f"unknown benchmark {name!r}")


def load_tasks(benchmark_name: str, api: str = "tmdb", max_queries: int = None):
    """
    Returns (benchmark, tasks) where each task is one retrieval unit:

        {"queryset": Queryset, "queries": [...], "base_url": str, "base_path": str}

    RestBench yields a single task sliced to one API; SOCBench-D yields one per
    GICS sector. Both stages iterate over tasks, so the rest of the pipeline
    does not need to know which benchmark it is running.
    """
    bench = load_benchmark(benchmark_name)

    if benchmark_name == "restbench":
        queryset = bench.queries[0]
        base_url, base_path = RESTBENCH_APIS[api]
        start = 0 if api == "spotify" else RESTBENCH_SPOTIFY_QUERY_COUNT
        end = start + max_queries if max_queries else (
            RESTBENCH_SPOTIFY_QUERY_COUNT if api == "spotify" else len(queryset.queries)
        )
        return bench, [{
            "queryset": queryset,
            "queries": queryset.queries[start:end],
            "base_url": base_url,
            "base_path": base_path,
        }]

    return bench, [{
        "queryset": qs,
        "queries": qs.queries[:max_queries] if max_queries else qs.queries,
        "base_url": SOCBENCHD_BASE_URL,
        "base_path": SOCBENCHD_BASE_PATH,
    } for qs in bench.queries]
