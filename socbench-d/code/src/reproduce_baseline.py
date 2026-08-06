"""
Reproduces a RestBench recall@k number from the socbench-d RAG pipeline,
without evaluate.py's full sweep (15 chunking strategies x 3 k x 2 benchmarks).

BGE-small embeddings, endpoint-split chunking. RestBench merges Spotify (57
queries) and TMDB (100) into one index, matching the paper; this splits the
results back apart by query position.

Also reports distinct endpoints per query, which is the direct evidence for the
fragmentation effect: a TMDB endpoint averaging ~11,700 chars does not fit in a
1024-token chunk, so it is SPLIT into several chunks that all carry the same
endpoint metadata. Those fragments occupy top-k slots without contributing new
endpoints, which is why TMDB recall trails Spotify's.

Run from socbench-d/code/ (NOT src/ -- get_restbench() resolves data paths
relative to code/):
    python src/reproduce_baseline.py
"""
import json
import os

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import benchmark
from socrag.index import get_retriever

TOP_K = 10
CHUNKING_STRATEGY = "ENDPOINT_SPLIT_1024_0"
EMBEDDING_DIMENSIONS = 384
SPOTIFY_QUERY_COUNT = 57

os.makedirs("data", exist_ok=True)  # gitignored; evaluate.py assumes it exists

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

restbench = benchmark.get_restbench()
queryset = restbench.queries[0]

print(f"Building/loading '{CHUNKING_STRATEGY}' index over RestBench (Spotify+TMDB)...")
retriever = get_retriever(
    restbench.name,
    "bge_small",
    queryset.name,
    CHUNKING_STRATEGY,
    queryset.openapis,
    embed_model,
    EMBEDDING_DIMENSIONS,
    TOP_K,
)


def evaluate(queries):
    """Returns (micro-averaged recall, mean distinct endpoints per query).

    Micro-averaging matches the Evaluator in benchmark/llama_index.py, so the
    number is comparable with socbench-d's own results.
    """
    total_solution = 0
    total_true_positive = 0
    distinct_counts = []

    for query in queries:
        retrieved_endpoints = set()
        for doc in retriever.retrieve(query.query):
            retrieved_endpoints.update(json.loads(doc.metadata["endpoints"]))
        distinct_counts.append(len(retrieved_endpoints))
        total_true_positive += len(retrieved_endpoints.intersection(query.solution))
        total_solution += len(query.solution)

    recall = total_true_positive / total_solution if total_solution else float("nan")
    mean_distinct = sum(distinct_counts) / len(distinct_counts) if distinct_counts else 0.0
    return recall, mean_distinct


groups = [
    ("Spotify-only", queryset.queries[:SPOTIFY_QUERY_COUNT]),
    ("TMDB-only   ", queryset.queries[SPOTIFY_QUERY_COUNT:]),
    ("Combined    ", queryset.queries),
]

print()
for label, queries in groups:
    recall, mean_distinct = evaluate(queries)
    print(f"{label} recall@{TOP_K}: {recall:.3f}  ({len(queries)} queries)  "
          f"distinct endpoints: {mean_distinct:.1f} / {TOP_K} chunks")
