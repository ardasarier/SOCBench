"""
STAGE 1: retrieval + AttentionRAG compression. Writes contexts to JSON.

Split from generation because running the compression models and a local
codegen model in one process caused MPS OOM and ~400s per query. Separated,
each stage has the machine to itself -- and the compressed contexts are cached,
so stage 2 can be rerun with a different codegen model for free.

Run from socbench-d/code/:
    python src/prepare_contexts.py
"""
import json
import time

from llama_index.core import Settings
from llama_index.core.schema import QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import benchmark
from attentionrag.models import empty_cache
from attentionrag.postprocessor import AttentionRAGPostprocessor
from socrag.index import get_retriever

# Qwen2.5-0.5B has 24 layers.
LAYER_RANGE = (8, 16)    # None | (0, 8) shallow | (8, 16) middle | (16, 24) deep
LAYER_LABEL = "middle"   # "all" | "shallow" | "middle" | "deep"
NUM_QUERIES = 10         # halved -- four configs to run in limited time
TOP_K = 10               # SOCBench's k (CHUNKS)
TOP_K_TOKENS = 10        # AttentionRAG's k (TOKENS)
API = "tmdb"             # "spotify" or "tmdb"
CHUNKING_STRATEGY = "ENDPOINT_SPLIT_1024_0"
EMBEDDING_DIMENSIONS = 384

# RestBench concatenates Spotify's 57 queries, then TMDB's 100.
SPOTIFY_QUERY_COUNT = 57
if API == "spotify":
    QUERY_START = 0
    BASE_URL, BASE_PATH = "https://api.spotify.com/v1", "/v1"
else:
    QUERY_START = SPOTIFY_QUERY_COUNT
    BASE_URL, BASE_PATH = "https://api.themoviedb.org/3", "/3"

OUTPUT_PATH = f"data/contexts_{API}_k{TOP_K}_{LAYER_LABEL}.json"

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

restbench = benchmark.get_restbench()
queryset = restbench.queries[0]
encoding = benchmark.get_encoding()

retriever = get_retriever(
    restbench.name, "bge_small", queryset.name, CHUNKING_STRATEGY,
    queryset.openapis, embed_model, EMBEDDING_DIMENSIONS, TOP_K,
)
postprocessor = AttentionRAGPostprocessor(top_k_tokens=TOP_K_TOKENS, layer_range=LAYER_RANGE)


def texts_of(nodes) -> list:
    return [n.node.get_content() for n in nodes]


def token_count(texts) -> int:
    return sum(len(encoding.encode(t)) for t in texts)


records = []
queries = queryset.queries[QUERY_START:QUERY_START + NUM_QUERIES]
start_all = time.time()

for i, query in enumerate(queries):
    empty_cache()
    print(f"[{i + 1}/{len(queries)}] {query.query!r}")
    started = time.time()

    retrieved = retriever.retrieve(query.query)
    compressed = postprocessor.postprocess_nodes(
        retrieved, query_bundle=QueryBundle(query.query)
    )

    raw_texts = texts_of(retrieved)
    compressed_texts = texts_of(compressed)

    # base_url/base_path travel with the record so stage 2 cannot drift out of
    # sync with whichever API stage 1 prepared.
    records.append({
        "query": query.query,
        "solution": query.solution,
        "base_url": BASE_URL,
        "base_path": BASE_PATH,
        "raw_chunks": raw_texts,
        "compressed_chunks": compressed_texts,
        "raw_tokens": token_count(raw_texts),
        "compressed_tokens": token_count(compressed_texts),
    })

    raw, comp = records[-1]["raw_tokens"], records[-1]["compressed_tokens"]
    print(f"    {len(raw_texts)} -> {len(compressed_texts)} chunks, "
          f"{raw} -> {comp} tokens (CR {raw / comp:.2f}x), {time.time() - started:.1f}s")

with open(OUTPUT_PATH, "w") as f:
    json.dump(records, f, indent=2)

total_raw = sum(r["raw_tokens"] for r in records)
total_compressed = sum(r["compressed_tokens"] for r in records)
print(f"\nwrote {len(records)} records to {OUTPUT_PATH}")
print(f"overall CR: {total_raw / total_compressed:.2f}x")
print(f"total time: {(time.time() - start_all) / 60:.1f} min")
