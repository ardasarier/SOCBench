"""
Measures AttentionRAG compression on real RestBench retrievals:
compression ratio, recall before vs. after, and wall-clock time per query.

Recall after compression is NOT identical to recall before. Compression only
rewrites chunk text and preserves metadata["endpoints"], so it cannot raise
recall -- but skip_on_none drops whole chunks, which lowers it. On the full
157-query run that cost 0.509 -> 0.504 for 2.84x compression.

Each query runs TOP_K forward passes with output_attentions=True, materialising
num_layers x heads x seq_len^2 floats (24 layers for Qwen2.5-0.5B). Start with
NUM_QUERIES small and raise it once a few queries survive.

Run from socbench-d/code/:
    python src/measure_compression.py
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

NUM_QUERIES = 20
TOP_K = 10               # SOCBench's k (CHUNKS)
TOP_K_TOKENS = 10        # AttentionRAG's k (TOKENS)
CHUNKING_STRATEGY = "ENDPOINT_SPLIT_1024_0"
EMBEDDING_DIMENSIONS = 384

encoding = benchmark.get_encoding()  # tiktoken gpt-4o, as socbench-d uses


def count_tokens(nodes) -> int:
    return sum(len(encoding.encode(n.node.get_content())) for n in nodes)


def endpoints_in(nodes) -> set:
    found = set()
    for node in nodes:
        found.update(json.loads(node.node.metadata["endpoints"]))
    return found


embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

restbench = benchmark.get_restbench()
queryset = restbench.queries[0]

retriever = get_retriever(
    restbench.name, "bge_small", queryset.name, CHUNKING_STRATEGY,
    queryset.openapis, embed_model, EMBEDDING_DIMENSIONS, TOP_K,
)
postprocessor = AttentionRAGPostprocessor(top_k_tokens=TOP_K_TOKENS, verbose=True)

total_before = total_after = 0
total_solution = true_positives_before = true_positives_after = 0
total_time = 0.0
nodes_dropped = 0

queries = queryset.queries[:NUM_QUERIES]

for i, query in enumerate(queries):
    empty_cache()
    print(f"\n[{i + 1}/{len(queries)}] {query.query!r}")

    retrieved = retriever.retrieve(query.query)

    started = time.time()
    compressed = postprocessor.postprocess_nodes(
        retrieved, query_bundle=QueryBundle(query.query)
    )
    elapsed = time.time() - started
    total_time += elapsed

    before_tokens = count_tokens(retrieved)
    after_tokens = count_tokens(compressed)
    total_before += before_tokens
    total_after += after_tokens

    solution = set(query.solution)
    true_positives_before += len(endpoints_in(retrieved) & solution)
    true_positives_after += len(endpoints_in(compressed) & solution)
    total_solution += len(solution)

    nodes_dropped += len(retrieved) - len(compressed)

    ratio = before_tokens / after_tokens if after_tokens else float("inf")
    print(f"    tokens {before_tokens} -> {after_tokens}  (CR {ratio:.2f}x)")
    print(f"    nodes  {len(retrieved)} -> {len(compressed)}")
    print(f"    {elapsed:.1f}s")

print("\n" + "=" * 60)
print(f"Queries:            {len(queries)}")
print(f"Tokens:             {total_before} -> {total_after}")
if total_after:
    print(f"Compression ratio:  {total_before / total_after:.2f}x")
print(f"Nodes dropped:      {nodes_dropped}")
print()
print(f"Recall before:      {true_positives_before / total_solution:.3f}")
print(f"Recall after:       {true_positives_after / total_solution:.3f}   "
      f"<- lower only where chunks were dropped")
print()
print(f"Time per query:     {total_time / len(queries):.1f}s")
print(f"Full RestBench:     ~{total_time / len(queries) * 157 / 60:.0f} min (157 queries)")
print("=" * 60)
