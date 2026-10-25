"""
STAGE 1: retrieval + AttentionRAG compression. Writes contexts to JSON.

Split from generation because running the compression models and a local
codegen model in one process caused MPS OOM and ~400s per query. Separated,
each stage has the machine to itself -- and the compressed contexts are cached,
so stage 2 can be rerun with a different codegen model for free.

Handles both benchmarks. RestBench is one queryset over one API; SOCBench-D is
11 GICS-sector querysets per instance, each with its own prebuilt FAISS index,
so the retriever is rebuilt per sector.

Every record also logs what was retrieved -- rank, retrieval score and the
endpoints of each chunk, its tokens before and after compression -- plus the
query's retrieval recall and the local Qwen token counts. Together with the
API usage stage 2 records, this makes each run traceable chunk by chunk.

Run from socbench-d/code/:
    python src/prepare_contexts.py
"""
import json
import os
import time

from llama_index.core import Settings
from llama_index.core.schema import QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import benchmark
from attentionrag import usage as qwen_usage
from attentionrag.compression import ENDPOINT_LINE, REQUEST_SCHEMA_FIELDS
from attentionrag.models import empty_cache
from attentionrag.postprocessor import AttentionRAGPostprocessor
from benchmark_loader import load_tasks
from socrag.index import get_retriever

# --- experiment scope ---
BENCHMARK = "socbenchd_1"           # "restbench" | "socbenchd_1".."socbenchd_5"
API = "tmdb"                        # RestBench only: "spotify" or "tmdb"
MAX_QUERIES_PER_SET = None          # None = all; set e.g. 2 for a pilot run

# --- retrieval ---
TOP_K = 40                          # SOCBench's k (CHUNKS)
CHUNKING_STRATEGY = "ENDPOINT_SPLIT_1024_0"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_LABEL = "bge_small"       # index directory name
EMBEDDING_DIMENSIONS = 384

# --- compression ---
TOP_K_TOKENS = 10                   # AttentionRAG's k (TOKENS)
THRESHOLD_RATIO = None              # None = fixed top-k; else 0.1 / 0.25 / 0.5
THRESHOLD_ABS = None                # fixed cutoff on raw attention score
LAYER_RANGE = None                  # None = all 24 (Qwen2.5-0.5B); (0,8) shallow,
LAYER_LABEL = "all"                 #   (8,16) middle, (16,24) deep
USE_ANCHOR = False
IDENTITY_ONLY = False               # ablation: identity prefix only, no attention
PATH_VERB_ONLY = False
RANDOM_SCORES = False               # control: random instead of attention scores
KEEP_IDENTITY = True                # False: no unconditional retention
KEEP_FIELDS = None                  # None = off; REQUEST_SCHEMA_FIELDS

ANCHOR_LABEL = "anchor" if USE_ANCHOR else "noanchor"
if KEEP_FIELDS:
    SELECT_LABEL = "reqschema"
elif PATH_VERB_ONLY:
    SELECT_LABEL = "pathverb"
elif IDENTITY_ONLY:
    SELECT_LABEL = "identity"
else:
    if THRESHOLD_ABS is not None:
        base = "abs" + f"{THRESHOLD_ABS:g}".replace(".", "p")
    elif THRESHOLD_RATIO is not None:
        base = f"t{int(THRESHOLD_RATIO * 100)}"
    else:
        base = f"k{TOP_K_TOKENS}"
    SELECT_LABEL = f"random-{base}" if RANDOM_SCORES else base
    if not KEEP_IDENTITY:
        SELECT_LABEL += "-noid"
SCOPE_LABEL = BENCHMARK if BENCHMARK != "restbench" else f"restbench-{API}"

embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
Settings.embed_model = embed_model

encoding = benchmark.get_encoding()
bench, tasks = load_tasks(BENCHMARK, api=API, max_queries=MAX_QUERIES_PER_SET)

total_queries = sum(len(t["queries"]) for t in tasks)
OUTPUT_PATH = (f"data/contexts/{SCOPE_LABEL}_k{TOP_K}_{LAYER_LABEL}_{ANCHOR_LABEL}"
               f"_{SELECT_LABEL}_n{total_queries}.json")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

postprocessor = AttentionRAGPostprocessor(
    top_k_tokens=TOP_K_TOKENS, threshold_ratio=THRESHOLD_RATIO,
    layer_range=LAYER_RANGE, use_anchor=USE_ANCHOR,
    identity_only=IDENTITY_ONLY, random_scores=RANDOM_SCORES,
    keep_identity=KEEP_IDENTITY, path_verb_only=PATH_VERB_ONLY,
    threshold_abs=THRESHOLD_ABS, keep_fields=KEEP_FIELDS,
)


def texts_of(nodes) -> list:
    return [n.node.get_content() for n in nodes]


def token_count(texts) -> int:
    return sum(len(encoding.encode(t)) for t in texts)


def endpoints_of(node_with_score) -> list:
    """Endpoints a chunk documents. socrag keeps them in node metadata (a JSON
    string or a list) -- what SOCBench computes retrieval recall from. Falls
    back to the chunk's Endpoint: header."""
    meta = node_with_score.node.metadata.get("endpoints")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            meta = [meta]
    if meta:
        return list(meta)
    match = ENDPOINT_LINE.match(node_with_score.node.get_content())
    return [match.group(1)] if match else []


def retrieval_log(retrieved, compressed) -> list:
    """One entry per retrieved chunk, in retrieval order. Compressed nodes are
    deep copies that keep their node_id, so they are matched by id -- a chunk
    the compression dropped shows up as kept=False."""
    compressed_by_id = {n.node.node_id: n.node.get_content() for n in compressed}
    log = []
    for rank, node_with_score in enumerate(retrieved):
        compressed_text = compressed_by_id.get(node_with_score.node.node_id)
        log.append({
            "rank": rank,
            # FAISS IndexFlatL2 (socrag/index.py): an L2 distance, so LOWER
            # means more similar -- not a similarity score.
            "distance": (round(node_with_score.score, 6)
                         if node_with_score.score is not None else None),
            "endpoints": endpoints_of(node_with_score),
            "raw_tokens": len(encoding.encode(node_with_score.node.get_content())),
            "compressed_tokens": (len(encoding.encode(compressed_text))
                                  if compressed_text is not None else 0),
            "kept": compressed_text is not None,
        })
    return log


print(f"benchmark {bench.name}: {len(tasks)} queryset(s), {total_queries} queries")
print(f"writing to {OUTPUT_PATH}\n")

records = []
done = 0
start_all = time.time()

for task in tasks:
    queryset = task["queryset"]

    # Each SOCBench-D sector has its own prebuilt index; RestBench has one.
    retriever = get_retriever(
        bench.name, EMBEDDING_LABEL, queryset.name, CHUNKING_STRATEGY,
        queryset.openapis, embed_model, EMBEDDING_DIMENSIONS, TOP_K,
    )

    for query in task["queries"]:
        empty_cache()
        done += 1
        print(f"[{done}/{total_queries}] ({queryset.name}) {query.query[:70]!r}")
        started = time.time()

        qwen_usage.take()          # discard anything recorded during setup
        retrieved = retriever.retrieve(query.query)
        compressed = postprocessor.postprocess_nodes(
            retrieved, query_bundle=QueryBundle(query.query)
        )
        qwen_tokens = qwen_usage.take()

        raw_texts = texts_of(retrieved)
        compressed_texts = texts_of(compressed)

        chunks_log = retrieval_log(retrieved, compressed)
        retrieved_endpoints = {e for c in chunks_log for e in c["endpoints"]}
        solution = set(query.solution)
        retrieval_recall = (len(solution & retrieved_endpoints) / len(solution)
                            if solution else 0.0)

        # benchmark/queryset/top_k and the URLs travel with the record so stage
        # 2 needs no configuration of its own and cannot drift out of sync.
        records.append({
            "benchmark": bench.name,
            "queryset": queryset.name,
            "top_k": TOP_K,
            "query": query.query,
            "solution": query.solution,
            "base_url": task["base_url"],
            "base_path": task["base_path"],
            "raw_chunks": raw_texts,
            "compressed_chunks": compressed_texts,
            "raw_tokens": token_count(raw_texts),
            "compressed_tokens": token_count(compressed_texts),
            "retrieval_recall": retrieval_recall,
            "retrieval": chunks_log,
            "qwen_tokens": qwen_tokens,
            "seconds": round(time.time() - started, 1),
        })

        raw, comp = records[-1]["raw_tokens"], records[-1]["compressed_tokens"]
        cr = raw / comp if comp else float("inf")
        print(f"    {len(raw_texts)} -> {len(compressed_texts)} chunks, "
              f"{raw} -> {comp} tokens (CR {cr:.2f}x), "
              f"retrieval recall {retrieval_recall:.2f}, {time.time() - started:.1f}s")

        # Persist after every query: a crash at query 400 of 550 would otherwise
        # throw away hours of compression work.
        with open(OUTPUT_PATH, "w") as f:
            json.dump(records, f, indent=2)

total_raw = sum(r["raw_tokens"] for r in records)
total_compressed = sum(r["compressed_tokens"] for r in records)
cr = total_raw / total_compressed if total_compressed else float("inf")

print(f"\nwrote {len(records)} records to {OUTPUT_PATH}")
print(f"overall CR: {cr:.2f}x")
print(f"retrieval recall: {sum(r['retrieval_recall'] for r in records) / len(records):.3f}")
qwen_total = {}
for r in records:
    for name, value in r["qwen_tokens"].items():
        qwen_total[name] = qwen_total.get(name, 0) + value
print(f"qwen: {qwen_total or 'not used'}")
print(f"total time: {(time.time() - start_all) / 60:.1f} min")