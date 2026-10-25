"""
Task 1: compression at EQUAL TOKEN BUDGET instead of equal k.

So far every condition kept k=10 and measured what compression COSTS. The
question that decides whether compression is worth anything is the reverse:
if compression frees up tokens, can the same budget hold MORE retrieved chunks
-- and does the composition benefit from the deeper retrieval?

This builds context files for exactly that comparison, offline, from the logs
prepare_contexts.py now writes:

  --reference  a k=10 context file. Its raw chunks are the baseline arm, and
               its per-query raw token count is that query's budget.
  --source     a deeper context file (e.g. k=50) whose log holds every chunk
               with rank and compressed tokens.

Per query, the compressed arm takes chunks of the source in retrieval order
for as long as they fit into that query's budget (a prefix -- i.e. a per-query
k). The raw arm stays the reference's k=10 chunks, so run_codegen.py hits the
existing k=10 baseline cache and every result row reads directly as
"raw k=10 vs compressed at the same budget".

--raw-k N instead builds the upper reference: the first N RAW chunks, no
budget -- what deeper retrieval would give without compression.

Run from socbench-d/code/:
    python src/make_budget_contexts.py --source <k50 file> --reference <k10 file> --label reqschema
    python src/make_budget_contexts.py --source <k50 file> --reference <k10 file> --label raw20 --raw-k 20
    python src/make_budget_contexts.py --source <k50 file> --curve
"""
import argparse
import json
import os


def chunk_texts(record: dict) -> list:
    """Compressed text per retrieved chunk, in rank order; None where the
    compression dropped the chunk. compressed_chunks holds only the kept ones,
    in order, so they are paired with the log's kept flags."""
    kept_texts = iter(record["compressed_chunks"])
    texts = [next(kept_texts) if c["kept"] else None for c in record["retrieval"]]
    assert next(kept_texts, None) is None, "more compressed chunks than kept flags"
    return texts


def recall(endpoints: set, solution: list) -> float:
    sol = set(solution)
    return len(sol & endpoints) / len(sol) if sol else 0.0


def curve(source: list) -> None:
    """Retrieval recall and tokens per k, averaged per query."""
    depth = min(len(r["retrieval"]) for r in source)
    print(f"{len(source)} queries, at least {depth} chunks retrieved per query\n")
    print(f"{'k':>4}{'retrieval recall':>18}{'raw tokens':>12}{'compressed':>12}")
    for k in [k for k in (5, 10, 15, 20, 25, 30, 35, 40, 45, 50) if k <= depth]:
        rr = raw = comp = 0.0
        for r in source:
            top = r["retrieval"][:k]
            rr += recall({e for c in top for e in c["endpoints"]}, r["solution"])
            raw += sum(c["raw_tokens"] for c in top)
            comp += sum(c["compressed_tokens"] for c in top)
        n = len(source)
        print(f"{k:>4}{rr / n:>18.3f}{raw / n:>12.0f}{comp / n:>12.0f}")


def build(source: list, reference: list, label: str, source_name: str,
          raw_k: int = None, budget_factor: float = 1.0) -> list:
    by_key = {(r["queryset"], r["query"]): r for r in source}
    out, prefix_mismatch = [], 0
    for ref in reference:
        src = by_key[(ref["queryset"], ref["query"])]
        if src["raw_chunks"][:len(ref["raw_chunks"])] != ref["raw_chunks"]:
            prefix_mismatch += 1

        budget = round(ref["raw_tokens"] * budget_factor)
        if raw_k is not None:
            chosen = [(c, text) for c, text in zip(src["retrieval"][:raw_k], src["raw_chunks"][:raw_k])]
            used = sum(c["raw_tokens"] for c, _ in chosen)
            depth = len(chosen)
        else:
            chosen, used, depth = [], 0, 0
            for c, text in zip(src["retrieval"], chunk_texts(src)):
                if used + c["compressed_tokens"] > budget:
                    break
                used += c["compressed_tokens"]
                depth += 1
                if text is not None:
                    chosen.append((c, text))

        visible = {e for c, _ in chosen for e in c["endpoints"]}
        out.append({
            **{k: ref[k] for k in ("benchmark", "queryset", "query", "solution",
                                   "base_url", "base_path")},
            "top_k": ref["top_k"],                  # k=10 -> existing baseline cache
            "raw_chunks": ref["raw_chunks"],
            "raw_tokens": ref["raw_tokens"],
            "compressed_chunks": [text for _, text in chosen],
            "compressed_tokens": used,
            "budget": {
                "label": label,
                "source": source_name,
                "tokens": budget if raw_k is None else None,
                "depth": depth,
                "retrieval_recall": recall(visible, ref["solution"]),
            },
            "retrieval": [c for c, _ in chosen],
        })
    if prefix_mismatch:
        print(f"WARNING: {prefix_mismatch} queries whose first chunks differ from the reference")
    return out


def summary(records: list, reference: list) -> None:
    n = len(records)
    mean = lambda f: sum(f(r) for r in records) / n
    ref_rr = sum(r.get("retrieval_recall", 0) for r in reference) / len(reference)
    print(f"queries:                  {n}")
    print(f"mean depth (chunks):      {mean(lambda r: r['budget']['depth']):.1f}")
    print(f"tokens used / raw k=10:   {mean(lambda r: r['compressed_tokens']) / mean(lambda r: r['raw_tokens']):.3f}")
    print(f"retrieval recall:         {mean(lambda r: r['budget']['retrieval_recall']):.3f}"
          f"   (raw k=10: {ref_rr:.3f})")
    over = sum(1 for r in records if r["budget"]["tokens"] is not None
               and r["compressed_tokens"] > r["budget"]["tokens"])
    print(f"queries over budget:      {over}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--reference")
    ap.add_argument("--label")
    ap.add_argument("--raw-k", type=int)
    ap.add_argument("--budget-factor", type=float, default=1.0)
    ap.add_argument("--curve", action="store_true")
    args = ap.parse_args()

    source = json.load(open(args.source))
    if args.curve:
        curve(source)
        raise SystemExit

    reference = json.load(open(args.reference))
    records = build(source, reference, args.label, os.path.basename(args.source),
                    raw_k=args.raw_k, budget_factor=args.budget_factor)
    bench = records[0]["benchmark"]
    path = f"data/contexts/{bench}_k10_budget-{args.label}_n{len(records)}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    summary(records, reference)
    print(f"\nwrote {path}")