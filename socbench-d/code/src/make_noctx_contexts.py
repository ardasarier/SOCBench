"""
A condition without any retrieved context: the composition model gets only
the query and the API's base URL.

It measures what the model brings from training. On RestBench the composition
recall exceeds the retrieval recall (TMDB: 0.647 called vs 0.392 in context),
which suggests the model knows these real-world APIs. Without context, recall
should then stay well above zero on RestBench and collapse on SOCBench-D,
whose synthetic services cannot be in any training data.

The uncompressed arm keeps the reference's k=10 chunks, so run_codegen hits
the cached baseline and every result row reads "uncompressed k=10 vs no
context".

Run from socbench-d/code/:
    python src/make_noctx_contexts.py --reference <k10 contexts>.json [--label spotify-noctx]
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="a k=10 context file")
    ap.add_argument("--label", default="noctx")
    args = ap.parse_args()

    reference = json.load(open(args.reference))
    out = []
    for ref in reference:
        record = {k: ref[k] for k in ("benchmark", "queryset", "query", "solution",
                                      "base_url", "base_path", "top_k",
                                      "raw_chunks", "raw_tokens")}
        record.update({
            "compressed_chunks": [],
            "compressed_tokens": 0,
            "budget": {"label": args.label, "source": None, "tokens": 0,
                       "depth": 0, "retrieval_recall": 0.0},
        })
        out.append(record)

    path = (f"data/contexts/{out[0]['benchmark']}_k10_budget-{args.label}"
            f"_n{len(out)}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"queries:                  {len(out)}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
