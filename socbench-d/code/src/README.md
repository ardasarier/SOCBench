# src/

Adapting AttentionRAG (Fang et al., arXiv:2503.10720) to the SOCBench OpenAPI
RAG pipeline. Bachelor's thesis, IAAS, University of Stuttgart.

## Running scripts

```bash
cd socbench-d/code
source venv/bin/activate
python src/prepare_contexts.py
```

Two things have to hold at once, and they pull in opposite directions:

- `get_restbench()` resolves data paths relative to `code/`, so the working
  directory must be `code/`.
- Python puts the *script's own directory* on the import path, which is what
  makes `import benchmark` and `import socrag` resolve.

## Pipeline

```
reproduce_baseline.py     retrieval recall + distinct endpoints per query
                          no LLM, no API, runs in seconds

prepare_contexts.py       STAGE 1: retrieval + AttentionRAG compression
                          local models only, free, writes data/contexts/*.json

run_codegen.py            STAGE 2: composition + scoring
                          reads the JSON, calls the API, writes data/logs/*.txt
  ├── composition.py      the OpenAI call and the composition prompt
  └── scoring.py          SOCBench-SC analysis + template matching

check_setup.py            verifies interpreter, packages, env vars, data
                          loads no models, makes no API calls
```

The two stages are split because running the compression models and a codegen
model in one process caused MPS OOM. The split also means compressed contexts
are cached, so stage 2 can be rerun with different settings for free.

`run_codegen.py` caches the uncompressed baseline per k under
`data/baselines/`, so every condition is measured against the same fixed
reference. Without it the reference drifts: identical input once gave recall
0.633 and 0.733, because gpt-4o is not deterministic even at temperature 0.

## attentionrag/

From-scratch implementation of Algorithm 1. No public code exists for the
paper.

```
models.py              loads both Qwen models once, picks mps/cuda/cpu
                       1.5B for hint generation, 0.5B for attention
hint_prefix.py         step 3.1 -- Appendix B.1 prompt
anchor_token.py        step 3.2a -- Appendix B.3 prompt (optional, see below)
attention_features.py  step 3.2b -- Eq. 2, heads averaged then summed over layers
compression.py         step 3.3 -- Eq. 3, JSON-structure segmentation
postprocessor.py       wraps all of it as a LlamaIndex BaseNodePostprocessor
```

Module demos run as `python -m attentionrag.<module>` from `src/` — relative
imports fail under direct script execution.

Two deviations from the paper, both documented in the modules:

- **Eq. 3 retains sentences**, which assumes prose. On minified OpenAPI JSON the
  sentence splitter cannot reach the endpoint's own identity at any k, so this
  implementation segments on JSON structure and always retains the identity
  prefix.
- **Eq. 2 sums over layers but never states how heads are combined.** This
  implementation averages heads within a layer first.

## Not mine

`evaluate.py` and `stats.py` are SOCBench's originals (unmodified).
`benchmark/` and `socrag/` are the upstream pipeline.

`socbenchsc` is installed editable (`pip install -e <path> --no-deps`) rather
than copied — the `--no-deps` avoids its hard `pydantic == 2.11.7` pin
downgrading the environment.