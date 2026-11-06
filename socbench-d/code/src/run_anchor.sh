#!/bin/sh
# AttentionRAG with a generated anchor token per chunk (as in the original
# method; chunks answered "none" are dropped), at equal budget, instance 1 --
# once with the generated hint prefix, once with the fixed one.
# Run from socbench-d/code/:
#   nohup caffeinate -i sh src/run_anchor.sh > data/logs/e2e/driver_anchor.txt 2>&1 &
set -e
LOGS=data/logs/e2e/socbenchd_1
REF=data/contexts/socbenchd_1_k10_all_noanchor_reqschema_n110.json
FIXED="The API endpoint required for this task is"
mkdir -p $LOGS

# 1. generated hint + anchor
python -u src/prepare_contexts.py --benchmark socbenchd_1 --top-k 40 --use-anchor --skip-existing \
  > $LOGS/prepare_k40_anchor.txt 2>&1
python src/make_budget_contexts.py --reference $REF \
  --source data/contexts/socbenchd_1_k40_all_anchor_k10_n110.json \
  --label attention40-anchor > $LOGS/budget_attention40-anchor.txt 2>&1
echo "generated hint + anchor: contexts done"

# 2. fixed hint + anchor
python -u src/prepare_contexts.py --benchmark socbenchd_1 --top-k 40 --use-anchor --fixed-hint "$FIXED" --skip-existing \
  > $LOGS/prepare_k40_fixhint_anchor.txt 2>&1
python src/make_budget_contexts.py --reference $REF \
  --source data/contexts/socbenchd_1_k40_all_anchor_k10-fixhint_n110.json \
  --label attention40-fixhint-anchor > $LOGS/budget_attention40-fixhint-anchor.txt 2>&1
echo "fixed hint + anchor: contexts done"

# 3. codegen, 3 runs each
for label in attention40-anchor attention40-fixhint-anchor; do
  for r in 1 2 3; do
    python -u src/run_codegen.py data/contexts/socbenchd_1_k10_budget-${label}_n110.json --run $r \
      > $LOGS/codegen_${label}_r$r.txt 2>&1
    echo "$label run $r done"
  done
done
echo "all done"