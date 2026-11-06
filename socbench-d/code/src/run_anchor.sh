#!/bin/sh
# AttentionRAG with a generated anchor token per chunk (as in the original
# method; chunks answered "none" are dropped), at equal budget, instance 1.
# Run from socbench-d/code/:
#   nohup caffeinate -i sh src/run_anchor.sh > data/logs/e2e/driver_anchor.txt 2>&1 &
set -e
LOGS=data/logs/e2e/socbenchd_1
mkdir -p $LOGS
python -u src/prepare_contexts.py --benchmark socbenchd_1 --top-k 40 --use-anchor --skip-existing \
  > $LOGS/prepare_k40_anchor.txt 2>&1
echo "contexts done"
python src/make_budget_contexts.py \
  --reference data/contexts/socbenchd_1_k10_all_noanchor_reqschema_n110.json \
  --source data/contexts/socbenchd_1_k40_all_anchor_k10_n110.json \
  --label attention40-anchor > $LOGS/budget_attention40-anchor.txt 2>&1
echo "budget done"
for r in 1 2 3; do
  python -u src/run_codegen.py data/contexts/socbenchd_1_k10_budget-attention40-anchor_n110.json --run $r \
    > $LOGS/codegen_attention40-anchor_r$r.txt 2>&1
  echo "codegen run $r done"
done
echo "all done"
