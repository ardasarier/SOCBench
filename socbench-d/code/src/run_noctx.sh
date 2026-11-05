#!/bin/sh
# No-context condition: SOCBench-D 1-5 (1 run each), RestBench Spotify and TMDB (3 runs each).
# Run from socbench-d/code/:  nohup caffeinate -i sh src/run_noctx.sh > data/logs/e2e/driver_noctx.txt 2>&1 &
set -e
for i in 1 2 3 4 5; do
  mkdir -p data/logs/e2e/socbenchd_$i
  python src/make_noctx_contexts.py --reference data/contexts/socbenchd_${i}_k10_all_noanchor_reqschema_n110.json
  python -u src/run_codegen.py data/contexts/socbenchd_${i}_k10_budget-noctx_n110.json --run 1 \
    > data/logs/e2e/socbenchd_$i/codegen_noctx_r1.txt 2>&1
  echo "socbenchd_$i done"
done
python src/make_noctx_contexts.py --reference data/contexts/restbench-spotify_k10_all_noanchor_reqschema_n57.json --label spotify-noctx
python src/make_noctx_contexts.py --reference data/contexts/restbench-tmdb_k10_all_noanchor_reqschema_n100.json --label tmdb-noctx
for r in 1 2 3; do
  python -u src/run_codegen.py data/contexts/restbench_k10_budget-spotify-noctx_n57.json --run $r \
    > data/logs/e2e/restbench-spotify/codegen_noctx_r$r.txt 2>&1
  python -u src/run_codegen.py data/contexts/restbench_k10_budget-tmdb-noctx_n100.json --run $r \
    > data/logs/e2e/restbench-tmdb/codegen_noctx_r$r.txt 2>&1
  echo "restbench run $r done"
done
echo "all done"
