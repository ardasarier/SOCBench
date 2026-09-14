RESTBENCH ARCHIVE

Completed experimental phase on RestBench (Spotify + TMDB), before switching to
SOCBench-D. Everything here is frozen; new runs go to data/contexts,
data/baselines, data/logs.

Naming scheme: <stage>_<benchmark>_k<retrieval>_<variant>_n<queries>[_r<run>]
  stage       prepare (retrieval + compression) or codegen (generation + scoring)
  k           retrieval depth, i.e. how many chunks were fetched
  variant     topk / t25 / middeep / identity / sentencesplit / ...
  n           number of queries
  r           repeat run


IMPORTANT: PREFIX-BUG

Files marked "_PREFIX-BUG" predate two scoring fixes and are NOT USABLE:

  a) match_to_template returned the first match while iterating a set.
     "/movie/now_playing" resolved to either the literal template or
     "/movie/{movie_id}" depending on Python's per-process hash seed.
     13 of 100 queries differed on identical cached code.
  b) path.strip("/") removed the trailing empty segment left when a runtime
     path parameter could not be resolved. "/movie/" counted as a hallucination.

Effect: recall 0.517 before vs 0.603 after. These files are kept only as
evidence of the bug's magnitude.


DIRECTORIES

01_json_segmentation/    n=20. Eq. 3 is broken on minified JSON. The first
                         sentence span starts at char 584, so endpoint identity
                         is unretainable at any k. Replaced with JSON-structure
                         segmentation + identity prefix.
                         sentencesplit 2.87x / recall 0.583
                         jsonseg250    1.65x / recall 0.633
                         jsonseg120    1.91x / recall 0.633

02_layer_bands_n10/      First layer comparison. shallow 2.03x, all 2.08x,
                         deep 2.22x. Confirmed at n=100 in 07.

03_anchor_ablation_n10/  Anchor token is optional. 2.08x with / 2.03x without,
                         25% faster, removes the "none" gate.
                         NO LOGS, ran without tee. Ratios are recomputable
                         from the JSONs.

04_threshold_sweep_n10/  Relative threshold: 0.10 -> 1.40x, 0.50 -> 2.42x,
                         0.70 -> 2.61x. t25 is missing -- overwritten by the
                         n=100 run (filenames had no n at the time).
                         NO LOGS.

05_scoring_fix_n10/      Before/after the scoring fixes at n=10.

06_main_comparison_n100/ The main comparison. Shared cached baseline
                         (recall 0.598 / precision 0.772).
                         topk 1.87x  recall 0.612 +-0.015  precision 0.814
                         t25  1.76x  recall 0.599 +-0.008  precision 0.790

07_middeep_n100/         Layers 8-23. 1.95x, compresses harder on 84/100
                         queries. recall 0.594 (unchanged), but precision
                         0.779 -- ranges do not overlap with all-layers.
                         Inverts Table 9 of the paper.

08_k_sweep_n100/         Retrieval depth. k=5 is an optimum:
                         k=10 recall 0.598 / k=5 0.618 / k=3 0.600
                         k=2 only prepared, never generated.

09_identity_ablation/    The key finding. Keep only the identity prefix, no
                         hint, no anchor, no forward pass.
                         167,005 tokens, recall 0.641 -- beats attention-guided
                         compression (250,363 / 0.613). Suggests a
                         prior-knowledge confound: gpt-4o knows TMDB.

baselines/               Cached uncompressed generations per k. Reused by
                         run_codegen.py so every condition is measured against
                         the same fixed reference.


MEASURED VARIANCE

+-0.015 recall at n=100 with a fixed baseline. Effects below ~0.03 require
repeats, hence 3 runs per condition.


OLD -> NEW

task1_contexts_tmdb_k10_sentencesplit.json -> 01/contexts_tmdb_k10_sentencesplit_n20.json
task1_contexts_tmdb_k10_jsonseg250.json    -> 01/contexts_tmdb_k10_jsonseg250_n20.json
task1_contexts_tmdb_k10_jsonsplit120.json  -> 01/contexts_tmdb_k10_jsonseg120_n20.json
task1_contexts_tmdb_k28_ss.json            -> 01/contexts_tmdb_k28_sentencesplit_n20.json
task2_tmdb_k10_sentencesplit.txt           -> 01/codegen_tmdb_k10_sentencesplit_n20.txt
task2_tmdb_k10_jsonseg250.txt              -> 01/codegen_tmdb_k10_jsonseg250_n20.txt
task2_tmdb_k10_jsonsplit120.txt            -> 01/codegen_tmdb_k10_jsonseg120_n20.txt
task2_tmdb_k28_ss.txt                      -> 01/codegen_tmdb_k28_sentencesplit_n20.txt

contexts_tmdb_k10_all.json                 -> 02/contexts_tmdb_k10_layer-all_n10.json
contexts_tmdb_k10_shallow.json             -> 02/contexts_tmdb_k10_layer-shallow_n10.json
contexts_tmdb_k10_middle.json              -> 02/contexts_tmdb_k10_layer-middle_n10.json
contexts_tmdb_k10_deep.json                -> 02/contexts_tmdb_k10_layer-deep_n10.json
task3_all.txt                              -> 02/codegen_tmdb_k10_layer-all_n10.txt
task3_deep.txt                             -> 02/codegen_tmdb_k10_layer-deep_n10.txt

contexts_tmdb_k10_all_anchor.json          -> 03/contexts_tmdb_k10_layer-all_anchor_n10.json
contexts_tmdb_k10_all_noanchor.json        -> 03/contexts_tmdb_k10_layer-all_noanchor_n10.json
contexts_tmdb_k10_middle_anchor.json       -> 03/contexts_tmdb_k10_layer-middle_anchor_n10.json
contexts_tmdb_k10_middle_noanchor.json     -> 03/contexts_tmdb_k10_layer-middle_noanchor_n10.json

contexts_tmdb_k10_all_noanchor_t0.1.json   -> 04/contexts_tmdb_k10_t10_n10.json
contexts_tmdb_k10_all_noanchor_t50.json    -> 04/contexts_tmdb_k10_t50_n10.json
contexts_tmdb_k10_all_noanchor_t70.json    -> 04/contexts_tmdb_k10_t70_n10.json

thresh_t25.txt                             -> 05/codegen_tmdb_k10_t25_n10_PREFIX-BUG.txt
thresh_topk.txt                            -> 05/codegen_tmdb_k10_topk_n10_PREFIX-BUG.txt
thresh_t25_v2.txt                          -> 05/codegen_tmdb_k10_t25_n10.txt
thresh_topk_v2.txt                         -> 05/codegen_tmdb_k10_topk_n10.txt

prepare_tmdb_k10_n100.txt                  -> 06/prepare_tmdb_k10_topk_n100.txt
contexts_tmdb_k10_all_noanchor_k10.json    -> 06/contexts_tmdb_k10_topk_n100.json
prepare_tmdb_k10_t25_n100.txt              -> 06/prepare_tmdb_k10_t25_n100.txt
contexts_tmdb_k10_all_noanchor_t25.json    -> 06/contexts_tmdb_k10_t25_n100.json
codegen_topk_n100.txt                      -> 06/codegen_tmdb_k10_topk_n100_PREFIX-BUG.txt
codegen_topk_n100_v2.txt                   -> 06/codegen_tmdb_k10_topk_n100_r1.txt
codegen_topk_n100_r2.txt                   -> 06/codegen_tmdb_k10_topk_n100_r2.txt
codegen_topk_n100_r3.txt                   -> 06/codegen_tmdb_k10_topk_n100_r3.txt
codegen_t25_n100.txt                       -> 06/codegen_tmdb_k10_t25_n100_PREFIX-BUG.txt
codegen_t25_n100_v2.txt                    -> 06/codegen_tmdb_k10_t25_n100_r1.txt
codegen_t25_n100_r2.txt                    -> 06/codegen_tmdb_k10_t25_n100_r2.txt
codegen_t25_n100_r3.txt                    -> 06/codegen_tmdb_k10_t25_n100_r3.txt

prepare_middeep_n100.txt                   -> 07/prepare_tmdb_k10_middeep_n100.txt
contexts_tmdb_k10_middeep_noanchor_k10.json-> 07/contexts_tmdb_k10_middeep_n100.json
codegen_middeep_n100.txt                   -> 07/codegen_tmdb_k10_middeep_n100_r1.txt
codegen_middeep_n100_r2.txt                -> 07/codegen_tmdb_k10_middeep_n100_r2.txt
codegen_middeep_n100_r3.txt                -> 07/codegen_tmdb_k10_middeep_n100_r3.txt

prepare_k5_n100.txt                        -> 08/prepare_tmdb_k5_topk_n100.txt
contexts_tmdb_k5_all_noanchor_k10.json     -> 08/contexts_tmdb_k5_topk_n100.json
codegen_k5_n100.txt                        -> 08/codegen_tmdb_k5_topk_n100_r1.txt
codegen_k5_n100_r2.txt                     -> 08/codegen_tmdb_k5_topk_n100_r2.txt
codegen_k5_n100_r3.txt                     -> 08/codegen_tmdb_k5_topk_n100_r3.txt
prepare_k3_n100.txt                        -> 08/prepare_tmdb_k3_topk_n100.txt
contexts_tmdb_k3_all_noanchor_k10.json     -> 08/contexts_tmdb_k3_topk_n100.json
codegen_k3_n100.txt                        -> 08/codegen_tmdb_k3_topk_n100_r1.txt
codegen_k3_n100_r2.txt                     -> 08/codegen_tmdb_k3_topk_n100_r2.txt
codegen_k3_n100_r3.txt                     -> 08/codegen_tmdb_k3_topk_n100_r3.txt
prepare_k2_n100.txt                        -> 08/prepare_tmdb_k2_topk_n100.txt
contexts_tmdb_k2_all_noanchor_k10.json     -> 08/contexts_tmdb_k2_topk_n100.json

prepare_k5_identity.txt                    -> 09/prepare_tmdb_k5_identity_n100.txt
contexts_tmdb_k5_all_noanchor_identity.json-> 09/contexts_tmdb_k5_identity_n100.json
codegen_k5_identity.txt                    -> 09/codegen_tmdb_k5_identity_n100_r1.txt
codegen_k5_identity_r2.txt                 -> 09/codegen_tmdb_k5_identity_n100_r2.txt

baseline_codegen_tmdb_k10.json             -> baselines/baseline_tmdb_k10.json
baseline_codegen_tmdb_k5.json              -> baselines/baseline_tmdb_k5.json
baseline_codegen_tmdb_k3.json              -> baselines/baseline_tmdb_k3.json