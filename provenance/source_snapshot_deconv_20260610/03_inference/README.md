# 03 — Inference (Route2) / 模型推断

Frozen Stage-1 decoder + priors → generate cells, and predict cell-type proportions.
Core formula: **`ê_t(b_s) = ψ(μ_t) + α·δθ(b_s^core, t)`** (α=1); the bulk-conditioned
residual `δθ = mask ⊙ 2·tanh(·/2) ∈ (−2,2)` is masked to marker∪core genes and trained
offline on Dirichlet synthetic bulks from the reference donors. Source `../paths.env` first.

## (a) Expression generation  → `$WORK_ROOT/<ds>/generated_data.h5ad`
| Script | Dataset / use | Proportions used |
|---|---|---|
| `route2_generate_hca.py` | HCA_fold2 (env `OUT`/`AG_FOLD`/`CPS`/`DIV`/`SMOOTH_K`) | softmax (self-solved) |
| `route2_generate_gse.py --dataset {GSE141115,GSE159585} [--min_cells_per_type N]` | GSE | softmax (self-solved) |
| `route2_generate_downstream.py --which {covid,normal_18_24}` | Fig5 COVID (→ `$SRC0605/downstream/`) | softmax |
| `route2_generate_gse226077.py` | Fig6 iMGL (reuses pretrained model) | softmax |
| `route2_generate_nnls.py --which … [--props_csv F] [--solver nnls_core\|softmax]` | pluggable proportions | external CSV / NNLS / softmax |

`--min_cells_per_type N` floors cells/type for full expression coverage **without changing
proportions** (expression is decoupled from the proportion vector).

## (b) Proportions — unified prophead + baselines
Driver lives in `../lib/proportions_unified.py` (it is dual-purpose: library `PU` **and**
CLI). Methods: `softmax / nnls_core / nnls_ridge / dirichlet_map / prophead / prophead_ref`.
`prophead` = amortised softmax head trained on π_ref-sampled synthetic pseudobulks + Scaden-style
input noise (the 2026-06-10 fix for the over-uniform/over-sparse proportion problem).

| Script | Role |
|---|---|
| `python ../lib/proportions_unified.py --dataset {hca_fold2,gse141115,gse159585} [--pseudobulk T] [--real_bulk covid,normal] [--tag T]` | fit π_ref, run all methods, write `proportions_<ds>[_<tag>]_<method>.csv` + cache `.npz` |
| `eval_proportions_unified.py --dataset … --tag … --gt G.csv` | metrics vs GT (sample/celltype/overall PCC) + composition realism |
| `identity_realbulk.py --cache cache_<ds>_real.npz --tag real` | real-bulk sample identity + composition health (no GT) |
| `expr_coverage_test.py --dataset …` (+ `_gse159585.py`) | proportion-free per-type profile/gene PCC + per-solver coverage |

Expected (GSE159585): real-bulk identity 57% (COVID 86 / normal 100), eff≈13.5; held-out
pseudobulk overall PCC 0.74 (GSE141115 0.92). See `../PIPELINE.md` Stage 3.
