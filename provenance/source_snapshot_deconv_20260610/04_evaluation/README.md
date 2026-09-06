# 04 — Evaluation / 评测

Scores expression reconstruction on one shared log1p-CP10K pseudobulk scale, across 4
Pearson axes (profile / sample / gene / pseudobulk) × 2 gene sets (all / marker).
Source `../paths.env` first.

| Script | Role |
|---|---|
| `run_eval_v3_route2.py` | **main expression eval**. Recomputes **only** DeconvSC (=Route2 cells, read from `$WORK_ROOT/<ds>/generated_data.h5ad`); the 4 competitors (BayesPrism/CIBERSORTx/DISSECT/TAPE) are reused **verbatim** from the cVAE eval tree `…/evaluation_unified_pearson_20260528/03_results/<ds>/<m>/pearson_*.csv`. Scores all 3 datasets in one run. `--outdir` to override the output tree. |
| `04_plot_pcc_per_dataset.R` | **Figure 2 / Figure 3 / Supplementary Figure 3** — per-(dataset,method) PCC figures. Reads `$WORK_ROOT/03_result_tables/<ds>/<method>/pearson_*.csv`, writes `$WORK_ROOT/03_result_figures/figures/`. Produces: per-cell-type bars (`<ds>_{profile,pseudobulk}_{fine,major}_pcc`), per-dataset method boxes (`main_{profile,genewise,samplewise}_pcc_<ds>` marker + `main_genewise_allgenes_pcc_<ds>` all-genes), and the combined 4-axis method boxes (`<ds>_methods_box_{marker,all}_genes`). Replaces `plot_v3_full_figures.py` + run_eval's Dark2 methods_box (2026-06-10). **Single `palette`** → every method's colour is identical across bars, boxes, and the proportion figure. Run with `$RSCRIPT`. |
| `plot_v3_full_figures.py` / `make_route2_figures.py` | (superseded by `04_plot_pcc_per_dataset.R` for Fig2/3/SuppFig3; kept for reference) |
| `05_calculate_attention_ablation.py` → `05_Figure4_visualization.py` | **Fig4** attention / co-expression network ablation (`network_pcc`, hub/module, MI). Tunable via `AA_TOPK`/`AA_SEEDS`. |
| `make_celltype_tables_by_dataset.py` / `make_summary_tables_by_dataset.py` / `export_benchmark_tables_v3.py` | per-type / summary / benchmark tables |

**Important**: `run_eval_v3_route2.py` reads its generated cells through `$WORK_ROOT`, which
must match the value used at generation time (Stage 3). With the default
`WORK_ROOT=$SRC0605`, it finds the published generated cells for all three datasets.

Expected: HCA fold2 gene-wise Route2 ≈ **0.54** vs DeltaZ −0.003 vs BayesPrism 0.257;
HCA 5-fold gene-wise & marker **5/5** beat BayesPrism; `network_pcc` **0.717** vs DeltaZ
0.556. **Do not recompute already-landed competitor PCCs** — they are reused verbatim.
See `../PIPELINE.md` Stage 4.
