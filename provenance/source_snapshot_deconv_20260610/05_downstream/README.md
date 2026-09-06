# 05 — Downstream / 下游分析

ssGSEA (GSVA + limma), CellChat, Monocle2 trajectories, and the sample-specificity
investigation. Source `../paths.env` first.

> **Path note**: downstream scripts read/write the **canonical working tree**
> `$SRC0605/downstream/` (and `$SRC0605/GSE226077/`) regardless of `WORK_ROOT` — that is
> where the COVID / normal / iMGL generated artifacts already live. The pinned MSigDB
> snapshot is `$MSIGDB` (= `/disk1/maijl/deconv/script/msigdb_homosapiens.rds`, 2025.1.Hs).

## Fig5 — COVID ssGSEA (`run/run_downstream_covid.sh` wraps this)
`prep_ssgsea_avg.py` → `ssGSEA_linux.R` (GSVA + limma COVID19−Normal + heatmaps).
- **Gotcha**: `gsva_scores.rds` is reused by **column names only**, with no gene-set check —
  `rm -f $SRC0605/downstream/ssGSEA/gsva_scores.rds` before any fresh/changed run.
- Expected: 1242 significant cell-type-pathways (952 COVID-high / 290 Normal-high);
  Fig5a = `ht_main_sdlogfc_specific`, Fig5b = `sample_heatmap`.

## Fig6 — iMGL (GSE226077)
`../03_inference/route2_generate_gse226077.py` → `prep_ssgsea_gse226077.py` →
`ssGSEA_GSE226077_linux.R` → `figure6_ab_gse226077.py` → `monocle2_GSE226077_linux.R`.

## CellChat
`cellchat_linux.R` — **must** run sequential: `CC_WORKERS=1 $RSCRIPT cellchat_linux.R`.

## Sample-specificity investigation (Δz vs Route2-anchored)
`beyond_centroid_analysis.py`, `faithfulness_vs_truth.py`, `within_condition_resolution.py`,
`identity_both_conditions.py`, `alpha_scan_shrinkage.py`, `residual_realmag_heatmap.py`,
`sample_specificity_heatmap.py`, `within_condition_old_vs_new.py`, plus legacy solver probes
`solver_search_covid.py` / `covid_nnls_proportions.py` / `plot_nnls_fix.py`, and ssGSEA
clarifications `extract_gsva_nozscore.R` / `plot_gsva_nozscore.py` / `export_gsva_full.R` /
`cov_check.R` / `ablation_pathway_recovery.R`.

> Use the **right** files: OLD (Δz) = `$CVAE_ROOT/GSE159585/application/`; truth =
> `$DATA_ROOT/GSE159585/GSE159585_snRNAseq_raw_counts.h5ad` (**exclude IPF**). Do **not** use
> `combined_geneembed` for OLD or `GSE159585_combined.h5ad` for truth. See `../PIPELINE.md`
> Stage 5 and the source docs' "known corrections & pitfalls".
