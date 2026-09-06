# GSE141115 within-cell-type topology 实验记录

## Material Passport

- Origin Skill: `academic-research-suite / experiment-agent`
- Origin Mode: run
- Origin Date: 2026-07-13
- Verification Status: `VERIFIED`
- Version Label: `within_celltype_topology_experiment_v2_edge_mae`

## Experiment Result

- **ID**: `GSE141115_within_celltype_topology_20260710`
- **Type**: analysis
- **Status**: completed
- **Working Directory**: `/disk1/maijl/deconv/deconv_20260610/prophead/GSE141115/topology_extended_benchmark/within_celltype`
- **Primary run duration**: 115.0 s
- **Deterministic rerun duration**: 117.7 s
- **GPU**: not used
- **Network/external upload**: not used
- **Anomalies**: Gate 0 falsified the original per-cell CP10K assumption; this was handled by a recorded protocol amendment and S7 native-scale sensitivity. Figure 4d was subsequently changed from per-gene connectivity-profile PCC to per-gene co-expression edge MAE after the PCC result had been inspected; this post-result revision is recorded and the original PCC outputs are retained as supplementary metrics.

## 1. Legacy bridge reproduction

The unmodified legacy script was run twice in isolated directories with:

```bash
env \
  MPLCONFIGDIR=cache/matplotlib \
  OURS_H5AD=/disk1/maijl/deconv/cVAE/GSE141115/attention_geneembed/generated_data.h5ad \
  ABLATION_DIR=cache/legacy_bridge_run1 \
  WT_MIN_CELLS=100 \
  /disk1/maijl/software/miniforge3/envs/pytorch/bin/python \
  /disk1/maijl/deconv/deconv_20260610/04_evaluation/within_type_module_l2.py
```

The second execution used `cache/legacy_bridge_run2`. Comparison command:

```bash
/disk1/maijl/software/miniforge3/envs/pytorch/bin/python \
  scripts/compare_legacy_bridge.py \
  --run1 cache/legacy_bridge_run1 \
  --run2 cache/legacy_bridge_run2 \
  --output legacy_bridge_reproduction.csv
```

Verdict: 5/5 arrays/tables exact.

## 2. Primary analysis

```bash
env \
  MPLCONFIGDIR=cache/matplotlib \
  PYTHONUNBUFFERED=1 \
  /disk1/maijl/software/miniforge3/envs/pytorch/bin/python \
  scripts/run_within_celltype_topology.py \
  --outdir /disk1/maijl/deconv/deconv_20260610/prophead/GSE141115/topology_extended_benchmark/within_celltype
```

Log: `logs/main_analysis.log`.

The initial native-scale execution is retained as `logs/main_analysis_native_initial.log`; it was superseded after the Gate 0 scale audit, not silently discarded.

## 3. Deterministic rerun

The same command was executed with:

```text
--outdir .../within_celltype/cache/repro_run
```

Log: `logs/repro_run.log`.

Comparison:

```bash
/disk1/maijl/software/miniforge3/envs/pytorch/bin/python \
  scripts/compare_reproducibility.py \
  --reference /disk1/maijl/deconv/deconv_20260610/prophead/GSE141115/topology_extended_benchmark/within_celltype \
  --rerun /disk1/maijl/deconv/deconv_20260610/prophead/GSE141115/topology_extended_benchmark/within_celltype/cache/repro_run \
  --output reproducibility_comparison.csv
```

Verdict: 21/21 CSV/NPZ artifacts were structurally identical and numerically exact; maximum numeric difference was zero.

Final validation command:

```bash
/disk1/maijl/software/miniforge3/envs/pytorch/bin/python \
  scripts/validate_outputs.py \
  --outdir /disk1/maijl/deconv/deconv_20260610/prophead/GSE141115/topology_extended_benchmark/within_celltype
```

Verdict: 60/60 integrity, statistical-unit, provenance, sensitivity, Figure 4 input and rendered-figure checks passed.

## 4. Figure 4-style 2×2 visualization

```bash
env MPLCONFIGDIR=cache/matplotlib \
  /disk1/maijl/software/miniforge3/envs/pytorch/bin/python \
  scripts/plot_figure4_within_celltype.py \
  --data figure4_inputs/topology_metrics_long.csv \
  --outdir figures
```

The combined A–D figure and four standalone panels were exported as PNG, PDF, SVG and EMF with editable embedded Arial text. Panel d displays per-gene co-expression edge MAE: for each gene, absolute generated-minus-ground-truth Pearson edge differences are averaged over the other 499 genes, and the median across 500 genes is averaged over ten seeds. This provides a dense gene-centred, all-panel edge-calibration view that complements the sparse hub-neighbour overlap in panel a, non-linear MI error in panel b and ground-truth-module block error in panel c. The final display contains box plots, all six cell-type points and paired lines, with no violin or kernel-density layer. All inferential annotations (brackets, stars, p values and `ns`) were intentionally omitted. Plot values, displayed layers and the absence of statistical annotations are recorded in `figures/Figure4_within_celltype_ABCD_plot_audit.csv`; statistical results remain in their original result tables.

## 5. Main output inventory

- Input/provenance: `input_manifest.csv`, `input_schema_audit.csv`, `provenance_checks.csv`.
- Eligibility/design: `eligibility_matrix.csv`, `cell_count_audit.csv`, `setting_eligibility.csv`, `analysis_config.json`.
- Architecture/protocol: `architecture_equivalence_audit.csv`, `protocol_deviations.csv`.
- Gene/module definitions: `within_type_gene_panel.csv`, `within_type_modules.csv`.
- Raw/aggregated metrics: `within_type_metrics_by_seed.csv`, `within_type_metrics_by_celltype.csv`, `module_l2_by_celltype_module_seed.csv`, `hub_jaccard_by_celltype_hub_seed.csv`, `raw_distributions.npz`.
- Statistics: `statistical_tests.csv`, `sensitivity_analysis.csv`.
- Figure inputs: `figure4_inputs/`.
- Reports/validation: `analysis_report_zh.md`, `validation_report.md`, `validation_checks.csv`, `output_manifest.csv`.

All source inputs remained read-only; the final validator re-hashes them against `input_manifest.csv`.
