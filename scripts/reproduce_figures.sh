#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

Rscript code/03_model_evaluation/expression_benchmark/plot_pcc_per_dataset.R
Rscript code/03_model_evaluation/expression_benchmark/plot_proportion_pcc.R

python code/04_ablation/within_celltype/scripts/plot_figure4_within_celltype.py \
  --data results/ablation/within_celltype/figure4_inputs/topology_metrics_long.csv \
  --edge-profile-data results/ablation/within_celltype/figure4_inputs/figure4d_pca50_symmetric_profile_pcc.csv \
  --outdir work/figure4

python code/03_model_evaluation/supplementary_figure5/build_supplementary_figure5.py

echo "Core benchmark, Figure 4 and Supplementary Figure 5 outputs are under work/."
