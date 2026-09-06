# DeconvSC Figshare Reproduction Package

Version 1.0.0, frozen on 2026-08-09.

This directory is the standalone release package for the DeconvSC manuscript
workflow. It contains the portable code, configuration files, processed input
data, locked model weights, generated single-cell-resolution expression
matrices, frozen benchmark tables, manuscript figures, downstream result
objects, environment files and provenance records needed to inspect or
reproduce the submitted results.

The fastest reviewer path is:

```bash
cd /disk1/maijl/deconv/papercode2
python scripts/validate_release.py
sha256sum -c MANIFEST.sha256
```

The validation report at `provenance/validation_report.json` records 78/78
package checks passing at release time. The checksum file covers the released
files and should pass after download or upload.

## Reproduction Overview

The recommended reproduction order follows the manuscript analysis logic:

1. Validate the release package and environment.
2. Reproduce model-evaluation outputs: expression-prediction benchmarks and
   generated-cell evaluation.
3. Reproduce ablation outputs: within-cell-type topology metrics and
   residual-strength ablation.
4. Reproduce downstream validation outputs: COVID-19 lung pathway/CellChat
   analysis and iMGL trajectory/pathway analysis.

The table below links each analysis block to its released data and code.

| Analysis block | Manuscript outputs | Source data in this package | Main reproduction code |
|---|---|---|---|
| Model evaluation | Figures 2 and 3, Supplementary Figure 3, benchmark tables | `benchmark_tables/expression_prediction/03_result_tables/`, `model_outputs/` | `code/03_model_evaluation/`, `scripts/evaluate_prediction.py` |
| Ablation analysis | Figure 4 and ablation result tables | `results/ablation/`, `model_outputs/GSE141115/` | `code/04_ablation/` |
| Downstream validation | Figure 5, Figure 6 and downstream result objects | `results/downstream/`, `model_outputs/GSE159585/`, `model_outputs/GSE226077/` | `code/05_downstream/` |
| Release validation | Package checksums and provenance | `MANIFEST.sha256`, `provenance/` | `scripts/validate_release.py`, `scripts/build_release_manifests.py` |

More detailed source mapping is available in:

- `provenance/FIGURE_SOURCE_MAP.tsv`
- `provenance/ARTIFACT_SOURCE_MAP.tsv`
- `provenance/ARTIFACT_MANIFEST.tsv`
- `provenance/DATA_DICTIONARY.tsv`

## Directory Guide

| Path | Meaning |
|---|---|
| `deconvsc/` | Portable Python implementation of the attention-based conditional VAE, prior-anchored inference, preprocessing, evaluation and plotting helpers. |
| `scripts/` | Main command-line entry points for preprocessing, training, prediction, evaluation, figure reproduction, validation and release packaging. |
| `configs/` | Dataset-specific YAML files controlling preprocessing, training, prediction, evaluation and full pipeline runs. |
| `code/01_preprocessing/` | Preprocessing module wrapper and documentation. |
| `code/02_model_training/` | CVAE training wrapper and GSE141115 architecture reference. |
| `code/03_model_evaluation/` | Expression-prediction benchmark code, Supplementary Figure 5 code and shared evaluation helpers. |
| `code/04_ablation/` | Figure 4 ablation code, including within-cell-type topology metrics and PCA <= 50 co-expression-profile analysis. |
| `code/05_downstream/` | Downstream CellChat, ssGSEA/GSVA, Monocle2 and Figure 6 plotting code. |
| `processed_data/` | Processed reference, bulk and held-out validation data used by the manuscript workflow. |
| `model_weights/` | Locked CVAE checkpoints and cell-type prior tensors used for released inference. |
| `model_outputs/` | Locked generated single-cell-resolution expression matrices, named by manuscript role. |
| `metadata/` | Cell-type label maps and compact `.obs` metadata exported from generated H5AD files. |
| `benchmark_tables/` | Frozen expression-prediction and proportion benchmark tables used for Figures 2 and 3 and Supplementary Figure 3. |
| `results/` | Frozen result tables, source objects and figures for ablation, Supplementary Figure 5 and downstream analyses. |
| `environment/` | Conda, pip, R package and recorded session information. |
| `provenance/` | Validation report, checksums, data dictionary, external-resource notes and source snapshots. |
| `docs/` | Reserved for additional documentation; currently not required for reproduction. |

The supported portable entry points are `scripts/`, `deconvsc/`, `configs/` and
`code/`. The directory `provenance/source_snapshot_deconv_20260610/` contains
an unmodified snapshot of the historical workstation scripts for audit only.
Those files retain original absolute paths and are not the recommended run
interface.

## Key Files By Purpose

### Portable Python package: `deconvsc/`

| File | Meaning |
|---|---|
| `deconvsc/model.py` | Attention-based conditional VAE architecture. |
| `deconvsc/trainer.py` | CVAE training loop. |
| `deconvsc/infer.py` | Prior-anchored bulk-conditioned inference and generated-cell decoding. |
| `deconvsc/evaluate.py` | Expression prediction evaluation utilities. |
| `deconvsc/preprocess.py` | Preprocessing utilities. |
| `deconvsc/config.py` | YAML configuration loading and validation. |
| `deconvsc/io.py` | H5AD/CSV input-output helpers. |
| `deconvsc/loss.py` | Training losses. |
| `deconvsc/visualization.py` and `deconvsc/plot_style.py` | Plotting helpers and style settings. |
| `deconvsc/main.py` | Package-level command interface. |
| `deconvsc/utils.py` | General helper functions. |

### Main scripts: `scripts/`

| File | Meaning |
|---|---|
| `scripts/run_pipeline.py` | Runs a configured preprocessing, inference and evaluation workflow. |
| `scripts/preprocess_data.py` | Top-level preprocessing command. |
| `scripts/train_model.py` | Top-level training command. |
| `scripts/predict_bulk.py` | Generates single-cell-resolution expression from bulk data using locked or newly trained weights. |
| `scripts/evaluate_prediction.py` | Computes prediction benchmark metrics. |
| `scripts/visualize_results.py` | General visualization entry point. |
| `scripts/reproduce_figures.sh` | Convenience script for figure reproduction. |
| `scripts/validate_release.py` | Validates required files, configs, figures and hashes. |
| `scripts/build_release_manifests.py` | Builds release manifests. |
| `scripts/export_release_metadata.py` | Exports compact metadata from released artifacts. |
| `scripts/make_archive.sh` | Creates the uploadable archive. |

### Input data: `processed_data/`

| Path | Meaning |
|---|---|
| `processed_data/real_data/mouse_kidney/` | GSE141115 mouse kidney reference, bulk and held-out real data. |
| `processed_data/real_data/mouse_kidney/train_data.h5ad` | GSE141115 training reference. |
| `processed_data/real_data/mouse_kidney/test_data.h5ad` | GSE141115 held-out ground-truth single-cell/snRNA data. |
| `processed_data/real_data/mouse_kidney/bulk_counts.txt` | GSE141115 bulk-count input. |
| `processed_data/real_data/mouse_kidney/train_data_canonical20.h5ad` | Canonical 20-cell-type GSE141115 reference used by the release checks. |
| `processed_data/real_data/human_lung/` | GSE159585 human lung reference, bulk and held-out real data. |
| `processed_data/real_data/human_iMGL/` | GSE226077 human iMGL reference and bulk TPM input. |
| `processed_data/simulation_data/HCA/` | HCA simulation reference, synthetic bulk and validation truth. |

### Locked model weights: `model_weights/`

| Path | Meaning |
|---|---|
| `model_weights/GSE141115/deconvsc/` | GSE141115 attention-CVAE checkpoint and prior tensors. |
| `model_weights/GSE159585/benchmark_reference/` | GSE159585 main benchmark checkpoint and prior tensors. |
| `model_weights/GSE159585/covid_application/` | COVID-19 lung downstream checkpoint, prior tensors and gene metadata. |
| `model_weights/GSE159585/normal_application/` | Normal-lung downstream checkpoint, prior tensors and gene metadata. |
| `model_weights/HCA/fold2/` | HCA fold-2 checkpoint and prior tensors. |
| `model_weights/GSE226077/application/` | iMGL downstream checkpoint, prior tensors and gene/signature metadata. |

The main checkpoint files are `scvae_best.pth`; the cell-type prior files are
`cell_type_mu_logvar_best.pt`.

### Locked generated outputs: `model_outputs/`

| File | Manuscript role |
|---|---|
| `model_outputs/GSE141115/prophead_generated.h5ad` | Main GSE141115 expression benchmark output. |
| `model_outputs/GSE141115/figure4_suppfigure5_deconvsc_generated.h5ad` | Locked DeconvSC matrix for Figure 4 and Supplementary Figure 5. |
| `model_outputs/GSE141115/figure4_ablated_vae_generated.h5ad` | Locked ablated-VAE matrix for Figure 4. |
| `model_outputs/GSE159585/prophead_generated.h5ad` | Main GSE159585 benchmark output. |
| `model_outputs/GSE159585/covid_application_generated.h5ad` | COVID-19 lung downstream output. |
| `model_outputs/GSE159585/normal_application_generated.h5ad` | Normal-lung downstream output. |
| `model_outputs/HCA/prophead_generated.h5ad` | HCA fold-2 benchmark output. |
| `model_outputs/GSE226077/prophead_generated.h5ad` | iMGL downstream output. |

Do not interchange the GSE141115 prophead benchmark matrix with the separate
Figure 4/Supplementary Figure 5 matrix. Their roles and hashes are documented in
`provenance/ARTIFACT_SOURCE_MAP.tsv` and
`provenance/ARTIFACT_MANIFEST.tsv`.

### Metadata: `metadata/`

| File type | Meaning |
|---|---|
| `*_generated_obs.csv.gz` | Compact cell-level metadata exported from the corresponding generated H5AD file. |
| `GSE141115/label_id_to_celltype_functional.csv` | GSE141115 label-id to functional cell-type map. |
| `GSE141115/figure4_source_label_map.csv` | Source and label map used by the Figure 4 analysis. |

### Benchmark and result files

| Path | Meaning |
|---|---|
| `benchmark_tables/expression_prediction/03_result_tables/summary_all_methods.csv` | Combined expression benchmark summary across methods and datasets. |
| `benchmark_tables/expression_prediction/03_result_tables/TABLE_*.csv` | Dataset-level expression benchmark tables. |
| `benchmark_tables/expression_prediction/03_result_tables/*_genewise_by_method.csv` | Gene-wise PCC tables by method. |
| `benchmark_tables/expression_prediction/03_result_tables/unified/` | Proportion estimates and cached unified benchmark objects. |
| `results/expression_prediction/figures/` | Frozen expression benchmark figures and the CSV files used to draw them. |
| `results/ablation/within_celltype/` | Figure 4a-d metrics, audits, validation files and plots. |
| `results/ablation/alpha_sweep/` | Figure 4e alpha-sweep result table and panel. |
| `results/supplementary_figure5/` | Corrected Supplementary Figure 5 data, figures and validation files. |
| `results/downstream/GSE159585/` | Figure 5 CellChat and ssGSEA/GSVA source objects and figures. |
| `results/downstream/GSE226077/` | Figure 6 Monocle2, ssGSEA/GSVA, UMAP, marker and trajectory outputs. |
| `results/manuscript_tables/SupplementaryTables.xlsx` | Supplementary tables used with the manuscript. |

## Environment Setup

Create the Python environment:

```bash
cd /disk1/maijl/deconv/papercode2
conda env create -f environment/conda_environment.yml
conda activate deconvsc
pip install -r environment/requirements.txt
```

Install R dependencies for plotting and downstream analyses:

```bash
Rscript environment/install_R_packages.R
```

See `environment/README.md`, `environment/R_packages.txt` and
`environment/session_info.txt` for exact package records.

The MSigDB 2025.1.Hs snapshot is license controlled and is not redistributed.
For ssGSEA/GSVA scripts, set:

```bash
export MSIGDB_SNAPSHOT=/path/to/authorized/msigdb_2025.1.Hs.rds
```

## Reproduction Commands

Run commands from the package root.

### Step 0: validate the release package

```bash
cd /disk1/maijl/deconv/papercode2
python scripts/validate_release.py
sha256sum -c MANIFEST.sha256
```

Expected result: all required release checks pass, and all files listed in
`MANIFEST.sha256` verify successfully.

### Step 1: reproduce model-evaluation results

Run the full released-checkpoint pipelines for the three benchmark datasets.
These commands write new outputs below `work/` and do not overwrite locked
release artifacts.

```bash
python scripts/run_pipeline.py --config configs/pipeline_hca.yaml
python scripts/run_pipeline.py --config configs/pipeline_gse141115.yaml
python scripts/run_pipeline.py --config configs/pipeline_gse159585.yaml
```

Regenerate the expression-prediction and proportion benchmark figures from the
frozen cross-method benchmark tables:

```bash
Rscript code/03_model_evaluation/expression_benchmark/plot_pcc_per_dataset.R
Rscript code/03_model_evaluation/expression_benchmark/plot_proportion_pcc.R
```

Recompute the GSE141115 generated-cell normalized-expression and co-expression
evaluation:

```bash
python code/03_model_evaluation/supplementary_figure5/build_supplementary_figure5.py
```

Model-evaluation inputs are in:

- `processed_data/`
- `model_weights/`
- `model_outputs/`
- `benchmark_tables/expression_prediction/03_result_tables/`

Frozen model-evaluation outputs are in:

- `results/expression_prediction/figures/`
- `results/supplementary_figure5/`

The cross-method reproduction boundary for CIBERSORTx, BayesPrism, TAPE and
DISSECT is the final method-specific CSV table set in
`benchmark_tables/expression_prediction/03_result_tables/`. The release does
not redistribute commercial web-service runtimes or every private intermediate
matrix from those third-party methods.

### Step 2: reproduce ablation results

Regenerate the within-cell-type ablation figure from locked metric tables:

```bash
python code/04_ablation/within_celltype/scripts/plot_figure4_within_celltype.py \
  --data results/ablation/within_celltype/figure4_inputs/topology_metrics_long.csv \
  --edge-profile-data results/ablation/within_celltype/figure4_inputs/figure4d_pca50_symmetric_profile_pcc.csv \
  --outdir work/figure4
```

Optionally recompute the ablation metric tables from cell-level matrices before
plotting:

```bash
python code/04_ablation/within_celltype/scripts/run_within_celltype_topology.py
python code/04_ablation/within_celltype/run_no_real_split.py
python code/04_ablation/within_celltype/scripts/validate_figure4_pca50.py
```

Ablation inputs are in:

- `processed_data/real_data/mouse_kidney/test_data.h5ad`
- `model_outputs/GSE141115/figure4_suppfigure5_deconvsc_generated.h5ad`
- `model_outputs/GSE141115/figure4_ablated_vae_generated.h5ad`
- `results/ablation/within_celltype/figure4_inputs/`
- `results/ablation/alpha_sweep/`

Ablation outputs are in:

- `results/ablation/within_celltype/`
- `results/ablation/alpha_sweep/`

### Step 3: reproduce downstream validation results

The expensive downstream result objects are included under `results/` and can
be inspected without rerunning CellChat, GSVA or Monocle2.

GSE159585 COVID-19/normal lung:

```bash
python code/05_downstream/prep_ssgsea_avg.py
MSIGDB_SNAPSHOT=/path/to/msigdb_2025.1.Hs.rds \
  Rscript code/05_downstream/ssGSEA_linux.R
Rscript code/05_downstream/cellchat_linux.R
```

GSE226077 iMGL:

```bash
python code/05_downstream/prep_ssgsea_gse226077.py
MSIGDB_SNAPSHOT=/path/to/msigdb_2025.1.Hs.rds \
  Rscript code/05_downstream/ssGSEA_GSE226077_linux.R
Rscript code/05_downstream/monocle2_GSE226077_linux.R
python code/05_downstream/figure6_ab_gse226077.py
```

Downstream inputs are in:

- `processed_data/real_data/human_lung/`
- `processed_data/real_data/human_iMGL/`
- `model_outputs/GSE159585/`
- `model_outputs/GSE226077/`

Downstream outputs are in:

- `results/downstream/GSE159585/`
- `results/downstream/GSE226077/`

### Optional: train a new model

The release provides locked checkpoints for reproducing the submitted results.
To train a new CVAE instead of using the locked checkpoint, run:

```bash
python code/01_preprocessing/preprocess_data.py --config configs/preprocess_gse141115.yaml
python code/02_model_training/train_model.py --config configs/train_gse141115.yaml
```

New training does not overwrite the released checkpoint. To evaluate a newly
trained model, point the relevant prediction config to:

```text
work/<dataset>/train/scvae_best.pth
```

## Reproducibility Boundary

The main benchmark matrices in `model_outputs/*/prophead_generated.h5ad` and
the tables in `benchmark_tables/expression_prediction/03_result_tables/` come
from the 20260610 prophead workflow.

Historical end-to-end bitwise retraining cannot be claimed because the original
online residual-network and proportion-head states were not saved as standalone
checkpoints with complete optimizer states and solver metadata. The released
code can train new versions of those components, but stochastic retraining is
expected to produce numerically different generated cells.

Figure 4 reports undirected expression dependency and co-expression fidelity.
It does not establish directed or causal gene-regulatory networks. Cell type is
the summary unit for the displayed Figure 4 distributions; cells, genes, gene
pairs, modules and random seeds are not independent biological replicates.

See `provenance/REPRODUCIBILITY_SCOPE.md` for the full boundary statement.

## Final Upload Check

Before uploading to figshare, run:

```bash
cd /disk1/maijl/deconv/papercode2
python scripts/validate_release.py
sha256sum -c MANIFEST.sha256
```

The prepared archive from this release was:

```text
/disk1/maijl/deconv/DeconvSC_Figshare_v1.0.0.tar.gz
```

The archive checksum is written to the sidecar file:

```text
/disk1/maijl/deconv/DeconvSC_Figshare_v1.0.0.tar.gz.sha256
```
