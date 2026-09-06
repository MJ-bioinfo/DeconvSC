# 02 — Stage-1 training / 模型训练

Trains the **attention conditional VAE** once per dataset, then **freezes** it. Inference
(Stage 3) consumes only two artifacts: `scvae_best.pth` (decode weights) and
`cell_type_mu_logvar_best.pt` (per-cell-type latent priors `{t: (μ_t, logσ²_t)}`).

> These scripts are **monoliths**: each loads + preprocesses the reference, synthesises
> training pseudobulks, trains the VAE, and writes the two artifacts. They embed the
> Stage-1 *data processing* recipe documented in `../01_data_processing/README.md`.

> **Weights already exist** under `$CVAE_ROOT` — retraining is **optional**. Inference and
> evaluation run directly against the existing checkpoints.

| Dataset | Script | Existing weights (under `$CVAE_ROOT`) |
|---|---|---|
| HCA_fold2 | `HCA_simulation_v3.py` (also does fold/bulk/GT simulation) | `Simulation_HCA/result/HCA/v3_pureprior/fold_2/` |
| GSE141115 | `GSE141115_geneembed_vae_noiter.py` | `GSE141115/attention_geneembed/` |
| GSE159585 | `GSE159585-geneembed.py` | `GSE159585/geneembed/` |
| GSE159585-COVID | `covid_train.py` (wraps `GSE159585-geneembed.py`) | `GSE159585/covid/` |
| GSE226077 (iMGL) | `GSE226077-geneembed.py` | reused at inference, **no retrain** |

## Architecture / hyperparameters (verified per dataset)
| dataset | core in | out G | hidden_size_list | mid_hidden | embed | heads | layers | batch | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|
| HCA_fold2 | 3000 | 25366 | [2048,1024,512] | 512 | 32 | 4 | 2 | 128 | 1e-4 | 50 |
| GSE141115 | 2940 | common | [4096,2048,1024] | 256 | 32 | 4 | 4 | 512 | 5e-5 | 50 |
| GSE159585 | 2022 | common | [4096,2048,1024] | 256 | 32 | 4 | 4 | 512 | 5e-5 | 50 |

**Composite loss**: `λ_recon·MSE + β·KL + λ_corr·gene-corr`; β warms 1e-5→0.05 over the first
20% of epochs; `λ_corr` anneals from 20.0; `AdamW(weight_decay=1e-4)`. See `../PIPELINE.md`
Stage 2 and the source docs for the full training detail.
