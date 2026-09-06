# Module 4: ablation analysis

The current primary ablation compares DeconvSC with the attention-ablated VAE
within six eligible GSE141115 kidney cell types. Panels a-d quantify hub-neighbor
overlap, mutual-information error, module L2 error and PCA <= 50/Ledoit-Wolf
per-gene co-expression-profile PCC. `within_celltype/scripts/` contains the
metric and plotting code; `results/ablation/within_celltype/` contains the
frozen tidy source tables.

The older global three-model violin/network-fidelity analysis is intentionally
excluded from this release.
