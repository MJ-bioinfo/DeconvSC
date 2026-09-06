# Figure 4-ready within-cell-type inputs

- `topology_metrics_long.csv`: source for panels a-c (hub-neighbour Jaccard, MI fidelity error and module L2 error).
- `figure4d_pca50_symmetric_profile_pcc.csv`: authoritative no-real-split source for panel d. It contains six cell types x two models, 200 genes, ten seeds and the PCA<=50 per-gene co-expression profile PCC.
- `paired_statistics.csv`: archived cell-type-level paired tests and effect sizes; no inferential annotations are displayed in the current figure.
- `within_celltype_panel_data.npz`: legacy compact arrays containing current and superseded endpoints.
- `module_l2_withinType_dists.npz`: compatibility input for the panel-c loader.

For all four panels, cells were balanced within cell type and donor across held-out real data, DeconvSC and the ablated VAE, with a maximum of 200 cells per donor. Ground-truth cells were not divided into reference subsets. Every plotted value is one cell type after metric-specific gene, gene-pair, hub or module summaries were calculated within each seed and then averaged across ten fixed seeds. Seeds and lower-level observations are not biological replicates. The current panel-d means are 0.678 for DeconvSC and 0.380 for the ablated VAE, with higher DeconvSC values in all six cell types.

These endpoints quantify undirected expression-topology fidelity, not directed or causal gene-regulatory-network reconstruction. DISSECT, BayesPrism, TAPE and CIBERSORTx are absent because their available sample-by-cell-type aggregate outputs cannot estimate within-cell-type cell-level covariance.
