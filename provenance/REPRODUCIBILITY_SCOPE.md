# Reproducibility scope

## Reproduced from locked artifacts

- Expression-prediction benchmark figures can be regenerated from the exact
  `deconv_20260610/prophead/03_result_tables` snapshot.
- Current Figure 4 can be regenerated from the released cell-level generated
  matrices and tidy within-cell-type result tables.
- Corrected Supplementary Figure 5 can be recomputed from the released real and
  generated GSE141115 matrices.
- Downstream Figure 5/6 result objects and source tables are included.

## Training limitation

The original CVAE checkpoint and cell-type prior tensors are provided. In the
historical workflow, however, the bulk-conditioned residual network and
proportion head were trained online and their exact historical standalone
checkpoints, optimizer states and complete solver metadata were not retained.
Consequently, this package supports current-workflow reruns and exact figure
reproduction from locked artifacts, but it does not claim bitwise reproduction
of every historical training/generation run from raw inputs.

## GSE141115 analysis endpoints

The main prophead benchmark endpoint and the later within-cell-type/
Supplementary Figure 5 analyses used different locked generated matrices. The
former is `prophead_generated.h5ad`; the latter is
`figure4_suppfigure5_deconvsc_generated.h5ad`. Figure 4 also requires the locked
attention-ablated VAE matrix. These files are stored under descriptive names and
must not be interchanged. Their hashes and original roles are explicit in the
artifact manifests.

## Cross-method benchmark boundary

The release preserves the final method-specific Pearson tables for BayesPrism,
TAPE, CIBERSORTx and DISSECT. It does not claim to redistribute all third-party
software, web-service runtimes or every private intermediate matrix. The frozen
tables are therefore the exact cross-method figure-reproduction boundary.

## Inferential scope

Figure 4 reports undirected expression dependency/co-expression fidelity. It
does not establish directed or causal gene-regulatory networks. Cell type is the
summary unit for the displayed Figure 4 distributions; cells, genes, gene pairs,
modules and random seeds are not independent biological replicates.
