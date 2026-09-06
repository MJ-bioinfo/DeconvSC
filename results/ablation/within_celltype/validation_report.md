# GSE141115 within-cell-type topology 验证报告

## Material Passport

- Origin Skill: `academic-research-suite / experiment-agent`
- Origin Mode: validate
- Origin Date: 2026-07-13
- Verification Status: `VERIFIED`
- Version Label: `within_celltype_topology_validation_v2_edge_mae`

## Validation Report

- **Source**: `GSE141115_within_celltype_topology_20260710`
- **Overall Confidence**: `CAUTION`
- **Reason**: primary structural result is exact-reproducible and robust, but only six cell-type units are available and the BaseVAE training configuration is not fully auditable; attention-only causal attribution is unsupported.

### Statistical findings

| Metric | Test | Result | Effect | Confidence |
|---|---|---|---|---|
| Module L2 | exact one-sided paired Wilcoxon across cell types | p=0.015625; two-sided p=0.03125 | 6/6 wins; rank-biserial=1.0; mean 65.81% lower | SOLID for this benchmark |
| Module RMSE | exact one-sided paired Wilcoxon | p=0.015625 | 6/6 wins; mean 64.20% lower | SOLID sensitivity |
| Module edge MAE | exact one-sided paired Wilcoxon | p=0.015625 | 6/6 wins; mean 67.53% lower | SOLID sensitivity |
| Hub Jaccard | exact one-sided paired Wilcoxon + Holm family | raw p=0.046875; Holm p=0.0625 | 5/6 wins; bootstrap CI crosses zero | CAUTION |
| MI fidelity error | exact one-sided paired Wilcoxon + Holm family | raw p=0.015625; Holm p=0.0625 | 6/6 wins | CAUTION after correction |
| Per-gene co-expression edge MAE | exact one-sided paired Wilcoxon + Holm family | raw p=0.015625; Holm p=0.0625 | 6/6 wins; mean 68.49% lower | CAUTION after correction |

Whole-network PCC and the original per-gene connectivity-profile PCC remain available as supplementary sensitivity metrics but are no longer displayed in Figure 4.

### Statistical assumptions and warnings

- The paired unit is cell type, not module, seed, gene, edge or cell.
- Exact Wilcoxon is appropriate for small paired N without a normality assumption, but N=6 gives coarse p-value resolution.
- Cell types share the same three donors; p-values quantify consistency across evaluated cell types, not population-level donor inference.
- Bootstrap CI across six cell types is descriptive and has limited precision.
- The four final Figure 4 endpoints use Holm correction; none remains below 0.05.
- The primary directional alternative was specified before final analysis; a two-sided p is also reported.
- Protocol scale amendment occurred after an initial result was observed and is transparently recorded. P0 and native-scale S7 agree.
- Figure 4d was changed after the original per-gene PCC result had been inspected. The revision is recorded as post-result, the original PCC outputs are retained, and the MAE endpoint is interpreted as a complementary error measure rather than a transformed PCC.

### Fallacy scan

- **Coverage**: 11/11 statistical fallacy types checked.

| Fallacy | Severity | Finding | Control/interpretation |
|---|---|---|---|
| Simpson's paradox | NOTE | Global and within-type topology can differ because cell-type composition changes correlations | Analyses are separated; no pooled result is used as within-type evidence |
| Ecological fallacy | CAUTION | Cell-type-level statistics cannot establish individual-cell or donor-population effects | Claim restricted to consistency across benchmark cell types |
| Berkson's paradox | CAUTION | Only sufficiently represented cell types enter the benchmark | Eligibility fixed before final metrics; rare types are not generalized to |
| Collider bias | NOTE | No outcome-induced covariate adjustment detected | Donor centering removes mean shifts without selecting on model performance |
| Base-rate neglect | NOTE | Not a diagnostic classification analysis | Not applicable |
| Regression to the mean | NOTE | No extreme-group pre/post design | Not applicable |
| Survivorship bias | CAUTION | Thresholding excludes rare cell types | Threshold 50/100/200 results are all reported |
| Look-elsewhere effect | CAUTION | Multiple topology endpoints exist and panel d was revised after viewing the PCC result | Holm correction covers the four final panels; original PCC metrics remain disclosed as supplementary outputs |
| Garden of forking paths | CAUTION | Many defensible topology choices, one post-audit scale amendment and one post-result display-endpoint revision exist | Full S1–S7/LODO matrix and both protocol deviations are disclosed |
| Correlation ≠ causation | CAUTION | Correlation/MI topology does not prove regulation or attention causality | Results are named expression topology; GRN/causal claims prohibited |
| Reverse causality | NOTE | No directional regulatory edge inference is made | Not applicable to symmetric expression topology |

### Reproducibility

- **Method**: deterministic rerun in `cache/repro_run`.
- **Verdict**: `REPRODUCIBLE`.
- Main analysis: 21/21 structured CSV/NPZ artifacts exact, maximum numeric difference 0.
- Legacy bridge: 5/5 fields/tables exact.
- Figure 3 provenance: 48 profiles × 16,801 genes, maximum absolute difference `3.41×10⁻⁶`.
- Raw-input integrity is checked by the final `scripts/validate_outputs.py` run and recorded in `validation_checks.csv`.

### Attribution verdict

The evidence supports a DeconvSC-vs-BaseVAE model comparison. It does not verify that attention is the only changed factor, because the exact BaseVAE training code/configuration is unavailable and generation dimensions differ. Any attention-specific claim remains `CANNOT_VERIFY` until a controlled orthogonal ablation is run.
