# Ablation scope — which methods can be scored on the GRN-topology metrics?

_Recorded 2026-06-16. Applies to the attention-advantage / GRN-topology ablation (Fig 4)._

## Question
Besides DISSECT and the attention-ablated VAE (BaseVAE), can **BayesPrism / CIBERSORTx / TAPE**
be scored on the topology metrics — **hub-gene Jaccard, |ΔMI| (MI fidelity), module L2, gene-network
fidelity**, and the **within-type module L2**?

## All five metrics are functions of a gene×gene co-expression matrix
- `04_evaluation/05_calculate_attention_ablation.py`: every metric flows from
  `corr_of(X) = np.corrcoef(X, rowvar=False)` on an **(observations × genes)** matrix
  (hub genes = degree of that network; modules = Ward clusters of it; |ΔMI| = binned MI of gene
  pairs; network fidelity = correlation-of-correlation rows).
- `04_evaluation/within_type_module_l2.py`: `wt_corr = mean over per-type np.corrcoef`, i.e. the
  gene×gene correlation estimated **inside each cell type**, equal-weight averaged across types
  (composition-invariant). Needs `>= WT_MIN_CELLS=100` cells of that type, matched-subsampled to `CAP=800`.

So eligibility is decided entirely by **what "observations" each method emits**.

## Three tiers of expression output (HCA fold2, verified shapes)
| tier | methods | output | obs / cell type | global topology (hubJ/|ΔMI|/moduleL2/netfid) | within-type module L2 |
|---|---|---|---|---|---|
| **1 — per-cell single cells** | DeconvSC `generated_data.h5ad` (15,739 cells, ~684/type); BaseVAE; DeltaZ `pred_validation_cells.h5ad` | cells × genes | hundreds | ✅ | ✅ |
| **2 — aggregated per-(sample,celltype) means** | DISSECT `est_expression.h5ad` (184 = 8×23, **8/type**); BayesPrism `pred_celltype_expr.h5ad` (2,415 = 105×23, 105/type); TAPE same; CIBERSORTx HiRes (genes×105/type, **44% NaN**) | (sample×celltype) × genes | a few – ~100 deconvolved means | ⚠ technically (pooled, between-type structure) | ❌ no single cells within a type |
| **3 — proportions only** | any method run in proportion mode | samples × celltypes | — | ❌ | ❌ |

## Conclusion
1. **Global topology metrics** need a genes×observations matrix whose observations span the
   population; a pooled gene×gene network is dominated by **between-cell-type** structure. Both
   single cells (Tier 1) and aggregated per-(sample,celltype) profiles (Tier 2) technically yield
   one. **Granularity does NOT gate these global metrics** — DISSECT's `est_expression` is the *same
   kind* of output as BayesPrism/TAPE (aggregated per-(sample,celltype); in fact coarser, 8 vs 105
   profiles/type), so in principle DISSECT, BayesPrism and TAPE could all be scored on hub-Jaccard /
   |ΔMI| / global module L2 / net fidelity (CIBERSORTx is the lone hard exception: HiRes is **44%
   NaN**, so the fixed top-K HVG network cannot be assembled). **DISSECT is the one shown not because
   the others *can't*, but by design**: it is the architecture-matched comparator — the cVAE-style
   generative deconvolution the attention mechanism is contrasted against (Results: *"attention
   captures non-linear dependencies better than a pure conditional VAE, e.g. DISSECT"*; Fig 4a–d).
   BayesPrism / CIBERSORTx / TAPE are positioned as the **expression-accuracy / proportion**
   baselines (Fig 2–3), not the GRN-topology comparator. (A residual caveat applies equally to
   DISSECT's own global numbers: the gene-gene covariance of per-sample deconvolved **means**
   reflects across-sample variation, not within-population single-cell co-expression.)
2. **Within-type module L2** (the composition-invariant, *intrinsic* co-regulation metric) is
   strictly tighter: it estimates a gene×gene correlation **inside each cell type** (≥100 cells/type),
   so it needs **many single cells per type** — only Tier-1 per-cell generators have that. It
   therefore excludes BayesPrism / CIBERSORTx / TAPE **and DISSECT**, and is reported only for
   **DeconvSC vs the attention-ablated VAE**.

## Why DISSECT (cell-level reconstruction) still cannot do within-type module L2
DISSECT's `est_expression` is **not single cells** — it is **1 aggregated expression profile per
(sample, cell type)** (HCA: 184 = 8 samples × 23 types = **8 profiles/type**; GSE141115: 126
profiles). The two metrics ask for different things:
- **Global** network pools all 184/126 profiles **across** types → between-type structure is
  estimable → DISSECT participates (`05_calculate_attention_ablation.py` treats it as
  `n_obs < 500 → native N, computed once`).
- **Within-type** network removes the between-type signal and needs the gene×gene correlation among
  **many cells of one type**; with only 8 aggregated **means** per type, the within-type correlation
  over 500 HVGs is rank-deficient / undefined → DISSECT is excluded (**→ NaN**;
  `within_type_module_l2.py` docstring: _"DISSECT is excluded (126 aggregated profiles, no
  within-type single cells)"_). DeconvSC has ~684 single cells/type, so its within-type network is
  well-estimated.

## Measured (2026-06-16): BayesPrism/TAPE *do* score on the global metrics — and BayesPrism WINS
Computed hub-Jaccard / module-L2 / |ΔMI| / netPCC / net-fidelity for BayesPrism + TAPE on HCA fold2,
same functions/params as `05_calculate_attention_ablation.py` (DeltaZ & DISSECT reproduced the
canonical run almost exactly → pipeline validated; `HCA/ablation_compare/topology_competitors_supplement.csv`):

| method | hubJ ↑ | modL2 ↓ | \|ΔMI\| ↓ | netPCC ↑ | netfid ↑ |
|---|---|---|---|---|---|
| Route2 (DeconvSC) | 0.369 | 0.013 | 0.055 | 0.790 | 0.807 |
| **BayesPrism** | **0.477** | 0.013 | 0.067 | 0.785 | **0.817** |
| TAPE | 0.220 | 0.027 | 1.117 | 0.391 | 0.423 |
| DISSECT | 0.213 | 0.029 | 0.960 | 0.456 | 0.480 |
| DeltaZ | 0.287 | 0.020 | 0.455 | 0.563 | 0.601 |

**BayesPrism is competitive with DeconvSC on the global metrics** — clearly higher only on **hub-gene
Jaccard** (0.48 vs 0.37); tied/within ~0.01 on module-L2, |ΔMI|, netPCC and net-fidelity (|ΔMI| 0.067
vs 0.055 and netPCC 0.785 vs 0.790 actually favour DeconvSC marginally). The point is **not** that
BayesPrism wins — it's that a **means-only method (zero single cells)** *ties* a single-cell generator
on a "co-expression network" metric. That tie is the tell: the GLOBAL gene×gene network (pooled across
cell types) is **largely shaped by between-cell-type marker structure** — replacing real cells by their
cell-type means (destroying all within-type info) still reproduces **~80% of the real network's
correlation pattern (off-diagonal corr 0.81)** — which deconvolved cell-type means capture by
construction. (It's a *major* component, not total: between-type is only ~19% of per-gene variance, but
it is the *coherent* part that drives gene-gene correlations; within-type variance is larger but mostly
incoherent.) TAPE scores poorly (its per-(sample,celltype) expression inflates MI like DISSECT). ⇒
**Putting deconvolution methods on the Fig-4 global panels would be misleading** (a means-only method
ties/edges DeconvSC there). The global metric does not isolate *within*-cell-type co-expression (the
part the attention module adds); the **within-type module L2** — which removes the between-type signal
and which BayesPrism/TAPE/DISSECT *cannot* compute — is the genuinely discriminating metric.

## Recommendation for the manuscript
- Keep the topology / within-type ablation scoped to single-cell generators (**DeconvSC vs ablated
  VAE**; DISSECT only on the global panels).
- Score **BayesPrism / CIBERSORTx / TAPE** on the **proportion** benchmark (PCC/RMSE). If their
  expression side is to be credited, add a **cell-type-mean expression-recovery PCC** (gene-wise /
  per-profile, the metric that matches their per-(sample,celltype) output) — buildable from each
  method's `pred_celltype_expr.h5ad`.

_Sources: `04_evaluation/05_calculate_attention_ablation.py`, `04_evaluation/within_type_module_l2.py`;
shapes verified on `benchmark_fold2/{BayesPrism,TAPE,DISSECT,CIBERSORTx}/results/` and
`dissect/HCA/.../est_expression.h5ad`._
