#!/usr/bin/env python
"""Composition-invariant Module-L2 (within-cell-type, equal-weight).

The global Module-L2 (Fig4 panel c) is computed on the gene-gene correlation matrix ACROSS all
cells, which is dominated by cell-type co-occurrence -> it depends on the cell-type composition
(hence on the proportion scheme). This panel instead computes the gene-gene correlation WITHIN each
sufficiently-large cell type, averages it UNWEIGHTED across types (so it is proportion/composition-
invariant), and then measures the per-module Frobenius distance to real. It therefore isolates the
INTRINSIC within-cell-state co-regulation that the attention module is meant to capture.

  - Rare cell types excluded: only types with >= WT_MIN_CELLS cells in real AND in every method.
  - Per seed, each kept type is subsampled to the shared min N (<= CAP) so the correlation estimate
    is matched across methods; modules are the top-10 cohesive Ward modules of the real within-type
    correlation matrix; Module-L2 = ||R_real - R_gen||_F / |module|^2 (same as Fig4 panel c).
  - DeconvSC (Route2) vs BaseVAE (attention-ablated). DISSECT is excluded (126 aggregated profiles,
    no within-type single cells). Significance = one-sided paired Wilcoxon (DeconvSC < BaseVAE).
  - Run per scheme:  OURS_H5AD=<scheme generated_data.h5ad>  ABLATION_DIR=<scheme>/.../ablation_compare
"""
import os, sys, numpy as np, pandas as pd, anndata as ad, scanpy as sc, scipy.sparse as sp
from scipy.stats import wilcoxon
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import seaborn as sns
import warnings; warnings.filterwarnings("ignore"); sc.settings.verbosity = 0
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import plot_style  # noqa

RELEASE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REAL = os.path.join(RELEASE_ROOT, "processed_data/real_data/mouse_kidney/test_data.h5ad")
OURS = os.environ.get("OURS_H5AD", os.path.join(RELEASE_ROOT, "model_outputs/GSE141115/figure4_suppfigure5_deconvsc_generated.h5ad"))
BASEVAE = os.path.join(RELEASE_ROOT, "model_outputs/GSE141115/figure4_ablated_vae_generated.h5ad")
AC = os.environ.get("ABLATION_DIR", os.path.join(RELEASE_ROOT, "work/legacy_within_type_module_l2"))
MIN_CELLS = int(os.environ.get("WT_MIN_CELLS", "100"))
TOPK, N_MOD, TOP_MOD, NSEED, CAP = 500, 20, 10, 5, 800
PAL = {"Route2": "#D8B2AE", "BaseVAE": "#7187A7"}
DISP = {"Route2": "DeconvSC", "BaseVAE": "Ablated VAE"}
ORDER = ["Route2", "BaseVAE"]


def ct_col(a):
    for c in ("Cell_type", "cell type", "labels", "celltype"):
        if c in a.obs.columns: return c
    raise KeyError(f"no cell-type column in {list(a.obs.columns)}")


def to_log(a):
    X = a.X
    mx = float(X.max() if not sp.issparse(X) else X.max())
    if mx > 30: sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)   # raw -> log1p CP10K
    return a


def dense(a, obs_idx, genes):
    sub = a[obs_idx, genes].X
    return np.asarray(sub.todense() if sp.issparse(sub) else sub, dtype=np.float64)


def wt_corr(mats):                       # equal-weight average of per-type correlation matrices
    return np.mean(np.stack([np.nan_to_num(np.corrcoef(m, rowvar=False)) for m in mats]), axis=0)


def main():
    real = to_log(ad.read_h5ad(REAL)); real.var_names_make_unique(); rcc = ct_col(real)
    ours = to_log(ad.read_h5ad(OURS)); ours.var_names_make_unique(); occ = ct_col(ours)
    base = to_log(ad.read_h5ad(BASEVAE)); base.var_names_make_unique(); bcc = ct_col(base)

    common = sorted(set(real.var_names) & set(ours.var_names) & set(base.var_names))
    rh = real[:, common].copy(); sc.pp.highly_variable_genes(rh, n_top_genes=TOPK)
    hvg = rh.var_names[rh.var.highly_variable].tolist()

    rc = real.obs[rcc].astype(str); oc = ours.obs[occ].astype(str); bc = base.obs[bcc].astype(str)
    rvc, ovc, bvc = rc.value_counts(), oc.value_counts(), bc.value_counts()
    kept = [t for t in rvc.index
            if rvc[t] >= MIN_CELLS and ovc.get(t, 0) >= MIN_CELLS and bvc.get(t, 0) >= MIN_CELLS]
    print(f"kept {len(kept)} cell types (>= {MIN_CELLS} cells in real+DeconvSC+BaseVAE): {kept}")
    if len(kept) < 3:
        print("too few cell types pass the threshold; lower WT_MIN_CELLS"); return
    ridx = {t: np.where(rc.values == t)[0] for t in kept}
    oidx = {t: np.where(oc.values == t)[0] for t in kept}
    bidx = {t: np.where(bc.values == t)[0] for t in kept}

    # modules: top-10 cohesive Ward modules of the real within-type (equal-weight) correlation
    R_real_full = wt_corr([dense(real, ridx[t], hvg) for t in kept])
    d = 1.0 - np.abs(R_real_full); np.fill_diagonal(d, 0.0)
    lab = fcluster(linkage(squareform(d, checks=False), method="ward"), N_MOD, criterion="maxclust")
    mods = []
    for m in range(1, N_MOD + 1):
        idx = np.where(lab == m)[0]
        if len(idx) >= 6:
            coh = np.abs(R_real_full[np.ix_(idx, idx)])[np.triu_indices(len(idx), 1)].mean()
            mods.append((coh, idx))
    mods = [idx for _, idx in sorted(mods, key=lambda x: -x[0])[:TOP_MOD]]
    print(f"{len(mods)} cohesive modules (top-{TOP_MOD} of {N_MOD}, >=6 genes)")

    rows = {m: [] for m in ORDER}
    for s in range(NSEED):
        rng = np.random.RandomState(2024 + s)
        rX, oX, bX = [], [], []
        for t in kept:
            n = min(len(ridx[t]), len(oidx[t]), len(bidx[t]), CAP)
            rX.append(dense(real, np.sort(rng.choice(ridx[t], n, replace=False)), hvg))
            oX.append(dense(ours, np.sort(rng.choice(oidx[t], n, replace=False)), hvg))
            bX.append(dense(base, np.sort(rng.choice(bidx[t], n, replace=False)), hvg))
        Rr, Ro, Rb = wt_corr(rX), wt_corr(oX), wt_corr(bX)
        for idx in mods:
            sz = len(idx)
            rows["Route2"].append(np.linalg.norm(Rr[np.ix_(idx, idx)] - Ro[np.ix_(idx, idx)]) / sz**2)
            rows["BaseVAE"].append(np.linalg.norm(Rr[np.ix_(idx, idx)] - Rb[np.ix_(idx, idx)]) / sz**2)

    vals = {m: np.array(rows[m]) for m in ORDER}
    p = wilcoxon(vals["Route2"], vals["BaseVAE"], alternative="less")[1]   # DeconvSC < BaseVAE
    star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"

    # ---- summary CSV ----
    summ = pd.DataFrame([{"method": DISP[m].replace("\n", " "), "median": float(np.median(vals[m])),
                          "mean": float(vals[m].mean()), "n_module_x_seed": len(vals[m])} for m in ORDER])
    summ["kept_types"] = len(kept); summ["min_cells"] = MIN_CELLS
    summ["p_DeconvSC_lt_BaseVAE_pairedWilcoxon"] = f"{p:.3e}"
    summ.to_csv(os.path.join(AC, "module_l2_withinType_summary.csv"), index=False)
    # also save the raw per-(module x seed) distributions so Fig4 panel c can plot the within-type violin
    np.savez(os.path.join(AC, "module_l2_withinType_dists.npz"),
             Route2=vals["Route2"], BaseVAE=vals["BaseVAE"], p=np.float64(p), kept=np.int64(len(kept)))

    # fold the within-type Module L2 into the attention-advantage summary (per method; DISSECT -> NaN,
    # it has no within-type single cells). module_l2 there is the global (mean); we add median + mean.
    asum_csv = os.path.join(AC, "attention_advantage_summary.csv")
    if os.path.exists(asum_csv):
        asum = pd.read_csv(asum_csv)
        asum["module_l2_withinType_median"] = asum["method"].map({m: float(np.median(vals[m])) for m in ORDER})
        asum["module_l2_withinType_mean"]   = asum["method"].map({m: float(vals[m].mean())     for m in ORDER})
        asum.to_csv(asum_csv, index=False)
        print(f"[updated] {asum_csv} += module_l2_withinType_{{median,mean}}")

    # ---- panel (seaborn violin + box; theme matches Fig4 panel_c_module_l2: y-axis "Module L2 error", no title) ----
    long = pd.DataFrame([{"Model": DISP[m], "v": float(x)} for m in ORDER for x in vals[m]])
    cats = [DISP[m] for m in ORDER]; pal = {DISP[m]: PAL[m] for m in ORDER}
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.violinplot(data=long, x="Model", y="v", order=cats, hue="Model", palette=pal,
                   legend=False, inner=None, alpha=0.35, cut=0, ax=ax)
    sns.boxplot(data=long, x="Model", y="v", order=cats, hue="Model", palette=pal,
                legend=False, width=0.35, fliersize=0, linewidth=1.0, ax=ax)
    ax.set_ylabel("Module L2 error", fontsize=16, fontweight="bold"); ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), fontsize=14, fontweight="bold")
    pstr = f"P = {p:.0e}" if p < 0.01 else f"P = {p:.3f}"
    y0 = long["v"].max(); rng = max(long["v"].max() - long["v"].min(), 1e-9); h = rng * 0.04; bb = y0 + rng * 0.05
    ax.plot([0, 0, 1, 1], [bb, bb + h, bb + h, bb], lw=1.0, c="black")
    ax.text(0.5, bb + h, f"{star}\n{pstr}", ha="center", va="bottom", fontsize=12, linespacing=1.7, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False); ax.set_ylim(top=bb + rng * 0.34)
    plot_style.save(fig, os.path.join(AC, "module_l2_withinType"))
    plt.close(fig)
    print(f"DeconvSC median {np.median(vals['Route2']):.4f} vs BaseVAE {np.median(vals['BaseVAE']):.4f} "
          f"| p(DeconvSC<BaseVAE)={p:.2e} ({star})")
    print(f"[saved] {AC}/module_l2_withinType.{{png,pdf,svg,emf}} + module_l2_withinType_summary.csv")


if __name__ == "__main__":
    main()
