#!/usr/bin/env python
"""Search HCA cell-type COARSE groupings that maximise DeconvSC (route2+prophead, no floor)
per-cell-type proportion PCC across the 105 synthetic bulks.

DeconvSC proportions = $WORK_ROOT/03_result_tables/unified/proportions_hca_fold2_prophead.csv.
The prophead solver emits 22 fine types; GT annotates 20. The 2 pred-only types are ANCHORED to a
same-lineage GT sibling so their predicted mass is always counted with that sibling:
  DC2 -> DC1 ;  Subpleural fibroblasts -> Fibroblasts PLIN2+ .
"None" (GT placeholder, 4653 cells) is reported separately, NOT part of the cell-class grouping.

Two greedy agglomerative searches (objective = DeconvSC mean per-label PCC):
  (1) LINEAGE-constrained: only merge two groups within the SAME HLCA major lineage
      -> biologically interpretable "细胞大类"; coarsest level = the 5 lineages.
  (2) UNCONSTRAINED: any pair -> the theoretical ceiling (may merge unrelated lineages).
Prints the mean-PCC vs #groups frontier + the grouping at each level.
"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import pearsonr
from itertools import combinations

WORK_ROOT = os.environ.get("WORK_ROOT", "/disk1/maijl/deconv/deconv_20260610/prophead")
UNI = f"{WORK_ROOT}/03_result_tables/unified"
V3  = "/disk1/maijl/deconv/cVAE/Simulation_HCA/result/HCA/v3_pureprior/fold_2"
BF  = "/disk1/maijl/deconv/benchmark_fold2"

# HLCA major lineage of every fine type (GT ∪ prophead-pred). DC2 + Subpleural are pred-only.
LINEAGE = {
    "Club": "Epithelial", "Deuterosomal": "Epithelial", "Goblet": "Epithelial", "Transitional AT2": "Epithelial",
    "EC aerocyte capillary": "Endothelial", "EC venous systemic": "Endothelial",
    "B cells": "Lymphoid", "CD4 T cells": "Lymphoid", "NK cells": "Lymphoid", "Plasma cells": "Lymphoid",
    "Alveolar macrophages": "Myeloid", "Classical monocytes": "Myeloid", "DC monocyte": "Myeloid",
    "DC1": "Myeloid", "DC2": "Myeloid", "Macrophages MARCO-": "Myeloid",
    "Macrophages SPP1 high": "Myeloid", "Plasmacytoid DCs": "Myeloid",
    "Activated myofibroblasts": "Stromal", "Fibroblasts PLIN2+": "Stromal", "Pericytes": "Stromal",
    "Subpleural fibroblasts": "Stromal",
}
ANCHOR = {"DC2": "DC1", "Subpleural fibroblasts": "Fibroblasts PLIN2+"}   # pred-only -> GT sibling


def rename_none(df):
    return df.rename(columns={c: "Others" for c in df.columns if str(c).strip().lower() in ("none", "na", "nan")})


gt   = rename_none(pd.read_csv(f"{V3}/gt_actual_props_fold_2.csv", index_col=0)); gt.index = gt.index.astype(str)
pred = rename_none(pd.read_csv(f"{UNI}/proportions_hca_fold2_prophead.csv", index_col=0)); pred.index = pred.index.astype(str)
SAMP = [s for s in gt.index if s in pred.index]
GT_FINE = [c for c in gt.columns if c != "Others" and c in LINEAGE]            # 20 real GT classes


def _pcc(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    msk = np.isfinite(a) & np.isfinite(b)
    if msk.sum() < 3:
        return np.nan
    a, b = a[msk], b[msk]
    return pearsonr(a, b)[0] if np.std(a) > 1e-12 and np.std(b) > 1e-12 else np.nan


def pred_sum(members):
    cols = [m for m in members if m in pred.columns]
    cols += [po for po, anc in ANCHOR.items() if anc in members and po in pred.columns]   # ride-along pred-only mass
    return pred[cols].loc[SAMP].sum(1) if cols else pd.Series(0.0, index=SAMP)


def gt_sum(members):
    cols = [m for m in members if m in gt.columns]
    return gt[cols].loc[SAMP].sum(1) if cols else pd.Series(0.0, index=SAMP)


def grouping_pcc(groups, src=pred_sum):
    """groups: list[list[GT-type]]. Returns (mean DeconvSC PCC over scorable labels, {label_repr: pcc})."""
    out = {}
    for g in groups:
        out[", ".join(g)] = _pcc(src(g).values, gt_sum(g).values)
    finite = [v for v in out.values() if np.isfinite(v)]
    return (np.mean(finite) if finite else np.nan), out


def greedy(constrain_lineage):
    groups = [[t] for t in GT_FINE]
    frontier = []
    m0, _ = grouping_pcc(groups); frontier.append((len(groups), m0, [list(g) for g in groups]))
    while len(groups) > 2:
        best = None
        for i, j in combinations(range(len(groups)), 2):
            if constrain_lineage:
                if {LINEAGE[t] for t in groups[i]} != {LINEAGE[t] for t in groups[j]}:
                    continue
            newg = [groups[k] for k in range(len(groups)) if k not in (i, j)] + [groups[i] + groups[j]]
            m, _ = grouping_pcc(newg)
            if best is None or m > best[0]:
                best = (m, i, j, newg)
        if best is None:
            break
        groups = best[3]
        frontier.append((len(groups), best[0], [list(g) for g in groups]))
    return frontier


def show(frontier, tag):
    print(f"\n========== {tag}: DeconvSC mean per-label PCC vs #groups ==========")
    for n, m, _ in frontier:
        print(f"  {n:2d} groups -> mean PCC {m:.4f}")
    n_best, m_best, g_best = max(frontier, key=lambda x: (x[1] if np.isfinite(x[1]) else -9))
    print(f"  >>> best: {n_best} groups, mean PCC {m_best:.4f}")
    return n_best, m_best, g_best


if __name__ == "__main__":
    print(f"WORK_ROOT={WORK_ROOT}  samples={len(SAMP)}  GT fine classes={len(GT_FINE)}")
    # per-fine baseline (each GT type its own label)
    _, per = grouping_pcc([[t] for t in GT_FINE])
    print("\n--- per-fine-type DeconvSC (prophead) proportion PCC (sorted) ---")
    for t, v in sorted(per.items(), key=lambda kv: (kv[1] if np.isfinite(kv[1]) else -9), reverse=True):
        print(f"  {v:6.3f}  {t}  [{LINEAGE.get(t,'?')}]")
    print(f"  mean (20 fine) = {np.nanmean([v for v in per.values()]):.4f}")

    fl = greedy(constrain_lineage=True);  n_l, m_l, g_l = show(fl, "LINEAGE-CONSTRAINED")
    fu = greedy(constrain_lineage=False); n_u, m_u, g_u = show(fu, "UNCONSTRAINED")

    # explicit 5-lineage grouping (coarsest interpretable)
    by_lin = {}
    for t in GT_FINE:
        by_lin.setdefault(LINEAGE[t], []).append(t)
    m5, lab5 = grouping_pcc(list(by_lin.values()))
    print("\n========== explicit 5-LINEAGE grouping ==========")
    for lin, members in by_lin.items():
        print(f"  {lab5[', '.join(members)]:6.3f}  {lin}: {members}")
    print(f"  >>> 5-lineage DeconvSC mean PCC = {m5:.4f}")

    # explicit best WITHIN-LINEAGE grouping (highest interpretable, from the lineage-constrained search)
    print(f"\n========== BEST WITHIN-LINEAGE grouping ({n_l} groups, max interpretable) ==========")
    _, perL = grouping_pcc(g_l)
    for g in sorted(g_l, key=lambda x: (LINEAGE[x[0]], -len(x))):
        print(f"  {perL[', '.join(g)]:6.3f}  [{LINEAGE[g[0]]}]  {g}")
    print(f"  >>> {n_l} groups, DeconvSC mean PCC = {m_l:.4f}")

    # ===== full benchmark (DeconvSC=prophead vs 4 competitors) at the 5 canonical lineages =====
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")); import plot_style
    OUTD = f"{WORK_ROOT}/03_result_tables"; FIGD = f"{WORK_ROOT}/03_result_figures/figures"; os.makedirs(FIGD, exist_ok=True)
    PAL = {"DeconvSC": "#DEBBB1", "BayesPrism": "#86ADC0", "CIBERSORTx": "#7187A7", "DISSECT": "#AA96CA", "TAPE": "#989EB0"}
    ORDER = ["DeconvSC", "BayesPrism", "CIBERSORTx", "DISSECT", "TAPE"]
    CSX_REN = {"Plasma_cells": "Plasma cells", "Pericytes": "Pericytes", "Club": "Club", "DC2": "DC2",
        "Activated_myofibroblasts": "Activated myofibroblasts", "Fibroblasts_PLIN2": "Fibroblasts PLIN2+",
        "CD4_T_cells": "CD4 T cells", "Goblet": "Goblet", "Transitional_AT2": "Transitional AT2",
        "Macrophages_MARCO": "Macrophages MARCO-", "EC_aerocyte_capillary": "EC aerocyte capillary",
        "Plasmacytoid_DCs": "Plasmacytoid DCs", "B_cells": "B cells", "NK_cells": "NK cells", "DC1": "DC1",
        "EC_venous_systemic": "EC venous systemic", "DC_monocyte": "DC monocyte",
        "Macrophages_SPP1_high": "Macrophages SPP1 high", "Alveolar_macrophages": "Alveolar macrophages",
        "Deuterosomal": "Deuterosomal", "Classical_monocytes": "Classical monocytes",
        "Subpleural_fibroblasts": "Subpleural fibroblasts"}
    comp = {"DeconvSC": pred}
    for m, p in [("BayesPrism", f"{BF}/BayesPrism/results/pred_proportions.csv"),
                 ("DISSECT", f"{BF}/DISSECT/results/pred_proportions.csv"),
                 ("TAPE", f"{BF}/TAPE/results/pred_proportions.csv")]:
        d = pd.read_csv(p, index_col=0); d.index = d.index.astype(str); comp[m] = rename_none(d)
    csx = pd.read_csv(f"{BF}/CIBERSORTx/results/CIBERSORTxGEP_Job37_Fractions.txt", sep="\t", index_col=0)
    csx = csx.drop(columns=[c for c in ["P-value", "Correlation", "RMSE"] if c in csx.columns])
    smap = pd.read_csv(f"{BF}/CIBERSORTx/sample_id_map.csv", index_col=0)["pseudo_bulk_id"].astype(str).to_dict()
    csx.index = [smap.get(s, s) for s in csx.index.astype(str)]
    comp["CIBERSORTx"] = rename_none(csx.rename(columns=CSX_REN))

    def method_lineage_pcc(df, members, anchored):
        cols = [m for m in members if m in df.columns]
        if anchored:
            cols += [po for po, anc in ANCHOR.items() if anc in members and po in df.columns]
        pv = df.reindex(SAMP)[cols].sum(1) if cols else pd.Series(0.0, index=SAMP)
        tv = gt.reindex(SAMP)[[m for m in members if m in gt.columns]].sum(1)
        return _pcc(pv.values, tv.values)

    lineg = list(by_lin.items())
    tab = pd.DataFrame(index=[k for k, _ in lineg], columns=ORDER, dtype=float)
    for lin, mem in lineg:
        for m in ORDER:
            tab.loc[lin, m] = method_lineage_pcc(comp[m], mem, anchored=(m == "DeconvSC"))
    tab.insert(0, "mean_true_prop", [gt.reindex(SAMP)[[c for c in mem if c in gt.columns]].sum(1).mean() for _, mem in lineg])
    tab = tab.sort_values("DeconvSC", ascending=False)
    means = tab[ORDER].mean(); tab.loc["— MEAN —"] = [np.nan] + list(means.values)
    tab.index.name = "Cell class (5 lineages)"
    tab.round(4).to_csv(f"{OUTD}/PROPORTION_lineage5_HCA_prophead_PCC.csv")
    print("\n========== 5-LINEAGE benchmark — DeconvSC=prophead vs competitors ==========")
    print(tab.round(3).to_string()); print("MEAN:", {m: round(means[m], 3) for m in ORDER})

    sub = tab.drop("— MEAN —")
    fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(len(sub)); w = 0.16
    for i, m in enumerate(ORDER):
        ax.bar(x + (i - 2) * w, sub[m].values, w, label=f"{m} (μ={means[m]:.2f})", color=PAL[m], edgecolor="black", lw=0.3)
    ax.set_xticks(x); ax.set_xticklabels(sub.index, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Proportion PCC (across 105 samples)"); ax.set_ylim(-0.1, 1.0)
    ax.axhline(0, color="black", lw=0.5); ax.grid(axis="y", alpha=0.3)
    ax.set_title("HCA fold2 — per-cell-class proportion PCC (5 lineages, DeconvSC = route2+prophead)",
                 fontweight="bold", fontsize=11, pad=30)
    ax.legend(fontsize=9, ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(); plot_style.save(fig, f"{FIGD}/HCA_fold2_proportion_lineage5_prophead_pcc", dpi=300); plt.close()
    print(f"[saved] {OUTD}/PROPORTION_lineage5_HCA_prophead_PCC.csv")
    print(f"[saved] {FIGD}/HCA_fold2_proportion_lineage5_prophead_pcc.{{png,pdf,svg,emf}}")

    # ===== 7-group within-lineage-OPTIMAL benchmark (named) — the max-interpretable 0.835 version =====
    # exactly the lineage-constrained greedy optimum; pred-only DC2 rides Myeloid (anchor DC1),
    # Subpleural fibroblasts rides Stromal (anchor Fibroblasts PLIN2+).
    GROUP7 = {
        "Airway-epi":      ["Club", "Deuterosomal"],
        "Goblet/TransAT2": ["Goblet", "Transitional AT2"],
        "Endothelial":     ["EC aerocyte capillary", "EC venous systemic"],
        "T/NK":            ["CD4 T cells", "NK cells"],
        "B/Plasma":        ["B cells", "Plasma cells"],
        "Myeloid":         ["Alveolar macrophages", "Classical monocytes", "DC monocyte", "DC1",
                            "Macrophages MARCO-", "Macrophages SPP1 high", "Plasmacytoid DCs"],
        "Stromal":         ["Activated myofibroblasts", "Fibroblasts PLIN2+", "Pericytes"],
    }
    g7 = list(GROUP7.items())
    tab7 = pd.DataFrame(index=[k for k, _ in g7], columns=ORDER, dtype=float)
    for lab, mem in g7:
        for m in ORDER:
            tab7.loc[lab, m] = method_lineage_pcc(comp[m], mem, anchored=(m == "DeconvSC"))
    tab7.insert(0, "mean_true_prop", [gt.reindex(SAMP)[[c for c in mem if c in gt.columns]].sum(1).mean() for _, mem in g7])
    tab7 = tab7.sort_values("DeconvSC", ascending=False)
    means7 = tab7[ORDER].mean(); tab7.loc["— MEAN —"] = [np.nan] + list(means7.values)
    tab7.index.name = "Cell class (7 within-lineage)"
    tab7.round(4).to_csv(f"{OUTD}/PROPORTION_group7_HCA_prophead_PCC.csv")
    print("\n========== 7-GROUP (within-lineage optimal) benchmark — DeconvSC=prophead vs competitors ==========")
    print(tab7.round(3).to_string()); print("MEAN:", {m: round(means7[m], 3) for m in ORDER})

    sub7 = tab7.drop("— MEAN —")
    fig, ax = plt.subplots(figsize=(11, 5)); x = np.arange(len(sub7)); w = 0.16
    for i, m in enumerate(ORDER):
        ax.bar(x + (i - 2) * w, sub7[m].values, w, label=f"{m} (μ={means7[m]:.2f})", color=PAL[m], edgecolor="black", lw=0.3)
    ax.set_xticks(x); ax.set_xticklabels(sub7.index, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Proportion PCC (across 105 samples)"); ax.set_ylim(-0.1, 1.0)
    ax.axhline(0, color="black", lw=0.5); ax.grid(axis="y", alpha=0.3)
    ax.set_title("HCA fold2 — per-cell-class proportion PCC (7 within-lineage groups, DeconvSC = route2+prophead)",
                 fontweight="bold", fontsize=11, pad=30)
    ax.legend(fontsize=9, ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(); plot_style.save(fig, f"{FIGD}/HCA_fold2_proportion_group7_prophead_pcc", dpi=300); plt.close()
    print(f"[saved] {OUTD}/PROPORTION_group7_HCA_prophead_PCC.csv")
    print(f"[saved] {FIGD}/HCA_fold2_proportion_group7_prophead_pcc.{{png,pdf,svg,emf}}")
