#!/usr/bin/env python
"""HCA fold2 — cell-type PROPORTION inference PCC at the S4 merged (14-label) taxonomy.

DeconvSC's proportion solver = NNLS+core on the anchored basis (the best internal solver on HCA's
well-conditioned basis). Route2 confuses some rare/similar subtypes but gets them right in
aggregate, so merging them within-lineage (21 fine -> 14 "S4" labels) raises DeconvSC's per-cell-
type mean proportion PCC from ~0.54 to ~0.68 — i.e. the S4 resolution maximises DeconvSC.
Compared against BayesPrism / CIBERSORTx / DISSECT / TAPE on the same 105 synthetic bulks.

Outputs (refresh of the older PROPORTION_merged14_*):
  $WORK_ROOT/03_result_tables/PROPORTION_merged14_by_celltype_PCC_HCA_fold2.csv   (per-label table)
  $WORK_ROOT/03_result_figures/figures/HCA_fold2_proportion_S4_pcc.{png,pdf,svg,emf}  (grouped bar)
Style = the shared pastel palette + Arial + EMF. "None" -> "Others" (CIBERSORTx has no Others -> gap).
"""
import os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import plot_style   # Arial (applied on import) + plot_style.save (png/pdf/svg + emf)

WORK_ROOT = os.environ.get("WORK_ROOT", "/disk1/maijl/deconv/deconv_20260605")
OUTD = f"{WORK_ROOT}/03_result_tables"
FIGD = f"{WORK_ROOT}/03_result_figures/figures"; os.makedirs(FIGD, exist_ok=True)
UNI  = f"{OUTD}/unified"
V3   = "/disk1/maijl/deconv/cVAE/Simulation_HCA/result/HCA/v3_pureprior/fold_2"
BF   = "/disk1/maijl/deconv/benchmark_fold2"

# pastel palette (identical to the expression figures); DeconvSC == Route2 NNLS+core
PAL = {"DeconvSC": "#DEBBB1", "BayesPrism": "#86ADC0", "CIBERSORTx": "#7187A7",
       "DISSECT": "#AA96CA", "TAPE": "#989EB0"}
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
# S4 coarse lineage map (user-specified 2026-06-11): fine HLCA type -> coarse label.
# Unmapped GT types stay as their own label. NOTE: "DC2" is NOT a GT column on this fold
# (HLCA fold2 has DC1 / DC monocyte / Plasmacytoid DCs) -> Dendritic = {DC1, DC monocyte};
# Plasmacytoid DCs is left ungrouped (its own label). Commented-out entries below are intentionally
# excluded (Transitional AT2, CD4 T cells, NK cells stay as singletons).
COARSE_MAP = {
    "Deuterosomal": "Ciliated",
    "Goblet": "Secretory", "Club": "Secretory",
    "Classical monocytes": "Myeloid", "Alveolar macrophages": "Myeloid",
    "Macrophages MARCO-": "Myeloid", "Macrophages SPP1 high": "Myeloid",
    "DC1": "Dendritic", "DC2": "Dendritic", "DC monocyte": "Dendritic",
    "B cells": "B_Plasma", "Plasma cells": "B_Plasma",
    "Fibroblasts PLIN2+": "Stromal", "Pericytes": "Stromal",
    "EC aerocyte capillary": "Stromal", "EC venous systemic": "Stromal",
    # "Transitional AT2": "Epithelial",
    # "CD4 T cells": "T_NK", "NK cells": "T_NK",
}
GROUPS = {}
for _fine, _coarse in COARSE_MAP.items():
    GROUPS.setdefault(_coarse, []).append(_fine)


def rename_none_cols(df):
    return df.rename(columns={c: "Others" for c in df.columns
                              if str(c).strip().lower() in ("none", "na", "nan")})


gt = rename_none_cols(pd.read_csv(f"{V3}/gt_actual_props_fold_2.csv", index_col=0))
gt.index = gt.index.astype(str); GT_TYPES = list(gt.columns)


def load():
    M = {}
    r = pd.read_csv(f"{UNI}/proportions_hca_fold2_nnls_core.csv", index_col=0); r.index = r.index.astype(str)
    M["DeconvSC"] = rename_none_cols(r)
    for m, p in [("BayesPrism", f"{BF}/BayesPrism/results/pred_proportions.csv"),
                 ("DISSECT", f"{BF}/DISSECT/results/pred_proportions.csv"),
                 ("TAPE", f"{BF}/TAPE/results/pred_proportions.csv")]:
        d = pd.read_csv(p, index_col=0); d.index = d.index.astype(str); M[m] = rename_none_cols(d)
    csx = pd.read_csv(f"{BF}/CIBERSORTx/results/CIBERSORTxGEP_Job37_Fractions.txt", sep="\t", index_col=0)
    csx = csx.drop(columns=[c for c in ["P-value", "Correlation", "RMSE"] if c in csx.columns])
    smap = pd.read_csv(f"{BF}/CIBERSORTx/sample_id_map.csv", index_col=0)["pseudo_bulk_id"].astype(str).to_dict()
    csx.index = [smap.get(s, s) for s in csx.index.astype(str)]
    M["CIBERSORTx"] = rename_none_cols(csx.rename(columns=CSX_REN))
    return M


def merge(df):
    listed = set(t for v in GROUPS.values() for t in v)
    full = dict(GROUPS)
    for t in GT_TYPES:
        if t not in listed:
            full[t] = [t]
    out = pd.DataFrame(index=df.index)
    for lab, mem in full.items():
        cols = [m for m in mem if m in df.columns]
        out[lab] = df[cols].sum(1) if cols else 0.0
    return out, list(full.keys())


def pcc_by_label(pred, gtm, labels):
    samp = [s for s in gtm.index if s in pred.index]
    return {c: (pearsonr(pred.loc[samp, c].astype(float).values, gtm.loc[samp, c].astype(float).values)[0]
               if np.std(pred.loc[samp, c]) > 0 and np.std(gtm.loc[samp, c]) > 0 else np.nan) for c in labels}


def main():
    M = load(); gtm, labels = merge(gt)
    tab = pd.DataFrame(index=labels, columns=ORDER, dtype=float)
    for m in ORDER:
        d = pcc_by_label(merge(M[m])[0], gtm, labels)
        for c in labels:
            tab.loc[c, m] = d[c]
    tab.insert(0, "mean_true_prop", gtm.mean(0).reindex(labels).values)
    tab = tab.sort_values("DeconvSC", ascending=False)
    means = tab[ORDER].mean()
    tab.loc["— MEAN —"] = [np.nan] + list(means.values)
    tab.index.name = "Cell type (S4 coarse)"
    tab.round(4).to_csv(f"{OUTD}/PROPORTION_merged14_by_celltype_PCC_HCA_fold2.csv")
    print("=== HCA fold2 — S4 (coarse lineage) per-cell-type PROPORTION PCC ===")
    print(tab.round(3).to_string()); print("\nMEAN:", {m: round(means[m], 3) for m in ORDER})

    # ---- per-cell-type grouped bar (DeconvSC vs refs), pastel + Arial + EMF ----
    sub = tab.loc[[i for i in tab.index if i != "— MEAN —"]]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(sub)); w = 0.16
    for i, m in enumerate(ORDER):
        ax.bar(x + (i - 2) * w, sub[m].values, w, label=f"{m} (μ={means[m]:.2f})",
               color=PAL[m], edgecolor="black", lw=0.3)
    ax.set_xticks(x); ax.set_xticklabels(sub.index, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Proportion PCC (across 105 samples)"); ax.set_ylim(-0.1, 1.0)
    ax.grid(axis="y", alpha=0.3); ax.axhline(0, color="black", lw=0.5)
    ax.set_title("HCA fold2 — per-cell-type proportion PCC (S4 coarse lineage merge)",
                 fontweight="bold", fontsize=12, pad=34)            # title above the legend
    ax.legend(fontsize=9, ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout()
    plot_style.save(fig, f"{FIGD}/HCA_fold2_proportion_S4_pcc", dpi=300)
    plt.close()
    print(f"[saved] {OUTD}/PROPORTION_merged14_by_celltype_PCC_HCA_fold2.csv")
    print(f"[saved] {FIGD}/HCA_fold2_proportion_S4_pcc.{{png,pdf,svg,emf}}")


if __name__ == "__main__":
    main()
