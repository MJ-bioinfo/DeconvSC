#!/usr/bin/env python
"""Reproduce the full 03_results/figures/ family for the v3 (cVAE = Route 2) results.
Reads the v3 per-axis pearson CSVs (03_results_v3_route2/<ds>/<method>/pearson_*.csv,
cVAE = Route2; other methods reused) and writes the same figure types into
03_result_figures/figures/:

  <ds>_profile_fine_pcc / _profile_major_pcc       (per-cell-type bars)
  <ds>_pseudobulk_fine_pcc / _pseudobulk_major_pcc
  main_profile_pcc / main_genewise_pcc             (cross-dataset marker-gene boxes)

Proportion figures (HCA_fold2_proportion*) are Route2-INVARIANT (proportions come from the
deconvolution solve, not the generation step), so the existing ones are copied verbatim.
"""
import os, shutil
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg")
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "lib")); import plot_style  # noqa: F401  (Arial font applied on import + plot_style.save/emf_dir for EMF)
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import seaborn as sns

for _fp in ["/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
            "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf"]:
    try: fm.fontManager.addfont(_fp)
    except Exception: pass
plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "pdf.fonttype": 42, "svg.fonttype": "none"})

WORKROOT = Path(os.environ.get("WORK_ROOT", "/disk1/maijl/deconv/deconv_20260605"))
RES = WORKROOT / "03_result_tables"                 # pearson CSVs written by run_eval_v3_route2 (this scheme)
SRC_FIG = WORKROOT / "03_result_tables" / "unified" # this scheme's proportion-benchmark figures
OUT = WORKROOT / "03_result_figures" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
DATASETS = ["HCA_fold2", "GSE141115", "GSE159585"]

METHODS_FINE = [("BayesPrism", "BayesPrism"), ("DISSECT", "DISSECT"), ("TAPE", "TAPE"), ("cVAE", "DeconvSC")]
METHOD_ORDER = ["BayesPrism", "CIBERSORTx", "DISSECT", "TAPE", "DeconvSC"]
PALETTE = {"BayesPrism": "#86ADC0", "CIBERSORTx": "#7187A7", "DISSECT": "#AA96CA",
           "TAPE": "#989EB0", "DeconvSC": "#DEBBB1"}

HLCA = {
    "AlveolarEpithelial": ["AT0","AT1","AT2","AT2 proliferating","pre-TB secretory","Basal resting","Suprabasal",
        "Club (nasal)","Club (non-nasal)","Goblet (bronchial)","Goblet (nasal)","Goblet (subsegmental)",
        "Multiciliated (non-nasal)","Deuterosomal","Ionocyte","Neuroendocrine","Mesothelium","Club","Goblet","Transitional AT2"],
    "CapillaryVenousEndothelial": ["EC aerocyte capillary","EC general capillary","EC venous pulmonary","EC venous systemic",
        "EC arterial","Lymphatic EC mature","Lymphatic EC differentiating"],
    "Lymphoid": ["B cells","CD4 T cells","CD8 T cells","NK cells","Plasma cells","T cells proliferating","Hematopoietic stem cells"],
    "Myeloid": ["Alveolar macrophages","Alveolar Mph CCL3+","Alveolar Mph MT-positive","Alveolar Mph proliferating",
        "Monocyte-derived Mph","Interstitial Mph perivascular","Classical monocytes","Non-classical monocytes","DC1","DC2",
        "Plasmacytoid DCs","Migratory DCs","Mast cells","Macrophages MARCO-","Macrophages SPP1 high","DC monocyte"],
    "StromalFibroblast": ["Adventitial fibroblasts","Alveolar fibroblasts","Peribronchial fibroblasts","Subpleural fibroblasts",
        "Myofibroblasts","Smooth muscle","Smooth muscle FAM83D+","SM activated stress response","Pericytes",
        "Activated myofibroblasts","Fibroblasts PLIN2+"],
}
MAJOR_MAPS = {
    "HCA_fold2": HLCA,
    "GSE141115": {
        "Collecting_Duct": ["CD_IC","CD_PC","CD_Trans","CD_IC_A","CD_IC_B"], "Glomerular": ["MC","Podo"],
        "Immune": ["B","MPH","NK","Neut","T"], "Tubular_Epithelial": ["PT","DCT","CNT","aLOH"],
        "Vascular_Stromal": ["Endo","Fib"]},
    "GSE159585": {
        "Bcell": ["B_cell","plasma"],
        "Epithelial": ["AT1","AT2","ciliated","club","goblet","basal","proliferating_epithelial","neuroendocrine","AT0",
            "rare_epithelial","alveolar_AT0","epithelial_proliferating","epithelial_airway_ciliated",
            "epithelial_airway_secretory","mesothelial"],
        "Fibroblast": ["fibroblast","adventitial_fibroblast","alveolar_fibroblast","myofibroblast","matrix_fibroblast",
            "peribronchial_fibroblast","fibroblast_intermediate","fibroblast_matrix","fibromyocyte"],
        "LEC": ["LEC","lymphatic_EC"],
        "Myeloid": ["macrophage_alv","macrophage_interstitial","monocyte_classical","monocyte_nonclassical",
            "dendritic_cell_plasmacytoid","dendritic_cell_conventional","mast_cell","neutrophil","macrophage","DC","monocyte",
            "macro_alveolar","macro_alveolar_SPP1","macro_HS3ST2","monocyte_IL1B","monocyte_non_classical","cDC_type_1",
            "cDC_type_2","migDC","pDC","granulocyte","proliferating_myeloid"],
        "NKcell": ["NK","NK_KLRC1"],
        "SMC_Pericyte": ["SMC","pericyte","vascular_SMC","airway_SMC","smooth_muscle"],
        "T_cell": ["CD4_EM","CD4_naive","CD8_EMRA","CD4_Treg","CD8_EM","CD4_Th17","CD8_RM","CD8_naive",
            "proliferating_lymphoid","CD4_Th1","CD8_MAIT","CD8_gamma_delta"],
        "Vascular": ["cap_1","cap_2","cap_CX3CL1","cap_PB","vein_1","vein_2","vein_PB","arterial","venous","artery_1",
            "artery_2","cap_arterial","cap_venous","large_vessel","proliferating_endothelial"]},
}


def read_metric(ds, method, metric):
    f = RES / ds / method / f"pearson_{metric}_all_genes.csv"
    if not f.exists(): return None
    df = pd.read_csv(f)
    if df.shape[1] == 2: df.columns = ["CellType", "Value"]
    elif df.shape[1] >= 3: df.columns = ["Sample", "CellType", "Value"] + list(df.columns[3:])
    else: return None
    df = df[["CellType", "Value"]].copy(); df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"]); df = df[df["CellType"].astype(str) != ""]
    return df.groupby("CellType", as_index=False)["Value"].mean() if len(df) else None


def to_major(fine_df, mmap):
    if fine_df is None or not len(fine_df): return None
    fine2major = {ct.lower(): mg for mg, cts in mmap.items() for ct in cts}
    g = fine_df.copy(); g["Major"] = g["CellType"].str.lower().map(fine2major)
    g = g.dropna(subset=["Major"])
    if not len(g): return None
    return g.groupby("Major", as_index=False)["Value"].mean().rename(columns={"Major": "CellType"})


def build(ds, metric, level):
    mmap = MAJOR_MAPS[ds]; rows = []
    for m, disp in METHODS_FINE:
        df = read_metric(ds, m, metric)
        if df is None: continue
        if level == "major":
            df = to_major(df, mmap)
            if df is None: continue
        df = df.copy(); df["Method"] = disp; rows.append(df)
    if level == "major":
        # CIBERSORTx is stored at MAJOR level natively for GSE (celltype names == major
        # groups) but at FINE level for HCA -> use directly if already major, else aggregate.
        cx = read_metric(ds, "CIBERSORTx", metric)
        if cx is not None:
            major_keys = set(mmap.keys())
            if set(cx["CellType"].astype(str)) & major_keys:
                cx = cx[cx["CellType"].astype(str).isin(major_keys)].copy()
            else:
                cx = to_major(cx, mmap)
            if cx is not None and len(cx):
                cx = cx.copy(); cx["Method"] = "CIBERSORTx"; rows.append(cx)
    if not rows: return None
    out = pd.concat(rows, ignore_index=True)
    return out[~out["CellType"].isin(["Others", "others_merged", "Unknown"])]


def make_bar(ds, df, metric, level):
    if df is None or not len(df): print(f"skip {ds} {metric}/{level}: empty"); return
    # GSE159585: show only cell types DeconvSC actually reconstructed — prophead's realistic-sparse
    # composition yields 0 cells for some rare types, so they have no DeconvSC profile. Drop them so
    # every plotted cell type has a focal (DeconvSC) bar rather than an empty slot.
    if ds == "GSE159585" and "DeconvSC" in set(df["Method"]):
        keep = set(df.loc[df["Method"] == "DeconvSC", "CellType"])
        dropped = sorted(set(df["CellType"]) - keep)
        df = df[df["CellType"].isin(keep)].copy()
        if dropped: print(f"[{ds} {metric}/{level}] dropped {len(dropped)} cell types absent in DeconvSC: {dropped}")
        if not len(df): print(f"skip {ds} {metric}/{level}: no DeconvSC cell types"); return
    metric_lab = "Profile-wise" if metric == "profile" else "Pseudo-bulk"
    order = df.groupby("CellType")["Value"].mean().sort_values(ascending=False).index.tolist()
    methods = [m for m in METHOD_ORDER if m in df["Method"].unique()]
    piv = df.pivot_table(index="CellType", columns="Method", values="Value").reindex(order)
    n_ct, n_m = len(order), len(methods)
    figw = max(4.0, (180 if level == "fine" else 95) / 25.4 * (n_ct / 16 + 0.3))
    fig, ax = plt.subplots(figsize=(figw, 100 / 25.4)); bw = 0.8 / n_m; x = np.arange(n_ct)
    for i, m in enumerate(methods):
        vals = piv[m].values if m in piv else np.full(n_ct, np.nan)
        ax.bar(x + (i - (n_m - 1) / 2) * bw, vals, bw, label=m, color=PALETTE[m], edgecolor="white", linewidth=0.1)
    ymin = min(0.0, np.nanmin(df["Value"].values)); ymax = max(1.0, np.nanmax(df["Value"].values))
    ax.set_ylim(ymin, ymax * 1.02); ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=9, fontweight="bold")
    ax.set_xlim(-0.5, n_ct - 0.5)
    ax.set_ylabel(f"{metric_lab} Pearson correlation coefficient", fontsize=9); ax.set_xlabel("Cell type", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(0.4); ax.spines["bottom"].set_linewidth(0.4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=n_m, frameon=False, fontsize=9, handlelength=1.1, columnspacing=1.2)
    ax.set_title(f"{ds} (cVAE = Route 2)", fontsize=9, loc="left", pad=18)
    fig.tight_layout()
    base = f"{ds}_{metric}_{level}_pcc"
    df.assign(CellType=pd.Categorical(df["CellType"], categories=order, ordered=True)).sort_values(["CellType", "Method"]).to_csv(OUT / f"{base}.csv", index=False)
    for ext in ("png", "pdf", "svg"): fig.savefig(OUT / f"{base}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig); print(f"saved {base} ({n_ct} cts, {n_m} methods)")


# ---- main cross-dataset marker-gene boxes ----
MAIN_METHODS = [("BayesPrism", "BayesPrism"), ("DISSECT", "DISSECT"), ("TAPE", "TAPE"), ("cVAE", "DeconvSC")]
MAIN_ORDER = ["BayesPrism", "DISSECT", "TAPE", "DeconvSC"]
MAIN_PAL = {"BayesPrism": "#86ADC0", "DISSECT": "#AA96CA", "TAPE": "#989EB0", "DeconvSC": "#DEBBB1"}
DS_LABEL = {"HCA_fold2": "HLCA simulation", "GSE141115": "GSE141115 (mouse kidney)", "GSE159585": "GSE159585 (human lung)"}


def load_vec(ds, method, axis):
    f = RES / ds / method / f"pearson_{axis}_marker_genes.csv"
    if not f.exists(): return None
    v = pd.to_numeric(pd.read_csv(f).iloc[:, -1], errors="coerce").dropna()
    return v.values if len(v) >= 3 else None


def main_boxes():
    rows = []
    for ds in DATASETS:
        for axis in ("profile", "gene"):
            for m, disp in MAIN_METHODS:
                v = load_vec(ds, m, axis)
                if v is None: continue
                for x in v: rows.append({"dataset": ds, "axis": axis, "method": disp, "r": x})
    long = pd.DataFrame(rows); long.to_csv(OUT / "main_profile_genewise_marker.csv", index=False)
    for axis, lab, fname in [("profile", "Profile-wise", "main_profile_pcc"), ("gene", "Gene-wise", "main_genewise_pcc")]:
        fig, axarr = plt.subplots(1, len(DATASETS), figsize=(5 * len(DATASETS), 5), squeeze=False)
        for c, ds in enumerate(DATASETS):
            ax = axarr[0][c]; sub = long[(long.dataset == ds) & (long.axis == axis)]
            present = [m for m in MAIN_ORDER if m in set(sub.method)]
            if not present: continue
            sns.boxplot(data=sub, x="method", y="r", order=present, hue="method", palette=MAIN_PAL,
                        legend=False, width=0.62, fliersize=1.4, linewidth=0.6, ax=ax)
            med = sub.groupby("method")["r"].median()
            for i, m in enumerate(present):
                ax.text(i, 1.02, f"{med[m]:.2f}", ha="center", va="bottom", fontsize=12,
                        fontweight="bold" if m == "DeconvSC" else "normal", color="#B5651D" if m == "DeconvSC" else "black")
            ax.set_ylim(-0.05, 1.12); ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.set_xlabel("")
            ax.set_ylabel(f"{lab} PCC" if c == 0 else "", fontsize=12); ax.set_title(DS_LABEL[ds], fontsize=11, pad=4)
            ax.tick_params(axis="x", labelsize=12)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", va="top", rotation_mode="anchor", fontweight="bold")
            sns.despine(ax=ax)
        fig.tight_layout(w_pad=1.8)
        for ext in ("png", "pdf", "svg"): fig.savefig(OUT / f"{fname}.{ext}", dpi=600, bbox_inches="tight")
        plt.close(fig); print(f"saved {fname}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Full v3 Route2 figure family")
    ap.add_argument("--outdir", default=None, help="figure output dir (default = the cVAE eval figures tree)")
    a = ap.parse_args()
    if a.outdir:
        OUT = Path(a.outdir); OUT.mkdir(parents=True, exist_ok=True)
        print(f"[outdir] figures -> {OUT}")
    for ds in DATASETS:
        for mt in ("profile", "pseudobulk"):
            for lv in ("fine", "major"):
                make_bar(ds, build(ds, mt, lv), mt, lv)
    main_boxes()
    # proportion-benchmark figures for THIS scheme (no longer Route2-invariant)
    for f in SRC_FIG.glob("UNIFIED_proportion*.png"):
        shutil.copy(f, OUT / f.name)
    print(f"[copied] {len(list(SRC_FIG.glob('UNIFIED_proportion*.png')))} proportion-benchmark figures")
    print(f"\nDONE -> {OUT}")
