#!/usr/bin/env python
"""Assemble Figure 4 (two-column layout) as a single vector figure — the ONLY Figure 4
plotting script (reads precomputed results and draws every panel).

  a Hub-gene Jaccard          b MI fidelity error
  c Module L2 error           d Co-expression network fidelity
  e Sample-adaptation ablation (alpha sweep, spans both columns)

Inputs (all produced upstream; nothing is computed or hardcoded here):
  a/b/c/d : ablation_compare/attention_advantage_dists.npz  + corr0_genenet_v2.npz
            (both written by attention_advantage.py --dataset GSE141115)
  e       : result_gse141115.csv  (alpha sweep written by route2_gse141115.py)

Panels a-d: violin+box over DeconvSC(Route2)/BaseVAE/DISSECT, with two significance brackets
            each: DeconvSC vs BaseVAE (paired Wilcoxon) and DeconvSC vs DISSECT (Mann-Whitney),
            annotated with stars + p-value.
Panel  e  : alpha = 0 / 0.5 / 1 sweep across the 5 unified-PCC axes (warm-pink ramp).
Palette: DeconvSC = warm pink #D8B2AE; baselines (BaseVAE ablation, DISSECT) = cool blue tones.
"""
import numpy as np, pandas as pd, os
import matplotlib; matplotlib.use("Agg")
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "lib")); import plot_style  # noqa: F401  (Arial font applied on import + plot_style.save/emf_dir for EMF)
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu
import seaborn as sns

plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], 
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"]      = "Arial"
plt.rcParams["mathtext.it"]      = "Arial:italic"
plt.rcParams["mathtext.bf"]      = "Arial:bold"

AC = os.environ.get("FIG4_AC", "/disk1/maijl/deconv/deconv_20260610/prophead/GSE141115/ablation_compare")
OUT = os.environ.get("FIG4_OUT", "/disk1/maijl/deconv/deconv_20260610/prophead/GSE141115/ablation_compare/Figure4_rebuilt")
# palette: DeconvSC = warm pink; baselines (BaseVAE ablation, DISSECT) = cool blue tones
PAL = {"Route2": "#D8B2AE", "BaseVAE": "#7187A7", "DISSECT": "#A7B8C8"}
DISP = {"Route2": "DeconvSC", "BaseVAE": "Ablated VAE", "DISSECT": "DISSECT"}
ORDER = ["Route2", "BaseVAE", "DISSECT"]

d = np.load(f"{AC}/attention_advantage_dists.npz")
nf = np.load(f"{AC}/corr0_genenet_v2.npz")
DAT = {"hubJ": {k: d[f"hubJ_{k}"] for k in ORDER},
       "modL2": {k: d[f"modL2_{k}"] for k in ORDER},
       "dMI": {k: d[f"dMI_{k}"] for k in ORDER},
       "netfid": {k: nf[f"netfid_{k}"] for k in ORDER}}

# Panel c = WITHIN-cell-type Module L2 (composition-invariant). Only DeconvSC vs BaseVAE
# (DISSECT has 126 aggregated profiles, no within-type single cells). Distributions are
# precomputed by within_type_module_l2.py -> module_l2_withinType_dists.npz.
WT_ORDER = ["Route2", "BaseVAE"]
WT_DISP  = {"Route2": "DeconvSC", "BaseVAE": "Ablated VAE"}
WT_PAL   = {"Route2": "#D8B2AE", "BaseVAE": "#7187A7"}
_wt = np.load(f"{AC}/module_l2_withinType_dists.npz")
WT_DAT = {"Route2": _wt["Route2"], "BaseVAE": _wt["BaseVAE"]}
WT_P   = float(_wt["p"])


def save_panel(draw_fn, fname, figsize=(6, 5), bbox="tight", tight_layout=False):
    """把单个面板画到独立 figure 并导出 png/pdf/svg。bbox=None 保留精确画布尺寸（不裁剪）。"""
    f = plt.figure(figsize=figsize, dpi=600)
    ax = f.add_subplot(111)
    draw_fn(ax)                       # 复用与主图完全相同的绘制逻辑
    if tight_layout: f.tight_layout()
    base = os.path.join(AC, fname)
    for ext in ("png", "pdf", "svg"):
        f.savefig(f"{base}.{ext}", dpi=600, bbox_inches=bbox)
    plt.close(f)
    plot_style.relabel_svg(f"{base}.svg")   # Liberation/DejaVu -> literal Arial in the SVG
    plot_style.svg_to_emf(f"{base}.svg")     # + emit <fname>.emf (MS Office vector)
    
def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def fmt_p(p):
    if p is None or not np.isfinite(p): return "P = NA"
    if p < 1e-300: return "P < 1e-300"
    if p < 0.01: return f"P = {p:.0e}"
    return f"P = {p:.3f}"


def paired_p(dat, higher_better):  # DeconvSC vs BaseVAE (same-length -> paired Wilcoxon)
    a, b = dat["Route2"], dat["BaseVAE"]; n = min(len(a), len(b))
    return wilcoxon(a[:n], b[:n], alternative="greater" if higher_better else "less")[1]


def mann_p(dat, higher_better):  # DeconvSC vs DISSECT (unequal length -> Mann-Whitney)
    a, b = dat["Route2"], dat["DISSECT"]
    return mannwhitneyu(a, b, alternative="greater" if higher_better else "less")[1]

def violin(ax, dat, ylabel, higher_better):
    long = pd.DataFrame([{"Model": DISP[k], "v": float(x)} for k in ORDER for x in dat[k]])
    cats = [DISP[k] for k in ORDER]; pal = {DISP[k]: PAL[k] for k in ORDER}
    sns.violinplot(data=long, x="Model", y="v", order=cats, hue="Model", palette=pal,
                   legend=False, inner=None, alpha=0.35, cut=0, ax=ax)
    sns.boxplot(data=long, x="Model", y="v", order=cats, hue="Model", palette=pal,
                legend=False, width=0.35, fliersize=0, linewidth=1.0, ax=ax)
    ax.set_ylabel(ylabel, fontsize=16,  fontweight="bold"); ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), fontsize=14, fontweight="bold")
    # two brackets: DeconvSC vs BaseVAE (paired Wilcoxon) and DeconvSC vs DISSECT (Mann-Whitney)
    p1 = paired_p(dat, higher_better); p2 = mann_p(dat, higher_better)
    y0 = long["v"].max(); rng = max(long["v"].max() - long["v"].min(), 1e-9); h = rng * 0.04

    def bracket(xa, xb, base, p):
        ax.plot([xa, xa, xb, xb], [base, base + h, base + h, base], lw=1.0, c="black")
        ax.text((xa + xb) / 2, base + h, f"{stars(p)}\n{fmt_p(p)}", ha="center", va="bottom",
                fontsize=12, linespacing=1.7, fontweight="bold")

    b1 = y0 + rng * 0.05; bracket(0, 1, b1, p1)
    b2 = b1 + rng * 0.34; bracket(0, 2, b2, p2)
    ax.set_ylim(top=b2 + rng * 0.34)


def violin_wt(ax, dat, ylabel, p):
    """Within-type Module-L2 panel: 2 methods (DeconvSC vs BaseVAE), ONE precomputed bracket
    (one-sided paired Wilcoxon, DeconvSC < BaseVAE). Same violin+box style as violin()."""
    long = pd.DataFrame([{"Model": WT_DISP[k], "v": float(x)} for k in WT_ORDER for x in dat[k]])
    cats = [WT_DISP[k] for k in WT_ORDER]; pal = {WT_DISP[k]: WT_PAL[k] for k in WT_ORDER}
    sns.violinplot(data=long, x="Model", y="v", order=cats, hue="Model", palette=pal,
                   legend=False, inner=None, alpha=0.35, cut=0, ax=ax)
    sns.boxplot(data=long, x="Model", y="v", order=cats, hue="Model", palette=pal,
                legend=False, width=0.35, fliersize=0, linewidth=1.0, ax=ax)
    ax.set_ylabel(ylabel, fontsize=16, fontweight="bold"); ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), fontsize=14, fontweight="bold")
    y0 = long["v"].max(); rng = max(long["v"].max() - long["v"].min(), 1e-9); h = rng * 0.04
    base = y0 + rng * 0.05
    ax.plot([0, 0, 1, 1], [base, base + h, base + h, base], lw=1.0, c="black")
    ax.text(0.5, base + h, f"{stars(p)}\n{fmt_p(p)}", ha="center", va="bottom",
            fontsize=12, linespacing=1.7, fontweight="bold")
    ax.set_ylim(top=base + rng * 0.34)

# ---- 每个面板的绘制逻辑（不含字母；字母是拼图专属标注）----
def panel_a(ax): violin(ax, DAT["hubJ"], "Hub-gene Jaccard", True)
def panel_b(ax): violin(ax, DAT["dMI"], "MI fidelity error ", False)
def panel_c(ax): violin(ax, DAT["modL2"], "Module L2 error", False)   # option 1: GLOBAL module L2, 3 comparable models
def panel_c_wt(ax):
    """option 2: within-type Module L2 for DeconvSC/BaseVAE + DISSECT's GLOBAL value as the 3rd bar.
    NOTE: DISSECT is a different metric (global, no within-type single cells) -> it sits on a different
    scale; only the DeconvSC-vs-BaseVAE bracket (precomputed within-type paired Wilcoxon) is like-for-like."""
    order = ["Route2", "BaseVAE", "DISSECT"]
    dat = {"Route2": WT_DAT["Route2"], "BaseVAE": WT_DAT["BaseVAE"], "DISSECT": DAT["modL2"]["DISSECT"]}
    long = pd.DataFrame([{"Model": DISP[k], "v": float(x)} for k in order for x in dat[k]])
    cats = [DISP[k] for k in order]; pal = {DISP[k]: PAL[k] for k in order}
    sns.violinplot(data=long, x="Model", y="v", order=cats, hue="Model", palette=pal,
                   legend=False, inner=None, alpha=0.35, cut=0, ax=ax)
    sns.boxplot(data=long, x="Model", y="v", order=cats, hue="Model", palette=pal,
                legend=False, width=0.35, fliersize=0, linewidth=1.0, ax=ax)
    ax.set_ylabel("Module L2 error", fontsize=16, fontweight="bold"); ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), fontsize=14, fontweight="bold")
    y0 = long["v"].max(); rng = max(y0 - long["v"].min(), 1e-9); h = rng * 0.03
    bb = max(float(WT_DAT["Route2"].max()), float(WT_DAT["BaseVAE"].max())) + rng * 0.04  # bracket above the 2 within-type violins
    ax.plot([0, 0, 1, 1], [bb, bb + h, bb + h, bb], lw=1.0, c="black")
    ax.text(0.5, bb + h, f"{stars(WT_P)}\n{fmt_p(WT_P)}", ha="center", va="bottom",
            fontsize=12, linespacing=1.7, fontweight="bold")
    ax.set_ylim(top=max(bb + rng * 0.22, y0 * 1.08))
def panel_d(ax): violin(ax, DAT["netfid"], "Co-expression network fidelity\n(per-gene r)", True)
def panel_e(ax, fs=1.0):   # fs scales all fonts; composite uses 1.0 (=a-c), standalone export uses ~0.6
    metrics = ['Profile\n(all genes)', 'Profile\n(marker)', 'Gene-wise\n(all genes)',
               'Gene-wise\n(marker)', 'Sample\n(all genes)']
    ECOLS = ['prof_all', 'prof_mk', 'gene_all', 'gene_mk', 'samp_all']
    edf = pd.read_csv(os.path.join(AC, "result_gse141115.csv")).set_index("variant")
    a0  = edf.loc["pure-prior", ECOLS].astype(float).tolist()   # alpha = 0
    a05 = edf.loc["a=0.5",      ECOLS].astype(float).tolist()    # alpha = 0.5
    a1  = edf.loc["Route2",     ECOLS].astype(float).tolist()    # alpha = 1 (DeconvSC)
    c0, c05, c1 = '#A7B8C8', '#7187A7', '#D8B2AE'   # alpha=0/0.5 in the baseline cool blues (light/medium), alpha=1 = DeconvSC pink
    x = np.arange(len(metrics)) * 1.45; w = 0.27; d = 0.30   # widen group spacing (×1.45) + within-group bar offset d
    for off, vals, col, lab in [(-d, a1, c1, 'alpha=1 (DeconvSC)'),
                                (0, a05, c05, 'alpha=0.5'),
                                (d, a0, c0, 'alpha=0 (pure prior)')]:   # DeconvSC(alpha=1) first, then ablation -> like other panels
        bars = ax.bar(x + off, vals, w, label=lab, color=col, edgecolor='#3a3a3a', lw=0.4)
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.008, f'{r.get_height():.2f}',
                    ha='center', va='bottom', fontsize=8 * fs, color='#222')
    # base sizes = panels a-c (y-label 16, x-ticks 14, y-ticks 10); fs scales them for the standalone export
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=14 * fs, fontweight="bold")
    ax.set_ylabel('Pearson r (median)', fontsize=16 * fs, fontweight="bold"); ax.set_ylim(0, 1.05)
    ax.legend(prop={"size": 13 * fs, "weight": "bold"}, frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.18))
    ax.tick_params(labelsize=10 * fs)

def panel_letter(ax, s):
    ax.text(-0.16, 1.06, s, transform=ax.transAxes, fontsize=18, fontweight="bold", va="top", ha="left")

fig = plt.figure(figsize=(12, 15), dpi=600)
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.94], hspace=0.45, wspace=0.30,
                      left=0.085, right=0.965, top=0.965, bottom=0.075)

# --- a: hub jaccard ---
axa = fig.add_subplot(gs[0, 0])
violin(axa, DAT["hubJ"], "Hub-gene Jaccard", True); panel_letter(axa, "a")
save_panel(panel_a, "panel_a_hub_jaccard", (8, 8))
# --- b: MI fidelity ---
axb = fig.add_subplot(gs[0, 1])
violin(axb, DAT["dMI"], "MI fidelity error ", False)
panel_letter(axb, "b")
save_panel(panel_b, "panel_b_mi_fidelity", (8, 8))
# --- c: module L2 ---
axc = fig.add_subplot(gs[1, 0])
violin_wt(axc, WT_DAT, "Module L2 error", WT_P); panel_letter(axc, "c")   # Fig 4c = 2-bar within-type module L2 (DeconvSC vs Ablated VAE; composition-invariant)
save_panel(panel_c,    "panel_c_module_l2", (8, 8))             # option 1: global module L2, 3 comparable models
save_panel(panel_c_wt, "panel_c_module_l2_withinType", (8, 8))  # option 2: within-type DeconvSC/BaseVAE + DISSECT(global)
# --- d: co-expression network fidelity violin (per-gene), 3 models ---
axd = fig.add_subplot(gs[1, 1])
violin(axd, DAT["netfid"], "Co-expression network fidelity\n(per-gene r)", True)
panel_letter(axd, "d")
save_panel(panel_d, "panel_d_network_fidelity", (8, 8))
# --- e: alpha ablation (spans both columns) ---
axe = fig.add_subplot(gs[2, :])
panel_e(axe)   # SAME drawing as the standalone panel_e -> composite & sub-figure colours/order always match
# panel_letter(axe, "e")
save_panel(lambda ax: panel_e(ax, fs=0.6), "panel_e_alpha_ablation", (18 / 2.54, 8 / 2.54), bbox=None, tight_layout=True)  # 18 cm × 8 cm; fs=0.6 so fonts match panel e as it appears in Figure4_rebuilt


for ext in ("png", "pdf", "svg"):
    fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight")
print("saved ->", OUT)
