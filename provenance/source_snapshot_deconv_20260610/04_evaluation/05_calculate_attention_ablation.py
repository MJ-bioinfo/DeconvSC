#!/usr/bin/env python
"""Attention-advantage triptych (robust): Hub-gene Jaccard, Module L2 error, MI fidelity |dMI|.

Replicates the three GRN-topology metrics of script/GSE141115_ablation_visualize.py
(Exp A hub Jaccard, Exp C module L2, Exp B mutual information) but made robust:
matched-N subsampling across methods + multiple seeds, all on a fixed top-K real-HVG set.
Hub-Jaccard and L2 keep the ORIGINAL code's definitions + violin/box style + palette
(#D8B2AE warm = DeconvSC, #7187A7 cool = baseline); the MI panel is reframed to the
magnitude-fidelity |MI_gen - MI_real| (lower = better) using the SAME palette.

Run:  python attention_advantage.py --dataset GSE141115   (4 methods)
      python attention_advantage.py --dataset HCA          (Route2/DeltaZ vs DISSECT; no BaseVAE)
"""
import os, sys, argparse, itertools, numpy as np, pandas as pd, anndata as ad, scanpy as sc, scipy.sparse as sp
from scipy.stats import wilcoxon, pearsonr
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "lib")); import plot_style  # noqa: F401  (Arial font applied on import + plot_style.save/emf_dir for EMF)
try: import seaborn as sns; HAVE_SNS = True
except Exception: HAVE_SNS = False
import warnings; warnings.filterwarnings("ignore"); sc.settings.verbosity = 0

# warm pink = DeconvSC (original Ours #D8B2AE); cool blues = baselines (#7187A7)
from matplotlib.colors import LinearSegmentedColormap
PAL = {"Route2": "#D8B2AE", "DeltaZ": "#C0857C", "BaseVAE": "#7187A7", "DISSECT": "#A7B8C8"}
DISP = {"Route2": "DeconvSC", "DeltaZ": "DeltaZ", "BaseVAE": "Ablation", "DISSECT": "DISSECT"}
NET_CMAP = LinearSegmentedColormap.from_list("custom_map", ['#D8B2AE', '#F7F3EF', '#7187A7'])

DATASETS = {
    "GSE141115": dict(
        real="/disk1/maijl/deconv/data/GSE141115/GSE141115_sc_raw_counts_test.h5ad", real_celltype="cell type",
        methods={"Route2": os.environ.get("OURS_H5AD", "/disk1/maijl/deconv/deconv_20260605/GSE141115/generated_data.h5ad"),
                 "BaseVAE": "/disk1/maijl/deconv/cVAE/GSE141115/ablation_basevae/BaseVAE-generated_data.h5ad",
                 "DISSECT": "/disk1/maijl/deconv/dissect/GSE141115/experiment_11.18.2025_10.43.44/est_expression.h5ad"}),
    "HCA": dict(
        real="/disk1/maijl/deconv/cVAE/Simulation_HCA/result/HCA/v3_pureprior/fold_2/gt_validation_cells.h5ad",
        real_celltype="Cell_type", real_is_log=True,
        methods={"Route2": "/disk1/maijl/deconv/deconv_20260605/HCA/generated_data.h5ad",
                 "DeltaZ": "/disk1/maijl/deconv/cVAE/Simulation_HCA/result/HCA/v2/fold_2/pred_validation_cells.h5ad",
                 "DISSECT": "/disk1/maijl/deconv/dissect/HCA/experiment_10.18.2025_09.22.47/est_expression.h5ad"}),
}
NBINS = 20; TOPK = int(os.environ.get("AA_TOPK", "500")); NSEED = int(os.environ.get("AA_SEEDS", "5"))
N_HUB = 50; K_NB = 20; N_MOD = 20; TOP_MOD = 10; MI_TOPK = 200
OUTROOT = os.environ.get("ABLATION_OUTROOT", "/disk1/maijl/deconv/deconv_20260605")


def dense(a, genes):
    X = a[:, genes].X; return np.asarray(X.toarray() if sp.issparse(X) else X)


def corr_of(X):
    return np.nan_to_num(np.corrcoef(X, rowvar=False))


def hub_jaccard(cr, cm, hubs, k=K_NB):
    out = []
    for h in hubs:
        rn = set(np.argsort(np.abs(cr[h]))[-k:]); mn = set(np.argsort(np.abs(cm[h]))[-k:])
        out.append(len(rn & mn) / len(rn | mn))
    return np.array(out)


def module_l2(cr, cm, modules):
    out = []
    for idx in modules:
        r = cr[np.ix_(idx, idx)]; m = cm[np.ix_(idx, idx)]
        out.append(np.linalg.norm(r - m) / r.size)
    return np.array(out)


def bin_matrix(X, nb=NBINS):
    codes = np.zeros(X.shape, np.int64)
    for g in range(X.shape[1]):
        c = X[:, g]; lo, hi = c.min(), c.max()
        if hi - lo > 1e-12: codes[:, g] = np.clip(np.digitize(c, np.linspace(lo, hi, nb + 1)[1:-1]), 0, nb - 1)
    return codes


def mi_vec(codes, pairs, nb=NBINS):
    out = np.empty(len(pairs))
    for p, (i, j) in enumerate(pairs):
        jh = np.bincount(codes[:, i] * nb + codes[:, j], minlength=nb * nb).reshape(nb, nb).astype(float)
        n = jh.sum()
        if n == 0: out[p] = 0; continue
        pp = jh / n; px = pp.sum(1, keepdims=True); py = pp.sum(0, keepdims=True); nz = pp > 0
        out[p] = (pp[nz] * np.log(pp[nz] / (px @ py)[nz])).sum()
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="GSE141115"); args = ap.parse_args()
    cfg = DATASETS[args.dataset]; outdir = f"{OUTROOT}/{args.dataset}/ablation_compare"; os.makedirs(outdir, exist_ok=True)
    real = ad.read_h5ad(cfg["real"]); real.var_names_make_unique()
    if not cfg.get("real_is_log"): sc.pp.normalize_total(real, target_sum=1e4); sc.pp.log1p(real)
    mats = {k: ad.read_h5ad(p) for k, p in cfg["methods"].items()}
    for k in mats: mats[k].var_names_make_unique()
    methods = list(cfg["methods"].keys())

    common = set(real.var_names)
    for k in methods: common &= set(mats[k].var_names)
    common = sorted(common); v = dense(real, common).var(0)
    topk = [common[i] for i in np.argsort(-v)[:TOPK]]
    mi_genes = topk[:MI_TOPK]; mi_pairs = list(itertools.combinations(range(MI_TOPK), 2))
    cell_methods = [k for k in methods if mats[k].n_obs > 500]
    N = min([real.n_obs] + [mats[k].n_obs for k in cell_methods] + [int(os.environ.get("AA_MAXN", "12000"))])
    print(f"[{args.dataset}] top-{TOPK} HVG; matched N={N} (cell-level); methods={methods}; "
          + ", ".join(f"{k} n={mats[k].n_obs}" for k in methods))

    # accumulate distributions over seeds; keep seed-0 corr matrices for the gene-network heatmaps
    hubJ = {k: [] for k in methods}; modL2 = {k: [] for k in methods}; dMI = {k: [] for k in methods}
    netPCC = {k: [] for k in methods}; netDist = {k: [] for k in methods}
    corr0 = {}; iu = np.triu_indices(TOPK, 1)
    for s in range(NSEED):
        rs = np.random.default_rng(s)
        def subX(a, genes):
            n = a.n_obs; idx = rs.choice(n, N, replace=False) if n > N else np.arange(n)
            return dense(a[idx], genes)
        Xr = subX(real, topk); cr = corr_of(Xr)
        if s == 0: corr0["real"] = cr
        hubs = np.argsort((np.abs(cr) - np.eye(TOPK)).sum(1))[-N_HUB:]
        Z = linkage(cr, method="ward"); lab = fcluster(Z, t=N_MOD, criterion="maxclust")
        mods = [np.where(lab == l)[0] for l in np.unique(lab) if (lab == l).sum() >= 6]
        mods = sorted(mods, key=lambda idx: -np.abs(cr[np.ix_(idx, idx)]).mean())[:TOP_MOD]
        mir = mi_vec(bin_matrix(subX(real, mi_genes)), mi_pairs)
        for k in methods:
            ak = mats[k]
            if k not in cell_methods and s > 0:
                continue  # DISSECT: native N, compute once
            Xk = subX(ak, topk) if k in cell_methods else dense(ak, topk)
            ck = corr_of(Xk)
            if s == 0: corr0[k] = ck
            hubJ[k].append(hub_jaccard(cr, ck, hubs)); modL2[k].append(module_l2(cr, ck, mods))
            netPCC[k].append(np.corrcoef(ck[iu], cr[iu])[0, 1]); netDist[k].append(np.linalg.norm(cr - ck))
            mik = mi_vec(bin_matrix(subX(ak, mi_genes) if k in cell_methods else dense(ak, mi_genes)), mi_pairs)
            dMI[k].append(np.abs(mik - mir))

    # pool distributions
    def pool(d): return {k: np.concatenate(v) for k, v in d.items() if v}
    HJ, ML, DM = pool(hubJ), pool(modL2), pool(dMI)
    rows = []
    print(f"\n{'method':>8} | {'hubJaccard↑':>11} {'moduleL2↓':>10} {'MI|dMI|↓':>9} {'netPCC↑':>8}")
    for k in methods:
        hj = HJ[k].mean(); ml = ML[k].mean(); dm = np.median(DM[k]); np_ = float(np.mean(netPCC[k]))
        flag = "" if k in cell_methods else f"  [N={mats[k].n_obs}]"
        print(f"{k:>8} | {hj:>11.4f} {ml:>10.4f} {dm:>9.4f} {np_:>8.4f}{flag}")
        rows.append(dict(dataset=args.dataset, method=k, hub_jaccard=hj, module_l2=ml,
                         mi_dMI_median=dm, network_pcc=np_, N=int(mats[k].n_obs)))
    # significance: DeconvSC vs BaseVAE (if present)
    if "BaseVAE" in methods:
        print("\n[sig] DeconvSC vs BaseVAE (Wilcoxon, pooled):")
        for dsc in [m for m in ["Route2", "DeltaZ"] if m in methods]:
            ph = wilcoxon(HJ[dsc], HJ["BaseVAE"], alternative="greater")[1]   # higher Jaccard better
            pl = wilcoxon(ML[dsc][:len(ML['BaseVAE'])], ML["BaseVAE"][:len(ML[dsc])], alternative="less")[1]
            pm = wilcoxon(DM[dsc], DM["BaseVAE"], alternative="less")[1]
            print(f"  {dsc}: hubJ>BaseVAE p={ph:.1e} | L2<BaseVAE p={pl:.1e} | dMI<BaseVAE p={pm:.1e}")
    pd.DataFrame(rows).to_csv(f"{outdir}/attention_advantage_summary.csv", index=False)

    # save raw distributions for re-plotting (Figure 4 panels a/b/c)
    np.savez(f"{outdir}/attention_advantage_dists.npz",
             **{f"hubJ_{k}": HJ[k] for k in methods}, **{f"modL2_{k}": ML[k] for k in methods},
             **{f"dMI_{k}": DM[k] for k in methods}, methods=np.array(methods))

    # per-gene co-expression network fidelity (Figure 4 panel d): for each gene g,
    # corr(C_gen[g, !g], C_real[g, !g]) from the seed-0 correlation matrices -> distribution.
    cr0 = corr0["real"]; offdiag = ~np.eye(TOPK, dtype=bool)

    def per_gene_fidelity(ck0):
        out = np.empty(TOPK)
        for g in range(TOPK):
            a = cr0[g][offdiag[g]]; b = ck0[g][offdiag[g]]
            out[g] = 0.0 if a.std() < 1e-12 or b.std() < 1e-12 else np.corrcoef(a, b)[0, 1]
        return np.nan_to_num(out)

    netfid = {k: per_gene_fidelity(corr0[k]) for k in methods}
    np.savez(f"{outdir}/corr0_genenet_v2.npz", real=cr0,
             **{f"netfid_{k}": netfid[k] for k in methods}, methods=np.array(methods))
    print("[saved] corr0_genenet_v2.npz (per-gene net fidelity median: "
          + ", ".join(f"{k}={np.median(netfid[k]):.3f}" for k in methods) + ")")

    # ---------------- THREE standalone figures (original Exp_C violin+box style) ----------------
    plt.rcParams.update({"font.size": 11, "font.family": ["Arial", "DejaVu Sans"],
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    MM = 1 / 25.4
    SINGLE_WIDTH_MM = 65
    SINGLE_HEIGHT_MM = 85
    TRIPTYCH_WIDTH_MM = 180
    TRIPTYCH_HEIGHT_MM = 85

    def stars(p):
        return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"

    def sig_p(dat, dsc, base, higher_better):
        if base not in dat or dsc not in dat: return None
        alt = "greater" if higher_better else "less"
        n = min(len(dat[dsc]), len(dat[base]))
        try: return wilcoxon(dat[dsc][:n], dat[base][:n], alternative=alt)[1]
        except Exception: return None

    def legacy_one_panel(dat, ylabel, fname, higher_better, order, odir):
        long = pd.DataFrame([{"Model": DISP[k], "v": float(x)} for k in order for x in dat[k]])
        cats = [DISP[k] for k in order]; pal = {DISP[k]: PAL[k] for k in order}
        fig, ax = plt.subplots(figsize=(3.6 if len(order) == 2 else 5.4, 5.2))
        if HAVE_SNS:
            sns.violinplot(data=long, x="Model", y="v", order=cats, hue="Model", palette=pal,
                           legend=False, inner=None, alpha=0.35, cut=0, ax=ax)
            sns.boxplot(data=long, x="Model", y="v", order=cats, hue="Model", palette=pal,
                        legend=False, width=0.35, fliersize=0, linewidth=1.0, ax=ax)
        ax.set_ylabel(ylabel, fontsize=11); ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), fontsize=9)
        # significance brackets: DeconvSC (Route2) vs each baseline in `order`
        bases = [m for m in order if m != "Route2"]
        y0 = long["v"].max(); rng_ = max(long["v"].max() - long["v"].min(), 1e-9); h = rng_ * 0.05; lvl = 0
        ia = order.index("Route2")
        for base in bases:
            p = sig_p(dat, "Route2", base, higher_better)
            if p is None: continue
            ib = order.index(base); yy = y0 + rng_ * 0.05 + lvl * (h + rng_ * 0.08)
            ax.plot([ia, ia, ib, ib], [yy, yy + h, yy + h, yy], lw=1.0, c="black")
            ax.text((ia + ib) / 2, yy + h, stars(p), ha="center", va="bottom", fontsize=12); lvl += 1
        ax.set_ylim(top=y0 + rng_ * (0.12 + lvl * 0.24))
        # ax.set_title(f"GSE141115 — {ylabel.splitlines()[0]}", fontsize=10, fontweight="bold", loc="left")
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"{odir}/{fname}.{ext}", dpi=600, bbox_inches="tight")
        plt.close(fig)

    def save_figure(fig, odir, fname):
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"{odir}/{fname}.{ext}", dpi=600)

    def draw_panel(ax, dat, ylabel, title, higher_better, order):
        order = [k for k in order if k in dat and len(dat[k]) > 0]
        long = pd.DataFrame([{"Model": DISP[k], "v": float(x)} for k in order for x in dat[k]])
        cats = [DISP[k] for k in order]; pal = {DISP[k]: PAL[k] for k in order}
        if HAVE_SNS:
            sns.violinplot(data=long, x="Model", y="v", order=cats, hue="Model", palette=pal,
                           legend=False, inner=None, alpha=0.35, cut=0, ax=ax)
            sns.boxplot(data=long, x="Model", y="v", order=cats, hue="Model", palette=pal,
                        legend=False, width=0.35, fliersize=0, linewidth=1.0, ax=ax)
        else:
            vals = [long.loc[long["Model"] == cat, "v"].to_numpy() for cat in cats]
            bp = ax.boxplot(vals, labels=cats, widths=0.35, patch_artist=True, showfliers=False)
            for patch, cat in zip(bp["boxes"], cats):
                patch.set_facecolor(pal[cat])
                patch.set_alpha(0.6)
        ax.set_ylabel(ylabel, fontsize=9, fontweight="bold")
        ax.set_xlabel("Method", fontsize=9, fontweight="bold")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.tick_params(axis="both", labelsize=8, width=0.3, length=2)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
                 rotation_mode="anchor", fontweight="bold")
        plt.setp(ax.get_yticklabels(), fontweight="bold")
        for side in ("bottom", "left"):
            ax.spines[side].set_linewidth(0.4)

        bases = [m for m in order if m != "Route2"]
        y0 = long["v"].max(); rng_ = max(long["v"].max() - long["v"].min(), 1e-9)
        h = rng_ * 0.05; lvl = 0
        ia = order.index("Route2")
        for base in bases:
            p = sig_p(dat, "Route2", base, higher_better)
            if p is None:
                continue
            ib = order.index(base); yy = y0 + rng_ * 0.05 + lvl * (h + rng_ * 0.08)
            ax.plot([ia, ia, ib, ib], [yy, yy + h, yy + h, yy], lw=1.0, c="black")
            ax.text((ia + ib) / 2, yy + h, stars(p), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
            lvl += 1
        ax.set_ylim(top=y0 + rng_ * (0.12 + lvl * 0.24))

    def one_panel(dat, ylabel, title, fname, higher_better, order, odir):
        fig, ax = plt.subplots(figsize=(SINGLE_WIDTH_MM * MM, SINGLE_HEIGHT_MM * MM))
        draw_panel(ax, dat, ylabel, title, higher_better, order)
        fig.tight_layout(pad=0.4)
        save_figure(fig, odir, fname)
        plt.close(fig)

    def triptych(metric_specs, fname, order, odir):
        fig, axes = plt.subplots(1, 3, figsize=(TRIPTYCH_WIDTH_MM * MM, TRIPTYCH_HEIGHT_MM * MM))
        for ax, spec in zip(axes, metric_specs):
            dat, ylabel, title, higher_better = spec
            draw_panel(ax, dat, ylabel, title, higher_better, order)
        fig.tight_layout(pad=0.4, w_pad=0.9)
        save_figure(fig, odir, fname)
        plt.close(fig)

    def gene_network_heatmap(compare, odir):
        cr, co, cc = corr0["real"], corr0["Route2"], corr0[compare]
        po, do = pearsonr(cr.flatten(), co.flatten())[0], np.linalg.norm(cr - co)
        pc, dc = pearsonr(cr.flatten(), cc.flatten())[0], np.linalg.norm(cr - cc)
        gc = sns.clustermap(pd.DataFrame(cr), cmap=NET_CMAP, center=0); idx = gc.dendrogram_row.reordered_ind; plt.close()
        ro = lambda m: m[idx, :][:, idx]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        kw = dict(cmap=NET_CMAP, center=0, vmin=-0.8, vmax=0.8, cbar=True, xticklabels=False, yticklabels=False)
        sns.heatmap(ro(cr), ax=axes[0], **kw); axes[0].set_title("Ground Truth (Real)", fontsize=20)
        sns.heatmap(ro(co), ax=axes[1], **kw); axes[1].set_title(f"DeconvSC\nPCC={po:.3f}, Dist={do:.2f}", fontsize=20)
        sns.heatmap(ro(cc), ax=axes[2], **kw); axes[2].set_title(f"{DISP[compare].replace(chr(10),' ')}\nPCC={pc:.3f}, Dist={dc:.2f}", fontsize=20)
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"{odir}/Gene_Network_Comparison.{ext}", dpi=600, bbox_inches="tight")
        plt.close(fig)
        return po, do, pc, dc

    YL = {"hub": "Hub-gene neighbor Jaccard",
          "l2": "Normalized module L2 distance",
          "mi": "MI fidelity error  |MI$_{gen}$ - MI$_{real}$|"}
    PANELS = [
        (HJ, YL["hub"], "Hub-gene Jaccard", True),
        (ML, YL["l2"], "Module L2", False),
        (DM, YL["mi"], "MI fidelity", False),
    ]

    # ---- overall figures: DeconvSC vs BaseVAE vs DISSECT ----
    print(f"\n[figures overall] -> {outdir}")
    one_panel(HJ, YL["hub"], "Hub-gene Jaccard", "fig_hub_jaccard", True, methods, outdir)
    one_panel(ML, YL["l2"], "Module L2", "fig_module_L2", False, methods, outdir)
    one_panel(DM, YL["mi"], "MI fidelity", "fig_MI_fidelity", False, methods, outdir)
    triptych(PANELS, "attention_advantage_triptych", methods, outdir)

    # ---- pairwise figures + gene network, stored under each baseline's dir ----
    for base in [m for m in methods if m != "Route2"]:
        bdir = f"{outdir}/{base}"; os.makedirs(bdir, exist_ok=True)
        pair_order = ["Route2", base]
        one_panel(HJ, YL["hub"], "Hub-gene Jaccard", "fig_hub_jaccard", True, pair_order, bdir)
        one_panel(ML, YL["l2"], "Module L2", "fig_module_L2", False, pair_order, bdir)
        one_panel(DM, YL["mi"], "MI fidelity", "fig_MI_fidelity", False, pair_order, bdir)
        triptych(PANELS, "attention_advantage_triptych", pair_order, bdir)
        po, do, pc, dc = gene_network_heatmap(base, bdir)
        print(f"  [{base}] singles + triptych + Gene_Network (DeconvSC PCC={po:.3f}/Dist={do:.1f} vs {base} PCC={pc:.3f}/Dist={dc:.1f}) -> {bdir}")
    print("[saved] single panels are 65 mm wide; triptychs are 180 mm wide")


if __name__ == "__main__":
    main()
