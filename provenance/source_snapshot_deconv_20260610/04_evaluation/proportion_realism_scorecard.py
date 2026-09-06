#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Proportion-realism scorecard: which solver predicts the most realistic cell proportions
on REAL GSE159585 COVID/normal bulk? (floor does NOT change p, so prophead±floor share one row.)
Triangulates 5 layers (see PROPORTION_REALISM_ANALYSIS.md). Reuses route2_lib.composition_metrics
and identity_realbulk.identity. Writes proportion_realism_scorecard.csv.
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

R10 = "/disk1/maijl/deconv/deconv_20260610"
sys.path.insert(0, f"{R10}/lib"); sys.path.insert(0, f"{R10}/03_inference")
import route2_lib as r2
from identity_realbulk import identity, COVID, NORMAL

U = f"{R10}/prophead/03_result_tables/unified"
CACHE = f"{U}/cache_gse159585_real.npz"
if not os.path.exists(CACHE):
    CACHE = "/disk1/maijl/deconv/deconv_20260605/03_result_tables/unified/cache_gse159585_real.npz"
METHODS = ["softmax", "prophead", "nnls_core"]           # prophead row = prophead±floor (same p)
CPS = 2000

def inv_simpson(P):                                       # mean per-sample effective #types
    P = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
    return float(np.mean(1.0 / np.clip((P ** 2).sum(1), 1e-12, None)))

# known dominant lung populations (for plausibility) and category matchers
DOMINANT = {"cap_1","cap_2","cap_arterial","cap_venous","cap_CX3CL1","cap_PB",
            "AT1","AT2","adventitial_fibroblast","alveolar_fibroblast","fibroblast_matrix",
            "macro_alveolar","macro_alveolar_SPP1","macrophage_interstitial"}
RARE = ["pDC", "migDC", "neuroendocrine"]
def is_immune(t):
    t=t.lower()
    return (t.startswith(("cd4","cd8","nk")) or t in {"b_cell","plasma","mast_cell","granulocyte"}
            or any(k in t for k in ["monocyte","macro","dc"]))
def is_fibro(t): return ("fibroblast" in t.lower()) or t.lower() in {"myofibroblast","fibromyocyte"}

# ---- cache: anchored core basis + real bulk ----
cache = {k: v for k, v in np.load(CACHE, allow_pickle=True).items()}
names = [str(x) for x in cache["names"]]; samples = [str(x) for x in cache["samples"]]
bc, yc = cache["basis_core"], cache["y_core"]; bulk_log = np.log1p(yc)
idx = {s: i for i, s in enumerate(samples)}
covid_i = [idx[s] for s in samples if s in COVID]; normal_i = [idx[s] for s in samples if s in NORMAL]

# ---- held-out pseudobulk GT ----
gt = pd.read_csv(f"{U}/pseudobulk/pb_gse159585_gt.csv", index_col=0)

rows = {}
for m in METHODS:
    # real-bulk predicted proportions, aligned to cache names/samples
    pr = pd.read_csv(f"{U}/proportions_gse159585_real_{m}.csv", index_col=0)
    if pr.shape[1] < pr.shape[0]: pr = pr.T
    P = pr.reindex(columns=names).fillna(0.0).reindex(samples).values
    P = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
    cm = r2.composition_metrics(P)
    recon = np.log1p(np.einsum("st,stg->sg", P, bc))
    id_all = identity(recon, bulk_log)
    id_cov = identity(recon[covid_i], bulk_log[covid_i])
    id_nor = identity(recon[normal_i], bulk_log[normal_i])
    # dominant-type plausibility
    mean_comp = pd.Series(P.mean(0), index=names)
    top8 = mean_comp.sort_values(ascending=False).head(8)
    dom_overlap = len(set(top8.index) & DOMINANT)
    top5 = "; ".join(f"{t}={v*100:.1f}%" for t, v in mean_comp.sort_values(ascending=False).head(5).items())
    # rare-real-type rendering (mean % and implied cells at CPS, no-floor)
    rare = {t: float(mean_comp.get(t, 0.0)) for t in RARE}
    rare_cells = {t: int(round(rare[t] * CPS)) for t in RARE}
    # COVID-normal biological direction (mean fraction per condition)
    Pc, Pn = P[covid_i].mean(0), P[normal_i].mean(0)
    imm = np.array([is_immune(t) for t in names]); fib = np.array([is_fibro(t) for t in names])
    at1 = np.array([t == "AT1" for t in names]); at2 = np.array([t == "AT2" for t in names])
    d_imm = (Pc[imm].sum() - Pn[imm].sum()) * 100
    d_fib = (Pc[fib].sum() - Pn[fib].sum()) * 100
    d_at1 = (Pc[at1].sum() - Pn[at1].sum()) * 100
    d_at2 = (Pc[at2].sum() - Pn[at2].sum()) * 100
    bio_ok = int(d_imm > 0) + int(d_fib > 0) + int(d_at1 < 0)   # expected: immune up, fibro up, AT1 down
    # held-out GT accuracy
    pp = pd.read_csv(f"{U}/proportions_gse159585_pb_{m}.csv", index_col=0)
    if pp.shape[1] < pp.shape[0]: pp = pp.T
    cols = sorted(set(gt.columns) | set(pp.columns)); idxs = sorted(set(gt.index) & set(pp.index))
    G = gt.reindex(index=idxs, columns=cols).fillna(0.0).values
    Q = pp.reindex(index=idxs, columns=cols).fillna(0.0).values
    overall_pcc = float(np.corrcoef(G.ravel(), Q.ravel())[0, 1])
    persample = float(np.mean([np.corrcoef(G[i], Q[i])[0, 1] for i in range(len(idxs))
                               if G[i].std() > 1e-9 and Q[i].std() > 1e-9]))
    rows[m] = dict(
        eff_realbulk=round(inv_simpson(P), 1), eff_compmetric=round(float(cm["eff_ntypes"]), 1),
        n_active_2pct=round(float(cm["n_active"]), 1),
        identity_all=round(id_all), identity_covid=round(id_cov), identity_normal=round(id_nor),
        dominant_top8_overlap=f"{dom_overlap}/8", top5=top5,
        rare_pDC_pct=round(rare["pDC"]*100, 3), rare_migDC_pct=round(rare["migDC"]*100, 3),
        rare_neuroend_pct=round(rare["neuroendocrine"]*100, 3),
        rare_cells_pDC=rare_cells["pDC"], rare_cells_migDC=rare_cells["migDC"],
        rare_cells_neuroend=rare_cells["neuroendocrine"],
        d_immune_pct=round(d_imm, 1), d_fibroblast_pct=round(d_fib, 1),
        d_AT1_pct=round(d_at1, 1), d_AT2_pct=round(d_at2, 1), bio_direction_ok=f"{bio_ok}/3",
        pb_overall_PCC=round(overall_pcc, 3), pb_persample_PCC=round(persample, 3),
        eff_pb_pred=round(inv_simpson(Q), 1), eff_pb_gt=round(inv_simpson(G), 1),
    )

df = pd.DataFrame(rows).T
out = f"{R10}/proportion_realism_scorecard.csv"
df.to_csv(out)
pd.set_option("display.width", 200, "display.max_columns", 40)
print(df.T.to_string())
print(f"\n[saved] {out}")
print(f"\nNOTE: floor (--min_cells_per_type) does NOT change p; prophead row = prophead+floor = prophead-no-floor for the PROPORTION.")
print(f"eff_pb_gt (held-out true effective #types) shown for reference.")
