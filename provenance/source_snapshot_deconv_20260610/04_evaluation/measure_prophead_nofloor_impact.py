#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXPERIMENT: does prophead WITHOUT --min_cells_per_type hurt ssGSEA / CellChat / Monocle2?

Measures the three already-computed downstream runs head-to-head:
  softmax  = deconv_20260605/downstream + GSE226077        (published, no floor, near-full coverage)
  prophead = deconv_20260610/prophead/downstream + GSE226077   (MIN="" -> NO floor)   <-- the case asked about
  nnls     = deconv_20260610/nnls/downstream + GSE226077       (--min_cells_per_type 20)

Outputs a console report:
  (1) COVID/normal cell-type COVERAGE in the generated h5ads (root cause)
  (2) ssGSEA: # testable cell types + n_cells distribution (from ssgsea_avg_meta.csv)
  (3) CellChat: # nodes (cell types) + # pathways (from comparison_df.csv / covid_pathway_counts.csv)
  (4) Monocle2/iMGL: cell-type coverage in the iMGL h5ad
Read-only; uses backed='r' so only obs is loaded.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import anndata as ad

R5 = "/disk1/maijl/deconv/deconv_20260605"
R10 = "/disk1/maijl/deconv/deconv_20260610"
SCHEMES = {
    "softmax(605,nofloor)":  f"{R5}/downstream",
    "prophead(610,NOfloor)": f"{R10}/prophead/downstream",
    "nnls(610,floor=20)":    f"{R10}/nnls/downstream",
}
IMGL = {
    "softmax(605,nofloor)":  f"{R5}/GSE226077/generated_data.h5ad",
    "prophead(610,NOfloor)": f"{R10}/prophead/GSE226077/generated_data.h5ad",
    "nnls(610,floor=20)":    f"{R10}/nnls/GSE226077/generated_data.h5ad",
}

CT_CANDS = ["cell type", "Cell_type", "cell_type", "CellType", "celltype", "labels", "Cell type"]
SA_CANDS = ["Sample", "sample", "sample_id", "SampleID", "donor", "orig.ident"]

def pick(cols, cands):
    for c in cands:
        if c in cols:
            return c
    return None

def coverage(h5, label):
    a = ad.read_h5ad(h5, backed="r")
    cols = list(a.obs.columns)
    ctc = pick(cols, CT_CANDS); sac = pick(cols, SA_CANDS)
    obs = a.obs[[c for c in (ctc, sac) if c]].copy()
    ct = obs[ctc].astype(str)
    res = {"n_cells": a.n_obs, "ct_col": ctc, "sa_col": sac,
           "types_union": sorted(ct.unique().tolist())}
    if sac:
        per = obs.groupby(sac, observed=True)[ctc].nunique()
        res["n_samples"] = int(per.shape[0])
        res["per_sample_mean"] = float(per.mean())
        res["per_sample_min"] = int(per.min())
        res["per_sample_max"] = int(per.max())
        # per-(sample,celltype) cell counts -> floor signature
        cnt = obs.groupby([sac, ctc], observed=True).size()
        res["combo_count"] = int(cnt.shape[0])
        res["combo_mincells"] = int(cnt.min())
        res["frac_combos_le5"] = float((cnt <= 5).mean())
        res["frac_combos_eq_floor20"] = float(((cnt >= 18) & (cnt <= 22)).mean())
    a.file.close()
    return res

print("=" * 90)
print("(1) COVID + NORMAL cell-type COVERAGE in generated h5ads")
print("=" * 90)
cov = {}
full_union = None
for lab, root in SCHEMES.items():
    cov[lab] = {}
    for cond in ["covid", "normal_18_24"]:
        h5 = f"{root}/{cond}/generated_data_{'covid' if cond=='covid' else 'normal_18_24'}.h5ad"
        if not os.path.exists(h5):
            print(f"[MISSING] {h5}"); continue
        cov[lab][cond] = coverage(h5, f"{lab}/{cond}")
# define "full" set = softmax union (covid ∪ normal)
sm = "softmax(605,nofloor)"
full_union = sorted(set(cov[sm]["covid"]["types_union"]) | set(cov[sm]["normal_18_24"]["types_union"]))
NTOT = len(full_union)
print(f"\nReference full cell-type set (softmax union, covid∪normal) = {NTOT} types\n")
hdr = f"{'scheme':<24}{'cond':<14}{'cells':>9}{'samp':>5}{'union':>7}{'/'}{'tot':<4}{'per-samp(mean/min/max)':>24}{'combos':>8}{'min#':>6}{'%<=5':>7}{'%~20':>7}"
print(hdr); print("-" * len(hdr))
for lab in SCHEMES:
    for cond in ["covid", "normal_18_24"]:
        d = cov[lab].get(cond)
        if not d: continue
        present = len(set(d["types_union"]) & set(full_union))
        ps = f"{d.get('per_sample_mean',float('nan')):.1f}/{d.get('per_sample_min','?')}/{d.get('per_sample_max','?')}"
        print(f"{lab:<24}{cond:<14}{d['n_cells']:>9}{d.get('n_samples','?'):>5}{present:>7}/{NTOT:<4}{ps:>24}"
              f"{d.get('combo_count','?'):>8}{d.get('combo_mincells','?'):>6}{d.get('frac_combos_le5',0)*100:>6.0f}%{d.get('frac_combos_eq_floor20',0)*100:>6.0f}%")

# which types prophead drops vs softmax (union covid∪normal)
print("\n-- cell types DROPPED by each scheme (absent from union, present in softmax) --")
for lab in SCHEMES:
    u = set(cov[lab]["covid"]["types_union"]) | set(cov[lab]["normal_18_24"]["types_union"])
    dropped = sorted(set(full_union) - u)
    print(f"\n{lab}: present {len(set(full_union)&u)}/{NTOT}, DROPPED {len(dropped)}:")
    print("   " + ("; ".join(dropped) if dropped else "(none)"))

print("\n" + "=" * 90)
print("(2) ssGSEA: # testable cell types (from ssgsea_avg_meta.csv)")
print("=" * 90)
print(f"{'scheme':<24}{'rows(sample,ct)':>16}{'uniq_celltypes':>16}{'n_cells med/min':>18}")
print("-" * 74)
for lab, root in SCHEMES.items():
    f = f"{root}/ssGSEA/ssgsea_avg_meta.csv"
    if not os.path.exists(f):
        print(f"{lab:<24}  [missing {f}]"); continue
    m = pd.read_csv(f)
    ctcol = "CellType" if "CellType" in m.columns else pick(list(m.columns), CT_CANDS)
    ncol = "n_cells" if "n_cells" in m.columns else None
    nuniq = m[ctcol].nunique()
    nmed = f"{m[ncol].median():.0f}/{m[ncol].min():.0f}" if ncol else "?"
    print(f"{lab:<24}{len(m):>16}{nuniq:>16}{nmed:>18}")

print("\n" + "=" * 90)
print("(3) CellChat: # nodes (cell types) + # pathways")
print("=" * 90)
print(f"{'scheme':<24}{'comparison_df rows':>20}{'covid pathways':>16}")
print("-" * 60)
for lab, root in SCHEMES.items():
    cdf = f"{root}/cellchat/comparison_df.csv"
    pcf = f"{root}/cellchat/covid_pathway_counts.csv"
    nnode = "?"; npath = "?"
    if os.path.exists(cdf):
        c = pd.read_csv(cdf); nnode = c.shape[0]
    if os.path.exists(pcf):
        p = pd.read_csv(pcf); npath = p.shape[0]
    print(f"{lab:<24}{str(nnode):>20}{str(npath):>16}")
# also dump comparison_df cell-type lists to compare nodes
print("\n-- CellChat comparison_df cell types per scheme --")
for lab, root in SCHEMES.items():
    cdf = f"{root}/cellchat/comparison_df.csv"
    if os.path.exists(cdf):
        c = pd.read_csv(cdf)
        ctcol = c.columns[0]
        print(f"\n{lab}: {c.shape[0]} nodes; cols={list(c.columns)}")
        print("   " + "; ".join(map(str, c[ctcol].tolist())))

print("\n" + "=" * 90)
print("(4) Monocle2 / iMGL: cell-type coverage in iMGL generated h5ad")
print("=" * 90)
imgl_full = None
imgl_cov = {}
for lab, h5 in IMGL.items():
    if not os.path.exists(h5):
        print(f"[MISSING] {h5}"); continue
    imgl_cov[lab] = coverage(h5, f"{lab}/iMGL")
imgl_full = sorted(imgl_cov[sm]["types_union"]) if sm in imgl_cov else []
print(f"\niMGL reference full set (softmax) = {len(imgl_full)} types: {imgl_full}\n")
print(f"{'scheme':<24}{'cells':>9}{'samp':>5}{'union':>7}{'per-samp(mean/min/max)':>24}{'min#cells':>10}")
print("-" * 80)
for lab in IMGL:
    d = imgl_cov.get(lab)
    if not d: continue
    present = len(set(d["types_union"]) & set(imgl_full))
    ps = f"{d.get('per_sample_mean',float('nan')):.1f}/{d.get('per_sample_min','?')}/{d.get('per_sample_max','?')}"
    print(f"{lab:<24}{d['n_cells']:>9}{d.get('n_samples','?'):>5}{present:>5}/{len(imgl_full):<3}{ps:>24}{d.get('combo_mincells','?'):>10}")
print("\n-- iMGL cell types DROPPED per scheme (vs softmax) --")
for lab in IMGL:
    if lab not in imgl_cov: continue
    u = set(imgl_cov[lab]["types_union"])
    dropped = sorted(set(imgl_full) - u)
    print(f"{lab}: DROPPED {len(dropped)}: {('; '.join(dropped) if dropped else '(none)')}")
print("\n[done]")
