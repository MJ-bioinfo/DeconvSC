#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""How much of the PUBLISHED COVID ssGSEA result (605 softmax, 1242 significant
cell-type x pathway associations) is carried by the cell types that prophead-no-floor DROPS?
If those types are absent, those associations simply cannot be produced."""
import pandas as pd

DROPPED = ["CD4_Th1", "CD8_gamma_delta", "large_vessel", "mesothelial", "migDC",
           "neuroendocrine", "pDC", "proliferating_endothelial",
           "proliferating_epithelial", "proliferating_lymphoid"]

# prep_ssgsea_avg.py rewrites "_"->"-" in cell-type names, so normalise before matching.
DROPPED_N = [d.replace("_", "-") for d in DROPPED]
t = pd.read_csv("/disk1/maijl/deconv/deconv_20260605/downstream/ssGSEA/limma_tidy.csv")
t["sig"] = t["sig"].astype(str).str.upper().isin(["TRUE", "1"])
sig = t[t["sig"]]
lost = sig[sig["CellType"].isin(DROPPED_N)]

print(f"limma_tidy rows                 : {len(t)}")
print(f"significant assoc (sig==TRUE)   : {len(sig)}")
print(f"  by direction                  : {sig['direction'].value_counts().to_dict()}")
print(f"distinct cell types in result   : {t['CellType'].nunique()}")
print()
print(f"prophead-DROPPED types present in result : {sorted(set(DROPPED_N) & set(t['CellType'].unique()))}")
print(f"significant assoc carried by DROPPED types : {len(lost)}  ({100*len(lost)/len(sig):.1f}% of {len(sig)})")
print(f"  by direction                  : {lost['direction'].value_counts().to_dict()}")
print()
print("significant associations LOST per dropped cell type:")
g = lost.groupby("CellType").size().sort_values(ascending=False)
for ct, n in g.items():
    print(f"  {ct:<28} {n}")
