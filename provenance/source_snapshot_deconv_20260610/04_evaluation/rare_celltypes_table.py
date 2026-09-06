#!/usr/bin/env python
"""Per-dataset RARE cell-type tables: cell types present in the real ground truth but DROPPED
(0 generated cells) by the prophead (no-floor) scheme, with their real count / fraction / rarity
rank. Writes one CSV per dataset + a combined markdown into prophead_nofloor_result_figures/."""
import os, sys, anndata as ad, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

ROOT = "/disk1/maijl/deconv/deconv_20260610/prophead"   # prophead no-floor scheme (this analysis is prophead-specific)
OUTD = "/disk1/maijl/deconv/deconv_20260610/prophead_nofloor_result_figures"
DS = {
    "HCA_fold2": dict(real="/disk1/maijl/deconv/cVAE/Simulation_HCA/result/HCA/v3_pureprior/fold_2/gt_validation_cells.h5ad",
                      real_col="Cell_type", gen=f"{ROOT}/HCA/generated_data.h5ad"),
    "GSE141115": dict(real="/disk1/maijl/deconv/data/GSE141115/GSE141115_sc_raw_counts_test.h5ad",
                      real_col="cell type", gen=f"{ROOT}/GSE141115/generated_data.h5ad"),
    "GSE159585": dict(real="/disk1/maijl/deconv/data/GSE159585/GSE159585_testset_level2celltype.h5ad",
                      real_col="cell type", gen=f"{ROOT}/GSE159585/generated_data.h5ad"),
}

md = ["# prophead (no-floor) dropped (rare) cell types per dataset\n",
      "For each dataset: cell types present in the real ground truth but generating **0 cells** under "
      "the prophead no-floor scheme (their predicted proportion rounds to 0 at the fixed cell budget). "
      "`rarity rank` = position by descending real abundance (1 = most abundant, N = rarest).\n"]
for ds, c in DS.items():
    real = ad.read_h5ad(c["real"], backed="r"); rc = real.obs[c["real_col"]].astype(str).value_counts()
    n_real_cells = int(rc.sum()); n_types = len(rc)
    gen = ad.read_h5ad(c["gen"], backed="r"); gset = set(gen.obs["Cell_type"].astype(str).unique())
    rank = {t: i + 1 for i, t in enumerate(rc.index)}            # descending-abundance rank
    rows = []
    for t in rc.index:
        if t not in gset:                                        # dropped by prophead
            rows.append((t, int(rc[t]), round(rc[t] / n_real_cells * 100, 3), f"{rank[t]}/{n_types}"))
    df = pd.DataFrame(rows, columns=["cell_type", "real_cells", "real_fraction_pct", "rarity_rank"])
    df.to_csv(f"{OUTD}/rare_celltypes_{ds}.csv", index=False)
    md.append(f"\n## {ds}  (real: {n_real_cells} cells, {n_types} types; dropped {len(df)} types)\n")
    md.append("| cell type | real cells | real fraction | rarity rank |")
    md.append("|---|---|---|---|")
    for _, r in df.iterrows():
        md.append(f"| {r.cell_type} | {r.real_cells} | {r.real_fraction_pct:.3f}% | {r.rarity_rank} |")
    print(f"[{ds}] dropped {len(df)} / {n_types} types -> rare_celltypes_{ds}.csv")
    print(df.to_string(index=False)); print()
open(f"{OUTD}/rare_celltypes.md", "w").write("\n".join(md) + "\n")
print(f"[saved] {OUTD}/rare_celltypes_{{HCA_fold2,GSE141115,GSE159585}}.csv + rare_celltypes.md")
