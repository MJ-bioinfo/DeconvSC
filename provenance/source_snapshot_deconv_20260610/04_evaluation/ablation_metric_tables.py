#!/usr/bin/env python
"""(Q1) Check whether the ablation_compare 'DeconvSC' input (20260605 generated_data) has the same
cell-type composition as the 0610 prophead/nnls strategy generations (the ablation metrics are
computed across cells, so composition matters). (Q2) From attention_advantage_dists.npz, tabulate
each model's actual Hub-gene Jaccard / Module-L2 / MI-fidelity values + the DeconvSC-vs-baseline
differences and Wilcoxon/Mann-Whitney p-values. Writes tables into the ablation_compare path."""
import os, numpy as np, pandas as pd, anndata as ad, scipy.stats as ss
AC = os.environ.get("ABLATION_DIR", "/disk1/maijl/deconv/deconv_20260605/GSE141115/ablation_compare")

# ---------- Q1: composition of the three GSE141115 generated_data.h5ad ----------
GENS = {"20260605 (ablation DeconvSC input)": "/disk1/maijl/deconv/deconv_20260605/GSE141115/generated_data.h5ad",
        "0610/prophead": "/disk1/maijl/deconv/deconv_20260610/prophead/GSE141115/generated_data.h5ad",
        "0610/nnls":     "/disk1/maijl/deconv/deconv_20260610/nnls/GSE141115/generated_data.h5ad"}
print("===== Q1: generated_data.h5ad composition (GSE141115) =====")
comp = {}
for name, p in GENS.items():
    if not os.path.exists(p): print(f"  {name}: MISSING"); continue
    a = ad.read_h5ad(p, backed="r"); vc = a.obs["Cell_type"].astype(str).value_counts()
    comp[name] = vc
    print(f"  {name}: n_cells={a.n_obs}, n_types={len(vc)}, top3={dict(vc.head(3))}")
names = list(comp)
if len(names) >= 2:
    ref = comp[names[0]]
    for other in names[1:]:
        common = ref.index.union(comp[other].index)
        same = (ref.reindex(common, fill_value=0) == comp[other].reindex(common, fill_value=0)).all()
        print(f"  composition identical [{names[0]}] vs [{other}]? {bool(same)}")

# ---------- Q2: per-model metric values + differences from the distributions ----------
d = np.load(f"{AC}/attention_advantage_dists.npz", allow_pickle=True)
METRICS = [("hub_jaccard", "hubJ", "higher=better (similarity to real)"),
           ("module_l2", "modL2", "lower=better (distance from real)"),
           ("mi_fidelity_error", "dMI", "lower=better |MI_gen-MI_real|")]
DISP = {"Route2": "DeconvSC", "BaseVAE": "BaseVAE (ablation)", "DISSECT": "DISSECT"}

per = []
for met, pre, note in METRICS:
    for m in ["Route2", "BaseVAE", "DISSECT"]:
        v = d[f"{pre}_{m}"]; v = v[np.isfinite(v)]
        per.append(dict(metric=met, direction=note, method=DISP[m], N=len(v),
                        median=round(float(np.median(v)), 6), mean=round(float(v.mean()), 6),
                        std=round(float(v.std(ddof=1)), 6), q25=round(float(np.percentile(v, 25)), 6),
                        q75=round(float(np.percentile(v, 75)), 6), min=round(float(v.min()), 6),
                        max=round(float(v.max()), 6)))
per_df = pd.DataFrame(per)
per_df.to_csv(f"{AC}/ablation_metric_values.csv", index=False)

# Significance computed exactly as the triptych bars in 05_calculate_attention_ablation.py:
# one-sided paired Wilcoxon in the DeconvSC-better direction, truncated to the shared min length.
ALT = {"hub_jaccard": "greater", "module_l2": "less", "mi_fidelity_error": "less"}  # DeconvSC-better direction
def _star(p): return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
def _wp(a, b, alt):
    n = min(len(a), len(b))
    try: return float(ss.wilcoxon(a[:n], b[:n], alternative=alt)[1])
    except Exception: return float("nan")
diff = []
for met, pre, note in METRICS:
    r, b, di = (d[f"{pre}_{m}"][np.isfinite(d[f"{pre}_{m}"])] for m in ["Route2", "BaseVAE", "DISSECT"])
    alt = ALT[met]; pb = _wp(r, b, alt); pdi = _wp(r, di, alt)
    diff.append(dict(metric=met, direction=note,
                     DeconvSC_median=round(float(np.median(r)), 6), BaseVAE_median=round(float(np.median(b)), 6),
                     DISSECT_median=round(float(np.median(di)), 6),
                     diff_DeconvSC_minus_BaseVAE=round(float(np.median(r) - np.median(b)), 6),
                     diff_DeconvSC_minus_DISSECT=round(float(np.median(r) - np.median(di)), 6),
                     p_vs_BaseVAE=f"{pb:.2e}", sig_vs_BaseVAE=_star(pb),
                     p_vs_DISSECT=f"{pdi:.2e}", sig_vs_DISSECT=_star(pdi)))
diff_df = pd.DataFrame(diff)
diff_df.to_csv(f"{AC}/ablation_metric_differences.csv", index=False)

with open(f"{AC}/ablation_metric_values.md", "w") as fh:
    fh.write("# GSE141115 attention-ablation: per-model metric values + DeconvSC-vs-baseline differences\n\n")
    fh.write("Values from `attention_advantage_dists.npz` (DeconvSC=Route2 full attention model; "
             "BaseVAE=attention-ablated; DISSECT=baseline). Subsampled to N=9,615 cells × 5 seeds; "
             "top-500 HVG (Jaccard/L2) or top-200 HVG / 19,900 pairs (MI).\n\n")
    fh.write("## Per-model values (median [IQR], mean±sd)\n\n")
    fh.write("| metric | model | median | mean | sd | IQR (q25–q75) | N |\n|---|---|---|---|---|---|---|\n")
    for _, r in per_df.iterrows():
        fh.write(f"| {r.metric} | {r.method} | {r['median']:.4f} | {r['mean']:.4f} | {r['std']:.4f} "
                 f"| {r.q25:.4f}–{r.q75:.4f} | {r.N} |\n")
    fh.write("\n## DeconvSC − baseline differences (median) + significance\n\n")
    fh.write("| metric | DeconvSC | BaseVAE | DISSECT | Δ(−BaseVAE) | Δ(−DISSECT) | p / sig vs BaseVAE | p / sig vs DISSECT |\n")
    fh.write("|---|---|---|---|---|---|---|---|\n")
    for _, r in diff_df.iterrows():
        fh.write(f"| {r.metric} | {r.DeconvSC_median:.4f} | {r.BaseVAE_median:.4f} | {r.DISSECT_median:.4f} "
                 f"| {r.diff_DeconvSC_minus_BaseVAE:+.4f} | {r.diff_DeconvSC_minus_DISSECT:+.4f} "
                 f"| {r.p_vs_BaseVAE} ({r.sig_vs_BaseVAE}) | {r.p_vs_DISSECT} ({r.sig_vs_DISSECT}) |\n")
    fh.write("\nSignificance: one-sided paired Wilcoxon in the DeconvSC-better direction (Jaccard: greater; "
             "L2/MI: less), truncated to the shared min N — the same test as the triptych significance bars "
             "in 05_calculate_attention_ablation.py. medians/means/differences computed from "
             "attention_advantage_dists.npz.\n")

print("\n===== Q2: per-model metric values =====")
print(per_df.to_string(index=False))
print("\n===== Q2: differences + p-values =====")
print(diff_df.to_string(index=False))
print(f"\n[saved] {AC}/ablation_metric_values.csv, ablation_metric_differences.csv, ablation_metric_values.md")
