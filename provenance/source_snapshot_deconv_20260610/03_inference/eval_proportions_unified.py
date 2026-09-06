#!/usr/bin/env python
"""Evaluate unified Route2 proportion methods against ground truth + composition realism.

Reads proportions_<dataset>[_<tag>]_<method>.csv emitted by proportions_unified.py, scores
each vs GT with metrics() (sample/celltype/overall PCC, RMSE, MAE) from
eval_proportion_variants_hca.py, AND with composition-realism metrics (n_active@2%, eff #types,
JSD to pi_ref) so over-uniform/over-sparse degeneracies are visible alongside accuracy.
For hca_fold2 it also scores the competitor software (BayesPrism/CIBERSORTx/DISSECT/TAPE).

Outputs <OUTD>/UNIFIED_proportion_<dataset>[_<tag>].csv + .png .

Usage:
  python eval_proportions_unified.py --dataset hca_fold2
  python eval_proportions_unified.py --dataset gse159585 --tag pb --gt <gt_props.csv>
"""
import os, glob, argparse, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "lib")); import plot_style  # noqa: F401  (Arial font applied on import + plot_style.save/emf_dir for EMF)
import matplotlib.pyplot as plt
import eval_proportion_variants_hca as EV
import route2_lib as r2

OUTD = os.environ.get("WORK_ROOT", "/disk1/maijl/deconv/deconv_20260605") + "/03_result_tables/unified"
GT_DEFAULT = {
    "hca_fold2": "/disk1/maijl/deconv/cVAE/Simulation_HCA/result/HCA/v3_pureprior/fold_2/gt_actual_props_fold_2.csv",
}
METHOD_ORDER = ["softmax", "nnls_core", "nnls_ridge", "dirichlet_map", "prophead", "prophead_ref"]


def load_gt(path):
    gt = pd.read_csv(path, index_col=0); gt.index = gt.index.astype(str)
    return gt


def score(pred, gt, ref_mean):
    m = EV.metrics(pred, gt)
    cm = r2.composition_metrics(pred.values, ref_mean=ref_mean)
    m.update({"n_active@2%": cm["n_active"], "eff_ntypes": cm["eff_ntypes"],
              "max_prop": cm["max_prop"], "jsd_to_ref": cm["jsd_to_ref"]})
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--gt", default=None, help="GT props csv (defaults to GT_DEFAULT[dataset])")
    a = ap.parse_args()
    gt = load_gt(a.gt or GT_DEFAULT[a.dataset])
    suffix = f"_{a.tag}" if a.tag else ""
    pi_ref = pd.read_csv(f"{OUTD}/pi_ref_{a.dataset}.csv", index_col=0)["mean"]

    rows = {}
    for m in METHOD_ORDER:
        f = f"{OUTD}/proportions_{a.dataset}{suffix}_{m}.csv"
        if not os.path.exists(f):
            continue
        pred = pd.read_csv(f, index_col=0); pred.index = pred.index.astype(str)
        rm = pi_ref.reindex(pred.columns).fillna(0.0).values
        rows[m] = score(pred, gt, rm)
    # GT's own realism (reference line) + competitors (hca only)
    rows["[GT]"] = {**EV.metrics(gt, gt),
                    **{f"{k2}": v for k2, v in zip(
                        ["n_active@2%", "eff_ntypes", "max_prop", "jsd_to_ref"],
                        [r2.composition_metrics(gt.values, ref_mean=pi_ref.reindex(gt.columns).fillna(0).values)[x]
                         for x in ["n_active", "eff_ntypes", "max_prop", "jsd_to_ref"]])}}
    if a.dataset == "hca_fold2" and not a.tag:
        for name, d in EV.load_competitors().items():
            rm = pi_ref.reindex(d.columns).fillna(0.0).values
            rows[f"[ref] {name}"] = score(d, gt, rm)

    cols = ["n_celltypes", "sample_PCC_mean", "celltype_PCC_mean", "overall_PCC", "RMSE", "MAE",
            "n_active@2%", "eff_ntypes", "max_prop", "jsd_to_ref"]
    tbl = pd.DataFrame({k: {c: rows[k].get(c, np.nan) for c in cols} for k in rows}).T[cols]
    for c in cols[1:]:
        tbl[c] = tbl[c].astype(float).round(4)
    tbl.index.name = "method"
    out_csv = f"{OUTD}/UNIFIED_proportion_{a.dataset}{suffix}.csv"
    tbl.to_csv(out_csv); print(tbl.to_string()); print(f"\n[saved] {out_csv}")

    # figure: PCC bars (left) + composition realism (right), GT as reference
    meth = [k for k in rows if not k.startswith("[")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(meth)); wb = 0.27
    for off, key, col, lab in [(-wb, "sample_PCC_mean", "#5DADE2", "per-sample PCC"),
                               (0, "celltype_PCC_mean", "#48C9B0", "per-cell-type PCC"),
                               (wb, "overall_PCC", "#C0392B", "overall PCC")]:
        ax[0].bar(x + off, [rows[k][key] for k in meth], wb, label=lab, edgecolor="black", lw=0.4, color=col)
    for name, c, ls in [("[ref] BayesPrism", "#2C3E50", ":"), ("[ref] CIBERSORTx", "#16A085", "--")]:
        if name in rows:
            ax[0].axhline(rows[name]["overall_PCC"], color=c, ls=ls, lw=1.2,
                          label=f"{name.split()[-1]} overall ({rows[name]['overall_PCC']:.2f})")
    ax[0].set_xticks(x); ax[0].set_xticklabels(meth, rotation=20, ha="right", fontsize=9)
    ax[0].set_ylabel("Pearson correlation"); ax[0].set_ylim(0, 1.0); ax[0].grid(axis="y", alpha=0.3)
    ax[0].set_title(f"{a.dataset}{suffix} — proportion accuracy vs GT", fontweight="bold", fontsize=11)
    ax[0].legend(fontsize=7, ncol=2)

    ax[1].bar(x, [rows[k]["n_active@2%"] for k in meth], 0.55, color="#AF7AC5", edgecolor="black", lw=0.4)
    ax[1].axhline(rows["[GT]"]["n_active@2%"], color="black", ls="--", lw=1.4,
                  label=f"GT n_active@2% ({rows['[GT]']['n_active@2%']:.1f})")
    ax[1].set_xticks(x); ax[1].set_xticklabels(meth, rotation=20, ha="right", fontsize=9)
    ax[1].set_ylabel("mean # active types (>2%)")
    ax[1].set_title("composition realism (uniform↑ / sparse↓)", fontweight="bold", fontsize=11)
    ax[1].grid(axis="y", alpha=0.3); ax[1].legend(fontsize=8)
    for j, k in enumerate(meth):
        ax[1].text(x[j], rows[k]["n_active@2%"] + 0.3, f"{rows[k]['n_active@2%']:.0f}", ha="center", fontsize=7)
    plt.tight_layout()
    base = f"{OUTD}/UNIFIED_proportion_{a.dataset}{suffix}"
    plot_style.save(fig, base, dpi=160)   # png/pdf/svg + emf, Arial
    plt.close(); print(f"[saved] {base}.{{png,pdf,svg,emf}}")


if __name__ == "__main__":
    main()
