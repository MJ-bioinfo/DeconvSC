#!/usr/bin/env python
"""Ablation: does removing prophead's noise augmentation (noise_std 0.2 -> 0.0) change cell-type
proportion PCC on the idealized HCA fold2 simulation (in-domain synthetic bulks)?
Reuses the saved basis cache (no GPU VAE rebuild). Multi-seed to isolate the noise effect."""
import os, sys, numpy as np, pandas as pd, torch
from scipy.stats import pearsonr
sys.path.insert(0, "/disk1/maijl/deconv/deconv_20260610/lib")
import route2_lib as r2

DEV = os.environ.get("ABL_DEV", "cpu")
UNI = "/disk1/maijl/deconv/deconv_20260610/prophead/03_result_tables/unified"
CACHE = f"{UNI}/cache_hca_fold2.npz"
BASELINE_CSV = f"{UNI}/proportions_hca_fold2_prophead.csv"
GT = "/disk1/maijl/deconv/cVAE/Simulation_HCA/result/HCA/v3_pureprior/fold_2/gt_actual_props_fold_2.csv"
SEEDS = [0, 1, 2, 3, 4]

c = {k: v for k, v in np.load(CACHE, allow_pickle=True).items()}
names = list(c["names"]); samples = [str(s) for s in c["samples"]]
Xb = torch.tensor(c["Xb_h"]); P = torch.tensor(c["P_h"])
bc, yc, ys = c["basis_core"], c["y_core"], c["y_full_sum"]
feat = np.log1p(yc / ys[:, None] * 1e4).astype(np.float32)      # head input (S, n_core), scale-invariant
n_core, n_types = bc.shape[2], c["mean"].shape[0]
gt = pd.read_csv(GT, index_col=0); gt.index = gt.index.astype(str)
print(f"cache: {len(samples)} samples x {n_types} types, n_core={n_core}, Xb={tuple(Xb.shape)}; DEV={DEV}\n")


def metrics(pred):
    samp = [s for s in gt.index if s in pred.index]; cts = [ct for ct in gt.columns if ct in pred.columns]
    Pp = pred.loc[samp, cts].astype(float).values; Gv = gt.loc[samp, cts].astype(float).values
    ps = [pearsonr(Pp[i], Gv[i])[0] for i in range(len(samp)) if np.std(Pp[i]) > 0 and np.std(Gv[i]) > 0]
    pc = {cts[j]: pearsonr(Pp[:, j], Gv[:, j])[0] for j in range(len(cts)) if np.std(Pp[:, j]) > 0 and np.std(Gv[:, j]) > 0}
    return dict(nT=len(cts), sample_PCC=float(np.mean(ps)), celltype_PCC=float(np.mean(list(pc.values()))),
                overall_PCC=float(pearsonr(Pp.ravel(), Gv.ravel())[0]),
                RMSE=float(np.sqrt(np.mean((Pp - Gv) ** 2))), MAE=float(np.mean(np.abs(Pp - Gv))), pc=pc)


def run(noise, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    head = r2.PropHead(n_core, n_types).to(DEV)
    r2.train_prop_head(head, Xb, P, DEV, epochs=120, noise_std=noise, log=lambda *a: None)
    with torch.no_grad():
        ph = head(torch.tensor(feat, device=DEV)).cpu().numpy()
    return pd.DataFrame(ph, index=samples, columns=names)


base = pd.read_csv(BASELINE_CSV, index_col=0); base.index = base.index.astype(str)
mb = metrics(base)
print("LANDED prophead CSV (noise=0.2, single run):",
      {k: round(mb[k], 4) for k in ["overall_PCC", "celltype_PCC", "sample_PCC", "RMSE", "MAE"]}, "\n")

summary, pc_by_noise = {}, {}
for noise in [0.2, 0.0]:
    res = [metrics(run(noise, s)) for s in SEEDS]
    summary[noise] = {k: np.array([r[k] for r in res]) for k in ["overall_PCC", "celltype_PCC", "sample_PCC", "RMSE", "MAE"]}
    cts = res[0]["pc"].keys()
    pc_by_noise[noise] = {ct: np.mean([r["pc"].get(ct, np.nan) for r in res]) for ct in cts}
    print(f"=== noise_std={noise}  ({len(SEEDS)} seeds) ===")
    for k, v in summary[noise].items():
        print(f"  {k:>12} = {v.mean():.4f} ± {v.std():.4f}")
    print()

print("=== delta (noise=0.0 minus noise=0.2), mean over seeds ===")
for k in ["overall_PCC", "celltype_PCC", "sample_PCC", "RMSE", "MAE"]:
    d = summary[0.0][k].mean() - summary[0.2][k].mean()
    print(f"  {k:>12}: {d:+.4f}")

print("\n=== per-cell-type PCC, fibroblast/stromal types (mean over seeds) ===")
print(f"  {'cell_type':36} {'noise0.2':>9} {'noise0.0':>9} {'delta':>8}")
for ct in pc_by_noise[0.2]:
    if any(t in ct.lower() for t in ["fibro", "myofib", "stromal", "peri"]):
        a, b = pc_by_noise[0.2][ct], pc_by_noise[0.0][ct]
        print(f"  {ct:36} {a:9.3f} {b:9.3f} {b-a:+8.3f}")
