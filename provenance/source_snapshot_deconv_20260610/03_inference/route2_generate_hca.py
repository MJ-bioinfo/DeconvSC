#!/usr/bin/env python
"""Generate HCA generated_data.h5ad with the NEW Route 2 approach end-to-end:
Route 2 solves its OWN cell-type proportions from each synthetic bulk (on the anchored
centroid+residual basis), then samples a controlled budget of pure-prior cells and adds the
residual — it does NOT borrow the Delta_Z pred_validation_cells counts. Written gzip-compressed.

Composition = Route2's own deconvolution; expression = decode(mu+eps*std) + alpha*delta(bulk,t).
Cell budget = CPS cells per sample (env CPS, default 200) distributed by the solved proportions.
"""
import sys, os, numpy as np, pandas as pd, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval_HCA_fold2 as R
import route2_lib as r2
import anndata as ad, scipy.sparse as sp, scanpy as sc

WORKROOT = os.environ.get("WORK_ROOT", "/disk1/maijl/deconv/deconv_20260605")  # deconv_20260610: outputs root (default = established working tree)
V3 = os.environ.get("AG_FOLD", "/disk1/maijl/deconv/cVAE/Simulation_HCA/result/HCA/v3_pureprior/fold_2")
GT_PATH = os.path.join(V3, "gt_validation_cells.h5ad")
OUT = os.environ.get("OUT", f"{WORKROOT}/HCA/generated_data.h5ad")
DEV = r2.device("cuda:1"); torch.manual_seed(0); np.random.seed(0)
CPS = int(os.environ.get("CPS", "200"))      # cells per bulk sample (budget)
DIV = float(os.environ.get("DIV", "0.6"))    # within-type diversity injection (zero-mean -> pseudobulk unchanged)
SMOOTH_K = int(os.environ.get("SMOOTH_K", "4"))   # avg K ref-cell deviations -> denoise


def solve_props(basis_sig, target_sig, valid_names, steps=300, temperature=0.5, lambda_entropy=0.1):
    logits = torch.zeros(basis_sig.shape[0], requires_grad=True, device=DEV)
    opt = torch.optim.Adam([logits], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        probs = F.softmax(logits / temperature, 0)
        pseudo = (probs.view(-1, 1) * basis_sig).sum(0)
        vx, vy = pseudo - pseudo.mean(), target_sig - target_sig.mean()
        pcc = (vx * vy).sum() / (vx.norm() * vy.norm() + 1e-8)
        loss = (1 - pcc) + lambda_entropy * (probs * torch.log(probs + 1e-8)).sum() + 0.01 * (logits ** 2).sum()
        loss.backward(); opt.step()
    p = F.softmax(logits / temperature, 0).detach().cpu().numpy(); p[p < 1e-4] = 0; p = p / (p.sum() + 1e-8)
    return {valid_names[i]: float(p[i]) for i in range(len(valid_names))}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--props_csv", default=None,
                    help="external proportions (synthetic-bulk id x cell types); overrides internal softmax solve")
    ap.add_argument("--min_cells_per_type", type=int, default=0,
                    help="floor cells/type for full expression coverage (0=off, default)")
    args = ap.parse_args()
    ext_props = None
    if args.props_csv:
        ext_props = pd.read_csv(args.props_csv, index_col=0); ext_props.index = ext_props.index.astype(str)
        print(f"[props] external proportions from {args.props_csv}: {ext_props.shape}")
    vae = r2.load_vae_decode(f"{V3}/scvae_best.pth", 25366, 23, input_size=3000, mid_hidden=512,
                             hidden_size_list=[2048, 1024, 512], device=DEV)
    pri = torch.load(f"{V3}/cell_type_mu_logvar_best.pt", map_location="cpu")
    ref = ad.read_h5ad(f"{V3}/reference_train.h5ad")
    labs = sorted(ref.obs["labels"].astype(str).unique())
    id2name = {i: n for i, n in enumerate(labs)}; name2id = {n: i for i, n in id2name.items()}
    genes = ref.var_names.astype(str); G = len(genes); gidx = {g: i for i, g in enumerate(genes)}
    cent_by_id = r2.decode_centroids(vae, pri, 23, DEV)
    cent_mat = torch.tensor(np.stack([cent_by_id[i] for i in range(23)]), device=DEV)
    sc.pp.highly_variable_genes(ref, n_top_genes=3000); core = np.where(ref.var["highly_variable"].values)[0]

    # train the anchored residual (same as route2_residual.py)
    X = ref.X.tocsr() if sp.issparse(ref.X) else sp.csr_matrix(ref.X)
    donors = ref.obs["donor_id"].astype(str).values; cells_t = ref.obs["labels"].astype(str).values
    dt_log, dt_lin = {}, {}
    for d in np.unique(donors):
        dm = donors == d
        for t in np.unique(cells_t[dm]):
            mk = dm & (cells_t == t)
            if mk.sum() < 5: continue
            ml = np.asarray(X[mk].mean(axis=0)).ravel()
            dt_log[(d, t)] = ml.astype(np.float32); dt_lin[(d, t)] = np.expm1(ml).astype(np.float32)
    train_donors = sorted({d for (d, _) in dt_log})
    # within-type diversity pools (X already in full gene space). dev has zero mean -> pseudobulk
    # (deconvolution) unchanged; restores realistic single-cell spread for UMAP (merges per-bulk blobs).
    ref_idx_div = {nm: np.where(cells_t == nm)[0] for nm in labs}
    cent_div = {nm: np.asarray(X[ix].mean(0)).ravel().astype(np.float32) for nm, ix in ref_idx_div.items() if len(ix) > 0}
    rng = np.random.default_rng(0)
    gt = ad.read_h5ad(GT_PATH); markers = R.derive_marker_genes(gt, "Cell_type", top_n=30)
    muni = sorted({g for v in markers.values() for g in v})
    gmask, w = r2.build_marker_core_mask(G, core, muni, gidx)
    net = r2.ResidualNet(len(core), 23, G, torch.tensor(gmask).to(DEV)).to(DEV)
    cmat_cpu = np.stack([cent_by_id[i] for i in range(23)])
    Xb, Xt, td = r2.build_examples(train_donors, labs, name2id, dt_lin, dt_log, core, cmat_cpu)
    r2.train_residual(net, Xb, Xt, td, torch.tensor(w), DEV); net.eval()

    sig_idx = torch.tensor([gidx[g] for g in muni if g in gidx], device=DEV)
    bulkv = pd.read_csv(f"{V3}/synthetic_bulks_cpm_linear.tsv", sep="\t", index_col=0).reindex(columns=genes).fillna(0.0)
    samples = list(bulkv.index.astype(str))
    print(f"[gen] {len(samples)} synthetic bulks, CPS={CPS} -> budget ~{len(samples)*CPS} cells; sig genes={len(sig_idx)}")

    generated, meta = [], []
    for si, s in enumerate(samples):
        blog = np.log1p(bulkv.loc[s].values).astype(np.float32)
        bcore = torch.tensor(blog[core], device=DEV).unsqueeze(0)
        with torch.no_grad():   # anchored basis is constant w.r.t. the proportion solve
            deltas = {t: net(bcore, torch.tensor([t], device=DEV)).reshape(-1) for t in range(23)}
            basis_lin = torch.stack([torch.expm1(torch.clamp(cent_mat[t] + deltas[t], min=0)) for t in range(23)])
            target_lin = torch.tensor(np.expm1(blog), device=DEV)
        if ext_props is not None and s in ext_props.index:          # external proportions
            row = ext_props.loc[s]; tot = sum(float(row.get(nm, 0.0)) for nm in labs)
            fr = {nm: float(row.get(nm, 0.0)) / tot for nm in labs} if tot > 0 else {nm: 0.0 for nm in labs}
        else:
            fr = solve_props(basis_lin[:, sig_idx], target_lin[sig_idx], labs)   # needs grad on logits
        with torch.no_grad():
            for t in range(23):
                count = max(int(round(fr.get(id2name[t], 0.0) * CPS)), args.min_cells_per_type)
                if count <= 0: continue
                mu = pri[t][0].to(DEV); std = torch.exp(0.5 * pri[t][1].to(DEV))
                z = mu + torch.randn(count, mu.shape[1], device=DEV) * std
                lab = F.one_hot(torch.tensor([t] * count, device=DEV), 23).float()
                base = vae.decode(z, lab)
                nm = id2name[t]
                if DIV > 0 and nm in cent_div and len(ref_idx_div[nm]) > 0:   # real within-type spread
                    sel = rng.choice(ref_idx_div[nm], count * SMOOTH_K, replace=True)
                    dv = X[sel].toarray().astype(np.float32).reshape(count, SMOOTH_K, -1) - cent_div[nm]
                    base = base + DIV * torch.tensor(dv.mean(1), device=DEV)
                cl = torch.clamp(base + deltas[t], min=0).cpu().numpy().astype(np.float32)
                generated.append(cl); meta.extend([(s, id2name[t])] * count)
        if (si + 1) % 20 == 0: print(f"  {si+1}/{len(samples)} samples, {sum(len(g) for g in generated)} cells")

    Xg = np.vstack(generated)
    obs = pd.DataFrame(meta, columns=["pseudo_bulk_id", "Cell_type"]); obs.index = [f"Cell_{i}" for i in range(len(obs))]
    a = ad.AnnData(X=Xg, obs=obs, var=pd.DataFrame(index=genes))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    a.write_h5ad(OUT, compression="gzip")
    print(f"[saved] {a.n_obs} cells x {a.n_vars} genes (gzip) -> {OUT}")


if __name__ == "__main__":
    main()
