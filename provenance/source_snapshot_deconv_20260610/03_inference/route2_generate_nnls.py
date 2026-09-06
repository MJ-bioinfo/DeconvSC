#!/usr/bin/env python
"""Route2 (prior-anchored) generation with PLUGGABLE proportions — latest inference script.

Route2 = frozen per-type prior centroid  +  bounded bulk-conditioned residual  +  proportions.
  ê_t(b_s) = ψ(μ_t) + α·δθ(b_s^core, t)                       # anchored per-celltype profile (log space)
  cells    = ReLU( decode(z~N(μ_t,σ_t)) + α·δθ ) ,  count_t = round(p_{s,t}·CPS)

⚠️ HONEST NOTE on proportions (verified 2026-06-10, see solver_search_covid.py):
Route2's internal proportion solve is a genuine weak point on collinear bases (e.g. GSE159585 COVID):
  - --solver softmax : over-UNIFORM (no cell type >2%, identity≈chance) — the original Fig5 behaviour
  - --solver nnls_core: over-SPARSE (~3 cell types/sample) — NOT a clean fix, just a different degeneracy
Neither recovers a realistic ~20–40 cell-type composition. **The recommended mode is external
proportions**: pass --props_csv (e.g. BayesPrism / CIBERSORTx output) and Route2 supplies only the
expression. That is the "division of labour" fix; the internal solvers are kept for A/B and for
datasets where the basis is well-conditioned (e.g. HCA, where nnls_core gave the best proportion PCC).

Usage:
  python route2_generate_nnls.py --which covid --props_csv bayesprism_covid_props.csv   # RECOMMENDED
  python route2_generate_nnls.py --which covid --solver nnls_core   # internal solve (over-sparse, A/B)
  python route2_generate_nnls.py --which covid --solver softmax     # internal solve (over-uniform, A/B)
--props_csv = rows samples × columns cell-type names (any extra cols ignored; rows renormalised to 1).
Outputs <out_root>/<dir>/generated_data_*_nnls.h5ad  (does NOT overwrite the original softmax files).
"""
import sys, os, json, argparse, numpy as np, pandas as pd, torch, torch.nn.functional as F
from scipy.optimize import nnls
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval_HCA_fold2 as R
import route2_lib as r2
import anndata as ad, scipy.sparse as sp, scanpy as sc

DEV = r2.device("cuda:0"); torch.manual_seed(0); np.random.seed(0)
BULK = "/disk1/maijl/deconv/data/GSE159585/GSE159585_bulk_RNA_seq_raw_counts_filtered.txt"
OUTROOT = os.environ.get("DOWN_ROOT", "/disk1/maijl/deconv/deconv_20260605/downstream")
COMBINED = "/disk1/maijl/deconv/cVAE/GSE159585/combined_geneembed"
CELLCOL, DONOR, MID, CPS, ALPHA = "cell type", "Sample ID", 256, 2000, 1.0

WHICH = {
    "covid": dict(ge="/disk1/maijl/deconv/cVAE/GSE159585/covid", gene_src="prediction_genes.csv", input_size=None,
                  ref="/disk1/maijl/deconv/data/GSE159585/GSE159585_covidset.h5ad",
                  samples=['GC1003242','GC1003243','GC1003244','GC1003245','GC1003246','GC1003247','GC1003248'],
                  status="COVID19", out="covid/generated_data_covid_nnls.h5ad"),
    "normal_18_24": dict(ge=COMBINED, gene_src=f"{COMBINED}/generated_data_covid.h5ad", input_size=2087,
                  ref="/disk1/maijl/deconv/data/GSE159585/GSE159585_trainset_level2celltype.h5ad",
                  samples=['GC1003234','GC1003233','GC1003230','GC1003229','GC1003226','GC1003225','GC1003221'],
                  status="Normal", out="normal_18_24/generated_data_normal_18_24_nnls.h5ad"),
}


def resolve_gene_space(c):
    ge = c["ge"]; pg = os.path.join(ge, "prediction_genes.csv")
    if c["gene_src"] == "prediction_genes.csv" and os.path.exists(pg):
        genes = pd.read_csv(pg)["prediction_genes"].astype(str).to_numpy()
    else:
        g = ad.read_h5ad(c["gene_src"], backed="r"); genes = g.var_names.astype(str).to_numpy(); g.file.close()
    ins = c["input_size"]; mp = os.path.join(ge, "meta.json")
    if ins is None and os.path.exists(mp): ins = int(json.load(open(mp))["input_size"])
    return genes, ins


def softmax_props(basis_sig, target_sig, names, steps=300, temp=0.5, le=0.1):
    """OLD solver (kept for A/B). softmax-parameterised, maximise corr on signature genes."""
    lg = torch.zeros(basis_sig.shape[0], requires_grad=True, device=DEV); opt = torch.optim.Adam([lg], lr=0.05)
    for _ in range(steps):
        opt.zero_grad(); p = F.softmax(lg / temp, 0); ps = (p.view(-1, 1) * basis_sig).sum(0)
        vx, vy = ps - ps.mean(), target_sig - target_sig.mean(); pcc = (vx * vy).sum() / (vx.norm() * vy.norm() + 1e-8)
        (((1 - pcc) + le * (p * torch.log(p + 1e-8)).sum() + 0.01 * (lg ** 2).sum())).backward(); opt.step()
    p = F.softmax(lg / temp, 0).detach().cpu().numpy(); p[p < 1e-4] = 0; p = p / (p.sum() + 1e-8)
    return {names[i]: float(p[i]) for i in range(len(names))}


def nnls_core_props(basis_lin, target_lin, core, names):
    """NEW solver. Non-negative least squares on the 3000 core (HVG) genes, then normalise.
       basis_lin: (n_types, G) linear; target_lin: (G,) linear bulk (CP10K)."""
    A = basis_lin[:, core].detach().cpu().numpy().T.astype(np.float64)   # n_core x n_types
    y = target_lin[core].detach().cpu().numpy().astype(np.float64)
    p, _ = nnls(A, y); s = p.sum(); p = p / s if s > 0 else p
    return {names[t]: float(p[t]) for t in range(len(names))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True, choices=list(WHICH))
    ap.add_argument("--solver", default="nnls_core", choices=["nnls_core", "softmax"])
    ap.add_argument("--props_csv", default=None, help="external proportions (samples x cell types); RECOMMENDED — overrides --solver")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    a = ap.parse_args(); c = WHICH[a.which]; GE = c["ge"]
    ext_props = None
    if a.props_csv:
        ext_props = pd.read_csv(a.props_csv, index_col=0); ext_props.index = ext_props.index.astype(str)
        print(f"[props] external proportions from {a.props_csv}: {ext_props.shape}")
    OUT = f"{OUTROOT}/{c['out']}"; os.makedirs(os.path.dirname(OUT), exist_ok=True)

    genes, INPUT_SIZE = resolve_gene_space(c)
    G = len(genes); gidx = {g: i for i, g in enumerate(genes)}
    pri = torch.load(f"{GE}/cell_type_mu_logvar_best.pt", map_location="cpu"); n_types = len(pri)
    tr = ad.read_h5ad(c["ref"]); tr.obs["labels"] = tr.obs[CELLCOL].astype(str)
    tr = tr[:, [g for g in genes if g in set(tr.var_names)]].copy(); present = tr.var_names.astype(str).to_numpy()
    sc.pp.normalize_total(tr, target_sum=1e4); sc.pp.log1p(tr)
    colmap = np.array([gidx[g] for g in present], dtype=int)
    Xtr = tr.X.tocsr() if sp.issparse(tr.X) else sp.csr_matrix(tr.X)

    vae = r2.load_vae_decode(f"{GE}/scvae_best.pth", G, n_types, input_size=INPUT_SIZE, mid_hidden=MID, device=DEV)
    cent_by_id = r2.decode_centroids(vae, pri, n_types, DEV)
    cent_mat = torch.tensor(np.stack([cent_by_id[i] for i in range(n_types)]), device=DEV)
    train_types = sorted(tr.obs["labels"].unique()); tmean = {}
    for t in train_types:
        full = np.zeros(G, np.float32); full[colmap] = np.asarray(Xtr[tr.obs["labels"].values == t].mean(0)).ravel(); tmean[t] = full
    id2name, corrs, dropped = r2.recover_id2name(cent_by_id, tmean, train_types, n_types)
    names = [id2name[i] for i in range(n_types)]; name2id = {n: i for i, n in id2name.items()}
    print(f"[{a.which}] solver={a.solver} alpha={a.alpha} G={G} types={n_types} centroid↔ref median={np.median(corrs):.3f}")

    sc.pp.highly_variable_genes(tr, n_top_genes=3000)
    core = np.array([gidx[g] for g in genes if g in set(tr.var_names[tr.var["highly_variable"].values].astype(str))], dtype=int)
    donors = tr.obs[DONOR].astype(str).values; cells_t = tr.obs["labels"].astype(str).values
    dt_log, dt_lin = {}, {}
    for d in np.unique(donors):
        dm = donors == d
        for t in np.unique(cells_t[dm]):
            mk = dm & (cells_t == t)
            if mk.sum() < 5: continue
            full = np.zeros(G, np.float32); full[colmap] = np.asarray(Xtr[mk].mean(0)).ravel()
            dt_log[(d, t)] = full; dt_lin[(d, t)] = np.expm1(full).astype(np.float32)
    train_donors = sorted({d for (d, _) in dt_log})
    markers = R.derive_marker_genes(tr, "labels", top_n=30)
    muni = sorted({g for v in markers.values() for g in v}); sig_idx = torch.tensor([gidx[g] for g in muni if g in gidx], device=DEV)
    gmask, w = r2.build_marker_core_mask(G, core, muni, gidx)
    net = r2.ResidualNet(len(core), n_types, G, torch.tensor(gmask).to(DEV)).to(DEV)
    Xb, Xt, td = r2.build_examples(train_donors, list(name2id), name2id, dt_lin, dt_log, core, np.stack([cent_by_id[i] for i in range(n_types)]))
    r2.train_residual(net, Xb, Xt, td, torch.tensor(w), DEV); net.eval()

    bulk = pd.read_csv(BULK, sep="\t", index_col=0).T
    bulk = bulk.loc[:, ~bulk.columns.duplicated()].reindex(columns=genes).fillna(0.0); bulk = bulk.div(bulk.sum(1), axis=0) * 1e4
    samples = [s for s in c["samples"] if s in bulk.index.astype(str).tolist()]
    print(f"[gen] {len(samples)} samples (status={c['status']}), CPS={CPS}")

    generated, meta = [], []
    for s in samples:
        bl = bulk.loc[s].values.astype(np.float32); bcore = torch.tensor(np.log1p(bl)[core], device=DEV).unsqueeze(0)
        with torch.no_grad():
            deltas = {t: net(bcore, torch.tensor([t], device=DEV)).reshape(-1) for t in range(n_types)}
            basis_lin = torch.stack([torch.expm1(torch.clamp(cent_mat[t] + a.alpha * deltas[t], min=0)) for t in range(n_types)])
            target_lin = torch.tensor(bl, device=DEV)
        if ext_props is not None and s in ext_props.index:          # RECOMMENDED: external proportions
            row = ext_props.loc[s]; tot = sum(float(row.get(nm, 0.0)) for nm in names)
            fr = {nm: float(row.get(nm, 0.0)) / tot for nm in names} if tot > 0 else {nm: 0.0 for nm in names}
        elif a.solver == "nnls_core":
            fr = nnls_core_props(basis_lin, target_lin, core, names)
        else:
            fr = softmax_props(basis_lin[:, sig_idx], target_lin[sig_idx], names)
        with torch.no_grad():
            for t in range(n_types):
                nm = names[t]
                if nm.startswith("__unmatched"): continue
                cnt = int(round(fr.get(nm, 0.0) * CPS))
                if cnt <= 0: continue
                mu = pri[t][0].to(DEV); std = torch.exp(0.5 * pri[t][1].to(DEV))
                z = mu + torch.randn(cnt, mu.shape[1], device=DEV) * std
                lab = F.one_hot(torch.tensor([t] * cnt, device=DEV), n_types).float()
                cl = torch.clamp(vae.decode(z, lab) + a.alpha * deltas[t], min=0).cpu().numpy().astype(np.float32)
                generated.append(cl); meta.extend([(s, nm)] * cnt)

    Xg = np.vstack(generated); sm = [m[0] for m in meta]; ct = [m[1] for m in meta]
    obs = pd.DataFrame({"Sample": sm, "Cell_type": ct, "Patient ID": sm, "cell type": ct, "disease_status": c["status"]})
    obs.index = [f"Cell_{i}" for i in range(len(obs))]
    ad.AnnData(X=Xg, obs=obs, var=pd.DataFrame(index=genes)).write_h5ad(OUT, compression="gzip")
    print(f"[saved] {len(obs)} cells x {G} genes -> {OUT}")


if __name__ == "__main__":
    main()
