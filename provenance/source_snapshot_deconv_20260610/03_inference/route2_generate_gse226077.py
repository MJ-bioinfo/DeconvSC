#!/usr/bin/env python
"""New-Route2 bulk-inference prediction for the GSE226077 iMGL (induced microglia-like) application.

The GSE226077 geneembed model ALREADY EXISTS and is route2-compatible (input_size=2314, G=16129,
5 marker_cell_type states, mid_hidden=256, trained 200 epochs) -> NO retraining needed; we reuse
its decoder + priors with the new Route2 inference (prior-anchored centroid + bounded marker∪core
residual + self-solved proportions), exactly like the normal_18_24 case.

Bulk = iMGL differentiation time course (D0,D1,D2,D4 rep1/rep2/rep3). Each bulk sample -> CPS cells.
D3 (GSM7062724/725/726) is excluded: the original GSE226077 study flagged it as a poor-quality timepoint.
obs carries Sample (bulk name, encodes the day as word 4) + Cell_type (marker_cell_type) for the
downstream Monocle2 trajectory and ssGSEA. Also writes ordering_genes.csv (top markers per
Cell_type, scanpy rank_genes_groups) so Monocle2 needs no Seurat.

  python route2_generate_gse226077.py
"""
import sys, os, numpy as np, pandas as pd, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval_HCA_fold2 as R
import route2_lib as r2
import anndata as ad, scipy.sparse as sp, scanpy as sc

DEV = r2.device("cuda:0"); torch.manual_seed(0); np.random.seed(0)
GE   = "/disk1/maijl/deconv/data/GSE226077/result/gse226077_embed16_head2_layer2/allday/application"
REF  = "/disk1/maijl/deconv/data/GSE226077/GSE226077_trainset.h5ad"
BULK = "/disk1/maijl/deconv/data/GSE226077/GSE226077_bulk_TPM_matrix.csv"
OUT  = os.environ.get("OUT", os.environ.get("IMGL_ROOT", "/disk1/maijl/deconv/deconv_20260605/GSE226077") + "/generated_data.h5ad")
CELLCOL, DONOR, MID, CPS = "marker_cell_type", "development_stage", 256, 400
DIV = float(os.environ.get("DIV", "0.6"))   # within-type diversity-injection strength (0=tight, 1=raw real)
SMOOTH_K = int(os.environ.get("SMOOTH_K", "4"))   # avg K ref-cell deviations -> denoise (helps rare types)
# D0/D1/D2/D4 × rep1/rep2/rep3 (12 samples). D3 (rep1/2/3 = GSM7062724/725/726) excluded:
# flagged as a poor-quality timepoint by the original GSE226077 study.
SAMPLES = ['GSM7062715_Bulk_iMGL_D0_rep1', 'GSM7062716_Bulk_iMGL_D0_rep2', 'GSM7062717_Bulk_iMGL_D0_rep3',
           'GSM7062718_Bulk_iMGL_D1_rep1', 'GSM7062719_Bulk_iMGL_D1_rep2', 'GSM7062720_Bulk_iMGL_D1_rep3',
           'GSM7062721_Bulk_iMGL_D2_rep1', 'GSM7062722_Bulk_iMGL_D2_rep2', 'GSM7062723_Bulk_iMGL_D2_rep3',
           'GSM7062727_Bulk_iMGL_D4_rep1', 'GSM7062728_Bulk_iMGL_D4_rep2', 'GSM7062729_Bulk_iMGL_D4_rep3']


def dmean(Xsub): return np.asarray(Xsub.mean(axis=0)).ravel()


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
                    help="external proportions (samples x cell types); overrides internal softmax solve")
    ap.add_argument("--min_cells_per_type", type=int, default=0, help="floor cells/type (0=off)")
    args = ap.parse_args()
    ext_props = None
    if args.props_csv:
        ext_props = pd.read_csv(args.props_csv, index_col=0); ext_props.index = ext_props.index.astype(str)
        print(f"[props] external proportions from {args.props_csv}: {ext_props.shape}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    genes = pd.read_csv(f"{GE}/prediction_genes.csv")["prediction_genes"].astype(str).to_numpy()
    G = len(genes); gidx = {g: i for i, g in enumerate(genes)}
    sd_emb = torch.load(f"{GE}/scvae_best.pth", map_location="cpu")["gene_embedding"]; INPUT_SIZE = int(sd_emb.shape[0])
    pri = torch.load(f"{GE}/cell_type_mu_logvar_best.pt", map_location="cpu"); n_types = len(pri)
    print(f"[gse226077] model={GE}\n  G={G} input_size={INPUT_SIZE} n_types={n_types} (REUSED — no retrain)")

    tr = ad.read_h5ad(REF); tr.obs["labels"] = tr.obs[CELLCOL].astype(str)
    tr = tr[:, [g for g in genes if g in set(tr.var_names)]].copy()
    present = tr.var_names.astype(str).to_numpy()
    if tr.X.max() > 50: sc.pp.normalize_total(tr, target_sum=1e4); sc.pp.log1p(tr)   # raw -> log1p CP10K
    colmap = np.array([gidx[g] for g in present], dtype=int)
    Xtr = tr.X.tocsr() if sp.issparse(tr.X) else sp.csr_matrix(tr.X)

    vae = r2.load_vae_decode(f"{GE}/scvae_best.pth", G, n_types, input_size=INPUT_SIZE, mid_hidden=MID, device=DEV)
    cent_by_id = r2.decode_centroids(vae, pri, n_types, DEV)
    cent_mat = torch.tensor(np.stack([cent_by_id[i] for i in range(n_types)]), device=DEV)

    train_types = sorted(tr.obs["labels"].unique()); tmean = {}
    for t in train_types:
        full = np.zeros(G, np.float32); full[colmap] = dmean(Xtr[tr.obs["labels"].values == t]); tmean[t] = full
    id2name, corrs, dropped = r2.recover_id2name(cent_by_id, tmean, train_types, n_types)
    name2id = {n: i for i, n in id2name.items()}; valid_names = [id2name[i] for i in range(n_types)]
    print(f"[gse226077] centroid↔ref corr median={np.median(corrs):.3f} dropped={len(dropped)}; types={valid_names}")

    sc.pp.highly_variable_genes(tr, n_top_genes=3000)
    hvg = set(tr.var_names[tr.var["highly_variable"].values].astype(str))
    core = np.array([gidx[g] for g in genes if g in hvg], dtype=int)
    donors = tr.obs[DONOR].astype(str).values; cells_t = tr.obs["labels"].astype(str).values
    dt_log, dt_lin = {}, {}
    for d in np.unique(donors):
        dm = donors == d
        for t in np.unique(cells_t[dm]):
            mk = dm & (cells_t == t)
            if mk.sum() < 5: continue
            full = np.zeros(G, np.float32); full[colmap] = dmean(Xtr[mk])
            dt_log[(d, t)] = full; dt_lin[(d, t)] = np.expm1(full).astype(np.float32)
    train_donors = sorted({d for (d, _) in dt_log})

    markers = R.derive_marker_genes(tr, "labels", top_n=30)
    muni = sorted({g for v in markers.values() for g in v})
    sig_idx = torch.tensor([gidx[g] for g in muni if g in gidx], device=DEV)
    gmask, w = r2.build_marker_core_mask(G, core, muni, gidx)
    net = r2.ResidualNet(len(core), n_types, G, torch.tensor(gmask).to(DEV)).to(DEV)
    Xb, Xt, td = r2.build_examples(train_donors, list(name2id), name2id, dt_lin, dt_log, core,
                                   np.stack([cent_by_id[i] for i in range(n_types)]))
    print(f"[train] donors={len(train_donors)} examples={len(Xb)} sig={len(sig_idx)}")
    r2.train_residual(net, Xb, Xt, td, torch.tensor(w), DEV); net.eval()

    bulk = pd.read_csv(BULK, sep=",", index_col=0).T   # genes x samples -> samples x genes (TPM)
    bulk = bulk.loc[:, ~bulk.columns.duplicated()].reindex(columns=genes).fillna(0.0)
    bulk = bulk.div(bulk.sum(axis=1), axis=0) * 1e4
    samples = [s for s in SAMPLES if s in bulk.index.astype(str).tolist()]
    print(f"[gen] {len(samples)} bulk samples, CPS={CPS}; samples={samples}")

    # Per-type real reference cells for within-type DIVERSITY injection. Each generated cell gets a
    # random real cell's deviation (x_ref_i - centroid_ref_t). E[dev]=0 -> pseudobulk / deconvolution
    # metrics are unchanged, but realistic cell-to-cell spread is restored (UMAP/trajectory look right
    # instead of collapsing to one tight blob per (sample,type)).
    colmap_t = torch.tensor(colmap, device=DEV)
    labels_ref = tr.obs["labels"].astype(str).values
    ref_idx = {nm: np.where(labels_ref == nm)[0] for nm in valid_names if not nm.startswith("__unmatched")}
    cent_ref_present = {nm: np.asarray(Xtr[ix].mean(0)).ravel().astype(np.float32)
                        for nm, ix in ref_idx.items() if len(ix) > 0}
    rng = np.random.default_rng(0)

    generated, meta = [], []
    for s in samples:
        bl = bulk.loc[s].values.astype(np.float32)
        bcore = torch.tensor(np.log1p(bl)[core], device=DEV).unsqueeze(0)
        with torch.no_grad():
            deltas = {t: net(bcore, torch.tensor([t], device=DEV)).reshape(-1) for t in range(n_types)}
            basis_lin = torch.stack([torch.expm1(torch.clamp(cent_mat[t] + deltas[t], min=0)) for t in range(n_types)])
            target_lin = torch.tensor(bl, device=DEV)
        if ext_props is not None and s in ext_props.index:          # external proportions
            row = ext_props.loc[s]; tot = sum(float(row.get(nm, 0.0)) for nm in valid_names)
            fr = {nm: float(row.get(nm, 0.0)) / tot for nm in valid_names} if tot > 0 else {nm: 0.0 for nm in valid_names}
        else:
            fr = solve_props(basis_lin[:, sig_idx], target_lin[sig_idx], valid_names)
        with torch.no_grad():
            for t in range(n_types):
                nm = valid_names[t]
                if nm.startswith("__unmatched"): continue
                count = max(int(round(fr.get(nm, 0.0) * CPS)), args.min_cells_per_type)
                if count <= 0: continue
                mu = pri[t][0].to(DEV); std = torch.exp(0.5 * pri[t][1].to(DEV))
                z = mu + torch.randn(count, mu.shape[1], device=DEV) * std
                lab = F.one_hot(torch.tensor([t] * count, device=DEV), n_types).float()
                base = vae.decode(z, lab)
                if DIV > 0 and nm in ref_idx and len(ref_idx[nm]) > 0:   # inject real within-type spread
                    sel = rng.choice(ref_idx[nm], count * SMOOTH_K, replace=True)
                    dv = Xtr[sel].toarray().astype(np.float32).reshape(count, SMOOTH_K, -1) - cent_ref_present[nm]
                    dev = torch.tensor(dv.mean(1), device=DEV)            # avg K ref cells -> denoise outliers
                    base[:, colmap_t] = base[:, colmap_t] + DIV * dev
                cl = torch.clamp(base + deltas[t], min=0).cpu().numpy().astype(np.float32)
                generated.append(cl); meta.extend([(s, nm)] * count)

    Xg = np.vstack(generated)
    sm = [m[0] for m in meta]; ct = [m[1] for m in meta]
    day = [s.split("_")[3] for s in sm]                       # word 4 = D0/D1/D2/D4 (matches Monocle2.R)
    obs = pd.DataFrame({"Sample": sm, "Cell_type": ct, "development_stage": day})
    obs.index = [f"Cell_{i}" for i in range(len(obs))]
    out = ad.AnnData(X=Xg, obs=obs, var=pd.DataFrame(index=genes))
    out.write_h5ad(OUT, compression="gzip")
    print(f"[saved] {out.n_obs} cells x {out.n_vars} genes (gzip) -> {OUT}")
    print("  composition:\n", obs.groupby(["development_stage", "Cell_type"]).size().unstack(fill_value=0))

    # ordering genes for Monocle2 (replaces Seurat FindAllMarkers): top markers per Cell_type
    og = out.copy()
    sc.pp.highly_variable_genes(og, n_top_genes=3000)  # ensure dispersion ok; rank uses all genes
    try:
        sc.tl.rank_genes_groups(og, "Cell_type", method="wilcoxon", n_genes=50)
        ordering = sorted({g for grp in og.obs["Cell_type"].cat.categories
                           for g in og.uns["rank_genes_groups"]["names"][grp]}) \
            if hasattr(og.obs["Cell_type"], "cat") else \
            sorted({g for grp in np.unique(ct) for g in og.uns["rank_genes_groups"]["names"][grp]})
    except Exception as e:
        print("[ordering] wilcoxon failed, using HVG:", e)
        ordering = sorted(og.var_names[og.var["highly_variable"].values].astype(str))
    pd.DataFrame({"ordering_genes": ordering}).to_csv(
        os.path.join(os.path.dirname(OUT), "ordering_genes.csv"), index=False)
    print(f"[saved] {len(ordering)} ordering genes -> ordering_genes.csv")


if __name__ == "__main__":
    main()
