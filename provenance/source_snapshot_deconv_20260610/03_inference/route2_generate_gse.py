#!/usr/bin/env python
"""Generate GSE141115 / GSE159585 generated_data.h5ad with the SAME new Route 2 approach as
route2_generate_hca.py: Route 2 solves its OWN cell-type proportions from each real bulk (on
the anchored centroid+residual basis, using train-derived signature genes), then samples a
controlled budget of pure-prior cells and adds the residual. It does NOT borrow the recorded
Delta_Z generation's per-(sample,cell-type) counts. Written gzip-compressed. Self-contained:
markers/signatures derived from the TRAIN reference (no test-set peeking).

  python route2_generate_gse.py --dataset GSE141115
  python route2_generate_gse.py --dataset GSE159585
"""
import sys, os, argparse, numpy as np, pandas as pd, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval_HCA_fold2 as R
import route2_lib as r2
import anndata as ad, scipy.sparse as sp, scanpy as sc

WORKROOT = os.environ.get("WORK_ROOT", "/disk1/maijl/deconv/deconv_20260605")  # deconv_20260610: outputs root (default = established working tree)
DEV = r2.device("cuda:0"); torch.manual_seed(0); np.random.seed(0)

DATASETS = {
    "GSE141115": dict(
        GE="/disk1/maijl/deconv/cVAE/GSE141115/attention_geneembed",
        gen_ref="/disk1/maijl/deconv/cVAE/GSE141115/attention_geneembed/generated_data.h5ad",
        train="/disk1/maijl/deconv/data/GSE141115/GSE141115_sc_raw_counts_train.h5ad",
        bulk="/disk1/maijl/deconv/data/GSE141115/GSE141115_bulk_counts_LD_testset.txt",
        celltype="cell type", donor="sample", input_size=2940, cps=1600, samples=None),
    "GSE159585": dict(
        GE="/disk1/maijl/deconv/cVAE/GSE159585/geneembed",
        gen_ref="/disk1/maijl/deconv/cVAE/GSE159585/geneembed/generated_data_normal.h5ad",
        train="/disk1/maijl/deconv/data/GSE159585/GSE159585_trainset_level2celltype.h5ad",
        bulk="/disk1/maijl/deconv/data/GSE159585/GSE159585_bulk_RNA_seq_raw_counts_filtered.txt",
        celltype="cell type", donor="Sample ID", input_size=2022, cps=2000,
        samples=['GC1003224', 'GC1003227', 'GC1003228', 'GC1003222', 'GC1003232', 'GC1003231', 'GC1003233']),
    "HRA000917": dict(
        GE="/disk1/maijl/deconv/cVAE/HRA000917/model/geneembed",
        genes_csv="/disk1/maijl/deconv/cVAE/HRA000917/model/geneembed/prediction_genes.csv",
        train="/disk1/maijl/deconv/cVAE/HRA000917/ref/HRA000917_train.h5ad",
        bulk="/disk1/maijl/deconv/cVAE/HRA000917/bulk/HRA000917_bulk_TPM_symbols.tsv",
        celltype="cell_type", donor="sample", input_size=2221, cps=1500, samples=None),
}


def dense_mean(Xsub):
    return np.asarray(Xsub.mean(axis=0)).ravel()


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
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    # Expression-coverage floor: generate >= N cells for EVERY cell type regardless of its
    # predicted proportion. Default 0 = OFF (unchanged behaviour). Set >0 when a sparse proportion
    # solver (prophead/NNLS) would otherwise give some types 0 cells and drop them from the
    # per-type EXPRESSION eval — expression is decoupled from proportion, so flooring coverage does
    # NOT change any per-type profile. NOTE: do not read proportions back from a floored h5ad
    # (the floor adds fixed mass); use the proportion vector directly for composition metrics.
    ap.add_argument("--min_cells_per_type", type=int, default=0,
                    help="floor cells/type for full expression coverage (0=off, default)")
    ap.add_argument("--props_csv", default=None,
                    help="external proportions (samples x cell types); RECOMMENDED — overrides internal softmax solve")
    a = ap.parse_args()
    c = DATASETS[a.dataset]; OUT = f"{WORKROOT}/{a.dataset}/generated_data.h5ad"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ext_props = None
    if a.props_csv:
        ext_props = pd.read_csv(a.props_csv, index_col=0); ext_props.index = ext_props.index.astype(str)
        print(f"[props] external proportions from {a.props_csv}: {ext_props.shape}")

    if c.get("genes_csv"):   # gene space from prediction_genes.csv (no gen_ref h5ad; e.g. hra000917)
        genes = pd.read_csv(c["genes_csv"])["prediction_genes"].astype(str).to_numpy()
    else:
        gen = ad.read_h5ad(c["gen_ref"], backed="r"); genes = gen.var_names.astype(str).to_numpy(); gen.file.close()
    G = len(genes); gidx = {g: i for i, g in enumerate(genes)}
    pri = torch.load(f"{c['GE']}/cell_type_mu_logvar_best.pt", map_location="cpu"); n_types = len(pri)

    tr = ad.read_h5ad(c["train"]); tr.obs["labels"] = tr.obs[c["celltype"]].astype(str)
    tr = tr[:, [g for g in genes if g in set(tr.var_names)]].copy()
    present = tr.var_names.astype(str).to_numpy()
    sc.pp.normalize_total(tr, target_sum=1e4); sc.pp.log1p(tr)
    colmap = np.array([gidx[g] for g in present], dtype=int)
    Xtr = tr.X.tocsr() if sp.issparse(tr.X) else sp.csr_matrix(tr.X)

    vae = r2.load_vae_decode(f"{c['GE']}/scvae_best.pth", G, n_types, input_size=c["input_size"], mid_hidden=256, device=DEV)
    cent_by_id = r2.decode_centroids(vae, pri, n_types, DEV)
    cent_mat = torch.tensor(np.stack([cent_by_id[i] for i in range(n_types)]), device=DEV)

    train_types = sorted(tr.obs["labels"].unique())
    tmean = {}
    for t in train_types:
        full = np.zeros(G, np.float32); full[colmap] = dense_mean(Xtr[tr.obs["labels"].values == t]); tmean[t] = full
    id2name, corrs, dropped = r2.recover_id2name(cent_by_id, tmean, train_types, n_types)
    name2id = {n: i for i, n in id2name.items()}; valid_names = [id2name[i] for i in range(n_types)]
    print(f"[{a.dataset}] G={G} types={n_types} centroid↔train corr median={np.median(corrs):.3f} dropped={dropped}")

    sc.pp.highly_variable_genes(tr, n_top_genes=3000)
    hvg = set(tr.var_names[tr.var["highly_variable"].values].astype(str))
    core = np.array([gidx[g] for g in genes if g in hvg], dtype=int)
    donors = tr.obs[c["donor"]].astype(str).values; cells_t = tr.obs["labels"].astype(str).values
    dt_log, dt_lin = {}, {}
    for d in np.unique(donors):
        dm = donors == d
        for t in np.unique(cells_t[dm]):
            mk = dm & (cells_t == t)
            if mk.sum() < 5: continue
            full = np.zeros(G, np.float32); full[colmap] = dense_mean(Xtr[mk])
            dt_log[(d, t)] = full; dt_lin[(d, t)] = np.expm1(full).astype(np.float32)
    train_donors = sorted({d for (d, _) in dt_log})

    # markers/signatures from TRAIN (self-contained, no test peeking)
    markers = R.derive_marker_genes(tr, "labels", top_n=30)
    muni = sorted({g for v in markers.values() for g in v})
    sig_idx = torch.tensor([gidx[g] for g in muni if g in gidx], device=DEV)
    gmask, w = r2.build_marker_core_mask(G, core, muni, gidx)
    net = r2.ResidualNet(len(core), n_types, G, torch.tensor(gmask).to(DEV)).to(DEV)
    Xb, Xt, td = r2.build_examples(train_donors, list(name2id), name2id, dt_lin, dt_log, core, np.stack([cent_by_id[i] for i in range(n_types)]))
    print(f"[train] donors={len(train_donors)} examples={len(Xb)} sig={len(sig_idx)}")
    r2.train_residual(net, Xb, Xt, td, torch.tensor(w), DEV); net.eval()

    bulk = pd.read_csv(c["bulk"], sep="\t", index_col=0).T
    bulk = bulk.loc[:, ~bulk.columns.duplicated()].reindex(columns=genes).fillna(0.0)
    bulk = bulk.div(bulk.sum(axis=1), axis=0) * 1e4
    samples = [s for s in (c["samples"] or list(bulk.index.astype(str))) if s in bulk.index.astype(str).tolist()]
    print(f"[gen] {len(samples)} bulk samples, CPS={c['cps']} -> budget ~{len(samples)*c['cps']} cells; samples={samples[:8]}")

    generated, meta = [], []
    for si, s in enumerate(samples):
        bl = bulk.loc[s].values.astype(np.float32)              # 1e4-CPM linear
        bcore = torch.tensor(np.log1p(bl)[core], device=DEV).unsqueeze(0)
        with torch.no_grad():
            deltas = {t: net(bcore, torch.tensor([t], device=DEV)).reshape(-1) for t in range(n_types)}
            basis_lin = torch.stack([torch.expm1(torch.clamp(cent_mat[t] + deltas[t], min=0)) for t in range(n_types)])
            target_lin = torch.tensor(bl, device=DEV)
        if ext_props is not None and s in ext_props.index:          # RECOMMENDED: external proportions
            row = ext_props.loc[s]; tot = sum(float(row.get(nm, 0.0)) for nm in valid_names)
            fr = {nm: float(row.get(nm, 0.0)) / tot for nm in valid_names} if tot > 0 else {nm: 0.0 for nm in valid_names}
        else:
            fr = solve_props(basis_lin[:, sig_idx], target_lin[sig_idx], valid_names)
        with torch.no_grad():
            for t in range(n_types):
                nm = valid_names[t]
                if nm.startswith("__unmatched"): continue
                count = max(int(round(fr.get(nm, 0.0) * c["cps"])), a.min_cells_per_type)
                if count <= 0: continue
                mu = pri[t][0].to(DEV); std = torch.exp(0.5 * pri[t][1].to(DEV))
                z = mu + torch.randn(count, mu.shape[1], device=DEV) * std
                lab = F.one_hot(torch.tensor([t] * count, device=DEV), n_types).float()
                cl = torch.clamp(vae.decode(z, lab) + deltas[t], min=0).cpu().numpy().astype(np.float32)
                generated.append(cl); meta.extend([(s, nm)] * count)

    Xg = np.vstack(generated)
    obs = pd.DataFrame(meta, columns=["Sample", "Cell_type"]); obs.index = [f"Cell_{i}" for i in range(len(obs))]
    out = ad.AnnData(X=Xg, obs=obs, var=pd.DataFrame(index=genes))
    out.write_h5ad(OUT, compression="gzip")
    print(f"[saved] {out.n_obs} cells x {out.n_vars} genes (gzip) -> {OUT}")


if __name__ == "__main__":
    main()
