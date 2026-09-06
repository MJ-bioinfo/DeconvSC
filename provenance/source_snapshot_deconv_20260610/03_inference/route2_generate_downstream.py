#!/usr/bin/env python
"""New-Route2 bulk-inference predictions for the GSE159585 downstream COVID-vs-Normal analysis.
Same approach as route2_generate_hca/gse: Route2 solves its own cell-type proportions from each
bulk on the anchored centroid+residual basis (train-derived signatures), then samples pure-prior
cells + bounded residual. gzip output.

Per-condition MODEL (the key fix):
  covid        -> covid/ : a COVID-SPECIFIC geneembed model retrained on GSE159585_covidset.h5ad
                  (combined_geneembed mixes normal+covid, so it is NOT covid-specific).
  normal_18_24 -> combined_geneembed : route2's existing model params (user: normal may reuse them),
                  trainset reference for the residual.
Each model carries its own gene space (prediction_genes.csv / h5ad var) + input_size (meta.json).

obs carries BOTH naming conventions so the downstream R scripts work directly:
  Sample / Cell_type   (cellchat.R)   and   'Patient ID' / 'cell type'  (ssGSEA.R)

  python route2_generate_downstream.py --which covid
  python route2_generate_downstream.py --which normal_18_24
"""
import sys, os, json, argparse, numpy as np, pandas as pd, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval_HCA_fold2 as R
import route2_lib as r2
import anndata as ad, scipy.sparse as sp, scanpy as sc

DEV = r2.device("cuda:0"); torch.manual_seed(0); np.random.seed(0)
BULK = "/disk1/maijl/deconv/data/GSE159585/GSE159585_bulk_RNA_seq_raw_counts_filtered.txt"
OUTROOT = os.environ.get("DOWN_ROOT", "/disk1/maijl/deconv/deconv_20260605/downstream")
COMBINED = "/disk1/maijl/deconv/cVAE/GSE159585/combined_geneembed"

WHICH = {
    "covid": dict(ge="/disk1/maijl/deconv/cVAE/GSE159585/covid",          # COVID-SPECIFIC model
                  gene_src="prediction_genes.csv", input_size=None,        # read from model dir
                  ref="/disk1/maijl/deconv/data/GSE159585/GSE159585_covidset.h5ad",
                  samples=['GC1003242', 'GC1003243', 'GC1003244', 'GC1003245', 'GC1003246', 'GC1003247', 'GC1003248'],
                  status="COVID19", out="covid/generated_data_covid.h5ad"),
    "normal_18_24": dict(ge=COMBINED,                                       # route2's existing params
                  gene_src=f"{COMBINED}/generated_data_covid.h5ad", input_size=2087,
                  ref="/disk1/maijl/deconv/data/GSE159585/GSE159585_trainset_level2celltype.h5ad",
                  samples=['GC1003234', 'GC1003233', 'GC1003230', 'GC1003229', 'GC1003226', 'GC1003225', 'GC1003221'],
                  status="Normal", out="normal_18_24/generated_data_normal_18_24.h5ad"),
}
CELLCOL, DONOR, MID, CPS = "cell type", "Sample ID", 256, 2000


def dmean(Xsub): return np.asarray(Xsub.mean(axis=0)).ravel()


def resolve_gene_space(c):
    """Genes (model output space) + input_size (core gene count) for this model dir."""
    ge = c["ge"]
    pg = os.path.join(ge, "prediction_genes.csv")
    if c["gene_src"] == "prediction_genes.csv" and os.path.exists(pg):
        genes = pd.read_csv(pg)["prediction_genes"].astype(str).to_numpy()
    else:
        g = ad.read_h5ad(c["gene_src"], backed="r"); genes = g.var_names.astype(str).to_numpy(); g.file.close()
    ins = c["input_size"]
    mp = os.path.join(ge, "meta.json")
    if ins is None and os.path.exists(mp):
        ins = int(json.load(open(mp))["input_size"])
    if ins is None:  # last resort: read from checkpoint encoder param
        sd = torch.load(os.path.join(ge, "scvae_best.pth"), map_location="cpu"); ins = int(sd["gene_embedding"].shape[0])
    return genes, ins


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
    ap = argparse.ArgumentParser(); ap.add_argument("--which", required=True, choices=list(WHICH))
    ap.add_argument("--props_csv", default=None, help="external proportions (samples x cell types); overrides internal softmax solve")
    ap.add_argument("--min_cells_per_type", type=int, default=0, help="floor cells/type (0=off)")
    a = ap.parse_args()
    c = WHICH[a.which]; GE = c["ge"]; OUT = f"{OUTROOT}/{c['out']}"; os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ext_props = None
    if a.props_csv:
        ext_props = pd.read_csv(a.props_csv, index_col=0); ext_props.index = ext_props.index.astype(str)
        print(f"[props] external proportions from {a.props_csv}: {ext_props.shape}")

    genes, INPUT_SIZE = resolve_gene_space(c)
    G = len(genes); gidx = {g: i for i, g in enumerate(genes)}
    pri = torch.load(f"{GE}/cell_type_mu_logvar_best.pt", map_location="cpu"); n_types = len(pri)
    print(f"[{a.which}] model={GE}\n  G={G} input_size={INPUT_SIZE} n_types={n_types}")

    tr = ad.read_h5ad(c["ref"]); tr.obs["labels"] = tr.obs[CELLCOL].astype(str)
    tr = tr[:, [g for g in genes if g in set(tr.var_names)]].copy()
    present = tr.var_names.astype(str).to_numpy()
    sc.pp.normalize_total(tr, target_sum=1e4); sc.pp.log1p(tr)
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
    print(f"[{a.which}] centroid↔ref corr median={np.median(corrs):.3f} dropped={len(dropped)}")

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
    Xb, Xt, td = r2.build_examples(train_donors, list(name2id), name2id, dt_lin, dt_log, core, np.stack([cent_by_id[i] for i in range(n_types)]))
    print(f"[train] donors={len(train_donors)} examples={len(Xb)} sig={len(sig_idx)}")
    r2.train_residual(net, Xb, Xt, td, torch.tensor(w), DEV); net.eval()

    bulk = pd.read_csv(BULK, sep="\t", index_col=0).T
    bulk = bulk.loc[:, ~bulk.columns.duplicated()].reindex(columns=genes).fillna(0.0)
    bulk = bulk.div(bulk.sum(axis=1), axis=0) * 1e4
    samples = [s for s in c["samples"] if s in bulk.index.astype(str).tolist()]
    print(f"[gen] {len(samples)} bulk samples (status={c['status']}), CPS={CPS}; samples={samples}")

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
                count = max(int(round(fr.get(nm, 0.0) * CPS)), a.min_cells_per_type)
                if count <= 0: continue
                mu = pri[t][0].to(DEV); std = torch.exp(0.5 * pri[t][1].to(DEV))
                z = mu + torch.randn(count, mu.shape[1], device=DEV) * std
                lab = F.one_hot(torch.tensor([t] * count, device=DEV), n_types).float()
                cl = torch.clamp(vae.decode(z, lab) + deltas[t], min=0).cpu().numpy().astype(np.float32)
                generated.append(cl); meta.extend([(s, nm)] * count)

    Xg = np.vstack(generated)
    sm = [m[0] for m in meta]; ct = [m[1] for m in meta]
    obs = pd.DataFrame({"Sample": sm, "Cell_type": ct, "Patient ID": sm, "cell type": ct,
                        "disease_status": c["status"]})
    obs.index = [f"Cell_{i}" for i in range(len(obs))]
    out = ad.AnnData(X=Xg, obs=obs, var=pd.DataFrame(index=genes))
    out.write_h5ad(OUT, compression="gzip")
    print(f"[saved] {out.n_obs} cells x {out.n_vars} genes (gzip) -> {OUT}")


if __name__ == "__main__":
    main()
