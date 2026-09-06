"""DeconvSC inference: the prior-anchored generation method + the prophead proportion head.

Generation (prior-anchored):
    ê(s,t) = centroid_t + alpha * delta(bulk_s, t)
      centroid_t = decode(prior_mu_t)               (frozen per-cell-type prior centroid)
      delta      = bounded residual on marker∪core genes, learned from reference donors
Cells for sample s are sampled from the per-type prior (z = mu_t + eps*std_t), decoded, and the
bulk-conditioned residual is added in gene space.

Proportions (prophead): cell-type fractions are predicted by an amortised softmax head trained on
synthetic pseudobulks drawn from the reference's realistic composition prior pi_ref (a Dirichlet
fit to the reference's per-donor compositions) with Scaden-style input noise. This replaced the
older per-sample softmax / latent-optimization solves, which were over-uniform or collapsed on collinear bases.

Both the residual and the proportion head are trained ONCE from the reference scRNA-seq h5ad
(donor + cell-type columns) at predict time; everything else (priors, core indices, mapping) comes
from the preprocess bundle. This module is the single source of truth for the DeconvSC method.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm


# ======================================================================
# Expression residual: bulk(core genes) + cell-type -> bounded per-gene residual
# ======================================================================
class ResidualNet(nn.Module):
    """bulk(core genes) + cell-type -> bounded per-gene residual, masked to marker∪core."""

    def __init__(self, n_core, n_types, n_genes, mask):
        super().__init__()
        self.benc = nn.Sequential(nn.Linear(n_core, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.temb = nn.Embedding(n_types, 64)
        self.head = nn.Sequential(nn.Linear(256 + 64, 512), nn.ReLU(),
                                  nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, n_genes))
        self.register_buffer("mask", mask)

    def forward(self, b, t):
        h = torch.cat([self.benc(b), self.temb(t)], dim=1)
        return self.mask * (2.0 * torch.tanh(self.head(h) / 2.0))


def decode_centroids(vae, priors, n_types, device):
    """centroid_t = decode(prior_mu_t) in the model's log1p output space -> {tid: [G]}."""
    out = {}
    vae.eval().to(device)
    with torch.no_grad():
        for tid in range(n_types):
            lab = F.one_hot(torch.tensor([tid], device=device), n_types).float()
            out[tid] = vae.decode(priors[tid][0].to(device), lab).reshape(-1).cpu().numpy()
    return out


def _derive_markers(adata, cell_col, top_n=30):
    a = adata.copy()
    X = a.X
    xmax = X.max() if not hasattr(X, "toarray") else X.max()
    if xmax > 50:
        sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    a.obs["__g"] = a.obs[cell_col].astype(str).values
    try:
        sc.tl.rank_genes_groups(a, "__g", method="wilcoxon", n_genes=top_n)
        nm = a.uns["rank_genes_groups"]["names"]
        return sorted({g for ct in nm.dtype.names for g in list(nm[ct][:top_n])})
    except Exception:
        Xd = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
        cells = a.obs["__g"].values; gns = list(a.var_names); out = set()
        for ct in np.unique(cells):
            mu = Xd[cells == ct].mean(0)
            out.update(gns[i] for i in np.argsort(-mu)[:top_n])
        return sorted(out)


# ======================================================================
# prophead: amortised proportion head trained on pi_ref-sampled synthetic pseudobulks
# ----------------------------------------------------------------------
# Diagnosis the head fixes: a per-sample softmax solve is over-uniform and an NNLS solve is
# over-sparse on collinear cell-type bases. The fix injects a realistic composition prior pi_ref
# (Dirichlet, method-of-moments from the reference's per-donor compositions) and learns an
# amortised mapping core-bulk -> proportions on synthetic pseudobulks drawn from that prior.
# ======================================================================
def donor_type_fractions(donors, cell_types, type_names):
    """Per-donor cell-type FRACTION matrix (n_donors x len(type_names)) from cell-level labels.
    donors, cell_types: 1-D arrays over cells. Missing (donor,type) -> 0."""
    donors = np.asarray(donors).astype(str); cell_types = np.asarray(cell_types).astype(str)
    name2col = {n: j for j, n in enumerate(type_names)}
    uds = sorted(np.unique(donors)); Fr = np.zeros((len(uds), len(type_names)), np.float64)
    for i, d in enumerate(uds):
        sub = cell_types[donors == d]
        if len(sub) == 0:
            continue
        for t, c in zip(*np.unique(sub, return_counts=True)):
            if t in name2col:
                Fr[i, name2col[t]] = c
        s = Fr[i].sum()
        if s > 0:
            Fr[i] /= s
    return Fr, uds


def fit_composition_prior(fractions, min_conc=1.0, max_conc=300.0, floor=1e-3, eps=1e-8):
    """Fit a Dirichlet prior to per-donor cell-type fraction rows (n_donors x T, rows ~ sum 1).
    Method of moments: a = s*m, mean m_t = mean_d f_{d,t}, concentration s robustly aggregated
    from Var_d[f_t] = m_t(1-m_t)/(s+1). Floors rare types so they stay sampleable.
    Returns dict(mean, conc, s): conc -> Dirichlet a (sampler for the synthetic pseudobulks)."""
    Fr = np.asarray(fractions, dtype=np.float64)
    Fr = Fr / np.clip(Fr.sum(1, keepdims=True), eps, None)
    m = Fr.mean(0); v = Fr.var(0)
    mask = (v > eps) & (m > eps) & (m < 1 - eps)
    s_est = float(np.median(m[mask] * (1 - m[mask]) / v[mask] - 1.0)) if mask.sum() else max_conc
    s = float(np.clip(s_est, min_conc, max_conc))
    a = np.maximum(s * m, floor)
    return dict(mean=m, conc=a, s=s, n_donors=int(Fr.shape[0]))


def build_prop_examples(train_donors, type_names, name2id, dt_lin, core_idx, conc,
                        n_types, K=200, seed=0, target_sum=1e4):
    """Simulate K pseudobulks per donor with KNOWN proportions drawn from Dir(conc restricted to
    the donor's types). Mixing + normalisation MATCH inference exactly: linear mix -> CP10K
    (target_sum) -> log1p -> core. Returns (Xb[core log1p], P[full T props])."""
    rng = np.random.default_rng(seed)
    Xb, P = [], []
    for d in train_donors:
        types_d = [t for t in type_names if (d, t) in dt_lin]
        if len(types_d) < 2:
            continue
        ids_d = [name2id[t] for t in types_d]
        lin_stack = np.stack([dt_lin[(d, t)] for t in types_d]).astype(np.float64)
        a_d = np.array([max(conc[name2id[t]], 1e-3) for t in types_d])
        for _ in range(K):
            p = rng.dirichlet(a_d)
            bl = (p[:, None] * lin_stack).sum(0); s = bl.sum()
            if s <= 0:
                continue
            bcp = bl / s * target_sum
            Xb.append(np.log1p(bcp)[core_idx].astype(np.float32))
            pv = np.zeros(n_types, np.float32)
            for j, tid in enumerate(ids_d):
                pv[tid] = p[j]
            P.append(pv)
    return torch.tensor(np.stack(Xb)), torch.tensor(np.stack(P))


class PropHead(nn.Module):
    """Amortised proportion head: core-gene bulk -> softmax over cell types."""

    def __init__(self, n_core, n_types, hidden=(512, 256)):
        super().__init__()
        layers, d = [], n_core
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)]; d = h
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(d, n_types)

    def forward(self, x):
        return F.softmax(self.out(self.body(x)), dim=1)


def train_prop_head(head, Xb, P, device, epochs=120, lr=1e-3, noise_std=0.2):
    """Train PropHead to map core-bulk -> proportions. Loss = cross-entropy to the known
    simulation proportions. noise_std adds per-batch Gaussian input noise (Scaden/DISSECT-style
    augmentation) that bridges the single-cell-pseudobulk -> real-bulk distribution shift."""
    head = head.to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-5)
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xb, P), batch_size=256, shuffle=True)
    for ep in range(epochs):
        head.train(); tot = 0.0
        for b, p in dl:
            b, p = b.to(device), p.to(device)
            if noise_std > 0:
                b = b + noise_std * torch.randn_like(b)
            opt.zero_grad(); q = head(b)
            loss = -(p * torch.log(q + 1e-8)).sum(1).mean()
            loss.backward(); opt.step(); tot += loss.item()
        if ep % 30 == 0 or ep == epochs - 1:
            print(f"    [prophead] ep{ep} ce={tot/len(dl):.4f}")
    head.eval()
    return head


def _prop_head_feat(bulk_log_full, core_idx, target_sum=1e4):
    """Scale-invariant head input: renormalise the (log1p) bulk to CP10K, log1p, take core.
    Matches build_prop_examples' normalisation so the head transfers from synthetic to real bulk."""
    lin = torch.expm1(bulk_log_full)
    s = lin.sum()
    cp = lin / s * target_sum if s > 0 else lin
    return torch.log1p(cp)[core_idx]


# ======================================================================
# Train the residual AND the proportion head from the reference scRNA-seq (one pass)
# ======================================================================
def train_anchored_residual(vae, priors, sc_ref_path, donor_col, celltype_label, mapping_dict,
                            common_genes, core_indices, device, n_types,
                            epochs=60, n_top_markers=30, lr=1e-3, K_bulks=30, seed=18, full_mask=False,
                            prophead_epochs=120, prophead_K=200, prophead_noise=0.2):
    """Train (a) the bulk-conditioned expression residual and (b) the prophead proportion head from
    the reference scRNA-seq (donor + cell-type columns), in a single load.
    Returns (residual_net, centroids_by_id, core, prop_head, prop_names)."""
    import anndata as ad, scipy.sparse as sp
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    genes = np.asarray(common_genes, dtype=str); G = len(genes); gidx = {g: i for i, g in enumerate(genes)}
    core = np.asarray(core_indices, dtype=int)

    a = ad.read_h5ad(sc_ref_path)
    a.obs["__lab"] = a.obs[celltype_label].astype(str).values
    a = a[:, [g for g in genes if g in set(a.var_names.astype(str))]].copy()
    present = a.var_names.astype(str).to_numpy()
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    colmap = np.array([gidx[g] for g in present], dtype=int)
    Xa = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)

    centroids_by_id = decode_centroids(vae, priors, n_types, device)
    cent_mat = np.stack([centroids_by_id[i] for i in range(n_types)])

    name2id = dict(mapping_dict)
    id2name = {v: k for k, v in mapping_dict.items()}
    prop_names = [id2name[i] for i in range(n_types)]   # cell-type name in prior-id order
    donors = a.obs[donor_col].astype(str).values; cells_t = a.obs["__lab"].astype(str).values

    # per-(donor, cell type) linear/log means (shared by residual targets + prophead pseudobulks)
    dt_log, dt_lin = {}, {}
    for d in np.unique(donors):
        dm = donors == d
        for t in np.unique(cells_t[dm]):
            if t not in name2id:
                continue
            mk = dm & (cells_t == t)
            if mk.sum() < 5:
                continue
            full = np.zeros(G, np.float32); full[colmap] = np.asarray(Xa[mk].mean(axis=0)).ravel()
            dt_log[(d, t)] = full; dt_lin[(d, t)] = np.expm1(full).astype(np.float32)
    train_donors = sorted({d for (d, _) in dt_log})

    # --- (a) expression residual ---
    muni = _derive_markers(a, "__lab", top_n=n_top_markers)
    gmask = np.ones(G, np.float32) if full_mask else np.zeros(G, np.float32)
    gmask[core] = 1.0; w = np.ones(G, np.float32)
    for g in muni:
        if g in gidx:
            gmask[gidx[g]] = 1.0; w[gidx[g]] = 5.0
    w = torch.tensor(w / w.mean(), device=device)

    cent_t = torch.tensor(cent_mat)
    Xb, Xt, Yd = [], [], []
    for d in train_donors:
        types_d = [t for t in name2id if (d, t) in dt_lin]
        if len(types_d) < 2:
            continue
        lin_stack = np.stack([dt_lin[(d, t)] for t in types_d])
        for _ in range(K_bulks):
            p = rng.dirichlet([0.5] * len(types_d))
            bl = (p[:, None] * lin_stack).sum(0); s = bl.sum()
            if s <= 0:
                continue
            bclog = np.log1p(bl / s * 1e6)[core].astype(np.float32)
            for t in types_d:
                Xb.append(bclog); Xt.append(name2id[t]); Yd.append(dt_log[(d, t)])
    if not Xb:
        raise RuntimeError("prior-anchored residual: no training examples (need >=2 cell types per donor; check donor_col).")
    Xb = torch.tensor(np.stack(Xb)); Xt = torch.tensor(Xt); Yd = torch.tensor(np.stack(Yd))
    target_delta = Yd - cent_t[Xt]

    net = ResidualNet(len(core), n_types, G, torch.tensor(gmask, device=device)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xb, Xt, target_delta), batch_size=256, shuffle=True)
    print(f">>> prior-anchored residual: {len(train_donors)} donors, {len(Xb)} examples, {int(gmask.sum())}/{G} mask genes")
    for ep in range(epochs):
        net.train(); tot = 0.0
        for b, t, y in dl:
            b, t, y = b.to(device), t.to(device), y.to(device)
            opt.zero_grad(); d = net(b, t)
            loss = (w * (d - y) ** 2).mean() + 0.01 * (d ** 2).mean()
            loss.backward(); opt.step(); tot += loss.item()
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"    ep{ep} loss={tot/len(dl):.4f}")
    net.eval()

    # --- (b) prophead proportion head ---
    fr, _ = donor_type_fractions(donors, cells_t, prop_names)
    prior = fit_composition_prior(fr)
    Xb_h, P_h = build_prop_examples(train_donors, prop_names, name2id, dt_lin, core,
                                    prior["conc"], n_types, K=prophead_K, seed=0)
    print(f">>> prophead: pi_ref s={prior['s']:.1f} on {prior['n_donors']} donors, {len(Xb_h)} synthetic pseudobulks")
    prop_head = PropHead(len(core), n_types).to(device)
    train_prop_head(prop_head, Xb_h, P_h, device, epochs=prophead_epochs, noise_std=prophead_noise)

    return net, centroids_by_id, core, prop_head, prop_names


# ======================================================================
# Generate cells: prior-anchored expression + prophead proportions
# ======================================================================
def generate_prior_anchored(vae, real_bulk_tensor, sample_names, mapping_dict, priors, device,
                            output_dir, common_genes, residual_net, centroids_by_id, core_indices,
                            prop_head, prop_names, alpha=1.0, total_cells=8000):
    """Per sample: predict cell-type proportions with the prophead head from the core-gene bulk,
    then for each type sample pure-prior cells z = mu + eps*std, decode, and add the residual in
    gene space. Output: AnnData with obs[Sample, Cell_type], X = log1p."""
    vae.eval().to(device)
    for p in vae.parameters():
        p.requires_grad = False
    num_classes = len(mapping_dict)
    genes = np.asarray(common_genes, dtype=str)
    core = torch.as_tensor(np.asarray(core_indices, dtype=int), device=device)
    sorted_ids = sorted(priors.keys())
    valid_names = [list(mapping_dict.keys())[list(mapping_dict.values()).index(tid)] for tid in sorted_ids]
    cent = {tid: torch.tensor(centroids_by_id[tid], device=device) for tid in sorted_ids}
    prop_head = prop_head.to(device).eval()

    generated_cells, generated_meta = [], []
    print(f"\n>>> prior-anchored generation: {len(sample_names)} samples (alpha={alpha}), proportions = prophead ...")
    for i, sname in tqdm(list(enumerate(sample_names)), desc="Samples"):
        target_log = real_bulk_tensor[i].to(device)
        bcore = target_log[core].unsqueeze(0)
        with torch.no_grad():
            # prophead proportions from the scale-invariant core-bulk feature
            feat = _prop_head_feat(target_log, core).unsqueeze(0)
            probs = prop_head(feat).reshape(-1).cpu().numpy()      # (n_types,) in prior-id order
            fr = {prop_names[tid]: float(probs[tid]) for tid in range(num_classes)}
            # bulk-conditioned residual per cell type (broadcast to all cells of that type)
            deltas = {tid: residual_net(bcore, torch.tensor([tid], device=device)).reshape(-1) for tid in sorted_ids}
            per_sample = total_cells // len(sample_names)
            for t_idx, tid in enumerate(sorted_ids):
                count = int(fr.get(valid_names[t_idx], 0.0) * per_sample)
                if count <= 0:
                    continue
                mu = priors[tid][0].to(device); std = torch.exp(0.5 * priors[tid][1].to(device))
                z = mu + torch.randn(count, mu.shape[1], device=device) * std
                lab = F.one_hot(torch.tensor([tid] * count, device=device), num_classes).float()
                cells_log = torch.clamp(vae.decode(z, lab) + alpha * deltas[tid], min=0).cpu().numpy()
                generated_cells.append(cells_log); generated_meta.extend([(sname, valid_names[t_idx])] * count)

    if not generated_cells:
        return None, None, None
    obs_df = pd.DataFrame(generated_meta, columns=["Sample", "Cell_type"])
    obs_df.index = [f"Cell_{i}" for i in range(len(obs_df))]
    adata_gen = sc.AnnData(X=np.vstack(generated_cells).astype(np.float32), obs=obs_df)
    adata_gen.var_names = genes
    obs_df.to_csv(os.path.join(output_dir, "generated_metadata.csv"))
    return adata_gen, None, obs_df
