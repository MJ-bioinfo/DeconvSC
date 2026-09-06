#!/usr/bin/env python
"""Train a COVID-19-specific geneembed AttentionVAE on GSE159585_covidset.h5ad.

The combined_geneembed model was trained on normal+covid jointly, so it is NOT covid-specific.
This trains a dedicated covid model using the *same* logic as GSE159585-geneembed.py:
  - AttentionVAE class            (imported verbatim from GSE159585-geneembed.py)
  - train_vae training loop       (imported verbatim — same losses/priors/saving)
  - preprocessing                 (faithful subset of load_sc_data lines 248-394: the
                                   data steps that feed train_vae; UMAP/PCA/signature/donut
                                   visualisation is skipped because it does NOT affect the
                                   trained model — train_vae takes none of those outputs)

Saves into OUT_DIR (same filenames the original pipeline uses, so route2 inference loads it
exactly like combined_geneembed): scvae_best.pth, cell_type_mu_logvar_best.pt,
prediction_genes.csv, Epoch*_scvae.pth + mapping_dict.json + meta.json.
"""
import os, sys, json, importlib.util, numpy as np, pandas as pd, torch, scanpy as sc
os.environ.setdefault("MPLBACKEND", "Agg")

COVIDSET = "/disk1/maijl/deconv/data/GSE159585/GSE159585_covidset.h5ad"
BULK     = "/disk1/maijl/deconv/data/GSE159585/GSE159585_bulk_RNA_seq_raw_counts_filtered.txt"
OUT_DIR  = "/disk1/maijl/deconv/cVAE/GSE159585/covid"
CELLTYPE = "cell type"
HIDDEN   = [4096, 2048, 1024]; MID = 256; EMB = 32; NHEAD = 4; NLAYERS = 4
EPOCHS   = 50; BATCH = 512; LR = 5e-5; SEED = 18
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- import the canonical training code (AttentionVAE + train_vae + helpers) ----
spec = importlib.util.spec_from_file_location("ge", "/disk1/maijl/deconv/script/GSE159585-geneembed.py")
ge = importlib.util.module_from_spec(spec); spec.loader.exec_module(ge)
ge.output_dir = OUT_DIR          # redirect every os.path.join(output_dir, ...) save into OUT_DIR
ge.set_seed(SEED)

# ============================================================================
# Faithful preprocessing — mirrors load_sc_data (GSE159585-geneembed.py:248-394)
# for exactly the tensors train_vae consumes.
# ============================================================================
print(f"[load] {COVIDSET}")
adata = sc.read_h5ad(COVIDSET)
adata.obs["labels"] = adata.obs[CELLTYPE].astype(str)
adata.X = adata.X.toarray()                                   # raw counts
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=False)
hvg_names = adata.var[adata.var["highly_variable"]].index.tolist()

# mapping_dict in appearance order (== load_sc_data), labels tensor
unique_labels = adata.obs["labels"].unique()
mapping_dict = {label: idx for idx, label in enumerate(unique_labels)}
labels = torch.LongTensor([mapping_dict[l] for l in adata.obs["labels"]])
K = len(mapping_dict); breed_2_list = list(mapping_dict.keys())
import matplotlib, seaborn as sns
palette = sns.color_palette("tab20", n_colors=K)
color_map = {ct: matplotlib.colors.rgb2hex(c) for ct, c in zip(breed_2_list, palette)}

# bulk: read, dedupe, common genes, save prediction_genes.csv, CPM-1e4, log1p
real_bulk_df = pd.read_csv(BULK, sep="\t", index_col=0).T
real_bulk_df = real_bulk_df.loc[:, ~real_bulk_df.columns.duplicated()]
all_genes = sorted(list(set(adata.var_names) & set(real_bulk_df.columns)))
pd.DataFrame({"prediction_genes": all_genes}).to_csv(os.path.join(OUT_DIR, "prediction_genes.csv"), index=False)
gene_to_idx = {g: i for i, g in enumerate(all_genes)}
core_genes = sorted(list(set(hvg_names) & set(all_genes)))
core_indices_tensor = torch.LongTensor([gene_to_idx[g] for g in core_genes])

real_bulk_df = real_bulk_df[all_genes]
real_bulk_df = real_bulk_df * 1e4 / real_bulk_df.sum(axis=1).values[:, None]
real_bulk_log = torch.FloatTensor(np.log1p(real_bulk_df).values)

adata = adata[:, all_genes].copy()
assert adata.var_names.tolist() == all_genes, "gene order mismatch"
X_tensor = torch.FloatTensor(adata.X)
print(f"[data] cells={X_tensor.shape[0]} all_genes={len(all_genes)} core_genes={len(core_genes)} "
      f"types={K} bulk={tuple(real_bulk_log.shape)}")

# ============================================================================
# Build + train (identical class + loop as the original)
# ============================================================================
scvae = ge.AttentionVAE(input_size=len(core_genes), output_size=len(all_genes),
                        hidden_size_list=HIDDEN, mid_hidden_size=MID, num_cell_types=K,
                        embedding_dim=EMB, nhead=NHEAD, num_layers=NLAYERS)
print(f"[train] AttentionVAE in={len(core_genes)} out={len(all_genes)} types={K}  "
      f"epochs={EPOCHS} bs={BATCH} lr={LR} dev={DEV}")
best_vae, priors = ge.train_vae(
    vae_model=scvae, X_tensor=X_tensor, labels=labels, real_bulk_log=real_bulk_log,
    used_device=DEV, batch_size=BATCH, core_indices_tensor=core_indices_tensor,
    feature_size=len(all_genes), epoch_num=EPOCHS, learning_rate=LR,
    hidden_list=HIDDEN, mid_hidden_size=MID, num_cell_types=K,
    breed_2_list=breed_2_list, color_map=color_map, seed=SEED)

# train_vae already saved scvae_best.pth + cell_type_mu_logvar_best.pt into OUT_DIR.
json.dump({int(v): k for k, v in mapping_dict.items()},
          open(os.path.join(OUT_DIR, "id2name.json"), "w"), indent=2, ensure_ascii=False)
json.dump({"input_size": len(core_genes), "output_size": len(all_genes), "n_types": K,
           "mid_hidden": MID, "hidden": HIDDEN, "embedding_dim": EMB, "nhead": NHEAD,
           "num_layers": NLAYERS, "epochs": EPOCHS, "n_cells": int(X_tensor.shape[0]),
           "source": "GSE159585_covidset.h5ad"},
          open(os.path.join(OUT_DIR, "meta.json"), "w"), indent=2)
print(f"[done] covid model -> {OUT_DIR}")
print("       scvae_best.pth, cell_type_mu_logvar_best.pt, prediction_genes.csv, id2name.json, meta.json")
