"""
GSE141115 Stage-1 training: attention conditional VAE (AttentionVAE) for DeconvSC.

Dataset: GSE141115 (mouse kidney). This monolith preprocesses the reference single
cells, synthesises training pseudobulks, and trains a Transformer-based conditional
VAE whose decoder is later reused by prior-anchored inference. Retraining is OPTIONAL: the
weights this script produces already exist under $CVAE_ROOT, so this file only needs
to be run when those artifacts must be regenerated.

Data-processing recipe (embedded here)
--------------------------------------
- Reference .h5ad raw counts -> sc.pp.normalize_total(target_sum=1e4) -> sc.pp.log1p.
- Highly variable genes: sc.pp.highly_variable_genes(n_top_genes=3000); the HVGs
  intersected with the reference/bulk common genes (~2940 genes) form the
  CORE / ENCODER input genes.
- Common genes = (reference var_names) intersected with (bulk columns); this set is
  the DECODER OUTPUT space (expression is predicted for every common gene).
- The reference single cells are partitioned 80/20 train/test during upstream data
  preparation; this script trains on the prepared training split, loaded from
  sc_metadata.pkl (plus the *_train.h5ad reference).

Architecture & key hyperparameters (as instantiated in main())
--------------------------------------------------------------
AttentionVAE = a Transformer gene-context encoder + a cell-type-conditioned decoder.
Each core gene's scalar value is projected to embedding_dim and added to a learned
per-gene embedding, passed through a TransformerEncoder, projected back to a scalar,
concatenated with a cell-type label embedding, and mapped by an MLP to (mu, logvar);
the decoder takes z (reparameterised) + the label embedding and outputs non-negative
(ReLU) expression over the common genes.
- input_size = #core genes (~2940 HVGs), output_size = #common genes
- hidden_size_list = [4096, 2048, 1024], mid_hidden_size (latent) = 256
- embedding_dim = 32, nhead = 4, num_layers = 4, ff_dim = 64
- batch_size = 512, lr = 5e-5, ~50 epochs, AdamW(weight_decay=1e-4, betas=(0.9, 0.999))
- Composite loss = recon MSE (lambda_recon=1.0) + beta * KL + lambda_corr * gene-gene
  correlation loss. beta linearly warms up to 0.05 over the first 20% of epochs
  (initialised at 1e-5); lambda_corr is held at 20.0 for the first half of training
  and then linearly annealed toward 1.0. The correlation term is the MSE between the
  gene-gene Pearson correlation matrices of the real vs reconstructed core genes.

Output artifacts (the ONLY two consumed by prior-anchored inference)
------------------------------------------------------------
- scvae_best.pth              : best-epoch AttentionVAE state_dict (decode weights).
- cell_type_mu_logvar_best.pt : per-cell-type latent priors (mu, logvar), computed
                                from the training set after training finishes.
prior-anchored inference loads only these two files; nothing else written here is required
downstream.
"""

import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, kl_divergence
from scipy.sparse import issparse
import scanpy as sc
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import rgb2hex
from sklearn.model_selection import train_test_split
import umap
import argparse
import os
from pathlib import Path
import swanlab 
import glob
from scipy.stats import wilcoxon, spearmanr, pearsonr, ttest_rel, entropy, chisquare
from scipy.spatial.distance import jensenshannon
import pickle
import random
import time
import math
from tqdm import tqdm
import anndata as ad
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import nnls
from scipy.stats import chi2_contingency, wasserstein_distance
from torch.optim import AdamW
import copy
from scipy.optimize import minimize
from sklearn.metrics import silhouette_score
import warnings
from torch.distributions import Dirichlet
warnings.filterwarnings("ignore")
from sklearn.preprocessing import scale

# Device configuration
device = torch.device(os.environ.get("DECONVSC_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")
print(f"GPU Count: {torch.cuda.device_count()}")
RELEASE_ROOT = Path(__file__).resolve().parents[2]
data_dir = os.environ.get("DECONVSC_DATA_DIR", str(RELEASE_ROOT / "processed_data/real_data/mouse_kidney"))
output_dir = os.environ.get("DECONVSC_OUTPUT_DIR", str(RELEASE_ROOT / "work/historical_gse141115_training"))
config_dir = os.environ.get("DECONVSC_CONFIG_DIR", str(RELEASE_ROOT / "configs"))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def set_seed(seed=18):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure cudnn determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed = 18

###############################
### Data preprocessing & loading utility functions ###
################################
def optimize_signature_matrix(adata, group_col=None, n_signatures_per_type=50, max_total_signatures=2000):
    """
    Intelligently select Signature Genes to minimize collinearity and lower the matrix condition number.
    """
    print("--- Starting Optimized Signature Gene Selection ---")
    # 1. Pre-filter: remove confounding genes (mitochondrial MT, ribosomal RP)
    # These genes are highly expressed across all cell types and severely interfere with deconvolution
    exclude_prefixes = ('MT-', 'RPS', 'RPL', 'HB') # HB is hemoglobin; pay attention if the tissue is lung
    gene_mask = [not g.startswith(exclude_prefixes) for g in adata.var_names]
    adata_clean = adata[:, gene_mask].copy()
    print(f"Filtered out MT/RP/HB genes. Remaining: {adata_clean.n_vars}")
    # 2. Compute differential expression (Rank Genes)
    # method='wilcoxon' is the standard, but we add a pts (expression fraction) constraint
    # pts=True computes the in-group and out-group expression fractions, which is very important
    sc.tl.rank_genes_groups(adata_clean, group_col, method='wilcoxon', 
                            pts=True, # compute expression fraction
                            use_raw=False) # assume X is already log1p-transformed
    candidates = set()
    result = adata_clean.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    
    # 3. Strict selection loop
    # We want genes with a high expression fraction in the target group (pts > 0.5) and a low fraction in other groups
    selected_genes_dict = {}
    for group in groups:
        # Get the statistics for this group
        groups = result['names'].dtype.names
        names = pd.DataFrame(result['names'])[group]
        scores = pd.DataFrame(result['scores'])[group]
        logfoldchanges = pd.DataFrame(result['logfoldchanges'])[group]
        # pts (fraction of cells expressing the gene)
        pts_group = pd.DataFrame(result['pts'])[group] 
        # pts_rest (fraction of rest cells expressing) - not in the default return; must be inferred indirectly or just use logfoldchange
        # CIBERSORTx logic: only select genes with large logFC and high score
        # Assemble DataFrame
        df_group = pd.DataFrame({
            'group': group,
            'gene': names,
            'score': scores,
            'logfc': logfoldchanges,
            'pts': pts_group.values if 'pts' in result else 0 # compatibility
        })
        df_group.to_csv(os.path.join(output_dir, f'raw_signature_candidates_{group}.csv'), index=False)
        # --- Core selection logic ---
        # 1. Significance filter
        df_filtered = df_group[ (df_group['logfc'] > 1.0) & (df_group['score'] > 0) ].copy()
        # 2. Expression-fraction filter (avoid selecting genes expressed in only 1% of cells)
        # If pts info is available, require at least 20% of cells of this type to express the gene
        if 'pts' in result:
             df_filtered = df_filtered[df_filtered['pts'] > 0.2]
        df_filtered.to_csv(os.path.join(output_dir, f'filtered_signature_candidates_{group}.csv'), index=False)
        # 3. Take Top N
        top_genes = df_filtered.head(n_signatures_per_type)['gene'].tolist()
        selected_genes_dict[group] = top_genes
        candidates.update(top_genes)
    sinagures = sorted(list(candidates))
    
    
    # # 4. (Optional) Reduce the total count to prevent overfitting
    # if len(sinagures) > max_total_signatures:
    #     # If there are too many genes, keep those with the highest score
    #     print(f"Reducing genes from {len(sinagures)} to {max_total_signatures}...")
    #     # Simplified here; in practice more sophisticated logic could be used
    #     sinagures = sinagures[:max_total_signatures]
    print(f"Selected {len(sinagures)} unique signature genes.")
    
    # 5. --- Key step: check the condition number ---
    # Build a temporary Signature Matrix (A)
    # Compute the mean expression of each cell type
    sig_matrix_list = []
    unique_labels = sorted(adata_clean.obs[group_col].unique())
    
    for label in unique_labels:
        # Take the mean of cells of this type
        # Note: here we use adata_clean (Log space)
        # True CIBERSORTx minimizes the condition number in Linear space, but selecting in Log space is also fine
        cells = adata_clean[adata_clean.obs[group_col] == label, sinagures]
        mean_expr = np.mean(cells.X, axis=0)
        # If it is a sparse matrix
        if issparse(mean_expr): mean_expr = mean_expr.toarray()
        sig_matrix_list.append(mean_expr.flatten())
    
    Sig_Matrix = np.array(sig_matrix_list).T # [Genes, CellTypes]
    
    # Compute the condition number
    # This matrix is usually in Log space. If NNLS uses Linear, apply expm1 before computing
    cond_num = np.linalg.cond(Sig_Matrix)
    print(f"Signature Matrix Condition Number (Log space): {cond_num:.2f}")
    with open(os.path.join(output_dir, 'metrics.txt'), 'a') as f:
                f.write(f"Signature matrix condition number: {cond_num:.2f}\n")
    
    if cond_num > 1000:
        print("WARNING: Condition number is high (>1000). Collinearity exists.")
        print("Suggest grouping similar cell types or increasing logfc threshold.")
    else:
        print("Condition number is good (stable).")

    return sinagures, np.array(sinagures)

def plot_smart_donut(type_counts, title, save_path):
    """
    Draw a smart donut chart:
    1. Automatically generate non-overlapping colors.
    2. The legend includes: category name, cell count, percentage.
    3. Only slices with a share > 2% show a label on the chart, to prevent overlap.
    4. The legend is placed on the side to keep the information complete and unobstructed.
    """
    # 1. Prepare data
    if isinstance(type_counts, dict):
        labels = list(type_counts.keys())
        sizes = list(type_counts.values())
    else: # assume it is a Series
        labels = type_counts.index.tolist()
        sizes = type_counts.values.tolist()
    
    # Sort by count (descending) so the legend looks nice
    sorted_indices = np.argsort(sizes)[::-1]
    labels = [labels[i] for i in sorted_indices]
    sizes = [sizes[i] for i in sorted_indices]
    total = sum(sizes)
    
    # 2. Set colors (handle up to 60+ colors)
    # Use the husl palette to maximize color distinguishability
    colors = sns.color_palette("husl", len(labels))
    
    # 3. Create the canvas
    fig, ax = plt.subplots(figsize=(14, 8), subplot_kw=dict(aspect="equal"))
    
    # 4. Define the slice-label generator (only shown when the slice is large enough)
    def my_autopct(pct):
        return f'{pct:.1f}%' if pct > 2.0 else '' # adjustable threshold; below 2% nothing is shown on the chart

    # 5. Draw the pie chart (Donut)
    wedges, texts, autotexts = ax.pie(
        sizes, 
        autopct=my_autopct,
        textprops=dict(color="w", fontweight='bold', fontsize=9),
        colors=colors,
        startangle=90,
        pctdistance=0.85, # distance of the value labels from the center
        wedgeprops=dict(width=0.4, edgecolor='w') # width controls the donut thickness
    )
    
    # 6. Build detailed legend labels (Name: Count (Pct%))
    legend_labels = [f"{l}: {s} ({s/total:.1%})" for l, s in zip(labels, sizes)]
    
    # 7. Add the legend (placed on the right, multi-column to avoid being too long)
    # Dynamically adjust the number of legend columns based on the number of categories
    ncols = 1
    if len(labels) > 20: ncols = 2
    if len(labels) > 40: ncols = 3
    
    ax.legend(wedges, legend_labels,
              title="Cell Types: Count (Ratio)",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1), # place it outside the chart on the right
              fontsize=8,
              ncol=ncols)
    
    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title(title, fontdict={'fontsize': 14, 'fontweight': 'bold'})
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Donut chart saved to: {save_path}")
    
def load_sc_data(sc_data, real_bulk_path, celltype_label=None):
    """
    Load single-cell and bulk data, computing the cell-type proportions for each bulk sample.
    Parameters:
        sc_data: path to the single-cell data (.h5ad file)
        real_bulk_path: path to the bulk data (.tsv file)
        celltype_label: column name of the cell-type label in the single-cell data
    Returns:
        dataset: PyTorch dataset (single-cell data)
        X_tensor: single-cell expression tensor
        labels: cell-type label tensor
        cell_type_fractions_list: list where each element is a dict of cell-type proportions for a single bulk sample
        mapping_dict: dict mapping cell types to integers
        common_genes: list of common genes
        real_bulk_tensor: bulk data tensor (all samples)
    """
    # Load single-cell data
    adata = sc.read_h5ad(sc_data)
    adata.obs['labels'] = adata.obs[celltype_label].astype(str)
    adata.X = adata.X.toarray()  # raw data
    # adata = adata[:, adata.var.highly_variable].copy()
    print(f"Range of adata.X: {adata.X.min().min()} - {adata.X.max().max()}")
    sc.pp.normalize_total(adata, target_sum=1e4)
    print(f"Range of normalized adata.X: {adata.X.min().min()} - {adata.X.max().max()}")
    sc.pp.log1p(adata)
    print(f"Range of log1p adata.X: {adata.X.min().min()} - {adata.X.max().max()}")

    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
    sc.tl.umap(adata, min_dist=0.5, spread=1.0, random_state=seed)
    fig = plt.figure(figsize=(25, 10))
    sc.pl.umap(adata, color='cell type', title='sc_lung_train UMAP by Cell Type', legend_loc='on data', size=30)
    plt.savefig(os.path.join(output_dir, 'sc_lung_train_umap_by_celltype.png'), dpi=300)
    plt.close()

    # min_cells = 3
    # label_counts = adata.obs['labels'].value_counts()
    # small_labels = label_counts[label_counts < min_cells].index
    # adata.obs['labels'] = adata.obs['labels'].replace(small_labels, 'others')
    # sc.tl.rank_genes_groups(adata, 'labels', method='wilcoxon', random_state=seed)  # use the already log-transformed data
    # marker_df = pd.DataFrame(adata.uns['rank_genes_groups']['names']).copy().head(100)
    # marker_df.to_csv(os.path.join(output_dir, 'marker_genes.csv'), index=False)
    # marker_array = sorted(np.unique(np.ravel(marker_df)))
    # missing_genes = [g for g in marker_array if g not in adata.var_names.values]
    # if missing_genes:
    #     print(f"Warning: {len(missing_genes)} genes in marker_array not found in adata.var['gene_name']: {missing_genes[:10]}...")
    
    unique_labels = adata.obs['labels'].unique()
    mapping_dict = {label: idx for idx, label in enumerate(unique_labels)}
    print(f"load_sc_data: mapping_dict: {mapping_dict}")
    # adata.obs['labels'] = adata.obs['labels'].astype('category')
    # Count the number of cells of each cell type
    cell_number_target_num = adata.obs['labels'].value_counts().to_dict()
    print(f"每种细胞类型的细胞数目: {cell_number_target_num}")
    labels = torch.LongTensor([mapping_dict.get(label, -1) for label in adata.obs['labels']])
    if -1 in labels:
        raise ValueError("某些细胞类型标签未在 mapping_dict 中定义")
    print(f"labels shape: {labels.shape}, unique labels: {labels.unique()}")
    plot_smart_donut(
        type_counts=cell_number_target_num,
        title=f"Training Set Composition (Total: {sum(cell_number_target_num.values())})",
        save_path=os.path.join(output_dir, 'training_data_composition_donut.png')
    )

    palette = sns.color_palette("tab20", n_colors=len(torch.unique(labels)))
    hex_colors = [matplotlib.colors.rgb2hex(color) for color in palette]
    color_map = {ct: color for ct, color in zip(adata.obs['labels'].cat.categories, hex_colors)}
    print(f"color_map: {color_map}")
    
     # Load bulk data
    real_bulk_df = pd.read_csv(real_bulk_path, sep='\t', index_col=0).T
    print(f"Range of original bulk data: {real_bulk_df.min().min()} - {real_bulk_df.max().max()}")
    real_bulk_df = real_bulk_df.loc[:, ~real_bulk_df.columns.duplicated()]

    # Compute common genes
    # common_genes = sorted(list(set(marker_array) & set(adata.var_names) & set(real_bulk_df.columns)))
    all_genes = sorted(list(set(adata.var_names) & set(real_bulk_df.columns)))
    pd.DataFrame({'prediction_genes': all_genes}).to_csv(os.path.join(output_dir, 'prediction_genes.csv'), index=False)
    print(f"共有基因数量: {len(all_genes)}")
    if len(all_genes) < 100:
        raise ValueError(f"Too few common genes: {len(all_genes)}")
    
    # signature genes
    raw_signatures, signature_array = optimize_signature_matrix(adata, group_col=celltype_label, n_signatures_per_type=50, max_total_signatures=5000) 
    final_signatures = sorted(list(set(raw_signatures) & set(all_genes)))
    print(f"Prediction Genes (Total): {len(all_genes)}")
    print(f"Signatures (Raw): {len(raw_signatures)}")
    print(f"Signatures (Final valid): {len(final_signatures)}")
    signature_df = pd.DataFrame({'signature_genes': final_signatures})
    signature_df.to_csv(os.path.join(output_dir, 'signature_genes.csv'), index=True)
    gene_to_idx = {gene: i for i, gene in enumerate(all_genes)}
    sig_indices = [gene_to_idx[g] for g in final_signatures]
    
    # HVG genes
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=False)
    hvg_names = adata.var[adata.var['highly_variable']].index.tolist()
    
    # Intersect to ensure the core genes are also among the full gene set
    core_genes = sorted(list(set(hvg_names) & set(all_genes)))
    
    # Compute the indices of the core genes within the full gene set (used for slicing)
    gene_to_idx = {g:i for i,g in enumerate(all_genes)}
    core_indices = [gene_to_idx[g] for g in core_genes]
    core_indices_tensor = torch.LongTensor(core_indices)
    
    # CPM normalization
    real_bulk_df = real_bulk_df[all_genes]
    real_bulk_df = real_bulk_df * 1e4 / real_bulk_df.sum(axis=1).values[:, None]
    print(f"Range of normalized bulk data: {real_bulk_df.min().min()} - {real_bulk_df.max().max()}")
    real_bulk_log = np.log1p(real_bulk_df)
    print(f"Range of log1p bulk data: {real_bulk_log.min().min()} - {real_bulk_log.max().max()}")
    bulk_sample_names = real_bulk_df.columns.tolist()

    # Subset the single-cell and bulk data
    adata = adata[:,all_genes].copy()
    # --- 3. Enforce a consistency check (Sanity Check) ---
    # This is a safeguard to prevent "gene misalignment" during later training that would drive correlation to 0
    print("\n--- Gene Order Check ---")
    print(f"SC Genes (first 5):   {adata.var_names[:5].tolist()}")
    print(f"Bulk Genes (first 5): {real_bulk_log.columns[:5].tolist()}")
    
    if adata.var_names.tolist() != real_bulk_log.columns.tolist():
        raise ValueError("CRITICAL ERROR: Gene order mismatch between SC and Bulk!")
    else:
        print(">>> Gene order is perfectly aligned.")

    # Build the cell-type matrix
    unique_cell_types = np.unique(adata.obs['labels'])
    label2id = {label: idx for idx, label in enumerate(unique_cell_types)}
    single_cell_matrix = []
    for label in unique_cell_types:
        mask = adata.obs['labels'] == label
        mean_expr = adata[mask].X.mean(axis=0)
        single_cell_matrix.append(mean_expr)
    single_cell_matrix = np.array(single_cell_matrix).T  # [num marker genes, num cell types]
    K = single_cell_matrix.shape[1]
    print(f"单细胞矩阵形状: {single_cell_matrix.shape}, 细胞类型数量: {K}")
    
    # Subset the single-cell and bulk data
    X_tensor = torch.FloatTensor(adata.X)
    dataset = TensorDataset(X_tensor, labels)
    real_bulk_log = torch.FloatTensor(real_bulk_log.values)
    print(f"X_tensor shape: {X_tensor.shape}, labels shape: {labels.shape}, real_bulk_log shape: {real_bulk_log.shape}")
    print(f"Range of X_tensor: {X_tensor.min().item()} - {X_tensor.max().item()}")
    print(f"Range of real_bulk_log: {real_bulk_log.min().item()} - {real_bulk_log.max().item()}")
    
    # 3. Debug print: check the means of the first 5 genes
    # ------------------------------------------------------------
    print("\n--- Data Alignment Check ---")
    print(f"SC  Tensor Shape: {X_tensor.shape}")
    print(f"Bulk Tensor Shape: {real_bulk_log.shape}")
    
    # Print the mean expression of the first 5 genes to check they are on the same scale and non-empty
    sc_mean = X_tensor.mean(dim=0)[:5].numpy()
    bulk_mean = real_bulk_log.mean(dim=0)[:5].numpy()
    print(f"Gene 1-5 ({all_genes[:5]}) Mean Expression:")
    print(f"  SC  : {sc_mean}")
    print(f"  Bulk: {bulk_mean}")
    
    # Check whether the Bulk is all zeros
    if real_bulk_log.sum() == 0:
        raise ValueError("Fatal Error: Real Bulk Tensor is all zeros!")
    
    return dataset, X_tensor, labels, K, mapping_dict, all_genes, real_bulk_log, single_cell_matrix, cell_number_target_num, signature_array, color_map, bulk_sample_names, final_signatures, sig_indices, core_genes, core_indices_tensor

def compute_gene_correlation_loss(x_real, x_recon):
    """
    Compute the difference between the gene-gene correlation matrices of two batches of data.
    Input: [Batch_Size, N_Genes]
    """
    # 1. For numerical stability, add a tiny epsilon to prevent a zero standard deviation
    epsilon = 1e-8
    
    # 2. Normalize (Z-score per gene)
    # subtract the mean
    real_mean = x_real.mean(dim=0, keepdim=True)
    recon_mean = x_recon.mean(dim=0, keepdim=True)
    real_centered = x_real - real_mean
    recon_centered = x_recon - recon_mean
    
    # compute the standard deviation
    real_std = x_real.std(dim=0, keepdim=True) + epsilon
    recon_std = x_recon.std(dim=0, keepdim=True) + epsilon
    
    # 3. Compute the correlation matrix (Pearson Correlation Matrix) [Genes, Genes]
    # formula: (X_centered.T @ X_centered) / (N - 1) / (std.T @ std)
    n = x_real.size(0)
    if n <= 1: return torch.tensor(0.0, device=x_real.device) # Batch too small to compute
    
    real_corr = (real_centered.t() @ real_centered) / (n - 1)
    real_corr = real_corr / (real_std.t() @ real_std)
    
    recon_corr = (recon_centered.t() @ recon_centered) / (n - 1)
    recon_corr = recon_corr / (recon_std.t() @ recon_std)
    
    # 4. Compute the MSE loss between the two matrices
    # We only care about the off-diagonal elements (gene-gene relationships); the diagonal is always 1
    # But computing MSE directly is also fine, because the diagonal error will be 0
    loss = F.mse_loss(real_corr, recon_corr)
    
    return loss

def compute_mmd_loss(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Compute the MMD distance between two distributions (used for Batch Size > 1)
    """
    batch_size = x.size(0)
    n_samples = batch_size

    total0 = total1 = total2 = 0
    L2_distance = torch.sum((x[:, None, :] - x[None, :, :]) ** 2, dim=2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples + 1e-6)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    total0 = sum(kernel_val)

    L2_distance = torch.sum((y[:, None, :] - y[None, :, :]) ** 2, dim=2)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    total1 = sum(kernel_val)

    L2_distance = torch.sum((x[:, None, :] - y[None, :, :]) ** 2, dim=2)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    total2 = sum(kernel_val)

    loss = torch.mean(total0 + total1 - 2 * total2)
    return loss

def get_cell_type_indices(labels, num_cell_types):
    """Precompute indices"""
    indices_dict = {}
    labels_np = labels.cpu().numpy()
    for i in range(num_cell_types):
        indices_dict[i] = np.where(labels_np == i)[0]
    return indices_dict

def generate_batch_dirichlet_pseudobulk(target_samples: torch.Tensor, 
                                                  target_labels: torch.Tensor, 
                                                  cell_type_indices: list, # List of NumPy arrays or Tensors of indices
                                                  X_tensor: torch.Tensor, 
                                                  num_cell_types: int, 
                                                  device: torch.device):
    """
    Generate the corresponding Dirichlet pseudo-Bulk data for an entire batch (GPU-optimized version).
    target_samples: [Batch, Genes] (Log1p scale)
    target_labels: [Batch] (indices)
    X_tensor: [Total_Cells, Genes] (Log1p scale, preferably on GPU)
    """
    
    batch_size = target_samples.size(0)
    n_genes = target_samples.size(1)
    
    # Ensure all tensors are on the target device
    target_samples = target_samples.to(device)
    target_labels = target_labels.to(device)
    
    # --- 1. Generate batch-level Dirichlet proportions [Batch, Num_Types] (GPU) ---
    
    # Use PyTorch's Dirichlet distribution (created on CPU by default, must be moved to GPU)
    alpha_tensor = torch.full((num_cell_types,), 0.5, device=device)
    dirichlet_dist = Dirichlet(alpha_tensor)
    proportions_tensor = dirichlet_dist.sample((batch_size,)) # [Batch, Num_Types]
    
    # result container
    pseudo_bulks_linear = torch.zeros_like(target_samples).to(device)
    
    # --- 2. Prepare data and parameters ---
    num_bg_cells = 2000
    # inverse Log1p (GPU)
    target_samples_linear = torch.expm1(target_samples) 
    
    # Check whether X_tensor is on GPU; if not, handle the index data transfer
    X_tensor_device = X_tensor.device
    
    # --- 3. Core loop (the loop remains, but the inner computation is GPU-ized) ---
    
    # Precompute the total number of background cells per sample (used for the later mean computation)
    # We need a boolean mask to exclude the target cell type itself
    # Since proportions_tensor is already on GPU, we can operate on it directly
    
    # target_labels_expanded: [Batch, 1]
    target_labels_expanded = target_labels.unsqueeze(1)
    # Build a tensor with the same shape as proportions, holding each sample's target label in the batch
    target_labels_matrix = target_labels_expanded.repeat(1, num_cell_types)
    
    # Build a Cell Type index tensor [1, Num_Types]
    ct_indices_tensor = torch.arange(num_cell_types, device=device).unsqueeze(0)
    
    # Create a mask that excludes the target cell type [Batch, Num_Types]
    non_target_mask = (ct_indices_tensor != target_labels_matrix)
    
    # remove the self proportion
    bg_proportions = proportions_tensor * non_target_mask.float()
    
    # total number of sampled cells (excluding the target cell type)
    total_bg_count_batch = (bg_proportions.sum(dim=1) * num_bg_cells).round().int() # [Batch]
    
    # Iterate over each sample in the batch
    for i in range(batch_size):
        
        # background mean vector for the current sample (Linear Scale)
        bg_vector_linear = torch.zeros(n_genes, device=device)
        
        # Iterate over all cell types
        for ct_idx in range(num_cell_types):
            
            if not non_target_mask[i, ct_idx].item(): 
                continue # skip itself
            
            # number of cells to sample for this type
            n_sample = int(bg_proportions[i, ct_idx].item() * num_bg_cells)
            
            if n_sample > 0:
                avail_idx = cell_type_indices[ct_idx]
                
                if len(avail_idx) > 0:
                    
                    # Index sampling (use PyTorch instead of np.random.choice, staying on GPU or transferring efficiently)
                    
                    # Optimization: if avail_idx is a NumPy array / Python list
                    # We choose indices on CPU, then send the index tensor to GPU, ensuring data gathering and computation happen on GPU
                    chosen_indices_cpu = np.random.choice(avail_idx, n_sample, replace=True)
                    chosen_tensor = torch.as_tensor(chosen_indices_cpu, device=device)
                    
                    # Note: if X_tensor is not on device, index_select will automatically perform a transfer.
                    # Ideally, X_tensor should be on GPU.
                    if X_tensor_device != device:
                        chunk_log = X_tensor.index_select(0, chosen_tensor.cpu()).to(device)
                    else:
                        chunk_log = X_tensor.index_select(0, chosen_tensor)
                        
                    # Convert to linear and sum (performed entirely on GPU)
                    chunk_linear = torch.expm1(chunk_log)
                    bg_vector_linear += chunk_linear.sum(dim=0)
        
        # compute the mean
        current_total_count = total_bg_count_batch[i].item()
        if current_total_count > 0:
            bg_mean_linear = bg_vector_linear / current_total_count
        else:
            bg_mean_linear = torch.zeros(n_genes, device=device)
            
        # 3. Signal Injection (GPU)
        signal_frac = random.uniform(0.05, 0.50)
        
        # Mix: Target (Linear) + Background (Linear)
        pseudo_bulks_linear[i] = signal_frac * target_samples_linear[i] + (1 - signal_frac) * bg_mean_linear

    # --- 4. Noise injection and post-processing (fully batched GPU operations) ---
    
    # 1. Multiplicative Noise
    noise_mult = torch.randn_like(pseudo_bulks_linear) * 0.2 + 1.0
    pseudo_bulks_linear = pseudo_bulks_linear * noise_mult
    
    # 2. Ensure non-negative
    pseudo_bulks = torch.clamp(pseudo_bulks_linear, min=0.0)
    
    # 3. Finally return the Log1p result
    return torch.log1p(pseudo_bulks)

def check_distribution_overlap(sc_tensor, pb_tensor, real_bulk_tensor, epoch, output_dir):
    """
    Visualize the value distributions of Single Cell, Pseudo-Bulk, and Real Bulk (Log1p space).
    
    Parameters:
        sc_tensor: [B, G] single-cell batch (Log1p)
        pb_tensor: [B, G] generated pseudo-Bulk batch (Log1p)
        real_bulk_tensor: [N, G] real Bulk dataset (Log1p) - used as the reference baseline
        epoch: current epoch number
        output_dir: save path
    """
    # 1. Convert to Numpy and flatten
    sc_vals = sc_tensor.detach().cpu().numpy().flatten()
    pb_vals = pb_tensor.detach().cpu().numpy().flatten()
    
    # Real Bulk may be on GPU or CPU; ensure conversion
    if torch.is_tensor(real_bulk_tensor):
        real_vals = real_bulk_tensor.detach().cpu().numpy().flatten()
    else:
        real_vals = np.array(real_bulk_tensor).flatten()
    
    # 2. Randomly subsample to speed up plotting (take 10,000 points each)
    sample_size = 10000
    if len(sc_vals) > sample_size:
        sc_vals = np.random.choice(sc_vals, sample_size, replace=False)
    if len(pb_vals) > sample_size:
        pb_vals = np.random.choice(pb_vals, sample_size, replace=False)
    if len(real_vals) > sample_size:
        real_vals = np.random.choice(real_vals, sample_size, replace=False)
        
    # 3. Plot
    plt.figure(figsize=(10, 6))
    
    # Plot Single Cell (blue)
    sns.kdeplot(sc_vals, label='Single Cell (Log1p)', fill=True, color='blue', alpha=0.2)
    
    # Plot Pseudo Bulk (red - generated by the model)
    sns.kdeplot(pb_vals, label='Pseudo Bulk (Log1p)', fill=True, color='red', alpha=0.3)
    
    # Plot Real Bulk (green - target reference)
    sns.kdeplot(real_vals, label='Real Bulk (Log1p)', fill=True, color='green', alpha=0.2, linestyle='--')
    
    plt.title(f"Distribution Overlap Check - Epoch {epoch}")
    plt.xlabel("Expression Value (Log1p)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, f'dist_check_epoch_{epoch}.png')
    plt.savefig(save_path)
    plt.close()
    
    # 4. Print statistics
    print(f"\n[Epoch {epoch} Distribution Stats]")
    print(f"  SC       : Mean={sc_vals.mean():.4f}, Max={sc_vals.max():.4f}")
    print(f"  Pseudo-Bk: Mean={pb_vals.mean():.4f}, Max={pb_vals.max():.4f}")
    print(f"  Real-Bk  : Mean={real_vals.mean():.4f}, Max={real_vals.max():.4f}")

##############################
### Stage 1: AttentionVAE ###
##############################  
class AttentionVAE(nn.Module):
    """
    Conditional variational autoencoder (c-VAE) model integrating a Transformer attention mechanism, used for generating and reconstructing single-cell data.
    """
    def __init__(self, input_size: int, output_size:int, hidden_size_list: list, mid_hidden_size: int, num_cell_types: int,
                 embedding_dim: int, nhead: int,  num_layers: int, ff_dim: int = 64, seed: int = seed):
        """
        Initialize the variational autoencoder model, including the attention mechanism.

        Parameters:
            input_size (int): number of input genes (embedding dimension).
            hidden_size_list (list): list of hidden-layer dimensions, e.g. [2048, 1024, 512].
            mid_hidden_size (int): dimension of the middle hidden layer (used for mean and variance).
            embedding_dim (int): embedding dimension of the attention mechanism.
            nhead (int): number of attention heads of the Transformer.
            ff_dim (int): dimension of the Transformer feed-forward network.
            num_layers (int): number of Transformer encoder layers.
        """
        super(AttentionVAE, self).__init__()
        set_seed(seed)
        # Save the input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.mid_hidden_size = mid_hidden_size
        self.embedding_dim = embedding_dim
        
        # 1. Define the gene identity embedding (must be a Parameter)
        self.gene_embedding = nn.Parameter(torch.randn(input_size, embedding_dim))
        nn.init.normal_(self.gene_embedding, mean=0, std=0.02)
        
        # Projection layer for the attention mechanism
        self.linear_proj = nn.Linear(1, embedding_dim)
        
         # Transformer encoder layer
        transformers_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformers_encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embedding_dim, 1)

        # Conditional embedding layer
        self.label_emb_dim = mid_hidden_size * 4
        self.label_embedding = nn.Linear(num_cell_types, self.label_emb_dim)
        nn.init.xavier_uniform_(self.label_embedding.weight)
        
        # Encoder input dimension (gene expression + cell-type label)
        self.enc_input_dim = self.input_size + mid_hidden_size*4
        
        # Build the encoder feature-dimension list
        self.enc_feature_size_list = [self.enc_input_dim] + self.hidden_size_list + [self.mid_hidden_size * 4]
    
        # Build the Encoder MLP
        self.encoder_layers = nn.ModuleList()
        in_dim = self.enc_input_dim
        for h_dim in self.enc_feature_size_list[1:]: # skip the input layer
            self.encoder_layers.append(nn.Sequential(
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim), # strongly recommended to add BN
                nn.LeakyReLU(),
                nn.Dropout(0.1)
            ))
            in_dim = h_dim
        
        # Mean and variance layers (assuming the last hidden layer is mid_hidden_size * 4)
        last_enc_dim = self.enc_feature_size_list[-1]
        self.fc_mu = nn.Linear(last_enc_dim, mid_hidden_size)
        self.fc_var = nn.Linear(last_enc_dim, mid_hidden_size)

        # Build the Decoder
        # Decoder input: Latent Z + Label Embedding
        self.dec_input_dim = mid_hidden_size + self.label_emb_dim
        self.dec_feature_size_list = [self.dec_input_dim] + self.hidden_size_list[::-1] + [self.input_size]
        self.decoder_layers = nn.ModuleList()
        in_dim = self.dec_input_dim
        for h_dim in self.dec_feature_size_list[1:-1]:
            self.decoder_layers.append(nn.Sequential(
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.1)
            ))
            in_dim = h_dim
        
        # The final layer outputs gene expression
        # self.final_layer = nn.Linear(in_dim, self.input_size)
        self.final_layer = nn.Linear(in_dim, self.output_size)
    
    def encode(self, x: torch.Tensor, labels: torch.Tensor) -> tuple:
        # 1. Transformer feature extraction
        # x: [B, G]
        x_in = x.unsqueeze(-1) # [B, G, 1]
        x_val = self.linear_proj(x_in) # [B, G, Dim] # project into the high-dimensional space
        x_embed = x_val + self.gene_embedding # [B, G, Dim] inject gene identity information
        x_trans = self.transformer_encoder(x_embed) # [B, G, Dim] Transformer encoding
        x_feat = self.output_proj(x_trans).squeeze(-1) # [B, G] project back to scalar space (feature fusion)
        # 2. Get the Label Embedding
        label_embed = self.label_embedding(labels) # [B, label_emb_dim]
        # 3. [Core change] Concatenate (Concat)
        h = torch.cat([x_feat, label_embed], dim=1)  # x_feat contains the gene-expression features adjusted by Attention
        # 4. MLP
        for layer in self.encoder_layers:
            h = layer(h)  
        mu = self.fc_mu(h)
        logvar = self.fc_var(h)
        return mu, logvar # return a tuple

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # 1. Get the Label Embedding
        label_embed = self.label_embedding(labels)
        # 2. Concatenate
        h = torch.cat([z, label_embed], dim=1)
        # 3. MLP
        for layer in self.decoder_layers:
            h = layer(h)
        out = self.final_layer(h)
        return F.relu(out) # ensure non-negative
    
    def forward(self, x: torch.Tensor,labels: torch.Tensor, core_indices: torch.Tensor) -> tuple:
        """
        Forward pass that performs encoding, reparameterization, and decoding.

        Parameters:
            x (torch.Tensor): input tensor with shape (batch_size, input_size).
            used_device (torch.device): the device to run on (CPU or GPU).

        Returns:
            tuple: reconstruction output (x_hat) and KL divergence (kl_div).
        """
        x = x[:, core_indices]  # use core genes only
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, labels)
        kl_div = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return x_recon, mu, logvar
    
def train_vae(vae_model, X_tensor, labels, real_bulk_log,
              used_device: torch.device, batch_size: int, core_indices_tensor: torch.Tensor,
              feature_size: int, epoch_num: int, learning_rate: float, hidden_list: list, mid_hidden_size: int, num_cell_types: int, breed_2_list: list, color_map: dict, seed: int = 18) -> AttentionVAE:

    set_seed(seed)
    
    # --- Automatically detect whether the data is on GPU ---
    is_on_gpu = X_tensor.is_cuda
    # If the data is on GPU, pin_memory is not needed; otherwise it is
    use_pin_memory = not is_on_gpu 
    print(f"Data is on GPU: {is_on_gpu}. DataLoader pin_memory set to: {use_pin_memory}")
    # 1. Precompute indices (to speed up sampling)
    cell_type_indices = get_cell_type_indices(labels, num_cell_types)
    
    # 2. Ensure X_tensor is in memory (if GPU memory is large enough, X_tensor = X_tensor.to(used_device) is recommended to speed up sampling)
    # Create the DataLoader
    print(f"Start training with Batch Size: {batch_size}")
    dataset = TensorDataset(X_tensor, labels)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True,
        num_workers=8
    )
    print(f"Number of batches per epoch: {len(dataloader)}")

    # # 2. Check whether there are multiple GPUs; if so, wrap with DataParallel
    # if torch.cuda.device_count() > 1:
    #     print(f"Let's use {torch.cuda.device_count()} GPUs!")
    #     # Wrap the model to automatically use all visible GPUs
    #     vae = nn.DataParallel(vae_model)
    #     vae = vae.to(used_device)
    # else:
    #     vae = vae_model.to(used_device)
    vae = vae_model.to(used_device)
    core_indices_tensor = core_indices_tensor.to(used_device)
    criterion = nn.MSELoss()
    optimizer = AdamW(vae.parameters(), lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # --- Weight settings ---
    beta = 1e-5           # KLD weight (kept as is)
    target_beta = 0.05
    lambda_denoise = 1.0 # denoising reconstruction weight (pixel-level alignment)
    lambda_recon = 1.0
    lambda_mmd = 0.5    # MMD weight (distribution-level alignment)
    initial_lambda_corr = 20.0
    # ---------------
    
    best_vae = None
    min_loss = float('inf')
    
    swanlab.init(
    project="bulk2space",
    workspace="dendrobium",
    config={
        "lr": learning_rate, "epochs": epoch_num, "mid_hidden_size": mid_hidden_size, "num_cell_types": num_cell_types,"beta": 1e-5, "beta1": 0.9, "beta2": 0.999, "initial_lambda_corr":20.0,"lambda_recon":100.0,"hidden_list": hidden_list,"weight_decay": 5e-4,"embedding_dim": 16,"nhead":2,"num_layers":2,"batch_size": batch_size,"pseudo_bulk_cell_count":2000,"seed": seed})
    
    pbar = tqdm(range(epoch_num), desc="Training")
    for epoch in pbar:
        set_seed(seed + epoch)
        vae.train()
        train_loss_epoch = 0
        warmup_epochs = int(epoch_num * 0.2)
        if epoch < warmup_epochs:
            beta = target_beta * (epoch / warmup_epochs)
        else:
            beta = target_beta
        if epoch < epoch_num * 0.5:
            lambda_corr = initial_lambda_corr
        else:
            # linear decay
            decay_steps = epoch_num * 0.5
            progress = (epoch - epoch_num * 0.5) / decay_steps
            lambda_corr = initial_lambda_corr * (1.0 - progress) + 1.0
        for batch_idx, (cell_features, label_indices) in enumerate(dataloader):
            # move data to device
            cell_features = cell_features.to(used_device, dtype=torch.float32, non_blocking=True)
            label_one_hot = F.one_hot(label_indices, num_cell_types).float().to(used_device, non_blocking=True)
            
            # ==========================================
            # Task 1: clean single-cell reconstruction (Standard VAE)
            # ==========================================
            sc_recon, sc_enc_mu, sc_enc_logvar = vae(cell_features, label_one_hot, core_indices_tensor) # changed to return a tuple directly
            
            loss_recon = criterion(sc_recon, cell_features)
            loss_kld = -0.5 * torch.mean(torch.sum(1 + sc_enc_logvar - sc_enc_mu.pow(2) - sc_enc_logvar.exp(), dim=1))
            # ==========================================
            # Task 2: pseudo-Bulk domain adaptation
            # ==========================================
            # # Generate a pseudo-Bulk batch
            # pseudo_bulk_linear = generate_batch_dirichlet_pseudobulk(
            #     cell_features, label_indices, cell_type_indices, X_tensor, num_cell_types, used_device
            # )
            # pseudo_bulk_log = torch.log1p(pseudo_bulk_linear)
            # # Check the distribution overlap
            # if batch_idx == 0 and (epoch == 0 or epoch % 10 == 0):
            #     check_distribution_overlap(
            #         cell_features,  # cell_features here should already be the log1p-transformed input
            #         pseudo_bulk_log, 
            #         real_bulk_log,
            #         epoch, 
            #         output_dir
            #     )
            
            # pseudo_bulk_core = pseudo_bulk_log[:, core_indices_tensor] # use core genes only
            # if isinstance(vae, nn.DataParallel):
            #     pb_mu, pb_logvar = vae.module.encode(pseudo_bulk_core, label_one_hot)
            #     pb_z = vae.module.reparameterize(pb_mu, pb_logvar)
            #     pb_recon = vae.module.decode(pb_z, label_one_hot)
            # else:
            #     pb_mu, pb_logvar = vae.encode(pseudo_bulk_core, label_one_hot)
            #     pb_z = vae.reparameterize(pb_mu, pb_logvar)
            #     pb_recon = vae.decode(pb_z, label_one_hot)
                
            # loss_denoise = criterion(pb_recon, cell_features)
            
            # 2.2 MMD loss (Distribution Alignment)
            # Force the Bulk mu distribution to overlap the Single Cell mu distribution
            # Note: this compares the distribution difference between the two batches
            # loss_mmd = compute_mmd_loss(sc_enc_mu, pb_mu)
            X_tensor = X_tensor.to(used_device)
            real_core = cell_features[:, core_indices_tensor]
            recon_core = sc_recon[:, core_indices_tensor]
            loss_corr = compute_gene_correlation_loss(real_core, recon_core)
            # ==========================================
            # total loss
            # ==========================================
            total_loss = lambda_recon * loss_recon + beta * loss_kld + lambda_corr * loss_corr
            
            optimizer.zero_grad()
            total_loss.backward()
            # gradient clipping (prevents gradient explosion, especially early on)
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss_epoch += total_loss.item()
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{epoch_num}], Batch [{batch_idx+1}/{len(dataloader)}], Total Loss: {total_loss.item():.4f}, Recon Loss: {loss_recon.item():.4f}, KLD Loss: {loss_kld.item():.4f}, Corr Loss: {loss_corr.item():.4f}")
            
        train_loss_epoch /= len(dataloader)
        scheduler.step(train_loss_epoch)
        
        # Compute monitoring metrics
        with torch.no_grad():
            # inspect the "activity" of the latent
            curr_std = torch.exp(0.5 * sc_enc_logvar).mean().item()
        print(f"Epoch [{epoch+1}/{epoch_num}], Total Loss: {total_loss.item():.4f}, Recon Loss: {loss_recon.item():.4f}, KLD Loss: {loss_kld.item():.4f}, Corr Loss: {loss_corr.item():.4f}, Latent Std: {curr_std:.4f}, Beta: {beta:.6f}, Lambda_Corr: {lambda_corr:.4f}")
        
        # best-model saving logic (simplified)
        if train_loss_epoch < min_loss:
            min_loss = train_loss_epoch
            best_vae = copy.deepcopy(vae)
            epoch_final = epoch
            
        # periodic save
        if epoch % 10 == 0 and epoch != 0:
            torch.save(vae.state_dict(), os.path.join(output_dir, f'Epoch{epoch}_scvae.pth'))

        # Evaluate training quality during the training process
        if epoch % 20 == 0: #and epoch != 0:
            vae.eval()  # switch to evaluation mode
            print(f"\n--- Epoch {epoch} Evaluation ---")
            
            eval_batch_size = 512 # can use a larger batch at inference time
            eval_dataset = TensorDataset(X_tensor, labels)
            # shuffle=False keeps the order consistent
            eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0)
            
            # containers
            all_mu_list = []
            all_logvar_list = []
            all_labels_list = []
            
            # accumulators used to compute Pseudo-Bulk
            # dict structure: {label_id: [sum_real, count_real, sum_recon, count_recon]}
            # Note: we aggregate in Linear space, so expm1 is required
            pb_stats = {i: {'real_sum': 0, 'real_count': 0, 'recon_sum': 0} for i in range(num_cell_types)}

            with torch.no_grad():
                for batch_x, batch_y in tqdm(eval_loader, desc="Evaluating"):
                    batch_x = batch_x.to(used_device, dtype=torch.float32)
                    batch_y_hot = F.one_hot(batch_y, num_cell_types).float().to(used_device)
                    
                    # 1. Forward pass
                    # Note: encode returns a tuple (mu, logvar)
                    batch_x_core = batch_x[:, core_indices_tensor] 
                    mu, logvar = vae.encode(batch_x_core, batch_y_hot)
                    z = vae.reparameterize(mu, logvar)
                    recon = vae.decode(z, batch_y_hot)
                    
                    # 2. Collect Latent for the subsequent UMAP and Prior computation
                    all_mu_list.append(mu.cpu().numpy())
                    all_logvar_list.append(logvar.cpu().numpy())
                    all_labels_list.append(batch_y.cpu().numpy())
                    
                    # 3. Accumulate Pseudo-Bulk (in Linear space)
                    # This step evaluates whether the aggregated generated data resembles the real Bulk
                    batch_x_linear = torch.expm1(batch_x)
                    recon_linear = torch.expm1(recon)
                    
                    batch_y_np = batch_y.cpu().numpy()
                    for local_idx, label_id in enumerate(batch_y_np):
                        # accumulate Real
                        pb_stats[label_id]['real_sum'] += batch_x_linear[local_idx].cpu().numpy()
                        pb_stats[label_id]['real_count'] += 1
                        # accumulate Recon
                        pb_stats[label_id]['recon_sum'] += recon_linear[local_idx].cpu().numpy()

            # Merge Latent
            all_mu = np.concatenate(all_mu_list, axis=0)
            all_logvar = np.concatenate(all_logvar_list, axis=0)
            all_labels = np.concatenate(all_labels_list, axis=0)

            # ----------------------------------------------------------
            # Metric 1: Pseudo-Bulk correlation (the most important metric)
            # ----------------------------------------------------------
            print("\n>>> Metric 1: Pseudo-Bulk Correlation (Reconstruction vs Real)")
            pcc_list = []
            
            # Also prepare the data for computing "Prior Generation"
            cell_type_mu_logvar = {} # store the priors computed on this validation set, to be saved
            print(f"breed_2_list: {breed_2_list}")
            for label_id in range(num_cell_types):
                stats = pb_stats[label_id]
                
                if stats['real_count'] < 5: continue # too few cells, skip
                
                # A. Compute the real mean expression profile (Ground Truth Profile)
                # Linear Mean -> Log1p (converting back to Log space for correlation is usually more intuitive; Linear works too)
                real_profile_linear = stats['real_sum'] / stats['real_count']
                real_profile_log = np.log1p(real_profile_linear)
                
                # B. Compute the reconstructed mean expression profile
                recon_profile_linear = stats['recon_sum'] / stats['real_count']
                recon_profile_log = np.log1p(recon_profile_linear)
                
                # C. Compute correlation
                # filter out all-zero genes (optional, prevents correlation errors)
                if np.std(real_profile_log) > 1e-9 and np.std(recon_profile_log) > 1e-9:
                    pcc = pearsonr(real_profile_log, recon_profile_log)[0]
                    pcc_list.append(pcc)
                    print(f"  Type {breed_2_list[label_id]:<15}: PCC = {pcc:.4f}")
                else:
                    pcc_list.append(0)
                
                # D. Also save this type's Latent prior (mu, logvar)
                # here mu is the mean Latent over all cells of this type
                mask = all_labels == label_id
                mu_mean = torch.tensor(all_mu[mask].mean(axis=0, keepdims=True))
                logvar_mean = torch.tensor(all_logvar[mask].mean(axis=0, keepdims=True))
                cell_type_mu_logvar[label_id] = (mu_mean, logvar_mean)

            avg_pcc = np.mean(pcc_list) if pcc_list else 0
            print(f"--> Average Pseudo-Bulk PCC: {avg_pcc:.4f}")

            # ----------------------------------------------------------
            # Metric 2: pure generation quality check (Check Prior Quality)
            # ----------------------------------------------------------
            print("\n>>> Metric 2: Pure Generation Quality (Prior -> Decoder vs Real)")
            # This step verifies: if we give the Decoder only a class's mean z and label, can it generate the correct class features?
            gen_pcc_list = []
            
            with torch.no_grad():
                for label_id in range(num_cell_types):
                    if label_id not in cell_type_mu_logvar: continue
                    
                    # Get the real Profile for this class (reusing the computation above)
                    stats = pb_stats[label_id]
                    real_profile_log = np.log1p(stats['real_sum'] / stats['real_count'])
                    
                    # Get the Prior Mu
                    prior_mu = cell_type_mu_logvar[label_id][0].to(used_device)
                    label_vec = F.one_hot(torch.tensor([label_id]), num_cell_types).float().to(used_device)
                    
                    # decode
                    gen_x = vae.decode(prior_mu, label_vec).cpu().numpy().flatten()
                    # Note: assume the decode output is already log1p (or the model output is followed by relu)
                    # If the model output is linear, log1p is needed. Usually the VAE output directly fits the input, and the input is log1p, so the output is log1p too.
                    
                    if np.std(real_profile_log) > 1e-9 and np.std(gen_x) > 1e-9:
                        gen_pcc = pearsonr(real_profile_log, gen_x)[0]
                        gen_pcc_list.append(gen_pcc)
                        print(f"  Type {breed_2_list[label_id]:<15}: Gen PCC = {gen_pcc:.4f}")
                    else:
                        gen_pcc_list.append(0)
            
            avg_gen_pcc = np.mean(gen_pcc_list) if gen_pcc_list else 0
            print(f"--> Average Generation PCC: {avg_gen_pcc:.4f}")

            # write to log
            with open(os.path.join(output_dir, 'metrics.txt'), 'a') as f:
                f.write(f"Epoch {epoch} | PseudoBulk_PCC: {avg_pcc:.4f} | Generation_PCC: {avg_gen_pcc:.4f}\n")

            # ----------------------------------------------------------
            # Metric 3: UMAP visualization of the Latent Space
            # ----------------------------------------------------------
            # downsample for plotting
            if all_mu.shape[0] > 10000:
                idx_plot = np.random.choice(all_mu.shape[0], 10000, replace=False)
                mu_plot = all_mu[idx_plot]
                labels_plot = [breed_2_list[id] for id in all_labels[idx_plot]]
            else:
                mu_plot = all_mu
                labels_plot = [breed_2_list[id] for id in all_labels]

            try:
                adata_latent = sc.AnnData(mu_plot, obs=pd.DataFrame({'Cell_type': labels_plot}))
                sc.pp.neighbors(adata_latent, use_rep='X') # compute neighbors directly on X (i.e. mu), no extra PCA needed
                sc.tl.umap(adata_latent)
                
                # save the figure
                fig = plt.figure(figsize=(10, 8))
                sc.pl.umap(adata_latent, color='Cell_type', title=f'Latent UMAP Epoch {epoch}', 
                          legend_loc='on data', show=False, palette=color_map)
                plt.savefig(os.path.join(output_dir, f'_latent_mu_epoch{epoch}.png'))
                plt.close()
            except Exception as e:
                print(f"UMAP Plotting failed: {e}")

            # restore training mode
            vae.train()

    #     swanlab.log({
    #         "epoch": epoch,
    #         "total_loss": train_loss_epoch,
    #         "recon_loss": loss_recon.item(),
    #         "kl_loss": loss_kld.item(),
    #         "denoise_loss": loss_denoise.item(),
    #         "mmd_loss": loss_mmd.item(),
    #         "Latent_STD": curr_std
    #     })
    # swanlab.finish()
    # Training finished; use the Best VAE to compute statistics
    print(f"Best Loss: {min_loss:.4f} at epoch {epoch_final}")
    # save the final result
    torch.save(best_vae.state_dict(), os.path.join(output_dir, 'scvae_best.pth'))
    # save mu/logvar for all cell types for later use
    print("\nTraining Finished. Computing FINAL Priors from TRAINING SET...")
    
    # 1. switch to the best model
    if best_vae is not None:
        vae = best_vae
    vae.eval()
    
    # 2. create a DataLoader for the training set (no shuffling, the batch size can be larger)
    # Note: here we use the training set X_tensor
    final_dataset = TensorDataset(X_tensor, labels)
    final_loader = DataLoader(final_dataset, batch_size=1024, shuffle=False, num_workers=0)
    
    # 3. containers
    all_mu_list = []
    all_logvar_list = []
    all_labels_list = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(final_loader, desc="Computing Final Priors"):
            batch_x = batch_x.to(used_device, dtype=torch.float32)
            batch_y_hot = F.one_hot(batch_y, num_cell_types).float().to(used_device)
            
            # only Encode is needed
            batch_x_core = batch_x[:, core_indices_tensor]
            mu, logvar = vae.encode(batch_x_core, batch_y_hot)
            
            all_mu_list.append(mu.cpu().numpy())
            all_logvar_list.append(logvar.cpu().numpy())
            all_labels_list.append(batch_y.cpu().numpy())
            
    # 4. merge
    all_mu = np.concatenate(all_mu_list, axis=0)
    all_logvar = np.concatenate(all_logvar_list, axis=0)
    all_labels = np.concatenate(all_labels_list, axis=0)
    
    # 5. compute and save the final cell_type_mu_logvar
    final_cell_type_mu_logvar = {}
    
    print("Final Prior Statistics:")
    for label_id in range(num_cell_types):
        mask = all_labels == label_id
        if mask.sum() == 0:
            print(f"Warning: Cell type {label_id} not found in training set!")
            continue
            
        # compute the mean
        # mu_mean: [1, hidden_dim]
        mu_mean = torch.tensor(all_mu[mask].mean(axis=0, keepdims=True))
        logvar_mean = torch.tensor(all_logvar[mask].mean(axis=0, keepdims=True))
        
        final_cell_type_mu_logvar[label_id] = (mu_mean, logvar_mean)
        print(f"  Type {breed_2_list[label_id]}: n={mask.sum()}")

    # 6. save the final file (overwriting the one generated earlier from the validation set)
    torch.save(final_cell_type_mu_logvar, os.path.join(output_dir, 'cell_type_mu_logvar_best.pt'))
    print("Final priors saved to cell_type_mu_logvar_best.pt")
    return best_vae, final_cell_type_mu_logvar

##############################
### Stage 2: deconvolution prediction ###
##############################
def solve_proportions_with_vae(vae, 
                               Basis_Used,          # [K_types, G_sig] - basis that has already been scaled and sliced
                               Target_Used,         # [G_sig] - target that has already been sliced and (if needed) scaled
                               valid_names,         # list of cell-type names, corresponding to the rows of Basis_Used
                               device, 
                               temperature=0.5,     # temperature coefficient (slightly below 1.0 to encourage sparsity)
                               lambda_entropy=0.1,  # entropy regularization weight
                               lambda_reg=0.01,     # Logits L2 regularization weight
                               lr=0.05,             # learning rate
                               steps=500):          # number of iterations
    """
    Use the Basis generated by the VAE Decoder to solve for the cell proportions via gradient descent.
    This function assumes that the input Basis_Used and Target_Used have already had
    Signature Genes slicing and global scaling correction applied before the call.
    """
    
    num_classes_valid = Basis_Used.shape[0]
    
    # 1. initialize the optimization variables
    # Logits initialized to 0 (i.e. uniform initial probabilities)
    logits = torch.zeros(num_classes_valid, requires_grad=True, device=device)
    
    # use the Adam optimizer
    optimizer = torch.optim.Adam([logits], lr=lr)
    
    # 2. optimization loop
    for step in range(steps):
        optimizer.zero_grad()
        
        # compute the proportions (Softmax)
        probs = F.softmax(logits / temperature, dim=0)
        
        # mix to obtain the Pseudo Bulk (Linear Space)
        # [K, G_sig] * [K, 1] -> sum -> [G_sig]
        # Note: Basis_Used is already in linear space and globally scaled
        pseudo = torch.sum(probs.view(-1, 1) * Basis_Used, dim=0)
        
        # --- Loss 1: Pearson Correlation ---
        vx = pseudo - pseudo.mean()
        vy = Target_Used - Target_Used.mean()
        
        pearson = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
        loss_corr = 1.0 - pearson
        
        # --- Loss 2: Entropy Regularization ---
        # minimize -H (i.e. encourage a sparse distribution)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        loss_entropy = -entropy 
        
        # --- Loss 3: L2 Regularization on Logits ---
        loss_l2 = torch.sum(logits ** 2)
        
        # total Loss
        total_loss = loss_corr + lambda_entropy * loss_entropy + lambda_reg * loss_l2
        
        total_loss.backward()
        optimizer.step()
        
    # 3. output the results
    final_probs = F.softmax(logits / temperature, dim=0).detach().cpu().numpy()
    
    # filter out tiny values and renormalize
    final_probs[final_probs < 1e-4] = 0
    final_probs = final_probs / (final_probs.sum() + 1e-8)
    
    result = {valid_names[idx]: float(prob) for idx, prob in enumerate(final_probs)}
        
    return result

def get_high_expression_mask(expression_tensor, top_percent=0.02):
    """
    Identify a mask of highly expressed genes.
    Input: [N_genes] or [Batch, N_genes]
    Output: a boolean mask of [N_genes] (True means keep, False means remove)
    """
    # if it is a Batch, take the mean expression
    if expression_tensor.dim() > 1:
        mean_expr = expression_tensor.mean(dim=0)
    else:
        mean_expr = expression_tensor
        
    # compute the threshold
    n_genes = mean_expr.shape[0]
    k = int(n_genes * top_percent)
    
    # find the Top K values
    # topk returns (values, indices)
    _, top_indices = torch.topk(mean_expr, k)
    
    # create the mask (all True by default)
    mask = torch.ones(n_genes, dtype=torch.bool, device=expression_tensor.device)
    # set the Top K positions to False
    mask[top_indices] = False
    
    return mask, top_indices

def optimize_z_and_generate_final(
    vae, 
    real_bulk_tensor,       # Bulk Tensor in Log1p format [N_samples, G_full]
    sample_names, 
    mapping_dict, 
    priors,                 # trained Prior (Mean, Var)
    device, 
    output_dir,
    sig_indices,            # integer indices of the Signature Genes
    # --- hyperparameters ---
    total_cells=8000,       # total number of cells generated per sample
    scaling_factor=None,    # gene scaling factor of dimension [G_full] (global technical bias)
    # Phase 1 parameters (unchanged, but must be passed in)
    lambda_entropy=0.1,     
    temperature=0.5,        
    # Phase 2A parameters (Scaler optimization)
    lr_scaler=1.0,          # Scaler learning rate (can be high)
    steps_scaler=500,       # number of Scaler optimization steps
    lambda_reg_scaler=1.0,  # Scaler L2 regularization weight
    # Phase 2B parameters (Delta Z optimization)
    lr_z=0.05,              # learning rate for Z optimization
    steps_z=1500,           # number of steps for Z optimization
    lambda_reg_z=0.1,       # L2 regularization on Z (prevents drifting too far from the Prior)
    lambda_mse=0.1,          # auxiliary MSE loss weight (Linear Space)
    common_genes=None
    ):
    """
    Two-stage generation pipeline:
    1. Use Signature Genes to determine the cell proportions (Robust Proportions).
    2. Staged optimization: (A) learn the global gene scaling factor; (B) optimize the Latent Delta Z.
    """
    
    vae.eval().to(device)
    for p in vae.parameters():
        p.requires_grad = False
        
    num_classes = len(mapping_dict)
    generated_cells = []
    generated_meta = []
    n_genes = real_bulk_tensor.shape[1]

    if scaling_factor is not None:
        init_scale = scaling_factor.clone().to(device)
    else:
        init_scale = torch.ones(n_genes, device=device)
        
    print(f"\n>>> Starting Two-Stage Generation for {len(sample_names)} samples...")
    
    # reverse mapping ID -> Name
    id2name = {v: k for k, v in mapping_dict.items()}

    # pre-build the VAE basis (Basis Construction)
    basis_vectors = []
    valid_ids = []
    valid_names = []
    
    sorted_ids = sorted(priors.keys())
    with torch.no_grad():
        for type_id in sorted_ids:
            prior_mu = priors[type_id][0].to(device)
            label = F.one_hot(torch.tensor([type_id]), num_classes).float().to(device)
            recon_log = vae.decode(prior_mu, label)
            recon_linear = torch.expm1(recon_log) 
            basis_vectors.append(recon_linear)
            valid_ids.append(type_id)
            valid_names.append(id2name[type_id])
    
    if not basis_vectors: return None
    Basis_Full_Linear = torch.cat(basis_vectors, dim=0) # [K_types, G_full]

    # iterate over each sample
    for i, sample_name in tqdm(enumerate(sample_names), desc="Processing Samples"):
        
        # Target Log1p (all genes)
        target_bulk_log = real_bulk_tensor[i].to(device)
        # Target Linear (all genes)
        target_bulk_linear = torch.expm1(target_bulk_log) 
        
        # =====================================================
        # Phase 1: solve the cell proportions (using Signature Genes)
        # =====================================================
        # 1.1 get the signature-gene basis and target (Linear Space)
        if not torch.is_tensor(sig_indices):
            sig_indices_tensor = torch.tensor(sig_indices, dtype=torch.long, device=device)
        else:
            sig_indices_tensor = sig_indices.to(device)

        Basis_Sig = Basis_Full_Linear[:, sig_indices_tensor]
        Target_Sig = target_bulk_linear[sig_indices_tensor]
        Scaling_Sig = init_scale[sig_indices_tensor] 

        # apply global scaling (correct technical bias)
        # key step: complete the global scaling correction of the basis and target before calling the proportion optimizer
        Basis_Sig_Scaled = Basis_Sig * Scaling_Sig
        Target_Sig_Scaled = Target_Sig * Scaling_Sig # the target must also apply the scaling factor to stay consistent
        
        # 1.2 optimize the proportions
        fractions = solve_proportions_with_vae(
            vae=vae, 
            Basis_Used=Basis_Sig_Scaled,     # pass in the preprocessed basis
            Target_Used=Target_Sig_Scaled,   # pass in the preprocessed target
            valid_names=valid_names,         # list of cell-type names
            device=device,
            temperature=temperature,
            lambda_entropy=lambda_entropy, 
            lambda_reg=0.01,
            lr=0.05,
            steps=500
        )
        
        # =====================================================
        # Phase 2: optimize Latent Z and the Scaler (using All Genes)
        # =====================================================
        
        # 2.1 build the initial Z and Labels (same as the original code)
        z_list = []
        label_list = []
        prior_list = [] 
        type_names = []
        sample_cells = total_cells // len(sample_names) if len(sample_names) > 0 else 0
        print(f"\n>>> Sample {sample_name}: Generating ~{sample_cells} cells based on estimated fractions...")
        for t_name, t_id in mapping_dict.items():
            count = int(fractions.get(t_name, 0) * sample_cells)
            if count <= 0 or t_id not in priors: continue
            
            p_mu = priors[t_id][0].to(device)
            # z_chunk = p_mu.repeat(count, 1)
            p_logvar = priors[t_id][1].to(device) # get the variance
            p_std = torch.exp(0.5 * p_logvar)
            eps = torch.randn(count, p_mu.shape[1], device=device)
            z_chunk = p_mu + eps * p_std
            label_chunk = F.one_hot(torch.tensor([t_id]*count), num_classes).float().to(device)
            
            z_list.append(z_chunk)
            prior_list.append(z_chunk) 
            label_list.append(label_chunk)
            type_names.extend([t_name] * count)
            
        if not z_list:
            continue
            
        Prior_Z = torch.cat(prior_list, dim=0).detach()
        Labels = torch.cat(label_list, dim=0).detach()
        
        # define the optimization variables Delta and Scaler
        Delta_Z = torch.zeros_like(Prior_Z, requires_grad=True, device=device)
        # optimize log(scale)
        log_gene_scaler = torch.nn.Parameter(torch.log(init_scale + 1e-6).clone()) 
        
        # --- Phase 2A: optimize the Scaler (freeze Delta_Z) ---
        print(f"  > P2A: Optimizing Scaler...")
        optimizer_scaler = torch.optim.Adam([log_gene_scaler], lr=lr_scaler)
        
        for step_a in range(steps_scaler):
            optimizer_scaler.zero_grad()
            
            # decode using Prior Z (equivalent to optimizing only the global correction)
            with torch.no_grad():
                recon_log_prior = vae.decode(Prior_Z, Labels)
                recon_linear_prior = torch.expm1(recon_log_prior)
                
            current_scaler = torch.exp(log_gene_scaler)
            pseudo_bulk_adjusted = torch.sum(recon_linear_prior, dim=0) * current_scaler
            
            # Loss 1: Pearson Loss (Log Space)
            Pseudo_Log = torch.log1p(pseudo_bulk_adjusted + 1e-6)
            
            vx = Pseudo_Log - Pseudo_Log.mean()
            vy = target_bulk_log - target_bulk_log.mean()
            pearson = torch.sum(vx * vy) / (torch.sqrt((vx**2).sum()) * torch.sqrt((vy**2).sum()) + 1e-8)
            loss_corr = 1.0 - pearson
            
            # Loss 2: Scaler Regularization (prevents the Scaler from drifting away from the global Init)
            loss_reg_scaler_a = torch.mean((current_scaler - init_scale)**2)
            
            total_loss_a = loss_corr + lambda_reg_scaler * loss_reg_scaler_a
            
            total_loss_a.backward()
            optimizer_scaler.step()

            if step_a % 100 == 0:
                 tqdm.write(f"    P2A Step {step_a}: Corr={pearson.item():.4f}, Scale_Reg={loss_reg_scaler_a.item():.4f}")

        # get the final Scaler
        Final_Scaler = torch.exp(log_gene_scaler).detach()
        print(f"  > P2A Finished. Final Scaler Corr: {pearson.item():.4f}")
        
        # --- Phase 2B: optimize Delta Z (freeze the Scaler) ---
        print(f"  > P2B: Optimizing Delta Z...")
        # freeze the Scaler parameter
        log_gene_scaler.requires_grad = False
        
        optimizer_z = torch.optim.Adam([Delta_Z], lr=lr_z)
        
        for step_b in range(steps_z):
            optimizer_z.zero_grad()
            Current_Z = Prior_Z + Delta_Z
            
            # Decode -> Linear Space
            recon_log = vae.decode(Current_Z, Labels)
            recon_linear = torch.expm1(recon_log)
            
            # aggregate & apply the fixed Scaler
            pseudo_bulk_raw = torch.sum(recon_linear, dim=0)
            pseudo_bulk_adjusted = pseudo_bulk_raw * Final_Scaler
            
            # 1. Pearson Loss (Log Space, all genes)
            Pseudo_Log = torch.log1p(pseudo_bulk_adjusted + 1e-6)
            vx = Pseudo_Log - Pseudo_Log.mean()
            vy = target_bulk_log - target_bulk_log.mean()
            pearson = torch.sum(vx * vy) / (torch.sqrt((vx**2).sum()) * torch.sqrt((vy**2).sum()) + 1e-8)
            loss_corr = 1.0 - pearson
            
            # 2. MSE Loss (Linear Space, ensures Read Counts alignment)
            # Since Target_Linear can be very large, the MSE weight lambda_mse must be set very small
            loss_mse = F.mse_loss(pseudo_bulk_adjusted, target_bulk_linear)
            
            # 3. Z Regularization (prevents Z from drifting)
            loss_reg_z = torch.mean(Delta_Z ** 2)
            
            total_loss_b = loss_corr + lambda_mse * loss_mse + lambda_reg_z * loss_reg_z
            
            total_loss_b.backward()
            # gradient clipping (prevents Delta Z from exploding early in optimization)
            torch.nn.utils.clip_grad_norm_([Delta_Z], max_norm=1.0) 
            optimizer_z.step()
            
            if step_b % 100 == 0:
                 tqdm.write(f"    P2B Step {step_b}: Corr={pearson.item():.4f}, MSE={loss_mse.item():.2f}, Z_Reg={loss_reg_z.item():.4f}")

        print(f"  > Optimized Delta Z. Final Corr (All Genes): {pearson.item():.4f}")

        # =====================================================
        # Phase 3: generate and collect the results
        # =====================================================
        with torch.no_grad():
            Final_Z = Prior_Z + Delta_Z
            # the finally generated cells should be in Log1p format
            Final_Cells_Log = vae.decode(Final_Z, Labels)
            
            # In theory, the VAE Decode output should already be Log1p and need no further scaling.
            # If applying the Scaler, it should be applied in Linear space and then converted back to Log space with Log1p.
            # However, since optimizing Z already accounts for the Scaler's effect, we directly use Final_Cells_Log as the final output.
            
            generated_cells.append(Final_Cells_Log.cpu().numpy())
            
            for t_name in type_names:
                generated_meta.append((sample_name, t_name))
        
    # assemble AnnData
    if not generated_cells: return None
    
    X_all = np.vstack(generated_cells)
    obs_df = pd.DataFrame(generated_meta, columns=['Sample', 'Cell_type'])
    obs_df.index = [f"Cell_{i}" for i in range(len(obs_df))]
    obs_df.to_csv(os.path.join(output_dir, f'generated_cells_metadata.csv'))
    
    # ensure AnnData contains the correct gene names
    adata_gen = sc.AnnData(X=X_all, obs=obs_df, var=pd.DataFrame(index=common_genes))
    
    # ... (the plotting part is the same as the original code)
    print("绘制生成数据的细胞类型分布图...")
    gen_type_counts = obs_df['Cell_type'].value_counts()
    plot_smart_donut(
        type_counts=gen_type_counts,
        title=f"Generated Data Composition (Total: {len(obs_df)})",
        save_path=os.path.join(output_dir, f'generated_data_composition_donut_{sample_name}.png')
    )
    gen_type_counts.to_csv(os.path.join(output_dir, f'generated_data_composition_counts_{sample_name}.csv'))
    return adata_gen, Final_Z, obs_df

def get_sample_indices_from_bulk(real_bulk_path, target_samples):
    """
    Read real_bulk.csv and return the indices of the target samples, along with the list of sample names actually found.
    """
    # read the Bulk data
    try:
        # assume TSV format, with the index in the first column
        real_bulk_df = pd.read_csv(real_bulk_path, sep='\t', index_col=0).T
    except Exception as e:
        print(f"Error reading {real_bulk_path}: {e}")
        return [], []
    
    # get all sample names
    all_sample_names = real_bulk_df.index.tolist()
    print(f"Total samples in bulk data: {len(all_sample_names)}")
    
    sample_indices = []
    valid_target_samples = [] # sample names actually found
    
    for sample in target_samples:
        try:
            idx = all_sample_names.index(sample)
            sample_indices.append(idx)
            valid_target_samples.append(sample)
        except ValueError:
            print(f"Warning: Sample '{sample}' not found in bulk data. Skipping.")
    
    if not sample_indices:
        print("Error: No target samples found in bulk data.")
    else:
        print(f"Found {len(sample_indices)} samples out of {len(target_samples)} targets.")
        print(f"Valid samples: {valid_target_samples}")
    
    return sample_indices, valid_target_samples

def plot_marker_expression_comparison(adata_real, adata_gen, 
                                      group_col_real='labels', 
                                      group_col_gen='Cell_type', # assumed label column name of the generated data
                                      top_n=5, 
                                      output_dir='./'):
    """
    Compare the expression levels of Marker genes between the real and generated data.
    Draw two paired heatmaps.
    """
    print("\n--- Comparing Marker Gene Expression (Real vs Gen) ---")
    
    # 1. Find the Top Markers of the real data
    # We use the real data as reference and check whether the generated data reproduces these Markers
    # assume adata_real has already been log1p-processed
    # create a copy to avoid modifying the original data
    real_temp = adata_real.copy()
    
    # compute Rank Genes (if not already computed)
    # use t-test or wilcoxon
    if 'rank_genes_groups' not in real_temp.uns:
        sc.tl.rank_genes_groups(real_temp, groupby=group_col_real, method='t-test_overestim_var')
    
    # extract the Marker list
    # format: {cell_type: [gene1, gene2...]}
    groups = real_temp.obs[group_col_real].unique().tolist()
    # sort to keep the plotting order consistent
    groups = sorted([str(g) for g in groups])
    
    marker_dict = {}
    var_names = set()
    
    for g in groups:
        # get the top_n genes of this group
        try:
            genes = sc.get.rank_genes_groups_df(real_temp, group=g).head(top_n)['names'].tolist()
            marker_dict[g] = genes
            var_names.update(genes)
        except KeyError:
            print(f"Warning: Group {g} not found in rank_genes_groups.")
            
    gene_list = [] # gene order used for plotting (arranged by cell type)
    for g in groups:
        if g in marker_dict:
            gene_list.extend(marker_dict[g])
            
    # deduplicate while preserving order (a bit tricky; for simplicity we allow a few duplicates, or use pd.unique)
    # here we allow duplicates, because different cells may share the same marker, and showing it repeatedly in the heatmap aids comparison
    
    # 2. Prepare the plotting data matrix
    # We need to compute the [mean expression] of these genes for each cell type
    
    # 2.1 real data matrix
    # [Cell_Types, Genes]
    real_mean_df = pd.DataFrame(index=groups, columns=gene_list)
    
    for g in groups:
        # find the cells of this type
        cells = real_temp[real_temp.obs[group_col_real] == g, gene_list]
        # compute the mean
        mean_expr = np.mean(cells.X, axis=0)
        # if it is a sparse matrix
        if hasattr(mean_expr, 'A1'): mean_expr = mean_expr.A1
        elif hasattr(mean_expr, 'toarray'): mean_expr = mean_expr.toarray().flatten()
        
        real_mean_df.loc[g] = mean_expr

    # 2.2 generated data matrix
    # ensure the genes exist in the generated data
    valid_genes = [gene for gene in gene_list if gene in adata_gen.var_names]
    if len(valid_genes) < len(gene_list):
        print(f"Warning: {len(gene_list) - len(valid_genes)} markers missing in generated data.")
    
    gen_mean_df = pd.DataFrame(index=groups, columns=valid_genes)
    
    # ensure the label column of the generated data exists
    if group_col_gen not in adata_gen.obs:
        print(f"Error: Column {group_col_gen} not found in adata_gen.obs")
        return

    for g in groups:
        # find the cells of this type
        # Note: the generated data may not contain all real-data types (if a type was filtered out)
        if g not in adata_gen.obs[group_col_gen].values:
            # print(f"Note: Cell type {g} not found in Generated Data.")
            gen_mean_df.loc[g] = 0 # fill 0
            continue
            
        cells = adata_gen[adata_gen.obs[group_col_gen] == g, valid_genes]
        mean_expr = np.mean(cells.X, axis=0)
        gen_mean_df.loc[g] = mean_expr

    # 3. Normalization (Z-score Scaling)
    # normalize by gene (column) to remove absolute-expression differences and focus on specificity patterns
    # the scale function operates on columns by default (axis=0)
    
    # for nicer plotting, convert the data to float
    real_plot_data = real_mean_df.astype(float)
    gen_plot_data = gen_mean_df.astype(float)
    
    # row normalization (compares this gene's level across different cells)
    # or column normalization (compares different genes within this cell)
    # DotPlot/Heatmap conventionally Z-score by Gene (column)
    
    # implement it manually to handle all-zero rows/columns
    def safe_zscore(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or df.shape[1] == 0 or df.shape[0] == 0:
            # return an all-zero DataFrame of the same shape
            return pd.DataFrame(0, index=df.index, columns=df.columns)
        # normal z-score
        z = scale(df, axis=0)          # axis=0 standardizes by column
        return pd.DataFrame(z, index=df.index, columns=df.columns).fillna(0)

    real_norm = safe_zscore(real_plot_data)
    gen_norm = safe_zscore(gen_plot_data)
    real_norm.to_csv(os.path.join(output_dir, 'real_marker_expression_zscore.csv'))
    gen_norm.to_csv(os.path.join(output_dir, 'gen_marker_expression_zscore.csv'))
    
    zero_rows = gen_norm.index[(gen_norm == 0).all(axis=1)]
    print(f"Removing {len(zero_rows)} rows with all zeros.")
    # 2. drop these rows from both DataFrames
    real_norm = real_norm.drop(zero_rows)
    gen_norm  = gen_norm.drop(zero_rows)
    real_norm = real_norm[~real_norm.index.isna()]
    gen_norm = gen_norm[~gen_norm.index.isna()]
    
    # clip extreme values to prevent the heatmap colors from being skewed by individual points
    vmax = 2.5
    vmin = -1.5
    
    # 4. draw the paired heatmaps
    fig, axes = plt.subplots(
    1, 2,
    figsize=(13, 7),
    gridspec_kw={"width_ratios": [1, 1]}  # force equal width left and right
    )
    # Panel 1: Real
    sns.heatmap(real_norm, ax=axes[0], cmap='RdBu_r', vmin=vmin, vmax=vmax, 
                xticklabels=False, cbar=False)
    axes[0].set_title("Real Data (Top Markers)", fontsize=14)
    axes[0].set_xlabel("Marker Genes (Grouped by Type)", fontsize=12)
    axes[0].set_ylabel("Cell Types", fontsize=13)
    axes[0].tick_params(axis="y", labelsize=11)
    
    # Panel 2: Generated
    sns.heatmap(gen_norm, ax=axes[1], cmap='RdBu_r', vmin=vmin, vmax=vmax, 
                xticklabels=False, cbar=True)
    axes[1].set_title("Generated Data (Same Markers)", fontsize=14)
    axes[1].set_xlabel("Marker Genes", fontsize=12)
    axes[1].tick_params(axis="y",left=False,labelleft=False)
    
    # plt.tight_layout()
    plt.subplots_adjust(
    wspace=0.05,   # horizontal spacing; 0.03-0.08 is common for journals
    left=0.18,
    right=0.92,
    top=0.90,
    bottom=0.10
)
    save_path = os.path.join(output_dir, 'comparison_marker_heatmap.png')
    plt.savefig(save_path, dpi=600)
    plt.close()
    print(f"Marker comparison heatmap saved to {save_path}")
    
    # 5. Optional: compute the correlation between the two matrices (quantitative evaluation)
    # flatten the two matrices to compute Pearson
    # only compare genes present in both
    common_cols = real_norm.columns.intersection(gen_norm.columns)
    if len(common_cols) > 0:
        flat_real = real_norm[common_cols].values.flatten()
        flat_gen = gen_norm[common_cols].values.flatten()
        from scipy.stats import pearsonr
        corr, _ = pearsonr(flat_real, flat_gen)
        print(f"Overall Pattern Consistency (Correlation): {corr:.4f}")
        
        # save the correlation to a file
        with open(os.path.join(output_dir, 'marker_consistency_score.txt'), 'w') as f:
            f.write(f"Marker Pattern Correlation: {corr:.4f}\n")
            
def compare_gene_coexpression_networks(
    adata_real, 
    adata_gen_ours, 
    adata_gen_ablation, 
    output_dir, 
    n_top_genes=500, 
    specific_genes=None
):
    """
    Compare the gene co-expression networks of the three datasets: Real, Ours, and Ablation.
    
    Parameters:
        adata_real: real single-cell data (AnnData)
        adata_gen_ours: data generated by the Attention model
        adata_gen_ablation: data generated by the MLP model (no Attention)
        output_dir: path to save the figure
        n_top_genes: if specific_genes is not specified, how many HVGs to use for the computation
        specific_genes: (optional) list of genes; if provided, only the network of these genes is computed
    """
    print("--- Starting Gene Co-expression Network Comparison ---")
    
    # 1. Determine the gene set to analyze
    # ensure the genes exist in all three datasets
    common_vars = list(set(adata_real.var_names) & set(adata_gen_ours.var_names) & set(adata_gen_ablation.var_names))
    
    if specific_genes is not None:
        target_genes = [g for g in specific_genes if g in common_vars]
        print(f"Using {len(target_genes)} specific genes provided by user.")
    else:
        # if no specific genes are given, use the HVGs of the Real data
        print(f"Selecting top {n_top_genes} HVGs from Real data...")
        temp_adata = adata_real[:, common_vars].copy()
        sc.pp.highly_variable_genes(temp_adata, n_top_genes=n_top_genes)
        target_genes = temp_adata.var[temp_adata.var['highly_variable']].index.tolist()
        print(f"Selected {len(target_genes)} HVGs.")

    # 2. Extract the expression matrix (Cells x Genes)
    # Note: co-expression is usually computed on Log1p-transformed data
    def get_matrix(adata, genes):
        sub = adata[:, genes]
        if isinstance(sub.X, np.ndarray):
            return sub.X
        else:
            return sub.X.toarray()

    X_real = get_matrix(adata_real, target_genes)
    X_ours = get_matrix(adata_gen_ours, target_genes)
    X_ablation = get_matrix(adata_gen_ablation, target_genes)

    # 3. Compute the gene-gene correlation matrix (Genes x Genes)
    # rowvar=False means each column is a variable (gene)
    # returned matrix shape: [n_genes, n_genes]
    print("Computing correlation matrices...")
    corr_real = np.corrcoef(X_real, rowvar=False)
    corr_ours = np.corrcoef(X_ours, rowvar=False)
    corr_ablation = np.corrcoef(X_ablation, rowvar=False)
    
    # handle possible NaN (if a gene's expression is all zeros)
    corr_real = np.nan_to_num(corr_real)
    corr_ours = np.nan_to_num(corr_ours)
    corr_ablation = np.nan_to_num(corr_ablation)

    # 4. Quantitative evaluation: matrix similarity
    # Method A: Pearson correlation of the flattened matrices (approximate Mantel Test)
    # measures "if A and B are positively correlated in the real data, are they also positively correlated in the generated data?"
    score_ours_pcc = pearsonr(corr_real.flatten(), corr_ours.flatten())[0]
    score_ablation_pcc = pearsonr(corr_real.flatten(), corr_ablation.flatten())[0]
    
    # Method B: matrix distance (Frobenius Norm)
    # measures the absolute error; smaller is better
    dist_ours = np.linalg.norm(corr_real - corr_ours)
    dist_ablation = np.linalg.norm(corr_real - corr_ablation)

    print(f"\n[Quantitative Results]")
    print(f"Matrix Similarity (PCC) [Higher is Better]:")
    print(f"  Ours (Attention): {score_ours_pcc:.4f}")
    print(f"  Ablation (MLP)  : {score_ablation_pcc:.4f}")
    print(f"Matrix Distance (L2 Norm) [Lower is Better]:")
    print(f"  Ours (Attention): {dist_ours:.4f}")
    print(f"  Ablation (MLP)  : {dist_ablation:.4f}")

    # 5. Visualization
    # for a nicer figure, cluster the genes so the red blocks group together
    # use seaborn's clustermap to obtain the clustered indices
    print("Plotting heatmaps...")
    g = sns.clustermap(pd.DataFrame(corr_real), cmap='vlag', center=0)
    reordered_idx = g.dendrogram_row.reordered_ind
    plt.close() # do not display this intermediate figure

    # reorder the three matrices by the Real data's clustering order for an intuitive comparison
    def reorder_mat(mat, idx):
        return mat[idx, :][:, idx]

    corr_real_sorted = reorder_mat(corr_real, reordered_idx)
    corr_ours_sorted = reorder_mat(corr_ours, reordered_idx)
    corr_ablation_sorted = reorder_mat(corr_ablation, reordered_idx)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # common plotting parameters
    heatmap_kwargs = {'cmap': 'RdBu_r', 'center': 0, 'vmin': -0.8, 'vmax': 0.8, 'cbar': True, 'xticklabels': False, 'yticklabels': False}

    sns.heatmap(corr_real_sorted, ax=axes[0], **heatmap_kwargs)
    axes[0].set_title("Ground Truth (Real)\nGene Co-expression")

    sns.heatmap(corr_ours_sorted, ax=axes[1], **heatmap_kwargs)
    axes[1].set_title(f"Ours (Attention)\nPCC={score_ours_pcc:.3f}, Dist={dist_ours:.2f}")

    sns.heatmap(corr_ablation_sorted, ax=axes[2], **heatmap_kwargs)
    axes[2].set_title(f"Ablation (MLP)\nPCC={score_ablation_pcc:.3f}, Dist={dist_ablation:.2f}")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "Gene_Network_Comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved to {save_path}")

    return {
        'ours_pcc': score_ours_pcc, 
        'ablation_pcc': score_ablation_pcc,
        'ours_dist': dist_ours,
        'ablation_dist': dist_ablation
    }
    
def evaluate_generated_data(X_tensor: torch.Tensor, labels: torch.Tensor, 
                           adata_gen: ad.AnnData, # <--- receives AnnData
                           breed_2_list: list, 
                           common_genes: list, 
                           color_map: dict,
                           output_dir: str,
                           real_bulk_tensor=None,
                           top_percent_remove=0.02) -> None:
    """
    Evaluate the generated data.
    X_tensor, labels: real single-cell data (Tensor)
    adata_gen: generated single-cell data (AnnData)
    """
    set_seed(seed)
    print("开始评估生成结果...")
    
    # # 1. Compute the Mask (based on Bulk)
    # if real_bulk_tensor is not None:
    #     # real_bulk_tensor: [N_samples, N_genes]
    #     # compute the mean Bulk expression across all samples
    #     bulk_mean = real_bulk_tensor.mean(dim=0).cpu().numpy() # [12220]
        
    #     # find the Top K
    #     n_genes = bulk_mean.shape[0]
    #     k = int(n_genes * top_percent_remove)
        
    #     # argsort is ascending; the last k are the largest
    #     top_indices = np.argsort(bulk_mean)[-k:]
        
    #     keep_mask = np.ones(n_genes, dtype=bool)
    #     keep_mask[top_indices] = False
        
    #     print(f"Evaluation: Masking Top {k} genes based on BULK expression.")
    # else:
    #     # if no Bulk is passed, fall back to keeping all genes or basing it on SC
    #     keep_mask = np.ones(X_tensor.shape[1], dtype=bool)
    
    
    # 1. Prepare the real data as Numpy
    x_real = X_tensor.numpy()
    labels_real = labels.numpy()
    print(f"真实数据形状: {x_real.shape}, 生成数据形状: {adata_gen.X.shape}")
    
    # 2. Prepare the generated data as Numpy
    x_gen = adata_gen.X
    
    # mask = ~np.isin(common_genes, ['ACTA2', 'TAGLN', 'COL1A1', 'MALAT1'])
    # x_gen = x_gen[:, mask]
    # x_real = x_real[:, mask]
    
    # === [Core code that fixes the KeyError] ===
    # Map the Cell_type strings back to integer indices to align with labels_real
    if 'Label_Idx' in adata_gen.obs:
        labels_gen = adata_gen.obs['Label_Idx'].values
    elif 'Cell_type' in adata_gen.obs:
        # create the mapping dict: { 'T cell': 0, 'B cell': 1, ... }
        name_to_idx = {name: i for i, name in enumerate(breed_2_list)}
        # map
        try:
            labels_gen = adata_gen.obs['Cell_type'].map(name_to_idx).astype(int).values
        except ValueError as e:
            print("Error mapping cell types to indices. Please check if breed_2_list matches adata_gen.obs['Cell_type']")
            raise e
    else:
        raise KeyError("adata_gen.obs must contain 'Cell_type' or 'Label_Idx'")
    # ==================================
    
    # ensure the gene dimensions are aligned
    if x_gen.shape[1] != x_real.shape[1]:
        raise ValueError(f"Gene dimension mismatch! Real: {x_real.shape[1]}, Gen: {x_gen.shape[1]}")

    # 3. Data diagnosis (Sanity Check)
    diagnose_label_idx = 0
    mask_gen = labels_gen == diagnose_label_idx
    if mask_gen.sum() > 0:
        gen_subset = x_gen[mask_gen]
        print(f"\n--- Diagnosis for Cell Type ID {diagnose_label_idx} ({breed_2_list[diagnose_label_idx]}) ---")
        print(f"Mean: {gen_subset.mean():.4f}, Max: {gen_subset.max():.4f}")
        # check the variance to detect Mode Collapse
        avg_var = np.mean(np.var(gen_subset, axis=0))
        print(f"Average Variance per gene: {avg_var:.6f}")
        if avg_var < 1e-6:
            print("WARNING: Variance is extremely low! Mode Collapse detected.")
    
    # --- 4. Compute Pearson Correlation (per Cell Type) ---
    # print("\nCalculating Pearson correlation per cell type...")
    pearson_list = []
    
    for label_idx, label_name in enumerate(breed_2_list):
        mask_real = labels_real == label_idx
        mask_gen = labels_gen == label_idx
        
        # skip a cell type if it has too few cells
        if mask_real.sum() < 1 or mask_gen.sum() < 1:
            print(f"Skipping {label_name}: insufficient cells (Real={mask_real.sum()}, Gen={mask_gen.sum()})")
            pearson_list.append(np.nan)
            continue
            
        # compute the Centroid (mean expression profile)
        mean_real = x_real[mask_real].mean(axis=0)
        mean_gen = x_gen[mask_gen].mean(axis=0)
        # mean_real_filtered = mean_real[keep_mask]
        # mean_gen_filtered = mean_gen[keep_mask]
        
        # handle the all-zero case to prevent errors
        if np.std(mean_real) < 1e-9 or np.std(mean_gen) < 1e-9:
            corr = 0.0
        else:
            # corr, _ = pearsonr(mean_real, mean_gen)
            corr, _ = pearsonr(mean_real, mean_gen)
            
        pearson_list.append(corr)
        print(f"Pearson for {label_name}: {corr:.4f}, Real n={mask_real.sum()}, Gen n={mask_gen.sum()}, for {mean_real.shape[0]} genes")
    print(f"Mean Pearson Correlation: {np.nanmean(pearson_list):.4f}")
        
    pd.DataFrame({'Cell_type': breed_2_list, 'Pearson': pearson_list}).to_csv(os.path.join(output_dir, '1_eval_pearson_correlation.csv'))
    
    # --- 5. Compute MSE (per Cell Type) ---
    # print("\nCalculating MSE per cell type...")
    mse_list = []
    
    for label_idx, label_name in enumerate(breed_2_list):
        mask_real = labels_real == label_idx
        mask_gen = labels_gen == label_idx
        
        if mask_real.sum() < 2 or mask_gen.sum() < 2:
            mse_list.append(np.nan)
            continue
            
        mean_real = x_real[mask_real].mean(axis=0)
        mean_gen = x_gen[mask_gen].mean(axis=0)
        
        mse = np.mean((mean_real - mean_gen) ** 2)
        mse_list.append(mse)
        
    pd.DataFrame({'Cell_Type': breed_2_list, 'MSE': mse_list}).to_csv(os.path.join(output_dir, '2_eval_mse_per_cell_type.csv'))
    
    # --- 6. UMAP visualization (Real vs Generated) ---
    print("\nPreparing UMAP visualization...")
    adata_X = adata_gen.copy()
    # since the data is already in normalized Log space, run PCA directly
    sc.tl.pca(adata_X, svd_solver='arpack', n_comps=50) 
    sc.pp.neighbors(adata_X, n_neighbors=15, n_pcs=50)
    sc.tl.umap(adata_X, min_dist=0.5, spread=1.0)
    
    # plot the UMAP colored by Sample and Cell Type
    fig_X, axes_X = plt.subplots(1, 2, figsize=(18, 8))
    
    # Panel 1: colored by the VAE-predicted Cell_type label (checks whether the generated data preserves the predicted-label structure)
    sc.pl.umap(adata_X, color='Cell_type', title='Generated X UMAP: VAE Predicted Type', 
               ax=axes_X[0], show=False)
               
    # Panel 2: colored by Sample (checks whether individual differences are captured)
    sc.pl.umap(adata_X, color='Sample', title='Generated X UMAP: Sample', 
               ax=axes_X[1], show=False)
               
    plt.tight_layout()
    umap_X_path = os.path.join(output_dir, '_generated_X_umap_comparison.png')
    plt.savefig(umap_X_path, dpi=300)
    plt.close(fig_X)
    print(f"Generated X UMAP saved to {umap_X_path}")
    
    
    # # downsample to prevent OOM (Limit to 10k cells each)
    # max_cells_plot = 10000
    
    # # 6.1 process the real data
    # if x_real.shape[0] > max_cells_plot:
    #     idx = np.random.choice(x_real.shape[0], max_cells_plot, replace=False)
    #     x_real_plot = x_real[idx]
    #     labels_real_plot = labels_real[idx]
    # else:
    #     x_real_plot = x_real
    #     labels_real_plot = labels_real

    # obs_real = pd.DataFrame({
    #     'Cell_Type': [breed_2_list[i] for i in labels_real_plot],
    #     'Source': 'Real'
    # })
    # adata_real_plot = sc.AnnData(X=x_real_plot, obs=obs_real)
    # adata_real_plot.var_names = index_2_gene # gene names must be aligned
    
    # # 6.2 process the generated data
    # adata_gen_plot = adata_gen.copy()
    # if adata_gen_plot.n_obs > max_cells_plot:
    #     sc.pp.subsample(adata_gen_plot, n_obs=max_cells_plot)
    
    # adata_gen_plot.obs['Source'] = 'Generated'
    # # keep only the needed columns for merging
    # adata_gen_plot.obs = adata_gen_plot.obs[['Cell_type', 'Source']].rename(columns={'Cell_type': 'Cell_Type'})
    
    # # 6.3 merge
    # # Note: ad.concat takes the intersection by default; mismatched var_names would yield empty data, but consistency was ensured earlier
    # adata_combined = ad.concat([adata_real_plot, adata_gen_plot], join='outer')
    # adata_combined.obs['Cell_Type'] = adata_combined.obs['Cell_Type'].astype('category')
    # adata_combined.obs['Source'] = adata_combined.obs['Source'].astype('category')
    
    # # 6.4 compute UMAP
    # # print("Running PCA and UMAP...")
    # sc.pp.pca(adata_combined)
    # sc.pp.neighbors(adata_combined)
    # sc.tl.umap(adata_combined)
    
    # # 6.5 set colors
    # # Source colors
    # adata_combined.uns['Source_colors'] = ['#f4df4e', '#949398'] # Gen: Yellow, Real: Grey
    
    # # Cell Type colors (using color_map)
    # if color_map:
    #     # Scanpy assigns colors in category order, so manual sorting is required
    #     categories = adata_combined.obs['Cell_Type'].cat.categories
    #     palette = [color_map.get(cat, '#808080') for cat in categories]
    #     adata_combined.uns['Cell_Type_colors'] = palette
    
    # # 6.6 plot and save
    # sc.pl.umap(adata_combined, color=['Source', 'Cell_Type'], 
    #            wspace=0.4, title=['Source (Real vs Gen)', 'Cell Type'],
    #            save='_real_vs_generated.png', show=False)
    
    # # --- 7. Compute the Silhouette Score ---
    # # evaluate how well Real and Generated mix (a lower score is better, meaning they are indistinguishable, i.e. realistic generation)
    # # Note: a low Score here is usually good (close to 0), indicating no clear boundary between Sources
    # try:
    #     sil_score = silhouette_score(adata_combined.obsm['X_umap'], adata_combined.obs['Source'])
    #     print(f"Silhouette Score (Source mixing): {sil_score:.4f}")
    # except:
    #     sil_score = -1.0
    #     print("Silhouette Score calculation failed.")

    # with open(os.path.join(output_dir, '3_eval_metrics.txt'), 'w') as f:
    #     f.write(f"Silhouette Score (Real vs Generated): {sil_score:.4f}\n")
    #     f.write(f"Mean Pearson Correlation: {np.nanmean(pearson_list):.4f}\n")
        
    print(f"评估完成，结果保存在 {output_dir}")

def debug_decoder_capability(net, mapping_dict, X_tensor, labels, device):
    print("\n=== Debugging Decoder Capability (Bypassing Encoder) ===")
    net.eval().to(device)
    
    # test cell type
    target_name = 'CD4_EM' # pick one that exists in your data
    if target_name not in mapping_dict:
        print(f"Error: {target_name} not in mapping dict")
        return

    target_idx = mapping_dict[target_name]
    
    # 1. Construct the input: a completely random Latent Z (independent of Bulk)
    if hasattr(net, 'mid_hidden_size'):
        latent_dim = net.mid_hidden_size * 2  # 256 * 2 = 512
    else:
        latent_dim = 512 
    
    print(f"Using Latent Dim: {latent_dim}")
    
    count = 50
    
    # sample z ~ N(0, 1)
    z_random = torch.randn(count, latent_dim).to(device)
    
    # 2. Construct the label
    num_classes = len(mapping_dict)
    label_tensor = F.one_hot(torch.tensor([target_idx] * count), num_classes=num_classes).float().to(device)
    
    # 3. decode
    try:
        with torch.no_grad():
            generated = net.decode(z_random, label_tensor).cpu().numpy()
    except RuntimeError as e:
        print(f"Decoder failed with error: {e}")
        print(f"Debug info: z shape {z_random.shape}, label shape {label_tensor.shape}")
        return
        
    # 3. evaluate
    real_indices = np.where(labels.numpy() == target_idx)[0]
    real_data = X_tensor[real_indices].numpy()
    
    mean_gen = generated.mean(axis=0)
    mean_real = real_data.mean(axis=0)
    
    corr, _ = pearsonr(mean_gen, mean_real)
    print(f"Target: {target_name}")
    print(f"Generated with Random Z Correlation: {corr:.4f}")
    
    # check the variance of the generated data to see whether it is active
    print(f"Generated Variance: {np.mean(np.var(generated, axis=0)):.6f}")
    
    if corr < 0.5:
        print(">>> FAIL: Decoder cannot generate correct profile even with clean Z.")
    else:
        print(">>> PASS: Decoder works fine. The problem is in Bulk -> Encoder path.")

def debug_prior_quality(net, cell_type_mu_logvar, mapping_dict, adata_real, common_genes, device):
    print("\n=== DEBUG: Checking Prior Quality (Fixed) ===")
    net.eval().to(device)
    
    # 1. strictly align the genes (this is the most critical step)
    # create a new AnnData to ensure the column order is exactly consistent
    # assume adata_real already contains these genes
    valid_genes = [g for g in common_genes if g in adata_real.var_names]
    if len(valid_genes) != len(common_genes):
        print(f"Warning: {len(common_genes) - len(valid_genes)} genes missing in adata_real.")
    
    # this step automatically reorders adata_real's columns to match the valid_genes list order
    adata_aligned = adata_real[:, valid_genes].copy()
    
    # 2. check the data range
    print(f"Real Data Range: {adata_aligned.X.min()} - {adata_aligned.X.max()}")
    # if the max value is large (>20), it may be Raw Counts and need Log1p
    if adata_aligned.X.max() > 20:
        print("Applying Log1p to real data for comparison...")
        sc.pp.log1p(adata_aligned)

    # take one cell type from the real data
    target_type = list(mapping_dict.keys())[0]
    type_id = mapping_dict[target_type]
    
    # 3. get the Prior and decode
    prior_mu = cell_type_mu_logvar[type_id][0].to(device)
    label_tensor = F.one_hot(torch.tensor([type_id]), len(mapping_dict)).float().to(device)
    
    with torch.no_grad():
        decoded_expr = net.decode(prior_mu, label_tensor).cpu().numpy().flatten()
    
    # 4. get the real mean
    real_cells = adata_aligned[adata_aligned.obs['labels'] == target_type].X
    if issparse(real_cells): real_cells = real_cells.toarray()
    real_mean_expr = real_cells.mean(axis=0).flatten()
    
    # 5. compute correlation
    # ensure equal length (in case common_genes has missing entries)
    min_len = min(len(decoded_expr), len(real_mean_expr))
    corr = np.corrcoef(decoded_expr[:min_len], real_mean_expr[:min_len])[0, 1]
    
    print(f"Cell Type: {target_type}")
    print(f"Prior decoded correlation with Real Mean: {corr:.4f}")
    
    if corr < 0.6:
        print(">>> 严重警告: Prior 本身与真实数据不相关！")
        print("    可能原因: 1. 基因顺序错乱 (最可能)")
        print("              2. 严重的 Batch Effect")
        print("              3. VAE 根本没训练好")
    else:
        print(">>> Prior 质量合格。问题出在优化过程。")

def check_vae_reconstruction(vae_model, 
                             X_tensor, 
                             labels, 
                             mapping_dict, 
                             common_genes, 
                             device, 
                             output_dir,
                             batch_size=512):
    """
    Evaluate the VAE's "compress-decompress" reconstruction ability on single-cell data.
    Core metric: the Pearson correlation between the input single cells and the reconstructed single cells.
    """
    print("\n" + "="*40)
    print(" >>> 开始检查 VAE 模型重建能力 (Sanity Check)")
    print("="*40)
    
    vae_model.eval()
    vae_model.to(device)
    
    # get the ID -> name mapping
    id2type = {v: k for k, v in mapping_dict.items()}
    num_cell_types = len(mapping_dict)
    
    # randomly sample part of the data for testing (e.g. 2000 cells)
    # or use all the data
    n_total = X_tensor.shape[0]
    indices = torch.randperm(n_total)[:2000] # testing 2000 is enough
    
    test_X = X_tensor[indices].to(device)
    test_y = labels[indices].to(device)
    
    recon_X_list = []
    
    with torch.no_grad():
        # process in batches to prevent GPU OOM
        for i in range(0, len(test_X), batch_size):
            batch_x = test_X[i : i+batch_size]
            batch_y = test_y[i : i+batch_size]
            
            # One-hot label
            batch_y_hot = F.one_hot(batch_y, num_cell_types).float()
            
            # standard VAE forward pass
            # Note: here we want to see the model's "best" achievable result
            # so usually no random noise is added (z = mu), or very little noise
            mu, logvar = vae_model.encode(batch_x, batch_y_hot)
            # z = vae.reparameterize(mu, logvar)
            # recon = vae.decode(z, batch_y_hot)
            # reconstruct directly from the mean (upper bound of the denoising ability)
            recon_batch = vae_model.decode(mu, batch_y_hot)
            recon_X_list.append(recon_batch.cpu())
            
    recon_X = torch.cat(recon_X_list, dim=0)
    
    # === Evaluation metrics ===
    
    # 1. Overall correlation (Flatten)
    # this measures the consistency of the overall value distribution
    input_flat = test_X.cpu().numpy().flatten()
    recon_flat = recon_X.numpy().flatten()
    corr_global, _ = pearsonr(input_flat, recon_flat)
    print(f"\n[总体] 全局 Pearson 相关性: {corr_global:.4f}")
    
    # 2. Per-cell correlation (Per Cell Correlation)
    # this measures whether each cell still resembles itself
    cell_corrs = []
    for i in range(len(test_X)):
        c_in = test_X[i].cpu().numpy()
        c_out = recon_X[i].numpy()
        if np.std(c_in) < 1e-9 or np.std(c_out) < 1e-9:
            cell_corrs.append(0)
        else:
            c, _ = pearsonr(c_in, c_out)
            cell_corrs.append(c)
    
    avg_cell_corr = np.mean(cell_corrs)
    print(f"[细胞级] 平均单细胞 Pearson 相关性: {avg_cell_corr:.4f}")
    
    # 3. Per-gene correlation (Per Gene Correlation)
    # this measures whether the model captured gene specificity
    gene_corrs = []
    input_np = test_X.cpu().numpy()
    recon_np = recon_X.numpy()
    
    for j in range(input_np.shape[1]): # iterate over genes
        g_in = input_np[:, j]
        g_out = recon_np[:, j]
        if np.std(g_in) < 1e-9 or np.std(g_out) < 1e-9:
             # if a gene is 0 in all cells, correlation is meaningless; set it to NaN or 0
            gene_corrs.append(np.nan)
        else:
            c, _ = pearsonr(g_in, g_out)
            gene_corrs.append(c)
            
    avg_gene_corr = np.nanmean(gene_corrs)
    print(f"[基因级] 平均基因 Pearson 相关性: {avg_gene_corr:.4f}")
    
    # 4. statistics per cell type
    print("\n--- 按细胞类型拆分 ---")
    type_stats = {}
    test_y_np = test_y.cpu().numpy()
    
    for type_id in range(num_cell_types):
        type_name = id2type.get(type_id, str(type_id))
        mask = (test_y_np == type_id)
        if mask.sum() == 0: continue
        
        # mean correlation for this type
        sub_corrs = np.array(cell_corrs)[mask]
        mean_corr = np.mean(sub_corrs)
        type_stats[type_name] = mean_corr
        print(f"  Type {type_name:<15} (n={mask.sum():<4}): {mean_corr:.4f}")
        
    # # === Plotting (optional) ===
    # # randomly pick a cell and draw a scatter plot
    # idx = np.random.randint(len(test_X))
    # plt.figure(figsize=(6, 6))
    # plt.scatter(test_X[idx].cpu().numpy(), recon_X[idx].numpy(), alpha=0.5, s=10)
    # plt.plot([0, test_X.max()], [0, test_X.max()], 'r--')
    # plt.xlabel("Original Expression (Log1p)")
    # plt.ylabel("Reconstructed Expression (Log1p)")
    # plt.title(f"Single Cell Reconstruction (Corr={cell_corrs[idx]:.2f})")
    # plt.savefig(os.path.join(output_dir, "vae_sanity_check_scatter.png"))
    # plt.close()
    
    # save the results
    pd.DataFrame({'Gene': common_genes, 'Pearson': gene_corrs}).to_csv(
        os.path.join(output_dir, "vae_gene_reconstruction_corr.csv"), index=False
    )
    
    print("="*40 + "\n")
    return avg_cell_corr
################################
### End-to-end workflow ###
################################
def main(sc_data, real_bulk_path, real_sc_path, output_dir, device=device, epochs=100, lr=0.001):
    """
    Main function that runs the complete pipeline from data loading to single-cell data generation.

    Parameters:
        sc_data: path to the single-cell data (.h5ad file)
        real_bulk_path: path to the bulk data (.tsv file)
        output_dir: output directory
        device: compute device ('cuda' or 'cpu')
        epochs: number of BulkVAE training epochs
        lr: learning rate
    """
    # # 1. Configure parameters
    # dataset, X_tensor, labels, n_cell_types, mapping_dict, all_genes, real_bulk_log, single_cell_matrix, cell_number_target_num, signature_array, color_map, bulk_sample_names, final_signatures, sig_indices, core_genes, core_indices_tensor = load_sc_data(
    #     sc_data, real_bulk_path, celltype_label='level2_cell_type'
    # )
    # print("Data loaded. n_cells:", len(dataset), "n_genes:", X_tensor.shape[1], "n_cell_types:", len(mapping_dict), "bulk samples:", real_bulk_log.shape)
    # breed_2_list = list(mapping_dict.keys())

    # sc_metadata = {
    # 'input_dim': len(all_genes),
    # 'X_tensor': X_tensor,
    # 'labels': labels,
    # 'n_cell_types': n_cell_types,
    # 'real_bulk_log': real_bulk_log,
    # 'mapping_dict': mapping_dict,
    # 'common_genes': all_genes,
    # 'cell_number_target_num': cell_number_target_num,
    # 'signature_array': signature_array,
    # 'breed_2_list': breed_2_list,
    # 'color_map': color_map,
    # 'bulk_sample_names': bulk_sample_names,
    # 'signatures': final_signatures,
    # 'sig_indices': sig_indices,
    # 'core_genes': core_genes,
    # 'core_indices_tensor': core_indices_tensor
    # }

    # # save to file
    # with open(os.path.join(output_dir, 'sc_metadata.pkl'), 'wb') as f:
    #     pickle.dump(sc_metadata, f)

    # # Stage 1: train the Attention GMVAE model
    hidden_size_list=[4096, 2048, 1024]
    with open(os.path.join(config_dir, 'sc_metadata.pkl'), 'rb') as f:
        sc_metadata = pickle.load(f)
    print(sc_metadata.keys())
    X_tensor = sc_metadata['X_tensor']
    labels = sc_metadata['labels']
    n_cell_types = sc_metadata['n_cell_types']
    real_bulk_log = sc_metadata['real_bulk_log']
    mapping_dict = sc_metadata['mapping_dict']
    common_genes = sc_metadata['common_genes']
    input_dim = len(common_genes)
    cell_number_target_num = sc_metadata['cell_number_target_num']
    # signature_array = sc_metadata['signature_array']
    breed_2_list = sc_metadata['breed_2_list']
    color_map = sc_metadata['color_map']
    bulk_sample_names = sc_metadata['bulk_sample_names']
    sig_indices = sc_metadata['sig_indices']
    core_genes = sc_metadata['core_genes']
    core_indices_tensor = sc_metadata['core_indices_tensor']
    print(f"mapping_dict: {mapping_dict}")
    print(f"breed_2_list: {breed_2_list}")
    
    scvae = AttentionVAE(input_size=len(core_genes), 
                         output_size=len(common_genes),
                         hidden_size_list=hidden_size_list,mid_hidden_size=256,num_cell_types=n_cell_types, embedding_dim=32,nhead=4,num_layers=4)
    # scvae.to(device).train()
    # scvae, cell_type_mu_logvar = train_vae(vae_model=scvae,
    #                                        X_tensor=X_tensor,labels=labels,
    #                                        real_bulk_log=real_bulk_log,
    #                                        batch_size=512,feature_size=input_dim,
    #                                        epoch_num=50,
    #                                        learning_rate=5e-5,hidden_list=hidden_size_list,mid_hidden_size=256,num_cell_types=n_cell_types,breed_2_list=breed_2_list, color_map=color_map, used_device=device, core_indices_tensor=core_indices_tensor)

    # # Stage 2: generate the single-cell expression matrix per bulk sample
    # # compute the global scaling correction factor for the bulk
    # 1. compute the Linear Mean of the Training SC
    # X_tensor is in Log1p
    sc_mean_linear = torch.expm1(X_tensor).mean(dim=0)
    # 2. compute the Linear Mean of the Real Bulk
    # real_bulk_log is assumed to be in Log1p too
    bulk_mean_linear = torch.expm1(real_bulk_log).mean(dim=0)
    # 3. compute the factor
    # add 1e-6 to prevent division by zero
    scaling_factor = bulk_mean_linear / (sc_mean_linear + 1e-6)
    print(f"Global scaling factor calculated: {scaling_factor.mean().item():.4f}.")
    scaling_factor = scaling_factor.to(device)

    state = torch.load(os.path.join(output_dir, 'scvae_best.pth'), map_location=device)
    scvae.load_state_dict(state)
    cell_type_mu_logvar_best = torch.load(os.path.join(output_dir, 'cell_type_mu_logvar_best.pt'), map_location=device)
    
    target_samples=['LDK1', 'LDK2', 'LDK3', 'LDK4', 'LDK5', 'LDK6']
    # 2. get the indices and the valid sample names (guards against a misspelled or missing sample name in the file)
    sample_indices, valid_sample_names = get_sample_indices_from_bulk(
        real_bulk_path, 
        target_samples=target_samples
    )
    if not sample_indices:
        raise ValueError("没有找到有效的样本，无法生成。")
    input_bulk_log = real_bulk_log[sample_indices]
    print(f"Selected bulk samples indices: {sample_indices}, shape: {input_bulk_log.shape}")

    # # 2. prepare the corresponding cell-proportion list (very important! must be sliced to align with input_bulk)
    # input_fractions_list = [cell_type_fractions_list[i] for i in sample_indices]

    # print(f"Selected bulk samples indices: {sample_indices}")
    # print(f"Input bulk shape: {input_bulk_log.shape}")
    # print("Start generating the single-cell expression matrix using the [optimization strategy]...")

    # 3. call the new optimized generation function
    scvae = scvae.to(device)
    total_cells = n_cell_types * 500
    adata_generated, Final_Z, obs_df = optimize_z_and_generate_final(
        vae=scvae,
        real_bulk_tensor=input_bulk_log,
        sample_names=valid_sample_names,
        mapping_dict=mapping_dict,
        priors=cell_type_mu_logvar_best, # trained prior
        device=device,
        output_dir=output_dir,
        sig_indices = sig_indices,
        total_cells = total_cells,
        scaling_factor=scaling_factor,
        lambda_entropy = 0.1,
        temperature = 0.5,
        lr_scaler=1.0,
        steps_scaler=400,
        lambda_reg_scaler=1.0,
        lr_z = 0.05,
        steps_z = 1000,
        lambda_reg_z = 1.0,
        lambda_mse=0.0,
        common_genes=common_genes,
    )
    # adata_generated = sc.read_h5ad(os.path.join(output_dir, 'generated_data_noncovid1_6.h5ad'))
    #  # [New] compute training-set statistics
    # print("Computing SC training statistics...")
    # # compute on CPU to avoid GPU OOM, or process in batches
    # sc_mean = X_tensor.mean(dim=0, keepdim=True)
    # sc_std = X_tensor.std(dim=0, keepdim=True)
    # X_tensor_stats = {'mean': sc_mean, 'std': sc_std}
    
    # # debug_decoder_capability(
    # #     net=scvae,
    # #     mapping_dict=mapping_dict,
    # #     X_tensor=X_tensor,
    # #     labels=labels,
    # #     device=device
    # # )

    adata_real = sc.read_h5ad(real_sc_path)
    adata_real.obs['labels'] = adata_real.obs['cell type'].astype(str)
    adata_real.obs['labels'] = adata_real.obs['labels'].astype('category')
    adata_real.X = adata_real.X.toarray()  # raw data
    # adata = adata[:, adata.var.highly_variable].copy()
    print(f"Range of adata.X: {adata_real.X.min().min()} - {adata_real.X.max().max()}")
    sc.pp.normalize_total(adata_real, target_sum=1e4)
    print(f"Range of normalized adata.X: {adata_real.X.min().min()} - {adata_real.X.max().max()}")
    sc.pp.log1p(adata_real)
    print(f"Range of log1p adata.X: {adata_real.X.min().min()} - {adata_real.X.max().max()}")
    # plot_smart_donut(
    #     type_counts=adata_real.obs['labels'].value_counts().to_dict(),
    #     title=f"Real Set Composition (Total: {adata_real.obs['labels'].shape[0]})",
    #     save_path=os.path.join(output_dir, 'real_data_composition_donut.png')
    # )
    min_cells = 3
    label_counts = adata_real.obs['labels'].value_counts()
    small_labels = label_counts[label_counts < min_cells].index
    adata_real.obs['labels'] = adata_real.obs['labels'].replace(small_labels, 'others')
    adata_real = adata_real[:, common_genes].copy()
    X_tensor = torch.tensor(adata_real.X, dtype=torch.float32)
    labels = torch.tensor(adata_real.obs['labels'].cat.codes.values, dtype=torch.long)
    
    plot_marker_expression_comparison(
        adata_real=adata_real,
        adata_gen = adata_generated,
        group_col_real='labels',
        group_col_gen='Cell_type',
        top_n=5,
        output_dir=output_dir
    )
    
    evaluate_generated_data(
        X_tensor=X_tensor, # real single-cell tensor
        labels=labels,     # real label tensor
        adata_gen = adata_generated, # pass in the generated AnnData
        breed_2_list=breed_2_list,
        common_genes=common_genes,
        color_map=color_map,
        output_dir=output_dir,
        real_bulk_tensor=input_bulk_log,
    )
    
    # ###debug: pseudo-bulk for debugging####
    # print("Debug: generate and evaluate using single-cell data...")
    # sanity_score = check_vae_reconstruction(
    #     vae_model=scvae,
    #     X_tensor=X_tensor,  # your training or test set Tensor
    #     labels=labels,      # the corresponding labels
    #     mapping_dict=mapping_dict,
    #     common_genes=common_genes,
    #     device=device,
    #     output_dir=output_dir
    # )
    # print(f"VAE Sanity Check Score: {sanity_score:.4f}")
    # debug_prior_quality(scvae, cell_type_mu_logvar_best, mapping_dict, adata_real, common_genes, device)
################################
### Example usage ###
################################

if __name__ == "__main__":
    main(
        sc_data=os.path.join(data_dir, 'GSE14115_sc_raw_counts_train.h5ad'),
         real_bulk_path=os.path.join(data_dir, 'GSE14115_bulk_counts_LD_testset.txt'),
         real_sc_path=os.path.join(data_dir, 'GSE14115_sc_raw_counts_LD.h5ad'),
         output_dir=output_dir,
         device=device)
