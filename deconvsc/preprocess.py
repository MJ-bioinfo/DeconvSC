import os
import torch
from torch.utils.data import TensorDataset
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from scipy.sparse import issparse
from .utils import plot_smart_donut

def optimize_signature_matrix(adata, output_dir, group_col=None, n_signatures_per_type=50, max_total_signatures=2000):
    """
    Minimize co-linearity, reduce matrix condition number.
    """
    print("--- Starting Optimized Signature Gene Selection ---")
    exclude_prefixes = ('MT-', 'RPS', 'RPL', 'HB') 
    gene_mask = [not g.startswith(exclude_prefixes) for g in adata.var_names]
    adata_clean = adata[:, gene_mask].copy()
    print(f"Filtered out MT/RP/HB genes. Remaining: {adata_clean.n_vars}")
    
    min_cells = 3
    grp_counts = adata_clean.obs[group_col].value_counts()
    to_replace = grp_counts[grp_counts < 3].index.tolist()
    adata_clean.obs[group_col] = adata_clean.obs[group_col].replace(to_replace, "others")
    adata_clean.obs[group_col] = pd.Categorical(adata_clean.obs[group_col])
    
    sc.tl.rank_genes_groups(adata_clean, group_col, method='wilcoxon', pts=True, use_raw=False) 
    candidates = set()
    result = adata_clean.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    
    selected_genes_dict = {}
    for group in groups:
        names = pd.DataFrame(result['names'])[group]
        scores = pd.DataFrame(result['scores'])[group]
        logfoldchanges = pd.DataFrame(result['logfoldchanges'])[group]
        pts_group = pd.DataFrame(result['pts'])[group] 
        
        df_group = pd.DataFrame({
            'group': group, 'gene': names, 'score': scores,
            'logfc': logfoldchanges, 'pts': pts_group.values if 'pts' in result else 0 
        })
        df_group.to_csv(os.path.join(output_dir, f'raw_signature_candidates_{group}.csv'), index=False)
        
        df_filtered = df_group[ (df_group['logfc'] > 1.0) & (df_group['score'] > 0) ].copy()
        if 'pts' in result:
             df_filtered = df_filtered[df_filtered['pts'] > 0.2]
        df_filtered.to_csv(os.path.join(output_dir, f'filtered_signature_candidates_{group}.csv'), index=False)
        
        top_genes = df_filtered.head(n_signatures_per_type)['gene'].tolist()
        selected_genes_dict[group] = top_genes
        candidates.update(top_genes)
        
    sinagures = sorted(list(candidates))
    print(f"Selected {len(sinagures)} unique signature genes.")
    
    sig_matrix_list = []
    unique_labels = sorted(adata_clean.obs[group_col].unique())
    for label in unique_labels:
        cells = adata_clean[adata_clean.obs[group_col] == label, sinagures]
        mean_expr = np.mean(cells.X, axis=0)
        if issparse(mean_expr): mean_expr = mean_expr.toarray()
        sig_matrix_list.append(mean_expr.flatten())
    
    Sig_Matrix = np.array(sig_matrix_list).T
    cond_num = np.linalg.cond(Sig_Matrix)
    print(f"Signature Matrix Condition Number (Log space): {cond_num:.2f}")
    
    with open(os.path.join(output_dir, 'metrics.txt'), 'a') as f:
        f.write(f"Signature matrix condition number: {cond_num:.2f}\n")
    
    if cond_num > 1000:
        print("WARNING: Condition number is high (>1000). Collinearity exists.")
    else:
        print("Condition number is good (stable).")

    return sinagures, np.array(sinagures)

def load_sc_data(sc_data, real_bulk_path, celltype_label, output_dir, sep="\t", seed=18):
    """
    load sc data, preprocess, visualize, select signature genes, and prepare for training.
    """
    adata = sc.read_h5ad(sc_data)
    adata.obs['labels'] = adata.obs[celltype_label].astype(str)
    adata.X = adata.X.toarray() if issparse(adata.X) else adata.X
    
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
    sc.tl.umap(adata, min_dist=0.5, spread=1.0, random_state=seed)
    
    fig = plt.figure(figsize=(25, 10))
    sc.pl.umap(adata, color='labels', title='sc UMAP by Cell Type', legend_loc='on data', size=30, show=False)
    plt.savefig(os.path.join(output_dir, 'sc_train_umap_by_celltype.png'), dpi=300)
    plt.close()
    
    unique_labels = adata.obs['labels'].unique()
    mapping_dict = {label: idx for idx, label in enumerate(unique_labels)}
    cell_number_target_num = adata.obs['labels'].value_counts().to_dict()
    
    labels = torch.LongTensor([mapping_dict.get(label, -1) for label in adata.obs['labels']])
    
    plot_smart_donut(
        type_counts=cell_number_target_num,
        title=f"Training Set Composition (Total: {sum(cell_number_target_num.values())})",
        save_path=os.path.join(output_dir, 'training_data_composition_donut.png')
    )

    palette = sns.color_palette("tab20", n_colors=len(torch.unique(labels)))
    hex_colors = [matplotlib.colors.rgb2hex(color) for color in palette]
    color_map = {ct: color for ct, color in zip(adata.obs['labels'].astype('category').cat.categories, hex_colors)}
    
    real_bulk_df = pd.read_csv(real_bulk_path, sep=sep, index_col=0).T
    real_bulk_df = real_bulk_df.loc[:, ~real_bulk_df.columns.duplicated()]
    real_bulk_df = real_bulk_df.dropna(axis=1, how='all')

    all_genes = sorted(list(set(adata.var_names) & set(real_bulk_df.columns)))
    pd.DataFrame({'prediction_genes': all_genes}).to_csv(os.path.join(output_dir, 'prediction_genes.csv'), index=False)
    
    if len(all_genes) < 100:
        raise ValueError(f"Too few common genes: {len(all_genes)}")
    
    raw_signatures, signature_array = optimize_signature_matrix(
        adata, output_dir=output_dir, group_col='labels', n_signatures_per_type=50, max_total_signatures=5000) 
    final_signatures = sorted(list(set(raw_signatures) & set(all_genes)))
    
    pd.DataFrame({'signature_genes': final_signatures}).to_csv(os.path.join(output_dir, 'signature_genes.csv'), index=True)
    gene_to_idx = {gene: i for i, gene in enumerate(all_genes)}
    sig_indices = [gene_to_idx[g] for g in final_signatures]
    
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=False)
    hvg_names = adata.var[adata.var['highly_variable']].index.tolist()
    core_genes = sorted(list(set(hvg_names) & set(all_genes)))
    core_indices = [gene_to_idx[g] for g in core_genes]
    core_indices_tensor = torch.LongTensor(core_indices)
    
    real_bulk_df = real_bulk_df[all_genes]
    real_bulk_df = real_bulk_df * 1e4 / real_bulk_df.sum(axis=1).values[:, None]
    real_bulk_log = np.log1p(real_bulk_df)
    bulk_sample_names = real_bulk_df.columns.tolist()

    adata = adata[:, all_genes].copy()
    if adata.var_names.tolist() != real_bulk_log.columns.tolist():
        raise ValueError("CRITICAL ERROR: Gene order mismatch between SC and Bulk!")

    unique_cell_types = np.unique(adata.obs['labels'])
    single_cell_matrix = []
    for label in unique_cell_types:
        mask = adata.obs['labels'] == label
        mean_expr = adata[mask].X.mean(axis=0)
        single_cell_matrix.append(mean_expr)
    single_cell_matrix = np.array(single_cell_matrix).T
    K = single_cell_matrix.shape[1]
    
    X_tensor = torch.FloatTensor(adata.X)
    dataset = TensorDataset(X_tensor, labels)
    real_bulk_log = torch.FloatTensor(real_bulk_log.values)
    
    return dataset, X_tensor, labels, K, mapping_dict, all_genes, real_bulk_log, single_cell_matrix, cell_number_target_num, signature_array, color_map, bulk_sample_names, final_signatures, sig_indices, core_genes, core_indices_tensor