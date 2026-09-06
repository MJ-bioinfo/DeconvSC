import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale

def plot_marker_expression_comparison(adata_real, adata_gen, 
                                      group_col_real='labels', 
                                      group_col_gen='Cell_type', 
                                      top_n=5, 
                                      output_dir='./'):
    """Heatmap comparing marker gene expression between real and generated data."""
    print("\n--- Comparing Marker Gene Expression (Real vs Gen) ---")
    
    real_temp = adata_real.copy()
    if 'rank_genes_groups' not in real_temp.uns:
        sc.tl.rank_genes_groups(real_temp, groupby=group_col_real, method='t-test_overestim_var')
    
    groups = sorted([str(g) for g in real_temp.obs[group_col_real].unique()])
    marker_dict = {}
    var_names = set()
    
    for g in groups:
        try:
            genes = sc.get.rank_genes_groups_df(real_temp, group=g).head(top_n)['names'].tolist()
            marker_dict[g] = genes
            var_names.update(genes)
        except KeyError:
            print(f"Warning: Group {g} not found in rank_genes_groups.")
            
    gene_list = []
    for g in groups:
        if g in marker_dict:
            gene_list.extend(marker_dict[g])
            
    real_mean_df = pd.DataFrame(index=groups, columns=gene_list)
    for g in groups:
        cells = real_temp[real_temp.obs[group_col_real] == g, gene_list]
        mean_expr = np.mean(cells.X, axis=0)
        if hasattr(mean_expr, 'A1'): mean_expr = mean_expr.A1
        elif hasattr(mean_expr, 'toarray'): mean_expr = mean_expr.toarray().flatten()
        real_mean_df.loc[g] = mean_expr

    valid_genes = [gene for gene in gene_list if gene in adata_gen.var_names]
    gen_mean_df = pd.DataFrame(index=groups, columns=valid_genes)
    
    if group_col_gen not in adata_gen.obs:
        print(f"Error: Column {group_col_gen} not found in adata_gen.obs")
        return

    for g in groups:
        if g not in adata_gen.obs[group_col_gen].values:
            gen_mean_df.loc[g] = 0
            continue
        cells = adata_gen[adata_gen.obs[group_col_gen] == g, valid_genes]
        gen_mean_df.loc[g] = np.mean(cells.X, axis=0)

    def safe_zscore(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or df.shape[1] == 0 or df.shape[0] == 0:
            return pd.DataFrame(0, index=df.index, columns=df.columns)
        z = scale(df, axis=0)
        return pd.DataFrame(z, index=df.index, columns=df.columns).fillna(0)

    real_norm = safe_zscore(real_mean_df.astype(float))
    gen_norm = safe_zscore(gen_mean_df.astype(float))
    
    vmax, vmin = 2.5, -1.5
    fig, axes = plt.subplots(1, 2, figsize=(16, 10), sharey=True)
    
    sns.heatmap(real_norm, ax=axes[0], cmap='RdBu_r', vmin=vmin, vmax=vmax, xticklabels=False, cbar=False)
    axes[0].set_title("Real Data (Top Markers)", fontsize=14)
    axes[0].set_ylabel("Cell Types")
    axes[0].set_xlabel("Marker Genes (Grouped by Type)")
    
    sns.heatmap(gen_norm, ax=axes[1], cmap='RdBu_r', vmin=vmin, vmax=vmax, xticklabels=False, cbar=True)
    axes[1].set_title("Generated Data (Same Markers)", fontsize=14)
    axes[1].set_xlabel("Marker Genes")
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'comparison_marker_heatmap.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Marker comparison heatmap saved to {save_path}")