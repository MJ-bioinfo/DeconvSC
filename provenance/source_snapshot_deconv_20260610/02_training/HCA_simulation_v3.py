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
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import rgb2hex
from sklearn.model_selection import train_test_split,StratifiedGroupKFold, GroupKFold, StratifiedKFold
import umap
import argparse
import os
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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, mean_squared_error
from scipy.optimize import nnls
from scipy.stats import chi2_contingency, wasserstein_distance
from torch.optim import AdamW
import copy
from scipy.optimize import minimize
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
import warnings
warnings.filterwarnings("ignore")

# 设备配置
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
data_dir = '/disk1/maijl/deconv/data/Simulation_HCA'
output_dir = '/disk1/maijl/deconv/cVAE/Simulation_HCA/result/HCA/v3_pureprior'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def set_seed(seed=18):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 保证cudnn的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(seed=18)
seed = 18

###############################
### 数据预处理与加载工具函数 ###
################################
def optimize_signature_matrix(adata, group_col=None, n_signatures_per_type=50, max_total_signatures=2000):
    """
    智能筛选 Signature Genes，最小化共线性，降低矩阵条件数。
    """
    print("--- Starting Optimized Signature Gene Selection ---")
    # 1. 预过滤：剔除干扰基因 (线粒体MT, 核糖体RP)
    # 这些基因在各种细胞都高表达，会严重干扰解卷积
    exclude_prefixes = ('MT-', 'RPS', 'RPL', 'HB') # HB是血红蛋白，如果是肺组织要注意
    gene_mask = [not g.startswith(exclude_prefixes) for g in adata.var_names]
    adata_clean = adata[:, gene_mask].copy()
    print(f"Filtered out MT/RP/HB genes. Remaining: {adata_clean.n_vars}")
    # 2. 计算差异表达 (Rank Genes)
    # method='wilcoxon' 是标准的，但我们要加 pts (表达比例) 约束
    # pts=True 会计算基因在组内和组外的表达比例，非常重要
    sc.tl.rank_genes_groups(adata_clean, group_col, method='wilcoxon', 
                            pts=True, # 计算表达比例
                            use_raw=False) # 假设 X 已经是 log1p 后的
    candidates = set()
    result = adata_clean.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    
    # 3. 严格筛选循环
    # 我们希望选出的基因：在目标组表达率高(pts > 0.5)，在其他组表达率低
    selected_genes_dict = {}
    for group in groups:
        # 获取该组的统计数据
        groups = result['names'].dtype.names
        names = pd.DataFrame(result['names'])[group]
        scores = pd.DataFrame(result['scores'])[group]
        logfoldchanges = pd.DataFrame(result['logfoldchanges'])[group]
        # pts (fraction of cells expressing the gene)
        pts_group = pd.DataFrame(result['pts'])[group] 
        # pts_rest (fraction of rest cells expressing) - 这一项不在默认return里，需要间接推断或只用 logfoldchange
        # CIBERSORTx 逻辑：只选 logFC 大且 score 高的
        # 组合 DataFrame
        df_group = pd.DataFrame({
            'group': group,
            'gene': names,
            'score': scores,
            'logfc': logfoldchanges,
            'pts': pts_group.values if 'pts' in result else 0 # 兼容性
        })
        df_group.to_csv(os.path.join(output_dir, f'raw_signature_candidates_{group}.csv'), index=False)
        # --- 核心筛选逻辑 ---
        # 1. 显著性过滤
        df_filtered = df_group[ (df_group['logfc'] > 1.0) & (df_group['score'] > 0) ].copy()
        # 2. 表达率过滤 (防止选到只在 1% 细胞里表达的基因)
        # 如果 pts 信息可用，要求至少 20% 的该类细胞表达该基因
        if 'pts' in result:
             df_filtered = df_filtered[df_filtered['pts'] > 0.2]
        df_filtered.to_csv(os.path.join(output_dir, f'filtered_signature_candidates_{group}.csv'), index=False)
        # 3. 取 Top N
        top_genes = df_filtered.head(n_signatures_per_type)['gene'].tolist()
        selected_genes_dict[group] = top_genes
        candidates.update(top_genes)
    sinagures = sorted(list(candidates))
    
    # # 4. (可选) 缩减总数以防止过拟合
    # if len(sinagures) > max_total_signatures:
    #     # 如果基因太多，保留 score 最高的那些
    #     print(f"Reducing genes from {len(sinagures)} to {max_total_signatures}...")
    #     # 这里简化处理，实际可以用更复杂的逻辑
    #     sinagures = sinagures[:max_total_signatures]
    print(f"Selected {len(sinagures)} unique signature genes.")
    
    # 5. --- 关键步骤：检查条件数 (Condition Number) ---
    # 构建临时 Signature Matrix (A)
    # 计算每种细胞的平均表达量
    sig_matrix_list = []
    unique_labels = sorted(adata_clean.obs[group_col].unique())
    
    for label in unique_labels:
        # 取该类型细胞的平均值
        # 注意：这里用 adata_clean (Log空间)
        # 真正的 CIBERSORTx 是在 Linear 空间最小化条件数，但在 Log 空间做筛选也没问题
        cells = adata_clean[adata_clean.obs[group_col] == label, sinagures]
        mean_expr = np.mean(cells.X, axis=0)
        # 如果是稀疏矩阵
        if issparse(mean_expr): mean_expr = mean_expr.toarray()
        sig_matrix_list.append(mean_expr.flatten())
    
    Sig_Matrix = np.array(sig_matrix_list).T # [Genes, CellTypes]
    
    # 计算条件数
    # 这里的矩阵通常是 Log 空间的。如果 NNLS 用 Linear，需要 expm1 后再算
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
    绘制智能甜甜圈图：
    1. 自动生成不重叠的颜色。
    2. 图例包含：类别名、细胞数、百分比。
    3. 只有占比 > 2% 的切片才在图上显示标签，防止重叠。
    4. 图例放在侧边，保证信息完整且不遮挡。
    """
    # 1. 准备数据
    if isinstance(type_counts, dict):
        labels = list(type_counts.keys())
        sizes = list(type_counts.values())
    else: # 假设是 Series
        labels = type_counts.index.tolist()
        sizes = type_counts.values.tolist()
    
    # 按数量排序（从大到小），这样图例好看
    sorted_indices = np.argsort(sizes)[::-1]
    labels = [labels[i] for i in sorted_indices]
    sizes = [sizes[i] for i in sorted_indices]
    total = sum(sizes)
    
    # 2. 设置颜色 (处理多达60+种颜色)
    # 使用 husl 调色板，确保颜色区分度尽可能大
    colors = sns.color_palette("husl", len(labels))
    
    # 3. 创建画布
    fig, ax = plt.subplots(figsize=(14, 8), subplot_kw=dict(aspect="equal"))
    
    # 4. 定义切片标签生成函数 (仅在切片够大时显示)
    def my_autopct(pct):
        return f'{pct:.1f}%' if pct > 2.0 else '' # 阈值可调，2%以下不显示在图上

    # 5. 绘制饼图 (Donut)
    wedges, texts, autotexts = ax.pie(
        sizes, 
        autopct=my_autopct,
        textprops=dict(color="w", fontweight='bold', fontsize=9),
        colors=colors,
        startangle=90,
        pctdistance=0.85, # 数值距离圆心的距离
        wedgeprops=dict(width=0.4, edgecolor='w') # width控制甜甜圈厚度
    )
    
    # 6. 构建详细图例标签 (Name: Count (Pct%))
    legend_labels = [f"{l}: {s} ({s/total:.1%})" for l, s in zip(labels, sizes)]
    
    # 7. 添加图例 (放在右侧，多列显示以防太长)
    # 根据类别数量动态调整图例列数
    ncols = 1
    if len(labels) > 20: ncols = 2
    if len(labels) > 40: ncols = 3
    
    ax.legend(wedges, legend_labels,
              title="Cell Types: Count (Ratio)",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1), # 放在图表右侧外
              fontsize=8,
              ncol=ncols)
    
    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title(title, fontdict={'fontsize': 14, 'fontweight': 'bold'})
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Donut chart saved to: {save_path}")
    
def plot_marker_expression_comparison(adata_gen, X_val_true, y_val_true, mapping_dict, all_genes, marker_dict, output_dir):
    """
    对比生成数据和真实数据的 Marker 表达。
    marker_dict: {cell_type: [genes...]} 动态计算出的 Marker 字典
    """
    # 1. 展平并去重基因，同时保持基于细胞类型的顺序
    ordered_genes = []
    gene_to_type_map = {} # 用于X轴标签颜色等(可选)
    
    for ctype, genes in marker_dict.items():
        for g in genes:
            if g in all_genes and g not in ordered_genes:
                ordered_genes.append(g)
                gene_to_type_map[g] = ctype
                
    if not ordered_genes:
        print("Warning: No valid marker genes found.")
        return

    gene_to_idx = {gene: i for i, gene in enumerate(all_genes)}
    marker_indices = [gene_to_idx[g] for g in ordered_genes]
    
    # 2. 计算真实验证集 (Ground Truth) 的平均表达矩阵
    # Rows: Cell Types, Cols: Genes
    cell_types_present = sorted(list(mapping_dict.keys()))
    inv_map = {v: k for k, v in mapping_dict.items()}
    
    # 准备容器
    true_expr_matrix = pd.DataFrame(index=cell_types_present, columns=ordered_genes)
    gen_expr_matrix = pd.DataFrame(index=cell_types_present, columns=ordered_genes)
    
    # 转换 tensor
    if torch.is_tensor(X_val_true): X_val_np = X_val_true.cpu().numpy()
    else: X_val_np = X_val_true
        
    if torch.is_tensor(y_val_true): y_val_np = y_val_true.cpu().numpy()
    else: y_val_np = y_val_true

    # 填充真实数据矩阵
    for cls_name in cell_types_present:
        cls_idx = mapping_dict[cls_name]
        mask = y_val_np == cls_idx
        if mask.sum() > 0:
            # 该类型细胞的平均表达
            mean_expr = X_val_np[mask].mean(axis=0)
            true_expr_matrix.loc[cls_name] = mean_expr[marker_indices]
        else:
            true_expr_matrix.loc[cls_name] = 0.0

    # 3. 填充生成数据矩阵
    # 确保 adata_gen 索引了所有基因
    gen_df = adata_gen.to_df() # [Cells, Genes]
    gen_df['Cell_type'] = adata_gen.obs['Cell_type'].values
    
    gen_means = gen_df.groupby('Cell_type')[ordered_genes].mean()
    
    # 对齐索引
    for cls_name in cell_types_present:
        if cls_name in gen_means.index:
            gen_expr_matrix.loc[cls_name] = gen_means.loc[cls_name]
        else:
            gen_expr_matrix.loc[cls_name] = 0.0
            
    # 类型转换
    true_expr_matrix = true_expr_matrix.astype(float)
    gen_expr_matrix = gen_expr_matrix.astype(float)
    
    true_expr_matrix.to_csv(os.path.join(output_dir, 'real_marker_expression_matrix.csv'))
    gen_expr_matrix.to_csv(os.path.join(output_dir, 'gen_marker_expression_matrix.csv'))

    # 4. 绘图
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    
    # 统一色标
    vmin = min(true_expr_matrix.min().min(), gen_expr_matrix.min().min())
    vmax = max(true_expr_matrix.max().max(), gen_expr_matrix.max().max())
    
    # Plot True
    sns.heatmap(true_expr_matrix, ax=axes[0], cmap='viridis', vmin=vmin, vmax=vmax, 
                cbar_kws={'label': 'Mean Log1p Expr'})
    axes[0].set_title('Ground Truth (Validation Set) - Dynamic Markers')
    axes[0].set_ylabel('Cell Type')
    
    # Plot Generated
    sns.heatmap(gen_expr_matrix, ax=axes[1], cmap='viridis', vmin=vmin, vmax=vmax,
                cbar_kws={'label': 'Mean Log1p Expr'})
    axes[1].set_title('Generated Cells - Dynamic Markers')
    axes[1].set_ylabel('Cell Type')
    axes[1].set_xlabel('Marker Genes (Grouped by Cell Type)')
    
    # 旋转 X 轴标签
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'marker_heatmap.png'), dpi=300)
    plt.close()
    print(f"Dynamic Marker comparison plot saved to {os.path.join(output_dir, 'marker_heatmap.png')}")

    # 5. 可选：计算两者矩阵的相关性（量化评估）
    # 将两个矩阵拉平计算 Pearson
    # 只比较两者都有的基因
    common_cols = true_expr_matrix.columns.intersection(gen_expr_matrix.columns)
    if len(common_cols) > 0:
        flat_real = true_expr_matrix[common_cols].values.flatten()
        flat_gen = gen_expr_matrix[common_cols].values.flatten()
        from scipy.stats import pearsonr
        corr, _ = pearsonr(flat_real, flat_gen)
        print(f"Overall Pattern Consistency (Correlation): {corr:.4f}")
        
        # 保存相关性到文件
        with open(os.path.join(output_dir, 'marker_consistency_score.txt'), 'w') as f:
            f.write(f"Marker Pattern Correlation: {corr:.4f}\n")
            
def get_cell_type_indices(labels, num_cell_types, allowed_indices=None):
    """
    预计算索引。
    如果提供了 allowed_indices (Numpy array of indices)，则只从这些索引中筛选。
    """
    indices_dict = {}
    labels_np = labels.cpu().numpy()
    
    # 如果有限制索引范围（例如只用训练集）
    if allowed_indices is not None:
        # 创建一个全量的 mask
        mask_base = np.zeros(len(labels_np), dtype=bool)
        mask_base[allowed_indices] = True
    else:
        mask_base = np.ones(len(labels_np), dtype=bool)

    for i in range(num_cell_types):
        # 既要是该类别，又要在允许的索引范围内
        type_mask = (labels_np == i) & mask_base
        indices_dict[i] = np.where(type_mask)[0]
    return indices_dict

def compute_pseudo_bulk(adata):
    """
    从单细胞adata合成pseudo-bulk数据，按样本求和细胞表达并归一化为TPM-like格式。
    
    参数：
        adata: 单细胞adata对象，需包含'donor_id'列
    
    返回：
        bulk_df: DataFrame，行是样本，列是基因，值为TPM-like表达量
    """
    if 'donor_id' not in adata.obs.columns:
        raise ValueError("adata.obs must contain 'donor_id' column for sample-wise splitting")
    samples = np.unique(adata.obs['donor_id'])
    pseudo_bulk = pd.DataFrame(index=samples, columns=adata.var_names, dtype=np.float64)
    for s in samples:
        s_mask = adata.obs['donor_id'] == s
        s_expr = adata[s_mask].X.sum(axis=0)
        if issparse(s_expr):
            s_expr = s_expr.toarray().flatten()
        pseudo_bulk.loc[s] = s_expr
    # 检查总和以避免除零
    row_sums = pseudo_bulk.sum(axis=1)
    row_sums[row_sums == 0] = 1.0  # 避免除零
    pseudo_bulk = pseudo_bulk.div(row_sums, axis=0) * 1e6  # TPM-like 归一化
    pseudo_bulk = pseudo_bulk.fillna(0.0)  # 填充任何 NaN
    print(f"pseudo_bulk contains NaN: {pseudo_bulk.isna().any().any()}")
    return pseudo_bulk

from torch.distributions import Dirichlet
from torch import numel
def compute_gene_correlation_loss(x_real, x_recon):
    """
    计算两个 Batch 数据的基因-基因相关性矩阵的差异。
    输入: [Batch_Size, N_Genes]
    """
    # 1. 为了数值稳定性，添加微小噪声防止标准差为0
    epsilon = 1e-8
    
    # 2. 归一化 (Z-score per gene)
    # 减去均值
    real_mean = x_real.mean(dim=0, keepdim=True)
    recon_mean = x_recon.mean(dim=0, keepdim=True)
    real_centered = x_real - real_mean
    recon_centered = x_recon - recon_mean
    
    # 计算标准差
    real_std = x_real.std(dim=0, keepdim=True) + epsilon
    recon_std = x_recon.std(dim=0, keepdim=True) + epsilon
    
    # 3. 计算相关性矩阵 (Pearson Correlation Matrix) [Genes, Genes]
    # 公式: (X_centered.T @ X_centered) / (N - 1) / (std.T @ std)
    n = x_real.size(0)
    if n <= 1: return torch.tensor(0.0, device=x_real.device) # Batch太小无法计算
    
    real_corr = (real_centered.t() @ real_centered) / (n - 1)
    real_corr = real_corr / (real_std.t() @ real_std)
    
    recon_corr = (recon_centered.t() @ recon_centered) / (n - 1)
    recon_corr = recon_corr / (recon_std.t() @ recon_std)
    
    # 4. 计算两个矩阵的 MSE Loss
    # 我们只关心非对角线元素（基因间的关系），对角线总是1
    # 但直接算 MSE 也可以，因为对角线误差会是0
    loss = F.mse_loss(real_corr, recon_corr)
    
    return loss

###################################################################
### 新增：per-donor 伪 bulk 合成 + GT adata + per-sample 评估   ###
###################################################################
def precompute_donor_celltype_pools(adata, donor_col='donor_id', type_col='labels',
                                     min_cells_per_type=5, allowed_indices=None):
    """
    构建 {(donor, cell_type) -> np.array(global_indices)} 池子。
    仅保留 donor 内该类型细胞数 >= min_cells_per_type 的 (donor, type) 组合。
    allowed_indices: 限定全局可用细胞索引（例如 fold 的 val_idx）。
    返回:
        pools, donors_in_pool (list[str]), types_in_pool (list[str])
    """
    donors_arr = adata.obs[donor_col].astype(str).values
    types_arr = adata.obs[type_col].astype(str).values
    n = adata.n_obs
    base = np.zeros(n, dtype=bool)
    if allowed_indices is None:
        base[:] = True
    else:
        base[np.asarray(allowed_indices, dtype=np.int64)] = True

    pools = {}
    unique_donors = sorted(np.unique(donors_arr[base]).tolist())
    unique_types  = sorted(np.unique(types_arr[base]).tolist())
    for d in unique_donors:
        for t in unique_types:
            mask = base & (donors_arr == d) & (types_arr == t)
            idx = np.where(mask)[0]
            if len(idx) >= min_cells_per_type:
                pools[(d, t)] = idx
    donors_kept = sorted({d for (d, _) in pools.keys()})
    types_kept  = sorted({t for (_, t) in pools.keys()})
    return pools, donors_kept, types_kept


def generate_synthetic_bulks_per_donor(
    adata,
    allowed_indices,
    X_log_tensor,
    mapping_dict,
    donor_col='donor_id',
    type_col='labels',
    n_bulks_per_donor=15,
    n_cells_per_bulk=500,
    alpha=0.5,
    min_cells_per_type=5,
    seed=18,
    device=None,
):
    """
    每个伪 bulk 的细胞全部来自同一个 donor。
    Dirichlet 仅在该 donor 可用的细胞类型上抽样，避免不存在的 type 被采样。
    所有 bulk 用 with-replace 内部采样以保证 n_cells_per_bulk 总量精确。

    返回:
        bulk_tensor   : torch.FloatTensor [N_bulks, G]  (log1p, CPM-like)
        gt_adata      : AnnData;obs 含 pseudo_bulk_id/donor_id/Cell_type/orig_cell_idx
        records       : list[dict]
                        keys: pseudo_bulk_id, donor_id, replicate_idx,
                              target_props (dict), actual_props (dict)
    """
    rng = np.random.default_rng(seed)
    if torch.is_tensor(X_log_tensor):
        X_log_np = X_log_tensor.cpu().numpy()
    else:
        X_log_np = X_log_tensor
    n_genes = X_log_np.shape[1]

    pools, donors_kept, _ = precompute_donor_celltype_pools(
        adata, donor_col=donor_col, type_col=type_col,
        min_cells_per_type=min_cells_per_type, allowed_indices=allowed_indices)

    if len(pools) == 0:
        print("Warning: precompute_donor_celltype_pools returned empty pool.")
        return None, None, []

    bulks = []
    gt_cell_indices = []
    gt_obs_rows = []
    records = []

    for d in donors_kept:
        avail_types = sorted([t for (dd, t) in pools.keys() if dd == d])
        if len(avail_types) == 0:
            continue
        for rep in range(n_bulks_per_donor):
            # 1) Dirichlet 仅在 avail_types 上
            p = rng.dirichlet([alpha] * len(avail_types))
            target_props = {t: float(p_t) for t, p_t in zip(avail_types, p)}

            # 2) 比例 -> 细胞数；修正取整偏差
            raw_counts = {t: max(0, int(round(p_t * n_cells_per_bulk)))
                          for t, p_t in target_props.items()}
            diff = n_cells_per_bulk - sum(raw_counts.values())
            if raw_counts:
                anchor = max(raw_counts, key=raw_counts.get)
                raw_counts[anchor] = max(0, raw_counts[anchor] + diff)

            # 3) 在 (donor, type) 池子内采样
            picked_chunks = []
            for t, k in raw_counts.items():
                if k <= 0:
                    continue
                pool = pools[(d, t)]
                replace = (k > len(pool))
                chosen = rng.choice(pool, size=k, replace=replace)
                picked_chunks.append(chosen)
            if not picked_chunks:
                continue
            picked = np.concatenate(picked_chunks).astype(np.int64)

            # 4) 聚合 bulk: log1p -> expm1 -> sum -> CPM -> log1p
            sub_log = X_log_np[picked]
            sub_lin = np.expm1(sub_log).sum(axis=0)
            row_sum = sub_lin.sum()
            if row_sum > 0:
                sub_lin = sub_lin / row_sum * 1e6
            bulk_log = np.log1p(sub_lin)

            # 5) 实际比例（防取整漂移）
            picked_types = adata.obs[type_col].astype(str).values[picked]
            n_total = float(len(picked))
            actual_props = {t: float((picked_types == t).sum()) / n_total for t in avail_types}

            pseudo_bulk_id = f"{d}__rep{rep:04d}"
            bulks.append(bulk_log.astype(np.float32))
            for ci in picked:
                gt_cell_indices.append(int(ci))
                gt_obs_rows.append({
                    'pseudo_bulk_id': pseudo_bulk_id,
                    'donor_id': d,
                    'Cell_type': str(adata.obs[type_col].iloc[int(ci)]),
                    'orig_cell_idx': int(ci),
                })
            records.append({
                'pseudo_bulk_id': pseudo_bulk_id,
                'donor_id': d,
                'replicate_idx': rep,
                'target_props': target_props,
                'actual_props': actual_props,
            })

    if not bulks:
        return None, None, []

    bulk_arr = np.stack(bulks, axis=0)
    bulk_tensor = torch.tensor(bulk_arr, dtype=torch.float32)
    if device is not None:
        bulk_tensor = bulk_tensor.to(device)

    gt_X = X_log_np[np.array(gt_cell_indices, dtype=np.int64)].astype(np.float32)
    gt_obs = pd.DataFrame(gt_obs_rows)
    gt_obs.index = [f"gt_{i}" for i in range(len(gt_obs))]
    gt_adata = sc.AnnData(X=gt_X, obs=gt_obs,
                          var=pd.DataFrame(index=list(adata.var_names)))

    print(f"[bulk synth] donors={len(donors_kept)} bulks={len(records)} "
          f"cells_per_bulk={n_cells_per_bulk} total_gt_cells={gt_adata.n_obs}")
    return bulk_tensor, gt_adata, records


def records_to_prop_matrices(records, output_dir, fold_idx=None):
    """记录的 target/actual 比例写成 2 个矩阵 csv。"""
    if not records:
        return None, None
    all_types = sorted({t for r in records for t in r['target_props'].keys()})
    rows = [r['pseudo_bulk_id'] for r in records]
    target_mat = pd.DataFrame(0.0, index=rows, columns=all_types)
    actual_mat = pd.DataFrame(0.0, index=rows, columns=all_types)
    for r in records:
        for t, v in r['target_props'].items():
            target_mat.loc[r['pseudo_bulk_id'], t] = v
        for t, v in r['actual_props'].items():
            actual_mat.loc[r['pseudo_bulk_id'], t] = v
    suffix = f"_fold_{fold_idx}" if fold_idx is not None else ""
    target_mat.to_csv(os.path.join(output_dir, f'gt_target_props{suffix}.csv'))
    actual_mat.to_csv(os.path.join(output_dir, f'gt_actual_props{suffix}.csv'))
    return target_mat, actual_mat


def evaluate_per_sample(gt_adata, pred_adata, output_dir, fold_idx=None, save_plots=True):
    """
    严格按 pseudo_bulk_id 做样本级评估。
    输出:
        per_sample_metrics.csv (列: pseudo_bulk_id, donor_id, rmse_prop, pcc_prop,
                                     mean_pcc_celltype_expr, pcc_bulk_expr)
        per_sample_metrics_boxplot.png
    """
    if 'pseudo_bulk_id' not in gt_adata.obs.columns:
        raise ValueError("gt_adata.obs missing 'pseudo_bulk_id'")
    if 'pseudo_bulk_id' not in pred_adata.obs.columns:
        raise ValueError("pred_adata.obs missing 'pseudo_bulk_id'")

    rows = []
    common_ids = sorted(set(gt_adata.obs.pseudo_bulk_id.unique()) &
                        set(pred_adata.obs.pseudo_bulk_id.unique()))
    print(f"[per-sample eval] {len(common_ids)} paired pseudo_bulk_ids "
          f"(gt={gt_adata.obs.pseudo_bulk_id.nunique()}, "
          f"pred={pred_adata.obs.pseudo_bulk_id.nunique()})")

    def _row_mean(X_block):
        if hasattr(X_block, 'toarray'):
            v = np.asarray(X_block.mean(axis=0)).flatten()
        else:
            v = np.asarray(X_block).mean(axis=0).flatten()
        return v

    def _agg_log(X_block):
        arr = X_block.toarray() if hasattr(X_block, 'toarray') else np.asarray(X_block)
        return np.log1p(np.expm1(arr).sum(axis=0)).flatten()

    for bid in common_ids:
        gt_sub   = gt_adata[gt_adata.obs.pseudo_bulk_id == bid]
        pred_sub = pred_adata[pred_adata.obs.pseudo_bulk_id == bid]
        if gt_sub.n_obs == 0 or pred_sub.n_obs == 0:
            continue

        gt_types   = gt_sub.obs['Cell_type'].astype(str)
        pred_types = pred_sub.obs['Cell_type'].astype(str)

        # 1) 比例
        gt_prop   = gt_types.value_counts(normalize=True)
        pred_prop = pred_types.value_counts(normalize=True)
        types_union = sorted(set(gt_prop.index) | set(pred_prop.index))
        g = gt_prop.reindex(types_union).fillna(0).values.astype(float)
        p = pred_prop.reindex(types_union).fillna(0).values.astype(float)
        rmse_prop = float(np.sqrt(np.mean((g - p) ** 2)))
        if g.std() > 1e-8 and p.std() > 1e-8 and len(types_union) > 1:
            pcc_prop = float(pearsonr(g, p)[0])
        else:
            pcc_prop = float('nan')

        # 2) per-celltype 表达 PCC（在该 bulk 内部）
        per_t = []
        common_types = sorted(set(gt_types) & set(pred_types))
        for t in common_types:
            mg = _row_mean(gt_sub[gt_types.values == t].X)
            mp = _row_mean(pred_sub[pred_types.values == t].X)
            if mg.std() > 1e-8 and mp.std() > 1e-8:
                per_t.append(float(pearsonr(mg, mp)[0]))
        mean_pcc_ct = float(np.mean(per_t)) if per_t else float('nan')

        # 3) 该 bulk 聚合 expression PCC
        gt_bulk   = _agg_log(gt_sub.X)
        pred_bulk = _agg_log(pred_sub.X)
        if gt_bulk.std() > 1e-8 and pred_bulk.std() > 1e-8:
            pcc_bulk = float(pearsonr(gt_bulk, pred_bulk)[0])
        else:
            pcc_bulk = float('nan')

        rows.append({
            'pseudo_bulk_id': bid,
            'donor_id': str(gt_sub.obs['donor_id'].iloc[0]),
            'n_gt_cells': int(gt_sub.n_obs),
            'n_pred_cells': int(pred_sub.n_obs),
            'rmse_prop': rmse_prop,
            'pcc_prop': pcc_prop,
            'mean_pcc_celltype_expr': mean_pcc_ct,
            'pcc_bulk_expr': pcc_bulk,
        })

    df = pd.DataFrame(rows)
    suffix = f"_fold_{fold_idx}" if fold_idx is not None else ""
    csv_path = os.path.join(output_dir, f'per_sample_metrics{suffix}.csv')
    df.to_csv(csv_path, index=False)
    print(f"[per-sample eval] saved metrics -> {csv_path}")

    if save_plots and len(df) > 0:
        plot_cols = ['rmse_prop', 'pcc_prop', 'mean_pcc_celltype_expr', 'pcc_bulk_expr']
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for ax, col in zip(axes, plot_cols):
            sub = df[col].dropna()
            if len(sub) > 0:
                sns.boxplot(y=sub, ax=ax)
                ax.set_title(f'{col}\n(median={sub.median():.3f})')
            else:
                ax.set_title(col)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'per_sample_metrics_boxplot{suffix}.png'), dpi=200)
        plt.close()
    return df


def generate_synthetic_validation_data(X_val, y_val, num_samples=10, n_genes=None, mapping_dict=None, device=None):
    """
    从验证集单细胞数据中合成 Pseudo-Bulk，用于评估模型。
    返回:
        syn_bulk_tensor: [num_samples, n_genes] (Log space)
        gt_proportions: DataFrame [num_samples, n_cell_types] (真实比例)
        sample_names: list of names
    """
    n_classes = len(mapping_dict)
    n_genes = X_val.shape[1]
    
    syn_bulk_list = []
    gt_props_list = []
    sample_names = []
    
    # 转换为 Linear 空间进行合成
    if torch.is_tensor(X_val):
        X_val_linear = torch.expm1(X_val).cpu().numpy()
        y_val_np = y_val.cpu().numpy()
    else:
        X_val_linear = np.expm1(X_val)
        y_val_np = y_val
        
    # 获取每个类别的索引
    indices_map = {v: np.where(y_val_np == v)[0] for k, v in mapping_dict.items()}
    
    for i in range(num_samples):
        # 1. 随机生成比例 (Dirichlet分布)
        # alpha < 1 模拟稀疏性，alpha > 1 模拟均匀性
        props = np.random.dirichlet(np.ones(n_classes) * 0.5)
        
        # 2. 依据比例采样细胞并聚合
        # 每个样本由约 2000 个细胞组成
        total_cells_per_sample = 2000
        sample_expr = np.zeros(n_genes)
        actual_counts = np.zeros(n_classes)
        
        for cls_name, cls_idx in mapping_dict.items():
            n_cls = int(props[cls_idx] * total_cells_per_sample)
            avail_indices = indices_map[cls_idx]
            
            if n_cls > 0 and len(avail_indices) > 0:
                # 随机抽样 (可放回)
                chosen = np.random.choice(avail_indices, n_cls, replace=True)
                sample_expr += X_val_linear[chosen].sum(axis=0)
                actual_counts[cls_idx] = n_cls
        
        # 归一化为 CPM 或均值，这里用均值模拟 Bulk
        if actual_counts.sum() > 0:
            bulk_profile = sample_expr / actual_counts.sum()
        else:
            bulk_profile = np.zeros(n_genes)
            
        # 转回 Log1p
        bulk_profile_log = np.log1p(bulk_profile)
        
        # 保存
        syn_bulk_list.append(bulk_profile_log)
        # 重新计算实际比例 (因为取整可能导致微小偏差)
        actual_props = actual_counts / actual_counts.sum()
        gt_props_list.append(actual_props)
        sample_names.append(f"Syn_Val_Rep_{i}")
        
    syn_bulk_tensor = torch.tensor(np.array(syn_bulk_list), dtype=torch.float32).to(device)
    
    # 构建真实比例 DataFrame
    inv_map = {v: k for k, v in mapping_dict.items()}
    cols = [inv_map[i] for i in range(n_classes)]
    gt_props_df = pd.DataFrame(np.array(gt_props_list), columns=cols, index=sample_names)
    
    return syn_bulk_tensor, gt_props_df, sample_names

def load_sc_data(sc_data, real_bulk_path, celltype_label=None):
    """
    加载单细胞和体视显微镜数据，为每个 bulk 样本计算细胞类型比例。
    参数：
        sc_data: 单细胞数据路径（.h5ad 文件）
        real_bulk_path: 体视显微镜数据路径（.tsv 文件）
        celltype_label: 单细胞数据中细胞类型标签的列名
    返回：
        dataset: PyTorch 数据集（单细胞数据）
        X_tensor: 单细胞表达张量
        labels: 细胞类型标签张量
        cell_type_fractions_list: 列表，每个元素是单个 bulk 样本的细胞类型比例字典
        mapping_dict: 细胞类型到整数的映射字典
        common_genes: 公共基因列表
        real_bulk_tensor: bulk数据张量（所有样本）
    """
    # 加载单细胞数据
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
    # sc.tl.rank_genes_groups(adata, 'labels', method='wilcoxon', random_state=seed)  # 使用已经log转换的数据
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
    # 统计每种细胞类型的细胞数目
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
    
     # 加载bulk数据
    real_bulk_df = pd.read_csv(real_bulk_path, sep='\t', index_col=0).T
    print(f"Range of original bulk data: {real_bulk_df.min().min()} - {real_bulk_df.max().max()}")
    real_bulk_df = real_bulk_df.loc[:, ~real_bulk_df.columns.duplicated()]

    # 计算公共基因
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
    
    # 取交集，确保核心基因也在全量基因里
    core_genes = sorted(list(set(hvg_names) & set(all_genes)))
    
    # 计算核心基因在全量基因中的索引 (用于切片)
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

    # 筛选单细胞和 bulk 数据
    adata = adata[:,all_genes].copy()
    # --- 3. 强制一致性检查 (Sanity Check) ---
    # 这步是保险，防止后续训练出现“基因错位”导致相关性为 0
    print("\n--- Gene Order Check ---")
    print(f"SC Genes (first 5):   {adata.var_names[:5].tolist()}")
    print(f"Bulk Genes (first 5): {real_bulk_log.columns[:5].tolist()}")
    
    if adata.var_names.tolist() != real_bulk_log.columns.tolist():
        raise ValueError("CRITICAL ERROR: Gene order mismatch between SC and Bulk!")
    else:
        print(">>> Gene order is perfectly aligned.")

    # 创建细胞类型矩阵
    unique_cell_types = np.unique(adata.obs['labels'])
    label2id = {label: idx for idx, label in enumerate(unique_cell_types)}
    single_cell_matrix = []
    for label in unique_cell_types:
        mask = adata.obs['labels'] == label
        mean_expr = adata[mask].X.mean(axis=0)
        single_cell_matrix.append(mean_expr)
    single_cell_matrix = np.array(single_cell_matrix).T  # [标记基因数, 细胞类型数]
    K = single_cell_matrix.shape[1]
    print(f"单细胞矩阵形状: {single_cell_matrix.shape}, 细胞类型数量: {K}")
    
    # 筛选单细胞和bulk数据
    X_tensor = torch.FloatTensor(adata.X)
    dataset = TensorDataset(X_tensor, labels)
    real_bulk_log = torch.FloatTensor(real_bulk_log.values)
    print(f"X_tensor shape: {X_tensor.shape}, labels shape: {labels.shape}, real_bulk_log shape: {real_bulk_log.shape}")
    print(f"Range of X_tensor: {X_tensor.min().item()} - {X_tensor.max().item()}")
    print(f"Range of real_bulk_log: {real_bulk_log.min().item()} - {real_bulk_log.max().item()}")
    
    # 3. 调试打印：核对前5个基因的均值
    # ------------------------------------------------------------
    print("\n--- Data Alignment Check ---")
    print(f"SC  Tensor Shape: {X_tensor.shape}")
    print(f"Bulk Tensor Shape: {real_bulk_log.shape}")
    
    # 打印前 5 个基因的平均表达量，看看是否在同一个量级，且是否有数值
    sc_mean = X_tensor.mean(dim=0)[:5].numpy()
    bulk_mean = real_bulk_log.mean(dim=0)[:5].numpy()
    print(f"Gene 1-5 ({all_genes[:5]}) Mean Expression:")
    print(f"  SC  : {sc_mean}")
    print(f"  Bulk: {bulk_mean}")
    
    # 检查 Bulk 是否全为 0
    if real_bulk_log.sum() == 0:
        raise ValueError("Fatal Error: Real Bulk Tensor is all zeros!")
    
    return dataset, X_tensor, labels, K, mapping_dict, all_genes, real_bulk_log, single_cell_matrix, cell_number_target_num, signature_array, color_map, bulk_sample_names, final_signatures, sig_indices, core_genes, core_indices_tensor

##############################
### 第一阶段: AttentionVAE ###
##############################  
class AttentionVAE(nn.Module):
    """
    条件变分自编码器（c-VAE）模型，集成了 Transformer 注意力机制，用于单细胞数据的生成和重构。
    """
    def __init__(self, input_size: int, output_size:int, hidden_size_list: list, mid_hidden_size: int, num_cell_types: int,
                 embedding_dim: int, nhead: int,  num_layers: int, ff_dim: int = 64, seed: int = seed):
        """
        初始化变分自编码器模型，包含注意力机制。

        参数：
            input_size (int): 输入基因数目（嵌入维度）。
            hidden_size_list (list): 隐藏层的维度列表，例如 [2048, 1024, 512]。
            mid_hidden_size (int): 中间隐藏层的维度（用于均值和方差）。
            embedding_dim (int): 注意力机制的嵌入维度。
            nhead (int): Transformer 的注意力头数。
            ff_dim (int): Transformer 前馈网络的维度。
            num_layers (int): Transformer 编码器层数。
        """
        super(AttentionVAE, self).__init__()
        set_seed(seed)
        # 保存输入参数
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.mid_hidden_size = mid_hidden_size
        self.embedding_dim = embedding_dim
        
        # 注意力机制的投影层
        self.linear_proj = nn.Linear(1, embedding_dim)
        
         # Transformer 编码器层
        transformers_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformers_encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embedding_dim, 1)

        # 条件嵌入层
        self.label_emb_dim = mid_hidden_size * 4
        self.label_embedding = nn.Linear(num_cell_types, self.label_emb_dim)
        nn.init.xavier_uniform_(self.label_embedding.weight)
        
        # 编码器输入维度（基因表达 + 细胞类型标签）
        self.enc_input_dim = self.input_size + mid_hidden_size*4
        
        # 构建编码器特征维度列表
        self.enc_feature_size_list = [self.enc_input_dim] + self.hidden_size_list + [self.mid_hidden_size * 4]
    
        # 构建 Encoder MLP
        self.encoder_layers = nn.ModuleList()
        in_dim = self.enc_input_dim
        for h_dim in self.enc_feature_size_list[1:]: # 跳过输入层
            self.encoder_layers.append(nn.Sequential(
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim), # 强烈建议加 BN
                nn.LeakyReLU(),
                nn.Dropout(0.1)
            ))
            in_dim = h_dim
        
        # 均值和方差层 (假设最后一层 hidden 是 mid_hidden_size * 4)
        last_enc_dim = self.enc_feature_size_list[-1]
        self.fc_mu = nn.Linear(last_enc_dim, mid_hidden_size)
        self.fc_var = nn.Linear(last_enc_dim, mid_hidden_size)

        # 构建 Decoder
        # Decoder 输入: Latent Z + Label Embedding
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
        
        # 最后一层输出基因表达
        # self.final_layer = nn.Linear(in_dim, self.input_size)
        self.final_layer = nn.Linear(in_dim, self.output_size)
    
    def encode(self, x: torch.Tensor, labels: torch.Tensor) -> tuple:
        # 1. Transformer 特征提取
        # x: [B, G]
        x_in = x.unsqueeze(-1)
        x_embed = self.linear_proj(x_in)
        x_trans = self.transformer_encoder(x_embed)
        x_feat = self.output_proj(x_trans).squeeze(-1) # [B, G]
        # 2. 获取 Label Embedding
        label_embed = self.label_embedding(labels) # [B, label_emb_dim]
        # 3. 【核心修改】拼接 (Concat)
        h = torch.cat([x_feat, label_embed], dim=1) 
        # 4. MLP
        for layer in self.encoder_layers:
            h = layer(h)  
        mu = self.fc_mu(h)
        logvar = self.fc_var(h)
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar # 返回 tuple

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # 1. 获取 Label Embedding
        label_embed = self.label_embedding(labels)
        # 2. 拼接
        h = torch.cat([z, label_embed], dim=1)
        # 3. MLP
        for layer in self.decoder_layers:
            h = layer(h)
        out = self.final_layer(h)
        return F.relu(out) # 保证非负
    
    def forward(self, x: torch.Tensor,labels: torch.Tensor, core_indices: torch.Tensor) -> tuple:
        """
        前向传播，完成编码、重参数化和解码过程。

        参数：
            x (torch.Tensor): 输入张量，形状为 (batch_size, input_size)。
            used_device (torch.device): 运行设备（CPU或GPU）。

        返回：
            tuple: 重构输出 (x_hat)、KL散度 (kl_div)。
        """
        x = x[:, core_indices]  # 仅使用核心基因
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, labels)
        kl_div = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return x_recon, mu, logvar
    
def train_vae(vae_model, X_tensor, labels, real_bulk_log,
              used_device: torch.device, batch_size: int, core_indices_tensor: torch.Tensor,
              feature_size: int, epoch_num: int, learning_rate: float, hidden_list: list, mid_hidden_size: int, num_cell_types: int, breed_2_list: list, color_map: dict, seed: int = 18, output_dir=None) -> AttentionVAE:

    set_seed(seed)
    # # --- 自动检测数据是否在 GPU 上 ---
    # is_on_gpu = X_tensor.is_cuda
    # # 如果数据在 GPU，不需要 pin_memory，否则需要
    # use_pin_memory = not is_on_gpu 
    # print(f"Data is on GPU: {is_on_gpu}. DataLoader pin_memory set to: {use_pin_memory}")
    print(f"output_dir: {output_dir}")
    # 1. 预计算索引 (加速采样)
    cell_type_indices = get_cell_type_indices(labels, num_cell_types)
    cell_type_indices = [
    torch.as_tensor(idx, device=device, dtype=torch.long)
    for idx in cell_type_indices
    ]
    # 2. 确保 X_tensor 在内存中（如果显存够大，建议 X_tensor = X_tensor.to(used_device) 以加速采样）
    # 创建 DataLoader
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

    # # 2. 检查是否有多个 GPU，如果有，使用 DataParallel 包装
    # if torch.cuda.device_count() > 1:
    #     print(f"Let's use {torch.cuda.device_count()} GPUs!")
    #     # 包装模型，自动利用所有可见 GPU
    #     vae = nn.DataParallel(vae_model)
    #     vae = vae.to(used_device)
    # else:
    #     vae = vae_model.to(used_device)
    vae = vae_model.to(used_device)
    core_indices_tensor = core_indices_tensor.to(used_device)
    criterion = nn.MSELoss()
    # [v3] marker/core 加权重构：上调信息基因(HVG/core)在 MSE 中的权重，锐化每类质心
    # （profile/marker-gene-wise 的瓶颈是质心保真度；架构其余部分保持原样不动）
    MARKER_RECON_W = 5.0
    gene_w = torch.ones(feature_size, device=used_device)
    gene_w[core_indices_tensor.to(used_device)] = MARKER_RECON_W
    gene_w = gene_w / gene_w.mean()
    optimizer = AdamW(vae.parameters(), lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # --- 权重设置 ---
    beta = 1e-5           # KLD 权重 (保持原样)
    target_beta = 0.05
    lambda_recon = 1.0
    initial_lambda_corr = 20.0
    # ---------------
    
    best_vae = None
    min_loss = float('inf')
    
    # swanlab.init(
    # project="bulk2space",
    # workspace="dendrobium",
    # config={
    #     "lr": learning_rate, "epochs": epoch_num, "mid_hidden_size": mid_hidden_size, "num_cell_types": num_cell_types,"beta": 1e-5, "beta1": 0.9, "beta2": 0.999, "lambda_denoise":100.0,"lambda_recon":100.0,"lambda_mmd":0.5,"hidden_list": hidden_list,"weight_decay": 5e-4,"embedding_dim": 16,"nhead":2,"num_layers":2,"batch_size": batch_size,"pseudo_bulk_cell_count":2000,"seed": seed})
    
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
        if epoch < warmup_epochs:
            beta = target_beta * (epoch / warmup_epochs)
        else:
            beta = target_beta
        if epoch < epoch_num * 0.5:
            lambda_corr = initial_lambda_corr
        else:
            # 线性衰减
            decay_steps = epoch_num * 0.5
            progress = (epoch - epoch_num * 0.5) / decay_steps
            lambda_corr = initial_lambda_corr * (1.0 - progress) + 1.0
        for batch_idx, (cell_features, label_indices) in enumerate(dataloader):
            # 数据上云
            time1 = time.time()
            cell_features = cell_features.to(used_device, dtype=torch.float32, non_blocking=True)
            label_one_hot = F.one_hot(label_indices, num_cell_types).float().to(used_device, non_blocking=True)
            
            # ==========================================
            # 任务 1: 纯净单细胞重构 (Standard VAE)
            # ==========================================
            sc_recon, sc_enc_mu, sc_enc_logvar = vae(cell_features, label_one_hot, core_indices_tensor) # 修改为直接返回 tuple
            is_on_gpu = cell_features.is_cuda
            # 如果数据在 GPU，不需要 pin_memory，否则需要
            use_pin_memory = not is_on_gpu 
            # print(f"cell_features is on GPU: {is_on_gpu}.")
            loss_recon = (gene_w * (sc_recon - cell_features) ** 2).mean()  # [v3] marker 加权 MSE
            loss_kld = -0.5 * torch.mean(torch.sum(1 + sc_enc_logvar - sc_enc_mu.pow(2) - sc_enc_logvar.exp(), dim=1))
            # ==========================================
            # 任务 2: 伪 Bulk 域适应 (Domain Adaptation)
            # ==========================================
            # 生成伪 Bulk Batch
            time3 = time.time()
            X_tensor = X_tensor.to(used_device)
            real_core = cell_features[:, core_indices_tensor]
            recon_core = sc_recon[:, core_indices_tensor]
            loss_corr = compute_gene_correlation_loss(real_core, recon_core)
            time4 = time.time()
            # # 检查分布重合情况
            # if batch_idx == 0 and (epoch == 0 or epoch % 10 == 0):
            #     check_distribution_overlap(
            #         cell_features,  # 这里的 cell_features 应该是已经 log1p 过的输入
            #         pseudo_bulk_log, 
            #         real_bulk_log,
            #         epoch, 
            #         output_dir
            #     )
            # ==========================================
            # 总损失
            # ==========================================
            total_loss = lambda_recon * loss_recon + beta * loss_kld + lambda_corr * loss_corr
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"\n[Error] Loss is NaN/Inf at Epoch {epoch}, Batch {batch_idx}")
                print(f"Recon: {loss_recon.item()}, KLD: {loss_kld.item()}")
                print(f"Logvar max: {sc_enc_logvar.max().item()}, min: {sc_enc_logvar.min().item()}")
                # 可以选择 break 或者 return，防止保存坏模型
                break
                return None, None
            
            optimizer.zero_grad()
            total_loss.backward()
            # 梯度裁剪 (防止梯度爆炸，特别是初期)
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss_epoch += total_loss.item()
            time2 = time.time()
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{epoch_num}], Batch [{batch_idx+1}/{len(dataloader)}], Time:{time2-time1:.2f}, pseudo-time:{time4-time3:.2f}, Total Loss: {total_loss.item():.4f}, Recon Loss: {loss_recon.item():.4f}, KLD Loss: {loss_kld.item():.4f}, Corr Loss: {loss_corr.item():.4f}")
            
        train_loss_epoch /= len(dataloader)
        scheduler.step(train_loss_epoch)
        
        # 计算监控指标
        with torch.no_grad():
            # 查看 latent 的“活性”
            curr_std = torch.exp(0.5 * sc_enc_logvar).mean().item()
        print(f"Epoch [{epoch+1}/{epoch_num}], Total Loss: {total_loss.item():.4f}, Recon Loss: {loss_recon.item():.4f}, KLD Loss: {loss_kld.item():.4f}, Corr Loss: {loss_corr.item():.4f}, Latent Std: {curr_std:.4f}, Beta: {beta:.6f}, Lambda_Corr: {lambda_corr:.4f}")
        
        # 保存最佳模型逻辑 (简化)
        if train_loss_epoch < min_loss:
            min_loss = train_loss_epoch
            best_vae = copy.deepcopy(vae)
            epoch_final = epoch
            
        # 定期保存
        if epoch % 10 == 0 and epoch != 0:
            torch.save(vae.state_dict(), os.path.join(output_dir, f'Epoch{epoch}_scvae.pth'))

        # 在训练过程中评估模型训练效果
        if epoch % 20 == 0 and epoch != 0:
            vae.eval()  # 切换到评估模式
            print(f"\n--- Epoch {epoch} Evaluation ---")
            
            eval_batch_size = 512 # 推理时可以用大一点的 batch
            eval_dataset = TensorDataset(X_tensor, labels)
            # shuffle=False 保证顺序一致
            eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0)
            
            # 容器
            all_mu_list = []
            all_logvar_list = []
            all_labels_list = []
            
            # 用于计算 Pseudo-Bulk 的累加器
            # 字典结构: {label_id: [sum_real, count_real, sum_recon, count_recon]}
            # 注意：我们在 Linear 空间做聚合，所以需要 expm1
            pb_stats = {i: {'real_sum': 0, 'real_count': 0, 'recon_sum': 0} for i in range(num_cell_types)}

            with torch.no_grad():
                for batch_x, batch_y in tqdm(eval_loader, desc="Evaluating"):
                    batch_x = batch_x.to(used_device, dtype=torch.float32)
                    batch_y_hot = F.one_hot(batch_y, num_cell_types).float().to(used_device)
                    
                    # 1. 前向传播
                    # 注意：encode 返回的是 tuple (mu, logvar)
                    batch_x_core = batch_x[:, core_indices_tensor] 
                    mu, logvar = vae.encode(batch_x_core, batch_y_hot)
                    z = vae.reparameterize(mu, logvar)
                    recon = vae.decode(z, batch_y_hot)
                    
                    # 2. 收集 Latent 用于后续 UMAP 和 Prior 计算
                    all_mu_list.append(mu.cpu().numpy())
                    all_logvar_list.append(logvar.cpu().numpy())
                    all_labels_list.append(batch_y.cpu().numpy())
                    
                    # 3. 累积计算 Pseudo-Bulk (在 Linear 空间)
                    # 这一步是为了评估：聚合后的生成数据是否像真实的 Bulk
                    batch_x_linear = torch.expm1(batch_x)
                    recon_linear = torch.expm1(recon)
                    
                    batch_y_np = batch_y.cpu().numpy()
                    for local_idx, label_id in enumerate(batch_y_np):
                        # 累加 Real
                        pb_stats[label_id]['real_sum'] += batch_x_linear[local_idx].cpu().numpy()
                        pb_stats[label_id]['real_count'] += 1
                        # 累加 Recon
                        pb_stats[label_id]['recon_sum'] += recon_linear[local_idx].cpu().numpy()

            # 合并 Latent
            all_mu = np.concatenate(all_mu_list, axis=0)
            all_logvar = np.concatenate(all_logvar_list, axis=0)
            all_labels = np.concatenate(all_labels_list, axis=0)

            # ----------------------------------------------------------
            # 评估 1: Pseudo-Bulk 相关性 (最重要的指标)
            # ----------------------------------------------------------
            print("\n>>> Metric 1: Pseudo-Bulk Correlation (Reconstruction vs Real)")
            pcc_list = []
            
            # 还要准备计算 "Prior Generation" 的数据
            cell_type_mu_logvar = {} # 存储本次验证集算出的先验，用于保存
            print(f"breed_2_list: {breed_2_list}")
            for label_id in range(num_cell_types):
                stats = pb_stats[label_id]
                
                if stats['real_count'] < 5: continue # 细胞太少跳过
                
                # A. 计算真实平均表达谱 (Ground Truth Profile)
                # Linear Mean -> Log1p (为了计算相关性，转回 Log 空间通常更符合人眼直觉，Linear 也行)
                real_profile_linear = stats['real_sum'] / stats['real_count']
                real_profile_log = np.log1p(real_profile_linear)
                
                # B. 计算重建平均表达谱
                recon_profile_linear = stats['recon_sum'] / stats['real_count']
                recon_profile_log = np.log1p(recon_profile_linear)
                
                # C. 计算相关性
                # 过滤掉全 0 基因 (可选，防止相关性报错)
                if np.std(real_profile_log) > 1e-9 and np.std(recon_profile_log) > 1e-9:
                    pcc = pearsonr(real_profile_log, recon_profile_log)[0]
                    pcc_list.append(pcc)
                    print(f"  Type {breed_2_list[label_id]:<15}: PCC = {pcc:.4f}")
                else:
                    pcc_list.append(0)
                
                # D. 同时保存该类型的 Latent 先验 (mu, logvar)
                # 这里的 mu 是该类型所有细胞 Latent 的平均值
                mask = all_labels == label_id
                mu_mean = torch.tensor(all_mu[mask].mean(axis=0, keepdims=True))
                logvar_mean = torch.tensor(all_logvar[mask].mean(axis=0, keepdims=True))
                cell_type_mu_logvar[label_id] = (mu_mean, logvar_mean)

            avg_pcc = np.mean(pcc_list) if pcc_list else 0
            print(f"--> Average Pseudo-Bulk PCC: {avg_pcc:.4f}")

            # ----------------------------------------------------------
            # 评估 2: 纯生成能力验证 (Check Prior Quality)
            # ----------------------------------------------------------
            print("\n>>> Metric 2: Pure Generation Quality (Prior -> Decoder vs Real)")
            # 这一步验证：如果我们只给 Decoder 一个类别的平均 z 和标签，它能生成正确的类别特征吗？
            gen_pcc_list = []
            
            with torch.no_grad():
                for label_id in range(num_cell_types):
                    if label_id not in cell_type_mu_logvar: continue
                    
                    # 获取该类别的真实 Profile (复用上面的计算)
                    stats = pb_stats[label_id]
                    real_profile_log = np.log1p(stats['real_sum'] / stats['real_count'])
                    
                    # 获取 Prior Mu
                    prior_mu = cell_type_mu_logvar[label_id][0].to(used_device)
                    label_vec = F.one_hot(torch.tensor([label_id]), num_cell_types).float().to(used_device)
                    
                    # 解码
                    gen_x = vae.decode(prior_mu, label_vec).cpu().numpy().flatten()
                    # 注意：假设 decode 输出已经是 log1p (或模型输出后接了relu)
                    # 如果模型输出是 linear，需要 log1p。通常 VAE 输出直接拟合 input，input 是 log1p，所以 output 也是 log1p。
                    
                    if np.std(real_profile_log) > 1e-9 and np.std(gen_x) > 1e-9:
                        gen_pcc = pearsonr(real_profile_log, gen_x)[0]
                        gen_pcc_list.append(gen_pcc)
                        print(f"  Type {breed_2_list[label_id]:<15}: Gen PCC = {gen_pcc:.4f}")
                    else:
                        gen_pcc_list.append(0)
            
            avg_gen_pcc = np.mean(gen_pcc_list) if gen_pcc_list else 0
            print(f"--> Average Generation PCC: {avg_gen_pcc:.4f}")

            # 写入日志
            with open(os.path.join(output_dir, 'metrics.txt'), 'a') as f:
                f.write(f"Epoch {epoch} | PseudoBulk_PCC: {avg_pcc:.4f} | Generation_PCC: {avg_gen_pcc:.4f}\n")

            # ----------------------------------------------------------
            # 评估 3: UMAP 可视化 Latent Space
            # ----------------------------------------------------------
            # 降采样绘图
            if all_mu.shape[0] > 10000:
                idx_plot = np.random.choice(all_mu.shape[0], 10000, replace=False)
                mu_plot = all_mu[idx_plot]
                labels_plot = [breed_2_list[id] for id in all_labels[idx_plot]]
            else:
                mu_plot = all_mu
                labels_plot = [breed_2_list[id] for id in all_labels]

            try:
                adata_latent = sc.AnnData(mu_plot, obs=pd.DataFrame({'Cell_type': labels_plot}))
                sc.pp.neighbors(adata_latent, use_rep='X') # 直接用 X (即 mu) 算 neighbor，不用再 PCA
                sc.tl.umap(adata_latent)
                
                # 保存图片
                fig = plt.figure(figsize=(10, 8))
                sc.pl.umap(adata_latent, color='Cell_type', title=f'Latent UMAP Epoch {epoch}', 
                          legend_loc='on data', show=False, palette=color_map)
                plt.savefig(os.path.join(output_dir, f'_latent_mu_epoch{epoch}.png'))
                plt.close()
            except Exception as e:
                print(f"UMAP Plotting failed: {e}")

            # 恢复训练模式
            vae.train()

    # #     swanlab.log({
    # #         "epoch": epoch,
    # #         "total_loss": train_loss_epoch,
    # #         "recon_loss": loss_recon.item(),
    # #         "kl_loss": loss_kld.item(),
    # #         "denoise_loss": loss_denoise.item(),
    # #         "mmd_loss": loss_mmd.item(),
    # #         "Latent_STD": curr_std
    # #     })
    # # swanlab.finish()
    #  训练结束，使用 Best VAE 计算统计量
    print(f"Best Loss: {min_loss:.4f} at epoch {epoch_final}")
    # 保存最终结果
    torch.save(best_vae.state_dict(), os.path.join(output_dir, 'scvae_best.pth'))
    # 保存全量细胞类型的 mu/logvar 以备后续使用
    print("\nTraining Finished. Computing FINAL Priors from TRAINING SET...")
    
    # 1. 切换到最佳模型
    state = torch.load(os.path.join(output_dir, 'scvae_best.pth'), map_location=device)
    vae.load_state_dict(state)
    if best_vae is not None:
        vae = best_vae
    vae.eval()
    
    # 2. 创建训练集的 DataLoader (不打乱顺序，batch size 可以大一点)
    # 注意：这里使用训练集 X_tensor
    final_dataset = TensorDataset(X_tensor, labels)
    final_loader = DataLoader(final_dataset, batch_size=1024, shuffle=False, num_workers=0)
    
    # 3. 容器
    all_mu_list = []
    all_logvar_list = []
    all_labels_list = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(final_loader, desc="Computing Final Priors"):
            batch_x = batch_x.to(used_device, dtype=torch.float32)
            batch_y_hot = F.one_hot(batch_y, num_cell_types).float().to(used_device)
            
            # 只需要 Encode
            batch_x_core = batch_x[:, core_indices_tensor]
            mu, logvar = vae.encode(batch_x_core, batch_y_hot)
            
            all_mu_list.append(mu.cpu().numpy())
            all_logvar_list.append(logvar.cpu().numpy())
            all_labels_list.append(batch_y.cpu().numpy())
            
    # 4. 合并
    all_mu = np.concatenate(all_mu_list, axis=0)
    all_logvar = np.concatenate(all_logvar_list, axis=0)
    all_labels = np.concatenate(all_labels_list, axis=0)
    
    # 5. 计算并保存最终的 cell_type_mu_logvar
    final_cell_type_mu_logvar = {}
    
    print("Final Prior Statistics:")
    for label_id in range(num_cell_types):
        mask = all_labels == label_id
        if mask.sum() == 0:
            print(f"Warning: Cell type {label_id} not found in training set!")
            continue
            
        # 计算均值
        # mu_mean: [1, hidden_dim]
        mu_mean = torch.tensor(all_mu[mask].mean(axis=0, keepdims=True))
        logvar_mean = torch.tensor(all_logvar[mask].mean(axis=0, keepdims=True))
        
        final_cell_type_mu_logvar[label_id] = (mu_mean, logvar_mean)
        print(f"  Type {breed_2_list[label_id]}: n={mask.sum()}")

    # 6. 保存最终文件 (覆盖之前验证集生成的)
    torch.save(final_cell_type_mu_logvar, os.path.join(output_dir, 'cell_type_mu_logvar_best.pt'))
    print("Final priors saved to cell_type_mu_logvar_best.pt")

    return best_vae, final_cell_type_mu_logvar


def solve_proportions_with_vae(vae, 
                               Basis_Used,          # [K_types, G_sig] - 已经缩放和切片的基底
                               Target_Used,         # [G_sig] - 已经切片和（如果需要）缩放的目标
                               valid_names,         # 细胞类型名称列表，与 Basis_Used 行对应
                               device, 
                               temperature=0.5,     # 温度系数 (略低于 1.0 鼓励稀疏)
                               lambda_entropy=0.1,  # 熵正则化权重 
                               lambda_reg=0.01,     # Logits L2 正则权重
                               lr=0.05,             # 学习率
                               steps=500):          # 迭代次数
    """
    使用 VAE Decoder 生成的 Basis，通过梯度下降求解细胞比例。
    此函数假定输入的 Basis_Used 和 Target_Used 已经在调用前进行了
    Signature Genes 切片和全局缩放校正。
    """
    
    num_classes_valid = Basis_Used.shape[0]
    
    # 1. 初始化优化变量
    # Logits 初始化为 0 (即初始概率均匀)
    logits = torch.zeros(num_classes_valid, requires_grad=True, device=device)
    
    # 使用 Adam 优化器
    optimizer = torch.optim.Adam([logits], lr=lr)
    
    # 2. 优化循环
    for step in range(steps):
        optimizer.zero_grad()
        
        # 计算比例 (Softmax)
        probs = F.softmax(logits / temperature, dim=0)
        
        # 混合得到 Pseudo Bulk (Linear Space)
        # [K, G_sig] * [K, 1] -> sum -> [G_sig]
        # 注意：Basis_Used 已经是线性空间且经过全局缩放
        pseudo = torch.sum(probs.view(-1, 1) * Basis_Used, dim=0)
        
        # --- Loss 1: Pearson Correlation ---
        vx = pseudo - pseudo.mean()
        vy = Target_Used - Target_Used.mean()
        
        pearson = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
        loss_corr = 1.0 - pearson
        
        # --- Loss 2: Entropy Regularization ---
        # 最小化 -H (即鼓励稀疏分布)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        loss_entropy = -entropy 
        
        # --- Loss 3: L2 Regularization on Logits ---
        loss_l2 = torch.sum(logits ** 2)
        
        # 总 Loss
        total_loss = loss_corr + lambda_entropy * loss_entropy + lambda_reg * loss_l2
        
        total_loss.backward()
        optimizer.step()
        
    # 3. 输出结果
    final_probs = F.softmax(logits / temperature, dim=0).detach().cpu().numpy()
    
    # 过滤掉极小值并重新归一化
    final_probs[final_probs < 1e-4] = 0
    final_probs = final_probs / (final_probs.sum() + 1e-8)
    
    result = {valid_names[idx]: float(prob) for idx, prob in enumerate(final_probs)}
        
    return result

def get_high_expression_mask(expression_tensor, top_percent=0.02):
    """
    识别高表达基因的掩码。
    输入: [N_genes] 或 [Batch, N_genes]
    输出: [N_genes] 的 boolean mask (True 表示保留，False 表示剔除)
    """
    # 如果是 Batch，取平均表达量
    if expression_tensor.dim() > 1:
        mean_expr = expression_tensor.mean(dim=0)
    else:
        mean_expr = expression_tensor
        
    # 计算阈值
    n_genes = mean_expr.shape[0]
    k = int(n_genes * top_percent)
    
    # 找到 Top K 的值
    # topk 返回 (values, indices)
    _, top_indices = torch.topk(mean_expr, k)
    
    # 创建掩码 (默认全 True)
    mask = torch.ones(n_genes, dtype=torch.bool, device=expression_tensor.device)
    # 将 Top K 位置设为 False
    mask[top_indices] = False
    
    return mask, top_indices

def optimize_z_and_generate_final(
    vae, 
    real_bulk_tensor,       # Log1p 格式的 Bulk Tensor [N_samples, G_full]
    sample_names, 
    mapping_dict, 
    priors,                 # 训练好的 Prior (Mean, Var)
    device, 
    output_dir,
    sig_indices,            # Signature Genes 的整数索引
    # --- 超参数 ---
    total_cells=8000,       # 每个样本生成的细胞总数
    scaling_factor=None,    # [G_full] 维度的基因缩放因子 (全局技术偏差)
    # Phase 1 参数 (保持不变，但需传入)
    lambda_entropy=0.1,     
    temperature=0.5,        
    # Phase 2A 参数 (Scaler 优化)
    lr_scaler=1.0,          # Scaler 学习率（可以高）
    steps_scaler=500,       # Scaler 优化步数
    lambda_reg_scaler=1.0,  # Scaler L2 正则权重
    # Phase 2B 参数 (Delta Z 优化)
    lr_z=0.05,              # Z 优化的学习率
    steps_z=1500,           # Z 优化的步数
    lambda_reg_z=0.1,       # Z 的 L2 正则 (防止偏离 Prior 太远)
    lambda_mse=0.1,          # 辅助 MSE 损失权重 (Linear Space)
    common_genes=None
    ):
    """
    两阶段生成流程：
    1. 使用 Signature Genes 确定细胞比例 (Robust Proportions)。
    2. 分阶段优化：(A) 学习全局基因缩放因子；(B) 优化 Latent Delta Z。
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
    
    # 反向映射 ID -> Name
    id2name = {v: k for k, v in mapping_dict.items()}

    # 预先构建 VAE 基底 (Basis Construction)
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

    # 遍历每个样本
    for i, sample_name in tqdm(enumerate(sample_names), desc="Processing Samples"):
        
        # Target Log1p (全基因)
        target_bulk_log = real_bulk_tensor[i].to(device)
        # Target Linear (全基因)
        target_bulk_linear = torch.expm1(target_bulk_log) 
        
        # =====================================================
        # Phase 1: 求解细胞比例 (使用 Signature Genes)
        # =====================================================
        # 1.1 获取签名基因基底和目标 (Linear Space)
        if not torch.is_tensor(sig_indices):
            sig_indices_tensor = torch.tensor(sig_indices, dtype=torch.long, device=device)
        else:
            sig_indices_tensor = sig_indices.to(device)

        Basis_Sig = Basis_Full_Linear[:, sig_indices_tensor]
        Target_Sig = target_bulk_linear[sig_indices_tensor]
        Scaling_Sig = init_scale[sig_indices_tensor] 

        # 应用全局缩放（修正技术偏差）
        # 关键步骤：在调用比例优化函数前，完成基底和目标的全局缩放校正
        Basis_Sig_Scaled = Basis_Sig * Scaling_Sig
        Target_Sig_Scaled = Target_Sig * Scaling_Sig # 目标也要应用缩放因子以保持一致性
        
        # 1.2 优化比例
        fractions = solve_proportions_with_vae(
            vae=vae, 
            Basis_Used=Basis_Sig_Scaled,     # 传入预处理好的基底
            Target_Used=Target_Sig_Scaled,   # 传入预处理好的目标
            valid_names=valid_names,         # 细胞类型名称列表
            device=device,
            temperature=temperature,
            lambda_entropy=lambda_entropy, 
            lambda_reg=0.01,
            lr=0.05,
            steps=500
        )
        
        # =====================================================
        # Phase 2: 优化 Latent Z 和 Scaler (使用 All Genes)
        # =====================================================
        
        # 2.1 构建初始 Z 和 Labels (与原代码相同)
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
            p_logvar = priors[t_id][1].to(device) # 获取方差
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
        
        # 定义优化变量 Delta 和 Scaler
        Delta_Z = torch.zeros_like(Prior_Z, requires_grad=True, device=device)
        # 优化 log(scale)
        log_gene_scaler = torch.nn.Parameter(torch.log(init_scale + 1e-6).clone()) 
        
        # --- Phase 2A: 优化 Scaler (冻结 Delta_Z) ---
        print(f"  > P2A: Optimizing Scaler...")
        optimizer_scaler = torch.optim.Adam([log_gene_scaler], lr=lr_scaler)
        
        for step_a in range(steps_scaler):
            optimizer_scaler.zero_grad()
            
            # 使用 Prior Z 进行解码 (相当于只优化全局修正)
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
            
            # Loss 2: Scaler Regularization (防止 Scaler 偏离全局 Init)
            loss_reg_scaler_a = torch.mean((current_scaler - init_scale)**2)
            
            total_loss_a = loss_corr + lambda_reg_scaler * loss_reg_scaler_a
            
            total_loss_a.backward()
            optimizer_scaler.step()

            if step_a % 100 == 0:
                 tqdm.write(f"    P2A Step {step_a}: Corr={pearson.item():.4f}, Scale_Reg={loss_reg_scaler_a.item():.4f}")

        # 获取最终 Scaler
        Final_Scaler = torch.exp(log_gene_scaler).detach()
        print(f"  > P2A Finished. Final Scaler Corr: {pearson.item():.4f}")
        
        # --- Phase 2B: 优化 Delta Z (冻结 Scaler) ---
        print(f"  > P2B: Optimizing Delta Z...")
        # 冻结 Scaler 参数
        log_gene_scaler.requires_grad = False
        
        optimizer_z = torch.optim.Adam([Delta_Z], lr=lr_z)
        
        for step_b in range(steps_z):
            optimizer_z.zero_grad()
            Current_Z = Prior_Z + Delta_Z
            
            # Decode -> Linear Space
            recon_log = vae.decode(Current_Z, Labels)
            recon_linear = torch.expm1(recon_log)
            
            # 聚合 & 应用固定 Scaler
            pseudo_bulk_raw = torch.sum(recon_linear, dim=0)
            pseudo_bulk_adjusted = pseudo_bulk_raw * Final_Scaler
            
            # 1. Pearson Loss (Log Space, 全基因)
            Pseudo_Log = torch.log1p(pseudo_bulk_adjusted + 1e-6)
            vx = Pseudo_Log - Pseudo_Log.mean()
            vy = target_bulk_log - target_bulk_log.mean()
            pearson = torch.sum(vx * vy) / (torch.sqrt((vx**2).sum()) * torch.sqrt((vy**2).sum()) + 1e-8)
            loss_corr = 1.0 - pearson
            
            # 2. MSE Loss (Linear Space, 确保 Read Counts 对齐)
            # 由于 Target_Linear 可能很大，MSE 权重 lambda_mse 必须设置很小
            loss_mse = F.mse_loss(pseudo_bulk_adjusted, target_bulk_linear)
            
            # 3. Z Regularization (防止 Z 漂移)
            loss_reg_z = torch.mean(Delta_Z ** 2)
            
            total_loss_b = loss_corr + lambda_mse * loss_mse + lambda_reg_z * loss_reg_z
            
            total_loss_b.backward()
            # 梯度裁剪 (防止 Delta Z 优化初期爆炸)
            torch.nn.utils.clip_grad_norm_([Delta_Z], max_norm=1.0) 
            optimizer_z.step()
            
            if step_b % 100 == 0:
                 tqdm.write(f"    P2B Step {step_b}: Corr={pearson.item():.4f}, MSE={loss_mse.item():.2f}, Z_Reg={loss_reg_z.item():.4f}")

        print(f"  > Optimized Delta Z. Final Corr (All Genes): {pearson.item():.4f}")

        # =====================================================
        # Phase 3: 生成并收集结果
        # =====================================================
        with torch.no_grad():
            Final_Z = Prior_Z + Delta_Z
            # 最终生成的细胞应该是 Log1p 格式
            Final_Cells_Log = vae.decode(Final_Z, Labels)
            
            # 理论上，VAE 的 Decode 输出应该已经是 Log1p 且无需再缩放。
            # 如果要应用 Scaler，应该在 Linear 空间应用 Scale，然后 Log1p 转换回 Log 空间。
            # 但是，由于我们优化 Z 已经考虑到 Scaler 的影响，我们直接使用 Final_Cells_Log 作为最终输出。
            
            generated_cells.append(Final_Cells_Log.cpu().numpy())
            
            for t_name in type_names:
                generated_meta.append((sample_name, t_name))
        
    # 组装 AnnData
    if not generated_cells: return None
    
    X_all = np.vstack(generated_cells)
    obs_df = pd.DataFrame(generated_meta, columns=['Sample', 'Cell_type'])
    obs_df.index = [f"Cell_{i}" for i in range(len(obs_df))]
    obs_df.to_csv(os.path.join(output_dir, f'generated_cells_metadata.csv'))
    
    # 确保 AnnData 包含正确的基因名
    adata_gen = sc.AnnData(X=X_all, obs=obs_df, var=pd.DataFrame(index=common_genes))
    
    # ... (绘图部分与原代码相同)
    print("绘制生成数据的细胞类型分布图...")
    gen_type_counts = obs_df['Cell_type'].value_counts()
    plot_smart_donut(
        type_counts=gen_type_counts,
        title=f"Generated Data Composition (Total: {len(obs_df)})",
        save_path=os.path.join(output_dir, f'generated_data_composition_donut_{sample_name}.png')
    )
    gen_type_counts.to_csv(os.path.join(output_dir, f'generated_data_composition_counts_{sample_name}.csv'))
    return adata_gen, Final_Z, obs_df

def get_dynamic_markers(X_data, y_labels, unique_labels, all_genes, n_top=5):
    """
    动态计算给定数据的差异表达基因 (Marker Genes)。
    
    参数:
        X_data: Tensor or Numpy array (Log1p expression)
        y_labels: Tensor or Numpy array (Label indices)
        unique_labels: List of label names
        all_genes: List of gene names
        n_top: 每种细胞类型提取前N个Marker
    
    返回:
        marker_dict: {cell_type: [gene1, gene2...]}
        marker_flat_list: 所有Marker的去重列表
    """
    # 1. 构建临时 AnnData
    if torch.is_tensor(X_data):
        X_np = X_data.cpu().numpy()
        y_np = y_labels.cpu().numpy()
    else:
        X_np = X_data
        y_np = y_labels
        
    obs_df = pd.DataFrame({'cell_type': [unique_labels[i] for i in y_np]})
    # 确保是 Categorical 以便排序
    obs_df['cell_type'] = obs_df['cell_type'].astype('category')
    
    adata_temp = sc.AnnData(X=X_np, obs=obs_df, var=pd.DataFrame(index=all_genes))
    
    # 2. 计算差异表达 (Wilcoxon)
    # 假设数据已经是 Log1p 的
    min_cells = 10
    label_counts = adata_temp.obs['cell_type'].value_counts()
    small_labels = label_counts[label_counts < min_cells].index
    adata_temp.obs['cell_type'] = adata_temp.obs['cell_type'].replace(small_labels, 'others')
    print(f"label_counts after filtering: \n{adata_temp.obs['cell_type'].value_counts()}")

    sc.tl.rank_genes_groups(adata_temp, 'cell_type', method='wilcoxon', use_raw=False)
    
    # 3. 提取基因
    marker_dict = {}
    flat_list = []
    
    result = adata_temp.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    
    for group in groups:
        # 获取该组的前 n_top 基因
        genes = result['names'][group][:n_top].tolist()
        marker_dict[group] = genes
        flat_list.extend(genes)
        
    # 去重但保持顺序 (这里简单排序，绘图时会重新按字典组织)
    return marker_dict, sorted(list(set(flat_list)))

def generate_pure_prior(vae, real_bulk_tensor, sample_names, mapping_dict, priors, device,
                        output_dir, sig_indices, total_cells=8000, scaling_factor=None,
                        temperature=0.5, lambda_entropy=0.1, common_genes=None):
    """[v3] 纯先验生成（统一替换 Delta_Z 生成）：
       每个样本用 signature 基因解出比例，再按比例发射 decode(prior_mu) 的类型质心细胞。
       不做 Delta_Z 优化——已实证 Delta_Z 会破坏 cross-celltype/表达保真。
       返回 (adata_gen, None, obs_df)，与原调用解包一致。"""
    vae.eval().to(device)
    for p in vae.parameters():
        p.requires_grad = False
    num_classes = len(mapping_dict)
    n_genes = real_bulk_tensor.shape[1]
    init_scale = scaling_factor.clone().to(device) if scaling_factor is not None else torch.ones(n_genes, device=device)
    id2name = {v: k for k, v in mapping_dict.items()}
    sorted_ids = sorted(priors.keys())
    basis_lin, basis_log, valid_names = [], [], []
    with torch.no_grad():
        for tid in sorted_ids:
            lab = F.one_hot(torch.tensor([tid]), num_classes).float().to(device)
            rl = vae.decode(priors[tid][0].to(device), lab)   # [1, G] log1p 质心
            basis_log.append(rl)
            basis_lin.append(torch.expm1(rl))
            valid_names.append(id2name[tid])
    Basis_Full_Linear = torch.cat(basis_lin, dim=0)   # [K, G]
    Basis_Full_Log = torch.cat(basis_log, dim=0)      # [K, G]
    sig_idx = sig_indices if torch.is_tensor(sig_indices) else torch.tensor(sig_indices, dtype=torch.long)
    sig_idx = sig_idx.to(device)
    name2row = {n: i for i, n in enumerate(valid_names)}
    sample_cells = total_cells // max(1, len(sample_names))
    gen_cells, gen_meta = [], []
    for i, sname in enumerate(tqdm(sample_names, desc="PurePrior Gen")):
        tgt_lin = torch.expm1(real_bulk_tensor[i].to(device))
        Bsig = Basis_Full_Linear[:, sig_idx] * init_scale[sig_idx]
        Tsig = tgt_lin[sig_idx] * init_scale[sig_idx]
        fractions = solve_proportions_with_vae(
            vae=vae, Basis_Used=Bsig, Target_Used=Tsig, valid_names=valid_names,
            device=device, temperature=temperature, lambda_entropy=lambda_entropy,
            lambda_reg=0.01, lr=0.05, steps=500)
        for t_name, t_id in mapping_dict.items():
            cnt = int(fractions.get(t_name, 0) * sample_cells)
            if cnt <= 0 or t_name not in name2row:
                continue
            row = Basis_Full_Log[name2row[t_name]].detach().cpu().numpy()  # [G] log 质心
            gen_cells.append(np.tile(row, (cnt, 1)))
            gen_meta.extend([(sname, t_name)] * cnt)
    if not gen_cells:
        return None
    X_all = np.vstack(gen_cells).astype(np.float32)
    obs_df = pd.DataFrame(gen_meta, columns=['Sample', 'Cell_type'])
    obs_df.index = [f"Cell_{i}" for i in range(len(obs_df))]
    obs_df.to_csv(os.path.join(output_dir, 'generated_cells_metadata.csv'))
    adata_gen = sc.AnnData(X=X_all, obs=obs_df, var=pd.DataFrame(index=common_genes))
    print(f"[pure-prior gen] {adata_gen.n_obs} cells x {adata_gen.n_vars} genes over {len(sample_names)} samples")
    return adata_gen, None, obs_df


def run_cross_validation(sc_data_path, output_dir,
                         donor_id_col='donor_id', k_folds=5, device=device, seed=18,
                         n_bulks_per_donor=15, n_cells_per_bulk=500,
                         alpha_dirichlet=0.5, min_cells_per_type=5):
    """
    保持 cVAE 不变。每个 fold 在 val donors 上做 per-donor 伪 bulk 合成。
    默认 38 donors × n_bulks_per_donor=15 = 570 个 bulk 总量（满足 >= 500）。
    """
    set_seed(seed)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
        
    # 1. 加载单细胞全量数据 (仅加载一次)
    # 这里我们只读取原始 adata，后续手动切分
    print(">>> Loading Raw Data...")
    adata = sc.read_h5ad(sc_data_path)
    print(f"Existing adata.raw.X:{adata.raw.X.shape}")
    adata.X = adata.raw.X.copy()
    print(f"Original Data: {adata.n_obs} cells, {adata.n_vars} genes.")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"Filtered Data: {adata.n_obs} cells, {adata.n_vars} genes.")
    
    # 预处理 (与原 load_sc_data 逻辑一致)
    celltype_label = 'original_ann_level_4' # 请根据你的数据修改列名
    adata.obs['labels'] = adata.obs[celltype_label].astype(str)
    
    # 基础过滤和标准化
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    if donor_id_col not in adata.obs.columns:
        raise ValueError(f"Donor ID column '{donor_id_col}' not found in adata.obs.")
    
    # Get donor groups for splitting
    donors = adata.obs[donor_id_col]
    donor_groups, _ = pd.factorize(donors) # Convert donor IDs to integer groups
    
    # 基因筛选 (取 HVG 和 预定义基因的交集)
    # 为了简化，这里假设我们使用 Top 3000 HVG
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    # 如果有特定基因列表文件，请在这里读取并求交集
    hvg_names = adata.var[adata.var['highly_variable']].index.tolist()
    # adata = adata[:, genes_to_use].copy()
    
    all_genes = adata.var_names.tolist()
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    hvg_names = adata.var[adata.var['highly_variable']].index.tolist()
    # 取交集，确保核心基因也在全量基因里
    core_genes = sorted(list(set(hvg_names) & set(all_genes)))
    

    # 准备 Tensor
    X_full = adata.X
    if issparse(X_full):
        X_full = X_full.toarray()
    X_tensor_full = torch.FloatTensor(X_full)
    
    unique_labels = sorted(adata.obs['labels'].unique())
    mapping_dict = {label: idx for idx, label in enumerate(unique_labels)}
    labels_full = torch.LongTensor([mapping_dict[l] for l in adata.obs['labels']])
    n_cell_types = len(unique_labels)
    
    print(f"Data Loaded: {X_tensor_full.shape[0]} cells, {X_tensor_full.shape[1]} genes, {n_cell_types} cell types.")
    print(f"Found {len(donors.unique())} unique donors for {k_folds}-fold group cross-validation.")
    
    # --- Initialize GroupKFold ---
    gkf = GroupKFold(n_splits=k_folds)
    
    cv_metrics = []
    
    # --- Cross-Validation Loop ---
    fold_records = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_tensor_full, labels_full, groups=donor_groups)):
        # [v3] 只跑 fold_2 验证（ONLY_FOLD 环境变量可改）
        only_fold = int(os.environ.get('ONLY_FOLD', '2'))
        if (fold + 1) != only_fold:
            print(f"  [skip] fold {fold+1} (ONLY_FOLD={only_fold})")
            continue
        print(f"\n{'='*20} Processing Fold {fold+1}/{k_folds} {'='*20}")
        fold_dir = os.path.join(output_dir, f'fold_{fold+1}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # --- A. Data Splitting by Donor ---
        X_train, y_train = X_tensor_full[train_idx], labels_full[train_idx]
        X_val, y_val = X_tensor_full[val_idx], labels_full[val_idx]
        
        # Check donor separation
        train_donors = set(donors.iloc[train_idx])
        val_donors = set(donors.iloc[val_idx])
        print(f"  Train donors: {len(train_donors)}, Val donors: {len(val_donors)}")
        fold_records.append({
        'fold': fold+1,
        'train_donors': list(train_donors),  # 转为列表便于保存
        'val_donors': list(val_donors)
    })
        assert len(train_donors.intersection(val_donors)) == 0, "Data leakage: Donors are in both train and val sets!"
        
        print("  > Computing Dynamic Marker Genes from Validation Set...")
        # 提取 top 5 markers 用于绘图验证
        val_marker_dict, val_marker_flat = get_dynamic_markers(
            X_data=X_val, 
            y_labels=y_val, 
            unique_labels=unique_labels, 
            all_genes=all_genes, 
            n_top=5
        )
        print(f"    Identified {len(val_marker_flat)} unique marker genes from validation donor(s).")

        # --- B. Signature Selection (on training set only) ---
        adata_train = sc.AnnData(X=X_train.numpy(), 
                                 obs=pd.DataFrame({'labels': [unique_labels[i] for i in y_train]}),
                                 var=pd.DataFrame(index=all_genes))
        print(" >Selecting core genes (excluding top 2% high expression genes)...")
        sc.pp.highly_variable_genes(adata_train, n_top_genes=3000)
        # 如果有特定基因列表文件，请在这里读取并求交集
        hvg_names = adata_train.var[adata_train.var['highly_variable']].index.tolist()
        core_genes = sorted(list(set(hvg_names) & set(all_genes)))
        print("  > Selecting Signature Genes from Training Fold...")
        sig_genes, _ = optimize_signature_matrix(adata_train, group_col='labels', n_signatures_per_type=50)
        final_signatures = sorted(list(set(sig_genes) & set(all_genes)))
        sig_indices = [gene_to_idx[g] for g in final_signatures]
        gene_to_idx = {g:i for i,g in enumerate(all_genes)}
        core_indices = [gene_to_idx[g] for g in core_genes]
        core_indices_tensor = torch.LongTensor(core_indices)
        print(f"    Selected {len(final_signatures)} signature genes, {len(core_indices_tensor)} core genes for this fold.")

        # --- C. Model Training ---
        print("  > Training Model...")
        vae_model = AttentionVAE(
            input_size=len(core_genes),
            output_size=len(all_genes),
            hidden_size_list=[2048, 1024, 512], mid_hidden_size=512,
            num_cell_types=n_cell_types, embedding_dim=32, nhead=4, num_layers=2
        )
        
        best_vae, priors = train_vae(
            vae_model=vae_model, 
            X_tensor=X_train, 
            labels=y_train,
            used_device=device,
            batch_size=128, 
            core_indices_tensor=core_indices_tensor,
            feature_size=len(all_genes),
            epoch_num=50, 
            learning_rate=1e-4, 
            hidden_list=[2048, 1024, 512],
            mid_hidden_size=512,
            num_cell_types=n_cell_types,
            breed_2_list=unique_labels,
            color_map={}, 
            seed=seed+fold, 
            real_bulk_log=None,
            output_dir=fold_dir
        )
        
        # --- D. Synthetic Bulk Generation (per-donor, single-donor bulks) ---
        print("  > Generating Synthetic Validation Bulk (per-donor)...")
        syn_bulk_tensor, gt_adata, records = generate_synthetic_bulks_per_donor(
            adata=adata,
            allowed_indices=val_idx,
            X_log_tensor=X_tensor_full,
            mapping_dict=mapping_dict,
            donor_col=donor_id_col,
            type_col='labels',
            n_bulks_per_donor=n_bulks_per_donor,
            n_cells_per_bulk=n_cells_per_bulk,
            alpha=alpha_dirichlet,
            min_cells_per_type=min_cells_per_type,
            seed=seed + fold,
            device=device,
        )
        if syn_bulk_tensor is None or len(records) == 0:
            print("  Synthetic bulk generation produced no records; skipping fold.")
            continue
        # 落盘 GT scRNA-seq + 目标/实际比例矩阵
        gt_adata.write_h5ad(os.path.join(fold_dir, 'gt_validation_cells.h5ad'))
        records_to_prop_matrices(records, fold_dir, fold_idx=fold+1)

        # --- Benchmark artifacts (for fair comparison with other deconv tools) ---
        # 同一份 bulk / reference 喂给所有方法 (CIBERSORTx / BayesPrism / TAPE / DISSECT)，
        # 保证输入完全一致。bulk_tensor 是 log1p(CPM)，反推 linear CPM 一并落盘。
        bulk_sample_names = [r['pseudo_bulk_id'] for r in records]
        syn_bulk_log_np = syn_bulk_tensor.detach().cpu().numpy().astype(np.float32)
        syn_bulk_lin_np = np.expm1(syn_bulk_log_np.astype(np.float64)).astype(np.float32)
        pd.DataFrame(syn_bulk_log_np, index=bulk_sample_names, columns=all_genes).to_csv(
            os.path.join(fold_dir, 'synthetic_bulks_log1p_cpm.tsv'), sep='\t')
        pd.DataFrame(syn_bulk_lin_np, index=bulk_sample_names, columns=all_genes).to_csv(
            os.path.join(fold_dir, 'synthetic_bulks_cpm_linear.tsv'), sep='\t')
        # CSV 别名（linear CPM, sample x gene），方便接受 .csv 的工具直接读
        pd.DataFrame(syn_bulk_lin_np, index=bulk_sample_names, columns=all_genes).to_csv(
            os.path.join(fold_dir, 'synthetic_bulks.csv'))
        # sample -> donor 映射
        pd.DataFrame(
            [{'pseudo_bulk_id': r['pseudo_bulk_id'], 'donor_id': r['donor_id'],
              'n_cells': n_cells_per_bulk} for r in records]
        ).set_index('pseudo_bulk_id').to_csv(
            os.path.join(fold_dir, 'synthetic_bulks_sample_map.tsv'), sep='\t')

        # Train-only reference scRNA. .X 保持 log1p(CPT)（cVAE 训练用的标度），
        # .raw 对齐到同一 gene set，便于其他工具按 raw counts / CPM 各取所需。
        ref_train = adata[train_idx].copy()
        if ref_train.raw is not None:
            ref_train.raw = ref_train.raw.to_adata()[:, list(ref_train.var_names)].copy()
        ref_train.write_h5ad(os.path.join(fold_dir, 'reference_train.h5ad'))
        print(f"  [bench] dumped synthetic_bulks*.{{tsv,csv}}, reference_train.h5ad "
              f"({ref_train.n_obs} cells x {ref_train.n_vars} genes) -> {fold_dir}")

        sc_mean = torch.expm1(X_train).mean(dim=0).to(device)
        bulk_mean = torch.expm1(syn_bulk_tensor).mean(dim=0).to(device)
        scaling_factor = bulk_mean / (sc_mean + 1e-6)

        # --- E. Deconvolution and Generation ---
        print(f"  > Optimizing and Generating Cells for {len(records)} bulks...")
        sample_names = [r['pseudo_bulk_id'] for r in records]
        sample_to_donor = {r['pseudo_bulk_id']: r['donor_id'] for r in records}
        gen_result = generate_pure_prior(
            vae=best_vae, real_bulk_tensor=syn_bulk_tensor,
            sample_names=sample_names,
            mapping_dict=mapping_dict,
            priors=priors,
            device=device,
            output_dir=fold_dir,
            sig_indices=sig_indices,
            total_cells=n_cells_per_bulk * len(sample_names),
            scaling_factor=scaling_factor,
            common_genes=all_genes
        )
        if gen_result is None:
            print("  Generation failed for this fold, skipping evaluation.")
            continue
        adata_gen, final_z, obs_df = gen_result
        # 对齐 obs：Sample -> pseudo_bulk_id；附加 donor_id
        if 'Sample' in adata_gen.obs.columns and 'pseudo_bulk_id' not in adata_gen.obs.columns:
            adata_gen.obs = adata_gen.obs.rename(columns={'Sample': 'pseudo_bulk_id'})
        adata_gen.obs['donor_id'] = adata_gen.obs['pseudo_bulk_id'].astype(str).map(sample_to_donor)
        adata_gen.write_h5ad(os.path.join(fold_dir, 'pred_validation_cells.h5ad'))

        # --- F. Per-Sample Evaluation (gt vs pred by pseudo_bulk_id) ---
        print("  > Per-Sample Evaluation...")
        per_sample_df = evaluate_per_sample(
            gt_adata=gt_adata,
            pred_adata=adata_gen,
            output_dir=fold_dir,
            fold_idx=fold+1,
        )
        # fold 级别汇总指标（中位数+均值），加入 cv_metrics
        if per_sample_df is not None and len(per_sample_df) > 0:
            fold_metrics = {
                'fold': fold + 1,
                'n_bulks': int(len(per_sample_df)),
                'rmse_prop_mean': float(per_sample_df['rmse_prop'].mean()),
                'rmse_prop_median': float(per_sample_df['rmse_prop'].median()),
                'pcc_prop_mean': float(per_sample_df['pcc_prop'].mean(skipna=True)),
                'mean_pcc_celltype_expr_mean': float(per_sample_df['mean_pcc_celltype_expr'].mean(skipna=True)),
                'pcc_bulk_expr_mean': float(per_sample_df['pcc_bulk_expr'].mean(skipna=True)),
            }
            cv_metrics.append(fold_metrics)

        del best_vae, vae_model, X_train, y_train, X_val, y_val, syn_bulk_tensor, adata_gen, gt_adata, obs_df
        torch.cuda.empty_cache()

    # --- 4. Final Summary ---
    df_folds = pd.DataFrame(fold_records)
    df_folds.to_csv(os.path.join(output_dir, 'cv_fold_donor_split.csv'), index=True)

    # 跨 fold 汇总所有 per_sample_metrics_fold_*.csv
    per_sample_files = sorted(glob.glob(os.path.join(output_dir, 'fold_*', 'per_sample_metrics_fold_*.csv')))
    if per_sample_files:
        all_dfs = [pd.read_csv(p) for p in per_sample_files]
        for fp, d in zip(per_sample_files, all_dfs):
            d['fold'] = os.path.basename(os.path.dirname(fp))
        all_per_sample = pd.concat(all_dfs, ignore_index=True)
        all_per_sample.to_csv(os.path.join(output_dir, 'cv_all_per_sample_metrics.csv'), index=False)
        print(f"\n[CV summary] total pseudo_bulks across folds: {len(all_per_sample)}")
        # 全局 boxplot
        plot_cols = ['rmse_prop', 'pcc_prop', 'mean_pcc_celltype_expr', 'pcc_bulk_expr']
        fig, axes = plt.subplots(1, len(plot_cols), figsize=(5*len(plot_cols), 5))
        for ax, col in zip(axes, plot_cols):
            sub = all_per_sample[col].dropna()
            if len(sub) > 0:
                sns.boxplot(y=sub, ax=ax)
                ax.set_title(f'{col}\nmed={sub.median():.3f} mean={sub.mean():.3f}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cv_all_per_sample_boxplot.png'), dpi=200)
        plt.close()

    if not cv_metrics:
        print("Cross-validation failed to produce any results.")
        return

    print(f"\n{'='*20} Cross Validation Finished {'='*20}")
    metrics_df = pd.DataFrame(cv_metrics)

    print("\n--- Per-Fold Aggregate Metrics ---")
    print(metrics_df.to_string())
    print("\n--- Average over folds ---")
    print(metrics_df.mean(numeric_only=True))

    metrics_df.to_csv(os.path.join(output_dir, 'cv_metrics_summary.csv'), index=False)

    plt.figure(figsize=(10, 5))
    plot_metric_cols = [c for c in metrics_df.columns if c not in ('fold', 'n_bulks')]
    sns.boxplot(data=metrics_df[plot_metric_cols], palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'{k_folds}-Fold Donor-Based CV Metrics (per-fold aggregates)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_metrics_boxplot.png'))
    plt.close()
    


##############################
### 第二阶段: 解卷积预测 ###
##############################

def evaluate_fold_results(adata_gen, gt_props_df, obs_df, X_val, y_val, mapping_dict, all_genes, output_dir, fold_idx, marker_dict):
    """
    评估单次 Fold 的结果
    """
    metrics = {}
    
    # --- 1. 细胞类型比例评估 (Proportions) ---
    # 汇总生成数据的比例
    gen_counts = obs_df.groupby(['Sample', 'Cell_type']).size().unstack(fill_value=0)
    gen_props = gen_counts.div(gen_counts.sum(axis=1), axis=0)
    
    # Save proportions to CSV
    prop_path = os.path.join(output_dir, f'fold_{fold_idx}_proportions.csv')
    gen_props.to_csv(prop_path)
    print(f"Predicted proportions saved to {prop_path}")
    
    # 对齐真实比例和预测比例的列
    common_cols = sorted(list(set(gen_props.columns) & set(gt_props_df.columns)))
    pred_vals = gen_props[common_cols].values.flatten()
    true_vals = gt_props_df[common_cols].values.flatten()
    
    # 计算 RMSE 和 Correlation
    mse = mean_squared_error(true_vals, pred_vals)
    rmse = np.sqrt(mse)
    pcc_prop, _ = pearsonr(true_vals, pred_vals)
    
    metrics['Prop_MSE'] = mse
    metrics['Prop_RMSE'] = rmse
    metrics['Prop_PCC'] = pcc_prop
    
    # 绘图: 比例散点图
    plt.figure(figsize=(6, 6))
    plt.scatter(true_vals, pred_vals, alpha=0.6)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Ground Truth Proportions')
    plt.ylabel('Predicted Proportions')
    plt.title(f'Fold {fold_idx}: Proportion Accuracy (RMSE={rmse:.3f})')
    plt.savefig(os.path.join(output_dir, f'fold_{fold_idx}_prop_scatter.png'))
    plt.close()

    # --- 2. Visualize Generated Composition ---
    gen_total_counts = obs_df['Cell_type'].value_counts()
    plot_smart_donut(
        type_counts=gen_total_counts,
        title=f"Fold {fold_idx} Generated Data Composition (Total: {len(obs_df)})",
        save_path=os.path.join(output_dir, f'fold_{fold_idx}_generated_donut.png')
    )
    
    # --- 3. Gene Expression Fidelity (Cell Type Profile Correlation) ---
    type_corrs = []
    type_mses = []
    type_rmses = []
    if torch.is_tensor(X_val): X_val = X_val.cpu().numpy()
    if torch.is_tensor(y_val): y_val = y_val.cpu().numpy()
        
    for cls_name, cls_id in mapping_dict.items():
        idx_true = np.where(y_val == cls_id)[0]
        
        # 样本太少，跳过但保留占位
        if len(idx_true) < 2:
            type_corrs.append(np.nan)
            type_mses.append(np.nan)
            type_rmses.append(np.nan)
            print(f"  Cell Type: {cls_name}, Skipped (only {len(idx_true)} samples).")
            continue
        true_profile = X_val[idx_true].mean(axis=0)
        
        if cls_name in adata_gen.obs['Cell_type'].values:
            gen_profile = adata_gen[adata_gen.obs['Cell_type'] == cls_name].X.mean(axis=0)
            valid_g = (true_profile > 0) & (gen_profile > 0)
            if valid_g.sum() > 10:
                corr, _ = pearsonr(true_profile[valid_g], gen_profile[valid_g])
                type_corrs.append(corr)
                mse_gene = mean_squared_error(true_profile[valid_g], gen_profile[valid_g])
                type_mses.append(mse_gene)
                rmse_gene = np.sqrt(mse_gene)
                type_rmses.append(rmse_gene)
                print(f"  Cell Type: {cls_name}, Profile PCC: {corr:.4f}, MSE: {mse_gene:.4f}, RMSE: {rmse_gene:.4f}")
            else:
                type_corrs.append(np.nan)
                type_mses.append(np.nan)
                type_rmses.append(np.nan)
                print(f"  Cell Type: {cls_name}, Not enough valid genes for correlation.")
        else:
            type_corrs.append(np.nan)
            type_mses.append(np.nan)
            type_rmses.append(np.nan)
            print(f"  Cell Type: {cls_name} not found in generated data for correlation.")

    pcc_df = pd.DataFrame({
        'Cell_Type': list(mapping_dict.keys()),
        'PCC': type_corrs
    })
    pcc_df.to_csv(os.path.join(output_dir, f'fold_{fold_idx}_celltype_profile_pcc.csv'), index=False)
    mse_df = pd.DataFrame({
        'Cell_Type': list(mapping_dict.keys()),
        'MSE': type_mses
    })
    mse_df.to_csv(os.path.join(output_dir, f'fold_{fold_idx}_celltype_profile_mse.csv'), index=False)
    rmse_df = pd.DataFrame({
        'Cell_Type': list(mapping_dict.keys()),
        'RMSE': type_rmses
    })
    rmse_df.to_csv(os.path.join(output_dir, f'fold_{fold_idx}_celltype_profile_rmse.csv'), index=False)
    
    avg_gene_pcc = np.mean(type_corrs) if type_corrs else 0
    metrics['Gene_Expr_PCC_Mean'] = avg_gene_pcc
    avg_gene_mse = np.mean(type_mses) if type_mses else 0
    metrics['Gene_Expr_MSE_Mean'] = avg_gene_mse
    avg_gene_rmse = np.mean(type_rmses) if type_rmses else 0
    metrics['Gene_Expr_RMSE_Mean'] = avg_gene_rmse
    
    # --- 4. Marker Gene Expression Fidelity Plot ---
    plot_marker_expression_comparison(
        adata_gen=adata_gen,
        X_val_true=X_val,
        y_val_true=y_val,
        mapping_dict=mapping_dict,
        all_genes=all_genes,
        marker_dict=marker_dict, # 传入字典
        output_dir=os.path.join(output_dir)
    )

    print(f"  [Fold {fold_idx} Eval] Prop RMSE: {rmse:.4f}, Prop PCC: {pcc_prop:.4f}, CellType Expr PCC: {avg_gene_pcc:.4f}")
    return metrics

if __name__ == "__main__":
    # 38 donors × n_bulks_per_donor=15 = 570 个 bulk 总量（>= 500）
    run_cross_validation(
        sc_data_path='/disk1/maijl/deconv/data/Simulation_HCA/HCA_sample38.h5ad',
        output_dir=output_dir,
        donor_id_col='donor_id',
        k_folds=5,
        device=device,
        n_bulks_per_donor=15,
        n_cells_per_bulk=500,
        alpha_dirichlet=0.5,
        min_cells_per_type=5,
    )
