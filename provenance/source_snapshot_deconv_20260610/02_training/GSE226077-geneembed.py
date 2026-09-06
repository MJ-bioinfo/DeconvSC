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
from sklearn.model_selection import train_test_split
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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import nnls
from scipy.stats import chi2_contingency, wasserstein_distance
from torch.optim import AdamW
import copy
from scipy.optimize import minimize
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import scale

# 设备配置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"GPU Count: {torch.cuda.device_count()}")
data_dir = '/disk1/maijl/deconv/data/GSE226077/result/gse226077_embed16_head2_layer2/allday'
output_dir = '/disk1/maijl/deconv/data/GSE226077/result/gse226077_embed16_head2_layer2/allday/application'

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
    
def load_sc_data(sc_data, real_bulk_path, celltype_label=None, sep=None):
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
    print(f"adata: {adata}")
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
    sc.pl.umap(adata, color=celltype_label, title='sc_lung_train UMAP by Cell Type', legend_loc='on data', size=30)
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
    real_bulk_df = pd.read_csv(real_bulk_path, sep=sep, index_col=0).T
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

def compute_mmd_loss(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    计算两个分布之间的 MMD 距离 (用于 Batch Size > 1)
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
    """预计算索引"""
    indices_dict = {}
    labels_np = labels.cpu().numpy()
    for i in range(num_cell_types):
        indices_dict[i] = np.where(labels_np == i)[0]
    return indices_dict

def generate_batch_dirichlet_pseudobulk(target_samples, target_labels, cell_type_indices, X_tensor, num_cell_types, device=None):
    """
    为整个 Batch 生成对应的 Dirichlet 伪 Bulk 数据。
    target_samples: [Batch, Genes]
    target_labels: [Batch] (indices)
    """
    batch_size = target_samples.size(0)
    n_genes = target_samples.size(1)
    
    # 结果容器
    pseudo_bulks = torch.zeros_like(target_samples).to(device)
    
    # 1. 生成 Batch 级的 Dirichlet 比例 [Batch, Num_Types]
    # alpha < 1 产生稀疏分布 (模拟真实组织)
    dirichlet_alpha = np.full((batch_size, num_cell_types), 0.5)
    proportions = np.random.dirichlet(np.full(num_cell_types, 0.5), size=batch_size)
    
    # 2. 每一个样本单独构建背景 (虽然有循环，但 batch=16 很快)
    num_bg_cells = 2000
    target_samples_linear = torch.expm1(target_samples) # 反 Log1p
    
    for i in range(batch_size):
        t_label = target_labels[i].item()
        bg_vector_linear = torch.zeros(n_genes).to(device)
        
        for ct_idx in range(num_cell_types):
            if ct_idx == t_label: continue # 跳过自身
            
            # 该类型需要采样的数量
            n_sample = int(proportions[i, ct_idx] * num_bg_cells)
            if n_sample > 0:
                avail_idx = cell_type_indices[ct_idx]
                if len(avail_idx) > 0:
                    # 随机采样并求和
                    chosen = np.random.choice(avail_idx, n_sample, replace=True)
                    # 注意：这里频繁的数据传输可能影响速度，若 X_tensor 在 GPU 则极快
                    if X_tensor.is_cuda:
                        chosen_tensor = torch.as_tensor(chosen, device=X_tensor.device)
                        chunk_log  = X_tensor.index_select(0, chosen_tensor)
                    else:
                        chunk_log  = X_tensor[chosen].to(device)
                    # --- 【关键修正 2】: 还原 Linear 再求和 ---
                    chunk_linear = torch.expm1(chunk_log)
                    bg_vector_linear += chunk_linear.sum(dim=0)
        
        total_bg_count = int(np.sum(proportions[i]) * num_bg_cells) - int(proportions[i, t_label] * num_bg_cells)
        if total_bg_count > 0:
            bg_mean_linear = bg_vector_linear / total_bg_count
        else:
            bg_mean_linear = torch.zeros(n_genes).to(device)
            
        # 3. 信号注入 (Signal Injection)
        # 随机目标比例 5% - 50%
        signal_frac = random.uniform(0.05, 0.50)
        # 混合：Target (Linear) + Background (Linear)
        pseudo_bulks[i] = signal_frac * target_samples_linear[i] + (1 - signal_frac) * bg_mean_linear
    # === [新增代码] 注入真实噪声 (Robustness Training) ===
    # 1. 乘性噪声 (Multiplicative Noise): 模拟 PCR 扩增偏差和测序深度波动
    # 生成均值为 1，标准差为 0.2 的噪声
    noise_mult = torch.randn_like(pseudo_bulks) * 0.2 + 1.0
    pseudo_bulks = pseudo_bulks * noise_mult
    
    # 2. 加性噪声 (Additive Noise): 模拟背景杂讯
    # noise_add = torch.randn_like(pseudo_bulks) * 0.1
    # pseudo_bulks = pseudo_bulks + noise_add
    
    # 3. Dropout (模拟 scRNA 到 Bulk 的稀疏性差异，可选)
    # dropout_mask = torch.rand_like(pseudo_bulks) > 0.1 # 随机丢弃 10% 的值
    # pseudo_bulks = pseudo_bulks * dropout_mask.float()
    
    # 确保非负
    pseudo_bulks = torch.clamp(pseudo_bulks, min=0.0)
    return pseudo_bulks

def check_distribution_overlap(sc_tensor, pb_tensor, real_bulk_tensor, epoch, output_dir):
    """
    可视化 Single Cell, Pseudo-Bulk 和 Real Bulk 的数值分布 (Log1p space)。
    
    参数:
        sc_tensor: [B, G] 单细胞 Batch (Log1p)
        pb_tensor: [B, G] 生成的伪 Bulk Batch (Log1p)
        real_bulk_tensor: [N, G] 真实的 Bulk 数据集 (Log1p) - 用作参考基准
        epoch: 当前轮数
        output_dir: 保存路径
    """
    # 1. 转换为 Numpy 并拉平
    sc_vals = sc_tensor.detach().cpu().numpy().flatten()
    pb_vals = pb_tensor.detach().cpu().numpy().flatten()
    
    # Real Bulk 可能在 GPU 或 CPU，确保转换
    if torch.is_tensor(real_bulk_tensor):
        real_vals = real_bulk_tensor.detach().cpu().numpy().flatten()
    else:
        real_vals = np.array(real_bulk_tensor).flatten()
    
    # 2. 随机采样以加快绘图速度 (各取 10,000 个点)
    sample_size = 10000
    if len(sc_vals) > sample_size:
        sc_vals = np.random.choice(sc_vals, sample_size, replace=False)
    if len(pb_vals) > sample_size:
        pb_vals = np.random.choice(pb_vals, sample_size, replace=False)
    if len(real_vals) > sample_size:
        real_vals = np.random.choice(real_vals, sample_size, replace=False)
        
    # 3. 绘图
    plt.figure(figsize=(10, 6))
    
    # 绘制 Single Cell (蓝色)
    sns.kdeplot(sc_vals, label='Single Cell (Log1p)', fill=True, color='blue', alpha=0.2)
    
    # 绘制 Pseudo Bulk (红色 - 模型生成的)
    sns.kdeplot(pb_vals, label='Pseudo Bulk (Log1p)', fill=True, color='red', alpha=0.3)
    
    # 绘制 Real Bulk (绿色 - 目标参考)
    sns.kdeplot(real_vals, label='Real Bulk (Log1p)', fill=True, color='green', alpha=0.2, linestyle='--')
    
    plt.title(f"Distribution Overlap Check - Epoch {epoch}")
    plt.xlabel("Expression Value (Log1p)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, f'dist_check_epoch_{epoch}.png')
    plt.savefig(save_path)
    plt.close()
    
    # 4. 打印统计信息
    print(f"\n[Epoch {epoch} Distribution Stats]")
    print(f"  SC       : Mean={sc_vals.mean():.4f}, Max={sc_vals.max():.4f}")
    print(f"  Pseudo-Bk: Mean={pb_vals.mean():.4f}, Max={pb_vals.max():.4f}")
    print(f"  Real-Bk  : Mean={real_vals.mean():.4f}, Max={real_vals.max():.4f}")

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
        
        # 1. 定义基因身份编码 (必须是 Parameter)
        self.gene_embedding = nn.Parameter(torch.randn(input_size, embedding_dim))
        nn.init.normal_(self.gene_embedding, mean=0, std=0.02)
        
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
        x_in = x.unsqueeze(-1) # [B, G, 1]
        x_val = self.linear_proj(x_in) # [B, G, Dim] # 投影到高维空间
        x_embed = x_val + self.gene_embedding # [B, G, Dim] 注入基因身份信息
        x_trans = self.transformer_encoder(x_embed) # [B, G, Dim] Transformer 编码
        x_feat = self.output_proj(x_trans).squeeze(-1) # [B, G] 投影回标量空间 (特征融合)
        # 2. 获取 Label Embedding
        label_embed = self.label_embedding(labels) # [B, label_emb_dim]
        # 3. 【核心修改】拼接 (Concat)
        h = torch.cat([x_feat, label_embed], dim=1)  # x_feat 包含了经过 Attention 调整过的基因表达特征
        # 4. MLP
        for layer in self.encoder_layers:
            h = layer(h)  
        mu = self.fc_mu(h)
        logvar = self.fc_var(h)
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
              feature_size: int, epoch_num: int, learning_rate: float, hidden_list: list, mid_hidden_size: int, num_cell_types: int, breed_2_list: list, color_map: dict, seed: int = 18) -> AttentionVAE:

    set_seed(seed)
    
    # --- 自动检测数据是否在 GPU 上 ---
    is_on_gpu = X_tensor.is_cuda
    # 如果数据在 GPU，不需要 pin_memory，否则需要
    use_pin_memory = not is_on_gpu 
    print(f"Data is on GPU: {is_on_gpu}. DataLoader pin_memory set to: {use_pin_memory}")
    # 1. 预计算索引 (加速采样)
    cell_type_indices = get_cell_type_indices(labels, num_cell_types)
    
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
    optimizer = AdamW(vae.parameters(), lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # --- 权重设置 ---
    beta = 1e-5           # KLD 权重 (保持原样)
    target_beta = 0.05
    lambda_denoise = 1.0 # 去噪重建权重 (像素级对齐)
    lambda_recon = 1.0
    lambda_mmd = 0.5    # MMD 权重 (分布级对齐)
    initial_lambda_corr = 20.0
    # ---------------
    
    best_vae = None
    min_loss = float('inf')
    
    swanlab.init(
    project="bulk2space",
    workspace="dendrobium",
    config={
        "lr": learning_rate, "epochs": epoch_num, "mid_hidden_size": mid_hidden_size, "num_cell_types": num_cell_types,"beta": 1e-5, "beta1": 0.9, "beta2": 0.999, "lambda_denoise":100.0,"lambda_recon":100.0,"lambda_mmd":0.5,"hidden_list": hidden_list,"weight_decay": 5e-4,"embedding_dim": 16,"nhead":2,"num_layers":2,"batch_size": batch_size,"pseudo_bulk_cell_count":2000,"seed": seed})
    
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
            # 线性衰减
            decay_steps = epoch_num * 0.5
            progress = (epoch - epoch_num * 0.5) / decay_steps
            lambda_corr = initial_lambda_corr * (1.0 - progress) + 1.0
        for batch_idx, (cell_features, label_indices) in enumerate(dataloader):
            # 数据上云
            cell_features = cell_features.to(used_device, dtype=torch.float32, non_blocking=True)
            label_one_hot = F.one_hot(label_indices, num_cell_types).float().to(used_device, non_blocking=True)
            
            # ==========================================
            # 任务 1: 纯净单细胞重构 (Standard VAE)
            # ==========================================
            sc_recon, sc_enc_mu, sc_enc_logvar = vae(cell_features, label_one_hot, core_indices_tensor) # 修改为直接返回 tuple
            
            loss_recon = criterion(sc_recon, cell_features)
            loss_kld = -0.5 * torch.mean(torch.sum(1 + sc_enc_logvar - sc_enc_mu.pow(2) - sc_enc_logvar.exp(), dim=1))
            # ==========================================
            # 任务 2: 伪 Bulk 域适应 (Domain Adaptation)
            # ==========================================
            # # 生成伪 Bulk Batch
            # pseudo_bulk_linear = generate_batch_dirichlet_pseudobulk(
            #     cell_features, label_indices, cell_type_indices, X_tensor, num_cell_types, used_device
            # )
            # pseudo_bulk_log = torch.log1p(pseudo_bulk_linear)
            # # 检查分布重合情况
            # if batch_idx == 0 and (epoch == 0 or epoch % 10 == 0):
            #     check_distribution_overlap(
            #         cell_features,  # 这里的 cell_features 应该是已经 log1p 过的输入
            #         pseudo_bulk_log, 
            #         real_bulk_log,
            #         epoch, 
            #         output_dir
            #     )
            
            # pseudo_bulk_core = pseudo_bulk_log[:, core_indices_tensor] # 仅使用核心基因
            # if isinstance(vae, nn.DataParallel):
            #     pb_mu, pb_logvar = vae.module.encode(pseudo_bulk_core, label_one_hot)
            #     pb_z = vae.module.reparameterize(pb_mu, pb_logvar)
            #     pb_recon = vae.module.decode(pb_z, label_one_hot)
            # else:
            #     pb_mu, pb_logvar = vae.encode(pseudo_bulk_core, label_one_hot)
            #     pb_z = vae.reparameterize(pb_mu, pb_logvar)
            #     pb_recon = vae.decode(pb_z, label_one_hot)
                
            # loss_denoise = criterion(pb_recon, cell_features)
            
            # 2.2 MMD 损失 (Distribution Alignment)
            # 强迫 Bulk 的 mu 分布与 Single Cell 的 mu 分布重合
            # 注意：这里比的是两个 Batch 的分布差异
            # loss_mmd = compute_mmd_loss(sc_enc_mu, pb_mu)
            X_tensor = X_tensor.to(used_device)
            real_core = cell_features[:, core_indices_tensor]
            recon_core = sc_recon[:, core_indices_tensor]
            loss_corr = compute_gene_correlation_loss(real_core, recon_core)
            # ==========================================
            # 总损失
            # ==========================================
            total_loss = lambda_recon * loss_recon + beta * loss_kld + lambda_corr * loss_corr
            
            optimizer.zero_grad()
            total_loss.backward()
            # 梯度裁剪 (防止梯度爆炸，特别是初期)
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss_epoch += total_loss.item()
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{epoch_num}], Batch [{batch_idx+1}/{len(dataloader)}], Total Loss: {total_loss.item():.4f}, Recon Loss: {loss_recon.item():.4f}, KLD Loss: {loss_kld.item():.4f}, Corr Loss: {loss_corr.item():.4f}")
            
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
        if epoch % 20 == 0: #and epoch != 0:
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

        swanlab.log({
            "epoch": epoch,
            "total_loss": train_loss_epoch,
            "recon_loss": loss_recon.item(),
            "kl_loss": loss_kld.item(),
            "corr_loss": loss_corr.item(),
            "Latent_STD": curr_std
        })
    swanlab.finish()
    #训练结束，使用 Best VAE 计算统计量
    print(f"Best Loss: {min_loss:.4f} at epoch {epoch_final}")
    # 保存最终结果
    torch.save(best_vae.state_dict(), os.path.join(output_dir, 'scvae_best.pth'))
    # 保存全量细胞类型的 mu/logvar 以备后续使用
    print("\nTraining Finished. Computing FINAL Priors from TRAINING SET...")
    
    # 1. 切换到最佳模型
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

##############################
### 第二阶段: 解卷积预测 ###
##############################
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
    priors,                 # 训练好的 Prior 字典 {type_id: (mu, logvar)}
    device, 
    output_dir,
    sig_indices,            # Signature Genes 的整数索引
    common_genes,           # 所有基因的名称列表
    # --- 超参数 ---
    total_cells=None,       # 每个样本生成的细胞总数
    scaling_factor=None,    # 全局初始缩放因子
    n_alternating_iters=3,  # [关键] P 和 Z 交替优化的轮数
    steps_z_per_iter=200,   # 每次迭代中 Z 的梯度下降步数
    lr_z=0.05,              # Z 的学习率
    lr_scaler=0.01,         # Scaler 的学习率
    lambda_reg_z=1.0,       # Z 的正则化权重 (防止偏离 Prior 太远)
    # P-Step 参数
    temperature=0.5,        
    lambda_entropy=0.1
    ):
    """
    Phase 2: 逆向优化与生成
    通过交替优化细胞比例(P)和潜在空间偏移(Delta Z)来解卷积并生成单细胞数据。
    """
    
    vae.eval().to(device)
    # 冻结 VAE 参数，我们只优化输入 Z
    for p in vae.parameters():
        p.requires_grad = False
        
    num_classes = len(mapping_dict)
    generated_cells = []
    generated_meta = []
    
    # 1. 准备全局初始 Scaler
    n_genes = real_bulk_tensor.shape[1]
    if scaling_factor is not None:
        init_scale = scaling_factor.clone().to(device)
    else:
        init_scale = torch.ones(n_genes, device=device)
        
    # 2. 准备 Prior Z 的 Centroids (用于作为优化的基准)
    # 将字典转为 Tensor，确保顺序一致
    sorted_ids = sorted(priors.keys())
    valid_names = [list(mapping_dict.keys())[list(mapping_dict.values()).index(tid)] for tid in sorted_ids]
    
    # Z_centroids_prior: [K_types, Latent_Dim]
    Z_centroids_prior = torch.cat([priors[tid][0] for tid in sorted_ids], dim=0).to(device)
    
    # 对应的 Labels (One-hot): [K_types, Num_Classes]
    Labels_K = torch.cat([F.one_hot(torch.tensor([tid]), num_classes) for tid in sorted_ids], dim=0).float().to(device)
    
    print(f"\n>>> Starting Iterative Optimization for {len(sample_names)} samples...")
    
    Final_Z_list = [] 
    # 遍历每个 Bulk 样本
    for i, sample_name in tqdm(enumerate(sample_names), desc="Processing Samples"):
        
        # --- 准备目标数据 ---
        target_bulk_log = real_bulk_tensor[i].to(device)
        target_bulk_linear = torch.expm1(target_bulk_log) #用于混合计算
        
        # --- 初始化优化变量 ---
        # Delta_Z: 我们只优化 K 个 Centroids 的偏移量，而不是几千个细胞
        # 初始化为 0，即从平均 Prior 开始
        Delta_Z_centroids = torch.zeros_like(Z_centroids_prior, requires_grad=True, device=device)
        
        # Log_Scaler: 针对该样本的基因缩放因子 (初始化为全局 Scale)
        log_gene_scaler = torch.nn.Parameter(torch.log(init_scale + 1e-6).clone())

        # 定义优化器 (同时优化 Z 和 Scaler)
        optimizer_latent = torch.optim.Adam([Delta_Z_centroids, log_gene_scaler], lr=lr_z)
        
        final_fractions = {}
        
        # =====================================================
        # Loop: 交替迭代 (Alternating Optimization)
        # =====================================================
        for iter_k in range(n_alternating_iters):
            print(f"\n--- Sample {sample_name} Alternating Iteration {iter_k+1}/{n_alternating_iters} ---")
            # -------------------------------------------------
            # Step A: 更新基底 (Basis Update)
            # 根据当前的 Z 偏移，生成新的细胞类型平均表达谱
            # -------------------------------------------------
            with torch.no_grad():
                Current_Z = Z_centroids_prior + Delta_Z_centroids
                
                # Decode -> [K, Genes] (Log space)
                recon_log = vae.decode(Current_Z, Labels_K)
                recon_linear = torch.expm1(recon_log)
                
                # 应用当前 Scaler
                current_scaler = torch.exp(log_gene_scaler)
                Basis_Current_Linear = recon_linear * current_scaler
            
            # -------------------------------------------------
            # Step B: 求解比例 (Solve Proportions)
            # 固定 Basis，解 P
            # -------------------------------------------------
            # 切片出 Signature Genes
            Basis_Sig = Basis_Current_Linear[:, sig_indices]
            Target_Sig = target_bulk_linear[sig_indices]
            
            # 调用 Phase 1 的求解函数
            final_fractions = solve_proportions_with_vae(
                vae, Basis_Sig, Target_Sig, valid_names, device, 
                temperature=temperature,
                lambda_entropy=lambda_entropy, 
                lr=0.05, steps=200 # 迭代内部步数
            )
            
            # 将比例转为 Tensor [K, 1] 用于下一步的梯度计算
            probs_list = [final_fractions.get(name, 0.0) for name in valid_names]
            Probs_Tensor = torch.tensor(probs_list, device=device, dtype=torch.float32).view(-1, 1)
            
            # -------------------------------------------------
            # Step C: 优化 Z 和 Scaler (Optimize Latent)
            # 固定 P，解 Delta_Z
            # -------------------------------------------------
            loss_history = []
            
            for step in range(steps_z_per_iter):
                optimizer_latent.zero_grad()
                
                # 1. 前向传播
                current_scaler_opt = torch.exp(log_gene_scaler)
                Current_Z_opt = Z_centroids_prior + Delta_Z_centroids # [K, Latent]
                
                recon_log_opt = vae.decode(Current_Z_opt, Labels_K)
                recon_linear_opt = torch.expm1(recon_log_opt)
                
                # 2. 混合 Pseudo Bulk (Linear Space)
                # 使用 Step B 算出的 Probs_Tensor (视为常量)
                # Formula: Sum(Proportion_k * Profile_k) * Scaler
                pseudo_bulk_raw = torch.sum(Probs_Tensor * recon_linear_opt, dim=0)
                pseudo_bulk_adjusted = pseudo_bulk_raw * current_scaler_opt
                
                # 3. 计算 Loss
                # 转换回 Log1p 空间计算 Pearson (更符合基因表达的统计特性)
                Pseudo_Log = torch.log1p(pseudo_bulk_adjusted + 1e-6)
                
                # Loss 1: Pearson Correlation (1 - PCC)
                vx = Pseudo_Log - Pseudo_Log.mean()
                vy = target_bulk_log - target_bulk_log.mean()
                # 加上 1e-8 防止除零
                pcc = torch.sum(vx * vy) / (torch.sqrt((vx**2).sum()) * torch.sqrt((vy**2).sum()) + 1e-8)
                loss_corr = 1.0 - pcc
                
                # Loss 2: Regularization (关键！)
                # 防止 Delta_Z 变得无限大以拟合噪音，强迫其保持在 Prior 附近
                loss_reg = torch.mean(Delta_Z_centroids ** 2)
                
                # Loss 3: Scaler Regularization (防止 Scaler 漂移太远)
                loss_scaler_reg = torch.mean((current_scaler_opt - init_scale)**2)
                
                # 总 Loss
                total_loss = loss_corr + lambda_reg_z * loss_reg + 0.1 * loss_scaler_reg
                
                total_loss.backward()
                # 梯度裁剪防止爆炸
                torch.nn.utils.clip_grad_norm_([Delta_Z_centroids], 1.0)
                optimizer_latent.step()
                
                loss_history.append(total_loss.item())

            # 打印本轮迭代的收敛情况
            tqdm.write(f"  Sample {sample_name} Iter {iter_k+1}/{n_alternating_iters}: PCC={pcc.item():.4f}, Reg={loss_reg.item():.4f}")

        # =====================================================
        # Final Generation: 采样生成 (Sampling with Heterogeneity)
        # =====================================================
        # 此时我们得到了最优的 Delta_Z_centroids (代表该病人的平均偏移)
        # 我们将其应用到 Prior 分布上进行采样
        
        optimized_delta = Delta_Z_centroids.detach()
        sample_cells_count = total_cells // len(sample_names)
        
        
        
        for t_idx, t_id in enumerate(sorted_ids):
            t_name = valid_names[t_idx]
            fraction = final_fractions.get(t_name, 0.0)
            count = int(fraction * sample_cells_count)
            
            if count <= 0: continue
            
            # 1. 获取训练好的 Prior 参数 (Mean & LogVar)
            prior_mu, prior_logvar = priors[t_id]
            prior_mu = prior_mu.to(device)
            prior_std = torch.exp(0.5 * prior_logvar.to(device))
            
            # 2. 获取优化后的偏移
            current_delta = optimized_delta[t_idx].unsqueeze(0) # [1, Latent]
            
            # 3. 核心步骤：构建目标分布并采样
            # Target_Mean = Prior_Mean + Optimized_Delta
            # Z ~ N(Target_Mean, Prior_Std)
            # 这样生成的 Z 既包含了病人的特异性(Delta)，又保留了原始的异质性(Std)
            eps = torch.randn(count, prior_mu.shape[1], device=device)
            z_sampled = (prior_mu + current_delta) + eps * prior_std
            
            # 4. 解码
            label_sampled = F.one_hot(torch.tensor([t_id]*count), num_classes).float().to(device)
            decoded_cells_log = vae.decode(z_sampled, label_sampled)
            
            # 5. 收集
            generated_cells.append(decoded_cells_log.cpu().numpy())
            generated_meta.extend([(sample_name, t_name)] * count)
            
            # 6. 保存最终的 Z 向量
            Final_Z_list.append(z_sampled.detach().cpu())

    # =====================================================
    # 结果组装
    # =====================================================
    if not generated_cells:
        print("Error: No cells generated.")
        return None, None, None, None
        
    X_all = np.vstack(generated_cells)
    obs_df = pd.DataFrame(generated_meta, columns=['Sample', 'Cell_type'])
    obs_df.index = [f"Cell_{i}" for i in range(len(obs_df))]
    
    # 创建 AnnData
    adata_gen = sc.AnnData(X=X_all, obs=obs_df)
    adata_gen.var_names = common_genes # 确保基因名正确
    
    print(f"Generation Complete. Created AnnData with shape {adata_gen.shape}")
    
    # 保存结果
    adata_gen.write_h5ad(os.path.join(output_dir, 'generated_data_final.h5ad'))
    obs_df.to_csv(os.path.join(output_dir, 'generated_metadata.csv'))
    
    return adata_gen, Delta_Z_centroids, obs_df, Final_Z_list

def get_sample_indices_from_bulk(real_bulk_path, target_samples, sep):
    """
    读取 real_bulk.csv 并返回目标样本的索引，同时返回实际找到的样本名称列表。
    """
    # 读取 Bulk 数据
    try:
        # 假设是 TSV 格式，如果 index 在第一列
        real_bulk_df = pd.read_csv(real_bulk_path, sep=sep, index_col=0).T
    except Exception as e:
        print(f"Error reading {real_bulk_path}: {e}")
        return [], []
    
    # 获取所有样本名
    all_sample_names = real_bulk_df.index.tolist()
    print(f"Total samples in bulk data: {len(all_sample_names)}")
    
    sample_indices = []
    valid_target_samples = [] # 实际找到的样本名
    
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


def get_latent_adata(Final_Z_list, adata_generated, latent_dim=256):
    """创建包含潜在向量和元数据的 AnnData 对象"""
    # --- 修复核心：确保 Final_Z_list 是一个 Tensor 序列 ---
    if isinstance(Final_Z_list, torch.Tensor):
        # 如果它本身就是一个 Tensor，说明只处理了一个样本，将其包装成列表
        print("Note: Final_Z_list is a single Tensor, wrapping it in a list.")
        Final_Z_list = [Final_Z_list]
    elif not isinstance(Final_Z_list, (list, tuple)) or not all(isinstance(z, torch.Tensor) for z in Final_Z_list):
        # 否则，如果它不是 Tensor 列表，可能是空列表或其他问题
        raise ValueError("Final_Z_list must be a list or tuple of PyTorch Tensors.")
    
    # 确保 Z 向量和细胞数量匹配
    # Z_all = torch.cat(Final_Z_list, dim=0).detach().cpu().numpy()
    Z_all = torch.cat(Final_Z_list, dim=0).cpu().numpy()
    print(Z_all.shape)
    # 将 Z 投影到一个合理的维度 (如果 Latent_Dim > 50)
    if Z_all.shape[1] > 50:
        reducer = umap.UMAP(n_components=50, random_state=18)
        Z_reduced = reducer.fit_transform(Z_all)
    else:
        Z_reduced = Z_all

    adata_latent = ad.AnnData(X=Z_reduced)
    adata_latent.obsm['Z_latent'] = Z_all # 存储全量 Z
    
    # 继承元数据
    adata_latent.obs = adata_generated.obs.copy()
    adata_latent.obs['Sample_Status'] = adata_latent.obs['Sample'].apply(
    lambda x: 'Day 0' if 'D0' in str(x) else ('Day 4' if 'D4' in str(x) else 'Other')
    ).astype('category')
    
    # 计算 Delta Z 的幅度 (假设我们在优化 Delta Z)
    # Note: 这一步需要 Delta_Z 向量，如果 Phase 2B 成功，你需要修改代码来保存它
    # 此处省略 Delta Z 幅度计算，专注于 Z_latent 比较
    
    sc.pp.neighbors(adata_latent, use_rep='X')
    sc.tl.umap(adata_latent)
    return adata_latent

def validate_iMGL_development(adata_generated, Final_Z_list, output_dir, device, all_genes):
    print("\n--- Validating iMGL Development and Heterogeneity ---")
    
    # 1. 潜在空间 UMAP 可视化
    adata_latent = get_latent_adata(Final_Z_list, adata_generated)
    
    fig = plt.figure(figsize=(18, 7))
    plt.subplot(1, 2, 1)
    sc.pl.umap(adata_latent, color='Cell_type', title='Latent UMAP by Cell Type', 
               show=False, ax=plt.gca())
    palette = {
    'Day 0': '#1f77b4',
    'Day 4': '#ff7f0e',
    'Other': '#d62728'  
    }
    plt.subplot(1, 2, 2)
    sc.pl.umap(adata_latent, color='Sample_Status', title='Latent UMAP by Sample Day (Day 0 vs Day 4)', 
               show=False, ax=plt.gca(), 
               # palette=palette
               )
    
    plt.tight_layout()
    umap_path = os.path.join(output_dir, 'iMGL_latent_umap_comparison.png')
    plt.savefig(umap_path, dpi=300)
    plt.close()
    print(f"Latent UMAP saved to {umap_path}")
    
    marker_genes_dict = {
    "Core markers": ["SPI1","CX3CR1","P2RY13","MAF","MEF2C","OLFML3","SELPLG","CSF1R","P2RY12","TMEM119"],
    "Antigen presentation": ["HLA-DRA","HLA-DRB1","HLA-DPA1","HLA-DPB1","CD74"],
    "Complement": ["C1QA","C1QC","C3"],
    "AD risk genes": ["TREM2","BIN1","SPP1","CD33","CLEC7A","MS4A6A","SORL1","GRN"],
    "Activation": ["TYROBP","LGALS9","IRF8","TGFBR1","TGFBR2","MARCKS","CXCL8","APOC1","LGALS3","CTSB","CTSD","MAFB","LYZ","CD68","PPARG","GPNMB","CCL2","MRC1","MERTK","CD83","S100A8","S100A9","CSF3R","LRRK2","IL1B","IL6R","IL17RA"],
    "Immediate early genes": ["TREM1","EGR1","EGR2","FOS","JUN"],
    "Cell cycle": ["MKI67","TOP2A","CDK1","CCNA2"],
    "Heat shock": ["HSP90AA1","HSPD1","HSPE1"],
    }
    # 在 AnnData 中添加 Sample_Status
    adata_generated.obs['Sample_Status'] = adata_latent.obs['Sample_Status']
    
    # 为了简化 DotPlot，我们只需要一个“细胞类型”（iMGL）
    # 但我们按样本状态分组
    fig = plt.figure(figsize=(15, 8))
    sc.pl.dotplot(
        adata_generated, 
        marker_genes_dict, 
        groupby='Sample_Status', 
        # use_raw=False, # 假设 X 已经是 Log1p
        cmap='viridis', 
        standard_scale='var', # 按基因标准化，对比相对表达量
        figsize=(12, 6),
        save='_iMGL_marker_dotplot_Samplestatus.png',
        show=False
    )
    plt.close()
    
    fig = plt.figure(figsize=(15, 8))
    sc.pl.dotplot(
        adata_generated, 
        marker_genes_dict, 
        groupby='Cell_type', 
        # use_raw=False, # 假设 X 已经是 Log1p
        cmap='viridis', 
        standard_scale='var', # 按基因标准化，对比相对表达量
        figsize=(12, 6),
        save='_iMGL_marker_dotplot_celltype.png',
        show=False
    )
    plt.close()
    print("Marker DotPlot saved.")
    
    # 3. 绘制adata_gen的UMAP，并按Sample_Status和Cell_type着色
    adata_X = adata_generated.copy()
    print(f"range of generated X: min={adata_X.X.min():.4f}, max={adata_X.X.max():.4f}, mean={adata_X.X.mean():.4f}, adata.shape={adata_X.shape}")
    
    sc.pp.highly_variable_genes(adata_X, n_top_genes=2000, subset=True, flavor='seurat_v3')
    adata_X = adata_X[:, adata_X.var.highly_variable]
    # 由于数据已经是归一化 Log 空间，直接进行 PCA
    sc.tl.pca(adata_X, svd_solver='arpack', n_comps=50) 
    sc.pp.neighbors(adata_X, n_neighbors=15, n_pcs=50)
    sc.tl.umap(adata_X, min_dist=0.5, spread=1.0)
    
    # 绘制按 Sample Status 和 Cell Type 着色的 UMAP
    fig_X, axes_X = plt.subplots(1, 2, figsize=(18, 8))
    
    # 图 1: 按 VAE 预测的 Cell_type 标签着色 (验证生成数据是否保留了预测标签的结构)
    sc.pl.umap(adata_X, color='Cell_type', title='Generated X UMAP: VAE Predicted Type', 
               ax=axes_X[0], show=False)
               
    # 图 2: 按 Sample Status (Day 0 vs Day 4) 着色 (验证是否捕捉到发育差异)
    sc.pl.umap(adata_X, color='Sample_Status', title='Generated X UMAP: Sample Status', 
               ax=axes_X[1], show=False, palette={'Day 0': 'skyblue', 'Day 4': 'darkred', 'Other': 'lightgrey'})
               
    plt.tight_layout()
    umap_X_path = os.path.join(output_dir, 'iMGL_generated_X_umap_comparison.png')
    plt.savefig(umap_X_path, dpi=300)
    plt.close(fig_X)
    print(f"Generated X UMAP saved to {umap_X_path}")
    
    # 3. 潜在向量差异量化
    # 目标：证明 Day 4 的 Z 中心与 Day 0 的 Z 中心在潜在空间中存在差异
    Z_all_tensor = torch.tensor(adata_latent.obsm['Z_latent'], device=device)
    
    # 假设 Day 0 对应 index 0, Day 4 对应 index 1
    mask_day0 = (adata_latent.obs['Sample_Status'] == 'Day 0').values
    mask_day4 = (adata_latent.obs['Sample_Status'] == 'Day 4').values
    
    Z_mean_day0 = Z_all_tensor[mask_day0].mean(dim=0)
    Z_mean_day4 = Z_all_tensor[mask_day4].mean(dim=0)
    
    # 计算两个 Z 中心点的欧氏距离
    dist_z = torch.norm(Z_mean_day0 - Z_mean_day4).item()
    print(f"\nQuantitative Z-space Check:")
    print(f"Euclidean Distance between Z-Centers (Day 0 vs Day 4): {dist_z:.4f}")
    
    if dist_z < 0.1:
        print("WARNING: Z-space centers are too close. Model may fail to capture GEP difference.")

    marker_genes =  [g for v in marker_genes_dict.values() for g in v]
    marker_genes_list = list(dict.fromkeys(marker_genes))
    # 4. (可选) 检查关键 Marker 的平均表达增益
    # 检查小胶质细胞成熟 Marker (如 TMEM119) 在 Day 4 是否显著高于 Day 0
    day0_df = pd.DataFrame(adata_generated[adata_generated.obs['Sample_Status'] == 'Day 0', marker_genes_list].X.mean(axis=0), index=marker_genes_list, columns=['Day 0 Mean'])
    day4_df = pd.DataFrame(adata_generated[adata_generated.obs['Sample_Status'] == 'Day 4', marker_genes_list].X.mean(axis=0), index=marker_genes_list, columns=['Day 4 Mean'])
    
    mean_comp = day0_df.join(day4_df)
    
    print("\nKey Marker Mean Expression Comparison (Log1p):")
    print(mean_comp.loc[["TMEM119", "P2RY12", "C1QA", "MKI67"]].to_string())
    
    if mean_comp.loc["TMEM119", 'Day 4 Mean'] > mean_comp.loc["TMEM119", 'Day 0 Mean'] * 1.5:
        print("SUCCESS: TMEM119 expression shows expected maturation signal.")
    else:
         print("FAILURE: TMEM119 signal is low or equal. Marker issue persists.")

################################
### 端到端工作流程 ###
################################
def main(sc_data, real_bulk_path, output_dir, device=device, epochs=100, lr=0.001):
    """
    主函数，执行从数据加载到单细胞数据生成的完整流程。

    参数：
        sc_data: 单细胞数据路径（.h5ad 文件）
        real_bulk_path: 体视显微镜数据路径（.tsv 文件）
        output_dir: 输出目录
        device: 计算设备（'cuda' 或 'cpu'）
        epochs: BulkVAE 训练轮数
        lr: 学习率
    """
    # 1. 配置参数
    dataset, X_tensor, labels, n_cell_types, mapping_dict, all_genes, real_bulk_log, single_cell_matrix, cell_number_target_num, signature_array, color_map, bulk_sample_names, final_signatures, sig_indices, core_genes, core_indices_tensor = load_sc_data(
        sc_data, real_bulk_path, celltype_label='marker_cell_type',sep=','
    )
    print("数据加载完成，细胞数量:", len(dataset), "基因数量:", X_tensor.shape[1], "细胞类型数量:", len(mapping_dict), "bulk样本:", real_bulk_log.shape)
    breed_2_list = list(mapping_dict.keys())

    sc_metadata = {
    'input_dim': len(all_genes),
    'X_tensor': X_tensor,
    'labels': labels,
    'n_cell_types': n_cell_types,
    'real_bulk_log': real_bulk_log,
    'mapping_dict': mapping_dict,
    'all_genes': all_genes,
    'cell_number_target_num': cell_number_target_num,
    'signature_array': signature_array,
    'breed_2_list': breed_2_list,
    'color_map': color_map,
    'bulk_sample_names': bulk_sample_names,
    'signatures': final_signatures,
    'sig_indices': sig_indices,
    'core_genes': core_genes,
    'core_indices_tensor': core_indices_tensor
    }

    # 保存到文件
    with open(os.path.join(output_dir, 'sc_metadata.pkl'), 'wb') as f:
        pickle.dump(sc_metadata, f)

    # # 第一阶段：训练Attention GMVAE模型
    hidden_size_list=[4096, 2048, 1024]
    with open(os.path.join(output_dir, 'sc_metadata.pkl'), 'rb') as f:
        sc_metadata = pickle.load(f)
    X_tensor = sc_metadata['X_tensor']
    labels = sc_metadata['labels']
    n_cell_types = sc_metadata['n_cell_types']
    real_bulk_log = sc_metadata['real_bulk_log']
    mapping_dict = sc_metadata['mapping_dict']
    all_genes = sc_metadata['all_genes']
    cell_number_target_num = sc_metadata['cell_number_target_num']
    breed_2_list = sc_metadata['breed_2_list']
    color_map = sc_metadata['color_map']
    bulk_sample_names = sc_metadata['bulk_sample_names']
    signatures = sc_metadata['signatures']
    sig_indices = sc_metadata['sig_indices']
    core_genes = sc_metadata['core_genes']
    core_indices_tensor = sc_metadata['core_indices_tensor']
    
    scvae = AttentionVAE(input_size=len(core_genes), 
                         output_size=len(all_genes),
                         hidden_size_list=hidden_size_list,mid_hidden_size=256,num_cell_types=n_cell_types, embedding_dim=32,nhead=4,num_layers=4)
    scvae.to(device).train()
    scvae, cell_type_mu_logvar = train_vae(vae_model=scvae,
                                           X_tensor=X_tensor,labels=labels,
                                           real_bulk_log=real_bulk_log,
                                           batch_size=512,
                                           feature_size=len(all_genes),
                                           epoch_num=200,
                                           learning_rate=5e-5,hidden_list=hidden_size_list,mid_hidden_size=256,num_cell_types=n_cell_types,breed_2_list=breed_2_list, color_map=color_map, used_device=device, core_indices_tensor=core_indices_tensor)

    # # 第二阶段：逐bulk样本生成单细胞表达矩阵
    # # 计算bulk全局的缩放校正因子
    # 1. 计算 Training SC 的 Linear Mean
    # X_tensor 是 Log1p 的
    sc_mean_linear = torch.expm1(X_tensor).mean(dim=0)
    # 2. 计算 Real Bulk 的 Linear Mean
    # real_bulk_log 假设也是 Log1p 的
    bulk_mean_linear = torch.expm1(real_bulk_log).mean(dim=0)
    # 3. 计算因子
    # 加上 1e-6 防止除以 0
    scaling_factor = bulk_mean_linear / (sc_mean_linear + 1e-6)
    print(f"Global scaling factor calculated: {scaling_factor.mean().item():.4f}.")
    scaling_factor = scaling_factor.to(device)
    
    state = torch.load(os.path.join(output_dir, 'scvae_best.pth'), map_location=device)
    scvae.load_state_dict(state)
    cell_type_mu_logvar_best = torch.load(os.path.join(output_dir, 'cell_type_mu_logvar_best.pt'), map_location=device)
    
    target_samples=[
        #'GSM7062715_Bulk_iMGL_D0_rep1',
        'GSM7062716_Bulk_iMGL_D0_rep2','GSM7062717_Bulk_iMGL_D0_rep3',
        # 'GSM7062718_Bulk_iMGL_D1_rep1',
        'GSM7062719_Bulk_iMGL_D1_rep2','GSM7062720_Bulk_iMGL_D1_rep3',
        #'GSM7062721_Bulk_iMGL_D2_rep1',
'GSM7062722_Bulk_iMGL_D2_rep2','GSM7062723_Bulk_iMGL_D2_rep3',#'GSM7062724_Bulk_iMGL_D3_rep1','GSM7062725_Bulk_iMGL_D3_rep2','GSM7062726_Bulk_iMGL_D3_rep3',
#'GSM7062727_Bulk_iMGL_D4_rep1',
       'GSM7062728_Bulk_iMGL_D4_rep2','GSM7062729_Bulk_iMGL_D4_rep3'
                    ]

    # 2. 获取索引和有效的样本名称 (防止某个样本名在文件中写错或不存在)
    sample_indices, valid_sample_names = get_sample_indices_from_bulk(
        real_bulk_path, 
        target_samples=target_samples,
        sep=','
    )
    if not sample_indices:
        raise ValueError("没有找到有效的样本，无法生成。")
    input_bulk_log = real_bulk_log[sample_indices]
    print(f"Selected bulk samples indices: {sample_indices}, shape: {input_bulk_log.shape}")

    # # 2. 准备对应的细胞比例列表 (非常重要！必须切片以对齐 input_bulk)
    # input_fractions_list = [cell_type_fractions_list[i] for i in sample_indices]

    # print(f"Selected bulk samples indices: {sample_indices}")
    # print(f"Input bulk shape: {input_bulk_log.shape}")
    # print("开始使用【优化策略】生成单细胞表达矩阵...")

    # 3. 调用新的优化生成函数
    scvae = scvae.to(device)
    total_cells = n_cell_types * 500
    # adata_generated, Delta_Z_centroids, obs_df, Final_Z_list = optimize_z_and_generate_final(
    #     vae=scvae,
    #     real_bulk_tensor=input_bulk_log,
    #     sample_names=valid_sample_names,
    #     mapping_dict=mapping_dict,
    #     priors=cell_type_mu_logvar_best, # 训练好的先验
    #     device=device,
    #     output_dir=output_dir,
    #     sig_indices = sig_indices,
    #     total_cells = total_cells,
    #     common_genes=all_genes,
    #     scaling_factor=scaling_factor,
    #     lambda_entropy = 0.1,
    #     temperature = 0.5,
    #     lr_scaler=1.0,
    #     # lambda_reg_scaler=1.0,
    #     lr_z = 0.1,
    #     # steps_z = 1000,
    #     lambda_reg_z = 0.5,
    #     # lambda_mse=0.0,
    #     n_alternating_iters=1,  # [关键] P 和 Z 交替优化的轮数
    #     steps_z_per_iter=1000, 
    # )
    
    # 4. 验证 iMGL 发育过程中的异质性捕捉
    validate_iMGL_development(
    adata_generated=adata_generated,
    Final_Z_list=Final_Z_list,
    output_dir=output_dir,
    device=device,
    all_genes=all_genes
    )

################################
### 示例用法 ###
################################

if __name__ == "__main__":
    main(
        # sc_data='/disk1/maijl/deconv/data/GSE226077/GSE226077_allgene_processed_afterR.h5ad',
        sc_data='/disk1/maijl/deconv/data/GSE226077/GSE226077_trainset.h5ad',
        real_bulk_path='/disk1/maijl/deconv/data/GSE226077/GSE226077_bulk_TPM_matrix.csv',
        output_dir=output_dir,
         device=device)
