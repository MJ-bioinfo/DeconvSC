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
from torch.distributions import Dirichlet
warnings.filterwarnings("ignore")
from sklearn.preprocessing import scale

# 设备配置
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"GPU Count: {torch.cuda.device_count()}")
data_dir = '/disk1/maijl/deconv/data/GSE141115'
output_dir = '/disk1/maijl/deconv/cVAE/GSE141115/attention_geneembed'
config_dir = '/disk1/maijl/deconv/cVAE/GSE141115'

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

def generate_batch_dirichlet_pseudobulk(target_samples: torch.Tensor, 
                                                  target_labels: torch.Tensor, 
                                                  cell_type_indices: list, # List of NumPy arrays or Tensors of indices
                                                  X_tensor: torch.Tensor, 
                                                  num_cell_types: int, 
                                                  device: torch.device):
    """
    为整个 Batch 生成对应的 Dirichlet 伪 Bulk 数据 (GPU 优化版本)。
    target_samples: [Batch, Genes] (Log1p scale)
    target_labels: [Batch] (indices)
    X_tensor: [Total_Cells, Genes] (Log1p scale, preferably on GPU)
    """
    
    batch_size = target_samples.size(0)
    n_genes = target_samples.size(1)
    
    # 确保所有张量都在目标设备上
    target_samples = target_samples.to(device)
    target_labels = target_labels.to(device)
    
    # --- 1. 生成 Batch 级的 Dirichlet 比例 [Batch, Num_Types] (GPU) ---
    
    # 使用 PyTorch 的 Dirichlet 分布（默认在 CPU 上创建，需要转移到 GPU）
    alpha_tensor = torch.full((num_cell_types,), 0.5, device=device)
    dirichlet_dist = Dirichlet(alpha_tensor)
    proportions_tensor = dirichlet_dist.sample((batch_size,)) # [Batch, Num_Types]
    
    # 结果容器
    pseudo_bulks_linear = torch.zeros_like(target_samples).to(device)
    
    # --- 2. 准备数据和参数 ---
    num_bg_cells = 2000
    # 反 Log1p (GPU)
    target_samples_linear = torch.expm1(target_samples) 
    
    # 检查 X_tensor 是否在 GPU 上，如果不在，需要处理索引数据传输
    X_tensor_device = X_tensor.device
    
    # --- 3. 核心循环 (循环仍在，但内部计算 GPU 化) ---
    
    # 预计算每个样本的总背景细胞数 (用于后面的均值计算)
    # 我们需要一个布尔掩码来排除目标细胞类型自身
    # 由于 proportions_tensor 已经在 GPU 上，我们可以直接操作
    
    # target_labels_expanded: [Batch, 1]
    target_labels_expanded = target_labels.unsqueeze(1)
    # 构造一个与 proportions 相同形状的张量，值为 batch 中每个样本的 target label
    target_labels_matrix = target_labels_expanded.repeat(1, num_cell_types)
    
    # 构造一个 Cell Type 索引张量 [1, Num_Types]
    ct_indices_tensor = torch.arange(num_cell_types, device=device).unsqueeze(0)
    
    # 创建排除目标细胞类型的掩码 [Batch, Num_Types]
    non_target_mask = (ct_indices_tensor != target_labels_matrix)
    
    # 移除自身比例
    bg_proportions = proportions_tensor * non_target_mask.float()
    
    # 总采样细胞数（不包括目标细胞类型）
    total_bg_count_batch = (bg_proportions.sum(dim=1) * num_bg_cells).round().int() # [Batch]
    
    # 遍历 Batch 中的每个样本
    for i in range(batch_size):
        
        # 当前样本的背景平均向量 (Linear Scale)
        bg_vector_linear = torch.zeros(n_genes, device=device)
        
        # 遍历所有细胞类型
        for ct_idx in range(num_cell_types):
            
            if not non_target_mask[i, ct_idx].item(): 
                continue # 跳过自身
            
            # 该类型需要采样的数量
            n_sample = int(bg_proportions[i, ct_idx].item() * num_bg_cells)
            
            if n_sample > 0:
                avail_idx = cell_type_indices[ct_idx]
                
                if len(avail_idx) > 0:
                    
                    # 索引采样 (使用 PyTorch 替代 np.random.choice, 保持在 GPU 或高效传输)
                    
                    # 优化：如果 avail_idx 是 NumPy 数组/Python list
                    # 我们在 CPU 上选择索引，然后将索引张量发送到 GPU，确保数据拉取和计算在 GPU 上
                    chosen_indices_cpu = np.random.choice(avail_idx, n_sample, replace=True)
                    chosen_tensor = torch.as_tensor(chosen_indices_cpu, device=device)
                    
                    # 注意：如果 X_tensor 不在 device 上，index_select 会自动进行一次转移。
                    # 理想情况下，X_tensor 应该在 GPU 上。
                    if X_tensor_device != device:
                        chunk_log = X_tensor.index_select(0, chosen_tensor.cpu()).to(device)
                    else:
                        chunk_log = X_tensor.index_select(0, chosen_tensor)
                        
                    # 转换到线性，求和 (完全在 GPU 上执行)
                    chunk_linear = torch.expm1(chunk_log)
                    bg_vector_linear += chunk_linear.sum(dim=0)
        
        # 计算平均值
        current_total_count = total_bg_count_batch[i].item()
        if current_total_count > 0:
            bg_mean_linear = bg_vector_linear / current_total_count
        else:
            bg_mean_linear = torch.zeros(n_genes, device=device)
            
        # 3. 信号注入 (Signal Injection) (GPU)
        signal_frac = random.uniform(0.05, 0.50)
        
        # 混合：Target (Linear) + Background (Linear)
        pseudo_bulks_linear[i] = signal_frac * target_samples_linear[i] + (1 - signal_frac) * bg_mean_linear

    # --- 4. 噪声注入和后处理 (完全批量化 GPU 操作) ---
    
    # 1. 乘性噪声 (Multiplicative Noise)
    noise_mult = torch.randn_like(pseudo_bulks_linear) * 0.2 + 1.0
    pseudo_bulks_linear = pseudo_bulks_linear * noise_mult
    
    # 2. 确保非负
    pseudo_bulks = torch.clamp(pseudo_bulks_linear, min=0.0)
    
    # 3. 最终返回 Log1p 结果
    return torch.log1p(pseudo_bulks)

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

def get_sample_indices_from_bulk(real_bulk_path, target_samples):
    """
    读取 real_bulk.csv 并返回目标样本的索引，同时返回实际找到的样本名称列表。
    """
    # 读取 Bulk 数据
    try:
        # 假设是 TSV 格式，如果 index 在第一列
        real_bulk_df = pd.read_csv(real_bulk_path, sep='\t', index_col=0).T
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

def plot_marker_expression_comparison(adata_real, adata_gen, 
                                      group_col_real='labels', 
                                      group_col_gen='Cell_type', # 假设生成数据的标签列名
                                      top_n=5, 
                                      output_dir='./'):
    """
    对比真实数据和生成数据中 Marker 基因的表达水平。
    绘制两张配对的热图。
    """
    print("\n--- Comparing Marker Gene Expression (Real vs Gen) ---")
    
    # 1. 寻找真实数据的 Top Marker
    # 我们以真实数据为准，看看生成数据是否复现了这些 Marker
    # 假设 adata_real 已经经过 log1p 处理
    # 为了避免修改原数据，创建副本
    real_temp = adata_real.copy()
    
    # 计算 Rank Genes (如果之前没算过)
    # 使用 t-test 或 wilcoxon
    if 'rank_genes_groups' not in real_temp.uns:
        sc.tl.rank_genes_groups(real_temp, groupby=group_col_real, method='t-test_overestim_var')
    
    # 提取 Marker 列表
    # 格式: {cell_type: [gene1, gene2...]}
    groups = real_temp.obs[group_col_real].unique().tolist()
    # 排序以保证绘图顺序一致
    groups = sorted([str(g) for g in groups])
    
    marker_dict = {}
    var_names = set()
    
    for g in groups:
        # 获取该组的 top_n 基因
        try:
            genes = sc.get.rank_genes_groups_df(real_temp, group=g).head(top_n)['names'].tolist()
            marker_dict[g] = genes
            var_names.update(genes)
        except KeyError:
            print(f"Warning: Group {g} not found in rank_genes_groups.")
            
    gene_list = [] # 用于绘图的基因顺序（按细胞类型排列）
    for g in groups:
        if g in marker_dict:
            gene_list.extend(marker_dict[g])
            
    # 去重但保持顺序 (稍微麻烦点，简单起见允许少量重复显示，或者用 pd.unique)
    # 这里我们允许重复，因为不同细胞可能有相同 marker，在热图中重复显示有助于对比
    
    # 2. 准备绘图数据矩阵
    # 我们需要计算每种细胞类型下，这些基因的【平均表达量】
    
    # 2.1 真实数据矩阵
    # [Cell_Types, Genes]
    real_mean_df = pd.DataFrame(index=groups, columns=gene_list)
    
    for g in groups:
        # 找到该类型的细胞
        cells = real_temp[real_temp.obs[group_col_real] == g, gene_list]
        # 计算均值
        mean_expr = np.mean(cells.X, axis=0)
        # 如果是稀疏矩阵
        if hasattr(mean_expr, 'A1'): mean_expr = mean_expr.A1
        elif hasattr(mean_expr, 'toarray'): mean_expr = mean_expr.toarray().flatten()
        
        real_mean_df.loc[g] = mean_expr

    # 2.2 生成数据矩阵
    # 确保基因存在于生成数据中
    valid_genes = [gene for gene in gene_list if gene in adata_gen.var_names]
    if len(valid_genes) < len(gene_list):
        print(f"Warning: {len(gene_list) - len(valid_genes)} markers missing in generated data.")
    
    gen_mean_df = pd.DataFrame(index=groups, columns=valid_genes)
    
    # 确保生成数据的标签列存在
    if group_col_gen not in adata_gen.obs:
        print(f"Error: Column {group_col_gen} not found in adata_gen.obs")
        return

    for g in groups:
        # 找到该类型的细胞
        # 注意：生成数据可能不包含所有真实数据的类型（如果该类型被过滤了）
        if g not in adata_gen.obs[group_col_gen].values:
            # print(f"Note: Cell type {g} not found in Generated Data.")
            gen_mean_df.loc[g] = 0 # 填 0
            continue
            
        cells = adata_gen[adata_gen.obs[group_col_gen] == g, valid_genes]
        mean_expr = np.mean(cells.X, axis=0)
        gen_mean_df.loc[g] = mean_expr

    # 3. 归一化 (Z-score Scaling)
    # 按基因（列）进行归一化，这样可以消除绝对表达量的差异，只看特异性模式
    # scale 函数默认对列进行操作 (axis=0)
    
    # 为了绘图美观，我们将数据转为 float
    real_plot_data = real_mean_df.astype(float)
    gen_plot_data = gen_mean_df.astype(float)
    
    # 行归一化（对比不同细胞间该基因的高低）
    # 或者列归一化（对比该细胞内不同基因的高低）
    # 通常 DotPlot/Heatmap 习惯对 Gene (列) 做 Z-score
    
    # 我们手动实现一下，为了处理全0行/列
    def safe_zscore(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or df.shape[1] == 0 or df.shape[0] == 0:
            # 返回同形状全 0 DataFrame
            return pd.DataFrame(0, index=df.index, columns=df.columns)
        # 正常 z-score
        z = scale(df, axis=0)          # axis=0 按列标准化
        return pd.DataFrame(z, index=df.index, columns=df.columns).fillna(0)

    real_norm = safe_zscore(real_plot_data)
    gen_norm = safe_zscore(gen_plot_data)
    real_norm.to_csv(os.path.join(output_dir, 'real_marker_expression_zscore.csv'))
    gen_norm.to_csv(os.path.join(output_dir, 'gen_marker_expression_zscore.csv'))
    
    zero_rows = gen_norm.index[(gen_norm == 0).all(axis=1)]
    print(f"Removing {len(zero_rows)} rows with all zeros.")
    # 2. 同时从两个 DataFrame 删除这些行
    real_norm = real_norm.drop(zero_rows)
    gen_norm  = gen_norm.drop(zero_rows)
    real_norm = real_norm[~real_norm.index.isna()]
    gen_norm = gen_norm[~gen_norm.index.isna()]
    
    # 截断极值，防止热图颜色被个别点拉偏
    vmax = 2.5
    vmin = -1.5
    
    # 4. 绘制双热图
    fig, axes = plt.subplots(
    1, 2,
    figsize=(13, 7),
    gridspec_kw={"width_ratios": [1, 1]}  # 强制左右等宽
    )
    # 图 1: Real
    sns.heatmap(real_norm, ax=axes[0], cmap='RdBu_r', vmin=vmin, vmax=vmax, 
                xticklabels=False, cbar=False)
    axes[0].set_title("Real Data (Top Markers)", fontsize=14)
    axes[0].set_xlabel("Marker Genes (Grouped by Type)", fontsize=12)
    axes[0].set_ylabel("Cell Types", fontsize=13)
    axes[0].tick_params(axis="y", labelsize=11)
    
    # 图 2: Generated
    sns.heatmap(gen_norm, ax=axes[1], cmap='RdBu_r', vmin=vmin, vmax=vmax, 
                xticklabels=False, cbar=True)
    axes[1].set_title("Generated Data (Same Markers)", fontsize=14)
    axes[1].set_xlabel("Marker Genes", fontsize=12)
    axes[1].tick_params(axis="y",left=False,labelleft=False)
    
    # plt.tight_layout()
    plt.subplots_adjust(
    wspace=0.05,   # 横向间距，0.03–0.08 期刊常用
    left=0.18,
    right=0.92,
    top=0.90,
    bottom=0.10
)
    save_path = os.path.join(output_dir, 'comparison_marker_heatmap.png')
    plt.savefig(save_path, dpi=600)
    plt.close()
    print(f"Marker comparison heatmap saved to {save_path}")
    
    # 5. 可选：计算两者矩阵的相关性（量化评估）
    # 将两个矩阵拉平计算 Pearson
    # 只比较两者都有的基因
    common_cols = real_norm.columns.intersection(gen_norm.columns)
    if len(common_cols) > 0:
        flat_real = real_norm[common_cols].values.flatten()
        flat_gen = gen_norm[common_cols].values.flatten()
        from scipy.stats import pearsonr
        corr, _ = pearsonr(flat_real, flat_gen)
        print(f"Overall Pattern Consistency (Correlation): {corr:.4f}")
        
        # 保存相关性到文件
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
    对比 Real, Ours, Ablation 三种数据的基因共表达网络。
    
    参数:
        adata_real: 真实单细胞数据 (AnnData)
        adata_gen_ours: Attention模型生成的数据
        adata_gen_ablation: MLP模型(无Attention)生成的数据
        output_dir: 图片保存路径
        n_top_genes: 如果没有指定specific_genes，选取多少个HVG进行计算
        specific_genes: (可选) 基因列表，如果提供则只计算这些基因的网络
    """
    print("--- Starting Gene Co-expression Network Comparison ---")
    
    # 1. 确定要分析的基因集
    # 确保基因在三个数据集中都存在
    common_vars = list(set(adata_real.var_names) & set(adata_gen_ours.var_names) & set(adata_gen_ablation.var_names))
    
    if specific_genes is not None:
        target_genes = [g for g in specific_genes if g in common_vars]
        print(f"Using {len(target_genes)} specific genes provided by user.")
    else:
        # 如果没给特定基因，就用 Real 数据的 HVG
        print(f"Selecting top {n_top_genes} HVGs from Real data...")
        temp_adata = adata_real[:, common_vars].copy()
        sc.pp.highly_variable_genes(temp_adata, n_top_genes=n_top_genes)
        target_genes = temp_adata.var[temp_adata.var['highly_variable']].index.tolist()
        print(f"Selected {len(target_genes)} HVGs.")

    # 2. 提取表达矩阵 (Cells x Genes)
    # 注意：计算共表达通常需要 Log1p 后的数据
    def get_matrix(adata, genes):
        sub = adata[:, genes]
        if isinstance(sub.X, np.ndarray):
            return sub.X
        else:
            return sub.X.toarray()

    X_real = get_matrix(adata_real, target_genes)
    X_ours = get_matrix(adata_gen_ours, target_genes)
    X_ablation = get_matrix(adata_gen_ablation, target_genes)

    # 3. 计算基因-基因相关性矩阵 (Genes x Genes)
    # rowvar=False 表示每一列是一个变量(基因)
    # 返回矩阵形状: [n_genes, n_genes]
    print("Computing correlation matrices...")
    corr_real = np.corrcoef(X_real, rowvar=False)
    corr_ours = np.corrcoef(X_ours, rowvar=False)
    corr_ablation = np.corrcoef(X_ablation, rowvar=False)
    
    # 处理可能的 NaN (如果某个基因表达全为0)
    corr_real = np.nan_to_num(corr_real)
    corr_ours = np.nan_to_num(corr_ours)
    corr_ablation = np.nan_to_num(corr_ablation)

    # 4. 定量评估：矩阵相似度
    # 方法 A: 矩阵展平后的 Pearson 相关性 (Mantel Test 近似)
    # 衡量“真实数据里 A和B 正相关，生成数据里 A和B 是否也正相关？”
    score_ours_pcc = pearsonr(corr_real.flatten(), corr_ours.flatten())[0]
    score_ablation_pcc = pearsonr(corr_real.flatten(), corr_ablation.flatten())[0]
    
    # 方法 B: 矩阵距离 (Frobenius Norm)
    # 衡量绝对误差，越小越好
    dist_ours = np.linalg.norm(corr_real - corr_ours)
    dist_ablation = np.linalg.norm(corr_real - corr_ablation)

    print(f"\n[Quantitative Results]")
    print(f"Matrix Similarity (PCC) [Higher is Better]:")
    print(f"  Ours (Attention): {score_ours_pcc:.4f}")
    print(f"  Ablation (MLP)  : {score_ablation_pcc:.4f}")
    print(f"Matrix Distance (L2 Norm) [Lower is Better]:")
    print(f"  Ours (Attention): {dist_ours:.4f}")
    print(f"  Ablation (MLP)  : {dist_ablation:.4f}")

    # 5. 可视化绘制
    # 为了图好看，我们对基因聚类一下，让红色的块聚在一起
    # 使用 seaborn 的 clustermap 获取聚类后的索引
    print("Plotting heatmaps...")
    g = sns.clustermap(pd.DataFrame(corr_real), cmap='vlag', center=0)
    reordered_idx = g.dendrogram_row.reordered_ind
    plt.close() # 不显示这个中间图

    # 按 Real 数据的聚类顺序重新排列三个矩阵，以便直观对比
    def reorder_mat(mat, idx):
        return mat[idx, :][:, idx]

    corr_real_sorted = reorder_mat(corr_real, reordered_idx)
    corr_ours_sorted = reorder_mat(corr_ours, reordered_idx)
    corr_ablation_sorted = reorder_mat(corr_ablation, reordered_idx)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 通用绘图参数
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
                           adata_gen: ad.AnnData, # <--- 接收 AnnData
                           breed_2_list: list, 
                           common_genes: list, 
                           color_map: dict,
                           output_dir: str,
                           real_bulk_tensor=None,
                           top_percent_remove=0.02) -> None:
    """
    评估生成数据。
    X_tensor, labels: 真实单细胞数据 (Tensor)
    adata_gen: 生成的单细胞数据 (AnnData)
    """
    set_seed(seed)
    print("开始评估生成结果...")
    
    # # 1. 计算 Mask (基于 Bulk)
    # if real_bulk_tensor is not None:
    #     # real_bulk_tensor: [N_samples, N_genes]
    #     # 计算所有样本的平均 Bulk 表达量
    #     bulk_mean = real_bulk_tensor.mean(dim=0).cpu().numpy() # [12220]
        
    #     # 找出 Top K
    #     n_genes = bulk_mean.shape[0]
    #     k = int(n_genes * top_percent_remove)
        
    #     # argsort 从小到大，取最后 k 个即为最大
    #     top_indices = np.argsort(bulk_mean)[-k:]
        
    #     keep_mask = np.ones(n_genes, dtype=bool)
    #     keep_mask[top_indices] = False
        
    #     print(f"Evaluation: Masking Top {k} genes based on BULK expression.")
    # else:
    #     # 如果没传 Bulk，退化为全保留或基于 SC
    #     keep_mask = np.ones(X_tensor.shape[1], dtype=bool)
    
    
    # 1. 准备真实数据 Numpy
    x_real = X_tensor.numpy()
    labels_real = labels.numpy()
    print(f"真实数据形状: {x_real.shape}, 生成数据形状: {adata_gen.X.shape}")
    
    # 2. 准备生成数据 Numpy
    x_gen = adata_gen.X
    
    # mask = ~np.isin(common_genes, ['ACTA2', 'TAGLN', 'COL1A1', 'MALAT1'])
    # x_gen = x_gen[:, mask]
    # x_real = x_real[:, mask]
    
    # === 【修复 KeyError 的核心代码】 ===
    # 将 Cell_type 字符串映射回整数索引，以便与 labels_real 对齐
    if 'Label_Idx' in adata_gen.obs:
        labels_gen = adata_gen.obs['Label_Idx'].values
    elif 'Cell_type' in adata_gen.obs:
        # 创建映射字典: { 'T cell': 0, 'B cell': 1, ... }
        name_to_idx = {name: i for i, name in enumerate(breed_2_list)}
        # 映射
        try:
            labels_gen = adata_gen.obs['Cell_type'].map(name_to_idx).astype(int).values
        except ValueError as e:
            print("Error mapping cell types to indices. Please check if breed_2_list matches adata_gen.obs['Cell_type']")
            raise e
    else:
        raise KeyError("adata_gen.obs must contain 'Cell_type' or 'Label_Idx'")
    # ==================================
    
    # 确保基因维度对齐
    if x_gen.shape[1] != x_real.shape[1]:
        raise ValueError(f"Gene dimension mismatch! Real: {x_real.shape[1]}, Gen: {x_gen.shape[1]}")

    # 3. 数据诊断 (Sanity Check)
    diagnose_label_idx = 0
    mask_gen = labels_gen == diagnose_label_idx
    if mask_gen.sum() > 0:
        gen_subset = x_gen[mask_gen]
        print(f"\n--- Diagnosis for Cell Type ID {diagnose_label_idx} ({breed_2_list[diagnose_label_idx]}) ---")
        print(f"Mean: {gen_subset.mean():.4f}, Max: {gen_subset.max():.4f}")
        # 检查方差，防止 Mode Collapse
        avg_var = np.mean(np.var(gen_subset, axis=0))
        print(f"Average Variance per gene: {avg_var:.6f}")
        if avg_var < 1e-6:
            print("WARNING: Variance is extremely low! Mode Collapse detected.")
    
    # --- 4. 计算 Pearson Correlation (按 Cell Type) ---
    # print("\nCalculating Pearson correlation per cell type...")
    pearson_list = []
    
    for label_idx, label_name in enumerate(breed_2_list):
        mask_real = labels_real == label_idx
        mask_gen = labels_gen == label_idx
        
        # 如果某类细胞太少，跳过
        if mask_real.sum() < 1 or mask_gen.sum() < 1:
            print(f"Skipping {label_name}: insufficient cells (Real={mask_real.sum()}, Gen={mask_gen.sum()})")
            pearson_list.append(np.nan)
            continue
            
        # 计算 Centroid (平均表达谱)
        mean_real = x_real[mask_real].mean(axis=0)
        mean_gen = x_gen[mask_gen].mean(axis=0)
        # mean_real_filtered = mean_real[keep_mask]
        # mean_gen_filtered = mean_gen[keep_mask]
        
        # 处理全0情况防止报错
        if np.std(mean_real) < 1e-9 or np.std(mean_gen) < 1e-9:
            corr = 0.0
        else:
            # corr, _ = pearsonr(mean_real, mean_gen)
            corr, _ = pearsonr(mean_real, mean_gen)
            
        pearson_list.append(corr)
        print(f"Pearson for {label_name}: {corr:.4f}, Real n={mask_real.sum()}, Gen n={mask_gen.sum()}, for {mean_real.shape[0]} genes")
    print(f"Mean Pearson Correlation: {np.nanmean(pearson_list):.4f}")
        
    pd.DataFrame({'Cell_type': breed_2_list, 'Pearson': pearson_list}).to_csv(os.path.join(output_dir, '1_eval_pearson_correlation.csv'))
    
    # --- 5. 计算 MSE (按 Cell Type) ---
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
    
    # --- 6. UMAP 可视化 (Real vs Generated) ---
    print("\nPreparing UMAP visualization...")
    adata_X = adata_gen.copy()
    # 由于数据已经是归一化 Log 空间，直接进行 PCA
    sc.tl.pca(adata_X, svd_solver='arpack', n_comps=50) 
    sc.pp.neighbors(adata_X, n_neighbors=15, n_pcs=50)
    sc.tl.umap(adata_X, min_dist=0.5, spread=1.0)
    
    # 绘制按 Sample和 Cell Type 着色的 UMAP
    fig_X, axes_X = plt.subplots(1, 2, figsize=(18, 8))
    
    # 图 1: 按 VAE 预测的 Cell_type 标签着色 (验证生成数据是否保留了预测标签的结构)
    sc.pl.umap(adata_X, color='Cell_type', title='Generated X UMAP: VAE Predicted Type', 
               ax=axes_X[0], show=False)
               
    # 图 2: 按 Sample着色 (验证是否捕捉到个体差异)
    sc.pl.umap(adata_X, color='Sample', title='Generated X UMAP: Sample', 
               ax=axes_X[1], show=False)
               
    plt.tight_layout()
    umap_X_path = os.path.join(output_dir, '_generated_X_umap_comparison.png')
    plt.savefig(umap_X_path, dpi=300)
    plt.close(fig_X)
    print(f"Generated X UMAP saved to {umap_X_path}")
    
    
    # # 降采样以防止内存溢出 (Limit to 10k cells each)
    # max_cells_plot = 10000
    
    # # 6.1 处理真实数据
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
    # adata_real_plot.var_names = index_2_gene # 必须对齐基因名
    
    # # 6.2 处理生成数据
    # adata_gen_plot = adata_gen.copy()
    # if adata_gen_plot.n_obs > max_cells_plot:
    #     sc.pp.subsample(adata_gen_plot, n_obs=max_cells_plot)
    
    # adata_gen_plot.obs['Source'] = 'Generated'
    # # 仅保留需要的列以便合并
    # adata_gen_plot.obs = adata_gen_plot.obs[['Cell_type', 'Source']].rename(columns={'Cell_type': 'Cell_Type'})
    
    # # 6.3 合并
    # # 注意: ad.concat 默认取交集，如果 var_names 不一致会导致空数据，前面已确保一致
    # adata_combined = ad.concat([adata_real_plot, adata_gen_plot], join='outer')
    # adata_combined.obs['Cell_Type'] = adata_combined.obs['Cell_Type'].astype('category')
    # adata_combined.obs['Source'] = adata_combined.obs['Source'].astype('category')
    
    # # 6.4 计算 UMAP
    # # print("Running PCA and UMAP...")
    # sc.pp.pca(adata_combined)
    # sc.pp.neighbors(adata_combined)
    # sc.tl.umap(adata_combined)
    
    # # 6.5 设置颜色
    # # Source 颜色
    # adata_combined.uns['Source_colors'] = ['#f4df4e', '#949398'] # Gen: Yellow, Real: Grey
    
    # # Cell Type 颜色 (使用 color_map)
    # if color_map:
    #     # Scanpy 按照 category 的顺序分配颜色，必须手动排序
    #     categories = adata_combined.obs['Cell_Type'].cat.categories
    #     palette = [color_map.get(cat, '#808080') for cat in categories]
    #     adata_combined.uns['Cell_Type_colors'] = palette
    
    # # 6.6 绘图与保存
    # sc.pl.umap(adata_combined, color=['Source', 'Cell_Type'], 
    #            wspace=0.4, title=['Source (Real vs Gen)', 'Cell Type'],
    #            save='_real_vs_generated.png', show=False)
    
    # # --- 7. 计算 Silhouette Score ---
    # # 评估 Real 和 Generated 混得好不好 (分数越低越好，说明分不开，即生成得像)
    # # 注意：通常这里的 Score 低是好事(接近0)，说明Source之间没有明显边界
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
    
    # 测试细胞类型
    target_name = 'CD4_EM' # 选一个你数据里有的
    if target_name not in mapping_dict:
        print(f"Error: {target_name} not in mapping dict")
        return

    target_idx = mapping_dict[target_name]
    
    # 1. 构造输入：完全随机的 Latent Z (不依赖 Bulk)
    if hasattr(net, 'mid_hidden_size'):
        latent_dim = net.mid_hidden_size * 2  # 256 * 2 = 512
    else:
        latent_dim = 512 
    
    print(f"Using Latent Dim: {latent_dim}")
    
    count = 50
    
    # 采样 z ~ N(0, 1)
    z_random = torch.randn(count, latent_dim).to(device)
    
    # 2. 构造标签
    num_classes = len(mapping_dict)
    label_tensor = F.one_hot(torch.tensor([target_idx] * count), num_classes=num_classes).float().to(device)
    
    # 3. 解码
    try:
        with torch.no_grad():
            generated = net.decode(z_random, label_tensor).cpu().numpy()
    except RuntimeError as e:
        print(f"Decoder failed with error: {e}")
        print(f"Debug info: z shape {z_random.shape}, label shape {label_tensor.shape}")
        return
        
    # 3. 评估
    real_indices = np.where(labels.numpy() == target_idx)[0]
    real_data = X_tensor[real_indices].numpy()
    
    mean_gen = generated.mean(axis=0)
    mean_real = real_data.mean(axis=0)
    
    corr, _ = pearsonr(mean_gen, mean_real)
    print(f"Target: {target_name}")
    print(f"Generated with Random Z Correlation: {corr:.4f}")
    
    # 检查生成数据的方差，看是否有活性
    print(f"Generated Variance: {np.mean(np.var(generated, axis=0)):.6f}")
    
    if corr < 0.5:
        print(">>> FAIL: Decoder cannot generate correct profile even with clean Z.")
    else:
        print(">>> PASS: Decoder works fine. The problem is in Bulk -> Encoder path.")

def debug_prior_quality(net, cell_type_mu_logvar, mapping_dict, adata_real, common_genes, device):
    print("\n=== DEBUG: Checking Prior Quality (Fixed) ===")
    net.eval().to(device)
    
    # 1. 严格对齐基因 (这是最关键的一步)
    # 创建一个新的 AnnData，确保列顺序完全一致
    # 假设 adata_real 已经包含了这些基因
    valid_genes = [g for g in common_genes if g in adata_real.var_names]
    if len(valid_genes) != len(common_genes):
        print(f"Warning: {len(common_genes) - len(valid_genes)} genes missing in adata_real.")
    
    # 这一步会自动按 valid_genes 的列表顺序重排 adata_real 的列
    adata_aligned = adata_real[:, valid_genes].copy()
    
    # 2. 检查数据范围
    print(f"Real Data Range: {adata_aligned.X.min()} - {adata_aligned.X.max()}")
    # 如果最大值很大 (>20)，说明可能是 Raw Count，需要 Log1p
    if adata_aligned.X.max() > 20:
        print("Applying Log1p to real data for comparison...")
        sc.pp.log1p(adata_aligned)

    # 取出真实数据中的某一类细胞
    target_type = list(mapping_dict.keys())[0]
    type_id = mapping_dict[target_type]
    
    # 3. 获取 Prior 并解码
    prior_mu = cell_type_mu_logvar[type_id][0].to(device)
    label_tensor = F.one_hot(torch.tensor([type_id]), len(mapping_dict)).float().to(device)
    
    with torch.no_grad():
        decoded_expr = net.decode(prior_mu, label_tensor).cpu().numpy().flatten()
    
    # 4. 获取真实均值
    real_cells = adata_aligned[adata_aligned.obs['labels'] == target_type].X
    if issparse(real_cells): real_cells = real_cells.toarray()
    real_mean_expr = real_cells.mean(axis=0).flatten()
    
    # 5. 计算相关性
    # 确保长度一致 (以防万一 common_genes 有缺失)
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
    评估 VAE 在单细胞数据上的“压缩-解压”还原能力。
    核心指标：输入单细胞 vs 重建单细胞 的 Pearson 相关性。
    """
    print("\n" + "="*40)
    print(" >>> 开始检查 VAE 模型重建能力 (Sanity Check)")
    print("="*40)
    
    vae_model.eval()
    vae_model.to(device)
    
    # 获取 ID 到 名称 的映射
    id2type = {v: k for k, v in mapping_dict.items()}
    num_cell_types = len(mapping_dict)
    
    # 随机采样一部分数据进行测试 (例如 2000 个细胞)
    # 或者是使用全部数据
    n_total = X_tensor.shape[0]
    indices = torch.randperm(n_total)[:2000] # 只测 2000 个够了
    
    test_X = X_tensor[indices].to(device)
    test_y = labels[indices].to(device)
    
    recon_X_list = []
    
    with torch.no_grad():
        # 分 Batch 处理防止显存溢出
        for i in range(0, len(test_X), batch_size):
            batch_x = test_X[i : i+batch_size]
            batch_y = test_y[i : i+batch_size]
            
            # One-hot label
            batch_y_hot = F.one_hot(batch_y, num_cell_types).float()
            
            # 标准 VAE 前向传播
            # 注意：这里我们想看的是模型“最好”能做到什么样
            # 所以通常不加随机噪声 (z = mu) 或者加很少的噪声
            mu, logvar = vae_model.encode(batch_x, batch_y_hot)
            # z = vae.reparameterize(mu, logvar)
            # recon = vae.decode(z, batch_y_hot)
            # 直接用均值重构 (去噪能力上限)
            recon_batch = vae_model.decode(mu, batch_y_hot)
            recon_X_list.append(recon_batch.cpu())
            
    recon_X = torch.cat(recon_X_list, dim=0)
    
    # === 评估指标 ===
    
    # 1. 整体相关性 (Flatten)
    # 这衡量了整体数值分布的一致性
    input_flat = test_X.cpu().numpy().flatten()
    recon_flat = recon_X.numpy().flatten()
    corr_global, _ = pearsonr(input_flat, recon_flat)
    print(f"\n[总体] 全局 Pearson 相关性: {corr_global:.4f}")
    
    # 2. 逐细胞相关性 (Per Cell Correlation)
    # 这衡量了每个细胞是否还像它自己
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
    
    # 3. 逐基因相关性 (Per Gene Correlation)
    # 这衡量了模型是否抓住了基因的特异性
    gene_corrs = []
    input_np = test_X.cpu().numpy()
    recon_np = recon_X.numpy()
    
    for j in range(input_np.shape[1]): # 遍历基因
        g_in = input_np[:, j]
        g_out = recon_np[:, j]
        if np.std(g_in) < 1e-9 or np.std(g_out) < 1e-9:
             # 如果基因在所有细胞都为0，相关性无意义，设为NaN或0
            gene_corrs.append(np.nan)
        else:
            c, _ = pearsonr(g_in, g_out)
            gene_corrs.append(c)
            
    avg_gene_corr = np.nanmean(gene_corrs)
    print(f"[基因级] 平均基因 Pearson 相关性: {avg_gene_corr:.4f}")
    
    # 4. 按细胞类型统计
    print("\n--- 按细胞类型拆分 ---")
    type_stats = {}
    test_y_np = test_y.cpu().numpy()
    
    for type_id in range(num_cell_types):
        type_name = id2type.get(type_id, str(type_id))
        mask = (test_y_np == type_id)
        if mask.sum() == 0: continue
        
        # 该类型下的平均相关性
        sub_corrs = np.array(cell_corrs)[mask]
        mean_corr = np.mean(sub_corrs)
        type_stats[type_name] = mean_corr
        print(f"  Type {type_name:<15} (n={mask.sum():<4}): {mean_corr:.4f}")
        
    # # === 绘图 (可选) ===
    # # 随机选一个细胞画散点图
    # idx = np.random.randint(len(test_X))
    # plt.figure(figsize=(6, 6))
    # plt.scatter(test_X[idx].cpu().numpy(), recon_X[idx].numpy(), alpha=0.5, s=10)
    # plt.plot([0, test_X.max()], [0, test_X.max()], 'r--')
    # plt.xlabel("Original Expression (Log1p)")
    # plt.ylabel("Reconstructed Expression (Log1p)")
    # plt.title(f"Single Cell Reconstruction (Corr={cell_corrs[idx]:.2f})")
    # plt.savefig(os.path.join(output_dir, "vae_sanity_check_scatter.png"))
    # plt.close()
    
    # 保存结果
    pd.DataFrame({'Gene': common_genes, 'Pearson': gene_corrs}).to_csv(
        os.path.join(output_dir, "vae_gene_reconstruction_corr.csv"), index=False
    )
    
    print("="*40 + "\n")
    return avg_cell_corr
################################
### 端到端工作流程 ###
################################
def main(sc_data, real_bulk_path, real_sc_path, output_dir, device=device, epochs=100, lr=0.001):
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
    # # 1. 配置参数
    # dataset, X_tensor, labels, n_cell_types, mapping_dict, all_genes, real_bulk_log, single_cell_matrix, cell_number_target_num, signature_array, color_map, bulk_sample_names, final_signatures, sig_indices, core_genes, core_indices_tensor = load_sc_data(
    #     sc_data, real_bulk_path, celltype_label='level2_cell_type'
    # )
    # print("数据加载完成，细胞数量:", len(dataset), "基因数量:", X_tensor.shape[1], "细胞类型数量:", len(mapping_dict), "bulk样本:", real_bulk_log.shape)
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

    # # 保存到文件
    # with open(os.path.join(output_dir, 'sc_metadata.pkl'), 'wb') as f:
    #     pickle.dump(sc_metadata, f)

    # # 第一阶段：训练Attention GMVAE模型
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
    
    target_samples=['LDK1', 'LDK2', 'LDK3', 'LDK4', 'LDK5', 'LDK6']
    # 2. 获取索引和有效的样本名称 (防止某个样本名在文件中写错或不存在)
    sample_indices, valid_sample_names = get_sample_indices_from_bulk(
        real_bulk_path, 
        target_samples=target_samples
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
    adata_generated, Final_Z, obs_df = optimize_z_and_generate_final(
        vae=scvae,
        real_bulk_tensor=input_bulk_log,
        sample_names=valid_sample_names,
        mapping_dict=mapping_dict,
        priors=cell_type_mu_logvar_best, # 训练好的先验
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
    #  # [新增] 计算训练集统计量
    # print("Computing SC training statistics...")
    # # 确保在 CPU 上计算以免显存不足，或者分批
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
        X_tensor=X_tensor, # 真实单细胞 tensor
        labels=labels,     # 真实标签 tensor
        adata_gen = adata_generated, # 传入生成的 AnnData
        breed_2_list=breed_2_list,
        common_genes=common_genes,
        color_map=color_map,
        output_dir=output_dir,
        real_bulk_tensor=input_bulk_log,
    )
    
    # ###debug调试用伪bulk####
    # print("调试：使用单细胞数据进行生成和评估...")
    # sanity_score = check_vae_reconstruction(
    #     vae_model=scvae,
    #     X_tensor=X_tensor,  # 你的训练集或测试集 Tensor
    #     labels=labels,      # 对应的标签
    #     mapping_dict=mapping_dict,
    #     common_genes=common_genes,
    #     device=device,
    #     output_dir=output_dir
    # )
    # print(f"VAE Sanity Check Score: {sanity_score:.4f}")
    # debug_prior_quality(scvae, cell_type_mu_logvar_best, mapping_dict, adata_real, common_genes, device)
################################
### 示例用法 ###
################################

if __name__ == "__main__":
    main(
        sc_data=os.path.join(data_dir, 'GSE14115_sc_raw_counts_train.h5ad'),
         real_bulk_path=os.path.join(data_dir, 'GSE14115_bulk_counts_LD_testset.txt'),
         real_sc_path=os.path.join(data_dir, 'GSE14115_sc_raw_counts_LD.h5ad'),
         output_dir=output_dir,
         device=device)
