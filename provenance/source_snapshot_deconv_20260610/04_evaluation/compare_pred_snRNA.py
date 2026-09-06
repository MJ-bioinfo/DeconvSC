#!/usr/bin/env python
# coding: utf-8

# Converted from compare_pred_snRNA.ipynb
# Markdown cells are preserved as comments; code cells are separated by # %%.
# Notebook-only magic/shell commands are commented out for valid Python syntax.

# %% [markdown] cell 1
# 对比生成数据和真实数据

# %% cell 2
import os
from pathlib import Path
import pandas as pd
import scanpy as sc
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import seaborn as sns

# %% cell 3
# Linux paths for the exact GSE141115 inputs used by the Figure 3 benchmark.
# Keep newly generated diagnostic plots separate from the manuscript figures.
project_root = Path('/disk1/maijl/deconv')
generated_h5ad = project_root / 'cVAE/GSE141115/attention_geneembed/generated_data.h5ad'
real_h5ad = project_root / 'data/GSE141115/GSE141115_sc_raw_counts_test.h5ad'
output_dir = Path(__file__).resolve().parent / 'outputs/compare_pred_snRNA'
output_dir.mkdir(parents=True, exist_ok=True)

# Retained only for the legacy exploratory notebook cells near the end of this
# converted script; those cells require additional cleanup before execution.
data_path = str(project_root)

# %% cell 4
adata_generated = sc.read_h5ad(generated_h5ad)
adata_generated

# %% cell 5
adata_real = sc.read_h5ad(real_h5ad)
adata_real

# %% cell 6
# 确保基因集一致
common_genes = list(set(adata_generated.var_names) & set(adata_real.var_names))
print(f"Analyzing {len(common_genes)} common genes.")

# 子集化
ad_gen = adata_generated[:, common_genes].copy()
ad_real = adata_real[:, common_genes].copy()

# %% [markdown] cell 7
# 1. 均值-方差关系 (Mean-Variance Trend)

# %% cell 8
print("1. Plotting Mean-Variance Relationship...")
def get_mean_var(adata):
    # 稀疏矩阵兼容
    if isinstance(adata.X, np.ndarray):
        X = adata.X
    else:
        X = adata.X.toarray()

    mean_expr = np.mean(X, axis=0)
    var_expr = np.var(X, axis=0)
    return mean_expr, var_expr

gen_mean, gen_var = get_mean_var(ad_gen)
real_mean, real_var = get_mean_var(ad_real)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
plt.rcParams["pdf.fonttype"]=42
plt.rcParams["font.size"] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']  # 优先使用 Arial
plt.rcParams['axes.unicode_minus'] = False   # 防止负号显示为方块

ax[0].scatter(real_mean, real_var, s=1, alpha=0.3, c='#7187A7')
ax[0].set_title("Mean-Variance of real scRNA-seq data")
ax[0].set_xlabel("Mean Expression")
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlim(1e-8, 1e2)   # x 轴范围
ax[0].set_ylim(1e-10, 1e4)

ax[1].scatter(gen_mean, gen_var, s=1, alpha=0.3, c='#7187A7')
ax[1].set_title("Mean-Variance of generated data")
ax[1].set_xlabel("Mean Expression")
ax[1].set_ylabel("Variance")
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_xlim(1e-8, 1e2)   # x 轴范围
ax[1].set_ylim(1e-10, 1e4)



x_ref = np.logspace(np.log10(1e-8), np.log10(1e4), 100)
y_ref = x_ref**2  #NB: 方差与均值的平方成正比
for a in ax:                         # 两个子图都画
    a.plot(x_ref, y_ref, color='red', lw=1, ls='--', alpha=0.8,
           label='var = μ^2')           # 图例可选
    a.legend(loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eval_1_mean_variance_comparison.svg"), format='svg', dpi=300)

plt.show()

# %% [markdown] cell 9
# 2. 基因-基因相关性矩阵 (Gene Correlation Matrix)

# %% cell 10
np.isinf(ad_real.X.data).any()

# %% cell 11
sc.pp.normalize_total(ad_real, target_sum=1e4)
sc.pp.log1p(ad_real)
sc.pp.highly_variable_genes(ad_real, n_top_genes=500)
hvg_genes = ad_real.var[ad_real.var['highly_variable']].index
ad_real

# %% cell 12
print("2. Computing Gene-Gene Correlation Matrix...")

# 设置颜色面板
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list(
    "custom_map",
    ['#D8B2AE', '#F7F3EF', '#7187A7']  # 👈 中间加浅色更自然
)

# 提取矩阵
X_gen_hvg = ad_gen[:, hvg_genes].X
X_real_hvg = ad_real[:, hvg_genes].X

if not isinstance(X_gen_hvg, np.ndarray): X_gen_hvg = X_gen_hvg.toarray()
if not isinstance(X_real_hvg, np.ndarray): X_real_hvg = X_real_hvg.toarray()

mask_keep = ~np.all(X_gen_hvg.T == 0, axis=1)   # 保留不全为 0 的基因
X_gen_hvg = X_gen_hvg[:, mask_keep]
X_real_hvg = X_real_hvg[:, mask_keep]
print(f"x_gen_hcg genes shape: {X_gen_hvg.shape}, x_real_hcg genes shape: {X_real_hvg.shape}")

# 计算相关性 (Transposed: gene vs gene)
corr_gen = np.corrcoef(X_gen_hvg.T)
corr_real = np.corrcoef(X_real_hvg.T)

print(np.isnan(corr_gen).any(), np.isinf(corr_gen).any())
print(np.isnan(corr_real).any(), np.isinf(corr_real).any())

# 对比矩阵相似度 (Matrix Frobenius Norm)
# 越小越好，或者计算两个矩阵 Flatten 后的相关性
matrix_corr = pearsonr(corr_gen.flatten(), corr_real.flatten())[0]
print(f"  -> Gene-Gene Correlation Matrix Similarity (PCC): {matrix_corr:.4f}")

# 绘图
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
plt.rcParams["pdf.fonttype"]=42
plt.rcParams["font.size"] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']  # 优先使用 Arial
plt.rcParams['axes.unicode_minus'] = False   # 防止负号显示为方块
sns.heatmap(corr_real, ax=ax[0], cmap=cmap, center=0, vmin=-0.5, vmax=0.5, cbar=False)
ax[0].set_title("Gene Correlation of real scRNA-seq data")
ax[0].axis('off')

sns.heatmap(corr_gen, ax=ax[1], cmap=cmap, center=0, vmin=-0.5, vmax=0.5)
# ax[1].set_title(f"Generated: Gene Correlation\nMatrix PCC: {matrix_corr:.3f}")
ax[1].set_title(f"Gene Correlation of generated data")
ax[1].axis('off')

plt.savefig(os.path.join(output_dir, "eval_2_gene_correlation_matrix.png"))
plt.show()

# %% [markdown] cell 13
# # 3. 差异表达基因 (DEG) 一致性 (Biological Utility)

# %% cell 14
print("3. Analyzing DEG Consistency...")
group_col = 'cell type'
# 找到数量最多的两个 Cluster
top_clusters = ad_real.obs[group_col].value_counts().index[:2].tolist()
if len(top_clusters) < 2:
    print("  -> Not enough clusters for DEG analysis. Skipping.")
else:
    grp1, grp2 = top_clusters[0], top_clusters[1]
    print(f"  -> Performing DEG: {grp1} vs {grp2}")

    # Real DEG
    sc.tl.rank_genes_groups(ad_real, groupby=group_col, groups=[grp1], reference=grp2, method='wilcoxon')
    deg_real = sc.get.rank_genes_groups_df(ad_real, group=grp1)
    top_genes_real = set(deg_real.head(100)['names'])

    # Generated DEG
    sc.tl.rank_genes_groups(ad_gen, groupby='Cell_type', groups=[grp1], reference=grp2, method='wilcoxon')
    deg_gen = sc.get.rank_genes_groups_df(ad_gen, group=grp1)
    top_genes_gen = set(deg_gen.head(100)['names'])

    # Overlap
    overlap = top_genes_real.intersection(top_genes_gen)
    jaccard = len(overlap) / len(top_genes_real.union(top_genes_gen))
    print(f"  -> Top 100 DEG Overlap: {len(overlap)} genes (Jaccard: {jaccard:.4f})")

# 绘制 Rank Comparison 散点图
# X轴: Real Score, Y轴: Gen Score (针对同一组基因)
common_deg_genes = deg_real['names'].tolist() # 取 Real 的所有基因排序

# 构建 score 字典
score_map_real = dict(zip(deg_real['names'], deg_real['scores']))
score_map_gen = dict(zip(deg_gen['names'], deg_gen['scores']))

scores_x = [score_map_real.get(g, 0) for g in common_deg_genes if g in score_map_gen]
scores_y = [score_map_gen.get(g, 0) for g in common_deg_genes if g in score_map_gen]

plt.figure(figsize=(6, 6))
plt.scatter(scores_x, scores_y, s=3, alpha=0.5, c='purple')

# 拟合线
m, b = np.polyfit(scores_x, scores_y, 1)
plt.plot(scores_x, m*np.array(scores_x) + b, color='black', linestyle='--', linewidth=1)

pcc = pearsonr(scores_x, scores_y)[0]
plt.title(f"DEG Scores Consistency ({grp1} vs {grp2})\nPCC = {pcc:.3f}")
plt.xlabel("Real Z-Score (Wilcoxon)")
plt.ylabel("Generated Z-Score (Wilcoxon)")
plt.grid(True, alpha=0.3)
plt.savefig(f"{output_dir}/eval_3_DEG_consistency.png")
plt.close()

# %% [markdown] cell 15
# # 4. Dropout (Sparsity) 模式分析

# %% cell 16
print("4. Analyzing Sparsity / Dropout Patterns...")

# 计算每个基因的非零比例 (Fraction of cells expressing gene)
def get_sparsity(adata):
    if isinstance(adata.X, np.ndarray):
        X = adata.X
    else:
        X = adata.X.toarray()
    # 计算 > 0 的比例
    frac_non_zero = np.count_nonzero(X, axis=0) / X.shape[0]
    return frac_non_zero

frac_real = get_sparsity(ad_real)
frac_gen = get_sparsity(ad_gen)

plt.figure(figsize=(6, 6))
plt.rcParams["pdf.fonttype"]=42
plt.rcParams["font.size"] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']  # 优先使用 Arial
plt.rcParams['axes.unicode_minus'] = False   # 防止负号显示为方块
plt.scatter(frac_real, frac_gen, s=2, alpha=0.4, c="#7187A7")
plt.plot([0, 1], [0, 1], 'r--', linewidth=1) # 对角线

plt.title("Gene Detection Rate (Sparsity Comparison)")
plt.xlabel("Fraction of cells expressing (Real data)")
plt.ylabel("Fraction of cells expressing (Generated data)")

# # 添加注释：Imputation Effect
# plt.text(0.1, 0.8, "Points above diagonal\nindicate Imputation\n(Recovered Signals)",
#             fontsize=9, color='darkred', bbox=dict(facecolor='white', alpha=0.8))

plt.grid(True, alpha=0.3)
plt.savefig(f"{output_dir}/eval_4_sparsity_comparison.png")
plt.show()

print("--- Evaluation Finished. Plots saved to output dir. ---")

# %% cell 17
import numpy as np
import matplotlib.pyplot as plt

def get_sparsity(adata):
    if isinstance(adata.X, np.ndarray):
        X = adata.X
    else:
        X = adata.X.toarray()
    # 每个基因在多少比例的细胞中 > 0
    frac_non_zero = np.count_nonzero(X, axis=0) / X.shape[0]
    return frac_non_zero

frac_real = get_sparsity(ad_real)
frac_gen = get_sparsity(ad_gen)

# 统一绘图风格
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

# 为了保证两张图可直接比较，使用相同的 bins 和相同的 y 轴范围
bins = np.linspace(0, 1, 51)  # 0~1, 50个bin

real_counts, _ = np.histogram(frac_real, bins=bins)
gen_counts, _ = np.histogram(frac_gen, bins=bins)
ymax = max(real_counts.max(), gen_counts.max()) * 1.05

# -------- Figure 1: Real --------
plt.figure(figsize=(5.5, 4.5))
plt.hist(frac_real, bins=bins, color="#7187A7", alpha=0.85, edgecolor="white")
plt.title("Gene Detection Rate (Real Data)")
plt.xlabel("Fraction of cells expressing")
plt.ylabel("Number of genes")
plt.xlim(0, 1)
plt.ylim(0, ymax)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/eval_4_sparsity_real.svg", dpi=300, bbox_inches="tight")
plt.show()

# -------- Figure 2: Generated --------
plt.figure(figsize=(5.5, 4.5))
plt.hist(frac_gen, bins=bins, color="#7187A7", alpha=0.85, edgecolor="white")
plt.title("Gene Detection Rate (Generated Data)")
plt.xlabel("Fraction of cells expressing")
plt.ylabel("Number of genes")
plt.xlim(0, 1)
plt.ylim(0, ymax)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/eval_4_sparsity_generated.svg", dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown] cell 18
# ### 1. 对比表达稀疏性

# %% cell 19
if issparse(pred_adata.X):
        pred_X = pred_adata.X.toarray()
pred_sparsity = np.mean(pred_X == 0)
print(pred_sparsity)

# %% cell 20
if issparse(real_adata.X):
        real_X = real_adata.X.toarray()
real_sparsity = np.mean(real_X == 0)
print(real_sparsity)

# %% cell 21
# 取pred_adata有的基因在real_adata中的子集
common_genes = pred_adata.var_names.intersection(real_adata.var_names)
print(f"common genes: {len(common_genes)}")
real_adata_sub = real_adata[:, common_genes].copy()
real_adata_sub

# %% cell 22
if issparse(real_adata_sub.X):
        real_marker_X = real_adata_sub.X.toarray()
real_marker_sparsity = np.mean(real_marker_X == 0)
print(real_marker_sparsity)

# %% [markdown] cell 23
# ### 2. 对比表达分布

# %% cell 24
import matplotlib.pyplot as plt

def to_dense(x):
    """把稀疏矩阵转为 dense（如果是稀疏格式）"""
    return x.A if hasattr(x, "A") else x

pred_X = to_dense(pred_adata.X)
real_X = to_dense(real_adata_sub.X)

# %% cell 25
plt.scatter(real_X.flatten(), pred_X.flatten(), s=1)
plt.xlabel("Real Expression")
plt.ylabel("Predicted Expression")
plt.title("Real vs Pred")

# %% cell 26
import matplotlib.pyplot as plt
# -----------------------
# 1. 全体表达值分布 (log1p)
# -----------------------
plt.figure(figsize=(6,4))
plt.hist(np.log1p(pred_X.flatten()), bins=100, alpha=0.5, label="Predicted", density=True)
plt.hist(np.log1p(real_X.flatten()), bins=100, alpha=0.5, label="Real", density=True)
plt.xlabel("log1p(Expression)")
plt.ylabel("Density")
plt.title("Global Expression Distribution")
plt.legend()
plt.show()

# -----------------------
# 2. 每个细胞的总UMI (library size)
# -----------------------
plt.figure(figsize=(6,4))
plt.hist(np.sum(pred_X, axis=1), bins=80, alpha=0.5, label="Predicted", density=True)
plt.hist(np.sum(real_X, axis=1), bins=80, alpha=0.5, label="Real", density=True)
plt.xlabel("Total counts per cell")
plt.ylabel("Density")
plt.title("Library Size Distribution")
plt.legend()
plt.show()

# -----------------------
# 3. 每个基因的平均表达分布
# -----------------------
plt.figure(figsize=(6,4))
plt.hist(np.log1p(np.mean(pred_X, axis=0)), bins=80, alpha=0.5, label="Predicted", density=True)
plt.hist(np.log1p(np.mean(real_X, axis=0)), bins=80, alpha=0.5, label="Real", density=True)
plt.xlabel("log1p(Mean expression per gene)")
plt.ylabel("Density")
plt.title("Per-Gene Mean Expression Distribution")
plt.legend()
plt.show()

# %% cell 27
plt.hist(np.log1p(pred_X[pred_X>0]), bins=80, alpha=0.5, label="Pred")
plt.hist(np.log1p(real_X[real_X>0]), bins=80, alpha=0.5, label="Real")
plt.legend()
plt.title("Non-zero expression distribution")
plt.show()

# %% [markdown] cell 28
# ### 3. 可视化特定基因的表达

# %% cell 29
pred_adata

# %% cell 30
target_list = ["CD74","CTSB","FOS","MKI67","SORL1","TREM2","APOE","BIN1"]  # 示例

genes_available = [g for g in target_list if g in pred_adata.var_names]
genes_missing = [g for g in target_list if g not in pred_adata.var_names]

print("存在的基因：", genes_available)
print("缺失的基因：", genes_missing)

sc.pl.violin(
    pred_adata,
    keys=genes_available,
    groupby="Cell_type",  # 如果你有 cluster 信息
    stripplot=True,
    rotation=90
)

# %% cell 31
sc.pl.dotplot(
    pred_adata,
    var_names=genes_available,
    groupby="Cell_type",
    standard_scale="var",
)

# %% cell 32


sc.pl.umap(
    pred_adata,
    color=genes_available,  # 使用上一步找到的存在基因
    cmap="viridis",
    ncols=3,
    vmax="p99",   # 避免高表达值拉高色阶
)

# %% [markdown] cell 33
# 细胞类型特异性分析：
#
# 使用已知的细胞类型标注（存储在real_adata.obs中），计算pred_adata的细胞类型分类准确性。
# 方法：使用聚类算法（如Louvain或Leiden）对pred_adata进行聚类，比较聚类结果与真实细胞类型的重叠率（调整兰德指数，Adjusted Rand Index, ARI）。

# %% cell 34
from sklearn.metrics import adjusted_rand_score
import scanpy as sc

# # 对生成的矩阵进行聚类
# sc.pp.neighbors(pred_adata)
# sc.tl.leiden(pred_adata, resolution=0.5)

# 计算调整兰德指数
ari = adjusted_rand_score(real_adata_sub.obs['cell type'], pred_adata.obs['leiden'])
print(f"调整兰德指数: {ari:.4f}")

# %% [markdown] cell 35
# 方法概述：使用现成的单细胞推断工具（例如scVI、MAGIC、SAVER等）对real_adata进行dropout推断，生成一个填补了技术零值的表达矩阵（imputed_adata），然后将imputed_adata与pred_adata进行比较，评估pred_adata是否合理填补了技术零值，同时保留了生物学零值。

# %% cell 36
# NOTE: notebook-only command skipped in script: %pip install magic-impute

# %% cell 37
import magic
from scipy.stats import pearsonr, spearmanr, ks_2samp
from sklearn.metrics import mean_squared_error

patient_col='Patient ID'
knn=5
t=3
"""
使用magic对真实单细胞数据进行dropout推断，并与预测数据比较。

参数：
    real_adata (AnnData): 真实单细胞表达矩阵
    pred_adata (AnnData): 预测单细胞表达矩阵
    marker_genes (list, optional): 乳腺癌相关marker基因列表
    patient_col (str): 存储患者ID的列名
    knn (int): MAGIC的k近邻数
    t (int): MAGIC的扩散步数

返回：
    dict: 比较结果（相关性、MSE、分布一致性、marker基因分析）
    """
# 预处理real_adata
sc.pp.filter_cells(real_adata_sub, min_genes=200)
sc.pp.filter_genes(real_adata_sub, min_cells=3)
sc.pp.normalize_total(real_adata_sub, target_sum=1e4)
sc.pp.log1p(real_adata_sub)

# 使用MAGIC进行推断
magic_op = magic.MAGIC(knn=knn, t=t)
imputed_expr = magic_op.fit_transform(real_adata_sub.X, genes='all_genes')
imputed_adata = sc.AnnData(imputed_expr, obs=real_adata_sub.obs, var=real_adata_sub.var)

# 确保pred_adata与imputed_adata的基因和细胞对齐
common_genes = list(set(pred_adata.var_names) & set(imputed_adata.var_names))
pred_adata = pred_adata[:, common_genes].copy()
imputed_adata = imputed_adata[:, common_genes].copy()

# 提取表达矩阵
pred_X = pred_adata.X.toarray() if issparse(pred_adata.X) else pred_adata.X
imputed_X = imputed_adata.X.toarray() if issparse(imputed_adata.X) else imputed_adata.X

# 计算稀疏性
pred_sparsity = np.mean(pred_X == 0)
imputed_sparsity = np.mean(imputed_X == 0)
print(f"pred_sparsity: {pred_sparsity:.4f}")
print(f"imputed_sparsity: {imputed_sparsity:.4f}")
# pred_sparsity: 0.5031
# imputed_sparsity: 0.0054


# KS检验比较表达分布
ks_stat, ks_pval = ks_2samp(pred_X.flatten(), imputed_X.flatten())
print(f"KS Statistic: {ks_stat:.4f}, p-value: {ks_pval:.4f}")
# KS Statistic: 0.4976, p-value: 0.0000

# 可视化UMAP
sc.pp.pca(pred_adata)
sc.pp.neighbors(pred_adata)
sc.tl.umap(pred_adata)
sc.pp.pca(imputed_adata)
sc.pp.neighbors(imputed_adata)
sc.tl.umap(imputed_adata)

# %% cell 38
sc.pl.umap(pred_adata, color=['Sample'], title='Predicted Data UMAP')
sc.pl.umap(imputed_adata, color=[patient_col], title='Imputed Data UMAP')

# %% [markdown] cell 39
# 小提琴图可视化每种细胞类型的marker基因的表达

# %% cell 40
marker_df = pd.read_csv(os.path.join(data_path, 'result','cVAE','covid','marker_genes.csv'))
marker_df

# %% cell 41
import seaborn as sns
import matplotlib.pyplot as plt

gene = 'RALGPS2'
cell_type = 'B_cell'

# 获取B_cell细胞的索引
bcell_idx_pred = pred_adata.obs['Cell_type'] == cell_type
bcell_idx_real = real_adata_sub.obs['cell type'] == cell_type

# 检查基因是否在共同基因中
if gene not in common_genes:
    raise ValueError(f"{gene} not in common_genes!")

gene_idx = list(common_genes).index(gene)

# 提取表达量
expr_pred = pred_adata.X[bcell_idx_pred, gene_idx].toarray().flatten() if hasattr(pred_adata.X, "toarray") else pred_adata.X[bcell_idx_pred, gene_idx]
expr_real = real_adata_sub.X[bcell_idx_real, gene_idx].toarray().flatten() if hasattr(real_adata_sub.X, "toarray") else real_adata_sub.X[bcell_idx_real, gene_idx]

# 合并数据
df_plot = pd.DataFrame({
    'Expression': np.concatenate([expr_pred, expr_real]),
    'Dataset': ['Predicted'] * len(expr_pred) + ['Real'] * len(expr_real)
})

plt.figure(figsize=(6, 4))
sns.violinplot(x='Dataset', y='Expression', data=df_plot, inner='box')
plt.title(f'{gene} expression in {cell_type} cells')
plt.ylabel('Expression')
plt.show()

# %% cell 42
# 获取B_cell marker基因列表，去除缺失值和重复
bcell_markers = marker_df['B_cell'].dropna().unique().tolist()
# 只保留在common_genes中的基因
bcell_markers = [g for g in bcell_markers if g in common_genes]

# 提取表达量
def get_expr(adata, idx, genes):
    gene_indices = [list(common_genes).index(g) for g in genes]
    X = adata.X[idx, :][:, gene_indices]
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X

expr_pred = get_expr(pred_adata, bcell_idx_pred, bcell_markers)
expr_real = get_expr(real_adata_sub, bcell_idx_real, bcell_markers)

# 合并为长表
df_pred = pd.DataFrame(expr_pred, columns=bcell_markers)
df_pred = df_pred.melt(var_name='Gene', value_name='Expression')
df_pred['Dataset'] = 'Predicted'

df_real = pd.DataFrame(expr_real, columns=bcell_markers)
df_real = df_real.melt(var_name='Gene', value_name='Expression')
df_real['Dataset'] = 'Real'

df_all = pd.concat([df_pred, df_real], ignore_index=True)

plt.figure(figsize=(len(bcell_markers)*0.7, 5))
sns.violinplot(x='Gene', y='Expression', hue='Dataset', data=df_all, split=True, inner='box')
plt.title('Marker gene expression in B_cell')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %% cell 43
import seaborn as sns
import matplotlib.pyplot as plt

# 展平表达矩阵
expr_pred_flat = pred_X.flatten()
expr_real_flat = real_marker_X.flatten()

# 数据框
df_expr = pd.DataFrame({
    'Expression': np.concatenate([expr_pred_flat, expr_real_flat]),
    'Dataset': ['Predicted'] * len(expr_pred_flat) + ['Real'] * len(expr_real_flat)
})

# ECDF图，美观设置：阶梯、置信区间
plt.figure(figsize=(6, 4), dpi=300)
sns.ecdfplot(data=df_expr, x='Expression', hue='Dataset', stat='proportion', complementary=False, palette='Set1')
plt.title('Empirical Cumulative Distribution of Expression Values')
plt.xlabel('Expression Value')
plt.ylabel('Cumulative Proportion')
plt.xlim([0, df_expr['Expression'].max() * 0.1])  # 聚焦低表达区
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ecdf_sparsity.png', bbox_inches='tight')
plt.show()

# %% cell 44
# NOTE: notebook-only command skipped in script: %pip install ggridges

# %% cell 45
bcell_markers = marker_df['B_cell'].dropna().unique().tolist()
# 只保留在common_genes中的基因
bcell_markers = [g for g in bcell_markers if g in common_genes]

gene_indices = [list(common_genes).index(g) for g in bcell_markers]

# 提取表达数据
expr_pred = pred_X[:, gene_indices]  # 预测B细胞标记基因表达
expr_real = real_marker_X[:, gene_indices]  # 真实B细胞标记基因表达

# 转换为长表格式
df_pred = pd.DataFrame(expr_pred, columns=bcell_markers)
df_pred = df_pred.melt(var_name='Gene', value_name='Expression')
df_pred['Dataset'] = 'Predicted'

df_real = pd.DataFrame(expr_real, columns=bcell_markers)
df_real = df_real.melt(var_name='Gene', value_name='Expression')
df_real['Dataset'] = 'Real'

df_all = pd.concat([df_pred, df_real], ignore_index=True)

# 确保表达值非负（若数据未log变换，建议log1p）
df_all['Expression'] = np.log1p(df_all['Expression'])  # log变换以压缩高表达值

# Ridge Plot
g = sns.FacetGrid(df_all, row='Gene', hue='Dataset', height=0.5, aspect=15, sharex=True)
g.map(sns.kdeplot, 'Expression', fill=True, alpha=0.6, linewidth=1)
g.set_titles('{row_name}', fontsize=10)
g.set_axis_labels('Log(Expression + 1)', 'Density')
g.add_legend(title='Dataset')
plt.suptitle('Ridge Plot of B Cell Marker Gene Expression', y=1.02, fontsize=12)
plt.savefig('ridge_plot_bcell_markers_seaborn.png', bbox_inches='tight', dpi=300)
plt.show()

# %% [markdown] cell 46
# 对比细胞比例
