import os
import random
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt


def set_seed(seed=18):
    """set random seed for reproducibility across random, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 保证 cudnn 的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_cell_type_indices(labels, num_cell_types):
    """Precompute the indices of each cell type in the batch for faster sampling."""
    indices_dict = {}
    labels_np = labels.cpu().numpy()
    for i in range(num_cell_types):
        indices_dict[i] = np.where(labels_np == i)[0]
    return indices_dict

def get_sample_indices_from_bulk(real_bulk_path, target_samples, sep):
    """
    Read bulk data (TSV/CSV) and return the indices of target samples, along with the list of actual found sample names.
    """
    try:
        real_bulk_df = pd.read_csv(real_bulk_path, sep=sep, index_col=0).T
    except Exception as e:
        print(f"Error reading {real_bulk_path}: {e}")
        return [], []
    
    all_sample_names = real_bulk_df.index.tolist()
    print(f"Total samples in bulk data: {len(all_sample_names)}")
    
    sample_indices = []
    valid_target_samples = []
    
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
   