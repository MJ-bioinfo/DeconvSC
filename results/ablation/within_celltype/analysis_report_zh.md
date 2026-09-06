# GSE141115 within-cell-type expression topology 分析报告

## Material Passport

- Origin Skill: `academic-research-suite / experiment-agent`
- Origin Mode: run + validate
- Origin Date: 2026-07-10
- Verification Status: `VERIFIED`
- Version Label: `within_celltype_topology_result_v1`

## 1. 结论摘要

在预先规定的 6 个可评估细胞类型（CD_PC、CNT、DCT、MC、PT、aLOH）中，DeconvSC 的主要终点 module L2 在 **6/6** 类型均低于 BaseVAE：

- DeconvSC mean：`0.01613`
- BaseVAE mean：`0.04719`
- 相对降低：`65.81%`
- cell-type-level exact one-sided paired Wilcoxon：`p = 0.015625`
- exact two-sided p：`0.03125`
- median paired advantage（BaseVAE − DeconvSC）：`0.02893`
- bootstrap 95% CI：`[0.01590, 0.04834]`
- paired rank-biserial effect：`1.0`

因此，在这个 GSE141115 benchmark 的可评估细胞类型中，DeconvSC 对**细胞类型内部表达相关结构**的保持显著优于 BaseVAE。该结果不能单独归因为 attention，也不能解释为有方向或因果的 GRN 恢复。

## 2. 数据来源与资格

### 2.1 来源核验

Figure 3 DeconvSC 单细胞 h5ad 聚合后，与已经用于 Figure 3 的 `pb_predicted_log1p_cp10k.csv.gz` 比较：

- 48 个 `LDK1–3 × cell type` profiles；
- 16,801 个基因；
- 最大绝对差异 `3.41 × 10⁻⁶ < 10⁻⁵`。

因此本分析使用的 DeconvSC Figure 3 h5ad 与 Figure 3 表达预测来源一致。

### 2.2 纳入方法

- 主分析：GT、DeconvSC Figure 3、BaseVAE。
- 敏感性：current prophead DeconvSC。
- 未纳入 within-cell-type：DISSECT、BayesPrism、TAPE、CIBERSORTx。它们只有聚合 profile，不能估计类型内单细胞协方差。

### 2.3 细胞类型与匹配

主分析冻结为 6 类，每个类型在 GT、DeconvSC、BaseVAE 的 LDK1–3 中均满足：

- pooled total cells ≥100；
- 每个 donor/method ≥20 cells；
- 每个 `cell type × donor` 三种方法抽取相同细胞数，单 donor cap=200；
- 10 个固定 seeds 只用于 Monte-Carlo 稳定性，不作为统计重复。

## 3. 执行期尺度审计与修订

Gate 0 发现，生成矩阵虽然是非负 decoder log 输出，但并不严格保持单细胞 CP10K library：

| 数据 | 共同面板 `expm1` library 中位数 | P0 变换后 |
|---|---:|---:|
| GT | 9,994.5 | 9,994.5 |
| DeconvSC Figure 3 | 2,040.8 | 10,000.0 |
| BaseVAE | 2,751.5 | 10,000.0 |
| DeconvSC current | 2,164.3 | 10,000.0 |

因此 P0 对所有方法统一使用 `expm1 → cell-wise CP10K → log1p`。原始 decoder log 尺度作为 `S7_native_decoder_scale` 保存。修订发生前初始 native 结果已被查看，因此该偏离被明确记录在 `protocol_deviations.csv`，而不是隐去。

S7 仍为 6/6 类型支持 DeconvSC，module L2 单侧精确 p 同为 `0.015625`，说明主要方向不依赖这次尺度修订。

## 4. 主终点：cell-type-specific module L2

每个细胞类型均独立使用 GT donor-centered residual variance 选择 top-500 genes，并在该类型 GT 矩阵上以 `1-|r|`、average linkage 定义模块。预测模型不参与基因选择、聚类或模块筛选。

| Cell type | DeconvSC | BaseVAE | Advantage (BaseVAE − DeconvSC) |
|---|---:|---:|---:|
| CD_PC | 0.01178 | 0.02980 | 0.01802 |
| CNT | 0.02271 | 0.08170 | 0.05899 |
| DCT | 0.01838 | 0.04635 | 0.02797 |
| MC | 0.01918 | 0.05687 | 0.03770 |
| PT | 0.00375 | 0.01752 | 0.01377 |
| aLOH | 0.02099 | 0.05088 | 0.02989 |

旧 `Frobenius/|M|²` 尺度可能受模块大小影响，所以同时检查：

| Module error | DeconvSC mean | BaseVAE mean | Relative improvement | Wins | One-sided exact p |
|---|---:|---:|---:|---:|---:|
| Legacy module L2 | 0.01613 | 0.04719 | 65.81% lower | 6/6 | 0.015625 |
| Matrix-element RMSE | 0.26295 | 0.73458 | 64.20% lower | 6/6 | 0.015625 |
| Upper-triangle edge MAE | 0.23676 | 0.72916 | 67.53% lower | 6/6 | 0.015625 |

三种尺度方向完全一致，因此 DeconvSC 优势不是 `|M|²` 除数单独造成的。

## 5. Figure 4 的互补拓扑指标

| Metric | DeconvSC mean/median | BaseVAE mean/median | Wins | Raw one-sided p | Holm p |
|---|---:|---:|---:|---:|---:|
| Hub-neighbor Jaccard (mean) | 0.12400 | 0.11612 | 5/6 | 0.046875 | 0.0625 |
| MI fidelity error (median) | 0.10924 | 0.82862 | 6/6 | 0.015625 | 0.0625 |
| Module L2 error (mean) | 0.01613 | 0.04719 | 6/6 | 0.015625 | 0.0625 |
| Per-gene co-expression edge MAE (median) | 0.24819 | 0.78753 | 6/6 | 0.015625 | 0.0625 |

解释：

- Figure 4a 的 hub Jaccard 只检验 50 个 ground-truth hub 的 top-20 局部邻居集合，忽略其余边和边权幅度。
- Figure 4b 的 MI error 检验非线性成对依赖；Figure 4c 的 module L2 检验 ground-truth 定义模块中的 Pearson 相关块结构。
- Figure 4d 的 per-gene edge MAE 对每个基因使用全部 499 条非自身 Pearson 边，先计算 model–real 绝对差的均值，再在 500 个基因上取中位数；它不依赖聚类，用于检验稠密、gene-centric 线性共表达边权的校准误差。
- 因此 A–D 分别覆盖局部稀疏邻居、非线性成对依赖、中观模块块结构和全网络基因中心的边幅度误差，而不是四次重复检验同一性质。
- 由于只有 6 个 cell types，精确 p 值分辨率有限；四个 Figure 4 指标经 Holm 后均为 `0.0625`，未达到 0.05。因此应报告为一致的数值优势，不能写成多重校正后显著。
- MI 的 rank/10-quantile 敏感性仍为 6/6，单侧 p=`0.015625`，说明 MI 方向不依赖 20-bin 等宽分箱。

原来的 network PCC 和 per-gene connectivity-profile PCC 仍保留在结果表中作为补充敏感性指标，但不再用于 Figure 4d。

## 6. 敏感性分析

| Setting | Types | DeconvSC mean L2 | BaseVAE mean L2 | Wins | One-sided exact p |
|---|---:|---:|---:|---:|---:|
| P0 primary | 6 | 0.01613 | 0.04719 | 6/6 | 0.015625 |
| S1 current DeconvSC | 6 | 0.01018 | 0.03977 | 6/6 | 0.015625 |
| S2a threshold 50 | 10 | 0.01728 | 0.04117 | 10/10 | 0.0009766 |
| S2b threshold 200 | 4 | 0.01373 | 0.03614 | 4/4 | 0.0625 |
| S3 no donor centering | 6 | 0.01604 | 0.04712 | 6/6 | 0.015625 |
| S4 shared/global panel | 6 | 0.000133 | 0.000250 | 6/6 | 0.015625 |
| S5 Ward linkage | 6 | 0.01938 | 0.04127 | 6/6 | 0.015625 |
| S6 donor Fisher-z | 6 | 0.01621 | 0.04714 | 6/6 | 0.015625 |
| S7 native decoder scale | 6 | 0.01367 | 0.04719 | 6/6 | 0.015625 |
| LODO without LDK1 | 6 | 0.01654 | 0.04712 | 6/6 | 0.015625 |
| LODO without LDK2 | 6 | 0.01638 | 0.04712 | 6/6 | 0.015625 |
| LODO without LDK3 | 6 | 0.01689 | 0.04757 | 6/6 | 0.015625 |

S2b 的 p=0.0625 是 4 个配对单位下精确检验的最小单侧 p，并非方向不一致。S4 的绝对值不可与 P0 直接比较：共同 average-linkage 模块中存在一个 480-gene 大模块，而旧 L2 除以 `|M|²`，会显著压缩数值；该 setting 只用于方向敏感性。

## 7. 与旧 within-type 脚本的关系

未修改的旧脚本已在两个隔离目录运行，两次 NPZ/CSV 5/5 完全一致。其 Figure 3 source 结果为：

- 11 个类型；
- 50 个 `module × seed` 数值；
- DeconvSC median `0.02212`，BaseVAE median `0.02192`；
- 但 pooled/module×seed 单侧 Wilcoxon `p=3.17×10⁻⁴`。

旧口径有三个核心问题：

1. GT 只有 LDK1–3，但生成数据未过滤 LDK4–6；
2. 先跨类型等权平均成一个 consensus matrix，不能检验真正的 cell-type-specific topology；
3. 把 module×seed 当作 50 个独立推断单位，产生伪重复。

新分析保留旧脚本作为可复现 bridge，但 Figure 4 显著性必须使用 6 个 cell-type-level 值，而不是旧 50 个值。

## 8. Attention 归因边界

DeconvSC 的精确训练代码存在，但 BaseVAE 输出目录没有保存 GSE141115 的精确训练脚本或配置；训练 split、loss 权重、latent/capacity 和生成设置无法逐项证明一致。此外两者原始生成细胞数和基因维度不同。

因此当前结果可支持：

> DeconvSC model 比现有 BaseVAE output 更好地保持 within-cell-type expression topology。

当前结果不能单独支持：

> attention 本身导致了全部改善。

若要写 attention-specific 结论，需要补齐严格正交消融，至少固定 correlation loss、训练/生成 seed、模型容量、细胞数和所有其他超参数。

## 9. Figure 4 输入建议

已生成与原 Figure 4 配色和版式一致的 2×2 A–D 成图：`figures/Figure4_within_celltype_ABCD.{png,pdf,svg,emf}`。图中保留 Arial、箱线图及 6 个未连线的 cell-type 散点，不再使用核密度、小提琴图层或细胞类型间的连接线。箱体表示六个细胞类型值的中位数和四分位距，须线延伸至 1.5×IQR 范围内的最远观测值，散点显示实际 cell-type 结果。全部 panel 均不显示检验括号、星号、p 值或 `ns`。检验结果仍完整保留在 `figure4_inputs/paired_statistics.csv` 和本报告中。四个独立 panel 也位于 `figures/`。

已在 `figure4_inputs/` 准备：

- `topology_metrics_long.csv`：首选 tidy 数据；每行一个 `cell type × method × metric`。
- `paired_statistics.csv`：cell-type-level 精确检验、效应量和 CI。
- `within_celltype_panel_data.npz`：hub Jaccard、MI error、module L2、per-gene co-expression edge MAE 及补充 PCC 指标的紧凑数组。
- `module_l2_withinType_dists.npz`：与现有 panel-C loader 的 `Route2/BaseVAE/p` 键兼容。

Figure 4 panel C 可以直接使用兼容 NPZ，但图注必须写明 `n=6 cell types`，10 seeds 已在类型内平均。不要加入 DISSECT 的 global L2 作为第三个同尺度分布，也不要使用旧 `module×seed n=50` 显著性。

若 Figure 4 的 a–d 全部改为 within-cell-type，应从 `topology_metrics_long.csv` 重画两模型 paired plots；BayesPrism/TAPE/CIBERSORTx/DISSECT 只能保留在 global/profile-level 图中。

## 10. 可写入 manuscript 的保守表述

> Across six sufficiently represented kidney cell types, DeconvSC showed lower within-cell-type module correlation error than the ablated VAE (6/6 cell types; exact one-sided paired Wilcoxon p=0.0156). The direction was stable across module-error scaling, cell-count thresholds, donor handling, clustering, model-output version, and leave-one-donor-out analyses. These metrics assess expression-topology fidelity rather than causal regulatory-network reconstruction.

不要使用 “reconstructed GRN”“identified regulatory edges” 或 “attention recovered causal regulation”。
