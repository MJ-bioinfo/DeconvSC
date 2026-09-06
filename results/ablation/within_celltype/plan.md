# GSE141115 within-cell-type expression-topology 分析计划

## Material Passport

- Origin Skill: `academic-research-suite / experiment-agent`
- Origin Mode: plan
- Origin Date: 2026-07-10
- Verification Status: `EXECUTED_VERIFIED`
- Version Label: `within_celltype_topology_plan_v1.1`

## 0. 当前状态与边界

- 本计划已于 2026-07-10 执行完成；结果见 `analysis_report_zh.md`，执行记录见 `experiment_record.md`，最终验证为 55/55 checks passed（包括 Figure 4 A–D 成图）。
- 原始预设规则保留在本文件中；执行期唯一实质修订是 Gate 0 触发的表达尺度修订，并已在下文及 `protocol_deviations.csv` 透明记录。
- 已完成的 pooled/global 分析及其全部产物位于相邻的 `../global/`。
- 所有输入只读；后续新增脚本、缓存、表、图和日志只能写入 `within_celltype/`。
- 不改动 `ablation_compare/`、Figure 3/4、manuscript、原始 h5ad 或 Figure 3 中间文件。
- 本分析检验的是**细胞类型内部的基因共表达/统计依赖拓扑保真度**，不是有方向的 GRN 重建，也不把 attention edge 直接解释为调控边。

## 1. 研究问题与预设假设

### 1.1 主要问题

在固定细胞类型后，DeconvSC 生成的单细胞表达是否比去除 attention 的 BaseVAE 更接近真实单细胞的基因—基因关系结构？

### 1.2 主要假设

- 主要比较：Figure 3 对应的 DeconvSC vs BaseVAE。
- 主要终点：cell-type-specific module L2 error，越小越好。
- 方向性假设：`module_L2(DeconvSC) < module_L2(BaseVAE)`。
- 同时报告双侧检验，避免只呈现有利方向。
- 当前 prophead DeconvSC 仅作 generation/version 敏感性分析，不替换 Figure 3 主版本。

### 1.3 可支持的结论

如果主要终点和预设敏感性分析方向一致，可表述为：

> DeconvSC 在本 benchmark 的可评估细胞类型中，更好地保持了细胞类型内部的表达相关结构。

不能据此表述为“恢复了真实 GRN”“attention 学到了因果调控边”，也不能把只含 3 位 donor 的结果外推为人群层面的生物学显著性。

## 2. 方法纳入资格

| 方法/数据 | 单细胞级输出 | within-cell-type 主分析 | 角色与原因 |
|---|---:|---:|---|
| GT raw test | 是 | 是 | 真实参照 |
| DeconvSC Figure 3 | 是 | 是 | 主要模型 |
| BaseVAE | 是 | 是 | attention ablation 对照 |
| DeconvSC current prophead | 是 | 敏感性 | 检查版本/generation 稳定性 |
| DISSECT | 否 | 否 | 每个 `sample × cell type` 只有聚合表达谱，无法估计类型内单细胞协方差 |
| BayesPrism | 否 | 否 | 同上；只能参加 global/profile-level 分析 |
| TAPE | 否 | 否 | 同上；只能参加 global/profile-level 分析 |
| CIBERSORTx | 否 | 否 | 还是聚合谱，且只有 major lineage 和受限基因面板 |

不能用每个 cell type 仅 3 个 donor profile 强行计算 500 基因的“within-cell-type network”：该矩阵秩极低，测到的是 donor 间变化而非类型内细胞状态，并且与单细胞模型粒度不公平。

## 3. 只读输入与版本锁定

| 角色 | 路径 |
|---|---|
| GT raw test | `/disk1/maijl/deconv/data/GSE141115/GSE141115_sc_raw_counts_test.h5ad` |
| DeconvSC Figure 3 | `/disk1/maijl/deconv/cVAE/GSE141115/attention_geneembed/generated_data.h5ad` |
| BaseVAE | `/disk1/maijl/deconv/cVAE/GSE141115/ablation_basevae/BaseVAE-generated_data.h5ad` |
| DeconvSC current sensitivity | `/disk1/maijl/deconv/deconv_20260610/prophead/GSE141115/generated_data.h5ad` |
| 旧 within-type 脚本（仅作口径参照） | `/disk1/maijl/deconv/deconv_20260610/04_evaluation/within_type_module_l2.py` |

只使用 `LDK1`、`LDK2`、`LDK3`。执行前生成 `input_manifest.csv`，记录 SHA256、shape、稀疏性、obs/var 字段、表达尺度、donor 与 cell-type 标签；执行后重新核对输入哈希。

## 4. 已完成的样本量初筛

以下是 LDK1–3 合计细胞数的只读初筛。按 GT、Figure 3 DeconvSC、BaseVAE 三者均不少于 100 个细胞，暂定 6 个主分析 cell types：

| Cell type | GT | DeconvSC Figure 3 | BaseVAE |
|---|---:|---:|---:|
| CD_PC | 1,046 | 358 | 505 |
| CNT | 893 | 151 | 293 |
| DCT | 871 | 245 | 536 |
| MC | 362 | 196 | 348 |
| PT | 9,414 | 1,096 | 1,831 |
| aLOH | 1,265 | 1,086 | 2,270 |

预设阈值敏感性：

- `min_total_cells = 50`：预计 11 类（CD_IC、CD_PC、CNT、DCT、Endo、Fib、MC、PT、Podo、T、aLOH）。
- `min_total_cells = 100`：主分析，预计上述 6 类。
- `min_total_cells = 200`：预计 4 类（CD_PC、DCT、PT、aLOH）。
- current DeconvSC 在阈值 100 下预计为 CD_PC、DCT、Endo、MC、PT、aLOH；该集合只用于敏感性分析。

这些只是 pooled count 初筛。最终资格还需通过 donor 内细胞数检查：每个纳入 cell type 至少有 2 个 donor，且每个可用 `cell type × donor × method` 至少 20 个细胞。不得在看到结果后降低阈值或挑选 cell types。

## 5. 预处理与匹配

> **执行期尺度修订（2026-07-10）**：Gate 0 发现生成矩阵并不严格满足单细胞 CP10K（共同面板上 `expm1` library 中位数约为 DeconvSC 2,041、BaseVAE 2,752，而 GT 约为 9,995）。因此 P0 对所有方法统一执行 `expm1 → cell-wise CP10K → log1p`；未重新归一化的 decoder log 值保留为 `S7_native_decoder_scale`。该修订及“修订前已查看初始结果”的事实记录在 `protocol_deviations.csv`，两种尺度必须方向一致才能支持稳健结论。

### 5.1 表达尺度

1. 明确检查每个输入是 raw count、CP10K 还是 log1p-CP10K，不依赖单一最大值阈值猜测。
2. GT raw count 转为 `log1p(CP10K)`。
3. 对已经是 log1p-CP10K 的生成结果不重复归一化；以聚合后与 Figure 3 中间文件的一致性作为 DeconvSC 来源校验。
4. 仅保留 GT、DeconvSC、BaseVAE 的共同基因；统一基因名并记录重复、缺失和常数基因。

### 5.2 donor 配平与残差化

对每个 cell type、donor、seed：

1. 令 `n(c,d) = min(n_GT, n_DeconvSC, n_BaseVAE, 200)`；三种方法从该组各抽取相同数量细胞。
2. 若 `n(c,d) < 20`，该 donor 不进入该 cell type；cell type 仍须至少保留 2 个 donor。
3. 在各方法内部，对每个 `cell type × donor` 的每个基因减去该组均值，再合并 donor 残差计算相关矩阵。这样去除 donor 均值位移，同时保持三种方法相同的 donor 组成。
4. 使用 10 个预先固定的随机种子（建议 `2024–2033`）。seed 只衡量抽样稳定性，不增加统计样本量。

补充敏感性：不做 donor centering 的 legacy pooled-within-type 口径；以及 donor 内分别计算相关后 Fisher-z 等权合并的口径。

## 6. 基因面板与模块定义

### 6.1 主分析：真正的 cell-type-specific topology

对每个 cell type 单独执行，且所有选择只使用 GT：

1. 在 GT 的 donor-centered residual 中按方差选择 top 500 genes。
2. 在该 cell type 的 GT 相关矩阵上定义模块；预测方法绝不参与选基因、聚类或筛模块。
3. 距离为 `1 - |r|`，average linkage，切为 20 个候选模块；过滤 `<6` genes 的模块，再按 GT 模块内平均 `|r|` 保留最多 10 个。
4. 每个 cell type 可有自己的基因面板与模块；这正是 cell-type-specific 检验，而不是把各类型先混成一个网络。

### 6.2 与旧代码衔接的 bridge analysis

另做一个明确标记为 bridge/legacy 的结果：GT 选择共同 top-500，先对每个类型算相关，再对 cell types 等权平均，随后定义共同模块。它用于与旧 `within_type_module_l2.py` 对照，不能取代主分析。

预设敏感性：

- 全局 GT HVG 面板（旧口径）；
- Ward linkage（旧口径）；
- 共同面板/共同模块 vs 每类独立面板/模块。

不得根据哪种设置能产生更小 p 值来选择主结果。

## 7. 拓扑指标

所有指标先在每个 `cell type × seed` 内计算，再汇总到 cell type；不把 cell、gene、edge、module 或 seed 当作独立生物学重复。

### 7.1 主要终点：module L2

延续 Figure 4 口径，对 cell type `c` 的 GT 模块 `M`：

\[
L2_{m,c,M,s}=\frac{\lVert R^{GT}_{c,M,s}-R^{m}_{c,M,s}\rVert_F}{|M|^2}
\]

同时报告两个尺度敏感性指标：

- 矩阵元素 RMSE：`Frobenius / |M|`；
- 上三角 edge MAE。

原因是旧式 `Frobenius / |M|²` 会随模块大小变化；只有在替代尺度下方向也稳定，才认为结论稳健。

### 7.2 次要终点

- Hub-neighbor Jaccard：GT 定义 top-50 hubs，比较每个 hub 的 top-20 邻居集合。
- Network PCC：比较 GT 与预测的相关矩阵上三角向量。
- Per-gene co-expression edge MAE：逐基因计算 model 与 ground truth 在其余 499 条 Pearson 相关边上的平均绝对误差，再在 500 个基因上取中位数。该指标不依赖聚类，作为对 hub 局部邻居、MI 非线性依赖和 module-level 块结构的全网络 gene-centric 补充。原 per-gene connectivity-profile PCC 保留为敏感性指标。
- MI fidelity：保留旧 20-bin MI 作为可比口径，同时增加 kNN-MI 或秩变换后的敏感性；小样本下不把大量 gene pairs 当作独立 N。

## 8. 统计检验：避免 module × seed 伪重复

### 8.1 推断单位

对每个方法与 cell type，先对 module 和 seed 求平均：

\[
E_{m,c}=mean_s\{mean_M(L2_{m,c,M,s})\}
\]

主要配对差值为 `D_c = E_BaseVAE,c - E_DeconvSC,c`；`D_c > 0` 有利于 DeconvSC。

### 8.2 主检验与报告

- exact one-sided paired Wilcoxon：DeconvSC `<` BaseVAE；
- 同时报告 exact two-sided p；
- 报告各 cell type 原始差值、median paired difference、paired rank-biserial effect、cell-type win fraction 和 bootstrap 95% CI；
- 以 exact sign-flip permutation 作为敏感性检查；
- 4 个次要 topology metrics 使用 Holm 校正。

主分析预计只有 6 个 cell types：若 6 类差值全部同方向，单侧精确检验的最小 p 约为 `1/64 = 0.015625`。因此应优先呈现效应大小、方向一致性和敏感性，而不是通过 module×seed 扩大 N。这个 p 值只描述“跨本 benchmark 细胞类型的一致性”，不代表 donor 层面的人群推断。

### 8.3 donor 稳定性

- 做 leave-one-donor-out（每次去掉 LDK1/2/3 中一个）并报告方向是否保持。
- donor 内样本充足时，补充 `donor × cell type` 误差；但仅 3 个 donor，不将其包装成有充分统计功效的人群检验。
- 若主要显著性只存在于把 module、seed 或 edge 当独立样本的分析，而 cell-type-level 检验不支持，则不得声称显著更优。

## 9. attention 归因审计

在把 DeconvSC 与 BaseVAE 的差异归因为 attention 前，必须核对两者是否只改变 attention：

- 相同 train/test split、预处理、训练数据和随机种子；
- 相同 latent size、encoder/decoder 容量、优化器、epoch 与 early stopping；
- 相同重建、KL、adversarial、correlation 等 loss 及权重；
- 相同生成路线、每个 donor/type 的目标细胞数和后处理。

若还存在其他差异，结果只能称为“DeconvSC model vs BaseVAE model”，不能单独归因于 attention。尤其现有 correlation loss 本身也可能改善 topology，必须与 attention 的贡献区分。

## 10. 预设执行矩阵

| ID | DeconvSC 来源 | Cell threshold | donor 处理 | 基因/模块 | 目的 |
|---|---|---:|---|---|---|
| P0 | Figure 3 | 100 | donor-centered、配平 | 每类 GT-specific、average | 主分析 |
| S1 | current prophead | 100 | 同 P0 | 同 P0 | 模型版本敏感性 |
| S2a | Figure 3 | 50 | 同 P0 | 同 P0 | 阈值敏感性 |
| S2b | Figure 3 | 200 | 同 P0 | 同 P0 | 阈值敏感性 |
| S3 | Figure 3 | 100 | 不中心化 | 同 P0 | 旧 pooled-within-type 敏感性 |
| S4 | Figure 3 | 100 | 同 P0 | global GT HVG/共同模块 | feature/module 敏感性 |
| S5 | Figure 3 | 100 | 同 P0 | Ward linkage | 旧聚类口径敏感性 |
| S6 | Figure 3 | 100 | donor 内相关、Fisher-z 合并 | 同 P0 | donor 权重敏感性 |
| S7 | Figure 3 | 100 | donor-centered、配平 | 原始 decoder log 尺度 | 输入尺度敏感性 |

## 11. 质量门与执行顺序

1. **Gate 0 — inventory**：输入存在、哈希、obs/var schema、表达尺度、每 donor/type 细胞数完整。
2. **Gate 1 — provenance**：Figure 3 DeconvSC h5ad 聚合后与已验证 Figure 3 中间表达在容差内一致。
3. **Gate 2 — ablation equivalence**：完成 attention 归因审计；若失败，降级结论措辞但仍可做模型比较。
4. **Gate 3 — eligibility lock**：按预设规则冻结 cell types、donors、genes，不看模型指标后再调整。
5. **Gate 4 — legacy reproduction**：新代码先在 bridge setting 复现旧 within-type 数值或解释确定差异。
6. **Gate 5 — primary run**：运行 P0，保存逐 type/module/seed 原始值。
7. **Gate 6 — statistics**：只从冻结结果生成 cell-type-level 检验。
8. **Gate 7 — sensitivities**：依次运行 S1–S6，区分 confirmatory 与 exploratory。
9. **Gate 8 — validation**：确定性复跑、输入哈希复核、表图数值一致性和 NaN/常数审计。
10. 用户审阅数值后，再决定是否重画 Figure 4；本计划本身不修改 manuscript。

## 12. 预期输出（已生成）

```text
within_celltype/
├── plan.md
├── scripts/
├── logs/
├── cache/
├── figures/
├── input_manifest.csv
├── architecture_equivalence_audit.csv
├── cell_count_audit.csv
├── eligibility_matrix.csv
├── within_type_gene_panel.csv
├── within_type_modules.csv
├── within_type_metrics_by_celltype.csv
├── module_l2_by_celltype_module_seed.csv
├── statistical_tests.csv
├── sensitivity_analysis.csv
├── raw_distributions.npz
├── experiment_record.md
├── validation_checks.csv
├── validation_report.md
└── analysis_report_zh.md
```

`analysis_report_zh.md` 必须分别列出：主分析、legacy bridge、敏感性、不能纳入的方法、统计功效限制和可/不可写入 manuscript 的结论。

## 13. 潜在下一步：attention-edge utility（不属于本次主分析）

若 within-cell-type topology 显示稳定改善，可进一步检验 attention 是否**有助于转录组重建**，而不是直接把它称为 GRN：

1. 分 cell type、layer、head、donor 提取 attention matrix；检查不同 seed/批次的边稳定性和头间一致性。
2. 将 attention edge 与 GT within-type co-expression、已知 TF–target/通路集合做富集或排序一致性分析；这些只算外部一致性证据，不算因果证明。
3. 做 matched perturbation：遮蔽 top-attention edges，与度数、权重和数量匹配的随机 edges 对照；比较表达重建 Pearson、cell-type marker 保真度和本计划的 topology metrics。
4. 做 attention randomization/head ablation，并与完整模型、BaseVAE、correlation-loss ablation 组成正交消融，区分 attention、correlation loss 和模型容量贡献。
5. 只有当 top-edge 定向遮蔽造成的损失稳定大于 matched-random，且在 donor/cell type/seed 间复现，才可称 attention edge 对转录组重建具有功能性贡献；仍不能仅凭此称为 causal regulatory edge。

该下一步应另建独立子目录和预注册计划，不能用本次 topology 结果反向选择最有利的 attention layer/head。
