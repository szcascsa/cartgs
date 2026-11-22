# 仅全局XYZ的Encoder与四Head补偿输入消融

**结论：两个seed的七档平均PSNR均下降；本轮XYZ-only组合没有收益，当前证据不支持将被移除属性一并视为冗余。**

七档non-Bank same-K平均变化 **-1.337 dB**，两seed分别 **-1.703/-0.971 dB**。

Gaussian属性输入仅保留原始全局XYZ；移除局部Offset、Scale、SH DC、Opacity及Rotation。保留logratio、每Anchor每ratio latent16、共享h32、四个属性Head及14维补偿输出。

## non-Bank same-K结果

| ratio组 | seed3100 ΔPSNR | seed3101 ΔPSNR | 平均ΔPSNR |
| --- | --- | --- | --- |
| all | -1.703 | -0.971 | -1.337 |
| train | -1.350 | -0.388 | -0.869 |
| unseen | -2.175 | -1.749 | -1.962 |

| 方案 | PSNR↑ | SSIM↑ | LPIPS↓ | photo↓ |
| --- | --- | --- | --- | --- |
| baseline | 17.24435 | 0.60185 | 0.60022 | 0.15107 |
| xyz_only | 15.90709 | 0.56602 | 0.62009 | 0.17450 |

## 各ratio

| ratio | K | seed3100 ΔPSNR | seed3101 ΔPSNR | 平均ΔPSNR |
| --- | --- | --- | --- | --- |
| 1% | 68 | -1.783 | -0.891 | -1.337 |
| 2.5% | 170 | -2.945 | -3.070 | -3.008 |
| 5% | 341 | -1.538 | -0.309 | -0.923 |
| 7.5% | 512 | -2.150 | -1.036 | -1.593 |
| 10% | 683 | -1.365 | +0.080 | -0.643 |
| 12.5% | 854 | -1.429 | -1.141 | -1.285 |
| 15% | 1025 | -0.712 | -0.433 | -0.572 |

## Bank与留出

| 评价集合 | 平均ΔPSNR |
| --- | --- |
| bank_train | -1.470 |
| validation | -1.401 |
| test | -1.337 |

## 实际结构

- Baseline：原始11维为Offset3+Scale3+SH DC3+Opacity1+logratio1，Encoder11→32；补偿输入raw11+latent16+h32=59。
- XYZ-only：原始4维为全局XYZ3+logratio1，Encoder4→32；补偿输入raw4+latent16+h32=52。全局XYZ不另做AABB归一化或动态重拟合，输入tensor和统计均保存。
- 选择Head仍32→32 ReLU→2；补偿四Head分别52→32 ReLU→32 ReLU→3/3/4/4。其余网络宽深、14D属性含义及变换完全保留，只对选中K行解码。
- 16维特征是每Anchor每训练ratio一张表，只给补偿使用；未见ratio仍线性插值latent，logratio输入保留。共享h32现在仅由XYZ和logratio生成。
- Renderer仍使用真实Gaussian的完整属性，这些属性从网络输入移除，并未从真实地图或渲染器删除。

## 初始化与输入独立性

- 复用原第一层前三列作为XYZ输入权重、原logratio列、latent/h32列、bias和所有后续层；删除其余输入列。坐标含义变化，隐藏函数不再相同，但step0概率与零补偿输出exact一致。
- 用只暴露get_xyz的Gaussian代理替换输入对象，并将旧派生属性tensor设为NaN，Encoder和Compensation输出仍exact一致，证明两条路径不再隐式读取被移除属性。未修改真实Gaussian张量。
- Encoder及四Head梯度、当前latent梯度与其他latent表grad=None通过检查；零输出初始化下隐藏层真实首步梯度可为零，另用非零投影探针验证路径。

## 公平性与局限

- 同一fr1_xyz冻结地图N6838/M2066，Master/关联/位姿/eligible/cached uniform固定；54 Bank训练、13内部留出、45历史non-Bank。网络仍联合优化Selector/Encoder/补偿/latent，选择集合可以变化。
- 两seed各12000真实LoD步、每ratio3000；同帧序、ratio顺序、fresh fused Adam lr=.001 eps1e-15、原photo+.01预算损失。复用上一轮锁定的两个无Rotation Baseline，不选择最好复跑。
- 最终12000步评价；训练1/5/10/15%，未见2.5/7.5/12.5%；same-K按logit(p)-logit(cached uniform)排名。实际Cached数量及四指标另存，未按non-Bank选择LR/步数/模型。
- 本轮同时减少输入信息、将局部Offset换成全局XYZ、减少首层参数1120个，不能将所有质量差异归因为信息过多。只验证该精简组合，没有自动追加特征。
- 两seed仍来自同一地图，历史评价轨迹已用于结构研究；不能宣称跨场景或在线Joint收益。GPU同时存在其他任务，不作无干扰性能结论。
- 生产默认及已有改动保留，未stage、commit或push。

![质量曲线](tests/diagnostics/s00_xyz_only_20261026/evidence/analysis/quality_curves.png)

## 原Cached Bernoulli实际预算评价

| ratio组 | seed3100 ΔPSNR | seed3101 ΔPSNR | 平均ΔPSNR |
| --- | --- | --- | --- |
| all | -1.802 | -1.118 | -1.460 |
| train | -1.423 | -0.571 | -0.997 |
| unseen | -2.307 | -1.847 | -2.077 |

上述保留实际选点数差异，严格同预算的主对照仍为same-K；两种评价均完整保存。

## 内部Bank留出收敛曲线

| 方案 | seed | 4000步 | 8000步 | 12000步 |
| --- | --- | --- | --- | --- |
| baseline | 3100 | 16.746 | 17.351 | 17.496 |
| baseline | 3101 | 16.721 | 16.887 | 17.307 |
| xyz_only | 3100 | 15.542 | 15.704 | 15.800 |
| xyz_only | 3101 | 15.712 | 16.086 | 16.201 |

统一用最终12000步，不按留出或non-Bank选择checkpoint；曲线与有限训练预算不能证明充分收敛的表示能力上限。下一项若继续按用户要求逐步加信息，应每次只恢复一种输入，不自动扩展为矩阵。

## 交付核验

原型源码与服务器归档SHA一致；2个新增模型及2个锁定参考完整保存，步数、曝光和Adam时钟匹配，四指标finite。Git HEAD、全部tracked diff及staged diff与实验前一致；保留已有17个tracked改动，未stage/commit/push。

[逐视图指标](tests/diagnostics/s00_xyz_only_20261026/evidence/analysis/quality_per_view.csv) · [配对统计](tests/diagnostics/s00_xyz_only_20261026/evidence/analysis/paired_statistics.csv) · [实际预算与集合重合](tests/diagnostics/s00_xyz_only_20261026/evidence/analysis/selected_set_overlap.csv)
