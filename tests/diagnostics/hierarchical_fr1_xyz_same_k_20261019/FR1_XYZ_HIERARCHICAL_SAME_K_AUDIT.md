# fr1_xyz：A/B/C/F3 Frozen Same-K 审计

## 结论

**Current C 相对 Raw 的低 budget 收益复现；F3 的全档优势没有跨场景复现。** F3 在1%比 C 高 **0.250820 dB**，但5/10/15%分别低 **0.110466 / 0.124511 / 0.059770 dB**，四档平均 **−0.010982 dB**，未通过沿用的 Combined gate。F3 相对 Fixed h8 的四档平均差为 **−0.063646 dB（BELOW）**。

F3 四档 LPIPS 均优于 C，但5/10/15%的 PSNR、SSIM、photo 同时回退。因此不能仅凭重复度下降或 LPIPS 改善判定 F3 更好。建议保留当前 C，F3 继续作为诊断候选；本轮没有修改任何算法或生产开关。

## 1. 数据、checkpoint 与冻结协议

- 数据集：`/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz`。GT直接来自用户指定目录。
- checkpoint：已有 `full/attempt_1/model/9900_shutdown/ply`，按最终 shutdown 选择，不读质量挑 checkpoint。
- Master共 **23733** rows，valid **23733**，observed及candidate均 **23643**；candidate=`valid AND observed`，effective maturity=1。
- 实际 Bank **18** views；全部归档KF共72，其余 **54** 个为 **fixed non-Bank archived evaluation views**。这些视角参与过原SLAM，不称为 unseen/test-only/independent novel views。
- 该场景原先没有本协议的贡献fixture，因此本轮仅用已归档实际Bank和原renderer导出一次完整 frame contribution并冻结；A/B/C/F3共用此矩阵及其mean Raw SI，不更新Bank。
- 归档 `view_bank_orb.json` 的 `scene` 文本为过期的 `fr3_long_office_household`；本轮不使用该标签。Bank只读取实际keyframe IDs，所有ID属于同checkpoint的72个相机，图像从明确的fr1_xyz路径读取并保存SHA。
- 相机/FOV/GT转换函数逐字复用上一轮；renderer、TF consumer、PSNR/SSIM/LPIPS及production photo实现也逐字复用。TF参数和AABB加载自该checkpoint，所有arm一致。
- production photo=`0.8*L1 + 0.2*(1-production fast_ssim)`；单独报告的SSIM沿用上一轮TorchMetrics实现，两者不混同。

| 输入 | 路径 | SHA256 |
|---|---|---|
| master | /home/prmi/zwj/mycartgs/resultsall/batch_20261018_tum_aligned/jobs/tum/rgbd/fr1_xyz/full/attempt_1/model/9900_shutdown/ply/point_cloud/iteration_9900/point_cloud.ply | a8abd19e63cc3079d4efd07947856e7d8e1f5e959e17142b62aa659b0faebb10 |
| tf | /home/prmi/zwj/mycartgs/resultsall/batch_20261018_tum_aligned/jobs/tum/rgbd/fr1_xyz/full/attempt_1/model/9900_shutdown/ply/transform_field.pt | bba18765352bced0e8359a5bae8501e7293cb4f907e7760a01e12b5b2f4f0384 |
| config | /home/prmi/zwj/mycartgs/resultsall/batch_20261018_tum_aligned/jobs/tum/rgbd/fr1_xyz/full/attempt_1/model/9900_shutdown/ply/selector_config.yaml | 2d0d0516bd299fff14b946b65c901e14e2970e20685229febd4c7bcbbb239690 |
| cameras | /home/prmi/zwj/hierarchical-fr1-xyz-same-k-20261019/inputs/fr1_xyz/cameras.pt | 684882bed264ddfc1dfdaca74eab0168f9b7d255e120e3d18aa2d3cdfb5fc074 |
| contributions.pt | /home/prmi/zwj/hierarchical-fr1-xyz-same-k-20261019/contributions.pt | 6949a04885d9eccebc16f89722758610b1ef5706c60e716bfc6f5d30841df79c |
| ranking_inputs.npz | /home/prmi/zwj/hierarchical-fr1-xyz-same-k-20261019/ranking_inputs.npz | 86a06e6b9d7f7cdce27a07ff30c8ce46e645a4ad9ebe725d5760c43005029602 |
| hierarchy_input.bin | /home/prmi/zwj/hierarchical-fr1-xyz-same-k-20261019/hierarchy_input.bin | 9518c1b79e2bd00f499fcbb81244c7a82be9cc10c3f0610ed745695a99a67fac |
| preregistration.json | /home/prmi/zwj/hierarchical-fr1-xyz-same-k-20261019/preregistration.json | d52a513b2f74bfe5f6c5366a8d4fdf880c6b2ad3c16ef0d9eb89a098f5a5803b |

全部原始相机图像、checkpoint JSON、数学源码、历史runtime等SHA见 `frozen_sources.json`；生成相机archive的SHA见 `camera_checks.json`，结束时全部重新检查。

## 2. Arms与固定参数

| Arm | 定义 |
|---|---|
| A | Raw SI全局降序，tie source row |
| B | Fixed h8 geometry-only完整local marginal greedy + predecessor-preserving merge |
| C | SparseHash当前3rep Delta + independent residual，全局score降序 |
| F3 | 同一冻结tree，ancestor-conditioned progressive 3新rep/node，完成后解锁child；final leaf继续conditional marginal tail |

target occupied roots=2048；origin=(0,0,0)；h搜索规则原样；3 representatives/node；Q<0.80 split；max L3；scalar-radius geometry-only κ；maturity=1。没有参数搜索、boundary correction、view/covariance、mass dedup、训练或online运行。

F3完整ranking为实际active-frontier heap emit顺序，未按priority重排。tie为priority降序、root key+octree path字典序、source row升序；已emit ancestor row不重复出现。其ordinal score只表达rank，不是守恒importance mass。

## 3. Correctness和same-K

| ratio | K |
|---|---|
| 1% | 236 |
| 5% | 1182 |
| 10% | 2364 |
| 15% | 3546 |

- A/B/C/F3均为同一candidate population的完整permutation，四档均为同一ranking的nested prefixes。
- B完整重算两次exact一致，并逐cell检查local predecessor order。
- C用上一轮原C++静态库构建两次：score、membership、h、tree、rep IDs/Delta/Q exact一致；对全部node独立重算first3 greedy及Q通过。
- F3完整构建两次，order/gain/emit/unlock trace exact一致。10项synthetic和耗尽child测试通过。
- 独立重放 **23643** 次argmax/gain、ancestor coverage及全局frontier/unlock，最大gain差 **0**；无重复、丢失row。
- C质量信号mass相对误差 **0.000e+00**；F3仅评价ranking，不宣称全局物理mass守恒。
- 本轮是新scene，没有fr1_xyz既有A/B/C同协议质量可做历史exact对照；复用的数学、worker及metric文件SHA与上一轮一致，本场景内部做exact repeat和独立核对。

## 4. Sparse hierarchy结构

| 统计 | 值 |
|---|---|
| occupied_roots | 2048 |
| h_root | 0.07246712689026927 |
| d50 | 0.007198099916696466 |
| h | 0.057584799333571725 |
| final_leaves | 4889 |
| final_representatives | 10889 |
| weighted_leaf_Q | 0.9403273217713178 |

weighted Q只解释结构，不参与质量判定。完整节点和membership见 `frozen_tree_nodes.json`、`frozen_tree_membership.npz`。

## 5. 四项真实render质量

### PSNR ↑

| ratio | K | A Raw | B FixedMarginal | C Current | F3 Progressive+Tail |
|---|---|---|---|---|---|
| 1% | 236 | 13.020670 | 14.584813 | 14.206011 | 14.456831 |
| 5% | 1182 | 17.814622 | 18.777373 | 18.822068 | 18.711601 |
| 10% | 2364 | 19.812102 | 20.261198 | 20.301973 | 20.177463 |
| 15% | 3546 | 20.890139 | 20.817547 | 20.900224 | 20.840454 |

### SSIM ↑

| ratio | K | A Raw | B FixedMarginal | C Current | F3 Progressive+Tail |
|---|---|---|---|---|---|
| 1% | 236 | 0.569450 | 0.582761 | 0.584646 | 0.584532 |
| 5% | 1182 | 0.656955 | 0.670624 | 0.671249 | 0.669939 |
| 10% | 2364 | 0.705598 | 0.706185 | 0.707379 | 0.706224 |
| 15% | 3546 | 0.725542 | 0.723764 | 0.724697 | 0.723950 |

### LPIPS ↓

| ratio | K | A Raw | B FixedMarginal | C Current | F3 Progressive+Tail |
|---|---|---|---|---|---|
| 1% | 236 | 0.704868 | 0.663592 | 0.669723 | 0.667035 |
| 5% | 1182 | 0.545347 | 0.490062 | 0.491978 | 0.489956 |
| 10% | 2364 | 0.442491 | 0.402928 | 0.405309 | 0.404475 |
| 15% | 3546 | 0.391351 | 0.361789 | 0.362828 | 0.360206 |

### Production photo ↓

| ratio | K | A Raw | B FixedMarginal | C Current | F3 Progressive+Tail |
|---|---|---|---|---|---|
| 1% | 236 | 0.197714 | 0.178564 | 0.179248 | 0.177929 |
| 5% | 1182 | 0.126302 | 0.119236 | 0.118568 | 0.119212 |
| 10% | 2364 | 0.103354 | 0.102742 | 0.102281 | 0.102993 |
| 15% | 3546 | 0.095069 | 0.096261 | 0.095808 | 0.096213 |

共4 arms ×4 ratios ×54views = **864次真实render**。所有arm除selected rows外均使用相同Master/TF/camera/renderer/GT。

### C − Raw

| ratio | ΔPSNR ↑ | ΔSSIM ↑ | ΔLPIPS ↓ | Δphoto ↓ |
|---|---|---|---|---|
| 1% | +1.185341 | +0.015196 | -0.035145 | -0.018466 |
| 5% | +1.007445 | +0.014294 | -0.053369 | -0.007735 |
| 10% | +0.489872 | +0.001781 | -0.037182 | -0.001073 |
| 15% | +0.010085 | -0.000844 | -0.028522 | +0.000739 |

### F3 − C

| ratio | ΔPSNR ↑ | ΔSSIM ↑ | ΔLPIPS ↓ | Δphoto ↓ |
|---|---|---|---|---|
| 1% | +0.250820 | -0.000114 | -0.002688 | -0.001319 |
| 5% | -0.110466 | -0.001310 | -0.002022 | +0.000645 |
| 10% | -0.124511 | -0.001155 | -0.000833 | +0.000712 |
| 15% | -0.059770 | -0.000747 | -0.002622 | +0.000405 |

### F3 − FixedMarginal

| ratio | ΔPSNR ↑ | ΔSSIM ↑ | ΔLPIPS ↓ | Δphoto ↓ |
|---|---|---|---|---|
| 1% | -0.127982 | +0.001771 | +0.003443 | -0.000634 |
| 5% | -0.065772 | -0.000685 | -0.000105 | -0.000024 |
| 10% | -0.083735 | +0.000039 | +0.001547 | +0.000251 |
| 15% | +0.022907 | +0.000186 | -0.001583 | -0.000048 |

## 6. Paired per-view结果

下表为F3−C，四档各54个配对视角；positive/improved fraction按指标方向区分，LPIPS/photo负差才为改善。全部比较和逐视角差值保存为CSV。

| ratio | metric | mean | median | p10 | p90 | improved views |
|---|---|---|---|---|---|---|
| 1% | psnr | +0.250820 | +0.240795 | +0.095179 | +0.445554 | 94.44% |
| 1% | ssim | -0.000114 | +0.000055 | -0.001834 | +0.001664 | 50.00% |
| 1% | lpips | -0.002688 | -0.003118 | -0.006020 | +0.001188 | 83.33% |
| 1% | photo | -0.001319 | -0.001514 | -0.002882 | +0.000071 | 88.89% |
| 5% | psnr | -0.110466 | -0.094290 | -0.203456 | -0.040878 | 3.70% |
| 5% | ssim | -0.001310 | -0.001762 | -0.003463 | +0.000565 | 22.22% |
| 5% | lpips | -0.002022 | -0.002710 | -0.005392 | +0.002265 | 68.52% |
| 5% | photo | +0.000645 | +0.000681 | -0.000055 | +0.001389 | 11.11% |
| 10% | psnr | -0.124511 | -0.123201 | -0.196973 | -0.047473 | 3.70% |
| 10% | ssim | -0.001155 | -0.001293 | -0.002499 | +0.000236 | 18.52% |
| 10% | lpips | -0.000833 | -0.000967 | -0.002947 | +0.001567 | 64.81% |
| 10% | photo | +0.000712 | +0.000694 | +0.000177 | +0.001238 | 7.41% |
| 15% | psnr | -0.059770 | -0.063790 | -0.122147 | +0.006368 | 14.81% |
| 15% | ssim | -0.000747 | -0.000806 | -0.001978 | +0.000433 | 20.37% |
| 15% | lpips | -0.002622 | -0.002851 | -0.004525 | -0.000110 | 88.89% |
| 15% | photo | +0.000405 | +0.000446 | -0.000057 | +0.000869 | 12.96% |

F3−C的PSNR改善视角比例依次为 **94.44% / 3.70% / 3.70% / 14.81%**；高档回退具有配对一致性，并非少数异常视角拉低均值。

## 7. 统一Fixed-h8 geometry重复度

与上一轮完全相同：只比较同一固定h8 cell中的selected rows；每row取最大off-diagonal κ，singleton记0，另统计unordered高相似pair。它是统一诊断，不是全局无边界重叠估计。

| ratio | arm | mean | median | p90 | κ>.8 | κ>.9 | κ>.95 |
|---|---|---|---|---|---|---|---|
| 1% | A Raw | 0.374380 | 0.148703 | 0.970094 | 54 | 36 | 21 |
| 5% | A Raw | 0.474609 | 0.504622 | 0.976130 | 348 | 210 | 133 |
| 10% | A Raw | 0.505341 | 0.610984 | 0.977895 | 844 | 503 | 308 |
| 15% | A Raw | 0.516748 | 0.621764 | 0.981327 | 1360 | 816 | 490 |
| 1% | B FixedMarginal | 0.024733 | 0.000000 | 0.003775 | 0 | 0 | 0 |
| 5% | B FixedMarginal | 0.158098 | 0.000001 | 0.583116 | 22 | 5 | 2 |
| 10% | B FixedMarginal | 0.217686 | 0.021227 | 0.710368 | 92 | 28 | 9 |
| 15% | B FixedMarginal | 0.246609 | 0.066893 | 0.749659 | 171 | 49 | 17 |
| 1% | C Current | 0.134665 | 0.000000 | 0.630343 | 10 | 4 | 1 |
| 5% | C Current | 0.226700 | 0.006920 | 0.737619 | 49 | 18 | 10 |
| 10% | C Current | 0.280846 | 0.091284 | 0.820299 | 169 | 64 | 31 |
| 15% | C Current | 0.296314 | 0.140379 | 0.822272 | 267 | 104 | 51 |
| 1% | F3 Progressive+Tail | 0.126496 | 0.000000 | 0.593684 | 9 | 4 | 1 |
| 5% | F3 Progressive+Tail | 0.214508 | 0.002408 | 0.716178 | 48 | 19 | 10 |
| 10% | F3 Progressive+Tail | 0.265370 | 0.073130 | 0.803401 | 151 | 55 | 23 |
| 15% | F3 Progressive+Tail | 0.279021 | 0.110627 | 0.805505 | 234 | 85 | 36 |

F3的mean redundancy四档均低于C，但仍明显高于B。10/15% κ>.9 pairs由C的64/104降至F3的55/85，仍不能阻止PSNR/SSIM/photo回退。重复度下降不是质量提升的充分条件。

## 8. Budget composition、空间覆盖与overlap

| ratio | arm | C-final-rep fraction | internal reps | final reps | tail | emit L1/L2/L3 |
|---|---|---|---|---|---|---|
| 1% | C Current | 100.00% |  |  |  | // |
| 5% | C Current | 97.72% |  |  |  | // |
| 10% | C Current | 92.81% |  |  |  | // |
| 15% | C Current | 89.31% |  |  |  | // |
| 1% | F3 Progressive+Tail | 99.15% | 50 | 186 | 0 | 236/0/0 |
| 5% | F3 Progressive+Tail | 95.18% | 426 | 728 | 28 | 1060/120/2 |
| 10% | F3 Progressive+Tail | 90.36% | 854 | 1372 | 138 | 1781/520/63 |
| 15% | F3 Progressive+Tail | 87.00% | 1225 | 2033 | 288 | 2316/1010/220 |

| ratio | arm | hit roots | hit leaves | hit-root Raw mass | hit-leaf Raw mass | dense-root budget | important-root budget |
|---|---|---|---|---|---|---|---|
| 1% | C Current | 206 | 215 | 61.79% | 55.63% | 24.58% | 80.51% |
| 5% | C Current | 665 | 910 | 91.92% | 85.15% | 36.55% | 53.21% |
| 10% | C Current | 978 | 1559 | 97.08% | 92.95% | 43.10% | 48.48% |
| 15% | C Current | 1181 | 2131 | 98.63% | 96.21% | 47.21% | 45.69% |
| 1% | F3 Progressive+Tail | 210 | 216 | 62.55% | 55.50% | 24.58% | 80.08% |
| 5% | F3 Progressive+Tail | 677 | 890 | 92.41% | 84.10% | 35.53% | 51.78% |
| 10% | F3 Progressive+Tail | 998 | 1538 | 97.30% | 92.37% | 41.96% | 47.50% |
| 15% | F3 Progressive+Tail | 1193 | 2103 | 98.71% | 95.92% | 46.76% | 45.26% |

| ratio | comparison | Jaccard | intersection | added/removed |
|---|---|---|---|---|
| 1% | B_FixedMarginal_vs_F3_ProgressiveMarginal | 0.552632 | 168 | 68/68 |
| 1% | C_Hierarchical_vs_F3_ProgressiveMarginal | 0.888000 | 222 | 14/14 |
| 5% | B_FixedMarginal_vs_F3_ProgressiveMarginal | 0.676596 | 954 | 228/228 |
| 5% | C_Hierarchical_vs_F3_ProgressiveMarginal | 0.859953 | 1093 | 89/89 |
| 10% | B_FixedMarginal_vs_F3_ProgressiveMarginal | 0.756315 | 2036 | 328/328 |
| 10% | C_Hierarchical_vs_F3_ProgressiveMarginal | 0.876190 | 2208 | 156/156 |
| 15% | B_FixedMarginal_vs_F3_ProgressiveMarginal | 0.790005 | 3130 | 416/416 |
| 15% | C_Hierarchical_vs_F3_ProgressiveMarginal | 0.883665 | 3327 | 219/219 |

1% F3全部236次emit来自L1，tail=0，因此该档F3−C提升不能归因于marginal tail，直接支持coarse scheduling在极低budget的作用。高档同时改变depth scheduling与tail，本轮只跑F3组合，不能唯一归因于其中某一个机制；不自动补跑F1/F2或调参数。

## 9. 预注册gate与跨场景解释

| 检查 | 结果 |
|---|---|
| C相对Raw的PSNR主gate | PASS：四档均不低，1/5%平均显著改善 |
| F3四档mean ΔPSNR≥0 | FAIL：-0.010982 dB |
| F3 1%优于C | PASS：+0.250820 dB |
| F3 10/15% mean redundancy<C | PASS |
| Combined supported | NO |
| Strong improvement | NO |
| F3相对B的PSNR水平 | BELOW：四档mean −0.063646 dB |

fr1_room上一轮F3−C四档平均 +0.059593 dB，本轮fr1_xyz为 −0.010982 dB。fr1_room的COMBINED SUPPORTED不能据此推广为跨场景稳定优于C；本轮也不能把F3低budget收益写成online或production GO。

## 10. Runtime、边界与交付

| 阶段 | 单次记录ms |
|---|---|
| B | 601.195 |
| C | 7057.882 |
| F3 | 972.081 |

CPU NumPy reference与GPU同步C构建的时间口径不同，不作速度比较；runtime不参与任何gate。

最初独立wrapper的CMake缺少CUDA架构设置，按项目已有CUDA11.8/GCC9/sm86设置修复；分析脚本上传前pipeline结束一次，随后只运行分析。失败日志均保留，无算法、输入或质量选优改动。

局限：单个frozen final map、归档训练时视角、实际Bank较少；没有online训练或ATE比较；所有结论限定same-K reconstruction ranking。

输出包括 `summary.json`、`quality_summary.csv`、`quality_per_view.csv`、`quality_deltas.csv`、`paired_statistics.csv`、`paired_per_view_deltas.csv`、`ranking_overlap.csv`、逐row变化、`selected_redundancy.csv`、`progressive_composition.csv`、`spatial_budget_coverage.csv`、完整emit/unlock trace、ranking数组及SHA/check日志。

**最终结论：Current C 的低budget收益在fr1_xyz复现；F3 Combined在本场景 NOT SUPPORTED，暂保留 Current C，不作全局替换。** 未commit，未push，未改production。
