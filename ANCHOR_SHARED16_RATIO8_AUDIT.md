# 16D共享＋8D ratio Anchor Feature质量实验

结果：16＋8＋log相对8＋8＋log的same-K non-Bank训练档差 **+0.1156 dB**、插值档 **+0.1825 dB**；相对原16D共享分别 **+0.2155 / -0.0964 dB**；相对同批R3分别 **-0.4899 / -0.2856 dB**。两seed与每档明细见下表，不能只按均值判定稳定性。

## 实际改动

每Anchor共享16D，训练ratio1/5/10/15%各存8D选择特征及14D补偿潜在特征，总计104个标量。35D输入：`共享16+ratio8+Offset3+Scale3+SH DC3+Opacity1+log(ratio)1`，共享Encoder35→32 ReLU；选择头h32→32 ReLU→2。选中K行h32加补偿潜在14D，通过46→32 ReLU→14解码原XYZ/Scale/Rotation/Opacity/SH DC补偿。8D与14D在未见ratio上均按raw ratio线性插值，范围外端点clamp；log(ratio)仍使用实际查询值。

保留8＋8版的原26D编码器列映射，将新增共享8D插入中间；其特征从零开始，新增权重采用普通Linear初始化。后续层、bias及随机时钟保留。再按原方式追加零权重log列。按构造16＋8与8＋8初始函数相同（浮点精度内），补偿初始均零；按用户要求未额外跑parity/单元套件，不把构造说明写成实测通过。

新增配置`Selector.anchor_ratio_shared_dim: 16`（旧默认8），配合`anchor_ratio_features: 1`、`anchor_ratio_scalar_input: "log"`和`compensation_method: "anchor_ratio_latent14"`。新checkpoint布局v9；C++/Python兼容34/35D输入，保留旧8＋8、16共享及R3加载路径。ratio表始终8D，不随共享宽度变为16D。

Anchor增长、同层max继承、细层零初始化、prune、成员重排、闭环加权迁移与Adam行迁移复用现有动态宽度路径。只改变特征参数化，没有增加Loss、改变采样器、调度或TF。

## 同批在线对照

| 标签 | 结构 | 补偿 |
|---|---|---|
| shared16 | 共享Anchor16＋log(ratio) | 额外ratio潜在14D，MLP解码 |
| shared8_ratio8 | 共享8＋ratio8＋log(ratio) | 同上 |
| shared16_ratio8 | 本轮共享16＋ratio8＋log(ratio) | 同上 |
| r3_independent14 | 原共享Anchor16＋log(ratio) | Gaussian/ratio直接14D系数 |

fr1_xyz、seed2026/2027，共八次真实SLAM，从头训练，同一源码/二进制；R3为同批参考，不沿用旧报告质量直接相减。Variable Anchor、当前Gaussian/Anchor交替拓扑、`anchor_only_pruning=0`、原joint、Cached Bernoulli、ratio轮转、原photo及预算Loss、TF/SI/GI关闭均保留。自然停止，不强制10000步，不按non-Bank选超参数或checkpoint。

| 运行 | Mapping | Gaussian | Anchor | 输入维数 | Full次数 | LoD次数 | 补偿次数 | 墙钟秒 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shared16_seed2026 | 6400 | 7732 | 2305 | 27 | 1280 | 5120 | 2720 | 43.43 |
| shared8_ratio8_seed2026 | 5904 | 7015 | 2432 | 27 | 1180 | 4724 | 2324 | 42.78 |
| shared16_ratio8_seed2026 | 5914 | 7388 | 2550 | 35 | 1182 | 4732 | 2332 | 42.68 |
| r3_independent14_seed2026 | 6400 | 6838 | 2066 | 27 | 1280 | 5120 | 2720 | 42.71 |
| shared16_seed2027 | 6400 | 8003 | 2303 | 27 | 1280 | 5120 | 2720 | 43.56 |
| shared8_ratio8_seed2027 | 5900 | 6975 | 2383 | 27 | 1180 | 4720 | 2320 | 42.82 |
| shared16_ratio8_seed2027 | 5908 | 7309 | 2520 | 35 | 1181 | 4727 | 2327 | 42.75 |
| r3_independent14_seed2027 | 6400 | 7479 | 2325 | 27 | 1280 | 5120 | 2720 | 42.87 |

固定64个RGB评价帧，剔除八次Bank并集后共 **45** 个non-Bank。各地图使用自己的最终估计位姿。训练ratio1/5/10/15%，未见2.5/7.5/12.5%；Full为独立原Master评价。

Cached生产选择与same-K诊断分别保存。same-K按`logit(p)−logit(cached_uniform)`排序，所有八个模型使用共同K与相同视图。它控制点数，不能消除在线地图、位姿与Adam次数差异。

## Same-K non-Bank PSNR

| ratio | K | 原16共享 | 8＋8＋log | 16＋8＋log | R3 |
|---|---:|---:|---:|---:|---:|
| 1% | 68 | 14.7476 | 14.8734 | 15.1866 | 16.1083 |
| 2.5% | 170 | 15.5719 | 14.9503 | 15.1829 | 15.5986 |
| 5% | 341 | 17.5926 | 17.6839 | 17.5110 | 18.4358 |
| 7.5% | 512 | 18.1990 | 17.9339 | 18.1450 | 18.4591 |
| 10% | 683 | 18.5453 | 18.5889 | 18.6921 | 18.9395 |
| 12.5% | 854 | 18.7036 | 18.7536 | 18.8575 | 18.9845 |
| 15% | 1025 | 18.7546 | 18.8935 | 19.1123 | 18.9780 |

以下均为16＋8减参考；PSNR/SSIM越高越好，LPIPS/photo越低越好。

| 参考 | 范围 | Cached ΔPSNR | same-K ΔPSNR | same-K ΔPSNR seed2026/2027 | ΔSSIM | ΔLPIPS | Δphoto |
|---|---|---:|---:|---:|---:|---:|---:|
| shared16 | train | +0.1106 | +0.2155 | +0.1761/+0.2549 | +0.00552 | -0.00905 | -0.00337 |
| shared16 | unseen | -0.2116 | -0.0964 | +0.0014/-0.1943 | -0.00118 | +0.00187 | +0.00175 |
| shared16 | all | -0.0275 | +0.0818 | +0.1012/+0.0624 | +0.00265 | -0.00437 | -0.00118 |
| shared8_ratio8 | train | +0.2116 | +0.1156 | +0.1730/+0.0581 | -0.00043 | -0.00428 | -0.00151 |
| shared8_ratio8 | unseen | +0.1736 | +0.1825 | +0.1013/+0.2637 | +0.00051 | -0.00566 | -0.00248 |
| shared8_ratio8 | all | +0.1953 | +0.1443 | +0.1423/+0.1463 | -0.00003 | -0.00487 | -0.00192 |
| r3_independent14 | train | -0.4692 | -0.4899 | -0.2505/-0.7293 | -0.01053 | +0.00102 | +0.00756 |
| r3_independent14 | unseen | -0.3405 | -0.2856 | +0.1621/-0.7333 | -0.00647 | -0.00175 | +0.00533 |
| r3_independent14 | all | -0.4140 | -0.4024 | -0.0737/-0.7310 | -0.00879 | -0.00017 | +0.00660 |

Full non-Bank PSNR差：相对16共享 **-0.0709 dB**、8＋8 **+0.2387 dB**、R3 **+0.3016 dB**。逐视图mean/median/p10/p90与改善比例保存于paired_statistics.csv。

归因限制：相对8＋8，Full本身也改善，因此不能将LoD增益全部解释为额外共享8维提高了Selector表达能力。相对R3，Full均值更高而LoD仍低，差距也不能简单归为Master总体质量不足；具体空间分布、优化次数和稀疏路径尚未完全隔离。2.5%平均预算仍不足，增加共享维度未解决插值预算泛化。

## 2.5%插值预算

| 方法 | 平均选择概率 | 实际Cached比例均值 | 期望预算相对误差 |
|---|---:|---:|---:|
| shared16 | 2.003% | 2.004% | -19.9% |
| shared8_ratio8 | 1.762% | 1.765% | -29.5% |
| r3_independent14 | 2.391% | 2.274% | -4.3% |
| shared16_ratio8 | 1.711% | 1.707% | -31.6% |

## 成本和局限

16＋8比8＋8每Anchor多8个FP32标量，Encoder多256个权重；参数、梯度及Adam一二阶状态合计增加约128×Anchor数＋4096 bytes（估算，不是峰值显存）。相对纯16共享，则多四档8D特征表。补偿头宽度和14D表数量不变。所有ratio-feature模型都使用现有LibTorch输入路径；未新增CUDA融合，服务器其他GPU作业使墙钟不能当独占benchmark。

本轮观察的是在线整体效果，不是冻结同一地图的纯容量结论；两seed仍属于一个场景。初始化按构造匹配不保证训练轨迹一致，same-K也不保证相同空间覆盖。未运行单元/回归套件，因此不能宣称所有几何迁移及历史checkpoint边界均已实测。未评价1%～15%以外质量。

服务器直接同步常用项目并复用build-performance-20261023-121515构建tum_rgbd，无隔离构建、无stage/commit/push。默认仍R3，未自动替换。

本轮只重建并运行tum_rgbd；其他应用入口未重建。因Mapper头文件布局有新增字段，使用其他入口前应基于当前源码重编，不应将本轮构建记录视为全部入口已更新。

## 证据

本地：`tests/diagnostics/anchor_shared16_ratio8_20261026/`。汇总为aggregate_comparison.csv、budget_interpolation.csv、audit_summary.json；evidence/validation含逐视图四指标、共同K、选择统计、真实结构、特征分布、训练配置、Full/LoD次数及原始日志。服务器完整模型：`/home/prmi/zwj/anchor-shared16-ratio8-20261026/`。
