# 原Baseline新增全局XYZ受控实验

## 结论

两seed七档non-Bank same-K平均变化-0.3767 dB；seed3100/3101分别为-0.2661/-0.4873 dB。
两seed是同一冻结地图上的初始化与训练顺序重复，不是两个独立SLAM场景。此次不依据non-Bank结果追加调参或替换生产配置。

**不建议将全局XYZ加入当前Baseline。** 两seed在七档ratio的均值均下降；Bank训练视图平均下降0.354 dB，non-Bank下降0.377 dB，因此此次退化不只发生于未见视图。原Cached实际数量评价同样下降0.296 dB。

本次排除了上一组原属性初始化不同的混淆，支持“当前固定训练预算及原始XYZ表示下，新增XYZ没有质量收益”。但仍不能断言世界坐标在所有表达或训练方式下都无用：新增连接改变优化轨迹，Encoder和Head同步接入，且Selector集合随训练变化。本轮未隔离这些细分因素，也未测试坐标归一化或更长训练。

## 结构与严格初始化对齐

| 项目 | 原Baseline | Baseline+全局XYZ |
| --- | --- | --- |
| 原始输入 | Offset3 + Scale3 + SH DC3 + Opacity1 + log(ratio) | 原11维全部保留，再追加世界XYZ3 |
| Encoder | 11→32→ReLU | 14→32→ReLU |
| 补偿输入 | 原始11 + 每Anchor/ratio latent16 + h32 = 59 | 原始14 + latent16 + h32 = 62 |
| 四Head | 各59→32→ReLU→32→ReLU→输出 | 各62→32→ReLU→32→ReLU→输出 |
| 输出 | XYZ3 / Scale3 / Rotation4 / Opacity1+SH DC3 | 完全相同 |
| 参数总量 | 146096 | 146576，增加480 |

与上一组沿逐步加回链路构建的浅DC不同：本次所有原属性列、bias、latent/h32投影和后续层直接复制原Baseline同seed初始化，只有新增XYZ列置零。step0原11维输入和旧权重exact相同；概率和零补偿exact一致，隐藏函数及临时非零输出投影做浮点容忍检查。真实photo反传检查新增连接随后获得有效梯度。

世界XYZ直接采用Master真实坐标，不额外归一化；局部Offset保留，Scale为activated scale，Opacity为sigmoid opacity，DC为原SH系数。Encoder和四个Head都添加XYZ，无法把影响单独归因到其中一条路径。

## 质量与实际预算

| 方案 | PSNR↑ | SSIM↑ | LPIPS↓ | photo↓ |
| --- | --- | --- | --- | --- |
| original_baseline | 17.24435 | 0.60185 | 0.60022 | 0.15107 |
| baseline_xyz | 16.86764 | 0.59068 | 0.60538 | 0.15697 |

| ratio | same-K | seed3100 ΔPSNR | seed3101 ΔPSNR | 均值ΔPSNR |
| --- | --- | --- | --- | --- |
| 1% | 68 | -0.1377 | -0.7369 | -0.4373 |
| 2.5% | 170 | -0.0307 | -0.1883 | -0.1095 |
| 5% | 341 | -0.3412 | -0.5305 | -0.4358 |
| 7.5% | 512 | -0.1228 | -0.2749 | -0.1989 |
| 10% | 683 | -0.4193 | -0.2794 | -0.3494 |
| 12.5% | 854 | -0.4926 | -0.7368 | -0.6147 |
| 15% | 1025 | -0.3188 | -0.6643 | -0.4915 |

| ratio组 | seed3100 ΔPSNR | seed3101 ΔPSNR | 均值ΔPSNR |
| --- | --- | --- | --- |
| all | -0.2661 | -0.4873 | -0.3767 |
| train | -0.3042 | -0.5528 | -0.4285 |
| unseen | -0.2153 | -0.4000 | -0.3077 |

| 评价 | same-K ΔPSNR | 实际Cached ΔPSNR |
| --- | --- | --- |
| bank_train | -0.3537 | -0.2807 |
| validation | -0.3394 | -0.2894 |
| test | -0.3767 | -0.2959 |

non-Bank逐视图差值median/p10/p90：-0.2873/-1.0837/+0.2327 dB；改善比例27.8%。完整四指标、visible与effective_proxy数量、cached实际K见CSV；effective_proxy仅为半径>0且补偿后opacity≥1/255的代理，并非逐点真实像素贡献。

## 实验边界与公平性

- 同源fr1_xyz冻结N=6838/M=2066，Master/拓扑/ID/Anchor归属、中心与尺度/位姿/eligible/cached uniform冻结。scene_manifest与原Baseline逐项比较。
- 54个Bank训练视图320×240，13个内部Bank留出和45个历史non-Bank视图640×480；评价的Bank训练指标只取54个实际训练视图。
- 新增模型两seed各12000真实LoD步，每ratio3000次曝光，复用同seed原Baseline最终12000步锁定参照。fresh fused Adam lr=.001 eps=1e-15，原.8L1+.2(1-fast_ssim)+.01预算Loss，无新增Loss或正则。
- 训练1/5/10/15%，未见2.5/7.5/12.5%使用原raw-ratio latent线性插值并输入真实logratio；不重新拟合。
- 同seed Bank顺序与ratio轮转相同，全部正式结果取预定最终12000步；4k/8k仅作收敛诊断。没有按non-Bank选择checkpoint、学习率或训练步数。
- Cached Bernoulli训练；same-K按logit(p)-logit(cached uniform)排序而非纯概率。原Cached实际数量另外评价。
- Encoder/Selector/Compensation及当前ratio补偿latent参与训练，选中身份可变，不是固定选择集合的纯Compensation因果。
- Adam实际时钟按空选择次数核验，未伪造零梯度step；两seed one-map有限预算，不能认定表达能力上界或跨场景最优。历史non-Bank多次用于研究，不是全新盲测。
- 服务器存在其他GPU任务，耗时只记录，不作无干扰速度比较。本次是冻结地图实验，不是在线SLAM质量证据。
- 生产网络/默认配置保持原状；本轮不增加其他输入、网络深度或维度，不stage、commit或push。

## 原始证据

[逐视图四指标](tests/diagnostics/s00_baseline_globalxyz_20261026/evidence/analysis/quality_per_view.csv) · [配对统计](tests/diagnostics/s00_baseline_globalxyz_20261026/evidence/analysis/paired_statistics.csv) · [预算与实际Adam时钟](tests/diagnostics/s00_baseline_globalxyz_20261026/evidence/analysis/fairness_checks.csv) · [选择重合率](tests/diagnostics/s00_baseline_globalxyz_20261026/evidence/analysis/selected_set_overlap.csv)

![质量曲线](tests/diagnostics/s00_baseline_globalxyz_20261026/evidence/analysis/quality_curves.png)

## 内部Bank留出收敛

| 方案 | seed | 4000步 | 8000步 | 12000步 |
| --- | --- | --- | --- | --- |
| original_baseline | 3100 | 16.746 | 17.351 | 17.496 |
| original_baseline | 3101 | 16.721 | 16.887 | 17.307 |
| baseline_xyz | 3100 | 16.756 | 17.301 | 17.275 |
| baseline_xyz | 3101 | 16.686 | 16.765 | 16.850 |

## 交付核验

两新增模型和两锁定参照均完成12000步、每ratio3000次曝光。step0及新增XYZ梯度检查通过；原始逐视图四指标finite，实际Adam时钟与空选择记录匹配。源码与下载归档SHA一致，Git HEAD、tracked diff及staged diff与实验前完全相同；未stage、commit或push。

实际更新记录：[{"kind": "BaselinePlusGlobalXYZSplit4", "seed": 3100, "empty_steps": {"0.01": 0, "0.05": 0, "0.1": 0, "0.15": 0}, "compensation_updates": 12000, "parameters": 146576}, {"kind": "BaselinePlusGlobalXYZSplit4", "seed": 3101, "empty_steps": {"0.01": 0, "0.05": 0, "0.1": 0, "0.15": 0}, "compensation_updates": 12000, "parameters": 146576}, {"kind": "original_baseline", "seed": 3100, "empty_steps": {"0.01": 0, "0.05": 0, "0.1": 0, "0.15": 0}, "compensation_updates": 12000, "parameters": 146096}, {"kind": "original_baseline", "seed": 3101, "empty_steps": {"0.01": 0, "0.05": 0, "0.1": 0, "0.15": 0}, "compensation_updates": 12000, "parameters": 146096}]

协议文字勘误：归档protocol的source描述沿用“四个参照”旧文字；实际目录、分析和本报告均为两个原Baseline参照（每seed一个）。仅文字计数错误，不影响预定训练、模型或统计。见实验目录protocol_metadata_erratum.json。
