# 全局XYZ逐步加回Scale与Opacity的质量对照

**结论：两seed均优于XYZ-only。两seed仍均低于原Baseline。**

七档non-Bank same-K相对XYZ-only平均 **+0.558 dB**，相对原Baseline **-0.780 dB**。

本轮只新增一组输入：在XYZ-only上同时加入激活后的Scale3和Opacity1。保留logratio、Anchor/ratio latent16、h32与四Head；不恢复Offset、SH DC或Rotation。

## non-Bank same-K结果

| 参照 | ratio组 | seed3100 ΔPSNR | seed3101 ΔPSNR | 平均ΔPSNR |
| --- | --- | --- | --- | --- |
| xyz_only | all | +0.593 | +0.523 | +0.558 |
| xyz_only | train | +0.498 | +0.345 | +0.421 |
| xyz_only | unseen | +0.720 | +0.760 | +0.740 |
| baseline | all | -1.111 | -0.449 | -0.780 |
| baseline | train | -0.852 | -0.043 | -0.448 |
| baseline | unseen | -1.455 | -0.989 | -1.222 |

| 方案 | PSNR↑ | SSIM↑ | LPIPS↓ | photo↓ |
| --- | --- | --- | --- | --- |
| baseline | 17.24435 | 0.60185 | 0.60022 | 0.15107 |
| xyz_only | 15.90709 | 0.56602 | 0.62009 | 0.17450 |
| xyz_scale_opacity | 16.46470 | 0.58149 | 0.61206 | 0.16512 |

## 逐ratio变化

| ratio | K | 相对XYZ-only | 相对原Baseline |
| --- | --- | --- | --- |
| 1% | 68 | +0.266 | -1.071 |
| 2.5% | 170 | +1.096 | -1.911 |
| 5% | 341 | +0.504 | -0.420 |
| 7.5% | 512 | +0.502 | -1.091 |
| 10% | 683 | +0.671 | +0.029 |
| 12.5% | 854 | +0.620 | -0.665 |
| 15% | 1025 | +0.243 | -0.329 |

## Bank、留出及实际Cached评价

| 参照 | Bank训练视图ΔPSNR | 内部留出ΔPSNR | non-Bank实际Cached ΔPSNR |
| --- | --- | --- | --- |
| xyz_only | +0.658 | +0.584 | +0.561 |
| baseline | -0.812 | -0.817 | -0.898 |

## 结构与初始化

- Encoder输入[世界XYZ3,activated Scale3,sigmoid Opacity1,logratio1]共8维，8→32 ReLU。没有新的归一化、log-scale或opacity-logit输入。
- 补偿输入为相同raw8+每Anchor每ratio latent16+h32，共56维；四Head各56→32 ReLU→32 ReLU→3/3/4/4。14维补偿输出含义、属性变换保持不变。
- 相对XYZ-only新增Scale/Opacity输入列零初始化，全部原XYZ/logratio/latent/h32权重、bias和后续层同seed复制；step0概率与零补偿exact一致，初始h32误差在gate保存。
- 只暴露XYZ/Scale/Opacity的Gaussian代理及NaN旧派生属性输入检查通过，网络不依赖SH、Rotation或局部Offset。Encoder及四Head新增连接梯度通过，其他ratio表grad=None。
- 相对XYZ-only参数增加640，总参数145616；四Head参数11982，原Baseline为12366。只对选中K行解码补偿。

## 协议与因果边界

- 同源fr1_xyz冻结N6838/M2066地图；Master/拓扑/位姿/关联/eligible/uniform保持一致。新训练seed3100/3101各12000步、每ratio3000曝光；复用匹配的两个XYZ-only与两个原Baseline最终模型。
- 同Bank帧序、四档ratio顺序、fresh fused Adam lr=.001 eps1e-15、photo=.8L1+.2(1-fast_ssim)+.01预算项；无non-Bank调参或选步数。
- 54 Bank训练320×240，Bank训练54/内部留出13/non-Bank45评价640×480；训练1/5/10/15%，另评2.5/7.5/12.5%，latent插值规则不变。
- primary same-K按logit(p)-logit(cached uniform)排序，实际Cached结果另存。Selector/Encoder仍联合训练，集合可变化；不能将全部变化归于补偿表达。
- Scale和Opacity按用户要求同时加入，本轮只识别联合增量，不能分离两者单独贡献。相对原Baseline还存在全局XYZ与局部Offset、是否输入SH的差异，不能只归因于缺SH。
- 两seed来自同一冻结地图、历史non-Bank反复用于研究，不是独立场景或在线SLAM验证。GPU同时有其他任务，耗时不用于性能结论。
- 原生产默认与已有改动不替换；未stage、commit或push；未自动新增其他输入实验。

![质量曲线](tests/diagnostics/s00_xyz_scale_opacity_20261026/evidence/analysis/quality_curves.png)

## 内部Bank留出收敛

| 方案 | seed | 4000步 | 8000步 | 12000步 |
| --- | --- | --- | --- | --- |
| baseline | 3100 | 16.746 | 17.351 | 17.496 |
| baseline | 3101 | 16.721 | 16.887 | 17.307 |
| xyz_only | 3100 | 15.542 | 15.704 | 15.800 |
| xyz_only | 3101 | 15.712 | 16.086 | 16.201 |
| xyz_scale_opacity | 3100 | 15.945 | 16.477 | 16.420 |
| xyz_scale_opacity | 3101 | 16.192 | 16.389 | 16.748 |

统一最终12000步，不按non-Bank选step；有限训练差距不等于充分收敛的容量上限。

## 交付核验

2个新增与4个参考模型完整保存，全部12000步、每ratio3000曝光与Adam时钟匹配；原型源码与归档SHA一致，逐视图四指标finite。Git HEAD、全部tracked diff及staged diff保持与实验前一致，保留已有17个tracked改动；未stage/commit/push。

[逐视图质量](tests/diagnostics/s00_xyz_scale_opacity_20261026/evidence/analysis/quality_per_view.csv) · [配对统计](tests/diagnostics/s00_xyz_scale_opacity_20261026/evidence/analysis/paired_statistics.csv) · [实际预算与集合](tests/diagnostics/s00_xyz_scale_opacity_20261026/evidence/analysis/selected_set_overlap.csv)
