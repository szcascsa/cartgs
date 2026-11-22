# 局部Offset与SH DC的等容量输入增量对照

相对世界XYZ+Scale+Opacity参照，Offset平均 **+0.480 dB**，SH DC平均 **-0.601 dB**；SH DC相对Offset平均 **-1.080 dB**。

**结论：Offset在两个seed上七档平均均高于SH DC。** 两个优化seed共享一张冻结地图，不能据此宣称跨场景显著性。

base为上一轮世界XYZ+Scale+Opacity+logratio。A只加局部Offset3，B只加SH DC3，两组均保留世界XYZ；不是用Offset替换世界XYZ，也没有同时加两项。

## non-Bank same-K结果

| 对照 | ratio组 | seed3100 ΔPSNR | seed3101 ΔPSNR | 平均ΔPSNR |
| --- | --- | --- | --- | --- |
| offset − base | all | +0.740 | +0.219 | +0.480 |
| offset − base | train | +0.386 | -0.126 | +0.130 |
| offset − base | unseen | +1.212 | +0.680 | +0.946 |
| shdc − base | all | -0.153 | -1.048 | -0.601 |
| shdc − base | train | -0.067 | -1.124 | -0.595 |
| shdc − base | unseen | -0.268 | -0.948 | -0.608 |
| shdc − offset | all | -0.893 | -1.268 | -1.080 |
| shdc − offset | train | -0.453 | -0.998 | -0.725 |
| shdc − offset | unseen | -1.479 | -1.628 | -1.553 |
| offset − baseline | all | -0.371 | -0.229 | -0.300 |
| offset − baseline | train | -0.466 | -0.169 | -0.318 |
| offset − baseline | unseen | -0.244 | -0.310 | -0.277 |
| shdc − baseline | all | -1.264 | -1.497 | -1.380 |
| shdc − baseline | train | -0.919 | -1.167 | -1.043 |
| shdc − baseline | unseen | -1.723 | -1.937 | -1.830 |

| 方案 | PSNR↑ | SSIM↑ | LPIPS↓ | photo↓ |
| --- | --- | --- | --- | --- |
| baseline | 17.24435 | 0.60185 | 0.60022 | 0.15107 |
| base | 16.46470 | 0.58149 | 0.61206 | 0.16512 |
| offset | 16.94428 | 0.59616 | 0.60172 | 0.15545 |
| shdc | 15.86399 | 0.56784 | 0.61973 | 0.17176 |

## 逐ratio变化

| ratio | K | Offset − base | SH DC − base | SH DC − Offset |
| --- | --- | --- | --- | --- |
| 1% | 68 | +0.564 | -0.316 | -0.880 |
| 2.5% | 170 | +1.534 | +0.215 | -1.318 |
| 5% | 341 | -0.000 | -0.700 | -0.699 |
| 7.5% | 512 | +0.991 | -1.235 | -2.226 |
| 10% | 683 | -0.022 | -0.736 | -0.714 |
| 12.5% | 854 | +0.312 | -0.804 | -1.116 |
| 15% | 1025 | -0.021 | -0.630 | -0.609 |

## Bank、留出及实际Cached评价

| 对照 | Bank训练视图ΔPSNR | 内部留出ΔPSNR | non-Bank实际Cached ΔPSNR |
| --- | --- | --- | --- |
| offset − base | +0.506 | +0.513 | +0.441 |
| shdc − base | -0.559 | -0.544 | -0.584 |
| shdc − offset | -1.066 | -1.057 | -1.025 |
| offset − baseline | -0.305 | -0.304 | -0.458 |
| shdc − baseline | -1.371 | -1.361 | -1.483 |

## 结构与初始化

- 两组Encoder输入均为原raw8+新增3维=11维，11→32 ReLU；补偿输入raw11+每Anchor每ratio latent16+h32=59维，四Head均59→32 ReLU→32 ReLU→3/3/4/4。
- Offset=(真实XYZ−冻结Anchor中心)/冻结局部参考尺度，原世界XYZ同时保留。SH DC是基底球谐DC系数3维，不是RGB；SH分支不读取Anchor中心/尺度。两组都不输入Rotation。
- 原common8/latent/h32权重、bias和后续层同seed复制，新增3维输入列零初始化；两组初始完整state dict exact相同，step0概率和零补偿与base exact一致。
- 仅暴露允许属性的代理及旧派生tensor设NaN检查通过；SH分支用不可访问的Anchor空间元数据对象进一步检查无Offset依赖。Encoder与四Head新增连接梯度、其他ratio表grad=None通过。
- 两组总参数均146096、四Head均12366；相对base各增加480个输入连接参数，彼此容量相同。共享h32、latent16插值及14D输出变换全部保留。

## 协议与因果边界

- 同源fr1_xyz冻结N6838/M2066地图；Master/拓扑/位姿/关联/eligible/uniform保持一致。两组seed3100/3101各12000步、每ratio3000曝光；第二seed反转两臂运行顺序。复用匹配的两个base和两个原Baseline最终模型。
- 同Bank帧序、四档ratio顺序、fresh fused Adam lr=.001 eps1e-15、photo=.8L1+.2(1-fast_ssim)+.01预算项；无non-Bank调参或选步数。
- 54 Bank训练320×240，Bank训练54/内部留出13/non-Bank45评价640×480；训练1/5/10/15%，另评2.5/7.5/12.5%，latent插值规则不变。
- primary same-K按logit(p)-logit(cached uniform)排序，实际Cached结果另存。Selector/Encoder仍联合训练，集合可变化；不能将全部变化归于补偿表达。
- Offset与SH分支等参数、同初值；其直接对照检验新信息类型的差异，但Selector/Encoder仍联合训练，不能把全部收益单独归于补偿。
- 原Baseline输入为局部Offset+Scale+SH DC+Opacity+logratio；与本轮各组比较仍涉及坐标/信息组合及输入列初始化方式差异，不能只归因于某一项。没有同时加Offset和SH的训练。
- 两seed来自同一冻结地图、历史non-Bank反复用于研究，不是独立场景或在线SLAM验证。GPU同时有其他任务，耗时不用于性能结论。
- 原生产默认与已有改动不替换；未stage、commit或push；未自动新增其他输入实验。

![质量曲线](tests/diagnostics/s00_offset_shdc_20261026/evidence/analysis/quality_curves.png)

## 内部Bank留出收敛

| 方案 | seed | 4000步 | 8000步 | 12000步 |
| --- | --- | --- | --- | --- |
| baseline | 3100 | 16.746 | 17.351 | 17.496 |
| baseline | 3101 | 16.721 | 16.887 | 17.307 |
| base | 3100 | 15.945 | 16.477 | 16.420 |
| base | 3101 | 16.192 | 16.389 | 16.748 |
| offset | 3100 | 16.533 | 17.192 | 17.174 |
| offset | 3101 | 16.556 | 17.152 | 17.021 |
| shdc | 3100 | 16.076 | 16.471 | 16.273 |
| shdc | 3101 | 15.640 | 15.272 | 15.807 |

最终12000步统一评价，不按留出或non-Bank选择checkpoint。有限训练预算不等于充分收敛的表示上限。

## 交付核验

4个新模型、4个参考模型完整保存；各12000步、每ratio3000曝光、Adam时钟和新组等参数量均通过核验。原型源码与归档SHA一致，逐视图四指标finite，Git HEAD、tracked diff和staged diff与实验前一致，保留已有17个tracked改动。未stage、commit或push。

[逐视图质量](tests/diagnostics/s00_offset_shdc_20261026/evidence/analysis/quality_per_view.csv) · [配对统计](tests/diagnostics/s00_offset_shdc_20261026/evidence/analysis/paired_statistics.csv) · [预算与集合](tests/diagnostics/s00_offset_shdc_20261026/evidence/analysis/selected_set_overlap.csv)
