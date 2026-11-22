# 加回局部Offset与16维瓶颈三层ReLU共享Encoder的质量对照

**结论：相对上一组两seed均下降。** 七档non-Bank same-K相对上一组平均 **-0.192 dB**，相对原Baseline **-0.971 dB**。

本轮结构已经修正为 **11→16→ReLU→32→ReLU→32→ReLU**。先前缺少中间ReLU的运行已停止并排除，不计入本报告。

先前缺中间ReLU的任务已停止并标记superseded，不计入本轮结果。本轮按用户更正同时加入局部Offset3，并将Encoder改为Linear11→16、ReLU、Linear16→32、ReLU、Linear32→32、ReLU。最终输出32维共享编码，每个Linear之后都有ReLU，不另加输出层。

## non-Bank same-K结果

| 参照 | ratio组 | seed3100 ΔPSNR | seed3101 ΔPSNR | 平均ΔPSNR |
| --- | --- | --- | --- | --- |
| baseline | all | -1.321 | -0.621 | -0.971 |
| baseline | train | -1.388 | -0.475 | -0.932 |
| baseline | unseen | -1.232 | -0.817 | -1.024 |
| xyz_scale_opacity | all | -0.211 | -0.173 | -0.192 |
| xyz_scale_opacity | train | -0.536 | -0.432 | -0.484 |
| xyz_scale_opacity | unseen | +0.223 | +0.173 | +0.198 |

| 方案 | PSNR↑ | SSIM↑ | LPIPS↓ | photo↓ |
| --- | --- | --- | --- | --- |
| baseline | 17.24435 | 0.60185 | 0.60022 | 0.15107 |
| xyz_scale_opacity | 16.46470 | 0.58149 | 0.61206 | 0.16512 |
| offset_encoder16 | 16.27296 | 0.58122 | 0.63618 | 0.16609 |

## 各ratio

| ratio | K | 相对上一组 | 相对原Baseline |
| --- | --- | --- | --- |
| 1% | 68 | -0.366 | -1.436 |
| 2.5% | 170 | +0.435 | -1.477 |
| 5% | 341 | -0.883 | -1.303 |
| 7.5% | 512 | +0.402 | -0.689 |
| 10% | 683 | -0.449 | -0.421 |
| 12.5% | 854 | -0.243 | -0.907 |
| 15% | 1025 | -0.238 | -0.567 |

## Bank、留出与原Cached评价

| 参照 | Bank训练视图ΔPSNR | 内部留出ΔPSNR | non-Bank实际Cached ΔPSNR |
| --- | --- | --- | --- |
| baseline | -0.885 | -0.884 | -0.970 |
| xyz_scale_opacity | -0.074 | -0.067 | -0.072 |

## 结构与初始化

- raw11=[全局XYZ3,activated Scale3,sigmoid Opacity1,局部Offset3,logratio1]。Offset=(XYZ−所属Anchor中心)/原冻结参考尺度；与旧scene.attrs前三维exact一致。没有SH DC或Rotation输入。
- Encoder=11→16→ReLU→32→ReLU→32→ReLU，三个Linear后均有ReLU；中间32维层包含非线性，不能合并为一个仿射映射。
- 补偿输入raw11+Anchor/ratio latent16+h32共59维，四Head仍各59→32 ReLU→32 ReLU→3/3/4/4；原14维属性变换、latent插值、Selector32→32→2均保留。只对选中K行补偿。
- 新Encoder首层取旧首层前16行公共权重/偏置，新增Offset列零；后两层用固定seed+110/+120初始化。补偿头公共列和后续层复制、新Offset列零。因16维瓶颈，隐藏函数不与原32维浅层一致；step0概率和零补偿exact一致。
- Encoder参数1792，较上一组8→32的288增加1504；四Head参数12366，增加384。总参数147504，较上一组增加1888。
- 输入代理仅开放XYZ/Scale/Opacity，将旧派生属性设NaN后输出exact不变；Offset由几何和所属Anchor重新计算。所有Encoder层及Offset连接梯度有效，其他ratio表grad=None。

## 协议和限制

- 同源fr1_xyz N6838/M2066冻结地图，Master/拓扑/位姿/关联/eligible/uniform固定；Selector/Encoder/补偿和当前ratio latent联合训练，最终集合可以不同。
- 两seed各12000步、每ratio3000，复用匹配的上一组和原Baseline最终模型，不挑最好重复；同Bank帧序、ratio顺序、fresh fused Adam lr=.001 eps1e-15、原photo+.01预算项。
- 54 Bank训练320×240，54 Bank训练/13内部留出/45历史non-Bank评价640×480；训练1/5/10/15%，另评2.5/7.5/12.5%。统一最终12000步，不根据评价调LR或选checkpoint。
- 主same-K按logit(p)-logit(cached uniform)排序；原Cached实际点数和四指标另存。effective_proxy是可见且opacity≥1/255，并非积分贡献。
- 本轮同时加入Offset并改变Encoder结构/初始化/参数量，不能单独归因于Offset、深度或16维瓶颈；需要时另做单变量对照，但本轮不自动扩展。
- 原Baseline虽然也有raw11，却是Offset3+Scale3+SH DC3+Opacity1+logratio1和11→32浅Encoder；同维数不代表同信息。
- 两seed为同一历史地图上的初始化/帧序重复，不是独立SLAM地图或新盲测。GPU同时存在其他任务，不作无干扰性能结论；生产源码/默认未替换，未stage/commit/push。

![质量曲线](tests/diagnostics/s00_offset_encoder16_relu_20261026/evidence/analysis/quality_curves.png)

## 内部Bank留出收敛

| 方案 | seed | 4000步 | 8000步 | 12000步 |
| --- | --- | --- | --- | --- |
| baseline | 3100 | 16.746 | 17.351 | 17.496 |
| baseline | 3101 | 16.721 | 16.887 | 17.307 |
| xyz_scale_opacity | 3100 | 15.945 | 16.477 | 16.420 |
| xyz_scale_opacity | 3101 | 16.192 | 16.389 | 16.748 |
| offset_encoder16 | 3100 | 16.249 | 16.606 | 16.242 |
| offset_encoder16 | 3101 | 16.254 | 16.516 | 16.792 |

使用固定最终12000步，没有按non-Bank选LR/step；有限预算结果不能直接等同表达能力上限。

## 本地交付核验

更正后的2个新增模型与4个参照模型完整保存，全部12000步、每ratio3000曝光和Adam时钟匹配。源码与服务器归档SHA一致，逐视图四指标finite；Git HEAD、全部tracked diff和staged diff与实验前完全一致，已有17个tracked改动保留。未stage、commit或push。

[逐视图指标](tests/diagnostics/s00_offset_encoder16_relu_20261026/evidence/analysis/quality_per_view.csv) · [配对统计](tests/diagnostics/s00_offset_encoder16_relu_20261026/evidence/analysis/paired_statistics.csv) · [预算与时钟核验](tests/diagnostics/s00_offset_encoder16_relu_20261026/evidence/analysis/fairness_checks.csv)
