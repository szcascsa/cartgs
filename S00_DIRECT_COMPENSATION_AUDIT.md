# Compensation直接输入：8/16维ratio特征与Offset对照

## 结论

**没有证明共享32维混合表示是主要瓶颈。** 同样8维latent下，绕过h32的Offset组下降0.294 dB，原始属性组下降0.147 dB。

16维直接输入产生正向信号：Offset组比E8旧路径高0.206 dB，原始属性组高0.268 dB。但本轮没有16维latent+h32的匹配对照，因此不能把16维组的提升单独归因于去掉混合表示。

本轮最高PSNR的新增方案为DirectRawE16，但仍比同seed的E50共享编码方案低0.365 dB、比R3低1.704 dB。所有四组均未接近R3，保留生产E50默认。

单次单场景结果仅用于快速筛查，不表述为稳定提高或统计显著，不从中自动选模型部署。


本轮按用户要求仅fr1_xyz、单优化顺序seed3100；新增四个模型，全部12000真实LoD步，每ratio3000次曝光。不作稳定收益或统计显著性判断。

| 方案 | 七档PSNR | 相对E8共享编码 | 训练ratio增益 | 未见ratio增益 |
| --- | --- | --- | --- | --- |
| S00R00E08D12 | 16.046 | +0.000 | +0.000 | +0.000 |
| S00R00E50D12 | 16.679 | +0.633 | +0.739 | +0.492 |
| DirectOffsetE08 | 15.752 | -0.294 | +0.059 | -0.764 |
| DirectRawE08 | 15.899 | -0.147 | -0.224 | -0.045 |
| DirectOffsetE16 | 16.252 | +0.206 | +0.061 | +0.399 |
| DirectRawE16 | 16.314 | +0.268 | +0.300 | +0.225 |
| R3 | 18.018 | +1.972 | +2.024 | +1.901 |

四指标与逐ratio指标、原Cached实际预算、有效点代理及逐视图配对分布见analysis/*.csv。

## 结构与协议

- DirectOffsetE08/E16：每Anchor每训练ratio的8/16维latent，加归一化局部Offset3，进入两隐藏层32→32、输出14的补偿头。
- DirectRawE08/E16：相同latent，加Offset3、activated scale3、SH DC3、activated opacity1及log(ratio)1，总输入19/27维。
- Selector仍使用原11→32共享层、32→32→2选择头；四组仅补偿绕过共享h32。未见ratio在1/5/10/15%节点之间先线性插值latent再解码。
- 训练ratio1/5/10/15%，未见ratio2.5/7.5/12.5%。源地图N6838/M2066、Master属性/Anchor关联/相机/eligible/cached uniform冻结，54训练Bank、13内部Bank留出、45历史non-Bank。
- 原photo和0.01预算项、Adam lr=.001/eps1e-15、训练帧顺序、最终12000步一致，无early stop或评价调参。8/16维各组初始选择概率及补偿相同，第二隐藏层和输出层初始化一致；直接输入组原始前3列、latent前8列权重一致。
- E8共享编码、E50、R3参考复用同地图、同seed、同12000步结果。R3同时有不同的Selector特征与独立补偿参数化，仅作系统目标参考。
- Master冻结，但Selector和Encoder继续依原协议训练；Direct补偿不再向Encoder回传。因此同时改变补偿信息路径、Encoder梯度耦合和从原属性到输出的有效深度。不能只凭质量变化证明32维压缩丢失信息或梯度冲突。
- same-K集合按logit(p)-logit(cached_uniform)排序；不同方案选中身份可不同。effective_proxy是radii>0且opacity≥1/255的点数，不是逐像素积分贡献。
- 本轮为隔离冻结地图GPU实验，生产默认E50及C++/Python生产实现未变，未进行新的在线SLAM。未stage/commit/push。

## 输入建议与归因边界

Offset区分Anchor内位置；Scale提供形状/物理尺度，Opacity与SH DC提供外观状态，log(ratio)直接提供连续预算条件。DirectRaw将这些旧输入原样送入私有补偿头，避免共享编码器变换。
本轮没有新增Rotation或全局XYZ，也没有逐项拆分11维属性。若Raw优于Offset，只能支持这组附加属性整体有价值，不能归到其中单个输入。

![质量曲线](tests/diagnostics/s00_direct_compensation_20261026/evidence/analysis/quality_curves.png)

## 固定视图

固定中间non-Bank视图、1%与5%，未按改善大小选图。

![1%](tests/diagnostics/s00_direct_compensation_20261026/evidence/analysis/1pct_examples.png)

![5%](tests/diagnostics/s00_direct_compensation_20261026/evidence/analysis/5pct_examples.png)

## 8→16维与原始信息的配对比较

| 比较（后者为参考） | 七档ΔPSNR | 训练四档ΔPSNR | 未见三档ΔPSNR |
| --- | --- | --- | --- |
| DirectOffsetE16 − DirectOffsetE08 | +0.500 | +0.002 | +1.163 |
| DirectRawE16 − DirectRawE08 | +0.415 | +0.524 | +0.270 |
| DirectRawE08 − DirectOffsetE08 | +0.146 | -0.283 | +0.719 |
| DirectRawE16 − DirectOffsetE16 | +0.062 | +0.239 | -0.174 |

8→16维两种输入路径均提高本次平均PSNR；Offset组的变化主要在未见ratio，不能解释成训练ratio普遍提高。16维Raw比Offset平均仅高约0.062 dB，训练ratio更好但未见ratio更低，说明附加属性整组也存在权衡。

输入建议：优先保留局部Offset及显式log(ratio)作为空间身份与连续预算条件；Scale、Opacity和SH DC可直接供补偿使用。本轮只比较这几项整体加入，尚未单独验证每项的贡献，不能宣称它们逐项必需或已确定最优输入组合。

若后续继续归因，价值最高的是匹配16维latent+h32控制，而不是继续盲目增加latent或层数；本轮没有追加该实验。

## 七档等权：四指标

| 方案 | PSNR↑ | SSIM↑ | LPIPS↓ | photo↓ |
| --- | --- | --- | --- | --- |
| S00R00E08D12 | 16.04607 | 0.56991 | 0.63066 | 0.17080 |
| S00R00E50D12 | 16.67913 | 0.58383 | 0.62502 | 0.16095 |
| DirectOffsetE08 | 15.75233 | 0.56791 | 0.64019 | 0.17517 |
| DirectRawE08 | 15.89873 | 0.57339 | 0.62259 | 0.17044 |
| DirectOffsetE16 | 16.25203 | 0.57422 | 0.62857 | 0.16833 |
| DirectRawE16 | 16.31417 | 0.57889 | 0.63330 | 0.16516 |
| R3 | 18.01770 | 0.62820 | 0.58884 | 0.13756 |

## 逐ratio PSNR与严格相同预算

| ratio | K | S00R00E08D12 | S00R00E50D12 | DirectOffsetE08 | DirectRawE08 | DirectOffsetE16 | DirectRawE16 | R3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1% | 68 | 14.388 | 15.188 | 14.268 | 13.926 | 14.516 | 14.477 | 16.246 |
| 2.5% | 170 | 13.481 | 14.171 | 12.037 | 13.880 | 14.219 | 14.345 | 15.170 |
| 5% | 341 | 16.226 | 17.040 | 16.513 | 16.242 | 15.713 | 16.702 | 18.620 |
| 7.5% | 512 | 16.079 | 16.489 | 15.320 | 15.893 | 16.170 | 16.073 | 18.234 |
| 10% | 683 | 17.432 | 17.930 | 17.344 | 16.925 | 17.651 | 17.654 | 19.361 |
| 12.5% | 854 | 17.078 | 17.453 | 16.988 | 16.730 | 17.447 | 16.896 | 18.939 |
| 15% | 1025 | 17.638 | 18.483 | 17.795 | 17.695 | 18.047 | 18.054 | 19.555 |

DirectOffsetE08在2.5%出现明显退化；DirectOffsetE16在5%仍低于E8，DirectRawE16在7.5%和12.5%也没有提升。平均值不能解释成所有档位均变好。所有5684条Bank留出/non-Bank、same-K/Cached逐视图记录的四项质量指标均finite。

## 训练与验证边界

- 四个新增模型均完成12000步，无空选择训练步；每ratio曝光3000次。初始选择相同、补偿exact零，所有源地图张量保持冻结。
- 私有latent梯度gate是在临时设置非零输出投影后验证链路；实际零输出初始化的第一步与旧方案一样，latent初始梯度可以为零。没有把gate解释为实际首步latent即更新。
- 本地6296个已有tracked文件SHA核对无变化，暂存区为空；本轮仅增加隔离实验与报告。服务器没有替换生产实现或在线二进制。
- 同GPU另有工作负载；耗时记录仅用于实验完整性，不作网络速度比较。
- 评价沿用历史non-Bank轨迹，不是全新盲测。优化顺序seed只有一个，不估计跨seed稳定性或以逐视图数量代替独立重复。
