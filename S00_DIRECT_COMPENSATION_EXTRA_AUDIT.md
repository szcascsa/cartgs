# Compensation追加：32维ratio特征与原始/共享编码联合输入

## 结果解读

相对上一轮16维ratio特征＋原始11维：

- 32维ratio特征＋原始11维：七档平均变化 **+0.041 dB**；训练四档 +0.100 dB，未见三档 -0.039 dB。
- 16维ratio特征＋原始11维＋共享h32：七档平均变化 **+0.135 dB**；训练四档 +0.179 dB，未见三档 +0.077 dB。

与同seed的E50共享编码方案相比，两组分别变化 -0.324 / -0.230 dB；与R3相比为 -1.663 / -1.568 dB。

本轮只跑一个优化顺序seed和一个冻结地图。它可以描述当前两组改动的质量影响，不能据此认定更高维度或共享表示普遍有益/有害。共享组还恢复了Encoder梯度通路，因此不把其变化单独归为信息压缩问题。生产默认保持E50。

补充判读：32维组虽然PSNR略升，SSIM和photo相比DirectRawE16更差；共享16维组LPIPS改善，但SSIM与photo没有同步改善，不能称为四指标全面提升。逐ratio看，32维组在2.5%比DirectRawE16低约1.036 dB，共享16维组在该档也低约0.405 dB；平均值不能掩盖这一插值档位退化。

本次证据不支持直接删除共享编码，也不支持继续单调扩大latent。加回共享h32有小幅正向信号，但仍不足以证明它是主要质量限制，更不能将共享组的收益全部解释为信息恢复。


仅fr1_xyz、单次seed3100。新增两组，均12000真实LoD步、每ratio3000曝光，最终checkpoint评价；不根据评价选择LR或步数。

| 方案 | 七档PSNR | 相对旧E8 | 训练ratio增益 | 未见ratio增益 |
| --- | --- | --- | --- | --- |
| S00R00E08D12 | 16.046 | +0.000 | +0.000 | +0.000 |
| S00R00E50D12 | 16.679 | +0.633 | +0.739 | +0.492 |
| DirectRawE08 | 15.899 | -0.147 | -0.224 | -0.045 |
| DirectRawE16 | 16.314 | +0.268 | +0.300 | +0.225 |
| DirectRawE32 | 16.355 | +0.309 | +0.401 | +0.186 |
| RawSharedE16 | 16.449 | +0.403 | +0.480 | +0.302 |
| R3 | 18.018 | +1.972 | +2.024 | +1.901 |

## 关键匹配对照

| 比较 | 七档ΔPSNR | 训练ratioΔPSNR | 未见ratioΔPSNR |
| --- | --- | --- | --- |
| DirectRawE32 − DirectRawE16 | +0.041 | +0.100 | -0.039 |
| RawSharedE16 − DirectRawE16 | +0.135 | +0.179 | +0.077 |
| RawSharedE16 − DirectRawE32 | +0.095 | +0.079 | +0.116 |

## 网络与梯度

- DirectRawE32：每Anchor每ratio latent32 + 原始属性11，总输入43，补偿头43→32 ReLU→32 ReLU→14，不读取共享h32。
- RawSharedE16：latent16 + 原始属性11 + 原共享h32，总输入59，补偿头59→32 ReLU→32 ReLU→14。共享h不detach，补偿可向共享Encoder回传。
- 原始11维为归一化局部Offset3、activated Scale3、SH DC3、activated Opacity1、log(ratio)1；Selector网络输入和结构保持11→32→32→2。
- 每Anchor四张独立latent表；训练ratio1/5/10/15%，未见ratio2.5/7.5/12.5%，沿用raw-ratio线性插值latent再解码。只对选中K行计算补偿。
- 与旧DirectRawE16的raw11、前16列latent投影、bias及第二层/输出层初始化exact一致。32维组延伸原28列随机reservoir，保留前16列；共享组为追加h32初始化小幅非零投影。
- 所有组初始p=.5与补偿exact零；私有特征梯度gate以临时非零输出层检查链路，不声称实际零输出初始化首步latent即更新。
- 同步验证：直接32组compensation→Encoder梯度为零，共享16组该梯度非零，Selector私有head不接收独立补偿forward的梯度；非当前ratio表gradient=None。

## 冻结地图与公平性

- 复用N6838/M2066的同一fr1_xyz地图、同一54 Bank训练/13内部Bank留出/45历史non-Bank、seed3100和帧序；源地图manifest、runtime及Renderer SHA匹配历史实验。
- Master、Anchor归属/中心/尺度、相机、eligible及cached uniform冻结。Selector/Encoder仍按原预算与photo共同训练，不同方案选择集合可以变化；不是冻结Selection的纯补偿Oracle。
- 同一Adam fused lr=.001/eps1e-15、原photo及.01预算项、同12000步和每ratio3000曝光；E8/E50/DirectRaw8/DirectRaw16/R3均复用同seed与相同步数结果。
- 正式same-K按logit(p)-logit(cached_uniform)排序；Cached实际点数另存。effective_proxy表示可见且opacity≥1/255，不是积分贡献测量。
- 新增共享输入也恢复了Compensation与Encoder的梯度耦合，并增加头参数；该对照测量整个附加h路径效果，不能单独证明信息压缩或梯度冲突因果。
- 单次单场景、历史评价视图，不作跨seed稳定收益、统计显著性或新盲测结论；R3有不同Selector特征与独立参数化，仅作系统目标参考。
- 生产算法和E50默认不变。本轮无在线SLAM、无新CUDA Mapping构建，未stage/commit/push。GPU另有负载，不以记录耗时作速度比较。

## 四指标：七档non-Bank same-K

| 方案 | PSNR↑ | SSIM↑ | LPIPS↓ | photo↓ |
| --- | --- | --- | --- | --- |
| S00R00E08D12 | 16.04607 | 0.56991 | 0.63066 | 0.17080 |
| S00R00E50D12 | 16.67913 | 0.58383 | 0.62502 | 0.16095 |
| DirectRawE08 | 15.89873 | 0.57339 | 0.62259 | 0.17044 |
| DirectRawE16 | 16.31417 | 0.57889 | 0.63330 | 0.16516 |
| DirectRawE32 | 16.35468 | 0.57007 | 0.63008 | 0.16847 |
| RawSharedE16 | 16.44933 | 0.57862 | 0.62675 | 0.16531 |
| R3 | 18.01770 | 0.62820 | 0.58884 | 0.13756 |

![质量曲线](tests/diagnostics/s00_direct_compensation_extra_20261026/evidence/analysis/quality_curves.png)

## 固定视图

固定中间non-Bank视图、1%/5%，不按改善幅度选图。

![1%](tests/diagnostics/s00_direct_compensation_extra_20261026/evidence/analysis/1pct_examples.png)

![5%](tests/diagnostics/s00_direct_compensation_extra_20261026/evidence/analysis/5pct_examples.png)
