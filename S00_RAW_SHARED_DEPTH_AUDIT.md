# 59维输入Compensation Head：两层与三层隐藏层对照

## 结果解读

相同59维输入下，三层补偿头七档non-Bank same-K PSNR为 **16.459 dB**，两层头为 **16.449 dB**，变化 **+0.010 dB**。训练四档变化 -0.141 dB，未见三档变化 +0.211 dB。

内部Bank留出七档变化 +0.002 dB。non-Bank逐视图/ratio配对PSNR差值中位数 -0.069 dB，p10/p90为 -0.513/+0.778 dB，改善比例 44.8%。这些视图不是独立seed重复。

三层头相比同seed的E50变化 -0.220 dB，相比R3变化 -1.559 dB。只在一个冻结地图上完成一个优化顺序seed，不据此认定稳定收益、容量极限或主要瓶颈；生产默认仍为E50，没有继续自动加层或扩大维度。

## 内部Bank留出收敛曲线

| 方案 | 4000步 | 8000步 | 12000步 |
| --- | --- | --- | --- |
| RawSharedE16 | 16.194 | 16.451 | 16.646 |
| RawSharedE16D3 | 16.107 | 16.492 | 16.647 |

统一使用最终12000步，不按上表回选checkpoint。固定步数的差异同时包含表示能力与优化条件影响，不等同于充分收敛后的质量上限。

## 本轮判断

加深一层没有形成明显的总体PSNR收益，主要是训练ratio与未见ratio之间的质量权衡。七档平均SSIM、LPIPS和photo略有改善，但PSNR差值中位数为负，不能解释为多数视图全面改善。该结果不支持立即以三层头替换当前默认，也不足以排除其他参数化或优化条件下的容量收益。

本轮只检验两层→三层、宽度32、单位初始化的这一种改法；没有同步加宽、改激活、改Loss或增加训练步数，完成后停止。


fr1_xyz单次seed3100；新增一个三层头，保持12000真实LoD步与每ratio3000曝光。所有指标来自最终checkpoint，无早停或评价调参。

| 方案 | 七档PSNR | 相对两层头 | 训练ratio增益 | 未见ratio增益 |
| --- | --- | --- | --- | --- |
| RawSharedE16 | 16.449 | +0.000 | +0.000 | +0.000 |
| DirectRawE16 | 16.314 | -0.135 | -0.179 | -0.077 |
| S00R00E50D12 | 16.679 | +0.230 | +0.260 | +0.190 |
| RawSharedE16D3 | 16.459 | +0.010 | -0.141 | +0.211 |
| R3 | 18.018 | +1.568 | +1.545 | +1.600 |

## 单变量结构与初始化

- 共同输入：每Anchor每ratio latent16 + 原始Offset3/Scale3/SH DC3/Opacity1/log(ratio)1 + 原共享h32，共59维；共享h不detach。
- 两层头RawSharedE16：59→32 ReLU→32 ReLU→14。三层头RawSharedE16D3：59→32 ReLU→32 ReLU→32 ReLU→14。只增加一个32→32 Linear+ReLU，宽度与特征维度均不变，额外1056个参数。
- 追加层单位矩阵、bias零，插在原第二个ReLU后；所有公共权重、latent表初始化与两层控制exact一致。不是残差网络。初始p和delta相同，并以临时非零输出投影验证隐藏函数一致及新增层梯度。
- 训练仍使用Cached Bernoulli、原photo+.01预算损失、Adam fused lr=.001/eps1e-15、相同帧序。新增层随其余补偿参数共同训练，共享Encoder仍接收补偿梯度。

## 质量与收敛证据

| 方案 | PSNR↑ | SSIM↑ | LPIPS↓ | photo↓ |
| --- | --- | --- | --- | --- |
| RawSharedE16 | 16.44933 | 0.57862 | 0.62675 | 0.16531 |
| DirectRawE16 | 16.31417 | 0.57889 | 0.63330 | 0.16516 |
| S00R00E50D12 | 16.67913 | 0.58383 | 0.62502 | 0.16095 |
| RawSharedE16D3 | 16.45914 | 0.57960 | 0.62281 | 0.16394 |
| R3 | 18.01770 | 0.62820 | 0.58884 | 0.13756 |

所有相对变化以同seed、同地图、相同12000步的两层RawSharedE16为参考。四训练ratio为1/5/10/15%，三未见ratio为2.5/7.5/12.5%；未见ratio先插值latent再解码，没有单独拟合。

![质量曲线](tests/diagnostics/s00_raw_shared_depth_20261026/evidence/analysis/quality_curves.png)

## 实验边界

- 同一N6838/M2066地图；Master、Gaussian身份、Anchor归属/中心/尺度、相机、eligible及cached uniform冻结。54 Bank训练、13内部Bank留出、45历史non-Bank。runtime/Renderer SHA与原协议匹配。
- Selector/Encoder和补偿仍按原隔离协议共同训练，改变补偿头可以间接改变最终选择集合。不是冻结选择集合的纯补偿Oracle。same-K按logit(p)-logit(cached_uniform)排序，各方案K严格一致；原Cached实际点数另存。
- 单次单场景且使用历史评价视图，不作跨seed稳定性或新盲测结论；固定步数结果不是容量上限。4k/8k/12k收敛曲线存档，不回选checkpoint。
- E50与R3是同seed相同步数的系统目标参考，输入及补偿参数化不同。R3剩余差距不能全归于补偿头。effective_proxy为可见且opacity≥1/255的点数，不是积分渲染贡献。
- 全部实验仅在隔离Python原型中执行；生产默认E50、生产源码与在线二进制不变。无新在线SLAM或CUDA Mapping构建；未stage/commit/push。另有GPU负载，不作耗时速度结论。

## 固定视图

固定中间non-Bank视图，1%/5%，没有按改善幅度选图。

![1%](tests/diagnostics/s00_raw_shared_depth_20261026/evidence/analysis/1pct_examples.png)

![5%](tests/diagnostics/s00_raw_shared_depth_20261026/evidence/analysis/5pct_examples.png)
