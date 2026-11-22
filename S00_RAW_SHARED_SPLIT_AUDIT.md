# 59维输入Compensation：XYZ / Scale / Rotation / Other四分支对照

## 结果解读

相同59维输入下，四分支补偿头七档non-Bank same-K PSNR为 **17.001 dB**，统一两层头为 **16.449 dB**，变化 **+0.552 dB**。训练四档变化 +0.376 dB，未见三档变化 +0.786 dB。

内部Bank留出七档变化 +0.510 dB。non-Bank逐视图/ratio配对PSNR差值中位数 +0.572 dB，p10/p90为 -0.289/+1.270 dB，改善比例 82.2%。这些视图不是独立seed重复。

四分支头相比同seed的E50变化 +0.322 dB，相比R3变化 -1.017 dB。只在一个冻结地图上完成一个优化顺序seed，不据此认定稳定收益、容量极限或主要瓶颈；生产默认仍为E50，没有继续自动调整分支、Loss或维度。

## 内部Bank留出收敛曲线

| 方案 | 4000步 | 8000步 | 12000步 |
| --- | --- | --- | --- |
| RawSharedE16 | 16.194 | 16.451 | 16.646 |
| RawSharedE16Split4 | 16.736 | 16.941 | 17.155 |

统一使用最终12000步，不按上表回选checkpoint。固定步数的差异同时包含表示能力与优化条件影响，不等同于充分收敛后的质量上限。

## 各ratio的non-Bank same-K结果

| ratio | K | 统一两层PSNR | 四分支PSNR | 差值 |
| --- | --- | --- | --- | --- |
| 1% | 68 | 14.707 | 15.590 | +0.882 |
| 2.5% | 170 | 13.940 | 14.814 | +0.874 |
| 5% | 341 | 16.921 | 17.304 | +0.382 |
| 7.5% | 512 | 16.505 | 17.359 | +0.854 |
| 10% | 683 | 17.743 | 17.949 | +0.206 |
| 12.5% | 854 | 17.098 | 17.727 | +0.629 |
| 15% | 1025 | 18.231 | 18.264 | +0.034 |

原Cached Bernoulli推理七档平均PSNR为 16.972 dB，对照为 16.435 dB，变化 +0.536 dB。实际点数有差异，因此该结果为补充验证，正式预算归因仍以same-K为准。

| ratio | 统一头Cached K | 四分支Cached K |
| --- | --- | --- |
| 1% | 65 | 65 |
| 2.5% | 165 | 162 |
| 5% | 343 | 333 |
| 7.5% | 512 | 521 |
| 10% | 699 | 696 |
| 12.5% | 900 | 874 |
| 15% | 1056 | 1028 |

## 本轮判断

本次四指标与七个ratio的平均PSNR均优于统一两层头，Bank留出与non-Bank收益方向一致；是值得继续确认的正向信号，但仍是同一冻结地图上的一个优化顺序seed，尚未达到生产替换依据。

本次只测量单seed、同预算下拆分四个属性分支的整体效果。新增8928个隐藏参数与改变属性梯度共享同时发生，因此不将质量变化全部归因于某一个因素。独立分支仍共享输入latent16与Encoder，图像损失仍耦合几何和外观，不能称为完全解耦优化。

四分支共享同一16维ratio特征表，所有分支各用两层32维隐藏层。Residual/Other只预测剩余Opacity1+SH DC3，不另行叠加全14维残差。本轮没有增加额外属性Loss或等参数量统一头控制，完成后停止。


fr1_xyz、seed3100，新增一组四分支模型；全部12000真实LoD步、每ratio3000曝光，最终checkpoint评价，不作早停或评价调参。

| 方案 | 七档PSNR | 相对统一两层头 | 训练ratio增益 | 未见ratio增益 |
| --- | --- | --- | --- | --- |
| RawSharedE16 | 16.449 | +0.000 | +0.000 | +0.000 |
| RawSharedE16D3 | 16.459 | +0.010 | -0.141 | +0.211 |
| S00R00E50D12 | 16.679 | +0.230 | +0.260 | +0.190 |
| RawSharedE16Split4 | 17.001 | +0.552 | +0.376 | +0.786 |
| R3 | 18.018 | +1.568 | +1.545 | +1.600 |

## 本轮结构

- 输入始终是每Anchor每ratio latent16 + 原始11维(Offset3/Scale3/SH DC3/Opacity1/log(ratio)1) + 原共享h32，共59维。所有分支接收同一输入，latent不按属性额外复制。
- XYZ：59→32 ReLU→32 ReLU→3；Scale：同样两层、输出3；Rotation：同样两层、输出4；Residual/Other：同样两层、输出4，含Opacity-logit1+SH DC3。
- 四组输出按XYZ3、Scale3、Rotation4、Opacity1、SH DC3拼回原14维。Other仅预测剩余四个属性分量，不另加一个全14维残差，也不叠加旧统一头输出。
- 四个分支的两层隐藏参数独立，不是只将同一线性输出层切片；初始化复制原统一头的隐藏参数四份，并复制相应输出行。初始函数/概率一致，零补偿exact相同。
- 补偿头参数3438→12366，增加8928。没有等参数量的统一宽头控制，不能将变化只归因于解除属性梯度共享。
- 共享h不detach；分支输入处的Encoder与latent梯度仍求和，所以并非完全解除属性间优化耦合。Selector私有Head、预算规则和原重建Loss均不改，不加属性Loss或权重。
- 14维几何含义保持原样：XYZ加法，Scale乘exp，Rotation为identity+delta归一化后与基础旋转相乘，Opacity在logit域相加后sigmoid，SH DC加法。

## 四指标：七档non-Bank same-K

| 方案 | PSNR↑ | SSIM↑ | LPIPS↓ | photo↓ |
| --- | --- | --- | --- | --- |
| RawSharedE16 | 16.44933 | 0.57862 | 0.62675 | 0.16531 |
| RawSharedE16D3 | 16.45914 | 0.57960 | 0.62281 | 0.16394 |
| S00R00E50D12 | 16.67913 | 0.58383 | 0.62502 | 0.16095 |
| RawSharedE16Split4 | 17.00095 | 0.59359 | 0.60141 | 0.15404 |
| R3 | 18.01770 | 0.62820 | 0.58884 | 0.13756 |

![质量曲线](tests/diagnostics/s00_raw_shared_split_20261026/evidence/analysis/quality_curves.png)

## 实验边界

- 同一N6838/M2066 Variable Anchor地图；Master、Gaussian身份、Anchor归属/中心/尺度、相机、eligible及cached uniform冻结。54 Bank训练、13内部Bank留出、45历史non-Bank；runtime/Renderer SHA一致。
- Selector/Encoder和补偿仍按原隔离协议共同训练。same-K预算严格一致，但各方案选中身份可变化；本轮不是固定Selection的纯补偿Oracle。same-K排序仍为logit(p)-logit(cached_uniform)。
- 原Adam fused lr=.001/eps1e-15、photo+.01预算项、帧序、四档ratio轮转和步数均保持。训练1/5/10/15%，未见2.5/7.5/12.5%，先插值相同latent16再通过四个头，不单独优化未见ratio。
- 旧统一两层/三层、E50、R3复用同地图、同seed与同12000步结果。R3是系统目标参考，输入和独立补偿参数化不同，剩余差距不能全部归因于拆头。
- 历史SPLIT_COMPENSATION_HEAD_AUDIT使用13919 Gaussian、冻结原27→32 Encoder和不同预算/帧协议。本轮6838 Gaussian、59维补偿输入，不混用旧历史质量数字。
- 单次单场景、历史评价轨迹，不作跨seed稳定性、新盲测或容量上限结论；保存4k/8k/12k曲线但不回选checkpoint。effective_proxy为可见且opacity≥1/255，不是积分贡献。
- 无生产算法/默认变更、无新在线SLAM或CUDA Mapping构建。TF/SI/GI等沿用既定关闭配置。未stage/commit/push；另有GPU负载，不作速度比较。

## 固定视图

固定中间non-Bank视图、1%/5%，未按改善幅度选图。

![1%](tests/diagnostics/s00_raw_shared_split_20261026/evidence/analysis/1pct_examples.png)

![5%](tests/diagnostics/s00_raw_shared_split_20261026/evidence/analysis/5pct_examples.png)
