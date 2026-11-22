# 同容量补偿专用Encoder：冻结选择集合的质量对照

在相同fr1_xyz冻结地图上，以两seed刚完成的48维四Head模型为源，各继续训练12000步、每ratio3000次曝光。A固定原编码，B训练同初值同容量的11→32 ReLU专用Encoder；两组Head和latent16均继续训练。原Selector、原Encoder和选中Gaussian身份完全冻结。

## 主要结果

在源模型固定Cached集合、每组exact相同K下，B相对A七档non-Bank PSNR平均 **+0.003 dB**，两个seed分别 **+0.072/-0.067 dB**。

训练ratio平均 -0.001 dB，未见ratio平均 +0.008 dB，内部Bank留出平均 +0.004 dB。辅助固定floor-ratio same-K集合的non-Bank平均变化 +0.017 dB。

逐视图/ratio配对差值中位数 +0.022 dB，p10/p90为 -0.190/+0.176 dB，改善比例 59.4%。这些视图不是独立seed重复。

## 两seed等权四指标：固定Cached选择集合

| 方案 | PSNR↑ | SSIM↑ | LPIPS↓ | photo↓ |
| --- | --- | --- | --- | --- |
| initial | 17.10570 | 0.59430 | 0.60502 | 0.15420 |
| frozen_encoder | 17.43724 | 0.60448 | 0.59772 | 0.14899 |
| private_encoder | 17.44001 | 0.60454 | 0.59717 | 0.14902 |

## 专用Encoder收益与继续训练收益

| 比较 | ratio组 | seed3100 ΔPSNR | seed3101 ΔPSNR | 平均ΔPSNR |
| --- | --- | --- | --- | --- |
| private_encoder − frozen_encoder | all | +0.072 | -0.067 | +0.003 |
| private_encoder − frozen_encoder | train | +0.018 | -0.020 | -0.001 |
| private_encoder − frozen_encoder | unseen | +0.144 | -0.129 | +0.008 |
| frozen_encoder − initial | all | +0.303 | +0.360 | +0.332 |
| frozen_encoder − initial | train | +0.696 | +0.598 | +0.647 |
| frozen_encoder − initial | unseen | -0.222 | +0.044 | -0.089 |
| private_encoder − initial | all | +0.375 | +0.294 | +0.334 |
| private_encoder − initial | train | +0.715 | +0.578 | +0.646 |
| private_encoder − initial | unseen | -0.079 | -0.085 | -0.082 |

只有B−A是专用编码适配的增量；A/B相对initial的大幅变化不能全部算到Encoder。该继续训练协议与历史从头Joint实验的绝对分数不可直接相减归因。

## 内部Bank留出收敛：固定Cached集合

| 方案 | seed | 4000步 | 8000步 | 12000步 |
| --- | --- | --- | --- | --- |
| frozen_encoder | 3100 | 17.589 | 17.639 | 17.671 |
| frozen_encoder | 3101 | 17.342 | 17.556 | 17.532 |
| private_encoder | 3100 | 17.625 | 17.693 | 17.747 |
| private_encoder | 3101 | 17.337 | 17.545 | 17.463 |

始终采用最终12000步，没有按non-Bank选step或LR；有限步数差异不是容量上限。

## 本轮判断

两个seed方向不一致，本轮Encoder适配结论为MIXED，不能仅凭平均值决定替换或继续扩容。

**当前没有稳定的同容量补偿专用Encoder质量收益，不建议据此替换生产或自动继续加深Encoder。** 真实photo梯度检查与参数变化均正常，不能把零收益解释为专用Encoder没有更新。它只能说明本次已训练源模型、固定集合、相同LR与预算下，额外编码适配价值很小；不能排除从头训练或在线场景下的差异。

两组继续训练都在训练ratio上恢复约0.65 dB，但未见ratio平均变化约−0.08～−0.09 dB。当前更值得关注补偿Head/latent的训练充分性与跨ratio泛化之间的权衡，而不是将总继续训练收益算作Encoder改善；该解释仍是下一项研究方向，不是已隔离的因果结论。

本实验固定了选择集合并冻结选择分支，因此没有直接测量原Joint路径中选择/预算/补偿的梯度竞争。A的Encoder冻结，B多384个可训练自由度；它检验编码适配，而非完全等训练自由度的纯参数共享因果。

两seed来自同一冻结地图的两个源模型，历史non-Bank已多次使用；不作独立场景稳定性、统计显著性或在线Joint质量保证。生产默认保持E50。

## 完整协议与对照


fr1_xyz、两个源48维四Head checkpoint，每组额外12000步、每ratio3000曝光；不是之前从头Joint训练的直接横比。

| 对照 | ratio组 | seed3100 ΔPSNR | seed3101 ΔPSNR | 平均ΔPSNR |
| --- | --- | --- | --- | --- |
| private_encoder − frozen_encoder | all | +0.072 | -0.067 | +0.003 |
| private_encoder − frozen_encoder | train | +0.018 | -0.020 | -0.001 |
| private_encoder − frozen_encoder | unseen | +0.144 | -0.129 | +0.008 |
| frozen_encoder − initial | all | +0.303 | +0.360 | +0.332 |
| frozen_encoder − initial | train | +0.696 | +0.598 | +0.647 |
| frozen_encoder − initial | unseen | -0.222 | +0.044 | -0.089 |
| private_encoder − initial | all | +0.375 | +0.294 | +0.334 |
| private_encoder − initial | train | +0.715 | +0.578 | +0.646 |
| private_encoder − initial | unseen | -0.079 | -0.085 | -0.082 |

## 实际结构与梯度

- 源模型为SharedE16Split4NoRaw各seed已训练12000步模型；冻结原Master、Selector、原Encoder、Anchor关系、相机和所有七档选择集合。
- A：固定复制的原11→32 ReLU编码，只训练四Head和四张Anchor/ratio latent16表。B：相同初值的独立11→32 ReLU编码也可训练，梯度仅来自补偿photo。
- 两组输入仍为latent16+编码32，共48维；四Head各48→32 ReLU→32 ReLU→3/3/4/4，14D输出变换不改。没有原始11维直连、额外Rotation、加深或加宽。
- 两组补偿模块存储参数数相同，B多384个可训练自由度。原Selector Encoder始终冻结；只对选中K行执行补偿编码和解码。
- 初始A/B输出exact相同；K行与原N行GEMM可能产生微小浮点差异，对原函数以gate记录误差容忍核对，不把此差异隐藏为逐位exact。
- A/B均fresh Adam fused lr=.001 eps1e-15，Head/latent优化器重置一致；Head12000步、各ratio表3000步，A编码0步、B编码12000步。没有伪造零梯度step。
- 训练固定原Cached Bernoulli集合，原opacity ST前向gate保留但无Selector梯度；原photo+.01预算项中预算成为冻结常数，不影响补偿梯度，不引入新Loss。

## 选点与数据

- primary为源模型固定Cached集合，每个seed内A/B/initial的Gaussian身份与实际K exact相同；不同seed可有不同K，配对比较不混淆。
- secondary为源模型logit(p)-logit(cached_uniform)排序后固定floor(rN)的same-K集合，同样身份冻结；并非训练时改换采样算法。
- 同N6838/M2066地图，54 Bank训练320×240、13内部Bank留出与45历史non-Bank评价640×480，源sha、选择rows sha及冻结检查完整保存。
- 四训练ratio1/5/10/15%，三未见ratio2.5/7.5/12.5%；latent插值与连续logratio编码保持，不为未见ratio重新训练。

## 因果与范围

- B−A隔离当前固定集合下的编码适配自由度；不等于证明原共享训练存在有害梯度冲突，也不能直接证明32维信息丢失或维度不足。
- 与initial的差距包含继续训练Head/latent的收益，不能全部记为专用Encoder收益。所有最终step固定，不按non-Bank选LR/step。
- 两seed对应相同冻结地图上的不同源模型/帧序，不是独立SLAM场景；历史评价轨迹已反复使用，没有新盲测或线上Joint证据。
- 所有生产文件与默认E50保持，未stage/commit/push。没有新的在线SLAM或CUDA Mapping构建。

![曲线](tests/diagnostics/s00_comp_private_encoder_20261026/evidence/analysis/quality_curves.png)

## 固定渲染示例

固定seed3100中间non-Bank视图，未按改善幅度选图。

![1%](tests/diagnostics/s00_comp_private_encoder_20261026/evidence/analysis/1pct_examples.png)

![5%](tests/diagnostics/s00_comp_private_encoder_20261026/evidence/analysis/5pct_examples.png)
