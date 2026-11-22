# Rotation同时输入共享Encoder与四个补偿Head的质量对照

**结论：两个seed的七档平均PSNR均下降；本轮不支持将Rotation同时加入Encoder与全部Head，建议保留59维四Head实验Baseline。**

七档non-Bank same-K平均变化为 **-0.253 dB**，两seed分别 **-0.234/-0.273 dB**。

本轮将59维四Head作为实验Baseline，重新训练两seed；候选仅增加基础Quaternion输入。生产默认和在线代码未替换。

## 主要结果

| ratio组 | seed3100 ΔPSNR | seed3101 ΔPSNR | 平均ΔPSNR |
| --- | --- | --- | --- |
| all | -0.234 | -0.273 | -0.253 |
| train | -0.330 | -0.418 | -0.374 |
| unseen | -0.105 | -0.080 | -0.092 |

以上为候选减本轮重新训练Baseline，non-Bank、deterministic same-K。两个seed是同一冻结地图的网络初始化/训练顺序重复，不能当作独立场景或统计显著性。

## 两seed等权质量

| 方案 | PSNR↑ | SSIM↑ | LPIPS↓ | photo↓ |
| --- | --- | --- | --- | --- |
| baseline | 17.24435 | 0.60185 | 0.60022 | 0.15107 |
| rotation_all | 16.99088 | 0.59552 | 0.60062 | 0.15521 |

## 逐ratio变化

| ratio | seed3100 ΔPSNR | seed3101 ΔPSNR | 平均ΔPSNR |
| --- | --- | --- | --- |
| 1% | -0.806 | -1.789 | -1.297 |
| 2.5% | -0.097 | -0.351 | -0.224 |
| 5% | -0.083 | -0.147 | -0.115 |
| 7.5% | -0.086 | +0.074 | -0.006 |
| 10% | -0.219 | +0.469 | +0.125 |
| 12.5% | -0.132 | +0.038 | -0.047 |
| 15% | -0.215 | -0.206 | -0.211 |

## 输入、结构与初始化

- 原始输入实际11维：Offset3、Scale3、SH DC3、Opacity1、logratio1。Rotation采用现有归一化Quaternion4，因此候选是15维，并非12维。q/-q按最大绝对分量为正统一符号。
- Baseline Encoder11→32 ReLU，四Head各59→32 ReLU→32 ReLU→3/3/4/4；候选Encoder15→32，四Head全部63→32→32→3/3/4/4，Other也读取Rotation。
- Head输入为原始属性/ratio、每Anchor每ratio latent16、共享h32；latent不参与Selector，未见ratio仍先线性插latent再解码。h不detach。
- 所有公共权重同seed复制，新增Quaternion列零初始化；step0选择概率和补偿输出exact一致，隐藏GEMM浮点误差另存。实际photo的前三步验证新输入梯度，未把临时非零输出投影探针当成真实step1梯度。
- Head参数12366→12878，Encoder增加128参数，总计增加640；仅增加输入连接，不改变隐藏层宽深、14D输出变换或Loss。
- 上一轮Rotation只输入三个几何Head，Encoder和Other无Rotation；本轮是同时加入两条路径，不能从本对照进一步单独分离Encoder与直连的贡献。

## 公平性与范围

- 同源fr1_xyz冻结N6838/M2066地图，Master、关联、Gaussian身份、相机、eligible和cached uniform不变。54 Bank训练、13内部Bank留出、45历史non-Bank。
- 两组都从初始零补偿训练12000真实LoD步，每ratio3000；fresh fused Adam lr=.001 eps1e-15，共同参数12000步、每latent表3000步；帧序和ratio顺序相同，第二seed反转两臂执行次序。
- 原Cached Bernoulli、photo=.8L1+.2(1-fast_ssim)及.01预算项不变。Encoder与Selector参与训练，最终选中身份可变；same-K只固定预算，并非固定集合。
- same-K按logit(p)-logit(cached uniform)排序；四指标、实际Cached点数及visible/effective_proxy完整保存。effective_proxy是可见且opacity≥1/255，并非积分渲染贡献。
- 训练1/5/10/15%，评价另含2.5/7.5/12.5%；统一最终12000步，无non-Bank调参/选checkpoint。Bank/留出/non-Bank完整逐视图配对结果保存。
- GPU同时存在其他任务，耗时仅为运行记录，不用作无干扰性能对比。历史non-Bank多次用于结构研究，不是全新盲测；没有在线SLAM、跨场景结论。

## 同协议Baseline重复波动

| seed | 本轮Baseline − 历史Baseline PSNR |
| --- | --- |
| 3100 | +0.011 |
| 3101 | +0.041 |

复跑波动与本轮两seed差值一起解释，不把微小均值当稳定收益。所有旧结果保留，不挑最好Baseline。

Bank字段为全部67视图（包含13留出）；bank_train字段仅54个实际训练视图，并用相同640×480分辨率评价，validation为13个内部留出，test为45个历史non-Bank。

![质量曲线](tests/diagnostics/s00_rotation_encoder_20261026/evidence/analysis/quality_curves.png)

## 本地交付核验

原型源码与服务器归档SHA一致；4模型各12000步、每ratio3000曝光及Adam时钟通过核对，全部逐视图指标finite。实验开始时已有17个tracked改动，Git HEAD、全部tracked diff和staged diff在实验前后完全一致；未stage、commit或push。

[完整逐视图指标](tests/diagnostics/s00_rotation_encoder_20261026/evidence/analysis/quality_per_view.csv) · [配对统计](tests/diagnostics/s00_rotation_encoder_20261026/evidence/analysis/paired_statistics.csv) · [预算与优化器核验](tests/diagnostics/s00_rotation_encoder_20261026/evidence/analysis/fairness_checks.csv)
