# S00R00 补偿特征 E8→E50 单变量实验

本轮仅将每Anchor、每训练ratio的补偿latent由8维提高到50维：输入40→82，隐藏32→32和输出14均不变。Selector仍为11→32→32→2，保留Gaussian属性及log(ratio)，没有Selector Anchor latent。

**结论：平均略升，尚未证明稳定优于E8，也没有改善未见ratio的平均质量。** fr1/fr2七档non-Bank same-K均约+0.159 dB，但95%区间都跨0，Holm p均为0.578。训练过的ratio平均提升0.309/0.338 dB，未见ratio平均下降0.041/0.079 dB。后者同样不能直接解释为已确证的泛化损害。

距离R3仍有1.639/1.899 dB，仅恢复原PSNR差距约8.9%/7.7%。因此，增加到50维存在正向信号，但不支持“只要增加latent维度就能接近R3”的结论。实验默认按用户要求保留E50，E8回退配置完整保留；不据此自动增加层数、宽度或继续调学习率。

## 结果

| 场景 | E50−E8 PSNR | 95%CI | 改善seed | Holm p |
| --- | --- | --- | --- | --- |
| fr1_xyz | +0.159 | [-0.086, +0.433] | 6/8 | 0.5781 |
| fr2_xyz | +0.159 | [-0.197, +0.498] | 5/8 | 0.5781 |

上表为七档等权、non-Bank、确定性same-K的配对结果。95%区间来自seed与连续相机块的配对重采样；Holm仅校正本轮两个场景主比较。评价轨迹已用于此前结构研究，本轮不将其称为全新盲测。

| 场景 | ratio组 | E8 PSNR | E50 PSNR | R3 PSNR | R3差距恢复率 |
| --- | --- | --- | --- | --- | --- |
| fr1_xyz | all | 16.257 | 16.417 | 18.056 | 8.9% |
| fr1_xyz | train | 16.552 | 16.862 | 18.481 | 16.0% |
| fr1_xyz | unseen | 15.865 | 15.824 | 17.489 | -2.5% |
| fr2_xyz | all | 16.409 | 16.568 | 18.467 | 7.7% |
| fr2_xyz | train | 16.879 | 17.217 | 18.815 | 17.5% |
| fr2_xyz | unseen | 15.782 | 15.702 | 18.002 | -3.6% |

差距恢复率按PSNR dB差值定义 `(E50-E8)/(R3-E8)`，不是MSE恢复比例，也不是理论上界。

R3仍使用其原有16维Selector Anchor特征与Gaussian/ratio独立14D补偿，是系统质量目标参考。它并非冻结相同Selector后的纯补偿Oracle；不能将剩余全部dB直接归因于Compensator。

| 场景 | 方案 | PSNR↑ | SSIM↑ | LPIPS↓ | photo↓ |
| --- | --- | --- | --- | --- | --- |
| fr1_xyz | S00R00E08D12 | 16.257 | 0.5796 | 0.6260 | 0.1669 |
| fr1_xyz | S00R00E50D12 | 16.417 | 0.5799 | 0.6209 | 0.1646 |
| fr1_xyz | R3 | 18.056 | 0.6249 | 0.5930 | 0.1384 |
| fr2_xyz | S00R00E08D12 | 16.409 | 0.5730 | 0.5869 | 0.1714 |
| fr2_xyz | S00R00E50D12 | 16.568 | 0.5772 | 0.5816 | 0.1690 |
| fr2_xyz | R3 | 18.467 | 0.6221 | 0.5508 | 0.1371 |

![Quality](tests/diagnostics/s00_e50_compensation_20261026/evidence/analysis/quality_curves.png)

## 公平性与边界

- E50新增训练16次：fr1_xyz与fr2_xyz各8seed；全部12000真实LoD步，每个训练ratio3000次。E8与R3复用完全匹配的上一轮12000步结果，没有混入不同地图或训练预算。
- 复用源checkpoint、Gaussian身份/数量、Anchor归属、冻结Master、相机与帧顺序、eligibility、cached uniform、Loss、Adam lr=0.001/eps=1e-15。Renderer二进制和原实验runtime SHA一致。
- 公共Encoder/Selector、补偿公共权重以及原有8列投影初始化exact相同；仅增加私有坐标。所有方案step0 p=0.5、补偿exact零。
- 固定的是Selector结构；Encoder、Selector和补偿仍按原隔离协议共同训练。选点集合可能随优化变化，因此这是同K整体系统效果，不是冻结选点后的纯解码容量上界。
- 训练ratio为1/5/10/15%，未训练ratio为2.5/7.5/12.5%。latent先按raw ratio插值再解码，不为未见ratio单独训练；没有据评价数据选LR、step或seed。
- 新增参数会改变优化条件；固定LR下的差异不能直接当作表示能力上限。完整收敛曲线、真实Adam时钟、空选择次数、逐视图mean/median/p10/p90及改善比例均保存。
- E50最终补偿及指数尺度等激活属性的非有限值行数：0。Master训练前后exact冻结。
- 本轮是冻结地图GPU质量实验，不是新在线SLAM。训练补偿从第1步启用；生产默认仍保留3000 Mapping步warmup、原Joint/Full调度与自然停止规则。

## 收敛与结构约束

留出Bank七档平均PSNR：

| 场景 | 方案 | 4000步 | 8000步 | 12000步 |
|---|---|---:|---:|---:|
| fr1_xyz | E8 | 15.596 | 16.177 | 16.459 |
| fr1_xyz | E50 | 16.018 | 16.493 | 16.582 |
| fr2_xyz | E8 | 16.169 | 16.436 | 16.504 |
| fr2_xyz | E50 | 16.536 | 16.632 | 16.681 |

E50的早期优势比最终优势更大，这与更快的早期优化一致，不能将其视为更高的已收敛质量上限。统一12000步仍是本轮比较标准，没有回选中间checkpoint。

还有一个确定的结构事实：第一层为 `t = ReLU(W_h h + U z + b)`，其中 `U ∈ R^(32×50)`。固定网络后，50维latent经U最多形成32个独立线性方向；在冻结拓扑和线性latent插值下，可将 `Uz` 吸收到32维表中表示相同静态函数。因此本结构中的E50并不意味着50个独立可用的补偿方向。这个秩上界不能证明本轮质量瓶颈就是32维hidden，也不能替代E32或更宽head的实际实验；本轮没有新增这些搜索。

## 当前配置与交付

项目实验默认为 `s00r00e50d12`，E8回退文件 `cfg/selector/s00_e08_reference.yaml`，R3质量参考 `cfg/selector/r3_reference.yaml`。C++/Python新50维保存、插值和生命周期支持已接入；本地CPU/推理核验通过，未将其等同于完整CUDA Mapping构建或在线质量验证。

以上默认修改位于本地项目。服务器本轮运行的是隔离Python质量实验，没有替换或重编译服务器在线SLAM二进制及其默认配置。

原始指标：`analysis/quality_summary.csv`、`quality_per_view.csv`、`paired_statistics.csv`、`r3_gap_recovery.csv`、`fairness_checks.csv`、`validation_curves.csv`。模型与训练日志保存在本次服务器目录；无早停、无容量/学习率追加搜索。未stage、commit或push。

## 固定视图示例

每场景固定第一个与中间non-Bank视图、固定第一个seed，显示GT/E8/E50/R3；不是挑选改善最大的图片。

![fr1_xyz 1%](tests/diagnostics/s00_e50_compensation_20261026/evidence/analysis/fr1_xyz_1pct_examples.png)

![fr1_xyz 5%](tests/diagnostics/s00_e50_compensation_20261026/evidence/analysis/fr1_xyz_5pct_examples.png)

![fr2_xyz 1%](tests/diagnostics/s00_e50_compensation_20261026/evidence/analysis/fr2_xyz_1pct_examples.png)

![fr2_xyz 5%](tests/diagnostics/s00_e50_compensation_20261026/evidence/analysis/fr2_xyz_5pct_examples.png)

## 原Cached Bernoulli推理与实际点数

原策略的七档non-Bank平均PSNR增益为fr1 +0.186、fr2 +0.237 dB。实际K随模型变化，以下同时列出K；主结论仍依据same-K。

| 场景 | ratio | E8 PSNR / 实际K均值 | E50 PSNR / 实际K均值 |
|---|---|---|---|
| fr1_xyz | 1% | 14.342 / 67.88 | 14.528 / 64.25 |
| fr1_xyz | 5% | 16.513 / 333.38 | 16.866 / 329.88 |
| fr1_xyz | 10% | 17.517 / 683.00 | 17.843 / 679.50 |
| fr1_xyz | 15% | 17.914 / 1021.00 | 18.284 / 1011.88 |
| fr2_xyz | 1% | 14.995 / 89.12 | 15.565 / 83.50 |
| fr2_xyz | 5% | 16.886 / 436.62 | 17.233 / 428.38 |
| fr2_xyz | 10% | 17.494 / 897.12 | 17.879 / 873.00 |
| fr2_xyz | 15% | 17.820 / 1321.88 | 18.207 / 1322.00 |
