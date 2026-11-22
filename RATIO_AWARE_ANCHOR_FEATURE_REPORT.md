# Ratio-aware Anchor 特征：实现与质量对照

**实现已完成，CPU／CUDA功能验证及真实训练质量对照已完成。** 本轮未获得稳定的 same-K 质量提升，不能据此认定 ratio 专属特征优于旧结构。

same-K 平均 ΔPSNR=-0.2225 dB、Δphoto=+0.003311；2/8 组同时改善两项。这里的平均对每个 seed／预算等权。完整四指标和逐视图配对数据均保留。

## 主要结果与解释

| 对照范围（两 seed 等权） | ΔPSNR dB | ΔSSIM | ΔLPIPS | Δphoto |
|---|---:|---:|---:|---:|
| 四个训练 ratio 的正式推理 | -0.1401 | -0.00189 | -0.00221 | +0.002578 |
| 三个插值 ratio 的正式推理 | -0.6912 | -0.01396 | +0.01377 | +0.009550 |
| Full | -0.1305 | -0.00341 | +0.00618 | +0.001743 |
| 四个相同绝对 K | -0.2225 | -0.00295 | +0.00187 | +0.003311 |

训练 ratio 的 LPIPS均值略好，但PSNR／photo均值变差；seed2026四档same-K均退化，seed2027仅较高两个K同时改善PSNR／photo，说明收益依赖seed／预算。

各ratio表的数值标准差约0.0286～0.0694，相邻表的平均L2差异约0.062～0.110，已经实际分化。真实模型C++／Python四训练ratio的Cached Bernoulli集合完全一致，几何／Opacity／SH DC补偿数值通过验证。这些证据排除了“特征完全未更新”以及本次训练模型在两种推理实现之间不一致的解释；尚不能据此确定容量分配、优化或插值中的哪一项是主要限制。

3%插值在两seed分别下降1.8113／1.4730 dB，同时实际选择数从458→372／547→418，点数减少与特征插值、补偿、地图差异同时存在，不能把全部下降归因于特征插值本身。闭环容量迁移在单测中通过，本次四次在线训练中迁移计数均为0，因此未覆盖真实闭环事件。

当前代码按批准方案保留新结构默认开启，`Selector.anchor_ratio_features=0`可运行旧结构。**本轮质量证据不支持将新结构视作质量改进；没有根据评价结果自动改架构、LR、训练步数或回滚用户指定结构。**

## 实际改动

- 每 Anchor 的旧16维 Feature 改为共享8维，加四个训练 ratio 各自独立的8维 Feature。五张表分别是叶子 Parameter，不将不同 ratio 塞进同一 Parameter 的不同列。
- 当前档位输入 `[shared8, ratio_feature8, offset3, activated_scale3, SH_DC3, activated_opacity1]`，合计26维；移除显式 `log(ratio)` 输入。ratio 仍决定特征查询、原预算损失与原选择预算。
- 网络为共享 `26→32 ReLU`，选择分支 `32→32 ReLU→2`；仍先选择，再仅对被选中点解码 `32→32 ReLU→14` 补偿。14维仍为 XYZ3、log-scale3、quaternion4、opacity-logit1、SH-DC3。
- 训练档位为1%、5%、10%、15%；仅共享表、当前档表和网络获得梯度。其他档位 gradient=None，不推进其 Adam step 或动量。
- 未见 ratio 在相邻训练节点按原始 ratio 线性插值特征，例如3%=0.5·feature(1%)+0.5·feature(5%)；范围外使用最近端点特征，选择预算仍用实际 ratio。Full继续 Master 路径。
- 五张表均支持初始零初始化、同层逐通道max继承、细层零初始化、Adam扩展／压缩、teacher冻结、checkpoint、闭环接收成员数加权及混合端动量清零。Gaussian的现有归属、坐标和优化语义保留。
- 新 checkpoint 为布局v7；C++与Python自动识别26维布局及ratio节点。旧27维 checkpoint 继续走旧路径，不进行有损截断迁移。当前默认 `Selector.anchor_ratio_features=1`，设为0可回到旧结构。

## 验证与实验协议

- 16项服务器 CPU／CUDA／Python／生产 Mapping 门控通过，详见 [correctness.json](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/ratio_feature_validation/evidence/validation/correctness.json)。
- 验证了五张独立表、当前档梯度、其他档参数／Adam历史不动、增长／prune／闭环加权迁移、插值／端点、紧凑行索引、teacher、新旧checkpoint和选后补偿。生产Mapping验证LoD联合更新以及Full参数／动量隔离。
- 真实训练4次：fr1_xyz × seed2026/2027 × 旧结构／新结构。仅特征布局开关不同；Adam、LR=0.001、compensation_start_iter=3000、joint、ratio调度、Cached Bernoulli、重建及预算损失不变。
- TF、SI/GI、ChangeDistill保持关闭；Anchor-only pruning、可变1..10成员、停用与原增长算法保留。`max_mapping_iterations=0`，使用原自然停止，不强制10000步，不按评价结果选checkpoint或调参。
- 固定相同64个RGB评价视图，最终shutdown模型；正式推理保持Cached Bernoulli，另按cached logit margin截取完全相同整数K作诊断。same-K输入ratio=K/N。正式选择策略没有替换。
- 每组在线SLAM使用自己的估计位姿、地图和异步Mapping轨迹；同seed不保证相同优化次数。故这是端到端对照，不能单独将变化归因于选择排序，也不能声称已证明跨场景稳定改善。

## 训练、地图规模与成本

| 方法／seed | 最终 iteration | Gaussian | Anchor | Full render | LoD render | 训练秒 | 采样峰值 MiB | 原生Selector ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_seed2026 | 7400 | 13834 | 4922 | 1989 | 6048 | 45.25 | 915 | 0.1723 |
| ratio_features_seed2026 | 7400 | 14878 | 5864 | 1991 | 6048 | 45.18 | 937 | 0.1666 |
| baseline_seed2027 | 7400 | 15027 | 5362 | 1892 | 6023 | 44.29 | 955 | 0.1747 |
| ratio_features_seed2027 | 7400 | 14608 | 5505 | 2050 | 6063 | 45.61 | 945 | 0.1668 |

训练时间包含在线SLAM、Mapping与最终保存，未做严格性能基准；GPU峰值来自0.5秒采样，可能漏掉瞬时峰值。原生Selector计时为各自真实N、5组重复取中位数，每组预热20次后测200次；N不同，不能将耗时比直接解释为网络结构开销。

同样Anchor数量时，Feature从16→40个FP32值／Anchor，参数本体增加96 bytes／Anchor；若计入gradient与Adam一／二阶状态，理论增加384 bytes／Anchor。26维第一层比旧27维少32个权重，网络其余宽度不变。

## 正式 Cached Bernoulli：四指标与实际点数

下表Δ为新−旧。PSNR／SSIM越大越好，LPIPS／photo越小越好；同名ratio不代表相同点数。

| seed | ratio | 旧K→新K | ΔPSNR dB | ΔSSIM | ΔLPIPS | Δphoto |
|---:|---|---:|---:|---:|---:|---:|
| 2026 | Full | 13834→14878 | -0.1115 | -0.00343 | +0.00297 | +0.001493 |
| 2026 | 1% | 134→143 | -0.5733 | -0.00971 | +0.00181 | +0.008472 |
| 2026 | 3% | 458→372 | -1.8113 | -0.03794 | +0.04087 | +0.024633 |
| 2026 | 5% | 680→736 | -0.4512 | -0.01171 | +0.01267 | +0.007094 |
| 2026 | 7.5% | 1083→1074 | -0.8309 | -0.01622 | +0.01045 | +0.010176 |
| 2026 | 10% | 1354→1474 | -0.5084 | -0.00888 | -0.00258 | +0.005985 |
| 2026 | 12.5% | 1738→1880 | -0.5535 | -0.00992 | -0.00214 | +0.006502 |
| 2026 | 15% | 2026→2205 | -0.5365 | -0.00843 | -0.00655 | +0.006044 |
| 2027 | Full | 15027→14608 | -0.1496 | -0.00340 | +0.00939 | +0.001992 |
| 2027 | 1% | 148→145 | -0.0219 | -0.00090 | -0.00065 | +0.001047 |
| 2027 | 3% | 547→418 | -1.4730 | -0.03192 | +0.04990 | +0.019607 |
| 2027 | 5% | 742→731 | +0.1071 | +0.00119 | +0.01229 | -0.000682 |
| 2027 | 7.5% | 1134→1152 | +0.1137 | +0.00086 | +0.00189 | -0.000130 |
| 2027 | 10% | 1458→1433 | +0.3704 | +0.00977 | -0.01259 | -0.003024 |
| 2027 | 12.5% | 1848→1832 | +0.4075 | +0.01139 | -0.01833 | -0.003485 |
| 2027 | 15% | 2190→2152 | +0.4928 | +0.01355 | -0.02211 | -0.004309 |

3%、7.5%、12.5%未参加训练，只做特征插值推理。插值公式连续不等于质量必然单调或优于旧结构。

## 相同绝对K的质量对照

| seed | K | 旧PSNR | 新PSNR | ΔPSNR dB | ΔSSIM | ΔLPIPS | Δphoto |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 151 | 16.0745 | 15.4314 | -0.6431 | -0.00744 | +0.00423 | +0.008576 |
| 2026 | 792 | 19.0288 | 18.2387 | -0.7901 | -0.01774 | +0.02581 | +0.010403 |
| 2026 | 1434 | 19.6816 | 19.1137 | -0.5679 | -0.01096 | +0.00315 | +0.006756 |
| 2026 | 2075 | 19.8352 | 19.2702 | -0.5649 | -0.01009 | -0.00058 | +0.006565 |
| 2027 | 151 | 16.4376 | 16.3680 | -0.0696 | -0.00167 | +0.00126 | +0.001641 |
| 2027 | 792 | 18.6862 | 18.6776 | -0.0085 | +0.00112 | +0.01468 | -0.000092 |
| 2027 | 1434 | 19.2219 | 19.6038 | +0.3819 | +0.01004 | -0.01237 | -0.003193 |
| 2027 | 2075 | 19.3534 | 19.8357 | +0.4823 | +0.01311 | -0.02124 | -0.004169 |

same-K只控制选择点数，未冻结两组各自训练出的Gaussian属性和SLAM位姿。该结果不能被表述为纯Selector排序的因果证据。

## 交付与结论

本轮未获得稳定的 same-K 质量提升，不能据此认定 ratio 专属特征优于旧结构。

- [quality_summary.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/ratio_feature_validation/evidence/validation/quality_summary.csv)：全部48组质量及实际点数；[quality_per_view.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/ratio_feature_validation/evidence/validation/quality_per_view.csv)：逐视图四指标。
- [paired_statistics.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/ratio_feature_validation/evidence/validation/paired_statistics.csv)：每个seed／ratio／预算的mean、median、p10、p90及improved-view fraction。
- [feature_statistics.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/ratio_feature_validation/evidence/validation/feature_statistics.csv)：各表数值统计；[model_diagnostics.json](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/ratio_feature_validation/evidence/validation/model_diagnostics.json)：各层Anchor、更新次数、Feature分化、显存和计时。
- [training_protocol.json](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/ratio_feature_validation/evidence/validation/training_protocol.json)、[evaluation_plan.json](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/ratio_feature_validation/evidence/validation/evaluation_plan.json)、[audit_summary.json](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/ratio_feature_validation/evidence/validation/audit_summary.json)保存锁定协议、模型SHA及复核结果。
- 代码和本轮模型／日志证据保留；未stage、commit或push。本轮没有修改网络深度、增长阈值、损失或训练停止规则。
