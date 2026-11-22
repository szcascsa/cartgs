# 可变成员 Anchor：原 joint 重建质量测试

**结论：减少存储点和低 opacity 成员的目标达成；当前结果不支持“整体重建质量稳定提高”。Full 四指标均退化，原 ratio 四档 LoD 均退化；相同 K 下有中等预算收益，但存在预算和指标依赖。**

## 实验定义与固定基线

正式对照为 `fr1_xyz / seed2026 / 10,000 Mapping`。候选使用原 joint：online 按 1%/5%/10%/15%/Full 轮转；LoD 联合优化 Gaussian、Selector、Anchor Feature 和 TF，Full 只优化 Master。尾部保留原机制的 Full 后一个轮转 joint LoD。`Training.serial_full_lod=0`、Full100=0。没有使用此前串行 joint 的可变成员模型。

- 固定十点：复用 `/home/prmi/zwj/anchor-slot-deactivation-20261021/enabled_seed2026` 的既有模型与 Full/原 ratio 质量结果。
- 可变成员：本次 `/home/prmi/zwj/variable-anchor-joint-quality-20261021/variable_seed2026`，只增加可变成员生命周期；其余参数沿用固定模型的输入配置，并显式关闭 serial/Full100。
- 两者均 SI/GI disabled、maturity=1、27 维网络、原 Loss、Cached Bernoulli、原 TF/CD，slot 停用设置保留。没有根据质量调参数或选择 checkpoint。
- 沿用此前锁定的 64 个共同时间采样 RGB 视图；不是按质量选帧，也不声称是严格的 non-Bank 集合。两模型使用各自在线估计位姿。

用户补充“复用固定实验”时，刚启动的固定重复训练已完成。该结果保留在本次目录的 `fixed_seed2026`，**不参与任何正式质量表**。后续没有追加训练；固定模型只补充新的 same-K 推理。原始协议和用户指令修订分别保存在 `training_protocol.json`、`protocol_amendment.json`。

## Full-map 质量

PSNR/SSIM 越高越好，LPIPS/photo 越低越好。photo 使用原生 `0.8 L1 + 0.2 (1-fast_ssim)`，不含训练 opacity 正则。

| 方法 | Gaussian | PSNR | SSIM | LPIPS | photo |
|---|---:|---:|---:|---:|---:|
| 固定十点 | 42,640 | 22.0451 | 0.76060 | 0.27537 | 0.08426 |
| 可变成员 | 7,099 | 21.7957 | 0.75245 | 0.29203 | 0.08713 |

Full 差值：PSNR **−0.2493 dB**、SSIM −0.00815、LPIPS +0.01666、photo +0.00287。PSNR 改善 28/64 视图，LPIPS 改善仅 3/64；不能将均值 PSNR 接近解释为感知质量保持不变。

## 原 ratio 的生产 Cached Bernoulli

这些是生产选择结果，点数随各自地图人口变化，不能作为相同预算的纯选择器比较。

| ratio | 方法 | 实际点数 | PSNR | SSIM | LPIPS | photo |
|---|---|---:|---:|---:|---:|---:|
| 1% | 固定十点 | 447 | 18.8471 | 0.64858 | 0.54860 | 0.12608 |
| 1% | 可变成员 | 89 | 15.8053 | 0.58157 | 0.70884 | 0.16567 |
| 5% | 固定十点 | 2,135 | 20.6425 | 0.70293 | 0.41076 | 0.10352 |
| 5% | 可变成员 | 350 | 18.2414 | 0.63271 | 0.58842 | 0.13247 |
| 10% | 固定十点 | 4,216 | 21.0699 | 0.71827 | 0.37663 | 0.09812 |
| 10% | 可变成员 | 703 | 19.4241 | 0.66333 | 0.51015 | 0.11873 |
| 15% | 固定十点 | 6,157 | 21.1361 | 0.72110 | 0.37011 | 0.09721 |
| 15% | 可变成员 | 1,052 | 19.6795 | 0.67351 | 0.48005 | 0.11501 |

四档 PSNR 差值分别 **−3.0418 / −2.4011 / −1.6458 / −1.4567 dB**；四个指标的均值均退化。地图显著缩小后，继续使用同一名义 ratio 会分配更少的绝对点数，这一预算变化必须与选择质量分开看。

## 相同绝对 K 的质量

预先按人口确定共同预算区间 `[ceil(0.01 × max N), floor(0.15 × min N)] = [427,1064]`，均匀取四个整数 K。每个模型以 **K/N** 同时输入 Selector 与 TF，均处于训练 ratio 的 1%～15% 数值范围内，避免外推；大多数不是直接训练过的四个 ratio，因此结果也包含 Selector/TF 的 ratio 插值行为。使用 `logit(p_effective)−logit(cached_uniform)` 加 source-row tie-break 取 exact K，仅作诊断，没有替换生产采样算法。

| K | 固定输入 ratio | 可变输入 ratio | 固定 PSNR | 可变 PSNR | ΔPSNR | PSNR 改善视图 |
|---|---:|---:|---:|---:|---:|---:|
| 427 | 1.001% | 6.015% | 18.7234 | 18.4868 | -0.2366 | 19/64 |
| 639 | 1.499% | 9.001% | 18.9714 | 19.3015 | +0.3301 | 50/64 |
| 852 | 1.998% | 12.002% | 18.9265 | 19.5797 | +0.6532 | 50/64 |
| 1064 | 2.495% | 14.988% | 19.5743 | 19.6713 | +0.0970 | 38/64 |

| K | ΔSSIM | ΔLPIPS | Δphoto |
|---|---:|---:|---:|
| 427 | -0.00295 | -0.00116 | +0.00085 |
| 639 | +0.00345 | +0.00067 | -0.00312 |
| 852 | +0.00895 | -0.01087 | -0.00609 |
| 1064 | +0.00249 | +0.00515 | -0.00187 |

K=852 的四指标均值全部改善，PSNR/LPIPS/photo 分别改善 50/56/53 个视图。K=639 的 PSNR/photo 改善，但 LPIPS 均值略差；K=1064 的 PSNR 略升，LPIPS 退化；K=427 的 PSNR/photo 退化。应判为 **same-K MIXED**，不能只引用最高的 +0.653 dB。

![原 joint 质量曲线](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/variable_anchor_joint_quality/quality_curves.png)

## 人口、低 opacity 与实际训练轨迹

| 项目 | 固定十点 | 可变成员 |
|---|---:|---:|
| Gaussian 总数 | 42,640 | 7,099 |
| Anchor 数 | 4,264 | 1,798 |
| opacity < 1/255 的存储点 | 29,272 | 0 |
| opacity < 0.02 的存储点 | 30,756 | 0 |
| 窗口停用点 | 20,739 | 0 |
| Full Render | 4,960 | 4,743 |
| LoD Render | 8,741 | 8,686 |
| Master Adam 更新 | 13,701 | 13,429 |
| Selector 正常更新 | 8,741 | 8,686 |
| TF 正常更新 | 6,341 | 6,286 |
| SLAM online 结束 Mapping | 6,299 | 6,571 |

存储 Gaussian 减少 **83.35%**。opacity ≥1/255 的存储点从 13,368 降至 7,099，减少 **46.90%**；这仍不是逐点真实渲染贡献统计。最终可变成员模型中，每个点当前 opacity 均 ≥0.02，四档生产选择均没有低于 renderer 截断的点。

平均成员数为 3.948；L0/L1/L2 Anchor 分别为 627/1,157/14。新模型人口、初始点覆盖、增长、剪枝和在线运行时序一起变化，不能将收益或损失单独归因于某一项。原 joint 调度规则相同，但异步 SLAM 结束时间不同，导致实际 Full/LoD 更新次数不同；本轮没有强行改调度以匹配次数。

原生 Selector forward 的五组均值为固定 0.380 ms、可变 0.169 ms，分别测各自实际地图。该差异包含人口变化，不是相同 N 下的算法加速对照。候选训练 wall time 为 68.61 秒，不作为严格性能基准。

## 验证与交付

- 18 个评价条目、1,152 条逐视图结果、36 条配对统计已在本地从原始数组重新计算；共同帧、实际选择行和 exact K 全部核验。
- 250 个证据文件与 57 个已验证源文件 SHA256 匹配；58 个固定评价文件直接复用且与此前本地归档 hash 一致。
- 两个正式模型的四档 C++/Python/导出 Cached Bernoulli mask exact 一致，评价前后模型 SHA 不变。
- 候选成员数、Anchor 归属、topology 交替、maturity=1、原 joint 与 Full100=0 的状态核验通过。

- [质量汇总 CSV](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/variable_anchor_joint_quality/evidence/validation/quality_summary.csv)
- [逐视图 CSV](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/variable_anchor_joint_quality/evidence/validation/quality_per_view.csv)
- [配对统计 CSV](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/variable_anchor_joint_quality/evidence/validation/paired_statistics.csv)
- [地图与训练统计](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/variable_anchor_joint_quality/evidence/validation/model_diagnostics.csv)
- [本地交付核验](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/variable_anchor_joint_quality/local_delivery_checks.json)

本轮仅支持这一个 scene/seed 的结论。减少低 opacity 点没有自动换来整体画质提升；可变成员方案在中等绝对预算上有收益，同时 Full 感知质量与最低预算仍有损失。本轮止于质量测试，没有依据结果调 pruning、容量、网络或 Loss，也没有 stage、commit 或 push。
