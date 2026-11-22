# Selector 的 Opacity 与颜色补偿：实现和质量测试

**已实现 14 维补偿并通过 14 项功能与回归门控。** 单次 `fr1_xyz / seed2026 / 10000 Mapping` 对照中，原 ratio 有 2/4 档同时改善 PSNR 和 photo；完整质量、相同 K 与同集合外观消融列于下方，不据单次结果推断跨场景稳定性。

10%／15% 的 PSNR 分别提升 0.3549／0.5395 dB；1%／5% 分别下降 0.2315／0.0478 dB。相同 K 下三档 PSNR 提升，最低预算下降；应判为有中高预算收益、最低预算尚未改善。新增外观补偿的同集合推理作用是 PSNR／photo 四档均改善，但 5%／10%／15% 的 LPIPS 略差，不能声称所有指标都提升。

## 补偿定义

共享浅层和选择分支保持不变；私有补偿头从 `32→32 ReLU→10` 扩为 `32→32 ReLU→14`。先 Cached Bernoulli 选择，再仅对 K 个选中点的缓存特征解码。

| 分量 | 维度 | 应用方式 |
|---|---:|---|
| XYZ | 3 | `XYZ′=XYZ+δXYZ` |
| Scale | 3 | `scale′=scale·exp(δlog-scale)` |
| Rotation | 4 | 原单位四元数残差与原旋转相乘并归一化 |
| Opacity | 1 | `opacity′=sigmoid(master_opacity_logit+δopacity_logit)` |
| 颜色基色 SH DC | 3 | `SH_DC′=master_SH_DC+δSH_DC` |

颜色只补偿 RGB 基色对应的 SH DC。高阶 SH 保留原值，不增加相机或视角输入。基色 RGB 的变化是 `0.28209479·δSH_DC`；沿用原 renderer 的颜色处理，不额外 clamp。

Opacity 直接读取 Master 的显式原始 logit；不通过 `logit(sigmoid(logit))` 反算，因此有限但饱和的 logit 不会被变成 Inf。ST gate 仅在补偿后乘一次：`render_opacity=ST_mask·opacity′`。

全部输出最后层 weight/bias 仍 exact zero 初始化，补偿在 iteration 3000 开始。LoD 联合更新 Master、Selector、两个分支和 Anchor Feature；Full 只优化原 Master Gaussian，不执行任何补偿。

正常 photo、ratio loss、27 维输入、Cached Bernoulli、原 joint online/tail 调度、可变成员生命周期、停用机制、SI/GI disabled 和 TF disabled 均保留。新增输出增加 132 个参数，FP32 权重增加 528 bytes；这不等于已测出整个训练峰值显存。

剪枝、停用与 Selector 输入仍读取 Master Opacity／SH DC，补偿只影响当前 LoD 渲染；本轮没有改变候选资格或 pruning 规则。

## 保存、导出与 Change Distill

- 新 checkpoint schema=2，对应 14 维；旧 schema=1 继续按 10 维加载、克隆和推理，旧真实模型的原生选择及几何输出 exact 不变。不会静默把旧权重升级为 14 维。

- Python 推理、直接 prepared renderer 和标准 PLY 导出均使用补偿后的 Opacity logit 与 SH DC；高阶 SH 保留原值。未选中点的完整属性在临时补偿结果中保持 exact 原值。

- 几何 Change Distill 继续运输 XYZ／Scale／Rotation；额外保留补偿后的 Opacity 和 SH DC。仅 CD 阶段增加新增属性的 L1 保真项：`mean|α′student−α′teacher| + C0·mean|DC′student−DC′teacher|`，不增加正常重建 loss 或正则。

- 教师冻结、Master 不被 CD 修改、正常 Adam 状态不变、event-local Adam 更新和新旧 schema 均已做功能验证。

## 实验协议

- 仅新训练一次 14 维候选；旧 10 维模型直接复用上一轮保存结果。两轮输入 YAML 的 SHA256 exact 相同，保持同 seed、10000 Mapping、最后 checkpoint，不根据评价调参或选步数。

- 原 joint，`serial_full_lod=0`、Full100=0、TF 关闭。online 为 1%/5%/10%/15%/Full 轮转，tail 保留 Full 后一个轮转 joint LoD。

- 使用原先锁定的 64 个共同 RGB 视图和各自在线估计位姿；不是按质量选帧，不声称是严格 non-Bank。

- 共同整数预算 K=[76, 421, 765, 1110]，每模型输入 ratio=K/N，均在 1%～15% 内；cached logit margin exact-K 仅作诊断，生产仍 Cached Bernoulli。

- 外观 OFF 只在新模型推理时忽略新增四维输出，保留同一权重、ratio、选中集合和几何补偿；其 PLY 几何和高阶 SH exact 一致。这不是重新训练一个没有外观补偿的模型。

## Full 和生产 Cached Bernoulli

| 档位 | 方法 | 实际 K | PSNR↑ | SSIM↑ | LPIPS↓ | photo↓ |
|---|---|---:|---:|---:|---:|---:|
| Full | 几何10维 | 7406 | 21.5113 | 0.74507 | 0.30666 | 0.08991 |
| Full | 几何＋外观14维 | 7563 | 21.5785 | 0.75144 | 0.29917 | 0.08815 |
| 1% | 几何10维 | 73 | 15.2089 | 0.56679 | 0.72361 | 0.17684 |
| 1% | 几何＋外观14维 | 75 | 14.9774 | 0.57235 | 0.71847 | 0.17739 |
| 5% | 几何10维 | 364 | 17.9123 | 0.62163 | 0.60354 | 0.13831 |
| 5% | 几何＋外观14维 | 362 | 17.8645 | 0.62868 | 0.61035 | 0.13644 |
| 10% | 几何10维 | 727 | 18.8811 | 0.64903 | 0.54145 | 0.12538 |
| 10% | 几何＋外观14维 | 727 | 19.2360 | 0.66130 | 0.52326 | 0.12042 |
| 15% | 几何10维 | 1065 | 19.0030 | 0.65751 | 0.52082 | 0.12255 |
| 15% | 几何＋外观14维 | 1069 | 19.5425 | 0.67004 | 0.49490 | 0.11687 |

Full 四指标均值改善，但 PSNR 只提高 0.0672 dB，中位数下降 0.0094 dB，仅 31/64 视图改善，不能解释为稳定的 Full 质量提升。原 ratio 的预算略有不同，因此同时提供 exact same-K 对照。

## 相同 K：14 维减 10 维

| K | 旧 PSNR | 新 PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | Δphoto | PSNR 改善视图 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 76 | 15.1738 | 14.9778 | -0.1959 | +0.00500 | -0.00302 | +0.00023 | 17/64 |
| 421 | 18.0528 | 18.2405 | +0.1877 | +0.00805 | -0.00007 | -0.00332 | 40/64 |
| 765 | 18.8851 | 19.2895 | +0.4044 | +0.01306 | -0.02106 | -0.00560 | 46/64 |
| 1110 | 18.9683 | 19.5178 | +0.5495 | +0.01316 | -0.02851 | -0.00584 | 47/64 |

## 同模型外观补偿 ON 减 OFF，几何补偿保留

| ratio | K | OFF PSNR | ON PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | Δphoto |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1% | 75 | 14.6961 | 14.9774 | +0.2813 | +0.01015 | -0.00491 | -0.00644 |
| 5% | 362 | 17.8219 | 17.8645 | +0.0427 | +0.00546 | +0.00126 | -0.00210 |
| 10% | 727 | 19.1617 | 19.2360 | +0.0744 | +0.00426 | +0.00068 | -0.00205 |
| 15% | 1069 | 19.4621 | 19.5425 | +0.0804 | +0.00470 | +0.00079 | -0.00218 |

## 外观与几何补偿幅度

| ratio | 选中点原 Opacity 均值 | 补偿后 Opacity 均值 | ΔRGB RMS | δXYZ RMS：10维→14维 |
|---|---:|---:|---:|---:|
| 1% | 0.8475 | 0.8765 | 0.0428 | 0.3567→0.2630 |
| 5% | 0.9146 | 0.9218 | 0.0200 | 0.1535→0.1255 |
| 10% | 0.9146 | 0.9205 | 0.0146 | 0.1073→0.0895 |
| 15% | 0.8853 | 0.8911 | 0.0123 | 0.0883→0.0740 |

![质量曲线](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/appearance_compensation_validation/quality_curves.png)

## 训练轨迹与验证边界

| 项目 | 几何10维 | 几何＋外观14维 |
|---|---:|---:|
| Gaussian | 7406 | 7563 |
| Anchor | 1775 | 1863 |
| Full Render | 4470 | 4612 |
| LoD Render | 8618 | 8654 |
| Master Adam 更新 | 13088 | 13266 |
| Selector 更新 | 8618 | 8654 |
| online 结束 Mapping | 6912 | 6734 |
| 补偿执行次数 | 6218 | 6254 |
| 实际补偿 CD 步数 | 0 | 0 |
| 网络参数，不含 Anchor Feature | 3404 | 3536 |

- **IMPLEMENTED / TESTED**：14 维选后补偿，真实 renderer 的新增通道梯度，Full 参数及 Adam 隔离，零初始化和饱和 logit 恒等，未选中属性及高阶 SH 保留，CPU/CUDA 保存加载，原生／Python／PLY／prepared renderer 一致。

- **TESTED**：旧 10 维真实模型输出 exact 兼容；旧可变成员 Mapping 和 CD 回归；14 项功能门控均通过。

- **PARTIALLY VERIFIED**：新增外观 CD 通过合成局部几何修正测试。本次实际补偿 CD 为 0 步，不能宣称已验证真实 Loop Closure 后的质量。

- 原 joint 为异步在线训练，实际人口、位姿、online 结束时刻和 Full／LoD 更新数不同。此轮只支持一个 scene/seed 的实测比较，不是固定地图的纯网络归因实验。

- 不单独归因于 Opacity 或颜色：两项本轮一起加入；外观 OFF 诊断也一起关闭。没有依据评价结果自动调 LR、增宽网络或加正则。

- 本地重算 22 条质量评价、1408 条逐视图记录及 52 条配对统计；341 个证据文件、177 个源文件 SHA256 核验通过。

## 交付

- [启用配置](H:/AIProjects/cartgs优化更细致版/mycartgs/cfg/selector/anchor_compensation.yaml)：新训练默认输出14维，TF关闭。

- [质量汇总](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/appearance_compensation_validation/evidence/validation/quality_summary.csv) / [逐视图](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/appearance_compensation_validation/evidence/validation/quality_per_view.csv) / [配对统计](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/appearance_compensation_validation/evidence/validation/paired_statistics.csv)

- [补偿统计](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/appearance_compensation_validation/evidence/validation/compensation_statistics.csv) / [验证记录](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/appearance_compensation_validation/evidence/validation/correctness.json) / [交付核验](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/appearance_compensation_validation/local_delivery_checks.json)

- [最终14维模型](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/appearance_compensation_validation/evidence/candidate_seed2026/10000_shutdown/ply/selector.pt)

没有 stage、commit、push；既有10维模型、源码快照和结果保留。完成后停止。
