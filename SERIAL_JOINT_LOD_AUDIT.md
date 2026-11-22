# 串行 Full + joint LoD：恢复 Master 梯度的质量对照

**结论：恢复 LoD 对 Master 的梯度后，四档 LoD 都明显优于串行隔离版；相对原 joint，中高预算改善，但最低预算仍退化。不能判为全面优于原 joint。**

`fr1_xyz / seed2026 / 10,000 Mapping` 已完成。相同 K 下，本轮相对隔离版的 PSNR 提升 **0.697／1.498／1.476／1.402 dB**；相对原 joint 为 **−1.296／+0.327／+0.556／+0.577 dB**（1%／5%／10%／15%）。Full PSNR 为 **22.617 dB**，与隔离版 22.610 dB 接近。

## 实现与实验定义

```yaml
Training.ultra_light_mode: 1
Training.post_topology_full_steps: 0
Training.serial_full_lod: 1
Training.serial_lod_detach_master: 0
```

每个 Mapping 先 Full Render、backward 并更新 Master，执行原有 Full 统计和必要拓扑维护，然后重新读取更新后的 Master，以 live Gaussian 属性执行一个轮转 ratio 的 LoD，更新 Master Gaussian、Selector、Anchor Feature 和 TF。

与串行隔离版相比，只关闭 Master detach；Full／LoD 顺序、ratio 顺序、分辨率、Loss、Cached Bernoulli、网络、SI/GI disabled、slot 停用、Anchor growing／pruning、TF／Change Distill 算法均保持原配置。Full100 关闭。

每个 Mapping 对 Master 最多执行两次 Adam 更新，Mapping／LR 时钟只推进一次。Full 阶段清除辅助参数残留梯度，仅更新 Master；LoD 阶段重新 backward，不累积上一阶段 Full 梯度。两阶段继续使用当前共享 Adam 参数组。

新开关 `Training.serial_lod_detach_master` 默认 `1`，保留旧串行隔离行为；设为 `0` 执行本轮串行 joint。`Training.serial_full_lod` 仍默认关闭，不改变原 joint 默认调度。

## 对照与验证标准

| 方法 | 每 Mapping 的 online 行为 | LoD 更新 Master | Full100 |
|---|---|---|---|
| 原 joint | 1%／5%／10%／15%／Full 五步轮转 | 是 | 关闭 |
| 串行隔离 | Full 后一个 LoD ratio | 否 | 关闭 |
| 串行 joint（本轮） | Full 后一个 LoD ratio | 是 | 关闭 |

原 joint 的 tail 已是每步 Full + joint LoD；两个串行版本的 tail 也保持串行，区别在 Master 梯度。LoD ratio 跨 online／tail 连续轮转。

本轮候选只训练一次：`fr1_xyz / seed2026 / 10,000 Mapping`，评价最终 checkpoint，不选最好 step、不调参数。共同 64 个视图沿用此前锁定集合，三个模型使用各自在线估计轨迹。

评价包含 Full、生产 Cached Bernoulli 和相同绝对 K，指标为 PSNR／SSIM／LPIPS／原生 photo。same-K 按固定 cached uniform 的 `logit(p_effective)-logit(u)` 顺序和 source row tie-break 取相同数量，仅作诊断，不替换训练或生产采样。

功能门控验证：LoD 对 Master 有 finite 图像梯度，确实改变参数与 Adam step／一阶／二阶动量；同时更新 Selector／Anchor／TF，保持 Full-only 拓扑统计和 ratio 顺序。旧串行隔离、Full100、旧 joint、pruning、Change Distill 继续回归。

## 功能与实际训练核验

八项门控全部通过：调度、串行 joint Mapping、串行隔离 Mapping、Full100 回归、旧 joint 关闭路径、Anchor pruning CPU／CUDA、Change Distill。

真实 CUDA Mapping 测试验证了 LoD 对 Master 的 finite 图像梯度，参数、Adam step 和两种动量的变化，以及 Selector／Anchor Feature／TF 更新。online／tail 的阶段计数和 ratio 顺序正确；LoD 不累积 Full-only 屏幕梯度与 Anchor opacity 访问统计。旧隔离测试仍验证 Master 在 LoD 中 exact 不变，logging 开关保持数值一致。

10,000 行真实 trace 逐行核验通过：

- 每步 Full 和 LoD 都执行，两个阶段各有一次 Master 更新，合计 **20,000 次**。
- 四个 ratio 各执行 2,500 次，Selector 共更新 10,000 次；TF 从原定阶段开始，共正常更新 7,000 次。
- Full100 配置与恢复窗口计数为零，无 non-finite LoD。
- 最终 native C++／Python／导出 Cached Bernoulli 在四个 ratio 的选择集合 exact 一致。

| 实际统计 | 原 joint | 串行隔离 | 串行 joint |
|---|---:|---:|---:|
| Mapping | 10,000 | 10,000 | 10,000 |
| Full Render | 4,960 | 10,000 | 10,000 |
| LoD Render | 8,741 | 10,000 | 10,000 |
| Master optimizer 更新 | 13,701 | 10,000 | **20,000** |
| Selector 正常更新 | 8,741 | 10,000 | 10,000 |
| TF 正常更新 | 6,341 | 7,000 | 7,000 |
| 最终 Gaussian | 42,640 | 44,130 | **50,220** |
| 最终 Anchor | 4,264 | 4,413 | 5,022 |
| 累计创建／删除 Anchor | 7,203／2,939 | 8,178／3,765 | 9,783／4,761 |
| SLAM online 结束 Mapping | 6,299 | 4,305 | 3,883 |

相对隔离版，Full、LoD、Selector、TF 的正常更新次数相同，增加的是 LoD 对 Master 的更新。但在线地图、轨迹和增长过程未冻结，最终人口增加 13.80%，不能当作纯固定地图的梯度 ablation。相对原 joint，人口增加 17.78%，Master 更新次数增加约 45.97%。

## Full-map 质量

64 个相同评价帧的均值。PSNR／SSIM 越高越好，LPIPS／photo 越低越好；photo 为原生 `0.8 L1 + 0.2 (1-fast_ssim)`，不含训练 opacity 正则。

| 方法 | 实际点数 | PSNR | SSIM | LPIPS | photo |
|---|---:|---:|---:|---:|---:|
| 原 joint | 42,640 | 22.0451 | 0.76060 | 0.27537 | 0.08426 |
| 串行隔离 | 44,130 | 22.6101 | 0.77671 | 0.25844 | 0.07750 |
| 串行 joint | 50,220 | 22.6174 | 0.77640 | 0.25280 | 0.07763 |

相对原 joint，Full 四指标均值改善，PSNR +0.5724 dB，53/64 个视图改善。相对隔离版，PSNR 仅 +0.0073 dB，SSIM／photo 略差，LPIPS 改善；**没有 Full 全面超过隔离版的证据**。

## 相同绝对 K 的主要质量结果

三个模型统一使用 `K=floor(r × min(42,640,44,130,50,220))`，没有用停用 slot 填充预算。

| LoD 档位 | K | 原 joint PSNR | 隔离 PSNR | 本轮 joint PSNR | 对隔离 Δ | 对原 joint Δ |
|---|---:|---:|---:|---:|---:|---:|
| 1% | 426 | 18.7211 | 16.7285 | 17.4253 | **+0.6969** | **−1.2958** |
| 5% | 2,132 | 20.6427 | 19.4714 | 20.9699 | **+1.4985** | **+0.3272** |
| 10% | 4,264 | 21.0693 | 20.1496 | 21.6256 | **+1.4760** | **+0.5563** |
| 15% | 6,396 | 21.1206 | 20.2949 | 21.6972 | **+1.4023** | **+0.5766** |

本轮相对隔离版，四档四指标均值均改善；PSNR 改善视图分别为 59／62／62／62 个（各 64 个）。

相对原 joint 的四指标差值：

| LoD | ΔPSNR | ΔSSIM | ΔLPIPS | Δphoto | PSNR 改善视图 |
|---|---:|---:|---:|---:|---:|
| 1% | −1.2958 | −0.01868 | +0.01861 | +0.01342 | 9/64 |
| 5% | +0.3272 | +0.00773 | −0.00322 | −0.00380 | 47/64 |
| 10% | +0.5563 | +0.01748 | −0.02340 | −0.00711 | 56/64 |
| 15% | +0.5766 | +0.01828 | −0.02499 | −0.00745 | 57/64 |

10%／15% 的收益较广泛。5% 虽然 LPIPS 均值改善，但中位数差值 **+0.00193（退化）**，仅 27/64 个视图改善，不能把这一档的 LPIPS 描述为稳定提升。1% 则四指标均退化。

same-K 仍以共同名义 ratio 输入 Selector，再从各自 cached margin 顺序取共同 K；没有将输入 ratio 重标定为 `K/N_model`。对于本轮更大的地图，426 点相当于约 0.848% 的存储人口，原 joint 则约为 0.999%。此诊断隔离了绝对点数差异，没有消除 ratio conditioning、地图和分布差异。

## 原生产 Cached Bernoulli

| LoD | 原 joint 点数／PSNR | 隔离点数／PSNR | 本轮 joint 点数／PSNR | 本轮对原 joint ΔPSNR |
|---|---:|---:|---:|---:|
| 1% | 447／18.8471 | 451／16.7223 | 524／19.4341 | +0.5871 |
| 5% | 2,135／20.6425 | 2,256／19.5568 | 2,535／21.2944 | +0.6519 |
| 10% | 4,216／21.0699 | 4,480／20.1630 | 5,028／21.6477 | +0.5778 |
| 15% | 6,157／21.1361 | 6,552／20.2984 | 7,409／21.7048 | +0.5686 |

生产同 ratio 四档四指标均值均优于两个对照，但点数也更多。尤其 1% 的生产结果改善，不能据此覆盖同 K 退化的事实：原 joint 从 447 缩至 K=426，本轮则从 524 缩至 426，采用的是预先锁定的同一诊断策略。

全部四指标的绝对值和 paired mean／median／p10／p90／improved fraction 已保存，没有挑选视图或最佳 checkpoint。

## 低 opacity 与预算占用

| 最终状态 | 原 joint | 串行隔离 | 串行 joint |
|---|---:|---:|---:|
| opacity < 1/255 | 29,272（68.65%） | 26,324（59.65%） | 33,892（67.49%） |
| opacity < 0.02 | 30,756（72.13%） | 28,340（64.22%） | 34,987（69.67%） |
| ≥8 个成员低于截断的 Anchor | 2,282（53.52%） | 1,882（42.65%） | 2,620（52.17%） |
| 10 个成员均低于截断的 Anchor | 870（20.40%） | 712（16.13%） | 778（15.49%） |
| 停用 slot | 20,739 | 19,991 | 25,351 |

相同 K 中，当前低于 `1/255` 的选中数：

| LoD | 原 joint | 隔离 | 本轮 joint |
|---|---:|---:|---:|
| 1% | 0 | 0 | 0 |
| 5% | 240 | 13 | 306 |
| 10% | 896 | 376 | 865 |
| 15% | 1,802 | 1,323 | 1,931 |

恢复联合梯度后，低 opacity 占比回升，但 LoD 明显优于隔离版；因此它与 LoD 质量并非单调关系。最低预算三者都未选择当前低于截断的点，仍存在明显质量差异。上述是当前快照，不把未充分访问的新点判成长期失效；本轮没有改 slot 停用或 pruning。

## 结论与边界

1. **恢复 Master LoD 梯度是否有帮助：本次 YES。** 相同 K 的四档均明显优于隔离版，支持显式 Gaussian 参与子集重建适配的价值。
2. **串行 joint 是否全面优于原 joint：NO。** 5%／10%／15% 改善，1% 退化，5% LPIPS 逐视图也不稳定。
3. **Full 是否保持：均值接近隔离版，并优于原 joint。** 不能将其解读为固定人口、固定轨迹下的单因素收益。
4. **默认切换：HOLD。** 保留实验开关，没有自动切换默认或扩大训练搜索。

限制：单场景、单 seed；SLAM 与 Mapping 异步，online 结束时刻、采样、增长、pruning、相机轨迹和 native tail 占比不同。固定的是 Mapping 预算，不是相同 Master optimizer 次数或人口。64 帧未严格划分 Bank／non-Bank，不能声称 held-out 泛化。

单次 wall time 为原 joint 73.627 s、隔离 86.293 s、本轮 96.396 s（相对隔离约 +11.7%，相对原 joint 约 +30.9%）。这是含在线轨迹差异的单次总时长，不是严格性能 benchmark；没有可比较峰值显存数据。

## 交付与证据

- [配置读取](H:/AIProjects/cartgs优化更细致版/mycartgs/src/gaussian_mapper.cpp:542)
- [live／detach 路由](H:/AIProjects/cartgs优化更细致版/mycartgs/src/gaussian_mapper.cpp:1853)
- [真实 joint 梯度测试](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/serial_joint_lod_mapping_test.cpp)
- [八项 correctness](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/serial_joint_lod_validation/evidence/validation/correctness.json)
- [quality_summary.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/serial_joint_lod_validation/evidence/validation/quality_summary.csv)
- [quality_per_view.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/serial_joint_lod_validation/evidence/validation/quality_per_view.csv)
- [paired_statistics.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/serial_joint_lod_validation/evidence/validation/paired_statistics.csv)
- [optimization_diagnostics.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/serial_joint_lod_validation/evidence/validation/optimization_diagnostics.csv)
- [逐 Mapping trace](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/serial_joint_lod_validation/evidence/joint_seed2026/serial_full_lod_trace.csv)
- [独立核验](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/serial_joint_lod_validation/local_delivery_checks.json)
- [相对隔离版代码差异](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/serial_joint_lod_validation/task.patch)

状态：实现 **IMPLEMENTED**；功能、梯度和本次在线质量 **TESTED**；跨场景／seed 结论 **NOT TESTED**。

382 个证据文件 SHA、21 个本地／实验源码 SHA、10,000 行 trace、三臂 27 个评价项的 1,728 条逐视图数据和 72 条 paired 统计，全部在本地独立复算通过。对照模型与 production 源码 SHA 保持不变。服务器实验目录：`/home/prmi/zwj/serial-joint-lod-20261021`，训练／评价进程已结束。

本轮未 stage、commit、push。
