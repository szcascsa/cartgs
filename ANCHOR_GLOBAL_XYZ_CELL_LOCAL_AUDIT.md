# Global XYZ + 当前 Anchor 局部 Offset 实验

## 实际改动

Selector 输入变为 30 维：

`Anchor Feature16 + Local Offset3 + Global XYZ3 + Scale3 + SH DC3 + Opacity1 + log(ratio)1`。

网络仍是两层 32 宽 ReLU，共享参数从 2,018 增至 2,114。Global XYZ 使用 TF 前显式 Gaussian 的原始世界坐标；未增加 AABB 归一化、位置编码或相机输入。既有 Scale、SH DC、Opacity 的含义和 joint LoD 梯度分流不变。

对所属叶子 Cell `C`，现在使用：

`offset_i = (xyz_i - center_C) / h_C`，`h_C = 0.1 / 2^level_C`。

细化时更新子中心和子尺度，重新计算 Offset；子 Feature 和已存在的两套 Adam 状态复制父行，之后独立优化。**不再要求 split 瞬间分数连续。** Feature 继承相同，不代表网络输出必然相同，因为 Offset 输入已改变。

仍只细化 occupancy>128 的叶子，最多 L0/L1/L2，只创建非空子节点；不直接增删 Gaussian。新增、prune、几何事件、frozen teacher 和 Change Distill 沿用已有生命周期接入。没有改变 Loss、LR、Ratio 调度、Cached Bernoulli、TF 或 Full Render，也没有启用 SI/GI。

新布局为 v4，checkpoint 保存实际 `centers` 和每节点 `offset_scales`；旧 27 维 v3 checkpoint 保持原 root-offset 推理语义。启用新路径需 `Selector.anchor_gaussian_attributes: 1`、`Selector.anchor_global_xyz: 1`；自适应组另设 `Selector.anchor_adaptive: 1`。

## 验证

10 项 gate 全部通过：CPU/CUDA 新坐标与 Feature/Adam 继承、CPU/CUDA native/Python probability 与 Cached Bernoulli mask 一致性、CPU/CUDA 固定 30 维、CPU/CUDA 旧 27 维回归、Mapping、Change Distill。Global XYZ 对输出和输入权重梯度有效；子中心/尺度公式、几何冻结、clone/load 和层级限制均通过。

950 个点的合成测试中，split 后 probability 最大变化为 CPU `0.00927`、CUDA `0.01328`；这次是允许的坐标变化，不再作为连续性失败。

更明显的扰动出现在训练完成的固定 30 维模型上：仅在其冻结地图上即时细化、复制 Feature 并重算 Offset，两个 seed 的单点 probability 最大变化达到 `0.999997/0.998310`；15% ratio 下，全体 population 的 Cached Bernoulli 状态变化比例为 `3.077%/2.113%`。Feature 本身仍 exact 继承。这是一项训练后固定模型的离线扰动诊断，不代表实际在线每次 split 都有同等幅度，也不能证明它单独造成了最终训练回退。

## 实验协议

`fr1_xyz`，seed 2026/2027，固定/自适应各两次全新训练，每次 10,000 Mapping；评价最终 checkpoint。两组都使用相同 30 维网络。另与上一轮 27 维固定/自适应最终模型比较。

评价前按轨迹覆盖锁定共同 GT 视图和统一 K；不按质量筛选视图、step 或 seed。Primary 为生产 Cached Bernoulli，记录实际点数；secondary 按 probability/source-row 排序作同 K 诊断，没有改变训练或导出策略。如果 K 与旧实验不一致，仅重新评价旧模型的同 K 选择，不重新训练旧模型。

最终共同视图为 792 个，统一 K 为 `233/1168/2336/3504`。完成新模型 36 项评价，另重评旧模型的 16 项 same-K，旧模型 20 项生产预算结果复用；本轮共执行 52 项渲染评价。没有额外训练旧模型。

## 质量结果

**结论：Global XYZ 在低预算下有收益信号；自适应细化并重算 Offset 未稳定优于同一套 30 维固定 Anchor。**

- 30 维固定−27 维固定，same-K 的 1% PSNR 为 `+1.643/+2.169 dB`，5% 为 `+0.982/+0.552 dB`；10%/15% 仍存在 seed 分歧。固定方案的 Offset 公式未改变，这组对照主要检验 Global XYZ 输入及相应网络初始化带来的端到端变化。
- 当前 30 维自适应−固定：生产 LoD PSNR 提升 `3/8`，SSIM `4/8`，LPIPS `7/8`；same-K 分别为 `1/8、3/8、4/8`。其中 same-K 2026/10% PSNR 仅 `−0.005 dB`，接近持平；不能把所有负号都解释为显著退化。
- 新自适应−旧自适应：same-K PSNR 提升 `6/8`，SSIM `5/8`，LPIPS `7/8`，但高 ratio 的 PSNR 在两个 seed 上方向相反；生产预算下三指标均只提升 `5/8`。不支持稳定、全面的质量优势。
- 新自适应 Full PSNR 比新固定分别低 `0.032/0.105 dB`。其最终 Master 点数在 seed 2026 增加、2027 减少，表明在线 joint 优化轨迹也发生了变化。

新路径已按要求实现并保留；本轮证据更支持继续保留固定 Anchor 对照，尚不支持将“重算子 Offset 的自适应”视为确定改进。没有据结果自动回滚实现、调 LR、增加网络或增加补偿规则。

当前 30 维：自适应−固定。ΔPSNR 分 seed 报告，ΔSSIM/ΔLPIPS 为两个 seed 的平均差；PSNR、SSIM 越高越好，LPIPS 越低越好。完整绝对值、实际选点数和逐视图数据在 CSV 中。

| 策略 | 级别 | ΔPSNR 2026 | ΔPSNR 2027 | 平均 ΔSSIM | 平均 ΔLPIPS |
|---|---|---:|---:|---:|---:|
| Cached | Full | -0.032 | -0.105 | -0.00019 | +0.00005 |
| Cached | 1% | +0.034 | -0.056 | +0.00167 | -0.00924 |
| Cached | 5% | -0.062 | -0.581 | -0.03629 | -0.01073 |
| Cached | 10% | +0.298 | -0.201 | -0.02608 | -0.01391 |
| Cached | 15% | +0.303 | -0.279 | -0.01976 | -0.01142 |
| same-K | 1% | -0.014 | -0.028 | +0.00085 | -0.00665 |
| same-K | 5% | -0.222 | -0.637 | -0.03980 | -0.00255 |
| same-K | 10% | -0.005 | -0.431 | -0.03221 | -0.00489 |
| same-K | 15% | +0.264 | -0.274 | -0.02010 | -0.01029 |

新自适应 30 维−旧自适应 27 维：

| 策略 | 级别 | ΔPSNR 2026 | ΔPSNR 2027 | 平均 ΔSSIM | 平均 ΔLPIPS |
|---|---|---:|---:|---:|---:|
| Cached | Full | +0.106 | -0.190 | +0.00223 | +0.00475 |
| Cached | 1% | +0.208 | +0.457 | +0.01802 | -0.00697 |
| Cached | 5% | +0.864 | -0.403 | -0.00281 | -0.01224 |
| Cached | 10% | +1.158 | -0.456 | +0.00354 | -0.00870 |
| Cached | 15% | +1.295 | -0.380 | +0.01172 | -0.01419 |
| same-K | 1% | +0.417 | +1.875 | +0.04569 | -0.00930 |
| same-K | 5% | +0.579 | +0.034 | +0.00498 | -0.02012 |
| same-K | 10% | +1.012 | -0.614 | +0.00263 | -0.01458 |
| same-K | 15% | +1.263 | -0.485 | +0.00986 | -0.01557 |

## 层级与开销

| 方案/seed | Gaussian | Feature 数 | 非空叶 L0/L1/L2 | 最大叶 occupancy | 最终 L2 超限 |
|---|---:|---:|---|---:|---:|
| fixed/2026 | 25248 | 1321 | 1321/0/0 | 472 | 0 |
| adaptive/2026 | 27536 | 2914 | 1263/611/336 | 127 | 0 |
| fixed/2027 | 24566 | 1223 | 1223/0/0 | 521 | 0 |
| adaptive/2027 | 23361 | 2964 | 1156/554/284 | 124 | 0 |

保留的内部/历史节点不参与 Gaussian 直接 lookup，但维护已有细分历史和后续成员的 Feature 继承，因此 Feature 总数超过当前非空叶子数；本轮没有增加空节点预分配或历史节点回收算法。最终两组均无 L2 超限。

RTX 3090，在全部评价结束、GPU 无并发训练/评价时，使用相同 `N=23,361`、ratio=10%、预计算属性和 source rows，50 次预热、7×500 次 CUDA-event forward：

| 布局/方案 | ms 2026 | ms 2027 |
|---|---:|---:|
| 27/fixed | 0.2592 | 0.2592 |
| 27/adaptive | 0.2595 | 0.2595 |
| 30/fixed | 0.2708 | 0.2725 |
| 30/adaptive | 0.2712 | 0.2745 |

30 维相对 27 维的 forward 中位开销约 `+4.5%–5.8%`；30 维内自适应相对固定约 `+0.16%/+0.74%`。这是等价 LibTorch 推理测量，不是 Mapping 总时延；实际 native C++ 全人口计时另存于 `runtime.csv`。

| 方案/seed | 训练秒 | 峰值 allocated MiB | Full renders | LoD renders | 累计细化 host ms |
|---|---:|---:|---:|---:|---:|
| fixed/2026 | 68.93 | 539.816 | 4739 | 8685 | 0.0 |
| adaptive/2026 | 70.47 | 560.628 | 4844 | 8711 | 224.8 |
| fixed/2027 | 68.73 | 536.52 | 4752 | 8688 | 0.0 |
| adaptive/2027 | 69.52 | 554.28 | 4806 | 8702 | 198.9 |

自适应两次训练的树处理累计约 `224.8/198.9 ms`，完成 156/139 个父 split。整图显存包括不同地图、TF、render 和优化器轨迹，不能把整图峰值差全部归因于 Selector；Feature 本体固定为每个存储节点 64 bytes，额外还有 Adam 状态、中心/尺度和历史树存储。未改变 optimizer 实际更新次数规则，但异步在线/tail 分配不同，详细时钟在结果文件中。

## 结论边界与文件

这是单场景、双 seed 的在线端到端实验。共同视图和同 K 消除了部分评价及预算差异，但不冻结地图、相机轨迹、densification、在线/tail 比例、TF 或实际 Full/LoD 更新次数。30 维初始化也改变了随机数消耗。新自适应与旧自适应的比较同时改变 Global XYZ 和 Offset 参考，不能把结果归因于其中一项。

细化改变局部坐标尺度，也改变 XYZ 经 Offset 分支的梯度尺度；这是本次设计的直接结果，没有额外调 LR 或补偿梯度。功能验证通过不等于已经证明重建质量更好。

本地结果目录：`tests/anchor_selector/xyz_local_validation/evidence/validation/`，包含 `quality_summary.csv`、`quality_per_view.csv`、`paired_statistics.csv`、`comparison_with_previous.csv`、`historical_quality_summary.json`、`anchor_statistics.csv`、每模型 Cell/mapping、`runtime.csv`、`matched_n_forward.json`、`correctness.json` 和源码 SHA 清单。

模型保存在隔离实验目录 `/home/prmi/zwj/anchor-xyz-local-20261020/{fixed,adaptive}_seed{2026,2027}/10000_shutdown/ply/`。旧模型及旧评价保留。

代码和报告均未暂存、未提交、未 push；HEAD 保持细化前基线 `456c410f2a638effd7c61f38868e20b2e076ba45`。本轮完成后停止，不扩大网络或调参。
