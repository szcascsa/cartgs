# Anchor Selector 三层 occupancy 八叉树细化

## 范围与基线

细化前基线 commit：`456c410f2a638effd7c61f38868e20b2e076ba45`，当前分支 `redundancy-aware-si`。基线选择性提交 55 个相关源码、测试、报告和小型证据文件；没有纳入模型、渲染图片、压缩包或无关研究产物。未 push。本轮细化代码不 stage、不 commit。

本轮保留 27 维 `feature16 + offset3 + activated scale3 + SH DC3 + activated opacity1 + log(ratio)1`，网络仍为 `27→32 ReLU→32 ReLU→2`。Loss、Cached Bernoulli、Ratio 轮转、joint LoD 更新、Master-only Full、TF 和 Change Distill 算法均未改变；实验保持 SI/GI disabled、maturity=1。没有梯度阈值、额外特征、额外损失或阈值搜索。

## 已实现

- `Selector.anchor_adaptive: 1` 启用；默认 `0` 保留固定 Anchor 对照。启用时要求 Anchor architecture、27 维输入、`h=0.1`。
- L0/L1/L2 边长固定为 `0.1/0.05/0.025`。叶 occupancy 严格 `>128` 才细分；只根据当前成员 XYZ 和父 Cell 的三个中分平面创建非空子 Cell。L2 不再细分。
- 保存 `(level, integer XYZ key)`、branch 标记和 Gaussian→Feature 行映射。Feature 仍为每个节点 16 维；只改变 Feature 归属，不增删或修改任何 Gaussian 属性。
- 子节点通过重复父行初始化，Feature 与所有已存在 Adam owner 的 `exp_avg/exp_avg_sq`、可选 AMSGrad maximum 一起复制，Adam step 保持不变。替换真实 autograd leaf，并同步修改两个 optimizer 的参数别名。
- `centers` 继续保存原 L0 Offset 参考中心；所有层仍除以原 `h`。实际 Cell 中心由 key/level 独立恢复。细化不把 Offset 改成 child-relative 坐标。
- 新增、clone/split 后沿现有树下沉；prune/reorder 只改变 Gaussian 映射，不折叠已细分拓扑。父节点及曾经占用后变空的节点作为历史/继承模板保留，**没有预先创建空子节点**。下文分别报告存储 Feature 数、内部/历史空节点和当前非空叶子，不能把它们混为一谈。
- 普通 Gaussian optimizer 更新沿用既有 Anchor 身份语义，Offset 使用实时 XYZ；不在每次梯度更新后全局重新 voxelize。细化当前成员时保持所属 root 的参考坐标。显式几何修正/loop closure 时按 world keys 和已有层级重新关联；Feature/Adam 延续既有按成员运输、碰撞平均的处理，再运行现有 Change Distill。
- append/几何事件仅标记待处理；细化在 render lock 内、当前 backward/optimizer 完成后、下一次 LoD forward 前执行。tail Full 的 densification 后、tail LoD 前也有安全边界；导出前处理最后一次待细化状态。没有逐 Mapping 全量重建 Scaffold。
- checkpoint 保存层级与映射；frozen teacher/guard clone、C++ 加载和 Python 推理支持新布局，并保留固定 Anchor checkpoint 兼容。

主要源码：`improvements/selector/anchor_bank.h`、`src/gaussian_mapper.cpp`、`python/selector.py`。网络文件 `gumbel_network.h` 相对基线没有修改。

## 验证

CPU/CUDA 八叉树、CPU/CUDA Python parity、固定布局 CPU/CUDA 回归、自适应/固定 Mapping、已细化 Anchor 的局部几何 Change Distill，共 9 项 gate 全部通过。

950 个 Gaussian、四个测试 Ratio 的细化前后 logits 和 probability 最大绝对误差均为 **0**；原 Offset 和逐 Gaussian Feature 输入 exact 相同。两个 Adam owner 的参数别名、Feature、两个动量、AMSGrad 和 step 继承通过。128 不拆、129 递归拆、仅创建非空子节点、L2 仍超限时保留、负坐标、prune 后重入、几何迁出再迁回、归档和 frozen teacher 均通过。显式 XYZ exact 不变。

Mapping 回归验证沿用在线 LoD/Full 顺序、tail Full+单个轮转 joint LoD、Gaussian/Selector/TF 梯度分流、Full 不更新 Selector 和几何事件蒸馏。它证明接入正确，不等于证明重建质量更好。

收尾补充了 Selector 对象未创建时的空指针保护，并重新编译、复跑固定/自适应 Mapping 回归；本轮启用 Selector 的训练路径和数学结果不变。训练与最终源码 SHA 清单分别保留。

## 质量对照协议

`fr1_xyz`，seed 2026/2027，固定/自适应各两次新训练，每次 10,000 Mapping；只评价最终 checkpoint。其他配置完全相同。四条轨迹共同覆盖的全部 792 个 GT 视图在评价前锁定，不按图像质量筛选。

Primary 使用原生产 Cached Bernoulli，报告实际点数；secondary 对两组、两个 seed 都使用统一绝对 K：`271/1358/2716/4074`，由四个最终 Master 的最小 N 决定，评价前锁定。secondary 仅用 probability 排序作 deterministic same-K 诊断，没有改变训练或正式导出策略。

在线 SLAM 具有异步性：即使 seed、Mapping 次数一致，地图、相机轨迹、densification、在线/tail 比例和实际 LoD/Full 更新次数仍可能不同。same-K 只消除预算差异，不能消除地图和优化轨迹差异。本实验只支持单场景、双 seed 的初步端到端结论。

## 质量结果：MIXED，尚未证明整体 LoD 改善

结论：**occupancy 细化有效且连续；1% 的 PSNR 在两个 seed 上都改善，但没有跨 ratio、跨指标的稳定质量优势，不建议据此替换默认固定 Anchor。** 不继续调阈值或扩大实验。

Primary 为原生产 Cached Bernoulli。SSIM/LPIPS 列是 Adaptive−Fixed；PSNR/SSIM 越高越好，LPIPS 越低越好。完整绝对值及逐视图结果保存在 `quality_summary.csv`、`quality_per_view.csv`。

| seed | 级别 | 固定点数 | 自适应点数 | 固定 PSNR | 自适应 PSNR | ΔSSIM | ΔLPIPS |
|---|---|---:|---:|---:|---:|---:|---:|
| 2026 | Full | 27307 | 27548 | 22.099 | 22.105 | -0.00070 | -0.00547 |
| 2026 | 1% | 288 | 273 | 16.324 | 16.705 | +0.00440 | -0.01935 |
| 2026 | 5% | 1438 | 1429 | 18.560 | 18.492 | -0.02515 | -0.00302 |
| 2026 | 10% | 2846 | 2856 | 19.153 | 19.376 | -0.01854 | -0.01771 |
| 2026 | 15% | 4142 | 4208 | 19.226 | 19.464 | -0.01841 | -0.01998 |
| 2027 | Full | 27166 | 27857 | 22.078 | 22.105 | -0.00507 | +0.00221 |
| 2027 | 1% | 279 | 286 | 16.303 | 16.834 | -0.00517 | +0.00062 |
| 2027 | 5% | 1443 | 1474 | 19.073 | 18.986 | -0.02767 | -0.00475 |
| 2027 | 10% | 2834 | 2929 | 19.639 | 19.572 | -0.03320 | -0.01222 |
| 2027 | 15% | 4093 | 4218 | 19.798 | 19.634 | -0.03524 | -0.00910 |

统一 K 的诊断：

| seed | Ratio | K | ΔPSNR | ΔSSIM | ΔLPIPS | PSNR 改善视图比例 |
|---|---|---:|---:|---:|---:|---:|
| 2026 | 1% | 271 | +0.653 | +0.00940 | -0.02040 | 87.2% |
| 2026 | 5% | 1358 | -0.050 | -0.02452 | -0.00261 | 43.8% |
| 2026 | 10% | 2716 | +0.306 | -0.01774 | -0.01907 | 74.0% |
| 2026 | 15% | 4074 | +0.245 | -0.01766 | -0.02128 | 64.8% |
| 2027 | 1% | 271 | +0.554 | -0.00732 | -0.00349 | 81.6% |
| 2027 | 5% | 1358 | -0.106 | -0.03004 | -0.00522 | 39.3% |
| 2027 | 10% | 2716 | -0.098 | -0.03512 | -0.01178 | 39.5% |
| 2027 | 15% | 4074 | -0.173 | -0.03601 | -0.00774 | 34.2% |

- 1%：生产 PSNR 分别 `+0.381/+0.531 dB`；same-K 为 `+0.653/+0.554 dB`，改善视图占 `87.2%/81.6%`，说明这部分收益不能仅归因于多选了 Gaussian。但 SSIM 的 seed 一致性仍不足。
- 5%：两个 seed 的 PSNR 均略降，SSIM 均下降；10%/15% 的 PSNR 在两个 seed 间方向不一致。
- LoD 生产 PSNR 改善 `4/8`，LPIPS 改善 `7/8`，SSIM 改善 `1/8`；same-K 分别为 `4/8、8/8、1/8`。不能把 LPIPS 改善表述为全面质量改善。
- Full PSNR 分别 `+0.006/+0.028 dB`，SSIM 均略降，LPIPS 一好一差。没有证据认为 Full-map 质量得到稳定提升。

## 层级、occupancy 与存储

| 方案/seed | Gaussian | 存储 Feature 数 | 当前非空叶 L0/L1/L2 | 内部节点 | 曾占用后变空的叶 | 最大叶 occupancy | 最终 L2 超限 Cell |
|---|---:|---:|---|---:|---:|---:|---:|
| fixed/2026 | 27307 | 1399 | 1399/0/0 | 0 | 0 | 634 | 0 |
| adaptive/2026 | 27548 | 2934 | 1302/530/290 | 131 | 681 | 128 | 0 |
| fixed/2027 | 27166 | 1401 | 1401/0/0 | 0 | 0 | 616 | 0 |
| adaptive/2027 | 27857 | 2987 | 1298/561/316 | 146 | 666 | 126 | 0 |

自适应全部存储节点的 L0/L1/L2 数分别为 `2025/607/302`、`2007/646/334`。保留内部/历史节点使 Feature 数超过当前非空叶子数，这是本实现维护细分历史和新成员继承的显式内存成本。最终两组没有 L2 超限；这不表示训练过程中从未发生过超限。L2 超限保留由专门构造的 129/150 个近重合点测试验证。

固定/自适应最终 Gaussian 总数的差别来自既有 densification/pruning 和在线优化轨迹；八叉树 split 本身仅复制 Feature 与更新映射，不生成 Gaussian。

两个真实固定 checkpoint 的冻结地图另做即时细化：分别产生 49、60 次父节点 split，全部 Gaussian 的原 Feature/Offset 输入 exact 不变；四个 ratio 的最大 probability 误差为 **0**，Cached Bernoulli selected set exact 不变。没有修改 checkpoint 文件，也未对该即时细化状态重新训练或用于正式质量结果。

## 计算与显存成本

GPU 为 RTX 3090。下表 forward 使用相同 `N=27,166`、ratio=10%、预先计算的 Scale/DC/Opacity、显式 source rows；质量评价/训练完全结束后独立执行，CUDA event 计时，每组 50 次预热、7×500 次 forward，两个 seed 交换先后顺序。该项为 Python 等价 LibTorch 路径；实际全人口 C++ native forward 也单独保存在 `runtime.csv`，不能将两种计时口径混比。

| seed | 固定 forward ms | 自适应 forward ms | 差异 | 固定训练秒 | 自适应训练秒 | 固定/自适应峰值 allocated MiB | 累计细化 host ms |
|---|---:|---:|---:|---:|---:|---|---:|
| 2026 | 0.2584 | 0.2591 | +0.26% | 67.79 | 68.42 | 549.958 / 542.737 | 208.3 |
| 2027 | 0.2600 | 0.2615 | +0.57% | 68.14 | 69.72 | 545.602 / 556.557 | 224.2 |

同 N forward 增量仅约 `0.26%/0.57%`，处于重复计时波动量级，没有观察到明显 forward 成本增加，也不主张加速。自适应新增 Feature 本体分别约 `95.9/99.1 KiB`；加上一个 Adam owner 的两个动量后为 `287.8/297.4 KiB`。若两个 owner 均拥有状态，Feature+两套 Adam 状态的增量上界为 `479.7/495.6 KiB`，另有布局 buffer、梯度及 CPU 历史树开销。整图 allocated 峰值一降一升，不能把它当作纯 Selector 增量。

两次训练共 80/82 次待细化处理，累计 split 131/146 个父节点；host 观测总耗时 `208.3/224.2 ms`，p95 `4.13/4.57 ms`。该计时包含 CPU/GPU 往返等待和调度，非独立 CUDA kernel benchmark；也不涵盖已有 append/几何重新关联的所有耗时。

训练墙钟分别增加约 `0.93%/2.32%`。以下实际训练轨迹不同，因此不能将墙钟差全部归因于树管理：

| 方案/seed | 在线结束 Mapping | tail Mapping | Full renders | LoD renders |
|---|---:|---:|---:|---:|
| fixed/2026 | 6710 | 3290 | 4632 | 8658 |
| adaptive/2026 | 6595 | 3405 | 4724 | 8681 |
| fixed/2027 | 6651 | 3349 | 4679 | 8670 |
| adaptive/2027 | 6495 | 3505 | 4804 | 8701 |

## 交付与验证边界

- **IMPLEMENTED / TESTED**：L0–L2/128、非空子节点创建、Feature/两套 Adam 继承、瞬时分数连续、append/prune/reorder、显式几何事件、TF/CD、旧/新 checkpoint 与 Python 推理。
- **TESTED**：一个 TUM 场景、两个 seed、四次新训练、36 项最终 checkpoint 质量评价；792 个共同视图；生产实际预算与统一 K 对照；同 N forward/显存统计。51 个相关源码文件的本地/实验副本 SHA 一致，评价前后模型 SHA 不变。
- **PARTIALLY VERIFIED**：跨场景质量、长期在线地图增长下历史节点的内存累积、复杂真实 loop closure 分布的稳健性；本轮只有针对性几何事件测试和单序列端到端证据。
- **NOT IMPLEMENTED**：梯度驱动 split、threshold sweep、合并/历史节点回收、新损失、新网络/输入、训练采样算法改动、完整 optimizer/时钟 checkpoint resume。没有据功能测试宣称质量优越。

结果根目录：`tests/anchor_selector/octree_validation/evidence/`。其中 `validation/` 包含 `correctness.json`、`quality_summary.csv`、`quality_per_view.csv`、`paired_statistics.csv`、`anchor_statistics.csv`、四个 `*_cells.csv` / `*_mapping.npz`、`runtime.csv`、`matched_n_forward.json`、`audit_summary.json`、源码 SHA 清单和日志。

实验模型保存在隔离目录 `/home/prmi/zwj/anchor-octree-20261020/{fixed,adaptive}_seed{2026,2027}/10000_shutdown/ply/`，保留 Selector、TF、显式 Gaussian 和 metadata；没有覆盖旧基线或部署到生产目录。

本轮新增细化代码和报告保持未暂存、未提交；HEAD 仍为上述细化前基线，未 push。完成后停止，等待验收。
