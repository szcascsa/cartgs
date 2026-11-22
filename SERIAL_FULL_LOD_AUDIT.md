# 串行 Full Master 与隔离 LoD 优化实验

**结论：本次配置改善 Full-map，但明显损害四档 LoD，不建议替代当前 joint 训练。**

`fr1_xyz / seed2026 / 10,000 Mapping` 中，Full PSNR 提高 **0.565 dB**，四项 Full 指标均改善；相同绝对 K 下，1%／5%／10%／15% 的 PSNR 分别下降 **1.993／1.171／0.920／0.826 dB**，SSIM、LPIPS、photo 也全部退化。低 opacity 比例下降，未转化为 LoD 质量改善。

## 实验定义与实现

```yaml
Training.ultra_light_mode: 1
Training.post_topology_full_steps: 0
Training.serial_full_lod: 1
```

Full100 本轮关闭。新开关默认 `0`，本次实验明确启用，不重新启用已废弃的 mode2。

每次 Mapping 顺序为：

1. 按现有 GausPyramid 执行 Full Render、原 Full Loss 和 backward。
2. 清除 Selector／TF 残留梯度，仅更新 Master Gaussian。
3. 执行该 Full 阶段原有的统计、计划中的 Anchor growing／pruning 和维护。
4. 从**已更新、已完成必要拓扑维护的 Master** 重新读取 XYZ／Scale／Rotation／Opacity／SH，全部 detach。
5. 执行一个轮转 ratio 的 LoD：`1% → 5% → 10% → 15% → …`，跨 online／tail 连续。
6. 使用原 Cached Bernoulli、Opacity ST、Selector Loss 和 TF 路径，仅更新 Selector MLP／Anchor Feature／TF。Master 无梯度，Adam step 和动量不因 LoD 变化。

每对 Full／LoD 只推进一次 Mapping 和 Gaussian LR 时钟。沿用已有分辨率：Full 按 GausPyramid，online LoD 使用原 0.5 scale，tail LoD native。本轮没有改网络、特征、Loss、Ratio Budget、采样、SI/GI disabled、slot 停用、Anchor growing／pruning、TF 或 Change Distill 算法。

当前优化器物理存储仍复用共享 Adam 参数组，隔离依赖 detached 输入与 undefined gradient；没有把 Gaussian／Anchor 参数状态复制到另一套优化器。实际测试同时验证参数、Adam step、一阶和二阶动量，不能仅凭 `detach` 代码推断隔离成功。运行时额外检查 LoD 不得产生 Master 梯度。

## 正确性验证

七项门控全部通过：调度、串行真实 CUDA Mapping、Full100 回归、旧 joint 关闭路径回归、Anchor pruning CPU／CUDA、Change Distill。

串行真实 Mapping 测试覆盖：

- 每步 Full 后跟 LoD，Master 每 Mapping 只更新一次。
- LoD 更新 Selector／Anchor Feature／TF。
- 单次 LoD 内，Master 参数、Adam step、一阶／二阶动量 **exact 不变**。
- 串行 Mapping 与独立 Full-only 对照的 Master 参数／动量一致（独立 CUDA 执行比较 `rtol=1e-5, atol=1e-6`），Adam step exact 一致。
- online／tail ratio 连续，拓扑维护后仍可 LoD，Full100 不触发。
- logging 开关不改变优化行为。

## 实验范围

主基线为上一轮 `slot_deactivation=1`、无 Full100 的 joint 模型；候选只改为串行隔离训练。场景 `fr1_xyz`，seed `2026`，总预算 `10,000 Mapping`，候选训练一次，评价最终 checkpoint。共同评价视图沿用此前已锁定的 64 帧，不按本轮质量选帧或选 checkpoint。

同时评价生产 Cached Bernoulli 与相同绝对 K。same-K 诊断按固定 cached uniform 的 `logit(p_effective)-logit(u)` 排序、source row 破 tie；它保持固定随机门限顺序，只用于消除数量差异，不替换生产采样，也不是纯 Selector score Top-K。

## 真实训练记录

| 统计项 | 原 joint 基线 | 串行隔离 |
|---|---:|---:|
| Mapping | 10,000 | 10,000 |
| Full Render | 4,960 | 10,000 |
| LoD Render／Selector 正常更新 | 8,741 | 10,000 |
| Gaussian optimizer 更新 | 13,701 | 10,000 |
| LoD 导致的 Master 更新 | 原路径允许 | **0** |
| TF 正常更新 | 6,341 | 7,000 |
| 最终 Gaussian／Anchor | 42,640／4,264 | 44,130／4,413 |
| 累计创建／删除 Anchor | 7,203／2,939 | 8,178／3,765 |
| SLAM online 结束 Mapping | 6,299 | 4,305 |

10,000 行 trace 逐行验证通过：每步都有 Full 和 LoD，Master 每步只在 Full 更新；四个 ratio 各 **2,500 次**，没有 non-finite LoD，Full100 窗口计数为零。Full 阶段统计仍只使用 Full 可见访问和屏幕空间梯度；LoD 不污染这些统计。Full 次数变化会影响相同窗口内累积的统计量，但没有更改增长／pruning 公式或阈值。

这里固定的是 Mapping 预算，不是渲染次数或墙钟预算。串行总渲染数为 20,000，基线 13,701；同时取消 LoD 对 Master 的更新，使 Master Adam 总更新数反而下降。

## 质量对照

以下均为 64 个相同帧的平均值，差值为“串行隔离 − 原 joint”。PSNR／SSIM 越高越好，LPIPS／photo 越低越好。photo 使用原生 `0.8 L1 + 0.2 (1-fast_ssim)`，不含训练 opacity 正则。

### Full-map

| 方法 | 实际点数 | PSNR | SSIM | LPIPS | photo |
|---|---:|---:|---:|---:|---:|
| joint | 42,640 | 22.0451 | 0.76060 | 0.27537 | 0.08426 |
| 串行隔离 | 44,130 | 22.6101 | 0.77671 | 0.25844 | 0.07750 |
| 差值 | +1,490 | **+0.5651** | **+0.01612** | **−0.01693** | **−0.00677** |

Full PSNR 在 52/64 个视图改善。新地图点数增加 3.49%，在线地图与姿态也不同，不能将全部 Full 收益单独归因于梯度隔离。

### 原生产 Cached Bernoulli

| LoD | joint／串行实际点数 | joint PSNR | 串行 PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | Δphoto |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1% | 447／451 | 18.8471 | 16.7223 | **−2.1247** | −0.05483 | +0.09689 | +0.03047 |
| 5% | 2,135／2,256 | 20.6425 | 19.5568 | **−1.0857** | −0.03488 | +0.05153 | +0.01362 |
| 10% | 4,216／4,480 | 21.0699 | 20.1630 | **−0.9069** | −0.03189 | +0.04435 | +0.01137 |
| 15% | 6,157／6,552 | 21.1361 | 20.2984 | **−0.8377** | −0.02996 | +0.03835 | +0.01053 |

即使串行在各档实际选点更多，LoD 仍全面退化；不是名义 ratio 导致实际预算更小。

### 相同绝对 K

`K=floor(r × min(42,640,44,130))`。双方严格同 K，停止 slot 不用于补足预算。

| LoD | K | joint PSNR | 串行 PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | Δphoto | PSNR 改善视图 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1% | 426 | 18.7211 | 16.7285 | **−1.9926** | −0.05375 | +0.08796 | +0.02975 | 0/64 |
| 5% | 2,132 | 20.6427 | 19.4714 | **−1.1712** | −0.03706 | +0.05472 | +0.01468 | 4/64 |
| 10% | 4,264 | 21.0693 | 20.1496 | **−0.9196** | −0.03235 | +0.04618 | +0.01153 | 2/64 |
| 15% | 6,396 | 21.1206 | 20.2949 | **−0.8257** | −0.03081 | +0.04229 | +0.01068 | 5/64 |

退化分布覆盖绝大多数视图。所有指标的 paired mean／median／p10／p90／improved-view fraction 完整保存在 CSV，不只报告平均 PSNR。

上一轮 Full100 结果仅复用为背景，不重新启用或训练。已核对同样的 64 个帧、上述四个相同 K，其 same-K PSNR 分别为 18.6216／20.8414／21.3737／21.4543 dB；本次串行隔离也全部低于这些已有结果。

## 低 opacity 与预算占用

以下为最终 Master 当前 opacity 快照，不等同于“每个点已充分访问后仍长期无效”。

| 指标 | joint | 串行隔离 |
|---|---:|---:|
| opacity < 1/255 | 29,272（68.65%） | 26,324（59.65%） |
| opacity < 0.02 | 30,756（72.13%） | 28,340（64.22%） |
| ≥8 个成员低于 1/255 的 Anchor | 2,282（53.52%） | 1,882（42.65%） |
| 10 个成员均低于 1/255 的 Anchor | 870（20.40%） | 712（16.13%） |
| 停用 slot | 20,739（48.64%） | 19,991（45.30%） |
| opacity 均值 | 0.15781 | 0.18861 |
| opacity 中位数 | 0.0002413 | 0.0005282 |

相同 K 的当前低 opacity 选中数：

| LoD | joint | 串行隔离 |
|---|---:|---:|
| 1% | 0/426 | 0/426 |
| 5% | 240/2,132 | 13/2,132 |
| 10% | 896/4,264 | 376/4,264 |
| 15% | 1,802/6,396 | 1,323/6,396 |

**低 opacity 地图成员及选中占比下降，同时 LoD 质量下降。** 这直接说明“减少低 opacity 点”不足以保证高质量的 LoD 子集。特别是 1% 双方均未选中低于截断的点，仍下降约 1.99 dB，当前 LoD 问题不能仅归因于选入这些失效点。

## 解释与限制

本次直接观察到：Full 专门优化 Master 后，Full 重建质量与 opacity 分布更好；取消 LoD 对 Master 的梯度后，即使 Selector／TF 获得更多正常更新，LoD 仍明显变差。**这与“Gaussian 需要参与 LoD 子集重建适配，单靠 Selector／TF 难以补偿”一致，但不是独立证明。**

本轮同时改变 Full 频率和 Master 梯度路由，不能分别量化两者的因果贡献。也不能从一次训练断言 Selector 已收敛，或把退化直接归因于某个网络容量／训练步数。没有据此改 Loss、增加训练、调整 LR 或搜索 checkpoint。

- 单场景、单 seed，仅代表本次配置。
- SLAM／Mapping 异步，online 结束从 6299 变为 4305；在线姿态、关键帧使用、增长／pruning 和 native tail 占比不同。same-K 只控制选择数量，没有冻结地图或在线训练轨迹。
- 评价沿用各模型自己的估计轨迹；64 帧没有严格 Bank／non-Bank 隔离，不能当作 held-out 泛化证明。
- 单次 wall time：joint 73.627 s、串行 86.293 s（约 +17.2%）。包含异步执行差异，不是严格性能 benchmark；没有可比较的峰值显存数据。
- 本轮 Full100 明确关闭；串行保留为默认关闭的实验开关，**不建议基于本次结果替代 joint 默认策略**。

## 可复核文件与交付状态

- [串行调度与状态](H:/AIProjects/cartgs优化更细致版/mycartgs/include/ultrafast_rotation.h:17)
- [配置、Full／LoD 路由](H:/AIProjects/cartgs优化更细致版/mycartgs/src/gaussian_mapper.cpp:539)
- [更新后 Master 重新读取与 LoD detach](H:/AIProjects/cartgs优化更细致版/mycartgs/src/gaussian_mapper.cpp:1849)
- [真实梯度／Adam 隔离测试](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/ultrafast_mapping_test.cpp:183)
- [correctness.json](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/serial_full_lod_validation/evidence/validation/correctness.json)
- [quality_summary.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/serial_full_lod_validation/evidence/validation/quality_summary.csv)
- [quality_per_view.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/serial_full_lod_validation/evidence/validation/quality_per_view.csv)
- [paired_statistics.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/serial_full_lod_validation/evidence/validation/paired_statistics.csv)
- [optimization_diagnostics.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/serial_full_lod_validation/evidence/validation/optimization_diagnostics.csv)
- [逐 Mapping trace](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/serial_full_lod_validation/evidence/serial_seed2026/serial_full_lod_trace.csv)
- [本地独立核验](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/serial_full_lod_validation/local_delivery_checks.json)

状态：实现 **IMPLEMENTED**，功能及梯度 **TESTED**，一次在线质量对照 **TESTED**，跨场景／seed 结论 **NOT TESTED**。

276 个证据文件 SHA、20 个实验源码 SHA、10,000 条 trace、18 个评价 arm 的 1,152 条逐视图数据、36 条 paired 统计均在本地独立核验通过。最终四个 ratio 的 native／Python／导出 Cached Bernoulli 集合 exact 一致。

服务器隔离目录：`/home/prmi/zwj/serial-full-lod-20261021`。production、原 joint 基线和上一轮 Full100 源码 SHA 均保持不变，训练／评价已结束。未 stage、commit、push。
