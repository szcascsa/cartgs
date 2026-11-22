# 拓扑更新后连续 100 步 Full Render：实现与质量验证

**结论：部分有效，但没有解决低 opacity 成员问题，也没有证明 Full 不足是主要根因。**

`fr1_xyz / seed2026 / 10,000 Mapping` 实际训练完成。相同绝对 K 下，5%／10%／15% 的 PSNR 分别提高 **0.199／0.304／0.334 dB**，四项质量指标均改善；1% 的 PSNR、SSIM、photo 退化。Full PSNR 只提高 **0.106 dB**，LPIPS 变差。最终低于 `1/255` 的 Gaussian 比例由 **68.65% 增至 70.25%**。

## 1. 已实现的调度

配置：

```yaml
Training.ultra_light_mode: 1
Training.post_topology_full_steps: 100
```

默认值为 `0`，保持旧行为；本次实验明确设置为 `100`。当前只允许 `0` 或 `100`，且开启恢复窗口要求现有 joint mode1。

- 每次计划中的 `densifyAndPrune` 调用结束后启动窗口，包括没有改变人口的检查。
- **随后 100 次完成的 Mapping iteration** 只执行 Full Render，更新 Master Gaussian，不执行正常 LoD／Selector／Anchor Feature／TF 优化。
- 如果窗口内再次发生拓扑检查，从该次检查后的下一 Mapping 重新计 100 步。
- Mapping iteration、Gaussian LR 时钟和总训练预算继续推进；普通 ratio 游标暂停，窗口结束后从原 ratio 恢复。
- 原 tail 阶段为每 Mapping 执行 Full + joint LoD。恢复期间跳过后置 LoD；触发拓扑的当前 Mapping 也跳过其尚未执行的后置 LoD，该次不占后续 100 步。
- pending Selector Replay 暂停。在线几何修正和必要 Change Distill 继续保留，因此“Selector／TF 不更新”限定为**正常 Mapping 优化**，不包括必要的几何迁移事件。
- 网络、现有 Loss、Cached Bernoulli、Anchor growing／pruning、slot 停用规则、TF／Change Distill 算法均未修改。

实现位置：

- [调度状态与 ratio 游标](H:/AIProjects/cartgs优化更细致版/mycartgs/include/ultrafast_rotation.h:28)
- [配置读取](H:/AIProjects/cartgs优化更细致版/mycartgs/src/gaussian_mapper.cpp:539)
- [Full 路由](H:/AIProjects/cartgs优化更细致版/mycartgs/src/gaussian_mapper.cpp:1483)
- [拓扑后触发](H:/AIProjects/cartgs优化更细致版/mycartgs/src/gaussian_mapper.cpp:1751)
- [tail 跳过 LoD](H:/AIProjects/cartgs优化更细致版/mycartgs/src/gaussian_mapper.cpp:1826)
- [逐 Mapping trace 与剩余步数](H:/AIProjects/cartgs优化更细致版/mycartgs/src/gaussian_mapper.cpp:1929)

## 2. 正确性与实际执行

RTX 3090 上完成六项门控：调度测试、真实 Full100 Mapping、关闭功能回归、Anchor pruning CPU／CUDA、Change Distill，全部通过。

真实 Mapping 测试检查了拓扑触发、在线及 tail 的连续 100 Full、第 101 步恢复 LoD、ratio 游标、Master 参数变化、Selector／Anchor Feature／TF 参数及正常更新计数保持不变，以及 logging 开关不改变行为。

完整训练的 10,000 行 trace 已逐行独立复算：

- Mapping 编号 1…10000 无缺口。
- 19 次拓扑触发：1000、1500、…、10000。
- **18 个完整恢复窗口，累计 1,800 个 Full-only Mapping**；其中在线 1,200，tail 600。
- 最后一次在 Mapping 10000 触发，剩余 100 步，因固定训练预算而右截断。没有额外延长训练，不能称 19 个窗口全部完成。
- 所有恢复步均为 `Full=1, LoD=0, Selector正常更新=0, TF正常更新=0`。
- 最终四个 ratio 的 native C++、Python evaluator、Cached Bernoulli 导出选择集合 exact 一致。
- 275 个证据文件 SHA 和 19 个本地／实验源码 SHA 校验通过；1,152 条逐视图结果及 36 条 paired 统计从原始数据独立复算通过。

证据：[correctness.json](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/full_recovery_validation/evidence/validation/correctness.json)、[窗口明细](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/full_recovery_validation/evidence/validation/recovery_windows.csv)、[逐 Mapping trace](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/full_recovery_validation/evidence/full100_seed2026/post_topology_full_trace.csv)、[本地独立校验](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/full_recovery_validation/local_delivery_checks.json)。

## 3. 实验与真实优化次数

基线为上一轮 **slot 停用开启、无 Full 恢复窗口** 的最终模型。本轮仅在相同输入配置末尾增加 `Training.post_topology_full_steps: 100`，保持 slot 停用开启、SI/GI disabled、27 维 Anchor Selector、TF 与 Change Distill 开启。

- 场景：`fr1_xyz`；seed：`2026`；总预算：`10,000 Mapping`。
- 候选只训练一次，使用最终 checkpoint，不按质量选 step。
- 使用上一轮已锁定的 64 个共同评价视图，没有按本轮结果筛选视图。
- 四指标：PSNR、SSIM、LPIPS、原生 `photo = 0.8 L1 + 0.2 (1-fast_ssim)`。评价 photo 不包含训练 opacity regularization。

| 实际计数 | 基线 | Full100 | 变化 |
|---|---:|---:|---:|
| Full render | 4,960 | 5,650 | +13.91% |
| LoD render／Selector 正常更新 | 8,741 | 7,106 | −18.70% |
| Gaussian optimizer 更新 | 13,701 | 12,756 | −6.90% |
| TF 正常更新 | 6,341 | 5,026 | −20.74% |
| SLAM online 结束 Mapping | 6,299 | 6,637 | 异步边界不同 |
| 最终 Gaussian | 42,640 | 48,400 | +13.51% |
| 最终 Anchor | 4,264 | 4,840 | +13.51% |
| 累计创建／删除 Anchor | 7,203／2,939 | 7,887／3,047 | 增长轨迹不同 |

这次改变增加的是**完整地图的连续优化覆盖**，没有增加总 Gaussian optimizer 次数。tail 原本每步已有 Full，跳过其后置 LoD 后，总 Adam 更新次数反而减少。TF 首次初始化／应用从 3000／3001 自然延后至 3100／3101，没有修改 TF 算法。

因此原实现不是“全程 Full 都很少”：在线为四个 ratio LoD + 一次 Full，tail 则为每步 Full + LoD。本实验更直接检验的是“拓扑更新后连续完整地图优化是否有帮助”。

## 4. 质量结果

以下差值均为 **Full100 − 基线**。PSNR／SSIM 越大越好，LPIPS／photo 越小越好。

### 4.1 原生产 Cached Bernoulli

| 档位 | 基线实际点数 | Full100 实际点数 | ΔPSNR | ΔSSIM | ΔLPIPS | Δphoto |
|---|---:|---:|---:|---:|---:|---:|
| Full | 42,640 | 48,400 | +0.1059 | +0.00282 | **+0.00306** | −0.00152 |
| 1% | 447 | 522 | +0.4985 | +0.01016 | −0.01989 | −0.00543 |
| 5% | 2,135 | 2,505 | +0.3742 | +0.01169 | −0.01645 | −0.00476 |
| 10% | 4,216 | 4,884 | +0.3255 | +0.01092 | −0.01656 | −0.00428 |
| 15% | 6,157 | 7,209 | +0.3252 | +0.01182 | −0.01860 | −0.00450 |

同 ratio 的四档 LoD 均改善，但新地图人口更大、实际预算更多，不能把这些差值全部解释为欠优化改善。

Full 的绝对结果：

| 方法 | PSNR | SSIM | LPIPS | photo |
|---|---:|---:|---:|---:|
| 基线 | 22.0451 | 0.76060 | 0.27537 | 0.08426 |
| Full100 | 22.1509 | 0.76341 | 0.27843 | 0.08274 |

### 4.2 相同绝对 K

先锁定 `K=floor(r × min(N_baseline,N_full100))`。使用各模型固定 cached uniform 的 `logit(p_effective)-logit(u)` 顺序，source row 打破 tie，取相同数量。它是为消除预算差异增加的**诊断策略**，没有替换训练或生产 Cached Bernoulli；不等同于按纯 Selector 分数 hard Top-K。被停用的点不用于补足 K。

| 档位 | 双方 K | 基线 PSNR | Full100 PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | Δphoto | PSNR 改善视图 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1% | 426 | 18.7211 | 18.6216 | **−0.0995** | **−0.00484** | −0.00648 | **+0.00183** | 26/64 |
| 5% | 2,132 | 20.6427 | 20.8414 | +0.1987 | +0.00606 | −0.00575 | −0.00268 | 45/64 |
| 10% | 4,264 | 21.0693 | 21.3737 | +0.3044 | +0.00979 | −0.01481 | −0.00390 | 51/64 |
| 15% | 6,396 | 21.1206 | 21.4543 | +0.3337 | +0.01067 | −0.01393 | −0.00429 | 54/64 |

中高 ratio 的收益不能完全归因于多选点；极低预算 1% 没有保持收益。所有指标的 paired mean／median／p10／p90／improved-view fraction 完整保存在 CSV，没有只报告最佳视图。

数据：[quality_summary.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/full_recovery_validation/evidence/validation/quality_summary.csv)、[quality_per_view.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/full_recovery_validation/evidence/validation/quality_per_view.csv)、[paired_statistics.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/full_recovery_validation/evidence/validation/paired_statistics.csv)。

## 5. 欠优化与低 opacity 诊断

以下是最终 Master 的**当前 opacity 快照**，不是所有点均已充分观察的长期历史结论。

| 指标 | 基线 | Full100 |
|---|---:|---:|
| 当前 opacity < 1/255 | 29,272 / 42,640（68.65%） | 34,003 / 48,400（70.25%） |
| 当前 opacity < 0.02 | 30,756（72.13%） | 35,513（73.37%） |
| ≥8 个成员低于 1/255 的 Anchor | 2,282（53.52%） | 2,705（55.89%） |
| 10 个成员均低于 1/255 的 Anchor | 870（20.40%） | 1,145（23.66%） |
| 停用 slot | 20,739（48.64%） | 23,283（48.11%） |
| opacity 均值 | 0.15781 | 0.14900 |
| opacity 中位数 | 0.0002413 | 0.0002688 |

相同 K 的 LoD 仍选到了部分当前低于 `1/255` 的点：

| 档位 | 基线低 opacity 选中数 | Full100 低 opacity 选中数 |
|---|---:|---:|
| 1% | 0 / 426 | 0 / 426 |
| 5% | 240 / 2,132 | 173 / 2,132 |
| 10% | 896 / 4,264 | 922 / 4,264 |
| 15% | 1,802 / 6,396 | 2,068 / 6,396 |

增加连续 Full **没有减少地图中的低 opacity 比例，也没有系统消除低 opacity 预算占用**。这些快照不能单独证明哪些点长期无效；停用仍遵守上一轮的访问窗口规则，不能把尚未充分观察的点直接视为永久失败。

源码还表明该假设存在边界：

- [forward 截断](H:/AIProjects/cartgs优化更细致版/mycartgs/cuda_rasterizer/forward.cu:412) 与 [backward 截断](H:/AIProjects/cartgs优化更细致版/mycartgs/cuda_rasterizer/backward.cu:603) 都跳过 `alpha < 1/255`。当 activated opacity 本身已低于阈值时，增加 Full 光度渲染次数也不能直接给该点提供恢复所需的图像损失梯度；这不排除已有 Adam 动量、opacity reset 等其他状态变化。
- 现有 [Full loss](H:/AIProjects/cartgs优化更细致版/mycartgs/src/gaussian_mapper.cpp:1602) 含 `Optimization.opacity_reg: 0.001`，本轮保持不变。更多 Full 也增加这项既有正则的执行次数，因此不能仅凭透明度降低推断优化更充分。
- Full 只在当前训练相机的可见范围产生有效渲染梯度，Full render 次数不等于每个 Gaussian 的有效优化次数。

诊断数据：[optimization_diagnostics.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/full_recovery_validation/evidence/validation/optimization_diagnostics.csv)。原逐 slot 窗口数据也保留在本次证据目录的 `full100_seed2026/anchor_pruning/`，没有更改判停／pruning 条件。

## 6. 结论与证据边界

- **功能：IMPLEMENTED / TESTED。** 100 步 Full 窗口、恢复与梯度路由符合设计。
- **质量：PARTIALLY VERIFIED。** 同 K 中高预算改善，但 1% 退化，Full LPIPS 变差，不是所有档位全面改善。
- **欠优化假设：部分支持“连续完整地图优化有价值”，不支持已确认其为主要根因。** 低 opacity 状态没有缓解，不能据此认定大量失效成员已被优化恢复。
- 只有一个场景、一个 seed。SLAM 与 Mapping 异步，各自在线结束时刻、相机轨迹、地图增长和更新次数不同；same-K 控制点数，不能消除这些差异。
- 64 个共同视图未严格拆分 Bank／non-Bank，不能声称独立 held-out 泛化。评价使用各模型自己的估计相机轨迹。
- 单次 wall time 为基线 73.627 s、本轮 68.091 s，包含在线运行轨迹差异；不是独立、无干扰性能 benchmark，不能当作确定加速结论。本轮没有测得可比较的峰值显存。
- 没有据此调整恢复长度、Renderer 截断、Loss、Anchor 初始化或任何阈值；没有追加训练或选择最佳 checkpoint。

服务器隔离实验目录：`/home/prmi/zwj/post-topology-full100-20261021`。既有基线 source 与 production 源码 SHA 均未改变；最终训练／评价进程已结束。当前本地 HEAD 仍为 `497463a664fa198dd1bec0be2f3124bd7f878899`，本轮未 stage、commit、push。
