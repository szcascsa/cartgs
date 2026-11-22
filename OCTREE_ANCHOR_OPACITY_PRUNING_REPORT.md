# Octree-GS 原版 Anchor pruning 与逐 slot 窗口统计

## 结论

已按要求先提交修改前基线：`497463a664fa198dd1bec0be2f3124bd7f878899`，分支 `redundancy-aware-si`，未 push。本轮 pruning 与诊断实现保留为未提交改动。

**原版整组删除规则与逐 slot 同窗口统计均已实现并通过验证。长期低 opacity 成员及 LoD 预算浪费仍然严重，但本轮没有加入候选屏蔽或任何额外删除条件。**

fr1_xyz、seed2026、10,000 Mapping 的一次诊断训练：创建 7113 个 Anchor，整组删除 2940 个，最终 4173 Anchor / 41730 Gaussian。最终 28795 个 Gaussian（69.00%）当前低于 renderer 截断；30135 个（72.21%）低于旧逐点阈值。

其中 3795 个最终 Anchor 有充分窗口记录，378 个没有充分记录；后者不会被统计成“长期失效”。充分记录者中 1408 个至少有 8/10 slot 全窗口始终低于 `1/255`，占充分记录者 37.10%。还有 210 个的十个 slot 都全窗口低于截断，但 Anchor opacity 总和仍达到原版保留门槛。

这是单场景、单 seed 的机制与预算诊断。没有进行配对质量实验，不能据此宣称 PSNR/SSIM/LPIPS 提升或退化。在线异步训练轨迹与旧模型不同，不能把旧/新 N 或预算差异直接解释为纯 pruning 因果效应。

## 1. 原版规则

核对本地官方源码快照 `city-super/Octree-GS@9db64fe8981f5d6a1002c41cbc275a14f4d5ea0c`，四个参考文件的 SHA 与 `SOURCE.json` 一致。原代码：[training_statis](https://github.com/city-super/Octree-GS/blob/9db64fe8981f5d6a1002c41cbc275a14f4d5ea0c/scene/gaussian_model.py#L634)、[adjust_anchor](https://github.com/city-super/Octree-GS/blob/9db64fe8981f5d6a1002c41cbc275a14f4d5ea0c/scene/gaussian_model.py#L852)。

每次可见访问：

`O_A += sum_k max(alpha_Ak, 0)`；`T_A += 1`。

删除严格要求同时满足：

`T_A > check_interval * 0.8`；`O_A < 0.005 * T_A`。

opacity 使用十个 slot 的**总和**，没有除以 10。`>` 与 `<` 均为严格不等式。没有失败成员比例、成员最大值、单点 opacity、screen size 或 world size 的附加删除条件。

达到次数门槛的 Anchor 检查后清零 O/T；不足门槛的继续累计。slot 统计完全同窗重置；不能把固定 500 Mapping 简单当作每个 Anchor 都有 500 次访问。

## 2. 本地适配与梯度边界

- 保留现有 500 Mapping 的拓扑检查调度，`check_interval=densifyInterval()`；因此本次门槛是 **超过 400 次 Full 可见访问**。LoD 轮转中 Full 较少时，统计可以跨多个检查周期累计。
- 官方模型使用 Anchor 可见 mask；本项目有十个显式且可独立移动的 Gaussian。定义“Anchor 可见”为 Full renderer 的 `radius>0` 集合中至少存在一个成员；同一次 Full view 只记一次 Anchor 访问，并统计该 Anchor 全部十个 slot。
- 只在 Full Render、Gaussian optimizer 更新前采集未变形 Master 的激活 opacity。LoD/TF/Change Distill 不累计该统计；Selector 输入和视角无关评分保持原状。
- 所有新统计都在 `NoGradGuard` 下更新并 detach，不进入 optimizer，不产生额外 loss，不进入 Selector 评分或候选 mask。
- 新增 Anchor 的 O/T 与 slot 统计补零；删除使用同一整组 survivor mask 压缩；几何/Loop/CD 迁移保留 UID、slot 归属和其统计。Gaussian 六组属性、Cached Uniform、Anchor Feature 及双 Adam owner 继续同步迁移。
- Legacy 非显式 Anchor 路径保持逐点 pruning。此次仅替换显式十点路径的政策。

## 3. 每个 slot 保存什么

同一 Anchor 窗口与分母 T：

|字段|含义|
|---|---|
|`slot_opacity_accum`|可见访问时 slot 正 opacity 累计值|
|`slot_mean_opacity`|累计值 / Anchor 访问次数|
|`slot_below_002_count/fraction`|低于 0.02 的次数及访问比例|
|`slot_below_cutoff_count/fraction`|低于 1/255 的次数及访问比例|
|`slot_current_opacity`|导出当时的当前激活 opacity|
|`anchor_visits`、`block_uids`、`levels`、`keys`|分母与稳定身份、空间位置|

未观察到的访问不更新 O 或 T，也不为 slot 补零。T=0 的平均值、比例保存为 NaN，CSV 中为缺失值。它与“T>0 且始终低 opacity”明确区分。

每次检查**重置前**保存 `anchor_pruning/window_<mapping>.pt` 和 `windows.csv`；最终 checkpoint 保存 `point_cloud/iteration_10000/anchor_pruning_state.pt`。因此即使最终时刻刚清零，也能读取上一完整窗口。统计随 checkpoint 导出供复核，本轮没有新增跨进程恢复训练协议。

末期诊断对每个存活 UID 选其最近一个观察充分的闭合窗口。没有充分窗口时标记观察不足或未观察，不把它算作长期失效。“始终低于截断”只指这个已记录窗口的全部可见访问，**不是整个训练生命周期或连续空间时间内所有视图**。

## 4. 实际例子：Anchor 保留，成员失效

Anchor UID **3149**，L1，窗口检查时 Mapping=10000，累计 **836 次**可见访问，平均 slot opacity 总和 **1.001973**，满足原版保留条件。

十个成员中，**8 个在该窗口的每次访问都低于 `1/255`**；另外两个成员占累计 opacity 的 **99.9656%**。这是“整体 opacity 达标，但多数成员长期低 opacity”的直接实例；没有因此额外删除这个 Anchor。

|slot|窗口平均 opacity|低于 0.02 比例|低于 1/255 比例|最终当前 opacity|
|---|---:|---:|---:|---:|
|0|1.661603e-06|100.00%|100.00%|9.424117e-07|
|1|1.197749e-05|100.00%|100.00%|6.791333e-06|
|2|0.9999952|0.00%|0.00%|0.9999977|
|3|4.686381e-05|100.00%|100.00%|2.654048e-05|
|4|1.496142e-06|100.00%|100.00%|8.487351e-07|
|5|2.468828e-06|100.00%|100.00%|1.400089e-06|
|6|0.0002746857|100.00%|100.00%|0.0001596567|
|7|4.128488e-06|100.00%|100.00%|2.341151e-06|
|8|1.740368e-06|100.00%|100.00%|9.869912e-07|
|9|0.001633003|100.00%|94.86%|0.0006215892|

另外存在“10 slot 全窗口都低于 renderer 截断，但总和仍超过 0.005”的保留 Anchor。这也符合原公式：十个较小正 opacity 的和可能达到 0.005；整体保留并不保证其中某个 slot 能跨过 renderer 截断。

## 5. 当前 LoD 预算利用率

直接使用原 `build_selector_mask`、保存的 Cached Uniform 与同一个 Selector，未屏蔽任何候选，也没有重新拟合模型。TF 目前只改 XYZ/scale/rotation，不改 opacity；因此当前 opacity<1/255 的点在该状态下无法通过 renderer 的 alpha 截断。

|ratio|实际选点数|当前低于 1/255|占选择预算|充分历史窗口始终低于截断且当前仍低|
|---|---:|---:|---:|---:|
|1%|438|2|0.46%|1 / 438（0.23%）|
|5%|2219|727|32.76%|524 / 2219（23.61%）|
|10%|4257|2021|47.47%|1560 / 4257（36.65%）|
|15%|6138|3515|57.27%|2657 / 6138（43.29%）|

5%/10%/15% 的当前预算浪费依然明显。上表历史指标仅使用有充分窗口记录的成员；观察不足者没有被默认归入失效。结果支持将“候选屏蔽”作为后续独立研究问题，**本轮尚未实现或验证该方案**。

![窗口与预算诊断](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/anchor_opacity_pruning_validation/pruning_diagnostics.png)

## 6. 验证状态与成本

**已测试**：11 项 gate 通过，覆盖 CPU/CUDA 原版公式、访问次数取等不删、opacity 取等不删、sum 而非 slot mean、时间平均而非当前值、观察不足跨窗口累计、NaN 缺失值、暂时低 opacity 的非100%比例、8个长期低 slot 的有效整体仍保留、slot/Anchor 同窗重置、追加与整组压缩、双 Adam owner、几何运输；另有生产 Mapping 的 Full-only 统计、joint/Full 路由、Selector+TF CD、Legacy 回归及 Python/native 兼容测试。

**已运行**：fr1_xyz / seed2026 / 10,000 Mapping，一次诊断训练，19 次 pruning 检查。每个窗口独立复算删除公式、slot sum=Anchor sum、分母与比例、未观察为 NaN、最终 UID 与人口闭合。候选屏蔽未实现。

**未测试**：多场景/多 seed 完整质量矩阵、same-K PSNR/SSIM/LPIPS 对照、严格的无干扰耗时基准。此次训练时长仅为 smoke 运行记录，不能用于声称更快。

统计驻留存储为每 Anchor 256 字节（两个 Anchor 累计量及三组十 slot 累计量），最终约 1.02 MiB；临时张量和定期写文件另计。每次 Full 有额外统计 kernel，检查窗口会同步导出诊断数据；没有将该诊断版本称为已完成低延迟性能验收。

## 7. 文件与源码

服务器隔离验证目录：`/home/prmi/zwj/octree-anchor-pruning-20261021`。原 production、原训练 source 和模型 SHA 已核对保持不变；新原生源码与本地上传的 8 个受控路径 SHA 对齐。

本地证据目录：[anchor_opacity_pruning_validation](<H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/anchor_opacity_pruning_validation>)。

- [逐 Anchor 窗口明细](<H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/anchor_opacity_pruning_validation/evidence/validation/anchor_window_statistics.csv>)、[每个窗口汇总](<H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/anchor_opacity_pruning_validation/evidence/validation/window_summary.csv>)。
- [最终存活 Anchor 诊断](<H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/anchor_opacity_pruning_validation/evidence/validation/final_anchor_diagnostics.csv>)、[逐 slot 四项统计](<H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/anchor_opacity_pruning_validation/evidence/validation/final_slot_diagnostics.csv>)。
- [LoD 预算明细](<H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/anchor_opacity_pruning_validation/evidence/validation/lod_budget_statistics.csv>)、[correctness](<H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/anchor_opacity_pruning_validation/evidence/validation/correctness.json>)、[汇总](<H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/anchor_opacity_pruning_validation/evidence/validation/audit_summary.json>)。
- 原始窗口 PT、最终统计 PT、日志和 SHA 清单均在 `evidence/` 下保留。

主要实现：[统计与删除](<H:/AIProjects/cartgs优化更细致版/mycartgs/src/gaussian_model.cpp:130>)、[Full 接入](<H:/AIProjects/cartgs优化更细致版/mycartgs/src/gaussian_mapper.cpp:1575>)、[边界测试](<H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/anchor_opacity_pruning_test.cpp:1>)。
