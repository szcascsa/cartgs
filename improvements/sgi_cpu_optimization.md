# SGI CPU 选帧优化

## 改动与完成标准

本次优化 CPU 覆盖收益计算，保持原有 30% Bank 配额、Anchor/Representative 比例、关键帧遍历顺序、严格增益替换、平局处理和 Bottom-K 采样规则。

完成标准：优化前后逐步选帧结果一致；在 `172.19.163.58:/home/prmi/zwj/mycartgs/` 编译并通过测试；使用相同输入对比 CPU 耗时。本地未编译。

## 算法

原实现对每个候选遍历所有 Anchor，在每个配对中重复扫描 Anchor 的独占点和候选点，再进行覆盖查询及二分查找。

新实现维护每个 MapPoint 的覆盖次数、覆盖它的 Anchor ID 的 XOR，以及每个 Anchor 的独占覆盖数量。覆盖次数为 1 时，XOR 精确等于唯一覆盖者；移除 Anchor 后，也可在覆盖次数从 2 降为 1 时直接恢复剩余覆盖者，支持关键帧 ID 为 0。

替换收益精确写成：

`gain(candidate, anchor) = uncovered(candidate) + retained_unique(candidate, anchor) - unique(anchor)`

每个候选只扫描一次自己的点集，累计未覆盖点和各 Anchor 的可保留独占点，再读取缓存计算所有 Anchor 的收益。每次替换后立即维护计数，后续候选使用更新后的覆盖状态。

替换收益主循环由 `候选 × Anchor × 点` 降为 `候选 × 点 + 候选 × Anchor`。关联变化、关键帧删除、Anchor 增减时同时更新缓存。

## 验证方式

- `tests/global_gi_bank_reference.h` 冻结优化前的原算法，仅用于测试和基准。
- 差分测试覆盖 8 组配置、2000 次变化/空集合输入和 2000 次未变化输入，逐步比较成员、Anchor、Representative、配额、覆盖率与成员变更标志；包含重复点、输入顺序变化、ID=0、最大 uint64 ID、关键帧删除、容量限制和不同 Anchor 比例。
- `bin/global_gi_bank_test --benchmark` 使用 243 个关键帧、每帧 1000 个输入关联，分别比较 Anchor 关联变化、仅候选关联变化和未变化三类输入。每类采样 30 次，交替运行新旧算法顺序，并对每次结果进行一致性检查。

## 日志

增加 `global_gi_bank_snapshot_ms`、`global_gi_bank_selection_ms` 及对应的 `*_total_ms`，保持原有 Bank 总耗时字段。快照时间包含 ORB 锁等待和关联读取；selection 时间包含规范化、变化检查和覆盖选帧。

## 性能结果与边界

服务器 Release 完整构建通过，全部 CTest 28/28 通过，耗时 12.72 秒；差分测试与全部基准样本的选帧结果一致。

| 输入场景 | 原算法平均耗时 | 优化后平均耗时 | 原算法 P95 | 优化后 P95 | 平均加速比 |
|---|---:|---:|---:|---:|---:|
| Anchor 关联变化 | 306.3441 ms | 24.1559 ms | 314.6873 ms | 25.3606 ms | 12.6819 倍 |
| 仅候选关联变化 | 8.9732 ms | 7.6814 ms | 9.1770 ms | 7.9516 ms | 1.1682 倍 |
| 关联未变化 | 7.3356 ms | 7.4700 ms | 7.5190 ms | 7.6580 ms | 基本持平 |

这是在同一服务器、同一输入、同一测试进程中对比原算法与优化算法的结果，含输入复制、排序和变化检测，不含 ORB 快照读取。

真实数据使用已有 Replica room0 前 400 帧独立配置，启用 Selector 和 Streaming GI，完成 2100 次 Mapping，退出码为 0。第 1000、2000 次完整刷新耗时分别为 31.0761 ms、31.3755 ms；第 2000 次及最终状态均为 invalid=0、有效 Mature 比例=1、guidance=1。最终 33 个有效 KF、10 个 Bank 视角（8 Anchor、2 Representative）、25711 个 Gaussian。

最终累计快照时间 9429.9844 ms、选帧及规范化时间 842.1365 ms；快照占这两项之和的约 91.8%。最后一次 Bank 更新 4.2044 ms，其中快照 3.8171 ms、选帧 0.3873 ms。该短测仅验证接入及计时，不与先前三模式短测或用户 TUM 大地图日志比较整体速度，也不代表完整序列的画质收益。

本次算法优化不会消除每次 Mapping 的 ORB 关联读取；实际实验中可用新增字段区分剩余快照成本与选帧成本。合成关联基准不等于整体 Mapping、Tracking 的加速比例。

源文件通过 SHA256 校验后同步至 `/home/prmi/zwj/mycartgs/`。源文件备份及原始验证日志保存在远端 `build-cuda118-sm86-sgi-cpu-validation/`，使用原有 `build-cuda118-sm86-gcc9` 构建目录。
