# 强制覆写尾优化

在 `cfg/selector/fast.yaml` 或指定的 Selector 配置中设置：

```yaml
Training.force_tail_optimization_enabled: 1
Training.tail_optimization_steps: -1
```

开关默认 `0`，旧配置缺少字段时保持原行为。步数默认 `-1`：记录 SLAM
结束边界已经完成的 `iteration_`，尾部再执行 `floor(iteration_/2)` 次普通
Mapping。例如结束时为 8001，尾部为 4000 次。非负值指定固定步数，`0`
表示不执行尾优化；小于 `-1` 的配置拒绝加载。

该策略适用于模式 0、1、2。模式1的 `1%→5%→10%→15%→Full` 是正常
训练流程，SLAM 结束后继续轮转；`online_end` 只冻结在线统计和模型快照。
关闭覆写会使用原版尾部策略。需要 no-tail 实验时，必须显式设置：

```yaml
Training.force_tail_optimization_enabled: 1
Training.tail_optimization_steps: 0
```

## 停止规则

开启覆写后，`Selector.max_mapping_iterations` 被忽略，包括非零值。
SLAM 在线阶段不会因它耗尽而提前结束；实际 `pSLAM_->isShutDown()` 后，
冻结结束迭代数和已完成 Mapping 计数，执行指定数量后停止。

尾部不会再因 `densification_interval × 0.8` 的最低步数、densification
相位对齐或 `keep_training` 延长。原配置 densification_interval=500 时，
400 是旧策略的最低延迟，并非旧策略始终只执行 400 步。
手动停止及没有关键帧时仍可提前退出，并在结束日志中说明原因。

关闭覆写时，保留旧规则：非零 `Selector.max_mapping_iterations` 指定总
Mapping 预算；为 `0` 时使用原最低延迟、相位对齐及 `keep_training`。
无 SLAM 的 COLMAP 训练保留原停止策略，不使用该 SLAM 尾部覆写。

## 优化与计数

尾部长度按 `mapping_iter_` 的增量计数，每次有效普通 Mapping 加一。
Replay、ChangeDistill、DSD 的辅助优化不计入该预算。
SLAM 结束时的快照在第一步尾优化之前冻结，避免多算一尾步或重新计算目标。

覆写尾部允许 Gaussian/Selector/TF 的正常 optimizer 更新，即使
`Optimization.max_num_iterations` 已耗尽。该放行仅在覆写尾部生效，在线
阶段保留原优化预算。进入尾部时清除可能因在线预算耗尽而遗留的未提交
梯度，保留 Adam 参数状态与所有原时钟。

模式1在尾部保留相同的五档轮转；GI、Selector/LOD/TF、Guard、到期的
Replay/AABB/DSD 按既有配置执行。Gaussian 的学习率、densification、
opacity reset、TF 启动、周期检查仍使用原时钟；没有新的增强步时钟。
回环/BA 的 transport/ChangeDistill 即时处理路径保持不变。

## 记录与验证

`[TailOptimization CONFIG]` 输出开关与配置步数；`[TailOptimization]`
仅在尾部开始、结束各记录一次，包含结束迭代快照、起始 Mapping、目标、
完成数和退出原因。导出的 `selector_config.yaml` 保存两个配置字段；已有
`transform_state.json` 额外保存尾部快照、目标及完成数。

调度回归测试覆盖自动取半与奇数取整、低于 400 的尾部、固定/零步、快照
只冻结一次、SLAM/COLMAP 预算边界、关闭模式及尾部 optimizer 预算放行。
编译和真实数据测试在 `/home/prmi/zwj/mycartgs/` 执行，本地不编译。

## 服务器验证结果

CUDA 11.8 / sm_86 / LibTorch 构建成功，30/30 项 CTest 通过，用时 13.20 秒。
新增调度测试及已有 GI、Guard、Replay、CD/DSD、拓扑、CUDA LOD 回归均通过。

Replica room0 前 120 帧，开启超轻量、增强间隔 20，五组均正常退出：

| 场景 | Selector 总上限 | 原 optimizer 上限 | SLAM 结束迭代 | 实际尾步 | 最终 Mapping |
|---|---:|---:|---:|---:|---:|
| 开启，自动取半 | 1 | 1 | 480 | 240 | 720 |
| 开启，固定 7 步 | 1 | 1 | 469 | 7 | 476 |
| 开启，固定 0 步 | 1 | 无限 | 338 | 0 | 338 |
| 关闭，总预算 50 | 50 | 无限 | 在线预算耗尽 | 0 | 50 |
| 关闭，原停止策略 | 0 | 无限 | 362 | 538 | 900 |

自动/固定组的 Selector 和 TF 实际正常更新分别为 240/240、7/7，证明即使
原 optimizer 上限为 1，尾部仍实际优化参数；没有按 400 步下限延长。
两组尾部每步更新 Bank，最终 GI invalid 为 0。旧策略组在 900 步停止，
900%500=400，保留最低延迟和相位约束。

零步组没有尾部训练及阶段刷新，GI 仍是在线缓存快照，部分条目可处于
未观察/待更新状态。首次验证脚本误对零步组要求 GI 全部有效，触发断言；
保留原日志后修正此判据，在 `zero_retry` 新目录重新运行，真实退出码为 0。
程序实现未因该断言修改。模型导出的两个配置字段及尾部状态均已核验。

短测只调整验证配置的 TF 启动、成熟门槛、预算与检查周期；用户已有实验
配置值保持不变，新增字段默认关闭、自动取半。该结果是行为验证，不是
训练质量或加速比结论。回环/BA 路径没有改动，短测不声称覆盖真实回环。

服务器结果、日志、验证配置、源文件备份：
`/home/prmi/zwj/mycartgs/build-cuda118-sm86-tail-validation/`。
本地归档：`../tail-optimization-validation-20261010/server-results/`。
