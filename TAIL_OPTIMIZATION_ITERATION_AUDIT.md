# 尾优化输入与实际迭代次数核查（2026-10-23）

结论：已定位 `room16400.yaml` 对应训练少于 16400 步的直接原因。16400 写在布尔开关 `Training.force_tail_optimization_enabled`，C++ 将其解释为开启；尾部仍按 `Training.tail_optimization_steps: -1` 自动取半。该次训练实际为在线 6162 + 尾部 3081 = 9243，配置、日志、导出状态一致。没有发现这次运行存在尾步漏计或配置没有加载的证据。

本轮只修改交互启动器、测试及说明；生产停止策略、Loss、ratio 调度、joint 优化和源配置保持原语义。没有启动新的 SLAM 训练。

## 1. 直接证据

服务器实验：

`/home/prmi/zwj/mycartgs/results/interactive/rgbd_dataset_freiburg1_room_20261022_190506_452323`

该实验的 `selector_input.yaml` 和当前 `cfg/selector/room16400.yaml` 均包含：

```yaml
Training.force_tail_optimization_enabled: 16400
Training.tail_optimization_steps: -1
Selector.max_mapping_iterations: 0
```

`launch.json` 确认 `--selector-config` 指向该配置。`run.log` 记录：

```text
[TailOptimization CONFIG] enabled=1 steps=-1
[Gaussian Mapper]Selector.max_mapping_iterations = 0
[TailOptimization] event=begin slam_stop_iter=6162 start_mapping=6162 target=3081 ignored_max_mapping_iterations=0
[TailOptimization] event=end mapping_iter=9243 completed=3081 target=3081 reason=complete
9243 iterations completed.
```

源码对应关系：

- `src/gaussian_mapper.cpp:580`：`enabled = force_tail_enabled != 0`，数值 16400 只表示 true。
- `include/tail_optimization_schedule.h:21`：steps=-1 时，尾目标为 `iteration / 2`。
- 同一头文件 `completed()` 按 `mapping - start_mapping` 计尾部已完成次数。
- 最终 `mapping_stage_state.json` 与 `ultrafast_rotation_state.json` 均确认在线 6162、尾部 3081、总 Mapping 9243。

因此文件名中的“16400”和开关的非零数值都不会成为总训练预算。旧交互界面也有选项编号与布尔值不一致的问题：1=保持、2=设置0、3=设置1，输入 1 并不总是代表开启。

## 2. 现有停止规则

| 配置 | 实际含义 |
| --- | --- |
| force_tail=0、max_mapping=0 | 原停止策略，仍有原有尾优化；总步数不固定 |
| force_tail=1、tail_steps=-1 | 尾部为 floor(SLAM 结束迭代 / 2) |
| force_tail=1、tail_steps=N≥0 | 在线阶段之后额外执行 N 个 Mapping；总数为在线数 + N |
| force_tail=1、tail_steps=0 | 不执行额外尾部；在线 SLAM 阶段保留 |
| force_tail=0、max_mapping=N>0 | 目标总 Mapping 为 N；当前 mode1 若另有有限 optimizer 上限仍受其约束 |

强制尾部开启时，`Selector.max_mapping_iterations` 被忽略；关闭时 `Training.tail_optimization_steps` 不参与控制。

原停止规则在 `src/gaussian_mapper.cpp:1307` 附近：未停止、存在关键帧、optimizer 预算允许，并且以下任一条件成立时继续：

```text
iteration - SLAM_stop_iter < 0.8 * densify_interval
或 iteration % densify_interval < 0.8 * densify_interval
或 keep_training
```

当前这些实验 `densify_interval=500`，最少延迟为 400，且需满足相位条件后才停止。因此开关为 0 时出现 400～763 个尾部 Mapping 与源码一致，并不表示 force 开关未生效。

此外，在线 Mapping 与实时 SLAM 异步运行。相同数据及自然停止设置下，在线实际完成的 Mapping 数可能随处理速度、关键帧时序等变化；本次只确认计数，不对这些变化做质量归因。

## 3. 已有 8 次真实训练核对

| 实验后缀 | 原输入 force | 尾部规则 | 在线 Mapping | 尾部 Mapping | 最终迭代 | Master 更新 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| room 20261022_190506 | 16400 | 自动取半 | 6162 | 3081 | 9243 | 12324 |
| room 20261022_193951_gsprune | 0 | 原策略 | 6572 | 400 | 6972 | 7372 |
| desk 20261022_200336 | 0 | 原策略 | 3193 | 707 | 3900 | 4607 |
| xyz 20261022_195245_gsprune | 0 | 原策略 | 4182 | 718 | 4900 | 5618 |
| xyz 20261022_200542 | 0 | 原策略 | 4137 | 763 | 4900 | 5663 |
| xyz 20261023_131140 | 0 | 原策略 | 5337 | 563 | 5900 | 6463 |
| xyz 20261023_132426 | 0 | 原策略 | 5447 | 453 | 5900 | 6353 |
| room 手动命令 20261022_184714 | 原输入副本缺失；导出值0 | 原策略 | 6170 | 730 | 6900 | 7630 |

8 次均满足“在线 + 尾部 = 最终迭代”；自动取半组精确完成 3081 个尾步。其余组日志和保存配置均为 force=0、max_mapping=0，没有实际设定固定总预算。

Master 更新次数不是 Mapping 次数：当前 joint 在线每 Mapping 一次 Master 更新，尾部每 Mapping 的 Full 和 LoD 分别更新 Master，因而表中为 `online + 2×tail`。Selector 仅在对应 LoD 更新，四档轮转计数也不是每档各执行总预算。

## 4. 本轮启动器改动

- `Training.force_tail_optimization_enabled` 直接输入 0/1，回车保留基底，不再使用三级选项编号。
- 开启后直接输入 `Training.tail_optimization_steps`：-1 自动取半、0 跳过额外尾部、正整数为尾部额外 Mapping。
- 关闭后直接输入 `Selector.max_mapping_iterations`：0 为原策略，正整数为总 Mapping 目标。
- `Selector.anchor_only_pruning` 同样直接输入 0/1。
- 在启动/命令预览前校验基底与覆盖后的值，阻止将 16400 等步数写入布尔开关；实际读取的文件仍是本次实验目录内的 effective YAML，原文件不动。
- 显示最终步数规则以及被忽略的字段，说明 force=0 不表示全部尾优化关闭。

若目的是总 Mapping 16400，在 Selector 本次覆盖中使用：

```yaml
Training.force_tail_optimization_enabled: 0
Selector.max_mapping_iterations: 16400
```

当前 TUM Mapping 配置 `Optimization.max_num_iterations: -1`，没有额外有限 optimizer 上限。若目的是在线结束后额外训练 16400 步，则使用 force=1、tail_steps=16400，两种实验不能混称。

## 5. 验证与限制

- 本地 19 项启动器测试通过，包括直接输入、错误布尔值拦截、总预算/尾预算分离及原文件不变。
- 服务器重新编译运行现有纯 C++ `tail_optimization_schedule_test`，自动/固定/零步、快照冻结及预算分支验证通过。
- 服务器对现有实验做只读证据核对，没有重新训练或修改模型。
- 未改变 production C++ 对非零布尔值的兼容行为。新校验位于交互启动器；绕过启动器直接执行原生命令时，错误值仍会按非零=true 处理。
- `Optimization.max_num_iterations` 有限时的 mode1 预检查和上限组合需要单独区分。本次实际 Mapping 输入均为 -1，不能用这一未触发条件解释本次少于预期的步数。
- 服务器有此前启动、仍在菜单中的 Python 进程；Python 已加载的代码不会因磁盘脚本更新自动替换。退出旧菜单、重新执行 `bash run_interactive.sh` 后才使用新版输入界面。

原始计数、配置副本、阶段状态及日志摘要保存在 `tests/diagnostics/tail_iteration_audit_20261023/`，其中 `training_counts.csv/json` 为逐实验汇总，`evidence.tar.gz` 保留核对依据，`source_sha256.json` 标识审查时服务器源码。
