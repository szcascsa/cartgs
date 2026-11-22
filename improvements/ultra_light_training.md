# Full / LoD 训练模式

日常配置使用 `cfg/selector/fast.yaml` 或 `selector_config.yaml`：

```yaml
# 1：联合训练，默认值；2：Full / LoD 梯度独立。
Training.ultra_light_mode: 1
Selector.enabled: 1
```

旧 `Training.ultra_light_enabled`、`Training.ultra_light_interval` 已从活动配置、
读取逻辑和模型导出中移除。缺少 mode 时默认使用 mode1。内部 mode0 保留供
原始训练基线使用，日常选择 mode1 / mode2。

## Mode1：五步轮转，LoD 联合更新主地图

在线正常 Mapping 固定按 `LoD 1% → LoD 5% → LoD 10% → LoD 15% → Full` 轮转。
LoD 步不 detach 主 Gaussian，可共同更新 Gaussian、Selector 和已启动的 TF；
Full 步只更新 Gaussian。Selector 必须启用。此路径不使用配置中的
`Selector.detach_gaussian_parameters`，也不允许 target_ratios 改写固定轮转。

Full 使用现有 GausPyramid 图像选择。Full 成功提交后，按已有 Full-to-Full
几何漂移计算成熟度，再处理到期的 densify/prune 和 opacity reset；LoD
及联合 Replay 的实际 Gaussian 更新可登记本周期 touched 点。

SLAM 实际 shutdown 后，每轮尾部 Mapping 改为：

1. 同一次 KF 采样，Full 继续按 GausPyramid 选层并执行 backward。
2. 提交 Gaussian 更新、计算一次成熟度并清梯度；Full loss/EMA、统计、到期
   densify/prune/opacity reset 保持 mode1 的 after-step 顺序。
3. 从最新 Gaussian/topology 建图，使用 native 原图执行一个 ratio 的 LoD。
   不 detach Gaussian，联合提交 Gaussian、Selector 和已经运行的 TF 更新。
4. LoD 仅登记 touched 点，不重算成熟度；相邻 Full 快照的漂移包含前一 LoD 的影响。

尾部从在线尚未执行的下一个 LoD ratio 接续，仅在四个 ratio 间轮转。
未进入 LoD 不消费 ratio，已尝试但被拒绝的 LoD 仍消费一次。尾部每轮最多
两次 Gaussian Adam 提交，但 iteration、Mapping 和尾部预算只前进一次；
LoD 被拒绝不回退已成功 Full 的学习率时钟。在线五 slot 计数在 shutdown
边界冻结，尾部 Full、LoD、ratio 及实际提交分别统计。

## Mode2：每轮 Full，然后执行独立 LoD

每次有效 Mapping 使用同一个 KF，按以下顺序执行：

1. 按现有 GausPyramid 配置选择 Full 分辨率，执行 Full render/loss/backward。
2. 登记 Full 可见点；按原 CaRtGS 顺序执行统计和到期 topology 操作，
   touched 状态随 append/prune 对齐。
3. 提交 Gaussian 更新并清梯度；只有存在实际 Gaussian 参数组提交时，
   才用更新后的属性计算成熟度。topology 导致梯度丢失的空 step 不产生新证据。
4. Selector 已启用、模型非空且预算允许时，立即执行正常 LoD 分支。
   不再让整个 LoD 等待 `TransformField.start_iter` 或 TF 的成熟度启动条件。
5. TF 仍在满足自身启动条件时初始化并加入 LoD；此前 LoD 可训练 Selector。
6. LoD 强制 detach XYZ、scale、rotation、opacity、SH DC/rest，仅提交
   Selector/TF 更新，不改变主 Gaussian 参数、动量或 Adam step。

Full 与 LoD 分别提交、分别清梯度，仍使用同一个 shared Adam state owner。
LoD 使用 Full/topology 完成后的最新地图和成熟度，不写入 Full densification
统计。首个有效 Full 建立快照，后续有 touched 证据的 Full 才形成成熟度观测。

`Selector.target_ratios` 按实际 LoD 分支执行次数轮转。Selector 关闭时为
Full-only；TF 关闭不阻塞 Selector-only LoD。mode2 当前要求
`GradientConstraint.enable: 0`，不支持辅助阶段 Guard。

## 分辨率、周期任务和尾优化

两种模式的 Full 均遵循现有 GausPyramid 配置。LoD 的
`Selector.lod_render_scale` 在 mode1 中仅在线生效，尾部使用 native 图像；
mode2 在线和尾部均使用该 scale，相对 native 图像缩放。

两种模式均保留现有尾优化配置。mode1 尾部每轮 Full 后执行联合 LoD，
mode2 尾部继续 Full 后执行独立 LoD。GI、AABB、Replay、CD、DSD 的周期和特殊训练语义
保持原调用链；mode2 的 detach 仅用于正常 LoD，不改变联合 Replay。

## 回归验证

`ultralight_mode2_mapping_test` 覆盖默认 mode1、显式模式、配置导出、
Full-only 原调度/native/pyramid 对照、两阶段梯度及 Adam 状态隔离、
TF 启动前 Selector LoD、TF 独立启动、成熟度从零建立、topology 空提交、
ratio/预算/尾部及 Replay/CD/DSD 隔离。真实 renderer 和状态检查需要
目标 CUDA 11.8 / LibTorch 环境，静态检查不能替代运行验证。

`ultrafast_mapping_test` 另外验证 mode1 尾部两次真实 Gaussian backward/Adam
提交、Full 提交先于 LoD backward、maturity 只在 Full 边界更新、GausPyramid
Full 与 native LoD、固定/零步/旧尾预算及非有限 LoD 保留 Full 更新。
