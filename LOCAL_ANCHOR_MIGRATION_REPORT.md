# 闭环 Anchor 容量迁移与 Feature 加权

已实现用户确认的迁移规则，ChangeDistill 默认关闭。原生 CPU 迁移测试通过；尚未运行完整 SLAM/CUDA 闭环或重建质量实验。代码未 stage、commit、push。

## 默认配置

- `cfg/selector/selector_config.yaml` 与 `anchor_compensation.yaml` 均使用 `ChangeDistill.enable=0`；C++ `ChangeDistillConfig` 缺省值也为 false。
- CD 功能保留，原 CD 功能测试显式开启。历史 `fast.yaml` 和 `octree_explicit.yaml` 的显式配置保留，不会自动继承新默认配置。
- Mapping 上限仍为0；27维输入、14维补偿、joint、Cached Bernoulli、TF/SI/GI disabled、原 Anchor pruning 与 densification 调度不变。

## 迁移规则

本规则仅用于可变成员地图的 `LoopClosingBA`，处理实际接受几何修正的 Gaussian。固定10成员历史路径和整体 `ScaleRefinement` 保留原 Anchor 几何运输。

1. 完成整次闭环的所有关键帧修正，期间保持 Gaussian 行号不变。
2. 按来源层级和修正后的真实 XYZ 计算目标 `(level, floor(XYZ / h_level))`，世界原点固定。
3. 同一 cell 内仍归原 Anchor 的点直接保留。对跨 cell 的点，先统一释放所有迁出容量，再按 Gaussian 行号递增接收；重复修正行只计一次。因此两个满员 Anchor 可以互换成员，不会因处理顺序误删。
4. 同层目标有空位时接收；容量达到10时丢弃后续迁入点。已有碰撞桶中存在多个 Anchor 时，依现有稳定存储顺序寻找有空位者；不会因桶中所有 Anchor 都满而创建绕过容量限制的新 Anchor。
5. 目标不存在时创建同层 Anchor，使用 cell 中心和该层尺度。初始 Feature 来自实际接收的来源成员；同一新目标接收多个来源时也按实际接收数加权。
6. 删除被拒绝的 Gaussian，清除空来源 Anchor。幸存 Gaussian 不重排，幸存 Anchor 保留稳定 UID，层级不会退化。

可变成员闭环不再移动原 Anchor 中心来容纳跨 cell 成员；Local Offset 始终由实际 XYZ 和迁入后的 Anchor 中心、尺度计算。

## Feature 定义

设目标最终留驻的原成员数为 `n_resident`，来源 s 被实际接收的成员数为 `m_s`：

```
f_after = (n_resident * f_target_before + sum_s(m_s * f_source_before))
          / (n_resident + sum_s(m_s))
```

全部来源 Feature 使用同一迁移前快照，不会读取本次迁移已更新的 Feature。迁出成员不再计入原节点的留驻权重；被丢弃成员不进入分子、分母。

目标原有5点、接收来源5点时，结果为 `(5*f_target + 5*f_source)/10`。目标原有8点而来源迁出5点时，只接收2点，结果为 `(8*f_target + 2*f_source)/10`，其余3点直接删除。

## 优化器与统计

- 混合 Feature 的接收端清零 Adam 一阶、二阶动量；未混合行保留原状态。新增行的动量为0，参数级 Adam step 保留。同步处理 Selector optimizer 和共享 Mapping optimizer 的参数引用、状态。
- 成员改变的 Anchor 重置 opacity 访问窗口、逐 slot 窗口、停用标记和对应增长梯度统计，避免将旧成员的分母用于新成员。
- 复用 `GaussianModel::prunePoints` 压缩六组显式属性及其 Adam、出生时间、Protection/cached-uniform、监督时间、梯度与可见性统计，并执行原拓扑回调。
- 容量迁移在 render lock 内完成。若手动开启 CD，先完成该事件的 CD，再迁移、删除，防止破坏 teacher 的旧行对应；迁移后 Feature 已改变，不能把迁移前的 CD loss 当成最终模型的保持误差。
- 容量拒绝单独计数，不混入 opacity/尺寸逐点 pruning 或原 Anchor opacity pruning 的删除计数。

## 验证

`tests/anchor_selector/local_anchor_migration_test.cpp` 已注册为 CTest `local_anchor_migration`。使用 Windows MSVC 14.44 与本地 LibTorch 2.4.1 CPU 对真实 AnchorBank 实现编译运行，退出码均为0。

| 场景 | 结果 |
|---|---|
| 目标5点＋迁入5点 | 接收5点，Feature 等权混合 |
| 目标8点＋迁入5点 | 接收2点、丢弃3点，只按接收2点加权 |
| 目标10点＋迁入5点 | 全部丢弃，目标 Feature、Adam 保留 |
| 多来源并入已有目标 | 使用冻结来源 Feature，结果按所有接收数加权 |
| 多来源迁入空目标且总数超过10 | 只创建一个目标，接收前10点，拒绝其余 |
| 两个满员目标互换 | 接收20点，无误删，Feature 交换 |
| 相同位置、不同层级 | 细层占用不消耗粗层容量 |
| 负坐标及重复修正索引 | floor 量化正确，重复行不重复计数 |
| cell 内修正 | 不改变归属、Feature 或 Adam |

同时验证空来源清理、稳定 UID、两个 Feature Adam owner、step/二阶动量，以及改变/未改变成员的 slot 历史。

原生证据保存在 `build-cuda118-sm86/local-anchor-migration/local_anchor_migration_test_build.log`、`local_anchor_migration_test.log` 和 `local_anchor_migration_test_result.json`。

**PARTIALLY VERIFIED**：GaussianModel/Mapper 接入、回调和删除时机已完成源码检查，尚未运行完整 SLAM/CUDA 集成测试。尝试在 Windows 构建旧 `explicit_anchor_test` 时，被已有 `gumbel_network.h` 中 `ScopedTorchRNG rng(torch::Device(torch::kCPU))` 的 MSVC C2751 解析错误阻塞；未为本任务改写旧网络代码，也不把该项记为通过。上述新 AnchorBank 迁移测试不依赖该网络头，已实际编译运行通过。

本轮未声称重建质量改善，也未启动训练或参数搜索。
