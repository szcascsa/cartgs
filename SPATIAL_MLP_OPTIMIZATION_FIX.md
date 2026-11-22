# Spatial MLP 优化梯度修复

已修复预算校准的反向梯度和 MLP 公共输出方向。CPU/CUDA 数值回归、三个 scene 的真实 Bank backward、三个 scene 各 100 步有限训练检查通过。**本轮没有证明 hard ranking 质量提高**：100 步结束时三个 scene 的 hard Top-K 均未改变，Bank hard photo 与初始化相同。

**实际改动**

- [calibrated_selection.h](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_implicit_si_20261020/calibrated_selection.h:7)：保留原 42 次二分和给定 score 下完全相同的 forward 概率，补回 `sum(p)=K` 的隐式一阶导数。
- [train_worker.cpp](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_implicit_si_20261020/train_worker.cpp:83)：最后层 `bias=false`；完整 candidate population 上使用 `correction=r-r.mean()`，mean 保持 autograd；调用修正的校准器，记录 score gradient 之和。
- 原 worker 按字节保留为 [train_worker_legacy.cpp](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_implicit_si_20261020/train_worker_legacy.cpp)。SHA256 仍为 `fca6741d92944e176e2185c9a83170a315409aad5d4edfc5cd9bbc38e5e06d3a`。
- [CMakeLists.txt](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_implicit_si_20261020/CMakeLists.txt:13) 中 `train_worker` 继续编译 legacy 源码，原 `run_training.py` 继续用于历史实验；修复版独立入口为 **`train_worker_fixed`**。其参数顺序与旧 worker 相同，输出应使用新的目录。

正式修复版仍为 `3→32→32→1 / SiLU`，参数数目由 1217 减至 1216。Gaussian、TF、Master、AABB、Raw SI、candidate population、cached uniform、Bank schedule、photo loss 与 Adam LR 沿用原设置。production Selector 未修改。

**梯度定义**

令 `d=p*(1-p)`、`D=sum(d)`、`h=dL/dp`，修复后的 score gradient 为：

```text
g_i = d_i*h_i - d_i*sum_j(d_j*h_j)/D
```

实现没有对二分过程逐步展开，也没有改变 forward offset。它增加一个 forward 严格为零、backward 为 `d/D` 的 offset 项，使上述 VJP 通过普通 autograd 生效。仅保证本实验 Adam 所需的一阶导数；不提供校准器的高阶导数。

与只做 centering 相比，此修复同时处理了预算校准引入的非均匀加权项。若导数质量退化为 `D<=0` 或非有限数，立即报错，不作 silent clamp 或 fallback。

**验证结果**

| 检查 | 结果 |
|---|---|
| 修复版 native C++ 编译 | 通过 |
| CPU / CUDA，给定相同 score 的 forward 概率 | 与原公式 exact 相同 |
| 一阶 gradient 对重新校准的连续目标有限差分 | 最大绝对误差 `1.84e-8` |
| 公共 score 平移、保留 bias 的 centered toy | 梯度接近 0 |
| 原审计中沿错误方向上升的 smooth toy 反例 | 修复后的方向降低该连续目标 |
| 三个 scene 已保存 checkpoint 的 Bank backward | 全部通过，冻结 tensor exact |
| 三个 scene 各 100 步，5% / seed2026 / LR=1e-4 | 全部数值 finite；Gaussian/TF/Master exact |
| 300 步中的 `abs(sum grad)/sum(abs grad)` | 最大约 `1.24e-16` |
| correction 均值绝对值 | 最大约 `1.00e-18` |
| 初始化 correction / Raw ranking | exact 对齐 |
| cached uniform / Bank frame schedule | 与 legacy exact 一致 |
| 保存权重独立 Python forward / 最终 hard 排序 | 通过 |

真实 backward 探针使用三个 scene 的旧 `5% / seed2026 / step2000` checkpoint，只用于检验修复路径。探针保留旧 head bias 以验证其梯度接近零；它们没有 optimizer update，也未用作修复版训练初始化。修复版 100 步检查均从最后层零权重重新开始。

**Bank 有限训练检查的边界**

| scene | 初始 Bank hard photo | step100 | 最终 correction std |
|---|---:|---:|---:|
| fr1_room | 0.1526204585 | 0.1526204585 | 0.00001128 |
| fr1_xyz | 0.1215807369 | 0.1215807369 | 0.00066006 |
| fr2_xyz | 0.1149300252 | 0.1149300252 | 0.00026675 |

网络输出已有更新，但 100 步尚未跨越 hard Top-K 边界。因此验证文件分别保存 `all_correctness_checks_passed=true` 与 `formal_training_ready=false`，不会把 finite/有更新误标为收敛成功。这不等于已经验证 2000 步无效，也不构成质量改善结论。本轮未调整 LR、未重跑 24 模型、未读取 non-Bank 质量、未加入其他优化策略。

此修复匹配的是预算校准后的 sigmoid relaxation。cached Bernoulli、selected-only opacity ST 与最终 deterministic hard Top-K 之间的目标差异仍然存在，不能把修复后的梯度称为无偏离散 subset 梯度。

**可复核文件**

- [verification.json](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_optimization_fix_20261020/verification.json)：数值、Bank backward、有限训练检查的结果及源码 SHA。
- [selection_regression.cpp](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_optimization_fix_20261020/selection_regression.cpp)：CPU/CUDA forward、有限差分、gauge 与上升方向反例回归。
- [native_probe.cpp](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_optimization_fix_20261020/native_probe.cpp)：真实 compact renderer 的 backward 检查。
- [run_checks.py](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_optimization_fix_20261020/run_checks.py)：固定检查协议，分别记录数值正确性和 Bank loss 改善；未通过后者不启动正式矩阵。
- [worker.patch](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_optimization_fix_20261020/worker.patch)：worker 相对按字节保留的 legacy 的改动。

远端构建与检查位于 `/home/prmi/zwj/spatial-mlp-optimization-fix-20261020`，源码副本与本地修复版 SHA 一致。日志与 step0/100 权重、correction、uniform 快照已保存到本地检查目录。没有 stage、commit、push。
