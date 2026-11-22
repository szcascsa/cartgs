# Spatial MLP 优化路径代码审计

后续状态：本报告记录修复前的审计证据。修复前 worker 已按字节保存在 `tests/diagnostics/spatial_mlp_implicit_si_20261020/train_worker_legacy.cpp`；当前 `train_worker.cpp` 为修复版。修复改动及验证结果见 [SPATIAL_MLP_OPTIMIZATION_FIX.md](H:/AIProjects/cartgs优化更细致版/mycartgs/SPATIAL_MLP_OPTIMIZATION_FIX.md)。

结论：当前 MLP 存在可复现的 **forward 预算校准与 backward surrogate 不一致**。无效的公共 score 平移仍获得梯度，并通过共享网络与相对空间场耦合。这是应优先处理的优化问题。当前负结果不足以判定 spatial sharing 或 tiny MLP 的表达能力无效；同样，现有证据不能保证 gauge-fix 后质量一定恢复。

审计日期：2026-10-20。代码基点：`8cb6b767e6a59d7ad697710cee773f8eb2b2d7d3`。本轮只检查代码、重读已保存模型与日志、运行隔离数值测试和 checkpoint backward 探针，新增 optimizer update 为 **0**。没有运行附件的 24 模型 Gauge-Fixed 训练，没有修改旧训练器、production Selector、TF、Current C；没有 stage、commit、push。

**审计对象与完成标准**

当前 XYZ MLP 位于 [train_worker.cpp](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_implicit_si_20261020/train_worker.cpp:82)，是独立研究 worker。production 的 [GumbelNetwork](H:/AIProjects/cartgs优化更细致版/mycartgs/improvements/selector/gumbel_network.h:29) 使用 position/rotation/scale/ratio embeddings；其训练入口为 [gaussian_mapper.cpp](H:/AIProjects/cartgs优化更细致版/mycartgs/src/gaussian_mapper.cpp:3098)。本报告不将两者混为同一模型，也不把这里的 detached-offset 问题外推到没有这套校准的 production 路径。

完成标准为：定位实际代码路径；用数值测试核对公共平移与校准梯度；用保存模型的真实 Bank rendering backward 交叉验证；区分确定问题、既定实验限制及尚未证明的原因；交付可复核结果，不自动修算法或扩大实验。

本地与探针所用服务器的原 worker SHA256 一致：

`fca6741d92944e176e2185c9a83170a315409aad5d4edfc5cd9bbc38e5e06d3a`

**1. 最高优先级：forward 抵消公共平移，backward 却奖励公共平移**

[worker:97](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_implicit_si_20261020/train_worker.cpp:97) 在 `NoGradGuard` 内通过二分求 offset，再将其存为普通 `double`；[worker:98](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_implicit_si_20261020/train_worker.cpp:98) 计算：

```text
p_i = sigmoid(score_i - offset)
sum_i p_i = K
```

重新校准的 forward 满足 `p(score + c) = p(score)`。所以共同增加所有 score，不改变概率、cached-uniform sample 或 hard ranking。但由于 offset 没有反向路径，当前 backward 使用 `dp_i/dscore_j = delta_ij * p_i(1-p_i)`，对共同平移一般给出非零导数。

这不是二分精度不足：原 48,000 次更新的 `|sum(p)-K|` 最大仅 `5.28e-8`。问题在反向传播的定义。

本轮真实 native 探针覆盖三个 scene、5%/seed2026、step 0/250/1000/2000，共 12 个 checkpoint。每次只使用该 scene 的第一个 Bank frame，执行一次 backward，不做 optimizer step。结果：

| 检查 | 结果 |
|---|---|
| score 全部加 100 并重新校准后，概率最大误差 | `2.15e-14` 以下 |
| 同一 cached uniform 下 sample 是否改变 | 12/12 完全不变 |
| 对公共输出平移的 surrogate gradient | 12/12 非零且为负，约 `-0.01106` 至 `-0.001095` |
| 最后层 bias gradient 与全部 output gradient 之和 | 12/12 相等 |
| Gaussian/TF/Master | 探针前后 exact，未产生梯度 |
| 保存 checkpoint 文件 SHA | 探针前后完全一致 |

因此，在这些 Bank 探针上，gradient descent 会增大公共输出，尽管该方向不能改变实际 selection。纯输出 bias 本身只影响公共方向；隐藏层和共享权重同时影响公共量与相对量，错误的公共梯度可以经由它们影响相对 field。这解释了为什么问题对共享网络尤其值得检查，但还不能将全部质量损失唯一归因于它。

证据：[native_gradient_probes.json](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_optimization_code_audit_20261020/native_gradient_probes.json)。这是 12 个检查点的局部梯度证据，不代表全 48,000 步梯度均已测量。

**2. 真实输出更新大部分落在无效公共分量；只删 output bias 不足以修复**

[worker:84](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_implicit_si_20261020/train_worker.cpp:84) 直接返回网络输出，没有 centering。对保存的 24 个最终 field 和 192 个单步 before/after 快照重新分解：

```text
delta = correction_after - correction_before
delta_common = mean(delta)
delta_relative = delta - mean(delta)
common_fraction = mean(delta)^2 / mean(delta^2)
```

| 数量 | 新诊断结果 |
|---|---:|
| 最终 correction 公共分量占输出平方范数比例 | 99.3442%–99.9957% |
| 单步更新公共分量占输出更新平方范数比例 | 99.2677%–99.9993% |
| 单步公共分量占比中位数 | **99.9676%** |
| 最终 correction 均值 | 14.36–49.23 |
| 最后输出层 bias | 0.13284–0.21718 |

这些比例衡量 **Gaussian 输出空间的平方范数**，不能直接解释成同等比例的 optimizer 时间、参数更新或质量损失。即使相对分量很小，也仍可能产生显著 reordering。

12 个真实 backward 探针进一步将最后层 weight gradient 分为：

```text
sum_i g_i h_i
  = mean(h) * sum_i g_i
  + sum_i g_i * (h_i - mean(h))
```

其中 `h_i` 为最后隐藏层特征。完整 weight gradient 与第一项的 cosine 为 **0.99789–0.99999992**，支持公共特征分量在这些探针中很强。

大部分 correction 均值并非最后一个 bias 产生；只将最后层 `bias=false`，其他共享参数仍可产生接近常数的 field。附件中真正关键的是 **全 candidate population 上、保持 autograd 的 mean-centering**。

此外，对已训练输出只在 export 时减均值，24/24 完整 ranking 完全不变。因此必须修训练路径，事后减均值不能恢复已经学坏的相对 ranking。

证据：[field_decomposition.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_optimization_code_audit_20261020/field_decomposition.csv)、[update_decomposition.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_optimization_code_audit_20261020/update_decomposition.csv)。

**3. 附件的 gauge-fix 合理，但并没有修复完整的预算校准 Jacobian**

令 `d_i=p_i(1-p_i)`、`D=sum(d_i)`，令 `g_i` 为当前 detached-offset 路径得到的 score gradient。由 `sum(p)=K` 的隐函数导数，校准 sigmoid relaxation 的正确 VJP 为：

```text
J_calibrated = diag(d) - d d^T / D
g_calibrated_i = g_i - d_i * sum(g) / D
```

附件采用：

```python
r = mlp(normalized_xyz).flatten()
a = r - r.mean()       # mean 必须保留梯度；population 必须完整
score = base + a
```

在 offset 仍 detach 的情况下，它产生：

```text
g_centered_i = g_i - mean(g)
```

二者都消除了公共 output gradient，但一般 **不相等**。`r-r.mean().detach()` 甚至不能消除 backward 的公共梯度；仅 forward 零均值不足以完成 gauge-fix。

FP64 隔离 toy 验证：

| 检查 | 结果 |
|---|---:|
| detached 公共平移梯度 | `-0.2605101358`，但 forward 不变 |
| differentiable centering 后公共平移梯度 | `5.55e-17` |
| implicit calibrated gradient 与重新校准的 smooth objective 中心差分最大误差 | `8.07e-12` 以下 |
| centered gradient 与 calibrated gradient 的 L2 差异 | `0.0559526` |

真实 12 个 Bank 探针中，两种修正 gradient 的相对 L2 差异约 **0.70%–6.77%**。toy 还构造了一个确定性反例：沿 detached gradient 做小步下降，重新校准的连续局部目标反而上升；centering 在该 score-space 反例中也未消除上升方向。

这里检验的是 **校准后的连续概率 relaxation / 局部 opacity upstream**，不是要求 ST 等于离散 hard Top-K 的有限差分梯度。即使正确实现 implicit VJP，也不能据此声称得到了无偏的离散 subset 梯度或保证 hard quality 提升。

证据：[toy_gradients.py](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_optimization_code_audit_20261020/toy_gradients.py)、[toy_gradients.json](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_optimization_code_audit_20261020/toy_gradients.json)。

**4. 优化目标仍是 selected-only opacity surrogate，不能用 output coverage 证明 ranking gradient 有效**

实际路径是 [worker:98](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_implicit_si_20261020/train_worker.cpp:98) 先将未选行 compact 掉，再用 selected mask 乘 opacity（[worker:57](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_implicit_si_20261020/train_worker.cpp:57)）。[sampleSelectorST](H:/AIProjects/cartgs优化更细致版/mycartgs/include/selector_handoff.h:14) 的 backward 使用 probability surrogate。

因此，直接从 photo loss 返回的信号是“在当前 compact set 中微调已选点 opacity 的局部效果”。它没有直接测量“用未选 j 替换已选 i”的等预算质量变化。12 个真实探针中，unselected correction 的直接梯度均为 0，和代码一致。MLP 共享参数会使其输出间接变化，但这不保证变化方向对 ranking 有利。

原报告的 unselected output update coverage=100% 本身没有算错：本轮去掉每步公共平移后，192/192 快照的 unselected relative output 仍全部变化。但该统计无法说明相对变化的质量和大小，必须与 centered 更新幅度、hard same-K Bank loss、boundary crossing 一起观察。

此外，旧训练器只校准期望 K、每模型复用固定 uniform，最终导出 deterministic hard Top-K。在原 48,000 步中，**47,492 步 sampled_N 不等于 K**。这是既定 PG-ST 协议限制，并非新发现的 MLP 实现疏漏；本轮不擅自替换 sampler，也不因为这项限制自动回到 PL。

**5. 当前收敛 gate 太弱：三档 LR 都未改善 Bank hard loss，仍被标记 all_passed**

[run_training.py:31](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_implicit_si_20261020/run_training.py:31) 只要求 finite/nonfrozen，再选择 final Bank hard photo 最小的候选，没有要求任何候选比初始化更好。

| LR，fr1_xyz / 5% / seed2026，250 steps | Bank hard photo 初始 | 最终 |
|---|---:|---:|
| `1e-4` | 0.12158074 | 0.12160411 |
| `3e-4` | 0.12158074 | 0.12375542 |
| `1e-3` | 0.12158074 | 0.24529764 |

选 `1e-4` 符合原预注册规则、没有使用 eval 调参，但只证明它在三者中退化最少，不能称为“优化路径已经有效”。原模型最终 Bank hard photo 有 23/24 case 比初始化更差；按每个 Bank frame 首末各五次访问等权比较，sampled photo 有 21/24 case 更差。这不支持把失败主要归因于 train/eval overfitting。

下一次小范围 sanity 应将“实现 finite/有更新”和“Bank 上是否有有效下降”分开记录。全部 LR 都无正向信号时，应进入优化路径诊断，不自动扩展 LR 搜索，也不从 non-Bank quality 选择 LR/step。

**6. 已观察到近似线性 field 与输入尺度问题，但尚不能直接判定网络容量不足**

[worker:43](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_implicit_si_20261020/train_worker.cpp:43) 确实使用冻结 TF AABB 做归一化，没有动态拟合。其 normalized XYZ std 为：

| scene | X / Y / Z std |
|---|---|
| fr1_room | 0.00697 / 0.01281 / 0.00964 |
| fr1_xyz | 0.07314 / 0.02244 / 0.13832 |
| fr2_xyz | 0.00657 / 0.01802 / 0.06669 |

许多点集中在全局 AABB 内的一小块区域，隐藏特征容易接近相同。对 24 个最终 centered correction 用 normalized XYZ 拟合普通线性平面，R² 为 **0.999211–0.99999962**。当前学出的 field 几乎是公共偏移加全局线性倾斜，而不是复杂的局部 correction。

这是保存模型的描述性证据，兼容“公共梯度驱动共享特征增长”“输入 conditioning”“空间假设过强”等多种解释。不能据此认定理论 MLP 只能表达平面，或直接决定扩大网络/重新归一化。原公式正确执行了用户冻结 AABB 的要求，不能把它叫作 normalization 代码写错。

**没有发现的错误与证据边界**

代码及已有独立复核未显示：Gaussian/TF 被意外优化、Raw SI 被覆盖、candidate row 错位、初始化未对齐 Raw、Bank/eval 混用、训练层数或输入特征与协议不符。新 native 探针再次确认 frozen tensor exact。正式 loss 的系数及调用为 `0.8 L1 + 0.2(1-fast_ssim)`，没有发现此 worker 相对既定 loss 的符号错误；本轮没有重新验证 CUDA renderer/fast_ssim 的全部导数。

最后一层零初始化导致第一步前两层 gradient=0 是正常链式法则结果；后续日志有非零 hidden gradient，不能据此误报 autograd 断开。correction 均值大也不自动表示 sigmoid 饱和，因为 forward 会重新减去 offset。

**建议的最小修复顺序**

| 选择 | 优点 | 局限 / 决策 |
|---|---|---|
| 完整 population differentiable mean-centering，output bias=false，其他配置保持不变 | 最小、可隔离附件的 gauge 假设；消除公共输出方向 | **推荐首先做小范围 Bank-only correctness/sanity**；不能声称完整校准梯度已修好 |
| 单独验证 implicit calibration VJP `g-d*sum(g)/sum(d)` | 匹配预算校准 sigmoid relaxation；可 O(N) 实现 | 应单独控制变量；仍保留 ST 与 hard-ranking 目标差异；本轮只做数学/梯度诊断，没有修改训练器 |
| 当前立即增大网络、改 normalization 或做 grid/split | 尚无足够证据支持优先执行 | 会混淆已确定的优化问题，暂不进入 |

应先检查公共 c/bias gradient、`sum grad_r`、centered output update 和固定 Bank hard same-K 下降；检查通过后再决定是否值得完整训练矩阵。不要把“字段100%变化”“梯度finite”当作 learnability 成功，也不要把一次 gauge-fix 的通过当作 spatial sharing 已被支持。

本轮最确定的判断是：**当前共同平移方向的 surrogate 梯度确实存在问题，且实际 MLP 更新强烈偏向公共输出；附件修复方向有依据，但仅修 gauge 还不足以证明优化路径完整正确。** Spatial sharing 是否有效、质量能恢复多少，仍需受控修复后的 Bank 与 non-Bank 证据回答。

**复核文件**

所有本轮新增证据位于 [spatial_mlp_optimization_code_audit_20261020](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_optimization_code_audit_20261020)。包括 toy 源码/结果、只做 backward 的 C++ 探针及构建日志、12 个 native gradient 记录、24 个 field 分解、192 个单步 update 分解，以及 [correctness.json](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_optimization_code_audit_20261020/correctness.json)。`all_passed` 仅指本轮诊断断言通过，不表示旧 MLP 的优化或质量通过。
