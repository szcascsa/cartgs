# 共享 Anchor Feature 梯度与 Adam 状态逐节点诊断

## 结论

**在本轮固定 checkpoint、选择集合与上游梯度的实测范围内，首个位差来自原生 ATen `index_select` 反向的 Scatter 原子累加，没有发现 Residual48 融合公式、局部 VJP 或生产 Adam 实现引入新的计算错误。**

原版重复运行也会发生这一差异。融合版与原版的前向、Loss、网络输入 VJP、进入 GatherBackward 的梯度、Master XYZ 梯度和全部 21 个 Compensation 网络参数梯度均逐 bit 一致；首差出现在多个 Gaussian 的梯度加回同一 Anchor Feature 行时。

**“来源是正常 FP32 舍入”不等于“对训练没有影响”，也不等于通过 Zero-Change 准入。** 在真实 Photo Loss 的单视图诊断中，最大 Feature 梯度差为 `4.65661e-10`，单步 Adam 后参数最大差为 `7.45058e-9`。随机 VJP 连续压力测试中，小差异可经 Adam 和网络反馈显著放大；尚未证明真实在线长期质量无损。因此，上一轮被拒绝的训练融合仍未启用，生产默认没有改变。

## 1. 实验范围与可追溯性

- 实验与节点核验时间：2026-11-01 23:05:16—23:26:03（UTC+8），约 **20.77 分钟**；此时间不包含随后报告整理与归档。
- 独立诊断目录：`/home/prmi/zwj/experiments/anchor_feature_gradient_diagnosis_20261101_2245`。目录名时间仅是标识，实际起止以 `start.json`、`FINAL_STATUS.json` 为准。
- 只读复用上一轮源码与构建：`/home/prmi/zwj/experiments/residual48_cuda_optimization_20261101_1937/source`。
- 服务器生产项目：`/home/prmi/zwj/mycartgs`。
- 当前研究对象：每 Anchor 一张 48D Base 表、五张独立 48D Ratio Residual 表，经原顺序累加、Offset3 及 Ratio3 输入共享网络。
- 恢复上一轮被拒绝的训练输入融合到独立 `diagnostic_cuda` 命名空间；未把诊断代码加入生产库。编译使用 FP32、CUDA 11.8、sm86、`--fmad=false`，关闭 TF32。
- Gather/Scatter 使用现有原生 ATen 路径；没有以自写 Scatter 替换原实现。

主逐节点 fixture 为真实 room0 checkpoint：Gaussian `N=62,555`、Anchor `29,387`，checkpoint SHA256：

```text
fb8ab5de2c11fe12c11d30f221dccc2662d53a00a2d9bac6e745d736a622cc640
```

六档训练 ratio 的实际 K 为：

| Ratio | 1% | 5% | 10% | 20% | 30% | 40% |
|---|---:|---:|---:|---:|---:|---:|
| K | 627 | 3,129 | 6,257 | 12,512 | 18,768 | 25,023 |

另外覆盖 `2.5%、7.5%、15%、25%、35%`。训练节点复用 fixture 的真实选择集合；未见 ratio 的节点测试使用 40% fixture 集合的前 K 子集，验证输入和反向数值路径，**没有将其当作未见 ratio 的 Selector 质量实验**。

每个 ratio 使用 3 个随机 VJP seed；分别比较一次原版重复与两次融合执行。所有分支使用相同参数、owner、上游梯度，禁止在差异发生后重新挑选输入。

## 2. 梯度节点的定义

对于第 j 张 Feature 表：

\[
E^{(j)}\in\mathbb R^{A\times48},\qquad
g^{(j)}_i=E^{(j)}_{a(i)},\qquad
z_i=g^{(0)}_i+\sum_j w_j(r)g^{(j)}_i.
\]

逐节点记录：

```text
Feature 表 E
    → index_select(owner) → Gather 值 g
    → 原序 Residual 累加 + Offset 输入打包
    → 原 Compensation 网络 → Loss
    ← packed input VJP
    ← dL/dg：进入 GatherBackward 的梯度
    ← 原生 Scatter：dL/dE[a] = Σ_{i:a(i)=a} dL/dg[i]
    ← leaf.grad
    → Adam 一阶矩 / 二阶矩 / 独立表时钟
    → 参数更新
```

CSV 的 `gather_input_VJP_j` 指 **GatherBackward 收到的 `dL/dg`**，不是对 Feature 表的最终梯度。`scatter_output_j` 是表 leaf hook 收到、进入 AccumulateGrad 前的梯度；在此隔离图中每张表只有对应的 Gather 分支。另以保存快照检查同一分支 Scatter hook 值与最终 `leaf.grad`，避免把 AccumulateGrad 混为 Scatter。

bitwise 比较将连续 FP32 Tensor 按 Int32 位模式比较，同时记录 shape、undefined grad、不同元素数、最大绝对差及相对 L2。不是用 `allclose` 或 PSNR 替代位检查。

## 3. 首个位差定位

正常非确定性 Scatter 模式下：

| 节点 | 原版重复 | 原版 vs 融合 |
|---|---|---|
| Gather 前向值 | 逐 bit 一致 | 逐 bit 一致 |
| 打包输入 / 网络输出 / Loss | 逐 bit 一致 | 逐 bit 一致 |
| packed input VJP | 逐 bit 一致 | 逐 bit 一致 |
| GatherBackward 输入 `dL/dg` | 123/123 一致 | 246/246 一致 |
| Scatter 输出 `dL/dE` | 123/123 比较出现位差 | 246/246 比较出现位差 |
| Scatter 输出最大绝对差 | `3.81470e-6` | `3.81470e-6` |
| Scatter 最大相对 L2 | `2.90066e-8` | `3.00402e-8` |
| Master XYZ VJP | 逐 bit 一致 | 逐 bit 一致 |
| 21 个网络参数梯度 | 逐 bit 一致 | 逐 bit 一致 |

这是每个活跃表分别统计后的比较数量。未激活表的 undefined gradient 也单独保留和比较。

两图 GatherBackward 输入的 stride 也相同：Base 为 `(51,1)`，Residual 为 `(48,1)`；Scatter 输出为 `(48,1)`。不能用“融合改变了梯度布局”解释此次首差。无 hook 的六档复测也出现相同类型的表梯度位差，排除了仅由观察 hook 造成的问题。

补充的同分支自检查覆盖 **44 份快照、328 项 Scatter→leaf.grad 比较，全部逐 bit 一致**。因此，已观测的首差在 Scatter 内部，而非 Scatter 之后的 AccumulateGrad。

证据：`nodes.csv`、`NODE_SUMMARY.csv`、`gradient_layout.csv`、`ACCUMULATEGRAD_SELF_CHECK.csv`。

## 4. 不依赖融合实现也可复现

### 4.1 只重放原始 Scatter

绕过整个 Compensation 网络，对同一份固定 `dL/dg` 执行原生 `index_select(...).backward(gradient)`。11 ratio，分别测试原 stride、contiguous、每 Anchor 仅一个 owner，每条件执行 24 次。

| 条件 | 结果 |
|---|---|
| 正常 Scatter，保留原 stride | 每条件除首份 reference 外，23/23 重放出现位差 |
| 正常 Scatter，contiguous 梯度 | 同样 23/23 出现位差 |
| 正常 Scatter，唯一 owner | 全部逐 bit 一致 |
| 确定性 Scatter，所有布局 | 全部逐 bit 一致 |

原生 Scatter 重放最大差可达 `5.72205e-6`。这证明产生该类误差不需要运行融合 kernel，也不需要网络反向。

### 4.2 Nsight 与对应版本源码

专用 Nsight trace `scatter_only.nsys-rep` / `scatter_only.sqlite` 显示正常模式实际执行：

```text
at::native::indexFuncLargeIndex<..., ReduceAdd>
```

官方 PyTorch v2.3.1 `aten/src/ATen/native/cuda/Indexing.cu` 中，`ReduceAdd` 第 307 行调用 `fastAtomicAdd`。当多个 Gaussian 共享同一 Anchor 时，同一 Feature 元素收到多次 FP32 原子加法；执行先后不固定，浮点加法又不满足结合律，因而结果可能不同。

该版本第 868—875 行在 `deterministicAlgorithms()` 下切换到 `index_put_(..., accumulate=true)` 路径；Nsight 也实际观察到排序相关 kernel。启用这一诊断模式后，本轮所有逐节点比较，包括完整表梯度及 Adam 状态，全部逐 bit 一致。

官方来源：<https://raw.githubusercontent.com/pytorch/pytorch/v2.3.1/aten/src/ATen/native/cuda/Indexing.cu>。下载副本与 SHA 保存在实验目录。确定性开关仅在独立诊断进程使用，没有修改生产行为。

## 5. FP32 舍入的定量验证

对 11 ratio 的 **Base 表**，使用同一 `dL/dg` 在 CPU FP64 求和作为参考；实际被选成员最大 occupancy 为 10。

采用 FP32 顺序求和的误差界：

\[
u=2^{-24},\quad\gamma_n=\frac{nu}{1-nu},\quad
|\widehat{\sum x_i}-\sum x_i|\le\gamma_n\sum|x_i|.
\]

- 原版和融合版均为 **0 个元素越界**；判定脚本保留 `1e-15` 的绝对比较余量。
- 相对 FP64 的 L2 误差约 `2.0e-8～2.7e-8`。
- occupancy 0、1、2 的行没有出现两图位差；位差集中于 3～10 成员的行。
- 从不同 ratio 选取 44 个具体差异元素，穷举 3～7 个加数的 FP32 顺序；**原版与融合结果全部能由同一组加数的合法累加顺序得到**。

一个具体例子：ratio 1%，Anchor 83，channel 18，四个输入梯度为：

```text
[-0.24763211607933044, -0.5285226702690125,
  0.24380213022232056, -0.04995004087686539]
```

| 求和 | 结果 | 一种合法加数顺序 |
|---|---:|---|
| 原版 | `-0.5823027491569519` | `[3,1,0,2]` |
| 融合 | `-0.5823026895523071` | `[3,2,1,0]` |
| FP64 参考 | `-0.5823026970028877` | — |

该例没有缺失梯度、错误 owner 或不同的乘法输入；不同的是合法 FP32 加法顺序。此穷举证明覆盖所选 44 个元素，不能声称穷举了所有表的全部加数排列。

证据：`FP32_ROUNDOFF_BOUNDS.csv`、`OCCUPANCY_ERRORS.csv`、`VALID_ACCUMULATION_ORDERS.csv`、`SCATTER_REPEAT_ENVELOPE.csv`。

## 6. Adam 是否另外引入错误

### 6.1 审计实际生产参数

生产 `src/gaussian_model.cpp` 中 Master Adam 的 epsilon 为 **`1e-15`**。`GaussianMapper::ensureSharedMappingOptimizer()` 把 `selector_network_->parameters()` 加入同一个 Master optimizer，只设置该组学习率；其中包含 Compensation 及 Anchor Feature，epsilon 继承 Master 默认。不能据单独 Selector optimizer 的构造默认值，误认为在线 Joint 的 Feature 使用 `1e-8`。

本轮使用生产 `mapping_cuda::tryFusedDenseAdam()`，`lr=0.001、eps=1e-15`，默认 betas，与该在线组一致。测试冷启动状态及经过 18 步相同历史更新的 warm 状态；六表独立 leaf、独立状态与时钟，非活跃表保留 undefined gradient。没有把 inactive grad 改成零 Tensor。

### 6.2 对照结果

| 对照 | 结果 |
|---|---|
| 相同梯度 → 两份生产 Adam | 参数、update、一阶矩、二阶矩、表时钟均逐 bit 一致 |
| 相同梯度 → LibTorch 原生 Adam vs 生产 fused Adam | 已测参数与状态均逐 bit 一致 |
| 确定性 Scatter → 原版 vs 融合 → Adam | 全部已测节点逐 bit 一致 |
| 正常 Scatter → 原版重复 → Adam | 参数最大差 `3.63216e-6` |
| 正常 Scatter → 原版 vs 融合 → Adam | 参数最大差 `5.85243e-6` |

**Adam 没有制造第一处差异；它将已不同的 Scatter 梯度传递到一、二阶矩，再传递到参数。**

### 6.3 小二阶矩与梯度抵消会放大差异

一个 warm 状态的实际元素：ratio 40%，table 4，Anchor 11186，channel 14。

| 项目 | 数值 |
|---|---:|
| 梯度 A | `2.20537185669e-6` |
| 梯度 B | `2.44379043579e-6` |
| 更新前一阶矩 | `1.23049846934e-6` |
| 更新前二阶矩 | `4.11643002942e-14` |
| 更新前参数 | `0.0210285913199` |
| 更新后参数 A | `0.0200369581580` |
| 更新后参数 B | `0.0200311057270` |
| 参数差 | `5.85243105888e-6` |

上游各项抵消使总梯度接近零时，相同的绝对舍入差对应的相对梯度差可能较大；小二阶矩下，Adam 的有效缩放也更敏感。冷 Adam 的 40% 测试参数最大差仅 `4.65661e-10`，所以只看冷状态会低估风险。此次未改 epsilon，也没有证明修改 epsilon 的收益。

证据：`ADAM_SUMMARY.csv`、`adam_nodes.csv`、`adam_amplification.csv`、`ADAM_CANCELLATION_SUMMARY.csv`。

## 7. 连续更新与因果干预

固定 checkpoint、Gaussian/Anchor owner、ratio 序列及逐步随机上游 VJP，联合更新网络和六张 Feature 表；每条轨迹 72 步，使用生产 Adam 参数。Master 和选择决策固定。

**这是随机 VJP 压力诊断，不是真实 Photo Loss 在线训练；数值不能换算成 PSNR 或实际 SLAM 质量损失。**

确定性 Scatter 下，2 个 seed 的 72 步全部已观测节点保持逐 bit 一致，包括输出、网络参数、Feature 参数及其一、二阶矩和时钟。

正常 Scatter 下，首次测试和四次复验合计每种比较 10 条轨迹：

| 72 步轨迹统计 | 原版重复 | 原版 vs 融合 |
|---|---:|---:|
| Feature 参数峰值绝对差的中位数 | `2.259931e-4` | `2.259931e-4` |
| Feature 参数峰值绝对差的最大值 | `0.00788910` | `0.00982821` |
| Feature 参数峰值差 > `1e-3` 的轨迹 | 1/10 | 4/10 |
| 网络输出最大峰值差 | `0.0791768` | `0.307289` |

原版自身也能进入差异显著放大的轨迹。与此同时，融合比较的极端轨迹次数更多；**这些小样本不支持宣布两者长期误差分布相同**。计算图调度可能改变合法原子累加顺序及其进入敏感优化状态的机会，仍需真实训练轨迹检验。

另作因果干预：保持正常原子 Scatter，每步仅在 Adam 前把两支 Feature 表梯度替换成 reference 原版的同一份梯度，其他运算不变。替换前 Scatter 仍出现位差，但 2 seed ×72 步的前向、packed VJP、Gather 输入 VJP、网络梯度、网络/表参数及表一、二阶矩全部恢复逐 bit 一致。

这一结果将长期分歧定位到 **Scatter 舍入 → Adam → 后续网络反馈**，没有暴露另一条独立的融合公式错误。替换梯度只是因果诊断工具，不是可部署修复。

证据：`TRAJECTORY_DISTRIBUTION.csv`、`TRAJECTORY_SUMMARY.csv`、各次 `trajectory_repeat*.csv`、`SCATTER_RESCUE_SUMMARY.csv`。

## 8. 真实 Photo Loss 上游梯度复核

另使用上一轮 office2 的 OFF checkpoint，Gaussian `N=14,846`，固定一个真实视图，原生 1200×680 Renderer、物理 Scale 加法解码和原 Loss：

\[
\mathcal L=0.8\,L_1+0.2(1-SSIM).
\]

每个 ratio 只渲染并求一次 `dL/dDelta14`，随后固定这份上游梯度，隔离 Rasterizer 自身 atomic 反向对 Compensation 首差定位的干扰。六档实际 K 分别为 `150、744、1486、2970、4455、5939`。

| 节点 | 原版重复最大差 | 原版 vs 融合最大差 |
|---|---:|---:|
| 前向 / Loss / packed VJP / Gather 输入 VJP | 0，逐 bit | 0，逐 bit |
| Master XYZ / 网络参数梯度 | 0，逐 bit | 0，逐 bit |
| Scatter / leaf Feature 梯度 | `5.82077e-11` | `4.65661e-10` |
| 单步 Adam 后 Feature 参数 | `7.45058e-9` | `7.45058e-9` |

固定相同表梯度后的 Adam 对照逐 bit 一致；确定性模式下全部已测节点与 Adam 状态逐 bit 一致。

真实 Photo 上游的单步差异远小于随机 VJP 压力测试。但本实验只有一个视图、固定上游和单步 Adam，不含完整 Selector/Maturity 双 VJP、动态 Master/拓扑及异步在线调度，不能冒充完整在线训练等价证明。

证据：`photo_cases.csv`、`PHOTO_NODE_SUMMARY.csv`、`PHOTO_ADAM_SUMMARY.csv`、`photo_nodes.csv`、`photo_adam.csv`。

## 9. 诊断实现修正与失败尝试

首次诊断原型使用硬编码 double ratio knots，checkpoint 实际保存的 knots 来自 float，导致两个训练节点的原 API 与展开 baseline 前向出现差异。已将实际 Capture 改为 `anchor->compensationResidualWeights(ratio)`，复用 checkpoint 的真实权重规则；重跑后原生产 API 与可观测 baseline 在六训练 ratio 的前向、Loss、Master XYZ 和参数梯度全部逐 bit 一致。

首次结果保留在 `attempt1_hardcoded_knots/`，**未纳入正式统计**。此问题属于诊断脚本；上一轮融合候选原本就从 AnchorBank 获取权重，不能据此推断生产存在 ratio 插值错误。Adam 工具中的硬编码 helper 仅用于确定需要读取的活跃表数量，不参与前向权重计算。

另有一次 Nsight 导出路径冲突，正式分析使用成功导出的 `scatter_only.*`；初始日志保留。汇总脚本也修正了 `table_gradient` 名称归类，只影响汇总断言，不影响原始 CSV 和 GPU 运算。

## 10. 对方法、数值误差及准入的判断

| 问题 | 证据支持的回答 |
|---|---|
| Gather 是否读错 Feature 或 owner？ | 已测前向值逐 bit 一致，未发现 |
| 融合的 Residual 累加 / Offset / 局部 VJP 是否算错？ | 已测前向与进入 GatherBackward 的 VJP 逐 bit 一致，未发现 |
| 是否 Scatter 后还有 AccumulateGrad 误差？ | 44 份快照的 328 项同分支比较全一致，未发现 |
| 为什么共享 Anchor 更容易出现差异？ | 多 Gaussian 对同一 Feature 元素做原子求和；唯一 owner 对照无差异 |
| 是否生产 Adam kernel 自身算错？ | 同梯度与原生 Adam 对照均逐 bit 一致，未发现 |
| 是否是正常浮点误差来源？ | 原生 Scatter 可独立复现，FP64 误差界及具体累加顺序支持这一判断 |
| 能否认为这些误差无害、直接启用训练融合？ | 不能；压力轨迹可放大，完整真实训练长期影响尚未证明 |

共享 Feature 方法在实数意义下的梯度是明确的求和。当前证据没有否定这一表示方法；它暴露的是实现上的非确定 FP32 归约与优化器敏感性。融合可能改变原子累加的时序，即使没有改变数学公式，也仍可能改变最终训练轨迹，因此原版非确定性不能作为自动放行依据。

## 11. 后续最有信息价值的验证

1. **固定真实 Photo 决策 Replay**：固定视图、ratio、owner、严格 K 与 Master，先做逐步真实 Photo 上游轨迹；重复多条原版及融合轨迹。分别记录首差、表梯度抵消程度、Adam 二阶矩、输出和图像差，不用单次 PSNR 接近代替数值定位。
2. **逐步统一 Scatter 输出的因果对照**：沿用本轮 rescue，检验真实 Photo 轨迹是否同样恢复 bitwise；这是定位工具，禁止加入训练路径作为“优化”。
3. **若研究可重复 Scatter，将其单独作为数值实现方案**：排序/固定顺序归约会改变原有非确定累加结果，并增加成本；需要独立性能与质量验证，不能夹带进 Zero-Change 融合。此次不部署确定性模式。
4. **若研究 Adam 敏感性，独立评估 epsilon 等优化器假设**：本轮只证明 `eps=1e-15` 下小二阶矩状态可能放大差异，没有验证更大 epsilon 的收益；它属于训练算法变更，不能作为本轮 CUDA 等价修复。

当前保留原生产训练路径，不放宽逐 bit 标准。此次没有继续运行新的完整在线 SLAM 或声称已完成长期 PSNR 回归。

## 12. 交付与完整性

- 新增内容限于独立诊断源码、编译/运行脚本、CSV、Nsight trace、原始 Tensor 快照、中文报告与 SHA 清单。
- 最终实验总断言 `DIAGNOSIS_VERIFICATION.json` 为 passed；追加自检查 `ACCUMULATEGRAD_VERIFICATION.json` 为 passed。
- 服务器生产项目 **328 个受保护文件前后 SHA256 全部一致**；复用研究源码 **224 个文件与上一轮 accepted manifest 全部一致**。范围以 `main_before.json`、`main_after.json` 与 accepted manifest 为准，不将其扩大为整个服务器文件系统的完整性声明。
- 没有修改生产源码、默认配置或原构建；没有修改共享 conda 环境；没有 commit/push；本轮诊断进程已结束。
- 正式 Tensor 快照 111 份，共 `6,019,965,704` 字节，留在服务器 `snapshots/`，清单 `SNAPSHOT_MANIFEST.json` 记录路径、大小与 SHA256。首次无效 attempt 也保留，不参与正式统计。
- 代码、CSV、日志、JSON、报告与正式 Nsight trace 归档到本地 `research/anchor_feature_gradient_diagnosis_20261101/downloaded/`；为避免重复占用磁盘，不下载全部 Tensor 快照。

复现优先阅读 `runtime.json`、`compile_*.json` 的真实命令、对应 `.log` 和各汇总 CSV。`common.py` 中服务器路径限定为本次隔离目录；不要在生产项目内直接执行诊断确定性开关或 rescue。
