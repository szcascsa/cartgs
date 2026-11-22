# SpatialLocalXYZ CUDA Performance Optimization
## 面向共享神经补偿网络的高性能CUDA实现与严格等价优化

你现在是 `mycartgs` 项目的CUDA性能工程师和神经网络系统优化研究员。

本次任务的目标是：

**以 SpatialFiLM + Anchor单位位移（SpatialLocalXYZ）作为唯一算法基线，在不改变模型数学定义、输出语义和训练行为的前提下，尽可能提高Forward、Backward及完整训练步骤的执行速度。**

本次只研究计算性能，不研究新的网络架构，也不改变Compensation算法。

需要自主完成：

1. 源码与执行路径审计。
2. GPU性能Profiling。
3. 性能瓶颈定位。
4. CUDA/Triton优化实现。
5. 并行计算和Kernel Fusion。
6. Backward专项优化。
7. 严格数值正确性验证。
8. 真实训练与在线集成可行性验证。
9. 最终性能对比及优化报告。

不要只给出理论建议，应实际修改实验实现、编译、运行Benchmark并验证。

---

# 1. 基线定义

首先阅读：

- `NEURAL_V1_RESEARCH_REPORT.md`
- `NEURAL_V1_BEST_ARCHITECTURE.md`
- 相关实验脚本及运行环境
- SpatialLocalXYZ完整实现
- 当前Selector与Compensation的C++代码
- 现有PyTorch Renderer和训练Runtime

不要依据模型名称猜测实现。必须以实际源码中的Forward、Backward、输入字段、数据类型和参数布局为准。

当前研究基线：

**SpatialLocalXYZ = SpatialFiLM + Anchor-unit XYZ displacement**

已知结构：

- 55维输入特征。
- 19维物理特征。
- 36维固定空间Fourier编码。
- 共享64维神经网络主干。
- Ratio-conditioned FiLM。
- XYZ、Scale、Rotation、Other四个输出Head。
- 14维Gaussian属性补偿输出。
- 33,806个共享神经网络参数。
- 不含Gaussian/Anchor私有可训练补偿参数。

其中XYZ补偿采用：

\[
\Delta x_i=s_{a(i)} f_{\theta,xyz}(x_i,r)
\]

即先由共享网络输出Anchor单位的相对位移，再根据对应Anchor参考尺度转换为世界坐标位移。

这个变换必须保留。

上一轮原型测得：

- Forward约1.139ms。
- Forward + Backward约2.614ms。

这些数字只是已有环境下的参考，不得直接作为新优化的严格比较基准。

必须先重新测量原实现。

---

# 2. 最高优先级：保持计算等价

本轮优化不得改变：

- 55维输入的实际定义和字段顺序。
- 特征归一化公式。
- Fourier编码频率与计算语义。
- 网络参数及其数学结构。
- Ratio Conditioning。
- FiLM公式。
- 四Head结构和输出解释。
- Anchor单位位移变换。
- Gaussian属性补偿公式。
- 选择集合和same-K语义。
- Loss。
- Optimizer配置。
- 梯度传播关系。
- 训练数据顺序。
- 网络初始化。
- 参数更新时钟。

允许重新组织计算图、改变Kernel边界、重排内存、融合等价运算。

不允许以改变计算结果换取速度。

特别禁止：

- FP16/BF16替换FP32。
- 未经验证开启TF32。
- 使用Fast Math近似代替原有数学函数。
- 改变Quaternion归一化。
- 改变XYZ/Scale/Rotation/Opacity的变换公式。
- 改变Fourier频率。
- 删除输入特征。
- 削减网络层数或隐藏维度。
- 减少训练曝光。
- 使用近似梯度。
- 停止某些输入或参数的梯度传播。
- 重新引入Gaussian或Anchor私有可训练补偿参数。

## 正确性要求

必须进行两级验证。

### Level 1：Direct Numerical Parity

对同一个Checkpoint和完全相同的输入，比较原PyTorch实现与优化实现：

- 原始55维输入。
- Encoder中间输出。
- FiLM输出。
- 各Head输出。
- 14D补偿。
- 补偿后Gaussian属性。
- 最终Renderer输出。
- Loss。
- 所有可训练参数梯度。
- 如果Master属性存在梯度路径，也检查对应输入梯度。
- Anchor尺度等参与计算的输入梯度。

优先追求FP32严格一致。

如果不同的CUDA归约顺序或FMA导致不能做到bitwise一致，允许有边界明确的浮点舍入差异，但不能改变算法语义。

建议初始验收阈值：

- Forward：`atol=1e-6, rtol=1e-5`
- Backward：`atol=1e-5, rtol=1e-4`

这些只是起始阈值，必须同时记录每个张量的max absolute error、max relative error和RMS error。

接近零的数值不能只用相对误差判断。

如果出现超过阈值的差异，优先定位原因，不要直接放宽阈值。

必要时区分原始数学表达一致、FP32浮点一致与bitwise一致。

### Level 2：Training Step Parity

从相同Checkpoint、Optimizer State、随机状态和训练帧开始：

1. 原实现执行Forward。
2. 原实现执行Backward。
3. 原实现执行Optimizer Step。
4. 优化实现进行完全相同的操作。
5. 对比更新后的全部网络参数。
6. 对比Optimizer Moments。
7. 对比下一步输出和Loss。

至少进行单步、多步以及跨ratio的测试。

注意CUDA浮点误差可能在长程训练中因数值非确定性放大，不能要求随机训练轨迹必然bitwise一致。

但是相同输入下的输出与梯度必须数值一致，且短程参数更新应保持合理的数值等价性。

不能用“最终PSNR变化不大”替代计算正确性检查。

---

# 3. 第一阶段：完整性能Profiling

不要一开始直接重写整个网络。

先对原实现进行精确拆分计时。

使用：

- PyTorch Profiler。
- CUDA Events。
- NVTX。
- Nsight Systems（如果环境可用）。
- Nsight Compute（如果环境可用）。

必须区分CPU Launch开销、GPU Kernel执行时间、显存读写以及GPU等待时间。

建议对以下模块分别统计：

| 模块 | 主要检查内容 |
|---|---|
| Gaussian Gather | 索引收集、非连续访问 |
| Input Encoding | 归一化、asinh、log、协方差 |
| Fourier Encoding | sin/cos、广播与中间张量 |
| Shared MLP | Linear、Activation、GEMM |
| Ratio FiLM | Ratio编码、Scale/Shift |
| Four Heads | 四分支矩阵计算 |
| Output Transform | 14D残差与Anchor尺度转换 |
| Autograd | 每个节点的Backward |
| Scatter/Accumulation | 梯度回写与归约 |
| Optimizer | Adam参数更新及Kernel Launch |

必须输出：

- 每个阶段的CPU时间。
- 每个阶段的GPU时间。
- Kernel Launch次数。
- Tensor Allocation次数（可测时）。
- Tensor读写量估计。
- 主要内存瓶颈。
- Forward/Backward时间占比。
- 实际GPU利用情况。
- 是否出现CPU-GPU同步。
- 是否出现大量极小Kernel。

同时测试不同K：

- 68
- 170
- 341
- 512
- 683
- 854
- 1025

这对应fr1已知的七档same-K预算。

fr2使用实际源地图中的K，而不是直接照搬fr1。

尤其关注K很小时的情况。

**当前网络参数量很小，不能假设主要瓶颈是矩阵乘法FLOPs。Kernel Launch、Tensor构造、内存访问和调度开销可能更重要。**

所有判断必须由Profiling确认。

不要根据理论猜测直接宣布瓶颈。

---

# 4. 第二阶段：输入编码CUDA融合

这是首个优先研究方向。

当前输入包含：

- Normalized World XYZ。
- asinh Offset。
- log Scale。
- SH DC。
- Opacity Logit。
- Covariance Shape。
- 固定Fourier编码。

检查这些计算是否包含：

- 多次Tensor索引。
- 多个逐元素CUDA Kernel。
- 多次cat/stack。
- 广播产生的临时张量。
- 重复归一化。
- 重复读取Gaussian属性。
- 重复计算世界坐标Fourier编码。

## 4.1 Fused Input Encoding

尝试设计单个或少量CUDA/Triton Kernel：

输入：

- Selected Gaussian indices。
- Gaussian原始属性。
- Anchor中心、尺度和必要几何信息。
- 全局归一化参数。
- 固定Fourier频率。

输出：

- 连续布局的55维网络输入Tensor。

尽可能在一次读取中完成：

1. 收集选中Gaussian。
2. 读取对应Anchor信息。
3. 归一化与物理特征计算。
4. Fourier编码。
5. 写入最终55维特征。

尽量避免生成19维物理特征、36维Fourier特征后再进行额外cat。

但必须确认原有操作的数值顺序与语义。

特别注意：

- sin/cos的数值一致性。
- asinh和log的导数。
- 协方差构建过程。
- Quaternion相关运算。
- Anchor尺度广播。
- Gaussian属性更新后的输入有效性。

## 4.2 Input Encoding Backward

如果输入来自可训练Master Gaussian属性，则不能使用detach后的缓存切断梯度。

需要提供正确Backward。

比较：

- PyTorch原生Autograd。
- 自定义Triton Backward。
- 自定义CUDA Backward。

优先消除不必要的中间存储。

但不能通过漏算或跳过任何有效梯度获得加速。

如果固定Master时可安全跳过输入梯度，允许作为显式的Frozen-Master执行路径。

必须与Master Joint路径分开，不允许将Frozen优化错误推广到在线Joint。

---

# 5. 第三阶段：MLP与FiLM融合

分析共享主干中：

- Linear。
- Bias。
- ReLU。
- Ratio Modulation。
- FiLM Scale。
- FiLM Shift。
- 输出Head。

是否存在大量小Tensor运算和Kernel Launch。

## 5.1 Shared Backbone

测试不同等价实现：

- PyTorch Eager。
- `torch.compile`。
- Torch Inductor。
- cuBLAS/cuBLASLt。
- Triton Fused MLP。
- 自定义CUDA Kernel。

优先利用已经优化的矩阵乘法实现。

不要仅因为Triton可写，就假设自定义小矩阵乘法会快于cuBLAS。

当前宽度只有64左右，且K在几十到千级，应针对实际shape测试。

对不同K采用不同实现分支是允许的，但所有分支必须完全遵循同一数学定义。

## 5.2 FiLM Fusion

检查能否把：

`Linear → Activation → FiLM Scale/Shift`

融合到更少Kernel。

特别考虑：

- Ratio条件化只需要计算多少次。
- 同一Ratio下的条件特征是否被重复计算。
- FiLM参数是否可以批量广播。
- 是否能避免每个Gaussian重复计算相同的Ratio网络输出。
- 中间激活是否可以在Backward中高效复用。

如果同一批次所有Gaussian使用同一个Ratio，可以预先计算共享Ratio条件向量，然后在融合Kernel中使用。

但是必须保留完整的共享参数梯度聚合。

不能错误地将Ratio网络的梯度计算次数减少为一次而丢失所有Gaussian的贡献。

## 5.3 Four-Head Parallelism

当前包含：

- XYZ Head。
- Scale Head。
- Rotation Head。
- Other Head。

研究以下方式：

- 多Head统一调度。
- Grouped GEMM。
- Batched GEMM。
- Fused Head Kernel。
- 按分支并行。
- 合并中间激活存储。
- 减少重复读取共享输入。

如果四个Head结构相近，可以研究将独立权重打包，使GPU在一次调度中处理多个分支。

但必须保留原来四Head的权重独立性与梯度。

不允许将其改为一个共享隐藏层的统一Head。

对于极小K，多个CUDA Stream同时启动四个Head未必更快。

必须测量Stream同步与Launch开销，再决定是否采用并行Stream。

---

# 6. 第四阶段：Backward专项优化

**Backward是本轮的重点之一。**

不要仅优化Forward之后便停止。

需要区分：

- Encoder Backward。
- FiLM Backward。
- FourHead Backward。
- Output Transform Backward。
- Input Encoding Backward。
- Gather/Scatter Backward。

分析：

1. 哪些中间张量必须保存？
2. 哪些梯度可以融合计算？
3. 是否存在重复归约？
4. 是否存在重复读取参数和激活？
5. 是否发生非必要的Tensor转置或contiguous拷贝？
6. 是否存在过多小矩阵GEMM？
7. 哪些操作适合融合到一个Kernel？
8. 是否有多个Head对共享输入的梯度可以高效累积？

探索：

- Fused Linear + Activation Backward。
- Fused FiLM Backward。
- Multi-head Gradient Accumulation。
- Fused Input Encoding VJP。
- 低开销Shared Feature Gradient Reduction。
- 减少Autograd Node数量。
- 保存最小必要Activation。
- 合理利用Register与Shared Memory。

允许通过Triton或CUDA Extension实现自定义Autograd Function。

但必须严格验证：

- 每个Head的参数梯度。
- Shared Backbone梯度。
- FiLM网络梯度。
- 原始输入梯度。
- Anchor尺度相关梯度。
- 所有ratio下的梯度。

不得通过改变梯度路径获得速度。

对每个自定义Backward实现，优先运行独立Gradient Parity Test，再进入真实训练Benchmark。

---

# 7. 第五阶段：缓存、并行调度与减少重复计算

研究哪些计算可以安全缓存。

必须区分：

## Frozen Master

在Master属性和Anchor拓扑不变时，可以考虑缓存：

- 固定空间Fourier特征。
- 不依赖Ratio的原始输入部分。
- 静态归一化参数。
- Gaussian到Anchor的映射。
- 其他经过证明不变的确定性特征。

## Online Joint

Master与拓扑会变化，必须研究缓存失效。

涉及：

- XYZ/Scale/Rotation更新。
- Gaussian Clone/Split。
- Prune。
- Anchor Growing。
- Anchor重新分组。
- Anchor中心与尺度变化。
- 归一化参考变化。

任何缓存必须具备正确的版本管理或失效机制。

不能在Master属性更新后继续使用过期特征。

不能缓存需要参与当前Autograd的中间值并错误地detach。

重点调查：

- Gather与编码能否融合。
- 相同Ratio条件是否可以复用。
- 当前训练步骤中是否存在重复前向。
- Tensor分配是否可以复用。
- CPU-GPU同步是否可以消除。
- 是否存在不必要的 `.item()`、CPU mask或同步操作。
- CUDA Graph能否在固定shape下减少Launch开销。

CUDA Graph只在形状、内存地址、Optimizer执行方式与训练语义满足条件时使用。

对于动态K、Gaussian增长和稀疏训练，需要单独评估Graph Capture与维护成本。

不能为适配CUDA Graph改变选中数量或训练行为。

---

# 8. 性能测试必须公平

建立统一Benchmark Harness。

对每种实现进行：

- GPU预热。
- CUDA Event计时。
- 正确同步。
- 多次重复。
- Median、P10、P90。
- 不同K。
- Forward-only。
- Forward+Backward。
- 完整Optimizer Step。
- 实际Renderer+Loss训练步骤。

确保Benchmark使用相同输入、网络权重、梯度路径与Batch。

不要将包含N行Encoder的旧Baseline59与仅K行编码的新网络直接比较，然后声称纯Kernel加速。

本轮主要比较对象应该是：

**原始SpatialLocalXYZ PyTorch实现 vs 优化后的SpatialLocalXYZ。**

两者必须执行同样范围的计算。

其他模型只能作为背景参考。

GPU存在其他作业时，应尽量获得无干扰测试窗口；无法排除干扰时必须标注并使用配对交错重复，不能对小幅性能差异作出过强结论。

每个优化需要报告：

- 原始Latency。
- 优化Latency。
- Speedup。
- Kernel Launch变化。
- 显存变化。
- 数值最大误差。
- Backward误差。
- 适用K范围。
- 是否适用于Online Joint。

以实际端到端收益决定是否保留，而不是按照Kernel本身的加速倍数判断价值。

---

# 9. 优化顺序应由Profiling决定

建议优先级：

### P0：低风险快速优化

- 消除重复Tensor构造。
- 合并冗余广播。
- 消除CPU-GPU同步。
- 减少不必要的contiguous。
- 静态输入缓存。
- Ratio条件复用。
- 编译器自动融合。

### P1：高价值CUDA融合

- Fused Input Encoding。
- Fourier特征融合。
- Fused FiLM。
- Multi-head融合。
- 自定义Backward。

### P2：深入调度优化

- K专用Kernel。
- CUDA Graph。
- 更合适的内存布局。
- 更高效的梯度归约。
- 融合Optimizer。
- Forward/Backward跨模块融合。

优先执行实际Profiling指出的热点。

如果某个模块仅占总时间很小一部分，不要花费大量时间将其优化到极致。

例如，一个只占总时间5%的模块即使加速10倍，也无法让整个系统获得10倍性能。

研究的目标是降低实际训练Latency，而不是制造局部Kernel Benchmark的漂亮数字。

---

# 10. 优化验收与回退机制

每完成一个优化阶段，必须执行：

1. Unit Forward Parity。
2. Unit Backward Parity。
3. Integrated Compensation Parity。
4. Renderer Output Parity。
5. Training Step Parity。
6. 不同K和不同Ratio的边界测试。
7. Performance Benchmark。

重点覆盖：

- K=0。
- K=1。
- 小K。
- 大K。
- 不同Anchor成员数。
- 不同Anchor尺度。
- 极端但合法的Gaussian Scale。
- 合法Quaternion。
- 非零补偿输出。
- 不同训练Ratio与未见Ratio。

如果性能提升明显，但数值误差不能通过验证，则该优化不得进入推荐实现。

如果Kernel正确却没有实际性能收益，也不需要保留在主路径。

每一轮保留当前最快且已经通过正确性验证的版本。

不要把多个尚未验证的优化直接叠加。

---

# 11. 工程集成要求

不要直接覆盖原始SpatialLocalXYZ实现。

建议建立三个路径：

- `reference`：原始PyTorch实现。
- `optimized`：当前最快、通过验收的实现。
- `experimental`：仍在验证的Kernel或调度方案。

通过配置开关切换。

优先保证研究原型的正确性。

如果C++/CUDA Extension实现明显有价值，可以进一步设计其接入原项目的接口。

但不要未经验证直接替换生产Compensation路径。

最终需要说明：

- Fixed Master离线训练能加速多少。
- Online Joint理论上能复用哪些优化。
- 哪些CUDA Kernel支持Master梯度。
- 动态Gaussian增删是否需要重建缓存。
- 哪些地方有新的同步开销。
- 哪些优化需要额外维护成本。

不修改网络结构、不修改Loss、不修改训练调度、不修改生产默认算法。

---

# 12. 最终交付

## `SPATIAL_LOCALXYZ_CUDA_OPTIMIZATION_REPORT.md`

包含：

1. 原始性能Profile。
2. 各模块耗时。
3. 实际瓶颈。
4. 已实施的优化。
5. 每项优化的原理。
6. Kernel融合方案。
7. Forward/Backward分别的收益。
8. 不同K的性能。
9. 正确性与误差。
10. 最终端到端加速比。
11. 显存变化。
12. 在线部署限制。
13. 尚未解决的热点。
14. 下一阶段优化建议。

## `CUDA_PARITY_REPORT.md`

提供：

- 原实现与CUDA实现Forward逐张量比较。
- 各Head输出比较。
- 14D补偿比较。
- Gaussian属性比较。
- Renderer输出比较。
- 全参数梯度比较。
- 输入梯度比较。
- Optimizer Step比较。
- 相同训练状态下的多步检查。

必须提供具体数值。

不能只写PASS而不提供误差统计。

## `CUDA_PERFORMANCE_RESULTS.csv`

每个优化记录：

- Method。
- K。
- Forward ms。
- Backward ms。
- Forward+Backward ms。
- Optimizer Step ms。
- End-to-end ms。
- Speedup。
- Kernel Launch Count。
- Peak Memory。
- Numerical Error。
- Validation Status。

## `CUDA_OPTIMIZED_IMPLEMENTATION.md`

说明：

- 哪些模块改用了Triton。
- 哪些模块改用了CUDA Extension。
- 哪些模块仍保留PyTorch/cuBLAS。
- 为什么采用这些实现。
- Kernel的输入输出。
- Thread/Block分配。
- 内存布局。
- Backward计算方法。
- 动态K处理。
- Cache机制。
- Online Joint接入方法。

保留所有源码、构建命令、测试脚本和Benchmark入口。

---

# 13. 最终目标

**保持SpatialFiLM + Anchor单位位移的模型函数和梯度语义不变，在FP32正确性约束下最大化计算速度。**

尤其关注：

- 小K情况下的Kernel Launch开销。
- 55维输入编码的CUDA融合。
- 共享64维网络的高效执行。
- 四Head并行或融合。
- FiLM的融合计算。
- 自定义高效Backward。
- 静态特征缓存。
- 避免不必要的Tensor构造。
- 真实训练步骤的端到端加速。

不要为了速度牺牲PSNR、改变输出、修改梯度或减少训练。

不要只完成Profiling报告。

**你需要实际实现、验证、Benchmark并反复优化，直到主要热点得到合理处理，或者进一步优化的成本已经明显超过可获得的实际收益。**

现在开始检查源码、运行Profiler，并根据实际性能数据自主决定优化顺序。不需要逐阶段请求用户确认。