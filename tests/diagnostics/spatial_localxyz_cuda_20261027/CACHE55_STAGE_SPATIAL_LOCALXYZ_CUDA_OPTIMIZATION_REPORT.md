# SpatialLocalXYZ CUDA性能优化报告

## 1. 结论与提交边界

当前代码基线已提交为 `74fc1524d8d6345eda8e732a1da2efa9a4934951`，包含Neural V1源码、复现配置和SpatialLocalXYZ研究默认。用户随后确认：**仅性能研究默认，生产配置保持不变**。本轮优化代码保留在新目录，未另行stage/commit/push，未修改生产Selector或Compensation。

推荐冻结地图后端为 **graph_cached55_fused：缓存确定性的完整55维输入＋严格FP32 FiLM/输出融合＋完整神经补偿前后向Graph＋复用预热Stream**。输入需要梯度时自动回到实时Triton打包。未改变33,806个共享参数、初始化、55维字段、FiLM、14D变换、原Loss、原Adam或ratio轮转。

同阶段、同K、交错配对的七档等权平均：

| 地图 | 方法 | Forward ms | Backward-only ms | F+B ms | Renderer+Loss+Adam ms |
| --- | --- | --- | --- | --- | --- |
| fr1_xyz | graph_cached55_fused | 0.117662 | 0.34343 | 0.498347 | 2.94712 |
| fr1_xyz | reference | 0.708852 | 1.06267 | 1.95048 | 4.38467 |
| fr2_xyz | graph_cached55_fused | 0.12171 | 0.361179 | 0.509806 | 2.96126 |
| fr2_xyz | reference | 0.720896 | 1.05706 | 1.97332 | 4.4367 |

| 地图 | Forward倍数 | F+B倍数 | 完整冻结步骤倍数 |
| --- | --- | --- | --- |
| fr1_xyz | 6.02445 | 3.91389 | 1.48778 |
| fr2_xyz | 5.92308 | 3.87073 | 1.49825 |

这些是**冻结Master的选后补偿与真实Renderer步骤**，不含SLAM跟踪、Mapping调度、拓扑维护或Full Gaussian优化，不能称完整SLAM提速。

## 2. 真实路径审计

原型为`neural.py:SharedNeural`，55→64→64，log-ratio小网络产生两层FiLM，四个独立64→64→输出Head，XYZ乘Anchor参考尺度。四Head权重仍完全独立；没有私有可训练表。所有Linear及其Backward保留PyTorch/cuBLAS，没有重写GEMM、TF32或低精度。

重要边界：原`Features.__init__`在`no_grad`下，预计算并detach了19维物理特征、归一化世界坐标和Anchor尺度。原每次Forward实际执行的是**19维Gather＋36维Fourier＋MLP**，并非从Master重新构建协方差/人口统计。首个融合核保持这一真实范围，没有把冻结优化冒充Online Joint。没有新加detach去截断既有梯度。

本地生产默认仍是原E50路径；服务器源码/Schema较旧，与本地不同，两端均未改动。`gumbel_network.h`/`gaussian_mapper.cpp`的输入、参数与Checkpoint布局不同。本轮没有将研究网络塞进不兼容生产布局，也未声称已在服务器部署E50或SpatialLocalXYZ生产版本。

## 3. 环境与原始Profile

RTX3090，Torch 2.3.1+cu118、Triton 2.3.1；TF32关闭。没有可用Nsight/nvcc，使用现有环境Triton实际JIT编译、PyTorch Profiler、CUDA Events及CUDA Graph原生DOT。没有安装或修改系统包。正式性能队列顺序执行，未终止其他用户作业；未锁GPU时钟。

原始K68：Forward约0.713ms、F+B约2.004ms、真实步骤约4.306ms。CPU提交与CUDA区间接近，大量极小Kernel之间存在提交空隙；不能把区间时长全当GPU算力时间。原始独立Profile在K68记录4次迭代共432个Kernel，包含额外独立input_pack探针及Adam，不能当纯Forward数量。

更严格的首轮端到端Profiler在K68记录reference约250个GPU Kernel/step、graph_fused约236个；CUDA Graph是减少**主机调度**，不是把所有设备工作变成一个Kernel。`profile_summary_fr1_xyz.csv`保存GPU active时间、allocation调用与Memcpy事件。Profiler本身扰动明显，GPU active和硬件占用率不是一回事；没有伪造occupancy或DRAM带宽。

全打包Graph参与同进程trace时，本环境Profiler在Collection后停滞，原因未最终归结为上游缺陷。仅终止本任务挂起进程，原日志保留。最终延迟用Events；captured Kernel节点用原生DOT补充：

| CUDA原生转储 | K | 实际捕获Kernel节点 |
| --- | --- | --- |
| captured_K1025_0.dot | 1025 | 22 |
| captured_K1025_1.dot | 1025 | 58 |
| captured_K68_0.dot | 68 | 22 |
| captured_K68_1.dot | 68 | 56 |
| decoded_graph_captured_K1025_0.dot | 1025 | 52 |
| decoded_graph_captured_K1025_1.dot | 1025 | 135 |
| decoded_graph_captured_K68_0.dot | 68 | 52 |
| decoded_graph_captured_K68_1.dot | 68 | 133 |
| graph_cached55_fused_captured_K1025_0.dot | 1025 | 22 |
| graph_cached55_fused_captured_K1025_1.dot | 1025 | 58 |
| graph_cached55_fused_captured_K68_0.dot | 68 | 22 |
| graph_cached55_fused_captured_K68_1.dot | 68 | 56 |

DOT是神经图内部节点，不包括外部Renderer、Adam、输入copy和返回clone，不能与整步骤Kernel数量混为一谈。

## 4. 已实现优化及对照

1. Triton将19维Gather与36维Fourier直接写入连续K×55；使用`__nv_sinf/__nv_cosf`，禁用浮点乘加融合，不用`tl.sin`近似。
2. 两层FiLM的multiply/add/ReLU融合；Backward保留全部Gaussian对共享ratio参数的归约。使用PyTorch sum维持原归约形式。
3. 四Head输出拼接及XYZ乘Anchor尺度融合；其余属性解码和Quaternion处理仍走原Runtime。
4. 用Graph捕获整个补偿前后向，cuBLAS保持原样，原Adam在图外。Graph输入IDs/ratio有私有写入缓冲，输出clone避免后续replay污染已返回Tensor。每个Forward/Backward必须顺序消费。
5. 复用一个预热Stream，减少cuBLAS工作区；**不共享不同Graph的激活池**。Torch helper独立副本仅替换一行Stream构造，原三次warmup、capture/autograd逻辑不变，未改site-packages；来源和SHA在`shared_helper_provenance.json`。
6. 用户允许空间换时间后，增加Frozen-only的N×55静态输入缓存，fr1额外约1.435MiB、fr2约1.946MiB。缓存来自Gaussian显式属性和固定Fourier，**不是可学习个体特征**。物理输入/坐标需要梯度时禁用该缓存；PyTorch版本或storage变化触发重建，外部CUDA写入必须显式invalidate。

完整55维缓存相对已优化shared_warmup的额外收益主要在Forward，完整Renderer步骤收益较小，不能把原始→优化的总倍数全部归于静态缓存。最终matched阶段每个计时范围都从同一Checkpoint和Adam恢复，Adam-only探针不污染后续Renderer权重状态；此前阶段同样A/B匹配，但阶段内有连续更新，单独保留。

首轮单独Triton融合未提速：主机Autograd/launch增加抵消了Kernel减少。Graph解决主要提交瓶颈后才保留融合。Inductor同Core K68控制通过初步阈值，但冷编译约5.915秒，Core Forward 0.6031→0.6154ms，无优势；仅K68，不进入推荐路径。

## 5. 逐K性能与公平性

fr1为68/170/341/512/683/854/1025；fr2为实际92/231/463/695/927/1159/1391。所有方法相同权重、输入、选择身份和真实photo VJP，预热后交错测量。fr1三块、fr2两块，每阶段35次样本。CSV每行是块内median再跨块平均，保留P10/P90及CPU提交时间。

| ratio | K | 原Forward | 优化Forward | 原F+B | 优化F+B | 原完整步骤 | 优化完整步骤 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.01 | 68 | 0.706219 | 0.116736 | 1.93741 | 0.4864 | 4.39125 | 2.93035 |
| 0.025 | 170 | 0.692224 | 0.116395 | 1.91556 | 0.490155 | 4.30899 | 2.90167 |
| 0.05 | 341 | 0.703829 | 0.121173 | 1.94014 | 0.496981 | 4.36941 | 2.95424 |
| 0.075 | 512 | 0.70144 | 0.114688 | 1.9398 | 0.49152 | 4.35302 | 2.93103 |
| 0.1 | 683 | 0.725675 | 0.119808 | 1.98895 | 0.516096 | 4.45542 | 2.97097 |
| 0.125 | 854 | 0.729088 | 0.118784 | 1.98007 | 0.502443 | 4.43904 | 2.99895 |
| 0.15 | 1025 | 0.703488 | 0.116053 | 1.9514 | 0.504832 | 4.37555 | 2.94263 |

Backward-only另用同一真实photo VJP交错测量，将Forward置于事件区间外，数据见`cached55_paired_backward.csv`。没有用F+B减Forward冒充Backward；独立计时不可要求严格相加。Adam-only是隔离计时，不能把该微基准当实际训练曝光。

## 6. 正确性与真实训练

最终推荐后端两地图及边界共655项直接比较、跨ratio12步共1176项状态比较均通过原阈值。正常输入下网络输出/参数梯度在已测样本精确相同；完整误差含近零relative指标见`CUDA_PARITY_REPORT.md`。

三组从同一12k模型及完整Adam状态续训256个真实Bank photo步：原实现、原实现重复、优化。每ratio64次，使用原随机帧序的后续64轮；没有修改Loss、学习率、时钟或曝光。

| 方法 | 稳态每步ms | 首次建图ms | 含初始化总ms | 最终Adam时钟 |
| --- | --- | --- | --- | --- |
| reference | 4.44525 | 8.27345 | 1146.26 | 12256 |
| reference_repeat | 4.41165 | 3.15028 | 1132.53 | 12256 |
| optimized | 2.93577 | 1131.05 | 1882.6 | 12256 |

| 方法 | Bank留出PSNR | Bank留出photo |
| --- | --- | --- |
| optimized | 17.8742 | 0.135255 |
| reference | 17.8759 | 0.135244 |
| reference_repeat | 17.8696 | 0.135271 |

这是真实Frozen Renderer训练，不是在线SLAM。独立Renderer反传存在非确定性：原实现与自身重复也越过最初严格轨迹阈值。单独状态比较没有掩盖失败；用同一真实photo VJP隔离神经实现后，原Adam单步及跨ratio多步参数/Moments精确一致。不能用上述PSNR接近替代数值证明，也不能把一次轻微差值归因为系统性质量变化。

## 7. 初始化与显存成本

256步优化稳态约1.514倍，但首次4个训练shape建图约1.131秒；包含初始化后，这段短任务反而更慢。按当前平均步耗时估算约744步后才抵消额外初始化。这只是当前固定shape的估算，不是在线保证。

以下每种后端由独立进程测量，含网络、原Adam、七shape Graph及最后Backward缓冲；不含共同Fixture/地图。fr1为该进程冷测，fr2继承运行库工作区，是warm测，不能把差异当场景复杂度效应。

| 地图/冷暖 | 后端 | 常驻增量MiB | Graph数 | 权重bytes | Adam双Moments bytes |
| --- | --- | --- | --- | --- | --- |
| fr1_xyz | reference | 9.61523 | 0 | 135224 | 270448 |
| fr2_xyz | reference | 0.763184 | 0 | 135224 | 270448 |
| fr1_xyz | graph_all_fused | 131.963 | 7 | 135224 | 270448 |
| fr2_xyz | graph_all_fused | 115.914 | 7 | 135224 | 270448 |
| fr1_xyz | graph_shared_fused | 34.4629 | 7 | 135224 | 270448 |
| fr2_xyz | graph_shared_fused | 18.4136 | 7 | 135224 | 270448 |
| fr1_xyz | graph_cached55_fused | 36.8813 | 7 | 135224 | 270448 |
| fr2_xyz | graph_cached55_fused | 20.7578 | 7 | 135224 | 270448 |

共享预热Stream显著降低工作区开销，但Graph仍比eager占更多显存。参数保持135224 bytes、Adam双Moments270448 bytes；没有新增可学习参数。每次步骤峰值增量另在CSV，不是整张地图显存。

## 8. 失败尝试和回退

- 最初全打包Graph把调用方ratio缓存当可写静态缓冲，单次Forward通过、第五次跨ratio更新失败；已定位、隔离缓冲并通过完整重验，旧336项失败记录保留在`common_vjp_tensors.csv`，不用于推荐。
- Torch2.3.1的`num_warmup_iters=0`触发未初始化局部变量错误；没有修补安装包，改用可审计的一行Stream替换独立副本。
- 原始完整步骤Graph捕获在Adam处失败：`Attempting CUDA graph capture of step() for an instance of Adam but param_groups' capturable is False.`。未改capturable或任何Optimizer配置，Adam保留图外。该错误不能直接证明Renderer本身一定不可capture。
- 源码另有num_rendered和bucket_sum的Device-to-Host计数及动态buffer布局；没有用过期CPU计数、固定有效点数或截断工作区来换取整Renderer捕获。若继续GPU化这些计数，需要单独的Renderer等价工程。
- Inductor未形成收益，单独eager融合略慢，未推荐。没有继续手写SGEMM或合并成共享Head：剩余GEMM设备工作很小，而Renderer及主机路径已占主要时间。

## 9. Online Joint与交付状态

**IMPLEMENTED / TESTED：** Frozen Master七档、两地图、FP32输入/中间层/输出/参数及输入VJP、真实短续训、生命周期缓冲与Graph重建。

**PARTIALLY VERIFIED：** 输入leaf的梯度正确，但原原型没有raw Master→19维统计的活跃梯度路径；这不是生产Joint梯度接入证明。合成feature表append/prune/reorder验证不等于Gaussian optimizer与闭环集成。

**NOT IMPLEMENTED：** 生产C++/CUDA接入、真实SLAM、Full优化竞争、动态地图归一化策略、缓存自动逐步维护、ATE或15%以上ratio质量。动态K/N需要新图或安全边界invalidate，频繁重建会损害收益。参数storage替换必须invalidate；普通原地Adam更新不用。当前优化接口针对顺序训练及`no_grad`评价，不宣称支持任意并发Forward或`inference_mode`/整体requires_grad关闭的全部组合。

性能研究默认在`execution_config.json`，reference可随时切换；原`neural.py`与生产默认未覆盖。四份报告、逐张量/逐K原始数据、代码版本、构建运行命令及Checkpoint随包交付。
