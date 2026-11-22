# Residual48 CUDA 等价优化报告

本轮目录：`/home/prmi/zwj/experiments/residual48_cuda_optimization_20261101_1937`。源码：`/home/prmi/zwj/experiments/residual48_cuda_optimization_20261101_1937/source`。实测平台：RTX 3090、CUDA 11.8、PyTorch/LibTorch 对应环境；Python PyTorch `2.3.1+cu118`。开工时间以 `main_before.json` 为准：2026-11-01T11:39:57.053843+00:00；截至证据汇总约 160.5 分钟，包含独立构建、失败候选与复测。报告归档另计少量时间。

## 1. 结论与保留范围

**已实现、编译并验证的收益集中于冻结模型推理；本轮没有保留任何在线训练融合，也没有证明完整在线 Mapping 加速。** 严格 Feature 梯度门禁未通过的训练候选已从最终源码移除，不能用原版原子非确定性或相近 PSNR 将其放行。

最终保留三个独立、默认关闭的 C++ 开关：

- `CUDA.fused_residual48_input`：无梯度推理时，将六表活跃前缀的读取/原序累加、Offset 归一化/asinh/clamp 直接写入 K×51 输入。
- `CUDA.fused_residual48_heads`：无梯度推理时融合主干首层 Add/ReLU，以及四 Head 输出的 XYZ 尺度乘法/拼接。八次原始 Head Linear、权重和运算形状不变。
- `CUDA.parallel_residual48_heads`：无梯度、K≥8192 时四个独立 Head 使用四条 CUDA Stream；小 K 继续串行，Graph capture 期间继续串行。

训练 GradMode 开启时三者均回退原计算图。因此网络、Loss、Master/Selector/Maturity/Feature 梯度隔离、Feature 独立 Adam 时钟及严格选点逻辑均沿用原版。**推理收益不能标为训练收益。** Python 离线评价尚未接这些 C++ 开关；验证中的优化推理由本轮原生 C++ 驱动调用。

固定 room0 checkpoint 的两次交错配对测试，完整 Compensation 推理复测加速为约 **1.37×–1.65×**；全部训练/未见 Ratio 复测保持正收益。输入部分由 9–24 个 kernel 降至 1 个，完整补偿从 29–44 个降至 19 个。固定 office2 Full→LoD 前向 Replay 的训练Ratio时间降低约 **1.0%–2.1%**。

服务器主项目 `/home/prmi/zwj/mycartgs` 的 **328 个关键源码/配置/二进制/动态库 SHA-256 前后完全一致**。没有切换生产默认、没有 commit/push。独立构建和独立输出未写入主项目 bin/lib/results。只读第三方依赖复用主项目。最终实现另同步到本地工作区，生产配置未修改。

## 2. 算法与测量协议

保持 `spatial_offset3_residual48`：每 Anchor 六张独立 48D 参数表，1% 为 Base，其余五级按原始 ratio 区间逐级打开：

\[
z_a(r)=b_a+\sum_{j=1}^{5}w_j(r)d_{a,j},\quad
x_i^{offset}=\operatorname{clamp}\left(\frac{\operatorname{asinh}((xyz_i-c_a)/s_a)-\mu_{3:6}}{\sigma_{3:6}},-15,15\right).
\]

保留原 `ratio<=knot+1e-8` 非活跃判定与 double 权重到 FP32 scalar 的规则，活跃层按 j 递增执行独立乘法和加法，不合成 FMA，不树形归约。51D 输入加原 Ratio3 投影形成逻辑54D，64/64 主干、四 Head，25230个网络参数。原 XYZ Anchor 单位、物理 Scale 加法、相对四元数和 Other 输出/Renderer 解码不变。

固定组件使用已训练 room0：N=62555、Anchor=29387，实际选择 K 沿用原 fixture，未跨地图做 same-K；不是重新训练模型。全部计算 FP32、TF32关闭。空/动态 K、未见 Ratio、节点附近 ±5e-9、极端值和拓扑均另测。

CUDA Event 以同一输入/权重交错轮转比较，首轮70次、独立复测70次，每条件8次预热；报告以复测中位数为主，保留首轮。Event 含 Host 提交间隙；Nsight 提供纯 kernel 时间。Nsight 按 NVTX Host 发射与 CUPTI correlation 归属，14次 scope/条件含8次预热和6次采集。多 Stream 的 kernel 时间总和可重叠，**不能当作延迟**；另报首个 kernel 开始到最后完成的 GPU span 与事件时间。

GPU 上同时存在他人 PID600972、约6068 MiB 显存进程，本轮没有操作它。配对减少负载差异，仍不是独占GPU基准。原始记录位于 `parallel_events*.csv`、`parallel_nodes.nsys-rep`、`analysis/parallel_nodes/`。

## 3. P0：Residual 与 Offset 输入融合

第一版保留各表 `index_select`/原 Scatter 反向，只融合残差累加和 Offset；局部前向/VJP的97组位测试通过，但完整共享表梯度未通过。故最终版本只保留 **no-grad 直接 Gather 前向**，训练实现和自定义 VJP均从编译目标移除，失败源码/日志保留在 `rejected/`、`attempt_grouped_relu/`及 attempt 日志中。

最终 kernel 用每Gaussian 64个线程槽：48个特征分量与3个 Offset 分量直接写入 K×51，256线程处理4行。按活跃表数实例化1–6级 kernel，固定展开内部原序乘加；显式 RN 操作，目标编译保留 `--fmad=false`。相比首版按51列除模/动态循环的实现，40%输入 Event 从约195.6µs进一步降到约40µs。

每次从当前 Gaussian→Anchor owner、中心、参考尺度和六表读取，不缓存拓扑/参数数值。中心/参考尺度 Gather 仍在 kernel 外；本轮没有声称已融合所有 Mapper Gather。每次只分配最终输入，不创建逐层 K×48 中间量。

| Ratio | 实际K | 输入kernel 原→新 | 纯kernel µs 原→新 | Event µs 原→新 | 输入Tensor分配 原→新 |
| --- | --- | --- | --- | --- | --- |
| 1% | 627 | 9→1 | 25.65→2.98 | 64.51→8.19 | 9→1 |
| 5% | 3129 | 12→1 | 37.85→4.44 | 88.06→9.22 | 12→1 |
| 10% | 6257 | 15→1 | 57.28→7.81 | 106.50→11.26 | 15→1 |
| 20% | 12512 | 18→1 | 103.37→16.09 | 130.05→21.50 | 18→1 |
| 30% | 18768 | 21→1 | 195.77→27.05 | 176.13→29.70 | 21→1 |
| 40% | 25023 | 24→1 | 290.57→37.44 | 284.67→40.96 | 24→1 |

分配数量来自 CUDACachingAllocator 的 allocation 事件，包含复用池中的 Tensor 分配；不等于每次执行对应次数的 `cudaMalloc`。51D 输入在40%时约4.87MiB，最终实现没有训练中间缓存。

## 4. P1：四 Head 与完整补偿推理

保留四 Head 原 Linear，先后顺序在串行路径完全不变。Add/ReLU 对 `-0` 使用与 ATen一致的 `+0`，NaN保留；输出kernel单次完成四头拼接和 XYZ×参考尺度。参考尺度非连续或不符合原生产K×1布局时回退原路径，另测strided参考尺度。

并行路径保留每个 Head 原始 GEMM形状和参数，将四个完整分支分发到四条池化Stream；用一个 ready Event和四个完成Event建立输入→分支→消费依赖。输入、参数及输出均使用 `recordStream`，避免缓存分配器提前复用。线程局部状态只保留4条Stream与5个Event，切设备替换，**不缓存Feature或模型数值**。K≥8192取自本轮探针的拐点；8191/8192/8193/10000/16384另做位门禁。

| Ratio | K | 完整kernel 原→新 | 原Event µs | 串行融合 µs | 融合+大K并行 µs | 复测加速 | 首轮加速 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1% | 627 | 29→19 | 328.70 | 238.59 | 240.64 | 1.37× | 1.36× |
| 5% | 3129 | 32→19 | 363.52 | 249.86 | 251.39 | 1.45× | 1.43× |
| 10% | 6257 | 35→19 | 416.77 | 287.74 | 303.62 | 1.37× | 1.41× |
| 20% | 12512 | 38→19 | 665.60 | 516.10 | 402.43 | 1.65× | 1.65× |
| 30% | 18768 | 41→19 | 915.46 | 743.42 | 609.79 | 1.50× | 1.50× |
| 40% | 25023 | 44→19 | 1267.71 | 991.23 | 864.26 | 1.47× | 1.47× |

| 未见Ratio | K | 原Event µs | 新Event µs | 复测加速 |
| --- | --- | --- | --- | --- |
| 2.5% | 1564 | 371.71 | 257.02 | 1.45× |
| 7.5% | 4692 | 379.90 | 252.93 | 1.50× |
| 15% | 9384 | 547.33 | 340.99 | 1.61× |
| 25% | 15639 | 797.70 | 505.86 | 1.58× |
| 35% | 21895 | 1113.09 | 762.88 | 1.46× |

串行融合在1%/40%的Nsight纯kernel时间为 114.37→85.47µs、1314.68→1028.78µs。并行40%的kernel时间**总和**是1376.39µs，因资源竞争比串行融合更高；GPU span却由1365.24降到862.55µs。收益来自重叠和调度，不是宣称每个GEMM都变快。

## 5. 严格正确性与失败结果

| 验证 | 结果 | 证据与边界 |
| --- | --- | --- |
| 输入局部前向/随机VJP | 97组逐bit通过 | p0_local_gate.log；不包含共享表Scatter |
| 最终输入/完整网络/生命周期 | 766项逐bit通过 | release_inference_gate.log；含全部7种开关组合、空K、极端值、ratio边界、阈值K和strided参考尺度回退 |
| 动态状态 | 通过 | 新增Anchor、Clone、Split替换父点、Prune/重排、Anchor迁移、统一Scale/平移几何修正、checkpoint恢复 |
| 四Head多Stream探针 | 27组前向逐bit通过 | head_stream_probe.log；含Inf/NaN；小K未采用并行 |
| 训练回退随机VJP/Adam | 18步逐bit通过 | release_training_fallback_gate.log；仅唯一owner受控fixture；全部参数grad/undefined、XYZ多路径、参数/一二阶矩/时钟比较 |
| 固定office2完整前向 | 通过 | 6视图×11ratio，逐bit比较Delta/图像/Loss；8次配对+2次预热；Full不变 |
| 真实Renderer反向Replay | 不满足完整逐bit | 原版重复和开关开启但训练回退均有差异；不是训练融合放行证据 |
| 独立在线训练 | 完成8次短程回归 | 每次400帧/4000 Mapping，所有LoD非有限计数0；不是完整长程等价或加速证明 |

已明确拒绝的候选：

- **训练输入融合**：合并计算图后真实room0步骤1（K=3129）Feature梯度出现最大绝对差1.90735e-6、2192元素不同；随机共享表测试K=627出现9.53674e-7。Gather输入VJP逐bit一致，差异在原子Scatter或共享梯度归约阶段。原版重复同样有差异，但本轮没有以此接受候选；最终源码删除此训练路径。
- **Batched/宽首层 GEMM**：K=1等形状前向不一致；大K部分前向一致，但hidden与参数梯度均存在位差，拒绝。保留 `head_candidates.csv` 全部27条结果，不按形状挑选训练通过点。
- **四ReLU一次融合并重排Head执行**：位门禁通过后，40%完整推理Event约1.25ms→1.67ms，实际变慢，拒绝。最终恢复原分支内部顺序，只保留输出拼接融合和单独验证的多Stream。
- **首版ReLU**：`-0`符号不同，已修正后重新门禁，没有放宽比较。

真实Renderer反向诊断从固定office2 checkpoint、相机、选点和Loss产生同样前向图像；未更新参数，比较一次原版、原版重复、开关开启但训练回退的梯度：

| Ratio | 原版重复不同Tensor数 | 原版重复max abs | 回退重复不同Tensor数 | 回退重复max abs |
| --- | --- | --- | --- | --- |
| 1% | 28 | 0.000000321 | 28 | 0.000000626 |
| 5% | 29 | 0.000000641 | 29 | 0.000001081 |
| 10% | 30 | 0.000001294 | 30 | 0.000003185 |
| 20% | 31 | 0.000001103 | 31 | 0.000001326 |
| 30% | 32 | 0.000003040 | 32 | 0.000003189 |
| 40% | 33 | 0.000009283 | 33 | 0.000004977 |

这不是通过的 Mapping Adam Replay。18步受控VJP证明的是**回退到原计算图**的参数/Adam一致；本轮未证明包含真实Rasterizer、共享Scatter、Selector双VJP和完整Master Adam的端到端逐bit训练序列。非确定性根源与本轮CUDA前向差异必须分开，不能用PSNR接近替代门禁。

## 6. P2：空间换时间、缓存与Graph

- 最终P0没有数值缓存，拓扑/Feature Adam后无需维护过期值；每次使用最新Tensor和owner。中间激活由原LibTorch缓存分配器管理，未引入无界Workspace。
- 大K四Stream首次建立cuBLAS工作区时，实测额外常驻 **34,078,720字节=32.5MiB**；后续Ratio未再增长。该值是PyTorch分配器观测，不含驱动侧Event/Stream管理内存。四个固定Stream保持有界。
- 仅研究驱动中为**不可变checkpoint快照**预计算六个Anchor前缀，实测额外33,856,512字节（约32.29MiB），构建约4.7ms。原序前缀加未见ratio下一层残差可逐bit通过，但额外完整补偿收益很小，未接入在线/最终Mapper。训练Adam或拓扑变化后不可复用。
- 单个固定model/ratio/exact-K CUDA Graph前向已实测逐bit输出和RNG状态不变；额外观测约8.2–10.4MiB。低K显著减少Host提交间隙，大K收益不稳定；没有接入在线Mapper，也没有跨拓扑、跨Adam、跨动态K复用。Graph capture自动走串行Head。
- Graph node级Nsight显示**21个真实kernel**，包含两个PyTorch Graph RNG状态管理Fill；普通Graph trace看到少量kernel不能误称剩两个。`--cuda-graph-trace=node`原始trace和修正汇总均保留。
- compact K、Renderer动态容量、finite检查的必要Host依赖未被推迟。训练优化未通过时，不尝试通过改变检查时机、Adam顺序或跳过动量换速度。本轮未删除这些同步点。

所有Ratio的临时分配、峰值增量及Host提交见 `PER_RATIO_MEMORY_HOST.csv`；固定模型前缀/Graph仅是独立研究，不是默认启用的安全在线缓存。

## 7. 固定office2渲染与完整前向收益

使用本轮OFF在线模型N=14846，按该地图N生成各档实际K。4个non-Bank视图和2个Bank视图在看指标前锁定，完整像素/Loss逐bit对比。不是跨地图same-K。该短程模型最高K=5939，低于8192并行阈值，实际验证的是输入/输出融合及并行开关的小K回退；大K多Stream完整网络输出由room0的766项门禁覆盖，未冒充office2的大K整图Replay。以下Event包含一次Full渲染/Loss和一次LoD渲染/Loss、补偿/解码，不含反向、Selector、Adam或拓扑，**不是完整Mapping**。

| Ratio | Full→LoD 原ms | 优化ms | 时间降低 |
| --- | --- | --- | --- |
| 1% | 2.9947 | 2.9618 | 1.10% |
| 5% | 3.1070 | 3.0691 | 1.22% |
| 10% | 3.0617 | 3.0304 | 1.02% |
| 20% | 3.1546 | 3.0968 | 1.83% |
| 30% | 3.2551 | 3.1976 | 1.77% |
| 40% | 3.3414 | 3.2696 | 2.15% |

| 固定non-Bank Ratio | K | PSNR | SSIM | LPIPS | 优化前后变化 |
| --- | --- | --- | --- | --- | --- |
| 1% | 150 | 15.6955 | 0.702742 | 0.603818 | 三项均0；像素逐bit一致 |
| 5% | 744 | 20.3745 | 0.754516 | 0.479922 | 三项均0；像素逐bit一致 |
| 10% | 1486 | 22.0139 | 0.790800 | 0.415007 | 三项均0；像素逐bit一致 |
| 20% | 2970 | 23.9731 | 0.825345 | 0.362308 | 三项均0；像素逐bit一致 |
| 30% | 4455 | 25.1907 | 0.840001 | 0.331941 | 三项均0；像素逐bit一致 |
| 40% | 5939 | 26.0801 | 0.847807 | 0.313811 | 三项均0；像素逐bit一致 |
| 100% | 14846 | 26.4626 | 0.881109 | 0.248169 | 三项均0；像素逐bit一致 |

此表为400帧/4000 Mapping短程模型的描述值，不能作为完整office2最终质量。SSIM为生产原生训练SSIM；LPIPS为AlexNet、输入clamp到[0,1]。原始逐视图与未见Ratio见 `render_quality_metrics.csv`。独立在线模型质量另用8个共同non-Bank视图、每图各自N预算评价，见 `online_quality_summary.csv` 和 `evaluation/`；其训练调度不同，不用于证明CUDA质量增益。

## 8. 真实在线回归与曝光

生产参数不变，研究配置仅固定前400帧/4000 Mapping以观测在线与tail、1000 Full预热、1200×680在线LoD、Full→单档轮转tail。没有给生产配置加4000上限。首轮/中间候选/最终候选所有运行均保留，不选择最快的运行代替对照。

| 运行 | Nsight | 最终N | online/tail Mapping | Full/LoD渲染总次数 | Master Adam更新 | Mapping s | 进程墙钟s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OFF | False | 14846 | 2693/1307 | 2548/2759 | 5307 | 33.096 | 43.898 |
| ON | False | 17319 | 2708/1292 | 2536/2756 | 5292 | 33.472 | 44.111 |
| NSYS_ON | True | 15246 | 2171/1829 | 2996/2833 | 5829 | 45.527 | 87.993 |
| ON_FINAL | False | 17716 | 2674/1326 | 2565/2761 | 5326 | 33.447 | 44.338 |
| NSYS_FINAL | True | 16329 | 2169/1831 | 2998/2833 | 5831 | 46.422 | 88.554 |
| ON_PARALLEL | False | 16788 | 2670/1330 | 2568/2762 | 5330 | 33.955 | 44.657 |
| NSYS_PARALLEL | True | 15985 | 2216/1784 | 2957/2827 | 5784 | 45.790 | 88.059 |
| ON_RELEASE | False | 15954 | 2699/1301 | 2543/2758 | 5301 | 33.115 | 43.891 |

这些运行均未启用新的训练kernel，N、online/tail比例、Full/LoD实际曝光随异步Tracking/Mapping时序变化。因而约33–34秒的未profile Mapping差异不代表训练提速/回退，也不能将Nsight约46秒直接和未profile比较。最终OFF与ON_PARALLEL质量、ATE及每ratio实际K/曝光已另存；不把不同N的LoD提升全部归因于Fusion。

各表Feature Adam时钟、6ratio曝光、FullBackward、Selector提交、拓扑次数、峰值显存都在 `ONLINE_BUDGETS.csv`；完整stage state保留在各 `runs/*/result.json`。无新增网络参数，无改SGI/Maturity、Scale数学或训练计划。

独立在线模型在8个共同non-Bank视图上的描述性质量如下。采用原有Python评价与TorchMetrics SSIM/LPIPS，和上节原生训练SSIM的统计口径不同。选点数按各图N计算，**没有将K拉齐**；表中差异不能视作推理kernel改变训练质量。

| 运行 | Ratio | 实际K | PSNR | SSIM | LPIPS |
| --- | --- | --- | --- | --- | --- |
| OFF | 100% | 14846 | 25.8515 | 0.880173 | 0.250611 |
| OFF | 1% | 150 | 15.5163 | 0.720681 | 0.597080 |
| OFF | 5% | 744 | 19.7510 | 0.765601 | 0.478866 |
| OFF | 10% | 1486 | 21.3528 | 0.796071 | 0.415200 |
| OFF | 40% | 5939 | 25.6524 | 0.850267 | 0.313204 |
| ON_PARALLEL | 100% | 16788 | 24.5068 | 0.881137 | 0.250033 |
| ON_PARALLEL | 1% | 169 | 14.9095 | 0.719805 | 0.594013 |
| ON_PARALLEL | 5% | 841 | 18.8767 | 0.772118 | 0.460727 |
| ON_PARALLEL | 10% | 1680 | 20.8834 | 0.801619 | 0.409170 |
| ON_PARALLEL | 40% | 6716 | 24.9006 | 0.851254 | 0.318370 |

ATE、共同帧身份、每视图p10/p90、几何长尾与失败列表均保存在 `evaluation/`；两模型评价的失败数均为0。

## 9. P3：最新系统热点与同步

| 模块 | kernel次数 | 累计GPU ms | 占比 |
| --- | --- | --- | --- |
| SSIM backward | 5784 | 5545.55 | 23.33% |
| ATen other | 933513 | 3842.79 | 16.16% |
| Rasterizer backward | 5784 | 3582.03 | 15.07% |
| SSIM forward | 5868 | 3158.41 | 13.28% |
| Gaussian to tile | 5868 | 2478.19 | 10.42% |
| GEMM | 154828 | 1350.74 | 5.68% |
| Rasterizer forward | 5868 | 1102.64 | 4.64% |
| CUB sort/scan | 121855 | 964.25 | 4.06% |
| Other | 83348 | 541.67 | 2.28% |
| L1 fusion | 11568 | 418.32 | 1.76% |
| Adam | 8611 | 345.27 | 1.45% |
| Gather/scatter | 56178 | 218.44 | 0.92% |
| Rasterizer preprocess | 11652 | 150.04 | 0.63% |
| Maturity | 11438 | 76.61 | 0.32% |

最新在线Trace总kernel为1,422,163。训练没有调用新前向kernel，因此小算子和渲染热点仍在。SSIM前后向约37%，ATen约16%，Rasterizer反向约15%，Gaussian→Tile约10%；本轮没有盲目改写这些已明确需要独立正确性门禁的模块。

在线Nsight来自并行版、最后的非连续参考尺度元数据保护之前；CUDA kernel源码相同、训练回退逻辑相同。增加该保护后重新构建、重跑全部门禁/组件Nsight/渲染Replay，并另跑ON_RELEASE在线回归。保留各构建日志与测量记录，不把多个二进制身份合并为一次运行。

| Host CUDA API | 次数 | 累计Host ms |
| --- | --- | --- |
| cudaLaunchKernel_v7000 | 1422163 | 7521.01 |
| cudaStreamSynchronize_v3020 | 36020 | 5941.46 |
| cudaMemcpy_v3020 | 17604 | 4029.22 |
| cudaDeviceSynchronize_v3020 | 186 | 21.06 |

Host API累计时间可能含等待/重叠，不能相加为可节省墙钟。≤32字节DtoH与详细API见 `analysis/online_parallel/copies.csv`、`cuda_api.csv`。Nsight Compute本轮再次实际调用，返回码255、`ERR_NVGPUCTRPERM`；未得到SM占用率、DRAM带宽或stall硬件计数，不虚构这些数据。Nsight Systems、NVTX和CUDA Event可用。

## 10. 源码、开关与回退

本地基准Git HEAD为`7dd333dc313863cef839dcbf45068ab713bcd05b`加已有未提交生产改动，**不能把HEAD当作完整源码版本**。版本以 `source_scope.json`、`accepted_source_manifest.json`、`accepted_binary_manifest.json`为准。

改动文件：CMake注册两个新源文件；`include/residual48_fusion.h` API；`src/residual48_fusion.cu`三个前向kernel；`src/residual48_streams.cpp`有界Stream调度；AnchorBank仅增加与原查询同规则的权重计算；SpatialCompensation仅增加可选已打包输入和推理开关；Mapper读取/导出三个默认0开关。没有修改cfg生产默认、网络参数注册或checkpoint张量schema。

研究时可在独立配置附加：

```yaml
CUDA.fused_residual48_input: 1
CUDA.fused_residual48_heads: 1
CUDA.parallel_residual48_heads: 1
```

三项全0即回到原路径；即使设1，GradMode开启的训练仍走原版。仅本轮 `replica_rgbd`入口与原生驱动完成重编译/加载身份验证，未重编译其他SLAM入口；不可把旧ABI入口与新库混用。服务器主项目没有这些改动，生产二进制保持原样。

复现依次使用本目录 `common.py`中的独立环境、CMake编译 `replica_rgbd`、`inference_gate.cpp`、`training_fallback_gate.cpp`、`inference_benchmark.cpp`、`render_replay.cpp`、`analyze.py`。逐次原始命令/返回码/耗时保存在对应JSON。所有失败attempt保留，研究checkpoint在各runs目录，无覆盖历史模型。

## 11. 下一阶段建议

1. **训练提速首先解决验证控制变量**：把原子Scatter/Rasterizer反向的确定性问题与候选的数学等价拆开。当前严格标准下无法凭原版非确定性接受新训练融合；如研究确定性排序归约，必须作为单独行为候选评审，不能悄悄改原累加顺序。本轮不给它生产准入。
2. **优先可独立验证的前向/中间链**：本轮已证明P0输入值与局部VJP一致，完整Feature Scatter位门禁才是阻塞。下一实验应固定原Gather/Scatter调用和所有分支梯度路由的完整Replay，逐节点定位首个差异；不继续盲目调整kernel块大小。
3. **部署推理收益前补生产消费路径**：当前Python评价没有调用新C++接口，需要单独接原生推理入口、固定checkpoint多场景像素门禁和真实渲染墙钟；不能把组件1.4×当作SLAM1.4×。
4. **同步优化先画精确依赖链**：compact count→分配/Renderer、tile scan末端计数→sort、finite→Adam提交；用调用点NVTX定位Host等待。只融合无依赖读取，不延后错误检查。现有截图或全局API总数不足以证明某个同步可删。
5. **系统级收益仍受大热点限制**：SSIM约37%、Rasterizer反向约15%、Tile约10%。若以后进入P3，单独验证缓存/并行候选；不为了完成清单改低占比Maturity行更新。多Stream推理已减少关键路径但增加kernel累计工作，继续验证共享GPU稳定性与目标硬件拐点。

## 12. 尚未达成的目标

- 没有通过严格共享Feature梯度/真实Mapping Adam门禁的**训练新优化**，因此未实现线上训练的5%或其他可证明加速。
- 在线回归为400帧/4000 Mapping短程，不是完整2000帧/Vanilla对齐长程质量证明。
- Loop/Scale生命周期门禁覆盖统一Scale/平移与Anchor迁移，不覆盖所有真实Loop Closure轨迹及任意并发几何事件。
- 固定模型Graph/前缀快照只在研究驱动使用，未完成在线参数更新失效管理；最终生产路径不使用它们。
- 没有硬件性能计数器权限；没有生产切换、服务器更新或自动提交。

**建议保留三个默认关闭的推理优化实现及全部原始证据；在线训练继续使用当前生产默认。**
