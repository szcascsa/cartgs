# SpatialLocalXYZ CUDA性能优化报告

## 1. 结论与版本边界

基线已提交：`74fc1524d8d6345eda8e732a1da2efa9a4934951`（anchor-si）。SpatialLocalXYZ只作为**性能研究默认**；生产配置保持原样。本轮新增优化不stage、不另行commit、不push。

最终推荐冻结地图后端 **decoded_graph**：只读55D输入缓存＋严格FP32 Triton FiLM/输出融合＋原cuBLAS网络与原生Gaussian解码共同CUDA Graph。33,806个共享参数、55D字段、Fourier、Anchor单位XYZ、Quaternion、Loss、Adam、训练顺序和ratio均不变，没有新增Gaussian/Anchor私有训练表。

同K、同Checkpoint、同Adam、交错测量的七档等权结果：

| 地图 | 原NN+解码 Forward ms | 优化Forward ms | Forward倍数 | 原F+B ms | 优化F+B ms | 原完整步ms | 优化完整步ms | 完整步倍数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fr1_xyz | 1.15371 | 0.17564 | 6.56857 | 3.37964 | 0.681984 | 4.4289 | 1.66361 | 2.66222 |
| fr2_xyz | 1.1479 | 0.173495 | 6.61636 | 3.40136 | 0.679936 | 4.42931 | 1.65654 | 2.67383 |

以上是Frozen Master真实Renderer+Loss+原Adam步骤，**不是完整在线SLAM加速**，也不是重建质量提升实验。独立Backward计时见下表；没有用F+B减Forward代替Backward。

| 地图 | 后端 | NN+解码Forward | Backward-only | NN+解码F+B | 加Adam | 加Renderer+Loss |
| --- | --- | --- | --- | --- | --- | --- |
| fr1_xyz | decoded_graph | 0.17564 | 0.478891 | 0.681984 | 0.821882 | 1.66361 |
| fr1_xyz | graph_cached55_fused | 0.539843 | 1.25152 | 1.97647 | 2.13221 | 2.96053 |
| fr1_xyz | reference | 1.15371 | 1.92878 | 3.37964 | 3.53919 | 4.4289 |
| fr2_xyz | decoded_graph | 0.173495 | 0.495177 | 0.679936 | 0.817883 | 1.65654 |
| fr2_xyz | graph_cached55_fused | 0.545134 | 1.2966 | 1.98012 | 2.13321 | 2.97618 |
| fr2_xyz | reference | 1.1479 | 1.95606 | 3.40136 | 3.55511 | 4.42931 |

## 2. 审计与原始Profile

原Features在`no_grad`下计算19维物理输入及世界坐标，训练时Gather和Fourier才实时执行。19维为世界XYZ3、asinh局部Offset3、logScale3、SH DC3、Opacity logit1、归一化协方差形状6，经原均值/标准差和clamp。另有六频带36维sin/cos，合计55维。主干55→64→64；ratio log归一化后1→32→256，两层FiLM；四Head64→64→3/3/4/4。XYZ输出乘原Anchor参考尺度。

fr1 N=6838、Anchor=2066；fr2 N=9275、Anchor=2724。原选择身份、Cached uniform和same-K排序未修改。源Checkpoint完整路径及SHA见`model_manifests.json`；网络、原Renderer与生产源码SHA分别在environment、production_source_verification等JSON。

RTX3090、Torch2.3.1+cu118、Triton2.3.1、FP32、TF32关闭。无nvcc/nsys/ncu，使用实际Triton JIT、PyTorch Profiler、CUDA Events及CUDA Graph DOT；没有修改系统安装。测试队列顺序运行，正式测量时未发现其他GPU计算作业，未锁时钟。使用用户现有conda环境，无隔离构建。

Profile发现小K的大量极小Kernel、Tensor构造与CPU提交间隙。K68原始整步骤Profiler约250个GPU Kernel，graph_fused约236个；不是GEMM计算量主导。基线独立探针含额外input_pack，432 kernels/4次不能当纯网络Kernel数。`profile_summary_*`和`operators_*`保留CPU/GPU active、分配事件；active时间不等于硬件occupancy。

最终打包Graph的CUPTI Collection后处理在本环境停滞；仅停止本任务挂起进程，保留日志。最终延迟来自CUDA Events。DOT报告如下：

| 原生DOT | K | 捕获Kernel节点 |
| --- | --- | --- |
| decoded_graph_captured_K1025_0.dot | 1025 | 52 |
| decoded_graph_captured_K1025_1.dot | 1025 | 135 |
| decoded_graph_captured_K68_0.dot | 68 | 52 |
| decoded_graph_captured_K68_1.dot | 68 | 133 |

这些是**图内**前向/反向节点，不包括Renderer、Adam、copy和clone。Graph主要减少主机调度，不能声称将所有GPU工作融合成一个Kernel。

## 3. 有效优化及空间换时间

1. Pack Triton直接写K×55，合并Gather、六频带Fourier与cat；使用libdevice `__nv_sinf/__nv_cosf`，不用近似`tl.sin`，所有内核关闭FP融合。
2. 两层FiLM multiply/add/ReLU融合；Backward保留dH和全部K行对共享ratio参数的sum。四个独立Head权重与cuBLAS矩阵运算保持原样。
3. 四Head拼接及XYZ×AnchorScale融合；尺度梯度保留。
4. 缓存只读确定性N×55输入。fr1约1.435MiB、fr2约1.946MiB；不是可训练身份表。输入需梯度时使用实时Pack，不能detach活跃路径。
5. 每种K/输入storage/梯度模式保留独立Graph及激活池。只共享预热Stream以减少cuBLAS工作区，**不共享不同Graph的激活池**。Torch helper独立副本仅改一行Stream创建，原三次warmup和Autograd不变，安装包未改。
6. 将原生Gaussian解码一起capture：XYZ加法、Scale×exp、Quaternion两次normalize与multiply、Opacity logit+sigmoid、DC加法，逐值与原Runtime一致。减少这些逐元素算子的Python/Autograd调度，Renderer和原Adam仍在图外。

仅缓存55D的Compensation-only后端仍推荐`graph_cached55_fused`。该接口的Forward约0.118/0.122ms、F+B约0.498/0.510ms；它不含Gaussian属性解码，不能直接与上表更大范围的0.176/0.173ms比较。旧阶段报告保存在`CACHE55_STAGE_*`。

## 4. 逐K与测量公平性

fr1三个交错块、fr2两个，每阶段35次＋预热。每一阶段恢复同一Checkpoint和完整Adam状态，Adam-only探针不污染后续Renderer状态；每块改变方法顺序。每行保留块内median/P10/P90，汇总为块间mean。Frozen NN+解码、Compensation-only分不同phase/scope，CSV不混算。

### fr1_xyz

| ratio | K | 原Forward | 优化Forward | 原Backward | 优化Backward | 原完整步 | 优化完整步 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.01 | 68 | 1.13562 | 0.174763 | 1.89372 | 0.440661 | 4.3479 | 1.66537 |
| 0.025 | 170 | 1.14278 | 0.176128 | 1.91829 | 0.462507 | 4.44279 | 1.67287 |
| 0.05 | 341 | 1.14825 | 0.175787 | 1.94219 | 0.462165 | 4.42129 | 1.66775 |
| 0.075 | 512 | 1.17146 | 0.175104 | 1.91488 | 0.474453 | 4.46566 | 1.66741 |
| 0.1 | 683 | 1.17999 | 0.174763 | 1.93365 | 0.490496 | 4.44348 | 1.64898 |
| 0.125 | 854 | 1.14961 | 0.176128 | 1.93946 | 0.500395 | 4.42709 | 1.66093 |
| 0.15 | 1025 | 1.14825 | 0.176811 | 1.95925 | 0.521557 | 4.45406 | 1.66195 |

### fr2_xyz

| ratio | K | 原Forward | 优化Forward | 原Backward | 优化Backward | 原完整步 | 优化完整步 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.01 | 92 | 1.11718 | 0.17152 | 1.96813 | 0.455168 | 4.42931 | 1.6599 |
| 0.025 | 231 | 1.13203 | 0.174592 | 1.95891 | 0.479232 | 4.40678 | 1.66502 |
| 0.05 | 463 | 1.13459 | 0.172544 | 1.97632 | 0.497664 | 4.43136 | 1.64813 |
| 0.075 | 695 | 1.15866 | 0.173056 | 1.95789 | 0.494592 | 4.42163 | 1.66502 |
| 0.1 | 927 | 1.14842 | 0.176128 | 1.96198 | 0.514048 | 4.41242 | 1.65325 |
| 0.125 | 1159 | 1.15405 | 0.173056 | 1.93229 | 0.516096 | 4.43238 | 1.63174 |
| 0.15 | 1391 | 1.1904 | 0.173568 | 1.9369 | 0.50944 | 4.4713 | 1.6727 |

## 5. 数值、梯度与真实续训

解码Graph两地图共842项直接/输入VJP/边界/生命周期比较，全部通过原阈值；两地图12步跨ratio共2304项参数、step、exp_avg、exp_avg_sq比较，最大绝对误差 **0**。另有Compensation-only 655项及1176项状态检查、配置默认入口14项。所有逐张量误差保留，无放宽阈值。

12步状态比较使用**同一真实Renderer photo的解码属性VJP**隔离网络实现与Renderer非确定性，既检查NN又检查解码Backward。输入VJP覆盖19D/coords/Anchor尺度以及XYZ、Scale、Rotation、DC、Opacity原属性leaf；既有缓存Graph梯度路径，也有输入需梯度时的实时fallback。无梯度的高阶SH不伪造活跃依赖。

原独立Renderer反传与自身重复也会越过初始严格轨迹阈值，失败没有被改为PASS；详见数值报告。因此完整真实256步用于确认可训练、时钟与轨迹波动，不能要求长程bitwise相同。

三组均从原12000步模型及完整Adam恢复，原Bank后续64轮×4ratio=256真实photo步，lr=.001、eps=1e-15、fused=True、capturable不变。每个参数Adam时钟都到12256，Master未更新。

| 方法 | 稳态ms/step | 首次4shape初始化ms | 含初始化总ms | Adam时钟 |
| --- | --- | --- | --- | --- |
| reference | 4.41596 | 7.97085 | 1138.46 | 12256 |
| reference_repeat | 4.35363 | 3.31493 | 1117.85 | 12256 |
| optimized | 1.68184 | 1161.25 | 1591.8 | 12256 |

| 方法 | ratio | Bank留出PSNR | photo |
| --- | --- | --- | --- |
| optimized | 0.01 | 16.4566 | 0.157644 |
| optimized | 0.025 | 15.2343 | 0.16877 |
| optimized | 0.05 | 18.2966 | 0.129844 |
| optimized | 0.075 | 18.5637 | 0.126185 |
| optimized | 0.1 | 18.7923 | 0.122424 |
| optimized | 0.125 | 18.8644 | 0.121339 |
| optimized | 0.15 | 18.9367 | 0.120186 |
| reference | 0.01 | 16.43 | 0.157943 |
| reference | 0.025 | 15.1963 | 0.169312 |
| reference | 0.05 | 18.3023 | 0.129907 |
| reference | 0.075 | 18.5561 | 0.126144 |
| reference | 0.1 | 18.7937 | 0.122389 |
| reference | 0.125 | 18.8694 | 0.12135 |
| reference | 0.15 | 18.9447 | 0.120224 |
| reference_repeat | 0.01 | 16.4305 | 0.158088 |
| reference_repeat | 0.025 | 15.2007 | 0.169176 |
| reference_repeat | 0.05 | 18.3025 | 0.129893 |
| reference_repeat | 0.075 | 18.5939 | 0.125939 |
| reference_repeat | 0.1 | 18.7906 | 0.122445 |
| reference_repeat | 0.125 | 18.8605 | 0.121357 |
| reference_repeat | 0.15 | 18.9357 | 0.120206 |

这些评价来自原Bank留出，不是全新独立场景质量认证；未重新做PSNR模型选择。数值验收才是等价依据，不以轻微PSNR差值代替。

## 6. 初始化与显存

真实续训稳态2.626倍；4训练shape初始化约1.161s。当前短256步含初始化后优化仍较慢，约422步可摊销额外初始化。此估算不包括反复拓扑/K变化的重捕获。

以下是每种后端、每张地图**独立新进程冷测**，相同NN+解码范围，含网络、原Adam、七shape及最后反传buffer，不含共同Fixture/完整地图：

| 地图 | 后端 | 常驻增量MiB | 图数 | 参数bytes | Adam双Moment bytes |
| --- | --- | --- | --- | --- | --- |
| fr1_xyz | decoded_graph | 37.3481 | 7 | 135224 | 270448 |
| fr2_xyz | decoded_graph | 37.7134 | 7 | 135224 | 270448 |
| fr1_xyz | graph_cached55_fused | 37.0312 | 7 | 135224 | 270448 |
| fr2_xyz | graph_cached55_fused | 37.2847 | 7 | 135224 | 270448 |
| fr1_xyz | reference | 11.9907 | 0 | 135224 | 270448 |
| fr2_xyz | reference | 12.1089 | 0 | 135224 | 270448 |

CSV另有每步临时峰值，不是GPU整进程/整地图峰值。参数135224 bytes、Adam双Moment270448 bytes不变；空间增长来自缓存、工作区、Graph静态激活和缓冲。

## 7. 失败、限制与剩余热点

- 单独eager融合不快，CPU launch抵消局部Kernel收益；Graph后才获明显收益。相同Core的Inductor K68初步通过，但冷编译约5.9秒且稳态略慢，未推荐；没有无目标重写cuBLAS GEMM。
- 初版Graph让静态IDs/ratio别名调用方缓存，跨ratio第五步失败336项；修复为私有clone后完整重验通过，原失败数据保留。
- 首次capture不能在同一组参数有未完成Autograd图时执行。接口探针曾违反此约束，日志保留；改在安全初始化边界/no_grad预热后通过。不是通过detach实际训练梯度规避。
- Torch helper三次预热保持；不采用上游版本中失败的zero-warmup。独立helper依赖Torch版本，升级需重验。
- 完整Renderer+Adam capture在原Adam capturable=False处失败，未修改Optimizer。不能据此断言Renderer本身不可捕获。源码另有num_rendered、bucket_sum的Device-to-Host计数和动态buffer；本轮未复用旧计数、截断tile或改变Renderer工作量。
- 剩余约1.66ms含Renderer、Loss、原Adam和图外调度。下一步若继续降延迟，应独立研究Renderer动态计数与buffer；需要新的严格等价工程，不应继续为很小的神经GEMM付出过高维护成本。

## 8. 在线适用性与交付状态

**IMPLEMENTED / TESTED：** 两地图七档K、Frozen Master、共享神经网络训练、原生解码Graph、全部网络与输入leaf VJP、真实256步、缓存/Graph失效与边界。

**PARTIALLY VERIFIED：** 输入leaf与bases梯度正确，但原Features生产者本来在no_grad下，raw Master→19D全链路及动态人口归一化不是本轮实现。合成append/prune/reorder不等于生产SLAM生命周期集成。

**NOT IMPLEMENTED：** 生产C++/在线SLAM接入、实时Master更新、闭环及Gaussian优化器迁移、ATE、动态shape重捕获策略、并发Forward/乱序Backward、inference_mode全部组合。不能将固定地图倍数推广为Mapping总收益。

缓存记录父表data_ptr、shape和PyTorch版本，原地PyTorch写入重建55D；外部CUDA写入必须invalidate。bases原地更新由图重放读取当前storage；更换storage/N/K重建。调用方必须同步更新19D生产者、Anchor尺度和render_bases，图不会自动替用户重新推导原19D统计。参数storage/requires_grad变化、append/prune/reorder后应在无未完成反传边界invalidate并释放旧图；普通原地Adam更新无需重建。

保持reference / optimized / experimental三路径；仅性能研究默认启用。生产源码SHA与研究开始前历史快照匹配，本地tracked生产无改动。全部源码、配置、指标、失败日志、代码版本、trace/DOT及续训模型随归档交付。
