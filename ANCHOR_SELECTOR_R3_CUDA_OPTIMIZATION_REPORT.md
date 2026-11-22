# Anchor / Selector / R3 CUDA 优化实施报告

> 部署状态更新（2026-10-23）：按用户要求，现有七项 CUDA 优化已默认启用，显式配置 `0` 可逐项回退 Legacy。本文保留上一轮的验收数据和当时的启用决策，不代表当前默认配置；后续验收限制见第二阶段报告。

## 1. 结论与启用决策

已实现、编译并接入**可切换**生产路径的模块为：Multi-Tensor Dense Adam、14D Decode/VJP、Anchor Input27/VJP。Legacy全部保留，三个默认开关均为0。

固定真实checkpoint的LoD完整训练步中位耗时降低 **27.7%～34.8%**（约1.38～1.53倍吞吐）；Full降低约5.1%。LoD的CUDA launch由**418→177**，Full由**143→96**。这不是完整在线SLAM的独占验收结果。

四次真实fr1_xyz运行中，在线Mapping均值由7.408→5.998ms，降低19.0%；Tail由13.846→11.122ms，降低19.7%。完整进程墙钟均值由52.346→45.831秒，降低12.4%（吞吐约1.14倍）。GPU始终存在其他任务，以上是共享负载下观测值，不能承诺稳定独占倍率。

**不默认开启的原因：** 在线最终地图与训练步数变化；non-Bank Full PSNR均值降低0.285dB，same-K 1%降低1.147dB。无法将其因果归于kernel误差，但也不能宣称在线质量未受影响。固定单步质量一致，200步受控连续训练末段均值差小于0.004dB，只支持局部数值/短程证据，不替代在线质量gate。

## 2. 版本、真实架构和环境

- HEAD：`4269d7e3bad6cf5af7e1df1763634b61e98e0b51`，与审计参考一致；本轮新增未提交修改。没有stage、commit或push。
- Variable Anchor 1～10成员、L0/L1/L2、Feature16、Input27、27→32共享Encoder、32→32→2 Selector、32→32→14冻结共享C。
- 生产R3：四张[N,14]独立参数表，`delta=detach(C(h))+b_ratio[ids]`；原Dense Adam，当前ratio表grad defined，其他ratio为None。未见ratio仍按原线性插值与端点规则。
- split继续采用R diag(scale)随机扰动；没有修改clone/split/prune、UID分配、Anchor growing/pruning、Loop closure迁移或Loss。
- 原始真实checkpoint：`/home/prmi/zwj/private-compensation-online-20261022/r3_independent14_seed2026/7465_shutdown/ply`；N=15,891，Anchor=6,121；level分布1686/4179/256。K为156/799/1582/2368。
- 隔离源码与构建：`/home/prmi/zwj/anchor-selector-r3-cuda-optimization-20261022/source`、`build`。原生产source与原checkpoint保留。
- RTX3090 24GB；CUDA11.8；LibTorch2.3.1 ABI1；g++9；sm86；FP32；`--fmad=false`。Python分析使用单独Torch2.3.1 ABI0环境，未混载ABI。
- 复用原Native Fixture、Gaussian布局和真实相机；完整重放显式恢复六组Master leaf的requires_grad。仅模块压力测试的10K/50K/100K为人工平铺，不能当作独立场景。

## 3. 调用路径与代码

|模块|文件 / 函数|接入位置 / 范围|
|---|---|---|
|Fused Adam host|include/mapping_cuda_optimizer.h；src/mapping_cuda_optimizer.cpp::tryFusedDenseAdam|GaussianMapper::stepWithHistoricalGuard，Guard未启用分支；不支持时整步fallback|
|Multi-Tensor kernel|src/transform_field_cuda_optimizer.cpp::tfCudaAdamDirect；src/transform_field_cuda_optimizer.cu::adamKernel|复用已存在的FP32多Tensor内核，本轮未改TF实现；每批最多32Tensor|
|Decode autograd|include/anchor_selector_cuda.h；src/anchor_selector_cuda.cpp::Decode14|accumulateLodGradients的14维属性decode，opacity ST gate只执行一次|
|Decode kernel|src/anchor_selector_cuda.cu::decodeForward / decodeBackward|各一次kernel，保留外部base属性activation和index_select/scatter|
|Input27 autograd|include/anchor_input_cuda.h；src/anchor_input_cuda.cpp::Input27|GumbelNetworkImpl::anchorInputs，经CARTGS_FUSED_MAPPING_CUDA编译定义接入|
|Input27 kernel|src/anchor_input_cuda.cu::pack / vjp|Feature16、Offset3、Scale3、DC3、Opacity1、logratio1；Feature用atomicAdd累加|
|配置和导出|include/gaussian_mapper.h；src/gaussian_mapper.cpp::readSettings / saveSelectorConfig；cfg/selector/selector_config.yaml|三个独立开关；冻结teacher复制InputPack开关；不改模型参数schema|

Header-only CPU单测不引入CUDA符号依赖；CMake将CARTGS_FUSED_MAPPING_CUDA作为transform_field_cuda的PUBLIC定义传播给生产和相应CUDA测试。原CPU测试仍可使用Legacy。

## 4. Phase 1：Dense Adam设计与正确性

整步预检CUDA FP32、contiguous、gradient/moment形状及device、AMSGrad、重复参数和重叠storage。预检失败在任何参数/state变更前返回false。CPU、非连续、AMSGrad、重叠view保留Legacy。

每个参数使用自身group的lr、beta、eps、weight decay以及AdamParamState.step；生产epsilon从options读取，为1e-15。grad=None不创建state、不递增时钟；梯度为0的未选行仍衰减历史m/v并更新参数。不是SparseAdam，不是selected-K更新，也未改成离线R3_matched。

复用现有32-Tensor descriptor，动态拓扑之后每次从当前optimizer组重新取Tensor，未缓存过期指针。更新保留parameter/m/v version bookkeeping。当前活跃leaf少于32，Full/LoD均只提交1个Adam kernel。

最终固定梯度测试 **13,640项**，参数/m/v的max absolute、relative、RMS误差均为0，bad_elements=0；仍使用预注册atol=2e-7、rtol=2e-6。覆盖五个真实fixture连续30步、100步1e-10量级梯度、零梯度历史moment、grad=None、独立ratio时钟、不同LR，以及67Tensor跨三批、beta(.8,.98)、weight decay、eps1e-15/1e-8切换和fallback。

拓扑测试额外启用Fused，在真实Gaussian append/clone/split/prune、Feature/dual Adam迁移后继续执行step；R3行、UID、optimizer alias、m/v尺寸和clock检查通过。完整六组Master与R3生命周期没有改规则。

最终额外重叠storage保护只增加CPU预检；正式整步/在线计时来自加入此防御检查前的候选，CUDA数值/launch完全相同。最终版模块重新编译并复测；不把早期整步时间说成这次CPU防御分支的精确最终耗时。相关版本源码均保存于evidence。

## 5. Phase 2：Decode14/VJP

输入为选中K行的XYZ、activated Scale、**已由原getRotationActivation正规化的base quaternion**、raw opacity logit、DC、delta14与ST mask。输出为XYZ、Scale、Rotation、gated Opacity、DC，保持[K,3]/[K,4]/[K,1]/[K,1,3]布局。

Forward保留XYZ相加、Scale乘exp、identity+delta quaternion归一化、quaternion乘法和再次归一化、sigmoid(logit+deltaOpacity)、DC相加、ST opacity gate。没有给base rotation额外加一次归一化。Backward保留全部base属性、delta14和mask的VJP；未被使用的输出对应grad=None，不制造零梯度leaf推进Adam。

首版独立算子通过，却在真实Renderer的5%/10%各3个通道出现超容差像素（max分别0.000688/0.001289）。原因定位到ATen归约树：norm4为(a0²+a1²)+(a2²+a3²)，三项点积为(a0b0+a2b2)+a1b1。已按真实ATen顺序修正，旧失败证据保存于reduction_v1_failure，**没有放宽容差**。

最终179项Decode检查全部bad_elements=0。真实属性Forward除rotation外可精确一致；rotation仍有约1e-7量级差异，完整render通过atol2e-5/rtol2e-4。覆盖K=0/1/7、identity、tiny/精确零quaternion、±100饱和logit、极大Scale/XYZ、极小Scale、非连续输入、重复ID的Backward scatter和unused output。精确零quaternion的Legacy NaN VJP也原样保留，不在性能优化中修复算法边界。

## 6. Phase 3：Input27/VJP

单kernel直接读取当前owner、Anchor Feature/center/reference scale与Gaussian属性，生成连续[N,27]；Offset=(xyz-center)/reference scale，不clamp、不缓存跨step变化的数值。原Encoder、Softmax、sampling和budget代码不变。

Backward按owner atomicAdd Feature16梯度，直接回传XYZ/reference scale、activated Scale、DC与activated Opacity梯度，再由原activation回传Master。center/reference scale继续是非训练buffer；没有增加它们的优化路径。支持非连续输入的contiguous拷贝，CPU/不支持Anchor布局走Legacy。

84项真实Variable Anchor检查通过，Input27、Encoder输出、概率在fixture上精确相同；四档Cached ID/顺序/K精确一致。重复owner梯度聚合有FP32 atomic顺序差异，max absolute约1.91e-6，预注册容差内。覆盖1/10成员、owner重排、长尾Offset、空集及非连续输入；实际拓扑单测同时验证变更后continued step。

大规模人工测试发现100K Forward回退（见第10节），故不能把真实15.9K的收益外推到所有地图规模。

## 7. 固定checkpoint完整训练步耗时

每分支10次warmup，60个正式完整step；两次Legacy、两次全候选，另保留Adam-only与Adam+Decode中间版本。表为两次run各自p50的均值。`forward_stage_event_sum`是各Forward scope的CUDA Event span之和，包含CPU提交空隙/外部排队，不是纯kernel时间；cycle为同步墙钟。正式计时与Nsight分开。

| case         | stage                   |   legacy_ms |   fused_ms |   latency_reduction_percent |   speedup |
|:-------------|:------------------------|------------:|-----------:|----------------------------:|----------:|
| Full         | cycle                   |      2.9436 |     2.7944 |                      5.0654 |    1.0534 |
| Full         | forward_stage_event_sum |      1.1663 |     1.1612 |                      0.4390 |    1.0044 |
| Full         | backward                |      1.4723 |     1.4372 |                      2.3821 |    1.0244 |
| Full         | adam                    |      0.1439 |     0.0358 |                     75.0890 |    4.0143 |
| lod_0.010000 | cycle                   |      5.2917 |     3.4622 |                     34.5737 |    1.5284 |
| lod_0.010000 | forward_stage_event_sum |      2.1066 |     1.8176 |                     13.7198 |    1.1590 |
| lod_0.010000 | backward                |      2.1729 |     1.4006 |                     35.5443 |    1.5515 |
| lod_0.010000 | adam                    |      0.6164 |     0.0471 |                     92.3588 |   13.0870 |
| lod_0.050000 | cycle                   |      5.3365 |     3.4802 |                     34.7845 |    1.5334 |
| lod_0.050000 | forward_stage_event_sum |      2.0923 |     1.8770 |                     10.2900 |    1.1147 |
| lod_0.050000 | backward                |      2.1688 |     1.4047 |                     35.2337 |    1.5440 |
| lod_0.050000 | adam                    |      0.6200 |     0.0471 |                     92.4030 |   13.1630 |
| lod_0.100000 | cycle                   |      5.3530 |     3.5809 |                     33.1053 |    1.4949 |
| lod_0.100000 | forward_stage_event_sum |      2.1862 |     1.7797 |                     18.5948 |    1.2284 |
| lod_0.100000 | backward                |      2.1130 |     1.5060 |                     28.7256 |    1.4030 |
| lod_0.100000 | adam                    |      0.6262 |     0.0471 |                     92.4775 |   13.2935 |
| lod_0.150000 | cycle                   |      5.3668 |     3.8828 |                     27.6516 |    1.3822 |
| lod_0.150000 | forward_stage_event_sum |      2.2070 |     1.8711 |                     15.2187 |    1.1795 |
| lod_0.150000 | backward                |      2.1338 |     1.5222 |                     28.6624 |    1.4018 |
| lod_0.150000 | adam                    |      0.6280 |     0.0481 |                     92.3359 |   13.0479 |

Profiler实际GPU kernel时间与CPU API/queue分别保存在kernel_execution_detail.csv、cuda_api_detail.csv；不能把microbenchmark耗时占比直接当作生产在线占比。

## 8. CUDA launch变化

| case         | stage               |   profile_fused |   profile_legacy |
|:-------------|:--------------------|----------------:|-----------------:|
| Full         | adam                |          1.0000 |          48.0000 |
| Full         | all                 |         96.0000 |         143.0000 |
| Full         | backward            |         48.0000 |          48.0000 |
| lod_0.010000 | adam                |          1.0000 |         112.0000 |
| lod_0.010000 | all                 |        177.0000 |         418.0000 |
| lod_0.010000 | anchor_input_pack   |          1.0000 |         nan      |
| lod_0.010000 | anchor_lookup       |        nan      |           5.0000 |
| lod_0.010000 | attribute_transform |          3.0000 |          34.0000 |
| lod_0.010000 | backward            |         91.0000 |         182.0000 |
| lod_0.010000 | input27             |        nan      |           4.0000 |
| lod_0.050000 | adam                |          1.0000 |         112.0000 |
| lod_0.050000 | all                 |        177.0000 |         418.0000 |
| lod_0.050000 | anchor_input_pack   |          1.0000 |         nan      |
| lod_0.050000 | anchor_lookup       |        nan      |           5.0000 |
| lod_0.050000 | attribute_transform |          3.0000 |          34.0000 |
| lod_0.050000 | backward            |         91.0000 |         182.0000 |
| lod_0.050000 | input27             |        nan      |           4.0000 |
| lod_0.100000 | adam                |          1.0000 |         112.0000 |
| lod_0.100000 | all                 |        177.0000 |         418.0000 |
| lod_0.100000 | anchor_input_pack   |          1.0000 |         nan      |
| lod_0.100000 | anchor_lookup       |        nan      |           5.0000 |
| lod_0.100000 | attribute_transform |          3.0000 |          34.0000 |
| lod_0.100000 | backward            |         91.0000 |         182.0000 |
| lod_0.100000 | input27             |        nan      |           4.0000 |
| lod_0.150000 | adam                |          1.0000 |         112.0000 |
| lod_0.150000 | all                 |        177.0000 |         418.0000 |
| lod_0.150000 | anchor_input_pack   |          1.0000 |         nan      |
| lod_0.150000 | anchor_lookup       |        nan      |           5.0000 |
| lod_0.150000 | attribute_transform |          3.0000 |          34.0000 |
| lod_0.150000 | backward            |         91.0000 |         182.0000 |
| lod_0.150000 | input27             |        nan      |           4.0000 |

LoD减少241次：Adam -111，Backward整体 -91，Input lookup+pack -8，Decode scope -31。Decode scope的3次包含raw logit gather和ST mask gather各1次；Decode自身为1次。R3 gather/add仍为2次，Selector MLP/Softmax/Sampling没有重写；不将相邻模块收益归功于R3 Lookup。

## 9. 完整在线SLAM、Tail与质量

运行顺序Legacy-A→Fused-A→Fused-B→Legacy-B，同一binary、数据、seed=2026和默认配置，仅改三个实现开关。自然停止，未设Mapping10000，serial_full_lod=false，TF关闭。每Mapping低开销trace保持一致；没有同时跑多个自己的训练进程。

以下是两次运行统计量的均值，单位ms。online/tail采样数不同，不视作同一训练时钟。

| phase   | variant   |    mean |     p50 |     p95 |     p99 |
|:--------|:----------|--------:|--------:|--------:|--------:|
| online  | Fused     |  5.9983 |  4.0652 | 14.6420 | 35.0349 |
| online  | Legacy    |  7.4078 |  5.2934 | 15.7442 | 39.4076 |
| tail    | Fused     | 11.1219 |  8.3150 | 20.5314 | 20.7759 |
| tail    | Legacy    | 13.8461 | 10.5408 | 21.8285 | 22.3041 |

| run             |     N |   Anchor |   online_steps |   tail_steps |   online_topology |   tail_topology |
|:----------------|------:|---------:|---------------:|-------------:|------------------:|----------------:|
| online_legacy_a | 13479 |     5344 |           4144 |          756 |                 7 |               1 |
| online_fused_a  | 14792 |     5817 |           5041 |          400 |                 9 |               0 |
| online_fused_b  | 15575 |     6366 |           5057 |          400 |                 9 |               0 |
| online_legacy_b | 12912 |     5041 |           4108 |          792 |                 7 |               1 |

加速改变异步SLAM在相同轨迹时间内执行的Mapping次数，在线topology7→9；Legacy还进行1次Tail topology而Fused没有。因此地图、Full总次数、R3各ratio训练次数和tail预算均不相同。原有调度规则没有修改，但最终训练序列没有保持相同，不能以这些结果证明等价算法质量。

评价复用先前预定64个RGB帧，按各自真实估计pose，四个模型Bank并集排除后有51个non-Bank帧。与先前离线70/44协议不是同一视图划分，不混合相减。四模型统一common-K=129/645/1291/1936；same-K沿原cached-uniform margin排序。保存2304条逐视图记录，PSNR/SSIM/LPIPS/photo均报告。

| policy   |   ratio | variant   |    psnr |   ssim |   lpips |   photo |
|:---------|--------:|:----------|--------:|-------:|--------:|--------:|
| Full     |  1.0000 | Fused     | 21.0461 | 0.7306 |  0.3397 |  0.0948 |
| Full     |  1.0000 | Legacy    | 21.3312 | 0.7391 |  0.3314 |  0.0918 |
| cached   |  0.0100 | Fused     | 17.0745 | 0.5981 |  0.6802 |  0.1511 |
| cached   |  0.0100 | Legacy    | 17.0333 | 0.5986 |  0.6834 |  0.1511 |
| cached   |  0.0500 | Fused     | 19.2047 | 0.6611 |  0.5347 |  0.1200 |
| cached   |  0.0500 | Legacy    | 19.2956 | 0.6608 |  0.5369 |  0.1193 |
| cached   |  0.1000 | Fused     | 19.6893 | 0.6785 |  0.4848 |  0.1133 |
| cached   |  0.1000 | Legacy    | 20.0407 | 0.6836 |  0.4829 |  0.1104 |
| cached   |  0.1500 | Fused     | 19.8103 | 0.6841 |  0.4663 |  0.1112 |
| cached   |  0.1500 | Legacy    | 20.1330 | 0.6877 |  0.4640 |  0.1089 |
| same_k   |  0.0100 | Fused     | 15.8505 | 0.5833 |  0.6875 |  0.1656 |
| same_k   |  0.0100 | Legacy    | 16.9972 | 0.5983 |  0.6845 |  0.1516 |
| same_k   |  0.0500 | Fused     | 19.0435 | 0.6552 |  0.5434 |  0.1228 |
| same_k   |  0.0500 | Legacy    | 19.2831 | 0.6604 |  0.5366 |  0.1195 |
| same_k   |  0.1000 | Fused     | 19.5964 | 0.6762 |  0.4900 |  0.1146 |
| same_k   |  0.1000 | Legacy    | 20.0472 | 0.6838 |  0.4822 |  0.1104 |
| same_k   |  0.1500 | Fused     | 19.7647 | 0.6821 |  0.4712 |  0.1121 |
| same_k   |  0.1500 | Legacy    | 20.1376 | 0.6877 |  0.4618 |  0.1088 |

**在线质量gate没有通过。** non-Bank平均Full -0.285dB；same-K四档分别约-1.147/-0.240/-0.451/-0.373dB；1%的一次候选明显较差。不能将不同地图的退化直接定性为CUDA错误，也不能用Cached K更大掩盖common-K退化。两个重复来自同一场景，不是独立场景验证。

## 10. 显存、分配与规模

真实模型未增加训练参数、R3表或Adam状态；三个候选只改变计算/中间分配。人工100K/5K下的allocator峰值额外分配：Dense R3 Adam Legacy约11.20MB→Fused0；Decode Forward约0.786→0.282MB，31→5次allocation；Decode Backward约0.622→0.583MB，60→7次allocation；Input Forward约22.00→10.80MB，8→1次allocation。Input Backward为了显式返回各base梯度，峰值约1.20→4.39MB，2→5次allocation，不能宣称全部内存项下降。

人工Input Pack Forward p50：10K 0.084→0.020ms，50K 0.112→0.073ms，100K **0.154→0.184ms**（回退约20%）。100K Backward 0.177→0.158ms。现有每线程一Gaussian的访存方式仍需按大N优化，不应全规模强开。

在线allocated峰值（项目计数器MB）Legacy-A/B为486.336/471.254，Fused-A/B为466.050/467.997；reserved峰值分别566/530与526/536。不同N/拓扑的进程峰值不能视为kernel本身的显存因果对照。外部负载保存在gpu_samples.json；人工allocator数据也不是整个在线进程峰值。

## 11. 完整Backward、Adam放大与连续训练

固定重放共3,945项比较：除下一段明确记录的parameter_after放大外，Forward、全部六组Master、Anchor Feature、Encoder、Selector、当前ratio R3梯度、其他ratio grad=None、m/v、clock、UID、owner、ID/K均通过原定gate。候选完整梯度max absolute约3.82e-6，RMS最大约1.67e-8，bad_elements=0。不能将这里的gate表述为全部post-step参数逐元素一致。

不将parameter_after差异藏进宽容差：Baseline-repeat已在1% Scale出现约5.54e-4的单元素Adam放大，候选约5.64e-4；R3约6e-5。eps1e-15会放大atomic造成的微小梯度差。固定grad/moment的最终Fused Adam为严格0误差；完整链parameter_after仍逐项记录，未宣称bitwise训练一致。

`worst_error_locations.csv`补充每项Tensor最大误差的flat/多维索引、参考值、候选值，以及可对应时的Gaussian row/UID、Anchor owner、base XYZ/Scale/Rotation/Opacity输入，供复现tiny-gradient放大或渲染边界。

补充四个连续分支：同一真实地图、相机、cached uniform、固定拓扑、fresh且相同Adam初态；每轮Full→轮转一个LoD，共100Full+100LoD、每ratio25次。持续更新参数/moments，不恢复权重，不做伪零梯度step。末5轮汇总如下（LoD混合ratio，不能当各档指标）：

| run                 | kind   |   photo_mean |   psnr_mean |     K_mean |
|:--------------------|:-------|-------------:|------------:|-----------:|
| continuous_fused_a  | Full   |       0.0288 |     31.7531 | 15891.0000 |
| continuous_fused_a  | LoD    |       0.0654 |     25.3936 |  1453.6000 |
| continuous_fused_b  | Full   |       0.0288 |     31.7556 | 15891.0000 |
| continuous_fused_b  | LoD    |       0.0654 |     25.3947 |  1431.6000 |
| continuous_legacy_a | Full   |       0.0288 |     31.7524 | 15891.0000 |
| continuous_legacy_a | LoD    |       0.0654 |     25.3933 |  1453.4000 |
| continuous_legacy_b | Full   |       0.0288 |     31.7553 | 15891.0000 |
| continuous_legacy_b | LoD    |       0.0654 |     25.3907 |  1453.4000 |

末段PSNR均值差<0.004dB、UID/owner/时钟精确一致；连续参数不满足strict allclose，99步时Baseline-repeat与一个候选均出现选点分歧。此结果支持短程没有明显系统性质量下降，不证明多视图长期在线等价。没有以验证集调LR/步数或选择best checkpoint。

## 12. Phase 4～6与并行化边界

Phase4的Anchor Full FP64统计/完整检查、Phase5 Stable Compaction、Phase6 Renderer workspace均**未实现**。在在线质量gate未通过时不继续叠加生产变更；保留原同步、统计、prune阈值和member规则。

Renderer的num_rendered/bucket_sum D2H、dynamic K同步、六组gather、R3全表Dense Adam读写仍存在。没有删除mutex_render，没有四ratio并行Backward，没有Full/LoD并发更新Master，也没有把只支持默认stream的Renderer强迁多stream。

Nsight Systems成功采集kernel、API和排队；Nsight Compute最终得到ERR_NVGPUCTRPERM，未取得硬件计数器。首次NCU还遇到LibTorch linalg动态库路径问题，补齐后确定权限阻塞。没有据此宣称SSIM受memory bandwidth限制，也未重写SSIM。

Nsight的5%候选重放中，纯kernel时长之和约2.595ms/步，其中SSIM Backward约0.799ms（30.8%，最大单kernel），duplicateWithKeys约0.212ms，SSIM Forward约0.199ms，PerGaussianRenderCUDA约0.187ms。只取3次被Profiler插桩的step且存在外部负载，不能代替正式端到端计时。该数据将下一轮kernel级优先级首先指向SSIM Backward，但其硬件瓶颈类型仍未确定。

## 13. 可复现命令和交付证据

诊断目录：`tests/diagnostics/anchor_selector_r3_cuda_optimization_20261022/`。脚本包括setup_build.py、run_validation.py、verify_replays.py、module_scaling.cpp、controlled_training_replay.cpp、run_continuous.py、evaluate_online.py、analyze_results.py与export_delivery.py。服务器完整证据包delivery_evidence.tar.gz约2.1GiB，保留各分支Tensor快照；本地compact_evidence.tar.gz包含完整CSV/日志/Nsight/source SHA以及一个5%候选完整fixture，旧Native fixture继续复用原审计目录。最终源码SHA见final_source_manifest.json/local_source_manifest.json；服务器保存完整在线checkpoint，不清理原始模型。

服务器构建/单测示例（不要混用Python ABI0库路径）：

```bash
R=/home/prmi/zwj/anchor-selector-r3-cuda-optimization-20261022
export LD_LIBRARY_PATH="$R/source/lib:/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib"
/home/prmi/.local/bin/cmake --build "$R/build" -j2 --target fused_mapping_adam_test fused_compensation_decode_test fused_anchor_input_test optimized_native_benchmark tum_rgbd
"$R/source/bin/fused_mapping_adam_test" /home/prmi/zwj/anchor-selector-r3-performance-20261022/native1 "$R/recheck_adam"
"$R/source/bin/fused_compensation_decode_test" /home/prmi/zwj/anchor-selector-r3-performance-20261022/native1 "$R/recheck_decode"
"$R/source/bin/fused_anchor_input_test" /home/prmi/zwj/private-compensation-online-20261022/r3_independent14_seed2026/7465_shutdown/ply /home/prmi/zwj/anchor-selector-r3-performance-20261022/native1 "$R/recheck_pack"
```

重放最后一个参数bits：0=Legacy，1=Adam，3=Adam+Decode，7=全部。现有run_validation.py会保留同名目录，不适合直接覆盖旧证据；需要新隔离输出根运行。默认生产开关保持：

```yaml
CUDA.fused_dense_adam: 0
CUDA.fused_compensation_decode: 0
CUDA.fused_anchor_input_pack: 0
```

非支持布局自动Legacy fallback；14D以外的旧补偿路径也保持Legacy。任一实验flag均不改变R3表/ratio插值/checkpoint schema。

## 14. 对八个验收问题的明确回答

1. **整体Mapping提高多少？** 固定地图LoD步延迟降低27.7%～34.8%，Full约5.1%；共享GPU真实在线单Mapping均值降低19.0%，Tail19.7%，完整进程12.4%。不同在线步数/地图和外部负载限制因果解释，尚无生产默认GO。
2. **LoD Forward/Backward/Adam分别？** 四档Forward scope Event span降低10.3%～18.6%，Backward降低28.7%～35.5%，Adam约92.3%～92.5%；对应详细表和run波动保留，纯kernel时间另表。
3. **Anchor/Selector/R3 launch？** Anchor input lookup+pack9→1；Selector MLP/Softmax/sampling未变；R3 lookup/add仍2，decode scope34→3，LoD Adam112→1，总418→177。没有虚报R3 lookup单独加速。
4. **同步和访存空间？** 仍有dynamic K、Renderer D2H、多组gather、Anchor检查scalar同步、大N Input Pack不连续访存；Dense R3全N moment更新属于当前语义，不可通过只更新K行消除。
5. **继续融合哪些？** 优先在独立gate下研究selected gather+R3 add+decode，再研究Anchor检查/Full统计、sampling/稳定compaction；不先重写Renderer全体或自研小GEMM。
6. **哪些不值得继续投入？** 当前单独R3 lookup、小共享C的GEMM不是优先热点；没有证据说这些模块完全无空间。Input Pack大N尚有反例，不能结束优化后全开。
7. **在线SLAM质量受影响吗？** 当前实际在线对照有退化，质量门未通过；固定单步质量一致及短程曲线不能覆盖这个结果。保留Legacy默认。
8. **最大剩余瓶颈？** 技术热点是Renderer/SSIM Backward、Renderer/动态K同步和Anchor Full/Topology处理；最大的接入阻碍是速度改变异步训练序列后的可比较在线质量验收。必须先建立同真实帧/拓扑/优化时钟的记录重放再给生产GO。
