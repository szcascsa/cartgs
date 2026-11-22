# 冻结部署CUDA深入优化：论文与官方资料调研

检索日期：2026-11-07。适用实现：mycartgs训练完成后的独立部署后端，当前扩展v7。本文只进行资料检索、源码核对与研究设计，没有修改部署、训练代码或服务器环境。

## 结论

有继续深入优化的依据。当前大地图方案失败说明“分块Histogram＋单CTA汇总＋按整图容量发射”这一实现的代价偏高，不能推导为GPU管理动态N/K没有价值。最值得验证的路线是：自适应精确选择、按小批点复用MLP权重，以及固定Buffer地址下复用准备阶段CUDA Graph。Rasterizer的Tile展开和排序有独立空间。

不同ratio需要不同优先级。不能将论文相对原始3DGS或torch.topk的倍数，直接当作相对当前v7的预期收益；当前实现已经做了网络融合、C++调度、稳定压缩和同步合并。

## 1. 当前实测证据与不确定性

核对本地deployment_device_counts_20261107的最终Nsight数据：

|工作段|代表条件|该段累计Kernel时间|主要Kernel|占该段累计Kernel时间|
|---|---|---:|---|---:|
|完整准备|局部1%|129.10μs|精确阈值选择43.47μs|33.7%|
|完整准备|局部70%|332.45μs|Compensator 222.21μs|66.8%|
|Rasterizer|局部70%|391.34μs|duplicateWithKeys 176.65μs|45.1%|
|Rasterizer|局部70%|391.34μs|主要Onesweep排序Kernel合计89.09μs|22.8%|

这些是各工作段的GPU累计时间占比，不是完整帧墙钟占比；排序还有其他辅助Kernel。共享GPU负载存在，完整耗时以原报告的两轮交错配对为准。

大地图分块设备选择在26596/66490点合成地图上出现退化；完整准备的正确性通过，但未完成大地图的独立模块Nsight归因。源码确认了以下额外工作：每轮仍扫描实际N个候选、每Block写2048桶、单CTA合并所有Block桶、容量上界发射与提前退出。它们各贡献多少退化仍待测量。

合成地图通过重复原地图和噪声构造，可能放大相同分数与相同Anchor访问的影响。最终静态16384点分流门限是保守工程选择，未证明是最佳门限，也不是硬件上限。下一轮需要实际大场景checkpoint，记录整图容量、实际N、K、资格比例、重复score比例和Tile实例数。

## 2. 直接相关的论文与资料

|资料|核实内容|对本项目的意义|限制|
|---|---|---|---|
|[RadiK，2025 arXiv版本](https://arxiv.org/html/2501.14336v1)|Radix Top-K的层次化原子、写缓冲、带宽优化及偏斜分布处理|最贴近大地图Histogram与候选访存瓶颈|论文数值缩放不能直接假定保持我们FP32同分结果；先借鉴数据流|
|[AIR Top-K / SC 2023论文](https://doi.org/10.1145/3581784.3607062)与[NVIDIA官方AgentTopK](https://github.com/NVIDIA/cccl/blob/main/cub/cub/agent/agent_topk.cuh)|官方实现区分已选中项、下一轮候选和丢弃项|后续轮次应收缩不确定候选，而非始终重扫全N|ACM全文请求403；算法判断以官方源码为依据，论文题名/作者/年份由DOI元数据核实|
|[Qrita，2026，v3](https://arxiv.org/html/2602.01518v3)|四分pivot搜索、重复值处理、预筛选失败回退；作者报告与稳定排序输出一致|适合研究饱和score、大量同分时的精确阈值和同分选取|LLM logits的统计结构不能套用到Selector概率；Gaussian σ-truncation指分布，不是3D Gaussian几何|
|[Prof-K，2026，v2](https://arxiv.org/html/2608.12573v2)|采样估阈值，一次全量过滤后精确选择，运行时失败检测与重试|很大N、很低K/N时可研究减少候选规模|论文明确小输入收益不足；理论使用不同值假设；我们必须处理同分，不能宣称当前十万级必然有效|
|[GPU-TopK / SIGMOD 2018作者实现](https://github.com/anilshanbhag/gpu-topk)|同时提供Bitonic、Radix Select与完整排序|支持按N、K、比例分流，不应强求单算法覆盖所有请求|旧实现与当前GPU/资格/同分语义需适配|
|[NVIDIA CUTLASS Grouped Schedulers](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/grouped_scheduler.md)|持久Block负责多个矩阵Tile，支持设备侧调度参数|为四Head小矩阵的批处理、负载分配提供实现参考|Grouped GEMM本身不是自动收益；示例常用半精度，不能直接照搬为FP32等价路径|
|[NVIDIA CUDA Graph官方指南](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)|执行计划与输入数据分离，可重复提交固定图|新的设备计数接口更适合少量固定图，变化相机/ratio不必逐K建图|固定图不自动解决动态Grid和排序有效长度|
|[Device Graph Launch官方资料](https://developer.nvidia.com/blog/enabling-dynamic-control-flow-in-cuda-graphs-with-device-graph-launch/)、[Conditional Nodes官方资料](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/)|设备侧调度与条件分支/循环|可在更高版本独立运行时研究设备内路径选择与重试|Device Graph Launch从CUDA12.0提供，文档描述12.4的IF/WHILE条件节点；当前11.8不能直接使用这些API|
|[FlashGS，2024](https://arxiv.org/html/2408.07967v2)|投影冗余、Tile交集、流水预取、调度和内存访问优化|直接对应duplicateWithKeys和后续排序成本|当前代码已有opacity感知Tile相交测试，不能把该部分重复算成全新收益；质量接近不证明逐bit|
|[HiGS，NVIDIA 2026研究页](https://research.nvidia.com/labs/sil/projects/higs/)|粗Macro-Tile分区、分段深度排序、细Render-Tile合成与动态调度|大场景/高分辨率渲染具有更大架构级优化空间|其发布结果包含fp16路径；保持前后Alpha合成语义不代表FP32逐bit等价，也不是当前低分辨率小地图的收益保证|

已读取RadiK、Qrita、Prof-K、FlashGS、Dr. Top-k的公开HTML全文，官方CUDA/CUTLASS/CCCL资料及相关作者仓库。检索得到但不匹配的结果没有作为证据。所有成功获取的内容保存在sources，附SHA256清单。

## 3. 推荐路线一：真正收缩候选的精确Top-K

### 现有实现的问题

上一版大地图Radix每轮只是改变prefix判断条件，实际仍重新读取整批N个score。每个Block写出完整Histogram，再由单CTA归并。候选不确定性已经减少，却没有同步减少后续读取量。

RadiK和官方AIR实现提供了直接参考：将确定入选的项计入结果，只压缩边界桶内尚未判定的项；下一轮只处理这批项。直方图采用层次化聚合与连续写缓冲，减少冲突和全局中间数据。[RadiK](https://arxiv.org/html/2501.14336v1)，[官方AgentTopK源码](https://github.com/NVIDIA/cccl/blob/main/cub/cub/agent/agent_topk.cuh)。

### 本项目的最小实现

1. 第一轮读取原score，生成有界整数Key、粗Histogram和有效计数。
2. 确定边界桶，把边界桶Key/原行ID压缩进两个有界Scratch之间；后续轮次只处理这个不确定桶。
3. 同分特别处理：若边界值已唯一或剩余Key全部相等，直接按原候选行序取所需数量，不反复扫描同分值。
4. 最终按原全局行序稳定发布K个ID，保留资格Mask与固定Gumbel噪声。

可用完整FP32排序Key加原ID形成明确的总顺序，也可只求分数阈值后独立解决同分。复合64bit Key更简单但增加带宽，需要和“32bit Key＋同分扫描”配对比较。正负零和NaN检查保持现有定义。

不能把score替换成logit差后声称严格等价：数学单调关系不消除FP32 softmax舍入造成的同分，点的身份可能改变。

建议按q=min(K,E−K)研究高保留率的补集选择，其中E是合资格数。补集必须选择原总顺序最末尾的E−K个点，同分时优先排除较后的行；最终仍要输出K个点，因此该技巧只能减少搜索代价，不能消除O(N)发布和O(K)补偿。

### Qrita与Prof-K如何借鉴

Qrita的多pivot与重复值处理值得借鉴，但Selector概率可能集中在0/1，不能假定其近似正态。统计阈值或采样阈值只能作为预筛选：对整个候选池按阈值过滤；过滤后不足K或Buffer溢出时回退/重试；满足条件后精确选择，并按现有规则处理边界同分。[Qrita](https://arxiv.org/html/2602.01518v3)。

Prof-K是很大N、很小保留比例的备选。其理论假设不同值，现有Selector有大量同分，需用总顺序或显式边界补齐；采样使用独立RNG，不能改变现有固定Gumbel噪声。论文自身提示小输入没有优势，不应将其1.5–15倍宣传结果用于我们当前N≈一万的视角。[Prof-K](https://arxiv.org/html/2608.12573v2)。

### 新版CUB不能直接替换

已核对[固定版本CCCL v3.2.0 DeviceTopK](https://github.com/NVIDIA/cccl/blob/v3.2.0/cub/cub/device/device_topk.cuh)：它是非稳定接口；本次检索main源码还明确限制输出顺序/确定性及tie-break要求。必须额外实现原索引优先级和最终稳定压缩。输出无序不等于集合不正确，但集合中同分点的身份也必须符合我们的定义。新接口的N/K仍有Host端参数，采用它不自动完成设备端动态计数。应先在独立工具链建立组件基线，再决定是否移植思路。

## 4. 推荐路线二：Compensator按小批点复用权重

当前Compensator每Gaussian一个128线程Block，四个Warp分别计算四Head。它减少了Kernel发射，但各Block仍反复读取相同权重；不能只根据“一个Gaussian一个Block”推导它已接近硬件上限。

建议在高K时测试一个Block处理8/16个Gaussian的SIMT FP32矩阵Tile：权重在Block内复用，bias/ReLU和四Head输出发布继续融合。四Head的第一层可以在部署时打包为64×256矩阵来计算，再按四个独立权重组切分；第二层依旧各自生成3/3/4/4维。只是冻结权重的布局与调度变化，不改训练架构、Head语义或参数。

低K继续使用现有逐点融合Kernel，高K才启用Tile路径。分流依据应由真实K分布和目标GPU实测决定，不把1024个固定Block当作通用最优。此前grid-stride循环仅改变谁处理下一点，未增加权重的批内复用，失败不代表这一矩阵Tile方案无效。[CUTLASS调度文档](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/grouped_scheduler.md)。

验收先保留FP32，不依赖Tensor Core半精度/TF32。累加顺序若变化，按已声明的部署质量门限验证，不能将它并入逐bit一致结论。四Head完全拼成新训练网络不在本方案范围内。

### 可选：checkpoint固定线性项预投影

由当前源码可推导：首层在ReLU之前是offset、Residual Feature、ratio投影、view投影的加法。冻结checkpoint后，可预计算W_feature乘各Anchor的Base/Residual表，以及固定offset的线性项；运行时按当前Residual权重求和并加入ratio/view项，最后执行原ReLU。

预投影内容只依赖checkpoint，不随相机、ratio改变失效；不新增每ratio的48D结果缓存。它会改变浮点分配律与加法顺序，需要单独误差验证。首层只占总MLP算术的一部分，四Head更重，不能承诺数倍整体收益。这个推导是针对当前代码的工程候选，不是上述论文已验证的mycartgs结论。

## 5. 推荐路线三：固定准备图与设备任务调度

### 当前CUDA11.8即可研究的部分

新的GPU控制记录使相机、ratio、N/K能作为本帧数据输入同一个固定准备图。固定算子连接和Buffer地址、每帧写入新相机与请求；不需要为每个K重新捕获Graph。这与此前动态K产生大量Graph构建开销的失败方案不同。

先只捕获准备阶段，末尾仍统一读取控制记录；Rasterizer保持外部调用。Graph输入输出Buffer采用有界槽位和事件管理。对外prepare返回的Tensor仍需独立所有权；应用持有上一帧输出时不能覆写，必要的复制成本计入完整耗时。

固定图不会消除容量尾部Block，也不会自动让CUB接受设备计数，必须独立测量。它主要减少Host发射和阶段间提交空隙。[CUDA Graph指南](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)。

### 需要独立较新工具链的部分

Device Graph Launch和Conditional Nodes能把路径选择/循环留在设备，适合候选压缩的终止与重试。不能据此认为GPU能够任意修改已实例化Graph的Grid尺寸；官方对设备图更新仍有限制。可以研究少量静态容量档位，或在条件循环中按块消费设备任务。

当前项目编译使用CUDA11.8，新能力需独立部署构建环境，不修改共享conda或训练环境。RTX3090上可用范围、驱动版本与收益仍需实测。普通持久Kernel也可在旧工具链研究，但必须按驻留资源设计同步；不能在超出同时驻留数量的Block之间使用自旋全局屏障。[Device Graph Launch](https://developer.nvidia.com/blog/enabling-dynamic-control-flow-in-cuda-graphs-with-device-graph-launch/)，[Conditional Nodes](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/)。

## 6. 推荐路线四：Rasterizer的Tile并行和分段排序

当前duplicateWithKeys由每Gaussian串行遍历Tile，footprint差异可能导致负载不均；现有高ratio trace该Kernel约占Rasterizer累计GPU时间45%。值得先做按Warp或小Block展开单Gaussian footprint，并用稳定前缀位置保留输出顺序。

源码已经具有opacity感知的Tile相交测试；不能照搬FlashGS的全部几何剔除再期待其论文倍数。更具体的候选是统计有效Tile对、减少无效哨兵进入排序、连续读取Gaussian属性，以及像素合成阶段预取。额外计数/压缩成本必须和节省排序量配对测量。[FlashGS](https://arxiv.org/html/2408.07967v2)。

更深入的HiGS将分区和合成拆为粗/细两级Tile，在Macro-Tile内使用深度分段排序，再调度细Tile合成。它说明全局复合Key排序不是唯一可行结构，但迁移规模更大；必须保留本项目阈值、同深度优先级和Alpha合成定义。发布结果含fp16，不能将“精确前后Alpha合成”理解成“FP32逐bit不变”。小地图、640×480和已经稀疏的K下也可能没有收益。[HiGS研究页](https://research.nvidia.com/labs/sil/projects/higs/)。

## 7. 下一轮建议实验顺序

|顺序|独立实验|最有信息量的对照|通过条件|
|---|---|---|---|
|0|实际大checkpoint专项Nsight|小/大实际N、低/高K、真实重复分数分布|分清Histogram、候选扫描、空Block、补偿与Tile排序成本；补齐目前归因空白|
|1|收缩候选的精确Radix|旧CUB、当前小CTA、收缩候选实现|ID/同分/严格K一致；低/高ratio都有合理分流，完整准备有重复收益|
|2|FP32小批点四Head Tile|现有逐点融合与小批Tile；同权重、同输入、同K|高K网络前向与整帧受益；低K不被强制切换；误差和质量单列|
|3|单一固定准备Graph|同一GPU算术，Host串行发射与Graph提交|动态相机、变化ratio与跳跃视角有效；计入捕获、槽位等待和结果复制|
|4|Tile展开并行化|保持原排序/像素Kernel，仅替换展开|深度Key/顺序/像素通过；观察不同footprint而非只看N|
|5|HiGS式粗细Tile|独立部署Rasterizer与当前Rasterizer|在真实大场景达到质量和墙钟门限，保留原入口|

所有实验使用已有checkpoint，不重新训练。继续沿用既有部署质量口径：每条件平均PSNR损失≤0.01dB，99%视角损失≤0.05dB，最差≤0.1dB；同时记录选点身份、Scale/XYZ长尾、SSIM/LPIPS及非有限值。精确选点候选首先要求ID一致，MLP/渲染算术变化另列误差。

不要在没有实际模块占比的情况下承诺额外百分比。若某模块占完整帧时间10%，单独将其加速2倍，按Amdahl估计整帧仅约1.053倍；这个算例不是对当前源码占比的测量。小ratio优先选点，大ratio优先Compensator与Rasterizer，是当前证据最明确的研究安排。

## 8. 证据归档

- 当前性能与失败数据：../deployment_device_counts_20261107/results/reports/。
- 文献原文、官方源码及页面快照：sources/。
- 每份来源的URL、获取状态、版本/日期与SHA256：SOURCE_MANIFEST.json。
- AIR原论文ACM页面403、部分仓库API限流与arXiv检索超时均有记录；不把抓取失败来源当作已阅读全文。

本报告提出后续研究方案，尚未实现或测得这些方案在mycartgs上的新增收益。
