# 冻结部署推理：精确Radix与小批点CUDA优化报告

日期：2026-11-07。仅适用于训练结束后的独立部署入口，比较对象是上一轮已验收v7。本轮没有训练、没有切换生产默认、没有commit/push，也没有修改常规训练／评价路径。

## 1. 结论

保留三项改动：收缩边界候选的精确Radix；高预算Compensator四点共享层权重；65536点以内连续GPU动态N/K准备。没有引入同帧视锥结果缓存、ratio特征结果缓存、FP16、TF32或网络结构变化。

最终验收的选点、补偿属性和渲染均相对v7逐位检查；验收输入下这些输出没有数值差异。GPU控制数学的末位差异单独报告，不能推导为所有未来输入均有逐位保证。部署专用网络与训练PyTorch之间的既有细小数值差异仍适用。

速度结论分为纯GPU计算和共享设备下的真实墙钟。大地图少一次同步的收益会随其他GPU负载变化，不能将共享负载下的加速比例直接当作独占GPU收益。

## 2. 场景、预算与验证口径

|场景|Master Gaussian|Anchor|相机数|来源|
|---|---:|---:|---:|---|
|TUM fr1_xyz RGB-D|13298|2790|792|已有联合模型，全轨迹估计位姿|
|Replica office3 RGB-D|49102|19936|137|真实训练模型保存的137个相机，并非Replica全轨迹|

Replica没有用复制TUM点构造合成地图。其模型与相机文件全程只读并独立校验SHA256。所有性能条件都按各视角自己的候选N和请求预算计算K，没有固定同K或固定相机Mask。

默认候选仍是当前名义Anchor视锥中的Gaussian，资格Mask、固定Gumbel噪声、K=min(预算,合资格数)、同分优先级及最终原行序不变。ratio预算为floor(ratio×当前候选N)。

## 3. 实际代码改动

|文件|改动|
|---|---|
|python/deployment_radix.cu|新增协作式精确Radix、边界桶压缩、同分提前停止和原序发布；支持Host与Device计数|
|python/deployment_runtime_kernels.cu|新增四点／八点共享层权重Tile；默认四点；低预算保留原单点Block|
|python/deployment_prepare.cpp|GPU动态计数上限从16384扩到65536，管理有界工作区与独立优化开关|
|python/deployment_runtime_bindings.cpp|独立候选入口和调优绑定，保持C++调度|
|python/deployment_runtime.py|独立v11扩展、两个可回退开关，正常训练不导入|
|python/DEPLOYMENT_RUNTIME.md|更新部署用法、数值边界、硬件要求和缓存语义|

### 3.1 精确Radix

将原FP32 score转换成保持排序的32bit Key，正负零归一；资格外Key为0。协作式Kernel先统计有效Key范围并跳过共同前导位，再构建256桶Histogram，只把边界桶Key压缩进下一轮Scratch。边界值全部相同时立即终止。临时Key压缩顺序不稳定，但只用于找精确阈值和所需同分个数；最终两次扫描原候选行，按原序输出K个全局ID。

最多四个8bit细化层。使用分层Warp同值聚合、Block Histogram和有界双缓冲；设备属性和occupancy元数据只查询一次。没有使用近似阈值、没有改用logit差，也没有重生成Gumbel噪声。协作式启动必须支持cooperative launch且Grid不超过可驻留Block数量。

### 3.2 Compensator小批点Tile

一个Block处理四Gaussian，每Gaussian一个Warp。当前层权重放入共享内存，再依次计算64/64主干和四个独立Head。每个输出仍沿原输入维度执行同一串FP32 FMA；Residual继续Base→各活跃Residual顺序相加，保留乘法与加法的原舍入。四Head权重、属性解码及有限值检查均未改变。

原单点Block在很小K时更快，因此预算容量或实际Host K低于1024时仍用原Kernel。Device路径阈值按输出容量判断，实际K由Kernel读取并提前退出；这一门限是本轮工程分流，尚未证明对全部端侧设备最优。

### 3.3 连续GPU控制与存储

冻结地图总容量≤65536时，候选压缩→N/K及ratio控制→Selector→Radix→Compensator连续在当前Stream执行；末尾统一读回96字节状态，建立精确K视图。相机、视锥、ratio和网络结果每帧重新计算。大于65536点时仍保留Host计数路径；这类容量尚未完成真实模型端到端验收。

有界Scratch与地图容量成比例；返回ID及属性有独立存储，下一请求不会覆盖。更换checkpoint或拓扑时重建实例，训练或更新Master后不能继续使用旧冻结实例。同实例串行调用，支持非默认CUDA Stream。Rasterizer自身的Tile容量读回保留，本轮没有宣称整个链条无同步。

## 4. 正确性与异常检查

- 精确选择随机／同分／饱和／正负零／极值／资格／空集：392组通过，最大524288候选。
- 六类checkpoint组件与Tile边界：2232项逐位比较通过。
- 五类checkpoint整体验收：1710组、17175项比较通过，含两种view模式、补偿开关、非默认Stream与输出所有权。
- tum逐视角回归：792相机×18条件=14256组、142560项比较，ID、属性、像素、radii全部逐位相同。
- tum随机控制量：311组，CPU/GPU控制有10组末位差（最大9.54e-07），ID差异0、属性差异0。
- replica逐视角回归：137相机×18条件=2466组、24660项比较，ID、属性、像素、radii全部逐位相同。
- replica随机控制量：311组，CPU/GPU控制有11组末位差（最大9.54e-07），ID差异0、属性差异0。
- 两场景Compute Sanitizer memcheck均为0 errors；Selector与补偿NaN报错、异常后下一请求恢复、空候选、源模型修改隔离、RNG和源模型状态不变均通过。
- 正式checkpoint加载与独立关闭开关：36组通过。>65536保留Host路径另以98204点合成容量检查39组；此项不作为真实大场景速度证据。
- 前后像素逐位相同，所以相同参考图下PSNR/SSIM/LPIPS差异为0；不以“PSNR接近”替代等价证据，不重复计算感知指标。没有保存渲染图片。

## 5. 两轮交错配对墙钟

每轮TUM每条件792请求；Replica每条件411请求，覆盖137保存相机三遍。A/B顺序逐请求交替，显式等待后测准备与准备＋Rasterizer墙钟。表中是两轮各自中位数的平均，不是把两轮原始帧混在一起求中位数。CSV同时保留每轮中位、p90与p99。

|场景／请求|准备v7→v11 ms|准备降低|准备＋渲染v7→v11 ms|总耗时降低|
|---|---:|---:|---:|---:|
|tum／1%|1.760→1.724|2.0%|5.622→5.519|1.8%|
|tum／70%|2.857→2.826|1.1%|8.768→8.706|0.7%|
|tum／变化ratio|2.756→2.716|1.4%|8.552→8.592|-0.5%|
|tum／变化点数|1.729→1.703|1.5%|5.555→5.417|2.5%|
|tum／跳转视角＋变化预算|1.556→1.536|1.3%|4.756→4.768|-0.3%|
|replica／1%|0.867→0.457|47.3%|2.709→2.336|13.8%|
|replica／70%|1.280→0.904|29.3%|3.221→2.964|8.0%|
|replica／变化ratio|1.012→0.650|35.8%|2.825→2.515|11.0%|
|replica／变化点数|0.986→0.611|38.0%|2.744→2.463|10.2%|
|replica／跳转视角＋变化预算|1.007→0.651|35.4%|2.802→2.331|16.8%|

![配对墙钟](DEPLOYMENT_SPEED.png)

这些测量是部署推理准备＋渲染，不包含图像编码、网络传输、UI、在线Tracking/Mapping或训练。服务器持续存在其他训练／评价负载，没有结束无关进程、没有锁频。共享负载会导致数毫秒等待和跨轮变化，绝对值与百分比均仅适用于记录的运行条件；不能据此保证所有端侧设备2ms。

## 6. Nsight：GPU工作量、发射与同步

使用Nsight Systems CUDA/NVTX/OS runtime关联每次prepare发射的Kernel。下表Kernel时间为本工作段累计GPU执行时间；同步次数为prepare自身，不含Rasterizer的读回，也不等于全部CPU等待。每项12个相机样本，统计其中位数。

|场景／ratio／实现|Kernel数|累计GPU μs|准备同步次数|DtoH次数|
|---|---:|---:|---:|---:|
|tum/v7/0.01|14|131.70|1|1|
|tum/optimized/0.01|14|109.52|1|1|
|tum/v7/0.7|14|359.71|1|1|
|tum/optimized/0.7|14|312.58|1|1|
|replica/v7/0.01|17.5|169.54|2|2|
|replica/optimized/0.01|14|143.25|1|1|
|replica/v7/0.7|19|514.47|2|2|
|replica/optimized/0.7|14|488.96|1|1|

具体Kernel、CUDA API等待与逐次统计见results/reports/PROFILE_ANALYSIS.json，原始Nsight报告保留在服务器独立目录。

### 6.1 相同输入组件微测

下表使用固定输入、相同ID、相同ratio控制的CUDA Event批量微测，每项8轮交替顺序。Event可能包含共享GPU排队间隙，与Nsight单Kernel执行时间不同。

|模型／ratio|候选N／K|原Comp μs|四点Tile μs|原选择 μs|收缩Radix μs|
|---|---:|---:|---:|---:|---:|
|both_li／0.01|8466／84|14.54|23.12|40.73|35.64|
|both_li／0.1|8466／846|42.32|37.63|41.11|29.44|
|both_li／0.7|8466／5926|472.63|319.95|39.76|29.88|
|replica／0.01|21756／217|15.74|24.52|65.74|31.36|
|replica／0.1|21756／2175|85.48|80.82|67.15|33.15|
|replica／0.7|21756／15229|1051.83|995.23|85.07|43.21|

### 6.2 独立开关消融

各条件120个变化相机，五种实现轮转执行。v11关闭两个开关应保留v7结果；该组用于归因，不能取其中最好的数字替代两轮主测。

|场景／ratio|v7准备ms|v11关闭优化|仅Radix|仅Tile|两项开启|
|---|---:|---:|---:|---:|---:|
|tum／0.01|2.681|2.682|2.638|2.665|2.668|
|tum／0.1|2.741|2.699|2.678|2.688|2.648|
|tum／0.7|2.893|2.907|2.855|2.822|2.841|
|replica／0.01|5.481|5.358|2.679|5.427|2.703|
|replica／0.1|5.447|5.378|2.759|5.617|2.805|
|replica／0.7|5.948|5.795|3.328|5.739|3.141|

## 7. 额外显存与缓存边界

|场景／实现|冻结实例保留显存MiB|100次变化请求峰值增量MiB|
|---|---:|---:|
|tum／v7|12.673|25.029|
|tum／optimized|12.777|25.132|
|replica／v7|61.014|75.380|
|replica／optimized|62.490|86.716|

计量从原场景已加载后开始，包含独立冻结实例与相关工作区；不是整卡显存。新Scratch按地图容量有界，没有视锥／ratio结果字典，没有跨请求复用网络计算结果，没有CUDA Graph形状缓存。

## 8. 失败与未采用候选

|候选|实测问题|处置|
|---|---|
|每Gaussian原Block完全改为共享Tile|K约84～217时Tile比原Kernel慢约10μs以上|保留低K原实现，1024以上才启用Tile|
|四点Tile双输出交错与部分unroll|相同输入配对中多数条件不如简单层权重共享，寄存器／调度变化未形成稳定收益|回退；源码与配对数据留在stage_tile2与TILE_COMPARISON.json|
|八点Tile作为统一默认|部分高K更快，但中小K更慢，且不同checkpoint不稳定|不设为默认，只保留独立测量入口与可选参数|
|每次Radix调用查询设备属性和occupancy|Host提交开销遮蔽GPU收益|改为每线程／设备缓存不变启动元数据|
|v9大地图仍Host计数|Kernel减少但中途同步仍在，共享GPU墙钟收益较小|v10单独验证后纳入v11的≤65536容量Device路径|

这些候选未改生产配置。前一轮不收缩候选的大Histogram和固定1024Block方案没有重新启用。

## 9. 隔离、身份与使用

服务器独立目录：/home/prmi/zwj/experiments/deployment_radix_tiling_20261107。正常训练、训练评价和Viewer默认后端未切换。关闭本轮优化：

```python
runtime = DeploymentRenderer.from_checkpoint(
    checkpoint, shrinking_radix=False, compensation_tile=0)
```

最终独立二进制：/home/prmi/zwj/experiments/deployment_radix_tiling_20261107/build/mycartgs_deployment_fp32_v11/mycartgs_deployment_fp32_v11.so。SHA256：cb41244904755c25c02a1dd601b84514c83b6982c6e862043a4eb9eca7e59899。基准Git HEAD：26c355b32bf373a75877fc4605a4e41b8de9b47f；本轮无新提交，实际源码身份以BINARY_IDENTITY.json逐文件SHA为准。

- PRODUCTION：604文件，修改0，缺失0。
- INPUTS：42文件，修改0，缺失0。
- LEGACY_INPUTS：28文件，修改0，缺失0。
- LARGE_INPUTS：14文件，修改0，缺失0。

完整前后清单见results/reports/*_BEFORE.json和*_AFTER.json。未修改共享conda、未删除模型或数据、未写生产build/bin/lib/results、未commit/push。

## 10. 剩余瓶颈与下一步

1. 70%准备中，Compensator约占TUM累计GPU时间66%、Replica约77%。当前小批点Tile只减少权重读取，尚未引入FP32 SIMT矩阵Tile或完整共享权重常驻。下一轮应针对实际K、寄存器与SM占用做Nsight Compute，比较按输出小Tile复用权重的实现，继续保持训练入口隔离。
2. 低ratio的准备已减少选择开销，后续收益更可能来自固定地址、有界Workspace与可更新相机／ratio参数的准备CUDA Graph；不能缓存Mask或Feature结果。当前11.8不使用新版条件Graph API。
3. Rasterizer的duplicateWithKeys在70%下分别约200μs（TUM）和476μs（Replica），约占各自渲染累计GPU时间46%和55%，另有Tile排序及动态容量读回。这些未在本轮改动。可按真实footprint做并行Tile展开，但必须验证同一Depth Key、排序稳定性和像素合成顺序。
4. 真实score诊断中存在K达到全部合资格点数的高ratio视角，例如Replica末相机70%下N=7774、K=合资格数5431。下一轮可精确绕过阈值搜索、只做资格稳定压缩；仍需执行Selector有限值检查，不能因预算饱和跳过必要异常检查。
5. >65536真实模型、目标端侧GPU、独占GPU与长时间交互尚未隔离。协作式启动在不同硬件或强竞争下的尾延迟需要单独测量；已有52万点选择单测不能替代这些端到端验收。

## 11. 研究来源

本轮依据上一轮已核实的候选收缩与同分处理思路独立实现，未直接照搬新版DeviceTopK的非稳定输出：

- [RadiK](https://arxiv.org/html/2501.14336v1)：Radix候选访存与层次化统计。
- [NVIDIA AIR/AgentTopK源码](https://github.com/NVIDIA/cccl/blob/main/cub/cub/agent/agent_topk.cuh)：已确定项与下一轮候选分离。
- [Qrita](https://arxiv.org/html/2602.01518v3)：大量重复值与精确边界处理。

论文报告的速度倍数没有用作本项目收益；本报告所有速度数字来自本轮编译产物和独立实验。
