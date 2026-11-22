# 低Ratio、大footprint渲染与Student Tensor Core联合推理：九小时研究报告

本轮授权窗口为香港时间 **2026-11-15 02:44:06—11:44:06**。用户最新重置预算覆盖此前暂停、网络中断和续算记录；本报告不将过去的F2、mode38、182、ThinDispatch或已有Tensor Core收益重复计为本轮新增收益。

## 结论与适用边界

本轮在独立研究目录完成CUDA候选实现、编译、跨场景配对测量、画质、Compute Sanitizer及原文件完整性核验。自然场景验收中，最终近似候选为 **Alpha 3/255＋起点已有182像素核＋轻量纯渲染入口**，联合执行另加入相机与请求一次提交、16 lane/Anchor动态View投影及原生Full入口。网络权重、Student结构和Feature表未改变。

主要Ratio 0.1—0.5的主验收结果：真实Student LoD纯渲染从起点最快Thin182的121.22μs降至113.39μs，配对几何平均加速 **1.0645×**，等价于延迟下降约6.1%；完整当前帧推理从263.72μs降至248.19μs，新增 **1.0613×**。固定属性删点纯渲染新增 **1.0266×**。这些新增收益没有达到“多数场景额外降低约10%”的工作目标，不能描述为又获得了一轮大幅提速。

该近似候选通过自然场景既定画质门限，但**不是对任意输入可靠的通用替代品**。合成Opacity=0.01且大量重叠的压力输入中，3/255高于各点峰值Opacity，会丢掉可累积成显著颜色的贡献；最坏参考图像PSNR为20.84dB、最大RGB误差0.551。保留所有反例，不以自然场景平均画质掩盖该问题。建议把Alpha3作为“当前测试分布内通过门限”的研究选项，保留182原图像算法及其新准备调度入口作为更保守的可回退结果。没有部署或修改生产默认。

减少K能够降低纯渲染延迟，但仍不能使T/T_Full普遍接近K/K_Full。真实LoD的footprint和Alpha交互量远未随K成比例下降；加入选择、补偿和严格预算后，部分低Ratio完整推理仍慢于直接Full。报告同时给出负结果和全部慢例。

## 目录、硬件与保护范围

远端：`/home/prmi/zwj/experiments/low_ratio_footprint_tc_9h_20261112_0011/student_resume_20261114_2250`。

本地：本报告所在目录。冻结F2保持在`render_count_scaling_20261111_2022/background_stage`；可回退的182联合实现为`source_joint_final/build_joint_final`；新实现为`source_joint_release/build_joint_release`，纯渲染为`source_tail_strong/build_tail_strong`。主报告、CSV、源码、实际ELF及hash均可复核。

测试GPU为RTX 3090，CUDA 11.8。Selector采用对应checkpoint原结构与权重，共1762参数；Student为冻结32/32网络，3534参数，保留XYZ、Scale、Rotation、Other四个独立Head。原始Feature表与地图拓扑不变。Selector为FP16 PTX MMA Tensor Core、FP32累加；Student为FP16 WMMA Tensor Core、FP32累加和属性解码。普通像素CUDA或half2指令不能称为Tensor Core。

所有网络前向只使用匹配的Student。读取checkpoint中的地图、Selector、指纹信息不等于运行Teacher；本轮未进行Teacher前向或性能实验、训练、蒸馏、commit、push、部署，也未停止无关进程。

## 测试集合与预先锁定口径

固定属性纯渲染覆盖32组：Replica的office0—office4、room0—room2，各mono/RGB-D共16组；TUM的fr1_desk、fr1_room、fr1_xyz、fr2_xyz、fr3_long_office_household，各mono/RGB-D共10组；VECtor的corner-slow、corridors-dolly、robot-normal，各mono/stereo共6组。地图N为4644—79351。分辨率分别为Replica 1200×680、TUM 640×480、VECtor 1224×1024，未降采样；诊断Tile总数分别3225、1200、4928，与原cameras.json尺寸对应，清单见`FINAL_CAMERA_RESOLUTIONS.csv`。

真实Student及联合推理覆盖18组：Replica全部16组、TUM fr1_desk RGB-D、VECtor corner-slow mono。其余14组没有可匹配的冻结Student，仅报告固定属性纯渲染；不临时改用Teacher或重新蒸馏。各Student路径、SHA256、地图指纹、Selector/Student结构及实际执行后端见`reports/FINAL_STUDENT_MANIFEST.json`和`release_execution_identity_a1/IDENTITIES.json`。

网络开发只用Replica office2/room0 RGB-D；沿用既有开发/保留划分。最终渲染参数于08:17冻结，未根据保留场景逐场景调参。候选3/255、4/255按已写入冻结清单的统一质量和双轮多数收益规则选择；4/255因跨输入质量失败而排除。

每组主验收使用10个真实视角：轨迹43%附近连续6帧，以及7%、31%、71%、97%的跳转视角。新增保留视角为58%附近连续2帧和23%、81%的跳转，均与原10视角不重合。每次重新提交当前相机和输入；无旧图像、旧预测或旧选点缓存。

Ratio为0.01、0.03、0.05、0.1、0.2、0.3、0.4、0.5、0.7，另含固定K=0、1、1000、5000及Full，共14条件。0.03/0.2/0.4等未见Ratio沿用冻结模型实际权重，无重新训练。每次发布严格预算K；局部Ratio分母为当前名义Anchor候选池，不能把ratio=0.1解释为全图N的10%。

两类纯渲染输入分开：A为固定Gaussian属性与视角、固定随机排列的嵌套前缀删点，并按原ID顺序发布；B为实际FP32 Student LoD选择与补偿结果。纯渲染的相机、K、Gaussian身份、属性、背景始终同输入配对；网络后端引起的ID差异只在完整推理中记录。

双轮独立进程逆转场景、条件、视角和版本次序，均衡版本轮转。主纯渲染每条件×视角×版本每轮12个同步墙钟样本；联合为10个。原始样本和p95保留，不以平滑曲线替换统计。报告汇总延迟为条件p50/p95的等权算术平均，加速比为配对条件加速比的几何平均，两者不混用。

同步墙钟、CUDA Event、Nsight Kernel累计分别保留。独立准备测量以GPU相机已就绪为起点；完整路径直接测量当前CPU位姿→准备→Selector→Student→属性发布→RGB完成。不同边界的中位数不能相加当作完整延迟。Profile末尾主动DeviceSynchronize属于测量，不全部算作必要部署同步。

## 资料与实现依据

资料索引见`LITERATURE_20261115.md`。本轮参考[3DGS CUDA优化记录](https://grgkopanas.github.io/3dgs_cuda_blog/index.html)中的像素布局、合成与负结果分析，[Warp-Level Culling](https://plse-splats.github.io/faster-gaussian-splatting/)的实际覆盖域/混合成本视角，以及[NVIDIA CUDA Graph启动成本](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/)和[动态Graph控制流](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/)资料。外部论文/较新Toolkit的收益不作为本3090、CUDA11.8的实测证据，当前环境不支持的条件节点未引入。已有子块筛除等负结果先读历史证据，未重复盲目实现。

## 实际保留的实现

|改动|作用|正确性/风险|本轮收益归因|
|---|---|---|---|
|Pixel Alpha阈值从1/255提高到3/255，并同步改变Tile成员贡献域|减少大footprint尾部Tile及Alpha交互；K、ID、属性、radii保持相同|近似；自然场景达门限，极端弱Opacity重叠失败|真实LoD相对相同Thin入口新增1.0645×，固定属性1.0266×|
|CPU相机与请求合并为一次有界HtoD提交|省去旧相机/请求重复装载与一个uploadRequest Kernel，当前帧仍重算|同后端ID/RGB/radii逐位一致，finite检查保留|独立关闭后主要Ratio平均多4.48μs，约1.0185×|
|16 lane/Anchor动态View投影|两个Anchor共享一个warp，保持每输出列加法和FMA顺序|同后端逐位一致，不变更Feature或权重|主验收整体约0.2%而非开发集约2%；71/180条件轮次更慢，增益很小|
|原生Full CPU相机入口，直接只读消费冻结Full属性|省去Full每帧不必要的属性clone、Python切片；图像与返回IDs仍独立|当前位姿每帧提交，非旧图缓存；Full输出逐位一致|独立关闭后Full平均多58.46μs，1.2365×；LoD本身不走此分支|
|轻量ThinDispatch纯渲染入口|复用空Tensor、内参标量及绑定方法|与对应普通入口RGB/radii逐位一致|这是起点已有收益；本轮只复用并与最快Thin182公平配对|

旧182像素核、快速bit-exp、相邻像素映射、严格选择、静态线性预投影、GPU N/K控制、已有Graph和Student TC属于起点既有实现。新增3/255不改变选点，只改变参与渲染的低贡献工作；必须与精确调度优化分开表述。

`new_prepare_182`保留旧182图像算法，仅加入新准备调度和Full入口；主要Ratio248.19μs的Alpha3最终路径与该保守路径256.52μs分别保留。后者相对I_previous新增1.0281×，相对A为1.5397×；不承担本轮提高Alpha阈值的额外压力反例。对18组×14条件×原10及新增4视角另做3528次直接比较，ID、RGB、radii全部与I_previous逐位一致，证据见`reports/final_conservative_parity/`。这里的逐位参考为已有182，不是逐位F2。

## 纯Rasterizer与完整延迟


|输入|路径|p50汇总 μs|p95汇总 μs|相对F2|相对起点Thin182|慢于Thin182的条件轮次|
|---|---|---|---|---|---|---|
|nested|F2|121.17|132.61|1.0000×|0.8277×|320|
|nested|thin182|99.63|108.13|1.2081×|1.0000×|0|
|nested|new_thin|96.79|103.93|1.2402×|1.0266×|2|
|student_lod|F2|171.32|194.36|1.0000×|0.7092×|180|
|student_lod|thin182|121.22|140.54|1.4100×|1.0000×|0|
|student_lod|new_thin|113.39|127.41|1.5009×|1.0645×|0|


主要Ratio：nested为32×5×2=320个条件轮次，Student LoD为18×5×2=180个。两个nested慢例为Replica office4 RGB-D的0.1第二轮（约0.12%）与TUM fr1_room mono的0.2第二轮（约0.06%），均保留；真实LoD主要Ratio无慢例。空集/单点和小K固定属性下有更多亚微秒波动，详见全部慢例CSV，不宣称所有输入严格更快。


|完整路径|p50汇总 μs|p95汇总 μs|相对A FP32+F2|相对C TC+F2|相对起点联合I|
|---|---|---|---|---|---|
|A_fp32_F2|395.33|431.76|1.0000×|0.8559×|0.6678×|
|B_fp32_new|347.79|378.21|1.1334×|0.9701×|0.7569×|
|C_native_F2|337.96|367.77|1.1684×|1.0000×|0.7802×|
|I_previous|263.72|291.34|1.4975×|1.2818×|1.0000×|
|new_prepare_182|256.52|281.85|1.5397×|1.3178×|1.0281×|
|D_final|248.19|268.52|1.5893×|1.3603×|1.0613×|


B为FP32网络加新渲染器、C为既有TC及准备加原F2、D为最终联合。A→D的1.5893×包含历史收益；本轮新增只能报I→D的1.0613×。I→new_prepare_182为1.0281×，保持同182渲染算法；new_prepare_182→D为1.0322×，对应本轮阈值变化在完整路径中的收益。不能把分离的纯渲染6.4%直接当完整推理收益。

按Ratio汇总如下。K为所有主验收视角和轮次的实际输入K中位数；延迟仍为各条件p50/p95等权汇总。因此K与延迟是各自统计，不代表同一个“中位帧”。


|条件|实际K中位|纯渲染p50/p95 μs|相对Thin182|完整p50/p95 μs|相对I|
|---|---|---|---|---|---|
|local0.01|91.0|70.98/75.38|1.0264×|198.12/209.34|1.0599×|
|local0.05|455.5|84.80/91.27|1.0481×|211.29/224.88|1.0616×|
|local0.1|911.5|94.02/102.26|1.0564×|222.69/233.96|1.0597×|
|local0.2|1823.0|105.19/115.14|1.0600×|242.03/258.66|1.0592×|
|local0.3|2734.5|115.57/127.10|1.0624×|249.27/268.29|1.0596×|
|local0.4|3647.0|123.10/142.07|1.0660×|260.05/286.48|1.0616×|
|local0.5|4558.5|129.09/150.48|1.0777×|266.91/295.21|1.0663×|
|local0.7|6286.5|137.38/161.06|1.0867×|274.22/310.05|1.0674×|
|full|30031.0|198.54/218.37|1.0750×|262.62/287.52|1.3003×|


CUDA Event独立结果：真实Student LoD的F2、Thin182、新Thin分别为152.98、102.20、92.80μs（主Ratio条件Event p50等权均值）；完整A、C、I、new_prepare_182、D分别为370.65、311.02、239.61、232.22、222.94μs。Event范围仍可包含Host提交形成的GPU空隙，不能冒充Kernel累计。

Full主验收：32组nested双轮64项、18组Student Full双轮36项，共100个条件轮次，新Thin均快于F2及起点最快Thin182。对Thin182的最差加速分别1.0184×、1.0348×；几何平均分别1.1066×、1.0750×。联合Full 36项也全部快于I，最差1.1948×、几何平均1.3003×。所有Full p95、两轮波动与慢例文件原样保留，不只看平均。

## 准备、网络与Profile归因

给定GPU相机的独立准备墙钟主要Ratio汇总：A 183.24/197.91μs、C 153.76/164.82μs、I 148.59/160.40μs、D 148.09/160.22μs（p50/p95）。D相对I仅约0.3%：主要新增收益来自CPU相机的完整提交路径及渲染，不应将它说成Student矩阵核又大幅加速。

代表性Nsight使用office2/room0 RGB-D、TUM fr1_desk RGB-D、VECtor corner-slow mono，在1%、10%、30%、50%、Full及移动/跳转视角采样。Profile四组代表的主要Ratio平均Kernel累计模块如下，仅作归因，不替代上文非Profile测速：

|模块|A独立准备 μs|C独立准备 μs|D独立准备 μs|D完整路径 μs|
|---|---:|---:|---:|---:|
|候选、View及控制准备|28.91|29.03|11.30|14.43|
|Student动态输入投影|并入原Comp核|并入旧Comp核|11.35|11.39|
|Selector|12.95|10.01|9.21|9.14|
|严格预算选择|35.62|35.58|30.11|29.98|
|Student主干、Head、属性发布|46.55|17.29|15.57|15.81|
|纯渲染相关节点|不在该scope|不在该scope|不在该scope|约93.37|

最终完整路径的纯渲染相关节点包括预处理、排序、位表清零、成员构建/压缩和像素合成；像素合成约46.37μs是其最大单模块，严格选择约29.98μs仍是网络之外的大项。相机合并提交减少一个uploadRequest节点；代表0.1输入D总节点17—19个，I为18—20个，C为19—21个。D每帧仍有准备与渲染两个Graph、一次必要Stream同步以完成finite/实际K检查及发布返回；Profile额外一次Device同步属于测量尾部。没有通过延后检查或返回旧结果加速。

Selector与Student继续使用已有MMA/WMMA，准备中的不规则候选、索引、View编码、严格选择和位表仍用普通CUDA；相机变换由CPU C++执行。没有将维度很小、低算术强度的准备工作强行搬到Tensor Core。

## 画质、边界与失败压力输入

主验收最终联合：全部14条件等权平均PSNR损失0.04617dB，仅主要Ratio平均0.06482dB，最差场景×条件均值0.23759dB，最差帧0.49971dB；平均SSIM损失约0.000602，LPIPS平均改善约0.005478。2520帧中PSNR损失p95为0.19985dB，最差单帧SSIM损失0.01134、LPIPS增加0.001931。改善不代表逐位相同；该路径明确为受控近似。

纯渲染主验收：nested平均PSNR损失0.05316dB、真实Student LoD为0.04592dB，最差条件0.34001dB。相应平均SSIM损失约0.003176/0.000601，LPIPS变化约−0.004407/−0.005478。质量均相对同输入FP32 Student＋F2；所有逐帧PSNR/SSIM/LPIPS保留。

4/255的Student LoD单独看可达门限，但nested的room0 RGB-D、Ratio0.7条件损失0.63250dB，room0 mono为0.54271dB，故全局排除。6/255、8/255在开发质量阶段失败，未进入最终候选。不能按场景选择不同阈值掩盖失败。

TC相对FP32的平均ID重合率99.9916%，最差视角99.1379%；严格K相同。这属于网络数值后端差异。相机/请求/分组View等调度开关之间ID、RGB、radii逐位一致；最终D也与同TC后端分离准备＋相同渲染器逐位一致。radii对同输入F2全程逐位相同，RGB对F2不逐位。

结构性边界覆盖空集、单点、31/32/33、255/256/257、511/512/513、1023/1024/1025、4097点、同深度稳定顺序、非整Tile分辨率、细长椭圆、大footprint、非有限输入、成员溢出、容量拒绝、不同Stream、Workspace复用与输出独立性。纯渲染和联合新核各完成memcheck/initcheck/synccheck，全部0错误。安全fallback保持成员顺序；跨Stream复用被明确拒绝。

合成压力另测97×65与641×479、K=0/1/16/128/512/1000/2500/5000，六种样式。该输入没有GT，其参考图PSNR只衡量与F2图像距离，不能冒充真实场景GT门限：

- Opacity=0.01的重叠输入，原182在5000点时与F2逐位相同；新3/255因阈值高于峰值直接排除大量点，参考图PSNR20.84dB、最大RGB误差0.551。这是本轮新增近似的真实失败。
- 此时出现最高约26×加速，来自移除了绝大部分合成贡献，不是纯运算效率提升，不能计入有效场景速度收益。
- 大footprint/随机等压力结果也记录了近似误差；所有radii和有限性检查通过不等于画质通过。

因此，**没有取得对任意大footprint/重叠输入都可靠的新Alpha阈值方案**。Alpha3仅在已验自然场景分布内符合用户数值门限；通用稳健推荐保留182及新准备调度，生产未切换。下一阶段应研究累计被省略贡献的保守界或显式误差预算，避免逐点“小贡献”累积成大误差；不能仅按K或Ratio绕开新路径。

## K—耗时缩放与反序

固定属性主验收：新Thin对每视角的T≈b+aK拟合，含Full的中位R²=0.9963，b≈63.29μs、a≈4.37μs/千点；起点Thin182为b≈62.28μs、a≈4.89μs/千点。主要改善每点成本，固定底噪几乎没有继续下降。归一化曲线因Full加速更多而略升高，不能把它说成绝对速度退化。

真实Student LoD：仅Ratio内部拟合中位R²从Thin182的0.9122升至0.9227；加入未经Student补偿的Full后，新Thin中位R²降至0.8578。原F2含Full为0.7482。该差异说明真实LoD改变footprint和遮挡，不能用固定属性实验的线性结论替代。

计时外诊断中，Alpha3主要Ratio的有效像素—Gaussian累加次数中位下降约21%，Tile实例下降约18—19%。但真实LoD的1%条件仍有约790万Alpha交互，Full约1592万，而K为91对30031；K减少约99.7%并没有消除固定像素输出及大footprint工作。`alpha_interactions`只统计weight>0的累加，不包括被测试后拒绝的候选；`alpha_participating`按唯一Gaussian ID去重，两者不混用。

最终纯渲染在1%、10%、50%时的T/T_Full中位数分别约0.393、0.504、0.652，对应实际K/K_Full约0.00324、0.03269、0.16350。没有接近斜率1的过原点直线。固定属性相应1%、10%、50%归一化耗时约0.365、0.436、0.696，仍有固定成本。

完整D的1%、10%、50% T/T_Full约0.794、0.883、1.031；50%条件有221/360个帧轮次慢于各自Full，0.1也有35/360。原因是Full直接渲染冻结Master，LoD额外承担候选、Selector、严格预算、Student及属性发布，而低K的footprint并未同比缩小。完整延迟未普遍随K单调下降。

全部反序均已导出。新Thin的真实LoD在仅Ratio比较中有3/360条视角序列出现超过2%反序；nested为11/640。完整D仅Ratio内部96/360条序列有超过2%反序，含Full则287/360。不能将这些反例解释为不存在或人为拉直曲线。

## 新视角复验


|新视角输入|p50汇总 μs|p95汇总 μs|相对各自起点最快路径|
|---|---|---|---|
|nested|99.39|105.00|1.0316×|
|student_lod|113.76|124.03|1.0626×|
|joint|248.02|264.75|1.0621×|


新增视角全部复验完成，仍使用冻结参数。联合平均PSNR损失0.03868dB，最差条件0.28663dB，最差帧0.53329dB；nested平均0.04692dB。两类均符合预定平均/条件门限，最差帧超过0.50dB单独披露，未把条件均值门限改成帧门限。新视角所有100项Full纯渲染和36项Full联合仍快于起点最快路径。nested主要Ratio有16/320个相对Thin182慢例，真实LoD与联合主要Ratio均无条件轮次慢例；全部保留CSV。

## 显存、预计算与生命周期

Student/Selector静态投影属于已有空间换时间设计：线性层按静态物理输入、Offset/Base Feature、各级Residual及动态Ratio/View分解，只在ReLU之前组合，不跨非线性重排。静态缓存依赖冻结地图、权重、Feature及拓扑；依赖变更必须重建部署实例。Ratio/View每帧重算。初始化使用批量矩阵投影，运行时不重复Gather完整48D再拼接。

Student静态投影约为128N字节（Offset+Base）＋128LA字节（L级Residual、A个Anchor）＋128A字节（当前动态投影工作区）；Selector静态首层为128N字节。实际缓存逐场景记录，不用该近似公式替代测量。当前地图指纹和Student来源逐个校验，原Feature表未改为half。

单实例部署测量（源地图已驻留）D常驻增量17.68—198.79MiB，峰值32.92—209.08MiB；Rasterizer Workspace约8.05—53.71MiB，预投影约1.70—29.31MiB。相对I没有新增大规模缓存，主要新增为有界请求包及Full帧控制。多对照同时驻留的显存未当作部署需求。初始化/首次预热19.78—135.66ms，不含源码编译、地图读取和初建CUDA上下文。

Rasterizer共享池总上限1GiB、位表256MiB，Graph计划最多512；超容量明确报错。池扩容或分辨率改变销毁旧地址计划。准备Graph使用有限容量桶，N/K实际计数和发布控制留GPU，必要时仍读回状态以检查finite和严格K。执行实例绑定device/Stream；同Stream按帧串行，跨Stream应使用独立实例和显式依赖。每次图像、radii、返回ID独立，后续帧不得覆盖已返回结果。

多Stream异步控制读回候选虽逐位通过，但没有稳定单帧收益，未采用。未把流水线吞吐冒充交互单帧加速。

## 未保留候选与原因

|候选|实测/失败原因|处理|
|---|---|---|
|Alpha 2/255|自然场景质量更保守，但新增收益低于3/255|保留Pareto与源码|
|Alpha 4/255|真实LoD更快，但nested最差条件损失0.6325dB|拒绝作为最终候选|
|Alpha 6/255、8/255|开发阶段画质失败|未进入最终验收|
|T早停0.015/0.02|无稳定收益|保留失败摘要；0.03/0.05质量失败|
|相邻像素共享Alpha|开发质量通过，但完整延迟慢约3—6%|撤回|
|提升后的快速exp2|完整收益约±0.5%；首次CUDA11.8接口编译失败后修为PTX复验|不保留，失败日志保留|
|直接half2 bit-exp编码|慢约19—20%，不能仅凭指令更少认定更快|撤回|
|CUB稳定深度排序|边界通过但慢约11—13%|撤回|
|动态K协作排序|逐位通过，慢约0.5—2.5%|撤回|
|Radix CTA4/8、16/32|小CTA慢7—15%；32基本持平|不保留|
|溢出列表前缀|局部room0 Full开发获益约14—15%，跨场景整体约1.0006×|不泛化，不保留|
|按Tile清零|约1%弱收益，未形成稳定组合收益|不单独采用|
|冻结预投影FP16存储|ID一致，RGB/radii改变，速度不稳定|撤回|
|每Anchor动态Selector首层MMA|省计算但新增Kernel/访存，完整慢约2%|撤回|
|放大准备Graph容量上限|空CTA和额外容量工作抵消Host省时|不保留|
|异步控制DtoH并发|逐位通过，单帧完整收益不稳定|不采用|

## 执行时间、外部干扰与完整性

02:44:06按用户最新授权重新计时；04:08后工具连接失效，08:14恢复，后台任务继续运行。其间计入墙钟，不能称为持续交互研究。08:17冻结渲染候选，08:45进入最终联合验收，早于第六小时冻结要求；原第4小时检查时因交互中断未及时报告，作为执行偏差记录，不补造历史进度。

08:45—09:37为主联合验收；09:02附近连续外部活动使控制器退出，09:30核实PID/start_ticks/boot_id及文件锁后只恢复剩余43项，保留72项已完成结果。09:38启动新视角验证；另一段交互工具不可用期间控制器因Xorg间歇活动停止，10:23核对现场后仅补齐5项，10:24完成。所有无效轮次保留，未杀Xorg或外部Python。主验收171项有效任务、44项干扰拒绝；新增视角94项有效、18项拒绝。额外重跑用于真实干扰复测，没有降低标准或挑最快轮次。每秒监测共3676条，认可测速任务的GPU忙样本温度53—83°C、SM时钟1282—1980MHz；该范围包括装载、预热和退出，不代表恒定时钟。未锁频；双轮均衡配对和全部监测样本保留，微小亚微秒收益不应外推到其他GPU或频率。

10:29完成末次冻结和保护核验：179项输入文件、53项原研究保护文件、248项生产/冻结源码/共享conda-meta文件，共480项manifest记录均无差异，18个Student SHA256匹配；检查范围严格限于清单，未宣称对整个共享环境作了全盘审计。最终源码、实际加载ELF及复用链接对象89项hash与冻结记录相同。没有保存渲染图片，性能曲线位于figures目录。

仅在本轮独立目录内删除已核绝对路径、UID和归属的失败候选`.o`及已导出的SQLite：43项、94.10MiB。清单含原hash见`FINAL_CLEANUP_ALLOWLIST.json`；保留原基线、182完整回退实现、最终ELF、源码、原始逐样本CSV/压缩CSV、关键Profile rep及分析。原模型、数据集、共享缓存和其他任务目录均未清理。

报告最终生成时间：2026-11-15T10:42:46.984543+08:00。研究在九小时上限内完成验收，不为用满预算重复无信息测试；实际封存时间以`FINAL_SESSION_CLOSED.json`为准。

## 逐场景结果

下表主要Ratio均为0.1—0.5。纯渲染为新Thin；有Student时列真实LoD，否则仅固定属性。完整列只对已有匹配Student报告。Full最差加速为两轮中最慢的一轮相对各自起点最快版本；PSNR损失是全部14条件等权平均。完整逐Ratio、逐帧与全部失败项见CSV，表中不取最好一次。


|场景/模式|N|输入|纯渲染μs|纯新增|Full最差|完整μs|完整新增|PSNR损失dB|
|---|---|---|---|---|---|---|---|---|
|Replica/office0/mono|20332|LoD|96.9|1.059×|1.056×|220.5|1.059×|0.0272|
|Replica/office0/rgbd|33431|LoD|104.2|1.054×|1.064×|232.1|1.066×|0.0304|
|Replica/office1/mono|16417|LoD|91.9|1.068×|1.067×|215.5|1.065×|0.0303|
|Replica/office1/rgbd|31503|LoD|98.8|1.053×|1.058×|229.2|1.058×|0.0595|
|Replica/office2/mono|24323|LoD|93.9|1.055×|1.057×|220.4|1.049×|0.0588|
|Replica/office2/rgbd|43197|LoD|101.7|1.064×|1.049×|238.9|1.065×|0.0290|
|Replica/office3/mono|21300|LoD|103.3|1.049×|1.051×|225.7|1.050×|-0.0030|
|Replica/office3/rgbd|49102|LoD|126.5|1.056×|1.044×|272.3|1.054×|0.0537|
|Replica/office4/mono|20861|LoD|88.1|1.059×|1.062×|212.1|1.057×|0.0222|
|Replica/office4/rgbd|51495|LoD|97.2|1.059×|1.051×|237.2|1.057×|0.0293|
|Replica/room0/mono|40282|LoD|147.8|1.074×|1.060×|289.2|1.066×|0.0812|
|Replica/room0/rgbd|79351|LoD|186.0|1.111×|1.394×|370.9|1.087×|0.0710|
|Replica/room1/mono|28559|LoD|120.3|1.073×|1.058×|251.6|1.069×|0.0389|
|Replica/room1/rgbd|54226|LoD|137.7|1.063×|1.056×|293.9|1.061×|0.0618|
|Replica/room2/mono|22807|LoD|110.9|1.071×|1.062×|237.1|1.058×|0.0720|
|Replica/room2/rgbd|53266|LoD|128.8|1.065×|1.051×|281.0|1.056×|0.0508|
|TUM_RGBD/freiburg1_desk/mono|18445|nested|81.6|1.022×|1.149×|缺少Student|—|0.0411|
|TUM_RGBD/freiburg1_desk/rgbd|20601|LoD|78.7|1.036×|1.035×|201.4|1.048×|0.0901|
|TUM_RGBD/freiburg1_room/mono|27577|nested|75.0|1.007×|1.018×|缺少Student|—|0.0193|
|TUM_RGBD/freiburg1_room/rgbd|72211|nested|99.1|1.014×|1.027×|缺少Student|—|0.0386|
|TUM_RGBD/freiburg1_xyz/mono|13630|nested|75.1|1.021×|1.039×|缺少Student|—|0.0496|
|TUM_RGBD/freiburg1_xyz/rgbd|13298|nested|74.7|1.021×|1.045×|缺少Student|—|0.0394|
|TUM_RGBD/freiburg2_xyz/mono|19910|nested|88.6|1.027×|1.868×|缺少Student|—|0.0536|
|TUM_RGBD/freiburg2_xyz/rgbd|15634|nested|84.8|1.025×|1.788×|缺少Student|—|0.0651|
|TUM_RGBD/freiburg3_long_office_household/mono|14431|nested|74.0|1.023×|1.047×|缺少Student|—|0.0539|
|TUM_RGBD/freiburg3_long_office_household/rgbd|42700|nested|92.3|1.023×|1.031×|缺少Student|—|0.0549|
|VECtor/corner-slow/mono|4644|LoD|128.5|1.094×|1.095×|238.6|1.078×|0.0234|
|VECtor/corner-slow/stereo|5716|nested|96.9|1.047×|1.088×|缺少Student|—|0.0607|
|VECtor/corridors-dolly/mono|15818|nested|95.6|1.033×|1.071×|缺少Student|—|0.0369|
|VECtor/corridors-dolly/stereo|26640|nested|104.6|1.026×|1.044×|缺少Student|—|0.0591|
|VECtor/robot-normal/mono|8189|nested|100.4|1.045×|1.083×|缺少Student|—|0.0649|
|VECtor/robot-normal/stereo|6774|nested|96.9|1.047×|1.079×|缺少Student|—|0.0365|


## 交付文件与复现

- `reports/RELEASE_THIN_SCENE_RATIO.csv`：32组nested＋18组LoD，逐场景/Ratio/轮次纯渲染p50/p95，与F2、普通182和Thin182配对。
- `reports/RELEASE_JOINT_SCENE_RATIO.csv`、`RELEASE_PREPARE_SCENE_RATIO.csv`：联合与独立准备，各18组、双轮。
- `reports/RELEASE_THIN_SAMPLES.csv.gz`、`RELEASE_JOINT_SAMPLES.csv.gz`、`RELEASE_PREPARE_SAMPLES.csv.gz`：原始有效逐次样本；Event独立CSV。
- `reports/FINAL_RASTER_ALL_SLOW_CASES.csv`、`FINAL_JOINT_FRAME_SLOW_CASES.csv.gz`、`FINAL_RASTER_ALL_INVERSIONS.csv`、`FINAL_JOINT_ALL_INVERSIONS.csv`：所有慢例和反序。
- `reports/STRONG_QUALITY_CONDITIONS.csv`、`RELEASE_JOINT_QUALITY.csv`、`UNSEEN_*`：逐场景条件画质及新增视角原始证据。
- `reports/FINAL_WORKLOAD_TIMING.csv.gz`：K、radii可见数、唯一Alpha参与点、像素交互、Tile实例、活跃Tile、成员分布、溢出及对应计时；诊断不在计时内。
- `reports/FINAL_PROFILE_COMPONENTS.csv`、`FINAL_PROFILE_CALLS.csv`、`FINAL_PROFILE_APIS.csv`：Kernel累计、节点、Graph/同步/API归因；三个代表`.nsys-rep`保留。SQLite可从rep重新导出。
- `reports/FINAL_STUDENT_MANIFEST.json`、`FINAL_DELIVERY_AUDIT.json`、`FINAL_PROTECTED_INTEGRITY.json`：模型/执行身份及保护核验。
- `reports/FINAL_ACCEPTANCE_DECISION.json`：明确区分自然场景门限通过、极端重叠失败、保守推荐和未实现的通用线性/新增10%目标，未修改门限。
- `reports/FINAL_JOINT_INCREMENTAL.patch`、最终源码和加载ELF：新旧版本差异及完整回退版本。
- `figures/release_20261115/`：32张纯渲染逐场景图、18张联合逐场景图、归一化、热图与Pareto，共54组PNG/SVG；450dpi，原数据无平滑，带CSV/hash清单。

远端环境使用`/home/prmi/anaconda3/envs/zwj/bin/python`，不可在共享conda安装或替换依赖。复现先加载`joint_release_runtime.JointReleaseRenderer`或`release_thin_control.ThinReleaseRenderer`，使用`FINAL_STUDENT_MANIFEST.json`的原路径/指纹，`torch.no_grad()`，每个Stream独立实例。mode301为自然场景门限内的Alpha3近似；mode182为保留旧渲染算法的准备调度版本。先预热有界容量计划，再使用当前帧CPU位姿和实际Ratio/预算请求；预热时间单独记录。

`summarize_release_joint.py`、`summarize_release_thin.py`、`summarize_unseen_validation.py`、`analyze_release_scaling.py`、`analyze_release_workloads.py`、`summarize_profile_components.py`和`plot_release_results.py`可从保留证据重建统计与曲线。控制器合并文件只引用已核验的原attempt，原失败控制器与日志未覆盖。

## 剩余优化空间

优先级最高的是**有误差预算的累计贡献裁剪**：Alpha3压力反例说明逐Gaussian贡献低不能推出总误差小。需要以实际Tile/像素的Opacity总和、透射率及被跳过贡献上界控制误差，保持通用调度，不能按场景、K或Ratio硬编码。

其次是严格选择约30μs的设备端固定成本、准备与渲染Graph边界、约60—80μs纯渲染固定底噪，以及大footprint每像素的依赖链。当前Student主干已很小，给定GPU相机的准备新增仅0.3%，继续压矩阵核不一定转化成完整延迟。

最后是低Ratio完整推理对Full的成本倒挂。Full省掉选择和补偿，且本轮Full入口获益更多；继续研究必须直接测完整当前帧路径，不能仅靠纯Rasterizer更快来宣称LoD系统整体线性。当前结论是可重复的自然场景小幅新增收益，加上明确的近似边界和仍未实现的通用缩放目标。
