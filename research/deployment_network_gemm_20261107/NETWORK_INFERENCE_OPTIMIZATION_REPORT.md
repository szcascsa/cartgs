# Selector与Compensator部署推理优化报告

日期：2026-11-07。最终实现v13，对照为已验收v11。根据用户追加要求，本轮没有修改Gaussian渲染；所有加速结果只统计Selector、Compensator或网络准备管线，Rasterizer不进入计时区。渲染只用于正确性检查。

## 1. 结论与范围

保留两项优化：Selector的16点寄存器Tile，以及高预算Compensator的16点寄存器Tile。二者均让一次权重读取供四个Gaussian的独立FMA链复用；每个输出仍按v11的输入维度顺序累加，没有分段归约、半精度、TF32、参数变化或训练变化。

没有改变名义Anchor候选、资格Mask、严格K、固定Gumbel噪声、同分优先级、最终点序、ratio编码、Residual累加、四个独立Head或属性变换。Radix、Gaussian→Tile、排序和像素合成源码均保持v11版本。

v13在本轮验收输入下相对v11的选点、属性与像素逐位一致。该结论不扩大为所有任意浮点输入的形式证明，也不扩大为部署专用网络与正常训练PyTorch版本逐位相同；后者的既有细小差异仍按之前报告理解。

## 2. 两网络实际改动

|模块|v11|v13|保持的语义|
|---|---|---|---|
|Selector|每Gaussian一个Warp，Block处理4点|Block处理16点；线程拥有一个神经元的四点累加器，权重只读取一次供四点使用|11D/19D输入、32/32隐藏层、两logit、FP32 FMA序、噪声及score计算|
|高预算Compensator|Block处理4点，逐层把权重装入共享内存|Block处理16点，线程将同一权重复用于四个独立点，减少权重读取和逐层装载|51D物理／Residual输入＋3D ratio投影、64/64主干、四Head、Scale／Rotation解码|
|低预算Compensator|每Gaussian一个Block、四Warp并行Head|保留原实现|低预算不强行使用较慢Tile|

Selector在16点Tile内使用四个独立累加器，最后仍分别计算每点的两logit。Compensator沿Base→活跃Residual顺序进行乘法和加法；主干后依次计算四个Head，每Head仍使用原权重及偏置，不是重新训练一个合并网络。

Compensator保留1024门限：Device路径按输出容量、Host路径按实际K分流。低于门限仍走v11单点Block。相机每次改变都会重新计算，不缓存视锥Mask、选点、ratio Feature或中间网络输出。

## 3. Nsight独立网络测量

同一checkpoint、相同输入与选点；每条件8个保存相机样本。下表为各调用范围内累计GPU Kernel时间的中位数。独立包装函数的状态清零Kernel计入对应网络；不含Host等待，也不含视锥、Top-K和Rasterizer。

|场景／ratio|模块|v11 μs|v13 μs|速度倍数|
|---|---|---:|---:|---:|
|both_li／0.01|selector|26.46|11.79|2.24×|
|both_li／0.1|selector|27.46|12.38|2.22×|
|both_li／0.7|selector|27.73|12.18|2.28×|
|replica／0.01|selector|45.42|22.74|2.00×|
|replica／0.1|selector|45.70|22.16|2.06×|
|replica／0.7|selector|45.86|20.43|2.24×|
|both_li／0.01|comp|20.64|20.21|1.02×|
|both_li／0.1|comp|47.84|48.30|0.99×|
|both_li／0.7|comp|204.67|126.69|1.62×|
|replica／0.01|comp|27.25|22.66|1.20×|
|replica／0.1|comp|82.67|50.30|1.64×|
|replica／0.7|comp|403.38|211.76|1.90×|

![独立网络性能](NETWORK_COMPONENT_SPEED.png)

## 4. 真实变化视角下完整准备耗时

完整准备包括当前相机的候选构建、动态N/K、视角编码、Selector、稳定精确选点、Compensator及末尾状态读回。计时函数只调用prepare，不调用Renderer。TUM每轮每条件792请求；Replica每轮每条件411请求，覆盖137个保存相机三遍。A/B顺序交替，表中为两轮中位数的平均。

|场景／请求|N范围|K范围|v11 ms|v13 ms|耗时降低|
|---|---:|---:|---:|---:|---:|
|tum／1%|4467–10822|44–108|0.1929|0.1813|6.0%|
|tum／10%|4467–10822|446–1082|0.2258|0.2157|4.5%|
|tum／70%|4467–10822|3126–7575|0.3760|0.3000|20.2%|
|tum／变化ratio|4467–10822|46–7571|0.2715|0.2359|13.1%|
|tum／变化点数|4467–10822|1000–8167|0.3171|0.2416|23.8%|
|tum／跳转视角＋变化预算|4467–10822|47–8254|0.3147|0.2445|22.3%|
|replica／1%|7774–27760|77–277|0.2325|0.2067|11.1%|
|replica／10%|7774–27760|777–2776|0.2791|0.2424|13.1%|
|replica／70%|7774–27760|5431–19432|0.6190|0.4254|31.3%|
|replica／变化ratio|7774–27760|104–19432|0.3602|0.3030|15.9%|
|replica／变化点数|7774–27760|1000–10000|0.3503|0.2602|25.7%|
|replica／跳转视角＋变化预算|7774–27760|138–16667|0.3850|0.3155|18.1%|

![完整准备性能](NETWORK_PREPARE_SPEED.png)

速度来自RTX3090，期间存在其他GPU进程，没有终止无关进程或锁频。每轮中位、p90、p99与逐请求原值保存在CSV／JSON。不是在线SLAM训练耗时，也不包含Rasterizer、图像编码、显示或网络传输；尚不能代表其他端侧设备。

## 5. 完整准备的GPU归因

|场景／ratio／实现|Kernel发射数|累计GPU μs|准备同步数|
|---|---:|---:|---:|
|tum/v11/0.01|14|111.30|1|
|tum/optimized/0.01|14|101.44|1|
|tum/v11/0.7|14|311.35|1|
|tum/optimized/0.7|14|225.39|1|
|replica/v11/0.01|14|140.51|1|
|replica/optimized/0.01|14|115.94|1|
|replica/v11/0.7|14|485.01|1|
|replica/optimized/0.7|14|342.38|1|

网络本来已各融合为单Kernel，本轮主要减少重复权重读取、改善独立FMA调度与资源占用，所以不应把Kernel数不变理解为没有优化。准备同步次数和GPU动态N/K路径沿用v11。

编译资源统计：高预算Compensator由v11四点Tile的92寄存器／线程、20496字节共享内存／Block，变为56寄存器、16704字节共享内存；Block线程数从128到256，每Block点数从4到16。Selector新19D Tile为48寄存器、5440字节共享内存，原为46寄存器、1024字节；这些是编译资源数据，不是独立的加速证明。

Nsight Systems两份采集均完成。另尝试Nsight Compute的LaunchStats／Occupancy，但驱动返回ERR_NVGPUCTRPERM，未获得硬件计数器数据；没有修改驱动权限、锁定时钟或清空其他任务的缓存。NCU_STATUS.json和logs/ncu.log保留该限制，不能把编译资源数据当作实测占用率。

## 6. 独立开关消融

各条件120个变化相机，五种实现轮转执行，只测prepare。off使用selector_tiled=False、compensation_tile=4，恢复v11网络执行方式。

|场景／ratio|v11 ms|off|仅Selector|仅Compensator|两项开启|
|---|---:|---:|---:|---:|---:|
|tum／0.01|2.3115|2.1356|2.3447|2.3778|2.3752|
|tum／0.1|2.4245|2.4284|2.4048|2.4161|2.4120|
|tum／0.7|2.5426|2.5587|2.5378|2.4688|2.4639|
|replica／0.01|2.4227|2.4012|2.3757|2.4024|2.3679|
|replica／0.1|2.4849|2.4829|2.4672|2.4682|2.4432|
|replica／0.7|2.7816|2.7869|2.7656|2.6207|2.5951|

这组消融的绝对耗时明显高于两轮主测，且TUM 1%出现非单调结果：off甚至快于v11，两项开启反而较慢。它不能单独证明各开关的墙钟收益；模块归因以相同输入的独立Nsight测量为主，实际准备收益以两轮A/B主测为主，不从消融中挑选最好的数字。

## 7. 数值、边界与异常验收

- 五类checkpoint整体验收：1710组、17175项逐位比较，通过view模式、补偿开关、空预算、动态K、非默认Stream和返回结果所有权检查。
- 独立Selector组件及尾部尺寸：1566项逐位比较；Compensator组件：1026项。
- 随机／极端幅度输入、11D/19D、0–65537候选、随机行序、不同Feature幅度：405项逐位比较通过，异常状态同样比较。
- tum：792相机×18条件=14256请求，142560项比较，ID／属性／像素／radii逐位一致。
- tum随机ratio／预算：311请求，ID差异0、属性差异0；CPU/GPU控制数学的既有末位差异仍单独记录。
- replica：137相机×18条件=2466请求，24660项比较，ID／属性／像素／radii逐位一致。
- replica随机ratio／预算：311请求，ID差异0、属性差异0；CPU/GPU控制数学的既有末位差异仍单独记录。
- Selector／Compensation非有限值检查、错误后恢复、源模型修改不影响冻结实例、源模型与RNG不变均通过；两场景Compute Sanitizer memcheck为0 errors。
- 重新加载正式checkpoint及独立关闭开关36组通过；98204点合成容量的Host保留路径39组通过，后者不是新大场景速度证据。
- 最终渲染像素逐位相同，因此在相同参考图下PSNR/SSIM/LPIPS不会变化；没有用PSNR接近替代数值验收，没有保存渲染图片。

## 8. 显存与状态生命周期

|场景／实现|冻结实例保留MiB|100次变化请求峰值增量MiB|
|---|---:|---:|
|tum／v11|12.777|25.132|
|tum／optimized|12.777|25.132|
|replica／v11|62.490|86.716|
|replica／optimized|62.490|86.716|

没有新增持久的网络中间Tensor或结果缓存，新增Tile使用Block内临时共享内存和寄存器。冻结实例、UID、拓扑和有界Workspace生命周期沿用v11；更换checkpoint或地图应重建实例，同一实例串行调用。

## 9. 其他候选与舍弃原因

|候选|实测|处置|
|---|---|
|批量FP32 GEMM＋四Head隐藏层合并调度|216组初步质量检查通过；高预算组件约快15%–20%，低预算更慢；与v11有属性末位差和局部像素差|未纳入最终版本；寄存器候选更快且逐位一致|
|32点寄存器Compensator Tile|与16点同样通过原组件逐位检查，但多数代表条件更慢，中小K尤其明显|未纳入最终路径，保留原阶段证据|
|低预算强行使用16点Tile|约84／217点时明显慢于v11单点Block|保留低预算旧Kernel与1024门限|
|Gaussian→Tile／Rasterizer优化|用户明确本轮不做|只读核查，没有实现、编译或替换渲染器|

GEMM原始测量和失败入口导入现场留在GEMM_TEST.json与logs；导入错误是研究目录缺少当前模型加载依赖，已复制只读v11依赖修复，没有修改checkpoint。另一次GEMM解码改写未做性能验证，没有据此声称收益。

## 10. 接入、回退与完整性

新增部署参数：selector_tiled=True；compensation_tile默认16。独立回退本轮网络优化：

```python
runtime = DeploymentRenderer.from_checkpoint(
    checkpoint, selector_tiled=False, compensation_tile=4)
```

正常训练、训练中的推理、常规评价和Viewer默认后端均未切换。服务器仅写独立目录：/home/prmi/zwj/experiments/deployment_network_gemm_20261107。没有commit/push或生产部署。

最终二进制：/home/prmi/zwj/experiments/deployment_network_gemm_20261107/build/mycartgs_deployment_fp32_v13/mycartgs_deployment_fp32_v13.so；SHA256=eb0195413f9b44a79fe93f4f1e313437469a6c367adf289b69006e190339e15e。基准v11 SHA256=cb41244904755c25c02a1dd601b84514c83b6982c6e862043a4eb9eca7e59899。逐文件源码身份见BINARY_IDENTITY.json，本轮源码未提交。

使用原Rasterizer二进制，SHA256=27b5b468cf385dd1c96aeabdd65426da38a938a38d52d8181b1b1da485090c86，前后未变化。

- PRODUCTION：604文件，修改0，缺失0。
- INPUTS：42文件，修改0，缺失0。
- LEGACY_INPUTS：28文件，修改0，缺失0。
- LARGE_INPUTS：14文件，修改0，缺失0。

本地修改仅涉及部署网络、调度、绑定和部署说明；cuda_rasterizer及正常训练文件相对本轮开始时均未变化。原始配置和模型保留，未清理无关文件或结束无关进程。

## 11. 剩余网络优化空间

1. 低预算现在主要剩候选准备、精确选择及Host调度；可研究固定地址、有界Workspace和CUDA Graph，每帧更新相机／ratio／预算，不缓存计算结果。
2. 1024阈值仍按容量或Host K判断，并非每个视角的最佳分流。可用GPU实际K研究更细的分流，但不能为此增加中途读回或牺牲小预算速度。
3. 更大真实地图和目标端侧GPU尚未测量；这里的RTX3090结果不能直接作为端侧性能保证。应在目标设备重新测寄存器限制、Tile大小及尾延迟。
4. 高斯渲染优化继续留待用户另行指示；本轮所有网络收益均未计入任何渲染器提速。
