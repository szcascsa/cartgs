# fr1_xyz局部500点预算的推理／渲染速度实测

当前默认为两网络真实视角＋LI=0.1。只读加载已训练模型，不训练、不改服务器生产代码、不commit/push，不保存渲染图片。

## 比较与口径

1. 原版CaRtGS RGB-D Full：此前三遍中Full PSNR最高的repeat1，35301点，实际使用原版Python Renderer与原版CUDA扩展。
2. 原版CaRtGS Full静态属性缓存对照：同模型、同原版CUDA渲染，仅一次准备固定Master的activated属性。用于避免把原版重复激活属性的开销当成LoD收益。
3. 原无视角联合模型：局部名义Anchor视锥候选，严格500点，原Selector／Compensator输入。
4. 当前默认模型：相同局部500点定义，两网络真实视角、LI=0.1训练所得checkpoint，实际`view_conditioned=True`。
5. 当前默认模型Master Full：13298点，使用当前原生inference-only CUDA Rasterizer，作为同地图、同位姿的计算对照。

同一792个RGB时间戳、640×480、FP32、TF32关闭、batch1。每个模型使用各自保存的SLAM位姿以对应实际图像；同时列出位姿差异，不声称不同地图具有完全相同几何工作量。新旧两个本方法模型相机序列应逐值一致。500保留点全部计入预算，不以Alpha点数或可见点数代替K。

## 计时

- 主测为同步墙钟延迟：可见候选池构建、Gather、Selector、严格预算选择、Compensator、属性打包及原有finite／状态检查全部计入推理准备。渲染计时含Renderer调用与等待图像计算完成。
- 原版常规Full的属性激活、屏幕点分配等原有Python准备均计入Full渲染调用；另有静态属性缓存对照，均不省略CUDA渲染。
- 预热各方法64次，随后完整792帧测3轮；帧顺序为顺序、逆序、固定seed打乱，方法顺序轮转，防止只先测某一方法。所有轮次保留，不按速度选最好一遍。
- 不使用已导出500点包缓存冒充动态推理，每次调用重新执行实际局部选点与补偿。
- 单独在均匀96帧测CUDA Event的stream时间；该时间包含CPU发射间隙和内部同步，不冒充纯Kernel累计时间。另做独立Profiler诊断，观察GPU Kernel累计时间与发射数，不用Profiler墙钟作为速度结果。
- 不计文件加载、相机构建、GT加载、指标计算、网络传输、Viewer显示；均在计时区间之外。首次模型加载和首次调用另列。
- GPU其他任务、温度／时钟／显存与时间块前后状态保存；出现活动训练任务则按实际记录结果污染，不杀进程或偷偷删样本。

输出逐样本CSV、各轮／全体mean、median、p10/p90、配对快慢比例、推理与渲染分解、GPU Event与Profiler归因、简单PSNR与点数、输入和生产文件前后SHA，以及中文报告和图。不将单场景设备延迟称为完整SLAM或Viewer FPS。
