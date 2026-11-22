# 两网络真实视角＋LI：局部500点与50%部署推理复测

日期：2026-11-06。独立目录`/home/prmi/zwj/experiments/deployment_local500_ratio50_20261106`。

使用原四Head默认checkpoint：`view_conditioned_li_20261106/runs/full_both_li/23560_shutdown/ply`，joint schema2、LI=0.1，N=13298、2790 Anchor。保留原训练结果，不重新训练、不切生产默认、不commit/push。

比较同一冻结Python源码、同一权重、同一相机及seed0固定Gumbel噪声的PyTorch、精确CUDA v3、新`deployment_runtime.DeploymentRenderer`。另测当前模型未补偿Full。局部500点取K=min(500,E)；局部50%取K=min(floor(.5Nv),E)，Nv为Viewer名义可见Anchor的成员数，E为可选成员数。新部署后端保持FP32，但改变MLP FMA、视角、Quaternion运算顺序，允许小误差，不声称逐bit等价。

完成标准：独立编译扩展；验证checkpoint身份、候选N和严格K；5个视角×两预算smoke；64视角预热；792帧×三轮交错速度测试（顺序、逆序、seed0乱序），全部轮次保留；96视角独立CUDA Event辅助；固定输入Selector及Compensator解码微测；792帧×两预算检查选点、属性、图像、PSNR，66帧补充SSIM/LPIPS。只保存数值和分析图，不保存渲染帧。

总耗时为同步墙钟prepare＋Rasterizer，保留有限值检查，prepare关闭额外分项同步`measure_time=False`。中间返回时间作为推理／渲染分界，prepare本身有finite结果完成依赖；另记录CUDA Event。所有后端共享相机构建，模型加载、JIT编译、GT、指标和显示在计时外。Selector/Compensation CUDA Event微测不含候选、严格K排序、finite读回和Rasterizer，不能替代总耗时。

部署质量门槛预先固定：相对精确CUDA，平均ΔPSNR≥−0.01dB，最差1%视角分位ΔPSNR≥−0.05dB，单帧最低ΔPSNR≥−0.1dB；所有N/K一致且finite。记录选点身份变化，不将相同数量等同于相同点。PyTorch与精确CUDA的smoke保留严格逐bit检查；新部署相对精确CUDA报告最大绝对误差与图像误差，不套用逐bit结论。

记录主项目604个关键源码／配置／库文件、原checkpoint和源码前后SHA、扩展二进制身份、GPU任务／频率／温度及显存。服务器存在其他任务时保留观察结果，不终止无关进程；只在同轮匹配下计算加速，不直接与历史34ms或8ms相减。
