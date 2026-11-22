# 最新native部署：每视角100–500点三栏图

日期2026-11-06。使用`deployment_native_select_20261106`最终验收的FP32 v4：C++整条准备、精确Radix选点／稳定压缩、共用finite状态。新独立目录`/home/prmi/zwj/experiments/native_budget100_500_fr1xyz_20261106`，复制并核对运行时源码，在独立build编译；不修改算法、权重、生产默认或共享环境，不commit/push。

模型仍是fr1_xyz RGB-D四Head“两网络真实视角＋LI=0.1”，13298点、2790 Anchor。每个视角在Viewer名义Anchor候选池上请求100/200/300/400/500点，严格K=min(B,E)，网络ratio=min(1,B/Nv)。100点部分视角ratio小于训练下界.01，明确记录，不补训。五预算统一相机／seed0固定噪声。

原版CaRtGS沿用已核验RGB-D repeat1的35301点地图、792帧Alpha贡献计数和PSNR；逐帧耗时复用上次local50测试中的cartgs_full三轮数据，不重新渲染原版。逐帧按RGB时间戳和文件对齐，各地图使用各自SLAM位姿。历史与本轮GPU负载不同，图中CaRtGS耗时标为历史参考，不计算严格配对加速。

先验证三帧×五预算的带计数Renderer与正常Renderer颜色／radii逐bit一致，然后64视角预热，792帧×三轮×五预算=11880次实际准备＋正常Renderer计时。顺序／逆序／seed0乱序遍历并逐帧轮转预算顺序；全部样本保留，外层墙钟包括C++准备、必要finite检查及同步，分别记录处理／准备和渲染。模型读取、JIT、相机构建、GT、指标、计数与显示不在计时内。

792帧×五预算=3960条质量／点数评价，记录输入K、radii>0投影数、实际参与Alpha合成点数、PSNR和photo-L1。图片不保存。左栏主图原版Alpha点数及地图总N，细节图本方法100–500点的真实Alpha贡献数；中栏PSNR；右栏上下分开处理、渲染耗时。

PSNR强高斯平滑σ=35帧、truncate=3、reflect边界，等效211帧窗口；处理／渲染先按帧平均三次再高斯平滑σ=12、truncate=3、reflect。点数不平滑。所有图例均值使用原始数值，另存原始PSNR图、CSV及PDF/SVG，平滑不改变报告指标。

保留604个生产关键文件、原模型、历史CSV及来源源码／实际.so SHA前后核验；不清理模型，不停止无关GPU进程。
