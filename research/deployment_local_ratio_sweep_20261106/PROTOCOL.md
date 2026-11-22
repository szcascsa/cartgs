# 新部署MLP局部ratio 0.1／0.3／0.5／0.8质量与速度

日期2026-11-06。独立目录`/home/prmi/zwj/experiments/deployment_local_ratio_sweep_20261106`。模型沿用四Head、两网络真实视角＋LI=0.1的`view_conditioned_li_20261106/runs/full_both_li/23560_shutdown/ply`，N13298、Anchor2790，不重训。

运行时源码直接复制上一轮已验收的独立快照，逐文件核对SHA；新目录重新构建两个CUDA扩展。原项目、既有checkpoint、历史结果只读，不改共享环境，不停止无关进程，不commit/push。

局部预算沿用Viewer名义可见Anchor成员数Nv，严格K=min(floor(r×Nv),E)，E为合格成员数，所有保留点计入K。训练ratio为.01/.05/.1/.3/.5/.7，.8明确标为超过训练上界的外推，Feature残差全部活跃但ratio编码继续取.8，不称插值。

主测仅新DeploymentRenderer与当前未补偿Master Full，640×480 FP32、TF32关闭。每方法64视角预热，792帧三轮顺序／逆序／seed0乱序，每帧轮换方法顺序。记录准备、Rasterizer及同步总墙钟，全部11880条保留。另96视角×五方法480次CUDA Event辅助。不保存渲染帧。

准备包含视锥／候选、view code、Selector／score、严格K、Residual48、Compensator、解码和finite检查。模型加载、JIT、相机构建、GT／指标、Viewer传输显示在计时外。prepare返回前有finite完成依赖；最后同步保证渲染完成。无CUDA Graph、无上帧Mask或ratio输出缓存。

五个代表视角×四ratio对PyTorch和精确CUDA packet逐bit检查；新后端允许FMA／归约次序差异，不要求逐bit。相同792帧评价新部署、精确CUDA的四ratio及Master Full，共7128条PSNR／SSIM（11×11）／LPIPS-AlexNet／photo-L1；记录Bank72/nonBank720、实际K／投影K和逐帧配对误差。相对精确CUDA仍使用运行前固定门槛：平均ΔPSNR≥−.01dB、最差1%视角分位≥−.05dB、单帧最低≥−.1dB；不以PSNR接近冒称逐bit等价。

记录GPU负载／频率／温度、显存、模型及实际扩展身份，604个生产关键文件和checkpoint前后SHA。最终提供中文报告、逐帧CSV、ratio–质量与速度曲线，并明确跨轮次性能及.8外推限制。
