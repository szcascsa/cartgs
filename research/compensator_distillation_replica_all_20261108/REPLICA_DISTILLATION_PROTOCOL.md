# Replica 全部场景蒸馏协议

覆盖office0、office1、office2、office3、office4、room0、room1、room2，按用户此前确认的Replica单目＋RGB-D两个入口，共16组。沿用fr1_xyz确定并在四个场景复验的四阶段流程，seed=0，每场景最大17600次学生更新，学生32/32和四个直接Head，共3534参数。

初始蒸馏6000步、图像校准2000步、几何校准6000步、连续Ratio校准3600步，配置详见PROTOCOL.json。每阶段从上阶段best初始化，重置Adam和采样状态。训练档位[.01,.05,.1,.3,.5,.7]，连续阶段50%保留轮转档位，50%采样相邻区间log均匀Ratio。另评价[.03,.08,.2,.4,.6]，这些具体点标为非锚点Ratio。

Teacher、Master、Selector、Anchor Feature与Gaussian拓扑冻结。每个场景训练独立学生并绑定其Teacher指纹。监督由原属性和同选点、同相机的Teacher RGB构成，仅学生更新。最后阶段原始残差权重0.01、每步图像监督权重200。

使用完整保存估计轨迹及原图分辨率，要求所有帧与RGB编号在0.001秒内一一匹配。每5帧池从学生训练排除，3个均匀视角验证，其余池均匀选32帧测试。模型选择依据Teacher验证误差，GT只用于评价。

office0 RGB-D及office3 RGB-D复用此前已完成结果。复用前核对同一个Teacher/数据/轨迹、全部阶段配置、步数、学生文件哈希、质量评价及部署验收，复制到本轮独立目录并保护原文件。其他14组本轮训练，新增计划246400步。

逐Ratio分别记录全局/局部PSNR、SSIM、LPIPS、Teacher图像MSE、实际K、选点ID哈希、XYZ与Scale尾部。全局保留原阈值及同分语义，局部保留名义Anchor候选和严格K。质量门限沿用平均PSNR损失≤0.01dB、p99≤0.05dB、最坏≤0.1dB。

单个本轮GPU进程串行执行，保留PID/start_ticks/boot_id、训练CSV增量和日志。仅有明确OOM证据允许额外重试2次，保留失败输出，不删除checkpoint。非OOM失败保留现场并继续其他组，主线程定位后修复。磁盘低于5GiB停止新增任务，不自动清理。

所有任务位于/home/prmi/zwj/experiments/compensator_distillation_replica_all_20261108。研究前后核验生产文件、原Teacher、复用结果和二进制哈希，不保存渲染图片。曲线与数值结果单独保存。共享GPU墙钟只作实验成本，推理速度待GPU空闲测量。
