# Replica 两级增生阈值1.5／2.5倍实验

范围：office0–office4、room0–room2，各跑 Mono、RGB-D，seed=0各一次，共16组。重新执行完整 Full SLAM＋正常 Full tail，保存 checkpoint；不追加 LoD Fine。

| 判定 | 原版 | 上轮3倍 | 本轮 |
|---|---|---|---|
| L0同层候选 | τ₀≤g<2τ₀ | 相同 | τ₀≤g<1.5τ₀ |
| L0→L1候选 | g≥2τ₀ | 相同 | g≥1.5τ₀ |
| L1同层候选 | 2τ₀≤g<4τ₀ | 2τ₀≤g<3τ₀ | 2τ₀≤g<2.5τ₀ |
| L1→L2候选 | g≥4τ₀ | g≥3τ₀ | g≥2.5τ₀ |
| L2规则 | 原实现 | 相同 | 相同 |

两项倍率均以基础阈值τ₀为参照，不是相对上一层阈值相乘。g仍为成员Gaussian在有效Full统计窗口内的时间平均梯度。τ₀仍由首个有效增生窗口中L0有限正平均梯度q90校准，不强制复用历史运行的数值。

只在独立源码的growBlocks中增加两行high覆盖。low、alpha=2、层级上限、满员条件、空间去重、观测插入、Gaussian和Feature生命周期不变。保留only anchor prune=1、正常tail终止、原Loss／学习率／优化器与相机配置。

服务器独立目录：`/home/prmi/zwj/realLOD/checkpoint/replica_l0_l1_grad1p5_l1_l2_grad2p5_20261108/`。源码、Mapper对象、入口二进制、动态库、模型、日志及评价结果全部独立保存；数据集与第三方预编译依赖只读复用。checkpoint附带阈值与源码哈希标记。

完整轨迹离线评价PSNR、SSIM、LPIPS、ATE、Gaussian数量及逐帧指标，不保存图片。Mono沿用尺度对齐，RGB-D不进行尺度对齐。全部训练后补齐评价，首个Mono／RGB-D模型先做全轨迹评价入口验收，保持上一轮时序。

验收：本地补丁与真实分支边界测试；全部Mapper对象独立编译；真实AnchorBank CPU／CUDA增长测试；16组无训练控制路径检查；真实进程maps核对独立库；Full／tail计数和零Fine验收；全轨迹指标逐帧核对；原项目及上轮checkpoint哈希前后核验。

OOM仅有明确证据才最多额外重试2次，失败现场保留；磁盘低于5GiB停止发射，不清理任何旧模型。独立监控每小时检查实际进程与日志增量。

后续比较包括上轮3倍、原2／4倍及Vanilla最佳。新方案各一遍，历史最佳可能来自三遍；异步SLAM曝光、实际Mapping步数及τ₀会变化，不能将单次差值直接归因为阈值收益。
