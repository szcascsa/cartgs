# Replica L1→L2 梯度阈值3倍：Full checkpoint实验

范围为 office0、office1、office2、office3、office4、room0、room1、room2，每个场景运行 Mono 和 RGB-D，seed=0各一次，共16组。

重新执行完整在线 Full SLAM 和正常 Full tail，保存新checkpoint，不追加LoD。原Full checkpoint的地图拓扑已经生成，不能在固定拓扑的Fine阶段检验增生阈值。

| 判定 | 原实现 | 本实验 |
|---|---|---|
| 基础阈值τ₀ | 首个有效增长窗口中，L0成员有效正平均梯度的q90 | 保留相同校准规则；不强制复用旧运行的数值 |
| L0同层候选 | τ₀≤g<2τ₀ | 相同 |
| L0→L1候选 | g≥2τ₀ | 相同 |
| L1同层候选 | 2τ₀≤g<4τ₀ | 2τ₀≤g<3τ₀ |
| L1→L2候选 | g≥4τ₀ | g≥3τ₀ |
| L2规则、最大层级 | 原实现 | 相同 |

g仍是单个成员Gaussian在有效Full优化统计窗口内的平均梯度，不改成Anchor成员之间求平均。空间键去重、观测插入、Clone/Split、轮转拓扑、Anchor prune及Feature生命周期均保持不变。

使用原Replica Full批次的源码、camera/mapping/selector配置作为匹配起点，仅在独立源码副本中的growBlocks增加一行L1上界覆盖。only Anchor prune=1、正常tail终止、原优化器/Loss/学习率不改；不混入新的LoD默认配置。

服务器结果目录：`/home/prmi/zwj/realLOD/checkpoint/replica_l1_l2_grad3_20261108/`。各组最终模型位于`jobs/Replica/<scene>/<mode>/prune1/seed0_repeat1/attempt_NN/model/*_shutdown/ply`。checkpoint旁附加`checkpoint_experiment.json`，明确阈值倍率与源码哈希。

完整轨迹离线评价导出PSNR、SSIM、LPIPS、ATE、N和逐帧数值，不保存渲染图片。Mono沿用尺度对齐，RGB-D不做尺度对齐；此处沿用原Replica批次评价口径。新16组Full在线训练沿用原Full批次时序，全部Full完成后统一补齐评价，首个Mono/RGB-D模型先做原生评价入口验收。这样避免把评价竞争引入异步在线Mapping预算；既有LoD消融并发队列不作修改。

OOM有明确证据才额外重试，最多2次，保留每次失败现场；磁盘低于5GiB暂停，不清理其他模型。每小时检查实际进程身份与日志增量。

验收顺序为独立编译全部Mapper对象和Replica入口、真实AnchorBank CPU/CUDA边界测试、16组无训练配置检查、首组Full及评价检查、正式队列继续。只报告实际执行通过的验证。

本地8项测试已通过，其中标量边界测试从真实C++判定片段提取编译；该测试不能替代服务器原生Anchor增生或在线训练验收。

异步Tracking/Mapping导致实际online步数、相机曝光和校准τ₀可能在两次运行间变化；后续比较必须同时记录这些差异。原先seed0三次中最佳PSNR与本轮单次结果的差异不能直接视为阈值的稳定因果收益。
