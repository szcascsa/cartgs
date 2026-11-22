# D组：全部完整方案场景的无重要性监督消融

2026-11-15，用户授权只运行D组：关闭GI和LI监督；不因Scale长尾终止，Selector全选0仍报错。

## 完成标准与范围

覆盖完整方案已完成的32个数据集／场景／相机模式：TUM 10组、Replica 16组、VECtor 6组。每组seed0运行一次；原始Full赢家、相机与训练步数沿用完整方案，共637377个Fine步骤。不重新运行在线SLAM，不在完整方案LoD终态上续训。失败及重试现场全部保留，不将失败记为质量结果。

每个成功训练的模型评价23项：全局与局部各11个ratio，加Fine后Full。训练ratio=[0.01,0.05,0.1,0.3,0.5,0.7]；未见ratio=[0.03,0.08,0.2,0.4,0.6]只评价。使用原完整轨迹对应RGB帧，不进行帧采样，不保存渲染或GT图片。VECtor双目沿用原训练校正，评价使用已统一的CaRtGS原RGB，不做单目式尺度对齐。

## 唯一算法消融

`Selector.gi_loss_weight=0`、`ViewBudget.li_weight=0`。两网络真实视角、Residual48结构、原Softplus Scale、原预算Loss、训练ratio、学习率、初始化、Master更新和固定拓扑均不改变。每步Full／全局LoD／局部LoD三路反传后一次Adam。全局条件为零，局部条件为真实view7＋mode1。

GI采集与原周期刷新保持开启，只用于对齐原统计流程，不产生GI监督梯度。LI权重0时不采集当前Full贡献、不执行定向LI VJP。此前采集颜色／radii逐位一致验证继续通过原生回归检查。D组不依赖GI／LI教师决定实际选点。

## 健康停止策略

新增独立研究开关`LoDHealth.zero_selection_only=1`，只在本轮源码和配置启用，旧默认false行为不变。

- 有合格候选时，保留原连续零选点判定：全局连续或相同ratio累计连续8次零选点报错。没有合格候选时不误判为Selector坍缩。
- Scale／XYZ长尾与概率饱和继续统计，不触发健康失败，不限幅、不重采样、不改变Loss。
- 保留非有限概率、补偿、Loss、梯度及CUDA运行错误检查；无效数值不能作为成功模型保存。
- 零选点失败不盲目重训，记录并继续其余场景。有明确OOM证据才额外重试最多2次，计数跨重启保留；非OOM故障保留现场并定位。
- 磁盘不足5GiB暂停，不擅自清理模型或历史结果。

原完整方案采用历史长尾停止及重试策略，本轮允许长尾，因此两边的成功训练筛选条件不同。这一差异与实际长尾记录必须随结果披露。

## 构建与保护

服务器独立目录：`/home/prmi/zwj/experiments/no_importance_all_scenes_20261115`。只读复用完整方案冻结快照、相同Full起点和第三方依赖；全部Mapper C++对象独立重编译，独立runtime/bin和runtime/lib。核对实际进程maps加载本轮Mapper。

完整方案来源为`realLOD/saved_experiments/dual_view_li_best_full_20261107`的26组及`dual_view_li_vector_best_full_20261107`的6组；两批源码和关键二进制相同。原Full、历史模型、源码／配置／runtime及生产项目共2645个文件建立前后SHA保护。本地已有未提交改动不混入研究源码，不修改生产默认、不commit/push、不改共享conda。

## 验收与评价

原生验证涵盖旧健康策略、关闭长尾后仍拒绝零选点／非有限值、D组12步三分支真实更新、LI旧配置梯度隔离和旧／新库90Tensor CUDA逐位回归。六种数据集／相机入口各6步smoke，Full初态所有原Tensor和PLY属性逐位核验；首／中／末三帧进行30项短程渲染验收。正式训练始终重新读取原Full。

完整训练核验真实Full／global／local次数、Master／Selector／Feature更新、ratio轮转及与完整方案相同的相机序列。分开记录累计训练K，不能将相同步数当相同选点曝光。保留GI刷新次数并确认LI采集／VJP均为0。

导出每场景、每scope、每ratio的PSNR／SSIM／LPIPS／ATE、N与实际K、逐帧结果和相对完整方案变化；按照各地图与各视角自己的人口计算预算，不添加跨模型same-K实验。单seed结果不作跨seed稳定性主张，不以nonBank选权重、终点或checkpoint。

预声明质量参考线：相对完整方案，全局六训练档平均PSNR下降不超过0.10dB、任一训练档不超过0.20dB、Full不超过0.10dB；局部1%／5%单列，不用高ratio均值覆盖。门槛只用于事后判读，不影响本轮训练停止或调度。
