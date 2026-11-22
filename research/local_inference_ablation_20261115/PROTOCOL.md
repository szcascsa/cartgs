# 32组场景的局部Ratio推理消融

完成标准：固定当前32组完整方案最终checkpoint，完成每组11档局部Ratio的两种消融，共704条新指标；导出逐帧PSNR／SSIM／LPIPS、N、实际K、候选数、选点哈希及与完整方案的配对差值。所有场景完整轨迹评价，不保存渲染图片。

参考表为`cartgs_pruning_baselines_20261112/reports/METHODS_ALL_METRICS.csv`中的352条`ours_view_ratio`，SHA256固定为`e56cc01dfe0dcfb1613f06d4211ce5d7e137b6c6ce341c2667db200cc027c891`。该表已采用corner-slow／robot-normal双目的11月11日最新复跑模型；不重新挑选实验赢家。

| 组别 | 选择 | 补偿 | 数据来源 |
|---|---|---|---|
| 完整方案 | 原视角Selector | 原四Head视角Compensator | 既有数值，只作抽样渲染复核 |
| selector_no_comp | 与完整方案逐帧相同ID | 完全关闭，使用冻结Master原属性 | 新评价 |
| random_with_comp | 当前候选池中均匀随机、无放回 | 原四Head真实视角补偿 | 新评价 |

这属于冻结网络的推理消融，不是去掉模块重新训练。随机组可揭示原Compensator对非Selector点集的表现，不能据此推断随机选择重新训练后的上限。

关闭补偿使用Fine终点的Master原属性，不回退到在线Full起点；地图与网络共同训练产生的适应仍然保留。对照表分别给出完整方案、Selector无补偿、随机选点＋补偿三种方法。

所有组使用同一Master、相机、名义Anchor可见候选、slot资格、预算K=min(floor(r×N_view), eligible)、连续ratio条件及同一未校正原图；每视角K与完整方案逐项核验。关闭补偿不改变选点。随机优先级使用现有FixedBudgetRenderer的seed0固定均匀随机分数，各Ratio取前K，保持随机集合嵌套；同一Gaussian跨视角复用优先级，不引入逐帧抽样抖动。

训练Ratio为[.01,.05,.1,.3,.5,.7]，未见Ratio为[.03,.08,.2,.4,.6]，两类均只推理。保持原全轨迹RGB关联、双目原图口径、原evo轨迹对齐逻辑与各模式correct_scale开关。各场景之间不强行匹配点数。

直接复用冻结批次Python和CUDA评价实现，独立复制Python快照及结果目录，CUDA扩展只读复用。每视角只构建一次候选池和视角编码。首／中／末帧逐Ratio验证手动复用数据与原公开prepare入口的选点及打包Tensor逐位相同；全部视角验证原Selector历史ID哈希与K；冻结网络状态及源文件前后哈希一致。

指标使用相同FP32定义，按像素数限制小批量调用，SSIM与LPIPS使用reduction=none后逐帧记值。首／中／末帧与原标量TorchMetrics及历史完整方案比较：PSNR误差<1e-4 dB，SSIM／LPIPS误差<2e-5，否则停止并修复。批量只影响指标执行，不改变渲染；不以指标接近替代输入Tensor逐位验证。

为避免同一GT在22项评价中重复经过LPIPS主干，六入口额外验收后允许缓存已归一化的GT特征，最多保留4帧；预测端、层间差分、线性层、空间平均和层间相加完整复用原TorchMetrics定义。实际使用与未使用缓存的运行均记录来源和数值核验，不改变指标公式或GT图像。

不训练、不改生产配置、不改共享conda、不提交或推送。独立进程最多使用32%显存，只有明确OOM才允许额外重试2次，所有失败attempt保留；非OOM先报错定位。磁盘不足5GiB暂停，不清理旧模型。

该任务不改变正在运行的无GI／LI训练队列；共享GPU时耗时仅记录实际运行值，不用于性能比较。
