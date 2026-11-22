# 新联合方案：TUM / VECtor / Replica最佳Full checkpoint批量Fine

实验目录：`/home/prmi/zwj/experiments/global_view_joint_batch_20261106`。独立复制上一轮fr1_xyz已验收的源码和二进制，不修改生产或原始模型，不commit/push。

## 范围和初始化

- TUM_RGBD：freiburg1_desk、freiburg1_room、freiburg1_xyz、freiburg2_xyz、freiburg3_long_office_household，mono/RGB-D，共10组。
- VECtor：corner-slow、corridors-dolly、robot-normal，mono/stereo，共6组。
- Replica：office0–4、room0–2，mono/RGB-D，共16组。
- 共32组。沿用前次已审计的Full全轨迹最高PSNR赢家，每组在seed0三次中选一次；双目选模使用修正后的左目矫正图像域。严格检查模型在用户指定两处saved_experiments中。
- 只接收LoD步数为0、Feature Adam历史为0的Full＋tail checkpoint。原Master、Anchor成员及网络读取；六节点Feature基座继承，其余残差清零。
- seed=0。Adam/RNG重置，XYZ学习率尺度由保存的最终训练关键帧中心半径重建。这不是完整SLAM/Adam恢复。
- 每组Fine步数=所选运行实际online Mapping步数，不含tail；合计637,377步。不重跑SLAM或tail，不另加预热，不固定为8k/13.4k。

## 训练设置

| 项目 | 本批设置 |
|---|---|
| 总目标 | Full＋全局LoD1＋局部LoD2，外层1:1:1 |
| 调度 | 同一相机、同一ratio、同一更新前参数，三路独立构图依次反传，一次共享Adam |
| ratio | 0.01、0.05、0.1、0.3、0.5、0.7循环；每步不改变列表 |
| 全局候选 | 全Master地图，无视锥裁剪 |
| 局部候选 | 当前可见名义Anchor AABB全部成员，从Master独立选择，不从LoD1子集再筛 |
| 训练采样 | 两路原Gumbel-ST；预算Loss约束数量，不使用严格Top-K |
| 预算Loss | abs(sum(mask)−rN)/max(1,rN)，权重0.01；局部N取局部候选人口 |
| Photo | 原0.8×L1＋0.2×DSSIM，各场景原mapping配置；不降分辨率 |
| 重要性 | FlexGS全训练关键帧Full alpha×T统计；全局GI权重1，刷新间隔1000步；局部无GI/LI |
| 参数更新 | Master、Selector、Compensator、Anchor Feature均按原梯度路径更新 |
| 拓扑 | Fine固定，不Clone/Split/Prune；起点均only_anchor_prune=1 |
| Selector | 原11→32→32→2，温度1，学习率0.001 |
| Compensator | spatial_offset3_residual48：每Anchor共享48D Base＋五级48D Residual，纯残差累加及连续ratio插值；offset3＋feature48＋ratio3=54D，64/64与四Head、14D输出 |
| Scale / XYZ | 当前Softplus Scale解码；无额外XYZ上限，不切回旧物理加法方案 |
| 关闭功能 | Maturity、Compensation Maturity Gate、SGI、TransformField/Hexplane、Feature Fourier、额外ViewResidual/ViewInput、局部Teacher |
| CUDA | 复用已验收的Residual48训练输入融合、原生Gather/Scatter、Selector/Adam/解码及SSIM缓存等当前开关 |

Master各属性学习率沿用每个场景保存的mapping.yaml，完整配置与输入/二进制哈希逐组保存。没有把不同数据集学习率强行改成TUM值。

## 评价和存储

训练六档与未见五档0.03、0.075、0.2、0.4、0.6，分别评价全局导出及名义Anchor局部ratio严格预算；另评Fine后的未补偿Full，共23配置/组、736条汇总记录。未见ratio只评价。

局部推理使用Viewer原固定seed0 Gumbel键及stable exact-K，K=min(floor(rN),Slot可用数)，不把训练软预算误写为严格K。评价是完整可关联RGB轨迹，无帧采样；TUM/Replica使用各自原图，VECtor stereo使用与训练一致的左目原生矫正图；mono沿用原尺度校正。ATE来自固定输入轨迹，不将Fine称作重新SLAM。

保存最终checkpoint、PSNR/SSIM/LPIPS/ATE、N/K、逐帧数值和必要训练日志；不保存渲染/GT图片。逐帧选点身份保存SHA256，避免保存重复的大量ID数组；评价临时导出PLY完成渲染后删除，仅限本批生成文件，原模型不动。

每种数据集/相机入口先6步smoke、初始Master/网络逐位检查与Full/全局低高ratio/局部低ratio评价，合格后进入正式训练。沿用全局与局部独立健康检测：连续零选点/概率饱和8次、Scale>100倍或XYZ>256 Anchor单位的长尾达到1%即在Adam前停止；仅记录和报错，不截断输出。非有限值立即失败。

串行执行，启动前GPU空闲显存至少6GiB；磁盘不足5GiB停止并保留现场。明确OOM才最多额外重试2次，仍失败跳过；其他错误停止队列定位。PID/start_ticks/boot_id、日志和训练CSV增量保存在控制文件中。不能仅凭JSON running判断健康。

队列另启用独立进度监测：每60秒核对本批活动子进程身份、真实日志和Mapping CSV增量；连续900秒无进展时保存CPU等待/GPU显存现场，只停止身份三元组一致的本批子进程，主控制器保留失败并停止，避免挂起持续占用时间。
