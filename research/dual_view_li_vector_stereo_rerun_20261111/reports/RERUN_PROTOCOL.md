# VECtor两组双目配对重跑协议

按用户要求重跑corner-slow双目及robot-normal双目的LoD Fine阶段，旧结果和Full起点只读保留。

使用原批次冻结的527个源码文件及16个运行文件，SHA256逐项一致，不使用工作树后续改动，不重新编译或修改生产配置。

- 相同seed=0；corner-slow 11,470步，robot-normal 13,006步。
- 双网络真实视角＋LI=0.1；Full＋全局LoD＋局部LoD，每步联合反传后一次Adam。
- Master可更新，拓扑固定；每次从原Full起点重新初始化Adam/RNG，不是完整SLAM恢复。
- 训练Ratio：[0.01,0.05,0.1,0.3,0.5,0.7]轮转；未见Ratio：[0.03,0.08,0.2,0.4,0.6]仅评价。
- 全局训练全图候选；局部使用原名义Anchor视锥候选预算。训练相机校正与原批次完全相同。
- 评价以CaRtGS未校正原图为准，双目无尺度对齐；每组23项完整轨迹数值评价，不保存渲染图片。
- 健康守卫、零选点和XYZ／Scale长尾阈值均保留；健康停止或明确OOM各最多额外重试2次，历史现场不覆盖。
- 初始化Master／网络逐bit验证；原训练及新训练的keyframe序列、更新预算、身份与数值结果分别比较。
- 只在新独立目录写配置、模型、二进制、日志与评价，不commit／push。

Full起点PSNR及选择赢家记录仍保留当时校正左图口径，最终对照以此次及11月8日更新的未校正原图结果为准。

| 场景 | Fine步数 | Full起点 |
|---|---:|---|
| VECtor/corner-slow/stereo | 11470 | /home/prmi/zwj/realLOD/saved_experiments/multi_dataset_full_tail_prune1_seed0_20261103/jobs/VECtor/corner-slow/stereo/prune1/seed0_repeat1/attempt_01/model/11900_shutdown/ply |
| VECtor/robot-normal/stereo | 13006 | /home/prmi/zwj/realLOD/saved_experiments/multi_dataset_full_tail_prune1_seed0_20261103/jobs/VECtor/robot-normal/stereo/prune1/seed0_repeat3/attempt_01/model/13406_shutdown/ply |
