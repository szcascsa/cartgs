# q80、alpha1.5最终方案：Replica各组三遍并选最佳

用户明确将当前q80、alpha1.5作为最终方案，本轮只补齐实验重复次数并选出最佳结果，不修改生产默认、不commit／push。

范围为Replica的office0–office4、room0–room2，Mono／RGB-D共16组，每组总计三遍seed0。原q80批次已有16遍，room0／room1额外批次已有8遍，均保留复用。此次仅对office0–office4和room2各模式补repeat2、repeat3，共新增24遍。每遍从头执行Full SLAM＋正常Full tail，不追加LoD Fine，不从最终checkpoint续训。

阈值t由首个有效增长窗口L0成员Gaussian的有限正Full屏幕XY梯度均值取q80，随后固定；alpha=1.5。L0同层区间[t,1.5t]，L1为[1.5t,2.25t]，L2为[2.25t,3.375t]，严格超过上界才细化，L2超过上界按已有最大层级规则跳过。only anchor prune=1，其余Loss／优化器／学习率／拓扑更新与当前q80配置完全相同。

独立目录为`/home/prmi/zwj/realLOD/checkpoint/replica_q80_alpha1p5_three_runs_20261109/`。源码、配置、运行时、模型、日志与评价独立保存。复用已编译q80二进制和库的独立副本，只重新定位RPATH，readelf核对.text机器码不变，ldd及/proc/PID/maps核对实际加载新目录的Mapper。原项目、前两批q80模型及旧配置只读保护。

新增24遍按repeat2的12组、repeat3的12组顺序串行训练。延续首个Mono和RGB-D完成后进行全轨迹评价入口验收、其余全部训练后统一补评的时序，避免并发评价扰动异步Mapping。保存完整轨迹PSNR／SSIM／LPIPS／ATE及逐帧指标、模型N、校准t、各层Anchor和成员数量、实际online／tail步数。Mono沿用尺度对齐，RGB-D不作尺度对齐；不保存任何渲染图片。

全部48遍核验后，对每个场景／模式以原始完整轨迹平均PSNR最大者作为最佳，同分优先repeat编号较小者。SSIM、LPIPS、Gaussian数、Anchor2占比、ATE、模型路径和评价路径必须绑定同一遍，不能逐指标择优。三遍原始记录、均值、样本标准差和选择来源全部保留。最佳结果用路径清单引用已有checkpoint，不复制或覆盖模型。

主交付：`Q80_FINAL_BEST.csv`／`Q80_FINAL_BEST.json`、`Q80_ALL_48_RUNS.csv`／`Q80_ALL_48_RUNS.json`、中文`Q80_FINAL_REPLICA_REPORT.md`。默认三遍均值和默认三遍最佳分别对照，同一对照的其他指标绑定同次或按明示规则平均。最佳对最佳用于结果汇总，不替代均值／波动报告。

验收包括24组配置、CPU／CUDA原生增长与校准测试、所有运行q80仅校准一次且t固定、零LoD、零图片、每个场景／模式恰有repeat1／2／3、最佳选择与同次指标映射、原始逐帧均值核对及历史文件SHA256前后核验。每小时检查实际进程和日志进展；OOM有明确证据才最多额外重试2次，非OOM失败保留现场，禁止因质量低重跑。磁盘不足5GiB停止新发射，不清理旧模型，不恢复其他队列。
