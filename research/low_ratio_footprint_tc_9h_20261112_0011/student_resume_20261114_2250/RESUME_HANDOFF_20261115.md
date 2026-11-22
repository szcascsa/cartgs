# 2026-11-15续算接续记录

## 时间与连接

- 用户最新明确重置九小时：香港时间2026-11-15 02:44:06至11:44:06；10:44:06起预留最终验收。已写远端及本地 `reports/PROTOCOL.json`，旧协议保存在 `BUDGET_RESET_20261115_0244.json`。不能再次重置。
- 最近时钟约04:00（UTC20:00）；已用约1小时16分。
- 远端根 `/home/prmi/zwj/experiments/low_ratio_footprint_tc_9h_20261112_0011/student_resume_20261114_2250`；本地为本文件目录。
- SSH bridge当前 unified exec session **75221**；`remote_bridge.py`凭据仅驻留进程内存。通过write_stdin传入JSON行 `{op:exec,cmd:...,timeout:...}` 或put/get。连接正常。不要输出或写入凭据。远端无rg，已确认后可用Python/grep。
- 不得派生agent。所有GPU任务串行；每个run_stage持有reports/controller.lock；不能启动重复控制器。不操作其他GPU进程，不commit/push/deploy，不改训练/生产/模型/共享conda。

## 已完成

- 修复 `verify_execution_identities.py`：原identity已有selector_parameters字典，新参数计数字段改为selector_parameter_count。18组实际执行Selector、Student及Feature hash通过。
- 旧 RESEARCH_ACCEPTANCE 在完成69项后因上述元数据错误退出；ACCEPTANCE_CONTINUED已经补齐余下11项，182及I_final各三项Compute Sanitizer均0错误；4个Nsight代表Profile已生成但**尚未export/analyze新Profile**。
- `summarize_joint_final.py --controller JOINT_FINAL02 --supplement RESEARCH_ACCEPTANCE`已完成；18场景×14条件×10视角×双轮：705600墙钟、84672 Event、35280质量、60480逐位调度检查。JOINT_FINAL_OVERVIEW/JOINT_SCENE_RATIO等已下载。
- 已有I_final相对A原FP32+F2 1.4755x、旧G_cpp_pose 1.2053x（主要Ratio）；**这些是重置预算前已有收益，不算本次新增**。质量平均损失0.00849dB、最差条件0.20278、最差帧0.30253。Student3534参数，保持32/32和独立Head。
- `summarize_thin_dispatch.py`已汇总旧50个任务：18组LoD+32组nested双轮。Thin相对同一182增益1.06685x/1.07796x，RGB/radii逐位相同。Full全部100个场景轮次均快于F2及原182；这些也是本次起点已有候选。
- 补充原C++位姿+F2对照 `joint_native_control.py`：18组双轮151200样本，I_final相对C_native_F2 1.28364x。补充single_instance_memory.py 18组；I与G部署显存同等级，峰值增量约34–228MiB。

## 当前活跃

- **TAIL_VALIDATION_20261115** 控制器PID3605014；必须读JSON核验PID/start_ticks/boot_id后确认，不依赖此PID。
- 共58任务，50个正式双轮渲染任务全部完成（18个Student LoD、32个固定Full属性nested），末尾6项Sanitizer在运行。
- 最近任务 `new_overflow_prefix_synccheck_a1`，前5项Sanitizer全0错误。单项启动可能约2分钟GPU0%：已核验这是Compute-Sanitizer/子Python装载期，后续正常完成，不盲目杀进程。
- 该计划只冻结 alpha2（2/255弱尾）和overflow_prefix，**当前尚未读取保留集质量/速度汇总**，以便后面的更强阈值只依开发集选择。
- `launch_deep_candidates_20261115.py`及其所有实现/Runtime/开发脚本已上传，等TAIL_VALIDATION complete后立即启动。29任务：tail_strong、encoded_half2、cub_depth、tile_clear、half_prefix、joint_incremental、full_pose的独立构建/双轮开发；CUB另有精确边界测试。

## 本次新增候选已测结果（开发仅office2/room0 RGB-D）

- CANDIDATES_20261115 complete：透射率.015/.02/.03/.05。前二者无稳定完整收益，后二者画质失败；不保留。弱尾1.25/1.5/2相对182约3%/4%/5–7%降时，开发画质通过，alpha2进入冻结保留验证。
- joint_cooperative：动态K协作排序逐位通过，但慢约0.5–2.5%；不保留。
- radix_grid CTA4/8慢7–15%，CTA16小幅慢，32基本持平；逐位通过，不保留。
- ADDITIONAL_20261115 首次exp2构建失败：CUDA11.8无__exp2f设备接口，改为PTX ex2.approx.ftz.f32恢复；旧日志保留。ADDITIONAL_CONTINUED已全部38项完成。
- pair_coarse：按空间变化近似相邻Alpha，画质通过但慢约3–6%；不保留。
- exp2_lifted：约±0.5%，无稳定收益；不保留。
- async_control：DtoH与Rasterizer多Stream重叠逐位通过，但完整无稳定增益；不保留。
- grouped_view：16lane/Anchor约2%增益，8lane不稳定，32lane控制接近原；全部ID/RGB/radii逐位一致。优先16lane。
- REMAINING_RESEARCH_20261115 complete12任务：direct_pose（CPU相机与请求合成单次有界HtoD，减少设备拷贝）约2–3%完整增益、同二进制control约2%增益，逐位一致。
- selector_anchor：每Anchor一次动态首层MMA，比逐Gaussian重算慢约2%（额外Kernel/访存）；逐位一致，不保留。
- wide_graph：把原32768容量上限改为128MiB发布容量，原理成立但与同二进制control相比无稳定收益，不保留；较大空CTA抵消Graph省下的Host成本。
- 两轮有Xorg/gnome间歇干扰，rejected均保留。必须从accepted列表取样，不能只取a1。例如 selector_anchor_r2认可的是a2，当前本地该a2未下载。

## 下一批已写未跑

- tail_strong：阈值3/4/6/8除255；独立Tile与Pixel阈值同步。先开发双轮择门限，通过再冻结，禁止根据保留场景改参数。
- encoded_half2：基于182的提升多项式，直接从bit-exp整数打包半精度Alpha/T/RGB；区别于旧失败half2。最新生成器已修正next表达式加法括号，远端上传的是修正生成器，尚未生成远端源码。
- cub_depth：CUB稳定32bit深度SortPairs替代归并，128MiB scratch和1GiB总上限。每个容量走同一路。cub_depth_edges.py已改为相对182逐位，并检查bitset、directory、counts和稳定sorted IDs。
- tile_clear：每CTA一Tile清零实际K位表，32/64/128/256线程，避免逐字动态整数除法，严格逐位。
- half_prefix：只把冻结Offset+Base线性投影缓存存为FP16，Feature表和权重不变，网络输出允许近似；质量和radii差异单独记录，ID必须相同。
- joint_incremental：组合direct_pose与16lane View，独立关闭两个改动，与原I/同库control比较。
- full_pose：组合以上，并新增完整Full的C++当前位姿入口，直接只读消费冻结full属性，省去旧Full每帧不必要clone与Python Tensor切片；**仍每帧新相机，不复用图像**。返回IDs仍clone、RGB/radii新存储。初始化额外核验Full属性finite。该候选改进Full速度可能让归一化曲线变差，不能隐藏。

## 之后必须完成

- 根据新候选开发结果择优；冻结后才读保留集结果；完成最终32组纯渲染和18组Student联合双轮、质量/慢例/Full/实际K关系。
- `summarize_tail_validation.py`已上传，控制器complete后可汇总但建议等tail_strong开发参数先冻结，防止见到保留集后调参。
- `stress_raster_final.py`已在本地写好，尚未上传：合成大footprint/细长/重叠/同深度/随机/临界Alpha，固定嵌套输入双轮；无GT，要把参考图误差与真实GT质量门限分开。
- 将最终渲染候选接入最佳Student原生路径；至少四组A原FP32+F2/B原FP32+新渲染/C最佳既有TC+F2/D新调度TC+新渲染，另保留当前I182对照及逐组件关闭归因。mode0在PoseRenderer已证实为同输入逐位F2，新的FinalStudentRenderer(0)需逐位验一次。
- 处理新Nsight Profile：reports/*profile_continued_a1.nsys-rep，4份均仅0.3–1.3MB；尚未生成.sqlite。导出分析只在编译/非计时阶段，避免CPU干扰。
- 组件准备/Selector/Compensator/纯Rasterizer与完整延迟分别报告；Nsight累计、Event、同步墙钟分开，不相加混称。
- 任何新最终kernel需Compute Sanitizer，网络新调度需当前帧、GPU K边界/Stream/输出独立性和参数指纹检查。
- verify_integrity.py核验生产/原checkpoint/conda-meta/Student/冻结基线；新增源和实际加载二进制记录hash；零渲染图片。
- 清理仅本轮可重建对象，保留build_joint_final完整对象（后续候选链接复用，需hash），保留基线与完整回退版本。删除前核绝对路径与归属并记清单。磁盘当前约57GiB空闲，无需动其他目录。
- 中文最终报告 LOW_RATIO_FOOTPRINT_AND_TENSOR_CORE_9H_REPORT.md、逐场景逐Ratio CSV、K—耗时/归一化/Pareto图。plots采用figures-python规范450DPI PNG+SVG，已读skill；现有plot_final_render.py和figures可复用作基础，但旧标签本轮182应注明重置前已有收益。
- SCALING_INTERPRETATION.md已明确局部Ratio分母≠全图N，真实LoD与Full不能当同域删点；Full提速可能抬高归一化曲线。所有反序和慢例保留。
- 最终回复若用了memory，末尾必须单个memory citation block；查过MEMORY.md约411–425及539–552，rollout id01a1c0e8-301e-7692-912a-ac2a300e95db。不得写新memory文件（用户没授权）。
