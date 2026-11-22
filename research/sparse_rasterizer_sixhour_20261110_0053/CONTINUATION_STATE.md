# 最新连续状态（优先于 TASK_STATE.md 旧记录）

**21:54 UTC最终验收状态：全部研究和测试已完成，候选R正式通过，交付包哈希校验进行中。** 原Q/C/G/F结果保留。R有效两轮为async_final03、async_final02，均无外部GPU活动；Full24组两轮均不慢于A/B/G/F/Q。R全44607帧相机和7256图像/选点/radii、Sanitizer和运行加载身份均通过。最终保护1037项0变化、无本轮活跃任务。最终报告LOW_POINT_RENDERING_SIX_HOUR_REPORT.md、CSV和24组最终曲线已生成。不要再启动新性能实验。待交付包下载并逐项哈希验证后才能将goal标记complete；生产、训练、checkpoint、共享环境未改，不commit/push/部署。下方为历史执行记录。

更新时间：2026-11-09 21:04 UTC。六小时开始仍是16:53:10，截止22:53:10；没有重置预算。goal active。SSH bridge PTY74656仍可用，JSON行op=exec/put/get。凭据只在旧桥进程内存，禁止写出。无子Agent、不改生产/训练/模型/共享conda、不commit/push/部署、不操作外部进程。所有文件只写当前research目录。

## 已完成，不要重复

- 真正冻结的渲染器revision2：reports/FROZEN_CANDIDATE.json，普通C和Graph G源码/二进制未变。C配置rv5/cap256/keys/stride/task/sort512/ordered，G512粒度共享workspace，device_clear=True。Tile coverage按A原SASS固定舍入已修复。
- edges04通过826项；sanitizer_mem04和init04各232项0errors。额外async01及sanitizer_async_mem01/sync01各166项，memcheck/synccheck0errors，覆盖真正无中间同步的48帧burst、同地址输入值每帧改动、输出全部存活、footprint255/256/257个Tile。
- formal02 round1、formal04 round2全部24组通过；两轮共822528个计时样本，24192份C/G数值输入及471744次fresh prepare检查。Full每场景每轮相对A和B均通过；G纯渲染1/5/10%相对B分别1.428–1.768、1.366–1.647、1.333–1.580×。prepare+render只提高1.05–1.15×，相机准备占主要开销。
- formal03受Xorg约7%SM活动中止，整轮无效，已完整重跑formal04，无外部活动。保留所有失败证据。
- 旧B在3个非空正式输入RGB不一致，最大abs.00103381276；6行（两轮）记录BASELINE_B_NUMERICAL_DIFFERENCES.csv。另负零和P=0产品背景语义差异单列，不更改B。
- coverage02全24组343份输入Tile所有成员和RGB/radii逐位一致。
- trajectory01覆盖24组44607个全轨迹帧，Full/.01/.05/.1、C/G共356856项RGB/radii逐位一致，298秒，无图片。
- count_scaling01全24组，6视角×6条件，25344样本；相同IDs换成Master属性只作独立归因，未改变最终补偿。1%补偿使Tile实例中位2.431×、Alpha交互2.526×、渲染延迟1.210×，低K不能按K比例预测速度。
- Nsight正式profile_suite02（8案例）无干扰；profile_suite01受Xorg3%活动，不用于kernel时间；nsight_final02为G低ratio，亦无干扰。NCU无计数器权限ERR_NVGPUCTRPERM，未改驱动。GPU_PROFILE.csv和GPU_KERNEL_DETAILS.csv已生成。A/B/C/G低ratio核18/8/6/8，G主机只有1kernel+1graph发射；C纯kernel82.7us vsB142.7us，G80.6us。Q不能混入之前G性能。
- 后续microtile01独立copy/build模块已编译并通过624左右比较（最终读SUMMARY），14400配对样本；8行/4行子Tile普遍慢，放弃，不更新冻结源。source+so位于late_candidates/microtile。清理可删该build/*.o但保留失败摘要。

## 新增固定内参与相机图

F = G renderer + CameraPreparer（camera_preparation.py）：只复用64B固定projection，所有位姿/CPU逆/BMM/torch.inverse/Selector/Comp重新算。内参、分辨率、device、Stream、dtype变化立即重建，只有一个固定矩阵。camera_study01调优5组逐位通过，省约60us。

F已冻结FROZEN_PREPARATION.json，并完成preparation01/02两轮24组，483840样本、12096数值输入，Full相对A/B/G均通过。analyze_preparation.py已运行，PREPARATION_FINAL_VERIFICATION.json：低ratio完整延迟相对B 1.165–1.274×。不得重测相同渲染器纯raster。

Q = G renderer + GraphCameraPreparer（camera_graph_preparation.py）：在F基础上将相同FP32 bmm和linalg_inv_ex放入一个有界CUDA Graph，逐帧仍调用原aten._linalg_check_errors(info,'linalg.inv',is_matrix=True)，不删除错误检查/必要同步。每帧更新view，clone full_proj及camera_center，所有输出独立。每实例一个固定4x4计划，内参/device/Stream等改变销毁重建。没有rigid逆公式近似，没有旧pose缓存。

camera_graph01调优5组比F再快约40–51us；camera_graph_edges01通过1592比较（192随机/仿射/极端条件数矩阵、旧输出存活、8个奇异矩阵原错误消息完全一致、不同Stream重建）。sanitizer_camera_graph01 memcheck1592项0errors，实际耗143s（主线程CPU99%是Sanitizer处理库/图，并非挂死；ptrace诊断被yama拒绝，未改权限或终止它）。

Q已冻结reports/FROZEN_CAMERA_GRAPH.json（21:04:20 UTC）。**当前运行camera_final01 round1**，runner1120692，实际身份以reports/jobs/camera_final01.json为准；脚本final_camera_graph_benchmark.py。A/B/G/F/Q五版本轮换、全24×14条件×18视角，scope pipeline/prepare，8重复。必须等结束且measurement_valid才启动camera_final02 round2。不要创建重复任务，不因JSONrunning就盲目再launch；先看日志/实际进度。

Q相关已准备但未运行：

- camera_trajectory_parity.py：44k所有帧相机四个Tensor逐位，另每25帧和末帧Full/.01/.05/.1核对IDs/RGB/radii，不存图。渲染器此前已经全轨迹，故新路径只额外查相机。
- profile_final_camera.py：Q每帧完整重建输入Nsight20调用，使用`nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o evaluation/Q_final ... profile_final_camera.py --output profile_q01`。
- analyze_camera_graph.py已上传，默认camera_final01,camera_final02，输出FINAL_CAMERA_GRAPH_PIPELINE_RESULTS.csv、FULL_GATES、COUNT_VS_FULL、MEMORY和CAMERA_GRAPH_FINAL_VERIFICATION.json。源码由generate_camera_graph_analysis.py生成沿用相同统计。
- plot_final_results.py仅本地已准备，等待Q结果；生成24场景最终曲线和总图，纯raster来自formal02/04，pipeline来自Q新批次，禁止跨批次构造配对。本地H:/Anaconda3/python.exe -X utf8可用numpy/pandas/mpl，无需新环境。已用figures-python技能，PNG450dpi+SVG。
- integrity_check.py、cleanup_rebuildable.py、frozen_renderer.py、README.md、FAILURE_CANDIDATES.md仅本地，最终需上传。完整性原保护1037项，先后sha无变化。局部tracked11项是原用户修改，后续不动；LOCAL_PREEXISTING_TRACKED_HASHES.json是21点的持续保护快照，不能伪称本轮开始快照。

## 交付还需

1. Q两轮所有场景Full不退化/低ratio、保留19组与无干扰验收；不合格则用已通过F/G，不调门限。
2. Q相机全轨迹与最终Nsight、必要Sanitizer补充；当前renderer mem/init/sync已0。
3. 全部原始CSV、监控、测试/失败摘要打包下载本地；当前ANALYSIS_EVIDENCE.tar.gz已解包formal02/04/count_scaling01/coverage02，PROFILE_EVIDENCE为旧profile_suite01（无效），最终profile_suite02需重新取证。生产/原checkpoint不复制。
4. 最终中文报告LOW_POINT_RENDERING_SIX_HOUR_REPORT.md（尚未生成），包括24组全表、实际K、p95/重复、相对A/B和renderer局部加速与完整pipeline区别、空集合反例、总延迟比自身Full差的反例（G有226/1296）、补偿footprint归因、内存与容量计划cold启动、全部失败。
5. 本轮可重建*.o和SQLite可删，长重复Sanitizer/SASS用gzip无损压缩；cleanup脚本只allowlist本目录，留清单，保留最终两.so、C完整回退、A/B外部基线只读、所有timing/metrics/model。
6. 最终核验1037项生产/模型SHA、冻结source/二进制、无图片、无残留本轮活跃任务，生成delivery包及哈希、下载确认。无commit/push/部署。

wall预算截止22:53:10，不要为了用满时间重复无信息测试。若已全部实现验收和报告，可提前交付并报告实际耗时；不要把未完成任务当完成。

## 21:29 UTC最新（前文21:04已过时，任务尚未交付）

Q两轮camera_final01/02全24已完成、0外部活动、628992样本，Full全部A/B/G/F通过。CAMERA_GRAPH_FINAL_VERIFICATION.json：1/5/10%总延迟相对B为1.2896–1.3866、1.2755–1.3731、1.2648–1.3663×。Q相对自身Full234/1296项更慢，最小.9082×，保留。camera_trajectory01全44607帧、178428相机Tensor、每25帧Full/.01/.05/.1共7256图像/选点/radii逐位通过。profile_q01无干扰，Nsight表final_Q01，Q camera NVTX中位272us vs旧G396us，GPU_API同步仍保留。

async_expansion01通过24项：原A先全部完成，连续候选扩容/分辨率变化、全部输出存活、中间无主动sync。delivery_smoke01用frozen_renderer.py实际只读加载两.so，10项通过，CUDA运行时11080。实际所有项目依赖源码/.so路径及SHA在DEPENDENCY_IDENTITY.json，proc maps和Torch版本在RUNTIME_IDENTITY.json。

最终完整性已运行：1037项0变化、核心冻结source/二进制0变化、无图片/活跃job（这只代表21:22时点，之后新研究需最后再核验）。cleanup_rebuildable.py已执行1次，31项重建object/SQLite清理和长log/SASS无损gzip，释放125986283bytes；清单不能覆盖。再次cleanup要改为追加新清理记录，或不再运行。已验收C/G二进制保留，build/*.o已删，因此常规测试要用native_modules/frozen_renderer只读加载，不调用JIT重建。

初版完整中文报告LOW_POINT_RENDERING_SIX_HOUR_REPORT.md已生成（Q版），DELIVERY_VERIFICATION.json全部pass，尚未package_delivery，也未最终答复/complete goal。原先准备提前交付，但用户要求剩余预算继续定位瓶颈，于21:24又独立研究R。

R = Q相机原数学+`.cuda(non_blocking=True)`替换两处上传，不改Q文件，不延后原inverse info检查；同Stream上传→view copy→graph→check，原不必要H2D后Host等待可删除。camera_async_preparation.py从Q独立复制仅改关键字。async_camera_edges01通过1592项，async_camera_study01在5调优组5760样本均快约7–10us，1–2%。sanitizer_camera_async01通过1592项memcheck0errors、143秒。**R已冻结reports/FROZEN_CAMERA_ASYNC.json于21:29:25UTC，符合原21:38冻结目标。**

**当前正在async_final01 round1**：runner1156527（以reports/jobs/async_final01.json和日志为准），脚本final_camera_async_benchmark.py，A/B/G/F/Q/R六版本，每场景14条件18视角8重复，pipeline+prepare，约7min；完成有效后依序async_final02 round2。R还要求Full不慢于Q（没有放宽旧门限）。不得与其他GPU测试并发。所有核心renderer源码/.so和Q源都冻结不动。

已准备上传：analyze_camera_async.py（默认async_final01,async_final02，输出FINAL_ASYNC_PIPELINE_RESULTS.csv、FULL_GATES、COUNT_VS_FULL、MEMORY和CAMERA_ASYNC_FINAL_VERIFICATION.json）；camera_async_trajectory.py（R全轨迹，待两轮后）；profile_async_camera.py（待Nsight独立capture，预计profile_r01，trace evaluation/final_R01）。

本地分析/交付脚本有最新未全部上传修改：plot_final_results.py自动在R通过时选FINAL_ASYNC_*、最终5曲线A/B/G/Q/R；generate_report.py自动读R并把主体Q替换R、插入R相对Q说明；integrity_check.py新增FROZEN_CAMERA_ASYNC哈希；verify_delivery.py新增R两轮/Sanitizer/轨迹/Profile检查。最后上传这些最新文件再生成/验收。

报告generator还有待严谨微调：R通过时不能照搬原Q相对F节省40–51us/600–850帧摊销，需按R正式F-R差值和R cold startup重新计算；R GPU Profile需在analyze_profiles.py加入R已验证分支（当前只有Q，未更新）。旧C/G/F/Q所有结果保留，不重跑未变渲染器。R若无普遍稳定收益则明确失败，保留Q（已有全部交付证据）。

最新图表已本地画好Q版并view_image检查清晰，服务器figures已上传，总27新图+27初期图PNG/SVG，FINAL_FIGURES.tar.gz33MB。R通过需用新CSV重画（输入变了应重画），更新同一final_scenes目录和DATA_MANIFEST，不拿Q图充R。用户只要求性能曲线，无渲染图片。

最终仍需：R两轮+轨迹+Nsight→分析→最终报告/图→完整性再核验→delivery验证→package_delivery.py打包→下载逐hash核验→goal complete再最终答复。打包只含本轮源、两.so、CSV、日志、少量Profile、图，无模型/数据/共享环境。DELIVERY_HELPERS.tar.gz等上传临时包不纳入最终包。不要再次覆盖CLEANUP_MANIFEST.json。

## 21:46 UTC补充

R的async_final01遭Xorg两次1%SM活动，已核验boot/PID/start_ticks/命令仅SIGTERM自有child1156528；现场reports/ASYNC_INTERFERENCE_INCIDENT.json，旧轮无效。**有效第一轮是async_final03（round1）**，已完成387072样本、0外部活动、Full24组相对Q均快。R对Q低ratio增益约.7–2.1%，部分global和local.7存在<.4%反例，新增ASYNC_VS_Q_REGRESSIONS.csv分析不要隐藏。

**当前async_final02 round2运行中**：runner1173461，child1173462，start_ticks6513807/6513809，已到TUM fr2_xyz。开始21:40:50UTC，预计21:48:30结束；最后看实际进程、JSON和日志。尚无干扰。默认分析脚本已改为async_final03,async_final02，不要使用无效async_final01。

最新helper打包FINAL_ACCEPTANCE_HELPERS.tar.gz已上传解包，包括自动R/Q选择的generate_report/plot_final_results、R完整性、按真实formal_names校验的verify_delivery、create_camera_preparer加载源SHA及验收结果检查。delivery_smoke.py现在若R合格会测试R，需用新任务名**delivery_smoke02**；记录运行TF32标志/precision/autograd/Stream及全部项目依赖SHA。

cleanup_rebuildable.py已改为**追加原清单**，最终可再运行清理R Nsight新SQLite，不会覆盖旧31项。generate_report最后又修正计时样本口径（首批822528包含所有scope，其中纯Rasterizer387072），该最新文件本地尚未单独上传（FINAL_ACCEPTANCE_HELPERS解包为修正前版本）。最终要再上传全部最新helper，或打包最新版。

R冻结源码没有再修改，核心C/G及F/Q也没有修改。新代码只在研究目录。最终仍未package，也未complete goal。截止仍22:53:10UTC，约1小时预算剩余，当前处于最后验收窗口，不再开发新候选。
