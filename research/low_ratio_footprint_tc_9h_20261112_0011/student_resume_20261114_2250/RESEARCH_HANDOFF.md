# 当前接续记录

预算仍为2026-11-14 22:52:35至2026-11-15 07:52:35，香港时间；06:52:35起至少预留1小时。不得因重连或接续重新计时。

本地目录与服务器目录同名 `student_resume_20261114_2250`。服务器位于 `/home/prmi/zwj/experiments/low_ratio_footprint_tc_9h_20261112_0011/`。主线程执行，禁止派生agent。

## 已冻结并跨场景复验

- 182像素核：bit-exp线性变换提升、8x8双像素warp、T符号和T-weight递推。仅两个原开发组调参。18个匹配Student组、10视角、14条件、两轮，主要Ratio相对F2加速1.3154x、相对历史mode38加速1.1601x；主要180组合均获益。平均PSNR损失0.008215dB，最差条件0.202781、最差帧0.288971。旧F2、mode38收益不计为新增。
- 32组Full属性嵌套删点：主要Ratio相对F2 1.1183x、相对38 1.0780x。TUM fr2_xyz mono的Full一轮慢2.3%，原始反例保留。`RENDER_SCENE_RATIO.csv`、`RENDER_PER_FRAME.csv.gz`、`RENDER_SCALING_FIT.csv`、`RENDER_INVERSIONS.csv`均已生成并下载。
- 35张PNG/SVG性能图已在本地figures生成并查看。`plot_final_render.py`最近修改了热图文字对比度，最终应重新生成。
- Student调度配置 `FROZEN_STUDENT.json`：reuse/fused/finish=true、members15；相机缓冲复用、View投影融合、radix小集合收尾、实际GPU K成员压缩清零。原32/32、四独立Head、3534参数不变。
- Student联合18组画质验收完成，I_final平均损失0.008487dB，最差条件0.202781。质量阶段20160次同后端ID/RGB/radii逐位检查通过。

## 活跃及失败控制器

- `JOINT_FINAL02_CONTROLLER.json`由于间歇Xorg GPU/CPU活动，g7_r2四次受干扰后停止。已认可的23个任务保存，9个quality组完成。不可覆盖。
- **当前活跃** `RESEARCH_ACCEPTANCE_CONTROLLER.json`，PID3526504、start_ticks48562437左右（必须读JSON实核，不依赖此近似值）、boot_id789690db-80cc-4794-9b49-e2c0127c7dee。计划80任务：新候选开发→缺失联合补测→Thin入口全覆盖→边界/身份→Sanitizer→Nsight。
- 18组联合缺失补测名称含 `_resumed`。汇总用 `summarize_joint_final.py --controller JOINT_FINAL02 --supplement RESEARCH_ACCEPTANCE`，只从accepted读取，27个联合任务齐后汇总。
- 当前Thin入口正式测量正在进行。不要同时编译或执行GPU实验；前台远端大量Pandas统计也会被CPU监控标记为干扰，应在编译/Sanitizer阶段做或下载后本地分析。
- 监控按1Hz采样，不改变其他进程。Xorg间歇4%–10%GPU，受扰轮次保留并重测。

## 最近候选

- 各种尾色扫描/补全、half2颜色/全状态、warp私有缓存、log-opacity预计算、unroll、4lane prefix scan均未保留；`failure_notes.py`记录早期负结果。
- 输入发布＋clear融合：逐位通过，增益不足1%。协作排序纯Rasterizer首版尾部sentinel未初始化，memcheck定位compactTileBits越界；v2已修复并过边界，但开发主要Ratio慢约10%、Full慢8%–18%，不保留纯渲染版本。
- Chunk2/4/8/16/32循环最好约1.5%，慢例明显，不保留。
- Selector单exp：2200227样本有限输入逐位通过，26项无效输入输出差异但原finite错误仍保留；实际完整速度无收益，不保留。
- Fast publish10/11/12：去Scale log-exp往返/一次旋转归一化/快速exp，开发完整收益约0%，不保留。仅候选源中存在，未进入最终Student。
- ThinDispatchRenderer：复用零元素Tensor、绑定方法、FoV标量；每帧所有相机矩阵/属性仍提交。开发快约6.6%，严格182图像一致；正在18组LoD＋32组nested双轮验证。新增脚本用 `--reuse-quality` 避免重做已逐位相同的PSNR/LPIPS。
- Overflow prefix：共享加载原有前256成员，再按位表处理剩余；主要Ratio基本持平，room0 Full快14%–15%，开发画质合格。**尚未保留集复验**。源码/库 `source_overflow_prefix` / `build_overflow_prefix`。

## 已写但尚未运行的新候选与补充验收

这些本地文件需要上传，待当前控制器结束或定位失败后再串行运行：

- `implement_joint_cooperative.py`、`joint_cooperative_runtime.py`、`joint_cooperative_development.py`：纯渲染已知K时协作排序慢，但联合GPU K保持在容量N内有多个空merge节点，单独检验此边界的收益。必须保留Full代价，不按场景/Ratio绕开。
- `implement_tail_cutoff.py`、`tail_cutoff_runtime.py`、`tail_cutoff_development.py`：1.25/1.5/2倍Alpha弱尾裁剪，同时改Tile阈值和Pixel阈值；近似独立开关，身份/K/radii不变，必须计参与点与交互量。未编译，未测试。
- `joint_native_control.py`：新增已有C++位姿路径＋逐位F2（PoseRenderer mode0）对照、旧G mode38、I_final；避免只与Python拆分C比较。需两轮18组，质量若逐位一致可复用。
- `single_instance_memory.py`：单实例G/I的启动、峰值、投影与Selector前缀显存，避免将多对照同时驻留的峰值当部署成本。9组各2场景即可。

## 当前计划中的可能阻点

- `verify_execution_identities.py`会对实际执行用的 `_selector`、`_compensator`、`_base.tables`逐个hash；如原FP32类属性命名不同，应读取实际参数修正元数据，不改变执行或盲目重跑训练。
- `joint_final_edges.py`覆盖GPU K容量、移动/跳转、Stream与输出独立性；跨Stream错误字符串要匹配实际保护信息，不能绕开保护。
- `verify_integrity.py`保存时若FROZEN_STUDENT的binary为绝对路径，Path拼接会自然采用绝对路径；sources校验原文件不删。
- ncu硬件计数器无权限ERR_NVGPUCTRPERM；不sudo，使用Nsight Systems与静态资源解释，不能虚构硬件计数。
- `analyze_student_profile.py`已修正pixel分类并支持 `--output`；旧分析保留，`student_raster_profile01_a1_ANALYSIS_V2`为正确分类。

## 最后必须完成

最终候选冻结及两轮独立配对；全场景质量/慢例/实际K缩放；正确性与Sanitizer（本计划有182渲染和I_final的memcheck/initcheck/synccheck）。若换成overflow等新像素核，需要重验受影响部分。

所有Student与Selector实际权重、Feature和地图指纹；原生产、原checkpoint、共享conda-meta哈希核验；零渲染图片。所有清理只限本轮目录的确认可重建对象，核绝对路径与uid，记录删除清单。

中文最终报告 `LOW_RATIO_FOOTPRINT_AND_TENSOR_CORE_9H_REPORT.md`、逐场景逐Ratio CSV、K—耗时/归一化/Pareto图。保留反序和Full慢例；嵌套与真实LoD不能混同。不要将低Ratio少点预设为严格线性，不把历史38/TC收益重复计为新增。

没有Teacher性能实验、训练、蒸馏、commit/push/deploy。不操作无关进程。凭据只在SSH进程内存，不写入此文件或输出。
