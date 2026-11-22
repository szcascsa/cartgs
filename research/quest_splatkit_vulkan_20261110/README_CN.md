# Quest 3 SplatKit Vulkan 基线

当前版本：**0.3.0-cpu-fast**。4线程、FP32、16384批量；启用并行准备／解码、输入直写、精确直方图TopK和有界工作区。
CPU前向2305组逐位检查通过，15552个正式原生配对样本与新版540个前台样本已记录。前台两遍仍有波动，不把原生约2.15倍加速等同于应用FPS收益。
最新说明：`report/QUEST_CPU_OPTIMIZATION_CN.md`；新前台数据：`evidence/cpu_optimization/APK_CPU_SUMMARY.csv`；所有原生逐配对数据：`evidence/cpu_optimization/PAIRED_RESULTS.csv`。
原0.2.2 APK和完整源码位于`evidence/cpu_optimization/baseline`，可回退。生产、训练和原checkpoint未改。
每个XR帧只使用当前左眼做一次Selector／Student推理，两眼共享选点、补偿属性及GPU Buffer，各自使用真实相机渲染。
主界面及场景详情页新增 **CPU INFERENCE TEST**：固定左眼相机，测试Ratio 0.1–0.5及固定1K／2K／5K／10K，每档5次预热＋30次测量。
面板显示候选N、实际K、CPU总时间中位数／p95及纯MLP时间；CSV包含准备／Selector／TopK／Student／解码分项。
CSV导出到该应用 `files/benchmarks/cpu_inference_*.csv`。
GPU点数基准和原先两个应用继续保留。
此前0.2.1／0.2.2说明与验证保存在 `report/LEFT_EYE_SHARED_CPU_BENCHMARK_CN.md`，其CPU结果仅对应旧版本。
此前前台两遍正式结果见 `evidence/left_eye_shared/CPU_APK_BENCHMARK_SUMMARY.csv`，不覆盖或混入本轮结果。
若ADB直接pull被Scoped Storage拦截，启动Intent附加 `--ez mycartgs.export_cpu_reports true`，再执行 `python T:\scripts\pull_cpu_reports.py T:\evidence\cpu_optimization\apk_cpu_benchmarks`；只导出最新两个CPU CSV并验证长度和SHA，不读取模型。
原左右眼分别推理的0.1.1 APK保留为 `delivery/mycartgs-splatkit-baseline-per-eye-0.1.1.apk`。

以下说明保留此前渲染基线的测试口径；当前CPU测速和左眼共享的差异以上述新报告为准。

本目录将 `Xget7/splatkit` 固定 commit `8a25c57c6c4f62cf682e4c6159cfcae048f7fefe` 的 GPU 筛选、稳定排序和 Quad 绘制管线接入原生 OpenXR。没有使用 Meta SplatFeature 或原项目自写 compute raster 作为渲染路径。

- APK：`delivery/mycartgs-splatkit-baseline.apk`，独立包名 `ai.mycartgs.quest.splatkit`。
- 原始上游：`source/upstream`，保留不改；工作副本：`source/splatkit`。
- Quest 宿主：`host`。原 Q: 应用和 S: 官方 SDK 应用均保留。
- GPU测速选项：主界面及场景详情页 `GPU POINT COUNT TEST`；CPU测速选择 `CPU INFERENCE TEST`。
- K：0／1K／2K／3K／4K／5K／10K／15K／Full；使用原Master的UID hash嵌套子集，不运行网络，不是学习型LoD质量测试。
- 面板显示输入K、GPU实际drawn、完整GPU管线中位数／p95及相对Full比值，原始CSV另外记录筛选、排序和绘制阶段。

两个已完成的头显内测试原始CSV在 `evidence/apk_benchmarks`；无窗口设备probe在 `evidence/pointcount_1200x680_run02.csv` 和 `run03.csv`。
`run01.csv` 的绘制阶段时间戳位置不适用于Tile GPU，**不作为正式绘制分项结果**。

## 基线口径

采用SplatKit flat world，SH3、RGBA8颜色／opacity、half协方差／SH、16位深度排序、1px subpixel阈值、renderScale=1，不建立内部LoD树。使用原MCGS直接生成SplatCloud，避免SPZ再量化。上游着色、覆盖范围、混合方式与原FP32 Raster不同，不能宣称画质或Alpha参与点数相同。

每帧重新执行GPU筛选和排序，没有复用同帧投影／排序结果。静态世界属性仅在场景或K变化时重新上传；缓存有单一世界边界。Benchmark固定相机、单眼离屏完整管线，排除UI和OpenXR present；普通预览保留头部跟踪与双眼绘制。CPU／提交等待与GPU时间分别记录。

## 构建

将本目录映射为 T:，前一个Android研究目录映射为 Q:。
构建使用独立 `tools/android-ndk-r27b`（27.1.12297006）、Q:已有JDK17／Gradle8.10.2／Android SDK34／CMake3.22.1。
依赖的Vulkan-Headers、vk-bootstrap和VMA源代码位于 `deps`，版本与SHA见证据文件。

```powershell
powershell -ExecutionPolicy Bypass -File T:\scripts\build_probe.ps1
powershell -ExecutionPolicy Bypass -File T:\scripts\build_apk.ps1
```

构建脚本不会部署。需要重新安装时明确运行 `scripts/install.ps1`。
源码包不含工具链、MCGS和第三方二进制；MCGS可从交付APK的 `assets/scenes` 提取，依赖路径、固定源码和只读XNNPACK archive要求见报告。

## 架构适配

SplatKit只增加借用OpenXR Vulkan句柄的Context入口，及为带深度附件的宿主RenderPass显式提供禁用深度／模板的pipeline state；不修改Shader和筛选、排序算法。宿主采用UNORM颜色目标，与SplatKit默认encoded-space混合一致。
GPU计时包围完整RenderPass，包含clear/store；不会使用Tile GPU渲染中途的时间戳假装纯绘制只需2微秒。
