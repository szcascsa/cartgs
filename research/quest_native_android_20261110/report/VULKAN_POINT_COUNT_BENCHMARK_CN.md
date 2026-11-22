# Quest 3 主界面 Vulkan 点数测速

日期：2026-11-10。版本：`0.4.1-point-benchmark`，versionCode 6。

## 功能与操作

主界面新增 **VULKAN POINT COUNT TEST**。手柄先选场景，再点此按钮，应用自动加载并开始测试；已加载场景的详情页也有同一入口。

自动测试输入 K：**0、1K、2K、3K、4K、5K、10K、15K、Full**。小地图将超出 N 的预算截为 N 并去重。K=0 用于量化固定开销，Full 为原地图全部 Gaussian。

每个 K 运行两轮：第一轮升序、第二轮降序；每轮每 K 预热 3 次、采样 15 次，共 30 个有效样本。office2 的九个 K 合计 324 次 GPU 渲染，其中 270 次用于统计。每个 XR 帧只测试一次，不将左右眼重复计数。

测试时菜单、头部和手柄继续跟踪；**CANCEL / BACK** 取消并保留部分结果，再按 BACK 返回；完成后可 **RUN AGAIN**。失去焦点时停止采样，恢复后额外丢弃 3 次预热，不将休眠后冷启动样本混入统计。

界面显示输入 K、投影可见数量、GPU 中位数、p95 和 Full 中位数除以当前中位数的倍率。测试进行中若 Full 尚无有效样本，倍率显示 `--`。部分结果不会伪装成完成结果。

## 测量口径

- **相机、分辨率、背景和 Gaussian 属性固定**。从原 Master 地图按固定 UID hash 排序取嵌套前缀，恢复 Master row 顺序后投影。各 K 的共同点具有完全相同的 XYZ、Scale、Rotation、Opacity、SH；不运行 Selector 或 Compensator，不重新训练。
- 从已加载场景开始测试时冻结开始时的视角；从场景列表直接测试时，沿用该场景的初始训练相机配准和当前 OpenXR FOV。测试期间移动头部不改变实验相机，菜单仍响应真实头部姿态。
- `VISIBLE` 是经过投影和屏幕 Tile footprint 检查后留下的 splat 数，不等于实际 Alpha 合成点数。输入 K 包含视野外点；原始 CSV 同时记录活跃 Tile 和 Tile 实例数。
- **GPU MS** 使用 Vulkan timestamp 包围原 `Dispatch` 及其内存屏障。它是离屏 Raster 时间，排除 CPU 准备/上传、MLP、全屏 Present、菜单、OpenXR 合成和等待；不是双目帧延迟或 FPS。
- CSV 的 `upload_ms` 是 Host 上传耗时；`submit_wait_with_ui_ms` 在 APK 中包含同一次提交内的菜单及等待，不能当作纯 GPU 时间。独立 shell probe 没有菜单，该列只含本次提交与等待。
- 每个 K 的固定输入仅准备一次。CSV 的 `prepare_once_ms` 是该 case 的准备元数据，在样本行重复记录，**不能相加或伪称为每帧端到端准备时间**。这是固定 fixture 测速，不以同帧复用冒充正常切换视角性能。
- 硬件时间戳不支持或读取失败时，界面明确报错，不以 Host 时间替代 GPU 时间。原正常 LoD/Full 推理路径不创建或写入查询。

这是点数与 Raster 负载的控制实验，**不是各 K 的 LoD 质量对比**。原地图属性与网络补偿属性的 footprint 可能差异很大，因此不能用本实验结果替代真实补偿 LoD 的帧时间。

## CSV 导出

结束或取消后，应用写入自身 external files 下的 `benchmarks/pointcount_<时间戳>.csv`，不需要额外存储权限，不保存渲染图片。界面显示文件名；日志 `POINT_BENCHMARK_EXPORT` 包含完整路径。

Quest 3 常用路径：

```text
/sdcard/Android/data/ai.mycartgs.quest/files/benchmarks/
```

```powershell
& tools/android-sdk/platform-tools/adb.exe pull /sdcard/Android/data/ai.mycartgs.quest/files/benchmarks ./quest-benchmarks
```

文件包含场景、完成/取消状态、固定相机、测量口径，以及每次原始采样的 round、K、warmup、投影点、活跃 Tile、Tile 实例、分辨率和各项时间。统计时必须排除 `warmup=1`。

## 实机验证

设备为 Quest 3 / Adreno 740。其 Graphics+Compute queue 支持 **48-bit timestamp，period=52.0833 ns**。独立原生 probe 使用与 APK 相同的 `GaussianFrame`、`VulkanGaussians` 和统计类，不依赖 QNN 或 MLP。

已通过：

1. Full 预算的投影属性、Tile ranges、members 与原 Full 路径逐位一致。
2. 启用 timestamp 前后的 Full RGBA 逐位一致。
3. 所有 K 的身份嵌套、实际数量、反向轮次身份复用、空输入、GPU Buffer 复用。
4. 1200×680 和非整 Tile 的 641×359，各 324 次采样，所有 K 均得到 30 个有效样本。
5. 主机测试：预热排除、两轮调度、median/p95、截断去重、空地图、取消/重跑、非有限时间拒绝、32/64-bit timestamp wrap、按钮 Trigger/就绪/容量检查。
6. Android Release 编译、V2 签名验证、覆盖安装与启动；设备 APK SHA256 与交付一致。原网络、解码、严格 K、Residual48、场景校验、shader、正常 Prepare/Project 与 0.4.0 源码核对未变；相机公式提取前后运算保持原顺序。

APK 新界面的原生菜单网格和手柄命中已通过主机验证；没有把用户对旧 0.4.0 实景的确认冒称为新测速界面的佩戴验收。安装后的 OpenXR 启动状态记录在日志中；完整 APK 内菜单并行采样和 external files 导出尚需用户实际启动该按钮确认。GPU 测速和 CSV 内容已经在独立 Quest 原生 probe 上实测。

## 实测结果

输入为 `Replica/office2/rgbd`，N=43,197，原地图固定相机。以下仅为该输入的独立 shell probe 实测，单位 ms：

| K | 1200×680 投影点 | GPU 中位数 | p95 | 641×359 投影点 | GPU 中位数 | p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.2676 | 0.3151 | 0 | 0.0944 | 0.1408 |
| 1,000 | 208 | 0.4913 | 0.5418 | 210 | 0.2769 | 0.3946 |
| 2,000 | 441 | 1.2564 | 1.2590 | 443 | 0.6897 | 0.9809 |
| 3,000 | 653 | 2.0568 | 2.0596 | 658 | 1.1175 | 1.5907 |
| 4,000 | 880 | 2.7353 | 2.7386 | 891 | 1.4938 | 2.1241 |
| 5,000 | 1,093 | 3.4522 | 3.4557 | 1,108 | 1.8856 | 2.6704 |
| 10,000 | 2,244 | 6.4866 | 6.4940 | 2,277 | 3.3137 | 5.0580 |
| 15,000 | 3,347 | 9.6654 | 9.6792 | 3,393 | 4.0838 | 6.6985 |
| Full 43,197 | 9,622 | 26.9898 | 27.0136 | 9,748 | 9.8638 | 14.0178 |

1200×680 下，Full 的 Tile 实例为 953,727，1K 为 13,421；减少实例解释了明显的 Raster 时间差异。该场景固定原始属性时，降低点数确实缩短了 Raster 延迟，但收益同时受可见点和 footprint 决定，不能只用输入 K 外推其他场景。

测试前仅停止本应用，不停止其他进程；系统合成器仍可能运行，没有锁频或长期温控测试。641×359 的中位数与 p95 差距较明显，原始波动完整保留，不能宣称为稳定双目帧率或进行跨分辨率性能归因。

原始 CSV、逐 K 汇总和日志分别位于 `evidence/point_benchmark/`、`evidence/POINT_BENCHMARK_DEVICE_*.log`。

## 实现、容量与回退

新增 `point_benchmark.h` 管理有界实验计划和统计。`scene_menu.*` 增加入口和结果页；`quest_runtime.*` 管理固定相机、单帧一次采样、取消和 CSV；`gaussian_frame.*` 增加仅测速用的固定属性子集；`vulkan_gaussians.*` 增加按需创建的两项 GPU timestamp query；OpenXR Vulkan 接线将测试离屏 Dispatch 与正常显示分开。

新增空间有界：一个两项 query pool、最多 N 个 uint32 UID 排序索引，以及最多 324 个样本；没有按视角或点数无限缓存。query pool 随 Vulkan 对象销毁；UID 顺序在场景 Bind 时清空；重跑清空统计；更换场景重建 fixture。每次读取时间戳前等待原提交 fence，GPU Buffer 沿用原有容量上限和同步依赖。

原训练、生产默认和 checkpoint 未改，未 commit/push。旧 0.4.0 APK 保留在 `delivery/mycartgs-quest-xnnpack-cpu-0.4.0-rollback.apk`。源码、APK 和原始测试证据可由 `delivery/POINT_BENCHMARK_DELIVERY_MANIFEST.json` 核对。
