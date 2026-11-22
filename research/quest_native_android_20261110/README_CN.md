# MyCartGS Quest 原生研究版本

当前交付：`0.4.1-point-benchmark`，包名 `ai.mycartgs.quest`。原 Selector 与 **32/32 蒸馏 Student Compensator** 明确运行在 XNNPACK ARM CPU，Gaussian 渲染运行在 Adreno Vulkan；不使用 Unity，不调用 HTP，不静默切换后端。

新增主界面 **VULKAN POINT COUNT TEST**。选择场景后点击即可加载并测试；已加载场景的详情页也有同一入口。自动比较 0/1K/2K/3K/4K/5K/10K/15K/Full，固定相机和原地图属性，不运行网络；界面显示输入 K、投影可见点、GPU Raster 中位数/p95 和相对 Full 的速度。结果保存在 Android 应用 `files/benchmarks/`，可取消或重跑。完整口径见 `report/VULKAN_POINT_COUNT_BENCHMARK_CN.md`。

Quest 3 实机已完成网络执行、严格 K、五组 PyTorch Student 图像对照及手柄交互。用户确认“真实场景正常，能切换预算”。当前仅封装 `Replica/office2/rgbd`；其他场景和 Quest 3S 尚未实测。

## 使用

1. 打开 MyCartGS，使用手柄射线与 Trigger 选择并加载场景。
2. 选择 **RUN CPU LOD**。Ratio 0.1–0.5 与固定 K 可在菜单切换。
3. **FULL PREVIEW** 明确绕过网络，只预览原地图。CPU LoD 面板显示准备、Selector、TopK、Student、解码、投影和绘制等待时间。

## 交付与证据

- APK：`delivery/mycartgs-quest-xnnpack-cpu-release.apk`。
- 中文报告：`report/XNNPACK_ARM_CPU_DEPLOYMENT_REPORT_CN.md`。
- 本次测速功能验收：`evidence/POINT_BENCHMARK_VALIDATION.json`；当前入口：`evidence/VALIDATION.json`。CPU 后端原验收见 `evidence/XNNPACK_VALIDATION.json`。
- 当前源码包：`delivery/mycartgs-quest-point-benchmark-source-0.4.1.zip`；0.4.0 源码包保留。
- 本次源码/交付哈希：`evidence/POINT_BENCHMARK_SOURCE_SHA256.json`、`delivery/POINT_BENCHMARK_DELIVERY_MANIFEST.json`。
- 性能原始记录：`evidence/XNNPACK_APK_EYE_TIMINGS.csv`。

五组对照的选点身份完全相同；这不等同于所有场景的 GT 质量验收。当前 Vulkan 单眼绘制与等待约 40–70 ms，双眼串行，**尚未达到实时 VR**；未完成长期热稳定测试。

## 命令行构建与安装

Windows 路径建议使用 ASCII 映射，例如将本研究目录映射为 `Q:`：

```powershell
Set-Location Q:\
& scripts/test_host.ps1
& scripts/build_apk.ps1 -Configuration Release
& scripts/install_xnnpack.ps1
```

脚本使用本目录 `tools/` 中的 JDK 17、Gradle 8.10.2、Android SDK 34、NDK 23.2.8568313 与 CMake 3.22.1，不修改共享环境。源码包不含工具链、大模型和第三方下载包；复现要求及从 APK 提取原场景的方法见技术报告。Release APK 使用已有研究 debug key 签名，属于实验交付包。

## 历史 HTP 版本

`report/HTP14001_DIAGNOSIS_CN.md` 保留原 FastRPC/SELinux 拒绝证据；旧 APK `delivery/mycartgs-quest-native-gaussians-0.3.1-rollback.apk` 保留。CPU 版本是用户明确选择的新后端，不代表 HTP 14001 已修复。

0.4.0 CPU APK 已保留为 `delivery/mycartgs-quest-xnnpack-cpu-0.4.0-rollback.apk`；历史报告中的 c32a3809… 哈希对应此文件。通用 `*-cpu-release.apk` 现在对应 0.4.1。

全部改动限定于独立研究目录；未改训练、生产默认或原 checkpoint，未 commit/push。
