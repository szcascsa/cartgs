# Quest 3 XNNPACK ARM CPU 部署报告

日期：2026-11-10。研究目录：`research/quest_native_android_20261110`。

版本说明：本文保留 0.4.0 部署验收数据。该版 APK 已归档为 `delivery/mycartgs-quest-xnnpack-cpu-0.4.0-rollback.apk`（SHA256 c32a3809…）；通用 Release 文件名现对应新增点数测速的 0.4.1，详见 `VULKAN_POINT_COUNT_BENCHMARK_CN.md`。

## 1. 结论与交付范围

已将原 Selector 与 **32/32 蒸馏 Student Compensator** 接入官方 XNNPACK C API，在 Quest 3 的 ARM CPU 执行；实景由原生 Adreno Vulkan 绘制，OpenXR 提供头部跟踪和双目显示。用户在头显中确认：**“真实场景正常，能切换预算”**。日志同时记录两个网络实际调用和左右眼 GPU 提交。

CPU 是用户明确选择的后端，不调用 HTP，也不将 HTP 失败静默回退为 CPU。未引入 Unity，未重新训练或蒸馏。采用直接 XNNPACK 算子 API，**没有生成 ExecuTorch `.pte`，不应将本实现称为 ExecuTorch 部署**。

当前 APK 仅打包 `Replica/office2/rgbd`。Quest 3S、多场景、长期热稳定及稳定双目帧率尚未验收。网络与图像对照误差很小，但完整帧仍受 Vulkan 绘制及等待限制，尚未达到实时 VR。

## 2. APK、设备和模型身份

| 项目 | 实际值 |
|---|---|
| APK | `delivery/mycartgs-quest-xnnpack-cpu-release.apk` |
| 版本 | `versionCode=5`，`0.4.0-xnnpack-cpu` |
| 包名 / Activity | `ai.mycartgs.quest/.QuestActivity` |
| ABI / SDK | ARM64 / minSdk 29 / targetSdk 34 |
| 大小 | 34,303,195 bytes |
| SHA256 | `c32a3809e6f4e9988f04aa0731dc009e707aae18aa7d5af94e37aa0b24f2cf95` |
| 安装核对 | ADB 覆盖安装成功；设备 `base.apk` SHA256 与交付一致；V2 签名验证通过 |
| 签名性质 | 已有研究 debug key；属于实验 Release 包，不是商店正式签名 |
| 实测设备 | Quest 3，SXR2230P / Adreno 740 / Android 14 SDK 34 |
| 固件 | `52641960034500520` |
| 地图 | 43,197 Gaussian，16,723 Anchor，SH degree 3 |
| Selector | 19→32→32→2，1,762 参数 |
| Student | 62→32→32→14，3,534 参数；未误用 64/64 Teacher |

场景 MCGS SHA256：`1d735b0871ff94cc6acbf039ad014e0d1490ca4e209caef92648a0b9e53a6685`。

Teacher/地图/UID fingerprint：`9e2d413fa7864127465394fa22bb3da4e05d912d4c9e0822a4e928933aa36804`。

Student checkpoint SHA256：`fd85285c6bd5871f4b41c85690597ccadc965d5a5ec6f474a983e4e09f547a75`。

加载时保留整文件 SHA、33 个 section SHA、shape、owner 范围、UID 唯一性及 Student schema/参数量校验。导出文件、APK 内 asset 和设备测试 asset 一致。原 checkpoint 路径和完整哈希见 `assets/Replica__office2__rgbd.manifest.json`。

## 3. 实现与数据流

每眼使用当前 OpenXR 姿态重新计算名义 Anchor 候选和视角条件，然后执行：

`候选/输入准备 → Selector → 严格 TopK → Residual48 Student 输入 → Student → 物理解码 → 协方差/SH 投影、排序和 Tile 表 → Vulkan 合成与显示`。

保留连续 Ratio 编码、Residual48 原顺序累加与插值、视角/模式条件、固定噪声、严格 K、分数同分时 UID 升序、最终 Master row 顺序和物理解码。固定 K 使用当前候选数量计算网络 Ratio 条件。初始化时的静态地图属性随场景卸载失效；没有复用旧视角 Mask、选点或补偿输出来制造加速。

XNNPACK 使用 FP32 fully-connected，前两层通过输出下界 0 融合 ReLU。Selector 最后加原固定噪声，再执行 FP32 softmax。Student 首层采用导出资产中已存在的 encode/ratio/view 权重拼接；四个学生输出 Head 的权重按原输出顺序拼接，没有使用 Teacher 权重。

加载时创建算子并打包权重。线程池持久复用，默认 2 线程。内部最大批量 4,096 行，尾块按真实行数 reshape；超过上限分块执行，不以补齐行改变 K。输入和中间激活由后端持有，提供 `XNN_EXTRA_BYTES` 尾部空间。空输入不增加执行计数，重新绑定场景后清空旧执行状态；输出非有限值直接报错。

工作区容量为 **3,719,312 bytes（3.55 MiB）**，与场景候选 N 无关；仅包含后端输入/激活/输出容量，不含 packed weights、线程栈、地图、帧准备和 Vulkan Buffer。重载场景时重建算子和权重。CPU 算子在线程池完成后才进入解码和渲染；当前每眼 GPU submit 后显式等待，未声称实现 CPU—GPU 零拷贝流水线。

APK 只含 `libhello_xr.so`、`libopenxr_loader.so`，不含 `libQnn*.so`；没有 GPU MLP。菜单明确显示 XNNPACK ARM CPU，Full Preview 明确绕过网络。`XNNPACK_ARM_CPU_EXECUTED` 仅在实际非空调用后设置。

### 主要文件

| 文件 | 改动 |
|---|---|
| `inference_backend.h`、`qnn_backend.h` | 公共状态/接口，保留历史 QNN 实现 |
| `xnnpack_backend.*` | ARM64/NEON、FP32 FC/softmax、分块、持久工作区、线程池和有限值检查 |
| `cmake/Xnnpack.cmake` | 固定依赖静态链接，关闭未使用的 ARM 扩展及基准/测试构建 |
| `quest_runtime.*`、`scene_menu.*` | CPU LoD 入口、场景加载、分段计时、明确错误提示 |
| `gaussian_frame.*` | 公共后端接线；保留选择、补偿、解码和投影语义 |
| OpenXR 副本 `main.cpp`、`graphicsplugin_vulkan.cpp`、`CMakeLists.txt` | Intent 测试入口、双眼接线、提交等待计时、XNNPACK 链接 |
| `app/build.gradle`、Manifest | CPU 版身份、ARM64 Release、排除 QNN 库和 DSP vendor 声明 |
| `tests/xnnpack_device_probe.cpp`、`probe_backend_io.h`、渲染 probe | 动态行数、重绑定、输入/输出捕获、PyTorch oracle 回放；捕获不进入 APK |
| `tools/validate_xnnpack.py` | 普通 PyTorch CPU FP32 对照及严格 K/身份核对 |
| `scripts/build_apk.ps1`、`install_xnnpack.ps1` | 构建、许可证打包、明确安装 CPU 版本 |

## 4. 正确性验证

### 网络与边界

Quest ARM64 probe 已覆盖 `rows=0,1,17,1023,1024,1027,4096,4097,43197`、动态尾块、重复调用、空输入计数和重绑定。1/2/4 线程均通过。主机 `native_core_test` 与 `scene_menu_test` 通过，包含严格 K/UID 同分、Residual 插值、Softplus 极值、非有限值拒绝、双手柄 Trigger、防抖、页面容量和就绪门禁。

普通 PyTorch CPU FP32 对照使用同一导出学生权重，分别执行首层拼接版本和原 encode/ratio/view 三次投影版本，区分首层融合舍入与 XNNPACK 执行差异。测试行数为 1,027、4,097、43,197。

| 项目 | 最大绝对误差 |
|---|---:|
| Selector 概率 | 2.1159649e-6 |
| Student 对原分拆首层 PyTorch 输出 | 1.1444092e-5 |
| PyTorch 拼接首层对原分拆首层，43,197 行样本 | 7.6293945e-6 |

全部输出有限。**不是 bitwise 等价**：ARM FP32 FMA/矩阵归约以及原首层投影相加的舍入顺序存在差异。未引入半精度或量化。这里比较部署误差，不包含此前 Teacher→Student 蒸馏质量损失。

### 实际视角的选点与图像

相同地图、相机、背景和 Vulkan rasterizer，分别喂入 XNNPACK 与普通 PyTorch Student oracle 输出。分辨率 1200×680；五组均保持选择身份完全相同、严格 K 正确。

| 条件 | 候选 N | K | 选点身份 | 输出间 PSNR/dB | mean abs |
|---|---:|---:|---|---:|---:|
| Ratio 0.10 | 9,075 | 907 | 一致 | 110.6386 | 4.6070e-7 |
| Ratio 0.30，camera shift=1 | 8,839 | 2,651 | 一致 | 104.6886 | 7.7596e-7 |
| Ratio 0.50，shift=-1 | 9,233 | 4,616 | 一致 | 95.3385 | 3.0520e-6 |
| 未见 Ratio 0.20，shift=4 | 8,171 | 1,634 | 一致 | 98.9307 | 3.3757e-6 |
| 固定 K=1,000 | 9,075 | 1,000 | 一致 | 100.6685 | 2.1410e-6 |

最高单像素绝对误差为 0.0078463，不能仅报告均值。真实视角 Student 对原 PyTorch 输出 max abs 不超过 6.67572e-6；probe 记录了最大 XYZ 补偿和 Scale 比率。当前样本未出现非有限残差，但这不构成全场景补偿长尾分布验收。

表中 PSNR 是**两后端输出图像之间的一致程度**，不是相对 GT 的场景 PSNR，也不是蒸馏质量。五个冻结视角不能外推所有场景、轨迹或预算。Alpha 合成采样对 CPU 检查最大误差不超过 2.4e-7。

### APK 现场与最终包

用户实际佩戴的 0.4.0 版本执行了两个网络，并切换 Ratio、固定 1,000/2,000/5,000 等预算；用户确认实景正常。`XNNPACK_APK_LIVE.log` 记录逐眼执行和提交，未发现对应 FATAL EXCEPTION、Fatal signal 或本任务非有限/网络错误。中途 NativeActivity 重建没有证据可认定为崩溃。

最终收尾修正仅涉及陈旧 requestedK 遥测、初始化失败信息/对象生命周期、移除无用成员和构建默认 Release；没有改变网络或渲染数学。最终包重新编译、验签、安装并核对设备 SHA256；启动日志进入 OpenXR IDLE，头显随后暂停，**没有将最终安装后的休眠状态冒称为再次完成双目画质验收**。五组数学对照沿用未改变的实现，不重复无信息价值测试。

两次 CPU 版截图分别捕获系统界面和休眠黑屏，不能充当实景截图。双目运行证据是实际执行日志和用户确认；历史 0.3 Full 预览截图不能替代 CPU LoD 截图。

## 5. 性能实测及界限

### 纯网络微测

原生 shell probe，每 case 2 次预热、9 次测量取中位。单位 ms，输入/输出拷贝和有限值检查计入后端调用，不含视角准备及绘制。

| 行数 | 线程 | Selector | Student |
|---:|---:|---:|---:|
| 1,027 | 1 | 0.188750 | 0.367396 |
| 1,027 | 2 | 0.108333 | 0.211406 |
| 1,027 | 4 | 0.125260 | 0.217500 |
| 43,197 | 1 | 6.909840 | 13.468900 |
| 43,197 | 2 | 4.446040 | 9.042760 |
| 43,197 | 4 | 4.007340 | 6.777920 |

两线程在小预算上更合适，作为 APK 默认；大负载四线程有优势，但不能据 shell 微测判断全部 APK 负载。shell 允许 CPU 为 0–5，APK 当时为 3–5，不能将该表当作 APK 帧延迟或宣称固定加速倍数。

### APK 每眼实际日志

以下是用户交互时的稀疏样本，姿态和预算不同，单位 ms。Draw 为 Vulkan submit＋wait 的 Host 时间，不是纯 GPU kernel 时间。CPU total 包含准备、两个网络、TopK、解码和投影。

| 候选 N | K | Selector | Student | CPU total | Draw＋wait | 每眼合计 |
|---:|---:|---:|---:|---:|---:|---:|
| 14,294 | 1,429 | 1.693 | 0.495 | 19.415 | 114.471 | 133.886 |
| 11,017 | 1,101 | 1.980 | 0.235 | 10.459 | 50.814 | 61.273 |
| 10,559 | 3,167 | 2.667 | 0.961 | 15.637 | 57.005 | 72.642 |
| 13,240 | 1,000 | 3.058 | 0.192 | 12.469 | 52.040 | 64.509 |
| 10,121 | 1,012 | 5.321 | 0.388 | 13.965 | 42.083 | 56.049 |
| 11,081 | 1,000 | 2.677 | 0.565 | 10.589 | 40.447 | 51.037 |
| 8,535 | 5,000 | 2.264 | 4.434 | 21.853 | 69.359 | 91.213 |

首行含冷启动。其余样本中两网络合计约 2.2–6.7 ms/眼，投影约 4.8–10.8 ms，绘制与等待约 40–70 ms/眼。完整字段见 `evidence/XNNPACK_APK_EYE_TIMINGS.csv`。这些不是相同负载配对基准，不能计算可靠 p95、稳定 FPS 或据此认定改变 Ratio 必然提高速度。

两眼目前串行准备、提交和等待，不能把单眼耗时直接换算双目帧率。主要剩余瓶颈是 GPU 绘制等待和 CPU 投影；本轮没有修改 rasterizer、训练或模型。渲染 probe 的网络/准备计时含测试用文件捕获 IO，不能拿它代替 APK 性能。

采样时应用 TOTAL PSS 229,743 KiB（约 224.4 MiB）、RSS 337,132 KiB、Swap PSS 2,995 KiB。这是单次快照，**不是峰值**。后端自有工作区 3.55 MiB 可准确计数。未完成 30 分钟持续运行、温控/降频和稳定双目帧时间测试。

## 6. 构建、依赖与复现

工具链：JDK 17.0.21+9、Gradle 8.10.2、Android SDK/build-tools 34、NDK 23.2.8568313、CMake 3.22.1。固定依赖：

| 依赖 | Commit |
|---|---|
| XNNPACK | `a5e4de3149dc68a26b063177e102c16c6cee594e` |
| cpuinfo | `731932997e55e84900e6f95aefb6edd167b0edcc` |
| pthreadpool | `15a6644ba1c45f1acc16ac1e883efc3e56c6bed2` |
| FXdiv | `63058eff77e11aa15bf531df5dd34395ec3017c8` |

下载 SHA256 固定于 `third_party/XNNPACK_DEPENDENCIES.json`，`tools/fetch_xnnpack.py` 校验后解压。四项许可证随 APK 打包；OpenXR 使用研究目录内 1.1.65 副本及实际改动。CPU APK 构建不需要 QAIRT。

OpenXR 继承给依赖的 warning-as-error 曾阻断 XNNPACK 编译；只对依赖 targets 解除相关警告升错误，不改数学或启用 fast-math。关闭当前 NDK 不适用且未使用的 SVE/SME/BF16/I8MM/KleidiAI 路径，保留 FP32 NEON。Release 编译、链接、V2 签名和安装通过。

```powershell
Set-Location Q:\
& scripts/test_host.ps1
& scripts/build_apk.ps1 -Configuration Release
& scripts/install_xnnpack.ps1
```

源码包含项目源码、Gradle/CMake、shader、脚本、测试、修改后的 OpenXR 副本、固定依赖清单和精简验收证据。不含 Android/JDK/Gradle 工具链、QAIRT、XNNPACK 下载大包、原 checkpoint 或重复场景模型；不是无需依赖的离线环境。将上述工具链放到 `tools/` 对应目录，并创建 `downloads/`。包内 OpenXR 已含实际改动及内置外部源码，不要覆盖为未修改版本。

场景可从交付 APK 提取，保持原字节：

```powershell
@'
from pathlib import Path
from zipfile import ZipFile
p = Path('app/src/main/assets/scenes/Replica__office2__rgbd.mcgs')
p.parent.mkdir(parents=True, exist_ok=True)
with ZipFile('delivery/mycartgs-quest-xnnpack-cpu-release.apk') as z:
    p.write_bytes(z.read('assets/scenes/Replica__office2__rgbd.mcgs'))
'@ | python -
```

网络 probe 命令接口：`xnnpack_device_probe scene.mcgs output-prefix threads`。普通 PyTorch 对照：`python tools/validate_xnnpack.py --asset scene.mcgs --prefix output-prefix --output parity.json`。已有对照使用服务器 Python 3.11 / Torch 2.3.1，以 CPU 执行，未修改共享 conda。历史 QNN probe 和混合渲染 probe 的 CMake 仍引用独立 QAIRT；CPU APK 本身不依赖它们。

## 7. 完整性、回退和未验证项

本轮前已有的 11 个生产/训练/Viewer 文件逐一 SHA256 复核不变；服务器原 Selector、原 PLY 和 Student checkpoint 哈希重新核对一致。场景资产未重导出或修改。证据见 `evidence/XNNPACK_INTEGRITY_FINAL.json`。未 commit、push 或修改生产默认。

历史 HTP APK 保留为 `delivery/mycartgs-quest-native-gaussians-0.3.1-rollback.apk`，SHA256：`ab46b78db84bc87074c7101f4daba6caa8c260afcaaed7097228e32ec064fbe9`。如需手动降级，ADB 要显式允许降级；本轮没有回退到该包。原 HTP 验收 JSON 和 14001 诊断报告保留。

后续需单独研究：多场景/Quest 3S、同负载连续运动帧时间与热稳定、完整补偿长尾分布、双眼并行与跨帧流水线、CPU 投影和 Vulkan rasterizer。HTP 权限问题未解决，但已不作为当前 CPU 后端部署的前置条件。
