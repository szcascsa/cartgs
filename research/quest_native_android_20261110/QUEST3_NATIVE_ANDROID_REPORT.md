# Quest 原生 Android 当前交付

当前版本：**0.4.1-point-benchmark**。主界面新增 **VULKAN POINT COUNT TEST**，固定视角/原地图属性测量不同 K 的纯 GPU Raster 时间，提供中位数、p95、相对 Full 的速度和 CSV 导出。实现及本轮 Quest 3 原生设备测试见 [点数测速报告](report/VULKAN_POINT_COUNT_BENCHMARK_CN.md)。

原 Selector 与 32/32 蒸馏 Student 保持 XNNPACK ARM CPU；0.4.0 的实景和手柄预算切换已获用户确认。0.4.1 菜单及数值测试范围单独记录，不将原用户确认冒称为新界面的佩戴验收。

CPU 部署报告：[XNNPACK ARM CPU 部署报告](report/XNNPACK_ARM_CPU_DEPLOYMENT_REPORT_CN.md)。本次机器可读结果：`evidence/POINT_BENCHMARK_VALIDATION.json`。仍不能把单眼离屏 Raster 测速当作双目 VR 帧率。

以下保留此前 0.3.1 的 HTP 诊断历史；原文另存于 `report/HTP14001_DIAGNOSIS_CN.md`，原验收 JSON 留档为 `evidence/VALIDATION_HTP_0.3.1.json`。本次是明确更换 CPU 后端，不声称 HTP 14001 已解决。

---

# 历史：Quest 3 HTP 14001 定位与 0.3.1 交付

## 结论

当前设备已能加载 QNN runtime、连接 V69 Stub；**HTP 仍未运行成功**。`14001` 的实际失败链是 FastRPC 设备访问被当前 Horizon OS 的 SELinux 策略拒绝，随后 HAL 打开 session 失败，最终 QNN `deviceCreate` 返回错误。普通应用的 Unsigned PD 与 Signed PD 两种路径均已在同一设备验证失败，不能把“增加 Signed PD 配置”当成解决方案。

**真实 Gaussian 的 Adreno Vulkan 渲染已经接通。** `Full Preview` 是明确选择的 Full 地图预览，不执行 Selector/Student，不是 NPU 失败后的静默回退。头显双目截图中已出现 office2 真实场景；HTP LoD 入口仍保持禁用。

此次可交付的是 0.3.1 诊断与实景预览版，不能称为完成 NPU 部署。继续满足“Selector 和 Student 均在 HTP V69 执行”的要求，需要 Meta/Qualcomm 确认可供第三方 APK 使用的 FastRPC/HAL 授权入口及配套 runtime/签名 DSP 镜像。当前未获得此入口，不承诺单纯更换 SDK、安装权限或系统升级即可解决。

## 设备与版本

- Quest 3，SXR2230P / SoC model 53 / HTP V69 / Adreno 740。
- Android 14 / SDK 34；固件 fingerprint：`oculus/eureka/eureka:14/UP1A.231005.007.A1/52641960034500520:user/abl_signing_keys:release,amss_signing_keys:release,release-keys`。
- 官方 QAIRT Community：`2.40.0.251030`，QNN interface headers 对应 API 2.30；已实际取得并用于 NDK 编译。
- 应用：`ai.mycartgs.quest/.QuestActivity`，`versionCode=4`，`versionName=0.3.1-htp-diagnostic`，minSdk 29，targetSdk 34。
- APK：`delivery/mycartgs-quest-native-gaussians-debug.apk`，78,282,039 bytes。
- APK SHA256：`ab46b78db84bc87074c7101f4daba6caa8c260afcaaed7097228e32ec064fbe9`。已覆盖安装；V2 签名验证通过。

## 已排除的应用侧加载问题

1. 旧版没有真实 QNN adapter；现已实现官方 C API，打包 HTP、Prepare、System、V69 Stub、Skel 和 DSP 算子库。
2. `NativeActivity android:hasCode=false` 导致 vendor linker namespace 不可用；已改为最小 Java `QuestActivity` 引导，并显式声明公开的 `libcdsprpc.so` / `libadsprpc.so`。Java 不执行推理。
3. Quest 不允许读取 sysfs SoC 信息；已在 backend API 前设置官方 SXR2230P SoC model override。`backendCreate` 已成功。
4. `ADSP_LIBRARY_PATH` 包含应用 nativeLibraryDir 与 vendor RFSA 路径；日志已经出现 `First connection to QNN stub established!`。

以上只证明加载和接口初始化阶段通过，不能证明 HTP 图构建、执行或数值正确。

## 直接证据

0.3.1 普通 APK 的 Signed PD 测试进程为 PID 6723；见 `evidence/HTP_SIGNED_APK.log`：

```text
DEVICE_REQUEST soc=53 arch=69 pd=signed
First connection to QNN stub established!
avc: denied { read } name="adsprpc-smd-secure" scontext=u:r:untrusted_app:...
avc: denied { read } name="adsprpc-smd" scontext=u:r:untrusted_app:...
open_hal_session: failed to open session, error -1
open_device_node failed for domain ID 3 ... errno 13, Permission denied
DspTransport.openSession qnn_open failed, 0x00000200
Transport layer setup failed: 14001
TRANSPORT_DIAGNOSTIC error=14001 pd=signed direct_errno=13 secure_errno=13
```

恢复默认 Unsigned PD 后，PID 6883 的同一 APK 仍得到对应拒绝和 `14001`；见 `evidence/HTP_UNSIGNED_APK_FINAL.log`。独立 shell probe 的 Signed PD 也失败，见 `HTP_SIGNED_TRANSPORT.log`；shell 结果仅作辅助，APK 进程才是主要证据。

`/vendor/etc/selinux/vendor_sepolicy.cil` 中相关设备 allow 规则已只读摘录到 `evidence/HTP_OS_ACCESS_POLICY.log`。实际观察到的授权对象为部分系统应用和服务，当前应用属于 `untrusted_app`。不能靠 Manifest 的普通运行时权限改变该域，也未修改固件、SELinux、root、系统节点权限或其他软件。

Signed PD 还要求配套签名 DSP 镜像；本次官方 Community 包只含 unsigned V69 Skel。当前失败发生在打开通道阶段，尚未走到镜像签名兼容性检查，因此不能声称补齐签名镜像后一定可运行。

## 本轮代码改动

所有修改位于独立研究目录 `research/quest_native_android_20261110`：

| 文件 | 实际作用 |
|---|---|
| `app/src/main/cpp/mycartgs/qnn_backend.*` | 真实 HTP C API、SoC override、固定 1024-row Selector/Student 图、设备身份检查、无回退门禁、执行 profile；新增 PD 对照和 FastRPC 访问诊断 |
| `app/src/main/java/ai/mycartgs/quest/QuestActivity.java`、Manifest | 修复 vendor runtime 链接器访问；保留纯原生界面与推理 |
| `gaussian_frame.*` | 每眼新姿态下的准备、选择、补偿、协方差/SH 投影、稳定深度排序及 tile 成员表；修正固定 K 时网络条件为 `min(1,K_requested/N_candidates)`；支持空候选 |
| `vulkan_gaussians.*`、`shaders/raster.comp`、`present.*` | 真 Gaussian compute 合成与 OpenXR 交换链显示，显式同步及有界 Buffer |
| `quest_runtime.*`、`scene_menu.cpp`、OpenXR sample 接线 | 实景预览/HTP LoD 分离，手柄菜单，真实训练视角初始化，PD 诊断 Intent；界面显示具体 HTP 拒绝原因 |
| `decode.cpp` | 对齐现有 CPU Student 后端的 Softplus threshold=20 与 FP32 最小正值 clamp；非有限 Student residual 直接报错 |
| `tests/*`、`scripts/diagnose_device.ps1`、`tools/compare_quest_full.py` | 主机行为测试、设备 HTP/渲染 probe、可重复日志和普通 PyTorch/CUDA Full 图像对照 |

QNN 图为原 Selector 19→32→32→2 与 **Student 62→32→32→14**。Student 输入是逻辑 54D（offset3 + Residual48 + ratio3）加 view7/mode1，按已审计学生后端打包。V69 采用 FP16 内部计算、FP32 IO；这不是 FP32 bitwise 路径。由于 `deviceCreate` 未通过，graphFinalize、graphExecute、NPU 数值误差、选点一致性和补偿长尾均未验证。

## 渲染验证

| 检查 | 结果 | 限制 |
|---|---|---|
| Quest Adreno 独立 Full render，1200×680 | K=43,197，projected=9,622，tile instances=953,727；64 个采样像素对同投影 CPU Alpha 合成 max abs=1.78814e-7 | 仅合成检查，不是完整端到端 bitwise |
| 非整 Tile，641×359 | projected=9,748，instances=303,424，sample max abs=1.19209e-7 | 非整 Tile 边界测试 |
| 同地图同训练相机，普通 PyTorch/CUDA renderer 对照 Quest FP32 图像 | 输出间 PSNR=65.2848 dB，mean abs=1.49590e-4，max abs=0.00388372，p99 abs=0.00226399 | 只含 Full；不是对 GT 的场景 PSNR，也不是蒸馏质量或 HTP 误差 |
| Quest OpenXR 双目 | `evidence/openxr_current.png` 可见左右眼真实 office2 场景，非示例方块 | 0.3.0 渲染路径截图；0.3.1 保留相同 Full 渲染代码，新增诊断/边界修复；最新前台复测状态见 `VALIDATION.json` |
| 主机 C++ | `native_core_test` 与 `scene_menu_test` PASS | 包含 Softplus 极值、非有限值拒绝、严格 K/UID 同分、插值、手柄防抖和菜单容量 |
| Android NDK / Gradle | BUILD SUCCESSFUL；设备安装 Success；签名通过 | 编译成功不能替代 HTP 执行 |

CPU 协方差/SH 投影、ARM 数学库、Vulkan exp 与 CUDA 的计算顺序/舍入不同，图像并非 bitwise 相同。保留原顺序和 FP32 合成；当前 Full 单相机误差已实测，未扩大为所有场景/视角质量结论。

设备 probe 首次 submit+wait 分别为 65.7447 ms 和 20.9891 ms，含首次开销，不能当成稳定帧率；当前未完成可靠的持续运行性能测量。菜单分段性能显示和 CPU—NPU—GPU 零拷贝尚未完成验收，不填造数字。

## 资产与完整性

- 场景：Replica/office2/rgbd，43,197 Gaussian，16,723 Anchor；原 Selector 1,762 参数，Student 3,534 参数。
- Teacher/地图/UID fingerprint：`9e2d413fa7864127465394fa22bb3da4e05d912d4c9e0822a4e928933aa36804`。
- Student checkpoint SHA256：`fd85285c6bd5871f4b41c85690597ccadc965d5a5ec6f474a983e4e09f547a75`。
- 原始 MCGS、打包资产、设备资产 SHA256 均为 `1d735b0871ff94cc6acbf039ad014e0d1490ca4e209caef92648a0b9e53a6685`；33 个 section 及 UID/shape/owner 校验未取消。
- 未修改训练/生产路径或原 checkpoint；已有其他工作区修改未覆盖。没有 commit/push；保留旧 APK。

## 使用与后续条件

1. 打开 MyCartGS，手柄射线选择场景并按 Trigger 加载。
2. `Full Preview` 查看真实 Gaussian 全图；`Run HTP LOD` 只有 QNN 真正就绪才可启用。
3. 本固件下仍会显示 `HTP 14001: FASTRPC / DSP ACCESS DENIED`。此信息不会自动消失；不能将本版误认为 HTP 已修复。
4. 复现默认路径：`scripts/diagnose_device.ps1`；复现 Signed PD：加 `-SignedPd`；明确选择 Full 预览：加 `-FullPreview`。保持头显佩戴和应用前台，`Ctrl+C` 只终止本机日志。

推荐继续取得 Meta/Qualcomm 的第三方 HTP 访问支持后，优先完成最小固定图的 graphExecute/profile 驻留验证，再比较 Student 输出、选点和渲染。若改为 ARM CPU 或 Adreno MLP，需要用户明确更改当前“仅 HTP”要求，本轮没有切换。

## 外部资料

- Qualcomm HTP backend 文档：<https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/htp_backend.html>。
- LiteRT Android XR Signed PD 需求记录：<https://github.com/google-ai-edge/LiteRT/issues/9754>；它不是本设备可运行的证明。
- Meta 官方 Inference Engine 说明：<https://developers.meta.com/vr/documentation/unity/unity-pca-sentis/>；其中 NPU 小节并未提供原生 QNN 的授权解决办法，不能把 Unity 当作当前阻塞的已验证修复。
