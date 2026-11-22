# Quest 3/3S 原生 Android 端侧实现报告

## 结论

本轮在独立目录完成了可重复构建的 arm64 原生 Android **诊断 APK**、真实 checkpoint/场景资产封装、32/32 Student 与原 Selector 的 ONNX 导出、严格绑定校验、OpenXR头部姿态与输入接线、Residual48/Ratio/严格K/物理解码参考实现，以及Vulkan Gaussian shader编译验证。

无法完成或验证 QNN HTP V69 执行、Gaussian live draw、Quest双目画质和持续性能。原因是当前环境没有 Qualcomm QAIRT/QNN SDK、HTP context compiler/runtime，也没有 Quest 3/3S 真机。代码没有加入CPU/GPU MLP回退，APK明确命名为`diagnostic`，不能视为最终部署版本。

## 资产与模型审计

场景为 `Replica/office2/rgbd`：43,197个Gaussian、16,723个Anchor。导出资产包含Master属性、SH、Anchor owner、UID、eligible mask、六级Residual48、原Selector参数及32/32 Student参数。

| 项目 | 校验结果 |
|---|---|
| Teacher/地图/UID fingerprint | `9e2d413f...aa36804` |
| 32/32 Student checkpoint SHA256 | `fd85285c...f547a75` |
| 场景资产 SHA256 | `1d735b08...53a6685` |
| Student参数量 | 3,534，符合32/32直接Head；拒绝64/64 Teacher |
| Selector参数量 | 1,762 |
| Section | 33/33 SHA256通过 |
| UID | 全部唯一 |
| Gaussian→Anchor | 全部owner位于 `[0, 16723)` |

APK运行时重复执行这些校验；整文件、任一section、fingerprint、参数量、网络shape或UID绑定不一致都会终止加载。

## 原生设计

### 输入与选择语义

- Selector输入维度19：静态Gaussian/Anchor信息、连续Ratio、view7和mode1。
- Student输入维度62：归一化offset3、累计Residual48、连续ratio3、view7和mode1。
- Residual48严格按原顺序执行：Base恒为1；高层Residual按相邻knot线性开启，低于前一knot时不访问该层。
- 连续Ratio编码保持 `t, sin(pi*t/2), cos(pi*t/2)`，`t`使用对数Ratio范围 `[0.01, 0.7]`。
- 严格K排序为score降序、UID升序，同分后恢复Master row顺序。非有限score直接报错。
- XYZ加法、physical Scale softplus、四元数delta乘法/规范化、Opacity sigmoid和DC加法均有原生实现。

### OpenXR与交互

- 使用Khronos OpenXR SDK Source `release-1.1.65`，commit `a9f14c4...b52a`。
- Android入口调用 `xrInitializeLoaderKHR`，使用Primary Stereo、每眼swapchain和`xrLocateViews`头部姿态。
- 左Trigger轮转Ratio 0.1/0.2/0.3/0.4/0.5；右Trigger进入并轮转固定K 1k/2k/3k/5k/10k/15k。
- 运行时保留资产、prepare、Selector、Top-K、Student、decode、render分段计时字段。QNN未就绪时字段为0且execution为`NOT_EXECUTED`，不能产生伪性能数据。

### QNN门禁

当前APK检查QNN System、HTP、HTP Prepare、V69 Stub和预编译context资产。任一缺失即阻止Selector/Student调用；`RunSelector`与`RunStudent`不会转到CPU或Vulkan。

由于没有QAIRT SDK，QNN API adapter尚未实现，状态固定停在`ApiAdapterUnavailable`。即使手工放入同名`.so/.bin`，也不会被误报为HTP已执行。

### Vulkan Gaussian渲染

已实现并编译验证Gaussian vertex/fragment shader：四元数与物理Scale生成3D covariance，Jacobian投影为2D covariance，生成3-sigma quad并计算Gaussian alpha。SPIR-V哈希记录在`evidence/VALIDATION.json`。

该shader尚未连接到OpenXR sample的live draw path；深度稳定排序、tile列表、SH视角颜色、premultiplied alpha blend和GPU buffer生命周期仍未实现。因此当前APK仍显示官方sample几何体，QNN失败时使用红色背景，不得称作Gaussian渲染结果。

## 构建与数值验证

### 已通过

- 主机C++测试：SHA256标准向量、严格K、UID同分、K=0/K>N、非有限值拒绝、Residual48插值、Ratio编码、物理解码。
- Android NDK arm64编译：`BUILD SUCCESSFUL`。
- APK：35,149,097 bytes，SHA256 `9bde0d7b49c8506153b9b0ec600ff2c4f4a682b14deffa15b8bd8d5ca576450f`。
- APK签名：debug V2签名验证通过。
- Manifest：minSdk 29、target/compileSdk 34、Vulkan required、headtracking required、仅arm64-v8a。
- APK内场景资产未压缩，SHA与源资产一致；包含`libhello_xr.so`和`libopenxr_loader.so`；不包含任何QNN库或context。
- Selector ONNX与PyTorch，257个随机输入：最大绝对误差 `5.5879e-08`。
- Student ONNX与PyTorch，257个随机输入：最大绝对误差 `7.6294e-06`。
- 两者在 `rtol=1e-5, atol=1e-6` 下通过。该差异属于ONNX Runtime CPU算子顺序差异，不是QNN部署误差。

### 未验证

- Quest 3/3S安装、OpenXR启动、头部追踪和双目显示。
- Selector/Student是否实际由HTP V69执行，以及QNN profile中的backend归属。
- QNN context相对PyTorch/ONNX的选点身份、补偿长尾和最终画质。
- Adreno live Gaussian renderer及CPU-NPU-GPU共享buffer。
- 完整帧延迟、分段性能面板显示、持续运行温度/降频/稳定性。

## 完成最终版本所需外部条件

1. 提供与Quest固件兼容的QAIRT/QNN SDK版本及许可文件。
2. 在该版本上实现并编译QNN API adapter，生成SoC model 53 / HTP V69固定batch contexts，并把所需Android stub与DSP skel按官方路径打包。
3. 在Quest 3/3S真机用QNN profile证明两个graph均在HTP执行；CPU/GPU fallback应作为测试失败。
4. 将已验证shader接入Vulkan稳定排序与alpha合成路径，完成普通PyTorch学生的逐帧同K、选点ID、属性长尾和PSNR/SSIM/LPIPS对照。
5. 实测运动视角、跳转视角、Ratio与固定K切换及30分钟以上持续运行。

在上述条件满足前，保留当前诊断APK作为构建、资产绑定和无回退门禁证据，不进入生产部署。
