# Meta 官方 SplatSample Quest 3 渲染测速

日期：2026-11-10。独立研究目录：`research/quest_meta_splatsample_20261110`。

## 结论与测试边界

本次测量的是 **Meta 官方渲染器在全局固定 K 静态 Gaussian 子集下的应用双目 GPU 时间**。它不是当前视角预算的局部 LoD，也不是 Selector／Compensator 导出的学习型全局 LoD。

输入为 Replica office2 RGB-D 的 43,197 点原 Master 地图。离线按固定 UID hash 排名取前 K 个点，再恢复原 Master 行顺序，保存成八份 FP32 PLY。各 K 子集嵌套，属性未经补偿。运行时切换 PLY；官方渲染器自行处理当前视角可见性。**输入 K 不等于视锥内 K，更不等于实际参与 Alpha 合成的点数。** 本次未取得官方 SDK 内部的逐帧可见点数或 Alpha 参与点数。

| 路径 | 预算与选点 | 网络 | 本次是否测量 |
|---|---|---|---|
| 本次官方 Renderer 基准 | 整图固定 K，UID hash 静态子集 | 无 Selector／Compensator | 是 |
| 我们的全局 LoD | 按整图 N 计算预算，由网络选择和补偿 | 有 | 否 |
| 我们的局部 LoD | 按当前可见名义 Anchor 的候选点计算预算并选点 | 有，带相应模式／视角输入 | 否 |

## 实际实现与官方来源

- 官方仓库：<https://github.com/meta-quest/Meta-Spatial-SDK-Samples>。
- 官方样例说明：<https://developers.meta.com/vr/documentation/spatial-sdk/spatial-sdk-sample-gaussiansplat/>。
- 固定源码 commit：`822f4d1ec967d4975ae8db386502b60cda5218c0`。
- 实际解析和打包的 Meta Spatial SDK：**0.15.0**。
- 官方 `VRFeature`＋`SplatFeature`＋`Splat(uri)` 负责渲染。日志确认 `SplatRendererFramework::startup`、`UnifiedGaussianSplatter` 与 Adreno 740 初始化。
- JNI 只查询 OpenXR 性能计数器，不执行高斯渲染或 MLP。没有接入原项目的自写 Vulkan Rasterizer。
- 实测 APK 和最终 APK 的官方 native library 均与解析的 0.15.0 官方 AAR 逐 byte 相同。七项来源记录中 zlib 在两个模块重复，实际为六个唯一官方库。最终 APK 与实测 APK 的官方库和 PLY 内容相同。
- 独立包名 `ai.mycartgs.quest.official`，与原 XNNPACK／自写 Vulkan 包 `ai.mycartgs.quest` 并存。

最初界面和 CSV 将 SDK 误标为 0.14.0，来自旧网页预览信息；实际 checkout TOML 和运行库从首次构建起即为 0.15.0。原 CSV 原样保留，`official_run_01_labeled.csv` 仅修正元数据，数据行逐字符核对相同。最终 0.1.1 APK 修正版本标签及统计口径说明；不把它称为新一轮性能复测。

## 输入与正确性检查

原输入：`../quest_native_android_20261110/assets/Replica__office2__rgbd.mcgs`。

SHA256：`1d735b0871ff94cc6acbf039ad014e0d1490ca4e209caef92648a0b9e53a6685`。

`scripts/export_fixtures.py` 验证 MCGS 各段 SHA，使用标准 binary little-endian PLY，保持 XYZ、opacity logits、log scale、quaternion 和 SH 属性。SH rest 从内部 `[N,15,3]` 转为标准 PLY `[N,3,15]` 顺序，导出后 readback 逐 byte 校验。没有经过 SPZ 量化。八份 PLY 合计 20,645,547 bytes。

参考相机到房间坐标使用 entity Transform，模型字段不改。变换检查最大 rotation 差约 `5.75e-8`，camera center 到原点误差约 `6.31e-8`。这只证明坐标转换正确，不等于官方渲染结果与原 Rasterizer 逐像素一致。尚未对官方图像做 GT PSNR／SSIM／LPIPS 验收。

## 测量方法

Quest 3 上完整执行两轮：第一轮 K 递增，第二轮递减；每档加载完成后预热 2 秒、读取约 3 秒。K 为 0／1K／2K／3K／4K／5K／10K／15K／43,197。K=0 隐藏 Splat entity。加载耗时单独记录，不计入稳态 GPU 指标。

测速期间通过公开 Transform 将模型相对当前头部 pose 固定，以减少切换档位时场景投影变化。实际头部 pose 保留在 CSV。该方式不保证与其他渲染器完全相同的预测 pose／FOV，也不能代表连续走动视角下的测试。

测量 API 为 `XR_META_performance_metrics` 的 app CPU/GPU frametime。**它是 runtime 聚合的应用双目指标，包含官方 SDK、面板及相关应用工作，不是纯 Raster kernel 时间。** compositor GPU 独立记录，不与 app GPU 相加充当同一指标。缺失或无效必选计数器会报错，不以目标刷新率替代。

共 3,902 次有效读取，`rejected=0`，逻辑 frame_id 不重复；每档均完成两轮。但全体 GPU 读取值仅有 72 种不同值，说明 runtime 计数器重复读取明显。**3,902 次读取不是 3,902 个独立 GPU 帧样本，表中 p95 是读取分布 p95，不是独立逐帧尾延迟或置信区间。** 聚合窗口由 runtime 定义，未证明其与测试档位边界严格对齐。

SDK 报告节拍中位约 13.889 ms，应用请求 72Hz。不能用这个数推算极限 FPS，也未做长时间热稳定、锁频或掉帧验收。当前 runtime 不支持请求的 app dropped-frame 和 compositor CPU 计数器，对应 CSV 为 NaN，不能当作零。

日志显示官方 SplatFeature 将 resolution 从 1 调为 0.9、MSAA 从 1 调为 2、shadows 从 true 调为 false。本次保留这些官方默认行为，不能把结果与旧版单眼离屏纯 Raster 时间直接相除，宣称严格加速倍数。

## 实测结果

下表为两轮读取合并后的统计，单位 ms。

| 全局输入 K | K / 43,197 | App GPU 中位 | GPU 读取分布 p95 | App CPU 中位 |
|---:|---:|---:|---:|---:|
| 0 | 0% | 1.678 | 1.916 | 0.712 |
| 1,000 | 2.31% | 2.211 | 2.394 | 0.732 |
| 2,000 | 4.63% | 2.283 | 2.496 | 0.754 |
| 3,000 | 6.94% | 2.431 | 2.616 | 0.747 |
| 4,000 | 9.26% | 2.471 | 2.538 | 0.706 |
| 5,000 | 11.57% | 2.548 | 2.608 | 0.733 |
| 10,000 | 23.15% | 3.066 | 3.202 | 0.732 |
| 15,000 | 34.72% | 3.507 | 3.589 | 0.765 |
| 43,197 | 100% | 6.107 | 6.225 | 0.707 |

| K | 第一轮 GPU 中位 ms | 第二轮 GPU 中位 ms |
|---:|---:|---:|
| 0 | 1.689 | 1.668 |
| 1,000 | 2.209 | 2.365 |
| 2,000 | 2.283 | 2.479 |
| 3,000 | 2.415 | 2.586 |
| 4,000 | 2.535 | 2.471 |
| 5,000 | 2.591 | 2.548 |
| 10,000 | 3.054 | 3.178 |
| 15,000 | 3.507 | 3.541 |
| 43,197 | 6.223 | 6.107 |

在本场景和当前投影下，全局输入点数减少对应更低的应用 GPU 时间。K=0 仍有约 1.68ms，说明应用固定开销明显；不能将 K=0 简单相减后称为精确的纯 Gaussian 时间。本次没有比较学习型 LoD 的质量、网络成本、不同场景或不同视角的泛化。

![官方渲染性能曲线](official_gpu_vs_global_k.png)

## 构建、验收与保护

- `:app:assembleRelease :app:testReleaseUnitTest` 成功，三项统计／顺序／CSV 单测通过；APK V2 签名验证通过。APK 使用开发 debug 签名，供侧载测试。
- 最终 APK 版本为 `0.1.1-meta-official`，SHA256：`9e24a41b16302d95d362a0ac2c0d88043b1f5591c7051e7cd8ee40002afafe5a`。设备安装 APK 的 SHA 与该文件一致，启动日志确认 SDK 0.15.0 和 Full 加载。
- 实测包 `0.1.0-meta-official` 的 SHA256：`e37f3883937f47876876f8e47842cf2893727b3b2440bbf9b51debfd2c627d2d`。保留该包以对应本次原始 CSV。最终包未重复进行佩戴完整测速。
- 原 MCGS、原 XNNPACK APK 和 11 项受保护生产文件 SHA 未变，上游 58 项文件 SHA 未变；原应用仍安装。没有修改训练／生产默认，没有 commit／push。
- 构建初期 NDK 默认版本冲突、Kotlin opt-in／API 调用错误已修复，失败日志保留，未混入测速。

主要证据：`evidence/FINAL_INTEGRITY.json`、`OFFICIAL_RENDERER_PROVENANCE_RUN01.json`、`FIXTURE_EXPORT.json`、`REFERENCE_POSE_TEST.json`、`OFFICIAL_BUILD.log`、`OFFICIAL_APK_SIGNATURE.log`、`OFFICIAL_LIVE.log`、`official_run_01.csv`、`RESULTS.csv`、`RESULTS.json`。

## 下一步

若要回答“官方渲染器运行我们全局／局部 LoD 有多快”，需将原 Selector 和学生 Compensator 的输出接入官方 Splat 数据接口或导出对应 PLY，保持点身份、补偿属性、视角和预算一致，另外统计网络准备、数据更新／上传和官方渲染的完整时间。尤其局部 LoD 需要真实切换视角后的重新选点，不能以本次静态子集切换替代。这部分本次未实现，不把当前 APK 描述为已完整迁移的神经 LoD 渲染链。
