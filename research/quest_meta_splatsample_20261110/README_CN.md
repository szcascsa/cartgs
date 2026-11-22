# 官方 SplatSample 独立测速交付

当前版本为 **0.2.1-network-lod**：新增原 Selector＋32/32 学生 Compensator，支持全局／局部 Ratio、全局／局部固定 K，提供实时分段计时、应用提交FPS、LoD更新Hz和实际N/K。

- 新功能报告：`report/OUR_METHOD_OFFICIAL_RENDERER_CN.md`。
- 当前APK：`delivery/mycartgs-meta-official-splat-bench.apk`（已安装；原包名不变）。
- 新结果：`evidence/METHOD_PERFORMANCE_SUMMARY.csv`；两个真机四模式测试原始结果在 `evidence/method_benchmark_run01/` 与 `method_benchmark_run02/`。
- 新源码包：`delivery/mycartgs-meta-official-network-lod-source-0.2.1.zip`。

菜单中选择全局／局部、比例／固定点数，输入预算后点击“应用并渲染”。“我们的方法四模式测速”将使用当前输入Ratio和K，测试四个组合各两轮。持续更新默认开启；测速结束会停止自动更新，可通过复选框重新开启。局部使用当前双眼名义Anchor视锥并集，视角条件来自头部中心。CPU推理由XNNPACK执行，官方SDK依然负责渲染。

**官方SDK的公开资源接口需要重新加载PLY，不能把约72FPS的显示／回调节拍等同于72Hz的新视角选点。** 面板单列CPU推理、PLY打包、SDK加载、LoD提交Hz及选点年龄，相关限制见新报告。

以下为保留的0.1.1静态对照说明；其APK保存在 `delivery/mycartgs-meta-official-splat-bench-0.1.1-rollback.apk`，历史 `META_OFFICIAL_SPLAT_BENCHMARK_CN.md` 和CSV对应该对照，不是当前网络模式的结果。

这是官方 Meta Spatial SDK 0.15.0 渲染性能测试入口。使用整图 UID hash 固定 K 子集；不运行 Selector／Compensator，不属于局部预算 LoD。

- APK：`delivery/mycartgs-meta-official-splat-bench.apk`。
- 原始实测 APK：`delivery/mycartgs-meta-official-splat-bench-run01.apk`。
- 报告：`report/META_OFFICIAL_SPLAT_BENCHMARK_CN.md`。
- 结果：`evidence/RESULTS.csv`、`evidence/official_run_01.csv`。
- 来源和保护检查：`evidence/FINAL_INTEGRITY.json`。

包名 `ai.mycartgs.quest.official`，与原 `ai.mycartgs.quest` 并存。菜单可切换固定输入 K、启动两轮自动测速。A 重置面板位置，B 取消测速；完成后 CSV 位于该包的 external files `official_benchmarks` 目录。

## 本机构建

已验证环境：Windows、JDK 17.0.21+9、Gradle 8.13、Android API 35／build-tools 35.0.0、NDK 23.2.8568313、CMake 3.22.1。Gradle 联网获取已锁定版本依赖；本源码包不是完整离线工具链。

当前研究目录映射为 S:，原 Android 研究目录映射为 Q:。`scripts/build_official.ps1` 使用本轮 `tools/gradle-8.13` 和 `tools/android-sdk`，JDK／NDK 使用 Q: 已有只读工具。映射丢失时可用 PowerShell 原生 `subst` 恢复映射，或修改脚本路径、`local.properties` 和 `app/build.gradle.kts` 的 ndkPath 为实际工具路径。不要改共享 conda。

```powershell
powershell -ExecutionPolicy Bypass -File S:\scripts\build_official.ps1
```

构建包含 Release APK 和三项 JVM 单测。安装脚本会侧载并启动本独立包，仅在需要再次安装时主动执行；打包或查看报告不会部署。

## 源码包的资产恢复

源码 ZIP 不重复包含约 20.6MB 的 PLY、工具链、Gradle 缓存、APK 或旧样例的 SPZ／场景贴图。**构建前须恢复 fixtures**：从已交付 APK 的 `assets/fixtures/` 解压到 `source/SplatSample/app/src/main/assets/fixtures/`，文件 SHA 见 `evidence/FIXTURE_EXPORT.json`。也可在保留原 MCGS 的本机运行 `scripts/export_fixtures.py`，该脚本只读原资产并验证绑定 SHA。无需复制模型到其他任务或修改 checkpoint。

源码包保留官方 MIT LICENSE。官方 native renderer 通过 Gradle 官方依赖加载，不在源码包中伪装成我们自己的实现。最终 APK 已内置所需 PLY，侧载无需恢复 fixtures。

网络版源码包还需从当前APK恢复 `assets/scenes/` 到本研究目录的 `method-assets/scenes/`。加载时按catalog校验MCGS／Student／Teacher绑定。四个已验收ARM64 XNNPACK静态库仍只读引用Q目录，SHA见 `evidence/METHOD_XNNPACK_ARCHIVES.json`；目录、源码和archive不可混用为x86版本。相关第三方许可证已包含于APK assets/licenses及源码包。
