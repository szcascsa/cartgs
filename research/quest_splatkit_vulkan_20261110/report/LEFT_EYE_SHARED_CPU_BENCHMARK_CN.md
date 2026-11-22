# Quest 3 左眼共享推理与 CPU 测速验证

历史版本说明：本文记录0.2.1／0.2.2。通用交付APK现已更新为0.3.0，当前性能、验证与回退路径见`QUEST_CPU_OPTIMIZATION_CN.md`。下文原始数据和旧版本身份保持不变。

日期：2026-11-10。当前版本：**0.2.2-cpu-csv-export**，versionCode=5；用户本次CPU测量版本为0.2.1。独立包名：`ai.mycartgs.quest.splatkit`。

## 实际改动与使用方法

主界面和场景详情页新增 **CPU INFERENCE TEST**，原 GPU 基准入口标为 **GPU POINT COUNT TEST**。CPU 测试自动加载所选场景，要求真实 XNNPACK 后端和对应 Student 就绪；不回退到 Teacher、HTP 或 GPU 网络。

CPU 测试冻结开始时的当前左眼相机，以局部名义 Anchor 可见候选建立 N。测试 Ratio 0.1／0.2／0.3／0.4／0.5，以及固定 K=1000／2000／5000／10000。每档 5 次预热、30 次测量，共 315 次执行、270 个正式样本。一个逻辑 XR 帧至多执行一次 CPU 样本。冻结相机用于公平比较预算，每个样本仍重新计算候选、特征、选点和补偿，不复用上一样本的推理输出。

测试期间不上传或绘制 Gaussian，OpenXR 菜单继续显示。面板展示当前进度、候选 N、实际 K、完整 CPU 中位数／p95 和两网络合计中位数。取消时保存已完成的样本并标记 partial；失去焦点时暂停，恢复后另做 5 次预热，不计入正式样本。

CSV 自动导出到：

`/sdcard/Android/data/ai.mycartgs.quest.splatkit/files/benchmarks/cpu_inference_<timestamp>.csv`

CSV 记录 `prepare_ms, selector_ms, topk_ms, student_ms, decode_ms, network_ms, cpu_total_ms`，以及 N、eligible、实际 K、每样本网络调用次数。`conditioning_ratio` 明确记录固定 K 时的条件：`min(1, requestedK/N)`；`ratio` 列在固定 K 行仅为测试计划的占位值，不能拿它解释实际网络条件。文件头保留相机矩阵、场景、Teacher／地图／UID绑定指纹与 Student checkpoint SHA256。

## 时间口径

| 字段 | 包含内容 |
|---|---|
| prepare | Anchor可见性、候选枚举、Selector输入构建，以及选中点的Residual48 Gather、Ratio／视角编码和Student输入打包 |
| selector | 真实XNNPACK Selector后端调用 |
| topk | 分数提取、原严格K／同分规则和选中身份映射 |
| student | 真实32/32蒸馏Student后端调用 |
| decode | 物理解码、输出组织和原有补偿统计 |
| network | 单样本selector+student耗时之和 |
| cpu_total | 整次GaussianFrame::Prepare墙钟时间 |

计时不包含模型加载、初次相机转换、Gaussian GPU上传、渲染或OpenXR present。后端调用时间包含后端自身调度、形状处理与必要后处理，不等同于纯矩阵乘Kernel时间。各分项中位数之和不等于合计中位数。此入口测局部名义Anchor候选，不能直接解释为全局导出成本，也不是连续移动视角的端到端FPS基准。

## 左眼推理与双眼共享

OpenXR获取视图后，在双眼Render loop之前明确取 `m_views[0]`、左眼FOV和尺寸执行一次Selector／Student及解码。两眼消费同一Gaussian rows和补偿属性，并各自更新真实相机、FOV，独立执行SplatKit GPU渲染。

每次BeginFrame使准备状态失效；revision／preparedFrame防止跨帧复用旧结果。同帧重复准备不会再次执行网络，右眼也不执行网络。同帧GPU属性上传复用同一revision，`uploadSerial`验证只上传一次。没有使用左右眼可见候选并集：仅左眼可见的名义Anchor决定本帧候选，右眼独有候选不会补入，这是用户要求的共享语义。

0.2.0真实前台日志验证了39个完整采样双眼帧，每帧Selector／Student各一次、左右眼rows和decoded hash相同、相机不同；用户确认“已打开，两眼正常”。0.2.1增加CPU基准和上传次数诊断；上述用户主观确认不能替代0.2.1新增CPU菜单的前台验收。

## 设备验证与本次测量

Quest 3原生测试直接运行同一QuestRuntime和CPU基准实现，使用真实XNNPACK ARM CPU、FP32、2线程，原Selector 19→32→32→2与Student 62→32→32→14。场景为Replica office2 RGB-D：地图43,197个Gaussian，固定左眼候选N=14,247，eligible=11,119。

8组Ratio／固定K测试完成真实双眼SplatKit GPU绘制，并检查：左眼与原独立左眼准备结果逐位一致；右眼共享选点、补偿属性和revision；两眼相机与FOV独立；每帧一次网络、一次GPU属性上传；新帧不会读取旧结果；同帧重复调用不再推理；Full、空候选和菜单边界有效。测试同时检查两眼非黑像素，不能把这个检查当作GT画质验证。

CPU sweep全部9档通过：45个预热、270个测量样本，每样本两网络各调用一次；时间均有限。原生数据源：`evidence\left_eye_shared\native_cpu_benchmarks\benchmarks\cpu_inference_1794318268740.csv`。以下是**设备原生测试入口**结果，不是前台APK已经完成的CPU sweep。

| 预算 | 实际K | CPU完整中位ms | CPU p95 ms | Selector中位ms | Student中位ms | 两网络合计中位ms |
|---|---:|---:|---:|---:|---:|---:|
| Ratio 0.1 | 1,424 | 3.570 | 6.185 | 1.239 | 0.235 | 1.482 |
| Ratio 0.2 | 2,849 | 7.304 | 11.033 | 2.372 | 0.544 | 3.003 |
| Ratio 0.3 | 4,274 | 7.469 | 11.592 | 2.232 | 0.898 | 3.386 |
| Ratio 0.4 | 5,698 | 6.541 | 12.519 | 1.261 | 0.838 | 2.085 |
| Ratio 0.5 | 7,123 | 7.178 | 8.791 | 1.234 | 1.039 | 2.281 |
| 固定1,000 | 1,000 | 3.402 | 4.012 | 1.235 | 0.161 | 1.395 |
| 固定2,000 | 2,000 | 5.026 | 7.498 | 1.730 | 0.387 | 2.136 |
| 固定5,000 | 5,000 | 8.224 | 13.868 | 1.691 | 1.060 | 3.172 |
| 固定10,000 | 10,000 | 13.031 | 19.837 | 2.599 | 1.719 | 5.834 |

本次部分档位p95明显高于中位数，较大预算也存在非单调波动；没有控制温度、核心频率或后台负载，因此该表用于验证功能与记录本次测量，不作为跨预算单调性、长期稳定延迟或前后加速的证明。正式现场测速请戴上头显，在应用内运行CPU INFERENCE TEST，使用其导出的数据。

## 用户前台APK实测结果（两遍）

用户在Quest 3前台实际完成两次CPU测试，每次315个执行／270个正式样本，共540个正式样本。两个文件均为complete；长度和SHA由应用显式导出后校验，Teacher／地图／UID指纹、Student SHA与原生测试一致。每档30个测量样本均有限、两网络每样本各调用一次，实际K逐行符合min(eligible,requestedK或floor(N×ratio))。

### cpu_inference_1794318528968：N=14,294，eligible=11,151

| 预算 | 实际K | 准备ms | Selector ms | 选点ms | Student ms | 解码ms | CPU完整中位ms | CPU p95 ms | 两网络合计中位ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ratio 0.1 | 1,429 | 2.235 | 1.618 | 0.315 | 0.298 | 0.385 | 4.938 | 6.038 | 2.001 |
| Ratio 0.2 | 2,858 | 2.860 | 1.650 | 0.432 | 0.564 | 0.696 | 6.302 | 7.483 | 2.279 |
| Ratio 0.3 | 4,288 | 3.219 | 1.634 | 0.479 | 0.911 | 0.914 | 7.207 | 7.851 | 2.561 |
| Ratio 0.4 | 5,717 | 3.649 | 1.502 | 0.519 | 1.028 | 1.104 | 7.980 | 8.685 | 2.506 |
| Ratio 0.5 | 7,147 | 3.554 | 1.673 | 0.454 | 1.279 | 1.319 | 8.400 | 10.029 | 2.932 |
| 固定1,000 | 1,000 | 2.430 | 2.097 | 0.381 | 0.250 | 0.331 | 5.701 | 6.688 | 2.336 |
| 固定2,000 | 2,000 | 2.498 | 1.572 | 0.410 | 0.393 | 0.503 | 5.433 | 6.410 | 1.979 |
| 固定5,000 | 5,000 | 3.380 | 1.602 | 0.476 | 0.905 | 0.964 | 7.448 | 9.850 | 2.592 |
| 固定10,000 | 10,000 | 4.009 | 1.547 | 0.483 | 1.650 | 1.751 | 9.672 | 11.149 | 3.228 |

### cpu_inference_1794318539041：N=12,243，eligible=9,563

| 预算 | 实际K | 准备ms | Selector ms | 选点ms | Student ms | 解码ms | CPU完整中位ms | CPU p95 ms | 两网络合计中位ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ratio 0.1 | 1,224 | 2.416 | 1.990 | 0.403 | 0.321 | 0.404 | 5.683 | 6.616 | 2.401 |
| Ratio 0.2 | 2,448 | 2.838 | 2.088 | 0.506 | 0.621 | 0.764 | 6.908 | 9.111 | 2.776 |
| Ratio 0.3 | 3,672 | 3.050 | 1.982 | 0.522 | 0.856 | 1.023 | 7.689 | 8.918 | 2.968 |
| Ratio 0.4 | 4,897 | 2.844 | 1.631 | 0.455 | 0.994 | 1.064 | 7.483 | 9.408 | 3.006 |
| Ratio 0.5 | 6,121 | 2.945 | 1.387 | 0.466 | 1.483 | 1.318 | 7.732 | 9.836 | 3.004 |
| 固定1,000 | 1,000 | 2.378 | 2.090 | 0.407 | 0.246 | 0.350 | 5.743 | 7.855 | 2.562 |
| 固定2,000 | 2,000 | 2.740 | 1.894 | 0.462 | 0.519 | 0.625 | 6.470 | 9.521 | 2.526 |
| 固定5,000 | 5,000 | 2.829 | 1.767 | 0.439 | 1.118 | 1.086 | 7.395 | 10.303 | 3.031 |
| 固定10,000 | 9,563 | 3.324 | 1.358 | 0.123 | 1.714 | 1.736 | 8.408 | 10.211 | 3.151 |

两个测试冻结的左眼相机不同，N分别14,294和12,243，不能作为同输入两次重复进行合并或归因。第一遍Ratio 0.1–0.5的CPU完整中位4.94–8.40ms，第二遍5.68–7.73ms；第二遍低Ratio并没有因N较少而普遍变快，说明本次调度／频率／后台运行波动仍影响耗时。没有用这些非配对数据宣称优化加速。

第二遍固定10K实际9563点，因为该视角eligible只有9563；这与原选点的可选点约束一致，不是丢失CSV或误将K写为10000。第一遍固定1K比2K慢也应保留，不据此声称模型反向扩展。

CPU总时长包括准备、选点、Student与物理解码；纯网络只占其中一部分，不能拿MLP时间代替完整部署准备成本。当前表不包含Gaussian上传、双眼GPU渲染或OpenXR present，不能直接换算应用FPS。

设备Scoped Storage允许列目录但阻止ADB直接读取新建CPU CSV。新增0.2.2显式导出：启动Intent携带mycartgs.export_cpu_reports=true时，只读取最新两个、每个不超过1MiB的cpu_inference_<timestamp>.csv，分块Base64写出专用日志并附字节数和SHA256；提取脚本逐文件校验。未读取模型或其他应用数据，不改变CSV内容。

0.2.2已构建、安装并完成原CSV导出验证；其libhello_xr.so SHA与实际测速的0.2.1逐位相同，只有Java导出入口／版本元数据变化，因此没有重新运行用户已完成的CPU测试。

## 交付身份、保护与复现

- APK：`delivery/mycartgs-splatkit-baseline.apk`。
- APK SHA256：`3eb17e16d8dcccb4e4a20447ca66eed9b31759d441b6b186c76cc81f3f131e71`。
- APK中的ARM64库SHA256：`ed128563d68fe0f5df7b1b09729093f061cd16129b4f54f09dbb26d5da40012c`。
- 旧逐眼版：`delivery/mycartgs-splatkit-baseline-per-eye-0.1.1.apk`，可独立回退。
- 新验证：`evidence/left_eye_shared/VALIDATION.json`。
- 原生逐预算结果：`evidence/left_eye_shared/CPU_BENCHMARK_SUMMARY.csv`。
- 用户前台两遍结果：`evidence/left_eye_shared/CPU_APK_BENCHMARK_SUMMARY.csv`；原始CSV在`apk_cpu_benchmarks`子目录。
- 双眼8组原始记录：`evidence/left_eye_shared/NATIVE_CASES.csv`。
- 原始实机前台共享日志：`evidence/left_eye_shared/APK_LIVE.log`与`APP_STEREO_PARITY.json`。
- 原0.2.1构建／安装／验签：`APK_BUILD_FINAL.log`、`INSTALL_FINAL.log`、`SIGNATURE_FINAL.log`；0.2.2为`APK_BUILD_022.log`、`INSTALL_022.log`、`SIGNATURE_022.log`。
- 源码与证据归档：`delivery/mycartgs-splatkit-baseline-source.zip`，清单见`DELIVERY_MANIFEST.json`与`SOURCE_CONTENTS.json`。

完整性审计确认：603个原始上游文件、14个Shader、5项原项目／资产保护项和11项生产源码均与保护清单一致；先前两个独立应用仍保留。只更改本独立SplatKit研究目录，不修改训练、生产默认或原checkpoint，无commit／push。

构建命令（T:映射本研究目录，Q:映射原独立Android研究目录）：

```powershell
powershell -ExecutionPolicy Bypass -File T:\scripts\build_probe.ps1
powershell -ExecutionPolicy Bypass -File T:\scripts\build_apk.ps1
python T:\scripts\audit.py
python T:\scripts\validate_shared_cpu.py
```

构建脚本本身不部署。设备原生probe目标为 `build-probe/stereo_runtime_probe`，使用当前MCGS；不需重新训练或修改模型。源码包不含工具链、大模型和第三方二进制，沿用本研究目录既有构建依赖，详见原SplatKit报告。

本任务未开展新一轮GT画质／PSNR验证、正式前后端到端加速或长时热稳定性测试；SplatKit渲染精度／排序口径与原FP32 Gaussian Rasterizer仍按历史报告披露。`FINAL_VALIDATION.json`保留历史渲染基线的验证范围，新共享与CPU测试结论以本报告和`left_eye_shared/VALIDATION.json`为准。
