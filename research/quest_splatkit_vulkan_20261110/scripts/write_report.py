from pathlib import Path
import json

root=Path(__file__).resolve().parents[1]
summary=json.loads((root/'evidence/POINT_COUNT_SUMMARY.json').read_text())
validation=json.loads((root/'evidence/FINAL_VALIDATION.json').read_text())
def table(source):
    lines=['| 输入 K | 实际绘制点 | Cull ms | Sort ms | RenderPass ms | 完整 GPU ms | 完整 GPU p95 ms |',
           '|---:|---:|---:|---:|---:|---:|---:|']
    for row in [x for x in summary if x['source']==source]:
        lines.append('| '+ ' | '.join([f"{row['k']:,}",f"{row['drawn_min']:,}"]+
            [f"{row[k]:.3f}" for k in ['cull_gpu_ms_median','sort_gpu_ms_median','raster_gpu_ms_median','total_gpu_ms_median','total_gpu_ms_p95']])+' |')
    return '\n'.join(lines)
report=f'''# Quest 3 SplatKit Vulkan 基线与不同点数渲染测速

日期：2026-11-10。独立目录：`research/quest_splatkit_vulkan_20261110`。

## 交付结论

已实际复用 [Xget7/splatkit](https://github.com/Xget7/splatkit) 的 Vulkan 渲染模块，构建并安装独立原生 OpenXR APK，加入不同输入点数的测速入口。
其baseline是**上游GPU可见性筛选／压缩＋稳定Radix排序＋Indirect Quad绘制**，不是此前 Meta SplatFeature，也不是我们的 compute Rasterizer。

Quest 3已完成两次头显内点数测试和两次正式无窗口原生测试。
正式计时共 **4,140个测量样本**，另保留预热样本。
头显1680×1760测试的第二次运行中，Full输入43,197点、实际drawn9,328点，单眼离屏完整GPU中位4.531ms；5K输入、drawn1,058点为0.731ms；1K输入、drawn204点为0.278ms。
这是当前场景和视角下实测的点数成本，不能据此宣称其他地图或同画质等价加速。

当前APK：`delivery/mycartgs-splatkit-baseline.apk`，版本 **0.1.1-splatkit-baseline**，独立包名 `ai.mycartgs.quest.splatkit`。
原CPU／自写Vulkan应用 `ai.mycartgs.quest`、官方Splat应用 `ai.mycartgs.quest.official`均保留。
该新包作为**本研究的SplatKit渲染基线**，未覆盖生产默认、训练或原checkpoint。

## 源码身份与最小适配

上游固定commit：`8a25c57c6c4f62cf682e4c6159cfcae048f7fefe`。
完整只读checkout位于 `source/upstream`，603个文件SHA已复核；工作副本位于 `source/splatkit`。

实际参与baseline的上游模块：

- `VulkanWorld`：SplatCloud打包与GPU上传。
- `VulkanCameraUniforms`：相机输入。
- `VulkanVisibilityPass`、`VulkanFrameCompute`：GPU筛选、投影、压缩、诊断和间接参数。
- `VulkanRadixSort`：GPU稳定深度排序。
- `VulkanSplatPipeline`：投影记录展开与硬件Quad混合。
- `VulkanRenderTarget`：离屏颜色目标。

上游渲染代码只改3个文件：VulkanContext.h／cpp新增借用OpenXR Vulkan句柄的入口，VulkanSplatPipeline.cpp显式提供全部禁用的DepthStencil state，适配宿主含菜单深度附件的RenderPass。
Context只销毁自己的VMA allocator和pipeline cache，不销毁宿主device／instance。
**14个Shader及其include文件SHA与上游逐byte相同**，未改算法、fragment逻辑或排序实现。
工作副本ARCHITECTURE.md已补充本适配。

OpenXR宿主基于既有独立原生Android研究版复制，保留头部跟踪、双目、手柄射线、场景菜单、Full预览和原局部CPU LoD示例入口。
新增桥位于 `host/app/src/main/cpp/mycartgs/splatkit_gaussians.*`。
正式点数测试绕过Selector／Compensator。
本轮**未将官方SDK版新增的任意全局／局部文本输入面板迁移到这个原生宿主**，原官方版该功能仍保留；本轮交付重点是SplatKit基线与点数测速。

## 基线策略与画质边界

使用SplatKit上游flat world策略：

| 项目 | 本次值 |
|---|---|
| Hierarchical LoD | 关闭，不构建内部树 |
| 输入子集 | 原Master，UID hash嵌套子集，恢复Master行序 |
| SH | degree3 |
| 输入分辨率／renderScale | 指定原尺寸／1.0 |
| Depth排序 | 上游默认16bit，稳定Radix |
| Subpixel阈值 | 上游默认1px |
| Color／opacity存储 | 上游RGBA8 |
| Covariance／SH存储 | 上游half；未启用SH8实验 |
| Fragment精度 | 上游mediump／RelaxedPrecision |
| 混合 | 上游从后到前over，encoded颜色空间 |
| 显示目标 | UNORM，单采样，与上游默认颜色策略一致 |

这套渲染器与原FP32 Alpha合成有固有差异：颜色／opacity量化、half协方差／SH、16bit排序、1px小footprint剔除、Quad覆盖和fragment截止等。
采用它作为baseline不等于证明与原渲染器逐位一致。
本次没有做GT PSNR／SSIM／LPIPS，不将速度结果描述为相同画质收益，也不把drawn称作实际Alpha参与点数。
没有通过SPZ额外量化：直接从已校验MCGS的原属性生成SplatCloud，再走上游打包。

## 输入、预算及缓存

场景Replica office2 RGB-D，43,197 Gaussian／16,723 Anchor。
MCGS SHA256：`1d735b0871ff94cc6acbf039ad014e0d1490ca4e209caef92648a0b9e53a6685`。
K为0／1K／2K／3K／4K／5K／10K／15K／43,197。
UID hash子集与此前官方SDK及原Vulkan点数测试一致；原生probe逐K UID SHA已核对。

本次是**全局静态子集的renderer成本测试**，不是学习型全局LoD或局部LoD质量测试。
输入K、GPU筛选后的drawn、实际Alpha参与点数是三个不同量；本次只记录前两者。

静态世界属性只在场景／K／补偿输出变化时重传，持有一个world、compute workspace和可复用pipelines；世界替换前等待相关提交完成。
这是有界的静态模型驻留，不缓存旧视角的筛选、排序或投影。
**每个测量帧重新执行GPU筛选／投影、Radix排序和绘制**。
K变化时workspace重新分配，加载／准备／上传与稳态GPU时间分别记录。

## 测量与时间戳修正

GPU计时直接使用Vulkan Timestamp，支持性从实际device和queue读取；不支持则报错，不用FPS替代。
记录Cull（含SH／投影和压缩）、Sort、完整RenderPass及完整GPU管线。
完整管线还含所需Barrier、间接参数和诊断复制，不保证分项中位数之和等于总中位数。
绘制分项包围**整个RenderPass，包括clear／store**，不声称只有fragment执行时间。

初次probe把绘制起点放在RenderPass内部，Adreno Tile GPU上读出几乎恒定约0.002ms，无法代表实际绘制。
已修正为Compute完成后的pass外起点和pass后终点。
初次 `pointcount_1200x680_run01.csv`保留失败口径但**排除正式汇总和曲线**。
正式使用run02／run03，图像非黑像素与RGB总和诊断放在计时外，没有保存渲染图片。

原生无窗口probe每档15次预热＋100次测量，K升序／降序各一轮，两次独立执行，共3,600测量样本。
头显内菜单测试每档3次预热＋15次测量，升序／降序各一轮，两次执行，共540测量样本。
头显每个XR逻辑帧只采一个离屏测试，不把左右眼重复计为同一视角两样本。
Benchmark离屏测试为**单眼**，排除菜单／present；CSV另记含菜单提交等待，不能把它当纯GPU时间。
正常预览双眼按宿主逐眼绘制，不能将单眼离屏毫秒直接换算成最终双目FPS。

设备：Quest 3，Adreno740，Vulkan1.3.295，driver `0x80345009`，subgroup64，共享内存32KiB。
原生probe和APK的CPU调度、设备频率和图像视场不同，必须分别报告。
短时测试未锁频、未记录完整温度／频率时间序列，未做持续热稳定；实测波动保留，不将不同运行的差异都归因于代码。

## 实测结果

### 头显测试：1680×1760，第二次完整运行

{table('pointcount_1794315668552.csv')}

第一／第二次头显运行的Full中位分别约5.076／4.531ms，15K约3.183／1.872ms，5K约0.763／0.731ms。
Full实际drawn为9,329／9,328，说明固定参考视角仍有头部预测和初始映射的微小差别；运行状态／频率也未严格控制。
全部原始样本和两轮分项保留，不能只选较快一轮当稳定上限。

### 原生设备probe：1200×680，最终运行

{table('pointcount_1200x680_run03.csv')}

此参考相机与头显FOV不同，Full实际drawn5,904点，不能把两组差异完全解释为分辨率。
1200×680中5K输入的完整管线约0.59ms，Full约1.99ms；可见点及大footprint覆盖共同决定速度。
空K仍有筛选／排序调度和清屏固定成本，因此低点数延迟不会随K线性降到零。

![不同点数渲染速度](SPLATKIT_POINT_COUNT.png)

此前Meta官方测试的6.11ms是包含SDK和面板的**双目应用系统聚合指标**；本表是单眼离屏GPU时间，且渲染策略不同。
禁止直接相除宣称SplatKit相对官方渲染器的严格加速倍数。
相同K只保证输入数量，不保证不同renderer筛选后drawn、像素覆盖或渲染质量相同。

## 验证、失败项及未验证范围

- Quest3上游相机投影测试通过。
- 上游SplatPipeline测试通过：Gaussian衰减、透明／LoD core、subpixel cutoff及CPU-order／projected图像对照；深度状态适配后重新执行通过。
- 上游Radix测试144个case通过，包含稳定性、无效输入、动态容量和frame slot；最大4M全容量case明确跳过。
- 上游FrameCompute测试通过：16／32bit链路、range、空视图、诊断、资源共享和LoD策略；本baseline不启用内部LoD。
- 两次正式probe覆盖全部9档K及两轮，K=0图像非黑像素为0，非零K有有效覆盖，drawn不超过K。
- 两次头显菜单测试均完整完成，CSV各324行，270个非预热样本。
- 原生测试Release没有可用Vulkan validation layer，不能称已通过Validation Layer零告警验收。
- 实际佩戴操作还运行了局部XNNPACK学生推理和双眼GPU提交，日志有实际调用；尚无本轮用户主观画质确认。
- 上游12M参考世界、大地图、多场景、长时间温度／功耗、Quest3S和对GT画质均未验证。

构建问题已解决并保留日志：旧NDK23无法处理上游返回unique_ptr的隐式move，使用上游要求的独立NDK27.1；旧OpenXR `ALooper_pollAll`替换为NDK27支持的pollOnce；vk-bootstrap被宿主全局warning-as-error误伤，依赖目标单独解除该策略；空世界SH buffer必须使用degree0。
头显CSV目录初始权限阻止ADB读取，已由应用创建时恢复可读／可遍历权限，并成功取回原始结果。

## 版本、保护及复现

当前APK SHA256：`{validation['apk_sha256']}`。
设备安装base.apk SHA与交付一致；签名为研究debug key的V2 Release包。
官方SplatKit渲染器库未提供二进制，本项目从锁定源码和Shader编译静态链接进libhello_xr.so，实际库SHA见FINAL_VALIDATION.json。
原0.1.0头显实测包保留为 `delivery/mycartgs-splatkit-baseline-run01.apk`。
最终0.1.1补充UNORM显示策略、显式禁用DepthStencil、读取权限及遥测；离屏数学不变，但不冒称最终包又跑过同一套头显测试。

原上游603项文件、原研究模型／应用5项SHA和生产源码11项SHA均未变；两个原应用继续安装。
没有修改训练、生产默认或原checkpoint，没有commit／push。
所有修改、构建和新依赖在本独立目录。

源码／构建脚本见README_CN.md、scripts/build_apk.ps1及build_probe.ps1。
GPU原始样本见 `evidence/POINT_COUNT_SAMPLES.csv`；逐运行逐K汇总见 `POINT_COUNT_SUMMARY.csv`；上游及保护验收见 `FINAL_VALIDATION.json`。
源码包不含工具链和模型；模型可从APK提取；第三方源码和许可证保留，依赖版本由manifest锁定。

## 下一步

本次已经建立可用的SplatKit baseline和点数成本测量入口。
后续若要比较官方Splat与SplatKit，需统一双目相机／分辨率／混合和质量策略，单独测完整应用帧，并补充图像质量对照。
若要替换我们完整推理应用，则在这条直接GPU上传路径上迁移现有全局／局部预算面板和Student推理，使推理耗时、属性更新、完整GPU时间与实际drawn同帧对应，保留本次纯renderer测试作为基准。
'''
(root/'report/SPLATKIT_VULKAN_BASELINE_CN.md').write_text(report,encoding='utf-8')
print('中文SplatKit基线报告已生成。')
