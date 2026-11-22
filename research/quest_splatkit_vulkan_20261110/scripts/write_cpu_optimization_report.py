from pathlib import Path
import csv
import difflib
import hashlib
import json
import math
import statistics
import zipfile

root=Path(__file__).resolve().parents[1]
e=root/'evidence/cpu_optimization'
v=json.loads((e/'VALIDATION.json').read_text(encoding='utf-8'))
def read(path):
    return list(csv.DictReader(path.open(encoding='utf-8-sig')))
paired=read(e/'PAIRED_RESULTS.csv')
final=[r for r in paired if r['source'].startswith('cpu_opt_final_')]
main=[r for r in final if not int(r['fixed_k'])]
gm=lambda values:math.exp(statistics.mean(math.log(x) for x in values))
speed=gm(float(r['speedup']) for r in main)
frontend=read(e/'APK_CPU_SUMMARY.csv')
old=read(root/'evidence/left_eye_shared/CPU_APK_BENCHMARK_SUMMARY.csv')
old=[r for r in old if r['run']=='cpu_inference_1794318528968']
runs=sorted({r['run'] for r in frontend})
table=['| Ratio | 旧前台记录 N=14,294 | 新第一遍 N=14,294 | 新第二遍 N=13,947 |',
       '|---|---:|---:|---:|']
for case in range(5):
    a=next(r for r in old if int(r['case'])==case)
    b=[next(r for r in frontend if r['run']==run and int(r['case'])==case) for run in runs]
    table.append(f"| {(case+1)/10:.1f} | {float(a['cpu_total_ms_median']):.3f} | {float(b[0]['cpu_total_ms_median']):.3f} | {float(b[1]['cpu_total_ms_median']):.3f} |")
native=['| Ratio | 原生原版中位ms | 原生优化中位ms | 配对几何平均加速 | 慢例/配对组 |',
        '|---|---:|---:|---:|---:|']
stage=['| Ratio | 准备 原→新 | Selector 原→新 | TopK 原→新 | Student 原→新 | 解码 原→新 |',
       '|---|---:|---:|---:|---:|---:|']
for ratio in [.1,.2,.3,.4,.5]:
    rr=[r for r in main if abs(float(r['ratio'])-ratio)<1e-6]
    native.append(f"| {ratio:.1f} | {statistics.median(float(r['original_cpu_total_ms']) for r in rr):.3f} | {statistics.median(float(r['optimized_cpu_total_ms']) for r in rr):.3f} | {gm(float(r['speedup']) for r in rr):.3f}× | {sum(float(r['speedup'])<1 for r in rr)}/{len(rr)} |")
    stage.append(f'| {ratio:.1f} | '+' | '.join(f"{statistics.median(float(r['original_'+key]) for r in rr):.3f}→{statistics.median(float(r['optimized_'+key]) for r in rr):.3f}" for key in ['prepare_ms','selector_ms','topk_ms','student_ms','decode_ms'])+' |')
app=['| 运行 | 预算 | 实际K | 准备ms | Selector ms | TopK ms | Student ms | 解码ms | 完整CPU中位ms | CPU p95 ms |',
     '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in frontend:
    label=f"固定{int(r['fixed_k']):,}" if int(r['fixed_k']) else f"Ratio {float(r['ratio']):.1f}"
    app.append(f"| {1+runs.index(r['run'])} | {label} | {int(r['K']):,} | "+' | '.join(f"{float(r[k]):.3f}" for k in ['prepare_ms_median','selector_ms_median','topk_ms_median','student_ms_median','decode_ms_median','cpu_total_ms_median','cpu_total_ms_p95'])+' |')
report=f'''# Quest 3 CPU整体推理优化报告

日期：2026-11-10。独立目录：`research/quest_splatkit_vulkan_20261110`。
当前APK：**0.3.0-cpu-fast**，versionCode=6，包名`ai.mycartgs.quest.splatkit`。

## 实际结果

已完成实现、构建、Quest 3逐位测试、逐帧切换视角的配对测速及前台验收。当前原Selector＋32/32学生Compensator、FP32、4线程，默认优化位掩码221；模型、Ratio语义、严格K、物理解码公式和SplatKit渲染算法保持原样。左眼每帧只推理一次，双眼共享结果。

前台第一遍主要Ratio完整CPU为**3.96–5.82ms**；旧记录为4.94–8.40ms，观察到约20%–34%的延迟降低。第二遍为5.30–6.12ms，显示仍有明显调度／运行状态波动。两次新测试各315次执行、270个正式样本；N分别14,294和13,947。

__FRONT_TABLE__

单位ms，均为完整CPU中位数，不含GPU上传／渲染／OpenXR present。旧记录来自0.2.1，其CPU库与0.2.2相同。旧记录与新第一遍N相同，但视图／投影矩阵最大差约5.96e-8，测量时间和运行状态也不同；此表是实际前台前后观察，**不是严格配对的因果加速证明**。第二遍使用另一个视角，不能与第一遍合并为相同输入的重复测试。全部p95及分项见后表和CSV，没有只选择较快的一遍。

严格原生配对在相同输入下得到主要Ratio **{speed:.4f}×** 加速、延迟降低 **{100*(1-1/speed):.2f}%**。开发视角为2.1376×，保留视角为2.1671×；该结果不等同前台应用或整帧FPS收益。

## 保留的实现

1. **复用同一个线程池并行准备。**视锥平面按帧提取；Anchor可见性／真实视角编码、Selector输入、Student输入分别按块并行。候选仍按原Gaussian行序枚举，UID和同分规则不变。分发只在阶段内部并行，保留Selector→严格K→Compensator的依赖。
2. **输入直接写入XNNPACK自有Buffer。**按冻结地图Gaussian容量预分配带`XNN_EXTRA_BYTES`尾部的FP32输入，免去准备Buffer到网络Buffer的再次复制。每次按当前相机和Ratio重新写入，未使用旧视角Mask或旧推理输出。
3. **精确直方图TopK。**借鉴此前x86中Ratio优化：256桶只缩小边界查找范围，边界仍用原始FP32分数、完整UID和行号比较；最后按原候选顺序输出。小于2048个候选、分数不在[0,1]、边界桶超过eligible的1/4时，使用原比较规则的nth_element路径。分数不量化，严格K不放宽。
4. **有界工作区。**复用可见标记、整数计数、TopK Key／输出行号和解码诊断空间。场景加载绑定容量；每帧覆盖有效区，不缓存跨帧选点结果。更换场景重新绑定模型和地图，GPU异步消费者继续使用独立Decoded结果。
5. **并行物理解码。**每点仍调用同一个原始`DecodeResidual`，保留系统exp/log1p/sqrt、Softplus分支、三次Quaternion归一化、非法Scale和非有限值检查。仅将独立点分块，最后归约原max统计。工作线程异常传回调用线程，按最早失败区间选择异常，异常后池可恢复。
6. **16384行网络批量与4线程。**原批量4096继续作为回退；增加中间激活容量，把原先多个FC／Softmax调用合并为较少的大批量调用。网络权重、层数、激活、FP32类型及输出完全相同；17种输入规模包括4096、16384附近边界均逐位对比。4线程与准备／解码共用一个池，阶段不会嵌套创建线程池。

此前x86的AVX／libmvec指令没有直接搬到ARM。本轮未引入近似exp、rsqrt、FP16或INT8；XNNPACK继续执行其ARM FP32实现。没有修改训练或GPU光栅化算法。

## 未启用候选与反例

- **共享Anchor 48D Feature，位2：**实现并逐位验证，但开关对照约1.005×，较长复测关闭后约1.011×；收益不稳定，默认关闭，相应3.2MB表也不在默认加载时分配。保留研究开关，不把旧x86批量上下文收益当成ARM证据。
- **完整块／尾块独立执行计划，位32：**实现并验证；四线程多次对照启用约0.907×，更大的16384批量也减少了它的必要性，默认关闭。一次短测曾给出相反结论；长复测和全部慢例均保留。默认不创建尾块模型／工作区。
- **16384批量：**相对相同四线程的4096路径，主要Ratio配对几何平均约1.092×，60组中20组更慢；最终选择依据组合路径的运动视角双轮验证，没有宣称它逐项、逐视角普遍加速。
- **两线程：**工程优化本身已有约1.46×原生收益。三线程候选约2.03×、四线程约2.09×属于开发轮次，不能重复加算到最终收益。实际前台网络没有复现原生约2倍加速，相关差距保留，不能把线程数翻倍当成必然吞吐翻倍。

所有开发原始CSV保留在`evidence/cpu_optimization`，`SLOWER_PAIRS.csv`含反例。没有按场景专门改变阈值或只保留较快样本。

## 原生配对协议与结果

同一Replica office2 RGB-D冻结地图：43,197 Gaussian、16,723 Anchor；Teacher／地图／UID绑定和学生SHA校验不变。优化使用相机0–5；保留验证相机8–13含旋转和位移。每个正式样本切换姿态，重新执行候选、准备、网络、选点与解码。

每套两轮、反转相机和版本先后、ABBA分块；每块3次预热和18次记录。测试Ratio .1/.2/.3/.4/.5，以及固定1K/2K/5K/10K。共**15,552个正式CPU样本、216个配对组**，包括固定预算的所有正式组均较原生基线快；主要Ratio共120个配对组。

原生基线使用从0.2.2保存的原始GaussianFrame代码及原4096分块FC调用；双方共用一个线程池，在测量块外切换原2线程／新4线程，避免两个活动池相互干扰。原版XNNPACK副本另用于网络逐位验证。配对基线共享了新后端的有界底层分配，因此此结果属于**原生计算路径对照**，不能写成旧APK与新APK的完整前台加速。细节见`FROZEN_PROTOCOL_CLARIFICATION.json`。

__NATIVE_TABLE__

延迟列为各配对组中位的中位，加速为每组比值的几何平均，两者不一定能直接相除。

__STAGE_TABLE__

单位ms。分项中位数不能机械相加成总中位数。计时不含相机矩阵构造、模型加载、CSV写入、诊断读回、GPU上传、渲染或显示。统计比较和身份／属性检查位于计时外。CPU频率观察到约960MHz至1516MHz变化，设备有两档CPU capacity 848／1024；没有锁频、修改调度器或超频。记录了热状态0及缓存温度，但没有进行长时热稳定性认证。

## 逐位、异常和双眼验证

普通及ASan／UBSan探针均通过：**2305组帧级逐位对照、17种网络规模、12组优化开关、30组TopK ties／饱和／域外回退、128组极端解码**。比较候选行、严格K身份、Selector概率／Score、Student原始残差、最终物理属性和max补偿统计。覆盖空候选、Full、未见Ratio、动态K、4096／16384附近尾部和线程池异常恢复。

ASan／UBSan初次在NDK libc++的`std::ifstream`构造报vptr错误；只有ifstream打开文件的独立最小程序也以同样错误退出134。原始报错及最小复现保留。最终检查仅排除vptr，其他ASan与UBSan检查保持开启；LeakSanitizer关闭，未称为泄漏验收；第三方预编译XNNPACK未重新插桩。成功日志为`ASAN_UBSAN_NO_VPTR.log`。

真实Adreno GPU完成8组双眼：与原2线程后端的左眼选点和属性逐位相同，右眼共享同一结果，左右眼相机独立，每帧两网络各一次、Gaussian属性只上传一次。Full、菜单和空集合门限保持有效。CPU完整9档315次测试也通过。

额外RGBA哈希检查未全部逐位相同；8组中6组在**相同输入、不重新上传、不切换CPU实现**的重复渲染控制中同样出现差异，原版重渲染控制也有差异。源码含subgroup后`atomicAdd(count,survivors)`追加与16位深度排序，这提供了一个与现象一致的顺序变化来源，但尚未逐像素追踪GPU成员列表验证全部因果。

因此本报告将CPU逐位一致与GPU重复渲染差异分开记录，不用GPU非确定性接受任何新的CPU数值误差，也不宣称RGBA全图逐位一致。最大观测RGBA通道差为29/255；原版重复控制最大22/255，完整差异计数和平方误差见`STEREO_CASES.csv`。未修改相关Shader或排序／混合逻辑。未新测GT PSNR／SSIM／LPIPS，RGBA相互差异不能当作GT质量下降。

## 前台完整分项

两遍各30个正式样本／预算；保留所有长尾。N分别14,294与13,947，eligible分别11,151与10,918，固定10K在这两遍均可选满。

__APP_TABLE__

两网络耗时仍只包含Selector＋Student，完整CPU额外包含筛选、枚举、打包、TopK和物理解码。新前台两遍最大CPU p95约9.65ms；没有将原生中位数换算成稳定FPS。

## 空间、完整性与回退

XNNPACK自有输入／激活／输出Workspace从**3,719,312字节**增至**23,564,228字节**，增加18.93MiB；新增Frame可复用工作区约1.66MiB，默认未分配共享Anchor Feature缓存。部分原输入副本不再需要，因此这些逻辑分配不能直接当作进程RSS净增。容量受当前冻结地图和固定16384批量约束，没有无界形状缓存。

安装APK SHA256：`{v['integrity']['apk_sha256']}`。
ARM64库SHA256：`{v['integrity']['renderer_library_sha256']}`。
APK资产与原MCGS SHA一致；603个原始上游文件、14个Shader、5项原项目／资产、11项生产源码保护检查通过。原Selector、32/32学生、Master和UID绑定保持。只修改本独立研究目录；未改训练、生产默认、原checkpoint、共享conda，无commit／push。

原0.2.2 APK和完整源码保存在`evidence/cpu_optimization/baseline`。研究开关：1并行准备、2共享Feature、4并行解码、8可复用TopK、16输入直写、32独立尾块计划、64精确直方图、128大批量；默认221。原始APK可直接覆盖回退，逐位参考代码在`tests/cpu_reference`。

构建与检查：

```powershell
powershell -ExecutionPolicy Bypass -File T:\\scripts\\build_probe.ps1
powershell -ExecutionPolicy Bypass -File T:\\scripts\\build_apk.ps1
powershell -ExecutionPolicy Bypass -File T:\\scripts\\build_cpu_sanitizer.ps1
python T:\\scripts\\analyze_cpu_optimization.py
python T:\\scripts\\validate_cpu_optimization.py
```

`T:`映射本研究目录，`Q:`映射原Android研究目录；NDK／SDK／XNNPACK依赖沿用既有独立构建环境。完整源码、补丁、APK及逐样本CSV均保留。最终冻结后只增加GPU诊断读回，不修改已经计时的CPU数值源码，差异清单写入`VALIDATION.json`。

## 仍需验证与下一处瓶颈

本轮只验证一台Quest 3和一个场景，未将以前x86的Replica16组结果当作ARM证据，也未实测Quest 3S。实际前台Selector约1.42–2.32ms，明显高于原生配对约0.62–0.64ms；接下来应测量OpenXR前台的核心分配、线程调度和频率，再决定是否采用官方线程／性能提示接口。这里尚未完成因果定位，不把差距直接归因于某个系统机制。

准备仍约1.5–2.7ms，并行后的数据扫描、视角编码和布局还有优化空间。后续继续以实际前台完整CPU为门限，分别看原生计算和OpenXR环境；本轮没有用同帧Mask命中、旧视角Feature或流水线吞吐替代每帧真实准备延迟。
'''
for key,value in [('__FRONT_TABLE__','\n'.join(table)),('__NATIVE_TABLE__','\n'.join(native)),('__STAGE_TABLE__','\n'.join(stage)),('__APP_TABLE__','\n'.join(app))]:report=report.replace(key,value)
(root/'report/QUEST_CPU_OPTIMIZATION_CN.md').write_text(report,encoding='utf-8')
clarification=dict(baseline_cpu_preparation='Copied original 0.2.2 GaussianFrame; original strict TopK and decode',
    baseline_network='Legacy 4096 FC/softmax calls, two threads, shared backend allocation and active pool with candidate',
    frozen_original_xnnpack='Independent original code verifies output bitwise; not used as a second active timing pool',
    scope='Native paired computation evidence; actual foreground results are independently recorded, not a paired old/new APK benchmark')
(e/'FROZEN_PROTOCOL_CLARIFICATION.json').write_text(json.dumps(clarification,indent=2),encoding='utf-8')
with zipfile.ZipFile(e/'baseline/source_0.2.2.zip') as z:
    paths=[p for p in (root/'host/app/src/main/cpp/mycartgs').glob('*') if p.suffix in {'.cpp','.h'}]
    paths+=[root/'host/app/build.gradle',root/'tests/CMakeLists.txt',root/'tests/stereo_runtime_probe.cpp',root/'tests/cpu_optimization_probe.cpp',root/'tests/cpu_recorded_cameras.h']
    patch=[]
    for path in paths:
        name=path.relative_to(root).as_posix();before=z.read(name).decode('utf-8-sig') if name in z.namelist() else ''
        patch.extend(difflib.unified_diff(before.splitlines(True),path.read_text().splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    (e/'CPU_OPTIMIZATION.patch').write_text(''.join(patch),encoding='utf-8')
print('WROTE QUEST_CPU_OPTIMIZATION_CN.md and source patch')
