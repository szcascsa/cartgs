"""由真实设备结果写中文报告及汇总表，不重新测速。"""
from pathlib import Path
import csv
import hashlib
import json
import statistics

root = Path(__file__).resolve().parents[1]
evidence = root/'evidence'
summary=[]
keys=['cpu_total_ms','prepare_ms','selector_ms','topk_ms','student_ms','decode_ms','ply_ms','sdk_load_ms','app_gpu_median_ms','app_fps_median']
raw_readings=[]
tables=[]
for run in ['method_benchmark_run01','method_benchmark_run02']:
    data=json.loads((evidence/run/'results.json').read_text())
    lines=['| 模式 | 预算 | 候选 N | 实际 K | CPU完整 ms | Selector ms | Student ms | App GPU ms | SDK加载 ms |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for global_scope in [True,False]:
        for kind in ['ratio','fixed_k']:
            rows=[c for c in data['cases'] if c['global']==global_scope and c['budget_kind']==kind]
            first=rows[0]; s={k:statistics.median(c[k] for c in rows) for k in keys}
            s.update(run=run,scope=first['scope'],budget_kind=kind,requested_ratio=first['ratio_requested'],
                     requested_k=first['fixed_k'],candidates=first['candidates'],eligible=first['eligible'],actual_k=first['k'],
                     readouts=sum(c['readouts'] for c in rows))
            summary.append(s)
            label='全局' if global_scope else '局部双眼并集'
            budget=f"Ratio={first['ratio_requested']:.1f}" if kind=='ratio' else f"K={first['fixed_k']}"
            lines.append(f"| {label} | {budget} | {first['candidates']:,} | {first['k']:,} | {s['cpu_total_ms']:.3f} | {s['selector_ms']:.3f} | {s['student_ms']:.3f} | {s['app_gpu_median_ms']:.3f} | {s['sdk_load_ms']:.1f} |")
    tables.append('\n'.join(lines))
    for case in data['cases']:
        for frame,gpu,cpu,fps in case['gpu_readings']:
            raw_readings.append(dict(run=run,round=case['round'],scope=case['scope'],budget_kind=case['budget_kind'],
                requested_ratio=case['ratio_requested'],actual_k=case['k'],frame_id=int(frame),app_gpu_readout_ms=gpu,app_cpu_readout_ms=cpu,app_callback_fps=fps))
for name, rows in [('METHOD_PERFORMANCE_SUMMARY.csv',summary),('METHOD_GPU_COUNTER_READOUTS.csv',raw_readings)]:
    with (evidence/name).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
validation=json.loads((evidence/'METHOD_FINAL_VALIDATION.json').read_text())
report=f'''# 官方 SplatSample 接入 Selector／学生 Compensator 与四模式预算

日期：2026-11-10。当前 APK：**0.2.1-network-lod**，包名 `ai.mycartgs.quest.official`。改动全部位于本独立研究目录。

## 交付结论

已把原 Selector 与 32/32 蒸馏 Student Compensator 接入官方 SplatFeature 0.15.0。四种模式为**全局 Ratio、局部 Ratio、全局固定 K、局部固定 K**。支持输入任意 `(0,1]` Ratio 或非负整数 K，以及 0.1—0.5 比例快捷键。当前仅打包 Replica office2 RGB-D 场景。

面板新增候选 N、eligible、实际提交 K、预算缺口、CPU准备／Selector／TopK／Student／解码、PLY打包、SDK加载、更新墙钟、应用提交 FPS、系统 App GPU 时间、LoD提交频率和选点年龄。原 UID hash 静态子集基准仍保留，可作为独立对照。

Quest 3 上已完成 **Ratio=0.3 和 Ratio=0.4 各一套四模式×两轮，共16个测试项**，全部成功导出；用户还实际触发了多个比例和固定预算的场景更新。日志证实模型执行和官方 Splat 加载完成。没有收到用户对本轮主观画质的单独确认，不把执行日志称为 GT 画质验收。

## 模型身份与网络路径

| 项目 | 身份 |
|---|---|
| 场景 | Replica / office2 / RGB-D |
| Gaussian／Anchor | 43,197／16,723 |
| 可参与选择 Gaussian | 33,535 |
| 原 Selector | 19→32→32→2，1,762参数 |
| Student Compensator | 62→32→32→14，3,534参数 |
| CPU后端 | XNNPACK FP32，2线程；分块4,096行 |
| GPU渲染 | 官方 Meta SplatFeature 0.15.0，Adreno；未使用我们自写 Rasterizer |
| MCGS SHA256 | `1d735b0871ff94cc6acbf039ad014e0d1490ca4e209caef92648a0b9e53a6685` |
| Teacher／地图／UID fingerprint | `9e2d413fa7864127465394fa22bb3da4e05d912d4c9e0822a4e928933aa36804` |
| Student checkpoint SHA256 | `fd85285c6bd5871f4b41c85690597ccadc965d5a5ec6f474a983e4e09f547a75` |

沿用原 SceneAsset 的整文件 SHA、33个 section SHA、shape、owner范围、UID唯一性、student schema和参数量校验。只加载已验收 Student32 权重，未误用64/64 Teacher；不调用 QNN／HTP，不在GPU上执行MLP。

保留固定噪声、Residual48六表原序累加／插值、连续Ratio编码、局部Offset3、模式／视角输入、严格K、UID同分升序和最终Master行序、物理解码与非有限值拒绝。当前资产的Scale解码为其原有softplus方案，未按历史其他实验设置改动。

## 四种预算口径

- **全局 Ratio**：候选为整图N，`K=min(eligible, floor(N×ratio))`，两网络的视角／模式输入均为0。
- **全局固定K**：候选为整图N，网络条件Ratio为 `min(1, K_requested/N)`；实际K受eligible限制。
- **局部 Ratio**：每次请求取得当前OpenXR双眼的真实FOV与相对头部姿态，使用**名义Anchor立方体与双眼视锥相交的并集**建立候选，按该候选N计算预算。不是按radii>0或Alpha贡献统计预算。
- **局部固定K**：同一局部候选域，条件Ratio为 `min(1, K_requested/N_local)`；实际K受eligible限制。

官方Splat对象为双眼共享，局部网络使用**头部中心视角条件**，不是左右眼各执行一套Selector／Compensator。双眼各自视锥用于候选并集，中心视角使用双眼FOV包络。地图Transform与头部姿态转换到同一map坐标系；OpenXR右／上／后转为原网络OpenCV右／下／前。

固定K=0和空候选直接提交空结果；不会用0去计算log Ratio。超过候选／eligible的固定K显示预算缺口。Ratio=1仍为网络LoD，受eligible约束，**不等同原Full Preview**。任意Ratio是输入能力，不代表已验证全部比例的画质；学生训练范围为1%—70%。

## 官方接口限制与实时行为

实际审计0.15.0 AAR：公开 `Splat`／`SplatSystem` 以PLY／SPZ URI加载资源；逐帧Gaussian属性DirectBuffer入口是private JNI。没有调用反射或私有native方法。当前路径是：

`当前头部／双眼参数 → CPU准备 → Selector → 严格TopK → Student → 解码 → FP32 PLY打包 → 官方异步加载 → 官方双目绘制`。

应用持有两个交替临时PLY文件和一个方法Splat实体，严格单个在途推理／加载。工作线程完成CPU流程，SDK事件确认加载完成后才更新提交状态和实际K。加载期间继续进行头部跟踪与绘制，但选点来自上次已提交结果；**选点年龄和LoD提交Hz直接显示这种延迟**。持续更新启用时，每轮重新取得最新视角，不把旧Mask／旧网络输出缓存命中当成加速。

PLY用原输出XYZ、DC、rotation和补偿后opacity logit；SH rest恢复标准PLY顺序；physical scale取log以满足官方标准PLY语义。该格式转换不是量化，但log／exp往返会有FP32舍入，实测最大Scale相对差约 `1.064e-6`。官方渲染器自身与原Vulkan Raster的画质差异尚未做GT定量评价。

公开资源加载是目前瓶颈：本次SDK加载约60—150ms，远高于纯MLP时间。该入口支持实际动态局部LoD实验，**不能宣称每个显示帧都已完成新视角推理和新地图提交**。停止按钮停止新的更新；模式切换或修改输入后需点击“应用并渲染”。四模式测速完成后暂停自动连续更新，以便查看结果，持续更新可通过复选框重新启用。

## 计时和FPS口径

CPU完整时间使用原生steady_clock，包含候选／特征准备、两网络、TopK和解码，排除模型初始化、PLY序列化和SDK加载。分项各自计时；分项中位数之和不保证等于总时间中位数。

四模式测速在每项执行2次CPU预热＋9次测量，报告中位数；然后加载一次PLY，预热2秒，再读取约3秒的OpenXR系统计数器。第二轮反向顺序。**CPU与GPU是分阶段测量，不是将一个GPU计数器值配给同一个CPU请求做端到端时间。** Benchmark“更新墙钟”包含11次CPU执行，面板明确标注，不能当作实时单次请求延迟。

App GPU为 `XR_META_performance_metrics` 的应用双目系统聚合值，包含SDK和面板，**不是纯Raster时间**。两套测试分别1,740／1,739次读取，每测试项约3秒中仅4种不同GPU值。读取分布p95不等于独立逐GPU帧p95，不能据此给出置信区间。原始计数器读取均导出。

实时FPS通过SDK每帧回调的实际到达时刻在有界一秒窗口计算，按predictedDisplayTime去重；它是**应用逻辑帧／提交节拍**，不是头显最大刷新率、不是 `1000/GPU_ms`，也不是独立统计的最终显示成功帧数。LoD提交Hz另外计算。测试采样段的应用提交FPS中位约72，不构成长期稳定72FPS验收。

## Quest 3实测结果

以下均为“每轮中位数再取两轮中位数”，GPU单位是上述系统App双目指标。不同模式的K或Gaussian身份不同，不能把差异全归因于renderer算法。

### 输入Ratio=0.3，固定预算=5,000

{tables[0]}

### 输入Ratio=0.4，固定预算=5,000

{tables[1]}

第二套中固定K模式的网络条件仍由K/N计算，因此与第一套固定K具有相同条件；耗时波动是真实运行条件变化，不代表把Ratio文本0.4用于固定K网络编码。表中全局N与局部N不同，原静态子集输入K也不能直接拿来当本次相同点身份的对照。

运行中手动切换视角／预算的日志显示局部候选N从6,873到21,919不等，说明并非一直复用同一个候选域。非Benchmark的单次提交墙钟实例约64—183ms；该范围来自本次交互实例，未控制工作量和设备频率，不能当正式延迟分布。Benchmark本身使用相对地图锁定视角，方便比较四个模式，不能替代完整运动轨迹测试。

## 正确性、边界和保护

- 22组原生预算测试覆盖Ratio 0.01／0.1／0.137／0.3／0.5／1及全局／局部、固定0／1／1K／5K／100K；另查空视锥、重复推理、双眼并集、坐标／投影、UID同分和11次测速计数。全部通过。
- 两组原旧CPU完全相同输入的局部case（Ratio0.1、固定1K）候选、Selector输入／噪声／输出、选点身份和Student输入／输出全部**逐位一致**。
- 22组独立NumPy FP32参考的选点身份全部一致；Selector输出最大绝对误差 `3.517e-6`，Student `1.526e-5`。NumPy与XNNPACK运算顺序不同，不宣称跨后端逐位一致。
- 两组原普通PyTorch CPU捕获参考与新ARM输出的Selector最大差 `3.338e-6`，Student最大差 `6.676e-6`。这是网络部署一致性，不是相对GT的PSNR，也不重复包装为新的蒸馏质量实验。
- PLY字段、SH排列和FP32有限值检查通过，XYZ／DC／Quaternion／opacity logit与已解码捕获逐byte一致；Scale格式往返误差如上。
- 8项JVM单测通过（原有4项＋新增4项），Release构建及APK V2验签通过。首次JNI函数与Json命名冲突已修复；重新push probe后重设可执行位再完成最终原生测试。
- 官方renderer库与保留0.1.1 APK逐byte一致。原Q目录源码／模型／APK的18项基线、生产源码11项及四个只读XNNPACK archive SHA均通过；原应用继续保留。
- 0.2.1仅补充休眠／恢复状态处理、FPS口径、空预算执行计数和界面说明／许可证等收尾，未改变网络数学。它已构建、安装并核对SHA；16项真机性能数据对应保留的0.2.0实测APK，不冒称为0.2.1又完整跑过一轮。

## 文件与复现

- 当前APK：`delivery/mycartgs-meta-official-splat-bench.apk`，SHA256 `{validation['apk_sha256']}`。
- 实测APK：`delivery/mycartgs-meta-official-splat-network-lod-run01.apk`，SHA256 `{validation['measured_apk_sha256']}`。
- 原官方静态测试0.1.1：`delivery/mycartgs-meta-official-splat-bench-0.1.1-rollback.apk`。
- 真机原始结果：`evidence/method_benchmark_run01/`、`method_benchmark_run02/` 的results.json／csv。
- 汇总：`evidence/METHOD_PERFORMANCE_SUMMARY.csv`；全部GPU读取：`METHOD_GPU_COUNTER_READOUTS.csv`。
- 逐值验证：`evidence/METHOD_CORRECTNESS.json`、`METHOD_DEVICE_PROBE_FINAL.log`；最终保护验收：`METHOD_FINAL_VALIDATION.json`。
- 运行日志：`evidence/OUR_METHOD_LIVE.log`；接口审计：`tools/api-audit-0.15`。
- 构建：`powershell -ExecutionPolicy Bypass -File S:\\scripts\\build_official.ps1`。

源码包不包含大MCGS、PLY和工具链；可从当前APK提取assets/scenes和assets/fixtures到源码包相应目录。XNNPACK使用先前已验收的四个只读ARM64静态库，完整路径和SHA见METHOD_XNNPACK_ARCHIVES.json；重建依赖时使用原固定commit及原配置，不能误用主机x86库。JNI和本轮新增源码全部在独立构建目录编译。未重新训练、未修改生产默认或原checkpoint，未commit／push。

## 下一步性能工作

首先应解决官方SDK公开接口的动态数据更新成本：寻求官方支持的持久Gaussian Buffer更新接口，或获得明确支持后设计可复用的原生数据桥。当前不得将private JNI描述成稳定公共API，也不能隐去重载成本。其次才是根据当前CPU分项优化全局Selector和局部候选准备。不同Gaussian集合的footprint差异也会影响官方GPU耗时，应在固定相机、相同点身份／补偿输出下单独比较渲染器。
'''
(root/'report/OUR_METHOD_OFFICIAL_RENDERER_CN.md').write_text(report,encoding='utf-8')
print('已生成中文报告、四模式逐实验汇总与完整系统计数器CSV。')
