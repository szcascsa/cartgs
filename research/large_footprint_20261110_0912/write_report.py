"""从最终CSV/哈希/检查日志生成中文报告；不从开发筛选推断最终收益。"""
from pathlib import Path
import json,re
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
def read(name):return json.loads((HERE/'reports'/name).read_text())
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
def gm(x):return float(np.exp(np.log(x).mean()))

result=read('RESULTS_SUMMARY.json');frozen=read('FROZEN_CANDIDATE.json');integrity=read('INTEGRITY_FINAL.json')
assert result['passed'] and integrity['passed']
s=pd.read_csv(HERE/'reports/RASTER_AND_PIPELINE_RESULTS.csv');a=pd.read_csv(HERE/'reports/AGGREGATE.csv')
w=pd.read_csv(HERE/'reports/WORKLOADS.csv');m=pd.read_csv(HERE/'reports/MEMORY.csv');agreement=pd.read_csv(HERE/'reports/ROUND_AGREEMENT.csv')
profile=pd.read_csv(HERE/'reports/PROFILE_COMPARISON.csv');modules=pd.read_csv(HERE/'reports/PROFILE_MODULES.csv')
env=pd.read_csv(HERE/'reports/GPU_ENVIRONMENT.csv');count=pd.read_csv(HERE/'reports/COUNT_VS_FULL.csv')
stress=pd.read_csv(HERE/'reports/SYNTHETIC_STRESS_RESULTS.csv');cleanup=read('CLEANUP.json')
assert read('jobs/footprint_stress02.json')['measurement_valid']
def value(scope,variant,condition):return a[(a.scope==scope)&(a.variant==variant)&(a.condition==condition)].iloc[0]
def drop(scope,variant,condition):return (1-1/value(scope,variant,condition).geometric_speedup_G)*100
full=s[(s.scope=='raster')&(s.condition=='full')]
worstC=float(full[full.variant=='C'].latency_change_pct_vs_G.max());worstD=float(full[full.variant=='D'].latency_change_pct_vs_G.max())
memory=m.pivot(index=['scene','round','condition'],columns='variant',values='workspace_bytes');extra=(memory.C-memory.G)/(1<<20)
checks=[]
for name in ['sanitizer_mem01','sanitizer_init01','sanitizer_sync01']:
    identity=read('jobs/'+name+'.json');summary=read(name+'/SUMMARY.json')
    log=(HERE/'logs'/(name+'_details.log')).read_text()
    assert identity['returncode']==0 and summary['passed'] and re.search(r'ERROR SUMMARY: 0 errors',log),name
    checks.append((name,summary['comparisons'],'0 errors'))
lowrows=[]
for name in ['C','D']:
    for ratio,condition in [('1%','local0.01'),('5%','local0.05'),('10%','local0.1')]:
        row=value('raster',name,condition);p=value('pipeline',name,condition)
        both=agreement[(agreement.variant==name)&(agreement.scope=='raster')&(agreement.condition==condition)].both_faster_G.sum()
        lowrows.append([name,ratio,f'{drop("raster",name,condition):.2f}%',f'{row.saved_us_G_median:.2f} μs',
            f'{row.min_speedup_G:.4f}–{row.max_speedup_G:.4f}×',f'{int(both)}/24',f'{drop("pipeline",name,condition):.2f}%'])
fullrows=[]
for scene in s.scene.unique():
    row=[scene,str(int(s[s.scene==scene].N.iloc[0]))]
    for name in ['C','D']:
        x=full[(full.scene==scene)&(full.variant==name)].sort_values('round')
        row.append(' / '.join(f'{v:+.2f}%' for v in x.latency_change_pct_vs_G))
        row.append(' / '.join(f'{v:.3f}' for v in x.p95_ms))
    fullrows.append(row)
profrows=[]
for condition in ['local0.01','local0.05','local0.1','full']:
    row=[condition]
    for variant in ['G','C','D']:
        p=profile[(profile.condition==condition)&(profile.variant==variant)].iloc[0]
        row.append(f'{p.kernel_launches:g} / {p.gpu_kernel_us:.2f}')
    profrows.append(row)
countrows=[]
for scope in ['raster','pipeline']:
    for condition in ['local0.01','local0.05','local0.1']:
        t=count[(count.variant=='C')&(count.scope==scope)&(count.condition==condition)]
        countrows.append([scope,condition,str(len(t)),str(int((t.speedup_vs_own_full<1).sum())),f'{t.speedup_vs_own_full.min():.3f}–{t.speedup_vs_own_full.max():.3f}×'])
single=[]
for condition in ['local0.01','local0.05','local0.1','full']:
    t=w[w.condition==condition]
    single.append([condition,f'{int(t.K.min())}–{int(t.K.max())}',f'{t.bbox_tiles_per_visible.median():.1f}',
        f'{(t.active_tiles/t.tiles).median()*100:.1f}%',str(int(t.members_max.max())),str(int(t.overflow_tiles.sum()))])
jobs=pd.read_csv(HERE/'reports/JOBS.csv');invalid=jobs[(jobs.measured==True)&(jobs.measurement_valid==False)]
text=f'''# 低点数、大 footprint CUDA 优化报告

## 结论

本轮在独立研究目录实现、编译并比较多条 CUDA 路径，保留 C（预处理融合任务）作为研究推荐，D（四 CTA 展开）作为取舍对照。C 相对上一轮 G/R，1%/5%/10% 纯渲染耗时分别下降 **{drop('raster','C','local0.01'):.2f}% / {drop('raster','C','local0.05'):.2f}% / {drop('raster','C','local0.1'):.2f}%**，24 组两轮均快于 G。节省约 4–6 μs，仍为微秒级收益，没有达到倍数级新增加速。

C 的完整准备＋渲染分别下降 {drop('pipeline','C','local0.01'):.2f}% / {drop('pipeline','C','local0.05'):.2f}% / {drop('pipeline','C','local0.1'):.2f}%。1% 有两组的其中一轮略慢于 G；5%/10% 全部两轮均快。Full 没有逐场景不退化：C 最差慢 **{worstC:.2f}%**，D 最差慢 **{worstD:.2f}%**。本轮按用户后续要求允许单列 Full 代价；若仍沿用上一轮的逐场景 Full 硬门限，C/D 都不能替换 G。

正式结果包含 {result['samples']:,} 个计时样本、{result['correctness_rows']:,} 份逐视角 RGB/radii/IDs/相机核对。所有核对逐位一致；没有以 PSNR 接近放行误差。**没有更新生产默认、训练代码、原 checkpoint，没有 commit/push/部署。**

## 对照、输入和测量口径

- A：原始推理渲染入口，为唯一数值 oracle。Pipeline A 包含原始相机准备。
- G：上一轮已验收 Graph 渲染；完整路径使用上一轮 R 异步相机准备。
- C：本轮 mode=21，64 线程预处理＋在预处理生成 footprint 任务＋每 Warp 一个 256 Tile 任务。
- D：本轮 mode=19，64 线程预处理＋每 Gaussian 固定四个 CTA 展开 footprint。

C/D 的完整路径使用与 G 完全相同的 R 准备。相对 G/R 的差异才是本轮新增收益。相对 A 的更大收益包含前几轮工作，不能重复归因。训练、Selector/Compensator、严格 K、Gaussian 身份、补偿属性、SH、FP32、背景、相机和像素 Alpha 合成顺序均保持不变。

覆盖 Replica 全部 16 个场景/模式、TUM 4 组、VECtor 4 组；地图 N={int(w.N.min()):,}–{int(w.N.max()):,}。5 组用于开发，19 组在冻结源码/二进制后验证。正式每场景连续 12 个运动视角＋6 个跳转视角；两轮独立测量，第二轮反转场景次序，版本次序持续轮换。每个完整推理样本重新构造位姿、上传相机、计算候选、选择与补偿，未依赖同帧 Mask/Feature 命中。

条件为局部 .01/.03/.05/.1/.3/.7、全局 .01、固定 0/1000/5000 和 Full。不同场景按自己的候选数量计算 K，不使用 same-K 跨场景口径。CSV 保留实际 K；固定预算不足候选时沿用既有逻辑。

每个场景/每轮作为独立测量单位，只有该段全过程无外部 GPU 活动才接受。整批 paired01 因两次 Xorg 活动，其计时全部作废；保留该轮已逐位验证的整数 workload 诊断。之后仅对受干扰场景重测，接受规则不查看性能值。全部失败/作废段保存在 JOBS.csv；共有 {len(invalid)} 个 measured 段未用于性能结论。没有停止 Xorg 或其他外部进程。

正式段时钟 {env.clock_MHz.min():g}–{env.clock_MHz.max():g} MHz，温度 {env.temperature_C.min():g}–{env.temperature_C.max():g}°C。没有锁频或修改系统设置。CUDA Event、同步墙钟、Nsight Kernel 累计时间分别保存；Profiler 时间不代替配对墙钟。

## 实现与数值等价性

### C：预处理融合 footprint 任务

旧 G 在稳定排序后先执行 scheduleBits，再执行 executeBitTasks。C 在已有 Gaussian 投影预处理完成后，用整数任务描述写出每个 Gaussian 的 256 Tile chunk，移除独立 schedule Kernel。LoD 的输入是原网络给出的补偿后属性，Full 的输入是原 Master 属性，两者均未改变。稳定排序的最后阶段同时写 Gaussian 输入索引到深度 rank 的逆映射，后续任务按这个 rank 对 Tile 位表执行整数 atomicOr。

任务入队次序可以不同，但整数位集合不依赖入队次序；像素仍按原稳定深度顺序枚举。没有新增浮点 atomic，也没有改写像素合成。小 footprint 同样使用这条任务路径，没有依据 Ratio/K 让 Full 绕开。

预处理 CTA 从 256 改为 64 线程，使 512 容量图从 2 个 CTA 增为 8 个 CTA，改善低点数时的可用并发。每点计算表达式保持原样，通过逐位检查验证编译后结果。

### D：每 Gaussian 四 CTA

四个 CTA 交错遍历同一 Gaussian 的真实 bbox Tile，使用已固定原二进制 FMA 顺序的 orderedTilePower，写入相同深度 rank 的成员位。它能增加大 footprint 的并行度，但 Full 中大量小 footprint 同样支付四 CTA 调度成本。跨场景结果表明这种代价显著，不推荐替换 C/G。

### Workspace、失效和 Stream

每个实例绑定单 device/Stream；跨 Stream 使用必须另建实例，否则明确报错。所有帧指针、P、相机和输出重新提交。任务计数在同一图中先显式清零，再预处理、稳定排序、执行任务、压缩成员、像素合成；不会读取旧帧计数。扩容或分辨率变化销毁旧地址调度计划。输出单独分配，后续调用不能覆盖已返回输出。

任务容量仍有界于 capacity×ceil(Tile数/256)，位表上限 256 MiB、总 Workspace 上限 1 GiB、最多 512 个图计划。额外 metadata 分配为 20×padded capacity 字节，其中 C 使用逆 rank；当前相对 G 增加 **{extra.min():.3f}–{extra.max():.3f} MiB**，C 最大 Workspace **{memory.C.max()/(1<<20):.2f} MiB**。没有缓存任何视角的渲染内容或使用过期 Feature。

## 两轮实际性能

下表耗时下降为场景/轮次等权的几何平均时间比例；“节省”是逐场景中位延迟差的中位数。“最小–最大加速比”保留所有 48 个场景轮次的范围。

'''
text+=table(['路径','Ratio','纯渲染耗时下降 vs G','节省中位','逐场景加速范围','两轮均更快','完整耗时下降 vs G/R'],lowrows)
text+='''
“两轮均更快”指每场景/每条件的样本中位，不代表每一帧都更快。完整 Pipeline 仍被相机准备、Selector/Compensator 和输入组织占用，不能把纯 Rasterizer 的 4%左右收益直接当作完整推理收益。所有中位、p95、标准差、两轮波动、实际 K 和逐视角结果见 RASTER_AND_PIPELINE_RESULTS.csv、PER_VIEW_RESULTS.csv、ROUND_AGREEMENT.csv。所有比 G 慢的条件保存在 ALL_SLOWER_CASES.csv，没有只筛选低 Ratio 的有利样本。

### Full 的逐场景代价

以下百分比为相对 G 的耗时变化，正数表示更慢；每格按第1轮 / 第2轮列出。p95 单位 ms。

'''
text+=table(['场景/模式','N','C Full耗时变化','C Full p95','D Full耗时变化','D Full p95'],fullrows)
text+='''
## Nsight 归因

同一次 Nsight 会话，对 Replica office2 RGB-D 的运动/跳转输入轮换 G/C/D，测 1%/5%/10%/Full，每条件每版本 12 次。以下每格为每帧平均 Kernel 数 / Kernel 累计 μs；不同视角的动态容量不同，因而均值可能不是整数。

'''
text+=table(['条件','G','C','D'],profrows)
text+='''
C 每帧减少一个 schedule Kernel。Host仍提交输入更新和Graph launch，这里减少的是图内Kernel节点，不能写成Host API发射次数同步下降。低点数有预处理并行度与任务调度收益；像素核完全沿用上一版，是主要剩余热点。Full 时 C 在预处理多做任务入队和稳定排序逆映射，节省 footprint 调度的同时也付出新增成本，净收益随负载变化。D 的 Full footprint 展开明显变慢，不能用减少一个 Kernel 的数量优势掩盖 Kernel 累计时间增加。

PROFILE_MODULES.csv 给出预处理、排序、footprint、成员压缩、像素与清零分项，PROFILE_COMPARISON.csv 给出逐条件统计；原始节点/资源使用在 profile_suite01/KERNELS.csv。Profile 内显式 cudaDeviceSynchronize 来自本次区间测量，不能写成产品内部每帧增加同步。没有使用受限的 NCU 硬件计数器臆测占用率。

## 为什么低点数仍没有倍数级新增收益

'''
text+=table(['条件','实际 K 范围','每可见点 bbox Tile 中位','活跃 Tile比例中位','成员最大值','溢出 Tile总数'],single)
text+='''
1% 输入的平均 footprint 仍覆盖大量 Tile，活跃 Tile 中位为100%。减少点数没有消除整幅像素的输出写入以及 Alpha/T/C 链；所以“跳过空 Tile”并不是这批低 Ratio 的主要机会。实际低 Ratio 的短列表都未溢出，本轮通用收益主要来自预处理与成员构建，不能假称已解决所有极端密集 footprint。

C 与自己的 Full 比较，纯渲染的低 Ratio 测试全部更快；完整路径仍有反例，因为局部 LoD 需要额外候选、选择与补偿。这是完整流程差异，不能只按 K 得出完整推理必然更快的结论。

'''
text+=table(['范围','条件','逐视角/轮次数','慢于自身Full','相对自身Full范围'],countrows)
text+='''
这些是同一冻结模型在不同 K/补偿条件下的实际部署路径比较，不是“相同属性仅删点”的因果实验，也不能推广为任意地图、分辨率、GPU 均有相同提升。

## 极端 footprint 压力输入

另测48种合成输入：K=1/32/128/512，两种分辨率641×479、1280×720，物理Scale=.03/.3/3，不透明度=.02/.5。144份G/C/D输出均与A逐位一致；3840个实际CUDA墙钟样本单独保存，不混入真实场景平均结果。压力输入没有额外渲染图片。

以下是一次固定输入微测的加速比分布，不能替代24组运动/跳转视角的两轮通用性验证；接近0.2%的差异也不能当作强统计收益。

'''
text+=table(['路径','物理Scale','相对G最小','中位','最大'],[
    [name,str(scale),f'{g.speedup_vs_G.min():.4f}×',f'{g.speedup_vs_G.median():.4f}×',f'{g.speedup_vs_G.max():.4f}×']
    for (name,scale),g in stress[stress.variant.isin(['C','D'])].groupby(['variant','scale'])])
text+='''
最大的Scale=3下，C的新增收益明显缩小，说明本轮减少固定调度开销后，大面积Alpha合成仍成为主导。尚未获得针对极端footprint的倍数级提速，不能把一般低Ratio的约4%收益无条件推广到更大footprint。

## 正确性与安全检查

- 最终 V4 全量开发验证 1,710 项，无差异；最终两候选额外生命周期检查 172 项通过。
- 两轮24组共28,512条初始逐视角检查全部 RGB/radii/IDs/相机逐位一致；每个新准备 Pipeline 调用还即时断言 IDs 与 RGB 一致。质量保持来自逐位证据，未重新训练或保存渲染图。
- 包括 0/1/31/32/33、255/256/257、511/512/513、1023/1024/1025、4097 点，短列表溢出、巨大/极薄/非有限 Gaussian、同深度稳定排序、负零、预计算颜色/协方差、非整 Tile 分辨率、异步形状变化、跨 Stream 显式依赖与拒绝、Workspace 复用和输出独立性。
- orderedTilePower 复用上轮按原 SASS 固定的 FP32 FMA 顺序；原极薄 Gaussian 反例再次通过。没有降低精度、改 Alpha 阈值或合成顺序。

'''
text+=table(['Compute Sanitizer','逐位检查','结果'],checks)
text+=f'''
## 失败候选与研究边界

| 候选 | 实测结果/处理 |
| --- | --- |
| Tile-word 全图扫描、Tile扫描融合压缩 | 重复扫描大量 Gaussian；动态实际 P 只使部分1%条件改善，5%及更高没有稳定优势。未纳入推荐。 |
| Warp shuffle、每线程4/8像素 | 增加广播/寄存器与长链成本；部分数值完全一致但速度回退。未纳入推荐。 |
| 直接从GPU缓存读取属性，取消9KiB shared缓存 | 全部通过逐位检查，但跨场景像素读取更慢。未纳入推荐。 |
| 每 Gaussian 两/四 CTA | 两者低Ratio有小幅收益；四CTA Full代价大。保留D作明确取舍对照，不推荐生产切换。 |
| 64/128线程预处理 | 单项只节省约1–3μs；与任务融合组成C。没有把相机或上轮已有收益算入该项。 |
| 早期模块 common.py 命名冲突 | 与上游导入冲突，已改为 study_common/footprint_runtime；失败日志保留。 |
| 压力测试旧G统计接口 | 旧G没有snapshot接口，首个压力工具报错；已仅在支持的C/D读取成员诊断，另名重跑通过。未改正式候选。 |
| Xorg活动干扰 | 受影响计时不使用，按预先确定的场景/轮次单位重测；没有删除反例或按速度筛选。 |

下一阶段优先测像素 Alpha 核的必要工作与内存写出下界，再研究大 footprint 的跨Tile属性复用或按真实负载的像素调度。低Ratio活跃Tile已基本覆盖全屏，不能继续以跳空Tile为主要假设。极端重叠导致短列表溢出的效率应独立研究；真实低Ratio当前最大成员数低于256，不应把只改善合成压力输入的收益写成当前所有场景收益。保持稳定 Alpha 次序是继续并行化的主要限制。

## 完整性、回退和交付

本轮目录：`/home/prmi/zwj/experiments/large_footprint_20261110_0912`。本地同名 research 目录保存报告、CSV、性能图、源码/补丁、冻结配置与证据。

复现沿用上一轮只读研究目录及既有TeacherSamples/输入路径；交付未复制原模型或数据集。预编译Graph与首次相机准备不计入稳态延迟，startup.csv记录三条研究路径合计的预编译耗时，不能解释为C独立冷启动耗时。

最终二进制：`{frozen['binary']}`。

SHA-256：`{frozen['binary_sha256']}`。

保护清单 {integrity['protected_files']:,} 个文件全部逐SHA复核，变更0、缺失0；覆盖服务器 mycartgs/realLOD 源码、配置、二进制、原模型和上一轮研究证据。冻结源码/二进制一致，无仍在运行的本轮实验，无渲染图片。INTEGRITY_FINAL.json 保存结果。生产默认、训练与共享conda未修改，没有commit/push/部署。C/D可通过独立 Renderer(21)/Renderer(19)复现实验；生产和上一轮G继续保留，不需要回滚生产。

仅清理本轮已确认可重建的 {len(cleanup['removed'])} 个object/SQLite中间文件，释放 {cleanup['freed_bytes']/(1<<20):.2f} MiB；保留.so、源码、CSV、原始nsys-rep和Sanitizer日志。逐文件SHA和删除清单见CLEANUP.json。核验时剩余磁盘 {integrity['disk_free_bytes']/(1<<30):.2f} GiB，未清理共享缓存或其他任务目录。

主要交付：

- `reports/RASTER_AND_PIPELINE_RESULTS.csv`：逐场景逐条件两轮中位/p95/样本数/实际K及A/G对照。
- `reports/PER_VIEW_RESULTS.csv`、每个正式段的 `samples.csv`：逐视角与原始样本。
- `reports/WORKLOADS.csv`：可见点、Alpha参与点、Tile实例、活跃Tile、成员分布；来自paired01计时外诊断，不使用其受干扰耗时。
- `reports/FULL_COST.csv`、`ALL_SLOWER_CASES.csv`：全部Full代价与慢例。
- `reports/PROFILE_COMPARISON.csv`、`PROFILE_MODULES.csv`、原始 `.nsys-rep`：真实节点归因。
- `reports/PAIRED_RUNS.json`、`JOBS.csv`、`GPU_ENVIRONMENT.csv`：接受/作废段、进程身份及硬件环境。
- `source_frozen.tar.gz`、V1/V2/V3独立快照、`build_v4/*.so`、Sanitizer日志与保护哈希：复现与回退证据。
- `figures/SUMMARY_SPEEDUP.png`、`LOW_RATIO_DETAIL.png`、`REPLICA_PER_SCENE.png`、`TUM_VECTOR_PER_SCENE.png`、`FOOTPRINT_SPEEDUP.png`及SVG/Data Manifest：性能曲线，非渲染图片。
'''
(HERE/'LARGE_FOOTPRINT_CUDA_OPTIMIZATION_REPORT.md').write_text(text,encoding='utf-8')
print('CHINESE_REPORT_WRITTEN',len(text))
