"""Student专项新增收益、失败候选与全部反例中文报告。"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
HERE=Path(__file__).resolve().parent
PRIMARY=['local0.1','local0.2','local0.3','local0.4','local0.5']
def read(n):return json.loads((HERE/'reports'/n).read_text())
def geo(x):return float(np.exp(np.log(x).mean()))
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
r=read('RESULTS_SUMMARY.json');f=read('FROZEN_CANDIDATE.json');integ=read('INTEGRITY_FINAL.json');assert r['passed'] and integ['passed']
for n in ['boundaries01','handoff_edges01','binding02','extremes01','asan01','ubsan01']:assert read('jobs/'+n+'.json')['returncode']==0,n
a=pd.read_csv(HERE/'reports/CPU_AGGREGATE.csv');cpu=pd.read_csv(HERE/'reports/CPU_PAIRED_RESULTS.csv');comp=pd.read_csv(HERE/'reports/COMPONENT_PAIRED_RESULTS.csv');h=pd.read_csv(HERE/'reports/HANDOFF_PAIRED_RESULTS.csv');q=pd.read_csv(HERE/'reports/QUALITY_ALL_FRAMES.csv');mem=pd.read_csv(HERE/'reports/CPU_MEMORY.csv');v=r['variant']
full=a[(a.variant==v)&(a.condition=='full')].iloc[0]
text=f'''# 蒸馏学生Compensator的CPU专项优化

## 实测结论

本轮以已经接入的32/32蒸馏学生作为真实基线，保持同一个学生权重、Teacher绑定、Master、Selector、Anchor Feature、Ratio和原GPU Rasterizer。新增收益不包含上一轮Teacher→Student的1.82倍补偿加速。

Ratio 0.1–0.5、Replica16组双轮配对：补偿及解码输出额外加速**{r['primary_compensator_speedup']:.4f}×**；全部CPU准备额外加速**{r['primary_CPU_speedup']:.4f}×**，延迟降低{100*(1-1/r['primary_CPU_speedup']):.2f}%；四代表场景准备＋H2D＋原GPU渲染额外加速**{r['primary_handoff_speedup']:.4f}×**，延迟降低{100*(1-1/r['primary_handoff_speedup']):.2f}%。没有把组件、GPU Event或参数量的收益直接当作整帧收益。

最终保留四个直接Head合并调度、32D专用循环、6 Anchor批量补偿上下文和8-lane几何解码。8点SoA、每worker Anchor缓存、固定循环界和rsqrt候选未获稳定额外收益，已从最终源码撤回；源及二进制在development_v1/v2/v3保留。

'''
rows=[]
for condition in PRIMARY:
    x=a[(a.variant==v)&(a.condition==condition)].iloc[0];g=h[h.condition==condition]
    rows.append([condition[5:],f'{x.comp_baseline_ms:.4f}→{x.comp_optimized_ms:.4f}',f'{x.compensator_speedup:.3f}×',f'{x.baseline_ms:.4f}→{x.optimized_ms:.4f}',f'{x.geometric_speedup:.3f}×',f'{geo(g.speedup):.3f}×'])
text+=table(['Ratio','原学生→新补偿ms','补偿加速','原学生→新CPU总ms','CPU加速','准备＋上传＋渲染加速'],rows)
text+='\n加速为场景轮次配对等权几何平均；延迟列为各配对中位数的中位，不能直接相除。原始样本、N/M/K、p95/最大值及所有退化项保留CSV。\n\n'
text+='''## 源码和实现

### 1. 四Head调度合并

学生四个Head都是32→3/3/4/4的线性层，原代码对四组权重分别调用SSE小输出微核。新路径将原独立权重按原输出顺序装入16列调度Buffer（14个有效输出＋2个固定零padding），一次AVX2微核生成14D残差。每个输出仍按原j=0…31执行FP32 FMA再加原Bias，没有把四组权重相加，没有修改模型结构、参数或训练。

派生权重仅在冻结实例加载时构建，额外约2KiB，随实例销毁释放；学生或地图改变须重建实例。原学生checkpoint仍是3534个MLP参数，没有新私有可训练参数。

### 2. 32维学生专用循环

原通用补偿函数同时为Teacher保留64维中间值、256维Head隐藏层和条件分支。学生专用循环只保存32D激活和16D输出，省去逐Head小Tensor复制及Teacher中间区。仍使用6点分块，SIMD尾部只发布实际R个点。最终未保留8点SoA：它在CPU输入布局转换和权重广播上的代价抵消了复用收益。

### 3. Anchor上下文批处理

原代码对每个实际被选择的独立Anchor执行一次8→32 View投影。新路径按最多6个Anchor批量计算，使同一权重在多点间复用；Ratio残差前缀、局部view/mode及加法顺序不变。仍只计算本次选择涉及的Anchor，每帧按当前相机与Ratio重新准备，不缓存旧视角或推理结果。

开发时也实现了每worker有界Anchor缓存，试图融合上下文与MLP并消除独立分发；它引入重复上下文和额外访存，没有胜过批处理，已撤回，因此最终没有新增每Anchor大缓存，也没有把同帧缓存命中计作切换视角收益。

### 4. 向量几何解码

一次按8-lane批量处理当前最多6点的XYZ、物理Scale、Quaternion、Opacity和DC。XYZ/DC及原始残差在开发逐节点检查中保持原数值；Scale/Opacity使用系统libmvec的AVX2 FP32 expf/logf，Rotation用向量sqrt/div并保留三次归一化流程。SH及means2D语义、发布顺序、严格K不变。

系统libmvec没有向量log1pf，因此使用`y=1+e; log1p(e)≈log(y)+(e-(y-1))/y`修正，避免小e直接被`log(1+e)`舍入掉。Softplus的v>20分支、极小Scale下限及输出finite检查仍保留。该路径允许已测量的FP32微差，不能称全属性逐位一致。

XYZ/DC/SH/means2D的变化会被正式测试拒绝；Score及UID哈希必须一致。开发逐节点记录raw残差和输出差异；正式16组检查发布属性、Score与UID，没有宣称逐场景重新检查了所有raw残差。真实图像额外质量门限沿用平均损失≤0.05dB、每场景每条件均值≤0.2dB并报告最差单帧。没有因Baseline本身存在舍入差异而跳过比较。

## 开发候选与失败记录

仅使用office2 RGB-D、room0 RGB-D、office1 mono开发，保留其余13组验证。开发编译过四个版本，分别保留二进制哈希和原始测试。调优结果只用于选定候选，正式收益来自冻结后的双轮实测。

'''
tunes=[]
for run in ['tune02','tune03','tune04','tune05']:
    rows=pd.read_csv(HERE/'reports'/run/'samples.csv');p=rows.groupby(['scene','condition','variant']).ms.median().unstack().reset_index();p=p[p.condition.isin(PRIMARY)]
    for variant in p.columns:
        if variant not in ['scene','condition','baseline']:tunes.append([run,variant,f'{geo(p.baseline/p[variant]):.4f}×',int((p[variant]>p.baseline).sum())])
text+=table(['开发轮次','候选','主要区间CPU加速','慢例/15'],tunes)
text+='''
- 单独Head合并与专用循环收益有限，说明原学生的矩阵计算已经较轻。没有用合并GEMM的理论算量宣称整帧大幅提速。
- 固定32D循环界的显式模板和两次Newton rsqrt均实际编译测试，未稳定超过向量sqrt/div。rsqrt还引入更多Rotation差异，最终移除。
- 每worker Anchor缓存占用更多内存且不如批量投影，已撤回；没有留下无收益的大缓存。
- tune01失败属于诊断脚本：K=0时没有计算Selector，读到了无消费者的旧Score缓冲。改为仅在有选点时核对Score，UID始终核对；原日志保留，未修改算法或计时数据掩盖失败。
- binding01因检查脚本尚未上传而退出，未执行模型测试；补齐脚本后使用独立binding02验收，失败日志保留，没有修改冻结后端。

## 正式协议与额外数值误差

Intel i5-9600K，6物理核、AVX2/FMA；原生池6线程、Torch辅助1线程；RTX3090只执行原Rasterizer和质量指标。CPU每次只驻留一个池，场景/版本顺序双轮轮换；每组3连续＋3跳转保存相机，各15条件、3块×18个同步CPU样本，另做24次分段计时。阶段中位不能机械相加得到未插桩总延迟。

四个完整路径代表为office1 mono、office2 RGB-D、room0 RGB-D、room2 mono。每个条件每版本每轮72样本，两轮反序；每次重新准备相机/候选/网络，包含CPU、相机属性H2D、依赖、原GPU光栅化及最终同步，不含磁盘、模型加载、相机矩阵构造或显示。没有用同帧输出缓存或流水线吞吐代替单帧延迟。

PROTOCOL.json的benchmark/quality描述沿用了上一轮Teacher→Student的模板文字，未覆盖它；本轮真实执行的common.py、冻结记录和样本均明确为同一学生优化前后。文字澄清另存PROTOCOL_CLARIFICATION.json，所有正式结论使用原学生CPU入口作为对照。

正式采用formal03/formal02与handoff01/handoff02，外部CPU/GPU活跃负载监测均通过。formal01虽正常完成数值检查，但3个采样时段检测到约64%–73%单核外部CPU活动，因此整轮不计入性能结论，原日志与样本保留，没有仅删除慢样本。采样不能排除OS调度、中断或频率瞬时波动，所有慢例保留。模型、初始化和选点一致，基线是原学生CPU后端，不是Teacher或最初PyTorch实现。

'''
text+=f'''CPU样本{r['CPU_samples']:,}个，属性/Score/身份检查{r['checks']:,}组；完整路径样本{r['handoff_samples']:,}个。主要区间CPU慢例{r['primary_CPU_slower_rounds']}/160，完整路径慢例{r['primary_handoff_slower_rounds']}/40。

Full的CPU属性准备并未获得加速：配对几何平均{full.geometric_speedup:.4f}×，{int(full.slower_rounds)}/{int(full.scene_rounds)}个场景轮次变慢；其中位延迟列为{full.baseline_ms:.4f}→{full.optimized_ms:.4f}ms。Full不调用学生补偿，不能把它当作本轮收益；本轮没有进行Full完整准备＋渲染的双轮速度验收，因此不声称Full渲染提速或逐场景不退化。低至1%的CPU准备收益也较弱，全部条件与反例保留在CSV。

同一学生720组渲染对照中，主要Ratio平均额外PSNR损失{r['primary_mean_psnr_loss']:+.8f}dB，全部条件最差单帧{r['worst_psnr_loss']:.8f}dB，最差场景条件均值{r['worst_condition_mean_loss']:.8f}dB；两图最低相互PSNR {r['minimum_image_psnr']:.4f}dB，最大RGB绝对差{r['max_rgb_error']:.7g}，radii不同{r['radii_mismatches']}项。GT图像clamp[0,1]后计算PSNR/SSIM/LPIPS，图像相互误差使用未clamp RGB。不保存渲染图片。

这只说明本轮优化相对原学生的新增差异。该学生本来相对Teacher存在上一轮主要Ratio平均约0.40dB损失，本轮没有改善蒸馏质量，也没有重训或把已有损失算成本轮代价。三个保存相机并非全轨迹质量保证。

PSNR差异较小不等于每像素相同；上述最大RGB差和radii变化均保留，不以全图均值掩盖局部差异。向量数学存在微差，不能作为逐位等价后端发布。

'''
text+=table(['最差额外质量项','帧','条件','PSNR损失dB','最大RGB差'],[[x.scene,x.frame,x.condition,f'{x.psnr_loss:.8f}',f'{x.max_rgb_error:.7g}'] for x in q.nlargest(8,'psnr_loss').itertuples()])
text+='\n## 每场景主要区间\n\n每行五档Ratio×双轮等权几何平均。\n\n'
rows=[]
for scene in cpu.scene.unique():
    p=cpu[(cpu.scene==scene)&(cpu.variant==v)&cpu.condition.isin(PRIMARY)];cc=comp[(comp.scene==scene)&(comp.variant==v)&comp.condition.isin(PRIMARY)&(comp.stage==4)]
    qq=q[(q.scene==scene)&q.condition.isin(PRIMARY)];rows.append([scene,int(p.N.iloc[0]),f'{geo(cc.speedup):.4f}×',f'{geo(p.speedup):.4f}×',f'{qq.psnr_loss.mean():+.7f}',int((p.speedup<1).sum())])
text+=table(['场景/模式','N','补偿及输出加速','全部CPU加速','额外PSNR损失','CPU慢例/10'],rows)
text+='\n## 逐场景逐Ratio CPU结果\n\n每格为两轮中位原学生→优化学生ms及配对几何平均加速。\n\n'
rows=[]
for scene in cpu.scene.unique():
    row=[scene]
    for condition in PRIMARY:
        p=cpu[(cpu.scene==scene)&(cpu.variant==v)&(cpu.condition==condition)];row.append(f'{p.baseline_ms.median():.4f}→{p.ms.median():.4f} ({geo(p.speedup):.3f}×)')
    rows.append(row)
text+=table(['场景','10%','20%','30%','40%','50%'],rows)
text+='\n## 真实完整路径\n\n学生权重与选择身份相同，GPU Rasterizer没有修改；微小属性变化仍可能改变Tile边界，因此另外保留只测原GPU渲染的Event诊断，不把它并入MLP收益。\n\n'
text+=table(['场景','Ratio','原学生完整ms','新学生完整ms','加速','旧较大p95','新较大p95'],[[scene,condition[5:],f'{g.baseline_ms.median():.4f}',f'{g.ms.median():.4f}',f'{geo(g.speedup):.4f}×',f'{g.baseline_p95.max():.4f}',f'{g.p95.max():.4f}'] for (scene,condition),g in h[h.condition.isin(PRIMARY)].groupby(['scene','condition'])])
text+=f'''
## 边界、安全、内存与回退

常规CPU边界33项、原生交接25项、学生绑定与快照隔离{read('BINDING_TESTS.json')['checks']}项通过，覆盖K=0及尾部、Buffer哨兵、独立输出、源Master修改隔离、非法输入拒绝和恢复、同实例并发串行化和不同消费Stream；原Rasterizer继续默认Stream。私有合成学生的Softplus/Opacity极端值、Quaternion边界和K尾部共{read('EXTREMES.json')['checks']}条节点检查通过，没有修改原学生文件。记录所有属性最大绝对/相对差，非法物理输出须与原路径同样拒绝。

新源码独立编译ASan+UBSan正常28项与UBSan完整33项。沿用已定位的系统ASan/Torch主动异常兼容限制，ASan明确排除5项主动异常，泄漏检测关闭；不称GPU内存或数据竞争验收。编译源/参数与日志保留。

派生Head权重及Bias约2112字节；专用循环和批量上下文使用有界线程栈，没有最终保留每Anchor/每worker大缓存。主要Workspace逻辑估算见CPU_MEMORY.csv，不含所有线程栈/分配器/进程共享页，不是RSS。三槽上传大小和SH数据量不变。模型/Feature/拓扑改变须重建快照；每帧视角与Ratio变化立即重算，无过期状态。

独立开关`student_kernel=0/1/2`分别为原通用、packed Head、32D专用循环；`student_decode=0/1`控制scalar/vector解码；`student_context=0/1`控制逐Anchor/批量投影。研究入口默认2/1/1，关闭即可回退到原算法。正式生产默认未覆盖。

只验证本机x86 AVX2与当前16组学生，不是ARM/移动端或在线SLAM训练证据。原始权重、Selector、Master、训练策略和checkpoint不变，没有新增网络或低精度权重。

## 剩余瓶颈与交付

学生MLP已经较小，继续只合并Head不能带来大幅整帧收益。补偿仍含几何解码、属性gather/发布及上下文，完整路径还含准备、上传和GPU渲染。本报告按实测报告新增收益，不承诺继续得到毫秒级或倍数级提升。后续若要更大收益，应根据剩余分段分析准备与网络的共同数据布局，或单独研究CPU/GPU重叠的吞吐；后者必须与单帧延迟分开验收，不以同帧缓存代替重新准备。

研究目录：`{HERE}`；基线：`{f['baseline']}`。

最终二进制：`{f['binary']}`；SHA256：`{f['binary_sha256']}`。

{integ['protected_files']}个受保护文件哈希变化0、缺失0；冻结源和二进制一致，无活跃本轮任务，无渲染图片。生产、原模型、原训练及共享环境未改；未commit/push/部署。最终源码/补丁、各开发版本、原始样本、逐场景逐Ratio CSV、质量/安全日志和曲线均保留，清理仅限本轮可重建中间对象并另存清单。
'''
(HERE/'CPU_STUDENT_SPECIALIZED_OPTIMIZATION_REPORT.md').write_text(text,encoding='utf-8');print('STUDENT_FAST_REPORT',len(text))
