"""实际CPU Student支持、双轮速度、蒸馏损失与后端误差分开报告。"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
HERE=Path(__file__).resolve().parent
PRIMARY=['local0.1','local0.2','local0.3','local0.4','local0.5']
def read(name):return json.loads((HERE/'reports'/name).read_text())
def geo(x):return float(np.exp(np.log(x).mean()))
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
r=read('RESULTS_SUMMARY.json');integrity=read('INTEGRITY_FINAL.json');f=read('FROZEN_CANDIDATE.json');assert r['passed'] and integrity['passed']
for run in ['boundaries01','handoff_edges01','binding01','asan02','ubsan01']:assert read('jobs/'+run+'.json')['returncode']==0,run
a=pd.read_csv(HERE/'reports/CPU_AGGREGATE.csv');cpu=pd.read_csv(HERE/'reports/CPU_PAIRED_RESULTS.csv');c=pd.read_csv(HERE/'reports/COMPONENT_PAIRED_RESULTS.csv');h=pd.read_csv(HERE/'reports/HANDOFF_PAIRED_RESULTS.csv');q=pd.read_csv(HERE/'reports/QUALITY_ALL_FRAMES.csv');qc=pd.read_csv(HERE/'reports/QUALITY_PER_CONDITION.csv');models=pd.read_csv(HERE/'reports/MODELS.csv');mem=pd.read_csv(HERE/'reports/CPU_MEMORY.csv');raster=pd.read_csv(HERE/'reports/RASTER_DIAGNOSTICS.csv')
text=f'''# 蒸馏Student Compensator：CPU接入与实际速度报告

## 结论

已为最新CPU冻结推理后端加入显式`student_checkpoint`加载功能，完成Teacher内容指纹绑定、32/32学生部署和真实速度验证。默认仍使用Teacher；学生只替换Compensator，Selector、Master、Anchor Feature、严格K和训练不变。

主要Ratio 0.1–0.5，16组Replica、两轮等权配对：**Compensator及属性解码/输出加速{r['primary_compensator_speedup']:.3f}×，完整CPU准备加速{r['primary_CPU_speedup']:.3f}×**；四代表场景的准备＋H2D＋原GPU渲染加速**{r['primary_handoff_speedup']:.3f}×**。对应CPU完整准备延迟降低{100*(1-1/r['primary_CPU_speedup']):.2f}%，真实完整路径降低{100*(1-1/r['primary_handoff_speedup']):.2f}%。不能把参数减少倍数当成实际加速倍数。

学生有既有蒸馏质量代价：本轮主要Ratio采样的平均Teacher−Student GT PSNR为{r['primary_mean_teacher_student_psnr_loss']:.5f}dB，最差单帧{r['primary_worst_frame_teacher_student_psnr_loss']:.5f}dB；不是Teacher质量等价替代。本功能保持可选，未替换生产默认或重训学生。

## 实现及使用

- 新入口`student_runtime.CpuRenderer`复用上一轮`mid_runtime.CpuRenderer`，新增可选文件加载及绑定校验；不复制和修改上一轮源码。
- `CpuRenderer.from_checkpoint(teacher_checkpoint, original_noise_path, student_checkpoint=path)`可在CPU独立恢复；显式传入学生实例也可使用。两种学生来源互斥；错误Teacher/地图/Feature/行序指纹直接报错，无静默回退。
- 不传学生即原Teacher，144组预检确认Teacher输出逐位不变。学生开启后同一Selector的Score、UID和K保持一致；不同的属性来自被蒸馏网络本身。
- 使用同一AVX2/FMA微核、直方图选点、6点分块、有界缓存和C++三槽交接。现有后端本来支持13个学生参数Tensor布局，本轮实际补上加载、验证和验收；无新增C++/CUDA源码需要重新编译。
- 学生输入保持offset3＋Residual48及连续Ratio3、局部view7＋mode1。32/32两层共享主干、四个直接线性Head输出XYZ/Scale/Rotation/Opacity/DC的14D属性；Teacher是64/64及四个两层Head。
- MLP参数：Teacher {int(models.teacher_parameters.iloc[0]):,}，Student {int(models.student_parameters.iloc[0]):,}，减少{100*(1-models.student_parameters.iloc[0]/models.teacher_parameters.iloc[0]):.2f}%。共享Anchor Feature和地图没有压缩，这不是整个模型磁盘或RAM缩小同样比例。
- 继续复用原GPU固定随机键；CPU不重采样替代。快照在加载时复制，模型/拓扑/Feature变化需重建，每帧视角、候选、网络和补偿均重新计算。

使用示例及Stream约定见README_CPU_STUDENT.md。训练、原checkpoint、共享环境和生产默认未改，无commit/push/部署。

## 配对协议

硬件Intel i5-9600K，6核6线程、AVX2/FMA；GPU RTX3090仅用于原Rasterizer及质量评价。原生CPU池6线程，Torch辅助1线程。Replica office0–4/room0–2，单目＋RGB-D共16组，使用每组已训练的最终continuous_calibration/best.pt，先核验Teacher指纹与模型哈希；没有根据本轮结果重新挑学生。

CPU双轮每次只驻留一个线程池。每组3连续＋3跳转保存相机，场景反序、版本顺序轮换；每条件三块，每块预热6次后计18次。正式未插桩总延迟与额外24次分段计时分别保留。分段包含相关准备、分发和输出，不能称纯MLP时间，也不能直接与独立总延迟相加。

主要条件local .1/.2/.3/.4/.5，另包括插值.15/.25/.45、额外.01/.7、fixed0/1000/5000、global.1及Full。总计15条件。所有条件保留实际N/M/K、连续/跳转标记、中位/p95/最大值及慢例。CPU测速结束CUDA未初始化。

四个完整路径代表场景：office1 mono、office2 RGB-D、room0 RGB-D、room2 mono，覆盖地图规模及此前敏感场景。每轮3个跳转相机、10条件、每条件每版本72个同步样本；单池驻留，轮换场景和版本次序。包含本帧51个相机FP32/属性H2D、事件依赖和原GPU Rasterizer；不含磁盘、相机构造、显示或模型加载。不是Viewer FPS或在线SLAM训练测速。

正式轮次逐秒监测CPU/GPU其他进程，未检测到外部活动。采样不能排除瞬时OS调度/中断或频率变化。原样保留各轮慢例，不能用平均加速声称每个场景、每个请求均更快。

## 主要Ratio速度

下表加速为各场景轮次配对的等权几何平均；ms为这些中位延迟的中位，二者不能直接互相相除。

'''
rows=[]
for condition in PRIMARY:
    x=a[a.condition==condition].iloc[0];g=h[h.condition==condition]
    rows.append([condition[5:],f'{x.compensator_teacher_ms:.4f}→{x.compensator_student_ms:.4f}',f'{x.compensator_geometric_speedup:.3f}×',f'{x.CPU_teacher_ms:.4f}→{x.CPU_student_ms:.4f}',f'{x.CPU_geometric_speedup:.3f}×',f'{g.teacher_ms.median():.4f}→{g.ms.median():.4f}',f'{geo(g.speedup):.3f}×'])
text+=table(['Ratio','Comp及输出ms','Comp加速','CPU全部准备ms','CPU加速','四场景准备＋渲染ms','完整加速'],rows)
text+=f'\nCPU总样本{r["CPU_samples"]:,}，分段记录{r["component_records"]:,}条，Score/身份/finite检查{r["checks"]:,}组；完整路径样本{r["handoff_samples"]:,}个。主要区间CPU有{r["primary_CPU_slower_rounds"]}/160项更慢，完整路径有{r["primary_handoff_slower_rounds"]}/40项更慢，均见CSV。\n\n'
text+='## 每场景主要Ratio结果\n\nCPU为五档Ratio×双轮等权几何平均；质量为各档三个保存相机的平均Teacher−原生CPU学生PSNR，正值为学生退化。不是此前32个留出全轨迹视角的重复评价。\n\n'
rows=[]
for scene in cpu.scene.unique():
    g=cpu[(cpu.scene==scene)&cpu.condition.isin(PRIMARY)];cc=c[(c.scene==scene)&c.condition.isin(PRIMARY)&(c.stage==4)];quality=qc[(qc.scene==scene)&qc.condition.isin(PRIMARY)];record=models[models.scene==scene].iloc[0]
    rows.append([scene,int(g.N.iloc[0]),f'{geo(cc.speedup):.3f}×',f'{geo(g.speedup):.3f}×',f'{quality.total_psnr_loss.mean():.5f}',f'{(record.teacher_workspace_bytes-record.student_workspace_bytes)/2**20:.2f}'])
text+=table(['场景/模式','N','Comp及输出加速','全部CPU加速','主要Ratio平均PSNR损失dB','Workspace减少MiB'],rows)
text+='\n## 逐场景逐Ratio CPU与质量\n\n每格为Teacher→Student CPU ms，括号为配对加速。\n\n'
rows=[]
for scene in cpu.scene.unique():
    row=[scene]
    for condition in PRIMARY:
        g=cpu[(cpu.scene==scene)&(cpu.condition==condition)];row.append(f'{g.teacher_ms.median():.3f}→{g.ms.median():.3f} ({geo(g.speedup):.2f}×)')
    rows.append(row)
text+=table(['场景/模式','10%','20%','30%','40%','50%'],rows)
text+='\n同口径GT PSNR损失逐Ratio：\n\n'
text+=table(['场景/模式']+[x[5:] for x in PRIMARY],[[scene]+[f'{qc[(qc.scene==scene)&(qc.condition==x)].total_psnr_loss.iloc[0]:.5f}' for x in PRIMARY] for scene in cpu.scene.unique()])
text+='\n## 质量代价和CPU部署误差\n\n三方对照使用相同CPU选择身份、相机和Master：原生CPU Teacher、普通PyTorch Student、原生CPU Student。普通学生的输入/Feature/视角使用原公式，输出经过原decode_student；GT PSNR/SSIM/LPIPS均对clamp[0,1]图像评价。后端图像差使用未clamp RGB。该对照不重新训练，不把学生的拟合误差归因CPU优化。\n\n'
text+=f'''共{r['quality_rows']}组（三个保存相机×15条件×16组），ID全部相同。主要Ratio的平均Teacher→普通学生损失与普通学生→CPU学生的后端损失分别见QUALITY_PER_CONDITION.csv；本轮主要区间额外后端平均损失{r['primary_mean_backend_psnr_loss']:+.8f}dB，全条件最差单帧后端损失{r['worst_backend_psnr_loss']:.8f}dB，CPU图与普通学生图最低相互PSNR {r['minimum_backend_image_psnr']:.4f}dB。部署通过既定平均0.05dB/条件平均0.2dB筛查，不能称逐像素一致。CPU与普通学生的radii存在微差条件，详情见原始数据。

蒸馏质量不是零损失；原Replica全场景报告中严格0.01dB门限没有通过，当前没有改写该历史结果。room2 mono此前GPU融合学生的70dB相互PSNR验收失败也保持原状态。本轮是独立CPU入口的额外误差评价，不替代或降低旧GPU路径门限。

主要区间最差8个单帧质量项：

'''
qp=q[q.condition.isin(PRIMARY)]
text+=table(['场景','帧','条件','Teacher PSNR','Student PSNR','总损失','其中CPU后端损失'],[[x.scene,x.frame,x.condition,f'{x.teacher_psnr:.5f}',f'{x.student_psnr:.5f}',f'{x.total_psnr_loss:.5f}',f'{x.backend_psnr_loss:+.6f}'] for x in qp.nlargest(8,'total_psnr_loss').itertuples()])
text+='\n## 完整路径及Rasterizer归因\n\nTeacher与学生几何预测不同，即使K和身份相同，实际投影footprint、Tile实例和像素合成工作量也可能不同。完整延迟是实际用户路径收益，但不能全部称为MLP加速。另采集render-only CUDA Event诊断，网络与H2D在区间外，不据此代替完整路径。\n\n'
text+=table(['场景','Ratio','Teacher完整ms','Student完整ms','完整加速','Teacher较大p95','Student较大p95','仅渲染Event加速'],[[scene,condition[5:],f'{g.teacher_ms.median():.4f}',f'{g.ms.median():.4f}',f'{geo(g.speedup):.3f}×',f'{g.teacher_p95.max():.4f}',f'{g.p95.max():.4f}',f'{geo(raster[(raster.scene==scene)&(raster.condition==condition)].speedup):.3f}×'] for (scene,condition),g in h[h.condition.isin(PRIMARY)].groupby(['scene','condition'])])
slow=cpu[cpu.speedup<1];slow.to_csv(HERE/'reports/CPU_ALL_SLOWER_CASES.csv',index=False)
text+=f'\n全条件CPU慢例{len(slow)}项，包括Full和空预算，均保留于CPU_ALL_SLOWER_CASES.csv；完整路径全部10条件及p95/最大值在HANDOFF_PAIRED_RESULTS.csv。Full无补偿，理论上无需学生，该项只作打包诊断，不能把波动宣称学生提升了Full。\n\n'
text+=f'''## 验证、完整性与限制

- Teacher绑定预检16组/144条件通过，Teacher入口相对原版本属性逐位不变；Score和选择身份正式检查通过。
- 错配学生文件、错配实例指纹拒绝；从CPU checkpoint独立加载与实例入口逐位相同，后续修改源学生不影响快照。绑定测试{read('BINDING_TESTS.json')['checks']}项通过，CUDA未初始化。
- 学生分支常规CPU边界33项、原生交接25项通过，覆盖空K、SIMD尾部、输入拒绝与恢复、输出独立性、Buffer哨兵及Stream依赖。
- 使用上一轮同源Sanitizer二进制，先核对除类名以外源码完全一致，再实际运行Student32分支。ASan正常路径28项、UBSan完整33项通过。当前系统ASan与Torch主动异常的既有兼容问题导致5项主动异常不纳入ASan；泄漏检测关闭，不能称GPU内存或数据竞争验收。首次ASan脚本因Python导入优先级选中上一轮边界脚本，被已有结果目录的禁止覆盖检查拦截；修正为显式本轮脚本路径后asan02完成。原结果没有被覆盖，失败日志保留。
- Workspace主要逻辑估算按32D学生投影降低，逐模型减少量见上表；不是进程RSS。上传属性数目/SH不变，三槽Host/Device容量相同，不能说蒸馏同时减少传输字节。学生只减少网络权重、CPU计算和静态投影缓存。
- 只测当前i5-9600K、x86 AVX2、FP32部署。不是ARM实机证据，不把过去GPU学生测速混入当前CPU加速。
- 所有学生复用既有权重，没有重训；只有三个保存相机的质量补验，不是全轨迹评价。需要画质优先时显式使用Teacher；学生属于速度/质量取舍。

研究目录：`{HERE}`。

后端二进制：`{f['binary']}`；SHA256：`{f['binary_sha256']}`。与本轮开始完全一致，没有新编译算子或修改原后端。

最终保护文件{integrity['protected_files']}个哈希变化0、缺失0；本轮新入口源码哈希及原二进制一致，无活跃本轮任务，无渲染图片。原生产、训练、原模型、共享环境未改，没有commit、push或部署。

源码、README、绑定/边界测试、逐场景逐Ratio CSV、原始样本、监测、质量三方比较和曲线全部保留。下一步若要继续提高学生质量，应针对0.1–0.5与最差场景单独训练质量消融；本轮没有擅自改变已有学生或质量门限。
'''
(HERE/'CPU_STUDENT_COMPENSATOR_SPEED_REPORT.md').write_text(text,encoding='utf-8');print('STUDENT_CHINESE_REPORT',len(text))
