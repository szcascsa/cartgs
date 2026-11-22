"""只分析本轮真实CPU/GPU分离样本，输出中文报告。"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
R=HERE/'reports'
s=pd.read_csv(R/'samples.csv');q=pd.read_csv(R/'quality.csv');checks=pd.read_csv(R/'checks.csv')
valid=json.loads((R/'VALIDATION.json').read_text())
job=json.loads((HERE/'logs/formal_job.json').read_text())
rejected=sorted(p.name for p in HERE.glob('rejected_attempt*') if p.is_dir())
assert valid['passed'] and job['measurement_valid'],job
assert len(checks)==1152 and len(q)==1200 and len(s)==55296,(len(checks),len(q),len(s))
assert checks.raster_bitwise.all() and checks.handoff_bitwise.all()
assert checks.groupby(['scene','ratio','frame']).ids_sha256.nunique().max()==1
assert checks.groupby(['scene','ratio','frame']).score_sha256.nunique().max()==1
agg=s.groupby(['scene','variant','ratio','scope']).agg(ms=('ms','median'),p95=('ms',lambda v:v.quantile(.95)),
    maximum=('ms','max'),N=('N','median'),M=('M','median'),K=('K','median'),samples=('ms','size')).reset_index()
agg.to_csv(R/'TIMING_SUMMARY.csv',index=False)
rounds=s.groupby(['scene','variant','ratio','scope','round']).ms.median().reset_index()
rounds.to_csv(R/'TIMING_ROUNDS.csv',index=False)
means=q.groupby(['scene','variant','ratio']).psnr.mean().reset_index()
means.to_csv(R/'QUALITY_SUMMARY.csv',index=False)
quality=q[q.variant!='full'].pivot(index=['scene','ratio','frame','round'],columns='variant',values='psnr')
quality['extra_loss_student_fast']=quality.student_cpu-quality.student_fast
quality['teacher_loss_student_fast']=quality.teacher_cpu-quality.student_fast
quality.to_csv(R/'PAIRED_QUALITY.csv')
extra=quality.extra_loss_student_fast
condition=extra.groupby(level=['scene','ratio']).mean()
assert extra.mean()<=.05 and condition.max()<=.2

def value(scene,variant,ratio,scope):
    x=agg[(agg.scene==scene)&(agg.variant==variant)&(agg.ratio==ratio)&(agg.scope==scope)]
    assert len(x)==1,(scene,variant,ratio,scope)
    return float(x.iloc[0].ms)

def geometric(v):return float(np.exp(np.log(v).mean()))

pairs=[]
frame=s.groupby(['scene','ratio','scope','round','frame','variant']).ms.median().unstack('variant')
for key,row in frame[frame.student_fast.notna()].iterrows():
    if key[2] not in ['cpu_prepare','pipeline_wall','stage_2','stage_4']:continue
    pairs.append(dict(zip(['scene','ratio','scope','round','frame'],key),student_vs_teacher=row.teacher_cpu/row.student_fast,
        fast_vs_original_student=row.student_cpu/row.student_fast))
pairs=pd.DataFrame(pairs);pairs.to_csv(R/'PAIRED_SPEEDUP.csv',index=False)
table=[]
for scene in s.scene.unique():
    for ratio in sorted(s[s.variant=='student_fast'].ratio.unique()):
        rec=dict(scene=scene,ratio=ratio)
        rec['K']=float(agg[(agg.scene==scene)&(agg.variant=='student_fast')&(agg.ratio==ratio)].K.iloc[0])
        for scope in ['cpu_prepare','stage_0','stage_2','stage_3','stage_4','h2d_event','raster_event','raster_wall','pipeline_wall']:
            rec[scope]=value(scene,'student_fast',ratio,scope)
        rec['full_raster_event']=value(scene,'full',1.,'raster_event')
        rec['full_raster_wall']=value(scene,'full',1.,'raster_wall')
        rec['full_pipeline_wall']=value(scene,'full',1.,'pipeline_wall')
        rec['raster_speedup']=rec['full_raster_event']/rec['raster_event']
        rec['pipeline_speedup']=rec['full_pipeline_wall']/rec['pipeline_wall']
        rec['cpu_teacher']=value(scene,'teacher_cpu',ratio,'cpu_prepare')
        rec['cpu_original_student']=value(scene,'student_cpu',ratio,'cpu_prepare')
        rec['extra_psnr_loss']=float(condition.loc[scene,ratio])
        rec['psnr']=float(means[(means.scene==scene)&(means.variant=='student_fast')&(means.ratio==ratio)].psnr.iloc[0])
        table.append(rec)
t=pd.DataFrame(table);t.to_csv(R/'CPU_GPU_SEPARATED_RESULTS.csv',index=False)
summary=dict(passed=True,samples=len(s),checks=len(checks),quality_rows=len(q),conditions=len(t),
    protected_files=valid['protected_files'],foreign_gpu=job['foreign_active_samples'],foreign_cpu=job['foreign_cpu_active_samples'],
    cpu_fast_vs_teacher=geometric(pairs[pairs.scope=='cpu_prepare'].student_vs_teacher),
    cpu_fast_vs_original_student=geometric(pairs[pairs.scope=='cpu_prepare'].fast_vs_original_student),
    pipeline_fast_vs_original_student=geometric(pairs[pairs.scope=='pipeline_wall'].fast_vs_original_student),
    raster_faster_conditions=int((t.raster_speedup>1).sum()),pipeline_faster_conditions=int((t.pipeline_speedup>1).sum()),
    mean_extra_psnr_loss=float(extra.mean()),worst_frame_extra_psnr_loss=float(extra.max()),worst_condition_extra_psnr_loss=float(condition.max()))
summary['rejected_attempts']=rejected
(R/'ANALYSIS_VALIDATION.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
lines=['# CPU蒸馏Compensator＋新低点数渲染器实测', '',
    '## 结论', '',
    f"同一冻结Teacher/地图/Selector/学生权重，CPU使用报告指定的最终context版本（2/1/1），GPU使用已验收G Renderer。共{len(t)}个场景Ratio条件，纯GPU渲染{summary['raster_faster_conditions']}项快于GPU常驻Full，包含CPU准备、实际属性上传及同步的完整流程{summary['pipeline_faster_conditions']}项快于Full。CPU推理时间单列，没有混入纯GPU渲染时间。", '',
    f"同视角/同Ratio/同轮配对后，全部CPU准备相对CPU Teacher加速{summary['cpu_fast_vs_teacher']:.3f}×，相对原CPU学生额外{summary['cpu_fast_vs_original_student']:.3f}×；新旧CPU学生的完整流程加速{summary['pipeline_fast_vs_original_student']:.3f}×。这些是本轮实测，不套用上一轮GPU推理速度。", '',
    '## 范围与计时口径', '',
    '- Intel i5-9600K／6个原生工作线程，Torch辅助1线程；RTX3090执行渲染。Teacher 25,742参数，学生3,534参数。CPU网络是AVX2/FMA，非CPU模拟GPU计时。',
    '- Replica office1单目、office2 RGB-D、room0 RGB-D、room2单目；Ratio=1%、5%、10%、20%、30%、40%、50%、70%。每场景3连续＋3跳转保存相机，双轮切换版本顺序。是冻结推理微测，不是全轨迹或在线SLAM。',
    '- 本轮采用CPU专项报告的六个保存相机，而上一轮GPU网络采用十八个轨迹相机；实际可见候选数与K不同。仅在本轮同输入之间配对，不跨两份报告直接相减归因。',
    '- CPU总准备：候选/Anchor上下文、Selector、预算选点、Compensator与物理解码、打包。CPU Selector与Compensator分项来自独立C++插桩；其中Compensator项包含解码/打包，不冒称纯MLP。候选阶段已经融合部分上下文。',
    '- CPU计时使用墙钟且区间内无CUDA调用；上传是同样字节数的连续Pinned Buffer单次H2D的CUDA Event诊断；GPU渲染是属性及相机已就绪后的CUDA Event。另给渲染墙钟，包含提交和最终同步。',
    '- 完整流程使用实际原生三槽位交接：每帧重新执行CPU候选及网络，上传属性和相机，等待依赖，G渲染并同步。不进行跨帧重叠，不缓存旧视角结果。该值独立实测，不等于表中各项中位数之和。',
    '- Full不执行网络：Master属性一次上传后常驻GPU，每视角仅上传51个相机float。LoD当前CPU部署每帧上传补偿后选中属性。两者数据驻留不同，完整流程比较包含这种真实部署成本。',
    '- 相机矩阵构造、checkpoint读取、GT读取、指标、图片编码和显示均在计时外。临时分配和Python入口调度在所属实际路径内，未使用历史GPU推理结果代替CPU测试。', '',
    '## 分离结果', '',
    '单位ms。K为六个视角实际选择数的中位数，可能为半整数。GPU项包含新G完整光栅过程；完整流程不是训练Mapping。', '',
    '| 场景 | Ratio | K | CPU全部准备 | CPU Selector | CPU补偿＋解码打包 | H2D Event | GPU渲染Event | 完整流程墙钟 | LoD PSNR |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for _,r in t.iterrows():
    lines.append(f"| {r.scene.replace('Replica__','').replace('__','/')} | {r.ratio:g} | {r.K:g} | {r.cpu_prepare:.4f} | {r.stage_2:.4f} | {r.stage_4:.4f} | {r.h2d_event:.4f} | {r.raster_event:.4f} | {r.pipeline_wall:.4f} | {r.psnr:.3f} |")
lines+=['','## 与直接Full的配对比较','','| 场景 | Ratio | Full GPU ms | LoD GPU ms | GPU加速 | Full完整ms | LoD完整ms | 完整加速 |','|---|---:|---:|---:|---:|---:|---:|---:|']
for _,r in t.iterrows():
    lines.append(f"| {r.scene.replace('Replica__','').replace('__','/')} | {r.ratio:g} | {r.full_raster_event:.4f} | {r.raster_event:.4f} | {r.raster_speedup:.3f}× | {r.full_pipeline_wall:.4f} | {r.pipeline_wall:.4f} | {r.pipeline_speedup:.3f}× |")
lines+=['','## CPU Teacher、原学生与本版本','','| 场景 | Ratio | Teacher CPU总ms | 原学生CPU总ms | 本版本CPU总ms |','|---|---:|---:|---:|---:|']
for _,r in t.iterrows():lines.append(f"| {r.scene.replace('Replica__','').replace('__','/')} | {r.ratio:g} | {r.cpu_teacher:.4f} | {r.cpu_original_student:.4f} | {r.cpu_prepare:.4f} |")
lines+=['','## 正确性、负载与质量限制','',
    f"- {len(checks)}个case均通过Score/ID身份核对、finite检查、同属性包原A与新G RGB/radii逐位比较，以及CPU直接上传与原生交接渲染逐位比较。学生优化前后XYZ/DC/SH/means2D保持相同，Scale/Opacity/Rotation差异保存在checks.csv的attribute_errors。G逐位结果不代表学生与Teacher逐位相同。",
    f"- 本轮六视角GT配对中，CPU专项优化相对原CPU学生平均额外PSNR损失{extra.mean():+.8f}dB，最差单帧{extra.max():.8f}dB，最差场景Ratio均值{condition.max():.8f}dB，通过此前既定平均≤.05dB、各条件均值≤.2dB门限。学生相对Teacher的已有损失单列PAIRED_QUALITY.csv，不能用本轮微小新增误差抹去蒸馏损失。",
    '- 这六个保存相机可能参与过训练/选模，不是新留出质量或全轨迹验证。向量解码包含已获允许的FP32微差，不能称全属性等价后端。',
    f"- 正式监控{job['observations']}次，外部活跃GPU样本{job['foreign_active_samples']}，外部单进程>10% CPU样本{job['foreign_cpu_active_samples']}；全部正式样本保留，p95/最大值和双轮值见CSV。低频监控无法排除所有中断/瞬时调度。",
    f"- 前{len(rejected)}次正式尝试数值验收通过，但监控发现Xorg约1%–2% SM活动，整次性能数据未采用，分别保留于{', '.join(rejected)}；随后完整重跑，没有只删除慢样本，也没有终止外部进程。",
    f"- {valid['protected_files']}个受保护文件前后SHA不变；读取已冻结.so且禁用JIT，没有修改原报告、网络权重、生产目录、共享conda或训练配置，没有commit/push，没有保存渲染图片。", '',
    '## 交付与后续判断','',
    '优先分别看CPU总准备与GPU渲染：降低高斯数量是否减少GPU渲染成本，与CPU网络开销能否被抵消，是两个独立结论。当前CPU方案的上传和单帧同步不能省略；若进一步研究CPU/GPU流水线，应另报吞吐与首帧延迟，不能把异步吞吐当成本轮单帧加速。', '',
    '目录：`/home/prmi/zwj/experiments/cpu_student_new_renderer_20261110`。原始samples/checks/quality、双轮汇总、逐视角配对、模型身份与完整性证据均在reports/，正式进程身份、返回码及负载在logs/。']
(HERE/'CPU_STUDENT_NEW_RENDERER_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
