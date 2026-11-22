"""学生对Teacher的组件/全准备/真实渲染加速，质量代价独立列出。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
PRIMARY=['local0.1','local0.2','local0.3','local0.4','local0.5']
def read(name):return json.loads((HERE/'reports'/name).read_text())
def geo(x):return float(np.exp(np.log(x).mean()))
def verify_job(name):
    job=read('jobs/'+name+'.json');assert job['measurement_valid'] and not job['foreign_active_samples'] and not job['foreign_cpu_active_samples'],job
def aggregate(rows,keys,column):return rows.groupby(keys+['variant']).agg(ms=(column,'median'),p95=(column,lambda v:v.quantile(.95)),maximum_ms=(column,'max'),samples=(column,'size')).reset_index()
def paired(rows,keys,column):
    d=aggregate(rows,keys,column);a=d[d.variant=='teacher'][keys+['ms','p95']].rename(columns={'ms':'teacher_ms','p95':'teacher_p95'});b=d[d.variant=='student'].merge(a,on=keys)
    b['speedup']=b.teacher_ms/b.ms;b['reduction_percent']=100*(1-b.ms/b.teacher_ms);return b

frames=[];stages=[];checks=[];memory=[]
for run in ['formal01','formal02']:
    verify_job(run);assert read(run+'/SUMMARY.json')['passed'] and not read(run+'/SUMMARY.json')['cuda_initialized']
    for sink,name in [(frames,'samples'),(stages,'stages'),(checks,'checks'),(memory,'memory')]:sink.append(pd.read_csv(HERE/'reports'/run/(name+'.csv')))
d=pd.concat(frames);st=pd.concat(stages);c=pd.concat(checks);m=pd.concat(memory);groups=c.groupby(['scene','round','condition','frame'])
assert groups.ids_sha256.nunique().eq(1).all() and groups.score_sha256.nunique().eq(1).all() and c.finite.all()
keys=['scene','round','condition'];cpu=paired(d,keys,'ms');sizes=d.groupby(keys).agg(N=('N','first'),M=('M','median'),K=('K','median')).reset_index();cpu=cpu.merge(sizes,on=keys)
component=paired(st,keys+['stage'],'ms')
for name,table in [('CPU_PAIRED_RESULTS',cpu),('COMPONENT_PAIRED_RESULTS',component),('CPU_MEMORY',m),('CPU_CORRECTNESS',c)]:table.to_csv(HERE/'reports'/(name+'.csv'),index=False)
ag=[]
for condition,g in cpu.groupby('condition'):
    comp=component[(component.condition==condition)&(component.stage==4)]
    ag.append(dict(condition=condition,CPU_geometric_speedup=geo(g.speedup),CPU_teacher_ms=g.teacher_ms.median(),CPU_student_ms=g.ms.median(),CPU_slower_rounds=int((g.speedup<1).sum()),scene_rounds=len(g),
        compensator_geometric_speedup=geo(comp.speedup) if len(comp) else None,compensator_teacher_ms=comp.teacher_ms.median() if len(comp) else None,compensator_student_ms=comp.ms.median() if len(comp) else None))
pd.DataFrame(ag).to_csv(HERE/'reports/CPU_AGGREGATE.csv',index=False)
hr=[];ras=[]
for run in ['handoff01','handoff02']:
    verify_job(run);assert read(run+'/SUMMARY.json')['passed'];hr.append(pd.read_csv(HERE/'reports'/run/'samples.csv'));ras.append(pd.read_csv(HERE/'reports'/run/'raster.csv'))
handoff=pd.concat(hr);h=paired(handoff,keys,'total_ms');host=paired(handoff,keys,'host_ms');raster=paired(pd.concat(ras),keys,'gpu_event_ms')
for name,table in [('HANDOFF_PAIRED_RESULTS',h),('HOST_PAIRED_RESULTS',host),('RASTER_DIAGNOSTICS',raster)]:table.to_csv(HERE/'reports'/(name+'.csv'),index=False)
q=pd.read_csv(HERE/'reports/quality01/quality.csv');assert q.same_ids.all() and read('quality01/SUMMARY.json')['passed']
qc=q.groupby(['scene','condition']).agg(total_psnr_loss=('total_psnr_loss','mean'),distillation_psnr_loss=('distillation_psnr_loss','mean'),backend_psnr_loss=('backend_psnr_loss','mean'),worst_backend_psnr_loss=('backend_psnr_loss','max'),
    teacher_psnr=('teacher_psnr','mean'),student_psnr=('student_psnr','mean'),teacher_ssim=('teacher_ssim','mean'),student_ssim=('student_ssim','mean'),teacher_lpips=('teacher_lpips','mean'),student_lpips=('student_lpips','mean'),
    backend_minimum_image_psnr=('backend_image_equivalence_psnr','min'),K=('K','median')).reset_index()
q.to_csv(HERE/'reports/QUALITY_ALL_FRAMES.csv',index=False);qc.to_csv(HERE/'reports/QUALITY_PER_CONDITION.csv',index=False)
assert q.backend_psnr_loss.mean()<=.05 and qc.backend_psnr_loss.max()<=.2
primary=cpu[cpu.condition.isin(PRIMARY)];comp=component[(component.condition.isin(PRIMARY))&(component.stage==4)];hp=h[h.condition.isin(PRIMARY)];qp=q[q.condition.isin(PRIMARY)]
result=dict(passed=True,scenes=int(d.scene.nunique()),CPU_samples=len(d),component_records=len(st),checks=len(c),handoff_samples=len(handoff),quality_rows=len(q),
    primary_CPU_speedup=geo(primary.speedup),primary_compensator_speedup=geo(comp.speedup),primary_handoff_speedup=geo(hp.speedup),primary_CPU_slower_rounds=int((primary.speedup<1).sum()),primary_handoff_slower_rounds=int((hp.speedup<1).sum()),
    primary_mean_teacher_student_psnr_loss=float(qp.total_psnr_loss.mean()),primary_worst_frame_teacher_student_psnr_loss=float(qp.total_psnr_loss.max()),primary_mean_backend_psnr_loss=float(qp.backend_psnr_loss.mean()),
    worst_backend_psnr_loss=float(q.backend_psnr_loss.max()),minimum_backend_image_psnr=float(q.backend_image_equivalence_psnr.min()),score_ids_identical=True)
(HERE/'reports/RESULTS_SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(pd.DataFrame(ag).round(5).to_string(index=False));print(json.dumps(result,indent=2))
