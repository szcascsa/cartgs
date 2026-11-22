"""分析成功的整进程Nsight；不伪造每Mapping/ratio的NVTX归属。"""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import tarfile

R=Path('/home/prmi/zwj/anchor-selector-r3-performance-20261022')
db=sqlite3.connect(f'file:{R}/online_profile_start2.sqlite?mode=ro',uri=True);db.row_factory=sqlite3.Row
strings={x['id']:x['value'] for x in db.execute('select * from StringIds')}
summary=dict(scope='whole production process: initialization+online+tail+shutdown;not per-Mapping',
             runtime='Nsight instrumented and external GPU contention',
             per_module_attribution='not available: production no per-stage NVTX;use separate MappingProfiler and native fixtures',
             previous_delayed_capture='exit139 near Mapping1500;cause unresolved;excluded from statistics')
kernel=[]
for x in db.execute('select demangledName,count(*) as n,sum(end-start) as total_ns,min(end-start) as min_ns,max(end-start) as max_ns from CUPTI_ACTIVITY_KIND_KERNEL group by demangledName'):
    kernel.append(dict(kernel=strings[x['demangledName']],launches=x['n'],kernel_total_ms=x['total_ns']/1e6,
                       kernel_mean_us=x['total_ns']/x['n']/1000,kernel_min_us=x['min_ns']/1000,kernel_max_us=x['max_ns']/1000,
                       scope='whole instrumented process;not per Mapping'))
def write(name,rows):
    with (R/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write('online_kernel_breakdown.csv',kernel)
apis=[]
for x in db.execute('select nameId,count(*) as n,sum(end-start) as total_ns,max(end-start) as max_ns from CUPTI_ACTIVITY_KIND_RUNTIME group by nameId'):
    apis.append(dict(api=strings[x['nameId']],calls=x['n'],cpu_api_total_ms=x['total_ns']/1e6,max_ms=x['max_ns']/1e6))
write('online_cuda_api_statistics.csv',apis)
summary['kernel_launches']=sum(x['launches'] for x in kernel)
summary['kernel_ms_sum']=sum(x['kernel_total_ms'] for x in kernel)
summary['gpu_kernel_names']=len(kernel)
summary['memcpy']=[dict(x) for x in db.execute('select copyKind,count(*) as calls,sum(bytes) as bytes,sum(end-start)/1000000.0 as gpu_ms from CUPTI_ACTIVITY_KIND_MEMCPY group by copyKind')]
summary['streams']=[dict(x) for x in db.execute('select streamId,count(*) as kernel_launches from CUPTI_ACTIVITY_KIND_KERNEL group by streamId')]
summary['api_waits']=[x for x in apis if any(k in x['api'] for k in ['Synchronize','Memcpy','Malloc','Free'])]
summary['mapping_state']=json.loads(next((R/'online_nsight_start2').glob('*_shutdown/ply/mapping_stage_state.json')).read_text())
summary['anchor_state']=json.loads(next((R/'online_nsight_start2').glob('*_shutdown/ply/point_cloud/iteration_*/anchor_selector_state.json')).read_text())
(R/'online_profile_summary.json').write_text(json.dumps(summary,indent=2))
columns=['kind','name','cpu_start_ns','cpu_end_ns','gpu_start_ns','gpu_end_ns','gpu_duration_us','queue_delay_us','bytes','stream','correlationId','scope']
samples=[];count=0
with gzip.open(R/'online_cpu_gpu_timeline.csv.gz','wt',newline='') as f:
    w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
    for table,kind in [('CUPTI_ACTIVITY_KIND_KERNEL','kernel'),('CUPTI_ACTIVITY_KIND_MEMCPY','memcpy')]:
        for x in db.execute('select g.*,r.start as cpu_start,r.end as cpu_end from '+table+' g left join CUPTI_ACTIVITY_KIND_RUNTIME r on g.correlationId=r.correlationId order by g.start'):
            row=dict(x);name=strings.get(row.get('demangledName'),'memcpy_'+str(row.get('copyKind','')))
            y=dict(kind=kind,name=name,cpu_start_ns=row['cpu_start'],cpu_end_ns=row['cpu_end'],gpu_start_ns=row['start'],gpu_end_ns=row['end'],
                   gpu_duration_us=(row['end']-row['start'])/1000,queue_delay_us=max(0,row['start']-(row['cpu_end'] or row['start']))/1000,
                   bytes=row.get('bytes',0),stream=row['streamId'],correlationId=row['correlationId'],scope='production_whole_process_unsegmented')
            w.writerow(y)
            if count%1000==0:samples.append(y)
            count+=1
write('online_timeline_sample.csv',samples)
files=[R/x for x in ['online_profile_summary.json','online_kernel_breakdown.csv','online_cuda_api_statistics.csv',
    'online_cpu_gpu_timeline.csv.gz','online_timeline_sample.csv','online_profile_start2.nsys-rep','online_profile_start2.sqlite',
    'online_profile_start2_command.json','online_profile_start2.log','online_profile.log','online_profile_command.json']]
files+=[R/'online_nsight_start2/training.log',R/'online_nsight_start2/mapping_trace.csv',R/'online_nsight/training.log']
manifest={str(x.relative_to(R)):dict(bytes=x.stat().st_size,sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in files}
(R/'online_profile_manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(R/'online_profile_evidence.tar.gz','w:gz',compresslevel=1) as t:
    for x in files+[R/'online_profile_manifest.json']:t.add(x,arcname=str(x.relative_to(R)))
print(json.dumps(dict(passed=True,kernels=summary['kernel_launches'],kernel_ms=summary['kernel_ms_sum'],timeline_rows=count,
                      mappings=sum(summary['mapping_state'][p]['mappings'] for p in ['online','tail']),
                      compensation_forwards=summary['anchor_state']['compensation_forwards'],archive_bytes=(R/'online_profile_evidence.tar.gz').stat().st_size)))
