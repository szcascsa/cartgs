"""从原始CSV/SQLite重算统计，禁止把event span写成kernel execution。"""
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import tarfile
import re

ROOT=Path(__file__).resolve().parent
E=ROOT/'evidence'
if not E.exists():
    E.mkdir()
    with tarfile.open(ROOT/'evidence.tar.gz') as t:t.extractall(E,filter='data')
manifest=json.loads((E/'evidence_manifest.json').read_text())
assert all(hashlib.sha256((E/p).read_bytes()).hexdigest()==h['sha256'] for p,h in manifest.items())
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def write(p,rows):
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def summary(v):
    a=sorted(v);return dict(samples=len(a),mean=statistics.mean(a),median=statistics.median(a),p95=a[min(len(a)-1,int(len(a)*.95))],p99=a[min(len(a)-1,int(len(a)*.99))])
stats=[]
for run in ['native1','native2']:
    rows=read(E/run/'native_stages.csv')
    for case in sorted({r['case'] for r in rows}):
        c=[r for r in rows if r['case']==case]
        for stage in sorted({r['stage'] for r in c}):
            s=[r for r in c if r['stage']==stage]
            for metric in ['host_submit_ms','cuda_elapsed_ms']:
                stats.append(dict(run=run,case=case,N=s[0]['N'],K=s[0]['K'],stage=stage,metric=metric,**summary([float(r[metric]) for r in s])))
        stats.append(dict(run=run,case=case,N=c[0]['N'],K=c[0]['K'],stage='cycle',metric='synchronized_wall_ms',
                          **summary([float(r['cycle_wall_ms']) for r in c if r['stage']=='adam'])))
write(ROOT/'native_statistics.csv',stats)
online=[]
for run in ['trace1','trace2']:
    rows=read(E/run/'mapping_trace.csv')
    for phase in ['all','0','1']:
        c=[r for r in rows if phase=='all' or r['phase']==phase]
        for stage in list(rows[0])[2:]:
            online.append(dict(run=run,phase=phase,stage=stage,scope='per_mapping_including_zero_calls',**summary([float(r[stage]) for r in c])))
            active=[float(r[stage]) for r in c if float(r[stage])>0]
            if active:online.append(dict(run=run,phase=phase,stage=stage,scope='nonzero_scope_calls',**summary(active)))
    # Current online schedule is 1/5/10/15/Full. Infer only when source confirms and no recovery.
    for label,phase in [('lod1',1),('lod5',2),('lod10',3),('lod15',4),('Full',0)]:
        c=[r for r in rows if r['phase']=='0' and int(r['iteration'])%5==phase and int(r['iteration'])>=3000]
        if c:online.append(dict(run=run,phase='online_post3000_'+label,stage='total',scope='source_schedule_inferred',**summary([float(r['total']) for r in c])))
    events={int(i) for i in re.findall(r'\[Alternating Topology\].*?mapping=(\d+)',(E/run/'training.log').read_text(encoding='utf-8'))}
    for label,filtered in [('topology_event',[r for r in rows if int(r['iteration']) in events]),
                            ('full_stats_without_topology',[r for r in rows if float(r['densify'])>0 and int(r['iteration']) not in events])]:
        for stage in ['total','densify']:
            if filtered:online.append(dict(run=run,phase=label,stage=stage,scope='actual_source_event',**summary([float(r[stage]) for r in filtered])))
write(ROOT/'online_statistics.csv',online)
db=sqlite3.connect(E/'native_profile.sqlite');db.row_factory=sqlite3.Row
tables=[r[0] for r in db.execute("select name from sqlite_master where type='table'")]
schema={t:[dict(r) for r in db.execute('pragma table_info('+t+')')] for t in tables}
(ROOT/'nsight_schema.json').write_text(json.dumps(schema,indent=2))
print('ONLINE',[(r['run'],r['phase'],round(r['mean'],3),round(r['p95'],3)) for r in online if r['stage']=='total' and r['scope']=='per_mapping_including_zero_calls'])
print('NATIVE',[(r['run'],r['case'],round(r['mean'],3),round(r['p95'],3)) for r in stats if r['stage']=='cycle'])
print('TABLES',tables)

# correlationId关联CUDA launch，归属NVTX scope；仅取每case rep1..3，排除warmup/fixture。
strings={r['id']:r['value'] for r in db.execute('select * from StringIds')}
nvtx=[]
for r in db.execute('select * from NVTX_EVENTS where end is not null'):
    x=dict(r);x['label']=x['text'] or strings.get(x['textId'],'');nvtx.append(x)
cases={}
for label in ['Full','lod_0.010000','lod_0.050000','lod_0.100000','lod_0.150000']:
    ranges=sorted([x for x in nvtx if x['label']==label],key=lambda x:x['start'])
    assert len(ranges)==14,(label,len(ranges))
    for i,x in enumerate(ranges[11:]):cases[(label,i+1)]=x
runtime={r['correlationId']:dict(r) for r in db.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')}
def location(start):
    for (case,rep),parent in cases.items():
        if parent['start']<=start<=parent['end']:
            children=[n for n in nvtx if n['start']>=parent['start'] and n['end']<=parent['end'] and n['start']<=start<=n['end'] and n['label']!=case]
            return case,rep,min(children,key=lambda n:n['end']-n['start'])['label'] if children else 'cycle_control'
    return None
timeline=[];kernels={};apis={}
for t,kind in [('CUPTI_ACTIVITY_KIND_KERNEL','kernel'),('CUPTI_ACTIVITY_KIND_MEMCPY','memcpy'),('CUPTI_ACTIVITY_KIND_MEMSET','memset')]:
    for raw in db.execute('select * from '+t):
        r=dict(raw);api=runtime.get(r['correlationId'])
        if not api:continue
        loc=location(api['start'])
        if not loc:continue
        case,rep,stage=loc
        name=strings.get(r.get('demangledName'),'') if kind=='kernel' else kind+'_'+str(r.get('copyKind',''))
        timeline.append(dict(case=case,repeat=rep,stage=stage,kind=kind,name=name,
                cpu_api=strings.get(api['nameId'],''),cpu_start_ns=api['start'],cpu_end_ns=api['end'],
                gpu_start_ns=r['start'],gpu_end_ns=r['end'],gpu_duration_us=(r['end']-r['start'])/1000,
                queue_delay_us=max(0,r['start']-api['end'])/1000,bytes=r.get('bytes',0),stream=r['streamId'],
                correlationId=r['correlationId'],measurement='Nsight instrumented;external GPU load present'))
        if kind=='kernel':
            key=(case,stage,name);kernels.setdefault(key,[]).append((r['end']-r['start'])/1e6)
for api in runtime.values():
    loc=location(api['start'])
    if not loc:continue
    case,rep,stage=loc;name=strings.get(api['nameId'],'')
    apis.setdefault((case,stage,name),[]).append((api['end']-api['start'])/1e6)
write(ROOT.parents[2]/'cpu_gpu_timeline.csv',timeline)
breakdown=[]
for (case,stage,name),times in kernels.items():
    breakdown.append(dict(case=case,stage=stage,kernel=name,launches=len(times),iterations=3,launches_per_iteration=len(times)/3,
                          kernel_total_ms=sum(times),kernel_ms_per_iteration=sum(times)/3,**summary(times)))
write(ROOT.parents[2]/'kernel_launch_breakdown.csv',breakdown)
apirows=[dict(case=c,stage=s,api=n,total_ms=sum(t),calls=len(t),per_iteration_ms=sum(t)/3) for (c,s,n),t in apis.items()]
write(ROOT/'cuda_api_statistics.csv',apirows)
stage=[]
for case in sorted({r['case'] for r in timeline}):
    for name in sorted({r['stage'] for r in timeline if r['case']==case}):
        c=[r for r in timeline if r['case']==case and r['stage']==name]
        k=[r for r in c if r['kind']=='kernel']
        stage.append(dict(case=case,stage=name,kernel_launches_per_iteration=len(k)/3,
                          kernel_ms_per_iteration=sum(r['gpu_duration_us'] for r in k)/3000,
                          memcpy_per_iteration=sum(r['kind']=='memcpy' for r in c)/3,
                          memcpy_bytes_per_iteration=sum(r['bytes'] for r in c if r['kind']=='memcpy')/3,
                          queue_delay_us_mean=statistics.mean(r['queue_delay_us'] for r in c)))
write(ROOT/'nsight_stage_statistics.csv',stage)
print('STAGES5',[(x['stage'],x['kernel_launches_per_iteration'],round(x['kernel_ms_per_iteration'],4)) for x in stage if x['case']=='lod_0.050000'])

if (ROOT/'evidence_online/online_kernel_breakdown.csv').exists():
    for x in read(ROOT/'evidence_online/online_kernel_breakdown.csv'):
        breakdown.append(dict(case='production_process_profiled',stage='initialization_mapping_shutdown_unsegmented',kernel=x['kernel'],
                              launches=x['launches'],iterations='NA',launches_per_iteration='NA',kernel_total_ms=x['kernel_total_ms'],
                              kernel_ms_per_iteration='NA',samples=x['launches'],mean=float(x['kernel_mean_us'])/1000,
                              median='NA',p95='NA',p99='NA'))
    write(ROOT.parents[2]/'kernel_launch_breakdown.csv',breakdown)
    for x in read(ROOT/'evidence_online/online_timeline_sample.csv'):
        timeline.append(dict(case='production_process_profiled',repeat='NA',stage='initialization_mapping_shutdown_unsegmented',
                             kind=x['kind'],name=x['name'],cpu_api='NA_see_full_SQLite',cpu_start_ns=x['cpu_start_ns'],cpu_end_ns=x['cpu_end_ns'],
                             gpu_start_ns=x['gpu_start_ns'],gpu_end_ns=x['gpu_end_ns'],gpu_duration_us=x['gpu_duration_us'],
                             queue_delay_us=x['queue_delay_us'],bytes=x['bytes'],stream=x['stream'],correlationId=x['correlationId'],
                             measurement='sample every1000 events;full production timeline in evidence_online/online_cpu_gpu_timeline.csv.gz'))
    write(ROOT.parents[2]/'cpu_gpu_timeline.csv',timeline)
