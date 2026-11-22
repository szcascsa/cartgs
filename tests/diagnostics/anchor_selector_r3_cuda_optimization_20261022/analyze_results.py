"""从原始计时/Profiler/在线trace生成机器可读结果；不混淆Event与纯kernel时间。"""
import csv
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

R=Path('/home/prmi/zwj/anchor-selector-r3-cuda-optimization-20261022')
def stats(values):
    a=np.asarray(values,dtype=float);return dict(samples=len(a),mean=float(a.mean()),p50=float(np.quantile(a,.5)),p95=float(np.quantile(a,.95)),p99=float(np.quantile(a,.99)))
def write(name,rows):
    if rows:pd.DataFrame(rows).to_csv(R/name,index=False)

native=[]
for run in ['legacy_a','legacy_b','adam','adam_decode','fused_a','fused_b']:
    a=pd.read_csv(R/run/'native_stages.csv')
    for case,c in a.groupby('case'):
        for stage,g in c.groupby('stage'):
            for key in ['host_submit_ms','cuda_elapsed_ms']:
                native.append(dict(scope='fixed_checkpoint',run=run,case=case,N=int(g.iloc[0].N),K=int(g.iloc[0].K),stage=stage,metric=key,**stats(g[key])))
        cycle=c[c.stage=='adam'];native.append(dict(scope='fixed_checkpoint',run=run,case=case,N=int(cycle.iloc[0].N),K=int(cycle.iloc[0].K),stage='cycle',metric='synchronized_wall_ms',**stats(cycle.cycle_wall_ms)))
        forward=c[~c.stage.isin(['adam','backward'])].groupby('repeat').cuda_elapsed_ms.sum()
        native.append(dict(scope='fixed_checkpoint',run=run,case=case,N=int(cycle.iloc[0].N),K=int(cycle.iloc[0].K),stage='forward_stage_event_sum',metric='cuda_elapsed_ms',**stats(forward)))
write('before_after_benchmarks.csv',native)
launches=[];detail=[];apis=[]
for run in ['profile_legacy','profile_fused']:
    db=sqlite3.connect(R/(run+'.sqlite'));db.row_factory=sqlite3.Row
    strings={x['id']:x['value'] for x in db.execute('select * from StringIds')}
    scopes=[dict(x) for x in db.execute('select * from NVTX_EVENTS where end is not null')]
    for x in scopes:x['label']=x['text'] or strings.get(x['textId'],'')
    parents=[]
    for case in ['Full','lod_0.010000','lod_0.050000','lod_0.100000','lod_0.150000']:
        candidates=sorted([x for x in scopes if x['label']==case],key=lambda x:x['start']);assert len(candidates)==14
        parents.extend([(case,rep+1,x) for rep,x in enumerate(candidates[11:])])
    def location(start):
        for case,rep,p in parents:
            if p['start']<=start<=p['end']:
                sub=[x for x in scopes if x['start']>=p['start'] and x['end']<=p['end'] and x['start']<=start<=x['end'] and x['label']!=case]
                return case,rep,min(sub,key=lambda x:x['end']-x['start'])['label'] if sub else 'cycle_control'
        return None
    runtime={x['correlationId']:dict(x) for x in db.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')}
    items=[]
    for x in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'):
        t=runtime.get(x['correlationId']);loc=location(t['start']) if t else None
        if loc:
            case,rep,stage=loc;items.append(dict(run=run,case=case,repeat=rep,stage=stage,kernel=strings[x['demangledName']],
                kernel_ms=(x['end']-x['start'])/1e6,queue_us=max(0,x['start']-t['end'])/1000))
    a=pd.DataFrame(items);detail.extend(items)
    for (case,stage),g in a.groupby(['case','stage']):
        launches.append(dict(run=run,case=case,stage=stage,launches_per_iteration=len(g)/3,kernel_ms_per_iteration=g.kernel_ms.sum()/3,queue_us_mean=g.queue_us.mean()))
    for case,g in a.groupby('case'):launches.append(dict(run=run,case=case,stage='all',launches_per_iteration=len(g)/3,kernel_ms_per_iteration=g.kernel_ms.sum()/3,queue_us_mean=g.queue_us.mean()))
    for t in runtime.values():
        loc=location(t['start'])
        if loc:apis.append(dict(run=run,case=loc[0],repeat=loc[1],stage=loc[2],api=strings[t['nameId']],host_ms=(t['end']-t['start'])/1e6))
write('kernel_launch_comparison.csv',launches);write('kernel_execution_detail.csv',detail);write('cuda_api_detail.csv',apis)
online=[];states=[]
for name in ['online_legacy_a','online_fused_a','online_fused_b','online_legacy_b']:
    p=R/name/'mapping_trace.csv'
    if not p.exists():continue
    a=pd.read_csv(p)
    for phase,g in a.groupby('phase'):
        for stage in list(a)[2:]:online.append(dict(run=name,phase='online' if phase==0 else 'tail',stage=stage,scope='all_mappings_including_zero_calls',**stats(g[stage])))
    # Tail中每轮Full+LoD；在线部分严格按生产iteration%5标注，排除初始化前段。
    g=a[(a.phase==0)&(a.iteration>=3000)]
    for modulo,label in [(0,'Full'),(1,'1%'),(2,'5%'),(3,'10%'),(4,'15%')]:
        selected=g[g.iteration%5==modulo]
        if len(selected):online.append(dict(run=name,phase='online_post3000_'+label,stage='total',scope='source_schedule_inferred',**stats(selected.total)))
    cp=next((R/name).glob('*_shutdown/ply'));state=json.loads((cp/'mapping_stage_state.json').read_text())
    anchor=json.loads(next((cp/'point_cloud').glob('iteration_*/anchor_selector_state.json')).read_text())
    states.append(dict(run=name,checkpoint=str(cp),mapping=state,anchor=anchor))
write('online_performance_comparison.csv',online);(R/'online_model_states.json').write_text(json.dumps(states,indent=2))
parity=[]
for phase,name in [('phase1_final','adam'),('phase2_final','decode'),('phase3_final','pack')]:
    a=pd.read_csv(R/phase/(name+'_parity.csv'));a['scope']=name;parity.extend(a.to_dict('records'))
for x in pd.read_csv(R/'fixed_replay_parity.csv').to_dict('records'):x['scope']='fixed_replay';parity.append(x)
write('correctness_parity.csv',parity)
scaling=pd.read_csv(R/'scaling.csv');summary=[]
for keys,g in scaling.groupby(['N','K','module','implementation','stage']):
    summary.append(dict(zip(['N','K','module','implementation','stage'],keys),**stats(g.cuda_ms),host_p50=g.host_ms.median(),peak_extra_bytes=g.peak_extra_bytes.max(),allocations_p50=g.allocations.median()))
write('scaling_summary.csv',summary)
print('PASS result aggregation; actual modules and scopes retained',flush=True)
