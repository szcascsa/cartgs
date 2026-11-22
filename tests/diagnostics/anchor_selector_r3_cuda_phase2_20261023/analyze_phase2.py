"""读取真实原始数据。事件时间、kernel时间、同步和固定序列质量分别报告。"""
import os
from pathlib import Path
import json,re,sqlite3
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(2)
R=Path(os.environ.get('CARTGS_PHASE2_RESULTS','/home/prmi/zwj/anchor-selector-r3-cuda-phase2-20261023'))
def stats(x):
    x=np.asarray(x,dtype=float)
    return dict(samples=len(x),mean=float(x.mean()),p50=float(np.quantile(x,.5)),p95=float(np.quantile(x,.95)),p99=float(np.quantile(x,.99)))
def write(name,rows):pd.DataFrame(rows).to_csv(R/name,index=False)
performance=[];cycles=[];memory=[];topology=[]
for path in sorted(R.glob('bench_*')):
    match=re.fullmatch(r'bench_(1\.0|0\.5)_(\d+)_(.+)',path.name)
    if not path.is_dir() or not match:continue
    resolution,round_,variant=match.groups();a=pd.read_csv(path/'native_stages.csv')
    for (case,stage),g in a.groupby(['case','stage']):
        for metric in ['host_submit_ms','cuda_elapsed_ms']:
            performance.append(dict(scope='fixed_checkpoint',variant=variant,round=int(round_),resolution=float(resolution),case=case,stage=stage,N=int(g.N.iloc[0]),K=int(g.K.iloc[0]),metric=metric,**stats(g[metric])))
    for case,g in a[a.stage=='adam'].groupby('case'):
        if case=='Full' and float(resolution)!=1.:continue
        row=dict(variant=variant,round=int(round_),resolution=float(resolution),case=case,scope='fixed_map_online_half' if float(resolution)==.5 else 'fixed_map_tail_native',N=int(g.N.iloc[0]),K=int(g.K.iloc[0]),**stats(g.cycle_wall_ms));cycles.append(row)
    m=pd.read_csv(path/'memory.csv');m['variant']=variant;m['round']=int(round_);m['resolution']=float(resolution);memory.extend(m.to_dict('records'))
    t=pd.read_csv(path/'topology_stages.csv')
    for operation,g in t.groupby('operation'):
        topology.append(dict(variant=variant,round=int(round_),operation=operation,scope='real_anchor_primitive_not_complete_event',**stats(g.completed_ms)))
write('phase2_performance_summary.csv',performance);write('phase2_online_tail_performance.csv',cycles);write('memory_measurements.csv',memory);write('phase2_topology_event_benchmark.csv',topology)
events=R/'topology_checkpoint/topology_events.csv'
if events.exists():
    a=pd.read_csv(events)
    for (scenario,fused),g in a.groupby(['scenario','fused']):
        topology.append(dict(variant='fused' if fused else 'legacy',round=-1,operation=scenario,scope='real_checkpoint_synthetic_event_gradient',N_before=int(g.N_before.iloc[0]),A_before=int(g.A_before.iloc[0]),N_after=int(g.N_after.iloc[0]),A_after=int(g.A_after.iloc[0]),**stats(g.cuda_ms)))
    write('phase2_topology_event_benchmark.csv',topology)

launch=[];sync=[];kernel_detail=[]
for path in sorted(R.glob('profile_*.sqlite')):
    if path.stem=='profile_topology':continue
    variant=path.stem.removeprefix('profile_');db=sqlite3.connect(path);db.row_factory=sqlite3.Row
    strings={x['id']:x['value'] for x in db.execute('select * from StringIds')}
    ranges=[dict(x) for x in db.execute('select * from NVTX_EVENTS where end is not null')]
    for x in ranges:x['label']=x['text'] or strings.get(x['textId'],'')
    parents=[]
    for case in ['Full','lod_0.010000','lod_0.050000','lod_0.100000','lod_0.150000']:
        options=sorted([x for x in ranges if x['label']==case],key=lambda x:x['start']);parents.extend([(case,i+1,p) for i,p in enumerate(options[-2:])])
    def location(t):
        for case,i,p in parents:
            if p['start']<=t<=p['end']:
                children=[r for r in ranges if r['start']>=p['start'] and r['end']<=p['end'] and r['start']<=t<=r['end'] and r['label']!=case]
                return case,i,min(children,key=lambda r:r['end']-r['start'])['label'] if children else 'control'
    runtime={x['correlationId']:dict(x) for x in db.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')};items=[];apis=[]
    for x in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'):
        api=runtime.get(x['correlationId']);loc=location(api['start']) if api else None
        if loc:items.append(dict(variant=variant,case=loc[0],repeat=loc[1],stage=loc[2],kernel=strings[x['demangledName']],kernel_ms=(x['end']-x['start'])/1e6,queue_us=max(0,x['start']-api['end'])/1000))
    kernel_detail.extend(items);f=pd.DataFrame(items)
    for (case,stage),g in f.groupby(['case','stage']):launch.append(dict(variant=variant,case=case,stage=stage,launches=len(g)/2,kernel_ms=g.kernel_ms.sum()/2,queue_us_mean=g.queue_us.mean()))
    for case,g in f.groupby('case'):launch.append(dict(variant=variant,case=case,stage='all',launches=len(g)/2,kernel_ms=g.kernel_ms.sum()/2,queue_us_mean=g.queue_us.mean()))
    for api in runtime.values():
        loc=location(api['start'])
        if loc:apis.append(dict(variant=variant,case=loc[0],repeat=loc[1],stage=loc[2],api=strings[api['nameId']],host_ms=(api['end']-api['start'])/1e6))
    af=pd.DataFrame(apis)
    for (case,stage,api),g in af.groupby(['case','stage','api']):sync.append(dict(variant=variant,case=case,stage=stage,api=api,calls_per_step=len(g)/2,host_ms_per_step=g.host_ms.sum()/2))
    db.close()
write('phase2_kernel_launch_comparison.csv',launch);write('phase2_cpu_gpu_sync.csv',sync);write('kernel_execution_detail.csv',kernel_detail)

def load(path):return {k:v.detach().cpu() for k,v in torch.jit.load(str(path),map_location='cpu').state_dict().items()}
parity=[];quality=[];worst=[]
def compare(name,a,b,envelope=None):
    for key in sorted(set(a)|set(b)):
        exact=any(s in key for s in ['selected_ids','owner','gaussian_uid','anchor_uid','member_counts','anchor_levels','uniform','_step']) or (key.startswith('anchor_stats_') and key not in ['anchor_stats_0','anchor_stats_2']) or (key in a and not a[key].is_floating_point())
        if key not in a or key not in b:
            parity.append(dict(comparison=name,tensor=key,shape_match=False,grad_none_match=False,exact=exact,bad_elements=-1));continue
        x,y=a[key],b[key]
        if x.shape!=y.shape:
            parity.append(dict(comparison=name,tensor=key,shape_match=False,grad_none_match=True,exact=exact,bad_elements=-1));continue
        d=(x.double()-y.double()).abs();r=d/y.double().abs().clamp_min(1e-300)
        tol=torch.zeros_like(d) if exact else (1e-12+2e-14*x.double().abs() if key.startswith('anchor_stats_') else 2e-5+2e-4*x.double().abs())
        bad=d>tol;outside=bad
        if envelope and key in envelope and envelope[key].shape==d.shape:outside=d>tol+envelope[key]
        idx=int(d.reshape(-1).argmax()) if d.numel() else -1
        parity.append(dict(comparison=name,tensor=key,shape_match=True,grad_none_match=True,exact=exact,max_abs=float(d.max()) if d.numel() else 0,max_rel=float(r.max()) if d.numel() else 0,rms=float(d.square().mean().sqrt()) if d.numel() else 0,bad_elements=int(bad.sum()),outside_baseline_envelope=int(outside.sum()),worst_index=idx))
        if bad.any():
            row=dict(comparison=name,tensor=key,flat_index=idx,reference=float(x.reshape(-1)[idx]),candidate=float(y.reshape(-1)[idx]))
            if y.ndim and 'gaussian_uid' in b:
                i=idx//(y.numel()//y.shape[0]);gid=None
                if y.shape[0]==b['gaussian_uid'].numel():gid=i
                elif 'selected_ids' in b and y.shape[0]==b['selected_ids'].numel():gid=int(b['selected_ids'][i])
                if gid is not None:row['gaussian_row']=gid;row['uid']=int(b['gaussian_uid'][gid]);row['owner']=int(b['owner'][gid])
            worst.append(row)
for resolution in ['1.0','0.5']:
  for case in ['Full','lod_0.010000','lod_0.050000','lod_0.100000','lod_0.150000']:
    base=load(R/f'bench_{resolution}_0_B'/f'{case}_parity.pt');repeat=load(R/f'bench_{resolution}_1_B'/f'{case}_parity.pt')
    envelope={k:(base[k].double()-repeat[k].double()).abs() for k in base.keys()&repeat.keys() if base[k].shape==repeat[k].shape}
    compare(f'{resolution}/{case}/B_repeat',base,repeat)
    for variant in ['A','C','D','E_workspace','E']:
        candidate=load(R/f'bench_{resolution}_0_{variant}'/f'{case}_parity.pt');compare(f'{resolution}/{case}/B_to_{variant}',base,candidate,envelope)
        diff=(candidate['render']-candidate['gt']).square().mean();quality.append(dict(scope='fixed_checkpoint',resolution=resolution,case=case,variant=variant,PSNR=float(-10*diff.log10()),SSIM=float(candidate['ssim']),photo=float(candidate['photo']),l1=float(candidate['l1']),budget=float(candidate.get('budget',torch.tensor(0.)))))
    quality.append(dict(scope='fixed_checkpoint',resolution=resolution,case=case,variant='B',PSNR=float(-10*(base['render']-base['gt']).square().mean().log10()),SSIM=float(base['ssim']),photo=float(base['photo']),l1=float(base['l1']),budget=float(base.get('budget',torch.tensor(0.)))))
write('phase2_correctness_parity.csv',parity);write('worst_error_locations.csv',worst);write('quality_metrics.csv',quality)
for scenario in ['noop_gaussian','gaussian_density','anchor_growth','anchor_prune']:
    a=load(R/'topology_checkpoint'/f'{scenario}_legacy.pt');b=load(R/'topology_checkpoint'/f'{scenario}_fused.pt');compare('topology/'+scenario,a,b)
curves=[]
for folder in sorted(R.glob('continuous_*')):
    if not folder.is_dir():continue
    a=pd.read_csv(folder/'training_curve.csv');a['run']=folder.name;curves.extend(a.to_dict('records'))
    prefix='continuous_online_' if folder.name.startswith('continuous_online_') else 'continuous_'
    if folder.name.endswith('B_a'):continue
    reference=R/(prefix+'B_a')
    for fixture in folder.glob('*.pt'):
        other=reference/fixture.name
        if other.exists():compare(folder.name+'/'+fixture.name,load(other),load(fixture))
write('continuous_training_curves.csv',curves);write('phase2_correctness_parity.csv',parity);write('worst_error_locations.csv',worst)
module_summary=[]
for folder,file,group in [('production_phase4','anchor_timing.csv',['implementation','stage']),('production_phase5','sampling_timing.csv',['ratio','implementation','stage']),('production_phase6','workspace_timing.csv',['case','reuse']),('production_ssim','ssim_timing.csv',['H','W','optimized'])]:
    a=pd.read_csv(R/folder/file)
    for keys,g in a.groupby(group):
        row=dict(zip(group,keys if isinstance(keys,tuple) else (keys,)));module_summary.append(dict(module=folder,**row,**stats(g.cuda_ms)))
write('module_paired_summary.csv',module_summary)
online=[];states=[]
for name in ['B','E']:
    folder=R/('online_'+name);path=folder/'mapping_trace.csv'
    if not path.exists():continue
    a=pd.read_csv(path)
    for phase,g in a.groupby('phase'):
        for stage in list(a)[2:]:online.append(dict(variant=name,phase='online' if phase==0 else 'tail',stage=stage,scope='actual_asynchronous_slam_unmatched_topology',**stats(g[stage])))
    for modulo,label in [(0,'Full'),(1,'1%'),(2,'5%'),(3,'10%'),(4,'15%')]:
        g=a[(a.phase==0)&(a.iteration>=3000)&(a.iteration%5==modulo)]
        if len(g):online.append(dict(variant=name,phase='online_post3000_'+label,stage='total',scope='source_schedule_inferred',**stats(g.total)))
    checkpoints=list(folder.glob('*_shutdown/ply'))
    if checkpoints:
        cp=checkpoints[0];states.append(dict(variant=name,mapping=json.loads((cp/'mapping_stage_state.json').read_text()),anchor=json.loads(next((cp/'point_cloud').glob('iteration_*/anchor_selector_state.json')).read_text())))
write('actual_online_mapping.csv',online);write('phase2_online_tail_performance.csv',cycles+online);(R/'online_final_states.json').write_text(json.dumps(states,indent=2))
for folder,file in [('production_fixed_adam','adam_parity.csv'),('production_phase4','anchor_parity.csv'),('production_phase5','sampling_parity.csv'),('production_phase6','workspace_parity.csv'),('production_ssim','ssim_parity.csv')]:
    a=pd.read_csv(R/folder/file)
    for row in a.to_dict('records'):
        parity.append(dict(row,comparison='isolated/'+folder,tensor=row.get('name',row.get('tensor',''))))
write('phase2_correctness_parity.csv',parity)
print('ANALYSIS_PASS',flush=True)
