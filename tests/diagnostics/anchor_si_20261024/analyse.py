"""汇总最终终点，不挑 checkpoint；逐视图配对与在线训练分配分开报告。"""
from pathlib import Path
import csv,json,sys
import numpy as np,torch

root=Path('/home/prmi/zwj/anchor-si-20261024')
sys.path.insert(0,str(root/'source/python'))
from selector import load_selector
torch.set_num_threads(2)

def read(path):
    with path.open() as f:return list(csv.DictReader(f))

def write(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

pairs=[]
for group,arms in [('frozen',['gaussian','anchor']),('online',['anchor'])]:
    rows=read(root/(group+'_quality_per_view.csv'))
    controls={(r['model'],r['policy'],r['ratio'],r['split'],r['image']):r for r in rows if r['model'].startswith('off_')}
    for seed in [2026,2027]:
        for arm in arms:
            cases=sorted({(r['policy'],r['ratio'],r['split']) for r in rows if r['model']==f'{arm}_{seed}'})
            for policy,ratio,split in cases:
                current=[r for r in rows if r['model']==f'{arm}_{seed}' and (r['policy'],r['ratio'],r['split'])==(policy,ratio,split)]
                baseline=[controls[(f'off_{seed}',policy,ratio,split,r['image'])] for r in current]
                assert len(current)==len(baseline)
                output=dict(group=group,seed=seed,arm=arm,policy=policy,ratio=ratio,split=split,views=len(current),
                    baseline_N=baseline[0]['N'],candidate_N=current[0]['N'],baseline_K=baseline[0]['K'],candidate_K=current[0]['K'])
                for metric in ['psnr','ssim','lpips','photo','visible','effective_proxy']:
                    a=np.array([float(r[metric]) for r in baseline]);b=np.array([float(r[metric]) for r in current]);d=b-a
                    output['baseline_'+metric]=float(a.mean());output['candidate_'+metric]=float(b.mean());output['delta_'+metric]=float(d.mean())
                    if metric=='psnr':
                        for tag,q in [('median',50),('p10',10),('p90',90)]:output['delta_psnr_'+tag]=float(np.percentile(d,q))
                        output['psnr_improved_fraction']=float((d>1e-6).mean())
                pairs.append(output)
write(root/'paired_quality.csv',pairs)

times=[]
for label in ['frozen_'+a+'_'+str(s) for s in [2026,2027] for a in ['off','gaussian','anchor']]:
    values=np.array([float(r['cuda_ms']) for r in read(root/label/'timing.csv')])
    row=dict(case=label,stage='full_plus_lod',capture='',samples=len(values),mean_ms=float(values.mean()))
    for tag,q in [('p50',50),('p95',95),('p99',99)]:row[tag+'_ms']=float(np.percentile(values,q))
    times.append(row)
for stage,capture in sorted({(r['stage'],r['capture']) for r in read(root/'micro/timing.csv')}):
    rows=[r for r in read(root/'micro/timing.csv') if (r['stage'],r['capture'])==(stage,capture)]
    for metric in ['host_ms','cuda_ms']:
        v=np.array([float(r[metric]) for r in rows]);row=dict(case='micro_'+metric,stage=stage,capture=capture,samples=len(v),mean_ms=float(v.mean()))
        for tag,q in [('p50',50),('p95',95),('p99',99)]:row[tag+'_ms']=float(np.percentile(v,q))
        times.append(row)
write(root/'performance_summary.csv',times)

models=[];states=[];phases=[];gradient=[]
for seed in [2026,2027]:
    for arm in ['off','gaussian','anchor']:
        rows=[r for r in read(root/f'frozen_{arm}_{seed}/training.csv') if float(r['si_grad_norm'])>0]
        if rows:
            norm=np.array([float(r['si_grad_norm'])/max(float(r['photo_budget_grad_norm']),1e-30) for r in rows]);cos=np.array([float(r['gradient_cosine']) for r in rows])
            gradient.append(dict(arm=arm,seed=seed,samples=len(rows),ratio_sampling='mainly 1pct every16, final15pct',norm_ratio_median=float(np.median(norm)),cosine_median=float(np.median(cos)),negative_cosine_fraction=float((cos<0).mean())))
    for arm in ['off','anchor']:
        label=f'online_{arm}_{seed}';cp=next((root/label).glob('*_shutdown/ply'));model=load_selector(str(cp/'selector.pt'),'cpu')
        a=model.anchors;n=len(a.rows);count=torch.bincount(a.rows,minlength=len(a.features));uid=model.gaussian_compensation.gaussian_uids
        assert torch.equal(count,a.member_counts) and bool(((count>=1)&(count<=10)).all())
        assert len(torch.unique(uid))==n
        row=dict(case=label,N=n,A=len(a.features),mean_members=float(count.float().mean()),uid_unique=True,counts_exact=True)
        for level in range(3):row['L'+str(level)]=int((a.levels==level).sum())
        for k in range(1,11):row['members_'+str(k)]=int((count==k).sum())
        clocks=json.loads((cp/'mapping_stage_state.json').read_text());row['selector_step']=clocks['selector_step']
        anchor_json=json.loads(next((cp/'point_cloud').glob('iteration_*/anchor_selector_state.json')).read_text())
        (root/(label+'_anchor_metadata.json')).write_text(json.dumps(anchor_json,indent=2))
        models.append(row)
        trace=read(root/label/'mapping_trace.csv')
        for phase in [0,1]:
            name='tail' if phase else 'online';p=clocks[name];v=np.array([float(r['total']) for r in trace if int(r['phase'])==phase])
            z=dict(case=label,phase=name,mean_ms=float(v.mean()),p50_ms=float(np.percentile(v,50)),p95_ms=float(np.percentile(v,95)),p99_ms=float(np.percentile(v,99)))
            for key in ['mappings','full_renders','lod_renders','gaussian_updates','selector_commits','topology','lod_nonfinite']:z[key]=p[key]
            for k in range(4):z['ratio_'+str(k)+'_exposures']=p['ratio_counts'][k]
            phases.append(z)
        if arm=='anchor':
            state=torch.jit.load(str(cp/'anchor_si_state.pt'),map_location='cpu').state_dict()
            assert torch.equal(state['anchor_uids'],a.block_uids)
            signatures=torch.full((len(a.features),10),-1,dtype=torch.long)
            for i in range(len(a.features)):
                ids=uid[a.rows==i].sort().values;signatures[i,:len(ids)]=ids
            assert torch.equal(signatures,state['member_uids'])
            visits=state['visits'];age=int(state['full_clock'])-state['last'];valid=(visits>=4)&(age<=128)&(state['active']>0)
            states.append(dict(case=label,clock=int(state['full_clock']),identity_exact=True,canonical_member_UID_exact=True,A=len(visits),
                min4_fraction=float((visits>=4).float().mean()),valid_fraction=float(valid.float().mean()),never_observed_fraction=float((visits==0).float().mean()),
                median_visits=float(visits.float().median()),p90_visits=float(torch.quantile(visits.float(),.9)),median_age=float(age.float().median())))
write(root/'online_model_summary.csv',models);write(root/'online_phase_summary.csv',phases);write(root/'si_state_summary.csv',states);write(root/'gradient_summary.csv',gradient)

try:
    from evo.tools import file_interface
    from evo.core import sync
    groundtruth=file_interface.read_tum_trajectory_file('/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz/groundtruth.txt')
    trajectory=[]
    for row in models:
        estimate=file_interface.read_tum_trajectory_file(str(root/row['case']/'CameraTrajectory_TUM.txt'))
        ref,est=sync.associate_trajectories(groundtruth,estimate,max_diff=.02);est.align(ref,correct_scale=False)
        error=np.linalg.norm(ref.positions_xyz-est.positions_xyz,axis=1)
        trajectory.append(dict(case=row['case'],aligned_SE3=True,scale_correction=False,matches=len(error),ATE_RMSE_cm=float(np.sqrt(np.mean(error**2))*100)))
    write(root/'trajectory_summary.csv',trajectory)
except Exception as e:
    (root/'trajectory_error.txt').write_text(str(e));raise
print('FINAL_ANALYSIS_PASS')
