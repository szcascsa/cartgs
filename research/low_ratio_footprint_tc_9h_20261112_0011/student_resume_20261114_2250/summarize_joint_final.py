"""Student组合、渲染和调度独立归因；不将历史收益计入新增收益。"""
from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np

HERE=Path(__file__).resolve().parent
PRIMARY=[f'local{x:g}' for x in [.1,.2,.3,.4,.5]]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--controller',default='JOINT_FINAL02');parser.add_argument('--supplement');args=parser.parse_args()
    state=json.loads((HERE/'reports'/(args.controller+'_CONTROLLER.json')).read_text())
    if args.supplement:
        assert state['state']=='complete' or (state['state']=='failed' and '持续资源干扰' in state.get('error',''))
        supplement=json.loads((HERE/'reports'/(args.supplement+'_CONTROLLER.json')).read_text())
        state['accepted']+=supplement['accepted'];state['rejected']+=supplement['rejected']
    else:assert state['state']=='complete'
    paths=[HERE/'reports'/name for name in state['accepted'] if name.startswith('student_joint_final_')]
    assert len(paths)==27,len(paths)
    for path in paths:
        job=json.loads((HERE/'reports/jobs'/(path.name+'.json')).read_text())
        assert job['state']=='complete' and job['measurement_valid'] and job['returncode']==0,path
    frames={}
    for name in ['samples','events','checks','quality','memory']:
        parts=[]
        for p in paths:
            f=p/(name+'.csv')
            if f.exists() and f.stat().st_size>1:
                part=pd.read_csv(f);part['job']=p.name;parts.append(part)
        frames[name]=pd.concat(parts,ignore_index=True)
        frames[name].to_csv(HERE/'reports'/('JOINT_FINAL_'+name.upper()+'.csv.gz'),index=False,compression='gzip')
    keys=['scene','condition','round','variant'];samples=frames['samples']
    stats=samples.groupby(keys).agg(p50_ms=('ms','median'),p95_ms=('ms',lambda x:x.quantile(.95)),mean_ms=('ms','mean'),
        K=('K','median'),N=('N','first'),samples=('ms','size')).reset_index()
    for variant,label in [('A_fp32_F2','A'),('C_tensor_best_F2','C'),('E_tensor_best_old38','E'),('G_cpp_pose','G'),('H_old38','H'),('D_tensor_best_new','D')]:
        ref=stats[stats.variant==variant][keys[:-1]+['p50_ms']].rename(columns={'p50_ms':label+'_p50_ms'})
        stats=stats.merge(ref,on=keys[:-1]);stats[label+'_speedup']=stats[label+'_p50_ms']/stats.p50_ms
    stats.to_csv(HERE/'reports/JOINT_SCENE_RATIO.csv',index=False)
    stats[(stats.A_speedup<1)|(stats.E_speedup<1)].to_csv(HERE/'reports/JOINT_ALL_SLOW_CASES.csv',index=False)
    q=frames['quality'];quality=q.groupby(['scene','condition','variant']).agg(psnr_loss=('psnr_loss','mean'),
        worst_frame_loss=('psnr_loss','max'),psnr=('psnr','mean'),baseline_psnr=('baseline_psnr','mean'),ssim=('ssim','mean'),
        baseline_ssim=('baseline_ssim','mean'),lpips=('lpips','mean'),baseline_lpips=('baseline_lpips','mean'),id_overlap=('id_overlap','mean'),
        worst_id_overlap=('id_overlap','min'),renderer_psnr_loss=('renderer_psnr_loss','mean'),rgb_max_error=('rgb_max_error','max')).reset_index()
    quality.to_csv(HERE/'reports/JOINT_QUALITY_CONDITIONS.csv',index=False)
    checks=frames['checks'];assert bool(checks.same_ids.all()) and bool(checks.rgb_bitwise.all()) and bool(checks.radii_bitwise.all())
    overview=[]
    for name,part in stats[stats.condition.isin(PRIMARY)].groupby('variant'):
        values=quality[quality.variant==name]
        overview.append(dict(variant=name,p50_us=part.p50_ms.mean()*1000,p95_us=part.p95_ms.mean()*1000,
            A_speedup=np.exp(np.log(part.A_speedup).mean()),E_speedup=np.exp(np.log(part.E_speedup).mean()),
            G_speedup=np.exp(np.log(part.G_speedup).mean()),H_speedup=np.exp(np.log(part.H_speedup).mean()),
            D_speedup=np.exp(np.log(part.D_speedup).mean()),worst_A_speedup=part.A_speedup.min(),worst_E_speedup=part.E_speedup.min(),
            mean_psnr_loss=values.psnr_loss.mean(),worst_condition=values.psnr_loss.max(),worst_frame=values.worst_frame_loss.max(),
            mean_id_overlap=values.id_overlap.mean(),worst_id_overlap=values.worst_id_overlap.min(),
            quality_pass=bool(values.psnr_loss.mean()<=.2 and values.psnr_loss.max()<=.5)))
    overview=pd.DataFrame(overview);overview.to_csv(HERE/'reports/JOINT_FINAL_OVERVIEW.csv',index=False)
    per=samples.groupby(['scene','condition','round','variant','frame']).agg(K=('K','first'),T=('ms','median')).reset_index()
    full=per[per.condition=='full'][['scene','round','variant','frame','K','T']].rename(columns={'K':'K_Full','T':'T_Full'})
    per=per.merge(full,on=['scene','round','variant','frame']);per['normalized_K']=per.K/per.K_Full;per['normalized_T']=per['T']/per.T_Full
    per.to_csv(HERE/'reports/JOINT_PER_FRAME.csv.gz',index=False,compression='gzip')
    per[(per.normalized_K<1)&(per.normalized_T>1)].to_csv(HERE/'reports/JOINT_LOW_RATIO_SLOWER_THAN_FULL.csv',index=False)
    summary=dict(passed=True,scenes=stats.scene.nunique(),conditions=stats.condition.nunique(),rounds=stats['round'].nunique(),
        samples=len(samples),events=len(frames['events']),quality=len(q),bitwise_checks=len(checks),all_checks_passed=True,
        accepted_jobs=state['accepted'],rejected_jobs=state['rejected'])
    (HERE/'reports/JOINT_FINAL_SUMMARY.json').write_text(json.dumps(summary,indent=2));print(overview.to_string(index=False));print(summary)


if __name__=='__main__':main()
