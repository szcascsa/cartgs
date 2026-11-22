"""最终Student联合/准备延迟、独立关闭归因及质量汇总，仅认可无干扰任务。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PRIMARY = [f'local{x:g}' for x in [.1,.2,.3,.4,.5]]


def main():
    r = HERE / 'reports'
    state = json.loads((r / 'RELEASE_JOINT_20261115_MERGED.json').read_text())
    assert state['state'] == 'complete', state.get('error', state['state'])
    parts, prep, prep_events, memories = {}, [], [], []
    for name in state['accepted']:
        if name.startswith('release_joint_g') or name.startswith('release_joint_quality_g'):
            for kind in ['samples', 'events', 'quality', 'checks']:
                path = r / name / (kind + '.csv')
                if path.exists() and path.stat().st_size > 1:
                    frame = pd.read_csv(path)
                    frame['job'] = name
                    parts.setdefault(kind, []).append(frame)
        elif name.startswith('release_prepare_g'):
            frame = pd.read_csv(r / name / 'samples.csv')
            frame['job'] = name
            prep.append(frame)
            event = pd.read_csv(r / name / 'events.csv')
            event['job'] = name
            prep_events.append(event)
        elif name.startswith('release_memory_g'):
            frame = pd.read_csv(r / name / 'memory.csv')
            frame['job'] = name
            memories.append(frame)
    frames = {key:pd.concat(value, ignore_index=True) for key,value in parts.items()}
    for kind, frame in frames.items():
        frame.to_csv(r / ('RELEASE_JOINT_' + kind.upper() + '.csv.gz'), index=False, compression='gzip')
    keys = ['scene','condition','round','variant']
    stats = frames['samples'].groupby(keys).agg(p50_ms=('ms','median'),p95_ms=('ms',lambda x:x.quantile(.95)),
        K=('K','median'),N=('N','first'),samples=('ms','size')).reset_index()
    for label, variant in [('A','A_fp32_F2'),('C','C_native_F2'),('I','I_previous'),('new_prepare','new_prepare_182')]:
        ref = stats[stats.variant == variant][keys[:-1]+['p50_ms']].rename(columns={'p50_ms':label+'_p50_ms'})
        stats = stats.merge(ref,on=keys[:-1])
        stats[label+'_speedup'] = stats[label+'_p50_ms'] / stats.p50_ms
    stats.to_csv(r/'RELEASE_JOINT_SCENE_RATIO.csv',index=False)
    frames['events'].groupby(keys).agg(event_p50_ms=('cuda_ms','median'),
        event_p95_ms=('cuda_ms',lambda x:x.quantile(.95))).reset_index().to_csv(r/'RELEASE_JOINT_CUDA_EVENT.csv',index=False)
    stats[(stats.A_speedup<1)|(stats.I_speedup<1)].to_csv(r/'RELEASE_JOINT_ALL_SLOW_CASES.csv',index=False)
    stats[stats.condition.eq('full')].to_csv(r/'RELEASE_JOINT_FULL.csv',index=False)
    q = frames['quality'].groupby(['scene','condition','variant']).agg(psnr_loss=('psnr_loss','mean'),
        worst_frame_loss=('psnr_loss','max'),psnr=('psnr','mean'),baseline_psnr=('baseline_psnr','mean'),
        ssim=('ssim','mean'),baseline_ssim=('baseline_ssim','mean'),lpips=('lpips','mean'),
        baseline_lpips=('baseline_lpips','mean'),id_overlap=('id_overlap','mean'),worst_id_overlap=('id_overlap','min')).reset_index()
    q.to_csv(r/'RELEASE_JOINT_QUALITY.csv',index=False)
    q[q.psnr_loss>.5].to_csv(r/'RELEASE_JOINT_QUALITY_FAILURES.csv',index=False)
    rows=[]
    for name, frame in stats[stats.condition.isin(PRIMARY)].groupby('variant'):
        values=q[q.variant==name]
        rows.append(dict(variant=name,p50_us=frame.p50_ms.mean()*1000,p95_us=frame.p95_ms.mean()*1000,
            A_speedup=np.exp(np.log(frame.A_speedup).mean()),C_speedup=np.exp(np.log(frame.C_speedup).mean()),
            I_speedup=np.exp(np.log(frame.I_speedup).mean()),prepare_speedup=np.exp(np.log(frame.new_prepare_speedup).mean()),
            mean_psnr_loss=values.psnr_loss.mean(),worst_condition=values.psnr_loss.max(),worst_frame=values.worst_frame_loss.max(),
            id_overlap=values.id_overlap.mean(),worst_id_overlap=values.worst_id_overlap.min(),
            quality_pass=bool(values.psnr_loss.mean()<=.2 and values.psnr_loss.max()<=.5)))
    pd.DataFrame(rows).to_csv(r/'RELEASE_JOINT_OVERVIEW.csv',index=False)
    per=frames['samples'].groupby(['scene','round','condition','frame','motion','variant']).agg(K=('K','first'),T=('ms','median')).reset_index()
    full=per[per.condition.eq('full')][['scene','round','frame','variant','K','T']].rename(columns={'K':'K_ref','T':'T_ref'})
    per=per.merge(full,on=['scene','round','frame','variant'])
    per['normalized_K'],per['normalized_T']=per.K/per.K_ref,per['T']/per.T_ref
    per.to_csv(r/'RELEASE_JOINT_PER_FRAME.csv.gz',index=False,compression='gzip')
    per[(per.normalized_K<1)&(per.normalized_T>1)].to_csv(r/'RELEASE_JOINT_LOW_RATIO_SLOWER_THAN_FULL.csv',index=False)
    assert frames['checks'].same_ids.all() and frames['checks'].rgb_bitwise.all() and frames['checks'].radii_bitwise.all()
    prepare=pd.concat(prep,ignore_index=True)
    prepare.to_csv(r/'RELEASE_PREPARE_SAMPLES.csv.gz',index=False,compression='gzip')
    prepare.groupby(keys).agg(p50_ms=('ms','median'),p95_ms=('ms',lambda x:x.quantile(.95)),K=('K','median'),N=('N','first')).reset_index().to_csv(r/'RELEASE_PREPARE_SCENE_RATIO.csv',index=False)
    pd.concat(prep_events,ignore_index=True).to_csv(r/'RELEASE_PREPARE_EVENTS.csv.gz',index=False,compression='gzip')
    if memories:pd.concat(memories,ignore_index=True).to_csv(r/'RELEASE_SINGLE_MEMORY.csv',index=False)
    (r/'RELEASE_JOINT_SUMMARY.json').write_text(json.dumps(dict(passed=True,scenes=stats.scene.nunique(),
        rounds=stats['round'].nunique(),conditions=stats.condition.nunique(),samples=len(frames['samples']),
        quality_rows=len(frames['quality']),checks=len(frames['checks']),prepare_samples=len(prepare),
        all_component_ablation_bitwise=True,accepted=state['accepted'],rejected=state['rejected']),indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':main()
