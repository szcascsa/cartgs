"""汇总全部尝试与配对逐视图差值；不筛选最好seed或checkpoint。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
METRICS=['psnr','ssim','lpips','photo'];TRAIN=[.01,.05,.1,.15]


def summary(data,keys):
    return data.groupby(keys,dropna=False).agg(**{m:(m,'mean') for m in METRICS if m in data},K=('selected','mean'),visible=('visible','mean'),effective_proxy=('effective_proxy','mean'),views=('view','count')).reset_index()


def main():
    infos=[];frames=[];bykey={}
    for p in sorted(ROOT.glob('results/*/*/*/*/complete.json')):
        d=json.loads(p.read_text());cfg=d['config'];key=(d['tier'],d['scene'],cfg['id'],d['seed']);bykey[key]=p.parent
        infos.append(dict(tier=d['tier'],scene=d['scene'],model=cfg['id'],feature=cfg.get('feature'),fusion=cfg.get('fusion','none'),dimension=cfg.get('dim',0),seed=d['seed'],route=d['route'],steps=d['steps'],prior_steps=d['prior_steps'],parameters=d['parameters'],seconds=d['seconds'],bank_holdout_psnr=d['bank_holdout_psnr'],bank_low_psnr=d['bank_low'],bank_unseen_psnr=d['bank_unseen'],status='TESTED' if (p.parent/'formal/test_per_view.csv').exists() else 'BANK_ONLY',checkpoint=d['checkpoint'],frame_order_sha256=d['frame_order_sha256'],source_checkpoint_sha256=d['source_checkpoint_sha256'],code_sha256=json.dumps(d['code_sha256'],sort_keys=True)))
        record=infos[-1];record.update(configuration=json.dumps(cfg,sort_keys=True),freeze_base=cfg.get('freeze_base',False),feature_lr=cfg.get('feature_lr',.001),fresh_base_adam=cfg.get('fresh_base_adam',False),
            parameters_total=sum(v['numel'] for v in d['parameter_inventory']),peak_memory_bytes=d['peak_memory_bytes'],optimizer_end_clocks=json.dumps(d['optimizer_steps'],sort_keys=True),formal_directory=str(p.parent/'formal'))
        for f in (p.parent/'formal').glob('*_per_view.csv'):
            df=pd.read_csv(f);prefix=f.name.replace('_per_view.csv','')
            for metric in METRICS:record[prefix+'_mean_'+metric]=df[metric].mean()
            if prefix=='test':
                for group,mask in [('low',df.ratio.isin([.01,.05])),('unseen',~df.ratio.isin(TRAIN))]:
                    for metric in METRICS:record['test_'+group+'_'+metric]=df.loc[mask,metric].mean()
                for r,part in df.groupby('ratio'):
                    record['test_r'+str(r)+'_K']=int(part.selected.iloc[0])
                    for metric in METRICS:record['test_r'+str(r)+'_'+metric]=part[metric].mean()
            df['directory']=str(p.parent);frames.append(df)
    for q in ROOT.glob('queue_*.json'):
        if q.name.endswith(('_results.json','_complete.json')):continue
        queue=json.loads(q.read_text(encoding='utf-8-sig'))
        if 'jobs' not in queue:continue
        for job in queue['jobs']:
            key=(job.get('tier',queue['tier']),queue['scene'],job['config']['id'],job['seed'])
            if key not in bykey:infos.append(dict(tier=key[0],scene=key[1],model=key[2],seed=key[3],route=job.get('route','continue'),steps=job['steps'],status='NOT COMPLETED',configuration=json.dumps(job['config'],sort_keys=True)))
    pd.DataFrame(infos).to_csv(ROOT/'ANCHOR_FEATURE_EXPERIMENTS.csv',index=False)
    if not frames:return
    all_data=pd.concat(frames,ignore_index=True);all_data.to_csv(ROOT/'all_formal_per_view.csv',index=False)
    keys=['tier','scene','config','seed','route','split','ratio']
    summary(all_data,keys).to_csv(ROOT/'quality_per_ratio.csv',index=False)
    macros=[]
    for label,mask in [('seven',np.ones(len(all_data),dtype=bool)),('train',all_data.ratio.isin(TRAIN)),('unseen',~all_data.ratio.isin(TRAIN)),('low',all_data.ratio.isin([.01,.05]))]:
        part=summary(all_data.loc[mask],keys[:-1]);part['ratios']=label;macros.append(part)
    pd.concat(macros,ignore_index=True).to_csv(ROOT/'quality_macro.csv',index=False)
    pairs=[];per_view=[];rng=np.random.default_rng(27812)
    for key,directory in bykey.items():
        tier,scene,name,seed=key
        if name=='Baseline':continue
        target=pd.read_csv(directory/'formal/test_per_view.csv') if (directory/'formal/test_per_view.csv').exists() else None
        if target is None:continue
        basekey=('scratch_confirm' if tier=='scratch_confirm' else 'same_seed_repeat' if tier in ['same_seed_repeat','frozen_same_seed_repeat'] else 'optimizer_control' if name=='FreshAdamHead8' else 'confirm_continue',scene,'FreshAdamBaseline' if name=='FreshAdamHead8' else 'Baseline',seed)
        base=bykey.get(basekey)
        if base is None or not (base/'formal/test_per_view.csv').exists():continue
        reference=pd.read_csv(base/'formal/test_per_view.csv')
        both=target.merge(reference,on=['ratio','path','split'],suffixes=('','_base'),validate='one_to_one')
        assert np.array_equal(both.selected,both.selected_base)
        for metric in METRICS:both[metric+'_delta']=both[metric]-both[metric+'_base']
        both['model']=name;both['tier']=tier;both['seed']=seed;both['scene']=scene;both['baseline']=basekey[2];per_view.append(both)
        selections=[(str(r),both.ratio==r) for r in sorted(both.ratio.unique())]+[('seven',np.ones(len(both),bool)),('low',both.ratio.isin([.01,.05])),('unseen',~both.ratio.isin(TRAIN))]
        for label,mask in selections:
            b=both.loc[mask];entry=dict(tier=tier,scene=scene,model=name,seed=seed,baseline=basekey[2],ratio=label,rows=len(b))
            # View-cluster bootstrap: all ratio rows of one camera travel together.
            view=b.groupby('path')[['psnr_delta','ssim_delta','lpips_delta','photo_delta']].mean();v=view.psnr_delta.to_numpy()
            boot=v[rng.integers(len(v),size=(2000,len(v)))].mean(1);entry.update(psnr_view_bootstrap_low=np.quantile(boot,.025),psnr_view_bootstrap_high=np.quantile(boot,.975))
            winners=[]
            for metric in METRICS:
                values=b[metric+'_delta'];direction=1 if metric in ['psnr','ssim'] else -1
                entry.update({metric+'_delta_mean':values.mean(),metric+'_delta_median':values.median(),metric+'_delta_p10':values.quantile(.1),metric+'_delta_p90':values.quantile(.9),metric+'_improved_fraction':(direction*values>0).mean()})
                if metric!='photo':winners.append('candidate' if direction*values.mean()>0 else 'baseline' if direction*values.mean()<0 else 'tie')
            entry['all_metrics_winner']=winners[0] if len(set(winners))==1 else 'mixed';pairs.append(entry)
    if pairs:pd.DataFrame(pairs).to_csv(ROOT/'paired_deltas.csv',index=False)
    if per_view:pd.concat(per_view,ignore_index=True).to_csv(ROOT/'paired_per_view.csv',index=False)
    refs=[]
    for f in ROOT.glob('references/*/*_per_view.csv'):
        df=pd.read_csv(f);df['reference']='Spatial_initial12k' if f.name.startswith('spatial_initial_') else 'R3_snapshot';refs.append(df)
    if refs:summary(pd.concat(refs),['scene','reference','split','ratio']).to_csv(ROOT/'references_per_ratio.csv',index=False)
    # Ablations quantify the dependency of final trained functions, not additive causal components.
    ablations=[]
    for f in ROOT.glob('results/*/*/*/*/mechanisms/counterfactual_per_view.csv'):
        d=pd.read_csv(f);baseline=d[d.ablation=='normal'];other=d.merge(baseline,on=['scene','split','ratio','path'],suffixes=('','_normal'),validate='many_to_one');other['tier']=f.parents[4].name
        other['psnr_delta']=other.psnr-other.psnr_normal;other['photo_delta']=other.photo-other.photo_normal;ablations.append(other)
    if ablations:
        data=pd.concat(ablations);data.groupby(['tier','model','scene','seed','split','ratio','ablation']).agg(psnr=('psnr','mean'),psnr_delta=('psnr_delta','mean'),photo_delta=('photo_delta','mean')).reset_index().to_csv(ROOT/'mechanism_ablation_summary.csv',index=False)
    print('SUMMARIZED',len(infos),'runs',len(all_data),'formal rows',flush=True)
if __name__=='__main__':main()
