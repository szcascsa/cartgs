"""完整账本与逐视图配对；不根据nonBank新增结构。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;TRAIN=[.01,.05,.1,.15]


def main():
    ledger=[];frames=[]
    for path in sorted((ROOT/'results').glob('*/*/*/*/complete.json')):
        v=json.loads(path.read_text());cfg=v['config'];tier,scene,name,seed=path.parts[-5:-1];directory=path.parent
        ledger.append(dict(experiment='/'.join([tier,scene,name,seed]),tier=tier,scene=scene,name=name,seed=int(seed),
            steps=v['steps'],source_steps=v.get('prior_steps',4000 if cfg.get('initial_checkpoint') else 0),
            training=v.get('training','photo only'),parameters=v['parameters'],trainable_parameters=v['trainable_parameters'],pure_shared=v['pure_shared'],
            validation_psnr=v['validation_psnr'],validation_train=v['validation_train'],validation_unseen=v['validation_unseen'],
            seconds=v['seconds'],peak_memory_bytes=v.get('peak_memory_bytes'),code_version=json.dumps(v['code_sha256'],sort_keys=True),
            code_sha_scope=v.get('code_sha_scope','earlyrun file SHA at completion; exact loaded snapshot unavailable'),
            config=json.dumps(cfg,sort_keys=True),checkpoint=v['checkpoint'],status='FORMAL_EVALUATED' if (directory/'formal/test_per_view.csv').exists() else 'TRAINED_BANK_ONLY'))
        for split in ['bank_train','validation','test']:
            p=directory/'formal'/(split+'_per_view.csv')
            if p.exists():f=pd.read_csv(p);f['directory']=str(directory);frames.append(f)
    for p in ROOT.glob('queue_*.json'):
        queue=json.loads(p.read_text())
        if not isinstance(queue,dict):continue
        keys={x['experiment'] for x in ledger}
        for job in queue.get('jobs',[]):
            key='/'.join([queue['tier'],queue.get('scene','fr1_xyz'),job['config']['id'],str(job['seed'])])
            if key not in keys:
                directory=ROOT/'results'/queue['tier']/queue.get('scene','fr1_xyz')/job['config']['id']/str(job['seed'])
                status=('INVALID_STOPPED_DIAGNOSTIC' if directory.exists() else 'NOT_STARTED_SUPERSEDED') if queue['tier']=='teacher_probe' else 'NOT_COMPLETED'
                ledger.append(dict(experiment=key,config=json.dumps(job['config']),steps=job['steps'],status=status))
    table=pd.DataFrame(ledger);table.to_csv(ROOT/'NEURAL_V1_EXPERIMENTS.csv',index=False,encoding='utf-8-sig')
    if not frames:return
    q=pd.concat(frames,ignore_index=True);q.to_csv(ROOT/'quality_per_view.csv',index=False)
    by=q.groupby(['tier','scene','config','seed','policy','split','ratio']).agg(
        psnr=('psnr','mean'),ssim=('ssim','mean'),lpips=('lpips','mean'),photo=('photo','mean'),selected=('selected','first'),
        visible=('visible','mean'),effective_proxy=('effective_proxy','mean')).reset_index()
    by.to_csv(ROOT/'quality_by_ratio.csv',index=False)
    macros=[]
    for key,g in by.groupby(['tier','scene','config','seed','policy','split']):
        for label,ratios in [('train',TRAIN),('unseen',[.025,.075,.125]),('seven',sorted(TRAIN+[.025,.075,.125])),('1pct',[.01]),('5pct',[.05])]:
            p=g[g.ratio.isin(ratios)];row=dict(zip(['tier','scene','config','seed','policy','split'],key));row['group']=label
            row.update({m:float(p[m].mean()) for m in ['psnr','ssim','lpips','photo']});macros.append(row)
    macro=pd.DataFrame(macros);macro.to_csv(ROOT/'quality_macro.csv',index=False)
    for index,row in table.iterrows():
        part=macro[(macro.tier==row.get('tier'))&(macro.scene==row.get('scene'))&(macro.config==row.get('name'))&
            (macro.seed==row.get('seed'))&(macro.policy=='same_k')&(macro.split=='test')]
        if part.empty:continue
        for group,label in [('train','test_train'),('unseen','test_unseen'),('seven','test_seven'),('1pct','test_1pct'),('5pct','test_5pct')]:
            g=part[part.group==group]
            if g.empty:continue
            for m in ['psnr','ssim','lpips','photo']:table.loc[index,label+'_'+m]=float(g[m].iloc[0])
    table.to_csv(ROOT/'NEURAL_V1_EXPERIMENTS.csv',index=False,encoding='utf-8-sig')
    pairs=[]
    for scene in q.scene.unique():
        pool=q[(q.scene==scene)&(q.policy=='same_k')]
        # 更强基线优先用同预算TrainEncoder；保留被冻结原Encoder控制另行比较。
        reference=pool[pool.config=='Reference59TrainEncoder']
        if reference.empty:reference=pool[pool.config=='Reference59']
        for (tier,name,seed,split),g in pool.groupby(['tier','config','seed','split']):
            if name.startswith('Reference59'):continue
            if tier=='long36k':b=pool[(pool.tier=='long36k')&(pool.config=='Reference59TrainEncoderLong')&(pool.seed==seed)&(pool.split==split)]
            elif tier=='baseline_sets':b=pool[(pool.tier==tier)&(pool.config=='Reference59TrainEncoder')&(pool.seed==seed)&(pool.split==split)]
            else:b=reference[(reference.seed==seed)&(reference.split==split)&(reference.tier!='baseline_sets')]
            if b.empty:continue
            m=g.merge(b,on=['ratio','path'],suffixes=('_new','_reference'));assert (m.selected_new==m.selected_reference).all()
            for label,ratios in [('train',TRAIN),('unseen',[.025,.075,.125]),('seven',sorted(TRAIN+[.025,.075,.125])),('1pct',[.01]),('5pct',[.05])]:
                part=m[m.ratio.isin(ratios)];delta=(part.psnr_new-part.psnr_reference).to_numpy()
                pairs.append(dict(tier=tier,scene=scene,config=name,seed=seed,split=split,group=label,
                    delta_psnr=float(delta.mean()),median_delta=float(np.median(delta)),p10_delta=float(np.quantile(delta,.1)),
                    p90_delta=float(np.quantile(delta,.9)),improved_fraction=float((delta>0).mean()),
                    delta_ssim=float((part.ssim_new-part.ssim_reference).mean()),delta_lpips=float((part.lpips_new-part.lpips_reference).mean()),
                    comparison='fixedMaster,fixedR3selectedidentities,matchedK; no population confidence claim'))
    pd.DataFrame(pairs).to_csv(ROOT/'paired_results.csv',index=False)
    print('SUMMARY',len(table),'jobs',len(q),'per-view metrics',flush=True)


if __name__=='__main__':main()
