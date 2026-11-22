"""分组回归、配对层次bootstrap及多重比较；不参与训练或测试集调参。"""
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t

from design import LEVELS,features,identity

ROOT=Path('/home/prmi/zwj/selector-capacity-matrix-20261026')
OUT=ROOT/'analysis';OUT.mkdir(exist_ok=True)
RATIOS=[.01,.025,.05,.075,.1,.125,.15]
METRICS=['psnr','ssim','lpips','photo']
def read(path):return list(csv.DictReader(Path(path).open()))
def write(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def dump(name,value):(OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False))

def effects():
    runs=json.loads((ROOT/'screen_runs.json').read_text());design=json.loads((ROOT/'screen_design.json').read_text())
    seeds=sorted({x['seed'] for x in runs});x=np.stack([features(c) for c in design]);scale=np.sqrt(np.mean(x*x,axis=0));x/=scale
    ys=[];xx=[];groups=[]
    for j,c in enumerate(design):
        for seed in seeds:
            ys.append(next(r['validation_score'] for r in runs if r['config']['id']==c['id'] and r['seed']==seed))
            xx.append(np.r_[x[j],-1 if seed==seeds[0] else 1]);groups.append(j)
    a=np.stack(xx);y=np.array(ys);groups=np.array(groups)
    inverse=np.linalg.pinv(a.T@a);beta=np.linalg.lstsq(a,y,rcond=None)[0];res=y-a@beta
    meat=np.zeros((a.shape[1],a.shape[1]));pred=[];observed=[]
    for g in range(len(design)):
        mask=groups==g;score=a[mask].T@res[mask];meat+=np.outer(score,score)
        fit=np.linalg.lstsq(a[~mask],y[~mask],rcond=None)[0]
        pred.append(float(np.mean(a[mask]@fit)));observed.append(float(np.mean(y[mask])))
    correction=len(design)/(len(design)-1)*(len(y)-1)/(len(y)-a.shape[1])
    cov=inverse@meat@inverse*correction
    allc=[dict(zip(LEVELS,v)) for v in itertools.product(*LEVELS.values())]
    full=np.stack([np.r_[features(c)/scale,0.] for c in allc])
    critical=student_t.ppf(.975,len(design)-1);rows=[];comparisons=[]
    reference_levels=dict(shared=16,ratio=0,extra=14,selector_depth=1,compensation_depth=1)
    for factor,values in LEVELS.items():
        reference=full[[c[factor]==reference_levels[factor] for c in allc]].mean(0)
        for value in values:
            contrast=full[[c[factor]==value for c in allc]].mean(0)
            est=float(contrast@beta);se=float(np.sqrt(max(0,contrast@cov@contrast)))
            rows.append(dict(factor=factor,level=value,model_adjusted_validation_psnr=est,ci_low=est-critical*se,ci_high=est+critical*se,
                interval='configuration-cluster robust linear-model interval; not a capacity bound',screen_seeds=len(seeds)))
            delta=contrast-reference;effect=float(delta@beta);de=float(np.sqrt(max(0,delta@cov@delta)))
            comparisons.append(dict(factor=factor,level=value,reference_level=reference_levels[factor],
                adjusted_delta_psnr=effect,ci_low=effect-critical*de,ci_high=effect+critical*de,
                caution='exploratory model-adjusted contrast, not a confirmatory architecture test'))
    write('factor_effects.csv',rows)
    write('factor_level_comparisons.csv',comparisons)
    observed=np.array(observed);pred=np.array(pred)
    info=dict(rows=len(y),configs=len(design),rank=int(np.linalg.matrix_rank(a)),columns=a.shape[1],
        train_r2=float(1-np.sum(res**2)/np.sum((y-y.mean())**2)),
        grouped_loo_r2=float(1-np.sum((pred-observed)**2)/np.sum((observed-observed.mean())**2)),
        grouped_loo_rmse_db=float(np.sqrt(np.mean((pred-observed)**2))),
        caution='registered interaction model only; weak grouped prediction means marginal effects are not reliable for unobserved architectures')
    dump('effect_model.json',info)
    write('effect_model_loo.csv',[dict(config=c['id'],observed=o,predicted=p) for c,o,p in zip(design,observed,pred)])
    return info

def cube(scene,phase,split):
    base=ROOT/'results'/scene/phase
    tables={}
    for path in base.glob('*/*/'+split+'_full.csv'):
        config=path.parent.parent.name;seed=int(path.parent.name)
        table=[x for x in read(path) if x['policy']=='same_k']
        tables[config,seed]=table
    configs=sorted({c for c,s in tables});seeds=sorted({s for c,s in tables})
    assert configs and seeds
    paths=sorted({r['path'] for r in next(iter(tables.values()))})
    values={m:np.zeros((len(configs),len(seeds),len(RATIOS),len(paths))) for m in METRICS}
    details=[]
    for ci,c in enumerate(configs):
        for si,s in enumerate(seeds):
            table=tables[c,s]
            keyed={(float(r['ratio']),r['path']):r for r in table}
            assert len(keyed)==len(RATIOS)*len(paths)
            for ri,ratio in enumerate(RATIOS):
                for vi,path in enumerate(paths):
                    r=keyed[ratio,path]
                    for metric in METRICS:values[metric][ci,si,ri,vi]=float(r[metric])
                    details.append(dict(scene=scene,phase=phase,split=split,config=c,seed=s,ratio=ratio,path=path,
                        selected=r['selected'],**{m:r[m] for m in METRICS}))
    write(f'{scene}_{phase}_{split}_per_view.csv',details)
    return configs,seeds,paths,values

def bootstrap(values,n=10000,seed=881):
    # Same seed/block draws for every architecture and ratio preserve pairing.
    c,s,r,v=values.shape;b=max(2,min(8,v//3));blocks=np.array_split(np.arange(v),b)
    sums=np.stack([values[:,:,:,j].sum(-1) for j in blocks],axis=-1)
    sizes=np.array([len(j) for j in blocks]);rng=np.random.default_rng(seed)
    sw=rng.multinomial(s,np.ones(s)/s,n).astype(float)
    vw=rng.multinomial(b,np.ones(b)/b,n).astype(float)
    draws=np.einsum('ts,csrb,tb->tcr',sw,sums,vw,optimize=True)/(s*(vw@sizes))[:,None,None]
    return draws,b

def permutation(seed_delta):
    x=np.asarray(seed_delta);signs=np.array(list(itertools.product([-1,1],repeat=len(x))))
    statistics=(signs*x).mean(1);actual=abs(x.mean())
    return float(np.mean(np.abs(statistics)>=actual-1e-12))
def holm(values):
    order=np.argsort(values);adjusted=np.zeros(len(values));current=0.
    for rank,j in enumerate(order):
        current=max(current,min(1.,values[j]*(len(values)-rank)));adjusted[j]=current
    return adjusted

def statistics(scene,phase,split,winner):
    configs,seeds,paths,data=cube(scene,phase,split)
    draws,blocks=bootstrap(data['psnr']);means=data['psnr'].mean((1,3))
    neural=[j for j,c in enumerate(configs) if c!='R3'];best=np.argmax(draws[:,neural].mean(-1),axis=1)
    probs={configs[j]:float(np.mean(best==i)) for i,j in enumerate(neural)}
    write(f'{scene}_{phase}_{split}_best_frequency.csv',[dict(config=c,bootstrap_best_frequency=p) for c,p in probs.items()])
    index=configs.index(winner);rows=[];pvalues=[];primary_rows=[]
    groups=[('all',list(range(7))),('train',[0,2,4,6]),('unseen',[1,3,5])]+[(str(r),[i]) for i,r in enumerate(RATIOS)]
    for j,reference in enumerate(configs):
        if j==index:continue
        delta=data['psnr'][index]-data['psnr'][j]
        bootdelta=draws[:,index]-draws[:,j]
        for group,indices in groups:
            effect=float(delta[:,indices].mean());bd=bootdelta[:,indices].mean(1)
            sd=delta[:,indices].mean((1,2));p=permutation(sd)
            ci=np.quantile(bd,[.025,.975])
            row=dict(scene=scene,phase=phase,split=split,winner=winner,reference=reference,group=group,
                seed_count=len(seeds),view_count=len(paths),camera_blocks=blocks,delta_psnr=effect,
                ci95_low=float(ci[0]),ci95_high=float(ci[1]),bootstrap_positive=float(np.mean(bd>0)),
                bootstrap_over_0p1db=float(np.mean(bd>.1)),permutation_p=p,holm_primary_p='',
                practically_equivalent_95=bool(ci[0]>-.1 and ci[1]<.1),
                interpretation='confirmatory secondary-scene primary only' if scene=='fr2_xyz' and split=='test' and group=='all' else 'exploratory or secondary metric')
            for metric in ['ssim','lpips','photo']:
                row['delta_'+metric]=float((data[metric][index,:,indices]-data[metric][j,:,indices]).mean())
            rows.append(row)
            if group=='all':pvalues.append(p);primary_rows.append(row)
    for row,p in zip(primary_rows,holm(pvalues)):row['holm_primary_p']=float(p)
    write(f'{scene}_{phase}_{split}_paired_statistics.csv',rows)
    seed_rows=[]
    for ci,c in enumerate(configs):
        for si,s in enumerate(seeds):
            for group,indices in groups[:3]:
                seed_rows.append(dict(scene=scene,phase=phase,split=split,config=c,seed=s,group=group,
                    **{m:float(v[ci,si,indices].mean()) for m,v in data.items()}))
    write(f'{scene}_{phase}_{split}_seed_metrics.csv',seed_rows)
    return dict(configs=configs,seeds=seeds,views=len(paths),blocks=blocks,best_frequency=probs,
        primary=[{k:v for k,v in row.items()} for row in primary_rows])

def budgets_and_convergence():
    budget=[];curves=[];fairness=[]
    for path in (ROOT/'results').glob('*/*/*/*/complete.json'):
        scene,phase,config,seed=path.parts[-5:-1]
        info=json.loads(path.read_text());expected=4000 if phase=='screen' else 12000
        assert info['steps']==expected and all(v==expected//4 for v in info['ratio_exposures'].values())
        assert info['master_frozen']
        fairness.append(dict(scene=scene,phase=phase,config=config,seed=seed,steps=expected,
            exposure_per_ratio=expected//4,empty_steps=sum(info['empty_steps'].values()),
            parameters=info['parameters'],min_adam_clock=min(info['adam_clocks'].values()),max_adam_clock=max(info['adam_clocks'].values())))
        for f in ['validation_full.csv','test_full.csv']:
            p=path.parent/f
            if not p.exists():continue
            table=read(p)
            for ratio in RATIOS:
                a=[r for r in table if r['policy']=='cached' and float(r['ratio'])==ratio]
                if a:budget.append(dict(scene=scene,phase=phase,split=a[0]['split'],config=config,seed=seed,ratio=ratio,
                    selected=int(a[0]['selected']),p_mean=float(a[0]['p_mean']),relative_expected_budget_error=float(a[0]['p_mean'])/ratio-1))
        p=path.parent/'hard_training_panel.csv'
        if p.exists():
            for row in read(p):row.update(scene=scene,phase=phase,config=config,seed=seed);curves.append(row)
    write('fairness_checks.csv',fairness);write('budget_errors.csv',budget);write('hard_training_curves.csv',curves)
    return dict(verified_models=len(fairness),iterations_equal_within_stage=True,exposures_equal=True,master_frozen=True,
        note='Shared parameters receive every step; ratio-specific tables receive one quarter; legitimate zero gradients remain distinct from fabricated optimizer steps')


def descriptive_tables():
    quality=[];paired=[];convergence=[];exposure=[]
    for scene,phase in [('fr1_xyz','confirm'),('fr2_xyz','external')]:
        base=ROOT/'results'/scene/phase
        winner=json.loads((ROOT/'winner_lock.json').read_text())['winner']['id']
        for split in ['validation','test']:
            tables={}
            for path in base.glob('*/*/'+split+'_full.csv'):
                config=path.parent.parent.name;seed=int(path.parent.name)
                tables[config,seed]=read(path)
            for config in sorted({c for c,s in tables}):
                seeds=sorted(s for c,s in tables if c==config)
                for policy in ['same_k','cached']:
                    for ratio in RATIOS:
                        selected=[[r for r in tables[config,s] if r['policy']==policy and float(r['ratio'])==ratio] for s in seeds]
                        assert all(selected)
                        row=dict(scene=scene,phase=phase,split=split,config=config,policy=policy,ratio=ratio,seeds=len(seeds),views=len(selected[0]))
                        for metric in METRICS+['selected','visible','effective_proxy','p_mean']:
                            seed_values=np.array([np.mean([float(r[metric]) for r in a]) for a in selected])
                            row[metric]=float(seed_values.mean());row[metric+'_seed_std']=float(seed_values.std(ddof=1))
                        quality.append(row)
                if config==winner:continue
                for policy in ['same_k','cached']:
                    for ratio in RATIOS:
                        for metric in METRICS:
                            differences=[]
                            for seed in seeds:
                                ref={r['path']:float(r[metric]) for r in tables[config,seed] if r['policy']==policy and float(r['ratio'])==ratio}
                                current={r['path']:float(r[metric]) for r in tables[winner,seed] if r['policy']==policy and float(r['ratio'])==ratio}
                                assert ref.keys()==current.keys()
                                differences.extend(current[p]-ref[p] for p in ref)
                            d=np.array(differences);sign=1 if metric in ['psnr','ssim'] else -1
                            paired.append(dict(scene=scene,split=split,winner=winner,reference=config,policy=policy,ratio=ratio,metric=metric,
                                mean=float(d.mean()),median=float(np.median(d)),p10=float(np.quantile(d,.1)),p90=float(np.quantile(d,.9)),
                                improved_fraction=float(np.mean(sign*d>0)),note='descriptive paired seed-view differences; not independent view trials'))
        for path in base.glob('*/*/validation_per_view.csv'):
            config=path.parent.parent.name;seed=int(path.parent.name)
            data=read(path)
            for step in [4000,8000,12000]:
                for label,ratios in [('all',RATIOS),('train',[.01,.05,.1,.15]),('unseen',[.025,.075,.125])]:
                    subset=[r for r in data if int(r['step'])==step and float(r['ratio']) in ratios]
                    convergence.append(dict(scene=scene,phase=phase,config=config,seed=seed,step=step,split='validation',group=label,
                        psnr=float(np.mean([float(r['psnr']) for r in subset])),photo=float(np.mean([float(r['photo']) for r in subset]))))
        for path in base.glob('*/*/selected_exposure_counts.npy'):
            x=np.load(path)
            for i,ratio in enumerate([.01,.05,.1,.15]):
                exposure.append(dict(scene=scene,phase=phase,config=path.parent.parent.name,seed=int(path.parent.name),ratio=ratio,
                    selected_ever_fraction=float(np.mean(x[i]>0)),count_min=int(x[i].min()),count_median=float(np.median(x[i])),
                    count_p90=float(np.quantile(x[i],.9)),count_max=int(x[i].max()),
                    caution='selection exposure only; not a claim of nonzero photo gradient or actual contribution'))
    write('quality_summary.csv',quality);write('paired_view_descriptive.csv',paired)
    write('validation_convergence.csv',convergence);write('selection_exposures.csv',exposure)


def matched_screen_contrasts():
    runs=json.loads((ROOT/'screen_runs.json').read_text());design=json.loads((ROOT/'screen_design.json').read_text())
    scores={(r['config']['id'],r['seed']):r['validation_score'] for r in runs};seeds=sorted({r['seed'] for r in runs});rows=[]
    for factor in LEVELS:
        others=[k for k in LEVELS if k!=factor]
        for a,b in itertools.combinations(design,2):
            if any(a[k]!=b[k] for k in others):continue
            if a[factor]>b[factor]:a,b=b,a
            values=[scores[b['id'],s]-scores[a['id'],s] for s in seeds]
            rows.append(dict(factor=factor,lower=a['id'],upper=b['id'],low_value=a[factor],high_value=b[factor],
                mean_delta_psnr=float(np.mean(values)),seed_min_delta=float(np.min(values)),seed_max_delta=float(np.max(values)),
                positive_seed_count=int(np.sum(np.array(values)>0)),seeds=len(seeds),
                note='same other four factors; exploratory screen4000; two order seeds on one map'))
    write('matched_screen_contrasts.csv',rows)


def matched_confirm_controls():
    configs,seeds,paths,data=cube('fr1_xyz','confirm','validation')
    draws,blocks=bootstrap(data['psnr'],seed=883)
    pairs=[('S16R00E14D11',c) for c in ['S16R00E14D01','S16R00E14D21','S16R00E14D10','S16R00E14D12']]
    pairs.extend([('S00R00E08D12','S16R00E08D12'),('S16R00E08D12','S16R00E14D12'),
                  ('S16R00E14D11','S16R08E14D11')]);rows=[]
    for baseline,candidate in pairs:
        a,b=configs.index(baseline),configs.index(candidate)
        delta=data['psnr'][b]-data['psnr'][a];boot=(draws[:,b]-draws[:,a]).mean(1);ci=np.quantile(boot,[.025,.975])
        rows.append(dict(baseline=baseline,candidate=candidate,delta_psnr=float(delta.mean()),
            ci95_low=float(ci[0]),ci95_high=float(ci[1]),seed_improved=int(np.sum(delta.mean((1,2))>0)),seeds=len(seeds),
            delta_training_ratios=float(delta[:,[0,2,4,6]].mean()),delta_unseen_ratios=float(delta[:,[1,3,5]].mean()),
            permutation_p=permutation(delta.mean((1,2))),note='Bank-only matched controlled follow-up; not independent-scene confirmation'))
    for row,p in zip(rows,holm([r['permutation_p'] for r in rows])):row['holm_p']=float(p)
    write('matched_confirm_controls.csv',rows)

if __name__=='__main__':
    lock=json.loads((ROOT/'winner_lock.json').read_text());winner=lock['winner']['id']
    outputs=dict(winner=lock['winner'],effect_model=effects(),fairness=budgets_and_convergence(),
        confirmation_validation=statistics('fr1_xyz','confirm','validation',winner),
        historical_fr1_test=statistics('fr1_xyz','confirm','test',winner),
        independent_fr2_test=statistics('fr2_xyz','external','test',winner))
    descriptive_tables();matched_screen_contrasts();matched_confirm_controls()
    dump('statistical_summary.json',outputs)
    print(json.dumps(outputs,indent=2),flush=True)
