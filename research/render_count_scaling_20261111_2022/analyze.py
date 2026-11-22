"""输出全部慢例、逐视角K相关性和配对速度，不通过筛选场景改善结论。"""
from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent
R=HERE/'reports'
OUT=R/'delivery'
V=('G','C','F')
PRIMARY=('local0.1','local0.2','local0.3','local0.4','local0.5')


def geometric(x):
    return float(np.exp(np.log(x).mean()))


def correlations(df, causal=False):
    keys=['scene','frame','variant']+([] if causal else ['round','scope'])
    selected=df if causal else df[df.condition.str.startswith('local')]
    rows=[]
    for key,q in selected.groupby(keys,sort=True):
        group=dict(zip(keys,key))
        cols=['K']+(['ratio'] if causal else ['condition'])
        data=q.groupby(cols,sort=False).ms.median().reset_index().sort_values('K')
        if data.K.nunique()<3:continue
        # 合并相同K，只用于相关性；逐条件原始耗时仍完整保留。
        distinct=data.groupby('K').ms.median().sort_index()
        k=distinct.index.to_numpy();t=distinct.to_numpy()
        delta=np.diff(t);relative=delta/t[:-1]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rho=float(spearmanr(k,t).statistic)
        rows.append(dict(**group,rho=rho,K_min=int(k.min()),K_max=int(k.max()),conditions=len(k),
                         adjacent_pairs=len(delta),reversed_pairs=int((delta<0).sum()),
                         reversed_over_3pct=int((relative<-.03).sum()),strictly_monotonic=bool((delta>=0).all()),
                         worst_reversal_pct=float(min(0.,relative.min())*100)))
    return pd.DataFrame(rows)


def main():
    controller=json.loads((R/'FORMAL_CONTROLLER.json').read_text())
    assert controller['state']=='complete' and len(controller['accepted'])==32
    for n in controller['accepted']:
        j=json.loads((R/'jobs'/(n+'.json')).read_text())
        assert j['measurement_valid'] and j['returncode']==0
    sources=lambda name:pd.concat([pd.read_csv(R/n/(name+'.csv')) for n in controller['accepted']
                                   if (R/n/(name+'.csv')).is_file()],ignore_index=True)
    d=sources('samples');checks=sources('checks');workloads=sources('workloads');events=sources('events');memory=sources('memory')
    assert len(d)==253440 and len(checks)==21120 and len(workloads)==3520
    assert checks.rgb_bitwise.all() and checks.radii_bitwise.all()
    assert d.scene.nunique()==32
    group=['scene','split','condition','round','scope','variant']
    summary=d.groupby(group).agg(p50_ms=('ms','median'),p95_ms=('ms',lambda v:np.quantile(v,.95)),
                                K_min=('K','min'),K_median=('K','median'),K_max=('K','max'),N=('N','first'),
                                samples=('ms','size')).reset_index()
    assert (summary.samples==60).all()
    p=summary.pivot(index=group[:-1],columns='variant',values=['p50_ms','p95_ms']).reset_index()
    p.columns=[('_'.join(filter(None,x)) if isinstance(x,tuple) else x) for x in p.columns]
    p['G_over_F']=p.p50_ms_G/p.p50_ms_F;p['C_over_F']=p.p50_ms_C/p.p50_ms_F
    p['G_minus_F_us']=1000*(p.p50_ms_G-p.p50_ms_F)
    for col in ('K_min','K_median','K_max','N'):
        p[col]=summary[summary.variant=='F'].set_index(group[:-1])[col].reindex(p.set_index(group[:-1]).index).to_numpy()
    slow=p[(p.G_over_F<1)|(p.C_over_F<1)].copy()
    local_cor=correlations(d)
    causal=pd.read_csv(R/'causal01/samples.csv')
    causal_cor=correlations(causal,causal=True)
    OUT.mkdir(exist_ok=False)
    files={'PER_SCENE_RATIO.csv':p,'PER_VARIANT_TIMINGS.csv':summary,'ALL_SLOWER_CASES.csv':slow,
           'LOCAL_COUNT_CORRELATION.csv':local_cor,'NESTED_COUNT_CORRELATION.csv':causal_cor,
           'WORKLOADS.csv':workloads,'CUDA_EVENTS.csv':events,'WORKSPACE.csv':memory,
           'CORRECTNESS.csv':checks}
    for path,value in files.items():value.to_csv(OUT/path,index=False)
    d.to_csv(OUT/'TIMING_SAMPLES.csv.gz',index=False)
    causal.to_csv(OUT/'NESTED_TIMING_SAMPLES.csv.gz',index=False)
    pd.read_csv(R/'causal01/workloads.csv').to_csv(OUT/'NESTED_WORKLOADS.csv',index=False)
    def speed(data):
        return dict(G_over_F=geometric(data.G_over_F),C_over_F=geometric(data.C_over_F),
                    median_us_saved_vs_G=float(data.G_minus_F_us.median()),
                    G_slower=int((data.G_over_F<1).sum()),C_slower=int((data.C_over_F<1).sum()),
                    comparisons=len(data),min_G_over_F=float(data.G_over_F.min()),
                    min_C_over_F=float(data.C_over_F.min()),
                    G_p50_ms=float(data.p50_ms_G.median()),C_p50_ms=float(data.p50_ms_C.median()),
                    F_p50_ms=float(data.p50_ms_F.median()),
                    F_p95_ms=float(data.p95_ms_F.median()))
    results=dict(samples=len(d),checks=len(checks),workloads=len(workloads),scene_modes=32,
                 accepted_jobs=controller['accepted'],excluded_jobs=controller['rejected'],
                 conditions={},dataset_primary={},full={},correlation={},memory={})
    for scope in ('raster','pipeline'):
        results['conditions'][scope]={cond:speed(z) for cond,z in p[p.scope==scope].groupby('condition')}
        results['full'][scope]=speed(p[(p.scope==scope)&(p.condition=='full')])
        q=p[(p.scope==scope)&p.condition.isin(PRIMARY)]
        results[scope+'_primary']=speed(q)
        results[scope+'_heldout_primary']=speed(q[q.split=='heldout'])
        for ds in ('Replica','TUM_RGBD','VECtor'):
            results['dataset_primary'][ds+'/'+scope]=speed(q[q.scene.str.startswith(ds+'/')])
    for name,df in [('local',local_cor),('nested',causal_cor)]:
        groups=[('raster',df)] if name=='nested' else list(df.groupby('scope'))
        for scope,df in groups:
            results['correlation'][name+'/'+scope]={}
            for v,q in df.groupby('variant'):
                results['correlation'][name+'/'+scope][v]=dict(views=len(q),median_rho=float(q.rho.median()),
                    min_rho=float(q.rho.min()),positive_views=int((q.rho>0).sum()),
                    monotonic_views=int(q.strictly_monotonic.sum()),adjacent_pairs=int(q.adjacent_pairs.sum()),
                    reversed_pairs=int(q.reversed_pairs.sum()),reversed_over_3pct=int(q.reversed_over_3pct.sum()),
                    worst_reversal_pct=float(q.worst_reversal_pct.min()))
    for v,q in memory.groupby('variant'):
        results['memory'][v]=dict(max_workspace_mib=float(q.workspace_bytes.max()/(1<<20)),
                                  max_plans=int(q.plans.max()))
    # 每个视角/轮次与自己的Full比较，不跨场景混淆地图规模。
    vv=d.groupby(['scene','round','scope','variant','frame','condition']).agg(ms=('ms','median'),K=('K','first')).reset_index()
    ff=vv[vv.condition=='full'].drop(columns=['condition','K']).rename(columns={'ms':'full_ms'})
    vv=vv.merge(ff,on=['scene','round','scope','variant','frame'],validate='many_to_one')
    vv['speedup_vs_own_full']=vv.full_ms/vv.ms
    vv.to_csv(OUT/'PER_VIEW_VS_FULL.csv',index=False)
    results['low_vs_full']={}
    for (scope,cond),q in vv[(vv.variant=='F')&vv.condition.isin(['local0.01','local0.05',*PRIMARY])].groupby(['scope','condition']):
        results['low_vs_full'][scope+'/'+cond]=dict(cases=len(q),faster=int((q.speedup_vs_own_full>1).sum()),
                    median_speedup=float(q.speedup_vs_own_full.median()),min_speedup=float(q.speedup_vs_own_full.min()))
    (OUT/'SUMMARY.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    plots(p,summary,causal)
    print(json.dumps({k:v for k,v in results.items() if k in ('raster_primary','pipeline_primary','full','correlation')},indent=2))


def plots(p,variants,causal):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    palette={'G':'#526E98','C':'#9E7B53','F':'#007F73'}
    labels={'G':'Previous G','C':'Footprint C','F':'New F'}
    ratios=[.01,.05,.1,.2,.3,.4,.5,.7]
    fig,ax=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for j,scope in enumerate(('raster','pipeline')):
        for v in V:
            q=p[p.scope==scope]
            y=[q.loc[q.condition==f'local{r:g}',f'p50_ms_{v}'].median() for r in ratios]
            ax[j].plot(ratios,y,'o-',color=palette[v],label=labels[v])
        ax[j].set(xlabel='Local ratio',ylabel='Latency (ms)',title='Rasterizer' if scope=='raster' else 'Fresh prepare + render')
    full=causal[causal.ratio==1.].groupby(['scene','frame','variant']).ms.median().rename('full_ms').reset_index()
    # 所有版本都除以G的Full，使Full差异仍可见。
    base=full[full.variant=='G'].drop(columns='variant')
    t=causal.merge(base,on=['scene','frame'],validate='many_to_one')
    t['normalized']=t.ms/t.full_ms
    for v in V:
        q=t[t.variant==v].groupby('ratio').normalized.median()
        ax[2].plot(q.index,q.values,'o-',label=labels[v],color=palette[v])
    ax[2].set(xlabel='Kept fraction of same frozen map',ylabel='Latency / previous Full',title='Nested deletion, unchanged attributes')
    for a in ax:a.grid(axis='y',alpha=.25);a.legend(frameon=False,fontsize=8)
    fig.savefig(OUT/'COUNT_SCALING_OVERVIEW.png',dpi=200);fig.savefig(OUT/'COUNT_SCALING_OVERVIEW.svg');plt.close(fig)
    # 每个场景用实际K作横轴，所有场景完整展示。
    for ds,cols in [('Replica',4),('TUM_RGBD',3),('VECtor',3)]:
        scenes=sorted(set(p[p.scene.str.startswith(ds+'/')].scene))
        nrows=math.ceil(len(scenes)/cols)
        fig,axes=plt.subplots(nrows,cols,figsize=(cols*4,nrows*2.8),squeeze=False,layout='constrained')
        for a,s in zip(axes.flat,scenes):
            for v in V:
                q=variants[(variants.scene==s)&(variants.scope=='raster')&
                           (variants.variant==v)&variants.condition.str.startswith('local')]
                q=q.groupby('condition').agg(K=('K_median','median'),ms=('p50_ms','median')).sort_values('K')
                a.plot(q.K,q.ms,'o-',color=palette[v],label=labels[v],markersize=3)
            a.set(title=s.split('/',1)[1],xlabel='Input Gaussians K',ylabel='Rasterizer (ms)')
            a.grid(axis='y',alpha=.25)
        for a in list(axes.flat)[len(scenes):]:a.set_visible(False)
        axes.flat[0].legend(frameon=False,fontsize=8)
        fig.savefig(OUT/(ds+'_ACTUAL_K.png'),dpi=180)
        fig.savefig(OUT/(ds+'_ACTUAL_K.svg'));plt.close(fig)


if __name__=='__main__':main()
