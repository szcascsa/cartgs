"""E50/E8配对质量、有限值、收敛与固定视图示例；不选择模型。"""
import csv
import itertools
import json
from pathlib import Path
import shutil
import tarfile
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import ROOT,OLD,PROTOCOL,Network,rt

OUT=ROOT/'analysis';OUT.mkdir(exist_ok=True)
RATIOS=rt.RATIOS
METRICS=['psnr','ssim','lpips','photo']
GROUPS=[('all',list(range(7))),('train',[0,2,4,6]),('unseen',[1,3,5])]+[(str(r),[i]) for i,r in enumerate(RATIOS)]

def read(p):return list(csv.DictReader(Path(p).open()))
def write(p,rows):rt.write_csv(OUT/p,rows)
def dump(name,x):rt.save_json(OUT/name,x)

def bootstrap(x,seed=910,n=10000):
    s,r,v=x.shape;b=max(2,min(8,v//3));blocks=np.array_split(np.arange(v),b)
    sums=np.stack([x[:,:,ids].sum(-1) for ids in blocks],-1);sizes=np.array([len(i) for i in blocks])
    rng=np.random.default_rng(seed);sw=rng.multinomial(s,np.ones(s)/s,n);vw=rng.multinomial(b,np.ones(b)/b,n)
    return np.einsum('ts,srb,tb->tr',sw,sums,vw,optimize=True)/(s*(vw@sizes))[:,None]

def permutation(x):
    signs=np.array(list(itertools.product([-1,1],repeat=len(x))))
    return float(np.mean(abs((signs*x).mean(1))>=abs(x.mean())-1e-12))

def holm(p):
    order=np.argsort(p);a=np.zeros(len(p));prev=0.
    for rank,i in enumerate(order):prev=max(prev,min(1,p[i]*(len(p)-rank)));a[i]=prev
    return a

def directory(scene,config,seed):
    if config=='S00R00E50D12':return ROOT/'results'/scene/config/str(seed)
    return OLD/'results'/scene/PROTOCOL['baseline_phases'][scene]/config/str(seed)

def collect():
    quality=[];paired=[];per_view=[];recovery=[];fairness=[];curves=[];groups=[]
    for scene,seeds in PROTOCOL['seeds'].items():
        for c in ['S00R00E08D12','S00R00E50D12','R3']:
            for seed in seeds:
                d=directory(scene,c,seed);info=json.loads((d/'complete.json').read_text())
                assert info['steps']==12000 and info['master_frozen'] and all(v==3000 for v in info['ratio_exposures'].values())
                fairness.append(dict(scene=scene,config=c,seed=seed,steps=12000,exposure_per_ratio=3000,
                    empty_steps=sum(info['empty_steps'].values()),parameters=info['parameters'],reused=c!='S00R00E50D12',
                    min_adam_clock=min(info['adam_clocks'].values()),max_adam_clock=max(info['adam_clocks'].values())))
                if c!='S00R00E50D12':
                    target=ROOT/'baseline_references'/scene/c/str(seed);target.mkdir(parents=True,exist_ok=True)
                    for name in ['complete.json','validation_full.csv','test_full.csv','validation_per_view.csv','hard_training_panel.csv']:
                        if (d/name).exists():shutil.copy2(d/name,target/name)
                if c=='R3':continue
                for row in read(d/'validation_per_view.csv'):
                    if row['policy']=='same_k':curves.append(dict(scene=scene,config=c,seed=seed,**row))
        for split in ['validation','test']:
            tables={(c,s):read(directory(scene,c,s)/(split+'_full.csv')) for c in ['S00R00E08D12','S00R00E50D12','R3'] for s in seeds}
            paths=sorted({r['path'] for r in next(iter(tables.values()))})
            for policy in ['same_k','cached']:
                cubes={}
                for c in ['S00R00E08D12','S00R00E50D12','R3']:
                    cubes[c]={m:np.zeros((8,7,len(paths))) for m in METRICS}
                    for si,s in enumerate(seeds):
                        keyed={(float(r['ratio']),r['path']):r for r in tables[c,s] if r['policy']==policy}
                        assert len(keyed)==7*len(paths)
                        for ri,ratio in enumerate(RATIOS):
                            for vi,path in enumerate(paths):
                                r=keyed[ratio,path]
                                for m in METRICS:cubes[c][m][si,ri,vi]=float(r[m])
                                per_view.append(dict(r,scene=scene,config=c,seed=s,split=split,policy=policy))
                    for ri,ratio in enumerate(RATIOS):
                        row=dict(scene=scene,config=c,split=split,policy=policy,ratio=ratio,seeds=8,views=len(paths))
                        for m in METRICS:
                            x=cubes[c][m][:,ri].mean(1);row[m]=float(x.mean());row[m+'_seed_std']=float(x.std(ddof=1))
                        counts=[int(r['selected']) for s in seeds for r in tables[c,s] if r['policy']==policy and float(r['ratio'])==ratio]
                        row['selected_mean']=float(np.mean(counts));row['selected_min']=min(counts);row['selected_max']=max(counts)
                        quality.append(row)
                    for label,ids in GROUPS[:3]:
                        groups.append(dict(scene=scene,config=c,split=split,policy=policy,group=label,
                            **{m:float(x[:,ids].mean()) for m,x in cubes[c].items()}))
                if policy=='same_k':
                    for ratio in RATIOS:
                        counts=[r['selected_mean'] for r in quality if r['scene']==scene and r['split']==split and r['policy']==policy and r['ratio']==ratio]
                        assert len(set(counts))==1,'Evaluation K mismatch'
                for metric in METRICS:
                    delta=cubes['S00R00E50D12'][metric]-cubes['S00R00E08D12'][metric]
                    boot=bootstrap(delta);sign=1 if metric in ['psnr','ssim'] else -1
                    for label,ids in GROUPS:
                        x=delta[:,ids];bd=boot[:,ids].mean(1);ci=np.quantile(bd,[.025,.975]);seed_delta=x.mean((1,2))
                        paired.append(dict(scene=scene,split=split,policy=policy,group=label,metric=metric,mean=float(x.mean()),
                            median=float(np.median(x)),p10=float(np.quantile(x,.1)),p90=float(np.quantile(x,.9)),
                            improved_fraction=float(np.mean(sign*x>0)),seed_improved=int(np.sum(sign*seed_delta>0)),
                            ci95_low=float(ci[0]),ci95_high=float(ci[1]),bootstrap_improved=float(np.mean(sign*bd>0)),
                            permutation_p=permutation(seed_delta),holm_primary_p='',
                            primary=split=='test' and policy=='same_k' and metric=='psnr' and label=='all'))
                for label,ids in GROUPS:
                    b=cubes['S00R00E08D12']['psnr'][:,ids].mean();x=cubes['S00R00E50D12']['psnr'][:,ids].mean();o=cubes['R3']['psnr'][:,ids].mean()
                    recovery.append(dict(scene=scene,split=split,policy=policy,group=label,psnr_e8=float(b),psnr_e50=float(x),psnr_r3=float(o),
                        e50_minus_e8=float(x-b),e50_minus_r3=float(x-o),recovery_psnr_gap=float((x-b)/(o-b)) if o-b>.05 else ''))
    primary=[x for x in paired if x['primary']]
    for row,p in zip(primary,holm([x['permutation_p'] for x in primary])):row['holm_primary_p']=float(p)
    for name,rows in [('quality_summary.csv',quality),('quality_per_view.csv',per_view),('paired_statistics.csv',paired),
                      ('r3_gap_recovery.csv',recovery),('fairness_checks.csv',fairness),('validation_curves.csv',curves),('group_summary.csv',groups)]:write(name,rows)
    return dict(primary=primary,recovery=[r for r in recovery if r['split']=='test' and r['policy']=='same_k'],
        fairness=dict(models=len(fairness),new_models=16,baseline_or_reference_models=32,steps=12000,exposure_per_ratio=3000),
        groups=[g for g in groups if g['split']=='test' and g['policy']=='same_k'])

@torch.no_grad()
def outputs_and_examples():
    checks=[]
    for scene,seeds in PROTOCOL['seeds'].items():
        sc=rt.Scene(scene)
        for seed in seeds:
            info=json.loads((directory(scene,'S00R00E50D12',seed)/'complete.json').read_text())
            net=Network(sc,info['config'],seed);payload=torch.load(info['checkpoint'],map_location='cuda');net.load_state_dict(payload['state_dict']);del payload
            for ratio in RATIOS:
                p,h,logits=net(ratio);margin=logits.double()-torch.logit(sc.uniform.double());margin[~sc.eligible]=-torch.inf
                ids=torch.argsort(margin,descending=True,stable=True)[:sc.K[ratio]];delta=net.compensate(h,ids,ratio)
                activated=torch.cat([delta,sc.g.get_xyz[ids]+delta[:,:3],sc.g.get_scaling[ids]*delta[:,3:6].exp()],1)
                checks.append(dict(scene=scene,seed=seed,ratio=ratio,selected=len(ids),nonfinite_rows=int((~activated.isfinite().all(1)).sum()),
                    xyz_delta_p90=float(torch.quantile(delta[:,:3].norm(dim=1),.9)),logscale_max=float(delta[:,3:6].abs().max())))
            del net
        views=sc.load_test();chosen=[views[0],views[len(views)//2]];seed=seeds[0]
        nets={}
        for c in ['S00R00E08D12','S00R00E50D12','R3']:
            info=json.loads((directory(scene,c,seed)/'complete.json').read_text());net=Network(sc,info['config'],seed)
            payload=torch.load(info['checkpoint'],map_location='cuda');net.load_state_dict(payload['state_dict']);nets[c]=net.eval();del payload
        for ratio in [.01,.05]:
            fig,axes=plt.subplots(2,4,figsize=(16,6))
            for row,v in enumerate(chosen):
                axes[row,0].imshow(v['gt'].permute(1,2,0).cpu());axes[row,0].set_title('GT / '+v['path'])
                for col,(c,net) in enumerate(nets.items(),1):
                    p,h,l=net(ratio);margin=l.double()-torch.logit(sc.uniform.double());margin[~sc.eligible]=-torch.inf
                    ids=torch.argsort(margin,descending=True,stable=True)[:sc.K[ratio]].sort().values
                    image,_,_=sc.render(ids,net.compensate(h,ids,ratio),v);image=image.clamp(0,1)
                    psnr=float(-10*torch.log10((image-v['gt']).square().mean()))
                    axes[row,col].imshow(image.permute(1,2,0).cpu());axes[row,col].set_title(f'{c} / {psnr:.2f} dB / K={len(ids)}')
                for ax in axes[row]:ax.axis('off')
            fig.suptitle(f'{scene} / ratio={100*ratio:g}% / seed={seed} / first and middle test views, no best-view selection')
            fig.tight_layout();fig.savefig(OUT/f'{scene}_{int(ratio*100)}pct_examples.png',dpi=130);plt.close(fig)
        assert sc.unchanged()
        del nets,sc;torch.cuda.empty_cache()
    write('output_diagnostics.csv',checks)
    return sum(r['nonfinite_rows'] for r in checks)

def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rows]])

def report(summary):
    quality=read(OUT/'quality_summary.csv');curves=read(OUT/'validation_curves.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,scene in zip(axes,PROTOCOL['seeds']):
        for c in ['S00R00E08D12','S00R00E50D12','R3']:
            rows=sorted([r for r in quality if r['scene']==scene and r['split']=='test' and r['policy']=='same_k' and r['config']==c],key=lambda r:float(r['ratio']))
            ax.errorbar([100*float(r['ratio']) for r in rows],[float(r['psnr']) for r in rows],yerr=[float(r['psnr_seed_std']) for r in rows],marker='o',capsize=2,label=c)
        ax.set_title(scene+' / non-Bank same-K');ax.set_xlabel('Ratio (%)');ax.set_ylabel('PSNR (dB)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'quality_curves.png',dpi=160);plt.close(fig)
    lines=['# S00R00 补偿特征 E8→E50 单变量实验','',
        '本轮仅将每Anchor、每训练ratio的补偿latent由8维提高到50维：输入40→82，隐藏32→32和输出14均不变。Selector仍为11→32→32→2，保留Gaussian属性及log(ratio)，没有Selector Anchor latent。','',
        '## 结果','',table(['场景','E50−E8 PSNR','95%CI','改善seed','Holm p'],[
            [r['scene'],f'{r["mean"]:+.3f}',f'[{r["ci95_low"]:+.3f}, {r["ci95_high"]:+.3f}]',f'{r["seed_improved"]}/8',f'{r["holm_primary_p"]:.4f}'] for r in summary['primary']]),'',
        '上表为七档等权、non-Bank、确定性same-K的配对结果。95%区间来自seed与连续相机块的配对重采样；Holm仅校正本轮两个场景主比较。评价轨迹已用于此前结构研究，本轮不将其称为全新盲测。','',
        table(['场景','ratio组','E8 PSNR','E50 PSNR','R3 PSNR','R3差距恢复率'],[
            [r['scene'],r['group'],f'{r["psnr_e8"]:.3f}',f'{r["psnr_e50"]:.3f}',f'{r["psnr_r3"]:.3f}',f'{100*r["recovery_psnr_gap"]:.1f}%']
            for r in summary['recovery'] if r['group'] in ['all','train','unseen']]),'',
        '差距恢复率按PSNR dB差值定义 `(E50-E8)/(R3-E8)`，不是MSE恢复比例，也不是理论上界。','',
        table(['场景','方案','PSNR↑','SSIM↑','LPIPS↓','photo↓'],[
            [r['scene'],r['config'],f'{r["psnr"]:.3f}',f'{r["ssim"]:.4f}',f'{r["lpips"]:.4f}',f'{r["photo"]:.4f}'] for r in summary['groups'] if r['group']=='all']),'',
        '![Quality](analysis/quality_curves.png)','',
        '## 公平性与边界','',
        '- E50新增训练16次：fr1_xyz与fr2_xyz各8seed；全部12000真实LoD步，每个训练ratio3000次。E8与R3复用完全匹配的上一轮12000步结果，没有混入不同地图或训练预算。',
        '- 复用源checkpoint、Gaussian身份/数量、Anchor归属、冻结Master、相机与帧顺序、eligibility、cached uniform、Loss、Adam lr=0.001/eps=1e-15。Renderer二进制和原实验runtime SHA一致。',
        '- 公共Encoder/Selector、补偿公共权重以及原有8列投影初始化exact相同；仅增加私有坐标。所有方案step0 p=0.5、补偿exact零。',
        '- 固定的是Selector结构；Encoder、Selector和补偿仍按原隔离协议共同训练。选点集合可能随优化变化，因此这是同K整体系统效果，不是冻结选点后的纯解码容量上界。',
        '- 训练ratio为1/5/10/15%，未训练ratio为2.5/7.5/12.5%。latent先按raw ratio插值再解码，不为未见ratio单独训练；没有据评价数据选LR、step或seed。',
        '- 新增参数会改变优化条件；固定LR下的差异不能直接当作表示能力上限。完整收敛曲线、真实Adam时钟、空选择次数、逐视图mean/median/p10/p90及改善比例均保存。',
        f'- E50最终补偿及指数尺度等激活属性的非有限值行数：{summary["nonfinite_output_rows"]}。Master训练前后exact冻结。',
        '- 本轮是冻结地图GPU质量实验，不是新在线SLAM。训练补偿从第1步启用；生产默认仍保留3000 Mapping步warmup、原Joint/Full调度与自然停止规则。',
        '',
        '## 当前配置与交付','',
        '项目实验默认为 `s00r00e50d12`，E8回退文件 `cfg/selector/s00_e08_reference.yaml`，R3质量参考 `cfg/selector/r3_reference.yaml`。C++/Python新50维保存、插值和生命周期支持已接入；本地CPU/推理核验通过，未将其等同于完整CUDA Mapping构建或在线质量验证。','',
        '原始指标：`analysis/quality_summary.csv`、`quality_per_view.csv`、`paired_statistics.csv`、`r3_gap_recovery.csv`、`fairness_checks.csv`、`validation_curves.csv`。模型与训练日志保存在本次服务器目录；无早停、无容量/学习率追加搜索。未stage、commit或push。','',
        '## 固定视图示例','',
        '每场景固定第一个与中间non-Bank视图、固定第一个seed，显示GT/E8/E50/R3；不是挑选改善最大的图片。','']
    for scene in PROTOCOL['seeds']:
        for ratio in [1,5]:lines += [f'![{scene} {ratio}%](analysis/{scene}_{ratio}pct_examples.png)','']
    (ROOT/'S00_E50_COMPENSATION_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    while not (ROOT/'run.exit').exists():time.sleep(20)
    assert (ROOT/'run.exit').read_text().strip()=='0'
    summary=collect();summary['nonfinite_output_rows']=outputs_and_examples();dump('summary.json',summary)
    report(summary)
    paths=[]
    for p in ROOT.rglob('*'):
        if p.is_file() and p.suffix in ['.json','.csv','.py','.md','.png','.npy','.log']:paths.append(p)
    paths.extend((ROOT/'results').glob('*/S00R00E50D12/*/step_12000.pt'))
    with tarfile.open(ROOT/'evidence.tar.gz','w:gz') as pack:
        for p in paths:pack.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    print('E50_REPORT_READY',json.dumps(summary['primary']),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        (ROOT/'finalize.exit').write_text('1');raise
    (ROOT/'finalize.exit').write_text('0')
