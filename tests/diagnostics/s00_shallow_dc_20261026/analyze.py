"""浅Encoder加回DC的配对质量汇总，比较无DC浅层和原Baseline。"""
import csv
import json
from pathlib import Path
import shutil
import tarfile
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import ROOT,SOURCE,PROTOCOL,Network,config,source,rt

KINDS=['original_baseline','shallow_no_dc','shallow_dc']
METRICS=['psnr','ssim','lpips','photo']
GROUPS=[('all',rt.RATIOS),('train',rt.TRAIN),('unseen',[r for r in rt.RATIOS if r not in rt.TRAIN])]
LEGACY=source.source.previous.xyz.baseline


def read(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def directory(kind,seed):
    if kind=='original_baseline':return source.source.previous.xyz.SOURCE/'results/fr1_xyz'/LEGACY.config('baseline')['id']/str(seed)
    if kind=='shallow_no_dc':return SOURCE/'results/fr1_xyz'/source.config('shallow32')['id']/str(seed)
    return ROOT/'results/fr1_xyz'/config()['id']/str(seed)


def model(scene,kind,seed):
    if kind=='original_baseline':net=LEGACY.Network(scene,LEGACY.config('baseline'),seed)
    elif kind=='shallow_no_dc':net=source.Network(scene,source.config('shallow32'),seed)
    else:net=Network(scene,config(),seed)
    net.load_state_dict(torch.load(directory(kind,seed)/'step_12000.pt',map_location='cuda')['state_dict'])
    return net.eval()


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])


@torch.no_grad()
def sets_for(scene,net):
    sets={}
    for ratio in rt.RATIOS:
        p,h,logits=net(ratio);margin=logits.double()-torch.logit(scene.uniform.double());margin[~scene.eligible]=-torch.inf
        sets[f'same_k_{ratio}']=torch.argsort(margin,descending=True,stable=True)[:scene.K[ratio]].sort().values.cpu().numpy()
        sets[f'cached_{ratio}']=torch.nonzero(p>scene.uniform).flatten().cpu().numpy()
    return sets


def main():
    assert (ROOT/'run.exit').read_text().strip()=='0'
    out=ROOT/'analysis';out.mkdir(exist_ok=True)
    scene=rt.Scene('fr1_xyz');train_paths={v['path'] for v in scene.train}
    per_view,quality,groups,paired,fairness,curves,overlap=[],[],[],[],[],[],[]
    values,sets={},{}
    for kind in KINDS:
        for seed in PROTOCOL['seeds']:
            d=directory(kind,seed);info=json.loads((d/'complete.json').read_text())
            assert info['steps']==12000 and info['master_frozen']
            assert all(v==3000 for v in info['ratio_exposures'].values())
            empty_total=sum(info['empty_steps'].values())
            for name,value in info['adam_clocks'].items():
                expected=(3000-info['empty_steps'][str(rt.TRAIN[int(name.split('.')[1])])]
                    if name.startswith('extra_tables.') else 12000-empty_total if name.startswith('compensation.') else 12000)
                assert value==expected,(kind,seed,name,value,expected)
            if kind=='shallow_dc':assert info['config']['encoder_layout']=='14-32-ReLU'
            dest=d
            if kind!='shallow_dc':
                dest=ROOT/'references'/kind/str(seed);dest.mkdir(parents=True,exist_ok=True)
                for name in ['complete.json','step_12000.pt','bank_full.csv','validation_full.csv','test_full.csv',
                    'training_curve.csv','validation_per_view.csv']:shutil.copy2(d/name,dest/name)
            net=model(scene,kind,seed);sets[kind,seed]=sets_for(scene,net);del net
            np.savez_compressed(dest/'selected_sets.npz',**sets[kind,seed])
            fairness.append(dict(kind=kind,seed=seed,steps=12000,exposure_per_ratio=3000,parameters=info['parameters'],
                seconds=info['seconds'],adam_clocks=json.dumps(info['adam_clocks']),empty_steps=json.dumps(info['empty_steps']),
                compensation_updates=12000-empty_total,master_frozen=True,reused=kind!='shallow_dc'))
            curves.extend([dict(r,kind=kind,seed=seed) for r in read(d/'validation_per_view.csv')])
            for split in ['bank_train','validation','test']:
                rows=read(d/(('bank' if split=='bank_train' else split)+'_full.csv'))
                if split=='bank_train':rows=[dict(r,split=split) for r in rows if r['path'] in train_paths]
                values[kind,seed,split]=rows;per_view.extend([dict(r,kind=kind,seed=seed) for r in rows])
                for policy in ['same_k','cached']:
                    for ratio in rt.RATIOS:
                        sub=[r for r in rows if r['policy']==policy and float(r['ratio'])==ratio]
                        counts={int(r['selected']) for r in sub};assert len(counts)==1
                        quality.append(dict(kind=kind,seed=seed,split=split,policy=policy,ratio=ratio,selected=counts.pop(),
                            visible=np.mean([int(r['visible']) for r in sub]),effective_proxy=np.mean([int(r['effective_proxy']) for r in sub]),
                            **{m:np.mean([float(r[m]) for r in sub]) for m in METRICS}))
                    for group,ratios in GROUPS:
                        sub=[r for r in quality if r['kind']==kind and r['seed']==seed and r['split']==split
                            and r['policy']==policy and r['ratio'] in ratios]
                        groups.append(dict(kind=kind,seed=seed,split=split,policy=policy,group=group,
                            **{m:np.mean([r[m] for r in sub]) for m in METRICS}))
    for split in ['bank_train','validation','test']:
        for ratio in rt.RATIOS:
            assert len({r['selected'] for r in quality if r['split']==split and r['policy']=='same_k' and r['ratio']==ratio})==1
        for reference in KINDS[:2]:
            for policy in ['same_k','cached']:
                for group,ratios in GROUPS+[(str(r),[r]) for r in rt.RATIOS]:
                    for metric in METRICS:
                        diffs=[]
                        for seed in PROTOCOL['seeds']:
                            index=lambda kind:{(r['path'],float(r['ratio'])):r for r in values[kind,seed,split]
                                if r['policy']==policy and float(r['ratio']) in ratios}
                            a,b=index('shallow_dc'),index(reference);assert a.keys()==b.keys()
                            diffs.append(np.array([float(a[k][metric])-float(b[k][metric]) for k in a]))
                        v=np.stack(diffs);sign=1 if metric in ['psnr','ssim'] else -1
                        paired.append(dict(reference=reference,split=split,policy=policy,group=group,metric=metric,
                            mean=float(v.mean()),seed3100=float(v[0].mean()),seed3101=float(v[1].mean()),
                            median=float(np.median(v)),p10=float(np.quantile(v,.1)),p90=float(np.quantile(v,.9)),
                            improved_fraction=float((sign*v>0).mean()),seed_improved=int((sign*v.mean(1)>0).sum())))
    for reference in KINDS[:2]:
        for seed in PROTOCOL['seeds']:
            for key,a in sets[reference,seed].items():
                b=sets['shallow_dc',seed][key];inter=len(np.intersect1d(a,b));union=len(np.union1d(a,b))
                overlap.append(dict(reference=reference,seed=seed,set=key,reference_k=len(a),candidate_k=len(b),
                    intersection=inter,jaccard=inter/union if union else 1.,recall=inter/len(a) if len(a) else 1.))
    assert scene.unchanged()
    assert all(np.isfinite(float(r[m])) for r in per_view for m in METRICS)
    for name,rows in [('quality_per_view.csv',per_view),('quality_summary.csv',quality),('group_summary.csv',groups),
        ('paired_statistics.csv',paired),('fairness_checks.csv',fairness),('validation_curves.csv',curves),('selected_set_overlap.csv',overlap)]:rt.write_csv(out/name,rows)
    pick=lambda ref,g='all',policy='same_k',split='test':next(r for r in paired if r['reference']==ref
        and r['group']==g and r['policy']==policy and r['split']==split and r['metric']=='psnr')
    summary=dict(primary=[pick(ref,g) for ref in KINDS[:2] for g,_ in GROUPS],
        cached=[pick(ref,g,'cached') for ref in KINDS[:2] for g,_ in GROUPS],same_k_equal=True,source_frozen=True,
        metrics_finite=True,new_models=2,reused_models=4,steps=12000,exposure_per_ratio=3000,production_change=False,online_slam=False)
    rt.save_json(out/'summary.json',summary)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,seed in zip(axes,PROTOCOL['seeds']):
        for kind in KINDS:
            sub=sorted([r for r in quality if r['kind']==kind and r['seed']==seed and r['split']=='test'
                and r['policy']=='same_k'],key=lambda r:r['ratio'])
            ax.plot([r['ratio']*100 for r in sub],[r['psnr'] for r in sub],marker='o',label=kind)
        ax.set(xlabel='Ratio (%)',ylabel='PSNR (dB)',title=f'fr1_xyz seed{seed}');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'quality_curves.png',dpi=150);plt.close(fig)
    lines=['# 浅层Encoder同时加回SH DC的质量对照','',
        '本轮在11→32 ReLU浅Encoder方案上，仅向Encoder和四Head直连输入增加SH DC3，网络实际变为14→32 ReLU。没有保持11维后挤掉其他输入；所有此前输入均保留。','',
        '## non-Bank same-K结果','',table(['参照','ratio组','seed3100 ΔPSNR','seed3101 ΔPSNR','平均ΔPSNR'],
        [[ref,g,f'{pick(ref,g)["seed3100"]:+.3f}',f'{pick(ref,g)["seed3101"]:+.3f}',f'{pick(ref,g)["mean"]:+.3f}'] for ref in KINDS[:2] for g,_ in GROUPS]),'',
        table(['方案','PSNR↑','SSIM↑','LPIPS↓','photo↓'],[[kind,*[f'{np.mean([r[m] for r in groups if r["kind"]==kind and r["split"]=="test" and r["policy"]=="same_k" and r["group"]=="all"]):.5f}'
            for m in METRICS]] for kind in KINDS]),'',
        '## 各ratio增量','',table(['ratio','K','相对不含DC浅层','相对原Baseline'],
        [[f'{r*100:g}%',scene.K[r],f'{pick("shallow_no_dc",str(r))["mean"]:+.3f}',f'{pick("original_baseline",str(r))["mean"]:+.3f}'] for r in rt.RATIOS]),'',
        '## Bank、留出及原Cached评价','',table(['参照','Bank训练视图ΔPSNR','内部留出ΔPSNR','non-Bank实际Cached ΔPSNR'],
        [[ref,f'{pick(ref,split="bank_train")["mean"]:+.3f}',f'{pick(ref,split="validation")["mean"]:+.3f}',
            f'{pick(ref,policy="cached")["mean"]:+.3f}'] for ref in KINDS[:2]]),'',
        '## 结构与初始化','',
        '- 输入顺序为[全局XYZ3,activated Scale3,sigmoid Opacity1,局部Offset3,logratio1,SH DC3]共14维；浅Encoder14→32 ReLU，输出32维共享编码。DC为原始SH系数，不转RGB、不额外归一化。',
        '- 补偿输入raw14+每Anchor每ratio latent16+h32共62维，四Head仍各62→32 ReLU→32 ReLU→3/3/4/4。只对选中K行补偿，Selector不读取补偿latent。原14维输出含义与变换、ratio latent插值保留。',
        '- Encoder前11列和四Head公共输入列/偏置/后续层与不含DC方案同seed复制，新增DC列全0；step0概率和零补偿exact一致，隐藏函数浮点容忍检查通过。',
        '- 输入DC与原Baseline场景派生属性中的DC tensor exact一致；仅开放允许属性的代理和NaN旧派生输入检查通过，不依赖Rotation或其他隐式输入。',
        '- 从真实零补偿起点的3次photo反传验证Encoder及全部四Head的新增DC连接随后获得梯度；其他ratio表grad=None。实际首步隐藏梯度为0属于零输出初始化，不误报未训练。',
        '- Encoder参数384→480，四Head参数12366→12750，总参数146096→146576，新增480；因此本轮不仅增加信息，也增加输入连接，不能完全排除参数化作用。','',
        '## 协议与范围','',
        '- 同源fr1_xyz冻结N6838/M2066地图，Master/拓扑/Anchor归属/位姿/eligible/cached uniform不变。54 Bank训练320×240，54 Bank训练/13留出/45历史non-Bank评价640×480。',
        '- 两seed各12000步、每ratio3000曝光，复用不含DC浅层和原Baseline各两个锁定最终模型；同帧序、ratio顺序、fresh fused Adam lr=.001 eps1e-15、原photo+.01预算Loss。',
        '- 训练1/5/10/15%，另评2.5/7.5/12.5%，未见ratio不额外拟合。所有评价用最终12000步，未按non-Bank调整LR、步数或追加架构；4k/8k/12k曲线保留。',
        '- primary same-K按logit(p)-logit(cached uniform)排名，K=floor(rN)并受eligible上限；实际Cached数量另存。Selector与Encoder联合训练，选择身份可变，不是固定选择的纯补偿因果。',
        '- 原Baseline为Offset3+Scale3+SH DC3+Opacity1+logratio1，不含全局XYZ；候选还额外包含XYZ，且DC输入初始化不同，不能将与原Baseline的差值全归因于加回DC。',
        '- 各参数Adam时钟按实际非空选择精确核验，不为公平伪造零梯度step；若存在空选择，辅助更新次数在fairness_checks中完整报告。',
        '- 两seed是同一历史地图上的初始化/帧序重复；评价轨迹已多次用于研究，不能当全新盲测或声称跨场景、在线Joint收益。GPU有其他任务，耗时只记录不作性能结论。',
        '- 生产源码/默认未替换，未stage、commit或push。本轮不做其他输入、深度、维度或LR搜索。','',
        '![质量曲线](analysis/quality_curves.png)','']
    (ROOT/'SHALLOW_DC_COMPENSATION_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')
    examples(scene)
    with tarfile.open(ROOT/'evidence.tar.gz','w:gz') as pack:
        for path in ROOT.rglob('*'):
            if path.is_file() and (path.suffix in ['.py','.json','.csv','.md','.png','.npy','.npz'] or path.name=='step_12000.pt'):
                pack.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
    print('PRIMARY',json.dumps(summary),flush=True);print('ARCHIVE_SHA256',rt.sha(ROOT/'evidence.tar.gz'),flush=True)


@torch.no_grad()
def examples(scene):
    view=scene.load_test()[len(scene.load_test())//2];nets={kind:model(scene,kind,3100) for kind in KINDS}
    for ratio in [.01,.05]:
        fig,axes=plt.subplots(1,4,figsize=(16,4));axes[0].imshow(view['gt'].permute(1,2,0).cpu());axes[0].set_title('GT')
        for ax,(kind,net) in zip(axes[1:],nets.items()):
            p,h,logits=net(ratio);margin=logits.double()-torch.logit(scene.uniform.double());margin[~scene.eligible]=-torch.inf
            ids=torch.argsort(margin,descending=True,stable=True)[:scene.K[ratio]].sort().values
            image,_,_=scene.render(ids,net.compensate(h,ids,ratio),view);image=image.clamp(0,1)
            psnr=float(-10*torch.log10((image-view['gt']).square().mean()))
            ax.imshow(image.permute(1,2,0).cpu());ax.set_title(f'{kind}\n{psnr:.2f}dB / K={len(ids)}')
        for ax in axes:ax.axis('off')
        fig.suptitle(f'seed3100 / {ratio*100:g}% / fixed middle non-Bank view');fig.tight_layout()
        fig.savefig(ROOT/f'analysis/{int(ratio*100)}pct_examples.png',dpi=140);plt.close(fig)
    assert scene.unchanged()


if __name__=='__main__':
    try:main()
    except Exception:
        (ROOT/'analyze.exit').write_text('1');raise
    (ROOT/'analyze.exit').write_text('0')
