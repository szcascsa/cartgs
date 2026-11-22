"""等参数量Offset与SH DC输入增量的独立对照。"""
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
from run import ROOT,SOURCE,PROTOCOL,Network,config,previous,rt

KINDS=['baseline','base','offset','shdc']
METRICS=['psnr','ssim','lpips','photo']
GROUPS=[('all',rt.RATIOS),('train',rt.TRAIN),('unseen',[r for r in rt.RATIOS if r not in rt.TRAIN])]
COMPARISONS=[('offset','base'),('shdc','base'),('shdc','offset'),('offset','baseline'),('shdc','baseline')]


def read(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def directory(kind,seed):
    if kind=='baseline':return previous.xyz.SOURCE/'results/fr1_xyz'/previous.xyz.baseline.config('baseline')['id']/str(seed)
    if kind=='base':return SOURCE/'results/fr1_xyz'/previous.config()['id']/str(seed)
    return ROOT/'results/fr1_xyz'/config(kind)['id']/str(seed)


def model(scene,kind,seed):
    if kind=='baseline':net=previous.xyz.baseline.Network(scene,previous.xyz.baseline.config('baseline'),seed)
    elif kind=='base':net=previous.Network(scene,previous.config(),seed)
    else:net=Network(scene,config(kind),seed)
    net.load_state_dict(torch.load(directory(kind,seed)/'step_12000.pt',map_location='cuda')['state_dict'])
    return net.eval()


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])


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
            assert all(v==(3000 if k.startswith('extra_tables.') else 12000) for k,v in info['adam_clocks'].items())
            net=model(scene,kind,seed);sets[kind,seed]={}
            with torch.no_grad():
                for ratio in rt.RATIOS:
                    p,h,logits=net(ratio);margin=logits.double()-torch.logit(scene.uniform.double());margin[~scene.eligible]=-torch.inf
                    sets[kind,seed][f'same_k_{ratio}']=torch.argsort(margin,descending=True,stable=True)[:scene.K[ratio]].sort().values.cpu().numpy()
                    sets[kind,seed][f'cached_{ratio}']=torch.nonzero(p>scene.uniform).flatten().cpu().numpy()
            del net
            dest=d
            if kind in ['baseline','base']:
                dest=ROOT/'references'/kind/str(seed);dest.mkdir(parents=True,exist_ok=True)
                for name in ['complete.json','step_12000.pt','bank_full.csv','validation_full.csv','test_full.csv',
                    'training_curve.csv','validation_per_view.csv']:shutil.copy2(d/name,dest/name)
            np.savez_compressed(dest/'selected_sets.npz',**sets[kind,seed])
            fairness.append(dict(kind=kind,seed=seed,steps=12000,exposure_per_ratio=3000,parameters=info['parameters'],
                seconds=info['seconds'],adam_clocks=json.dumps(info['adam_clocks']),master_frozen=True,reused=kind in ['baseline','base']))
            curves.extend([dict(r,kind=kind,seed=seed) for r in read(d/'validation_per_view.csv')])
            for split in ['bank_train','validation','test']:
                rows=read(d/(('bank' if split=='bank_train' else split)+'_full.csv'))
                if split=='bank_train':rows=[dict(r,split=split) for r in rows if r['path'] in train_paths]
                values[kind,seed,split]=rows
                per_view.extend([dict(r,kind=kind,seed=seed) for r in rows])
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
        for candidate,reference in COMPARISONS:
            for policy in ['same_k','cached']:
                for group,ratios in GROUPS+[(str(r),[r]) for r in rt.RATIOS]:
                    for m in METRICS:
                        differences=[]
                        for seed in PROTOCOL['seeds']:
                            index=lambda k:{(r['path'],float(r['ratio'])):r for r in values[k,seed,split]
                                if r['policy']==policy and float(r['ratio']) in ratios}
                            a,b=index(candidate),index(reference);assert a.keys()==b.keys()
                            differences.append(np.array([float(a[k][m])-float(b[k][m]) for k in a]))
                        v=np.stack(differences);sign=1 if m in ['psnr','ssim'] else -1
                        paired.append(dict(candidate=candidate,reference=reference,split=split,policy=policy,group=group,metric=m,
                            mean=float(v.mean()),seed3100=float(v[0].mean()),seed3101=float(v[1].mean()),
                            median=float(np.median(v)),p10=float(np.quantile(v,.1)),p90=float(np.quantile(v,.9)),
                            improved_fraction=float((sign*v>0).mean())))
    for candidate,reference in COMPARISONS:
        for seed in PROTOCOL['seeds']:
            for key,a in sets[reference,seed].items():
                b=sets[candidate,seed][key];inter=len(np.intersect1d(a,b));union=len(np.union1d(a,b))
                overlap.append(dict(candidate=candidate,reference=reference,seed=seed,set=key,reference_k=len(a),candidate_k=len(b),
                    intersection=inter,jaccard=inter/union if union else 1.,recall=inter/len(a) if len(a) else 1.))
    assert scene.unchanged()
    assert all(np.isfinite(float(r[m])) for r in per_view for m in METRICS)
    for name,rows in [('quality_per_view.csv',per_view),('quality_summary.csv',quality),('group_summary.csv',groups),
        ('paired_statistics.csv',paired),('fairness_checks.csv',fairness),('validation_curves.csv',curves),('selected_set_overlap.csv',overlap)]:
        rt.write_csv(out/name,rows)
    pick=lambda candidate,ref,group='all',policy='same_k',split='test':next(r for r in paired if r['candidate']==candidate and r['reference']==ref and r['group']==group
        and r['policy']==policy and r['split']==split and r['metric']=='psnr')
    summary=dict(primary=[pick(a,b,g) for a,b in COMPARISONS for g,_ in GROUPS],
        cached=[pick(a,b,g,'cached') for a,b in COMPARISONS for g,_ in GROUPS],
        same_k_equal=True,source_frozen=True,metrics_finite=True,new_models=4,reused_models=4,
        steps=12000,exposure_per_ratio=3000,production_change=False,online_slam=False)
    rt.save_json(out/'summary.json',summary)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,seed in zip(axes,PROTOCOL['seeds']):
        for kind in KINDS:
            sub=sorted([r for r in quality if r['kind']==kind and r['seed']==seed and r['split']=='test'
                and r['policy']=='same_k'],key=lambda r:r['ratio'])
            ax.plot([r['ratio']*100 for r in sub],[r['psnr'] for r in sub],marker='o',label=kind)
        ax.set(xlabel='Ratio (%)',ylabel='PSNR (dB)',title=f'fr1_xyz seed{seed}');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'quality_curves.png',dpi=150);plt.close(fig)
    lines=['# 局部Offset与SH DC的等容量输入增量对照','',
        'base为上一轮世界XYZ+Scale+Opacity+logratio。A只加局部Offset3，B只加SH DC3，两组均保留世界XYZ；不是用Offset替换世界XYZ，也没有同时加两项。','',
        '## non-Bank same-K结果','',table(['对照','ratio组','seed3100 ΔPSNR','seed3101 ΔPSNR','平均ΔPSNR'],
        [[a+' − '+b,g,f'{pick(a,b,g)["seed3100"]:+.3f}',f'{pick(a,b,g)["seed3101"]:+.3f}',f'{pick(a,b,g)["mean"]:+.3f}']
            for a,b in COMPARISONS for g,_ in GROUPS]),'',
        table(['方案','PSNR↑','SSIM↑','LPIPS↓','photo↓'],[[kind,*[f'{np.mean([r[m] for r in groups if r["kind"]==kind and r["split"]=="test" and r["policy"]=="same_k" and r["group"]=="all"]):.5f}'
            for m in METRICS]] for kind in KINDS]),'',
        '## 逐ratio变化','',table(['ratio','K','Offset − base','SH DC − base','SH DC − Offset'],
            [[f'{r*100:g}%',scene.K[r],f'{pick("offset","base",str(r))["mean"]:+.3f}',f'{pick("shdc","base",str(r))["mean"]:+.3f}',
                f'{pick("shdc","offset",str(r))["mean"]:+.3f}'] for r in rt.RATIOS]),'',
        '## Bank、留出及实际Cached评价','',table(['对照','Bank训练视图ΔPSNR','内部留出ΔPSNR','non-Bank实际Cached ΔPSNR'],
            [[a+' − '+b,f'{pick(a,b,split="bank_train")["mean"]:+.3f}',f'{pick(a,b,split="validation")["mean"]:+.3f}',
                f'{pick(a,b,policy="cached")["mean"]:+.3f}'] for a,b in COMPARISONS]),'',
        '## 结构与初始化','',
        '- 两组Encoder输入均为原raw8+新增3维=11维，11→32 ReLU；补偿输入raw11+每Anchor每ratio latent16+h32=59维，四Head均59→32 ReLU→32 ReLU→3/3/4/4。',
        '- Offset=(真实XYZ−冻结Anchor中心)/冻结局部参考尺度，原世界XYZ同时保留。SH DC是基底球谐DC系数3维，不是RGB；SH分支不读取Anchor中心/尺度。两组都不输入Rotation。',
        '- 原common8/latent/h32权重、bias和后续层同seed复制，新增3维输入列零初始化；两组初始完整state dict exact相同，step0概率和零补偿与base exact一致。',
        '- 仅暴露允许属性的代理及旧派生tensor设NaN检查通过；SH分支用不可访问的Anchor空间元数据对象进一步检查无Offset依赖。Encoder与四Head新增连接梯度、其他ratio表grad=None通过。',
        '- 两组总参数均146096、四Head均12366；相对base各增加480个输入连接参数，彼此容量相同。共享h32、latent16插值及14D输出变换全部保留。','',
        '## 协议与因果边界','',
        '- 同源fr1_xyz冻结N6838/M2066地图；Master/拓扑/位姿/关联/eligible/uniform保持一致。两组seed3100/3101各12000步、每ratio3000曝光；第二seed反转两臂运行顺序。复用匹配的两个base和两个原Baseline最终模型。',
        '- 同Bank帧序、四档ratio顺序、fresh fused Adam lr=.001 eps1e-15、photo=.8L1+.2(1-fast_ssim)+.01预算项；无non-Bank调参或选步数。',
        '- 54 Bank训练320×240，Bank训练54/内部留出13/non-Bank45评价640×480；训练1/5/10/15%，另评2.5/7.5/12.5%，latent插值规则不变。',
        '- primary same-K按logit(p)-logit(cached uniform)排序，实际Cached结果另存。Selector/Encoder仍联合训练，集合可变化；不能将全部变化归于补偿表达。',
        '- Offset与SH分支等参数、同初值；其直接对照检验新信息类型的差异，但Selector/Encoder仍联合训练，不能把全部收益单独归于补偿。',
        '- 原Baseline输入为局部Offset+Scale+SH DC+Opacity+logratio；与本轮各组比较仍涉及坐标/信息组合及输入列初始化方式差异，不能只归因于某一项。没有同时加Offset和SH的训练。',
        '- 两seed来自同一冻结地图、历史non-Bank反复用于研究，不是独立场景或在线SLAM验证。GPU同时有其他任务，耗时不用于性能结论。',
        '- 原生产默认与已有改动不替换；未stage、commit或push；未自动新增其他输入实验。','',
        '![质量曲线](analysis/quality_curves.png)','']
    (ROOT/'OFFSET_SHDC_INPUT_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')
    examples(scene)
    with tarfile.open(ROOT/'evidence.tar.gz','w:gz') as pack:
        for path in ROOT.rglob('*'):
            if path.is_file() and (path.suffix in ['.py','.json','.csv','.md','.png','.npy','.npz'] or path.name=='step_12000.pt'):
                pack.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
    print('PRIMARY',json.dumps(summary),flush=True);print('ARCHIVE_SHA256',rt.sha(ROOT/'evidence.tar.gz'),flush=True)


@torch.no_grad()
def examples(scene):
    view=scene.load_test()[len(scene.load_test())//2];nets={k:model(scene,k,3100) for k in KINDS}
    for ratio in [.01,.05]:
        fig,axes=plt.subplots(1,5,figsize=(19,4));axes[0].imshow(view['gt'].permute(1,2,0).cpu());axes[0].set_title('GT')
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
