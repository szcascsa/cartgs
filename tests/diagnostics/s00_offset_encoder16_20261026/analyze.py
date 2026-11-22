"""局部Offset与指定Encoder的匹配步数质量汇总。"""
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
from run import ROOT,PREVIOUS,PROTOCOL,Network,config,previous,rt

KINDS=['baseline','xyz_scale_opacity','offset_encoder16']
METRICS=['psnr','ssim','lpips','photo']
GROUPS=[('all',rt.RATIOS),('train',rt.TRAIN),('unseen',[r for r in rt.RATIOS if r not in rt.TRAIN])]


def read(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def directory(kind,seed):
    if kind=='baseline':return previous.xyz.SOURCE/'results/fr1_xyz'/previous.xyz.baseline.config('baseline')['id']/str(seed)
    if kind=='xyz_scale_opacity':return PREVIOUS/'results/fr1_xyz'/previous.config()['id']/str(seed)
    return ROOT/'results/fr1_xyz'/config()['id']/str(seed)


def model(scene,kind,seed):
    if kind=='baseline':net=previous.xyz.baseline.Network(scene,previous.xyz.baseline.config('baseline'),seed)
    elif kind=='xyz_scale_opacity':net=previous.Network(scene,previous.config(),seed)
    else:net=Network(scene,config(),seed)
    net.load_state_dict(torch.load(directory(kind,seed)/'step_12000.pt',map_location='cuda')['state_dict'])
    return net.eval()


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    assert (ROOT/'run.exit').read_text().strip()=='0'
    out=ROOT/'analysis';out.mkdir(exist_ok=True)
    scene=rt.Scene('fr1_xyz');train_paths={v['path'] for v in scene.train}
    per_view,quality,groups,paired,fairness,curves=[],[],[],[],[],[]
    values={}
    for kind in KINDS:
        for seed in PROTOCOL['seeds']:
            d=directory(kind,seed);info=json.loads((d/'complete.json').read_text())
            assert info['steps']==12000 and info['master_frozen']
            assert all(v==3000 for v in info['ratio_exposures'].values())
            assert all(v==(3000 if k.startswith('extra_tables.') else 12000) for k,v in info['adam_clocks'].items())
            if kind!='offset_encoder16':
                dest=ROOT/'references'/kind/str(seed);dest.mkdir(parents=True,exist_ok=True)
                for name in ['complete.json','step_12000.pt','bank_full.csv','validation_full.csv','test_full.csv',
                    'training_curve.csv','validation_per_view.csv']:shutil.copy2(d/name,dest/name)
            fairness.append(dict(kind=kind,seed=seed,steps=12000,exposure_per_ratio=3000,parameters=info['parameters'],
                seconds=info['seconds'],adam_clocks=json.dumps(info['adam_clocks']),master_frozen=True,reused=kind!='offset_encoder16'))
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
                    for m in METRICS:
                        diffs=[]
                        for seed in PROTOCOL['seeds']:
                            index=lambda kind:{(r['path'],float(r['ratio'])):r for r in values[kind,seed,split]
                                if r['policy']==policy and float(r['ratio']) in ratios}
                            a,b=index('offset_encoder16'),index(reference);assert a.keys()==b.keys()
                            diffs.append(np.array([float(a[k][m])-float(b[k][m]) for k in a]))
                        v=np.stack(diffs);sign=1 if m in ['psnr','ssim'] else -1
                        paired.append(dict(reference=reference,split=split,policy=policy,group=group,metric=m,
                            mean=float(v.mean()),seed3100=float(v[0].mean()),seed3101=float(v[1].mean()),
                            median=float(np.median(v)),p10=float(np.quantile(v,.1)),p90=float(np.quantile(v,.9)),
                            improved_fraction=float((sign*v>0).mean())))
    assert scene.unchanged()
    assert all(np.isfinite(float(r[m])) for r in per_view for m in METRICS)
    for name,rows in [('quality_per_view.csv',per_view),('quality_summary.csv',quality),('group_summary.csv',groups),
        ('paired_statistics.csv',paired),('fairness_checks.csv',fairness),('validation_curves.csv',curves)]:rt.write_csv(out/name,rows)
    pick=lambda ref,group='all',policy='same_k',split='test':next(r for r in paired if r['reference']==ref and r['group']==group
        and r['policy']==policy and r['split']==split and r['metric']=='psnr')
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
    lines=['# 加回局部Offset与16维瓶颈共享Encoder的质量对照','',
        '本轮按用户指定同时加入局部Offset3，并将Encoder改为Linear11→16、ReLU、Linear16→32、Linear32→32、ReLU。最终输出32维共享编码，不另加输出层或中间ReLU。','',
        '## non-Bank same-K结果','',table(['参照','ratio组','seed3100 ΔPSNR','seed3101 ΔPSNR','平均ΔPSNR'],
        [[ref,g,f'{pick(ref,g)["seed3100"]:+.3f}',f'{pick(ref,g)["seed3101"]:+.3f}',f'{pick(ref,g)["mean"]:+.3f}']
            for ref in KINDS[:2] for g,_ in GROUPS]),'',
        table(['方案','PSNR↑','SSIM↑','LPIPS↓','photo↓'],[[kind,*[f'{np.mean([r[m] for r in groups if r["kind"]==kind and r["split"]=="test" and r["policy"]=="same_k" and r["group"]=="all"]):.5f}'
            for m in METRICS]] for kind in KINDS]),'',
        '## 各ratio','',table(['ratio','K','相对上一组','相对原Baseline'],
        [[f'{r*100:g}%',scene.K[r],f'{pick("xyz_scale_opacity",str(r))["mean"]:+.3f}',f'{pick("baseline",str(r))["mean"]:+.3f}'] for r in rt.RATIOS]),'',
        '## Bank、留出与原Cached评价','',table(['参照','Bank训练视图ΔPSNR','内部留出ΔPSNR','non-Bank实际Cached ΔPSNR'],
        [[ref,f'{pick(ref,split="bank_train")["mean"]:+.3f}',f'{pick(ref,split="validation")["mean"]:+.3f}',
            f'{pick(ref,policy="cached")["mean"]:+.3f}'] for ref in KINDS[:2]]),'',
        '## 结构与初始化','',
        '- raw11=[全局XYZ3,activated Scale3,sigmoid Opacity1,局部Offset3,logratio1]。Offset=(XYZ−所属Anchor中心)/原冻结参考尺度；与旧scene.attrs前三维exact一致。没有SH DC或Rotation输入。',
        '- Encoder=11→16→ReLU→32→32→ReLU；16→32与32→32之间没有激活。两个连续Linear函数上可合并为16→32仿射映射，分解后参数优化不同，不将其解释为多一次非线性表达能力。',
        '- 补偿输入raw11+Anchor/ratio latent16+h32共59维，四Head仍各59→32 ReLU→32 ReLU→3/3/4/4；原14维属性变换、latent插值、Selector32→32→2均保留。只对选中K行补偿。',
        '- 新Encoder首层取旧首层前16行公共权重/偏置，新增Offset列零；后两层用固定seed+110/+120初始化。补偿头公共列和后续层复制、新Offset列零。因16维瓶颈，隐藏函数不与原32维浅层一致；step0概率和零补偿exact一致。',
        '- Encoder参数1792，较上一组8→32的288增加1504；四Head参数12366，增加384。总参数147504，较上一组增加1888。',
        '- 输入代理仅开放XYZ/Scale/Opacity，将旧派生属性设NaN后输出exact不变；Offset由几何和所属Anchor重新计算。所有Encoder层及Offset连接梯度有效，其他ratio表grad=None。','',
        '## 协议和限制','',
        '- 同源fr1_xyz N6838/M2066冻结地图，Master/拓扑/位姿/关联/eligible/uniform固定；Selector/Encoder/补偿和当前ratio latent联合训练，最终集合可以不同。',
        '- 两seed各12000步、每ratio3000，复用匹配的上一组和原Baseline最终模型，不挑最好重复；同Bank帧序、ratio顺序、fresh fused Adam lr=.001 eps1e-15、原photo+.01预算项。',
        '- 54 Bank训练320×240，54 Bank训练/13内部留出/45历史non-Bank评价640×480；训练1/5/10/15%，另评2.5/7.5/12.5%。统一最终12000步，不根据评价调LR或选checkpoint。',
        '- 主same-K按logit(p)-logit(cached uniform)排序；原Cached实际点数和四指标另存。effective_proxy是可见且opacity≥1/255，并非积分贡献。',
        '- 本轮同时加入Offset并改变Encoder结构/初始化/参数量，不能单独归因于Offset、深度或16维瓶颈；需要时另做单变量对照，但本轮不自动扩展。',
        '- 原Baseline虽然也有raw11，却是Offset3+Scale3+SH DC3+Opacity1+logratio1和11→32浅Encoder；同维数不代表同信息。',
        '- 两seed为同一历史地图上的初始化/帧序重复，不是独立SLAM地图或新盲测。GPU同时存在其他任务，不作无干扰性能结论；生产源码/默认未替换，未stage/commit/push。','',
        '![质量曲线](analysis/quality_curves.png)','']
    (ROOT/'OFFSET_ENCODER16_COMPENSATION_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')
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
