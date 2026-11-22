"""仅全局XYZ输入的匹配预算质量对照；不追加其他输入实验。"""
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
from run import ROOT,SOURCE,PROTOCOL,Network,config,baseline,rt

METRICS=['psnr','ssim','lpips','photo']
KINDS=['baseline','xyz_only']
GROUPS=[('all',rt.RATIOS),('train',rt.TRAIN),('unseen',[r for r in rt.RATIOS if r not in rt.TRAIN])]


def read(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def directory(kind,seed):
    return (SOURCE/'results/fr1_xyz'/baseline.config('baseline')['id']/str(seed) if kind=='baseline'
        else ROOT/'results/fr1_xyz'/config()['id']/str(seed))


def make_model(scene,kind,seed):
    net=baseline.Network(scene,baseline.config('baseline'),seed) if kind=='baseline' else Network(scene,config(),seed)
    net.load_state_dict(torch.load(directory(kind,seed)/'step_12000.pt',map_location='cuda')['state_dict'])
    return net.eval()


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])


@torch.no_grad()
def selected_sets(scene,net):
    sets={}
    for ratio in rt.RATIOS:
        p,h,logits=net(ratio)
        margin=logits.double()-torch.logit(scene.uniform.double());margin[~scene.eligible]=-torch.inf
        sets[f'same_k_{ratio}']=torch.argsort(margin,descending=True,stable=True)[:scene.K[ratio]].sort().values.cpu().numpy()
        sets[f'cached_{ratio}']=torch.nonzero(p>scene.uniform).flatten().cpu().numpy()
    return sets


def main():
    assert (ROOT/'run.exit').read_text().strip()=='0'
    out=ROOT/'analysis';out.mkdir(exist_ok=True)
    scene=rt.Scene('fr1_xyz');train_paths={x['path'] for x in scene.train}
    quality,per_view,groups,paired,curves,fairness,overlap=[],[],[],[],[],[],[]
    values,sets={},{}
    for kind in KINDS:
        for seed in PROTOCOL['seeds']:
            d=directory(kind,seed);info=json.loads((d/'complete.json').read_text())
            assert info['steps']==12000 and info['master_frozen']
            assert all(v==3000 for v in info['ratio_exposures'].values())
            assert all(v==(3000 if k.startswith('extra_tables.') else 12000) for k,v in info['adam_clocks'].items())
            net=make_model(scene,kind,seed);sets[kind,seed]=selected_sets(scene,net);del net
            if kind=='baseline':
                dest=ROOT/'references'/kind/str(seed);dest.mkdir(parents=True,exist_ok=True)
                for name in ['complete.json','step_12000.pt','bank_full.csv','validation_full.csv','test_full.csv',
                    'training_curve.csv','validation_per_view.csv']:
                    shutil.copy2(d/name,dest/name)
            np.savez_compressed((ROOT/'references'/kind/str(seed) if kind=='baseline' else d)/'selected_sets.npz',**sets[kind,seed])
            fairness.append(dict(kind=kind,seed=seed,steps=12000,exposure_per_ratio=3000,parameters=info['parameters'],
                seconds=info['seconds'],adam_clocks=json.dumps(info['adam_clocks']),master_frozen=info['master_frozen'],
                reused=kind=='baseline'))
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
        for policy in ['same_k','cached']:
            for group,ratios in GROUPS+[(str(r),[r]) for r in rt.RATIOS]:
                for m in METRICS:
                    differences=[]
                    for seed in PROTOCOL['seeds']:
                        index=lambda k:{(r['path'],float(r['ratio'])):r for r in values[k,seed,split]
                            if r['policy']==policy and float(r['ratio']) in ratios}
                        a,b=index('xyz_only'),index('baseline');assert a.keys()==b.keys()
                        differences.append(np.array([float(a[k][m])-float(b[k][m]) for k in a]))
                    v=np.stack(differences);sign=1 if m in ['psnr','ssim'] else -1
                    paired.append(dict(split=split,policy=policy,group=group,metric=m,mean=float(v.mean()),
                        seed3100=float(v[0].mean()),seed3101=float(v[1].mean()),median=float(np.median(v)),
                        p10=float(np.quantile(v,.1)),p90=float(np.quantile(v,.9)),improved_fraction=float((sign*v>0).mean())))
    for seed in PROTOCOL['seeds']:
        for key in sets['baseline',seed]:
            a,b=sets['baseline',seed][key],sets['xyz_only',seed][key]
            inter=len(np.intersect1d(a,b));union=len(np.union1d(a,b))
            overlap.append(dict(seed=seed,set=key,baseline_k=len(a),candidate_k=len(b),intersection=inter,
                jaccard=inter/union if union else 1.,recall=inter/len(a) if len(a) else 1.))
    assert all(np.isfinite(float(r[m])) for r in per_view for m in METRICS)
    assert scene.unchanged()
    for name,rows in [('quality_per_view.csv',per_view),('quality_summary.csv',quality),('group_summary.csv',groups),
        ('paired_statistics.csv',paired),('validation_curves.csv',curves),('fairness_checks.csv',fairness),('selected_set_overlap.csv',overlap)]:
        rt.write_csv(out/name,rows)
    pick=lambda group='all',split='test':next(r for r in paired if r['group']==group and r['split']==split
        and r['policy']=='same_k' and r['metric']=='psnr')
    summary=dict(primary=[pick(g) for g,_ in GROUPS],bank_train=pick(split='bank_train'),validation=pick(split='validation'),
        same_k_equal=True,source_frozen=True,metrics_finite=True,new_models=2,reused_models=2,steps=12000,exposure_per_ratio=3000,
        production_change=False,online_slam=False)
    rt.save_json(out/'summary.json',summary)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,seed in zip(axes,PROTOCOL['seeds']):
        for kind in KINDS:
            sub=sorted([r for r in quality if r['kind']==kind and r['seed']==seed and r['split']=='test'
                and r['policy']=='same_k'],key=lambda r:r['ratio'])
            ax.plot([r['ratio']*100 for r in sub],[r['psnr'] for r in sub],marker='o',label=kind)
        ax.set(xlabel='Ratio (%)',ylabel='PSNR (dB)',title=f'fr1_xyz seed{seed}');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'quality_curves.png',dpi=150);plt.close(fig)
    lines=['# 仅全局XYZ的Encoder与四Head补偿输入消融','',
        'Gaussian属性输入仅保留原始全局XYZ；移除局部Offset、Scale、SH DC、Opacity及Rotation。保留logratio、每Anchor每ratio latent16、共享h32、四个属性Head及14维补偿输出。','',
        '## non-Bank same-K结果','',table(['ratio组','seed3100 ΔPSNR','seed3101 ΔPSNR','平均ΔPSNR'],
        [[g,f'{pick(g)["seed3100"]:+.3f}',f'{pick(g)["seed3101"]:+.3f}',f'{pick(g)["mean"]:+.3f}'] for g,_ in GROUPS]),'',
        table(['方案','PSNR↑','SSIM↑','LPIPS↓','photo↓'],[[kind,*[f'{np.mean([r[m] for r in groups if r["kind"]==kind and r["split"]=="test" and r["policy"]=="same_k" and r["group"]=="all"]):.5f}'
            for m in METRICS]] for kind in KINDS]),'',
        '## 各ratio','',table(['ratio','K','seed3100 ΔPSNR','seed3101 ΔPSNR','平均ΔPSNR'],
        [[f'{r*100:g}%',scene.K[r],f'{pick(str(r))["seed3100"]:+.3f}',f'{pick(str(r))["seed3101"]:+.3f}',f'{pick(str(r))["mean"]:+.3f}'] for r in rt.RATIOS]),'',
        '## Bank与留出','',table(['评价集合','平均ΔPSNR'],[[s,f'{pick(split=s)["mean"]:+.3f}'] for s in ['bank_train','validation','test']]),'',
        '## 实际结构','',
        '- Baseline：原始11维为Offset3+Scale3+SH DC3+Opacity1+logratio1，Encoder11→32；补偿输入raw11+latent16+h32=59。',
        '- XYZ-only：原始4维为全局XYZ3+logratio1，Encoder4→32；补偿输入raw4+latent16+h32=52。全局XYZ不另做AABB归一化或动态重拟合，输入tensor和统计均保存。',
        '- 选择Head仍32→32 ReLU→2；补偿四Head分别52→32 ReLU→32 ReLU→3/3/4/4。其余网络宽深、14D属性含义及变换完全保留，只对选中K行解码。',
        '- 16维特征是每Anchor每训练ratio一张表，只给补偿使用；未见ratio仍线性插值latent，logratio输入保留。共享h32现在仅由XYZ和logratio生成。',
        '- Renderer仍使用真实Gaussian的完整属性，这些属性从网络输入移除，并未从真实地图或渲染器删除。','',
        '## 初始化与输入独立性','',
        '- 复用原第一层前三列作为XYZ输入权重、原logratio列、latent/h32列、bias和所有后续层；删除其余输入列。坐标含义变化，隐藏函数不再相同，但step0概率与零补偿输出exact一致。',
        '- 用只暴露get_xyz的Gaussian代理替换输入对象，并将旧派生属性tensor设为NaN，Encoder和Compensation输出仍exact一致，证明两条路径不再隐式读取被移除属性。未修改真实Gaussian张量。',
        '- Encoder及四Head梯度、当前latent梯度与其他latent表grad=None通过检查；零输出初始化下隐藏层真实首步梯度可为零，另用非零投影探针验证路径。','',
        '## 公平性与局限','',
        '- 同一fr1_xyz冻结地图N6838/M2066，Master/关联/位姿/eligible/cached uniform固定；54 Bank训练、13内部留出、45历史non-Bank。网络仍联合优化Selector/Encoder/补偿/latent，选择集合可以变化。',
        '- 两seed各12000真实LoD步、每ratio3000；同帧序、ratio顺序、fresh fused Adam lr=.001 eps1e-15、原photo+.01预算损失。复用上一轮锁定的两个无Rotation Baseline，不选择最好复跑。',
        '- 最终12000步评价；训练1/5/10/15%，未见2.5/7.5/12.5%；same-K按logit(p)-logit(cached uniform)排名。实际Cached数量及四指标另存，未按non-Bank选择LR/步数/模型。',
        '- 本轮同时减少输入信息、将局部Offset换成全局XYZ、减少首层参数1120个，不能将所有质量差异归因为信息过多。只验证该精简组合，没有自动追加特征。',
        '- 两seed仍来自同一地图，历史评价轨迹已用于结构研究；不能宣称跨场景或在线Joint收益。GPU同时存在其他任务，不作无干扰性能结论。',
        '- 生产默认及已有改动保留，未stage、commit或push。','',
        '![质量曲线](analysis/quality_curves.png)','']
    (ROOT/'XYZ_ONLY_COMPENSATION_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')
    examples(scene)
    with tarfile.open(ROOT/'evidence.tar.gz','w:gz') as pack:
        for path in ROOT.rglob('*'):
            if path.is_file() and (path.suffix in ['.py','.json','.csv','.md','.png','.npy','.npz'] or path.name=='step_12000.pt'):
                pack.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
    print('PRIMARY',json.dumps(summary),flush=True)
    print('ARCHIVE_SHA256',rt.sha(ROOT/'evidence.tar.gz'),flush=True)


@torch.no_grad()
def examples(scene):
    view=scene.load_test()[len(scene.load_test())//2]
    nets={kind:make_model(scene,kind,3100) for kind in KINDS}
    for ratio in [.01,.05]:
        fig,axes=plt.subplots(1,3,figsize=(12,4));axes[0].imshow(view['gt'].permute(1,2,0).cpu());axes[0].set_title('GT')
        for ax,(kind,net) in zip(axes[1:],nets.items()):
            p,h,logits=net(ratio);margin=logits.double()-torch.logit(scene.uniform.double());margin[~scene.eligible]=-torch.inf
            ids=torch.argsort(margin,descending=True,stable=True)[:scene.K[ratio]].sort().values
            image,_,_=scene.render(ids,net.compensate(h,ids,ratio),view);image=image.clamp(0,1)
            psnr=float(-10*torch.log10((image-view['gt']).square().mean()))
            ax.imshow(image.permute(1,2,0).cpu());ax.set_title(f'{kind} / {psnr:.2f}dB / K={len(ids)}')
        for ax in axes:ax.axis('off')
        fig.suptitle(f'seed3100 / {ratio*100:g}% / fixed middle non-Bank view')
        fig.tight_layout();fig.savefig(ROOT/f'analysis/{int(ratio*100)}pct_examples.png',dpi=140);plt.close(fig)
    assert scene.unchanged()


if __name__=='__main__':
    try:main()
    except Exception:
        (ROOT/'analyze.exit').write_text('1');raise
    (ROOT/'analyze.exit').write_text('0')
