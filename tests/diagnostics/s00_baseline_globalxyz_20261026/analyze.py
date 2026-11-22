"""原Baseline添加全局XYZ的配对质量汇总。"""
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

KINDS=['original_baseline','baseline_xyz']
METRICS=['psnr','ssim','lpips','photo']
GROUPS=[('all',rt.RATIOS),('train',rt.TRAIN),('unseen',[r for r in rt.RATIOS if r not in rt.TRAIN])]
LEGACY=source


def read(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def directory(kind,seed):
    if kind=='original_baseline':return SOURCE/'results/fr1_xyz'/source.config('baseline')['id']/str(seed)
    return ROOT/'results/fr1_xyz'/config()['id']/str(seed)


def model(scene,kind,seed):
    net=source.Network(scene,source.config('baseline'),seed) if kind=='original_baseline' else Network(scene,config(),seed)
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
            if kind=='baseline_xyz':assert info['config']['encoder_layout']=='14-32-ReLU'
            dest=d
            if kind!='baseline_xyz':
                dest=ROOT/'references'/kind/str(seed);dest.mkdir(parents=True,exist_ok=True)
                for name in ['complete.json','step_12000.pt','bank_full.csv','validation_full.csv','test_full.csv',
                    'training_curve.csv','validation_per_view.csv']:shutil.copy2(d/name,dest/name)
            net=model(scene,kind,seed);sets[kind,seed]=sets_for(scene,net);del net
            np.savez_compressed(dest/'selected_sets.npz',**sets[kind,seed])
            fairness.append(dict(kind=kind,seed=seed,steps=12000,exposure_per_ratio=3000,parameters=info['parameters'],
                seconds=info['seconds'],adam_clocks=json.dumps(info['adam_clocks']),empty_steps=json.dumps(info['empty_steps']),
                compensation_updates=12000-empty_total,master_frozen=True,reused=kind!='baseline_xyz'))
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
        for reference in KINDS[:-1]:
            for policy in ['same_k','cached']:
                for group,ratios in GROUPS+[(str(r),[r]) for r in rt.RATIOS]:
                    for metric in METRICS:
                        diffs=[]
                        for seed in PROTOCOL['seeds']:
                            index=lambda kind:{(r['path'],float(r['ratio'])):r for r in values[kind,seed,split]
                                if r['policy']==policy and float(r['ratio']) in ratios}
                            a,b=index('baseline_xyz'),index(reference);assert a.keys()==b.keys()
                            diffs.append(np.array([float(a[k][metric])-float(b[k][metric]) for k in a]))
                        v=np.stack(diffs);sign=1 if metric in ['psnr','ssim'] else -1
                        paired.append(dict(reference=reference,split=split,policy=policy,group=group,metric=metric,
                            mean=float(v.mean()),seed3100=float(v[0].mean()),seed3101=float(v[1].mean()),
                            median=float(np.median(v)),p10=float(np.quantile(v,.1)),p90=float(np.quantile(v,.9)),
                            improved_fraction=float((sign*v>0).mean()),seed_improved=int((sign*v.mean(1)>0).sum())))
    for reference in KINDS[:-1]:
        for seed in PROTOCOL['seeds']:
            for key,a in sets[reference,seed].items():
                b=sets['baseline_xyz',seed][key];inter=len(np.intersect1d(a,b));union=len(np.union1d(a,b))
                overlap.append(dict(reference=reference,seed=seed,set=key,reference_k=len(a),candidate_k=len(b),
                    intersection=inter,jaccard=inter/union if union else 1.,recall=inter/len(a) if len(a) else 1.))
    assert scene.unchanged()
    assert all(np.isfinite(float(r[m])) for r in per_view for m in METRICS)
    for name,rows in [('quality_per_view.csv',per_view),('quality_summary.csv',quality),('group_summary.csv',groups),
        ('paired_statistics.csv',paired),('fairness_checks.csv',fairness),('validation_curves.csv',curves),('selected_set_overlap.csv',overlap)]:rt.write_csv(out/name,rows)
    pick=lambda ref,g='all',policy='same_k',split='test':next(r for r in paired if r['reference']==ref
        and r['group']==g and r['policy']==policy and r['split']==split and r['metric']=='psnr')
    summary=dict(primary=[pick(ref,g) for ref in KINDS[:-1] for g,_ in GROUPS],
        cached=[pick(ref,g,'cached') for ref in KINDS[:-1] for g,_ in GROUPS],same_k_equal=True,source_frozen=True,
        metrics_finite=True,new_models=2,reused_models=2,steps=12000,exposure_per_ratio=3000,production_change=False,online_slam=False)
    rt.save_json(out/'summary.json',summary)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,seed in zip(axes,PROTOCOL['seeds']):
        for kind in KINDS:
            sub=sorted([r for r in quality if r['kind']==kind and r['seed']==seed and r['split']=='test'
                and r['policy']=='same_k'],key=lambda r:r['ratio'])
            ax.plot([r['ratio']*100 for r in sub],[r['psnr'] for r in sub],marker='o',label=kind)
        ax.set(xlabel='Ratio (%)',ylabel='PSNR (dB)',title=f'fr1_xyz seed{seed}');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'quality_curves.png',dpi=150);plt.close(fig)
    primary=pick('original_baseline')
    lines=['# 原Baseline新增全局XYZ受控实验','',
        '## 结论','',
        f'两seed七档non-Bank same-K平均变化{primary["mean"]:+.4f} dB；seed3100/3101分别为{primary["seed3100"]:+.4f}/{primary["seed3101"]:+.4f} dB。',
        '两seed是同一冻结地图上的初始化与训练顺序重复，不是两个独立SLAM场景。此次不依据non-Bank结果追加调参或替换生产配置。','',
        '## 结构与严格初始化对齐','',
        '| 项目 | 原Baseline | Baseline+全局XYZ |','| --- | --- | --- |',
        '| 原始输入 | Offset3 + Scale3 + SH DC3 + Opacity1 + log(ratio) | 原11维全部保留，再追加世界XYZ3 |',
        '| Encoder | 11→32→ReLU | 14→32→ReLU |',
        '| 补偿输入 | 原始11 + 每Anchor/ratio latent16 + h32 = 59 | 原始14 + latent16 + h32 = 62 |',
        '| 四Head | 各59→32→ReLU→32→ReLU→输出 | 各62→32→ReLU→32→ReLU→输出 |',
        '| 输出 | XYZ3 / Scale3 / Rotation4 / Opacity1+SH DC3 | 完全相同 |',
        '| 参数总量 | 146096 | 146576，增加480 |','',
        '与上一组沿逐步加回链路构建的浅DC不同：本次所有原属性列、bias、latent/h32投影和后续层直接复制原Baseline同seed初始化，只有新增XYZ列置零。step0原11维输入和旧权重exact相同；概率和零补偿exact一致，隐藏函数及临时非零输出投影做浮点容忍检查。真实photo反传检查新增连接随后获得有效梯度。','',
        '世界XYZ直接采用Master真实坐标，不额外归一化；局部Offset保留，Scale为activated scale，Opacity为sigmoid opacity，DC为原SH系数。Encoder和四个Head都添加XYZ，无法把影响单独归因到其中一条路径。','',
        '## 质量与实际预算','',
        table(['方案','PSNR↑','SSIM↑','LPIPS↓','photo↓'],[[kind,*[f'{np.mean([r[m] for r in groups if r["kind"]==kind and r["split"]=="test" and r["policy"]=="same_k" and r["group"]=="all"]):.5f}'
            for m in METRICS]] for kind in KINDS]),'',
        table(['ratio','same-K','seed3100 ΔPSNR','seed3101 ΔPSNR','均值ΔPSNR'],
            [[f'{r*100:g}%',scene.K[r],*[f'{pick("original_baseline",str(r))[key]:+.4f}' for key in ['seed3100','seed3101','mean']]] for r in rt.RATIOS]),'',
        table(['ratio组','seed3100 ΔPSNR','seed3101 ΔPSNR','均值ΔPSNR'],
            [[group,*[f'{pick("original_baseline",group)[key]:+.4f}' for key in ['seed3100','seed3101','mean']]] for group,_ in GROUPS]),'',
        table(['评价','same-K ΔPSNR','实际Cached ΔPSNR'],
            [[split,f'{pick("original_baseline",split=split)["mean"]:+.4f}',f'{pick("original_baseline",split=split,policy="cached")["mean"]:+.4f}'] for split in ['bank_train','validation','test']]),'',
        f'non-Bank逐视图差值median/p10/p90：{primary["median"]:+.4f}/{primary["p10"]:+.4f}/{primary["p90"]:+.4f} dB；改善比例{primary["improved_fraction"]:.1%}。完整四指标、visible与effective_proxy数量、cached实际K见CSV；effective_proxy仅为半径>0且补偿后opacity≥1/255的代理，并非逐点真实像素贡献。','',
        '## 实验边界与公平性','',
        '- 同源fr1_xyz冻结N=6838/M=2066，Master/拓扑/ID/Anchor归属、中心与尺度/位姿/eligible/cached uniform冻结。scene_manifest与原Baseline逐项比较。',
        '- 54个Bank训练视图320×240，13个内部Bank留出和45个历史non-Bank视图640×480；评价的Bank训练指标只取54个实际训练视图。',
        '- 新增模型两seed各12000真实LoD步，每ratio3000次曝光，复用同seed原Baseline最终12000步锁定参照。fresh fused Adam lr=.001 eps=1e-15，原.8L1+.2(1-fast_ssim)+.01预算Loss，无新增Loss或正则。',
        '- 训练1/5/10/15%，未见2.5/7.5/12.5%使用原raw-ratio latent线性插值并输入真实logratio；不重新拟合。',
        '- 同seed Bank顺序与ratio轮转相同，全部正式结果取预定最终12000步；4k/8k仅作收敛诊断。没有按non-Bank选择checkpoint、学习率或训练步数。',
        '- Cached Bernoulli训练；same-K按logit(p)-logit(cached uniform)排序而非纯概率。原Cached实际数量另外评价。',
        '- Encoder/Selector/Compensation及当前ratio补偿latent参与训练，选中身份可变，不是固定选择集合的纯Compensation因果。',
        '- Adam实际时钟按空选择次数核验，未伪造零梯度step；两seed one-map有限预算，不能认定表达能力上界或跨场景最优。历史non-Bank多次用于研究，不是全新盲测。',
        '- 服务器存在其他GPU任务，耗时只记录，不作无干扰速度比较。本次是冻结地图实验，不是在线SLAM质量证据。',
        '- 生产网络/默认配置保持原状；本轮不增加其他输入、网络深度或维度，不stage、commit或push。','',
        '## 原始证据','',
        '[逐视图四指标](analysis/quality_per_view.csv) · [配对统计](analysis/paired_statistics.csv) · [预算与实际Adam时钟](analysis/fairness_checks.csv) · [选择重合率](analysis/selected_set_overlap.csv)','',
        '![质量曲线](analysis/quality_curves.png)','']
    (ROOT/'BASELINE_GLOBAL_XYZ_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')
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
        fig,axes=plt.subplots(1,3,figsize=(12,4));axes[0].imshow(view['gt'].permute(1,2,0).cpu());axes[0].set_title('GT')
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
