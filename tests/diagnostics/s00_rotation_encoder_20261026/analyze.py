"""汇总锁定两组Rotation输入对照，不根据评价选择模型。"""
import csv
import json
from pathlib import Path
import tarfile
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import ROOT, PREVIOUS, PROTOCOL, Network, config, rt

METRICS = ['psnr','ssim','lpips','photo']


def read(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def directory(kind, seed):
    return ROOT/'results/fr1_xyz'/PROTOCOL['configs'][kind]/str(seed)


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    assert (ROOT/'run.exit').read_text().strip() == '0'
    out = ROOT/'analysis'
    out.mkdir(exist_ok=True)
    per_view, quality, groups, paired, curves, fairness, overlap, repeat = [],[],[],[],[],[],[],[]
    values = {}
    ratio_groups = [('all',rt.RATIOS),('train',rt.TRAIN),('unseen',[r for r in rt.RATIOS if r not in rt.TRAIN])]
    scene=rt.Scene('fr1_xyz')
    train_paths=set(json.loads((ROOT/'results/fr1_xyz/scene_manifest.json').read_text())['train_paths'])
    for kind in PROTOCOL['configs']:
        for seed in PROTOCOL['seeds']:
            d = directory(kind,seed)
            info = json.loads((d/'complete.json').read_text())
            net=Network(scene,config(kind),seed)
            net.load_state_dict(torch.load(d/'step_12000.pt',map_location='cuda')['state_dict'])
            sets={}
            with torch.no_grad():
                for ratio in rt.RATIOS:
                    p,h,logits=net(ratio)
                    margin=logits.double()-torch.logit(scene.uniform.double())
                    margin[~scene.eligible]=-torch.inf
                    sets[f'same_k_{ratio}']=torch.argsort(margin,descending=True,stable=True)[:scene.K[ratio]].sort().values.cpu().numpy()
                    sets[f'cached_{ratio}']=torch.nonzero(p>scene.uniform).flatten().cpu().numpy()
            np.savez_compressed(d/'selected_sets.npz',**sets)
            del net
            assert info['steps'] == 12000 and info['master_frozen']
            assert all(v == 3000 for v in info['ratio_exposures'].values())
            assert all(v == (3000 if name.startswith('extra_tables.') else 12000)
                for name,v in info['adam_clocks'].items())
            fairness.append(dict(kind=kind,seed=seed,steps=info['steps'],exposures=json.dumps(info['ratio_exposures']),
                adam_clocks=json.dumps(info['adam_clocks']),parameters=info['parameters'],seconds=info['seconds'],
                empty_steps=sum(info['empty_steps'].values()),master_frozen=info['master_frozen']))
            curves.extend([dict(row,kind=kind,seed=seed) for row in read(d/'validation_per_view.csv')])
            for split in ['bank','bank_train','validation','test']:
                rows = read(d/(('bank' if split=='bank_train' else split)+'_full.csv'))
                if split=='bank_train':
                    rows=[dict(r,split=split) for r in rows if r['path'] in train_paths]
                values[kind,seed,split] = rows
                per_view.extend([dict(row,kind=kind,seed=seed) for row in rows])
                for policy in ['same_k','cached']:
                    for ratio in rt.RATIOS:
                        sub = [r for r in rows if r['policy']==policy and float(r['ratio'])==ratio]
                        counts = {int(r['selected']) for r in sub}
                        assert len(counts)==1
                        quality.append(dict(kind=kind,seed=seed,split=split,policy=policy,ratio=ratio,views=len(sub),
                            selected=counts.pop(),visible=np.mean([int(r['visible']) for r in sub]),
                            effective_proxy=np.mean([int(r['effective_proxy']) for r in sub]),
                            **{m:np.mean([float(r[m]) for r in sub]) for m in METRICS}))
                    for group,ratios in ratio_groups:
                        sub = [r for r in quality if r['kind']==kind and r['seed']==seed and r['split']==split
                            and r['policy']==policy and r['ratio'] in ratios]
                        groups.append(dict(kind=kind,seed=seed,split=split,policy=policy,group=group,
                            **{m:np.mean([r[m] for r in sub]) for m in METRICS}))
    assert all(np.isfinite(float(row[m])) for row in per_view for m in METRICS)
    for split in ['bank','bank_train','validation','test']:
        for ratio in rt.RATIOS:
            assert len({r['selected'] for r in quality if r['split']==split and r['policy']=='same_k' and r['ratio']==ratio})==1
        for policy in ['same_k','cached']:
            for group,ratios in ratio_groups+[(str(r),[r]) for r in rt.RATIOS]:
                for m in METRICS:
                    deltas=[]
                    for seed in PROTOCOL['seeds']:
                        index=lambda kind:{(float(r['ratio']),r['path']):r for r in values[kind,seed,split]
                            if r['policy']==policy and float(r['ratio']) in ratios}
                        a,b=index('rotation_all'),index('baseline')
                        assert a.keys()==b.keys()
                        deltas.append(np.array([float(a[k][m])-float(b[k][m]) for k in a]))
                    v=np.stack(deltas);sign=1 if m in ['psnr','ssim'] else -1
                    paired.append(dict(split=split,policy=policy,group=group,metric=m,mean=float(v.mean()),
                        seed3100=float(v[0].mean()),seed3101=float(v[1].mean()),median=float(np.median(v)),
                        p10=float(np.quantile(v,.1)),p90=float(np.quantile(v,.9)),
                        improved_fraction=float((sign*v>0).mean()),seed_improved=int((sign*v.mean(1)>0).sum())))
    for seed in PROTOCOL['seeds']:
        a=np.load(directory('baseline',seed)/'selected_sets.npz')
        b=np.load(directory('rotation_all',seed)/'selected_sets.npz')
        for key in a.files:
            x,y=a[key],b[key];inter=len(np.intersect1d(x,y));union=len(np.union1d(x,y))
            overlap.append(dict(seed=seed,set=key,baseline_k=len(x),candidate_k=len(y),intersection=inter,
                jaccard=inter/union if union else 1.,recall=inter/len(x) if len(x) else 1.))
        old = read(PREVIOUS/f'results/fr1_xyz/RawSharedE16Split4/{seed}/test_full.csv')
        fresh={(r['ratio'],r['policy'],r['path']):r for r in values['baseline',seed,'test']}
        historic={(r['ratio'],r['policy'],r['path']):r for r in old}
        assert fresh.keys()==historic.keys()
        for m in METRICS:
            repeat.append(dict(seed=seed,metric=m,mean=float(np.mean([float(fresh[k][m])-float(historic[k][m])
                for k in fresh if k[1]=='same_k'])),note='baseline fresh minus previous matched protocol, descriptive repeatability'))
    for name,rows in [('quality_per_view.csv',per_view),('quality_summary.csv',quality),('group_summary.csv',groups),
        ('paired_statistics.csv',paired),('training_dynamics.csv',curves),('fairness_checks.csv',fairness),
        ('selected_set_overlap.csv',overlap),('baseline_repeatability.csv',repeat)]:
        rt.write_csv(out/name,rows)
    pick=lambda group='all',split='test',metric='psnr':next(r for r in paired if r['policy']=='same_k'
        and r['group']==group and r['split']==split and r['metric']==metric)
    summary=dict(scene='fr1_xyz',seeds=PROTOCOL['seeds'],primary=[pick(g) for g,_ in ratio_groups],
        cached=[r for r in paired if r['policy']=='cached' and r['split']=='test' and r['metric']=='psnr'
            and r['group'] in ['all','train','unseen']],same_k_equal=True,finite=True,
        baseline_repeatability=repeat,steps=12000,exposure_per_ratio=3000,
        source_frozen=True,production_change=False,online_slam=False)
    rt.save_json(out/'summary.json',summary)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,seed in zip(axes,PROTOCOL['seeds']):
        for kind in PROTOCOL['configs']:
            sub=sorted([r for r in quality if r['kind']==kind and r['seed']==seed and r['split']=='test'
                and r['policy']=='same_k'],key=lambda r:r['ratio'])
            ax.plot([r['ratio']*100 for r in sub],[r['psnr'] for r in sub],marker='o',label=kind)
        ax.set(xlabel='Ratio (%)',ylabel='PSNR (dB)',title=f'fr1_xyz seed{seed}')
        ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'quality_curves.png',dpi=150);plt.close(fig)
    lines=['# Rotation同时输入共享Encoder与四个补偿Head的质量对照','',
        '本轮将59维四Head作为实验Baseline，重新训练两seed；候选仅增加基础Quaternion输入。生产默认和在线代码未替换。','',
        '## 主要结果','',table(['ratio组','seed3100 ΔPSNR','seed3101 ΔPSNR','平均ΔPSNR'],
        [[g,f'{pick(g)["seed3100"]:+.3f}',f'{pick(g)["seed3101"]:+.3f}',f'{pick(g)["mean"]:+.3f}'] for g,_ in ratio_groups]),'',
        '以上为候选减本轮重新训练Baseline，non-Bank、deterministic same-K。两个seed是同一冻结地图的网络初始化/训练顺序重复，不能当作独立场景或统计显著性。','',
        '## 两seed等权质量','',table(['方案','PSNR↑','SSIM↑','LPIPS↓','photo↓'],
        [[kind,*[f'{np.mean([r[m] for r in groups if r["kind"]==kind and r["split"]=="test" and r["policy"]=="same_k" and r["group"]=="all"]):.5f}'
            for m in METRICS]] for kind in PROTOCOL['configs']]),'',
        '## 逐ratio变化','',table(['ratio','seed3100 ΔPSNR','seed3101 ΔPSNR','平均ΔPSNR'],
        [[f'{ratio*100:g}%',f'{pick(str(ratio))["seed3100"]:+.3f}',f'{pick(str(ratio))["seed3101"]:+.3f}',f'{pick(str(ratio))["mean"]:+.3f}'] for ratio in rt.RATIOS]),'',
        '## 输入、结构与初始化','',
        '- 原始输入实际11维：Offset3、Scale3、SH DC3、Opacity1、logratio1。Rotation采用现有归一化Quaternion4，因此候选是15维，并非12维。q/-q按最大绝对分量为正统一符号。',
        '- Baseline Encoder11→32 ReLU，四Head各59→32 ReLU→32 ReLU→3/3/4/4；候选Encoder15→32，四Head全部63→32→32→3/3/4/4，Other也读取Rotation。',
        '- Head输入为原始属性/ratio、每Anchor每ratio latent16、共享h32；latent不参与Selector，未见ratio仍先线性插latent再解码。h不detach。',
        '- 所有公共权重同seed复制，新增Quaternion列零初始化；step0选择概率和补偿输出exact一致，隐藏GEMM浮点误差另存。实际photo的前三步验证新输入梯度，未把临时非零输出投影探针当成真实step1梯度。',
        '- Head参数12366→12878，Encoder增加128参数，总计增加640；仅增加输入连接，不改变隐藏层宽深、14D输出变换或Loss。',
        '- 上一轮Rotation只输入三个几何Head，Encoder和Other无Rotation；本轮是同时加入两条路径，不能从本对照进一步单独分离Encoder与直连的贡献。','',
        '## 公平性与范围','',
        '- 同源fr1_xyz冻结N6838/M2066地图，Master、关联、Gaussian身份、相机、eligible和cached uniform不变。54 Bank训练、13内部Bank留出、45历史non-Bank。',
        '- 两组都从初始零补偿训练12000真实LoD步，每ratio3000；fresh fused Adam lr=.001 eps1e-15，共同参数12000步、每latent表3000步；帧序和ratio顺序相同，第二seed反转两臂执行次序。',
        '- 原Cached Bernoulli、photo=.8L1+.2(1-fast_ssim)及.01预算项不变。Encoder与Selector参与训练，最终选中身份可变；same-K只固定预算，并非固定集合。',
        '- same-K按logit(p)-logit(cached uniform)排序；四指标、实际Cached点数及visible/effective_proxy完整保存。effective_proxy是可见且opacity≥1/255，并非积分渲染贡献。',
        '- 训练1/5/10/15%，评价另含2.5/7.5/12.5%；统一最终12000步，无non-Bank调参/选checkpoint。Bank/留出/non-Bank完整逐视图配对结果保存。',
        '- GPU同时存在其他任务，耗时仅为运行记录，不用作无干扰性能对比。历史non-Bank多次用于结构研究，不是全新盲测；没有在线SLAM、跨场景结论。','',
        '## 同协议Baseline重复波动','',table(['seed','本轮Baseline − 历史Baseline PSNR'],
        [[r['seed'],f'{r["mean"]:+.3f}'] for r in repeat if r['metric']=='psnr']),'',
        '复跑波动与本轮两seed差值一起解释，不把微小均值当稳定收益。所有旧结果保留，不挑最好Baseline。','',
        'Bank字段为全部67视图（包含13留出）；bank_train字段仅54个实际训练视图，并用相同640×480分辨率评价，validation为13个内部留出，test为45个历史non-Bank。','',
        '![质量曲线](analysis/quality_curves.png)','']
    (ROOT/'ROTATION_ENCODER_COMPENSATION_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')
    examples()
    with tarfile.open(ROOT/'evidence.tar.gz','w:gz') as pack:
        for path in ROOT.rglob('*'):
            if path.is_file() and (path.suffix in ['.py','.json','.csv','.md','.png','.npy','.npz'] or path.name=='step_12000.pt'):
                pack.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
    print('SUMMARY',json.dumps(summary),flush=True)
    print('ARCHIVE_SHA256',rt.sha(ROOT/'evidence.tar.gz'),flush=True)


@torch.no_grad()
def examples():
    scene=rt.Scene('fr1_xyz');views=scene.load_test();view=views[len(views)//2]
    nets={}
    for kind in PROTOCOL['configs']:
        net=Network(scene,config(kind),3100)
        payload=torch.load(directory(kind,3100)/'step_12000.pt',map_location='cuda')
        net.load_state_dict(payload['state_dict']);nets[kind]=net.eval()
    for ratio in [.01,.05]:
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        axes[0].imshow(view['gt'].permute(1,2,0).cpu());axes[0].set_title('GT')
        for ax,(kind,net) in zip(axes[1:],nets.items()):
            p,h,logits=net(ratio);margin=logits.double()-torch.logit(scene.uniform.double())
            margin[~scene.eligible]=-torch.inf
            ids=torch.argsort(margin,descending=True,stable=True)[:scene.K[ratio]].sort().values
            image,_,_=scene.render(ids,net.compensate(h,ids,ratio),view)
            image=image.clamp(0,1)
            psnr=float(-10*torch.log10((image-view['gt']).square().mean()))
            ax.imshow(image.permute(1,2,0).cpu());ax.set_title(f'{kind} / {psnr:.2f}dB / K={len(ids)}')
        for ax in axes:ax.axis('off')
        fig.suptitle(f'seed3100 / {ratio*100:g}% / fixed middle non-Bank view')
        fig.tight_layout();fig.savefig(ROOT/f'analysis/{int(ratio*100)}pct_examples.png',dpi=140);plt.close(fig)
    assert scene.unchanged()


if __name__=='__main__':
    try:
        main()
    except Exception:
        (ROOT/'analyze.exit').write_text('1')
        raise
    (ROOT/'analyze.exit').write_text('0')
