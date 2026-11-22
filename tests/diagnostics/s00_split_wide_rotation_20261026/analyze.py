"""两阶段锁定结果：完整逐视图指标、配对seed和历史重复性核对。"""
import csv
import itertools
import json
from pathlib import Path
import shutil
import sys
import tarfile
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import ROOT, OLD, EXTRA, SPLIT, PROTOCOL, Network, config, rt

METRICS = ['psnr', 'ssim', 'lpips', 'photo']
SEEDS = PROTOCOL['seeds']
CONFIGS = PROTOCOL['configs']


def read(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def directory(kind, seed):
    if kind == 'unified32' and seed == 3100:
        return EXTRA / 'results/fr1_xyz/RawSharedE16/3100'
    return ROOT / 'results/fr1_xyz' / CONFIGS[kind] / str(seed)


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] +
        ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def repeatability():
    current = directory('split4', 3100)
    original = SPLIT / 'results/fr1_xyz/RawSharedE16Split4/3100'
    a = torch.load(current/'step_12000.pt', map_location='cpu')['state_dict']
    b = torch.load(original/'step_12000.pt', map_location='cpu')['state_dict']
    assert a.keys() == b.keys()
    max_error = max(float((a[k]-b[k]).abs().max()) for k in a)
    x = {(r['ratio'],r['policy'],r['path']):r for r in read(current/'test_full.csv')}
    y = {(r['ratio'],r['policy'],r['path']):r for r in read(original/'test_full.csv')}
    assert x.keys() == y.keys()
    result = dict(state_dict_exact=all(torch.equal(a[k],b[k]) for k in a), maximum_parameter_error=max_error,
        maximum_metric_error={m:max(abs(float(x[k][m])-float(y[k][m])) for k in x) for m in METRICS},
        same_k_mean_metric_change={m:float(np.mean([float(x[k][m])-float(y[k][m]) for k in x if k[1]=='same_k'])) for m in METRICS},
        same_seed=3100, note='repeatability is not an independent optimization-order seed')
    rt.save_json(ROOT/'repeatability.json', result)
    return result


def main():
    phase = sys.argv[1]
    assert phase in ['A', 'B']
    assert (ROOT/f'phase_{phase}.exit').read_text().strip() == '0'
    kinds = ['unified32', 'unified80', 'split4'] + (['split4_rotation'] if phase=='B' else [])
    out = ROOT / ('analysis_'+phase)
    out.mkdir(exist_ok=True)
    per_view, quality, groups, paired, fairness, curves, seeds = [], [], [], [], [], [], []
    comparisons = [('split4','unified32'), ('unified80','unified32'), ('split4','unified80')]
    if phase=='B':
        comparisons.append(('split4_rotation','split4'))
    tables = {}
    for kind in kinds:
        for seed in SEEDS:
            d = directory(kind, seed)
            info = json.loads((d/'complete.json').read_text())
            assert info['steps']==12000 and info['master_frozen']
            assert all(v==3000 for v in info['ratio_exposures'].values())
            assert all(v==3000 for k,v in info['adam_clocks'].items() if k.startswith('extra_tables.'))
            assert all(v==12000 for k,v in info['adam_clocks'].items() if not k.startswith('extra_tables.'))
            fairness.append(dict(kind=kind,seed=seed,steps=12000,exposure_per_ratio=3000,
                parameters=info['parameters'],seconds=info['seconds'],empty_steps=sum(info['empty_steps'].values()),
                reused=kind=='unified32' and seed==3100,adam_clocks=json.dumps(info['adam_clocks'])))
            if kind=='unified32' and seed==3100:
                dest=ROOT/'references'/CONFIGS[kind]/str(seed)
                dest.mkdir(parents=True,exist_ok=True)
                for name in ['complete.json','validation_full.csv','test_full.csv','training_curve.csv','validation_per_view.csv']:
                    shutil.copy2(d/name,dest/name)
            for row in read(d/'validation_per_view.csv'):
                curves.append(dict(kind=kind,seed=seed,**row))
            for split in ['validation','test']:
                rows=read(d/(split+'_full.csv'))
                tables[kind,seed,split]=rows
                for row in rows:
                    per_view.append(dict(row,kind=kind,seed=seed,split=split))
                for policy in ['same_k','cached']:
                    for ratio in rt.RATIOS:
                        subset=[r for r in rows if r['policy']==policy and float(r['ratio'])==ratio]
                        assert subset
                        counts={int(r['selected']) for r in subset}
                        assert len(counts)==1
                        quality.append(dict(kind=kind,seed=seed,split=split,policy=policy,ratio=ratio,
                            views=len(subset),selected=counts.pop(),
                            effective_proxy=float(np.mean([float(r['effective_proxy']) for r in subset])),
                            **{m:float(np.mean([float(r[m]) for r in subset])) for m in METRICS}))
                    for group, ratios in [('all',rt.RATIOS),('train',rt.TRAIN),('unseen',[r for r in rt.RATIOS if r not in rt.TRAIN])]:
                        subset=[r for r in quality if r['kind']==kind and r['seed']==seed and r['split']==split and r['policy']==policy and r['ratio'] in ratios]
                        groups.append(dict(kind=kind,seed=seed,split=split,policy=policy,group=group,
                            **{m:float(np.mean([r[m] for r in subset])) for m in METRICS}))
    for split in ['validation','test']:
        for ratio in rt.RATIOS:
            assert len({r['selected'] for r in quality if r['split']==split and r['policy']=='same_k' and r['ratio']==ratio})==1
        for policy in ['same_k','cached']:
            for a,b in comparisons:
                for group, ratios in [('all',rt.RATIOS),('train',rt.TRAIN),('unseen',[r for r in rt.RATIOS if r not in rt.TRAIN])]+[(str(r),[r]) for r in rt.RATIOS]:
                    deltas={m:[] for m in METRICS}
                    for seed in SEEDS:
                        x={(float(r['ratio']),r['path']):r for r in tables[a,seed,split] if r['policy']==policy and float(r['ratio']) in ratios}
                        y={(float(r['ratio']),r['path']):r for r in tables[b,seed,split] if r['policy']==policy and float(r['ratio']) in ratios}
                        assert x.keys()==y.keys()
                        for m in METRICS:
                            delta=np.array([float(x[k][m])-float(y[k][m]) for k in x])
                            deltas[m].append(delta)
                            seeds.append(dict(candidate=a,reference=b,seed=seed,split=split,policy=policy,group=group,
                                metric=m,mean=float(delta.mean())))
                    for m in METRICS:
                        values=np.stack(deltas[m])
                        sign=1 if m in ['psnr','ssim'] else -1
                        paired.append(dict(candidate=a,reference=b,split=split,policy=policy,group=group,metric=m,
                            mean=float(values.mean()),seed3100=float(values[0].mean()),seed3101=float(values[1].mean()),
                            seed_improved=int((sign*values.mean(1)>0).sum()),median=float(np.median(values)),
                            p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9)),
                            improved_fraction=float((sign*values>0).mean()),
                            note='two paired optimization seeds, descriptive only, no significance claim'))
    assert all(np.isfinite(float(r[m])) for r in per_view for m in METRICS)
    for name,rows in [('quality_per_view.csv',per_view),('quality_summary.csv',quality),('group_summary.csv',groups),
            ('paired_statistics.csv',paired),('paired_seed_statistics.csv',seeds),('fairness_checks.csv',fairness),('validation_curves.csv',curves)]:
        rt.write_csv(out/name,rows)
    repeat=repeatability()
    primary=[r for r in paired if r['split']=='test' and r['policy']=='same_k' and r['metric']=='psnr' and r['group'] in ['all','train','unseen']]
    summary=dict(phase=phase,scene='fr1_xyz',seeds=SEEDS,comparisons=primary,repeatability=repeat,
        metrics_finite=True,new_models=len(fairness)-1,reused_models=1,steps=12000,exposure_per_ratio=3000)
    rt.save_json(out/'summary.json',summary)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,seed in zip(axes,SEEDS):
        for kind in kinds:
            rows=sorted([r for r in quality if r['kind']==kind and r['seed']==seed and r['split']=='test' and r['policy']=='same_k'],key=lambda r:r['ratio'])
            ax.plot([r['ratio']*100 for r in rows],[r['psnr'] for r in rows],marker='o',label=kind)
        ax.set(xlabel='Ratio (%)',ylabel='PSNR (dB)',title=f'fr1_xyz / seed{seed} / non-Bank same-K')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'quality_curves.png',dpi=150)
    plt.close(fig)
    lines=['# 四Head复现、统一80宽头与基础Rotation输入对照','',
        '本报告阶段 '+phase+'；相同fr1_xyz冻结地图、两个网络初始化与Bank帧序seed3100/3101、统一12000步和每ratio3000曝光。以下PSNR为non-Bank exact same-K。','',
        table(['对照','ratio组','seed3100 ΔPSNR','seed3101 ΔPSNR','平均ΔPSNR'],
            [[r['candidate']+' − '+r['reference'],r['group'],f'{r["seed3100"]:+.3f}',f'{r["seed3101"]:+.3f}',f'{r["mean"]:+.3f}'] for r in primary]),'',
        '## 各方案绝对质量','',
        table(['方案','seed','七档PSNR','SSIM','LPIPS','photo'],
            [[g['kind'],g['seed'],f'{g["psnr"]:.4f}',f'{g["ssim"]:.5f}',f'{g["lpips"]:.5f}',f'{g["photo"]:.5f}']
             for g in groups if g['split']=='test' and g['policy']=='same_k' and g['group']=='all']),'',
        '![各seed曲线]('+str(out.name)+'/quality_curves.png)','',
        '## 结构、初始化与公平性','',
        '- 输入为Anchor/ratio latent16、原始属性11、共享Encoder输出h32，共59维。原Selector保持原11→32→32→2，h不detach。',
        '- unified32：59→32→32→14，3438个Head参数；unified80：59→80→80→14，12414参数；split4：四个59→32→32→3/3/4/4，12366参数。宽头与四Head差48参数(约0.39%)。',
        '- 宽头保留原前32隐藏通道函数，其余通道固定seed初始化；输出新增列零初始化，第二层前32行对新增通道权重零。零输出起点一致，但参数化和优化几何仍不同，不声称完全隔离全部容量因素。',
        '- 四Head复制原两层隐藏权重四份、按XYZ/Scale/Rotation/Other切分输出行。其同seed重复训练独立保存，不把重复运行当第三个seed。',
        '- split4_rotation：仅XYZ/Scale/Rotation头追加基础归一化四元数4维，输入63；Other仍59。按最大绝对分量为正统一q/-q符号；新增列初始为零，保持原函数，训练中可收到梯度。Head参数12750，较split4多384。Selector不读取Rotation。',
        '- Rotation只改变补偿输入，不改变输出14D语义：XYZ加法、Scale乘exp、增量四元数归一化与基础旋转复合、Opacity在logit域相加、SH DC加法。',
        '- 相同Master/IDs/Anchor关联中心尺度/相机/eligible/cached uniform冻结，N6838/M2066；54 Bank训练、13内部留出、45历史non-Bank。原photo+.01预算项、Adam fused lr=.001 eps1e-15、帧序与ratio轮转不变。',
        '- Selector与Encoder共同训练，选中集合可变化。same-K严格同预算，原Cached实际点数另存；不是冻结Selection的纯Compensation因果实验。',
        '- 训练ratio1/5/10/15%，未见2.5/7.5/12.5%；先raw-ratio线性插值latent再解码，不为未见ratio拟合参数；最终12000步评价，不回选4k/8k。',
        '- 两个seed同时控制网络初始化与Bank顺序，不是两个独立SLAM地图，不以逐视图数量代替独立重复，不报告不可靠的显著性；历史non-Bank轨迹不是新盲测。',
        '- 生产算法、Loss、默认E50与在线二进制未修改，无新在线SLAM或CUDA Mapping构建；未stage/commit/push。','',
        '## 重复性','',json.dumps(repeat,ensure_ascii=False),'']
    (ROOT/('SPLIT_WIDE_ROTATION_PHASE_'+phase+'.md')).write_text('\n'.join(lines),encoding='utf-8')
    print('ANALYSIS_READY',json.dumps(primary),flush=True)
    if phase=='B':
        examples(out)
        from rotation_diagnostics import main as rotation_diagnostics
        rotation_diagnostics()
        with tarfile.open(ROOT/'evidence.tar.gz','w:gz') as pack:
            for path in ROOT.rglob('*'):
                if path.is_file() and (path.suffix in ['.py','.json','.csv','.md','.png','.npy','.log'] or path.name=='step_12000.pt'):
                    pack.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)


@torch.no_grad()
def examples(out):
    scene=rt.Scene('fr1_xyz')
    views=scene.load_test()
    view=views[len(views)//2]
    kinds=['unified32','unified80','split4','split4_rotation']
    nets={}
    for kind in kinds:
        seed=3100
        info=json.loads((directory(kind,seed)/'complete.json').read_text())
        net=Network(scene,info['config'],seed)
        net.load_state_dict(torch.load(info['checkpoint'],map_location='cuda')['state_dict'])
        nets[kind]=net.eval()
    for ratio in [.01,.05]:
        fig,axes=plt.subplots(1,5,figsize=(16,3.5))
        axes[0].imshow(view['gt'].permute(1,2,0).cpu())
        axes[0].set_title('GT')
        for ax,(kind,net) in zip(axes[1:],nets.items()):
            p,h,logits=net(ratio)
            margin=logits.double()-torch.logit(scene.uniform.double())
            margin[~scene.eligible]=-torch.inf
            ids=torch.argsort(margin,descending=True,stable=True)[:scene.K[ratio]].sort().values
            delta=net.compensate(h,ids,ratio)
            assert delta.isfinite().all()
            image,_,_=scene.render(ids,delta,view)
            image=image.clamp(0,1)
            psnr=float(-10*torch.log10((image-view['gt']).square().mean()))
            ax.imshow(image.permute(1,2,0).cpu())
            ax.set_title(f'{kind}\n{psnr:.2f} dB / K={len(ids)}',fontsize=9)
        for ax in axes:
            ax.axis('off')
        fig.suptitle(f'seed3100 / {ratio*100:g}% / {view["path"]} / fixed middle view')
        fig.tight_layout()
        fig.savefig(out/f'{int(ratio*100)}pct_examples.png',dpi=140)
        plt.close(fig)
    assert scene.unchanged()


if __name__=='__main__':
    main()
