"""48/59维四Head：固定最终checkpoint，保存两seed及逐视图配对质量。"""
import csv
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
from run import ROOT,BASE,PROTOCOL,Network,config,rt

OUT=ROOT/'analysis'
METRICS=['psnr','ssim','lpips','photo']
KINDS=['full59','compact48']


def directory(kind,seed):
    return (BASE/'results/fr1_xyz/RawSharedE16Split4'/str(seed) if kind=='full59'
        else ROOT/'results/fr1_xyz'/PROTOCOL['candidate']/str(seed))


def read(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(value) for value in row)+' |' for row in rows])


def main():
    while not (ROOT/'run.exit').exists():
        time.sleep(10)
    assert (ROOT/'run.exit').read_text().strip()=='0'
    OUT.mkdir(exist_ok=True)
    quality,groups,views,paired,fairness,curves=[],[],[],[],[],[]
    tables={}
    for kind in KINDS:
        for seed in PROTOCOL['seeds']:
            d=directory(kind,seed)
            info=json.loads((d/'complete.json').read_text())
            assert info['steps']==12000 and info['master_frozen']
            assert all(value==3000 for value in info['ratio_exposures'].values())
            assert all(value==3000 for name,value in info['adam_clocks'].items() if name.startswith('extra_tables.'))
            assert all(value==12000 for name,value in info['adam_clocks'].items() if not name.startswith('extra_tables.'))
            fairness.append(dict(kind=kind,seed=seed,steps=12000,exposure_per_ratio=3000,
                parameters=info['parameters'],empty_steps=sum(info['empty_steps'].values()),
                reused=kind=='full59',adam_clocks=json.dumps(info['adam_clocks'])))
            if kind=='full59':
                dest=ROOT/'references'/kind/str(seed)
                dest.mkdir(parents=True,exist_ok=True)
                for name in ['complete.json','validation_full.csv','test_full.csv','training_curve.csv','validation_per_view.csv']:
                    shutil.copy2(d/name,dest/name)
            for row in read(d/'validation_per_view.csv'):
                curves.append(dict(kind=kind,seed=seed,**row))
            for split in ['validation','test']:
                rows=read(d/(split+'_full.csv'))
                tables[kind,seed,split]=rows
                for row in rows:
                    views.append(dict(row,kind=kind,seed=seed,split=split))
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
                    for group,ratios in [('all',rt.RATIOS),('train',rt.TRAIN),('unseen',[r for r in rt.RATIOS if r not in rt.TRAIN])]:
                        subset=[r for r in quality if r['kind']==kind and r['seed']==seed and r['split']==split
                            and r['policy']==policy and r['ratio'] in ratios]
                        groups.append(dict(kind=kind,seed=seed,split=split,policy=policy,group=group,
                            **{m:float(np.mean([r[m] for r in subset])) for m in METRICS}))
    for split in ['validation','test']:
        for ratio in rt.RATIOS:
            assert len({r['selected'] for r in quality if r['split']==split and r['policy']=='same_k' and r['ratio']==ratio})==1
        for policy in ['same_k','cached']:
            for group,ratios in [('all',rt.RATIOS),('train',rt.TRAIN),('unseen',[r for r in rt.RATIOS if r not in rt.TRAIN])]+[(str(r),[r]) for r in rt.RATIOS]:
                delta={m:[] for m in METRICS}
                for seed in PROTOCOL['seeds']:
                    a={(float(r['ratio']),r['path']):r for r in tables['compact48',seed,split]
                        if r['policy']==policy and float(r['ratio']) in ratios}
                    b={(float(r['ratio']),r['path']):r for r in tables['full59',seed,split]
                        if r['policy']==policy and float(r['ratio']) in ratios}
                    assert a.keys()==b.keys()
                    for m in METRICS:
                        delta[m].append(np.array([float(a[key][m])-float(b[key][m]) for key in a]))
                for m in METRICS:
                    values=np.stack(delta[m])
                    sign=1 if m in ['psnr','ssim'] else -1
                    paired.append(dict(candidate='compact48',reference='full59',split=split,policy=policy,
                        group=group,metric=m,mean=float(values.mean()),seed3100=float(values[0].mean()),
                        seed3101=float(values[1].mean()),seed_improved=int((sign*values.mean(1)>0).sum()),
                        median=float(np.median(values)),p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9)),
                        improved_fraction=float((sign*values>0).mean())))
    assert all(np.isfinite(float(r[m])) for r in views for m in METRICS)
    for name,rows in [('quality_summary.csv',quality),('group_summary.csv',groups),('quality_per_view.csv',views),
            ('paired_statistics.csv',paired),('fairness_checks.csv',fairness),('validation_curves.csv',curves)]:
        rt.write_csv(OUT/name,rows)
    primary=[r for r in paired if r['split']=='test' and r['policy']=='same_k' and r['metric']=='psnr']
    rt.save_json(OUT/'summary.json',dict(primary=primary,groups=[r for r in groups if r['split']=='test' and r['policy']=='same_k'],
        new_models=2,reused_models=2,seeds=PROTOCOL['seeds'],scene='fr1_xyz',steps=12000,exposure_per_ratio=3000))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,seed in zip(axes,PROTOCOL['seeds']):
        for kind in KINDS:
            rows=sorted([r for r in quality if r['kind']==kind and r['seed']==seed and r['split']=='test' and r['policy']=='same_k'],key=lambda r:r['ratio'])
            ax.plot([r['ratio']*100 for r in rows],[r['psnr'] for r in rows],marker='o',label=kind)
        ax.set(xlabel='Ratio (%)',ylabel='PSNR (dB)',title=f'fr1_xyz seed{seed} / non-Bank same-K')
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT/'quality_curves.png',dpi=150)
    plt.close(fig)
    examples()
    lines=['# 四Head输入消融：16维ratio latent＋共享h32','',
        '仅删除补偿头的原始11维直连，59维→48维；四个Head各保持两层32维隐藏层，输出3/3/4/4。fr1_xyz、两个seed、12000步。','',
        table(['ratio组','seed3100 ΔPSNR','seed3101 ΔPSNR','平均ΔPSNR','改善视图比例'],
            [[r['group'],f'{r["seed3100"]:+.3f}',f'{r["seed3101"]:+.3f}',f'{r["mean"]:+.3f}',f'{100*r["improved_fraction"]:.1f}%'] for r in primary]),'',
        '## 各seed质量','',
        table(['输入','seed','PSNR↑','SSIM↑','LPIPS↓','photo↓'],
            [[g['kind'],g['seed'],*[f'{g[m]:.5f}' for m in METRICS]] for g in groups
             if g['split']=='test' and g['policy']=='same_k' and g['group']=='all']),'',
        '![质量曲线](analysis/quality_curves.png)','','## 结构与归因边界','',
        '- Selector/Encoder仍读Offset3、Scale3、SH DC3、Opacity1、log(ratio)1；共享h32仍包含这些信息。本轮只删除它们对补偿头的直接连接，不是整个系统去掉属性或ratio条件。',
        '- 原输入顺序raw11+latent16+h32改为latent16+h32；48维首层精确保留原权重第11:59列与bias，后续隐藏层/输出层、Encoder/Selector/latent初始化相同。初始p与零补偿exact相同，非零隐藏函数本身因删输入而不同。',
        '- 每Anchor每ratio latent16仍共享给四Head，未见ratio仍先按raw-ratio线性插值latent再解码，不为未见ratio优化。只计算选中K行补偿。',
        '- 四Head各48→32 ReLU→32 ReLU→3/3/4/4，参数12366→10958，减少1408(约11.4%)；同时减少首层参数，不能称纯信息路径的完全等参数控制。',
        '- 原14D属性变换、photo+.01预算项、Cached Bernoulli、Adam fused lr=.001/eps1e-15、四ratio轮转与帧序保持。总12000步，每ratio3000曝光，保存4k/8k/12k但只评价最终checkpoint。',
        '- 同一冻结N6838/M2066地图、54 Bank训练/13内部Bank留出/45历史non-Bank，Master/IDs/Anchor关联中心尺度/相机/eligible/cached uniform冻结。runtime/Renderer与源manifest一致。',
        '- 复用上一轮59维四Head的seed3100/3101；它们分别是17.26895/17.16740 dB。历史更早seed3100为17.00095 dB且保留，不选最好重跑。GPU训练有已记录约0.268 dB同seed重复波动。',
        '- Encoder/Selector共同训练，不同方案选中身份可不同；same-K严格同预算，原Cached实际K另存。effective_proxy为可见且opacity≥1/255，不是积分贡献。',
        '- 两seed同时控制网络初始化与Bank顺序，不是两张独立SLAM地图；没有独立场景或全新盲测，不由小差值声明稳定收益。',
        '- 本轮只做隔离原型，生产算法与E50默认、在线二进制不变，无新在线SLAM或CUDA Mapping构建。未stage/commit/push。','',
        '## 固定视图','',
        '固定seed3100和中间non-Bank视图，未按效果选图。','',
        '![1%](analysis/1pct_examples.png)','','![5%](analysis/5pct_examples.png)','']
    (ROOT/'SPLIT_HEAD_NO_RAW_COMPENSATION_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')
    with tarfile.open(ROOT/'evidence.tar.gz','w:gz') as pack:
        for path in ROOT.rglob('*'):
            if path.is_file() and (path.suffix in ['.py','.json','.csv','.md','.png','.npy','.log'] or path.name=='step_12000.pt'):
                pack.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
    print('REPORT_READY',json.dumps(primary),flush=True)


@torch.no_grad()
def examples():
    scene=rt.Scene('fr1_xyz')
    views=scene.load_test()
    view=views[len(views)//2]
    nets={}
    for kind in KINDS:
        info=json.loads((directory(kind,3100)/'complete.json').read_text())
        net=Network(scene,info['config'],3100)
        net.load_state_dict(torch.load(info['checkpoint'],map_location='cuda')['state_dict'])
        nets[kind]=net.eval()
    for ratio in [.01,.05]:
        fig,axes=plt.subplots(1,3,figsize=(10,3.5))
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
            ax.set_title(f'{kind}\n{psnr:.2f} dB / K={len(ids)}')
        for ax in axes:
            ax.axis('off')
        fig.suptitle(f'{ratio*100:g}% / {view["path"]} / fixed middle view')
        fig.tight_layout()
        fig.savefig(OUT/f'{int(ratio*100)}pct_examples.png',dpi=140)
        plt.close(fig)
    assert scene.unchanged()


if __name__=='__main__':
    try:
        main()
    except Exception:
        (ROOT/'finalize.exit').write_text('1')
        raise
    (ROOT/'finalize.exit').write_text('0')
