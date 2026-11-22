"""固定选点的同容量补偿Encoder对照：初始值、继续训练控制与专用编码。"""
import csv
import json
from pathlib import Path
import tarfile
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import ROOT,PROTOCOL,FrozenSelection,Compensation,rt

OUT=ROOT/'analysis'
KINDS=['initial','frozen_encoder','private_encoder']
METRICS=['psnr','ssim','lpips','photo']


def read(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def directory(kind,seed):
    return ROOT/'results/fr1_xyz'/kind/str(seed)


def main():
    while not (ROOT/'run.exit').exists():
        time.sleep(10)
    assert (ROOT/'run.exit').read_text().strip()=='0'
    OUT.mkdir(exist_ok=True)
    views,quality,groups,paired,curves,fairness=[],[],[],[],[],[]
    tables={}
    comparisons=[('private_encoder','frozen_encoder'),('frozen_encoder','initial'),('private_encoder','initial')]
    for kind in KINDS:
        for seed in PROTOCOL['seeds']:
            if kind!='initial':
                d=directory(kind,seed)
                info=json.loads((d/'complete.json').read_text())
                assert info['steps']==12000 and info['frozen_source_unchanged']
                assert all(v==3000 for v in info['ratio_exposures'].values())
                assert all(v==(12000 if kind=='private_encoder' else 0) for k,v in info['adam_clocks'].items() if k.startswith('encoder.'))
                fairness.append(dict(kind=kind,seed=seed,steps=12000,exposure_per_ratio=3000,
                    trainable_parameters=info['trainable_parameters'],stored_parameters=info['stored_parameters'],
                    encoder_change_l2=info['encoder_change_l2'],adam_clocks=json.dumps(info['adam_clocks']),
                    source_checkpoint_sha256=info['source_checkpoint_sha256'],frozen_source_unchanged=True))
                for row in read(d/'validation_per_view.csv'):
                    curves.append(dict(kind=kind,seed=seed,**row))
            for split in ['validation','test']:
                path=ROOT/f'initial_{seed}_{split}.csv' if kind=='initial' else directory(kind,seed)/(split+'_full.csv')
                rows=read(path)
                tables[kind,seed,split]=rows
                for row in rows:
                    views.append(dict(row,kind=kind,seed=seed,split=split))
                for policy in ['cached','same_k']:
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
    for seed in PROTOCOL['seeds']:
        for split in ['validation','test']:
            for policy in ['cached','same_k']:
                for ratio in rt.RATIOS:
                    assert len({r['selected'] for r in quality if r['seed']==seed and r['split']==split
                        and r['policy']==policy and r['ratio']==ratio})==1
    for split in ['validation','test']:
        for policy in ['cached','same_k']:
            for a,b in comparisons:
                for group,ratios in [('all',rt.RATIOS),('train',rt.TRAIN),('unseen',[r for r in rt.RATIOS if r not in rt.TRAIN])]+[(str(r),[r]) for r in rt.RATIOS]:
                    delta={m:[] for m in METRICS}
                    for seed in PROTOCOL['seeds']:
                        x={(float(r['ratio']),r['path']):r for r in tables[a,seed,split] if r['policy']==policy and float(r['ratio']) in ratios}
                        y={(float(r['ratio']),r['path']):r for r in tables[b,seed,split] if r['policy']==policy and float(r['ratio']) in ratios}
                        assert x.keys()==y.keys()
                        for m in METRICS:
                            delta[m].append(np.array([float(x[k][m])-float(y[k][m]) for k in x]))
                    for m in METRICS:
                        values=np.stack(delta[m])
                        sign=1 if m in ['psnr','ssim'] else -1
                        paired.append(dict(candidate=a,reference=b,split=split,policy=policy,group=group,metric=m,
                            mean=float(values.mean()),seed3100=float(values[0].mean()),seed3101=float(values[1].mean()),
                            seed_improved=int((sign*values.mean(1)>0).sum()),median=float(np.median(values)),
                            p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9)),
                            improved_fraction=float((sign*values>0).mean())))
    assert all(np.isfinite(float(r[m])) for r in views for m in METRICS)
    for name,rows in [('quality_summary.csv',quality),('group_summary.csv',groups),('quality_per_view.csv',views),
            ('paired_statistics.csv',paired),('fairness_checks.csv',fairness),('validation_curves.csv',curves)]:
        rt.write_csv(OUT/name,rows)
    primary=[r for r in paired if r['split']=='test' and r['policy']=='cached' and r['metric']=='psnr'
        and r['group'] in ['all','train','unseen']]
    rt.save_json(OUT/'summary.json',dict(primary=primary,groups=[r for r in groups if r['split']=='test'],
        scene='fr1_xyz',seeds=PROTOCOL['seeds'],new_models=4,steps=12000,exposure_per_ratio=3000,
        frozen_sets_exact=True,primary_policy='fixed original cached sets, exact same identities and K within each seed'))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,seed in zip(axes,PROTOCOL['seeds']):
        for kind in KINDS:
            rows=sorted([r for r in quality if r['kind']==kind and r['seed']==seed and r['split']=='test'
                and r['policy']=='cached'],key=lambda r:r['ratio'])
            ax.plot([r['ratio']*100 for r in rows],[r['psnr'] for r in rows],marker='o',label=kind)
        ax.set(xlabel='Ratio (%)',ylabel='PSNR (dB)',title=f'seed{seed} / fixed cached sets / non-Bank')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT/'quality_curves.png',dpi=150)
    plt.close(fig)
    diagnostics_and_examples()
    lines=['# 同容量Compensation-private Encoder受控对照','',
        'fr1_xyz、两个源48维四Head checkpoint，每组额外12000步、每ratio3000曝光；不是之前从头Joint训练的直接横比。','',
        table(['对照','ratio组','seed3100 ΔPSNR','seed3101 ΔPSNR','平均ΔPSNR'],
            [[r['candidate']+' − '+r['reference'],r['group'],f'{r["seed3100"]:+.3f}',f'{r["seed3101"]:+.3f}',f'{r["mean"]:+.3f}'] for r in primary]),'',
        '## 实际结构与梯度','',
        '- 源模型为SharedE16Split4NoRaw各seed已训练12000步模型；冻结原Master、Selector、原Encoder、Anchor关系、相机和所有七档选择集合。',
        '- A：固定复制的原11→32 ReLU编码，只训练四Head和四张Anchor/ratio latent16表。B：相同初值的独立11→32 ReLU编码也可训练，梯度仅来自补偿photo。',
        '- 两组输入仍为latent16+编码32，共48维；四Head各48→32 ReLU→32 ReLU→3/3/4/4，14D输出变换不改。没有原始11维直连、额外Rotation、加深或加宽。',
        '- 两组补偿模块存储参数数相同，B多384个可训练自由度。原Selector Encoder始终冻结；只对选中K行执行补偿编码和解码。',
        '- 初始A/B输出exact相同；K行与原N行GEMM可能产生微小浮点差异，对原函数以gate记录误差容忍核对，不把此差异隐藏为逐位exact。',
        '- A/B均fresh Adam fused lr=.001 eps1e-15，Head/latent优化器重置一致；Head12000步、各ratio表3000步，A编码0步、B编码12000步。没有伪造零梯度step。',
        '- 训练固定原Cached Bernoulli集合，原opacity ST前向gate保留但无Selector梯度；原photo+.01预算项中预算成为冻结常数，不影响补偿梯度，不引入新Loss。','',
        '## 选点与数据','',
        '- primary为源模型固定Cached集合，每个seed内A/B/initial的Gaussian身份与实际K exact相同；不同seed可有不同K，配对比较不混淆。',
        '- secondary为源模型logit(p)-logit(cached_uniform)排序后固定floor(rN)的same-K集合，同样身份冻结；并非训练时改换采样算法。',
        '- 同N6838/M2066地图，54 Bank训练320×240、13内部Bank留出与45历史non-Bank评价640×480，源sha、选择rows sha及冻结检查完整保存。',
        '- 四训练ratio1/5/10/15%，三未见ratio2.5/7.5/12.5%；latent插值与连续logratio编码保持，不为未见ratio重新训练。','',
        '## 因果与范围','',
        '- B−A隔离当前固定集合下的编码适配自由度；不等于证明原共享训练存在有害梯度冲突，也不能直接证明32维信息丢失或维度不足。',
        '- 与initial的差距包含继续训练Head/latent的收益，不能全部记为专用Encoder收益。所有最终step固定，不按non-Bank选LR/step。',
        '- 两seed对应相同冻结地图上的不同源模型/帧序，不是独立SLAM场景；历史评价轨迹已反复使用，没有新盲测或线上Joint证据。',
        '- 所有生产文件与默认E50保持，未stage/commit/push。没有新的在线SLAM或CUDA Mapping构建。','',
        '![曲线](analysis/quality_curves.png)','','## 固定渲染示例','',
        '固定seed3100中间non-Bank视图，未按改善幅度选图。','',
        '![1%](analysis/1pct_examples.png)','','![5%](analysis/5pct_examples.png)','']
    (ROOT/'COMPENSATION_PRIVATE_ENCODER_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')
    with tarfile.open(ROOT/'evidence.tar.gz','w:gz') as pack:
        for path in ROOT.rglob('*'):
            if path.is_file() and (path.suffix in ['.py','.json','.csv','.md','.png','.npz','.log'] or path.name=='step_12000.pt'):
                pack.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
    print('REPORT_READY',json.dumps(primary),flush=True)


@torch.no_grad()
def diagnostics_and_examples():
    scene=rt.Scene('fr1_xyz')
    rows=[]
    for seed in PROTOCOL['seeds']:
        with np.load(ROOT/f'selected_sets_{seed}.npz') as stored:
            original_sets={key:stored[key].copy() for key in stored.files}
        frozen=FrozenSelection(scene,seed)
        for ratio,sets in frozen.ids.items():
            for policy,ids in sets.items():
                assert np.array_equal(original_sets[f'{ratio:g}_{policy}'],ids.cpu().numpy())
        initial=Compensation(frozen,False)
        nets={'initial':initial}
        for kind in ['frozen_encoder','private_encoder']:
            net=Compensation(frozen,kind=='private_encoder')
            net.load_state_dict(torch.load(directory(kind,seed)/'step_12000.pt',map_location='cuda')['state_dict'])
            nets[kind]=net.eval()
            for ratio in rt.RATIOS:
                ids=frozen.ids[ratio]['cached']
                raw=frozen.raw[ratio][ids]
                h0=torch.relu(initial.encoder(raw))
                h=torch.relu(net.encoder(raw))
                rows.append(dict(kind=kind,seed=seed,ratio=ratio,K=len(ids),
                    encoder_output_change_rms=float((h-h0).square().mean().sqrt()),
                    encoder_feature_cosine=float(torch.nn.functional.cosine_similarity(h,h0,dim=1).mean()),
                    initial_relu_active_fraction=float((h0>0).float().mean()),
                    final_relu_active_fraction=float((h>0).float().mean()),
                    note='representation change only, not proof of image benefit or gradient conflict'))
        if seed==3100:
            views=scene.load_test()
            view=views[len(views)//2]
            for ratio in [.01,.05]:
                fig,axes=plt.subplots(1,4,figsize=(13,3.5))
                axes[0].imshow(view['gt'].permute(1,2,0).cpu())
                axes[0].set_title('GT')
                ids=frozen.ids[ratio]['cached']
                for ax,(kind,net) in zip(axes[1:],nets.items()):
                    image,_,_=scene.render(ids,net(ratio,ids),view)
                    image=image.clamp(0,1)
                    psnr=float(-10*torch.log10((image-view['gt']).square().mean()))
                    ax.imshow(image.permute(1,2,0).cpu())
                    ax.set_title(f'{kind}\n{psnr:.2f} dB / K={len(ids)}',fontsize=9)
                for ax in axes:
                    ax.axis('off')
                fig.suptitle(f'{ratio*100:g}% / fixed source cached rows / {view["path"]}')
                fig.tight_layout()
                fig.savefig(OUT/f'{int(ratio*100)}pct_examples.png',dpi=140)
                plt.close(fig)
        assert frozen.unchanged()
        del nets,initial,frozen
        torch.cuda.empty_cache()
    rt.write_csv(OUT/'encoder_representation_statistics.csv',rows)


if __name__=='__main__':
    try:
        main()
    except Exception:
        (ROOT/'finalize.exit').write_text('1')
        raise
    (ROOT/'finalize.exit').write_text('0')
