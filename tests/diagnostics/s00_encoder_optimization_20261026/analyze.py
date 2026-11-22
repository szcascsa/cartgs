"""固定输入的Encoder优化对照：匹配时钟、两seed、最终checkpoint。"""
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

KINDS=['original_baseline','bottleneck16','shallow32','deep32','residual32']
CANDIDATES=['shallow32','deep32','residual32']
METRICS=['psnr','ssim','lpips','photo']
GROUPS=[('all',rt.RATIOS),('train',rt.TRAIN),('unseen',[r for r in rt.RATIOS if r not in rt.TRAIN])]


def read(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def directory(kind,seed):
    if kind=='original_baseline':
        old=source.previous.xyz.baseline
        return source.previous.xyz.SOURCE/'results/fr1_xyz'/old.config('baseline')['id']/str(seed)
    if kind=='bottleneck16':return SOURCE/'results/fr1_xyz'/source.config()['id']/str(seed)
    return ROOT/'results/fr1_xyz'/config(kind)['id']/str(seed)


def model(scene,kind,seed):
    if kind=='original_baseline':
        old=source.previous.xyz.baseline;net=old.Network(scene,old.config('baseline'),seed)
    elif kind=='bottleneck16':net=source.Network(scene,source.config(),seed)
    else:net=Network(scene,config(kind),seed)
    net.load_state_dict(torch.load(directory(kind,seed)/'step_12000.pt',map_location='cuda')['state_dict'])
    return net.eval()


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])


@torch.no_grad()
def sets_for(scene,net):
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
    scene=rt.Scene('fr1_xyz');train_paths={v['path'] for v in scene.train}
    per_view,quality,groups,paired,fairness,curves,overlap,encoder_stats=[],[],[],[],[],[],[],[]
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
            if kind in CANDIDATES:
                assert info['config']['encoder_variant']==kind
                assert info['config']['encoder_layout']==config(kind)['encoder_layout']
            dest=d
            if kind not in CANDIDATES:
                dest=ROOT/'references'/kind/str(seed);dest.mkdir(parents=True,exist_ok=True)
                for name in ['complete.json','step_12000.pt','bank_full.csv','validation_full.csv','test_full.csv',
                    'training_curve.csv','validation_per_view.csv']:shutil.copy2(d/name,dest/name)
            net=model(scene,kind,seed);sets[kind,seed]=sets_for(scene,net);del net
            np.savez_compressed(dest/'selected_sets.npz',**sets[kind,seed])
            fairness.append(dict(kind=kind,seed=seed,steps=12000,exposure_per_ratio=3000,parameters=info['parameters'],
                seconds=info['seconds'],adam_clocks=json.dumps(info['adam_clocks']),empty_steps=json.dumps(info['empty_steps']),
                compensation_updates=12000-empty_total,master_frozen=True,reused=kind not in CANDIDATES))
            if kind in CANDIDATES:
                encoder_stats.extend([dict(r,kind=kind,seed=seed) for r in read(d/'encoder_output_statistics.csv')])
            elif kind=='bottleneck16':
                net=model(scene,kind,seed)
                with torch.no_grad():
                    for ratio in rt.RATIOS:
                        p,h,_=net(ratio)
                        encoder_stats.append(dict(kind=kind,seed=seed,ratio=ratio,shared_rms=float(h.square().mean().sqrt()),
                            shared_max=float(h.abs().max()),shared_zero_fraction=float((h==0).float().mean()),
                            dead_channels=int((h==0).all(0).sum()),p_mean=float(p.mean()),p10=float(p.quantile(.1)),
                            p90=float(p.quantile(.9)),cached_k=int((p>scene.uniform).sum())))
                del net
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
    comparisons=[(kind,'bottleneck16') for kind in CANDIDATES]+[('deep32','shallow32'),('residual32','shallow32'),
        ('residual32','deep32')]+[(kind,'original_baseline') for kind in CANDIDATES]
    for split in ['bank_train','validation','test']:
        for ratio in rt.RATIOS:
            assert len({r['selected'] for r in quality if r['split']==split and r['policy']=='same_k' and r['ratio']==ratio})==1
        for a,b in comparisons:
            for policy in ['same_k','cached']:
                for group,ratios in GROUPS+[(str(r),[r]) for r in rt.RATIOS]:
                    for m in METRICS:
                        diffs=[]
                        for seed in PROTOCOL['seeds']:
                            index=lambda kind:{(r['path'],float(r['ratio'])):r for r in values[kind,seed,split]
                                if r['policy']==policy and float(r['ratio']) in ratios}
                            x,y=index(a),index(b);assert x.keys()==y.keys()
                            diffs.append(np.array([float(x[k][m])-float(y[k][m]) for k in x]))
                        v=np.stack(diffs);sign=1 if m in ['psnr','ssim'] else -1
                        paired.append(dict(candidate=a,reference=b,split=split,policy=policy,group=group,metric=m,
                            mean=float(v.mean()),seed3100=float(v[0].mean()),seed3101=float(v[1].mean()),
                            median=float(np.median(v)),p10=float(np.quantile(v,.1)),p90=float(np.quantile(v,.9)),
                            improved_fraction=float((sign*v>0).mean()),seed_improved=int((sign*v.mean(1)>0).sum())))
    for kind in CANDIDATES:
        for seed in PROTOCOL['seeds']:
            for key,a in sets['bottleneck16',seed].items():
                b=sets[kind,seed][key];inter=len(np.intersect1d(a,b));union=len(np.union1d(a,b))
                overlap.append(dict(candidate=kind,seed=seed,set=key,reference_k=len(a),candidate_k=len(b),
                    intersection=inter,jaccard=inter/union if union else 1.,recall=inter/len(a) if len(a) else 1.))
    assert scene.unchanged()
    assert all(np.isfinite(float(r[m])) for r in per_view for m in METRICS)
    for name,rows in [('quality_per_view.csv',per_view),('quality_summary.csv',quality),('group_summary.csv',groups),
        ('paired_statistics.csv',paired),('fairness_checks.csv',fairness),('validation_curves.csv',curves),
        ('selected_set_overlap.csv',overlap),('encoder_output_statistics.csv',encoder_stats)]:rt.write_csv(out/name,rows)
    pick=lambda a,b='bottleneck16',g='all',policy='same_k',split='test':next(r for r in paired if r['candidate']==a and r['reference']==b
        and r['group']==g and r['policy']==policy and r['split']==split and r['metric']=='psnr')
    summary=dict(primary=[pick(a,b,g) for a,b in comparisons for g,_ in GROUPS],
        cached=[pick(a,g=g,policy='cached') for a in CANDIDATES for g,_ in GROUPS],same_k_equal=True,source_frozen=True,
        metrics_finite=True,new_models=6,reused_models=4,steps=12000,exposure_per_ratio=3000,production_change=False,online_slam=False)
    rt.save_json(out/'summary.json',summary)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,seed in zip(axes,PROTOCOL['seeds']):
        for kind in KINDS:
            sub=sorted([r for r in quality if r['kind']==kind and r['seed']==seed and r['split']=='test'
                and r['policy']=='same_k'],key=lambda r:r['ratio'])
            ax.plot([r['ratio']*100 for r in sub],[r['psnr'] for r in sub],marker='o',label=kind)
        ax.set(xlabel='Ratio (%)',ylabel='PSNR (dB)',title=f'fr1_xyz seed{seed}');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'quality_curves.png',dpi=150);plt.close(fig)
    lines=['# 固定11维输入的共享Encoder优化实验','',
        '本轮只改变当前11维输入之后的共享Encoder。输入、四Head、Anchor/ratio latent16、Loss、采样、ratio顺序和训练曝光保持一致；生产默认未替换。','',
        '## 结构与问题','',table(['方案','Encoder','参数量','检验问题'],[
        ['bottleneck16','11→16 ReLU→32 ReLU→32 ReLU',1792,'上一轮当前参照'],
        ['shallow32','11→32 ReLU',384,'去掉16瓶颈和额外深度'],
        ['deep32','11→32 ReLU→32 ReLU→32 ReLU',2496,'保留深度但去掉16瓶颈'],
        ['residual32','h=ReLU(W0u+b0); y=ReLU(h+W2 ReLU(W1h+b1)+b2)',2496,'同参数深度的残差连接与函数保持起点']]),'',
        '- 原始输入固定为全局XYZ3、activated Scale3、sigmoid Opacity1、局部Offset3、logratio1；没有新特征或归一化。补偿输入raw11+Anchor/ratio latent16+h32=59，四Head仍各59→32→32→3/3/4/4。',
        '- 三个候选共同首层取同seed上一组8→32的32行公共权重，新增Offset列0；Selector/Head/latent所有共同权重exact相同。deep32后两层固定seed初始化，residual32末层W2/b2为0，因此step0隐藏函数exact等于shallow32。',
        '- 所有方案初始选择概率和零补偿exact一致；当前16瓶颈与新32首层不保证隐藏函数相同，不能称完全隔离初始化因素。',
        '- 残差末层为0时inner层首个梯度可以为0；真实零补偿初始状态的4次photo+预算反传验证其随后收到非零梯度，不将零初始化的正常延迟误判为失效。','',
        '## 相对当前16瓶颈方案的non-Bank same-K结果','',
        table(['候选','seed3100 ΔPSNR','seed3101 ΔPSNR','七档均值','训练ratio','未见ratio'],
        [[a,f'{pick(a)["seed3100"]:+.3f}',f'{pick(a)["seed3101"]:+.3f}',f'{pick(a)["mean"]:+.3f}',
            f'{pick(a,g="train")["mean"]:+.3f}',f'{pick(a,g="unseen")["mean"]:+.3f}'] for a in CANDIDATES]),'',
        table(['方案','PSNR↑','SSIM↑','LPIPS↓','photo↓'],[[kind,*[f'{np.mean([r[m] for r in groups if r["kind"]==kind and r["split"]=="test" and r["policy"]=="same_k" and r["group"]=="all"]):.5f}'
            for m in METRICS]] for kind in KINDS]),'',
        '## 候选之间的控制','',table(['比较','seed3100 ΔPSNR','seed3101 ΔPSNR','七档均值','训练ratio','未见ratio'],
        [[a+' − '+b,f'{pick(a,b)["seed3100"]:+.3f}',f'{pick(a,b)["seed3101"]:+.3f}',f'{pick(a,b)["mean"]:+.3f}',
            f'{pick(a,b,"train")["mean"]:+.3f}',f'{pick(a,b,"unseen")["mean"]:+.3f}']
            for a,b in [('deep32','shallow32'),('residual32','shallow32'),('residual32','deep32')]]),'',
        '## 各ratio相对当前参照','',table(['ratio','K',*CANDIDATES],
        [[f'{r*100:g}%',scene.K[r],*[f'{pick(a,g=str(r))["mean"]:+.3f}' for a in CANDIDATES]] for r in rt.RATIOS]),'',
        '## Bank、留出及原Cached评价','',table(['候选','Bank训练视图ΔPSNR','内部留出ΔPSNR','non-Bank实际Cached ΔPSNR'],
        [[a,f'{pick(a,split="bank_train")["mean"]:+.3f}',f'{pick(a,split="validation")["mean"]:+.3f}',
            f'{pick(a,policy="cached")["mean"]:+.3f}'] for a in CANDIDATES]),'',
        '## 协议与证据边界','',
        '- 同源fr1_xyz冻结N6838/M2066地图，Master/拓扑/相机/Anchor关联/eligible/cached uniform保持；Encoder/Selector/四Head和当前ratio latent联合训练，因此选择身份可变。same-K保证预算，不等于固定集合的补偿因果。',
        '- 预注册三候选×两seed，共6次完整训练，各12000步、每ratio3000；第二seed反转三臂执行顺序。复用当前16瓶颈与原始Baseline的4个最终模型，不选择最好复跑。',
        '- 固定54 Bank训练320×240、13内部留出和45历史non-Bank评价640×480；训练1/5/10/15%，评价另含2.5/7.5/12.5%；latent插值规则保留。',
        '- fresh fused Adam lr=.001 eps1e-15，原photo=.8L1+.2(1-fast_ssim)加.01相对预算L1；Encoder/Selector各12000更新、每ratio3000次训练曝光，未活跃表grad=None；没有零梯度伪造步数。',
        '- deep32/seed3100在1%档有2次Cached空集合，该两步仅预算分支有实际优化路径：补偿Head实际更新11998次、1% latent表2998次，其他表3000次。其余所有模型无空集合，补偿Head12000次、各表3000次。实际Adam时钟逐参数与空集合记录精确核对；相同训练曝光不等于所有辅助参数的更新次数绝对相同。',
        '- 统一最终12000步评价，保存4k/8k/12k曲线，未依据non-Bank选择LR、step、模型或追加候选。有限步数差距包含优化与初始化，不能称容量极限。',
        '- 原始Baseline使用局部Offset+Scale+SH DC+Opacity+logratio及浅Encoder，输入信息与本轮不同，只是系统效果参照，不能将其差值全归因于Encoder。',
        '- 两seed是同一地图的网络初始化和Bank顺序重复，不是独立SLAM地图。历史non-Bank已多次用于结构设计；本轮仅探索性结果，不作统计显著、跨场景或在线Joint质量保证。',
        '- deep32和residual32参数量相同，但连接方式、末层初始化和优化条件不同；对照检验整套结构，不能将差值只归为单一梯度性质。',
        '- GPU有其他任务，耗时仅作运行记录；新原型不接入生产，不stage/commit/push。',
        '- 训练预检查发现checkpoint继承的encoder_layout描述需更新；该预检查运行中止并保留metadata_preflight，本轮全部6个正式模型从step0重训，未用预检查质量做选择。','',
        '![质量曲线](analysis/quality_curves.png)','']
    lines+=['## 共享编码的激活情况','',table(['方案','平均零激活比例','完全为零通道数范围','编码RMS均值'],
        [[kind,f'{np.mean([float(r["shared_zero_fraction"]) for r in encoder_stats if r["kind"]==kind]):.3f}',
            str(min(int(r['dead_channels']) for r in encoder_stats if r['kind']==kind))+'～'+str(max(int(r['dead_channels']) for r in encoder_stats if r['kind']==kind)),
            f'{np.mean([float(r["shared_rms"]) for r in encoder_stats if r["kind"]==kind]):.3f}']
            for kind in ['bottleneck16',*CANDIDATES]]),'',
        '这些是最终checkpoint全部Gaussian上的数值统计，不以激活更稠密或编码幅度更大直接认定渲染更好。','']
    (ROOT/'ENCODER_OPTIMIZATION_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')
    examples(scene)
    with tarfile.open(ROOT/'evidence.tar.gz','w:gz') as pack:
        for path in ROOT.rglob('*'):
            if 'metadata_preflight' in path.relative_to(ROOT).parts:continue
            if path.is_file() and (path.suffix in ['.py','.json','.csv','.md','.png','.npy','.npz'] or path.name=='step_12000.pt'):
                pack.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
    print('PRIMARY',json.dumps(summary),flush=True);print('ARCHIVE_SHA256',rt.sha(ROOT/'evidence.tar.gz'),flush=True)


@torch.no_grad()
def examples(scene):
    view=scene.load_test()[len(scene.load_test())//2]
    kinds=['bottleneck16',*CANDIDATES];nets={kind:model(scene,kind,3100) for kind in kinds}
    for ratio in [.01,.05]:
        fig,axes=plt.subplots(1,5,figsize=(18,4));axes[0].imshow(view['gt'].permute(1,2,0).cpu());axes[0].set_title('GT')
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
