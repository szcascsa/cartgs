"""锁定单组加深实验最终checkpoint，汇总same-K及Cached质量，不选择seed或训练步。"""
import csv
import json
import shutil
import tarfile
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import ROOT, OLD, PREVIOUS, EXTRA, PROTOCOL, Network, rt

OUT = ROOT / 'analysis'
METRICS = ['psnr', 'ssim', 'lpips', 'photo']
SEED = PROTOCOL['seed']
NEW = [PROTOCOL['candidate']['id']]
BASE = 'RawSharedE16'


def read(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def directory(config):
    if config in NEW:
        return ROOT / 'results/fr1_xyz' / config / str(SEED)
    if config == 'RawSharedE16':
        return EXTRA / 'results/fr1_xyz' / config / str(SEED)
    if config in ['DirectRawE08', 'DirectRawE16']:
        return PREVIOUS / 'results/fr1_xyz' / config / str(SEED)
    if config == 'S00R00E50D12':
        return Path('/home/prmi/zwj/s00-e50-compensation-20261026/results/fr1_xyz') / config / str(SEED)
    return OLD / 'results/fr1_xyz/confirm' / config / str(SEED)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
        ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows])


def main():
    while not (ROOT / 'run.exit').exists():
        time.sleep(10)
    assert (ROOT / 'run.exit').read_text().strip() == '0'
    OUT.mkdir(exist_ok=True)
    configs = [BASE, 'DirectRawE16', 'S00R00E50D12', *NEW, 'R3']
    quality, groups, paired, all_views, fairness, curves = [], [], [], [], [], []
    for config in configs:
        d = directory(config)
        info = json.loads((d / 'complete.json').read_text())
        assert info['steps'] == 12000 and info['master_frozen']
        assert all(v == 3000 for v in info['ratio_exposures'].values())
        fairness.append(dict(config=config, seed=SEED, steps=info['steps'], exposure_per_ratio=3000,
            parameters=info['parameters'], seconds=info['seconds'], empty_steps=sum(info['empty_steps'].values()),
            reused=config not in NEW, adam_clocks=json.dumps(info['adam_clocks'])))
        if config not in NEW:
            dest = ROOT / 'references' / config
            dest.mkdir(parents=True, exist_ok=True)
            for name in ['complete.json', 'validation_full.csv', 'test_full.csv', 'training_curve.csv', 'validation_per_view.csv']:
                if (d / name).exists():
                    shutil.copy2(d / name, dest / name)
        if config != 'R3':
            for row in read(d / 'validation_per_view.csv'):
                curves.append(dict(config=config, **row))
        for split in ['validation', 'test']:
            rows = read(d / (split + '_full.csv'))
            for row in rows:
                row['config'] = config
                row['split'] = split
            all_views.extend(rows)
            for policy in ['same_k', 'cached']:
                for ratio in rt.RATIOS:
                    subset = [r for r in rows if r['policy'] == policy and float(r['ratio']) == ratio]
                    assert subset
                    selected = {int(r['selected']) for r in subset}
                    assert len(selected) == 1
                    quality.append(dict(config=config, split=split, policy=policy, ratio=ratio,
                        selected=selected.pop(), views=len(subset),
                        effective_proxy=float(np.mean([float(r['effective_proxy']) for r in subset])),
                        **{m:float(np.mean([float(r[m]) for r in subset])) for m in METRICS}))
                for group, ratios in [('all', rt.RATIOS), ('train', rt.TRAIN),
                                      ('unseen', [r for r in rt.RATIOS if r not in rt.TRAIN])]:
                    subset = [r for r in quality if r['config'] == config and r['split'] == split and
                              r['policy'] == policy and r['ratio'] in ratios]
                    groups.append(dict(config=config, split=split, policy=policy, group=group,
                        **{m:float(np.mean([r[m] for r in subset])) for m in METRICS}))
    for split in ['validation', 'test']:
        for ratio in rt.RATIOS:
            assert len({r['selected'] for r in quality if r['split'] == split and r['policy'] == 'same_k' and r['ratio'] == ratio}) == 1
        for policy in ['same_k', 'cached']:
            for config in configs:
                if config == BASE:
                    continue
                a = {(float(r['ratio']), r['path']):r for r in all_views if r['config'] == config and r['split'] == split and r['policy'] == policy}
                b = {(float(r['ratio']), r['path']):r for r in all_views if r['config'] == BASE and r['split'] == split and r['policy'] == policy}
                assert a.keys() == b.keys()
                for label, ratios in [('all', rt.RATIOS), ('train', rt.TRAIN),
                        ('unseen', [r for r in rt.RATIOS if r not in rt.TRAIN])] + [(str(r), [r]) for r in rt.RATIOS]:
                    keys = [key for key in a if key[0] in ratios]
                    for metric in METRICS:
                        delta = np.array([float(a[key][metric])-float(b[key][metric]) for key in keys])
                        sign = 1 if metric in ['psnr', 'ssim'] else -1
                        paired.append(dict(config=config, reference=BASE, split=split, policy=policy, group=label,
                            metric=metric, mean=float(delta.mean()), median=float(np.median(delta)),
                            p10=float(np.quantile(delta, .1)), p90=float(np.quantile(delta, .9)),
                            improved_fraction=float(np.mean(sign*delta > 0))))
    assert all(np.isfinite(float(row[m])) for row in all_views for m in METRICS)
    for name, rows in [('quality_summary.csv', quality), ('group_summary.csv', groups),
            ('paired_statistics.csv', paired), ('quality_per_view.csv', all_views),
            ('fairness_checks.csv', fairness), ('validation_curves.csv', curves)]:
        rt.write_csv(OUT / name, rows)
    primary = [g for g in groups if g['split'] == 'test' and g['policy'] == 'same_k']
    rt.save_json(OUT / 'summary.json', dict(groups=primary, fairness=fairness, single_seed=True,
        scene='fr1_xyz', new_models=1, reused_models=4, production_modified=False))
    fig, ax = plt.subplots(figsize=(10, 4))
    for config in configs:
        rows = sorted([r for r in quality if r['config'] == config and r['split'] == 'test' and r['policy'] == 'same_k'], key=lambda r:r['ratio'])
        ax.plot([r['ratio']*100 for r in rows], [r['psnr'] for r in rows], marker='o', label=config)
    ax.set(xlabel='Ratio (%)', ylabel='PSNR (dB)', title='fr1_xyz / seed3100 / non-Bank / exact same-K')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'quality_curves.png', dpi=150)
    plt.close(fig)
    examples()
    base = {r['group']:r for r in primary if r['config'] == BASE}
    rows = []
    for config in configs:
        values = {r['group']:r for r in primary if r['config'] == config}
        rows.append([config, f'{values["all"]["psnr"]:.3f}',
            f'{values["all"]["psnr"]-base["all"]["psnr"]:+.3f}',
            f'{values["train"]["psnr"]-base["train"]["psnr"]:+.3f}',
            f'{values["unseen"]["psnr"]-base["unseen"]["psnr"]:+.3f}'])
    def score(config, group='all', metric='psnr'):
        return next(float(r[metric]) for r in primary if r['config']==config and r['group']==group)
    lines = ['# 59维输入Compensation Head：两层与三层隐藏层对照','',
        'fr1_xyz单次seed3100；新增一个三层头，保持12000真实LoD步与每ratio3000曝光。所有指标来自最终checkpoint，无早停或评价调参。','',
        table(['方案','七档PSNR','相对两层头','训练ratio增益','未见ratio增益'],rows),'',
        '## 单变量结构与初始化','',
        '- 共同输入：每Anchor每ratio latent16 + 原始Offset3/Scale3/SH DC3/Opacity1/log(ratio)1 + 原共享h32，共59维；共享h不detach。',
        '- 两层头RawSharedE16：59→32 ReLU→32 ReLU→14。三层头RawSharedE16D3：59→32 ReLU→32 ReLU→32 ReLU→14。只增加一个32→32 Linear+ReLU，宽度与特征维度均不变，额外1056个参数。',
        '- 追加层单位矩阵、bias零，插在原第二个ReLU后；所有公共权重、latent表初始化与两层控制exact一致。不是残差网络。初始p和delta相同，并以临时非零输出投影验证隐藏函数一致及新增层梯度。',
        '- 训练仍使用Cached Bernoulli、原photo+.01预算损失、Adam fused lr=.001/eps1e-15、相同帧序。新增层随其余补偿参数共同训练，共享Encoder仍接收补偿梯度。','',
        '## 质量与收敛证据','',
        table(['方案','PSNR↑','SSIM↑','LPIPS↓','photo↓'],
            [[c,*[f'{score(c,metric=m):.5f}' for m in METRICS]] for c in configs]),'',
        '所有相对变化以同seed、同地图、相同12000步的两层RawSharedE16为参考。四训练ratio为1/5/10/15%，三未见ratio为2.5/7.5/12.5%；未见ratio先插值latent再解码，没有单独拟合。','',
        '![质量曲线](analysis/quality_curves.png)','','## 实验边界','',
        '- 同一N6838/M2066地图；Master、Gaussian身份、Anchor归属/中心/尺度、相机、eligible及cached uniform冻结。54 Bank训练、13内部Bank留出、45历史non-Bank。runtime/Renderer SHA与原协议匹配。',
        '- Selector/Encoder和补偿仍按原隔离协议共同训练，改变补偿头可以间接改变最终选择集合。不是冻结选择集合的纯补偿Oracle。same-K按logit(p)-logit(cached_uniform)排序，各方案K严格一致；原Cached实际点数另存。',
        '- 单次单场景且使用历史评价视图，不作跨seed稳定性或新盲测结论；固定步数结果不是容量上限。4k/8k/12k收敛曲线存档，不回选checkpoint。',
        '- E50与R3是同seed相同步数的系统目标参考，输入及补偿参数化不同。R3剩余差距不能全归于补偿头。effective_proxy为可见且opacity≥1/255的点数，不是积分渲染贡献。',
        '- 全部实验仅在隔离Python原型中执行；生产默认E50、生产源码与在线二进制不变。无新在线SLAM或CUDA Mapping构建；未stage/commit/push。另有GPU负载，不作耗时速度结论。','',
        '## 固定视图','',
        '固定中间non-Bank视图，1%/5%，没有按改善幅度选图。','',
        '![1%](analysis/1pct_examples.png)','','![5%](analysis/5pct_examples.png)','']
    (ROOT / 'S00_RAW_SHARED_DEPTH_AUDIT.md').write_text('\n'.join(lines), encoding='utf-8')
    with tarfile.open(ROOT / 'evidence.tar.gz', 'w:gz') as pack:
        for path in ROOT.rglob('*'):
            if path.is_file() and (path.suffix in ['.py', '.json', '.csv', '.md', '.png', '.npy', '.log'] or path.name == 'step_12000.pt'):
                pack.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    print('REPORT_READY', json.dumps(primary), flush=True)


@torch.no_grad()
def examples():
    scene = rt.Scene('fr1_xyz')
    views = scene.load_test()
    view = views[len(views)//2]
    nets = {}
    for config in [BASE, *NEW, 'R3']:
        info = json.loads((directory(config) / 'complete.json').read_text())
        net = Network(scene, info['config'], SEED)
        net.load_state_dict(torch.load(info['checkpoint'], map_location='cuda')['state_dict'])
        nets[config] = net.eval()
    for ratio in [.01, .05]:
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
        axes[0].imshow(view['gt'].permute(1, 2, 0).cpu())
        axes[0].set_title('GT')
        for ax, (config, net) in zip(axes[1:], nets.items()):
            p, h, logits = net(ratio)
            margin = logits.double() - torch.logit(scene.uniform.double())
            margin[~scene.eligible] = -torch.inf
            ids = torch.argsort(margin, descending=True, stable=True)[:scene.K[ratio]].sort().values
            delta = net.compensate(h, ids, ratio)
            assert delta.isfinite().all()
            image, _, _ = scene.render(ids, delta, view)
            image = image.clamp(0, 1)
            psnr = float(-10*torch.log10((image-view['gt']).square().mean()))
            ax.imshow(image.permute(1, 2, 0).cpu())
            ax.set_title(f'{config}\n{psnr:.2f} dB / K={len(ids)}', fontsize=9)
        for ax in axes:
            ax.axis('off')
        fig.suptitle(f'{ratio*100:g}% / {view["path"]} / fixed middle non-Bank view')
        fig.tight_layout()
        fig.savefig(OUT / f'{int(ratio*100)}pct_examples.png', dpi=140)
        plt.close(fig)
    assert scene.unchanged()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        (ROOT / 'finalize.exit').write_text('1')
        raise
    (ROOT / 'finalize.exit').write_text('0')
