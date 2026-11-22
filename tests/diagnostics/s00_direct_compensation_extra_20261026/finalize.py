"""锁定两组追加实验最终checkpoint，汇总same-K及Cached质量，不选择seed或训练步。"""
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
from run import ROOT, OLD, PREVIOUS, PROTOCOL, Network, rt

OUT = ROOT / 'analysis'
METRICS = ['psnr', 'ssim', 'lpips', 'photo']
SEED = PROTOCOL['seed']
NEW = [c['id'] for c in PROTOCOL['candidates']]
BASE = 'S00R00E08D12'


def read(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def directory(config):
    if config in NEW:
        return ROOT / 'results/fr1_xyz' / config / str(SEED)
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
    configs = [BASE, 'S00R00E50D12', 'DirectRawE08', 'DirectRawE16', *NEW, 'R3']
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
        scene='fr1_xyz', new_models=2, reused_models=5, production_modified=False))
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
    comparisons=[('DirectRawE32','DirectRawE16'),('RawSharedE16','DirectRawE16'),('RawSharedE16','DirectRawE32')]
    lines = ['# Compensation追加：32维ratio特征与原始/共享编码联合输入','',
        '仅fr1_xyz、单次seed3100。新增两组，均12000真实LoD步、每ratio3000曝光，最终checkpoint评价；不根据评价选择LR或步数。','',
        table(['方案','七档PSNR','相对旧E8','训练ratio增益','未见ratio增益'],rows),'',
        '## 关键匹配对照','',
        table(['比较','七档ΔPSNR','训练ratioΔPSNR','未见ratioΔPSNR'],
            [[a+' − '+b,*[f'{score(a,g)-score(b,g):+.3f}' for g in ['all','train','unseen']]] for a,b in comparisons]),'',
        '## 网络与梯度','',
        '- DirectRawE32：每Anchor每ratio latent32 + 原始属性11，总输入43，补偿头43→32 ReLU→32 ReLU→14，不读取共享h32。',
        '- RawSharedE16：latent16 + 原始属性11 + 原共享h32，总输入59，补偿头59→32 ReLU→32 ReLU→14。共享h不detach，补偿可向共享Encoder回传。',
        '- 原始11维为归一化局部Offset3、activated Scale3、SH DC3、activated Opacity1、log(ratio)1；Selector网络输入和结构保持11→32→32→2。',
        '- 每Anchor四张独立latent表；训练ratio1/5/10/15%，未见ratio2.5/7.5/12.5%，沿用raw-ratio线性插值latent再解码。只对选中K行计算补偿。',
        '- 与旧DirectRawE16的raw11、前16列latent投影、bias及第二层/输出层初始化exact一致。32维组延伸原28列随机reservoir，保留前16列；共享组为追加h32初始化小幅非零投影。',
        '- 所有组初始p=.5与补偿exact零；私有特征梯度gate以临时非零输出层检查链路，不声称实际零输出初始化首步latent即更新。',
        '- 同步验证：直接32组compensation→Encoder梯度为零，共享16组该梯度非零，Selector私有head不接收独立补偿forward的梯度；非当前ratio表gradient=None。','',
        '## 冻结地图与公平性','',
        '- 复用N6838/M2066的同一fr1_xyz地图、同一54 Bank训练/13内部Bank留出/45历史non-Bank、seed3100和帧序；源地图manifest、runtime及Renderer SHA匹配历史实验。',
        '- Master、Anchor归属/中心/尺度、相机、eligible及cached uniform冻结。Selector/Encoder仍按原预算与photo共同训练，不同方案选择集合可以变化；不是冻结Selection的纯补偿Oracle。',
        '- 同一Adam fused lr=.001/eps1e-15、原photo及.01预算项、同12000步和每ratio3000曝光；E8/E50/DirectRaw8/DirectRaw16/R3均复用同seed与相同步数结果。',
        '- 正式same-K按logit(p)-logit(cached_uniform)排序；Cached实际点数另存。effective_proxy表示可见且opacity≥1/255，不是积分贡献测量。',
        '- 新增共享输入也恢复了Compensation与Encoder的梯度耦合，并增加头参数；该对照测量整个附加h路径效果，不能单独证明信息压缩或梯度冲突因果。',
        '- 单次单场景、历史评价视图，不作跨seed稳定收益、统计显著性或新盲测结论；R3有不同Selector特征与独立参数化，仅作系统目标参考。',
        '- 生产算法和E50默认不变。本轮无在线SLAM、无新CUDA Mapping构建，未stage/commit/push。GPU另有负载，不以记录耗时作速度比较。','',
        '## 四指标：七档non-Bank same-K','',
        table(['方案','PSNR↑','SSIM↑','LPIPS↓','photo↓'],
            [[c,*[f'{score(c,metric=m):.5f}' for m in METRICS]] for c in configs]),'',
        '![质量曲线](analysis/quality_curves.png)','','## 固定视图','',
        '固定中间non-Bank视图、1%/5%，不按改善幅度选图。','',
        '![1%](analysis/1pct_examples.png)','','![5%](analysis/5pct_examples.png)','']
    (ROOT / 'S00_DIRECT_COMPENSATION_EXTRA_AUDIT.md').write_text('\n'.join(lines), encoding='utf-8')
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
    for config in [BASE, 'DirectRawE16', *NEW, 'R3']:
        info = json.loads((directory(config) / 'complete.json').read_text())
        net = Network(scene, info['config'], SEED)
        net.load_state_dict(torch.load(info['checkpoint'], map_location='cuda')['state_dict'])
        nets[config] = net.eval()
    for ratio in [.01, .05]:
        fig, axes = plt.subplots(1, 6, figsize=(18, 3.5))
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
