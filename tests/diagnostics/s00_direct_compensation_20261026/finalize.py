"""锁定四组最终checkpoint，汇总same-K及Cached质量，不选择seed或训练步。"""
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
from run import ROOT, OLD, PROTOCOL, Network, rt

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
    configs = [BASE, 'S00R00E50D12', *NEW, 'R3']
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
    for name, rows in [('quality_summary.csv', quality), ('group_summary.csv', groups),
            ('paired_statistics.csv', paired), ('quality_per_view.csv', all_views),
            ('fairness_checks.csv', fairness), ('validation_curves.csv', curves)]:
        rt.write_csv(OUT / name, rows)
    primary = [g for g in groups if g['split'] == 'test' and g['policy'] == 'same_k']
    rt.save_json(OUT / 'summary.json', dict(groups=primary, fairness=fairness, single_seed=True,
        scene='fr1_xyz', new_models=4, reused_models=3, production_modified=False))
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
    lines = ['# Compensation直接输入：8/16维ratio特征与Offset对照', '',
        '本轮按用户要求仅fr1_xyz、单优化顺序seed3100；新增四个模型，全部12000真实LoD步，每ratio3000次曝光。不作稳定收益或统计显著性判断。', '',
        table(['方案', '七档PSNR', '相对E8共享编码', '训练ratio增益', '未见ratio增益'], rows), '',
        '四指标与逐ratio指标、原Cached实际预算、有效点代理及逐视图配对分布见analysis/*.csv。', '',
        '## 结构与协议', '',
        '- DirectOffsetE08/E16：每Anchor每训练ratio的8/16维latent，加归一化局部Offset3，进入两隐藏层32→32、输出14的补偿头。',
        '- DirectRawE08/E16：相同latent，加Offset3、activated scale3、SH DC3、activated opacity1及log(ratio)1，总输入19/27维。',
        '- Selector仍使用原11→32共享层、32→32→2选择头；四组仅补偿绕过共享h32。未见ratio在1/5/10/15%节点之间先线性插值latent再解码。',
        '- 训练ratio1/5/10/15%，未见ratio2.5/7.5/12.5%。源地图N6838/M2066、Master属性/Anchor关联/相机/eligible/cached uniform冻结，54训练Bank、13内部Bank留出、45历史non-Bank。',
        '- 原photo和0.01预算项、Adam lr=.001/eps1e-15、训练帧顺序、最终12000步一致，无early stop或评价调参。8/16维各组初始选择概率及补偿相同，第二隐藏层和输出层初始化一致；直接输入组原始前3列、latent前8列权重一致。',
        '- E8共享编码、E50、R3参考复用同地图、同seed、同12000步结果。R3同时有不同的Selector特征与独立补偿参数化，仅作系统目标参考。',
        '- Master冻结，但Selector和Encoder继续依原协议训练；Direct补偿不再向Encoder回传。因此同时改变补偿信息路径、Encoder梯度耦合和从原属性到输出的有效深度。不能只凭质量变化证明32维压缩丢失信息或梯度冲突。',
        '- same-K集合按logit(p)-logit(cached_uniform)排序；不同方案选中身份可不同。effective_proxy是radii>0且opacity≥1/255的点数，不是逐像素积分贡献。',
        '- 本轮为隔离冻结地图GPU实验，生产默认E50及C++/Python生产实现未变，未进行新的在线SLAM。未stage/commit/push。', '',
        '## 输入建议与归因边界', '',
        'Offset区分Anchor内位置；Scale提供形状/物理尺度，Opacity与SH DC提供外观状态，log(ratio)直接提供连续预算条件。DirectRaw将这些旧输入原样送入私有补偿头，避免共享编码器变换。',
        '本轮没有新增Rotation或全局XYZ，也没有逐项拆分11维属性。若Raw优于Offset，只能支持这组附加属性整体有价值，不能归到其中单个输入。', '',
        '![质量曲线](analysis/quality_curves.png)', '',
        '## 固定视图', '',
        '固定中间non-Bank视图、1%与5%，未按改善大小选图。', '',
        '![1%](analysis/1pct_examples.png)', '', '![5%](analysis/5pct_examples.png)', '']
    (ROOT / 'S00_DIRECT_COMPENSATION_AUDIT.md').write_text('\n'.join(lines), encoding='utf-8')
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
        fig, axes = plt.subplots(1, 7, figsize=(21, 3.5))
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
