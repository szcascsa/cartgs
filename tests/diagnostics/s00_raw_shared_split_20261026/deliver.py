"""校验证据包，补充四属性分支的中文判读，不改变生产文件。"""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
archive = ROOT / 'evidence.tar.gz'
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
assert digest == sys.argv[1]
destination = ROOT / 'evidence'
destination.mkdir(exist_ok=True)
with tarfile.open(archive) as pack:
    for member in pack.getmembers():
        target = (destination / member.name).resolve()
        assert member.isfile() and not member.issym() and not member.islnk()
        assert target.is_relative_to(destination.resolve()) and not Path(member.name).is_absolute()
    pack.extractall(destination, filter='data')
models = list((destination / 'results/fr1_xyz').glob('*/*/complete.json'))
assert len(models) == 1
info = json.loads(models[0].read_text())
assert info['steps'] == 12000 and info['master_frozen']
assert all(value == 3000 for value in info['ratio_exposures'].values())
with (destination / 'analysis/quality_per_view.csv').open() as stream:
    views = list(csv.DictReader(stream))
assert all(math.isfinite(float(row[key])) for row in views for key in ['psnr', 'ssim', 'lpips', 'photo'])
production = json.loads((ROOT / 'production_verification.json').read_text(encoding='utf-8-sig'))
assert not production['production_mismatches'] and not production['staged_files']
with (destination / 'analysis/group_summary.csv').open() as stream:
    groups = {(r['config'], r['split'], r['policy'], r['group']):r for r in csv.DictReader(stream)}
with (destination / 'analysis/paired_statistics.csv').open() as stream:
    paired = list(csv.DictReader(stream))
with (destination / 'analysis/validation_curves.csv').open() as stream:
    curves = list(csv.DictReader(stream))
with (destination / 'analysis/quality_summary.csv').open() as stream:
    quality = list(csv.DictReader(stream))


def score(config, group='all', metric='psnr', split='test'):
    return float(groups[config, split, 'same_k', group][metric])


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] +
        ['| '+' | '.join(str(value) for value in row)+' |' for row in rows])


a, b = 'RawSharedE16Split4', 'RawSharedE16'
delta = score(a)-score(b)
p = next(r for r in paired if r['config']==a and r['split']=='test' and r['policy']=='same_k' and r['group']=='all' and r['metric']=='psnr')
lead = f'''## 结果解读

相同59维输入下，四分支补偿头七档non-Bank same-K PSNR为 **{score(a):.3f} dB**，统一两层头为 **{score(b):.3f} dB**，变化 **{delta:+.3f} dB**。训练四档变化 {score(a,'train')-score(b,'train'):+.3f} dB，未见三档变化 {score(a,'unseen')-score(b,'unseen'):+.3f} dB。

内部Bank留出七档变化 {score(a,split='validation')-score(b,split='validation'):+.3f} dB。non-Bank逐视图/ratio配对PSNR差值中位数 {float(p['median']):+.3f} dB，p10/p90为 {float(p['p10']):+.3f}/{float(p['p90']):+.3f} dB，改善比例 {100*float(p['improved_fraction']):.1f}%。这些视图不是独立seed重复。

四分支头相比同seed的E50变化 {score(a)-score('S00R00E50D12'):+.3f} dB，相比R3变化 {score(a)-score('R3'):+.3f} dB。只在一个冻结地图上完成一个优化顺序seed，不据此认定稳定收益、容量极限或主要瓶颈；生产默认仍为E50，没有继续自动调整分支、Loss或维度。

'''
convergence = []
for config in [b, a]:
    row = [config]
    for step in [4000, 8000, 12000]:
        values = [float(r['psnr']) for r in curves if r['config']==config and int(r['step'])==step and r['policy']=='same_k']
        assert values
        row.append(f'{sum(values)/len(values):.3f}')
    convergence.append(row)
lead += '## 内部Bank留出收敛曲线\n\n' + table(['方案', '4000步', '8000步', '12000步'], convergence) + '\n\n'
lead += '统一使用最终12000步，不按上表回选checkpoint。固定步数的差异同时包含表示能力与优化条件影响，不等同于充分收敛后的质量上限。\n\n'
same = {(r['config'], float(r['ratio'])):r for r in quality if r['split']=='test' and r['policy']=='same_k'}
ratios = sorted({ratio for config, ratio in same})
assert all(float(same[a, ratio]['psnr']) > float(same[b, ratio]['psnr']) for ratio in ratios)
lead += '## 各ratio的non-Bank same-K结果\n\n' + table(['ratio', 'K', '统一两层PSNR', '四分支PSNR', '差值'],
    [[f'{ratio*100:g}%', same[a,ratio]['selected'], f'{float(same[b,ratio]["psnr"]):.3f}',
      f'{float(same[a,ratio]["psnr"]):.3f}', f'{float(same[a,ratio]["psnr"])-float(same[b,ratio]["psnr"]):+.3f}'] for ratio in ratios]) + '\n\n'
cached = {(r['config'], float(r['ratio'])):r for r in quality if r['split']=='test' and r['policy']=='cached'}
cached_a = float(groups[a, 'test', 'cached', 'all']['psnr'])
cached_b = float(groups[b, 'test', 'cached', 'all']['psnr'])
lead += f'原Cached Bernoulli推理七档平均PSNR为 {cached_a:.3f} dB，对照为 {cached_b:.3f} dB，变化 {cached_a-cached_b:+.3f} dB。实际点数有差异，因此该结果为补充验证，正式预算归因仍以same-K为准。\n\n'
lead += table(['ratio', '统一头Cached K', '四分支Cached K'],
    [[f'{ratio*100:g}%', cached[b,ratio]['selected'], cached[a,ratio]['selected']] for ratio in ratios]) + '\n\n'
lead += '''## 本轮判断

本次四指标与七个ratio的平均PSNR均优于统一两层头，Bank留出与non-Bank收益方向一致；是值得继续确认的正向信号，但仍是同一冻结地图上的一个优化顺序seed，尚未达到生产替换依据。

本次只测量单seed、同预算下拆分四个属性分支的整体效果。新增8928个隐藏参数与改变属性梯度共享同时发生，因此不将质量变化全部归因于某一个因素。独立分支仍共享输入latent16与Encoder，图像损失仍耦合几何和外观，不能称为完全解耦优化。

四分支共享同一16维ratio特征表，所有分支各用两层32维隐藏层。Residual/Other只预测剩余Opacity1+SH DC3，不另行叠加全14维残差。本轮没有增加额外属性Loss或等参数量统一头控制，完成后停止。

'''
name = 'S00_RAW_SHARED_SPLIT_AUDIT.md'
original = (destination / name).read_text(encoding='utf-8')
title, body = original.split('\n', 1)
report = title+'\n\n'+lead+body
(destination / name).write_text(report, encoding='utf-8')
target = ROOT.parents[2] / name
target.write_text(report.replace('(analysis/', '(tests/diagnostics/s00_raw_shared_split_20261026/evidence/analysis/'), encoding='utf-8')
verification = dict(archive_sha256=digest, new_models=1, reused_models=4, steps=12000, exposure_per_ratio=3000,
    scene='fr1_xyz', seed=3100, finite_quality_rows=len(views), all_metrics_finite=True,
    production=production, online_slam=False, report_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
(ROOT / 'delivery_verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
print(json.dumps(verification))
