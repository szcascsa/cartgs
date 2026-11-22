"""校验证据包，补充加深头的中文判读，不改变生产文件。"""
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


def score(config, group='all', metric='psnr', split='test'):
    return float(groups[config, split, 'same_k', group][metric])


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] +
        ['| '+' | '.join(str(value) for value in row)+' |' for row in rows])


a, b = 'RawSharedE16D3', 'RawSharedE16'
delta = score(a)-score(b)
p = next(r for r in paired if r['config']==a and r['split']=='test' and r['policy']=='same_k' and r['group']=='all' and r['metric']=='psnr')
lead = f'''## 结果解读

相同59维输入下，三层补偿头七档non-Bank same-K PSNR为 **{score(a):.3f} dB**，两层头为 **{score(b):.3f} dB**，变化 **{delta:+.3f} dB**。训练四档变化 {score(a,'train')-score(b,'train'):+.3f} dB，未见三档变化 {score(a,'unseen')-score(b,'unseen'):+.3f} dB。

内部Bank留出七档变化 {score(a,split='validation')-score(b,split='validation'):+.3f} dB。non-Bank逐视图/ratio配对PSNR差值中位数 {float(p['median']):+.3f} dB，p10/p90为 {float(p['p10']):+.3f}/{float(p['p90']):+.3f} dB，改善比例 {100*float(p['improved_fraction']):.1f}%。这些视图不是独立seed重复。

三层头相比同seed的E50变化 {score(a)-score('S00R00E50D12'):+.3f} dB，相比R3变化 {score(a)-score('R3'):+.3f} dB。只在一个冻结地图上完成一个优化顺序seed，不据此认定稳定收益、容量极限或主要瓶颈；生产默认仍为E50，没有继续自动加层或扩大维度。

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
lead += '''## 本轮判断

加深一层没有形成明显的总体PSNR收益，主要是训练ratio与未见ratio之间的质量权衡。七档平均SSIM、LPIPS和photo略有改善，但PSNR差值中位数为负，不能解释为多数视图全面改善。该结果不支持立即以三层头替换当前默认，也不足以排除其他参数化或优化条件下的容量收益。

本轮只检验两层→三层、宽度32、单位初始化的这一种改法；没有同步加宽、改激活、改Loss或增加训练步数，完成后停止。

'''
name = 'S00_RAW_SHARED_DEPTH_AUDIT.md'
original = (destination / name).read_text(encoding='utf-8')
title, body = original.split('\n', 1)
report = title+'\n\n'+lead+body
(destination / name).write_text(report, encoding='utf-8')
target = ROOT.parents[2] / name
target.write_text(report.replace('(analysis/', '(tests/diagnostics/s00_raw_shared_depth_20261026/evidence/analysis/'), encoding='utf-8')
verification = dict(archive_sha256=digest, new_models=1, reused_models=4, steps=12000, exposure_per_ratio=3000,
    scene='fr1_xyz', seed=3100, finite_quality_rows=len(views), all_metrics_finite=True,
    production=production, online_slam=False, report_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
(ROOT / 'delivery_verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
print(json.dumps(verification))
