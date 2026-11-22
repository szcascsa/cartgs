"""核对服务器证据包并生成本地中文交付报告。"""
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
assert len(models) == 2
for model in models:
    info = json.loads(model.read_text())
    assert info['steps'] == 12000 and info['master_frozen']
    assert all(value == 3000 for value in info['ratio_exposures'].values())
with (destination / 'analysis/quality_per_view.csv').open() as stream:
    views = list(csv.DictReader(stream))
assert all(math.isfinite(float(row[key])) for row in views for key in ['psnr', 'ssim', 'lpips', 'photo'])
production = json.loads((ROOT / 'production_verification.json').read_text(encoding='utf-8-sig'))
assert not production['production_mismatches'] and not production['staged_files']
with (destination / 'analysis/group_summary.csv').open() as stream:
    groups = {(r['config'], r['group']):r for r in csv.DictReader(stream) if r['split']=='test' and r['policy']=='same_k'}


def score(config, group='all'):
    return float(groups[config, group]['psnr'])


old = 'DirectRawE16'
a, b = 'DirectRawE32', 'RawSharedE16'
lead = f'''## 结果解读

相对上一轮16维ratio特征＋原始11维：

- 32维ratio特征＋原始11维：七档平均变化 **{score(a)-score(old):+.3f} dB**；训练四档 {score(a,'train')-score(old,'train'):+.3f} dB，未见三档 {score(a,'unseen')-score(old,'unseen'):+.3f} dB。
- 16维ratio特征＋原始11维＋共享h32：七档平均变化 **{score(b)-score(old):+.3f} dB**；训练四档 {score(b,'train')-score(old,'train'):+.3f} dB，未见三档 {score(b,'unseen')-score(old,'unseen'):+.3f} dB。

与同seed的E50共享编码方案相比，两组分别变化 {score(a)-score('S00R00E50D12'):+.3f} / {score(b)-score('S00R00E50D12'):+.3f} dB；与R3相比为 {score(a)-score('R3'):+.3f} / {score(b)-score('R3'):+.3f} dB。

本轮只跑一个优化顺序seed和一个冻结地图。它可以描述当前两组改动的质量影响，不能据此认定更高维度或共享表示普遍有益/有害。共享组还恢复了Encoder梯度通路，因此不把其变化单独归为信息压缩问题。生产默认保持E50。

补充判读：32维组虽然PSNR略升，SSIM和photo相比DirectRawE16更差；共享16维组LPIPS改善，但SSIM与photo没有同步改善，不能称为四指标全面提升。逐ratio看，32维组在2.5%比DirectRawE16低约1.036 dB，共享16维组在该档也低约0.405 dB；平均值不能掩盖这一插值档位退化。

本次证据不支持直接删除共享编码，也不支持继续单调扩大latent。加回共享h32有小幅正向信号，但仍不足以证明它是主要质量限制，更不能将共享组的收益全部解释为信息恢复。

'''
name = 'S00_DIRECT_COMPENSATION_EXTRA_AUDIT.md'
original = (destination / name).read_text(encoding='utf-8')
title, body = original.split('\n', 1)
report = title + '\n\n' + lead + body
(destination / name).write_text(report, encoding='utf-8')
local_report = report.replace('(analysis/', '(tests/diagnostics/s00_direct_compensation_extra_20261026/evidence/analysis/')
target = ROOT.parents[2] / name
target.write_text(local_report, encoding='utf-8')
verification = dict(archive_sha256=digest, new_models=2, reused_models=5, steps=12000,
    exposure_per_ratio=3000, scene='fr1_xyz', seed=3100, finite_quality_rows=len(views),
    all_metrics_finite=True, production=production, online_slam=False,
    report_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
(ROOT / 'delivery_verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
print(lead)
print(json.dumps(verification))
