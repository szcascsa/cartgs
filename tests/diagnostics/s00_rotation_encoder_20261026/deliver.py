"""本地验证服务器原始证据与生产未改动状态，交付中文报告。"""
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
archive=ROOT/'evidence.tar.gz'
assert sha(archive)==sys.argv[1]
dest=ROOT/'evidence'
dest.mkdir(exist_ok=True)
with tarfile.open(archive) as pack:
    for member in pack.getmembers():
        target=(dest/member.name).resolve()
        assert member.isfile() and not member.issym() and not member.islnk()
        assert target.is_relative_to(dest.resolve()) and not Path(member.name).is_absolute()
    pack.extractall(dest,filter='data')
for name in ['run.py','analyze.py','matrix_runtime.py','protocol.json','scenes.json','controller.py']:
    assert sha(ROOT/name)==sha(dest/name),name
models=list((dest/'results/fr1_xyz').glob('*/*/complete.json'))
assert len(models)==4
for path in models:
    info=json.loads(path.read_text())
    assert info['steps']==12000 and info['master_frozen']
    assert all(v==3000 for v in info['ratio_exposures'].values())
    assert all(v==(3000 if k.startswith('extra_tables.') else 12000) for k,v in info['adam_clocks'].items())
    assert (path.parent/'step_12000.pt').exists()
for seed in [3100,3101]:
    gate=json.loads((dest/f'correctness_{seed}.json').read_text())
    assert all(gate[k] for k in ['initial_probabilities_exact','initial_compensation_exact',
        'quaternion_sign_invariant','inactive_ratio_gradient_none','master_frozen'])
with (dest/'analysis/quality_per_view.csv').open() as stream:
    rows=list(csv.DictReader(stream))
assert all(math.isfinite(float(r[m])) for r in rows for m in ['psnr','ssim','lpips','photo'])
cmd=lambda *args:subprocess.check_output(['git',*args],cwd=REPO)
before=json.loads((ROOT/'production_before.json').read_text(encoding='utf-8'))
now=dict(head=cmd('rev-parse','HEAD').decode().strip(),tracked_status=cmd('status','--short','--untracked-files=no').decode(),
    tracked_diff_sha256=hashlib.sha256(cmd('diff','--binary')).hexdigest(),
    staged_diff_sha256=hashlib.sha256(cmd('diff','--cached','--binary')).hexdigest())
assert before==now,'已有生产改动或索引在实验期间变化，需核对来源'
proof=dict(archive_sha256=sha(archive),models=4,seeds=[3100,3101],steps=12000,
    exposure_per_ratio=3000,source_scripts_exact=True,quality_rows=len(rows),quality_finite=True,
    tracked_and_staged_state_unchanged=True,preexisting_dirty_files=len(before['tracked_status'].splitlines()),
    production_change=False,online_slam=False)
(ROOT/'delivery_verification.json').write_text(json.dumps(proof,indent=2,ensure_ascii=False),encoding='utf-8')
report=(dest/'ROTATION_ENCODER_COMPENSATION_AUDIT.md').read_text(encoding='utf-8')
report=report.replace('(analysis/','(tests/diagnostics/s00_rotation_encoder_20261026/evidence/analysis/')
summary=json.loads((dest/'analysis/summary.json').read_text())
p=next(r for r in summary['primary'] if r['group']=='all')
if p['seed3100']>0 and p['seed3101']>0:
    verdict='两个seed的七档平均PSNR均提高；是否超出复跑波动仍需结合幅度和逐ratio结果判断，不据此宣称跨场景稳定。'
elif p['seed3100']<0 and p['seed3101']<0:
    verdict='两个seed的七档平均PSNR均下降；本轮不支持将Rotation同时加入Encoder与全部Head，建议保留59维四Head实验Baseline。'
else:
    verdict='两个seed方向不一致；没有稳定的Rotation输入收益，建议保留59维四Head实验Baseline。'
insert=f'\n**结论：{verdict}**\n\n七档non-Bank same-K平均变化为 **{p["mean"]:+.3f} dB**，两seed分别 **{p["seed3100"]:+.3f}/{p["seed3101"]:+.3f} dB**。\n'
first,rest=report.split('\n',1)
report=first+'\n'+insert+rest
report+='\n## 本地交付核验\n\n'
report+='原型源码与服务器归档SHA一致；4模型各12000步、每ratio3000曝光及Adam时钟通过核对，全部逐视图指标finite。实验开始时已有17个tracked改动，Git HEAD、全部tracked diff和staged diff在实验前后完全一致；未stage、commit或push。\n'
report+='\n[完整逐视图指标](tests/diagnostics/s00_rotation_encoder_20261026/evidence/analysis/quality_per_view.csv) · [配对统计](tests/diagnostics/s00_rotation_encoder_20261026/evidence/analysis/paired_statistics.csv) · [预算与优化器核验](tests/diagnostics/s00_rotation_encoder_20261026/evidence/analysis/fairness_checks.csv)\n'
(REPO/'ROTATION_ENCODER_COMPENSATION_AUDIT.md').write_text(report,encoding='utf-8')
print(json.dumps(dict(proof=proof,primary=summary['primary']),ensure_ascii=False))
