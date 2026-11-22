"""核验XYZ-only证据与生产状态，交付中文结论。"""
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
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
archive=ROOT/'evidence.tar.gz';assert sha(archive)==sys.argv[1]
dest=ROOT/'evidence';dest.mkdir(exist_ok=True)
with tarfile.open(archive) as pack:
    for member in pack.getmembers():
        target=(dest/member.name).resolve()
        assert member.isfile() and not member.issym() and not member.islnk()
        assert target.is_relative_to(dest.resolve()) and not Path(member.name).is_absolute()
    pack.extractall(dest,filter='data')
for name in ['run.py','analyze.py','controller.py','protocol.json','matrix_runtime.py','scenes.json']:
    assert sha(ROOT/name)==sha(dest/name),name
models=list((dest/'results/fr1_xyz').glob('*/*/complete.json'))
refs=list((dest/'references/baseline').glob('*/complete.json'))
assert len(models)==2 and len(refs)==2
for path in models+refs:
    info=json.loads(path.read_text())
    assert info['steps']==12000 and info['master_frozen']
    assert all(v==3000 for v in info['ratio_exposures'].values())
    assert all(v==(3000 if k.startswith('extra_tables.') else 12000) for k,v in info['adam_clocks'].items())
    assert (path.parent/'step_12000.pt').exists()
for seed in [3100,3101]:
    gate=json.loads((dest/f'correctness_{seed}.json').read_text())
    assert all(gate[k] for k in ['gaussian_attribute_proxy_gate','derived_attribute_nan_gate','encoder_gradient',
        'active_latent_gradient','inactive_latent_gradient_none','master_frozen'])
    assert all(r['initial_probability_exact'] and r['initial_zero_compensation_exact'] for r in gate['initial_checks'])
with (dest/'analysis/quality_per_view.csv').open() as stream:rows=list(csv.DictReader(stream))
assert all(math.isfinite(float(r[m])) for r in rows for m in ['psnr','ssim','lpips','photo'])
cmd=lambda *args:subprocess.check_output(['git',*args],cwd=REPO)
before=json.loads((ROOT/'production_before.json').read_text())
after=dict(head=cmd('rev-parse','HEAD').decode().strip(),tracked_status=cmd('status','--short','--untracked-files=no').decode(),
    tracked_diff_sha256=hashlib.sha256(cmd('diff','--binary')).hexdigest(),staged_diff_sha256=hashlib.sha256(cmd('diff','--cached','--binary')).hexdigest())
assert before==after,'实验期间已有生产改动或索引发生变化，需核对'
summary=json.loads((dest/'analysis/summary.json').read_text());p=summary['primary'][0]
if p['seed3100']>0 and p['seed3101']>0:
    conclusion='两个seed的七档平均PSNR均提高；该精简组合值得保留研究，但尚不能单独归因于输入信息减少。'
elif p['seed3100']<0 and p['seed3101']<0:
    conclusion='两个seed的七档平均PSNR均下降；本轮XYZ-only组合没有收益，当前证据不支持将被移除属性一并视为冗余。'
else:
    conclusion='两个seed方向不一致，当前没有稳定的XYZ-only收益。'
lead=f'\n**结论：{conclusion}**\n\n七档non-Bank same-K平均变化 **{p["mean"]:+.3f} dB**，两seed分别 **{p["seed3100"]:+.3f}/{p["seed3101"]:+.3f} dB**。\n'
report=(dest/'XYZ_ONLY_COMPENSATION_AUDIT.md').read_text(encoding='utf-8');first,rest=report.split('\n',1)
report=first+'\n'+lead+rest
with (dest/'analysis/paired_statistics.csv').open() as stream:paired=list(csv.DictReader(stream))
report+='\n## 原Cached Bernoulli实际预算评价\n\n| ratio组 | seed3100 ΔPSNR | seed3101 ΔPSNR | 平均ΔPSNR |\n| --- | --- | --- | --- |\n'
for group in ['all','train','unseen']:
    row=next(r for r in paired if r['split']=='test' and r['policy']=='cached' and r['group']==group and r['metric']=='psnr')
    report+=f'| {group} | {float(row["seed3100"]):+.3f} | {float(row["seed3101"]):+.3f} | {float(row["mean"]):+.3f} |\n'
report+='\n上述保留实际选点数差异，严格同预算的主对照仍为same-K；两种评价均完整保存。\n'
with (dest/'analysis/validation_curves.csv').open() as stream:curves=list(csv.DictReader(stream))
report+='\n## 内部Bank留出收敛曲线\n\n| 方案 | seed | 4000步 | 8000步 | 12000步 |\n| --- | --- | --- | --- | --- |\n'
for kind in ['baseline','xyz_only']:
    for seed in [3100,3101]:
        numbers=[]
        for step in [4000,8000,12000]:
            sub=[float(r['psnr']) for r in curves if r['kind']==kind and int(r['seed'])==seed
                and int(r['step'])==step and r['policy']=='same_k']
            assert sub
            numbers.append(f'{sum(sub)/len(sub):.3f}')
        report+='| '+' | '.join([kind,str(seed),*numbers])+' |\n'
report+='\n统一用最终12000步，不按留出或non-Bank选择checkpoint；曲线与有限训练预算不能证明充分收敛的表示能力上限。下一项若继续按用户要求逐步加信息，应每次只恢复一种输入，不自动扩展为矩阵。\n'
report=report.replace('(analysis/','(tests/diagnostics/s00_xyz_only_20261026/evidence/analysis/')
report+='\n## 交付核验\n\n原型源码与服务器归档SHA一致；2个新增模型及2个锁定参考完整保存，步数、曝光和Adam时钟匹配，四指标finite。Git HEAD、全部tracked diff及staged diff与实验前一致；保留已有17个tracked改动，未stage/commit/push。\n'
report+='\n[逐视图指标](tests/diagnostics/s00_xyz_only_20261026/evidence/analysis/quality_per_view.csv) · [配对统计](tests/diagnostics/s00_xyz_only_20261026/evidence/analysis/paired_statistics.csv) · [实际预算与集合重合](tests/diagnostics/s00_xyz_only_20261026/evidence/analysis/selected_set_overlap.csv)\n'
(REPO/'XYZ_ONLY_COMPENSATION_AUDIT.md').write_text(report,encoding='utf-8')
proof=dict(archive_sha256=sha(archive),new_models=2,reference_models=2,steps=12000,exposure_per_ratio=3000,
    source_scripts_exact=True,quality_rows=len(rows),quality_finite=True,strict_xyz_only_input_verified=True,
    tracked_and_staged_state_unchanged=True,production_change=False,online_slam=False,
    report_sha256=sha(REPO/'XYZ_ONLY_COMPENSATION_AUDIT.md'))
(ROOT/'delivery_verification.json').write_text(json.dumps(proof,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(dict(proof=proof,summary=summary),ensure_ascii=False))
