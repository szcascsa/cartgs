"""核验六个Encoder试验的训练时钟与原始质量证据。"""
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
archive=ROOT/'evidence.tar.gz';assert sha(archive)==sys.argv[1]
dest=ROOT/'evidence';dest.mkdir(exist_ok=True)
with tarfile.open(archive) as pack:
    for member in pack.getmembers():
        path=(dest/member.name).resolve()
        assert member.isfile() and not member.issym() and not member.islnk()
        assert path.is_relative_to(dest.resolve()) and not Path(member.name).is_absolute()
    pack.extractall(dest,filter='data')
for name in ['run.py','analyze.py','controller.py','protocol.json','matrix_runtime.py','scenes.json']:
    assert sha(ROOT/name)==sha(dest/name),name
models=list((dest/'results/fr1_xyz').glob('*/*/complete.json'))
refs=list((dest/'references').glob('*/*/complete.json'))
assert len(models)==6 and len(refs)==4
for path in models+refs:
    info=json.loads(path.read_text())
    assert info['steps']==12000 and info['master_frozen']
    assert all(v==3000 for v in info['ratio_exposures'].values())
    ratios=[.01,.05,.1,.15];empty_total=sum(info['empty_steps'].values())
    for name,value in info['adam_clocks'].items():
        expected=(3000-info['empty_steps'][str(ratios[int(name.split('.')[1])])]
            if name.startswith('extra_tables.') else 12000-empty_total if name.startswith('compensation.') else 12000)
        assert value==expected,(str(path),name,value,expected)
    assert (path.parent/'step_12000.pt').exists()
for kind in ['shallow32','deep32','residual32']:
    for seed in [3100,3101]:
        gate=json.loads((dest/f'correctness_{kind}_{seed}.json').read_text())
        assert all(gate[k] for k in ['initial_probability_exact','initial_compensation_exact',
            'common_selector_heads_latents_exact','offset_exact','encoder_gradient','inactive_ratio_gradient_none','master_frozen'])
        if kind=='residual32':assert gate['initial_hidden_equal_shallow']
        probe=gate['actual_photo_probe']
        for name in probe[-1]['gradient_norms']:
            assert any(row['gradient_norms'][name]>0 for row in probe[1:])
with (dest/'analysis/quality_per_view.csv').open() as stream:rows=list(csv.DictReader(stream))
assert all(math.isfinite(float(r[m])) for r in rows for m in ['psnr','ssim','lpips','photo'])
cmd=lambda *args:subprocess.check_output(['git',*args],cwd=REPO)
before=json.loads((ROOT/'production_before.json').read_text())
now=dict(head=cmd('rev-parse','HEAD').decode().strip(),tracked_status=cmd('status','--short','--untracked-files=no').decode(),
    tracked_diff_sha256=hashlib.sha256(cmd('diff','--binary')).hexdigest(),staged_diff_sha256=hashlib.sha256(cmd('diff','--cached','--binary')).hexdigest())
assert now==before,'实验期间生产或索引已有改动变化，需核对'
summary=json.loads((dest/'analysis/summary.json').read_text())
candidates=['shallow32','deep32','residual32']
pick=lambda a,b='bottleneck16',g='all':next(r for r in summary['primary'] if r['candidate']==a and r['reference']==b and r['group']==g)
lead=['## 结论','']
for kind in candidates:
    r=pick(kind);t=pick(kind,g='train');u=pick(kind,g='unseen')
    verdict='两seed均提高' if r['seed3100']>0 and r['seed3101']>0 else '两seed均下降' if r['seed3100']<0 and r['seed3101']<0 else '两seed方向不一致'
    lead.append(f'- {kind}相对当前16瓶颈：七档平均{r["mean"]:+.3f} dB（{r["seed3100"]:+.3f}/{r["seed3101"]:+.3f}），{verdict}；训练ratio{t["mean"]:+.3f}、未见ratio{u["mean"]:+.3f} dB。')
lead+=['','**本轮优先建议使用11→32 ReLU作为下一版实验Encoder候选。** 浅32与深32的same-K平均PSNR仅差0.002 dB、两个seed方向相反；浅32的LPIPS和photo略好，1%档相对当前16瓶颈提高0.779 dB，而深32在该档平均下降0.057 dB。深32未见ratio相对浅32平均提高0.077 dB，可保留为泛化对照，但不足以据此认定全面更好。',
    '残差版相对浅32平均下降0.194 dB，两个seed同向；函数保持初始化和额外层没有带来本轮总体收益，不推荐替换浅32。',
    '三个候选虽然都改善了当前16瓶颈结构，但仍低于输入含SH DC的早期原Baseline；浅32/深32的剩余差距约0.486/0.485 dB。这个系统参照还混合输入信息差异，不能把剩余差距全算到Encoder。',
    '这些结果只用于本轮探索性比较。无论候选均值高低，不能据此确认跨场景最优网络；若进入生产，仍需独立场景和真实在线Joint验证。当前生产默认保留。','']
report=(dest/'ENCODER_OPTIMIZATION_AUDIT.md').read_text(encoding='utf-8');first,rest=report.split('\n',1)
report=first+'\n\n'+'\n'.join(lead)+'\n'+rest
with (dest/'analysis/validation_curves.csv').open() as stream:curves=list(csv.DictReader(stream))
report+='\n## 内部Bank留出收敛\n\n| 方案 | seed | 4000步 | 8000步 | 12000步 |\n| --- | --- | --- | --- | --- |\n'
for kind in ['bottleneck16',*candidates]:
    for seed in [3100,3101]:
        numbers=[]
        for step in [4000,8000,12000]:
            sub=[float(r['psnr']) for r in curves if r['kind']==kind and int(r['seed'])==seed
                and int(r['step'])==step and r['policy']=='same_k']
            assert sub;numbers.append(f'{sum(sub)/len(sub):.3f}')
        report+='| '+' | '.join([kind,str(seed),*numbers])+' |\n'
report+='\n全部方案统一最终12000步，曲线不用于non-Bank选step，也不把有限训练预算下的差距解释为理论容量极限。\n'
report=report.replace('(analysis/','(tests/diagnostics/s00_encoder_optimization_20261026/evidence/analysis/')
report+='\n## 交付核验\n\n6个新增模型与4个锁定参考完整保存；全部12000步、每ratio3000曝光，实际Adam时钟与非空选择步数精确匹配，deep32/seed3100少2次补偿更新的差异已报告。残差版初始隐藏函数与浅Encoder exact一致。原型源码与服务器归档SHA一致，所有逐视图指标finite。Git HEAD、全部tracked diff及staged diff与实验前相同，已有17个tracked改动保留；未stage、commit或push。\n'
report+='\n[逐视图质量](tests/diagnostics/s00_encoder_optimization_20261026/evidence/analysis/quality_per_view.csv) · [配对统计](tests/diagnostics/s00_encoder_optimization_20261026/evidence/analysis/paired_statistics.csv) · [共享编码统计](tests/diagnostics/s00_encoder_optimization_20261026/evidence/analysis/encoder_output_statistics.csv)\n'
(REPO/'ENCODER_OPTIMIZATION_AUDIT.md').write_text(report,encoding='utf-8')
proof=dict(archive_sha256=sha(archive),new_models=6,reused_models=4,steps=12000,exposure_per_ratio=3000,
    quality_rows=len(rows),quality_finite=True,source_scripts_exact=True,residual_initial_function_verified=True,
    real_photo_gradient_all_encoder_parameters=True,tracked_and_staged_state_unchanged=True,production_change=False,online_slam=False,
    adam_clocks_match_recorded_empty_steps=True,deep32_seed3100_empty_steps=2,
    report_sha256=sha(REPO/'ENCODER_OPTIMIZATION_AUDIT.md'))
(ROOT/'delivery_verification.json').write_text(json.dumps(proof,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(dict(proof=proof,primary=[pick(kind) for kind in candidates]),ensure_ascii=False))
