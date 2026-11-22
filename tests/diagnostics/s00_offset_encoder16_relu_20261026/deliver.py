"""核验更正后每层ReLU的训练证据，保留旧任务但不混入结果。"""
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
assert len(models)==2 and len(refs)==4
for path in models+refs:
    info=json.loads(path.read_text())
    assert info['steps']==12000 and info['master_frozen']
    assert all(v==3000 for v in info['ratio_exposures'].values())
    assert all(v==(3000 if k.startswith('extra_tables.') else 12000) for k,v in info['adam_clocks'].items())
    assert (path.parent/'step_12000.pt').exists()
for seed in [3100,3101]:
    g=json.loads((dest/f'correctness_{seed}.json').read_text())
    assert g['encoder_layout']=='Linear11-16,ReLU,Linear16-32,ReLU,Linear32-32,ReLU'
    assert all(g[k] for k in ['initial_probability_exact','initial_compensation_exact','offset_exact',
        'allowed_attribute_proxy','forbidden_derived_attribute_nan','encoder_all_layers_gradient',
        'offset_input_gradient','inactive_ratio_gradient_none','master_frozen'])
with (dest/'analysis/quality_per_view.csv').open() as stream:rows=list(csv.DictReader(stream))
assert all(math.isfinite(float(r[m])) for r in rows for m in ['psnr','ssim','lpips','photo'])
cmd=lambda *args:subprocess.check_output(['git',*args],cwd=REPO)
before=json.loads((ROOT/'production_before.json').read_text())
now=dict(head=cmd('rev-parse','HEAD').decode().strip(),tracked_status=cmd('status','--short','--untracked-files=no').decode(),
    tracked_diff_sha256=hashlib.sha256(cmd('diff','--binary')).hexdigest(),staged_diff_sha256=hashlib.sha256(cmd('diff','--cached','--binary')).hexdigest())
assert now==before,'实验期间生产或索引已有改动变化，需核对'
summary=json.loads((dest/'analysis/summary.json').read_text())
pick=lambda ref:next(r for r in summary['primary'] if r['reference']==ref and r['group']=='all')
p,b=pick('xyz_scale_opacity'),pick('baseline')
if p['seed3100']>0 and p['seed3101']>0:verdict='相对上一组两seed均提高。'
elif p['seed3100']<0 and p['seed3101']<0:verdict='相对上一组两seed均下降。'
else:verdict='相对上一组两seed方向不一致。'
lead=f'\n**结论：{verdict}** 七档non-Bank same-K相对上一组平均 **{p["mean"]:+.3f} dB**，相对原Baseline **{b["mean"]:+.3f} dB**。\n\n本轮结构已经修正为 **11→16→ReLU→32→ReLU→32→ReLU**。先前缺少中间ReLU的运行已停止并排除，不计入本报告。\n'
report=(dest/'OFFSET_ENCODER16_RELU_COMPENSATION_AUDIT.md').read_text(encoding='utf-8')
first,rest=report.split('\n',1);report=first+'\n'+lead+rest
with (dest/'analysis/validation_curves.csv').open() as stream:curves=list(csv.DictReader(stream))
report+='\n## 内部Bank留出收敛\n\n| 方案 | seed | 4000步 | 8000步 | 12000步 |\n| --- | --- | --- | --- | --- |\n'
for kind in ['baseline','xyz_scale_opacity','offset_encoder16']:
    for seed in [3100,3101]:
        scores=[]
        for step in [4000,8000,12000]:
            sub=[float(r['psnr']) for r in curves if r['kind']==kind and int(r['seed'])==seed
                and int(r['step'])==step and r['policy']=='same_k']
            assert sub;scores.append(f'{sum(sub)/len(sub):.3f}')
        report+='| '+' | '.join([kind,str(seed),*scores])+' |\n'
report+='\n使用固定最终12000步，没有按non-Bank选LR/step；有限预算结果不能直接等同表达能力上限。\n'
report=report.replace('(analysis/','(tests/diagnostics/s00_offset_encoder16_relu_20261026/evidence/analysis/')
report+='\n## 本地交付核验\n\n更正后的2个新增模型与4个参照模型完整保存，全部12000步、每ratio3000曝光和Adam时钟匹配。源码与服务器归档SHA一致，逐视图四指标finite；Git HEAD、全部tracked diff和staged diff与实验前完全一致，已有17个tracked改动保留。未stage、commit或push。\n'
report+='\n[逐视图指标](tests/diagnostics/s00_offset_encoder16_relu_20261026/evidence/analysis/quality_per_view.csv) · [配对统计](tests/diagnostics/s00_offset_encoder16_relu_20261026/evidence/analysis/paired_statistics.csv) · [预算与时钟核验](tests/diagnostics/s00_offset_encoder16_relu_20261026/evidence/analysis/fairness_checks.csv)\n'
(REPO/'OFFSET_ENCODER16_RELU_COMPENSATION_AUDIT.md').write_text(report,encoding='utf-8')
proof=dict(archive_sha256=sha(archive),new_models=2,reused_models=4,steps=12000,exposure_per_ratio=3000,
    corrected_three_relu_layout_verified=True,source_scripts_exact=True,quality_rows=len(rows),quality_finite=True,
    tracked_and_staged_state_unchanged=True,production_change=False,online_slam=False,
    report_sha256=sha(REPO/'OFFSET_ENCODER16_RELU_COMPENSATION_AUDIT.md'))
(ROOT/'delivery_verification.json').write_text(json.dumps(proof,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(dict(proof=proof,primary=summary['primary']),ensure_ascii=False))
