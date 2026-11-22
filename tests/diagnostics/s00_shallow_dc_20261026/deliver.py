"""核验浅Encoder加回DC的质量与是否达到原Baseline。"""
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
models=list((dest/'results/fr1_xyz').glob('*/*/complete.json'));refs=list((dest/'references').glob('*/*/complete.json'))
assert len(models)==2 and len(refs)==4
empty_records=[]
for path in models+refs:
    info=json.loads(path.read_text());assert info['steps']==12000 and info['master_frozen']
    assert all(v==3000 for v in info['ratio_exposures'].values())
    ratios=[.01,.05,.1,.15];empty_total=sum(info['empty_steps'].values())
    for name,value in info['adam_clocks'].items():
        expected=(3000-info['empty_steps'][str(ratios[int(name.split('.')[1])])]
            if name.startswith('extra_tables.') else 12000-empty_total if name.startswith('compensation.') else 12000)
        assert value==expected,(str(path),name,value,expected)
    assert (path.parent/'step_12000.pt').exists()
    if path in models:empty_records.append(dict(seed=info['seed'],empty_steps=info['empty_steps'],compensation_updates=12000-empty_total))
for seed in [3100,3101]:
    gate=json.loads((dest/f'correctness_{seed}.json').read_text())
    assert gate['input_width']==14 and gate['head_input_width']==62
    assert all(gate[k] for k in ['initial_probability_exact','initial_compensation_exact','raw_dc_exact',
        'allowed_attribute_proxy','forbidden_derived_attribute_nan','dc_input_gradient','inactive_ratio_gradient_none','master_frozen'])
    probe=gate['actual_photo_probe']
    assert all(any(row['dc_gradient_norms'][k]>0 for row in probe[1:]) for k in probe[-1]['dc_gradient_norms'])
with (dest/'analysis/quality_per_view.csv').open() as stream:rows=list(csv.DictReader(stream))
assert all(math.isfinite(float(r[m])) for r in rows for m in ['psnr','ssim','lpips','photo'])
cmd=lambda *args:subprocess.check_output(['git',*args],cwd=REPO)
before=json.loads((ROOT/'production_before.json').read_text())
now=dict(head=cmd('rev-parse','HEAD').decode().strip(),tracked_status=cmd('status','--short','--untracked-files=no').decode(),
    tracked_diff_sha256=hashlib.sha256(cmd('diff','--binary')).hexdigest(),staged_diff_sha256=hashlib.sha256(cmd('diff','--cached','--binary')).hexdigest())
assert now==before,'实验期间已有生产改动或索引变化，需核对'
summary=json.loads((dest/'analysis/summary.json').read_text())
pick=lambda ref,g='all':next(r for r in summary['primary'] if r['reference']==ref and r['group']==g)
r,b=pick('shallow_no_dc'),pick('original_baseline')
describe=lambda p:'两seed均提高' if p['seed3100']>0 and p['seed3101']>0 else '两seed均下降' if p['seed3100']<0 and p['seed3101']<0 else '两seed方向不一致'
lead=['## 结论','',f'- 加回DC相对不含DC浅层：{describe(r)}，七档non-Bank same-K平均{r["mean"]:+.3f} dB（{r["seed3100"]:+.3f}/{r["seed3101"]:+.3f}）。',
    f'- 相对原Baseline：{describe(b)}，平均{b["mean"]:+.3f} dB（{b["seed3100"]:+.3f}/{b["seed3101"]:+.3f}）。',
    f'- 相对原Baseline，训练ratio平均{pick("original_baseline","train")["mean"]:+.3f} dB、未见ratio平均{pick("original_baseline","unseen")["mean"]:+.3f} dB。','',
    '这是两个seed、同一冻结历史地图上的探索性结果，不能仅凭均值决定跨场景最优结构或声称在线Joint提升。原生产默认保留，没有根据本轮结果继续调整网络或训练。','']
if b['seed3100']<0 and b['seed3101']<0:
    lead+=['**本轮没有恢复或超过原Baseline。** 若加回DC的两个seed方向不一致，不能把很小的平均正差当成稳定收益；原Cached实际预算结果与same-K结果同时报告。本轮不据此替换生产默认。','']
report=(dest/'SHALLOW_DC_COMPENSATION_AUDIT.md').read_text(encoding='utf-8');first,rest=report.split('\n',1)
report=first+'\n\n'+'\n'.join(lead)+'\n'+rest
with (dest/'analysis/validation_curves.csv').open() as stream:curves=list(csv.DictReader(stream))
report+='\n## 内部Bank留出收敛\n\n| 方案 | seed | 4000步 | 8000步 | 12000步 |\n| --- | --- | --- | --- | --- |\n'
for kind in ['original_baseline','shallow_no_dc','shallow_dc']:
    for seed in [3100,3101]:
        values=[]
        for step in [4000,8000,12000]:
            sub=[float(row['psnr']) for row in curves if row['kind']==kind and int(row['seed'])==seed
                and int(row['step'])==step and row['policy']=='same_k']
            assert sub;values.append(f'{sum(sub)/len(sub):.3f}')
        report+='| '+' | '.join([kind,str(seed),*values])+' |\n'
report+='\n全部使用固定最终12000步，有限预算差异不当作表示能力上限。\n'
report=report.replace('(analysis/','(tests/diagnostics/s00_shallow_dc_20261026/evidence/analysis/')
report+='\n## 交付核验\n\n2个新增模型、4个锁定参照均完整保存，全部12000步、每ratio3000次曝光。实际Adam时钟与非空集合记录精确匹配；DC输入、step0函数及真实photo梯度通过检查，源码与服务器归档SHA一致，逐视图四指标finite。Git HEAD、全部tracked diff和staged diff与实验前完全相同，已有17个tracked改动保留；未stage、commit或push。\n'
report+='\n新增模型空选择与补偿更新记录：'+json.dumps(empty_records,ensure_ascii=False)+'\n'
report+='\n[逐视图质量](tests/diagnostics/s00_shallow_dc_20261026/evidence/analysis/quality_per_view.csv) · [配对统计](tests/diagnostics/s00_shallow_dc_20261026/evidence/analysis/paired_statistics.csv) · [预算与时钟](tests/diagnostics/s00_shallow_dc_20261026/evidence/analysis/fairness_checks.csv)\n'
(REPO/'SHALLOW_DC_COMPENSATION_AUDIT.md').write_text(report,encoding='utf-8')
proof=dict(archive_sha256=sha(archive),new_models=2,reused_models=4,steps=12000,exposure_per_ratio=3000,
    source_scripts_exact=True,dc_input_and_real_photo_gradients_verified=True,quality_rows=len(rows),quality_finite=True,
    adam_clocks_match_recorded_empty_steps=True,empty_records=empty_records,tracked_and_staged_state_unchanged=True,
    production_change=False,online_slam=False,report_sha256=sha(REPO/'SHALLOW_DC_COMPENSATION_AUDIT.md'))
(ROOT/'delivery_verification.json').write_text(json.dumps(proof,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(dict(proof=proof,primary=summary['primary']),ensure_ascii=False))
