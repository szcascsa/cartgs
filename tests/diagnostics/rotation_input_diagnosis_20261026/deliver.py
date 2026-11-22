"""验证并交付Rotation输入退化研究的原始证据。"""
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
archive=ROOT/'evidence.tar.gz'
assert sha(archive)==sys.argv[1]
dest=ROOT/'evidence';dest.mkdir(exist_ok=True)
with tarfile.open(archive) as pack:
    for member in pack.getmembers():
        path=(dest/member.name).resolve()
        assert member.isfile() and not member.issym() and not member.islnk()
        assert path.is_relative_to(dest.resolve()) and not Path(member.name).is_absolute()
    pack.extractall(dest,filter='data')
for name in ['probe.py','train.py','summarize.py','controller.py','diagnostic_protocol.json',
    'representation_protocol.json','matrix_runtime.py','scenes.json']:
    assert sha(ROOT/name)==sha(dest/name),name
models=list((dest/'results/fr1_xyz').glob('*/*/complete.json'))
references=list((dest/'references').glob('*/*/complete.json'))
assert len(models)==4 and len(references)==4
for path in models+references:
    info=json.loads(path.read_text())
    assert info['steps']==12000 and info['master_frozen']
    assert all(v==3000 for v in info['ratio_exposures'].values())
    assert all(v==(3000 if k.startswith('extra_tables.') else 12000) for k,v in info['adam_clocks'].items())
    assert (path.parent/'step_12000.pt').exists()
for path in dest.glob('correctness_*.json'):
    gate=json.loads(path.read_text())
    assert all(gate[k] for k in ['initial_probability_exact','initial_compensation_exact',
        'rotation_path_gradient','inactive_ratio_gradient_none','master_frozen'])
with (dest/'analysis/quality_per_view.csv').open() as stream:rows=list(csv.DictReader(stream))
assert all(math.isfinite(float(r[m])) for r in rows for m in ['psnr','ssim','lpips','photo'])
cmd=lambda *args:subprocess.check_output(['git',*args],cwd=REPO)
before=json.loads((ROOT/'production_before.json').read_text())
after=dict(head=cmd('rev-parse','HEAD').decode().strip(),tracked_status=cmd('status','--short','--untracked-files=no').decode(),
    tracked_diff_sha256=hashlib.sha256(cmd('diff','--binary')).hexdigest(),
    staged_diff_sha256=hashlib.sha256(cmd('diff','--cached','--binary')).hexdigest())
assert before==after,'生产或索引已有改动在实验期间发生变化，需核对'
summary=json.loads((dest/'analysis/summary.json').read_text())
pick=lambda kind,ref='baseline',group='all':next(r for r in summary['primary'] if r['candidate']==kind
    and r['reference']==ref and r['group']==group)
lines=['## 结论与建议','',
    '输入预处理值得按几何语义验证，直接对Quaternion分量取log不合适。现有证据能确认等价表示与选点/补偿适配问题，但尚不能把正常评价退化唯一归于某一因素。','']
for kind in ['rot6','cov6']:
    r=pick(kind);rq=pick(kind,'rotation_all')
    lines.append(f'- {kind}相对无Rotation Baseline：两seed {r["seed3100"]:+.3f}/{r["seed3101"]:+.3f} dB，平均{r["mean"]:+.3f} dB；相对Quaternion平均{rq["mean"]:+.3f} dB。')
stable=[kind for kind in ['rot6','cov6'] if pick(kind)['seed3100']>0 and pick(kind)['seed3101']>0]
if stable:
    lines+=['','有处理方式在两个seed上超过Baseline，但仍需结合未见ratio和幅度解释；本轮单一历史场景不足以替换生产或确立最优结构。']
else:
    lines+=['','本轮没有任何处理方式在两个seed上都超过无Rotation Baseline，建议继续保留59维四Head实验Baseline，不自动扩展编码/网络搜索。']
lines+=['','Quaternion在本轮固定地图训练中本身冻结，微扰符号跳变是表示诊断，不能描述成训练期间实际发生了这些跳变。其范围小，暂不支持将其认定为主要退化原因。',
    '原始输入尺度有界；最终梯度快照没有显示Rotation使方向冲突更严重。跨模块交换揭示支持集适配，但其分布外性质限制了纯因果分解。','']
report=(dest/'ROTATION_INPUT_DIAGNOSIS.md').read_text(encoding='utf-8')
head,rest=report.split('\n',1)
report=head+'\n\n'+'\n'.join(lines)+'\n'+rest
report=report.replace('(analysis/','(tests/diagnostics/rotation_input_diagnosis_20261026/evidence/analysis/')
report+='\n## 本地交付核验\n\n'
report+='4个新增模型与4个原始参考完整保存，全部12000步、每ratio3000曝光及Adam时钟一致；原型源码与归档SHA一致，逐视图四指标finite，Git HEAD、全部tracked diff和staged diff与实验前一致。没有修改生产默认或执行stage/commit/push。\n'
(REPO/'ROTATION_INPUT_DIAGNOSIS.md').write_text(report,encoding='utf-8')
proof=dict(archive_sha256=sha(archive),new_models=4,reference_models=4,quality_rows=len(rows),
    finite=True,source_scripts_exact=True,steps=12000,exposure_per_ratio=3000,
    tracked_and_staged_state_unchanged=True,no_production_change=True,online_slam=False,
    report_sha256=sha(REPO/'ROTATION_INPUT_DIAGNOSIS.md'))
(ROOT/'delivery_verification.json').write_text(json.dumps(proof,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(dict(proof=proof,primary=[pick(kind) for kind in ['rotation_all','rot6','cov6']]),ensure_ascii=False))
