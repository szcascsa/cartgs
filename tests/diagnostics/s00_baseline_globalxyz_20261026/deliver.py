"""下载后核验质量证据、训练公平性与生产Git状态。"""
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
assert len(models)==2 and len(refs)==2
records=[]
for path in models+refs:
    info=json.loads(path.read_text());assert info['steps']==12000 and info['master_frozen']
    assert all(v==3000 for v in info['ratio_exposures'].values())
    ratios=[.01,.05,.1,.15];empty_total=sum(info['empty_steps'].values())
    for name,value in info['adam_clocks'].items():
        expected=(3000-info['empty_steps'][str(ratios[int(name.split('.')[1])])]
            if name.startswith('extra_tables.') else 12000-empty_total if name.startswith('compensation.') else 12000)
        assert value==expected,(str(path),name,value,expected)
    assert (path.parent/'step_12000.pt').exists()
    records.append(dict(kind=path.parts[-3],seed=info['seed'],empty_steps=info['empty_steps'],
        compensation_updates=12000-empty_total,parameters=info['parameters']))
for seed in [3100,3101]:
    gate=json.loads((dest/f'correctness_{seed}.json').read_text())
    assert gate['input_width']==14 and gate['head_input_width']==62
    assert all(gate[k] for k in ['common_weights_exact','new_xyz_columns_zero','original_raw11_exact',
        'global_xyz_exact','initial_probability_exact','initial_compensation_exact',
        'inactive_ratio_gradient_none','master_frozen'])
    assert gate['initial_hidden_max_error']<2e-6 and gate['nonzero_output_probe_max_error']<2e-6
    probe=gate['actual_photo_probe']
    assert all(any(row['xyz_gradient_norms'][k]>0 for row in probe[1:]) for k in probe[-1]['xyz_gradient_norms'])
with (dest/'analysis/quality_per_view.csv').open() as stream:rows=list(csv.DictReader(stream))
assert all(math.isfinite(float(r[m])) for r in rows for m in ['psnr','ssim','lpips','photo'])
cmd=lambda *args:subprocess.check_output(['git',*args],cwd=REPO)
before=json.loads((ROOT/'production_before.json').read_text())
now=dict(head=cmd('rev-parse','HEAD').decode().strip(),tracked_status=cmd('status','--short','--untracked-files=no').decode(),
    tracked_diff_sha256=hashlib.sha256(cmd('diff','--binary')).hexdigest(),staged_diff_sha256=hashlib.sha256(cmd('diff','--cached','--binary')).hexdigest())
assert now==before,'实验期间生产改动或索引变化，需核对'
report=(dest/'BASELINE_GLOBAL_XYZ_AUDIT.md').read_text(encoding='utf-8')
summary=json.loads((dest/'analysis/summary.json').read_text())
primary=summary['primary'][0]
if primary['seed3100']<0 and primary['seed3101']<0:
    report=report.replace('## 结构与严格初始化对齐',
        '**不建议将全局XYZ加入当前Baseline。** 两seed在七档ratio的均值均下降；Bank训练视图平均下降0.354 dB，non-Bank下降0.377 dB，因此此次退化不只发生于未见视图。原Cached实际数量评价同样下降0.296 dB。\n\n'
        '本次排除了上一组原属性初始化不同的混淆，支持“当前固定训练预算及原始XYZ表示下，新增XYZ没有质量收益”。但仍不能断言世界坐标在所有表达或训练方式下都无用：新增连接改变优化轨迹，Encoder和Head同步接入，且Selector集合随训练变化。本轮未隔离这些细分因素，也未测试坐标归一化或更长训练。\n\n'
        '## 结构与严格初始化对齐')
with (dest/'analysis/validation_curves.csv').open() as stream:curves=list(csv.DictReader(stream))
report+='\n## 内部Bank留出收敛\n\n| 方案 | seed | 4000步 | 8000步 | 12000步 |\n| --- | --- | --- | --- | --- |\n'
for kind in ['original_baseline','baseline_xyz']:
    for seed in [3100,3101]:
        values=[]
        for step in [4000,8000,12000]:
            sub=[float(row['psnr']) for row in curves if row['kind']==kind and int(row['seed'])==seed
                and int(row['step'])==step and row['policy']=='same_k']
            assert sub;values.append(f'{sum(sub)/len(sub):.3f}')
        report+='| '+' | '.join([kind,str(seed),*values])+' |\n'
report+='\n## 交付核验\n\n两新增模型和两锁定参照均完成12000步、每ratio3000次曝光。step0及新增XYZ梯度检查通过；原始逐视图四指标finite，实际Adam时钟与空选择记录匹配。源码与下载归档SHA一致，Git HEAD、tracked diff及staged diff与实验前完全相同；未stage、commit或push。\n'
report+='\n实际更新记录：'+json.dumps(records,ensure_ascii=False)+'\n'
report+='\n协议文字勘误：归档protocol的source描述沿用“四个参照”旧文字；实际目录、分析和本报告均为两个原Baseline参照（每seed一个）。仅文字计数错误，不影响预定训练、模型或统计。见实验目录protocol_metadata_erratum.json。\n'
report=report.replace('(analysis/','(tests/diagnostics/s00_baseline_globalxyz_20261026/evidence/analysis/')
(REPO/'BASELINE_GLOBAL_XYZ_AUDIT.md').write_text(report,encoding='utf-8')
proof=dict(archive_sha256=sha(archive),new_models=2,reused_models=2,steps=12000,exposure_per_ratio=3000,
    source_scripts_exact=True,step0_and_real_photo_xyz_gradients_verified=True,quality_rows=len(rows),quality_finite=True,
    adam_clocks_match_recorded_empty_steps=True,update_records=records,tracked_and_staged_state_unchanged=True,
    production_change=False,online_slam=False,report_sha256=sha(REPO/'BASELINE_GLOBAL_XYZ_AUDIT.md'))
(ROOT/'delivery_verification.json').write_text(json.dumps(proof,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(dict(proof=proof,primary=summary['primary']),ensure_ascii=False))
