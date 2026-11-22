"""核对48维输入实验交付，生成中文结果判读。"""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parent
archive=ROOT/'evidence.tar.gz'
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
assert digest==sys.argv[1]
dest=ROOT/'evidence'
dest.mkdir(exist_ok=True)
with tarfile.open(archive) as pack:
    for member in pack.getmembers():
        target=(dest/member.name).resolve()
        assert member.isfile() and not member.issym() and not member.islnk()
        assert target.is_relative_to(dest.resolve()) and not Path(member.name).is_absolute()
    pack.extractall(dest,filter='data')
models=list((dest/'results/fr1_xyz').glob('*/*/complete.json'))
assert len(models)==2
for path in models:
    info=json.loads(path.read_text())
    assert info['steps']==12000 and info['master_frozen']
    assert all(value==3000 for value in info['ratio_exposures'].values())
with (dest/'analysis/quality_per_view.csv').open() as stream:
    views=list(csv.DictReader(stream))
assert all(math.isfinite(float(r[m])) for r in views for m in ['psnr','ssim','lpips','photo'])
with (dest/'analysis/paired_statistics.csv').open() as stream:
    pairs=list(csv.DictReader(stream))
with (dest/'analysis/group_summary.csv').open() as stream:
    groups=list(csv.DictReader(stream))
with (dest/'analysis/validation_curves.csv').open() as stream:
    curves=list(csv.DictReader(stream))
production=json.loads((ROOT/'production_verification.json').read_text(encoding='utf-8-sig'))
assert not production['production_mismatches'] and not production['staged_files']


def metric(group='all',split='test',policy='same_k',m='psnr'):
    return next(r for r in pairs if r['group']==group and r['split']==split and r['policy']==policy and r['metric']==m)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


primary=metric()
lead=f'''# 四Head仅输入16维ratio特征＋共享32维的消融

补偿头输入59→48维，删除原始11维直连，仍保留16维Anchor/ratio latent与共享h32。四个Head仍各两层32维隐藏层，输出XYZ3/Scale3/Rotation4/Other4。

## 主要结果

相对59维四Head，48维方案七档non-Bank same-K平均变化 **{float(primary['mean']):+.3f} dB**，两个seed分别 **{float(primary['seed3100']):+.3f}/{float(primary['seed3101']):+.3f} dB**。训练四档平均 {float(metric('train')['mean']):+.3f} dB，未见三档平均 {float(metric('unseen')['mean']):+.3f} dB。

内部Bank留出七档变化 {float(metric(split='validation')['mean']):+.3f} dB，原Cached Bernoulli推理变化 {float(metric(policy='cached')['mean']):+.3f} dB；Cached实际K可不同，正式预算比较使用same-K。

逐视图/ratio配对PSNR差值中位数 {float(primary['median']):+.3f} dB，p10/p90为 {float(primary['p10']):+.3f}/{float(primary['p90']):+.3f} dB，改善比例 {100*float(primary['improved_fraction']):.1f}%。这些视图不是独立seed重复。

'''
lead+='## 两seed等权四指标\n\n'
average=[]
for kind in ['full59','compact48']:
    subset=[r for r in groups if r['kind']==kind and r['split']=='test' and r['policy']=='same_k' and r['group']=='all']
    assert len(subset)==2
    average.append([kind,*[f'{sum(float(r[m]) for r in subset)/2:.5f}' for m in ['psnr','ssim','lpips','photo']]])
lead+=table(['输入','PSNR↑','SSIM↑','LPIPS↓','photo↓'],average)+'\n\n'
lead+='## 各seed四指标\n\n'+table(['输入','seed','PSNR↑','SSIM↑','LPIPS↓','photo↓'],
    [[r['kind'],r['seed'],*[f'{float(r[m]):.5f}' for m in ['psnr','ssim','lpips','photo']]] for r in groups
     if r['split']=='test' and r['policy']=='same_k' and r['group']=='all'])+'\n\n'
lead+='## 内部Bank留出收敛\n\n'
convergence=[]
for kind in ['full59','compact48']:
    for seed in [3100,3101]:
        row=[kind,seed]
        for step in [4000,8000,12000]:
            values=[float(r['psnr']) for r in curves if r['kind']==kind and int(r['seed'])==seed
                    and int(r['step'])==step and r['policy']=='same_k']
            assert values
            row.append(f'{sum(values)/len(values):.3f}')
        convergence.append(row)
lead+=table(['输入','seed','4000步','8000步','12000步'],convergence)+'\n\n'
lead+='统一用最终12000步，没有回选checkpoint或调学习率。固定步数差异包括表示与优化条件影响，不是容量上限。\n\n'
value=float(primary['mean'])
directions=[float(primary[k]) for k in ['seed3100','seed3101']]
if all(x>0 for x in directions):
    decision='两个seed方向一致：本轮支持48维作为更简洁的候选，但不能由此认定所有场景下原始属性直连都多余。'
elif all(x<0 for x in directions):
    decision='两个seed七档平均均下降：质量优先时，暂保留59维四Head作为候选基线。但差距偏小且有已知重复波动，尚未确证11维直连是必需输入。'
else:
    decision='两个seed方向冲突，当前输入消融结论为MIXED；不能按一个平均值直接决定生产替换。'
lead+='## 决策与局限\n\n'+decision+'\n\n'
lead+='共享h仍由原始11维属性产生，当前实验只检查直连的额外作用，不能表述为系统不再需要Scale、Opacity、SH DC或ratio。删直连还减少1408个首层参数(约11.4%)，不把差值全归因于单一信息压缩因素。\n\n'
lead+=f'同一冻结fr1_xyz地图、两个网络初始化/帧序seed，不是独立场景重复。已知历史四Head同seed重跑约0.268 dB波动。若把seed3100的59维参考替换成更早历史结果，平均差值约为 {value+.268/2:+.3f} dB；若先平均这两次参考，平均差值约为 {value+.268/4:+.3f} dB。当前对比主要使用上一轮锁定的两seed参考，不选择最好重复，也不把该敏感性范围解释为置信区间。\n\n'
lead+='验证脚本最初错误要求零latent输入对应的投影权重梯度非零，已修正为零；latent表和共享Encoder的有效梯度分别验证。该修正发生在正式训练前，没有改模型、Loss或训练步数；首次gate失败日志保留。\n\n'
lead+='生产默认保持E50，本轮不做生产替换或新在线SLAM；未stage/commit/push。\n\n'
original=(dest/'SPLIT_HEAD_NO_RAW_COMPENSATION_AUDIT.md').read_text(encoding='utf-8')
lead+='## 完整协议与逐ratio对照\n\n'+original.split('\n',1)[1]
name='SPLIT_HEAD_NO_RAW_COMPENSATION_AUDIT.md'
(dest/name).write_text(lead,encoding='utf-8')
target=ROOT.parents[2]/name
target.write_text(lead.replace('(analysis/','(tests/diagnostics/s00_split_no_raw_20261026/evidence/analysis/'),encoding='utf-8')
verification=dict(archive_sha256=digest,new_models=2,reused_models=2,steps=12000,exposure_per_ratio=3000,
    scene='fr1_xyz',seeds=[3100,3101],finite_quality_rows=len(views),production=production,
    online_slam=False,report_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
(ROOT/'delivery_verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
print(json.dumps(verification))
