"""验证专用Encoder受控实验数据并生成中文交付报告。"""
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
assert len(models)==4
for path in models:
    info=json.loads(path.read_text())
    assert info['steps']==12000 and info['frozen_source_unchanged']
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


def metric(a='private_encoder',b='frozen_encoder',group='all',split='test',policy='cached',m='psnr'):
    return next(r for r in pairs if r['candidate']==a and r['reference']==b and r['group']==group
        and r['split']==split and r['policy']==policy and r['metric']==m)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


p=metric()
lead=f'''# 同容量补偿专用Encoder：冻结选择集合的质量对照

在相同fr1_xyz冻结地图上，以两seed刚完成的48维四Head模型为源，各继续训练12000步、每ratio3000次曝光。A固定原编码，B训练同初值同容量的11→32 ReLU专用Encoder；两组Head和latent16均继续训练。原Selector、原Encoder和选中Gaussian身份完全冻结。

## 主要结果

在源模型固定Cached集合、每组exact相同K下，B相对A七档non-Bank PSNR平均 **{float(p['mean']):+.3f} dB**，两个seed分别 **{float(p['seed3100']):+.3f}/{float(p['seed3101']):+.3f} dB**。

训练ratio平均 {float(metric(group='train')['mean']):+.3f} dB，未见ratio平均 {float(metric(group='unseen')['mean']):+.3f} dB，内部Bank留出平均 {float(metric(split='validation')['mean']):+.3f} dB。辅助固定floor-ratio same-K集合的non-Bank平均变化 {float(metric(policy='same_k')['mean']):+.3f} dB。

逐视图/ratio配对差值中位数 {float(p['median']):+.3f} dB，p10/p90为 {float(p['p10']):+.3f}/{float(p['p90']):+.3f} dB，改善比例 {100*float(p['improved_fraction']):.1f}%。这些视图不是独立seed重复。

'''
lead+='## 两seed等权四指标：固定Cached选择集合\n\n'
average=[]
for kind in ['initial','frozen_encoder','private_encoder']:
    subset=[r for r in groups if r['kind']==kind and r['split']=='test' and r['policy']=='cached' and r['group']=='all']
    assert len(subset)==2
    average.append([kind,*[f'{sum(float(r[m]) for r in subset)/2:.5f}' for m in ['psnr','ssim','lpips','photo']]])
lead+=table(['方案','PSNR↑','SSIM↑','LPIPS↓','photo↓'],average)+'\n\n'
lead+='## 专用Encoder收益与继续训练收益\n\n'
rows=[]
for a,b in [('private_encoder','frozen_encoder'),('frozen_encoder','initial'),('private_encoder','initial')]:
    for group in ['all','train','unseen']:
        r=metric(a,b,group)
        rows.append([a+' − '+b,group,f'{float(r["seed3100"]):+.3f}',f'{float(r["seed3101"]):+.3f}',f'{float(r["mean"]):+.3f}'])
lead+=table(['比较','ratio组','seed3100 ΔPSNR','seed3101 ΔPSNR','平均ΔPSNR'],rows)+'\n\n'
lead+='只有B−A是专用编码适配的增量；A/B相对initial的大幅变化不能全部算到Encoder。该继续训练协议与历史从头Joint实验的绝对分数不可直接相减归因。\n\n'
lead+='## 内部Bank留出收敛：固定Cached集合\n\n'
rows=[]
for kind in ['frozen_encoder','private_encoder']:
    for seed in [3100,3101]:
        row=[kind,seed]
        for step in [4000,8000,12000]:
            values=[float(r['psnr']) for r in curves if r['kind']==kind and int(r['seed'])==seed
                and int(r['step'])==step and r['policy']=='cached']
            assert values
            row.append(f'{sum(values)/len(values):.3f}')
        rows.append(row)
lead+=table(['方案','seed','4000步','8000步','12000步'],rows)+'\n\n'
lead+='始终采用最终12000步，没有按non-Bank选step或LR；有限步数差异不是容量上限。\n\n'
value=float(p['mean'])
directions=[float(p['seed3100']),float(p['seed3101'])]
if all(v>0 for v in directions):
    decision='本轮两个seed均支持给补偿增加独立编码适配自由度。优先保留同容量专用Encoder作为后续候选；不能将此解释为已经证明原共享梯度有害，也不由此自动加深或加宽。'
elif all(v<0 for v in directions):
    decision='本轮两个seed均未从可训练专用Encoder获得收益，固定编码更好。现阶段没有证据支持将Encoder适配作为首要改进，不能据此排除其他初始化或优化协议下的效果。'
else:
    decision='两个seed方向不一致，本轮Encoder适配结论为MIXED，不能仅凭平均值决定替换或继续扩容。'
lead+='## 本轮判断\n\n'+decision+'\n\n'
lead+='**当前没有稳定的同容量补偿专用Encoder质量收益，不建议据此替换生产或自动继续加深Encoder。** 真实photo梯度检查与参数变化均正常，不能把零收益解释为专用Encoder没有更新。它只能说明本次已训练源模型、固定集合、相同LR与预算下，额外编码适配价值很小；不能排除从头训练或在线场景下的差异。\n\n'
lead+='两组继续训练都在训练ratio上恢复约0.65 dB，但未见ratio平均变化约−0.08～−0.09 dB。当前更值得关注补偿Head/latent的训练充分性与跨ratio泛化之间的权衡，而不是将总继续训练收益算作Encoder改善；该解释仍是下一项研究方向，不是已隔离的因果结论。\n\n'
lead+='本实验固定了选择集合并冻结选择分支，因此没有直接测量原Joint路径中选择/预算/补偿的梯度竞争。A的Encoder冻结，B多384个可训练自由度；它检验编码适配，而非完全等训练自由度的纯参数共享因果。\n\n'
lead+='两seed来自同一冻结地图的两个源模型，历史non-Bank已多次使用；不作独立场景稳定性、统计显著性或在线Joint质量保证。生产默认保持E50。\n\n'
original=(dest/'COMPENSATION_PRIVATE_ENCODER_AUDIT.md').read_text(encoding='utf-8')
lead+='## 完整协议与对照\n\n'+original.split('\n',1)[1]
name='COMPENSATION_PRIVATE_ENCODER_AUDIT.md'
(dest/name).write_text(lead,encoding='utf-8')
target=ROOT.parents[2]/name
target.write_text(lead.replace('(analysis/','(tests/diagnostics/s00_comp_private_encoder_20261026/evidence/analysis/'),encoding='utf-8')
verification=dict(archive_sha256=digest,new_models=4,steps=12000,exposure_per_ratio=3000,
    scene='fr1_xyz',seeds=[3100,3101],finite_quality_rows=len(views),production=production,
    fixed_master_and_selector=True,fixed_sets=True,online_slam=False,
    report_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
(ROOT/'delivery_verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
print(json.dumps(verification))
