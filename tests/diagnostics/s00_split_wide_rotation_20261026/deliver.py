"""验证服务器归档、汇总中文结论并生成本地可点击报告。"""
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
assert len(models)==7
for path in models:
    info=json.loads(path.read_text())
    assert info['steps']==12000 and info['master_frozen']
    assert all(value==3000 for value in info['ratio_exposures'].values())
with (dest/'analysis_B/quality_per_view.csv').open() as stream:
    views=list(csv.DictReader(stream))
assert all(math.isfinite(float(r[m])) for r in views for m in ['psnr','ssim','lpips','photo'])
with (dest/'analysis_B/paired_statistics.csv').open() as stream:
    pairs=list(csv.DictReader(stream))
with (dest/'analysis_B/group_summary.csv').open() as stream:
    groups=list(csv.DictReader(stream))
with (dest/'analysis_B/validation_curves.csv').open() as stream:
    curves=list(csv.DictReader(stream))
production=json.loads((ROOT/'production_verification.json').read_text(encoding='utf-8-sig'))
assert not production['production_mismatches'] and not production['staged_files']


def metric(a,b,group='all',split='test',m='psnr'):
    return next(r for r in pairs if r['candidate']==a and r['reference']==b and r['group']==group
        and r['split']==split and r['policy']=='same_k' and r['metric']==m)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


split=metric('split4','unified32')
wide=metric('unified80','unified32')
sw=metric('split4','unified80')
rotation=metric('split4_rotation','split4')
lead=f'''# 四Head复现、统一80宽头与基础Rotation输入实验

本轮完成七次新增训练，复用一次匹配的统一32头参考。全部fr1_xyz冻结地图、seed3100/3101、12000真实LoD步，每ratio3000曝光。两个seed同时控制网络初始化与Bank帧序，不是只改变顺序，也不是两张独立SLAM地图。

## 主要结果

- 四Head相对统一32头：两个seed分别 **{float(split['seed3100']):+.3f}/{float(split['seed3101']):+.3f} dB**，平均 **{float(split['mean']):+.3f} dB**。
- 统一80宽头相对统一32头：分别 **{float(wide['seed3100']):+.3f}/{float(wide['seed3101']):+.3f} dB**，平均 **{float(wide['mean']):+.3f} dB**。
- 等参数量附近，四Head相对统一80宽头：分别 **{float(sw['seed3100']):+.3f}/{float(sw['seed3101']):+.3f} dB**，平均 **{float(sw['mean']):+.3f} dB**。
- 基础Rotation输入相对不加Rotation的四Head：分别 **{float(rotation['seed3100']):+.3f}/{float(rotation['seed3101']):+.3f} dB**，平均 **{float(rotation['mean']):+.3f} dB**。

以上都是七档等权、non-Bank、deterministic same-K。各ratio、其他指标、Bank留出、原Cached实际预算与逐视图配对分布完整保存在CSV；不能只凭平均值或两个seed认定统计显著。

'''
comparisons=[('split4','unified32'),('unified80','unified32'),('split4','unified80'),('split4_rotation','split4')]
lead+='## 训练ratio与未见ratio\n\n'+table(['比较','训练四档ΔPSNR','未见三档ΔPSNR','内部Bank留出ΔPSNR'],
    [[a+' − '+b,f'{float(metric(a,b,"train")["mean"]):+.3f}',f'{float(metric(a,b,"unseen")["mean"]):+.3f}',
      f'{float(metric(a,b,split="validation")["mean"]):+.3f}'] for a,b in comparisons])+'\n\n'
lead+='## 每个seed的四指标\n\n'+table(['方案','seed','PSNR↑','SSIM↑','LPIPS↓','photo↓'],
    [[r['kind'],r['seed'],*[f'{float(r[m]):.5f}' for m in ['psnr','ssim','lpips','photo']]] for r in groups
     if r['split']=='test' and r['policy']=='same_k' and r['group']=='all'])+'\n\n'
averages=[]
for kind in ['unified32','unified80','split4','split4_rotation']:
    subset=[r for r in groups if r['kind']==kind and r['split']=='test' and r['policy']=='same_k' and r['group']=='all']
    averages.append([kind,*[f'{sum(float(r[m]) for r in subset)/len(subset):.5f}' for m in ['psnr','ssim','lpips','photo']]])
lead+='## 两seed等权平均四指标\n\n'+table(['方案','PSNR↑','SSIM↑','LPIPS↓','photo↓'],averages)+'\n\n'
lead+='## 内部Bank留出收敛\n\n'
convergence=[]
for kind in ['unified32','unified80','split4','split4_rotation']:
    for seed in [3100,3101]:
        values=[]
        for step in [4000,8000,12000]:
            rows=[float(r['psnr']) for r in curves if r['kind']==kind and int(r['seed'])==seed
                  and int(r['step'])==step and r['policy']=='same_k']
            assert rows
            values.append(f'{sum(rows)/len(rows):.3f}')
        convergence.append([kind,seed,*values])
lead+=table(['方案','seed','4000步','8000步','12000步'],convergence)+'\n\n'
lead+='统一用最终12000步，未回选中间checkpoint；观察到的差距包括有限训练预算和初始化/优化几何影响，不能当作充分收敛后的表达上限。\n\n'
repeat=json.loads((dest/'repeatability.json').read_text())
lead+='## 历史四Head重复性\n\n'+json.dumps(repeat,ensure_ascii=False)+'\n\n'
lead+='重复seed3100仅检查重现性，不算第三个独立seed。原始历史结果保留，没有用重复结果覆盖历史文件。\n\n'
initialization=json.loads((dest/'repeat_initialization.json').read_text())
assert initialization['initial_state_dict_exact'] and initialization['schedule_exact']
lead+='独立进程核对初始全部权重exact相同，完整3000帧循环与日志ratio/frame/step序列exact相同。第1/2步photo相同，第3步差约1.49e-8，随后选择和训练轨迹分化；这与GPU数值非确定性被长程优化放大一致，但未单独隔离具体CUDA算子。不能把固定seed当成完全确定性，也不能忽略约0.268 dB的同seed平均PSNR波动。\n\n'
repeat_change=repeat['same_k_mean_metric_change']['psnr']
sensitivity=[]
for label,result,sign in [('split4 − unified32',split,-1),('split4 − unified80',sw,-1),
                          ('split4_rotation − split4',rotation,1)]:
    value=float(result['mean'])
    sensitivity.append([label,f'{value:+.3f}',f'{value+sign*repeat_change/2:+.3f}',
        f'{value+sign*repeat_change/4:+.3f}'])
lead+='## 对同seed重复波动的敏感性\n\n'+table(
    ['比较','本轮重跑作seed3100','历史结果作seed3100','seed3100两次平均后再等权两个seed'],sensitivity)+'\n\n'
lead+='三种算法均保留seed3101不变，仅展示同seed重复的影响，不是挑选最好seed。四Head相对统一80宽头平均优势在约0.177～0.311 dB之间，加入Rotation的平均变化在约−0.197～−0.063 dB之间，方向均未翻转。该敏感性分析不是置信区间，仍需独立场景与更多重复才能作稳定因果结论。\n\n'
weights=[]
for seed in [3100,3101]:
    weights.extend(json.loads((dest/f'rotation_input_weight_updates_{seed}.json').read_text()).values())
lead+='## Rotation输入是否真的被使用\n\n'
lead+=f'三个几何Head的新增Rotation输入列权重从零变为非零，最终范数范围 {min(r["rotation_columns_norm"] for r in weights):.3f}～{max(r["rotation_columns_norm"] for r in weights):.3f}。q/-q输入等价性通过，四元数norm最大误差约1.19e−7；数值输出依赖诊断保存在analysis_B/rotation_input_dependence.csv。因而不能把本次退化解释为新增输入完全没有训练或输入无效，但输出依赖不等于图像收益。\n\n'
lead+='## 本轮决策\n\n'
lead+='**保留16维ratio latent＋原始11维＋共享h32＋四个两层私有Head作为优先候选，暂不加入基础Quaternion输入。** 增大统一Head容量也有效，但本轮在近似等参数量下四Head的PSNR、SSIM、LPIPS、photo均更好，两个seed方向一致。这支持属性专用隐藏层有价值，尚不能排除优化几何与可变选择集合的共同影响。\n\n'
lead+='Rotation加入后两个seed七档平均PSNR、SSIM与photo均较差，未见ratio方向不一致；本次没有证据支持该4维输入方案。不能推广为所有Rotation表征无价值，也不据此自动改用其他Rotation编码或调参。\n\n'
lead+='生产默认继续保持E50。下一步先在另一独立地图/场景按相同步数与same-K验证四Head，再决定是否进入真实在线Joint；本轮没有自动运行该扩展。\n\n'
lead+='## 决策边界\n\n'
lead+='- split4与wide80参数量近似，但参数化、初始化嵌入与梯度路由不同；对照支持或削弱“属性拆分优于统一容量”的假设，不能当作完全隔离所有优化条件。\n'
lead+='- Rotation对照只追加基础四元数输入，几何Head多384参数；不会改Selector输入或输出补偿语义。不能由该对照推断所有Rotation表示都有效或无效。\n'
lead+='- 所有方案Selector/Encoder仍共同训练，集合可变；同K固定预算不等于固定点身份。不能将全部PSNR变化归因于Compensation表达。\n'
lead+='- 历史non-Bank集合已反复使用，只有fr1_xyz一个冻结地图；本轮不声称独立场景泛化，进入生产前仍需要另一地图/场景和在线Joint确认。\n'
lead+='- 生产源码、Loss、默认E50、在线二进制均未修改；无新在线SLAM或CUDA Mapping构建；未stage/commit/push。\n\n'
original=(dest/'SPLIT_WIDE_ROTATION_PHASE_B.md').read_text(encoding='utf-8')
lead+='## 完整协议与对照表\n\n'+original.split('\n',1)[1]
lead+='\n\n## 固定渲染示例\n\n固定seed3100与中间non-Bank视图，未按效果选图。\n\n![1%](analysis_B/1pct_examples.png)\n\n![5%](analysis_B/5pct_examples.png)\n'
name='SPLIT_WIDE_ROTATION_COMPENSATION_AUDIT.md'
(dest/name).write_text(lead,encoding='utf-8')
target=ROOT.parents[2]/name
target.write_text(lead.replace('(analysis_B/','(tests/diagnostics/s00_split_wide_rotation_20261026/evidence/analysis_B/'),encoding='utf-8')
verification=dict(archive_sha256=digest,new_models=7,reused_models=1,steps=12000,exposure_per_ratio=3000,
    scene='fr1_xyz',seeds=[3100,3101],finite_quality_rows=len(views),production=production,
    online_slam=False,report_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),repeatability=repeat,
    initial_weights_and_schedule_exact=initialization['initial_state_dict_exact'] and initialization['schedule_exact'])
(ROOT/'delivery_verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
print(json.dumps(verification))
