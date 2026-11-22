"""只读完整证据生成中文报告和标准科学图，不根据结果改变实验。"""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
LOCAL='H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_implicit_si_20261020'
load=lambda name:list(csv.DictReader(open(ROOT/name)))
s=json.loads((ROOT/'audit_summary.json').read_text());grid=load('grid_metadata.csv');q=load('quality_summary.csv');d=load('quality_deltas.csv');regional=load('regional_summary.csv');grad=load('cell_gradient_statistics.csv');corr=load('occupancy_correction_correlations.csv');lr=json.loads((ROOT/'lr_sanity.json').read_text());coeff=load('cell_coefficient_statistics.csv');pairs=load('paired_statistics.csv');seeds=load('seed_consistency.csv')
qm={(z['scene'],z['split'],z['arm'],float(z['ratio']),int(z['seed'])):z for z in q};dm={(z['scene'],z['split'],z['arm'],float(z['ratio']),int(z['seed']),z['reference']):z for z in d}
scenes=['fr1_room','fr1_xyz','fr2_xyz'];ratios=[.01,.05,.1,.15]
lines=['# Grid-Level Spatial MLP SI Audit','',
       '**结论：'+s['decision']+'。** '+s['interpretation'],'',
       '**最终问题**','', '| 问题 | 回答 |','|---|---|']
questions=['MLP只对occupied cells forward？','同cell严格共享一个coefficient？','发生真实cell-to-cell预算移动？','Grid-MLP稳定优于Raw？','优于已有calibrated Gaussian-input MLP？','接近或超过PG-ST？','缩小Fixed/C gap？','Independent-Cell说明grid参数化本身有效？','主要限制？','进入gradient-EMA threshold split？']
for i,question in enumerate(questions,1):lines.append(f'| Q{i}. {question} | {s["answers"]["Q"+str(i)]} |')
lines+=['','**判读与因果边界**','']+[f'- {text}'for text in s['mechanism_interpretation']]
lines+=['','**结构与冻结grid**','',
        '本轮MLP的对象是occupied cell：`normalized cell center[M,3] → 3→32→32→1 / SiLU → raw r[M]`。最后层bias=false、weight=0初始化，共1216参数。按occupancy计算 `mu=sum(n_C*r_C)/N`，保留autograd；`a_C=r_C-mu`，通过固定cell_id广播到N个Gaussian。score=`log(RawSI+1e-12)+a_C(i)`，严格保留cell内部Raw排序。','',
        'grid固定world origin=(0,0,0)，cell key=floor(XYZ/h)，center=(key+0.5)*h，cell ID按key字典序确定。h仅由冻结valid AND observed XYZ的occupied count选取：seed=max XYZ span/cbrt(2048)，doubling bracket，128点log probes，再对最佳邻域固定128点细化；误差相同选较大h。沿用已有sparse root的key/搜索结构，但不调用其radius-based seed；本轮没有读取radius、hierarchy、GT或quality决定grid。','',
        '| scene | N | M | M/N | h | occupancy min/median/p90/max |','|---|---:|---:|---:|---:|---|']
for z in grid:lines.append(f'| {z["scene"]} | {z["N"]} | {z["M"]} | {100*float(z["M_over_N"]):.2f}% | {float(z["h"]):.12g} | {z["occupancy_min"]} / {float(z["occupancy_median"]):g} / {float(z["occupancy_p90"]):g} / {z["occupancy_max"]} |')
lines+=['','cell centers按各scene occupied-center min/max box映射到[-1,1]^3；构造后h、mapping、occupancy、center及normalization全程冻结。occupancy只用于gauge和posthoc统计，未作为MLP输入。Gaussian XYZ也未逐行输入MLP。','',
        '**LR gate与训练协议**','',
        '| fr1_xyz /5%/seed2026，250steps LR | initial Bank photo | final Bank photo | Δphoto | ΔPSNR |','|---|---:|---:|---:|---:|']
for z in lr['results']:lines.append(f'| {z["lr"]:g} | {z["initial_photo"]:.9f} | {z["final_photo"]:.9f} | {z["photo_delta"]:+.9f} | {z["PSNR_delta"]:+.6f} |')
lines+=['',f'按预注册Bank hard photo优先准则选定LR={lr["selected_lr"]:g}，在读取任何baseline/eval质量前锁定；未触发三档全部恶化的停止条件。全部Grid-MLP和Independent-Cell均使用该LR、Adam(.9,.999,eps1e-8,wd0)、2000steps，不early stop。', '',
        '复用已修复的implicit expected-K VJP，未改校准器实现。采样仍为cached-uniform ST Bernoulli，compact selected-row renderer与opacity ST，photo=0.8L1+0.2(1-fast_ssim)，Bank schedule与旧实验exact。正式评估统一采用step2000 deterministic hard same-K。没有额外loss/regularizer/feature，也没有PG/C teacher。', '',
        'Independent-Cell control每cell直接学习一个scalar，不经过MLP；使用完全相同的gauge/calibration/sampler/Bank/loss/LR/budget。共完成24个Primary和24个control模型，正式96,000次更新，另有750次LR sanity更新。','',
        '**质量主表：每项24个scene/ratio/seed case**','',
        '| arm | reference | split | PSNR改善/相同/变差 | SSIM改善 | LPIPS改善 | photo改善 | mean ΔPSNR |','|---|---|---|---:|---:|---:|---:|---:|']
for arm in ['Grid-MLP','Independent-Cell']:
    references=['Raw','PG-ST','Fixed','C','Per-Gaussian-MLP']+(['Grid-MLP']if arm=='Independent-Cell'else[])
    for reference in references:
        for split in ['train','eval']:
            z=s['counts'][arm][reference][split];p=z['psnr']
            lines.append(f'| {arm} | {reference} | {"Bank"if split=="train"else"non-Bank"} | {p["improved"]}/{p["equal"]}/{p["worse"]} | {z["ssim"]["improved"]} | {z["lpips"]["improved"]} | {z["photo"]["improved"]} | {p["mean_delta"]:+.6f} |')
lines+=['','四指标全部absolute aggregate/per-view保存在quality_summary.csv和quality_per_view.csv。photo/LPIPS下降为改善；PSNR/SSIM上升为改善。未定义新的dB GO阈值，未挑最好seed或checkpoint。这里Per-Gaussian-MLP指已有calibrated XYZ-per-Gaussian-input MLP，直接复用既有质量结果；本轮不做coefficient MSE或PG coefficient拟合。','',
        '**non-Bank各case**','',
        '| scene | ratio | seed | Grid PSNR | ΔRaw | ΔPG-ST | ΔFixed | ΔC | ΔGaussian-MLP | Grid Δphoto Raw | Independent ΔPSNR Raw |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for scene in scenes:
    for ratio in ratios:
        for seed in [2026,2027]:
            z=qm[scene,'eval','Grid-MLP',ratio,seed];differences=[float(dm[scene,'eval','Grid-MLP',ratio,seed,reference]['psnr'])for reference in ['Raw','PG-ST','Fixed','C','Per-Gaussian-MLP']]
            photo=float(dm[scene,'eval','Grid-MLP',ratio,seed,'Raw']['photo']);control=float(dm[scene,'eval','Independent-Cell',ratio,seed,'Raw']['psnr'])
            lines.append(f'| {scene} | {100*ratio:g}% | {seed} | {float(z["psnr"]):.5f} | '+' | '.join(f'{x:+.5f}'for x in differences)+f' | {photo:+.7f} | {control:+.5f} |')
lines+=['','**Paired-view与seed一致性**','',
        '| Grid-MLP reference | split | PSNR paired median>0 cases | 多于一半views改善cases | improved-view fraction跨case中位数 |','|---|---|---:|---:|---:|']
for reference in ['Raw','PG-ST','Fixed','C','Per-Gaussian-MLP']:
    for split in ['train','eval']:
        z=s['counts']['Grid-MLP'][reference][split];fractions=[float(x['improved_view_fraction'])for x in pairs if x['arm']=='Grid-MLP'and x['reference']==reference and x['split']==split and x['metric']=='psnr']
        lines.append(f'| {reference} | {"Bank"if split=="train"else"non-Bank"} | {z["PSNR_positive_median_cases"]}/24 | {z["PSNR_majority_views_improved_cases"]}/24 | {np.median(fractions):.4f} |')
lines+=['','每个scene/ratio/seed、每项指标的paired mean/median/p10/p90/improved-view fraction完整保存在paired_statistics.csv，未把不同view池混合作为独立重复。','',
        '| arm | non-Bank PSNR两seed绝对差，中位数/最大值(dB) |','|---|---:|']
for arm in ['Grid-MLP','Independent-Cell','Per-Gaussian-MLP','PG-ST']:
    values=[float(x['psnr_seed_abs_difference'])for x in seeds if x['arm']==arm and x['split']=='eval']
    lines.append(f'| {arm} | {np.median(values):.6f} / {max(values):.6f} |')
lines+=['','**区域预算移动必须与质量同时评估**','',
        '| scene | ratio | seed | Grid changed cells | increase/decrease | transferred points | max abs delta_k | ΔBank photo | control transferred |','|---|---:|---:|---:|---|---:|---:|---:|---:|']
for z in [x for x in regional if x['arm']=='Grid-MLP']:
    other=next(x for x in regional if x['scene']==z['scene']and x['ratio']==z['ratio']and x['seed']==z['seed']and x['arm']=='Independent-Cell')
    lines.append(f'| {z["scene"]} | {100*float(z["ratio"]):g}% | {z["seed"]} | {z["changed_cells"]} | {z["increased_cells"]}/{z["decreased_cells"]} | {z["budget_transferred"]} | {z["max_abs_delta"]} | {float(z["hard_Bank_photo_delta"]):+.8f} | {other["budget_transferred"]} |')
lines+=['','每个模型均满足sum(delta_k)=0；sum(positive delta_k)=-sum(negative delta_k)。intra-cell排序保持使得transferred points等于相对Raw新增Gaussian数。完整per-cell值、系数、occupancy、direct选中事件与final/Raw count见cell_budget_redistribution.csv；histogram见cell_budget_histogram.csv。非零coefficient或大量changed cells本身不作为成功判据。','',
        '**occupancy与gradient空间分布**','',
        '| arm | scene | final corr(occupancy, a_C) min/max | final corr(occupancy, EMA) min/max | final top10% EMA mass share min/max |','|---|---|---:|---:|---:|']
for arm in ['Grid-MLP','Independent-Cell']:
    for scene in scenes:
        a=[float(z['correlation_occupancy_coefficient'])for z in corr if z['arm']==arm and z['scene']==scene and z['correlation_occupancy_coefficient']!='']
        b=[float(z['correlation_occupancy_gradient_EMA'])for z in corr if z['arm']==arm and z['scene']==scene and z['correlation_occupancy_gradient_EMA']!='']
        c=[float(z['top10_EMA_mass_share'])for z in grad if z['arm']==arm and z['scene']==scene and int(z['step'])==2000 and z['top10_EMA_mass_share']!='']
        lines.append(f'| {arm} | {scene} | {min(a):+.4f}/{max(a):+.4f} | {min(b):+.4f}/{max(b):+.4f} | {min(c):.4f}/{max(c):.4f} |')
lines+=['','gradient统计保留raw cell output r_C的|gradient|与EMA(beta=.95)，每100step及step250记录，含mean/std/p50/p90/p99和top10% mass share。这些数据仅用于描述空间集中性，没有据此选择threshold或split。相关性只作posthoc描述，不能把cell coefficient直接称为occupancy predictor或认为集中就值得split。完整系数/exp(a_C)分布见cell_coefficient_statistics.csv。','',
        '**Correctness与证据边界**','',
        f'- 所有Primary训练记录的MLP forward count=M，control为0；M/N为上述4%–9%左右，每个candidate恰有一个cell_id，按cell广播的correction exact相同。',
        f'- 全部weighted gauge和calibration检查通过；最大normalized gradient sum={s["correctness"]["maximum_normalized_gradient_sum"]:.3g}，最大weighted centering误差={s["correctness"]["maximum_weighted_center_error"]:.3g}。CPU/CUDA加权gauge有限差分和原calibration regression均通过。',
        '- 全部cell的Raw local order与final local order保持一致，violated cell count=0；initial full ranking及四ratio Top-K exact Raw。',
        '- Native每步验证共享系数/gauge/intra-cell顺序，定期验证Gaussian/TF/Master/grid tensor exact冻结；Python从保存MLP权重独立forward复核cell输出，并独立重算最终排序和预算转移。',
        '- 原始五组基线直接复用已有冻结质量文件，文件SHA核对；Raw的1420次Bank/non-Bank view render重新执行并逐view四指标exact，随后新48模型5680次view render。',
        '- 训练不读取C/Fixed/PG coefficient或non-Bank quality；LR选择只使用预指定Bank场景。全部新集合在quality render前锁定SHA。',
        '- 相对旧Gaussian-input MLP，本轮同时改变了forward对象、cell tying和normalization定义；不能把质量差异单独归因其中某一项。Independent-Cell control用于帮助区分fixed regional tying和MLP sharing，但不同参数化的优化条件仍可能不同。','',
        '**输出文件**','',
        f'- [grid_metadata.csv]({LOCAL}/grid_metadata.csv)、[cell_mapping.npz]({LOCAL}/cell_mapping.npz)、[occupancy_statistics.csv]({LOCAL}/occupancy_statistics.csv)。',
        f'- [quality_summary.csv]({LOCAL}/quality_summary.csv)、[quality_per_view.csv]({LOCAL}/quality_per_view.csv)、[paired_statistics.csv]({LOCAL}/paired_statistics.csv)、[independent_cell_quality.csv]({LOCAL}/independent_cell_quality.csv)。',
        f'- [training_curves.csv]({LOCAL}/training_curves.csv)、[hard_training_dynamics.csv]({LOCAL}/hard_training_dynamics.csv)、[cell_coefficient_statistics.csv]({LOCAL}/cell_coefficient_statistics.csv)。',
        f'- [cell_budget_redistribution.csv]({LOCAL}/cell_budget_redistribution.csv)、[cell_gradient_statistics.csv]({LOCAL}/cell_gradient_statistics.csv)、[selected_set_overlap.csv]({LOCAL}/selected_set_overlap.csv)、[raw_rank_promotion.csv]({LOCAL}/raw_rank_promotion.csv)。',
        f'- [correctness.json]({LOCAL}/correctness.json)、[audit_summary.json]({LOCAL}/audit_summary.json)、[质量对照图]({LOCAL}/quality_deltas.png)。',
        '- 每scene保存key/center/occupancy/mapping/normalization/h；每模型保存step0/250/1000/2000权重或独立scalar、raw output、cell coefficient、Gaussian broadcast、gradient EMA、final score/ranking/selected set。','',
        '完成后停止：未实现adaptive split、threshold、child机制，未改grid size/normalization/network/feature/loss；未修改production Selector，未stage、commit、push。','']
(ROOT/'GRID_MLP_IMPLICIT_SI_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')
fig,axes=plt.subplots(3,3,figsize=(13,9),sharex=True)
for i,scene in enumerate(scenes):
    for j,reference in enumerate(['Raw','PG-ST','Per-Gaussian-MLP']):
        ax=axes[i,j]
        for seed,marker in [(2026,'o'),(2027,'s')]:ax.plot([1,5,10,15],[float(dm[scene,'eval','Grid-MLP',ratio,seed,reference]['psnr'])for ratio in ratios],marker=marker,label=str(seed))
        if reference=='Raw':ax.plot([1,5,10,15],[np.mean([float(dm[scene,'eval','Independent-Cell',ratio,seed,'Raw']['psnr'])for seed in [2026,2027]])for ratio in ratios],linestyle='--',label='Independent mean')
        ax.axhline(0,color='black',linewidth=.7);ax.grid(alpha=.2);ax.set_xticks([1,5,10,15])
        if i==0:ax.set_title('Grid-MLP - '+reference)
        if j==0:ax.set_ylabel(scene+'\nPSNR delta (dB)')
        if i==2:ax.set_xlabel('ratio (%)')
axes[0,0].legend();fig.tight_layout();fig.savefig(ROOT/'quality_deltas.png',dpi=160);plt.close(fig)
print('REPORT_WRITTEN')
