"""完整证据完成后生成中文报告；不改模型或集合。"""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
LOCAL='H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_cached_bernoulli_inference_20261020'
load=lambda name:list(csv.DictReader(open(ROOT/name)))
s=json.loads((ROOT/'audit_summary.json').read_text());q=load('quality_summary.csv');d=load('quality_deltas.csv');points=load('selection_counts.csv');pairs=load('paired_statistics.csv')
qm={(z['scene'],z['split'],z['variant'],float(z['ratio']),int(z['seed'])):z for z in q}
dm={(z['scene'],z['split'],z['arm'],float(z['ratio']),int(z['seed']),z['reference']):z for z in d}
scenes=['fr1_room','fr1_xyz','fr2_xyz'];ratios=[.01,.05,.1,.15]
lines=['# Grid MLP Cached Bernoulli 推理质量审计','',s['interpretation'],'',
       '## 推理变化与完成范围','',
       '本轮不训练。复用3scene×4ratio×2seed的24个Grid-MLP与24个Independent-Cell的step2000 checkpoint。推理读取保存的score和cached_uniform，直接调用训练相同的C++ calibratedSelectionProbability与sampleSelectorST；保持42次expected-K二分、原nominal ratio、冻结TF与renderer。没有生成新noise、没有补点或截断。','',
       '训练ST mask的forward严格为0/1，因此推理只render被选rows，opacity保持原值；没有把opacity乘以概率。重复相同checkpoint推理得到exact相同集合。','',
       '**K是nominal expected budget，L是Cached实际选中数。** Cached不保证L=K。本轮同时真实render learned-score hard Top-L、Raw hard Top-L，严格匹配该Cached集合的实际点数；TF仍输入原nominal ratio，不改为L/N。原hard Top-K只作为历史导出方式对照，不能当成same-count比较。','',
       '| arm | actual L/K min | median | max | max abs(L−K) |','|---|---:|---:|---:|---:|']
for arm in ['Grid-MLP','Independent-Cell']:
    z=s['point_counts'][arm];lines.append(f'| {arm} | {z["min_L_over_K"]:.6f} | {z["median_L_over_K"]:.6f} | {z["max_L_over_K"]:.6f} | {z["maximum_abs_L_minus_K"]} |')
lines += ['','## 核心质量结果：每项24个scene/ratio/seed case','',
          '| Cached arm | reference | split | PSNR改善 | photo改善 | SSIM改善 | LPIPS改善 | mean ΔPSNR | mean Δphoto | actual-count相同case |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
for arm in ['Grid-MLP','Independent-Cell']:
    refs=['Raw-Cached',arm+'-Hard-L','Raw-Hard-L-'+arm,arm+'-Hard-K','Raw-Hard-K','Fixed-Hard-K','C-Hard-K']
    for ref in refs:
        for split in ['train','eval']:
            z=s['counts'][arm][ref][split];lines.append(f'| {arm} | {ref} | {"Bank"if split=="train"else"non-Bank"} | {z["psnr"]["improved"]}/24 | {z["photo"]["improved"]}/24 | {z["ssim"]["improved"]}/24 | {z["lpips"]["improved"]}/24 | {z["psnr"]["mean_delta"]:+.6f} | {z["photo"]["mean_delta"]:+.7f} | {z["same_count_cases"]}/24 |')
lines += ['','Raw-Cached为a=0、同一组saved uniform的初始化基线，可观察Cached策略下训练前后的变化；其实际点数与训练后可能不同。Hard-L两组是严格same-count对照。PSNR/SSIM越高越好，photo/LPIPS越低越好。所有差值均为Cached−reference。','',
          '## Grid-MLP non-Bank逐case','',
          '| scene | ratio | seed | nominal K | Cached L | Cached PSNR | ΔRaw-Cached | Δ原Grid hard-K | ΔGrid hard-L | ΔRaw hard-L |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for scene in scenes:
    for ratio in ratios:
        for seed in [2026,2027]:
            z=qm[scene,'eval','Grid-MLP-Cached',ratio,seed];values=[float(dm[scene,'eval','Grid-MLP',ratio,seed,ref]['psnr'])for ref in ['Raw-Cached','Grid-MLP-Hard-K','Grid-MLP-Hard-L','Raw-Hard-L-Grid-MLP']]
            lines.append(f'| {scene} | {100*ratio:g}% | {seed} | {z["K"]} | {z["L"]} | {float(z["psnr"]):.6f} | '+' | '.join(f'{value:+.6f}'for value in values)+' |')
lines += ['','## Paired-view改善范围','',
          '| arm | reference | split | paired PSNR median>0 cases | 多数views改善cases |','|---|---|---|---:|---:|']
for arm in ['Grid-MLP','Independent-Cell']:
    for ref in ['Raw-Cached',arm+'-Hard-L','Raw-Hard-L-'+arm,arm+'-Hard-K']:
        for split in ['train','eval']:
            z=s['counts'][arm][ref][split];lines.append(f'| {arm} | {ref} | {"Bank"if split=="train"else"non-Bank"} | {z["PSNR_positive_median_cases"]}/24 | {z["PSNR_majority_views_improved_cases"]}/24 |')
lines += ['','四指标的mean/median/p10/p90/improved-view fraction逐case保存在paired_statistics.csv；没有挑最好seed、view或checkpoint。','',
          '## 判读','']+[f'- {text}'for text in s['discussion']]
lines += ['','## Correctness','',
          '- 96项step0/2000检查：原生CUDA C++与Python CUDA重建集合exact，重复Cached推理exact，ST forward mask exact为0/1；step0点数与训练首步相同。',
          '- Grid-MLP与Independent-Cell共享同scene/ratio/seed的原始uniform，初始Raw-Cached集合exact相同；uniform没有被重新生成。',
          '- 推理读取保存checkpoint的score；没有更新MLP/coefficients。全部96个state文件本轮推理前后SHA不变，导出集合在quality前锁SHA。',
          '- 所有hard-L对照与对应Cached实际点数exact一致、rows unique；不把L当成K。',
          '- 原Raw hard-K的1420次Bank/non-Bank view render四指标逐view exact复现；新168个case共19880次view render。所有Gaussian/TF/Master与输入runtime冻结核验通过。',
          '- 仍使用原Bank/non-Bank划分、cameras、GT和metrics；没有用quality反向选score、uniform、ratio或checkpoint。没有新训练、grid或split，production未修改。','',
          '## 结果文件','',
          f'- [quality_summary.csv]({LOCAL}/quality_summary.csv)、[quality_per_view.csv]({LOCAL}/quality_per_view.csv)、[quality_deltas.csv]({LOCAL}/quality_deltas.csv)、[paired_statistics.csv]({LOCAL}/paired_statistics.csv)。',
          f'- [selection_counts.csv]({LOCAL}/selection_counts.csv)、[selected_sets.npz]({LOCAL}/selected_sets.npz)、[inference_arrays.npz]({LOCAL}/inference_arrays.npz)、[selected_set_overlap.csv]({LOCAL}/selected_set_overlap.csv)。',
          f'- [correctness.json]({LOCAL}/correctness.json)、[audit_summary.json]({LOCAL}/audit_summary.json)、[原生推理代码]({LOCAL}/cached_inference.cpp)、[推理与导出脚本]({LOCAL}/prepare.py)。',
          f'- [质量差值图]({LOCAL}/quality_deltas.png)。','',
          '完成后停止。未修改production Selector；未stage、commit、push。','']
(ROOT/'GRID_MLP_CACHED_BERNOULLI_INFERENCE_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')
fig,axes=plt.subplots(3,3,figsize=(13,9),sharex=True)
refs=['Raw-Cached','Grid-MLP-Hard-L','Raw-Hard-L-Grid-MLP']
for i,scene in enumerate(scenes):
    for j,ref in enumerate(refs):
        ax=axes[i,j]
        for seed,marker in [(2026,'o'),(2027,'s')]:ax.plot([1,5,10,15],[float(dm[scene,'eval','Grid-MLP',ratio,seed,ref]['psnr'])for ratio in ratios],marker=marker,label=str(seed))
        ax.axhline(0,color='black',lw=.7);ax.grid(alpha=.2);ax.set_xticks([1,5,10,15])
        if i==0:ax.set_title('Cached - '+ref)
        if j==0:ax.set_ylabel(scene+'\nPSNR delta (dB)')
        if i==2:ax.set_xlabel('nominal ratio (%)')
axes[0,0].legend();fig.tight_layout();fig.savefig(ROOT/'quality_deltas.png',dpi=160);plt.close(fig)
print('REPORT_WRITTEN')
