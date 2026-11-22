"""由全部固定结果生成中文表格、静态科学图；不用于选择训练配置。"""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
LOCAL='H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_pgst_representability_20261020'
load=lambda name:list(csv.DictReader(open(ROOT/name)))
summary=json.loads((ROOT/'audit_summary.json').read_text());reg=load('regression_metrics.csv');ov=load('topk_overlap.csv');quality=load('quality_comparison.csv');paired=load('paired_statistics.csv');conv=load('convergence_statistics.csv')
answer=summary['answers'];scenes=['fr1_room','fr1_xyz','fr2_xyz'];ratios=[.01,.05,.1,.15]
lookup={(z['scene'],float(z['ratio']),int(z['seed']),z['arm']):z for z in ov}
lines=['# Spatial MLP 对 PG-ST coefficient 的表示能力审计','',
       '**结论：'+summary['decision']+'。** '+summary['interpretation'],'',
       '关键证据：24组拟合的Pearson为0.0195–0.1209（中位0.0460），Spearman中位0.0355，R²中位仅0.00211。Top-K recall中位91.300%，但Raw本身已达91.244%；PG-ST新增行的恢复率中位仅0.344%、最高2.244%。这组较高的总体overlap主要来自共同Raw base，不能当作correction表示成功。','',
       '真实non-Bank质量：MLP-fit相对PG-ST的PSNR/photo均为23/24变差，平均ΔPSNR=-0.20883 dB（范围-1.11733至+0.04637 dB）；SSIM为23/24变差，LPIPS为22/24变差。PG-ST相对Raw的平均PSNR收益+0.18531 dB，在拟合ranking中变为-0.02352 dB。因此Q3的NO指未稳定保留PG-ST的有效质量收益，不声称所有case均出现相同程度的绝对质量下降。','',
       '| 必须回答的问题 | 结论 |','|---|---|',
       '| Q1. MLP能否拟合PG-ST coefficient？ | '+answer['Q1']+' |',
       '| Q2. MLP能否复现PG-ST hard Top-K？ | '+answer['Q2']+' |',
       '| Q3. MLP-fit ranking是否基本保留PG-ST质量？ | '+answer['Q3']+' |',
       '| Q4. 当前主要问题更像什么？ | '+answer['Q4']+' |','',
       '**固定实验协议**','',
       '覆盖全部现有 PG-ST 模型：三个 scene × 1/5/10/15% × seed2026/2027，共24个。目标是既有 step2000 per-Gaussian coefficient。每个target独立拟合一个 `3→32→32→1 / SiLU` 网络，输入为完整 valid AND observed population 的冻结 AABB normalized XYZ。沿用当前修复版无输出 bias 的 head，共1216参数；hidden为默认初始化、最后层weight=0。target和prediction均在完整population上去均值。','',
       '只优化 full-batch `mean((pred-target)^2)`，FP64，Adam lr=1e-3、betas=(.9,.999)、eps=1e-8、weight_decay=0，固定5000步。没有超参数搜索、early stop、target reweighting、normalization修改、其他feature或architecture。训练模块不导入renderer，不读取GT，不执行sample或Top-K训练。所有正式导出均使用step5000。','',
       '24个模型全部结束、selected sets和rankings锁定SHA后，才独立运行真实quality evaluation。此次是 REPRESENTABILITY ONLY；拟合权重未用于正式SI模型初始化，也不构成render-loss学习结果。','',
       '**Regression指标：全部24个目标**','',
       '| scene | ratio | seed | MSE | RMSE | Pearson | Spearman | target std | pred std | R² |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for z in reg:
    lines.append(f'| {z["scene"]} | {100*float(z["ratio"]):g}% | {z["seed"]} | {float(z["MSE"]):.8f} | {float(z["RMSE"]):.6f} | {float(z["Pearson"]):.5f} | {float(z["Spearman"]):.5f} | {float(z["target_std"]):.6f} | {float(z["prediction_std"]):.6f} | {float(z["R2"]):.6f} |')
lines+=['','R²以零预测器的MSE=`target variance`为基准。Spearman使用含tie的平均rank；PG-ST大批exact-zero coefficient在centering后形成同值，并未加jitter或过滤。','',
        '**Same-K hard ranking：MLP-fit vs PG-ST**','',
        '| scene | ratio | seed | K | intersection | recall | Jaccard | Raw→PG recall | PG新增行数 | 拟合恢复新增行 | 新增行recall |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for z in reg:
    key=z['scene'],float(z['ratio']),int(z['seed']);a=lookup[key+('MLP-fit',)];raw=lookup[key+('Raw',)]
    lines.append(f'| {z["scene"]} | {100*float(z["ratio"]):g}% | {z["seed"]} | {a["K"]} | {a["intersection"]} | {100*float(a["recall"]):.3f}% | {float(a["Jaccard"]):.6f} | {100*float(raw["recall"]):.3f}% | {a["PG_additions_vs_Raw"]} | {a["PG_additions_recovered"]} | {100*float(a["PG_additions_recall"]):.3f}% |')
lines+=['','`PG新增行=PG-ST Top-K 去掉 Raw Top-K`。score_pg与score_mlp均包含相同log RawSI base；因此即使correction拟合很差，总体Top-K仍可很接近。新增行recall用于解释这个共同base效应，未参与训练或改变K。','',
        '**真实render质量：MLP-fit − PG-ST**','',
        '| scene | ratio | seed | split | ΔPSNR | ΔSSIM | ΔLPIPS | Δphoto |','|---|---:|---:|---|---:|---:|---:|---:|']
for z in quality:
    lines.append(f'| {z["scene"]} | {100*float(z["ratio"]):g}% | {z["seed"]} | {"Bank"if z["split"]=="train"else"non-Bank"} | {float(z["psnr_delta"]):+.5f} | {float(z["ssim_delta"]):+.6f} | {float(z["lpips_delta"]):+.6f} | {float(z["photo_delta"]):+.7f} |')
lines+=['','PSNR/SSIM增大为好，LPIPS/photo减小为好。quality_comparison.csv同时保留两组绝对值；paired_statistics.csv保留四指标mean/median/p10/p90/improved-view fraction。未设置新的固定dB质量阈值。','',
        '**收敛与解释边界**','',
        '| scene | ratio | seed | 最后500步MSE相对下降 | 最后100步loss范围/final |','|---|---:|---:|---:|---:|']
for z in conv:lines.append(f'| {z["scene"]} | {100*float(z["ratio"]):g}% | {z["seed"]} | {100*float(z["last500_relative_decrease"]):+.5f}% | {100*float(z["last100_range_over_final"]):.5f}% |')
lines+=['','本轮只检验这套固定tiny XYZ MLP在固定Adam预算下是否实际拟合已有correction；有限优化结果不构成该网络对任意参数的数学不可表示证明。若尾段仍有下降或波动，将这些数值作为限制保留，不自动延长步数、换LR或增大网络。','',
        '**Correctness与可复核产物**','',
        '- coefficient文件逐一SHA核对；candidate source row、Raw base与原training_input二进制exact对齐；normalized XYZ按原冻结AABB重新计算exact，无refit/clamp。',
        '- centered target导出的same-K集合与历史PG-ST selected set exact；输入和target在回归前后exact不变。',
        '- 每个最终MLP权重在CPU独立forward复算prediction；全部MSE/Pearson/Spearman与hard集合独立复算通过。',
        '- 两组各2840次view render，共5680次。PG-ST真实重渲染的逐view四指标与历史记录exact。冻结Gaussian/TF/Master检查通过，Bank/non-Bank列表不交叉。',
        '- 排序沿用source row ascending tie break；两组相同K且行唯一；全部集合在render前锁定，没有以GT/quality选择模型。','',
        f'- [regression_metrics.csv]({LOCAL}/regression_metrics.csv)、[topk_overlap.csv]({LOCAL}/topk_overlap.csv)、[quality_comparison.csv]({LOCAL}/quality_comparison.csv)、[training_curve.csv]({LOCAL}/training_curve.csv)。',
        f'- [paired_statistics.csv]({LOCAL}/paired_statistics.csv)、[convergence_statistics.csv]({LOCAL}/convergence_statistics.csv)、[correctness.json]({LOCAL}/correctness.json)、[audit_summary.json]({LOCAL}/audit_summary.json)。',
        '- 每个target保存step0/250/1000/2500/5000权重、完整5001点loss curve、最终target/prediction/score/selected rows及normalized XYZ/source SHA metadata。',
        f'- [拟合曲线与ranking对照图]({LOCAL}/representability.png)。','',
        '本轮完成后停止，未扩大MLP、改变normalization或输入、加入Fourier feature、重新训练SI、修改production；未stage、commit、push。','']
(ROOT/'SPATIAL_MLP_PGST_REPRESENTABILITY_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')
curve=load('training_curve.csv');fig,axes=plt.subplots(3,2,figsize=(12,10))
for i,scene in enumerate(scenes):
    for ratio in ratios:
        a=[z for z in curve if z['scene']==scene and float(z['ratio'])==ratio and int(z['seed'])==2026]
        initial=float(a[0]['MSE']);axes[i,0].plot([int(z['step'])for z in a],[float(z['MSE'])/initial for z in a],label=f'{100*ratio:g}%')
    axes[i,0].set_ylabel(scene+'\nMSE / target variance');axes[i,0].grid(alpha=.2)
    for seed,marker in [(2026,'o'),(2027,'s')]:
        axes[i,1].plot([100*r for r in ratios],[float(lookup[scene,r,seed,'MLP-fit']['recall'])for r in ratios],marker=marker,label=f'fit {seed}')
        axes[i,1].plot([100*r for r in ratios],[float(lookup[scene,r,seed,'Raw']['recall'])for r in ratios],marker=marker,linestyle='--',label=f'Raw {seed}')
    axes[i,1].set_ylabel('PG-ST Top-K recall');axes[i,1].grid(alpha=.2);axes[i,1].set_xticks([1,5,10,15])
axes[0,0].set_title('Full-population supervised regression (seed 2026)');axes[0,0].legend()
axes[0,1].set_title('Fit vs Raw, same PG-ST target');axes[0,1].legend()
axes[-1,0].set_xlabel('step');axes[-1,1].set_xlabel('ratio (%)')
fig.tight_layout();fig.savefig(ROOT/'representability.png',dpi=160);plt.close(fig)
print('REPORT_WRITTEN')
