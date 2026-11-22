"""从锁定的完整质量表生成中文报告与独立科学图，不选择训练配置。"""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
LOCAL='H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_calibrated_quality_20261020'
def load(n):return list(csv.DictReader(open(ROOT/n)))
s=json.loads((ROOT/'audit_summary.json').read_text());q=load('quality_summary.csv');d=load('quality_deltas.csv');p=load('paired_statistics.csv');t=load('training_dynamics_summary.csv');o=load('selected_set_overlap.csv')
arm='MLP-ST-fixed';refs=['MLP-ST-legacy','PG-ST','Raw','Fixed','C'];scenes=['fr1_room','fr1_xyz','fr2_xyz'];ratios=[.01,.05,.1,.15]
qm={(z['scene'],z['split'],z['arm'],float(z['ratio']),int(z['seed'])):z for z in q}
dm={(z['scene'],z['split'],float(z['ratio']),int(z['seed']),z['reference']):z for z in d}
lines=['# 修复版 Spatial MLP 正式质量测试','',
       '**结论：修复明显减轻了旧 MLP 的退化，但当前 Spatial MLP 仍没有稳定的质量优势。** non-Bank 相对旧 MLP 的 PSNR/photo 均为 23/24 改善，平均 ΔPSNR=+2.28307 dB；相对 PG-ST 的 PSNR 仅 1/24 改善、photo 0/24；相对 Raw 的 PSNR/photo 均为 7 改善、1 相同、16 变差，平均 ΔPSNR=-0.10241 dB。相对 Fixed/C 的平均 ΔPSNR 分别为 -0.48990/-0.49688 dB，LPIPS 均为 24/24 更差。','',
       '因此：已证实的梯度缺陷得到修正，但空间共享的质量假设仍未被支持。当前不进入 shared grid 或 split，保留 Current C；本轮停止。失败不能仅归因于 generalization：Bank hard photo 也只有 8/24 改善，sampled photo 只有 7/24 改善。残余的 optimization、ST objective 与空间参数化限制尚未被单独隔离，不能唯一归因于表达能力或平滑性。','',
       '本轮固定三个 scene × 四个 ratio × 两个 seed，完成 24 个独立模型、各 2000 steps，共 48,000 次更新。统一 LR=1e-4，固定 cached-uniform ST Bernoulli、Bank schedule、photo loss 与 Adam；完整 population differentiable centering + implicit expected-K VJP + 无输出 bias。正式结果均为 step2000 deterministic hard same-K，未用 non-Bank 选择 LR、step 或模型。','',
       '本轮相对 legacy MLP 同时修正了公共输出方向与完整预算校准导数，因此不能从本轮单独归因两者各自的质量贡献。', '',
       '**主要结果：每项均为 24 个 scene/ratio/seed case，改善方向使用未舍入数值**','',
       '| 对照 | split | PSNR 改善/相同/变差 | SSIM 改善 | LPIPS 改善 | photo 改善 | 平均 ΔPSNR | ΔPSNR 范围 |',
       '|---|---|---:|---:|---:|---:|---:|---:|']
for ref in refs:
    for split in ['train','eval']:
        a=s['counts'][ref][split];b=a['psnr']
        lines.append(f'| {ref} | {"Bank"if split=="train"else"non-Bank"} | {b["improved"]}/{b["equal"]}/{b["worse"]} | {a["ssim"]["improved"]} | {a["lpips"]["improved"]} | {a["photo"]["improved"]} | {b["mean_delta"]:+.5f} | {b["min_delta"]:+.5f} … {b["max_delta"]:+.5f} |')
lines+=['','改善数目不能单独表示收益大小，必须结合下面的 delta、paired-view 分布及最终集合改动。四指标的完整 aggregate 与 per-view 数值均保存在 CSV。','',
        '**non-Bank：全部 24 个模型**','',
        '| scene | ratio | seed | PSNR | SSIM | LPIPS | photo | ΔPSNR legacy MLP | ΔPSNR PG | ΔPSNR Raw | ΔPSNR Fixed | ΔPSNR C |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for name in scenes:
    for ratio in ratios:
        for seed in [2026,2027]:
            a=qm[name,'eval',arm,ratio,seed];delta=[float(dm[name,'eval',ratio,seed,ref]['psnr'])for ref in refs]
            lines.append(f'| {name} | {100*ratio:g}% | {seed} | {float(a["psnr"]):.5f} | {float(a["ssim"]):.6f} | {float(a["lpips"]):.6f} | {float(a["photo"]):.6f} | '+' | '.join(f'{x:+.5f}'for x in delta)+' |')
lines+=['','**Paired-view：non-Bank PSNR**','',
        '| 对照 | 正 median 的 case | 多数 view 改善的 case | case improved-view fraction 中位数 | case improved-view fraction 范围 |','|---|---:|---:|---:|---:|']
for ref in refs:
    a=[z for z in p if z['split']=='eval'and z['reference']==ref and z['metric']=='psnr'];f=[float(z['improved_view_fraction'])for z in a];b=s['counts'][ref]['eval']
    lines.append(f'| {ref} | {b["PSNR_paired_positive_median_cases"]}/24 | {b["PSNR_majority_views_improved_cases"]}/24 | {np.median(f):.3f} | {min(f):.3f} … {max(f):.3f} |')
lines+=['','每个 case 的 mean、median、p10、p90、improved/equal view fraction，以及四指标完整统计见 paired_statistics.csv；没有使用新的 dB 阈值。','',
        '**训练动态与真实 reordering**','',
        f'Bank hard photo 比初始化降低：{s["training"]["Bank_hard_photo_improved"]}/24。按每个 Bank frame 首末各五次访问等权比较，sampled photo 降低：{s["training"]["sampled_photo_improved"]}/24。最终相对 Raw 存在 boundary crossing：{s["training"]["Raw_boundary_crossing_cases"]}/24。','',
        '| scene | ratio | seed | Bank hard Δphoto | Bank hard ΔPSNR | final correction std | Raw added | Jaccard vs Raw |','|---|---:|---:|---:|---:|---:|---:|---:|']
for z in t:
    a=next(a for a in o if a['scene']==z['scene']and a['ratio']==z['ratio']and a['seed']==z['seed']and a['reference']=='Raw')
    lines.append(f'| {z["scene"]} | {100*float(z["ratio"]):g}% | {z["seed"]} | {float(z["hard_Bank_photo_delta"]):+.7f} | {float(z["hard_Bank_PSNR_delta"]):+.5f} | {float(z["correction_std_final"]):.7f} | {a["added"]} | {float(a["Jaccard"]):.6f} |')
lines+=['','seed_consistency.csv 保存新旧 MLP 与 PG 的跨 seed 四指标差异。若新模型更接近 Raw、集合变化很少，较小 seed 差异不能单独解释为有效 regularization。','',
        '**Correctness 与冻结条件**','',
        '- 五组历史基线均用原冻结 renderer 重渲染，9940 个 view 的 PSNR/SSIM/LPIPS/photo 与历史记录逐项 exact，aggregate exact。',
        '- 修复版 2840 个 view：Bank 808、non-Bank 2032；全部为唯一且严格 same-K 的 hard 集合。训练 admission 仍为期望 K 的旧 Bernoulli 协议。',
        '- 24 个模型的 cached uniform 与 Bank frame schedule exact 对齐 PG-ST，初始化完整排序 exact Raw；step0/250/1000/2000 的权重经独立 Python forward 核对。',
        '- Gaussian/TF/Master native tensor exact，输入/相机/Bank/AABB/runtime 文件训练前后 SHA 检查通过。',
        f'- 全部更新中最大 normalized gradient sum 为 {max(float(z["gradient_sum_ratio_max"])for z in t):.3g}；correction 均值最大绝对值 {max(float(z["correction_mean_abs_max"])for z in t):.3g}。',
        '- 最终 selected sets/rankings 在新模型 quality render 前锁定 SHA；没有 GT-driven 选点、eval tuning、early stopping、regularization、grid 或 split。','',
        '首次 gate 因新脚本额外关闭 cuDNN TF32 导致 LPIPS 差异而停止，未训练任何模型。单帧对照确认历史 LPIPS 使用 cuDNN TF32=True；恢复原 metric 环境后才重跑并通过全部 exact gate，没有改变 metric 公式或放宽容差。首次失败日志与环境诊断均保留。','',
        '**输出**','',
        f'- [quality_summary.csv]({LOCAL}/quality_summary.csv)、[quality_per_view.csv]({LOCAL}/quality_per_view.csv)、[paired_statistics.csv]({LOCAL}/paired_statistics.csv)。',
        f'- [training_curves.csv]({LOCAL}/training_curves.csv)、[hard_training_dynamics.csv]({LOCAL}/hard_training_dynamics.csv)、[training_dynamics_summary.csv]({LOCAL}/training_dynamics_summary.csv)。',
        f'- [selected_set_overlap.csv]({LOCAL}/selected_set_overlap.csv)、[raw_rank_promotion.csv]({LOCAL}/raw_rank_promotion.csv)、[correction_statistics.csv]({LOCAL}/correction_statistics.csv)、[seed_consistency.csv]({LOCAL}/seed_consistency.csv)。',
        f'- [correctness.json]({LOCAL}/correctness.json)、[audit_summary.json]({LOCAL}/audit_summary.json)、rankings.npz、selected_sets.npz。',
        '- 每个模型均保存 step0/250/1000/2000 的 MLP 权重和 tensor 快照、final_field.npz、normalization.json 与原始 training.log。','',
        '本轮完成后停止，未修改 architecture、LR、loss、sampling protocol，也未实现 grid/split；没有 stage、commit、push。','']
(ROOT/'SPATIAL_MLP_CALIBRATED_QUALITY_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')
fig,axes=plt.subplots(3,3,figsize=(12,8),sharex=True,sharey='col')
for i,name in enumerate(scenes):
    for j,ref in enumerate(refs[:3]):
        ax=axes[i,j]
        for seed,marker in [(2026,'o'),(2027,'s')]:ax.plot([1,5,10,15],[float(dm[name,'eval',ratio,seed,ref]['psnr'])for ratio in ratios],marker=marker,label=str(seed),linewidth=1)
        ax.axhline(0,color='black',linewidth=.7);ax.set_xticks([1,5,10,15]);ax.grid(alpha=.2)
        if i==0:ax.set_title('Fixed MLP - '+ref)
        if j==0:ax.set_ylabel(name+'\nPSNR delta (dB)')
        if i==2:ax.set_xlabel('ratio (%)')
axes[0,0].legend();fig.tight_layout();fig.savefig(ROOT/'quality_deltas.png',dpi=170);plt.close(fig)
print('REPORT_GENERATED')
