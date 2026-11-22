"""所有固定结果完成后生成中文报告；结论由独立证据审阅写入。"""
from pathlib import Path
import json,csv
ROOT=Path(__file__).resolve().parent
def read(n):return json.loads((ROOT/n).read_text())
def load(n):return list(csv.DictReader(open(ROOT/n)))
def table(head,rows):
    def c(v):return str(v).replace('|','\\|')
    return '\n'.join(['| '+' | '.join(map(c,head))+' |','|'+'|'.join(['---']*len(head))+'|']+['| '+' | '.join(map(c,z))+' |'for z in rows])
def f(v,n=6):
    try:return f'{float(v):.{n}f}'
    except ValueError:return str(v)
def sign(v,n=6):return f'{float(v):+.{n}f}'
reg=read('preregistration.json');audit=read('audit_summary.json');interp=read('interpretation.json');check=read('correctness.json');ind=read('independent_checks.json');assert check['all_passed']and ind['all_passed']
q=load('quality_summary.csv');paired=load('paired_statistics.csv');outputs=load('mlp_output_statistics.csv');indirect=load('indirect_update_statistics.csv');sm=load('spatial_smoothness_statistics.csv');smcorr=load('spatial_smoothness_correlations.csv');prom=load('promotion_comparison.csv');seedvar=load('seed_consistency.csv');sampling=load('sampling_statistics.csv');dynamics=load('stochastic_hard_dynamics.csv');groups=load('direct_selection_groups.csv');overlap=load('selected_set_overlap.csv');lr=read('lr_sanity.json')
def get(s,r,arm,seed,split):return next(z for z in q if z['scene']==s and float(z['ratio'])==r and z['arm']==arm and int(z['seed'])==seed and z['split']==split)
def match(rows,s,r,seed):return next(z for z in rows if z['scene']==s and float(z['ratio'])==r and int(z['seed'])==seed)
parts=['# Spatial MLP Implicit SI Audit','','## 结论','',f"**{interp['classification']}**。",'']
for p in interp['conclusions']:parts += [p,'']
parts += ['研究决策：**'+interp['next_step']+'**。','','## 1. 唯一机制变化与冻结边界','',
 'score=`log(RawSI+1e-12)+f_theta(normalized XYZ)`。MLP固定3→32→32→1、两层SiLU，共1217个参数。仅normalized XYZ进入MLP；没有Raw/opacity/scale/SH/visibility/age/cell/hierarchy/C/residual输入。Raw仅以固定base加入最终score。最后Linear权重和bias全零，因此step0输出exact zero。', '',
 '唯一参数化变化是independent per-Gaussian scalar→shared spatial MLP。训练保持旧PG-ST的cached-uniform ST Bernoulli、temperature=1、42轮detached-offset expected-K校准、compact selected renderer、opacity ST、原photo=.8L1+.2(1-fast_ssim)、相同Bank camera顺序。Primary没有使用PL/Exact-K，也没有额外smoothness/TV/Laplacian/LPIPS/rank/distillation loss。', '',
 '三scene×四ratio×两seed，共24个独立MLP，全部从零correction重新训练2000step。Gaussian xyz/scale/rotation/opacity/SH、Master、TF、AABB、Raw/contribution/Bank、candidate mask/cameras/GT、renderer/metrics均冻结。只theta进入Adam；betas=.9/.999、eps=1e-8、weight_decay=0。没有densify/prune/reset/reorder或TF更新。','']
rows=[]
for name in reg['scenes']:
    info=json.loads((ROOT/name/'input_manifest.json').read_text());N=read('input_checks.json')['scenes'][name]['N'];rows.append([name,N,len(info['bank_frames']),len(info['evaluation_frames']),'/'.join(str(int(r*N+.5))for r in reg['ratios']),Path(info['master']).parts[-5]])
parts += [table(['scene','candidate N','Bank','non-Bank eval','K @1/5/10/15%','checkpoint'],rows),'',
 '冻结数据来自`/home/prmi/zwj/per-gaussian-implicit-si-20261020`。PG-ST模型/系数/selected sets/质量全部保留；训练前重新构建A/B/C/F完整ranking、same-K sets与tree并真实渲染exact复现。PG-ST对所有scene/ratio/seed、Bank/eval的逐view四指标重渲染差值全部为0。', '',
 '## 2. Frozen AABB normalization','',
 '严格使用冻结TF的`grid.aabb_min/max`，FP64计算`2*(xyz-min)/(max-min)-1`。不重新拟合AABB、不per-batch normalization、不clip。原冻结candidate中确有少量点在AABB外，公式对其自然外推，故不能宣称所有row都严格位于[-1,1]。这些rows保留在同一candidate population中。','']
rows=[]
for name in reg['scenes']:
    z=json.loads((ROOT/name/'normalization.json').read_text());fmt=lambda a:', '.join(f(v,6)for v in a)
    rows.append([name,fmt(z['aabb_min']),fmt(z['aabb_max']),fmt(z['min']),fmt(z['max']),fmt(z['std']),z['outside_rows']])
parts += [table(['scene','AABB min','AABB max','normalized min','normalized max','normalized std XYZ','outside rows'],rows),'',
 '每scene的`normalized_xyz.npz`和`normalization.json`保存source rows、XYZ、AABB、normalized数组SHA与统计；每模型复制normalization metadata。native训练输入与Python公式逐row exact一致。这里部分scene的大部分candidate集中在很小的normalized坐标范围；该冻结尺度可能影响tiny MLP的可学空间变化，但本轮没有调整它。', '',
 '## 3. LR sanity与初始exact gate','',
 '预注册case为fr1_xyz /5% /seed2026；三个LR各训练250step，数值稳定且更新后，按最终全Bank deterministic hard photo最小者选LR，tie取更小LR。只读取Bank GT/loss与Raw base，没有查询Fixed/C/evaluation结果。sanity结果与正式模型独立保存，未作正式初始化。','']
parts += [table(['LR','initial Bank photo','final Bank photo','Δphoto','final Bank PSNR','max theta grad norm','max |correction|','finite/updated'],[[z['lr'],f(z['initial_Bank_photo']),f(z['final_Bank_photo']),sign(z['photo_delta']),f(z['final_Bank_PSNR']),f(z['theta_grad_norm_max']),f(z['correction_max_abs']),str(z['finite'])+'/'+str(z['updated'])]for z in lr['results']]),'',
 f"统一LR锁定为 **{lr['selected_lr']}**，适用于全部三scene/四ratio/两seed，正式训练和evaluation前已经固定。没有扩大LR搜索或按最终质量重选LR。",'',
 '所有正式模型step0：MLP输出exact zero、完整Raw ranking exact、四ratio Top-K selected sets exact；同scene/ratio/seed的cached uniform与旧PG-ST逐tensor exact相同，全部2000个Bank frame的顺序也exact相同。MLP初始化在cached uniform生成之后完成，避免网络初始化改变旧采样noise。', '',
 '## 4. Primary quality：Bank hard same-K与non-Bank hard same-K','',
 '全部主表采用最终step2000 deterministic hard Top-K，score降序/source row升序打破tie。训练sample quality不用于主结论。PSNR/SSIM越高越好，LPIPS/photo越低越好。Bank/eval均复用原四指标实现；non-Bank是原冻结archived views，参与过原SLAM，但不参与本轮优化。']
for name in reg['scenes']:
    for split,label in [('train','Bank'),('eval','non-Bank evaluation')]:
        parts += ['',f'### {name} / {label}']
        for metric in ['psnr','ssim','lpips','photo']:
            rows=[]
            for r in reg['ratios']:
                for seed in reg['seeds']:
                    vals=[get(name,r,a,-1,split)[metric]for a in ['Raw','Fixed','C']]+[get(name,r,a,seed,split)[metric]for a in ['PG-ST','MLP-ST']];rows.append([f'{r:.0%}',seed]+[f(v)for v in vals]+[sign(float(vals[-1])-float(vals[-2]))])
            parts += ['',f'**{metric.upper()}**','',table(['ratio','seed','Raw','Fixed h8','C','PG-ST','MLP-ST','MLP-PG'],rows)]
parts += ['','## 5. 跨scene/ratio/seed质量判断','',table(['reference','split','PSNR↑ /24','SSIM↑ /24','LPIPS↓ /24','photo↓ /24'],[[reference,split,z['PSNR_positive'],z['SSIM_positive'],z['LPIPS_improved'],z['photo_improved']]for reference,by in audit['counts'].items()for split,z in by.items()]),'']
for p in interp['quality_analysis']:parts += [p,'']
parts += ['![MLP evaluation differences](quality_deltas.png)','','### Paired non-Bank PSNR','',
 'CSV保留四指标paired mean/median/p10/p90/improved-view fraction。下表列出全部模型与四个baseline的PSNR paired结果；没有发明固定dB成功门槛。','']
rows=[]
for z in paired:
    if z['split']=='eval'and z['metric']=='psnr':rows.append([z['scene'],f"{float(z['ratio']):.0%}",z['seed'],z['comparison'],sign(z['mean']),sign(z['median']),sign(z['p10']),sign(z['p90']),f"{float(z['improved_view_fraction']):.2%}"])
parts += [table(['scene','ratio','seed','comparison','mean','median','p10','p90','improved views'],rows),'','## 6. Indirect spatial propagation','',
 'direct_selected_count是2000次Bernoulli admission中实际进入compact renderer的次数，不等价于该点每次都有非零photo gradient。never-selected=count0；rarely=1..1000；frequently=>1000，只用作描述分组，不是新训练gate。每Gaussian的step0/250/1000/2000 correction及direct count均保存在模型`.pt/.npz`；完整final field保存在`final_field.npz`。','',
 '下面用|correction|>1e-12判断数值改变；`audit_summary.json`同时保留never-selected rows中exact非零输出数量。mean score shift就是correction；rank shift由Raw rank减final rank定义，正数表示提升。','']
rows=[]
for name in reg['scenes']:
    for r in reg['ratios']:
        for seed in reg['seeds']:
            z=next(z for z in indirect if z['scene']==name and float(z['ratio'])==r and int(z['seed'])==seed and int(z['step'])==2000 and z['diagnostic']=='cumulative_never_selected');updates=[u for u in indirect if u['scene']==name and float(u['ratio'])==r and int(u['seed'])==seed and u['diagnostic']=='single_step_update']
            rows.append([name,f'{r:.0%}',seed,z['unselected_N'],f"{int(z['unselected_N'])/int(z['N']):.2%}",f"{float(z['unselected_changed_fraction']):.2%}",f"{min(float(u['selected_changed_fraction'])for u in updates):.2%}",f"{min(float(u['unselected_changed_fraction'])for u in updates):.2%}"])
parts += [table(['scene','ratio','seed','never selected N','never share','never correction changed','single-step selected changed min','single-step unselected changed min'],rows),'','![indirect propagation](indirect_updates.png)','']
for p in interp['propagation_analysis']:parts += [p,'']
rows=[]
for name in reg['scenes']:
    for r in reg['ratios']:
        for seed in reg['seeds']:
            for group in ['frequently','rarely','never']:
                a=next(z for z in groups if z['scene']==name and float(z['ratio'])==r and int(z['seed'])==seed and z['group']==group and z['quantity']=='correction');rank=next(z for z in groups if z['scene']==name and float(z['ratio'])==r and int(z['seed'])==seed and z['group']==group and z['quantity']=='Raw_rank_promotion');rows.append([name,f'{r:.0%}',seed,group,a['N'],f(a['mean']),f(a['std']),f(rank['mean'],2),f(rank['p90'],2)])
parts += [table(['scene','ratio','seed','direct group','N','correction mean','correction std','mean Raw rank promotion','p90 promotion'],rows),'','## 7. Spatial smoothness与local field variation','',
 '每scene固定随机种子81020采100000 pairs，所有模型与对应PG-ST使用相同pair fixture。按normalized Euclidean distance的四分位分为very_near/near/medium/far；不是质量gate，也不参与训练。`spatial_smoothness_statistics.csv`保存各bin距离与|a_i-a_j|完整统计，correlations另表同时报告normalized/world distance。','']
rows=[]
for name in reg['scenes']:
    for r in reg['ratios']:
        for seed in reg['seeds']:
            for arm in ['MLP-ST','PG-ST']:
                z=[next(z for z in sm if z['scene']==name and float(z['ratio'])==r and int(z['seed'])==seed and z['arm']==arm and z['distance_bin']==b)for b in ['very_near','near','medium','far']];c=next(z for z in smcorr if z['scene']==name and float(z['ratio'])==r and int(z['seed'])==seed and z['arm']==arm);rows.append([name,f'{r:.0%}',seed,arm]+[f(a['mean'])for a in z]+[f(c['corr_normalized_distance_abs_difference']),f(c['corr_world_distance_abs_difference'])])
parts += [table(['scene','ratio','seed','arm','very-near mean |Δa|','near','medium','far','corr normalized','corr world'],rows),'','![smoothness](spatial_smoothness.png)','',
 '冻结target-2048 root grid仅在训练完成后参与分组。`root_field_statistics.csv`逐root保存occupancy、correction mean/std/range、hard selected count/ratio、direct selected count sum、mean Raw SI；root ID没有进入MLP、score或训练。','']
for p in interp['smoothness_analysis']:parts += [p,'']
parts += ['## 8. Correction分布、采样与训练动态','',
 '公共correction shift会被offset calibration抵消；只有relative score变化影响ranking。因此输出变化覆盖、mean(a)或exp(a)增大都不能单独作为有效传播的质量证据。std(a)与pair difference去除了公共shift的影响。本轮没有中心化或clip，严格保留旧ST训练语义。','']
rows=[]
for name in reg['scenes']:
    for r in reg['ratios']:
        for seed in reg['seeds']:
            z=next(z for z in outputs if z['scene']==name and float(z['ratio'])==r and int(z['seed'])==seed and int(z['step'])==2000 and z['quantity']=='correction');e=next(z for z in outputs if z['scene']==name and float(z['ratio'])==r and int(z['seed'])==seed and int(z['step'])==2000 and z['quantity']=='exp_correction');s=match(sampling,name,r,seed)
            rows.append([name,f'{r:.0%}',seed,f(z['mean']),f(z['median']),f(z['std']),f(z['p1']),f(z['p99']),f(z['max_abs']),f(e['p10'],3),f(e['p90'],3),s['sampled_initial'],s['sampled_final']])
parts += [table(['scene','ratio','seed','mean a','median a','std a','p1','p99','max|a|','exp(a) p10','exp(a) p90','sample N0','sample Nfinal'],rows),'',
 '完整p1/p10/p90/p99与exp correction分布在`mlp_output_statistics.csv`。每step记录theta total/layer-wise gradient norm、sampled photo/PSNR、correction mean/std/max。每250step保存整个population更新前后输出和selected mask；全部192个output-update snapshots保留。派生exp factor若超出FP64表示范围会明确标记，不clip原correction。','']
rows=[]
for z in dynamics:rows.append([z['scene'],f"{float(z['ratio']):.0%}",z['seed'],sign(z['stochastic_photo_delta']),sign(z['hard_photo_delta']),sign(z['hard_psnr_delta']),f(z['final_Jaccard_Raw']),f(z['last100_Jaccard'])])
parts += [table(['scene','ratio','seed','sample Δphoto first5→last5','hard Δphoto','hard ΔPSNR','final Jaccard Raw','last100 Jaccard'],rows),'','![hard Bank dynamics](hard_training_curves.png)','',
 'sample Δphoto按每Bank frame最早5次与最后5次sample平均后再等权汇总；采样数量可变化，不是主same-K结果。每100step和250step的hard诊断始终same-K。没有根据中途曲线挑checkpoint，formal export统一step2000。','']
for p in interp['training_analysis']:parts += [p,'']
parts += ['## 9. Boundary crossing与Raw低排名promotion','']
rows=[]
for z in prom:rows.append([z['scene'],f"{float(z['ratio']):.0%}",z['seed'],z['MLP_added'],z['PG_added'],f(z['MLP_Raw_pct_mean']),f(z['PG_Raw_pct_mean']),f(z['MLP_Raw_pct_p90']),f(z['PG_Raw_pct_p90']),z['MLP_max_Raw_rank'],z['PG_max_Raw_rank'],z['MLP_gt2K'],z['PG_gt2K']])
parts += [table(['scene','ratio','seed','MLP added','PG added','MLP pct mean','PG pct mean','MLP pct p90','PG pct p90','MLP max rank','PG max rank','MLP rank>2K','PG rank>2K'],rows),'',
 'Raw percentile=(Raw 1-based rank-1)/(N-1)，越高代表更靠后。完整新增source rows见`raw_rank_promotion_rows.csv`，四个baseline的intersection/Jaccard/added/removed见`selected_set_overlap.csv`及`selected_sets.npz`。','']
for p in interp['promotion_analysis']:parts += [p,'']
parts += ['## 10. Seed sensitivity与capacity判断','']
rows=[]
for name in reg['scenes']:
    for r in reg['ratios']:
        z=[next(z for z in seedvar if z['scene']==name and float(z['ratio'])==r and z['split']=='eval'and z['arm']==arm)for arm in ['PG-ST','MLP-ST']];rows.append([name,f'{r:.0%}',f(z[0]['psnr_seed_abs_difference']),f(z[1]['psnr_seed_abs_difference']),f(z[0]['photo_seed_abs_difference']),f(z[1]['photo_seed_abs_difference'])])
parts += [table(['scene','ratio','PG seed |ΔPSNR|','MLP seed |ΔPSNR|','PG seed |Δphoto|','MLP seed |Δphoto|'],rows),'']
for p in interp['capacity_analysis']:parts += [p,'']
parts += ['没有追加capacity sweep、Fourier features、grid/hash encoding或optional teacher fitting；没有MLP-PL secondary arm。tiny MLP本身的可表示范围明显小于独立a，但本轮不能凭训练失败将全部差距归于representation或平滑性。', '',
 '## 11. Correctness与限制','',
 f"96个snapshot的native权重由独立Python MLP复算，最大输出误差{ind['independent_Python_weight_forward_max_abs']:.3e}（CPU/GPU FP64数值差）；独立tuple sort验证完整ranking和same-K sets exact。归一化从冻结AABB重算exact，逐view质量聚合误差为0。",'',
 '全部模型step0 output zero/full Raw ranking/四ratio sets exact；只有theta进optimizer；Gaussian/TF/Master及camera/GT tensor训练前后exact无变化、无gradient。Raw/Bank/TF/Master/cameras/candidate mask/runtime文件SHA终验不变。所有2000-step Bank schedule和cached uniform与PG-ST exact一致。MLP训练不读取C/Fixed/evaluation质量；LR仅从预注册Bank case确定。', '',
 'final hard sets在新质量渲染前固定SHA，全部来源于同一candidate population，唯一且same-K。Native训练模型只接XYZ，训练函数在取GT前强制frame属于Bank。checkpoint/midpoint不用于evaluation选择。', '',
 '解释边界：本轮是tiny MLP + 指定frozen-AABB normalization + 原PG-ST surrogate的组合。MLP公共bias共享可改变优化动力学；offset detach仍保留旧ST近似梯度。低capacity smooth sharing效果、normalization压缩、LR/优化器和surrogate相互作用未被单独干预隔离。本轮不通过新规则消除这些问题，也不把输出传播正确等同于sharing有质量价值。', '',
 '## 12. 对十项问题的明确回答','',table(['问题','回答','证据/解释'],[[f'Q{i+1}',z['answer'],z['reason']]for i,z in enumerate(interp['answers'])]),'',
 '## 13. 交付与停止','',
 '`quality_summary.csv`/`quality_per_view.csv`/`paired_statistics.csv`保留Bank、eval四指标和两个seed。`training_curves.csv`/`hard_training_dynamics.csv`保存48000step及全Bank hard曲线；`lr_sanity.json`保存三次Bank-only LR检查与锁定依据。', '',
 '`mlp_output_statistics.csv`/`indirect_update_statistics.csv`/`direct_selection_groups.csv`/`spatial_smoothness_statistics.csv`/`root_field_statistics.csv`保存共享传播和场变化。`selected_set_overlap.csv`/`raw_rank_promotion.csv`及row明细保存真实boundary crossing。', '',
 '`rankings.npz`/`selected_sets.npz`保存全部source rows和prefix；每模型保存step0/250/1000/2000 MLP weights、normalization metadata、correction/score/ranking/direct counts，以及每250step输出更新快照；`final_field.npz`为完整population的最终field。`correctness.json`/`independent_checks.json`/`audit_summary.json`/`artifact_manifest.json`保存门控与SHA。', '',
 '本轮完成后停止。没有自动改变MLP、loss、训练器、grid、split或online；没有stage、commit、push。','']
(ROOT/'SPATIAL_MLP_IMPLICIT_SI_AUDIT.md').write_text('\n'.join(parts),encoding='utf-8');audit['decision']=interp['classification'];audit['next_step']=interp['next_step'];audit['answers']=interp['answers'];(ROOT/'audit_summary.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n');print('REPORT_WRITTEN',flush=True)
