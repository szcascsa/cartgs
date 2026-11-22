"""根据全部固定实验生成中文机制报告；结论必须先由证据审阅写入interpretation.json。"""
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parent
def read(n):return json.loads((ROOT/n).read_text())
def load(n):return list(csv.DictReader(open(ROOT/n)))
reg=read('preregistration.json');audit=read('audit_summary.json');correct=read('correctness.json');interpret=read('interpretation.json')
assert correct['all_passed']
q=load('quality_summary.csv');paired=load('paired_statistics.csv');coef=load('coefficient_statistics.csv');grad=load('gradient_statistics.csv');overlap=load('selected_set_overlap.csv');sampling=load('sampling_train_quality.csv');dyn=load('hard_training_dynamics.csv');models=load('model_training_checks.csv')
def getq(scene,ratio,arm,split='eval',seed=-1,step=0):return next(z for z in q if z['scene']==scene and float(z['ratio'])==ratio and z['arm']==arm and z['split']==split and int(z['seed'])==seed and int(z['step'])==step)
def match(rows,s,r,seed):return next(z for z in rows if z['scene']==s and float(z['ratio'])==r and int(z['seed'])==seed)
def table(headers,rows):
    def cell(v):return str(v).replace('|','\\|')
    return '\n'.join(['| '+' | '.join(map(cell,headers))+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(cell,r))+' |'for r in rows])
def ff(v,n=6):return f'{float(v):.{n}f}'
def signed(v,n=6):return f'{float(v):+.{n}f}'
parts=['# Per-Gaussian Implicit SI Upper-Bound Audit','', '## 结论','',f"**{interpret['classification']}**。",'',*interpret['conclusions'],'','研究决策：**'+interpret['next_step']+'**。本轮结束后停止；没有实现 shared grid coefficient、adaptive split、regularization、C distillation 或 online mapping。没有 stage、commit、push。','',
 '## 1. 实验对象与完成边界','',
 '本轮测试的是 `score_i = log(RawSI_i + 1e-12) + a_i`，每个 valid AND observed Gaussian 一个独立 FP64 scalar。Stage A 单独验证可表示性；Stage B 的 24 个模型全部从 a=0 开始，只使用 Bank rendering photo loss。独立 ratio 模型不共享 a，也不要求彼此构成同一跨 ratio prefix。', '',
 '所有 non-Bank evaluation views 保持原审计冻结列表。它们参与过原 SLAM，但从未进入本轮 coefficient optimization；本实验不宣称全新场景泛化。']
rows=[]
for s in reg['scenes']:
    info=json.loads((ROOT/s/'input_manifest.json').read_text());pop=json.loads((ROOT/s/'population.json').read_text());N=read('representability_check.json')['scenes'][s]['N']
    rows.append([s,N,len(info['bank_frames']),len(info['evaluation_frames']),'/'.join(str(int(r*N+.5))for r in reg['ratios']),Path(info['master']).parts[-5]])
parts+=['',table(['scene','candidate N','Bank views','eval views','K @1/5/10/15%','checkpoint'],rows),'',
 '冻结来源：`/home/prmi/zwj/final-rep-residual-fallback-20261020`。每场景 `input_manifest.json`、`population.json` 与根目录 `representability_check.json` 保存模型、TF、GT/cameras、Raw/contribution、运行库路径及 SHA。未重建 Bank、未训练 Gaussian/TF、未 densify/prune。', '',
 '## 2. Baseline gate 与 representability','',
 'A=Raw、B=Fixed h8 geometry-only marginal、C=Current C、附加 F=Final reps + C score，均重新构建完整 ranking、selected sets 与 tree，并进行了真实渲染复现。三个场景四 ratio 的 aggregate 和每 view 四指标最大绝对差全部为 **0**。所有正式新训练/评价在 baseline gate 之后执行。', '',
 'a=0 的完整 score ranking 与 Raw exact 相同。独立构造 `target=-rank_C/N`、`aC=target-base` 后，完整排序和四 ratio selected sets 与 C exact 相同。aC 仅保存在 `representability_only.npz`，没有用于 PG 初始化、loss、采样或监督。这证明参数化能够表示 C，**不代表 rendering loss 学会了 C**。', '',
 '## 3. 正式训练协议及采样边界','',
 '源码 `include/selector_handoff.h::sampleSelectorST` 是 cached-uniform ST Bernoulli；`GumbelNetwork::gumbelSoftmax` 是逐 Gaussian 两类别采样，也不是全局 Gumbel Top-K。用户确认本轮采用原 ST Bernoulli，加不参与学习的公共 offset 校准期望预算。没有新建 differentiable Top-K。', '',
 '`p_i = sigmoid(score_i - offset)`；每步 42 次二分使 `sum(p)=K`，offset detach，temperature=1。`mask = 1[u_i<p_i] + (p_i-p_i.detach())`，每模型按 seed 固定 cached uniform。仅实际 selected rows 进入 compact renderer，mask 乘到 opacity 上；反向只能回到被采样 rows 的 a。offset 不改变 hard ranking；训练实际点数可以偏离 K，**主评价始终是 hard same-K**。', '',
 table(['项目','固定值'],[['优化器','Adam，lr=.001，betas=(.9,.999)，eps=1e-8，weight decay=0'],['步数','每模型 2000，无 early stopping；3 scene ×4 ratio ×2 seed=24 模型，48000 正式更新'],['记录','step 0/250/1000/2000 tensor、完整 ranking、selected set；每50步记录训练 sample loss/PSNR 与 hard-set 变化'],['数据','仅 Bank；固定 seed shuffle 后逐 epoch 遍历；seed 2026、2027 全部报告'],['Loss','.8 L1 + .2 (1-fast_ssim)，复用项目实现'],['数值','base/a/score FP64，opacity/rasterizer FP32；现有 1e-12 安全 epsilon，未调 epsilon'],['选择规则','score降序，source row升序打破tie；K=floor(r*N+.5)'],['禁止项','没有 ratio loss、distillation、rank/coverage loss、正则、coefficient clip 或 LR sweep']]), '',
 'lr 直接复用 `cfg/selector/fast.yaml` 的 Selector.learning_rate。最大 step、最终 checkpoint、全部 sampling/optimizer 参数在新质量评价前固定；没有按 evaluation PSNR 选参数或挑 seed。', '',
 '训练输入为单独的 `training_input.bin`（rows、base）与 `training_manifest.json`（冻结 map/TF/cameras、Bank IDs），训练程序不读取 C/B ranking、tree、representative/residual label。cameras archive 含冻结完整列表，但训练函数在索引 GT 前强制 frame 属于 Bank。', '',
 '## 4. Primary quality：所有 seed、same-K','', 'PSNR/SSIM 越高越好，LPIPS/photo 越低越好。photo 为原始 production loss；其余三指标完全复用旧 frozen evaluation worker/metric 实现。']
for s in reg['scenes']:
    parts+=['',f'### {s}']
    for m in ['psnr','ssim','lpips','photo']:
        rows=[]
        for r in reg['ratios']:
            vals=[getq(s,r,arm)[m]for arm in['A','B','C']]+[getq(s,r,f'PG-{int(r*100)}',seed=seed,step=reg['steps'])[m]for seed in reg['seeds']]
            rows.append([f'{r:.0%}']+[ff(v)for v in vals])
        parts+=['',f'**{m.upper()}**','',table(['ratio','Raw','Fixed h8','C','PG seed2026','PG seed2027'],rows)]
parts+=['','## 5. PG 相对三条强基线的结果','',table(['reference','split','PSNR↑ /24','SSIM↑ /24','LPIPS↓ /24','photo↓ /24'],[[ref,split,z['PSNR_positive'],z['SSIM_positive'],z['LPIPS_improved'],z['photo_improved']]for ref,a in audit['counts'].items()for split,z in a.items()]),'',*interpret['quality_analysis'],'','![evaluation quality deltas](eval_quality_deltas.png)','',
 '下表按每 scene/ratio/seed 输出 PSNR aggregate 差；paired 统计紧随其后。CSV 保留全部四指标和所有 view，未以 aggregate 替代 per-view 证据。']
rows=[]
for s in reg['scenes']:
    for r in reg['ratios']:
        for seed in reg['seeds']:
            a=float(getq(s,r,f'PG-{int(r*100)}',seed=seed,step=reg['steps'])['psnr'])
            rows.append([s,f'{r:.0%}',seed]+[signed(a-float(getq(s,r,arm)['psnr']))for arm in['A','B','C']])
parts+=['',table(['scene','ratio','seed','PG-Raw dB','PG-Fixed dB','PG-C dB'],rows),'','### Paired evaluation PSNR','']
rows=[]
for z in paired:
    if z['split']=='eval'and z['metric']=='psnr':rows.append([z['scene'],f"{float(z['ratio']):.0%}",z['seed'],z['comparison'],signed(z['mean']),signed(z['median']),signed(z['p10']),signed(z['p90']),f"{float(z['improved_view_fraction']):.2%}"])
parts+=[table(['scene','ratio','seed','comparison','mean','median','p10','p90','improved views'],rows),'','## 6. Training views、目标函数与 hard-ranking export','',
 '下表使用与训练严格相同的 Bank camera 集合，但将参数导出为 deterministic hard same-K Top-K；因此可以直接对比 Raw。训练日志中的单帧 Bernoulli sample loss 会受 camera/sample 数变化影响，不能与该表混为同一统计量。','']
rows=[]
for s in reg['scenes']:
    for r in reg['ratios']:
        for seed in reg['seeds']:
            a=getq(s,r,f'PG-{int(r*100)}','train',seed,reg['steps']);b=getq(s,r,'A','train')
            rows.append([s,f'{r:.0%}',seed,ff(b['psnr']),ff(a['psnr']),signed(float(a['psnr'])-float(b['psnr'])),ff(b['photo']),ff(a['photo']),signed(float(a['photo'])-float(b['photo']))])
parts+=[table(['scene','ratio','seed','Raw train PSNR','PG train PSNR','ΔPSNR','Raw photo','PG photo','Δphoto'],rows),'','![Bank hard ranking dynamics](training_hard_curves.png)','',
 '为核实实际训练目标是否改善，另对每个模型 step0/final 的 cached Bernoulli mask 在全部 Bank views 重渲染。该项只诊断 optimization，**不是 same-K quality arm，也不用于选择任何 checkpoint**。采样数量如下，不能把它的质量改变量直接解释为相同预算下的 ranking 改善。','']
rows=[]
for s in reg['scenes']:
    for r in reg['ratios']:
        for seed in reg['seeds']:
            a=next(z for z in sampling if z['scene']==s and float(z['ratio'])==r and int(z['seed'])==seed and int(z['step'])==0);b=next(z for z in sampling if z['scene']==s and float(z['ratio'])==r and int(z['seed'])==seed and int(z['step'])==reg['steps'])
            rows.append([s,f'{r:.0%}',seed,a['sampled_N'],b['sampled_N'],ff(a['photo']),ff(b['photo']),signed(float(b['photo'])-float(a['photo'])),signed(float(b['psnr'])-float(a['psnr']))])
parts+=[table(['scene','ratio','seed','sample N0','sample Nfinal','photo0','photo final','Δphoto','ΔPSNR'],rows),'',*interpret['training_analysis'],'','## 7. Selected-set 调整','',
 'Added/removed 是相对同 K baseline 的真实 source-row 变化，不要求 PG 模仿 C。完整 source-row 差集保存在 `selected_sets.npz` 的 added_vs/removed_vs 键。','']
rows=[]
for s in reg['scenes']:
    for r in reg['ratios']:
        for seed in reg['seeds']:
            z=[next(z for z in overlap if z['scene']==s and float(z['ratio'])==r and int(z['seed'])==seed and z['reference']==arm)for arm in ['A','B','C']]
            rows.append([s,f'{r:.0%}',seed]+[f"{float(a['jaccard']):.6f} / {a['added']}"for a in z])
parts+=[table(['scene','ratio','seed','Raw Jaccard / added','Fixed Jaccard / added','C Jaccard / added'],rows),'','## 8. Coefficient 与梯度诊断','',
 '下表为 final a 分布摘要。`coefficient_statistics.csv` 完整保存 a、exp(a) 的 mean/median/std/p1/p10/p90/p99/max|·| 和与 log Raw 的相关性；`coefficient_raw_percentiles.csv` 按 Raw rank decile 分组。coefficient 大小只作诊断，不被当作错误或筛选条件。','']
rows=[]
for s in reg['scenes']:
    for r in reg['ratios']:
        for seed in reg['seeds']:
            a=next(z for z in coef if z['scene']==s and float(z['ratio'])==r and int(z['seed'])==seed and int(z['step'])==reg['steps']and z['quantity']=='a')
            e=next(z for z in coef if z['scene']==s and float(z['ratio'])==r and int(z['seed'])==seed and int(z['step'])==reg['steps']and z['quantity']=='exp_a')
            rows.append([s,f'{r:.0%}',seed,ff(a['mean']),ff(a['median']),ff(a['std']),ff(a['max_abs']),ff(e['p10']),ff(e['p90']),ff(a['corr_with_log_Raw'])])
parts+=[table(['scene','ratio','seed','mean a','median a','std a','max|a|','exp(a) p10','exp(a) p90','corr(a,logRaw)'],rows),'',
 '每个 a 保存 gradient EMA（decay=.99）、历次最大绝对 gradient、累计绝对 update、非零 gradient step 数。所有梯度来自 compact selected opacity 路径；未选 Gaussian 不存在有效的 addition gradient。本轮没有为未选 rows 伪造梯度。','']
rows=[]
for z in grad:rows.append([z['scene'],f"{float(z['ratio']):.0%}",z['seed'],z['ever_gradient'],z['N'],f"{float(z['never_gradient_fraction']):.2%}",f"{float(z['max_abs_gradient']):.4e}",f"{float(z['mean_gradient_EMA']):.4e}"])
parts+=[table(['scene','ratio','seed','ever nonzero gradient','N','never gradient','max|gradient|','mean gradient EMA'],rows),'',*interpret['gradient_analysis'],'','## 9. 空间集中性：仅诊断，不参与训练','',
 '只在所有训练结束后读取已冻结 target-2048 root ID。下表的 top10% 是描述分位，不是 GO 或 split threshold。按 root 总量排序可能受 occupancy 影响，因此同时给出同一组 roots 内的 candidate population 占比，以及 per-root 平均 gradient 的 CV。不能仅由梯度总量集中推出 adaptive split 有效。','']
rows=[]
for key,z in audit['states'].items():
    scene=key.split('_PG_')[0];ratio=key.split('_PG_')[1].split('_')[0];seed=key.split('_seed_')[1].split('_')[0]
    rows.append([scene,ratio+'%',seed,f"{z['gradient_top10percent_root_mass_fraction']:.2%}",f"{z['gradient_top10percent_roots_candidate_fraction']:.2%}",f"{z['coefficient_top10percent_root_abs_mass_fraction']:.2%}",f"{z['coefficient_top10percent_roots_candidate_fraction']:.2%}",ff(z['root_mean_gradient_CV'],3)])
parts+=[table(['scene','ratio','seed','top gradient roots: grad mass','same roots: candidates','top coefficient roots: |a| mass','same roots: candidates','mean-gradient CV'],rows),'','![spatial concentration](gradient_spatial_concentration.png)','',*interpret['spatial_analysis'],'','## 10. Correctness、复现性与限制','',
 f"全部 correctness gate 通过，详见 `correctness.json`。2000 步中每步校准期望 K 的最大绝对误差为 {correct['max_expected_K_error']:.3e}；这是数值残差，不是实际 Bernoulli sampled N 与 K 的差。",'',
 '所有 Gaussian 原始 tensor、TF parameter/buffer、camera/GT tensor、maturity 在训练前后逐 tensor exact 比较，均未变化且无 gradient。Master/TF/config/cameras、Raw/contribution、candidate mask、tree/rankings、运行库同时验证文件 SHA 未变化。Optimizer 仅持有一个 a tensor；没有任何模型参数更新。', '',
 f"重复 5-step CUDA smoke：coefficient 最大绝对差 {correct['determinism']['repeat']['coefficient']['max_abs']:.3e}，bitwise exact={correct['determinism']['repeat']['coefficient']['exact']}；完整 ranking exact={correct['determinism']['full_ranking_exact']}，四个 selected sets exact={correct['determinism']['all_four_selected_sets_exact']}。cached uniform/hit steps exact；反向不宣称 bitwise deterministic。固定 coefficient 下重复 score 构造/排序 exact。两 seed 正式结果全部保留。",'',
 '限制：只测试一个已预注册 LR、固定 2000-step 预算和现有 cached-uniform ST sampling。不是对所有 optimizer、温度、采样刷新策略或可微排序方法的全称否定；独立 per-ratio 模型也不能证明单一 static ranking 的跨 ratio 能力。由于 Stage A 已证明可表示 C，正式学习的失败不能轻易归因于 REPRESENTATION。', '',
 '## 11. 对八项问题的明确回答','',table(['问题','回答','证据/解释'],[[f'Q{i+1}',a['answer'],a['reason']]for i,a in enumerate(interpret['answers'])]),'',
 '## 12. 交付索引','',
 '`quality_summary.csv`、`quality_per_view.csv`、`paired_statistics.csv`：全部四指标、Bank/eval、两个 seed。`training_curves.csv`：单帧训练 sample 日志；`hard_training_dynamics.csv`：同 K Bank hard 导出；`sampling_train_quality.csv`：真实训练采样目标诊断。', '',
 '`coefficient_statistics.csv`、`coefficient_rows.csv`、`gradient_statistics.csv`、`grid_diagnostic_statistics.csv`、`coefficient_raw_percentiles.csv`：每 Gaussian 与冻结 root 的统计。`selected_set_overlap.csv`、`selected_sets.npz`、`rankings.npz`：所有 source rows、hard ranking 和差集。', '',
 '每场景 `models/PG_{ratio}_seed_{seed}/step_{0,250,1000,2000}.pt` 保存 coefficient、gradient EMA/max/update/hit、cached uniform tensor；对应 `.npz` 保存完整 score/order/source rows 与 hard selected set。`representability_check.json`、`training_gate.json`、`repeat_checks.json`、`correctness.json`、`audit_summary.json` 和 `artifact_manifest.json` 保存门控与 SHA。', '',
 '未接入 production；未实现任何后续 grid/split/regularization/loss/online 改动。本轮到此停止。','']
(ROOT/'PER_GAUSSIAN_IMPLICIT_SI_UPPER_BOUND_AUDIT.md').write_text('\n'.join(parts),encoding='utf-8')
audit['decision']=interpret['classification'];audit['next_step']=interpret['next_step'];audit['answers']=interpret['answers'];(ROOT/'audit_summary.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print('REPORT_WRITTEN',flush=True)
