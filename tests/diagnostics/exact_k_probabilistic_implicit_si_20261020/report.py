"""全部固定结果经审阅后生成中文报告；不使用质量结果反向选参。"""
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parent
def read(n):return json.loads((ROOT/n).read_text())
def load(n):return list(csv.DictReader(open(ROOT/n)))
def table(headers,rows):
    def c(v):return str(v).replace('|','\\|')
    return '\n'.join(['| '+' | '.join(map(c,headers))+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(c,z))+' |'for z in rows])
def f(v,n=6):return f'{float(v):.{n}f}'
def signed(v,n=6):return f'{float(v):+.{n}f}'
reg=read('preregistration.json');audit=read('audit_summary.json');interp=read('interpretation.json');correct=read('correctness.json');independent=read('independent_checks.json');assert correct['all_passed']and independent['all_passed']
q=load('quality_summary.csv');paired=load('paired_statistics.csv');deltas=load('quality_deltas.csv');coverage=load('gradient_coverage.csv');sampling=load('sampling_statistics.csv');grads=load('gradient_statistics.csv');overlap=load('selected_set_overlap.csv');promotion=load('promotion_comparison.csv');objective=load('stochastic_objective_dynamics.csv');seedvar=load('eval_seed_consistency.csv');runtime=read('runtime.json')['models'];toy=read('pl_probability_tests.json');gt=read('reinforce_gradient_tests.json')
def get(scene,ratio,arm,seed,split):return next(z for z in q if z['scene']==scene and float(z['ratio'])==ratio and z['arm']==arm and int(z['seed'])==seed and z['split']==split)
def match(rows,scene,ratio,seed):return next(z for z in rows if z['scene']==scene and float(z['ratio'])==ratio and int(z['seed'])==seed)
parts=['# Exact-K Probabilistic Implicit SI Audit','','## 结论','',f"**{interp['classification']}**。",'']
for p in interp['conclusions']:parts += [p,'']
parts += ['研究决策：**'+interp['next_step']+'**。','','## 1. 冻结与实验范围','',
 '24个独立模型：3 scene ×4 ratio ×2 seed；每模型从a=0训练2000步，共48000次正式更新。完整 valid AND observed population 不筛选。score保持 `log(RawSI+1e-12)+a`，a/score/PL计算均FP64。temperature=1，Adam lr=.001、betas=(.9,.999)、eps=1e-8、weight_decay=0。', '',
 '冻结 Gaussian xyz/scale/rotation/opacity/SH、Master、TF/AABB、Raw/contribution/Bank、candidate mask、cameras/GT、renderer和四指标实现。没有grid、split、representative/residual、coverage、温度/LR/baseline搜索、正则、entropy、C distillation、multi-sample、learned critic或online。旧PG-ST路径完整保留，新模式为 `pl_reinforce_exact_k`，仅在独立diagnostic中运行。','']
rows=[]
for s in reg['scenes']:
    info=json.loads((ROOT/s/'input_manifest.json').read_text());N=read('input_checks.json')['scenes'][s]['N'];rows.append([s,N,len(info['bank_frames']),len(info['evaluation_frames']),'/'.join(str(int(r*N+.5))for r in reg['ratios']),Path(info['master']).parts[-5]])
parts += [table(['scene','N candidate','Bank','non-Bank eval','K @1/5/10/15%','checkpoint'],rows),'',
 '数据来源：`/home/prmi/zwj/per-gaussian-implicit-si-20261020`。同scene/ratio/seed Bank camera顺序与旧PG-ST一致，均用std::mt19937(seed)重排完整epochs。evaluation严格沿用既有fixed non-Bank archived views；它们参与过原SLAM，但未用于本轮optimization。','',
 'A/B/C/F完整ranking、selected sets、tree重新构建；A/B/C/F的aggregate及逐view四指标exact复现。旧PG-ST的三scene×四ratio×两seed，在Bank与non-Bank均真实重渲染，逐view四指标差值全部为0。正式训练在上述gates通过后启动。','',
 '## 2. 新selection路径','',
 '每步新生成FP64 `u~Uniform(0,1)`、`g=-log(-log(u))`，按 `score+g` 取有序Top-K，不放回。该顺序用于PL log probability；实际renderer输入将这K个source rows升序排序，以复用原compact-render行为。没有refill/truncate、expected-K、Bernoulli、offset、opacity ST或cached noise。','',
 '`shift=max(z).detach(); w=exp(z-shift); previous=[0,cumsum(w[order])[:-1]]; denominator=sum(w)-previous; logP=sum(z[order]-shift-log(denominator))`。时间/存储O(N+K)，没有N×K矩阵。所有weights、denominator、logP、gradient逐步检查finite/positive；任何underflow/overflow或非正denominator直接停止。未使用safe clamp。','',
 '选择梯度为 `L_selection = detach(photo - baseline_before) * logP`，执行gradient descent。rendered photo全detach，Gaussian/TF无autograd更新；a只通过logP反向。所有未选Gaussian的weight仍在denominator内，因而有selection-level gradient。','',
 '每Bank frame的baseline以该frame下Raw hard Top-K photo初始化。固定beta=.99；当前advantage使用旧baseline，在backward/Adam之后才做EMA更新。baseline不参与梯度，不读取C、Fixed、PG-ST或evaluation结果。','',
 'RNG策略：完成FrozenScene初始化后重新`torch::manual_seed(seed)`；每步仅进行一次新的CUDA FP64 rand_like(N)，没有其他CUDA随机draw。保存step0/250/1000/2000的CUDA RNG state；全部48000步保留noise signature，并检查每模型step之间不同。保存旧signature仅作重复性检查，不参与采样。','',
 '## 3. Toy gates：概率、梯度、符号','',
 table(['检查','结果'],[['N6 K2 概率归一',f(toy['sum_probability'],16)],['Gumbel采样数',toy['samples']],['频率最大绝对误差',f(toy['max_frequency_abs_error'],8)],['频率最大标准误倍数',f(toy['max_frequency_error_SE'],4)],['MC/exact gradient cosine',f(gt['MC_gradient_cosine'],9)],['独立有限差分最大误差',f"{gt['finite_difference_max_abs_error']:.3e}"],['exact gradient descent前/后J',f(gt['baseline'],10)+' → '+f(gt['after_exact_gradient_descent'],10)],['N8 K3 selected/unselected gradient','全部finite且非零'],['gradient shift invariance',f"abs(sum grad)={gt['gradient_sum_abs']:.3e}"]]),'',
 'Toy使用20万次sample，预注册统计检查为频率/gradient误差在6 Monte Carlo standard errors内，manual probability/normalization误差<1e-12，finite difference误差<1e-8。这些是实现正确性检查，不是质量GO门槛。生产训练与toy共享同一`pl_selection.h`；MC抽样频率另与显式sequential PL公式比较。', '',
 '## 4. Exact K、fresh noise与梯度覆盖','',
 f"独立重读48000条日志验证：每一步sampled_count==K；每模型noise signatures无重复；baseline更新顺序exact；所有a gradient finite。最小每步gradient coverage={independent['minimum_per_step_gradient_coverage']:.2%}。",'']
rows=[]
for s in reg['scenes']:
    for r in reg['ratios']:
        for seed in reg['seeds']:
            z=match(sampling,s,r,seed);c=match(coverage,s,r,seed);rows.append([s,f'{r:.0%}',seed,z['K'],z['min'],z['max'],f(z['mean'],1),f"{float(c['per_step_nonzero_fraction_min']):.2%}",f"{float(c['ever_nonzero_fraction']):.2%}",f"{float(c['legacy_ever_nonzero_fraction']):.2%}"])
parts += [table(['scene','ratio','seed','K','sample min','sample max','sample mean','PL per-step min grad coverage','PL ever grad','old ST ever grad'],rows),'','![gradient coverage](gradient_coverage.png)','',
 '旧ST统计为photo经selected-only opacity ST形成的gradient；新PL统计为selection-level score-function gradient。两者不是同一种导数，表中比较的是“每个a有没有学习信号”，不是梯度方向质量。full coverage本身不能推出有效ranking。', '',
 '## 5. Primary quality：Bank hard same-K与non-Bank hard same-K','',
 '以下全部为deterministic hard Top-K，无Gumbel、无stochastic mask；PSNR/SSIM越高越好，LPIPS/photo越低越好。所有模型统一使用最终step2000，无early stop、不按eval挑seed。']
for s in reg['scenes']:
    for split,label in [('train','Bank'),('eval','non-Bank evaluation')]:
        parts += ['',f'### {s} / {label}']
        for metric in ['psnr','ssim','lpips','photo']:
            rows=[]
            for r in reg['ratios']:
                for seed in reg['seeds']:
                    vals=[get(s,r,a,-1,split)[metric]for a in['Raw','Fixed','C']]+[get(s,r,a,seed,split)[metric]for a in['PG-ST','PG-PL']];rows.append([f'{r:.0%}',seed]+[f(v)for v in vals]+[signed(float(vals[-1])-float(vals[-2]))])
            parts += ['',f'**{metric.upper()}**','',table(['ratio','seed','Raw','Fixed h8','C','PG-ST','PG-PL','PL-ST'],rows)]
parts += ['','## 6. 跨scene、seed与paired-view判断','',table(['reference','split','PSNR↑ /24','SSIM↑ /24','LPIPS↓ /24','photo↓ /24'],[[ref,split,z['PSNR_positive'],z['SSIM_positive'],z['LPIPS_improved'],z['photo_improved']]for ref,by in audit['counts'].items()for split,z in by.items()]),'']
for p in interp['quality_analysis']:parts += [p,'']
parts += ['![evaluation PSNR deltas](quality_deltas.png)','','### Paired non-Bank PSNR','',
 '完整四指标paired mean/median/p10/p90/improved-view fraction保存于`paired_statistics.csv`。下表保留每个scene/ratio/seed及四个reference的PSNR，不以aggregate代替view证据。','']
rows=[]
for z in paired:
    if z['split']=='eval'and z['metric']=='psnr':rows.append([z['scene'],f"{float(z['ratio']):.0%}",z['seed'],z['comparison'],signed(z['mean']),signed(z['median']),signed(z['p10']),signed(z['p90']),f"{float(z['improved_view_fraction']):.2%}"])
parts += [table(['scene','ratio','seed','comparison','mean','median','p10','p90','improved views'],rows),'','## 7. Stochastic objective与hard-ranking dynamics','',
 '每step记录sample loss、baseline-before/after、advantage、logP、surrogate、sample order首末source row/Raw percentile摘要。每100step及250step对全部Bank views重渲染hard Top-K；`hard_training_dynamics.csv`与`hard_training_per_view.csv`保存完整结果。中途观察未反馈给训练参数或停止时机。','',
 '下表比较每个Bank frame最早5次与最后5次stochastic sample loss，再在frames间等权平均。每次sample均exact K；这是一条含采样/视角噪声的描述曲线，不是paired same-subset或已证明无偏的收敛估计。hard列比较step2000与step0。','']
rows=[]
for z in objective:rows.append([z['scene'],f"{float(z['ratio']):.0%}",z['seed'],signed(z['stochastic_photo_delta']),signed(z['hard_photo_delta']),signed(z['hard_psnr_delta']),f(z['advantage_std']),f(z['log_prob_std'],2),f"{float(z['last100_hard_changed_fraction']):.2%}"])
parts += [table(['scene','ratio','seed','stochastic Δphoto','hard Δphoto','hard ΔPSNR','advantage std','logP std','last100 changed'],rows),'','![hard training curves](hard_training_curves.png)','']
for p in interp['training_analysis']:parts += [p,'']
parts += ['## 8. Variance、score concentration与numerical sanity','',
 '没有gradient clipping、a中心化或temperature调整。PL对公共score shift不变；因此每步记录abs(sum gradient)、sum abs gradient及两者归一残差。Adam是逐坐标缩放，其update不必零和，不能把coefficient公共漂移直接当作错误。', '',
 f"正式最大gradient shift residual={correct['max_gradient_shift_residual']:.3e}。所有denominator>0，weights严格正且finite，没有underflow、overflow、NaN/Inf或silent clamp。",'']
rows=[]
for s in reg['scenes']:
    for r in reg['ratios']:
        for seed in reg['seeds']:
            z=match(sampling,s,r,seed);a=next(a for a in grads if a['scene']==s and float(a['ratio'])==r and int(a['seed'])==seed and a['quantity']=='gradient_norm');u=next(a for a in grads if a['scene']==s and float(a['ratio'])==r and int(a['seed'])==seed and a['quantity']=='Adam_update_norm')
            rows.append([s,f'{r:.0%}',seed,f(z['score_spread_initial'],3)+' → '+f(z['score_spread_final'],3),f(z['ESS_initial'],1)+' → '+f(z['ESS_final'],1),f(z['max_weight_fraction']),f(a['mean']),f(a['std']),f(u['mean']),f"{float(z['max_shift_gradient_residual']):.2e}"])
parts += [table(['scene','ratio','seed','score/z spread','weight ESS','max weight share','grad norm mean','grad norm std','Adam norm mean','shift residual max'],rows),'',
 'ESS定义为`1/sum((w/sum(w))²)`，仅描述第一抽selection concentration，不能等同于K-subset entropy或总体inclusion ESS。完整advantage/logP/grad norm/max/update magnitude分布在`gradient_statistics.csv`，每step magnitude分位在`training_curves.csv`。','']
rows=[]
for s in reg['scenes']:
    for r in reg['ratios']:
        z=[next(z for z in seedvar if z['scene']==s and float(z['ratio'])==r and z['arm']==a)for a in['PG-ST','PG-PL']]
        rows.append([s,f'{r:.0%}',f(z[0]['PSNR_seed_abs_difference']),f(z[1]['PSNR_seed_abs_difference']),f(z[0]['photo_seed_abs_difference']),f(z[1]['photo_seed_abs_difference'])])
parts += [table(['scene','ratio','ST |Δseed PSNR|','PL |Δseed PSNR|','ST |Δseed photo|','PL |Δseed photo|'],rows),'']
for p in interp['variance_analysis']:parts += [p,'']
parts += ['## 9. Selected set与low-Raw-rank promotion','',
 '新增points定义为对应hard Top-K减Raw Top-K。Raw percentile为`(1-based rank-1)/(N-1)`，数值越大代表原Raw ranking越靠后。所有新增rows完整保存在`raw_rank_promotion_rows.csv`，每comparison added/removed source rows保存在`selected_sets.npz`。','']
rows=[]
for z in promotion:rows.append([z['scene'],f"{float(z['ratio']):.0%}",z['seed'],z['PL_added'],z['ST_added'],f(z['PL_Raw_percentile_mean']),f(z['ST_Raw_percentile_mean']),f(z['PL_Raw_percentile_p90']),f(z['ST_Raw_percentile_p90']),z['PL_below2K'],z['ST_below2K']])
parts += [table(['scene','ratio','seed','PL added','ST added','PL Raw pct mean','ST Raw pct mean','PL Raw pct p90','ST Raw pct p90','PL Raw rank>2K','ST Raw rank>2K'],rows),'','![Raw rank promotion](raw_rank_promotion.png)','']
for p in interp['promotion_analysis']:parts += [p,'']
rows=[]
for s in reg['scenes']:
    for r in reg['ratios']:
        for seed in reg['seeds']:
            z=[next(z for z in overlap if z['scene']==s and float(z['ratio'])==r and int(z['seed'])==seed and z['reference']==a)for a in['PG-ST','Raw','Fixed','C']]
            rows.append([s,f'{r:.0%}',seed]+[f"{float(a['Jaccard']):.6f} / {a['added']}"for a in z])
parts += [table(['scene','ratio','seed','vs ST Jaccard/added','vs Raw Jaccard/added','vs Fixed Jaccard/added','vs C Jaccard/added'],rows),'',
 '## 10. Runtime：研究诊断','',
 'CUDA synchronize包围采样、logP、render、backward+Adam。core排除全Bank hard diagnostics与后续统计输出；total包括当前step的记录/间隔hard diagnostics。数据只描述此独立审计实现，不作production speedup claim。logP没有N×K中间量。','']
rows=[]
for s in reg['scenes']:
    for r in reg['ratios']:
        for seed in reg['seeds']:
            rt=runtime[f'{s}_PL_{int(r*100)}_seed_{seed}'];rows.append([s,f'{r:.0%}',seed]+[f(rt[k]['mean'],3)for k in['sampling_ms','log_prob_ms','render_ms','backward_Adam_ms','core_ms','total_step_ms']]+[f(rt['process_seconds'],2)])
parts += [table(['scene','ratio','seed','sample ms','logP ms','render ms','backward+Adam ms','core ms','total step ms','process s'],rows),'',
 '## 11. Correctness与解释边界','',
 'A：a=0完整Raw ranking exact；B：48000步exact K；C：fresh noise每步不同；D/E：toy概率归一与频率匹配；F：score-function梯度方向与exact enumeration/finite difference匹配；G：selected/unselected全population gradient；H：Gaussian/TF/Master无gradient且tensor exact不变；I：所有最终export为deterministic hard同K；J：训练不读C/Fixed/evaluation结果。详见`correctness.json`。','',
 '源码训练器仅读取source rows、Raw base、Bank frame列表、冻结map/TF/camera文件。camera archive虽含完整冻结views，但photo函数在读取GT前强制frame属于Bank。旧PG-ST与强基线文件仅供门控、最终评价和posthoc统计，不参与优化。','',
 '每模型step0/250/1000/2000保存a、gradient EMA/max/hit、累计update、per-frame baseline与CUDA RNG state；所有hard前缀在最终质量渲染前固定并计算SHA。独立Python tuple sort复核96个snapshot的完整ranking与same-K集合；基线/旧PG-ST输入文件及模型/TF/Bank/Raw SHA终验全部不变。','',
 '5-step重复native smoke的系数、gradient统计、baseline、RNG state均bitwise exact。正式两个seed全部保留；这不等价于对跨GPU/软件版本determinism的承诺。', '',
 f"Native hard-dynamics与标准batch metric的photo逐view exact；PSNR最大差{correct['hard_training_diagnostic_vs_standard_metrics_max_abs']['psnr']:.3e} dB，来自FP32单图/批处理reduction，使用1e-5纯数值一致性容差。主quality表全部用原standard metric实现；该容差不是质量GO门槛。",'',
 'PL ordered-sample REINFORCE正确且对stochastic rendering objective无偏，不保证最终deterministic hard Top-K必然最优。full gradient coverage不等于有效探索，不等于低variance；两seed与一个固定预算不能独立证明剩余缺口全部来自某个机制。未用GT质量调candidate、score、temperature、LR、baseline或steps。', '',
 '## 12. 对十项问题的明确回答','',table(['问题','回答','证据/解释'],[[f'Q{i+1}',z['answer'],z['reason']]for i,z in enumerate(interp['answers'])]),'',
 '## 13. 交付与停止','',
 '`quality_summary.csv`、`quality_per_view.csv`、`paired_statistics.csv`：Bank/eval、全部四指标与所有seed。`training_curves.csv`：48000 step；`hard_training_dynamics.csv`/`hard_training_per_view.csv`：每100step和250step全Bank hard导出。', '',
 '`pl_probability_tests.json`、`reinforce_gradient_tests.json`、`gradient_coverage.csv`、`gradient_statistics.csv`、`sampling_statistics.csv`、`runtime.json`：正确性、覆盖、数值与开销。`selected_set_overlap.csv`、`raw_rank_promotion.csv`及完整row明细：搜索范围。', '',
 '`rankings.npz`、`selected_sets.npz`、`coefficients.pt`及每模型`.pt/.npz`：冻结导出、完整tensor与RNG state。`correctness.json`、`independent_checks.json`、`audit_summary.json`、`artifact_manifest.json`：证据链与SHA。', '',
 '没有修改Current C或production Selector，没有添加后续训练修复或grid/split。实验完成后停止；没有stage、commit、push。','']
(ROOT/'EXACT_K_PROBABILISTIC_IMPLICIT_SI_AUDIT.md').write_text('\n'.join(parts),encoding='utf-8')
audit['decision']=interp['classification'];audit['next_step']=interp['next_step'];audit['answers']=interp['answers'];(ROOT/'audit_summary.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n');print('REPORT_WRITTEN',flush=True)
