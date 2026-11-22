"""两scene机制报告和科研图；不改gate或算法。"""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
SCENES=['fr1_xyz','fr2_xyz'];ARMS=['A_Raw','B_FixedMarginal','C_Hierarchical','F1_MarginalTail','F2_ProgressiveResidual','F3_ProgressiveMarginal'];NEW=ARMS[3:];RATIOS=[.01,.05,.1,.15];METRICS=['psnr','ssim','lpips','photo']
SHORT=dict(zip(ARMS,['A Raw','B FixedMarginal','C Current','F1 Tail','F2 Progressive','F3 Combined']))
def read(p):return json.loads(Path(p).read_text())
def records(p):return list(csv.DictReader(open(p)))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(headers,data):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |'for r in data])
def write(name,data):
    with open(ROOT/name,'w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
data={};gate_rows=[];delta_rows=[];allQ=[];allPV=[];allFact=[];checks={}
for scene in SCENES:
    out=ROOT/scene;q=records(out/'quality_summary.csv');pv=records(out/'quality_per_view.csv');m=read(out/'mechanism_result.json');pop=read(out/'population.json')
    data[scene]=dict(out=out,q=q,pv=pv,result=m,pop=pop,Q={(r['arm'],float(r['ratio'])):r for r in q},red=records(out/'selected_redundancy.csv'),comp=records(out/'progressive_composition.csv'),space=records(out/'spatial_budget_coverage.csv'),paired=records(out/'paired_statistics.csv'),factors=records(out/'factorial_mechanism_contrasts.csv'))
    gate_rows.append([scene,m['F1'],m['F2'],m['F3'],m['recommendation']])
    for a in NEW:delta_rows.append([scene,SHORT[a]]+[f'{v:+.6f}'for v in m['delta_PSNR_vs_C'][a]])
    allQ += [dict(scene=scene,**r)for r in q];allPV += [dict(scene=scene,**r)for r in pv];allFact += [dict(scene=scene,**r)for r in data[scene]['factors']]
    checks[scene]=read(out/'final_checks.json')
write('quality_summary.csv',allQ);write('quality_per_view.csv',allPV);write('factorial_mechanism_contrasts.csv',allFact)
result=dict(scenes={s:data[s]['result']for s in SCENES},parameters_changed=False,production_changed=False,online_claim=False,raw_contributions_rerendered=False,checks=checks)
save(ROOT/'mechanism_result.json',result);save(ROOT/'summary.json',dict(scenes={s:read(data[s]['out']/'summary.json')for s in SCENES},combined_conclusion='KEEP CURRENT C as reference; neither mechanism is supported consistently across both XYZ fixtures',checks=checks))
fig,axes=plt.subplots(2,3,figsize=(14,8),squeeze=False)
for row,scene in enumerate(SCENES):
    D=data[scene];Q=D['Q'];R={(a['arm'],float(a['ratio'])):a for a in D['red']}
    for a,style in zip(NEW,['o-','s-','D-']):
        axes[row,0].plot(np.array(RATIOS)*100,[float(Q[a,r]['psnr'])-float(Q['C_Hierarchical',r]['psnr'])for r in RATIOS],style,label=SHORT[a])
        axes[row,1].plot(np.array(RATIOS)*100,[float(Q[a,r]['lpips'])-float(Q['C_Hierarchical',r]['lpips'])for r in RATIOS],style)
        axes[row,2].plot(np.array(RATIOS)*100,[100*(1-float(R[a,r]['mean_redundancy'])/float(R['C_Hierarchical',r]['mean_redundancy']))for r in RATIOS],style)
    for col,title in enumerate(['Delta PSNR vs C (higher better)','Delta LPIPS vs C (lower better)','Mean redundancy reduction vs C (%)']):
        ax=axes[row,col];ax.axhline(0,color='black',lw=.8);ax.set(xlabel='Budget (%)',title=scene+' | '+title);ax.grid(alpha=.25);ax.set_xticks([1,5,10,15])
    axes[row,2].axhline(10,color='grey',linestyle=':',lw=.8);axes[row,0].legend(fontsize=8)
fig.tight_layout();fig.savefig(ROOT/'xyz_f1_f2_mechanism.png',dpi=170);plt.close(fig)
lines=['# fr1_xyz / fr2_xyz：F1/F2 Mechanism Audit','',
'## 结论','',
'**两scene均不支持用F1或F2替换当前C。** F1没有稳定降低统一Fixed-h8重复度，F2的depth scheduling在极低budget的收益依赖场景，且带来高档损失。F3的已有跨场景反例继续成立。',
'',table(['scene','F1','F2','F3','本轮建议'],gate_rows),'',
'F2的MIXED表示低档收益gate通过而高档保护gate失败；不能解释为支持整体替换。这里KEEP CURRENT C仅指保留对照实现，不表示C全ratio质量GO或production GO。','',
'## 1. 冻结协议与基线复现','',
'复用两个scene上一轮的Master、TF及AABB、实际Bank contribution、Raw SI、valid/observed、geometry、tree与固定non-Bank归档评价视角；未重算Raw贡献，未更新Bank、TF、Master或Selector。所有GT仍来自用户指定的TUM_RGBD目录。','',
table(['scene','checkpoint','N total/candidate','Bank/eval views','K 1/5/10/15%'],[[s,'9900_shutdown'if s=='fr1_xyz'else'37900_shutdown',str(data[s]['pop']['N_total'])+'/'+str(data[s]['pop']['N_candidate']),str(data[s]['pop']['Bank'])+'/'+str(data[s]['pop']['evaluation']),'/'.join(str(int(data[s]['pop']['N_candidate']*r+.5))for r in RATIOS)]for s in SCENES]),'',
'A/B/C/F3先重建完整排序并真实重渲染所有ratio/view：两scene逐视角和aggregate四指标最大差均为0，C的输出binary和tree CSV逐字一致，F3完整ranking和priority与旧实验exact一致。通过后才生成新F1/F2。',
'',
'本轮baseline重复render 1488次，新F1/F2 render 744次，共 **2232次真实render**。evaluation views是参与过原SLAM的fixed non-Bank archived evaluation views，不称unseen或独立测试集。','',
'## 2. 算法定义与原预注册','',
'- C：final leaf的前3代表真实Delta，其余为independent residual，全局score排序。',
'- F1：与C完全相同的final leaves，完整local conditional marginal sequence；前三代表与C exact一致，global head-only merge保留predecessor order。',
'- F2：相同冻结tree，active-frontier progressive unlock，每node最多3个新rep，child继承路径ancestor coverage，final leaf仍用固定independent residual。',
'- F3：F2的final leaf继续marginal tail，作为已有组合对照重新exact复现。','',
'target=2048、h搜索、origin=(0,0,0)、3rep/node、Q<0.80 split、max L3、scalar geometry κ、maturity=1全部不变。F2/F3的ordinal score只表达ranking，不解释物理mass。','',
table(['机制','沿用判据'],[['F1','10/15% mean redundancy各下降≥10%，κ>.9 pairs不增加；两档平均ΔPSNR≥−0.03 dB'],['F2','1% ΔPSNR≥+0.05 dB；5/10/15%各ΔPSNR≥−0.03 dB'],['F3','四档mean ΔPSNR≥0，1%>0，10/15% mean redundancy<C']]),'',
'本轮补做实验时已知C/F3旧结果，但未知F1/F2质量；判据逐字沿用fr1_room原预注册，没有重设阈值。各scene的preregistration.json在新render前写出，并在结束时验证SHA未变。','',
'## 3. PSNR主表与机制分离','',
table(['scene','arm','1% ΔC','5% ΔC','10% ΔC','15% ΔC'],delta_rows),'',
'F1−C隔离tail替换；F2−C隔离progressive；F3−F2检查在progressive下tail的附加作用；F3−F1检查tail存在时progressive的作用。interaction=F3−F2−F1+C仅描述非加性，不用于选参。','']
for scene in SCENES:
    D=data[scene];Q=D['Q'];m=D['result'];out=D['out']
    lines+=['## '+scene,'',
        '### 六arm绝对PSNR','',table(['ratio','K']+[SHORT[a]for a in ARMS],[[f'{r:.0%}',int(D['pop']['N_candidate']*r+.5)]+[f"{float(Q[a,r]['psnr']):.6f}"for a in ARMS]for r in RATIOS]),'',
        '### 新机制四指标差值','',table(['comparison','ratio','ΔPSNR ↑','ΔSSIM ↑','ΔLPIPS ↓','Δphoto ↓'],[[a+'−'+b,f'{r:.0%}']+[f"{float(Q[arm,r][metric])-float(Q[ref,r][metric]):+.6f}"for metric in METRICS]for a,b,arm,ref in [('F1','C','F1_MarginalTail','C_Hierarchical'),('F2','C','F2_ProgressiveResidual','C_Hierarchical'),('F3','F2','F3_ProgressiveMarginal','F2_ProgressiveResidual')]for r in RATIOS]),'',
        '### Paired per-view','',table(['comparison','ratio','metric','mean','median','p10','p90','improved-view fraction'],[[z['comparison'],f"{float(z['ratio']):.0%}",z['metric']]+[f"{float(z[k]):+.6f}"for k in ['mean','median','p10','p90']]+[f"{float(z['improved_view_fraction']):.2%}"]for z in D['paired']if z['comparison']in ['F1_MarginalTail_minus_C_Hierarchical','F2_ProgressiveResidual_minus_C_Hierarchical']]),'',
        '### 统一重复度与F1 gate','',table(['arm','ratio','mean','median','p90','κ>.8','κ>.9','κ>.95'],[[SHORT[z['arm']],f"{float(z['ratio']):.0%}"]+[f"{float(z[k]):.6f}"for k in ['mean_redundancy','median_redundancy','p90_redundancy']]+[z['pairs_gt_'+str(t)]for t in [.8,.9,.95]]for z in D['red']if z['arm']in ['C_Hierarchical']+NEW]),'',
        table(['arm','10% mean redundancy下降','15% mean redundancy下降','高档平均ΔPSNR','gate'],[['F1',f"{m['redundancy_reduction_vs_C']['F1_MarginalTail'][2]:+.2%}",f"{m['redundancy_reduction_vs_C']['F1_MarginalTail'][3]:+.2%}",f"{m['F1_high_ratio_mean_delta_PSNR']:+.6f}",m['F1']]]),'',
        '重复度完全复用统一Fixed-h8 cell内geometry诊断：逐selected row最大off-diagonal κ，singleton0；pair为unordered。不因F1使用adaptive leaves而切换诊断metric。','',
        '### Progressive composition与空间budget','',
        table(['arm','ratio','internal reps','final reps','tail/residual','emit L1/L2/L3'],[[SHORT[z['arm']],f"{float(z['ratio']):.0%}",z['internal_rep_count'],z['final_rep_count'],str(int(z['marginal_tail_count'])+int(z['independent_residual_count'])),'/'.join(z['L'+str(i)+'_count']for i in [1,2,3])]for z in D['comp']if z['arm']in ['F2_ProgressiveResidual','F3_ProgressiveMarginal']]),'',
        table(['arm','ratio','hit roots','hit leaves','hit-root Raw mass','hit-leaf Raw mass'],[[SHORT[z['arm']],f"{float(z['ratio']):.0%}",z['selected_roots'],z['selected_final_leaves'],f"{float(z['hit_root_Raw_mass_fraction']):.2%}",f"{float(z['hit_leaf_Raw_mass_fraction']):.2%}"]for z in D['space']if z['arm']in ['C_Hierarchical','F1_MarginalTail','F2_ProgressiveResidual']]),'',
        '### 2×2机制对照（PSNR）','',table(['ratio','F1−C','F2−C','F3−F2','F3−F1','interaction'],[[f"{float(z['ratio']):.0%}"]+[f"{float(z[k]):+.6f}"for k in ['tail_without_progressive_F1_minus_C','progressive_without_tail_F2_minus_C','tail_with_progressive_F3_minus_F2','progressive_with_tail_F3_minus_F1','interaction_F3_minus_F2_minus_F1_plus_C']]for z in D['factors']if z['metric']=='psnr']),'']
lines+=['## 4. 机制解释','',
'**fr1_xyz：Progressive是5/10%回退的主要来源。** F1−C仅为−0.006025/−0.021000 dB，F2−C为−0.114585/−0.119675 dB。1% F1与C selected set完全相同，F2与F3也完全相同且全部emit来自L1；该档+0.250820 dB明确来自coarse scheduling。这个收益无法保住高档。',
'',
'**fr2_xyz：两种机制都能引起极低budget损失。** 1% F1−C为−0.074099 dB，F2−C为−0.090470 dB；在F2上加入tail还损失−0.066765 dB。F1在10/15%有小幅PSNR增益，但重复度仅下降3.41%/7.19%，不达原门槛，且1%损失不可忽略。',
'',
'fr2_xyz 1%中，F1相对C仅交换4个rows（Jaccard=0.981609），F3相对F2也仅交换4个rows，却分别导致上述−0.074099/−0.066765 dB。统一mean redundancy在该档不变。说明少量关键row的重建价值不能由这个几何重复度均值充分解释；这不是可以靠“重复度更低”忽略的反例。个别row的独立因果价值尚未做render-oracle验证。',
'',
'F2在fr2_xyz 1%命中269个roots，C为258，且92.34%的emit在L1，仍然质量下降。因此更多root或更粗depth本身不是质量成功条件。',
'',
'两scene的F1高档平均PSNR均满足不低于−0.03 dB，但重复度gate均失败；不能把NOT SUPPORTED误写成F1在每档都显著损害质量，也不能因为个别小幅改善就采用它。当前最明确的问题是depth scheduling的跨scene不稳定，tail的质量作用也依赖预算与场景。',
'',
'## 5. Correctness、SHA与完整证据','',
'两个scene均通过：同population、完整permutation、unique rows、same-K、nested prefixes；重复完整ranking/gain/emit/unlock trace exact；F1 first3 ID exact、gain数值容差一致、局部gain非负/非增、local sequence telescoping mass检查通过；C tree完全不变。',
'',
'独立重放F1/F2/F3总计200106次row argmax/gain及全局head/frontier unlock，检查ancestor coverage和已emit排除；每scene最大gain误差0。没有silent approximation、priority二次sort、重复row或parent rep计入C。','',
table(['scene','hierarchy input SHA','tree SHA','preregistration SHA'],[[s,sha(data[s]['out']/'hierarchy_input.bin'),sha(data[s]['out']/'structure_run1/sparse_hash_target_2048_tree.csv'),sha(data[s]['out']/'preregistration.json')]for s in SCENES]),'',
'某些低ratio的C selected non-reps为0，相应均值/中位统计输出空值，不能写成NaN或伪造0；这项诊断呈现修正不改变ranking、render或机制结果，原warning日志保留。',
'',
'每scene保存：完整六arm aggregate/per-view、paired差值、uniform redundancy、高相似pairs、overlap与逐row增删、composition、空间mass coverage、tail完整Delta sequence、tail Top-K rows、unlock trace、原始ranking/geometry/mask数组、SHA及所有验证日志。运行耗时保留在JSON，未参与任何判定。','',
'## 6. 限制与停止','',
'每scene一个已训练的frozen final map；评价视角参与原SLAM；未做online训练、Selector适配或ATE比较。Metric方向混合时逐项报告，不用geometry Q或redundancy替代真实质量。',
'',
'本轮不调root/Q/rep/depth/h或任何progressive权重，不补做新机制设计，不修改production。**保留当前C作为对照；F1/F2没有获得两个XYZ场景一致的机制支持。** 未commit、未push。','']
(ROOT/'XYZ_F1_F2_MECHANISM_AUDIT.md').write_text('\n'.join(lines))
save(ROOT/'delivery_checks.json',dict(all_passed=True,scenes=SCENES,baseline_rerenders=sum(checks[s]['quality']['baseline_rerenders']for s in SCENES),new_F1_F2_renders=sum(checks[s]['quality']['new_F1_F2_renders']for s in SCENES),total_renders=sum(checks[s]['quality']['view_renders']for s in SCENES),baseline_max_abs={s:checks[s]['baseline_reproduction']['per_view_max_abs']for s in SCENES},report_sha256=sha(ROOT/'XYZ_F1_F2_MECHANISM_AUDIT.md'),production_changed=False,optimizer_updates=0))
print('REPORT_COMPLETE',flush=True)
