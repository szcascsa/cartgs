"""三scene尺度因果比较报告；lambda只有32，所有gate在render前固定。"""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def rows(p):return list(csv.DictReader(open(p)))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(headers,data):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |'for r in data])
reg=read(ROOT/'preregistration.json');S=reg['scenes'];R=reg['ratios'];A=reg['arms'];res=read(ROOT/'mechanism_result.json');structure=read(ROOT/'structure_statistics.json');checks=read(ROOT/'final_checks.json');q=rows(ROOT/'quality_summary.csv');delta=rows(ROOT/'quality_deltas.csv');paired=rows(ROOT/'paired_statistics.csv');Q={(a['scene'],a['arm'],float(a['ratio'])):a for a in q};metrics=['psnr','ssim','lpips','photo']
scale_rows=[];physical=[];rootdist=[];alllevels=[];composition=[];red=[];overlap=[]
for s in S:
    old=structure[s]['old'];new=structure[s]['new'];scale_rows.append([s,f"{old['h_L1']:.12g}",f"{old['h_over_d50']:.6f}",f"{new['h_L1']:.12g}",f"{new['h_over_d50']:.6f}",old['occupied_roots'],new['occupied_roots']])
    physical.append([s,new['N_candidate'],f"{new['d50']:.12g}",f"{new['h_L1']:.12g}",f"{new['h_L2']:.12g}",f"{new['h_L3']:.12g}"])
    for kind in ['old','new']:
        v=structure[s][kind];d=v['root_occupancy'];rootdist.append([s,kind]+[f'{d[k]:.3f}'for k in ['min','mean','median','p90','p95','p99','max']]+[f"{v['largest_root_candidate_fraction']:.2%}",f"{v['largest_N_root_Raw_fraction']:.2%}",f"{v['largest_root_Raw_fraction']:.2%}"])
    alllevels += [dict(scene=s,**z)for z in rows(ROOT/s/'level_statistics.csv')];composition += [dict(scene=s,**z)for z in rows(ROOT/s/'progressive_composition.csv')];red += [dict(scene=s,**z)for z in rows(ROOT/s/'selected_redundancy.csv')];overlap += [dict(scene=s,**z)for z in rows(ROOT/s/'ranking_overlap.csv')]
def write(name,records):
    with open(ROOT/name,'w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
write('level_statistics.csv',alllevels);write('progressive_composition.csv',composition);write('selected_redundancy.csv',red);write('ranking_overlap.csv',overlap)
fig,axes=plt.subplots(1,2,figsize=(10,4));idx=np.arange(len(S));axes[0].bar(idx-.18,[structure[s]['old']['h_over_d50']for s in S],.36,label='target-2048');axes[0].bar(idx+.18,[structure[s]['new']['h_over_d50']for s in S],.36,label='d50x32');axes[0].set(xticks=idx,xticklabels=S,ylabel='L1 / d50',title='Gaussian-normalized physical root scale');axes[0].legend();axes[1].bar(idx-.18,[structure[s]['old']['occupied_roots']for s in S],.36,label='target-2048');axes[1].bar(idx+.18,[structure[s]['new']['occupied_roots']for s in S],.36,label='d50x32');axes[1].set(xticks=idx,xticklabels=S,ylabel='Occupied L1 roots',title='No occupied-count target in d50x32');axes[1].grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(ROOT/'root_scale_comparison.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(14,4))
for ax,s in zip(axes,S):
    ax.plot(np.array(R)*100,res['old_progressive_delta_PSNR'][s],'o-',label='P0 - C0');ax.plot(np.array(R)*100,res['new_progressive_delta_PSNR'][s],'s-',label='P32 - C32');ax.axhline(0,color='black',lw=.8);ax.axhline(-.05,color='grey',linestyle=':',lw=.8);ax.set(title=s,xlabel='Budget (%)',ylabel='Progressive delta PSNR (dB)');ax.grid(alpha=.25);ax.set_xticks([1,5,10,15]);ax.legend()
fig.tight_layout();fig.savefig(ROOT/'progressive_effect_comparison.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(14,4))
for ax,s in zip(axes,S):
    for label in ['C32_minus_C0','P32_minus_P0']:
        d=[next(float(z['delta_psnr'])for z in delta if z['scene']==s and z['comparison']==label and float(z['ratio'])==r)for r in R];ax.plot(np.array(R)*100,d,'o-',label=label.replace('_minus_',' - '))
    ax.axhline(0,color='black',lw=.8);ax.set(title=s,xlabel='Budget (%)',ylabel='Grid-scale effect on PSNR (dB)');ax.grid(alpha=.25);ax.set_xticks([1,5,10,15]);ax.legend()
fig.tight_layout();fig.savefig(ROOT/'grid_effect_comparison.png',dpi=170);plt.close(fig)
baseline_render=sum(checks['scenes'][s]['quality']['baseline_rerenders']for s in S);new_render=sum(checks['scenes'][s]['quality']['new_C32_P32_renders']for s in S)
question_labels={'Q1':'target-2048是否存在显著跨scene h/d50漂移','Q2':'C32质量是否保持（PSNR±.02等价，四指标另列）','Q3':'fr2_xyz 1%是否恢复至少+.05 dB','Q4':'fr1_xyz 5/10%均值是否恢复至少+.05 dB','Q5':'fr1_xyz 1%优势是否至少保留+.10 dB','Q6':'跨三scene Progressive 是否更稳定','Q7':'最终机制判断','Q8':'下一步'}
lines=['# Gaussian-Scale-Normalized Sparse Hierarchy Audit','',
'## 结论','',
f"**{res['Q7']}**。Primary A/B/C/D gate结果：{res['gates']}。Strong support：**{'YES'if res['strong_supported']else'NO'}**。唯一测试值为lambda=32，没有sweep，没有生产接入。",
'尺度漂移确实存在，但固定32并未稳定救回Progressive：fr2_xyz 1%和fr1_xyz 5/10%的相对效果改善，伴随C32 baseline自身变差；fr1_xyz 1%的旧优势则由+0.250820变为−0.165817 dB。当前应保留target-2048。',
'',table(['问题','回答'],[[question_labels[k],res[k]]for k in question_labels]),'',
'## 1. 假设、变量与冻结输入','',
'本轮检验occupied-count归一化是否让不同scene的level具有不同Gaussian-normalized物理意义，并进一步导致Progressive不稳定。唯一变量为L1 root size：旧target occupied≈2048，对比固定h=32*d50。',
'',
'三scene复用各自既有final checkpoint、Master拓扑/xyz/activated scale/opacity/SH、TF/AABB、实际Bank contribution、Raw SI、valid/observed、固定non-Bank归档evaluation cameras与GT。没有新Raw contribution导出、训练、Bank更新、GT选h或选view。',
'',
table(['scene','checkpoint','candidate N','Bank / evaluation','K 1/5/10/15%'],[[s,{'fr1_room':'16900_shutdown','fr1_xyz':'9900_shutdown','fr2_xyz':'37900_shutdown'}[s],structure[s]['new']['N_candidate'],str(read(ROOT/s/'population.json').get('Bank',read(ROOT/s/'population.json').get('bank_count')))+' / '+str(len(read(ROOT/s/'input_manifest.json')['evaluation_frames'])),'/'.join(str(int(structure[s]['new']['N_candidate']*r+.5))for r in R)]for s in S]),'',
'评价视角参与过原SLAM，称为fixed non-Bank archived evaluation views，不能称unseen或独立novel-view测试集。XYZ归档Bank的scene文本标签过期，始终通过实际checkpoint、keyframe ID与指定TUM_RGBD图像路径确认dataset；沿用上一轮同一归档Bank。','',
'## 2. Why target-2048 changes physical hierarchy scale','',
'旧backend用candidate geometry寻找使occupied root数量接近2048的h。occupied count约束与Gaussian的全局典型直径不是同一约束；不同scene的空间跨度、空间分布与Gaussian尺度分布会产生不同h/d50。实际结果如下，不能只说“更normalized”。','',
table(['scene','old h (m)','old h/d50','new h (m)','new h/d50','old roots','new roots'],scale_rows),'',
f"旧h/d50最大/最小比为 **{max(structure[s]['old']['h_over_d50']for s in S)/min(structure[s]['old']['h_over_d50']for s in S):.4f}倍**；新backend三scene均为32。occupied roots允许自然变化，不补齐、合并、split以凑2048。",'',
'对当前candidate valid AND observed，仅使用activated positive scale计算 r=exp(mean(log(sx,sy,sz)))=cbrt(sx*sy*sz)，d50=median(2*r)。固定L1=32*d50、L2=16*d50、L3=8*d50；32不是质量最优值，只把最细L3对齐既有Fixed h8 reference。',
'',table(['scene','candidate N','d50','h L1','h L2','h L3'],physical),'',
'新factory不接收Raw SI、Q或质量输入，不调用bracket/log search/root-count objective。origin固定0，int64 triplet floor(x/h)，std::map仅保存occupied keys并按字典序遍历。没有根据key span分配dense数组。','',
'## 3. Arms与数学边界','',
table(['Arm','定义'],[['A','Raw SI直接降序'],['B','原Fixed h8 geometry-only完整local marginal greedy + predecessor-preserving merge'],['C0','旧target-2048 tree，当前C score'],['P0','旧tree，原F2 progressive + independent residual'],['C32','新32*d50 tree，当前C score完全不变'],['P32','新tree，原F2 progressive + independent residual完全不变']]),'',
'所有其它数学保持：scalar κ=exp(−d²/(2σ²+1e−12))、σ=(ri+rj)/2、κii=1；最多3个代表，gain=Σ Ij max(κij−coverj,0)；Q=Σ Ij coverj/(Σ Ij+eps)；Q<.80且level<3才split。parent provisional reps不进入C输出，leaf代表用选中时Delta，non-rep用I*(1-cover)。',
'',
'P0/P32按active-frontier heap实际emit sequence排序；child继承路径ancestor coverage，已emit row排除，完成最多3个新rep后unlock children；final leaf为固定independent residual，不做marginal tail。Ordinal N-rank只表示order，不是物理importance mass。本轮没有新增F1/F3运行或render。','',
'## 4. Baseline reproduction gate','',
table(['scene','A/B/C0/P0 complete ranking','C0 binary/tree','per-view最大差 PSNR/SSIM/LPIPS/photo','baseline renders'],[[s,'exact','exact','/'.join(f"{checks['scenes'][s]['baseline']['per_view_max_abs'][m]:.3g}"for m in metrics),checks['scenes'][s]['quality']['baseline_rerenders']]for s in S]),'',
f"三scene全部baseline gate通过后才生成C32/P32。baseline重复 **{baseline_render}** 次真实render，新C32/P32 **{new_render}** 次，共 **{baseline_render+new_render}** 次。全部arm使用各scene完全相同candidate、K、renderer、TF consumer和metric。",'',
'## 5. 结构统计（不作为quality判据）','',
table(['scene','backend','min','mean','median','p90','p95','p99','max','largest N fraction','该root Raw mass','max-root Raw mass'],rootdist),'',
table(['scene','backend','final leaves','final reps','L3 unresolved','unresolved Raw mass','leaf Raw-weighted Q'],[[s,k,structure[s][k]['total_final_leaves'],structure[s][k]['total_final_representatives'],structure[s][k]['L3_unresolved_count'],f"{structure[s][k]['L3_unresolved_Raw_fraction']:.2%}",f"{structure[s][k]['global_leaf_weighted_Q']:.2%}"]for s in S for k in ['old','new']]),'',
table(['scene','backend','level','nodes','splits','final leaves','final reps','all provisional reps','occupancy mean/median/p90/max','Q mean/median/p10/p90','weighted Q'],[[z['scene'],z['backend'],z['level'],z['evaluated_nodes'],z['split_nodes'],z['final_leaves'],z['final_representatives'],z['provisional_representatives'],'/'.join(z['occupancy_'+k]for k in ['mean','median','p90','max']),'/'.join(z['Q_'+k]for k in ['mean','median','p10','p90']),z['Raw_SI_weighted_Q']]for z in alllevels]),'']
for s in S:
    lines+=['## 6. '+s+'：四项完整质量','']
    for m in metrics:
        lines += ['### '+m+(' ↑'if m in ['psnr','ssim']else' ↓'),'',table(['ratio','K']+A,[[f'{r:.0%}',int(structure[s]['new']['N_candidate']*r+.5)]+[f"{float(Q[s,a,r][m]):.6f}"for a in A]for r in R]),'']
    lines += ['### 四种因果差值','',table(['comparison','ratio','ΔPSNR','ΔSSIM','ΔLPIPS','Δphoto'],[[z['comparison'],f"{float(z['ratio']):.0%}"]+[f"{float(z['delta_'+m]):+.6f}"for m in metrics]for z in delta if z['scene']==s]),'',
        'C32−C0为grid对C的影响；P0−C0为旧Progressive效果；P32−C32为新Progressive效果；P32−P0为固定Progressive机制下grid的直接影响。不能用P32−C0替代这四项比较。','']
lines += ['## 7. Paired per-view','',
'所有四种比较的四metric均保存mean/median/p10/p90和improved-view fraction。下表给出核心P32−C32全metric结果；完整数据见paired_statistics.csv和paired_per_view_deltas.csv。LPIPS/photo负差才为改善。','',
table(['scene','ratio','metric','mean','median','p10','p90','improved views'],[[z['scene'],f"{float(z['ratio']):.0%}",z['metric']]+[f"{float(z[k]):+.6f}"for k in ['mean','median','p10','p90']]+[f"{float(z['improved_view_fraction']):.2%}"]for z in paired if z['comparison']=='P32_minus_C32']),'',
'## 8. Progressive composition与unlock','',
table(['scene','arm','ratio','internal reps','final reps','residual','emit L1/L2/L3','L1/L2/L3 fraction'],[[z['scene'],z['arm'],f"{float(z['ratio']):.0%}",z['internal_rep_count'],z['final_rep_count'],z['independent_residual_count'],'/'.join(z['L'+str(i)+'_count']for i in [1,2,3]),'/'.join(f"{float(z['L'+str(i)+'_fraction']):.2%}"for i in [1,2,3])]for z in composition]),'',
'每scene P32_unlock_trace.csv记录internal node的level、side length、side/d50、occupancy、Raw mass、Q、实际coarse reps及global ranks、child unlock rank、child数量。P32物理边长按32/16/8 d50对应各level；new tree AABB和child membership同时独立核对。Composition一致性只作为diagnostic，不参与quality gate。','',
'## 9. 统一Fixed-h8 redundancy与ranking overlap','',
'全部arm继续使用同一个原Fixed-h8 diagnostic：同cell内selected row最大off-diagonal κ，singleton0，unordered pairs阈值.8/.9/.95。未用新hierarchy自己的cell充当评价邻域。','',
table(['scene','arm','ratio','mean','median','p90','κ>.8','κ>.9','κ>.95'],[[z['scene'],z['arm'],f"{float(z['ratio']):.0%}"]+[f"{float(z[k]):.6f}"for k in ['mean','median','p90']]+[z['pairs_gt_'+str(t)]for t in [.8,.9,.95]]for z in red]),'',
table(['scene','comparison','ratio','Jaccard','intersection','added/removed'],[[z['scene'],z['comparison'],f"{float(z['ratio']):.0%}",f"{float(z['Jaccard']):.6f}",z['intersection'],z['added']+'/'+z['removed']]for z in overlap]),'',
'各scene ranking_row_changes.csv保存source row/xyz/Raw SI、old/new rank、old/new root与leaf level、C代表身份、P0/P32 emit身份及depth、selected状态。','',
'## 10. Correctness与mass语义','',
table(['Test','结果/证据'],[['A/B','deterministic median；h exact=32*d50；新backend无search/target'],['C/D','负坐标floor与巨大key范围，occupied-only sparse entries'],['E','101主体点+极远candidate outlier，h变化比例=1；无AABB驱动变化'],['F','invalid/unobserved极远xyz/scale变化不影响d50/root/tree；非法candidate scale明确拒绝'],['G/H','synthetic与real child occupancy conservation，无重复/丢失row'],['I','C32两次完整输出/tree/rep/Delta/Q exact'],['J','P32两次完整order/gain/emit/unlock exact'],['K','独立argmax、ancestor cover、residual和frontier replay exact'],['L','六arm相同K、完整permutation与nested prefix']]),'',
table(['scene','C32 mass relative','max leaf mass relative','replay rows','max replay gain error'],[[s,f"{checks['scenes'][s]['new_ranking']['global_mass_relative']:.3e}",f"{checks['scenes'][s]['new_ranking']['max_leaf_mass_relative']:.3e}",checks['scenes'][s]['replay']['rows_checked'],f"{checks['scenes'][s]['replay']['max_gain_error']:.3e}"]for s in S]),'',
'C32守恒来自原leaf Delta+residual数学，未renormalize。P32是ordinal ranking，不把Σ(N-rank)解释为Raw mass。geometry/selectCell/recursive split+redistribution代码块逐字不变，SHA见mathematical_parity.json；实验source diff见source_modification.diff。',
'',
'## 11. 预注册gate与假设判断','',
table(['gate','数值','要求','通过'],[['A fr2_xyz 1%恢复',f"{res['gate_values']['A_recovery_dB']:+.6f}",'≥+.05 dB',res['gates']['A_fr2_1percent_recovery']],['B fr1_xyz 5/10%平均恢复',f"{res['gate_values']['B_recovery_dB']:+.6f}",'≥+.05 dB',res['gates']['B_fr1_5_10_recovery']],['C fr1_xyz新1% Progressive效果',f"{res['gate_values']['C_new_delta_dB']:+.6f}",'≥+.10 dB',res['gates']['C_fr1_1percent_preserved']],['D fr1_room平均效果变化',f"{res['gate_values']['D_mean_change_dB']:+.6f}",'≥−.03 dB',res['gates']['D_room_preserved']]]),'',
table(['scene','old P0−C0 1/5/10/15%','new P32−C32 1/5/10/15%','old mean','new mean'],[[s,' / '.join(f'{v:+.6f}'for v in res['old_progressive_delta_PSNR'][s]),' / '.join(f'{v:+.6f}'for v in res['new_progressive_delta_PSNR'][s]),f"{np.mean(res['old_progressive_delta_PSNR'][s]):+.6f}",f"{np.mean(res['new_progressive_delta_PSNR'][s]):+.6f}"]for s in S]),'',
f"负于−.02 dB的Progressive cases：old={res['old_negative_outside_equivalence']}，new={res['new_negative_outside_equivalence']}；最差old/new={res['old_worst']:+.6f}/{res['new_worst']:+.6f} dB；12case标准差old/new={res['old_std']:.6f}/{res['new_std']:.6f}。这些辅助统计不替代A-D gate。",
'',
f"Strong逐项：{res['strong_checks']}。本轮假设判定：**{res['Q7']}**。C32质量变化必须与Progressive相对效果分别读表，不能将C32自身损失当成Progressive的稳定收益。",'',
'### 因果解释：相对恢复不等于绝对质量恢复','',
'- fr2_xyz 1%：C32−C0=−0.489482 dB，P32−P0=−0.425408 dB；P32−C32相对P0−C0改善+0.064073 dB，但两条新方案的绝对质量都下降。A gate通过不能单独支持新方案。',
'- fr1_xyz 5/10%：C32平均下降约0.112056 dB，P32平均比P0下降约0.032451 dB；相对Progressive效果改善+0.079605 dB，仍没有绝对P质量收益。',
'- fr1_xyz 1%：C32−C0=+0.571904 dB，P32−P0=+0.155267 dB；因为C32提升更多，Progressive相对效果从+0.250820反转到−0.165817 dB，C gate明确失败。',
'- fr1_room：新Progressive四档平均效果比旧增加+0.006367 dB，保住了原先成功scene；不能用这个局部成功覆盖XYZ反例。',
'',
'本轮符合“C32自身也有明显退化”的失败分支：**GRID SCALE HYPOTHESIS NOT SUPPORTED（固定32方案）**。已确认root-scale漂移是事实，但没有证明它是当前Progressive失败的主要可修复原因。因为32*d50也改变了Current C的结构质量，本轮不能进一步断言所有Progressive设计本身都错误；也不据此调其它lambda。','',
'## 12. Runtime、SHA与限制','',
table(['scene','new grid ms','new hierarchy ms','P32 ranking ms'],[[s,f"{checks['scenes'][s]['new_ranking']['new_grid_ms']:.3f}",f"{checks['scenes'][s]['new_ranking']['new_hierarchy_ms']:.3f}",f"{checks['scenes'][s]['new_ranking']['progressive_ms']:.3f}"]for s in S]),'',
'这些时间不作性能评价或gate；CPU reference和GPU同步构建不可直接当speedup比较。没有CUDA/knn/cache近似或batch优化。',
'',
'SHA冻结证明保存在各scene prior_checks.json：原Master/TF/cameras/GT/Bank文件和已冻结贡献/geometry/masks/old tree/ranking/quality，结束时重新核对；preregistration SHA保持不变。初次baseline诊断漏复制ranking_io依赖，已修复并保留失败日志；这发生在baseline数值验证前，没有baseline mismatch或新Grid结果选优。',
'',
'本轮仅一个预定义lambda=32，重复构建只用于correctness，不是多lambda实验。negative结果不证明所有可能的尺度归一化都错误，但不足以把target-2048的scale drift认定为当前Progressive失败的主要可修复原因。global d50对齐也不保证每个node拥有相同局部复杂度；本轮不引入local/node-specific scale。',
'',
'三个已训练frozen final maps、原SLAM归档视角，不能外推online teacher训练、SLAM稳定性或production性能。最终建议只为研究结论，不实际切换production backend。',
'',
f"**{res['Q7']}；下一步：{res['Q8']}。** 到此停止，不测试其它lambda、不改数学、不stage/commit/push。",'']
(ROOT/'D50_NORMALIZED_HIERARCHY_AUDIT.md').write_text('\n'.join(lines))
save(ROOT/'delivery_checks.json',dict(all_passed=True,baseline_rerenders=baseline_render,new_C32_P32_renders=new_render,total_renders=baseline_render+new_render,report_sha256=sha(ROOT/'D50_NORMALIZED_HIERARCHY_AUDIT.md'),no_new_F1_F3_renders=True,only_lambda=32,production_changed=False))
print('REPORT_COMPLETE',flush=True)
