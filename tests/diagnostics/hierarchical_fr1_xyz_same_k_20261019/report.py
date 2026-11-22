"""中文结果报告与静态科研图；不参与算法和评价。"""
from common import *
import csv,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def rows(n):return list(csv.DictReader(open(ROOT/n)))
q=rows('quality_quality.csv');d=rows('quality_deltas.csv');paired=rows('paired_statistics.csv');red=rows('selected_redundancy.csv');sp=rows('spatial_budget_coverage.csv');comp=rows('progressive_composition.csv');overlap=rows('ranking_overlap.csv')
pop=read(ROOT/'population.json');p=read(ROOT/'preregistration.json');result=read(ROOT/'mechanism_result.json');checks=read(ROOT/'ranking_sanity.json');frames=read(ROOT/'input_manifest.json');ratios=p['ratios'];arms=p['arms'];metrics=['psnr','ssim','lpips','photo']
Q={(r['arm'],float(r['ratio'])):r for r in q};R={(r['arm'],float(r['ratio'])):r for r in red}
short={'A_Raw':'A Raw','B_FixedMarginal':'B FixedMarginal','C_Hierarchical':'C Current','F3_ProgressiveMarginal':'F3 Progressive+Tail'}
def table(headers,data):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |'for r in data])
def metric_table(metric):return table(['ratio','K']+[short[a]for a in arms],[[f'{r:.0%}',int(pop['N_candidate']*r+.5)]+[f'{float(Q[a,r][metric]):.6f}'for a in arms]for r in ratios])
def delta_table(first,second):
    return table(['ratio','ΔPSNR ↑','ΔSSIM ↑','ΔLPIPS ↓','Δphoto ↓'],[[f'{r:.0%}']+[f'{float(Q[first,r][m])-float(Q[second,r][m]):+.6f}'for m in metrics]for r in ratios])
fig,axes=plt.subplots(1,3,figsize=(14,4.2));styles=['o-','s-','^-','D-']
for a,style in zip(arms,styles):
    axes[0].plot(np.array(ratios)*100,[float(Q[a,r]['psnr'])for r in ratios],style,label=short[a])
    axes[1].plot(np.array(ratios)*100,[float(Q[a,r]['lpips'])for r in ratios],style)
    axes[2].plot(np.array(ratios)*100,[float(R[a,r]['mean_redundancy'])for r in ratios],style)
for ax,title in zip(axes,['PSNR (higher better)','LPIPS (lower better)','Fixed-h8 mean redundancy']):ax.set(xlabel='Budget (%)',title=title);ax.grid(alpha=.25);ax.set_xticks([1,5,10,15])
axes[0].legend(fontsize=8);fig.suptitle('fr1_xyz | frozen Master + TF | 54 non-Bank archived views');fig.tight_layout();fig.savefig(ROOT/'fr1_xyz_same_k_quality.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4))
for a in ['B_FixedMarginal','C_Hierarchical','F3_ProgressiveMarginal']:ax.plot(np.array(ratios)*100,[float(Q[a,r]['psnr'])-float(Q['A_Raw',r]['psnr'])for r in ratios],'o-',label=short[a])
ax.axhline(0,color='black',lw=.8);ax.set(xlabel='Budget (%)',ylabel='Delta PSNR vs Raw (dB)',title='fr1_xyz zero-tuning same-K');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(ROOT/'fr1_xyz_delta_psnr.png',dpi=170);plt.close(fig)
shutil.copy2(ROOT/'quality_quality.csv',ROOT/'quality_summary.csv')
frozen=read(ROOT/'frozen_sources.json');source_rows=[]
for key in ['master','tf','config','cameras']:
    path=frames[key];digest=frozen[path]['sha256']if path in frozen else sha(path);source_rows.append([key,path,digest])
source_rows += [[name,str(ROOT/name),sha(ROOT/name)]for name in ['contributions.pt','ranking_inputs.npz','hierarchy_input.bin','preregistration.json']]
lines=['# fr1_xyz：A/B/C/F3 Frozen Same-K 审计','',
'## 结论','',
'**Current C 相对 Raw 的低 budget 收益复现；F3 的全档优势没有跨场景复现。** F3 在1%比 C 高 **0.250820 dB**，但5/10/15%分别低 **0.110466 / 0.124511 / 0.059770 dB**，四档平均 **−0.010982 dB**，未通过沿用的 Combined gate。F3 相对 Fixed h8 的四档平均差为 **−0.063646 dB（BELOW）**。',
'',
'F3 四档 LPIPS 均优于 C，但5/10/15%的 PSNR、SSIM、photo 同时回退。因此不能仅凭重复度下降或 LPIPS 改善判定 F3 更好。建议保留当前 C，F3 继续作为诊断候选；本轮没有修改任何算法或生产开关。','',
'## 1. 数据、checkpoint 与冻结协议','',
f"- 数据集：`{frames['dataset']}`。GT直接来自用户指定目录。",
'- checkpoint：已有 `full/attempt_1/model/9900_shutdown/ply`，按最终 shutdown 选择，不读质量挑 checkpoint。',
f"- Master共 **{pop['N_total']}** rows，valid **{pop['valid_count']}**，observed及candidate均 **{pop['N_candidate']}**；candidate=`valid AND observed`，effective maturity=1。",
f"- 实际 Bank **{pop['Bank']}** views；全部归档KF共72，其余 **{pop['evaluation']}** 个为 **fixed non-Bank archived evaluation views**。这些视角参与过原SLAM，不称为 unseen/test-only/independent novel views。",
'- 该场景原先没有本协议的贡献fixture，因此本轮仅用已归档实际Bank和原renderer导出一次完整 frame contribution并冻结；A/B/C/F3共用此矩阵及其mean Raw SI，不更新Bank。',
'- 归档 `view_bank_orb.json` 的 `scene` 文本为过期的 `fr3_long_office_household`；本轮不使用该标签。Bank只读取实际keyframe IDs，所有ID属于同checkpoint的72个相机，图像从明确的fr1_xyz路径读取并保存SHA。',
'- 相机/FOV/GT转换函数逐字复用上一轮；renderer、TF consumer、PSNR/SSIM/LPIPS及production photo实现也逐字复用。TF参数和AABB加载自该checkpoint，所有arm一致。',
'- production photo=`0.8*L1 + 0.2*(1-production fast_ssim)`；单独报告的SSIM沿用上一轮TorchMetrics实现，两者不混同。','',
table(['输入','路径','SHA256'],source_rows),'',
'全部原始相机图像、checkpoint JSON、数学源码、历史runtime等SHA见 `frozen_sources.json`；生成相机archive的SHA见 `camera_checks.json`，结束时全部重新检查。','',
'## 2. Arms与固定参数','',
table(['Arm','定义'],[['A','Raw SI全局降序，tie source row'],['B','Fixed h8 geometry-only完整local marginal greedy + predecessor-preserving merge'],['C','SparseHash当前3rep Delta + independent residual，全局score降序'],['F3','同一冻结tree，ancestor-conditioned progressive 3新rep/node，完成后解锁child；final leaf继续conditional marginal tail']]),'',
'target occupied roots=2048；origin=(0,0,0)；h搜索规则原样；3 representatives/node；Q<0.80 split；max L3；scalar-radius geometry-only κ；maturity=1。没有参数搜索、boundary correction、view/covariance、mass dedup、训练或online运行。',
'',
'F3完整ranking为实际active-frontier heap emit顺序，未按priority重排。tie为priority降序、root key+octree path字典序、source row升序；已emit ancestor row不重复出现。其ordinal score只表达rank，不是守恒importance mass。','',
'## 3. Correctness和same-K','',
table(['ratio','K'],[[f'{r:.0%}',int(pop['N_candidate']*r+.5)]for r in ratios]),'',
'- A/B/C/F3均为同一candidate population的完整permutation，四档均为同一ranking的nested prefixes。',
'- B完整重算两次exact一致，并逐cell检查local predecessor order。',
'- C用上一轮原C++静态库构建两次：score、membership、h、tree、rep IDs/Delta/Q exact一致；对全部node独立重算first3 greedy及Q通过。',
'- F3完整构建两次，order/gain/emit/unlock trace exact一致。10项synthetic和耗尽child测试通过。',
f"- 独立重放 **{pop['N_candidate']}** 次argmax/gain、ancestor coverage及全局frontier/unlock，最大gain差 **0**；无重复、丢失row。",
f"- C质量信号mass相对误差 **{checks['mass_relative_error']:.3e}**；F3仅评价ranking，不宣称全局物理mass守恒。",
'- 本轮是新scene，没有fr1_xyz既有A/B/C同协议质量可做历史exact对照；复用的数学、worker及metric文件SHA与上一轮一致，本场景内部做exact repeat和独立核对。','',
'## 4. Sparse hierarchy结构','',
table(['统计','值'],[[k,pop[k]]for k in ['occupied_roots','h_root','d50','h','final_leaves','final_representatives','weighted_leaf_Q']]),'',
'weighted Q只解释结构，不参与质量判定。完整节点和membership见 `frozen_tree_nodes.json`、`frozen_tree_membership.npz`。','',
'## 5. 四项真实render质量','',
'### PSNR ↑','',metric_table('psnr'),'',
'### SSIM ↑','',metric_table('ssim'),'',
'### LPIPS ↓','',metric_table('lpips'),'',
'### Production photo ↓','',metric_table('photo'),'',
'共4 arms ×4 ratios ×54views = **864次真实render**。所有arm除selected rows外均使用相同Master/TF/camera/renderer/GT。','',
'### C − Raw','',delta_table('C_Hierarchical','A_Raw'),'',
'### F3 − C','',delta_table('F3_ProgressiveMarginal','C_Hierarchical'),'',
'### F3 − FixedMarginal','',delta_table('F3_ProgressiveMarginal','B_FixedMarginal'),'',
'## 6. Paired per-view结果','',
'下表为F3−C，四档各54个配对视角；positive/improved fraction按指标方向区分，LPIPS/photo负差才为改善。全部比较和逐视角差值保存为CSV。','',
table(['ratio','metric','mean','median','p10','p90','improved views'],[[f"{float(z['ratio']):.0%}",z['metric']]+[f"{float(z[k]):+.6f}"for k in ['mean','median','p10','p90']]+[f"{float(z['improved_view_fraction']):.2%}"]for z in paired if z['comparison']=='F3_ProgressiveMarginal_minus_C_Hierarchical']),'',
'F3−C的PSNR改善视角比例依次为 **94.44% / 3.70% / 3.70% / 14.81%**；高档回退具有配对一致性，并非少数异常视角拉低均值。','',
'## 7. 统一Fixed-h8 geometry重复度','',
'与上一轮完全相同：只比较同一固定h8 cell中的selected rows；每row取最大off-diagonal κ，singleton记0，另统计unordered高相似pair。它是统一诊断，不是全局无边界重叠估计。','',
table(['ratio','arm','mean','median','p90','κ>.8','κ>.9','κ>.95'],[[f"{float(z['ratio']):.0%}",short[z['arm']]]+[f"{float(z[k]):.6f}"for k in ['mean_redundancy','median_redundancy','p90_redundancy']]+[z['pairs_gt_'+str(t)]for t in [.8,.9,.95]]for z in red]),'',
'F3的mean redundancy四档均低于C，但仍明显高于B。10/15% κ>.9 pairs由C的64/104降至F3的55/85，仍不能阻止PSNR/SSIM/photo回退。重复度下降不是质量提升的充分条件。','',
'## 8. Budget composition、空间覆盖与overlap','',
table(['ratio','arm','C-final-rep fraction','internal reps','final reps','tail','emit L1/L2/L3'],[[f"{float(z['ratio']):.0%}",short[z['arm']],f"{float(z['C_representative_fraction']):.2%}",z.get('internal_rep_count',''),z.get('final_rep_count',''),z.get('marginal_tail_count',''),'/'.join(z.get('emitted_L'+str(i)+'_count','')for i in [1,2,3])]for z in comp if z['arm']in ['C_Hierarchical','F3_ProgressiveMarginal']]),'',
table(['ratio','arm','hit roots','hit leaves','hit-root Raw mass','hit-leaf Raw mass','dense-root budget','important-root budget'],[[f"{float(z['ratio']):.0%}",short[z['arm']],z['selected_roots'],z['selected_final_leaves']]+[f"{float(z[k]):.2%}"for k in ['hit_root_Raw_mass_fraction','hit_leaf_Raw_mass_fraction','dense_roots_budget_fraction','important_roots_budget_fraction']]for z in sp if z['arm']in ['C_Hierarchical','F3_ProgressiveMarginal']]),'',
table(['ratio','comparison','Jaccard','intersection','added/removed'],[[f"{float(z['ratio']):.0%}",z['comparison'],f"{float(z['Jaccard']):.6f}",z['intersection'],z['added']+'/'+z['removed']]for z in overlap if z['comparison']in ['C_Hierarchical_vs_F3_ProgressiveMarginal','B_FixedMarginal_vs_F3_ProgressiveMarginal']]),'',
'1% F3全部236次emit来自L1，tail=0，因此该档F3−C提升不能归因于marginal tail，直接支持coarse scheduling在极低budget的作用。高档同时改变depth scheduling与tail，本轮只跑F3组合，不能唯一归因于其中某一个机制；不自动补跑F1/F2或调参数。','',
'## 9. 预注册gate与跨场景解释','',
table(['检查','结果'],[['C相对Raw的PSNR主gate','PASS：四档均不低，1/5%平均显著改善'],['F3四档mean ΔPSNR≥0',f"FAIL：{result['F3_mean_delta_PSNR_C']:+.6f} dB"],['F3 1%优于C','PASS：+0.250820 dB'],['F3 10/15% mean redundancy<C','PASS'],['Combined supported','NO'],['Strong improvement','NO'],['F3相对B的PSNR水平','BELOW：四档mean −0.063646 dB']]),'',
'fr1_room上一轮F3−C四档平均 +0.059593 dB，本轮fr1_xyz为 −0.010982 dB。fr1_room的COMBINED SUPPORTED不能据此推广为跨场景稳定优于C；本轮也不能把F3低budget收益写成online或production GO。',
'',
'## 10. Runtime、边界与交付','',
table(['阶段','单次记录ms'],[[a,f"{v:.3f}"]for a,v in checks['runtime_ms'].items()]),'',
'CPU NumPy reference与GPU同步C构建的时间口径不同，不作速度比较；runtime不参与任何gate。',
'',
'最初独立wrapper的CMake缺少CUDA架构设置，按项目已有CUDA11.8/GCC9/sm86设置修复；分析脚本上传前pipeline结束一次，随后只运行分析。失败日志均保留，无算法、输入或质量选优改动。',
'',
'局限：单个frozen final map、归档训练时视角、实际Bank较少；没有online训练或ATE比较；所有结论限定same-K reconstruction ranking。',
'',
'输出包括 `summary.json`、`quality_summary.csv`、`quality_per_view.csv`、`quality_deltas.csv`、`paired_statistics.csv`、`paired_per_view_deltas.csv`、`ranking_overlap.csv`、逐row变化、`selected_redundancy.csv`、`progressive_composition.csv`、`spatial_budget_coverage.csv`、完整emit/unlock trace、ranking数组及SHA/check日志。',
'',
'**最终结论：Current C 的低budget收益在fr1_xyz复现；F3 Combined在本场景 NOT SUPPORTED，暂保留 Current C，不作全局替换。** 未commit，未push，未改production。','']
(ROOT/'FR1_XYZ_HIERARCHICAL_SAME_K_AUDIT.md').write_text('\n'.join(lines))
print('REPORT_COMPLETE',flush=True)
