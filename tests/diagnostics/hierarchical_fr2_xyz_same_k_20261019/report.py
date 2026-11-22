"""fr2_xyz 中文结果报告；仅展示已冻结实验，不参与排序和选参。"""
from common import *
import csv,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def rows(n):return list(csv.DictReader(open(ROOT/n)))
q=rows('quality_quality.csv');paired=rows('paired_statistics.csv');red=rows('selected_redundancy.csv');sp=rows('spatial_budget_coverage.csv');comp=rows('progressive_composition.csv');overlap=rows('ranking_overlap.csv')
pop=read(ROOT/'population.json');p=read(ROOT/'preregistration.json');res=read(ROOT/'mechanism_result.json');checks=read(ROOT/'ranking_sanity.json');manifest=read(ROOT/'input_manifest.json');camera=read(ROOT/'camera_checks.json');ratios=p['ratios'];arms=p['arms'];metrics=['psnr','ssim','lpips','photo']
Q={(a['arm'],float(a['ratio'])):a for a in q};R={(a['arm'],float(a['ratio'])):a for a in red}
short={'A_Raw':'A Raw','B_FixedMarginal':'B FixedMarginal','C_Hierarchical':'C Current','F3_ProgressiveMarginal':'F3 Progressive+Tail'}
def table(headers,data):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,a))+' |'for a in data])
def metric_table(metric):return table(['ratio','K']+[short[a]for a in arms],[[f'{r:.0%}',int(pop['N_candidate']*r+.5)]+[f'{float(Q[a,r][metric]):.6f}'for a in arms]for r in ratios])
def delta_table(a,b):return table(['ratio','ΔPSNR ↑','ΔSSIM ↑','ΔLPIPS ↓','Δphoto ↓'],[[f'{r:.0%}']+[f'{float(Q[a,r][m])-float(Q[b,r][m]):+.6f}'for m in metrics]for r in ratios])
fig,axes=plt.subplots(1,3,figsize=(14,4.2))
for a,style in zip(arms,['o-','s-','^-','D-']):
    axes[0].plot(np.array(ratios)*100,[float(Q[a,r]['psnr'])for r in ratios],style,label=short[a])
    axes[1].plot(np.array(ratios)*100,[float(Q[a,r]['lpips'])for r in ratios],style)
    axes[2].plot(np.array(ratios)*100,[float(R[a,r]['mean_redundancy'])for r in ratios],style)
for ax,title in zip(axes,['PSNR (higher better)','LPIPS (lower better)','Fixed-h8 mean redundancy']):ax.set(xlabel='Budget (%)',title=title);ax.grid(alpha=.25);ax.set_xticks([1,5,10,15])
axes[0].legend(fontsize=8);fig.suptitle(f"fr2_xyz | frozen Master + TF | {pop['evaluation']} non-Bank archived views");fig.tight_layout();fig.savefig(ROOT/'fr2_xyz_same_k_quality.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4))
for a in arms[1:]:ax.plot(np.array(ratios)*100,[float(Q[a,r]['psnr'])-float(Q['A_Raw',r]['psnr'])for r in ratios],'o-',label=short[a])
ax.axhline(0,color='black',lw=.8);ax.set(xlabel='Budget (%)',ylabel='Delta PSNR vs Raw (dB)',title='fr2_xyz zero-tuning same-K');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(ROOT/'fr2_xyz_delta_psnr.png',dpi=170);plt.close(fig)
shutil.copy2(ROOT/'quality_quality.csv',ROOT/'quality_summary.csv')
frozen=read(ROOT/'frozen_sources.json');source_rows=[]
for key in ['master','tf','config','cameras']:
    path=manifest[key];digest=frozen[path]['sha256']if path in frozen else sha(path);source_rows.append([key,path,digest])
source_rows += [[name,str(ROOT/name),sha(ROOT/name)]for name in ['contributions.pt','ranking_inputs.npz','hierarchy_input.bin','preregistration.json']]
fc=res['F3_delta_PSNR_vs_C'];ca=res['C_delta_PSNR_vs_A'];f3_label='COMBINED SUPPORTED'if res['F3_combined_supported']else'NOT SUPPORTED'
q_delta=' / '.join(f'{v:+.6f}'for v in fc)
trend=[]
for ratio in ratios:
    trend.append([f'{ratio:.0%}']+['改善'if(float(Q['F3_ProgressiveMarginal',ratio][m])-float(Q['C_Hierarchical',ratio][m]))*(1 if m in ['psnr','ssim']else-1)>0 else'持平'if float(Q['F3_ProgressiveMarginal',ratio][m])==float(Q['C_Hierarchical',ratio][m])else'回退'for m in metrics])
cross=[];cross_sources={}
for scene,path,qfile in [('fr1_room',Path('/home/prmi/zwj/hierarchical-progressive-marginal-20261019'),'progressive_marginal_quality_summary.csv'),('fr1_xyz',Path('/home/prmi/zwj/hierarchical-fr1-xyz-same-k-20261019'),'quality_summary.csv')]:
    old=list(csv.DictReader(open(path/qfile)));oldQ={(a['arm'],float(a['ratio'])):float(a['psnr'])for a in old};cross_sources[str(path/qfile)]=sha(path/qfile)
    vals=[oldQ['F3_ProgressiveMarginal',r]-oldQ['C_Hierarchical',r]for r in ratios]
    cross.append([scene]+[f'{v:+.6f}'for v in vals]+[f'{np.mean(vals):+.6f}'])
cross.append(['fr2_xyz']+[f'{v:+.6f}'for v in fc]+[f"{res['F3_mean_delta_PSNR_C']:+.6f}"])
save(ROOT/'cross_scene_reference_sources.json',cross_sources)
one=next(a for a in comp if a['arm']=='F3_ProgressiveMarginal'and float(a['ratio'])==.01)
lines=['# fr2_xyz：A/B/C/F3 Frozen Same-K 审计','',
'## 结论','',
f"**F3 本场景判定：{f3_label}；Strong improvement：{'YES'if res['F3_strong_improvement']else'NO'}。** F3−C的1/5/10/15% PSNR差依次为 **{q_delta} dB**，四档平均 **{res['F3_mean_delta_PSNR_C']:+.6f} dB**。F3相对Fixed h8平均 **{res['F3_mean_delta_PSNR_B']:+.6f} dB（{res['F3_vs_B_PSNR']}，仅PSNR水平）**。",
f"C相对Raw的PSNR主gate：**{'PASS'if res['C_primary_PSNR_gate']else'FAIL'}**；对应四档ΔPSNR：{' / '.join(f'{v:+.6f}'for v in ca)} dB。其余指标方向必须结合下表判断，不能把PSNR gate等同所有指标全面占优。",
'',
'本轮只新增fr2_xyz离线结果，没有改变算法、参数或production开关，也没有训练或online运行。跨场景结论以各scene分别呈现，不用总体平均掩盖某个scene的回退。','',
'## 1. 冻结数据与评价协议','',
f"- 数据集：`{manifest['dataset']}`；GT直接从用户指定目录读取。",
'- checkpoint：已有 `full/attempt_1/model/37900_shutdown/ply`，按最终shutdown选择，没有读质量挑checkpoint。',
f"- Master **{pop['N_total']}** rows；valid **{pop['valid_count']}**；observed/candidate **{pop['N_candidate']}**。candidate=`valid AND observed`，effective maturity=1。",
f"- 共{camera['cameras']}个归档KF，其中{pop['Bank']}个实际Bank views只用于Raw SI，其余{pop['evaluation']}个全部作为 **fixed non-Bank archived evaluation views**。这些视角参与原SLAM，不称unseen/test-only/independent novel views。",
'- 没有既有fr2_xyz同协议贡献fixture；本轮用已归档Bank和原renderer导出一次完整frame contribution并冻结，四arm共用Raw SI=Bank contribution mean，不更新Bank。',
f"- `view_bank_orb.json`的scene文本仍为过期标签`{camera['source_scene_label']}`；本轮不使用标签选择数据。实际Bank IDs全属于同一checkpoint相机，所有GT图像来自明确的fr2_xyz路径，均保存SHA。",
'- 相机/FOV/GT转换、renderer、TF consumer和四项metric实现逐字复用上一轮。每个ratio所有arm共用同一TF参数与AABB，除了selected rows没有其它输入差别。',
'- production photo=`0.8*L1 + 0.2*(1-production fast_ssim)`；单独SSIM沿用TorchMetrics实现，不与photo里的fast_ssim混同。','',
table(['输入','路径','SHA256'],source_rows),'',
'完整原始输入SHA见 `frozen_sources.json`；相机archive SHA和GT图像SHA见 `camera_checks.json`。全部输入结束时重新核对。','',
'## 2. Arms、参数与correctness','',
table(['Arm','定义'],[['A','Raw SI全局降序，tie source row'],['B','Fixed h8 geometry-only完整local marginal greedy + predecessor-preserving merge'],['C','SparseHash当前3rep Delta + independent residual，全局score排序'],['F3','同一冻结tree，ancestor-conditioned progressive coarse-to-fine unlock + final-leaf conditional marginal tail']]),'',
'target occupied roots=2048，origin=(0,0,0)，h搜索不变，3 reps/node，Q<0.80 split，max L3，scalar-radius geometry κ，maturity=1。没有参数搜索、view/covariance、dedup或boundary correction。',
'',
'F3最终ranking等于heap实际emit顺序，不按priority二次排序；tie为priority降序、root key+octree path字典序、source row升序。已emit ancestor row被排除，每row恰好一次。Ordinal score只表达order，不解释物理importance mass。','',
table(['ratio','K'],[[f'{r:.0%}',int(pop['N_candidate']*r+.5)]for r in ratios]),'',
'- 四arm同candidate population、完整permutation、same-K与nested prefixes通过。',
'- B两次完整重算exact一致，逐cell predecessor order通过。',
'- C用原C++静态库构建两次，score、membership、h、tree、rep IDs/Delta/Q exact一致；全部节点first3 greedy和Q独立重算通过。',
f"- F3完整order/gain/emit/unlock重复运行exact一致；synthetic测试通过；独立重放全部{pop['N_candidate']}次argmax/gain及frontier/unlock，最大gain差0。",
f"- C mass相对误差：{checks['mass_relative_error']:.3e}。F3评价ranking，不宣称ordinal score守恒物理mass。",
'- 这是新scene，没有历史fr2_xyz同协议质量可做exact对照；本轮核对沿用数学/worker/metric的SHA，并在本scene做repeat和独立验证。','',
'## 3. Sparse hierarchy结构','',
table(['统计','值'],[[k,pop[k]]for k in ['occupied_roots','h_root','d50','h','final_leaves','final_representatives','weighted_leaf_Q']]),'',
'结构Q不参与质量gate。完整冻结tree/membership和内部provisional reps已保存。','',
'## 4. Same-K真实render质量','',
'### PSNR ↑','',metric_table('psnr'),'',
'### SSIM ↑','',metric_table('ssim'),'',
'### LPIPS ↓','',metric_table('lpips'),'',
'### Production photo ↓','',metric_table('photo'),'',
f"合计4 arms ×4 ratios ×{pop['evaluation']}views = **{16*pop['evaluation']}次真实render**。",
'',
'### C − Raw','',delta_table('C_Hierarchical','A_Raw'),'',
'### F3 − C','',delta_table('F3_ProgressiveMarginal','C_Hierarchical'),'',
'### F3 − FixedMarginal','',delta_table('F3_ProgressiveMarginal','B_FixedMarginal'),'',
'### F3相对C的指标方向','',table(['ratio','PSNR','SSIM','LPIPS','photo'],trend),'',
'## 5. Paired per-view','',
'F3−C的全部配对统计如下。LPIPS/photo为负差才是改善；positive_delta_fraction与improved_view_fraction在CSV中分别保存。所有其它比较及逐view差值也已保存。','',
table(['ratio','metric','mean','median','p10','p90','improved views'],[[f"{float(a['ratio']):.0%}",a['metric']]+[f"{float(a[k]):+.6f}"for k in ['mean','median','p10','p90']]+[f"{float(a['improved_view_fraction']):.2%}"]for a in paired if a['comparison']=='F3_ProgressiveMarginal_minus_C_Hierarchical']),'',
'## 6. 统一Fixed-h8 geometry重复度','',
'完全沿用上一轮：同一固定h8 cell内selected rows，逐row最大off-diagonal κ，singleton记0；另统计unordered高相似pair。它是统一局部diagnostic，不是无边界全局overlap。','',
table(['ratio','arm','mean','median','p90','κ>.8','κ>.9','κ>.95'],[[f"{float(a['ratio']):.0%}",short[a['arm']]]+[f"{float(a[k]):.6f}"for k in ['mean_redundancy','median_redundancy','p90_redundancy']]+[a['pairs_gt_'+str(t)]for t in [.8,.9,.95]]for a in red]),'',
'不将重复度下降本身当作quality GO；重建质量由四项真实render指标判断。','',
'## 7. Composition、空间覆盖和ranking overlap','',
table(['ratio','arm','C-final-rep fraction','internal reps','final reps','tail','emit L1/L2/L3'],[[f"{float(a['ratio']):.0%}",short[a['arm']],f"{float(a['C_representative_fraction']):.2%}",a.get('internal_rep_count',''),a.get('final_rep_count',''),a.get('marginal_tail_count',''),'/'.join(a.get('emitted_L'+str(i)+'_count','')for i in [1,2,3])]for a in comp if a['arm']in ['C_Hierarchical','F3_ProgressiveMarginal']]),'',
table(['ratio','arm','hit roots','hit leaves','hit-root mass','hit-leaf mass','dense-root budget','important-root budget'],[[f"{float(a['ratio']):.0%}",short[a['arm']],a['selected_roots'],a['selected_final_leaves']]+[f"{float(a[k]):.2%}"for k in ['hit_root_Raw_mass_fraction','hit_leaf_Raw_mass_fraction','dense_roots_budget_fraction','important_roots_budget_fraction']]for a in sp if a['arm']in ['C_Hierarchical','F3_ProgressiveMarginal']]),'',
table(['ratio','comparison','Jaccard','intersection','added/removed'],[[f"{float(a['ratio']):.0%}",a['comparison'],f"{float(a['Jaccard']):.6f}",a['intersection'],a['added']+'/'+a['removed']]for a in overlap if a['comparison']in ['C_Hierarchical_vs_F3_ProgressiveMarginal','B_FixedMarginal_vs_F3_ProgressiveMarginal']]),'',
f"1% F3的internal/final/tail数量为{one['internal_rep_count']}/{one['final_rep_count']}/{one['marginal_tail_count']}，emit L1/L2/L3为{one['emitted_L1_count']}/{one['emitted_L2_count']}/{one['emitted_L3_count']}。高档F3同时改变depth scheduling与tail，本轮未跑F1/F2，不能唯一分离二者因果；不自动追加实验或调参数。",'',
'## 8. 预注册判定和三场景对照','',
table(['gate','结果'],[['C相对Raw的PSNR主gate','PASS'if res['C_primary_PSNR_gate']else'FAIL'],['F3四档mean ΔPSNR≥0',f"{'PASS'if np.mean(fc)>=0 else'FAIL'}：{np.mean(fc):+.6f} dB"],['F3 1%优于C',f"{'PASS'if fc[0]>0 else'FAIL'}：{fc[0]:+.6f} dB"],['F3 10/15% mean redundancy<C','PASS'if res['F3_high_redundancy_gate']else'FAIL'],['Combined',f3_label],['Strong improvement','YES'if res['F3_strong_improvement']else'NO'],['F3相对B PSNR',res['F3_vs_B_PSNR']]]),'',
'以下为各scene内部F3−C的ΔPSNR，不能当作跨scene绝对PSNR直接比较，也不按view数量加权掩盖反例。','',
table(['scene','1%','5%','10%','15%','四档平均'],cross),'',
'fr1_room支持F3，fr1_xyz未通过Combined；本轮fr2_xyz结果增加一条零调参证据，不改写已有反例。任何跨scene整体替换结论都必须同时保留这些差异。','',
'本场景C的1/5% PSNR收益不能掩盖10/15%回退，因此C的四档Primary PSNR gate也未通过。F3的四档平均相对B为正主要由1%贡献，5/10/15%均低于B；不能把EXCEEDS均值标签解释为逐档或全指标支配。当前不支持用F3统一替代C，也不据此宣称C具有全ratio质量GO。','',
'## 9. Runtime与已知限制','',
table(['阶段','单次记录ms'],[[a,f'{v:.3f}']for a,v in checks['runtime_ms'].items()]),'',
'CPU NumPy reference与GPU同步C构建口径不同，不作速度比较，runtime不参与gate。',
'',
'单个frozen final map、原SLAM归档视角、实际Bank较少；未做online训练、ATE比较或Selector适配。本轮只能评价same-K reconstruction ranking。SHA与构建/执行日志均保留，结束时重新核对输入与源码。','',
'## 10. 交付','',
'主结果为 `summary.json`、`quality_summary.csv`、`quality_per_view.csv`；完整四metric差值、paired统计、ranking overlap、逐row变化、selected redundancy、composition、空间覆盖、完整emit/unlock traces、ranking数组、SHA和validation均已保存。',
'',
f"**fr2_xyz：{f3_label}；Strong improvement：{'YES'if res['F3_strong_improvement']else'NO'}。结论限定本场景离线same-K，未commit、未push、未改production。**",'']
(ROOT/'FR2_XYZ_HIERARCHICAL_SAME_K_AUDIT.md').write_text('\n'.join(lines))
print('REPORT_COMPLETE',flush=True)
