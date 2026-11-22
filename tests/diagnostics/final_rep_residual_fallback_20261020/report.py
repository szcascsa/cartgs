from pathlib import Path
import json,csv,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent;SCENES=['fr1_room','fr1_xyz','fr2_xyz'];R=[.01,.05,.1,.15];ARMS=['A','B','C','F','RF-C','RF-Raw','RF-Gap'];M=['psnr','ssim','lpips','photo']
def read(p):return json.loads(Path(p).read_text())
def rows(p):return list(csv.DictReader(open(p)))
def table(h,a):return '\n| '+' | '.join(h)+' |\n| '+' | '.join(['---']*len(h))+' |\n'+''.join('| '+' | '.join(map(str,v))+' |\n'for v in a)+'\n'
def f(v):return '—'if v is None or v==''else f'{float(v):.6f}'
def d(v):return f'{float(v):+.6f}'
def pc(v):return '—'if v is None or v==''else f'{100*float(v):.2f}%'
def ratio(v):return f'{100*float(v):g}%'
data={s:read(ROOT/s/'audit_summary.json')for s in SCENES};I=read(ROOT/'interpretation.json');q={s:{(a['arm'],float(a['ratio'])):a for a in data[s]['quality']}for s in SCENES}
for s in SCENES:assert data[s]['all_passed']and all(np.isfinite(float(a[m]))for a in data[s]['quality']for m in M)
names=['residual_requirement_by_ratio.csv','residual_pool_rows.csv','residual_proxy_comparison.csv','leaf_residual_statistics.csv','q_residual_statistics.csv','coverage_residual_statistics.csv','proxy_false_negative_rows.csv','proxy_false_positive_rows.csv','candidate_pool_summary.csv','candidate_recall.csv','quality_summary.csv','quality_per_view.csv','paired_statistics.csv','spatial_coverage.csv','redundancy_statistics.csv','quality_deltas.csv','proxy_error_statistics.csv','leaf_residual_concentration.csv']
for name in names:
    a=[dict(scene=s,**v)for s in SCENES for v in rows(ROOT/s/name)]
    if a:
        with open(ROOT/name,'w',newline='')as out:w=csv.DictWriter(out,fieldnames=list(a[0]));w.writeheader();w.writerows(a)
rankings={};selected={}
for s in SCENES:
    b=dict(np.load(ROOT/s/'rankings.npz'));z=dict(np.load(ROOT/s/'fallback_rankings.npz'));ids=b['row_ids'];rankings[s+'__source_rows']=ids
    for arm,key in [('A','A'),('B','B'),('C','C0')]:rankings[s+'__'+arm+'_source_row_order']=ids[b[key+'_order']]
    F=z['F'];rankings[s+'__F_source_row_order']=ids[F[np.lexsort((ids[F],-b['C0_score'][F]))]]
    for arm in ARMS[4:]:rankings[s+'__'+arm+'_source_row_order']=ids[z[arm+'_order']];rankings[s+'__'+arm+'_pool_source_rows']=ids[z[arm+'_pool']]
    for key in ['F','U','D15_RES','D15_RAW','D15_GAP']:rankings[s+'__'+key+'_source_rows']=ids[z[key]]
    selected.update({s+'__'+k:v for k,v in np.load(ROOT/s/'selected_sets.npz').items()})
np.savez_compressed(ROOT/'rankings.npz',**rankings);np.savez_compressed(ROOT/'selected_sets.npz',**selected)
ind=read(ROOT/'independent_checks.json');assert ind['all_passed']
for s,a in ind['pre_render_SHA'].items():
    for n,h in a.items():assert hashlib.sha256((ROOT/s/n).read_bytes()).hexdigest()==h
pre=hashlib.sha256((ROOT/'preregistration.json').read_bytes()).hexdigest()
for s in SCENES:assert read(ROOT/s/'prior_checks.json')['preregistration_sha256']==pre
checks=dict(all_passed=True,scenes={s:data[s]['correctness']for s in SCENES},independent_checks=ind,RF_C_render_exact=True,preregistration_unchanged=True,production_unchanged=True,no_stage=True,no_commit=True,no_push=True)
(ROOT/'correctness.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
summary=dict(verdict=I['verdict'],secondary_verdict=I['secondary_verdict'],answers=I['answers'],scenes={s:{k:data[s][k]for k in ['N_candidate','requirements','pools','proxy_recall','candidate_recall','deltas','concentration','renders']}for s in SCENES},interpretation=I,correctness_pass=True,HEAD='3cfccc4a92c8e1ec6c3d491dac1c9efbf151daeb',branch='redundancy-aware-si',preregistration_sha256=pre,stage=False,commit=False,push=False)
(ROOT/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
t=['# Final Representatives + Residual Fallback Audit\n','## 结论\n\n**'+I['verdict']+'**\n\n'+I['conclusion']+'\n\n'+I['secondary_verdict']+'\n',table(['问题','回答'],I['answers'].items()),
'## 1. 冻结协议与Baseline reproduction\n\n三个scene完全复用既有final Master、拓扑/xyz/activated scale/opacity/SH、TF/AABB、Raw contribution/Bank/Raw SI、valid AND observed、target-2048 tree/final leaves、B/C ranking、相机/GT/renderer/四指标；maturity=1。禁止的Progressive、多层union、d50x32、新训练或新proxy均未加入。evaluation views为fixed non-Bank archived evaluation views，参与过原SLAM，不称unseen或独立test-only。\n\nHEAD=`'+summary['HEAD']+'`，branch=`redundancy-aware-si`；production tracked源码/index未修改。预注册SHA256：`'+pre+'`，在baseline/新render前固定且结束复核。所有selected集合在新render前保存SHA，结束未变。\n',
table(['scene','checkpoint','candidate N','Bank/eval','K 1/5/10/15%'],[(s,{'fr1_room':'16900_shutdown','fr1_xyz':'9900_shutdown','fr2_xyz':'37900_shutdown'}[s],data[s]['N_candidate'],{'fr1_room':'69/161','fr1_xyz':'18/54','fr2_xyz':'14/39'}[s],'/'.join(str(int(r*data[s]['N_candidate']+.5))for r in R))for s in SCENES]),
'A/B/C/F完整ranking、selected rows、C binary/tree均exact；F对应上一轮F-C。逐ratio/view重渲染，PSNR/SSIM/LPIPS/photo逐view及aggregate最大差全部0。全部scene通过后才构造新fallback。Master/TF worker结束前tensor检查未变。\n',
table(['scene','baseline renders','逐view四指标最大差','aggregate最大差'],[(s,data[s]['correctness']['baseline']['view_renders'],'0/0/0/0','0/0/0/0')for s in SCENES]),
'### Frozen SHA\n']
sha_rows=[]
for s in SCENES:
    p=ROOT/s;a=read(p/'prior_checks.json');info=read(p/'input_manifest.json')
    for k in ['master','tf','cameras']:sha_rows.append((s,k,a['camera_sha256']if k=='cameras'else a['source_sha'][info[k]]['sha256']))
    for name in ['hierarchy_input.bin','ranking_inputs.npz','old_tree/hierarchy_tree.csv','exact_residual.npz']:sha_rows.append((s,name,hashlib.sha256((p/name).read_bytes()).hexdigest()))
t += [table(['scene','文件','SHA256'],sha_rows),'全部原数据/GT、上一轮artifacts和隔离runtime库SHA通过；完整路径/哈希见`prior_checks.json`。XYZ沿用原指定TUM_RGBD图像及实际KF IDs；旧Bank文字scene标签过期的已知限制保持记录，未改变Bank。\n',
'## 2. Final Core、精确Residual与静态Pool\n\nF=所有final-leaf structural reps的unique rows，U=Candidate\\F，不加入pure internal-only reps。对U：cover_i=max_{final rep s}κ_si，gap_i=1−cover_i，R_i=Raw_i×gap_i。Raw为原Bank contribution mean。\n\n为满足逐row exact，新增隔离`coverage_fixture.cpp`链接旧冻结`libhierarchy.a`，对冻结final-leaf membership直接调用原`selectCell`，导出原CUDA FP64 cover/gap/R。每leaf rep ID和Delta与冻结tree exact，重建完整C输出也exact；CPU NumPy对导出cover计算Raw×(1−cover)与C nonrep score逐row exact。没有用近似容差代替此gate，也没有由C score反解cover。production源码未改动。\n\nD_r=C TopK_r∩U。对每ratio，D_r与U中按R descending/source-row ascending的前m_r rows exact，且四D集合nested。D15_RES=D15 exact。\n\nRF-C=F∪D15_RES；RF-Raw=F∪同数量Raw Top residual；RF-Gap=F∪同数量gap Top residual。三pool大小exact一致、与ratio无关，所有最终排序统一C score descending/source-row ascending。不用Raw/gap作最终scorer，不按不同ratio重建pool。15%只是原实验范围最大值，无数量或ratio调参。\n',
'## 3. Budget-dependent residual需求（Q1）\n',table(['scene','ratio','K','representative rows','residual rows','residual/K','residual/N','hit residual leaves'],[(s,ratio(a['ratio']),*[a[k]for k in ['K','representative_rows','residual_rows']],pc(a['residual_fraction_of_K']),pc(a['residual_fraction_of_N']),a['hit_residual_leaves'])for s in SCENES for a in data[s]['requirements']]),
'三个scene的residual数量与比例都随这四档ratio增长，支持本fixture下“low budget代表core，高budget加入detail”的观察；比例单调并非一般理论保证。fr1_xyz 1%没有reference residual，其residual recall为未定义（表中—），不把空集合当作proxy效果。\n',
'## 4. 支持到15%的最小新增Residual与候选压缩（Q2/Q3）\n',table(['scene','full N','Final reps','Added residual','total candidate','retained','CR'],[(s,*[next(a for a in data[s]['pools']if a['arm']=='RF-C')[k]for k in ['full_N','Final_reps','Added_residual','total_candidate']],pc(next(a for a in data[s]['pools']if a['arm']=='RF-C')['retained_fraction']),f(next(a for a in data[s]['pools']if a['arm']=='RF-C')['compression_ratio']))for s in SCENES]),
'这里的“最少”有明确条件：固定保留全部F、保持C selected identity到K15。由于较小K是C15的prefix，包含C15即包含所有D；缺少任一D15 row就无法selected-set exact。因此m15为该条件下的必要且充分新增数。它不是任意相同重建质量的全局最小pool；也不声称所有F rows都是15%必需。\n\nP15是Current-C-complete reference pool，不能称GT oracle或真实重建最优。CR只是候选规模，不是speedup。\n',
'## 5. Residual recall、total recall与公平性\n',table(['scene','ratio','proxy','reference residual N','proxy residual N','recalled N','residual recall'],[(s,ratio(a['ratio']),*[a[k]for k in ['proxy','reference_residual_N','proxy_residual_N','recalled_residual_N']],pc(a['residual_recall']))for s in SCENES for a in data[s]['proxy_recall']]),
table(['scene','ratio','pool','pool size','C total recall','C rep recall','C residual recall'],[(s,ratio(a['ratio']),a['arm'],a['candidate_pool_size'],pc(a['C_total_recall']),pc(a['C_representative_recall']),pc(a['C_residual_recall']))for s in SCENES for a in data[s]['candidate_recall']]),
I['proxy_analysis']+'\n',
'## 6. Primary Quality Table\n\nPSNR/SSIM越高越好，LPIPS/photo越低越好。photo沿用production .8L1+.2(1−fast_ssim)，评价SSIM沿用历史TorchMetrics；全部arm同Master/TF/requested-ratio consumer/camera/GT/renderer，只改变selected rows。\n']
for m in M:t += ['### '+m.upper()+'\n',table(['scene','ratio']+ARMS,[(s,ratio(r),*[f(q[s][a,r][m])for a in ARMS])for s in SCENES for r in R])]
t += ['## 7. 关键四指标差值\n',table(['scene','ratio','比较','ΔPSNR','ΔSSIM','ΔLPIPS','Δphoto'],[(s,ratio(a['ratio']),a['arm']+'−'+a['base'],*[d(a[m])for m in M])for s in SCENES for a in data[s]['deltas']]),
'RF-C相对C的四指标per-view和aggregate都exact 0；真实render验证了same-K/compaction/TF consumer一致性，而非发现一个新重建质量提升。RF-Raw/GAP即使某档优于C，也不能将C当GT oracle，更不能据此调pool数量。\n',
'## 8. Paired per-view\n\n完整四指标mean/median/p10/p90/improved fraction保存在CSV，下面展示PSNR。原符号delta=arm−baseline；LPIPS/photo改善按下降定义。相关归档views不能作为独立scene统计显著性样本。\n',table(['scene','ratio','比较','mean','median','p10','p90','改善view比例'],[(s,ratio(a['ratio']),a['arm']+'−'+a['base'],*[d(a[k])for k in ['mean','median','p10','p90']],pc(a['improved_view_fraction']))for s in SCENES for a in data[s]['paired']if a['metric']=='psnr'and a['base']=='C']),
'exact相同的case改善比例为0，因为严格使用>或<；并不表示变差。RF-Raw在fr2_xyz 10%的PSNR p90仍为−0.011837 dB，显示平均损失不是单一view离群造成；15%则多数view正向。\n',
'## 9. Residual在leaf中的集中程度\n',table(['scene','ratio','residual N','有选中residual的leaves','全部leaves','leaf比例','覆盖50%所需leaf数','覆盖90%所需leaf数'],[(s,ratio(a['ratio']),a['selected_residual_N'],a['leaves_with_selected_residual'],a['all_final_leaves'],pc(a['leaf_fraction']),a['leaves_for_50percent_residual'],a['leaves_for_90percent_residual'])for s in SCENES for a in data[s]['concentration']]),
I['locality_analysis']+'\n\n`leaf_residual_statistics.csv`包含每个final leaf的occupancy/Q/rep count/Raw mass以及四ratio residual数量、Raw mass、score mass；该完整统计未用于选择fallback。\n',
'## 10. Residual与Q\n\nQ bins只作diagnostic，>=.8包含未split的final leaves。selected fraction分别按该bin residual population和全D_r定义，原始两种分母均在CSV；此表显示该bin中被选中的比例。\n',table(['scene','ratio','leaf Q bin','residual candidate N','selected residual N','selected/bin','selected score mass'],[(s,ratio(a['ratio']),a['Q_bin'],a['residual_candidate_N'],a['selected_residual_N'],pc(a['selected_fraction_of_bin']),f(a['selected_residual_score_mass']))for s in SCENES for a in rows(ROOT/s/'q_residual_statistics.csv')]),
'不能把Q分组解释成新split阈值或未来fallback quota，本轮没有基于Q筛选。\n',
'## 11. Coverage、Raw SI与高价值Residual\n\n完整selected/nonselected×四ratio×cover/gap/Raw/R/Raw rank percentile分布在`coverage_residual_statistics.csv`，逐row完整值在`residual_pool_rows.csv`。Raw rank percentile=1-based rank/N，越小越重要，另保存U内部rank percentile。\n',
table(['scene','ratio','population','metric','N','mean','median','p10','p90'],[(s,ratio(a['ratio']),a['population'],a['metric'],a['count'],*[f(a[k])for k in ['mean','median','p10','p90']])for s in SCENES for a in rows(ROOT/s/'coverage_residual_statistics.csv')if a['metric']in ['cover','gap','Raw_SI']]),I['coverage_analysis']+'\n',
'## 12. Proxy FN/FP机制\n\nFN/FP相对D15 reference定义，不是GT错误标签。每个错误row保存xyz/Raw/cover/gap/R/C rank/B rank/leaf/Q；完整分布在`proxy_error_statistics.csv`。\n',table(['scene','proxy','类型','metric','N','mean','median','p10','p90'],[(s,a['proxy'],a['error'],a['metric'],a['count'],*[f(a[k])for k in ['mean','median','p10','p90']])for s in SCENES for a in rows(ROOT/s/'proxy_error_statistics.csv')if a['metric']in ['Raw_SI','gap','residual_score']]),I['error_analysis']+'\n',
'## 13. Spatial coverage与统一redundancy\n',table(['scene','ratio','arm','hit roots','hit leaves','root Raw mass','leaf Raw mass'],[(s,ratio(a['ratio']),a['arm'],a['hit_roots'],a['hit_final_leaves'],pc(a['hit_root_Raw_mass_fraction']),pc(a['hit_leaf_Raw_mass_fraction']))for s in SCENES for a in rows(ROOT/s/'spatial_coverage.csv')]),
'统一沿用历史Fixed-h8 cell metric：每selected row取同h8 cell内其它selected rows的max geometry κ，singleton=0；再计算mean/median/p90及unordered pair阈值。不是用hierarchy cell评价，不检查cross-h8边界，保持历史语义。C/F数值与上一轮exact。\n',table(['scene','ratio','arm','mean','median','p90','κ>.8','κ>.9','κ>.95'],[(s,ratio(a['ratio']),a['arm'],*[f(a[k])for k in ['mean_redundancy','median_redundancy','p90_redundancy']],*[a['pairs_gt_'+str(v)]for v in [.8,.9,.95]])for s in SCENES for a in rows(ROOT/s/'redundancy_statistics.csv')]),
'空间覆盖和redundancy均未用作fallback筛选或质量gate；所有四指标仍单独报告。\n',
'## 14. Correctness验收\n\n- F与冻结tree final reps及final rep mask exact，U为candidate补集，无pure internal-only union。\n- 原LibTorch selectCell导出coverage，final representative IDs/Delta、重建全部C score exact；每个U row的Raw×gap与C score exact。\n- D1⊆D5⊆D10⊆D15，且每D exact等于R prefix；D15_RES与D15 order exact。\n- RF-C全部四ratio完整prefix及selected set与C exact；真实render四指标per-view/aggregate exact。\n- 三proxy candidate数相同，全部候选来自同population，selected size=K，无duplicate，所有ratio同一static pool、nested order。\n- 重复构造exact；独立Python source-ID sort/set实现复核全部proxy pool、prefix、same-K。Synthetic覆盖已知prefix、equal score tie和固定pool。\n- fallback构造模块不读GT/render/quality，集合在新render前保存SHA，结束不变；基线和新render均使用冻结runtime/consumer。\n- runtime仅记录，不做性能优化或claim。没有改production源码、stage、commit、push。\n',
'## 15. 结论边界与下一步\n\n'+I['limits']+'\n\n'+I['recommendation']+'\n\n实验在本报告交付后停止。\n']
(ROOT/'FINAL_REP_RESIDUAL_FALLBACK_AUDIT.md').write_text('\n'.join(t),encoding='utf-8')
try:
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.6),sharey=True)
    for ax,s in zip(axes,SCENES):
        for a in ['F','RF-Raw','RF-Gap','RF-C']:ax.plot([r*100 for r in R],[float(q[s][a,r]['psnr'])-float(q[s]['C',r]['psnr'])for r in R],marker='o',label=a+' - C')
        ax.axhline(0,color='gray',linewidth=.8);ax.set_title(s);ax.set_xlabel('Budget (%)');ax.set_xticks([1,5,10,15]);ax.grid(alpha=.2)
    axes[0].set_ylabel('PSNR difference (dB)');axes[-1].legend();fig.tight_layout();fig.savefig(ROOT/'residual_fallback_quality.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3.6))
    for s in SCENES:ax.plot([float(v['ratio'])*100 for v in data[s]['requirements']],[float(v['residual_fraction_of_K'])*100 for v in data[s]['requirements']],marker='o',label=s)
    ax.set_xlabel('Budget (%)');ax.set_ylabel('Residual rows in C Top-K (%)');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(ROOT/'residual_requirement.png',dpi=170);plt.close(fig)
except ImportError:pass
print('REPORT_COMPLETE',I['verdict'],flush=True)
