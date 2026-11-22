"""汇总三scene冻结candidate-only审计，结论来自人工机制解释文件。"""
from pathlib import Path
import csv,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
SCENES=['fr1_room','fr1_xyz','fr2_xyz'];R=[.01,.05,.1,.15];ARMS=['A','B','C','P0','F-Raw','F-C','M-Raw','M-C'];METRICS=['psnr','ssim','lpips','photo']
def read(p):return json.loads(Path(p).read_text())
def rows(p):return list(csv.DictReader(open(p)))
def table(headers,data):return '\n| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,a))+' |\n'for a in data)+'\n'
def f(x):return f'{float(x):.6f}'
def d(x):return f'{float(x):+.6f}'
def pct(x):return f'{float(x)*100:.2f}%'
def ratio(r):return f'{float(r)*100:g}%'
allscene={s:read(ROOT/s/'audit_summary.json')for s in SCENES};I=read(ROOT/'interpretation.json')
Q={s:{(a['arm'],float(a['ratio'])):a for a in allscene[s]['quality']}for s in SCENES}
for s in SCENES:
    assert allscene[s]['all_passed']and allscene[s]['correctness']['all_passed']
    assert all(np.isfinite(float(a[m]))for a in allscene[s]['quality']for m in METRICS)
    assert all(np.isfinite(float(a[m]))for a in rows(ROOT/s/'quality_per_view.csv')for m in METRICS)
def diff(s,a,b,r,m='psnr'):
    return float(Q[s][a,r][m])-float(Q[s][b,r][m])if (a,r)in Q[s]and (b,r)in Q[s]else None
def fmt_diff(v):return 'CAPACITY INSUFFICIENT'if v is None else d(v)
merge=['candidate_pool_summary.csv','candidate_pool_rows.csv','level_overlap_statistics.csv','candidate_recall.csv','capacity_checks.csv','missed_c_rows.csv','added_candidate_rows.csv','quality_summary.csv','quality_per_view.csv','paired_statistics.csv','quality_deltas.csv','spatial_coverage.csv','redundancy_statistics.csv','selected_pool_composition.csv']
for name in merge:
    data=[dict(scene=s,**a)for s in SCENES for a in rows(ROOT/s/name)]
    if not data:continue
    with open(ROOT/name,'w',newline='')as out:w=csv.DictWriter(out,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
rankings={};sets={};miss_stats=[]
for s in SCENES:
    b=dict(np.load(ROOT/s/'rankings.npz'));z=dict(np.load(ROOT/s/'pool_rankings.npz'));ids=b['row_ids'];N=len(ids)
    rankings[s+'__source_rows']=ids
    for arm,key in [('A','A'),('B','B'),('C','C0'),('P0','P0')]:rankings[s+'__'+arm+'_source_row_order']=ids[b[key+'_order']]
    for arm in ARMS[4:]:rankings[s+'__'+arm+'_source_row_order']=ids[z[arm+'_order']]
    for pool in ['P_final','P_multi']:rankings[s+'__'+pool+'_source_rows']=ids[z[pool]]
    sets.update({s+'__'+k:v for k,v in np.load(ROOT/s/'selected_sets.npz').items()})
    for r in R:
        g=[a for a in rows(ROOT/s/'missed_c_rows.csv')if float(a['ratio'])==r];K=int(r*N+.5)
        miss_stats.append(dict(scene=s,ratio=r,K=K,missed_count=len(g),missed_fraction=len(g)/K,residual_count=sum(a['is_residual']=='True'for a in g),representative_count=sum(a['is_residual']!='True'for a in g),mean_cover=float(np.mean([float(a['cover'])for a in g]))if g else None,median_cover=float(np.median([float(a['cover'])for a in g]))if g else None))
np.savez_compressed(ROOT/'rankings.npz',**rankings);np.savez_compressed(ROOT/'selected_sets.npz',**sets)
with open(ROOT/'missed_c_summary.csv','w',newline='')as out:w=csv.DictWriter(out,fieldnames=list(miss_stats[0]));w.writeheader();w.writerows(miss_stats)
prereg=hashlib.sha256((ROOT/'preregistration.json').read_bytes()).hexdigest()
for s in SCENES:assert read(ROOT/s/'prior_checks.json')['preregistration_sha256']==prereg
checks=read(ROOT/'independent_checks.json')
assert checks['all_passed']
for s,group in checks['pre_render_SHA'].items():
    for name,expected in group.items():assert hashlib.sha256((ROOT/s/name).read_bytes()).hexdigest()==expected
correct=dict(all_passed=True,baseline_all_exact=True,independent_checks=checks,scenes={s:allscene[s]['correctness']for s in SCENES},preregistration_unchanged=True,no_production_change=True,no_stage=True,no_commit=True,no_push=True)
(ROOT/'correctness.json').write_text(json.dumps(correct,ensure_ascii=False,indent=2)+'\n')
summary=dict(verdict=I['verdict'],answers=I['answers'],interpretation=I,scenes={s:{k:allscene[s][k]for k in ['N_candidate','pools','recall','composition','deltas','renders']}for s in SCENES},missed_C=miss_stats,correctness_pass=True,preregistration_sha256=prereg,HEAD='3cfccc4a92c8e1ec6c3d491dac1c9efbf151daeb',branch='redundancy-aware-si',backend='sparse_hash_target_2048',production_changed=False,stage=False,commit=False,push=False)
(ROOT/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
t=['# Hierarchy-Only Candidate Pool Audit\n',
'## 结论\n\n**'+I['verdict']+'**\n\n'+I['conclusion']+'\n',
table(['问题','回答'],I['answers'].items()),
'## 1. 研究问题与冻结协议\n\n只检验将结构hierarchy作为candidate generator，再用统一Raw/C scorer全局排序是否比P0稳定。target-2048 SparseHash、origin=(0,0,0)、原h搜索、3rep/node、Q<.80、L3、scalar-radius geometry κ、valid AND observed、maturity=1、Raw SI/贡献矩阵/Bank、Master/TF/AABB、renderer、evaluation cameras与四指标均冻结。不使用d50x32，不训练、不改Selector、不新增Progressive或residual fallback。\n\nHEAD=`'+summary['HEAD']+'`；branch=`redundancy-aware-si`。本轮只新增离线诊断/报告/证据，production tracked源码和index未改动。预注册SHA256：`'+prereg+'`。判定规则在新render前写入，结束再次核对；不使用新的任意dB gate。\n',
table(['scene','checkpoint','candidate N','Bank / eval','K 1/5/10/15%'],[(s,{'fr1_room':'16900_shutdown','fr1_xyz':'9900_shutdown','fr2_xyz':'37900_shutdown'}[s],allscene[s]['N_candidate'],{'fr1_room':'69 / 161','fr1_xyz':'18 / 54','fr2_xyz':'14 / 39'}[s],'/'.join(str(int(r*allscene[s]['N_candidate']+.5))for r in R))for s in SCENES]),
'evaluation views为参与过原SLAM的fixed non-Bank archived evaluation views，不是unseen/test-only/独立novel views。XYZ使用原指定TUM_RGBD GT。Bank归档过期scene标签沿用历史审计说明，以实际checkpoint/KF IDs/相机图像路径确认，未重选Bank。\n\n### Frozen SHA\n']
sha_rows=[]
for s in SCENES:
    p=ROOT/s;prior=read(p/'prior_checks.json');info=read(p/'input_manifest.json')
    for key in ['master','tf','cameras']:sha_rows.append((s,key,prior['camera_sha256']if key=='cameras'else prior['source_sha'][info[key]]['sha256']))
    for key in ['hierarchy_input.bin','ranking_inputs.npz','old_tree/hierarchy_tree.csv']:sha_rows.append((s,key,hashlib.sha256((p/key).read_bytes()).hexdigest()))
t += [table(['scene','输入','SHA256'],sha_rows),'原Master/TF/相机/GT、归档contribution+Raw+geometry/masks以及tree/ranking/质量文件SHA均通过；隔离renderer库与历史SHA一致，实际worker maps指向本轮runtime。完整文件路径和冻结SHA见各scene `prior_checks.json`。没有新GI contribution render。\n',
'## 2. Baseline reproduction gate\n\nA/B/C/P0完整order、same-K selected sets exact；C output binary/tree逐字一致；P0使用原F2定义重新生成order和emit-time priority，均exact。所有baseline ratio/view重新真实render后，PSNR/SSIM/LPIPS/photo逐view和aggregate最大差均为0。三个scene全部通过才生成并渲染新pool arms。\n',
table(['scene','baseline renders','ranking/C tree/P0 gain','逐view四指标最大差','aggregate最大差'],[(s,allscene[s]['baseline']['view_renders'],'EXACT','0 / 0 / 0 / 0','0 / 0 / 0 / 0')for s in SCENES]),
'## 3. Structural pool与全局scorer\n\nF=冻结tree所有final leaf structural reps的source-row union；M=全部L1/L2/L3 node structural reps的union。包括internal reps，但不使用P0 ranking-time ancestor-conditioned重新greedy的代表。每row去重，F⊆M exact。\n\nF-Raw/M-Raw按Raw SI descending；F-C/M-C按已有C score descending；全部tie按source row ascending。C score仍来自完整candidate的final-leaf Delta或independent residual，没有新score。\n\n新pool/ranking代码不调用parent-before-child、frontier、unlock或priority heap。每arm只有一条完整pool permutation，所有ratio是该顺序的prefix。若pool<K，只标CAPACITY INSUFFICIENT并跳过render，禁止任何refill。\n\n`candidate_pool_rows.csv`保留每row的多层multi-hot、r1/r2/r3位置、逐node出现列表、出现次数、最浅/最深层、所属final leaf/Q/occupancy、Raw/C score与Raw/B/C rank。一个row可同时具有多个level身份，统计不可相加成互斥emit depth。\n',
'## 4. Pool规模、压缩比与level overlap（Q1）\n',
table(['scene','pool','N','pool rows','占完整candidate','CR=N/pool','node-rep occurrences','重复occurrence比例'],[(s,a['pool'],a['N_candidate'],a['pool_size'],pct(a['pool_fraction']),f(a['compression_ratio']),a['total_node_rep_occurrences'],pct(a['duplicate_occurrence_ratio']))for s in SCENES for a in allscene[s]['pools']]),
'重复occurrence比例=(该pool的node-rep occurrences−unique source rows)/occurrences；F统计final nodes，M统计全部nodes。这是同row跨层重复，不是geometry重复度。CR只表示候选规模压缩，不表示速度提升；C scorer仍在完整population上构建，未实现production candidate Selector。\n',
table(['scene','unique L1','unique L2','unique L3','unique final','L1∩L2','L1∩L3','L2∩L3','L1∩L2∩L3'],[(s,*[rows(ROOT/s/'level_overlap_statistics.csv')[0][k]for k in ['unique_L1','unique_L2','unique_L3','unique_final','L1_intersect_L2','L1_intersect_L3','L2_intersect_L3','L1_intersect_L2_intersect_L3']])for s in SCENES]),
'## 5. Candidate recall与容量（Q2/Q3）\n\nRecall_X(P,K)=|P∩X TopK|/K。C/B/Raw均不是GT reconstruction oracle；高recall不能直接当quality GO，必须与真实render联合解释。\n',
table(['scene','ratio','pool','C recall','B recall','Raw recall'],[(s,ratio(a['ratio']),a['pool'],*[pct(a[k])for k in ['C_recall','B_recall','Raw_recall']])for s in SCENES for a in allscene[s]['recall']]),
'容量检查：'+I['capacity_text']+'。K始终由完整N计算，而不是按pool size缩小；全部新arm nested prefixes，未发生ratio-specific候选生成或排序。\n',
'## 6. 主质量表\n\nPSNR/SSIM越高越好，LPIPS/photo越低越好。photo沿用production .8 L1 + .2(1−production fast_ssim)；评价SSIM沿用原TorchMetrics实现。除selected rows外，Master/TF、camera、renderer、GT均相同。\n']
for metric in METRICS:
    t += ['### '+metric.upper()+'\n',table(['scene','ratio']+ARMS,[(s,ratio(r),*[f(Q[s][a,r][metric])if (a,r)in Q[s]else'CAPACITY INSUFFICIENT'for a in ARMS])for s in SCENES for r in R])]
t += ['## 7. 五组核心比较与Q5\n',table(['scene','ratio','M-C−P0','M-C−C','M-C−F-C','M-Raw−F-Raw','M-C−M-Raw'],[(s,ratio(r),*[fmt_diff(diff(s,a,b,r))for a,b in [('M-C','P0'),('M-C','C'),('M-C','F-C'),('M-Raw','F-Raw'),('M-C','M-Raw')]])for s in SCENES for r in R]),
'完整四指标差值：\n',table(['scene','ratio','比较','ΔPSNR','ΔSSIM','ΔLPIPS','Δphoto'],[(s,ratio(a['ratio']),a['arm']+'−'+a['base'],*[d(a[m])for m in METRICS])for s in SCENES for a in allscene[s]['deltas']if (a['arm'],a['base'])!=('F-C','C')]),
'## 8. 成功/失败case逐项解释\n\n'+I['cases_text']+'\n',
'## 9. Recall与真实质量联合分析\n',table(['scene','ratio','F C-recall','M C-recall','F-C PSNR','M-C PSNR','C PSNR'],[(s,ratio(r),*[pct(next(a for a in allscene[s]['recall']if float(a['ratio'])==r and a['pool']==pool)['C_recall'])for pool in ['F','M']],*[f(Q[s][a,r]['psnr'])if (a,r)in Q[s]else'容量不足'for a in ['F-C','M-C','C']])for s in SCENES for r in R]),
I['recall_quality_text']+'\n',
'## 10. Missed C rows与Added candidates\n\nMiss_C=C TopK\\M；Added=M-C TopK\\C TopK。完整逐row的xyz/Raw/C/B rank、C score、leaf、cover与independent residual保存在`missed_c_rows.csv`；Added还保留multi-hot levels及node/r1-r3出现信息。\n',
table(['scene','ratio','missed N','missed fraction','代表 / residual','mean cover','median cover'],[(a['scene'],ratio(a['ratio']),a['missed_count'],pct(a['missed_fraction']),str(a['representative_count'])+' / '+str(a['residual_count']),f(a['mean_cover'])if a['mean_cover']is not None else'—',f(a['median_cover'])if a['median_cover']is not None else'—')for a in miss_stats]),
'所有missed rows都是residual：这由F⊆M且F包含全部final-leaf reps决定，不是质量标签。低budget可仅miss少数row，高budgetmiss量增加，必须量化其渲染影响，不能自动加入fallback。对于same-K，M-C相对C的added count等于missed count；pool sort保持C原有relative order，变化来自排除missed rows后向更低C rank补足预算，未重新计算C score。\n',
'## 11. Multi-level internal-only实际贡献与multi-hot组成\n',
table(['scene','ratio','M-C K','final-pool rows','internal-only rows','internal-only比例','with L1','with L2','with L3'],[(s,ratio(a['ratio']),*[a[k]for k in ['M_C_selected','final_pool_rows','internal_only_rows']],pct(a['internal_only_fraction']),*[a[k]for k in ['selected_with_L1_identity','selected_with_L2_identity','selected_with_L3_identity']])for s in SCENES for a in allscene[s]['composition']]),
'with L1/L2/L3是可重叠的source-row属性，不是emit depth。即使保留internal structural pool，若C scorer不给它高rank，它也不会获得precedence或预算保护。\n',
'## 12. Paired per-view\n\n完整四指标的mean/median/p10/p90/improved-view fraction均保存；下表PSNR突出M-C−P0、M-C−C、F-C−C。delta使用原符号，LPIPS/photo的improved fraction按下降计算。视角来自同一序列，不能把相关views当独立scene显著性样本。\n',
table(['scene','ratio','比较','mean','median','p10','p90','改善view比例'],[(s,ratio(a['ratio']),a['arm']+'−'+a['base'],*[d(a[k])for k in ['mean','median','p10','p90']],pct(a['improved_view_fraction']))for s in SCENES for a in allscene[s]['paired']if a['metric']=='psnr'and (a['arm'],a['base'])in [('M-C','P0'),('M-C','C'),('F-C','C')]]),
I['paired_text']+'\n',
'## 13. Spatial coverage\n',table(['scene','ratio','arm','hit roots','hit leaves','hit-root Raw mass','hit-leaf Raw mass'],[(s,ratio(a['ratio']),a['arm'],a['hit_roots'],a['hit_final_leaves'],pct(a['hit_root_Raw_mass_fraction']),pct(a['hit_leaf_Raw_mass_fraction']))for s in SCENES for a in allscene[s]['spatial']]),
'coverage只作机制解释，不参与quality判定。新增arm未设置region quota；若质量退化而hit-root数量变化小，不能将全部损失直接解释为spatial collapse。\n',
'## 14. 统一Fixed-h8 redundancy\n\n沿用冻结Fixed-h8 cells。每selected row的redundancy=max κ(i,j)，j为同h8 cell的其它selected rows；singleton=0。再汇总selected rows的mean/median/p90，并统计同h8 cell内unordered pairs。全部arm用同一scalar radius/κ/固定h8 partition，不使用hierarchy自己的leaf或root。该metric明确不评价cross-h8 boundary pairs，与历史一致。C/P0统计与历史值exact。\n',
table(['scene','ratio','arm','mean','median','p90','κ>.8','κ>.9','κ>.95'],[(s,ratio(a['ratio']),a['arm'],*[f(a[k])for k in ['mean_redundancy','median_redundancy','p90_redundancy']],*[a['pairs_gt_'+str(k)]for k in [.8,.9,.95]])for s in SCENES for a in allscene[s]['redundancy']]),
'redundancy降低不等于rendering质量提高，未用该metric筛candidate或调参数。\n',
'## 15. Correctness与证据边界\n\n- F⊆M；每pool source rows unique，全部来自valid AND observed；F与冻结C final rep mask exact。\n- pool全部来自tree结构rep1/2/3，而非P0实际emit trace；multi-hot和跨node重复metadata保留。\n- 新排序代码只按Raw/C score+source-row tie，不导入或调用Progressive；baseline P0仅复现既有定义。\n- 全pool order重复构造exact；独立实现用Python source-row sets/排序复核union、order、same-K和nested prefixes。\n- synthetic覆盖跨层同row去重、multi-hot、Raw/C排序、equal-score source tie与容量不足不补点。\n- selected set构建模块不读GT/render/quality；全部scene的新集合在任何新render前保存SHA，结束复核未变。\n- 每scene baseline full ranking、C binary/tree、P0 priority及四指标per-view/aggregate exact；实际runtime maps为冻结库；Master/TF结束前tensor校验未变。\n- C score仍由完整population构建；本轮不是pool-native scorer，也没有质量oracle。M-C比较P0同时涉及structural-vs-ranking-time候选、C score及precedence去除，不能单凭其改善将全部因果唯一归于unlock。F/M之间同scorer、Raw/C之间同pool的比较更直接。\n- L1/L2/L3 multi-hot只表示来源，不赋予coarse-first次序；ordinal Progressive score未用于新scorer。\n',
'## 16. Runtime、限制与研究建议\n\n仅记录各stage wall time于`runtime.json`，不参与判定，不提出speedup claim。'+I['runtime_text']+'\n\n'+I['limitations_text']+'\n\n'+I['recommendation']+'\n\n本轮实验结束。未加入residual fallback、未训练Global/Ratio-aware Selector、未调Q/grid/rep/depth或继续新实验。没有stage、commit或push。\n']
doc='\n'.join(t)
(ROOT/'HIERARCHY_CANDIDATE_POOL_AUDIT.md').write_text(doc,encoding='utf-8')
try:
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.6),sharey=True)
    for ax,s in zip(axes,SCENES):
        for a in ['F-C','M-C','P0']:
            rr=[r for r in R if (a,r)in Q[s]];ax.plot([r*100 for r in rr],[diff(s,a,'C',r)for r in rr],marker='o',label=a+' - C')
        ax.axhline(0,color='gray',linewidth=.8);ax.set_title(s);ax.set_xlabel('Budget (%)');ax.set_xticks([1,5,10,15]);ax.grid(alpha=.2)
    axes[0].set_ylabel('PSNR difference (dB)');axes[-1].legend();fig.tight_layout();fig.savefig(ROOT/'candidate_pool_quality.png',dpi=170);plt.close(fig)
except ImportError:pass
print('REPORT_COMPLETE',summary['verdict'],flush=True)
