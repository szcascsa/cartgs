"""三个冻结场景的中文机制报告；不修改ranking或生成新实验。"""
from pathlib import Path
import csv,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
SCENES=['fr1_room','fr1_xyz','fr2_xyz'];R=[.01,.05,.1,.15];ARMS=['A','B','C','P0','S','L','G'];METRICS=['psnr','ssim','lpips','photo']
def load(p):return json.loads(Path(p).read_text())
def rows(p):return list(csv.DictReader(open(p)))
def table(headers,rs):return '\n| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,a))+' |\n'for a in rs)+'\n'
def f(x):return f'{float(x):.6f}'
def d(x):return f'{float(x):+.6f}'
def pct(x):return f'{float(x)*100:.2f}%'
def ratio(r):return f'{r*100:g}%'
S={s:load(ROOT/s/'audit_summary.json')for s in SCENES}
assert load(ROOT/'independent_checks.json')['all_passed']and load(ROOT/'setup_checks.json')['all_passed']
for s in SCENES:assert S[s]['baseline']['all_passed']and S[s]['construction']['all_passed']
Q={s:{(a['arm'],float(a['ratio'])):a for a in S[s]['quality']}for s in SCENES}
for s in SCENES:
    assert all(np.isfinite(float(a[m]))for a in S[s]['quality']for m in METRICS)
    assert all(np.isfinite(float(a[m]))for a in rows(ROOT/s/'quality_per_view.csv')for m in METRICS)
def delta(s,a,b,r,m='psnr'):return float(Q[s][a,r][m])-float(Q[s][b,r][m])
# 合并所有用户要求的CSV；逐scene仍保留原文件。
names=['internal_rep_inventory.csv','internal_rep_survival.csv','internal_rep_rank_statistics.csv','internal_rep_q_statistics.csv','internal_rep_r123_statistics.csv','internal_symmetric_difference.csv','local_replacement_trace.csv','quality_summary.csv','quality_per_view.csv','paired_statistics.csv','spatial_coverage.csv','root_budget_distribution.csv']
for name in names:
    merged=[dict(scene=s,**a)for s in SCENES for a in rows(ROOT/s/name)]
    with open(ROOT/name,'w',newline='')as out:
        w=csv.DictWriter(out,fieldnames=list(merged[0]));w.writeheader();w.writerows(merged)
selected={s+'__'+k:v for s in SCENES for k,v in np.load(ROOT/s/'diagnostic_selected_sets.npz').items()};np.savez_compressed(ROOT/'diagnostic_selected_sets.npz',**selected)
prereg_sha=hashlib.sha256((ROOT/'preregistration.json').read_bytes()).hexdigest()
for s in SCENES:assert load(ROOT/s/'prior_checks.json')['preregistration_sha256']==prereg_sha
case_deltas=[dict(scene=s,ratio=r,**{a+'_minus_P0':delta(s,a,'P0',r)for a in ['S','L','G']},P0_minus_C=delta(s,'P0','C',r),L_minus_C=delta(s,'L','C',r))for s in SCENES for r in R]
answers=dict(Q1='见按scene/ratio/depth的survival表',Q2='MIXED',Q3='NO',Q4='MIXED',Q5='BOTH：identity有局部证据，remaining region/unlock/ordering未被排除；尚不能确定主因',Q6='KEEP C')
result=dict(hypothesis_verdict='PARTIAL',answers=answers,interpretation='定向同subtree替换恢复部分失败case，但损害两个成功1% case；全删internal在12/12 case退化。不能认定internal reps是Progressive跨scene不稳定的主要原因。',case_deltas=case_deltas,baseline_all_exact=True,independent_replay_exact=True,L_root_budget_exact=True,preregistration_sha256=prereg_sha,no_new_dB_gate=True,render_counts={s:S[s]['render_counts']for s in SCENES},source_HEAD='3cfccc4a92c8e1ec6c3d491dac1c9efbf151daeb',backend='sparse_hash_target_2048',production_source_changed=False,commit=False,stage=False,push=False)
(ROOT/'audit_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
t=['# Progressive Internal-Node Representative Utility Audit\n',
'## 结论\n\n**INTERNAL REP HYPOTHESIS：PARTIAL。建议 KEEP C。**\n\ninternal representative identity确实影响部分失败case，但现有证据不足以将其认定为Progressive跨scene不稳定的主要原因。S全删internal在12/12 case降低PSNR；L保留root预算、在同subtree内替换后，fr1_xyz 5/10%和fr2_xyz 1%恢复明显，fr2_xyz 5/10%仅小幅恢复；两个原本成功的1% case反而下降。不能把“C低rank”直接解释为真实渲染utility低。\n',
table(['问题','回答'],[(k,v)for k,v in answers.items()]),
'Q3的NO指Skip Internal没有恢复任何失败case；Q4的MIXED表示L在五个重点失败case均正向，但幅度、paired-view证据不同，且成功case出现反向证据。BOTH不是证明两者贡献相等，而是identity有局部干预证据、region/unlock/后续ordering仍无法排除。\n',
'## 1. 研究假设、冻结协议与边界\n\n本轮仅诊断P0实际emit的internal代表是否不适合最终渲染。不调用d50x32；target-2048、原h搜索、origin=(0,0,0)、3rep/node、Q<0.80 split、L3、scalar-radius κ、candidate=valid AND observed、maturity=1、Raw贡献/Bank、Master/TF、renderer与四指标实现均冻结。没有新训练、GI贡献重渲染、Bank更新或参数搜索。HEAD=`'+result['source_HEAD']+'`，branch=`redundancy-aware-si`；production tracked源码和index未改动。\n\n预注册SHA256：`'+prereg_sha+'`。规则在baseline和新diagnostic render之前写入，结束时与三个scene保存的SHA再次核对；没有新增dB阈值。实验目录保留20261019批次名，本轮交付日期为2026-10-20。\n',
table(['scene','checkpoint','candidate N','Bank/eval','K 1/5/10/15%'],[(s,{'fr1_room':'16900_shutdown','fr1_xyz':'9900_shutdown','fr2_xyz':'37900_shutdown'}[s],S[s]['N_candidate'],{'fr1_room':'69/161','fr1_xyz':'18/54','fr2_xyz':'14/39'}[s],'/'.join(str(int(S[s]['N_candidate']*r+.5))for r in R))for s in SCENES]),
'评价视角为参与过原SLAM的fixed non-Bank archived evaluation views，不称unseen或独立novel-view测试。XYZ使用指定TUM_RGBD数据；旧Bank归档的scene文本标签过期，沿用已冻结的实际keyframe IDs和相机/GT路径。所有arm使用同一requested-ratio TF consumer。\n\n### 冻结SHA\n']
sha_rows=[]
for s in SCENES:
    prior=load(ROOT/s/'prior_checks.json');info=load(ROOT/s/'input_manifest.json')
    for k in ['master','tf','cameras']:
        sha_rows.append((s,k,prior['camera_sha256']if k=='cameras'else prior['source_sha'][info[k]]['sha256']))
    sha_rows.append((s,'hierarchy输入',hashlib.sha256((ROOT/s/'hierarchy_input.bin').read_bytes()).hexdigest()))
t += [table(['scene','输入','SHA256'],sha_rows),'全部consumed原始数据/GT和历史ranking/tree/质量artifact SHA通过；15个隔离runtime库SHA与历史一致，实际worker maps指向本轮runtime。完整路径/哈希见各scene `prior_checks.json`、`setup_checks.json`及`artifact_manifest.json`。\n',
'## 2. Baseline reproduction gate\n\n三个scene均重建A/B/C/P0完整ranking；C binary和tree逐字一致；P0完整order与emit-time priority exact一致；same-K与selected rows exact。重新真实渲染所有baseline ratio/view，PSNR/SSIM/LPIPS/photo的逐view及aggregate最大差全部为0。全部通过后才渲染S/L/G。Master/TF tensor在worker结束前再次验证未变。\n',
table(['scene','baseline真实render次数','逐view最大差','aggregate最大差','ranking/tree'],[(s,S[s]['baseline']['view_renders'],'0 / 0 / 0 / 0','0 / 0 / 0 / 0','EXACT')for s in SCENES]),
'## 3. Arms与identity语义\n\n- A：Raw SI descending，B：原Fixed h8 geometry-only marginal，C：final-leaf Delta + independent residual descending。\n- P0：原F2 active-frontier，node最多3个新coarse reps，继承ancestor coverage，排除已emit row，leaf继续independent residual。\n- S：从P0完整顺序稳定过滤全部INTERNAL_REP，其余相对顺序exact保持，再取前K。S是完整non-internal ranking，非完整candidate permutation。\n- L：每ratio从P0 TopK开始，按原P0 rank，仅替换不在C TopK的internal row；从其同一node.members/subtree选择当前未selected、从未用作proxy的最高C rank row；否则留原。允许已经移除且未曾用作proxy的row在后续重新成为proxy，遵循顺序更新selected状态。\n- G：删除P0 TopK中不在C TopK的internal rows，按缺失C global rank refill。它不是GT oracle，也不是数学保证的渲染质量上界。\n\nS按同一完整过滤顺序取nested prefixes；L/G是用户规定的ratio-specific集合，不主张nested。L/G不接Teacher，不构造成production Progressive。\n\n**重要源码事实：P0会在ancestor coverage条件下重新greedy选代表。** inventory中的local Delta/r1-r3是P0实际ranking-time值，不一定等于结构阶段empty-cover provisional reps。`matches_structural_provisional_ID`记录实际row是否属于该node原结构3rep。因此本轮严格检验“split internal node的实际输出”，不能假设每个P0 row与原provisional ID逐字相同。\n',
'## 4. Internal reps最终存活情况（Q1）\n\n下表分母为该budget中P0实际emit的internal reps，分子为其中同budget也进入C的数量。L3没有internal nodes。\n']
sv=[]
for s in SCENES:
    a={(float(z['ratio']),z['group']):z for z in S[s]['survival']}
    for r in R:
        z=a[r,'ALL'];sv.append((s,ratio(r),str(int(r*S[s]['N_candidate']+.5)),str(z['C_TopK_survivors'])+'/'+str(z['internal_count']),pct(z['C_TopK_survival_fraction']),*[str(a[r,'L'+str(l)]['C_TopK_survivors'])+'/'+str(a[r,'L'+str(l)]['internal_count'])for l in [1,2,3]],z['C_final_leaf_rep_count'],z['C_final_residual_count']))
t += [table(['scene','ratio','K','存活/内部总数','存活率','L1','L2','L3','C最终rep','C最终residual'],sv),
'成功的两个1% case存活率分别79.79%和74.00%，失败的fr2_xyz 1%为75.00%，没有清楚分离。高budget失败case的存活率反而可超过89%/94%；不能仅用C overlap解释质量方向。\n',
'## 5. Internal C/Raw/B rank分布（Q2）\n\nrank percentile=1-based rank/N，越小越靠前。只统计P0 TopK的internal reps；完整mean/median/p10/p90与按depth/r1r2r3细分见CSV。\n',
table(['scene','ratio','C mean','C median','C p90','Raw median','B median'],[(s,ratio(float(a['ratio'])),*[pct(a[k])for k in ['C_rank_percentile_mean','C_rank_percentile_median','C_rank_percentile_p90','Raw_rank_percentile_median','B_rank_percentile_median']])for s in SCENES for a in S[s]['rank_statistics']if a['group']=='ALL']),
'Q2=MIXED。fr2_xyz 1%的C median为0.684%，介于两个成功1% case的0.618%/0.789%之间；并非失败internal reps普遍rank明显更低。跨ratio必须考虑budget本身变化，不能直接比较1%与10%的绝对percentile后推断低utility。\n',
'## 6. 主质量表\n\nPSNR/SSIM越高越好；LPIPS/photo越低越好。photo沿用production .8 L1 + .2(1−production fast_ssim)，评价SSIM沿用原TorchMetrics实现。\n']
for m in METRICS:
    t+=['### '+m.upper()+'\n',table(['scene','ratio']+ARMS,[(s,ratio(r))+tuple(f(Q[s][a,r][m])for a in ARMS)for s in SCENES for r in R])]
t += ['## 7. 核心因果差值\n',table(['scene','ratio','P0−C','S−P0','L−P0','G−P0','L−C'],[(s,ratio(r),*[d(delta(s,a,b,r))for a,b in [('P0','C'),('S','P0'),('L','P0'),('G','P0'),('L','C')]])for s in SCENES for r in R]),
'S在12/12 case退化，幅度为−0.724至−2.056 dB。它删除了同时被C选择的大量有效internal rows，不能解释成“所有parent reps都有害”。L只定向替换P0-only internal，构成更接近identity干预的证据。\n\n重点失败case中，L恢复原P0−C损失的约：fr1_xyz 5%=80.49%、10%=43.59%；fr2_xyz 1%=70.29%、5%=4.97%、10%=34.36%。该比例只是对本次PSNR差值的描述，不是线性可加的utility估计。L仍未在这些case超过C。两个成功1% case分别损失0.0450与0.1509 dB，说明C的ranking不是final rendering utility oracle。\n\nG在失败case通常更接近C，尤其fr2_xyz 5%比L恢复更多；这提示同subtree限制下剩余budget/ordering仍有影响，但不能据此证明具体unlock timing规则错误。\n',
'### 四指标差值\n',table(['scene','ratio','比较','ΔPSNR','ΔSSIM','ΔLPIPS','Δphoto'],[(s,ratio(r),a+'−'+b,*[d(delta(s,a,b,r,m))for m in METRICS])for s in SCENES for r in R for a,b in [('S','P0'),('L','P0'),('G','P0'),('L','C')]]),
'## 8. Paired per-view\n\n以下为PSNR；四项metric完整分布均在`paired_statistics.csv`。LPIPS/photo的improved fraction按下降定义，delta仍为arm−baseline的原符号。视角来自同一序列且相关，未将view数量当作独立scene统计显著性。\n',
table(['scene','ratio','比较','mean','median','p10','p90','改善视角比例'],[(s,ratio(float(a['ratio'])),a['arm']+'−P0',*[d(a[k])for k in ['mean','median','p10','p90']],pct(a['improved_view_fraction']))for s in SCENES for a in S[s]['paired']if a['metric']=='psnr'and a['base']=='P0'and a['arm']in ['S','L','G']]),
'fr1_xyz的L 5%/10%改善视角占98.15%/85.19%；fr2_xyz 1%占94.87%，是较一致的identity改善证据。fr2_xyz 5%/10%仅58.97%/61.54%，恢复较弱。两个成功1% case的L改善视角仅35.40%/9.26%，构成明确反向证据。\n',
'## 9. Symmetric difference来源\n',table(['scene','ratio','P0 only','C only','internal only','internal占P0 only','final rep only','residual only'],[(s,ratio(float(a['ratio'])),a['P0_only'],a['C_only'],a['INTERNAL_REP_count'],pct(int(a['INTERNAL_REP_count'])/int(a['P0_only']))if int(a['P0_only'])else'0',a['FINAL_LEAF_REP_count'],a['RESIDUAL_count'])for s in SCENES for a in S[s]['symmetric_difference']]),
'这区分了“internal identity变化”和“全部ranking变化”。P0-only还可包含final reps/residual；仅修internal无法保证恢复整个C ranking。\n',
'## 10. Split-node Q与r1/r2/r3\n\nQ来自冻结结构阶段；实际emit Delta来自P0条件greedy，二者语义不同。Q bins只作诊断，不改变split。额外真实渲染L_Q0..3，每次只替换相应固定Q区间的bad internal，报告的是joint subset effect，不能相加分解成每row贡献。\n']
qrows=[]
for s in SCENES:
    for a in rows(ROOT/s/'internal_rep_q_statistics.csv'):
        qrows.append((s,ratio(float(a['ratio'])),['<.25','.25–.50','.50–.65','.65–.80'][int(a['group'][-1])],a['internal_count'],a['replacements'],pct(a['C_TopK_survival_fraction'])if a['internal_count']!='0'else'—',d(a['L_Qbin_minus_P0_psnr'])))
t += [table(['scene','ratio','Q bin','internal N','替换数','C存活率','仅该bin L−P0 PSNR'],qrows),
'Q越低不必然意味着每个parent代表越差；低Q衡量的是最多3rep的总体覆盖能力不足。例如fr1_xyz 1%仅替换Q=.25–.50的两个row就下降0.195957 dB；fr2_xyz 1%的主要正向效果反而来自Q=.65–.80组（+0.079264 dB）。这不支持“低Q parent普遍更不适合渲染”的单调解释。各bin干预不相加，也不据此设计新threshold。\n',
table(['scene','ratio','rep位置','N','C存活率','C median','Q median'],[(s,ratio(float(a['ratio'])),a['group'],a['internal_count'],pct(a['C_TopK_survival_fraction'])if a['internal_count']!='0'else'—',pct(a['C_rank_percentile_median'])if a['internal_count']!='0'else'—',f(a['node_Q_median'])if a['internal_count']!='0'else'—')for s in SCENES for a in rows(ROOT/s/'internal_rep_r123_statistics.csv')]),
'r1不表现为跨scene始终比r2/r3可靠：例如fr2_xyz 1%的r1 C存活率72.53%，低于r2/r3的77.78%/77.42%；fr1_xyz 5%的r3存活率88.89%，高于r1/r2。不能据此自动改成unlock_after_1或_2。\n',
'## 11. Same-region与空间覆盖\n\n**L在全部12个case中，每个root的selected count与P0 exact相同。** 每次替换严格留在原internal subtree；没有换root或改变K。因此重点失败case的L改善不能归因于在root之间重新分配预算。\n\nL允许在同subtree内改变final leaf身份，hit leaves/leaf mass可以变化；所以该干预不能隔离subtree内部的全部细粒度spatial allocation。root预算exact不等于每个final leaf预算exact。\n',
table(['scene','ratio','arm','hit roots','hit leaves','root Raw mass','leaf Raw mass','root预算=P0'],[(s,ratio(float(a['ratio'])),a['arm'],a['hit_roots'],a['hit_leaves'],pct(a['hit_root_Raw_mass_fraction']),pct(a['hit_leaf_Raw_mass_fraction']),a['root_budget_exact_to_P0'])for s in SCENES for a in S[s]['spatial']]),
'所有逐root计数见`root_budget_distribution.csv`；L replacement完整行级trace包含original rank/node/Q、proxy C rank/Raw/final leaf、same-subtree与C TopK标志。\n',
'## 12. Correctness与执行证据\n\n- Baseline full ranking、C binary/tree、P0 priority、selected sets，以及四指标per-view/aggregate均exact。\n- S为P0稳定filter，non-internal relative order exact；全部S剩余数量足够Kmax。\n- L/G每ratio same-K、无重复、全部来自同candidate；L构造两次selected set与trace exact。\n- L逐次断言same subtree，所有case的replacement_failed=0，逐root预算分布与P0 exact。\n- G严格按C rank refill；独立实现再次验证。\n- `independent_check.py`不用construct或质量/GT，通过冻结final-leaf path prefix独立重建descendant sets，逐step复核L最优可用proxy、trace与最终集合；12/12 case通过。\n- 补充synthetic覆盖same-subtree proxy、无unused descendant保留原row、G rank refill、same-K/unique与repeat determinism。\n- construct阶段不读取GT、图像、render或质量CSV；replacement只读C排序/树结构。质量是在固定集合生成后评价。\n- 原metric/worker/progressive/fixed-marginal模块逐字复用，runtime SHA与实际maps校验通过，结束前Master/TF tensors未变。\n',
table(['scene','S剩余N','完整P0 internal N','L替换数 1/5/10/15%','真实render总次数'],[(s,S[s]['construction']['S_remaining'],S[s]['construction']['internal_total'],'/'.join(str(a['replacement_count'])for a in S[s]['construction']['L_root_budget_parity']),S[s]['render_counts']['baseline_renders']+S[s]['render_counts']['new_diagnostic_renders'])for s in SCENES]),
'共11176次真实render：A/B/C/P0 baseline 4064次，主诊断S/L/G 3048次，固定Q-bin probes 4064次。未重渲染Bank contribution，未训练。性能不参与判定，本轮没有优化实现。\n',
'## 13. 因果解释、已知限制与停止结论\n\n1. **identity有局部因果证据**：L保持root预算并限制同subtree，在fr1_xyz 5/10%与fr2_xyz 1%恢复一部分质量；不能再把这些变化单纯归因于L1比例。\n2. **internal不等于无效点**：S全部退化；多数internal本身也在C TopK，且L在两个成功1% case反向。Q<.8只表示总体3rep容量不足，并不证明r1/r2/r3各自无价值。\n3. **C rank不是渲染oracle**：L选更高C rank的proxy仍可能比原row差；G也会损害成功case。“bad_internal_candidates”只是用户规定的diagnostic集合名，不是已证实的坏点标签。\n4. **remaining scheduling/ordering不能排除**：L并未恢复全部失败差距，部分case的G比L好；region内leaf分配、ancestor-conditioned选择、unlock后的ordering仍耦合。本轮不能量化它们各自的独立贡献。\n5. **不支持主因断言**：依据多scene、多ratio、paired views，最终只能PARTIAL；不能认定“重新输出parent provisional reps”是Progressive不稳定的主要/唯一原因。\n6. 数据来自三个冻结TUM final maps和既有归档views；不等于online/production验证。S/L/G为机制诊断，L/G不满足单一nested ranking，不可直接接percentile Teacher。\n\n下一步：**KEEP C**。若继续研究Progressive，需要同时保留identity与region/unlock/ordering两条机制解释；本轮不设计nested reps、不改unlock count、不调Q/grid/lambda、不自动发起下一轮。没有stage、commit或push。\n']
identity=[]
for s in SCENES:
    a=rows(ROOT/s/'internal_rep_inventory.csv')
    for depth in [1,2]:
        g=[z for z in a if z['emit_type']=='INTERNAL_REP'and int(z['depth'])==depth]
        identity.append((s,'L'+str(depth),len(g),sum(z['matches_structural_provisional_ID']=='True'for z in g)))
identity_text=table(['scene','depth','完整P0 internal数','属于原结构3rep的row数'],identity)
document='\n'.join(t).replace('## 4. Internal reps最终存活情况（Q1）',identity_text+'\n## 4. Internal reps最终存活情况（Q1）')
document+='\n![同subtree替换与全局C refill相对P0的PSNR变化](internal_rep_replacement_psnr.png)\n'
(ROOT/'PROGRESSIVE_INTERNAL_REP_UTILITY_AUDIT.md').write_text(document,encoding='utf-8')
try:
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.6),sharey=True)
    for ax,s in zip(axes,SCENES):
        for a in ['L','G']:ax.plot([r*100 for r in R],[delta(s,a,'P0',r)for r in R],marker='o',label=a+' - P0')
        ax.axhline(0,color='gray',linewidth=.8);ax.set_title(s);ax.set_xlabel('Budget (%)');ax.set_xticks([1,5,10,15]);ax.grid(alpha=.2)
    axes[0].set_ylabel('Paired mean PSNR difference (dB)');axes[-1].legend();fig.tight_layout();fig.savefig(ROOT/'internal_rep_replacement_psnr.png',dpi=170);plt.close(fig)
except ImportError:pass
print('REPORT_COMPLETE',result['hypothesis_verdict'],flush=True)
