"""用已冻结CSV生成中文报告；不进行参数或方法选择。"""
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parent
LOCAL='H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/hierarchical_si_quality_20261019'
def rows(n):return list(csv.DictReader(open(ROOT/n)))
def read(n):return json.loads((ROOT/n).read_text())
def table(headers,data):return '\n\n|'+'|'.join(headers)+'|\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('|'+'|'.join(map(str,r))+'|'for r in data)+'\n\n'
def fmt(v,p=6):return '—'if v is None else f'{float(v):.{p}f}'
def signed(v,p=6):return f'{float(v):+.{p}f}'
def percent(v):return f'{100*float(v):.2f}%'
q=rows('quality_summary.csv');delta=rows('quality_deltas.csv');pair=rows('paired_per_view_statistics.csv');overlap=rows('ranking_overlap.csv');red=rows('selected_redundancy.csv');comp=rows('hierarchical_topk_composition.csv');levels=rows('representative_level_composition.csv');resid=rows('residual_statistics.csv');coverage=rows('leaf_budget_coverage.csv');dense=rows('dense_region_budget_share.csv');cases=read('mechanism_cases.json');sanity=read('ranking_sanity.json');input_info=read('input_manifest.json');frozen=read('frozen_sha_checks.json');rep=read('representative_delta_summary.json');ratios=read('preregistration.json')['ratios']
Q={(r['arm'],float(r['ratio'])):r for r in q};R={(r['arm'],float(r['ratio'])):r for r in red};C={(r['arm'],float(r['ratio'])):r for r in comp}
out=['# Hierarchical Representative SI：Frozen Same-K Reconstruction Quality Audit\n\n']
out+=['结论：**整体QUALITY INCONCLUSIVE；低budget收益明确，不认定Strong GO。** Hierarchical相对Raw四档PSNR和LPIPS改善，1%/5%四项指标均改善、平均PSNR提升0.620024 dB；但10%/15%出现SSIM和photo回退，按任务原文“指标方向混合必须降级”作保守判定。PSNR表现接近Case A，不能将其写成无条件的整体Case A成功。\n\n与Fixed h8 Marginal的四档平均PSNR差为−0.017225 dB，满足预注册“接近”标准。该MATCHES仅表示预定义质量距离，不表示ranking、selected set或每项指标相同。\n\n']
out+=['本轮仅评价冻结final map；没有训练Gaussian/Selector/TF，没有重算GI，没有修改算法、参数、Bank、Master或production代码。未commit、未stage、未push。运行时间未参与任何GO判定。\n\n']
out+=['## 1. 研究问题与预注册\n\nQ1比较C相对Raw的same-K质量；Q2比较C与已验证的Fixed h8 Marginal；Q3利用D/E、residual组成、代表层级、统一重复度和空间budget解释机制。结构Q不作为质量gate。\n\n']
out+=['`preregistration.json`在质量渲染前写入，SHA256为 `ebb36b8fdb16b0583e59ffd27c92b78127c7565127e24d3d23c3a2e374bfc6cb`。Primary要求至少3/4档ΔPSNR≥−0.02 dB、低ratio平均ΔPSNR>+0.10 dB或满足+0.20/−0.02替代条件，且无系统性SSIM/LPIPS交换。Strong另要求四档非负/等价、LPIPS多数改善、低ratio paired明显偏正、重复度下降、接近FixedMarginal。\n\n']
out+=['将“明显”固定为低ratio paired PSNR改善视角≥60%且median>0，mean redundancy至少下降10%且κ>0.9 pairs减少。指标交换的诊断容差固定为SSIM 0.002、LPIPS 0.005；未调整这些阈值。\n\n**判读修正记录：** 初始量化实现只在低ratio或至少两个ratio超幅度容差时降级，范围窄于任务原文“PSNR/SSIM/LPIPS方向混合必须降级”。复核发现高ratio确有混合方向，因此保留原始preregistration、全部数值gate和结果，按优先级更高的用户原文进行保守降级。`mechanism_cases.json`同时记录量化gate通过和最终INCONCLUSIVE；不重设阈值、不修改算法、不重跑/筛选质量结果，也不把修正后的判读伪称为最初代码已正确实现。\n\n']
out+=['## 2. 冻结输入、SHA与runtime恢复\n\nHEAD：`40751d07e194aaaa997d06bdea14d90f813b1222`，分支 `redundancy-aware-si`。工作树保留上一轮未提交改动，详细开始状态见 `audit_before.json`。\n\n']
out+=['完全复用fr1_room final `16900_shutdown`：Master topology、xyz、activated scale、opacity/SH/rotation、TF参数及AABB buffers、69-view贡献矩阵、Raw SI、mask、evaluation cameras与GT。实际N_total=52635，valid AND observed候选N=50330，effective maturity=1。\n\n']
items=[]
for label,path in [('Master',input_info['master']),('TF（含AABB buffers）',input_info['tf']),('camera/GT archive',input_info['cameras']),('checkpoint config',input_info['config'])]:items.append([label,frozen['checked'][path]['actual']])
for label,suffix in [('69-view contribution matrix','/redundancy-si-ranking-audit-20261019/contributions.pt'),('Raw/geometry ranking input','/redundancy-si-ranking-audit-20261019/ranking_inputs.npz'),('hierarchy binary input','/sparse-hash-root-grid-20261019/fr1_room_input.bin')]:
    v=next(v for p,v in frozen['checked'].items()if p.endswith(suffix));items.append([label,v['actual']])
out+=[table(['冻结输入','SHA256'],items)]
out+=['发现当前production mapper库SHA已从旧审计的 `523bba…` 变为 `2f381f…`，因此暂停启动质量渲染。随后从历史 `geometry-marginal-online-20261019/runtime_before/lib` 找到字节完全一致的旧mapper与rasterizer库，复制到本轮隔离runtime，通过LD_LIBRARY_PATH选择，并核对ldd及实际worker `/proc/PID/maps`。未覆盖当前production库。使用的旧mapper SHA为 `523bba9fea150989e07f93eb4977e95bbd91cafb4c58cc0ed3fcdb2964345b1c`；rasterizer为 `019cbd9fb052c8fa01dd03a977ec0f0c262cb01c963c03e2bb87c4e383f5e6a7`。\n\n']
out+=['全部实际冻结输入SHA匹配后才继续。配置/camera清单中69 Bank与161 evaluation frames严格不交叉并覆盖归档230个KF；评价集合完整复用旧协议。正确称为 **fixed non-Bank archived evaluation views**：这些视角参与过原SLAM，不将其视为新的独立采集数据。\n\n']
out+=['## 3. Arm定义与TF/renderer语义\n']
out+=[table(['Arm','冻结定义'],[
['A Raw SI','I=mean_v(c_iv)，直接降序，source row打破tie'],['B Fixed h8 Marginal','旧GeometryOnlyReference：固定h8=8*d50、geometry-only κ、完整local marginal greedy及保留predecessor的global merge'],['C Hierarchical','当前SparseHash2048/3rep/Q<0.80/L3，代表score=真实Delta，non-rep=I*(1-cover)，直接全局score降序'],['D Representative-only','仅把C的non-rep score置0；相同tree/reps/Delta，不替代primary'],['E L1-only','直接复用C每个root已记录的临时3-rep与Delta，无split；按同一公式重分配，仅diagnostic']])]
out+=['所有score tie按source row升序；B沿用其冻结global merge tie规则并验证完整local predecessor顺序。C没有第二次marginal ranking。E没有改production depth/threshold，只用已有root计算记录构造诊断score。\n\n每arm每ratio使用相同Master、requested ratio、同一冻结TF consumer与全1 maturity；保留原TF candidate/AABB筛选，selected rows先按source row排序再compact render。worker在退出前逐tensor检查Master及TF参数/buffers未改变。\n\n指标沿用原worker/common：PSNR为逐view MSE的dB后取mean；SSIM与LPIPS使用旧PyTorch指标；production photo为0.8*L1+0.2*(1-fast_ssim)。photo在原render tensor计算，质量指标按原协议clamp到[0,1]。\n\n']
out+=['## 4. Same-K、nested与结构复现\n']
out+=[table(['ratio','N_candidate','K=floor(r*N+0.5)','全部arm exact K / nested'],[[percent(r),50330,int(r*50330+.5),'PASS']for r in ratios])]
out+=['每个arm均生成完整50330行确定性permutation，四档是同一ranking前缀。C在当前binary重新构建两轮，每轮内部另有重复检查：完整score、assignment、tree及complete ranking exact一致，并与上一轮SparseHash fixture exact一致。\n']
out+=[table(['结构量','本轮复现'],[['occupied roots',2048],['h_root (m)',fmt(sanity['structure']['h_root'],12)],['final leaves',5601],['final representatives',13715],['Raw-SI weighted leaf Q',percent(sanity['structure']['weighted_leaf_Q'])]])]
out+=['Input Raw/xyz/activated scale及candidate rows与上轮完全一致。Q和这些结构量只用于确认复现，不参与质量GO。\n\n']
out+=['## 5. FixedMarginal与Raw基线复现\n\n重新执行旧GeometryOnlyReference的完整local greedy，完整order/gain/local rank/cell/key与旧GeometryOnly8及已冻结reference.npz逐值一致。随后独立重渲染A/B，四档各161视角；photo/PSNR/SSIM/LPIPS每项最大逐view差均为 **0**。只有该gate通过后才渲染C/D/E。详见 `baseline_reproduction.json`。\n\n']
out+=['## 6. Primary质量表\n']
out+=[table(['ratio','K','Raw PSNR','FixedMarginal PSNR','Hierarchical PSNR','C−Raw dB','C−Fixed dB'],[[percent(r),int(r*50330+.5),fmt(Q['A_Raw',r]['psnr']),fmt(Q['B_FixedMarginal',r]['psnr']),fmt(Q['C_Hierarchical',r]['psnr']),signed(float(Q['C_Hierarchical',r]['psnr'])-float(Q['A_Raw',r]['psnr'])),signed(float(Q['C_Hierarchical',r]['psnr'])-float(Q['B_FixedMarginal',r]['psnr']))]for r in ratios])]
for metric in ['ssim','lpips','photo']:
    out+=[table(['ratio','Raw '+metric,'FixedMarginal '+metric,'Hierarchical '+metric],[[percent(r),fmt(Q['A_Raw',r][metric]),fmt(Q['B_FixedMarginal',r][metric]),fmt(Q['C_Hierarchical',r][metric])]for r in ratios])]
out+=['## 7. 四项指标差值\n\n差值均为前者减后者。PSNR/SSIM正值改善，LPIPS/photo负值改善。\n']
out+=[table(['比较','ratio','ΔPSNR','ΔSSIM','ΔLPIPS','Δphoto'],[[d['comparison'],percent(d['ratio']),*[signed(d['delta_'+m])for m in ['psnr','ssim','lpips','photo']]]for d in delta if d['comparison']in ['C_vs_Raw','C_vs_FixedMarginal']])]
out+=['C相对Raw：1%/5%四项指标同向改善；10%/15%的PSNR及LPIPS改善，而SSIM分别下降0.002677/0.001282、photo分别增加0.000997/0.000307。因此不能用PSNR收益覆盖指标交换。相对FixedMarginal，C在5% PSNR/SSIM/photo略好；其他PSNR略低，LPIPS在1/5/15%略差、10%略好。MATCHES不等于全面超过B。\n\n']
out+=['FixedMarginal自身在10%/15%相对Raw也出现SSIM和photo回退：SSIM约−0.003173/−0.001219，photo约+0.000935/+0.000075。因此这个高budget权衡不是C独有的新现象；本轮仍按当前任务的严格mixed-direction条款降级，不能拿历史B的GO标签替代当前评价规则。\n\n']
out+=[f'![四项指标的方向与高budget交换]({LOCAL}/metric_tradeoffs.png)\n\n']
out+=['## 8. Paired per-view统计\n\n全部比较严格配对同161个frame_index。下表“改善视角”按指标方向：PSNR/SSIM差值>0，LPIPS/photo差值<0；CSV另存原始positive-delta fraction，避免对lower-is-better指标产生歧义。\n']
for comparison in ['C_vs_Raw','C_vs_FixedMarginal']:
    out+=[table(['比较','ratio','指标','mean Δ','median Δ','p10 Δ','p90 Δ','改善视角'],[[comparison,percent(d['ratio']),d['metric'],*[signed(d[k])for k in ['mean','median','p10','p90']],percent(d['improved_view_fraction'])]for d in pair if d['comparison']==comparison])]
out+=['每view四指标差值保存在 `paired_per_view_deltas.csv`，原始值在 `quality_per_view.csv`。归档KF在空间/时间上相关，不能将161个paired样本当作161个独立场景。\n\n']
out+=['## 9. Ranking overlap与增删rows\n']
out+=[table(['比较','ratio','Jaccard','intersection','added rows','removed rows'],[[a['comparison'],percent(a['ratio']),fmt(a['Jaccard']),a['intersection'],a['added'],a['removed']]for a in overlap if a['comparison']in ['C_vs_A_Raw','C_vs_B_FixedMarginal']])]
out+=['`ranking_selected_set_changes.csv`逐row保存增删方向、source row ID、Raw I、hierarchical score、代表标签/层级、leaf/root及xyz；不是只给Jaccard或只给数量。\n\n']
out+=['## 10. 统一geometry重复度\n\n对所有arm使用同一个冻结Fixed-h8分组和scalar-radius κ：selected Gaussian的redundancy是同h8 cell中其余selected κ的最大值，singleton为0；unordered pair只计一次。该诊断不使用view、C的leaf尺寸或自适应阈值，不统计跨h8 boundary的pairs。\n']
out+=[table(['arm','ratio','mean','median','p90','κ>0.8 pairs','κ>0.9 pairs','κ>0.95 pairs'],[[a['arm'],percent(a['ratio']),fmt(a['mean']),fmt(a['median']),fmt(a['p90']),a['unordered_pairs_gt_0.8'],a['unordered_pairs_gt_0.9'],a['unordered_pairs_gt_0.95']]for a in red if a['arm']in ['A_Raw','B_FixedMarginal','C_Hierarchical']])]
out+=['C相对Raw明显减少重复选择，但各档重复度仍高于B；更低重复度不保证每个ratio都更高PSNR，例如5%的C比B重复度高而PSNR更好。这里不存在把“dense region选得更少”当作成功条件。\n\n']
out+=['## 11. Representative与residual组成\n']
out+=[table(['ratio','K','final-leaf reps','rep fraction','non-rep residual','residual fraction'],[[percent(r),C['C_Hierarchical',r]['K'],C['C_Hierarchical',r]['representative_count'],percent(C['C_Hierarchical',r]['representative_fraction']),C['C_Hierarchical',r]['residual_count'],percent(C['C_Hierarchical',r]['residual_fraction'])]for r in ratios])]
out+=['各档全部selected score为正；D也没有为了凑K选入零score non-rep。composition CSV中的所有arm统一按C的final-leaf代表身份分类；E行不是E自身L1代表占比。\n\n']
out+=['## 12. Final representative层级\n\n下表count/fraction的fraction以该档C选中的representatives为分母；split parent的临时代表不计入C。CSV另存占K的比例。\n']
out+=[table(['ratio','level','count','占selected representatives'],[[percent(a['ratio']),a['level'],a['count'],percent(a['fraction_of_selected_representatives'])]for a in levels if a['arm']=='C_Hierarchical'])]
out+=['1%同时使用L1/L2/L3代表，未出现预算全部被粗层或细层单独占据；层级计数用于解释，不以层级分布本身判断质量。\n\n']
out+=['## 13. Residual score诊断\n\nnon-rep residual_ratio=I′/(I+eps)，近似1-cover；仅对non-rep统计。\n']
out+=[table(['ratio','population','count','mean','median','p10','p90','cover≥0.9','cover≥0.95','cover≥0.99'],[[('all'if float(a['ratio'])==0 else percent(a['ratio'])),a['population'],a['count'],*[fmt(a[k])for k in ['mean','median','p10','p90']],a['cover_ge_0.9'],a['cover_ge_0.95'],a['cover_ge_0.99']]for a in resid])]
out+=['1%仅1个non-rep，其residual_ratio约0.999714，基本没有被代表覆盖；5%的126个selected non-reps没有cover≥0.9者。10%/15%分别出现16/62个cover≥0.9者，但所有ratio均无selected non-rep的cover≥0.99。当前低budget不符合“大Raw SI的高cover duplicate经residual大规模回流”的模式。\n\n']
out+=['## 14. Representative Delta诊断\n\n全部13715个final reps逐row保存于 `representative_delta_statistics.csv`：Raw I_r、Delta、Delta/(I_r+eps)、level、leaf occupancy/Q/mass、Raw与C全局rank。\n\n']
out+=[f"Delta与Raw I的Spearman={rep['Spearman_Delta_vs_Raw']:.9f}。这说明二者相关但发生了实质性重排；并非重新命名Raw score。\n"]
dr=rows('representative_delta_rank_jaccard.csv')
out+=[table(['ratio','K（取全candidate档位）','仅在代表集合内：Delta Top-K vs Raw Top-K Jaccard'],[[percent(a['ratio']),a['K'],fmt(a['Jaccard'])]for a in dr if a['population']=='candidate_K'])]
out+=['这里两条ranking的population均限定在13715个representatives，K使用全candidate实验档位；CSV还保存以representative population自身ratio计算K的版本，避免混淆。\n\n']
out+=['## 15. Selected leaf与Raw-SI mass coverage\n']
out+=[table(['arm','ratio','selected leaves / 5601','leaf fraction','hit-leaf Raw SI / global SI'],[[a['arm'],percent(a['ratio']),a['selected_leaves'],percent(a['selected_leaf_fraction']),percent(a['selected_Raw_SI_leaf_mass_fraction'])]for a in coverage if a['arm']in ['A_Raw','B_FixedMarginal','C_Hierarchical']])]
out+=['hit-leaf mass把至少有一个selected点的leaf全部Raw mass计入，是空间分布诊断，不等于实际render贡献覆盖或设施目标F。\n\n']
out+=['## 16. Dense-region budget share\n\n运行前固定：Fixed-h8 occupied cells中occupancy最高的前ceil(10%)为dense；按Raw-SI mass降序取最短前缀覆盖≥80%总mass为important。所有tie按key升序。另对Sparse roots重复相同规则，并保存top10 dense明细；不看quality定义region。\n']
out+=[table(['region','arm','ratio','selected count','budget share','区域内统一κ mean redundancy'],[[a['region'],a['arm'],percent(a['ratio']),a['selected_count'],percent(a['budget_fraction']),fmt(a['uniform_geometry_mean_redundancy_inside_region'])]for a in dense if a['domain']=='fixed_h8'and a['region']in ['dense_top10percent','important_cover80percent']and a['arm']in ['A_Raw','B_FixedMarginal','C_Hierarchical']])]
out+=['空间名额分配和内部去重复分别报告；区域candidate密度不参与split或score。`region_definitions.json`保存明确cell/root列表，`dense_region_budget_share.csv`包含全部arm与两种空间分组。\n\n']
out+=['具体地，1%时dense前10% cells中的budget从Raw的134/503（26.64%）增至C的199/503（39.56%），区域内κ mean redundancy却从0.511576降至0.117168；5%则在近似预算1041→1031下从0.609344降至0.239574。这支持“重要dense区域可以保留甚至增加名额，同时内部减少重复”的机制，而非简单density penalty。\n\n']
out+=['## 17. Arm D：是否应去掉residual\n']
out+=[table(['ratio','C PSNR','D PSNR','D−C PSNR','C/D Jaccard','D rep fraction','C redundancy','D redundancy'],[[percent(r),fmt(Q['C_Hierarchical',r]['psnr']),fmt(Q['D_RepresentativeOnly',r]['psnr']),signed(float(Q['D_RepresentativeOnly',r]['psnr'])-float(Q['C_Hierarchical',r]['psnr'])),fmt(next(a['Jaccard']for a in overlap if a['comparison']=='C_vs_D_RepresentativeOnly'and float(a['ratio'])==r)),percent(C['D_RepresentativeOnly',r]['representative_fraction']),fmt(R['C_Hierarchical',r]['mean']),fmt(R['D_RepresentativeOnly',r]['mean'])]for r in ratios])]
out+=[table(['ratio','D−C ΔSSIM','D−C ΔLPIPS','D−C Δphoto'],[[percent(a['ratio']),signed(a['delta_ssim']),signed(a['delta_lpips']),signed(a['delta_photo'])]for a in delta if a['comparison']=='D_vs_C'])]
out+=['D在1%只增加0.006891 dB，5/10/15%分别下降0.024537/0.106304/0.241640 dB，不满足“D明显优于C”。高ratio虽然D重复度更低，质量反而下降，支持residual保留有用未覆盖结构；不能据此把正式non-rep score置0。Case E不成立。\n\n']
out+=['## 18. Arm E：Octree的可观察价值\n']
out+=[table(['ratio','L1-only PSNR','full hierarchy PSNR','C−E ΔPSNR','ΔSSIM','ΔLPIPS','Δphoto'],[[percent(a['ratio']),fmt(Q['E_L1Only',float(a['ratio'])]['psnr']),fmt(Q['C_Hierarchical',float(a['ratio'])]['psnr']),*[signed(a['delta_'+m])for m in ['psnr','ssim','lpips','photo']]]for a in delta if a['comparison']=='C_vs_L1Only'])]
out+=['Full hierarchy在5/10/15%提高PSNR，LPIPS全部四档优于L1-only；但1% PSNR下降0.099203 dB，且SSIM并非每档改善。可回答有可观察的质量价值，但不支持“split对所有预算都更好”。Full与L1-only未达到预注册近似等价，Case F不成立。\n\n']
out+=['## 19. Case分类与质量gate\n']
out+=[table(['Case','判定','依据'],[['A','有条件接近','低ratio及PSNR/LPIPS支持，PSNR距离MATCHES FixedMarginal；高ratio指标混合，整体结论降级'],['B','NO','不是明显低于FixedMarginal的PSNR档位'],['C','NO','不是所有质量都近似Raw，低ratio有清晰增益'],['D','NO','未出现PSNR整体回退，但不能忽视高ratio SSIM/photo损失'],['E','NO','Representative-only未明显优于C，较高ratio反而更差'],['F','NO','Full与L1-only在多档质量不同']])]
out+=[table(['Primary/Strong gate','结果'],[[k,str(v)]for k,v in cases['primary_gates'].items()]+[[k,str(v)]for k,v in cases['strong_gates'].items()])]
out+=[f"低ratio平均ΔPSNR={cases['low_mean_delta_PSNR']:+.6f} dB；四ratio平均C−B={cases['mean_C_minus_B_PSNR']:+.6f} dB。数值Strong条目通过，但原文mixed-direction降级条款触发，不能宣布Strong GO。最终 **{cases['primary_decision']}；Strong GO={cases['Strong_GO']}**。\n\n"]
out+=['## 20. Runtime与完整性\n\n本轮不以runtime判GO。\n']
out+=[table(['记录量','时间'],[['B完整reference重算',fmt(sanity['runtime_ms']['B_reference'],3)+' ms'],['C一次完整build',fmt(sanity['runtime_ms']['C_build'],3)+' ms'],['20个arm-ratio、3220次view render及指标',fmt(read('quality_complete.json')['runtime_seconds'],3)+' s']])]
out+=['hierarchy程序使用上轮已验证binary，源码SHA与当前本地core一致；排序及tree两轮重建通过。worker完整检查frozen Master/TF未变，源码及输入在结束时再次SHA核对。图表只从最终CSV读取，不改变实验。\n\n']
out+=['最终 `final_validation.json` 独立核对240个冻结输入/worker条目、原ranking archive SHA、完整input manifest、same-K/nested以及3220条逐view指标与20个aggregate的一致性。`initial_quantitative_cases.json`仅保留判读修正前的审计记录，最终结论以 `mechanism_cases.json` 为准。\n\n']
out+=['## 21. 输出文件\n\n必需文件全部保存：`quality_summary.csv`、`quality_per_view.csv`、`ranking_overlap.csv`、`selected_redundancy.csv`、`hierarchical_topk_composition.csv`、`representative_level_composition.csv`、`residual_statistics.csv`、`representative_delta_statistics.csv`、`leaf_budget_coverage.csv`、`mechanism_cases.json`。\n\n额外保存完整ranking/selected rows、逐row增删、paired delta/statistics、dense-region budget、代表Delta统计/Jaccard、pre-registration、SHA/runtime/TF验证以及结构重建输出。\n\n']
out+=[f'![冻结same-K质量与机制]({LOCAL}/quality_and_mechanism.png)\n\n',f'![Raw与Hierarchical完整rank重排]({LOCAL}/raw_vs_hierarchical_rank.png)\n\n']
out+=['## 22. 已知限制与机制结论\n\n本结论仅适用于一个冻结fr1_room final map和固定归档评价协议。未检验第二scene、online teacher训练、refresh staleness、ATE或production运行；不能从本轮quality结论推导在线/production结论。\n\nHierarchical与Fixed h8使用不同competition partition，且C只记录每leaf前三步Delta，再保留residual；B生成完整conditional marginal序列。两者并非等价算法。C的重复度仍高于B，少量质量差距可能同时来自partition、代表容量与score近似，当前消融不能唯一归因。\n\n低budget由final representatives主导，residual主要来自未充分覆盖的信息；D不支持删除residual。Octree具有ratio相关的可观察收益，不能宣称对1%也必然有利。高ratio的SSIM/photo交换同时见于B，尚无证据将其唯一归因于hierarchical representative/residual/split某一层。\n\n下一步首先是明确是否接受这种高budget指标取舍，再判断近似/容量分配差异能否跨场景稳定；不是继续调整Grid、0.80、2048、rep数或residual公式。本轮到此停止，不自动开展下一轮实验。\n\n']
out+=['## 23. 最终明确回答\n']
out+=[table(['问题','回答','限定'],[['Q1 Hierarchical优于Raw？',cases['Q1'],'低ratio YES；整体PSNR/LPIPS改善，但10/15% SSIM/photo回退'],['Q2 1%/5%稳定收益？',cases['Q2'],'四项mean和paired均支持；单冻结scene'],['Q3 接近/超过Fixed h8？',cases['Q3'],'四档平均PSNR差−0.017225 dB；不是所有指标完全一致'],['Q4 代表真正进入Top-K？',cases['Q4'],'低budget代表99.80%/94.99%，不是大量高cover residual回流'],['Q5 Adaptive Octree有质量价值？',cases['Q5'],'5/10/15%有PSNR收益；1%负收益，不能泛化为全预算胜出'],['Q6 Case分类',cases['Q6'],'接近A的条件性表现，原文指标混合条款触发；不属于E或F'],['Q7 最终',cases['Q7'],'不认定整体QUALITY GO或Strong GO，不外推至online/production']])]
(ROOT/'HIERARCHICAL_REPRESENTATIVE_SI_QUALITY_AUDIT.md').write_text(''.join(out),encoding='utf-8')
print('REPORT_WRITTEN',flush=True)
