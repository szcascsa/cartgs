"""只读生成三份中文报告；所有GO门控来自quality前冻结的分析方案。"""
import json,re,hashlib,csv,os
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;R=Path('/home/prmi/zwj/mycartgs')
def read(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |'for row in rows)
def csvout(name,rows):
    with open(ROOT/name,'w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def describe(v):
    v=np.asarray(v,float);v=v[np.isfinite(v)]
    return dict(count=len(v),mean=float(v.mean()),median=float(np.median(v)),p90=float(np.quantile(v,.9)),p95=float(np.quantile(v,.95)),p99=float(np.quantile(v,.99)),max=float(v.max()),sum=float(v.sum()))if len(v)else dict(count=0,mean=None,median=None,p90=None,p95=None,p99=None,max=None,sum=0.)
def plot(name):plt.tight_layout();plt.savefig(ROOT/name,dpi=160);plt.close()
P=read(ROOT/'online_protocol.json');A=read(ROOT/'analysis_preregistration.json');models=read(ROOT/'online_training_results.json');ev=read(ROOT/'online_evaluation_results.json');assert len(models)==8 and len(ev)==40
shadow=read(ROOT/'shadow_validation.json');st=pd.read_csv(Path(shadow['model']['model'])/'gi_teacher_trace.csv');assert not st.marginal.any()
Q=[]
for e in ev:
    row=dict(scene=e['scene'],seed=e['seed'],arm=e['arm'],policy=e['policy'],ratio=e['ratio']if e['ratio']is not None else 1.,**{k:float(e['metrics'][k])for k in ['psnr','ssim','lpips']},ate_rmse=e['ate']['translation_ape_m']['rmse'],associated_poses=e['ate']['associated_poses'],selected_K=e['selection']['active_gaussians'],N=e['selection'].get('total_gaussians'),rendering_ms=e['metrics']['rendering ms'])
    Q.append(row)
q=pd.DataFrame(Q);q.to_csv(ROOT/'online_quality.csv',index=False)
paired=[]
for (scene,seed,policy,ratio),g in q.groupby(['scene','seed','policy','ratio']):
    raw=g[g.arm=='raw'].iloc[0];new=g[g.arm=='marginal'].iloc[0];paired.append(dict(scene=scene,seed=int(seed),policy=policy,ratio=ratio,delta_psnr=float(new.psnr-raw.psnr),delta_ssim=float(new.ssim-raw.ssim),delta_lpips=float(new.lpips-raw.lpips),delta_ATE=float(new.ate_rmse-raw.ate_rmse),raw_psnr=float(raw.psnr),marginal_psnr=float(new.psnr),raw_ATE=float(raw.ate_rmse),marginal_ATE=float(new.ate_rmse),raw_K=int(raw.selected_K),marginal_K=int(new.selected_K)))
d=pd.DataFrame(paired);d.to_csv(ROOT/'online_paired_deltas.csv',index=False)
gaps=[]
for z in q[q.policy!='full'].itertuples():
    full=q[(q.scene==z.scene)&(q.seed==z.seed)&(q.arm==z.arm)&(q.policy=='full')].iloc[0]
    gaps.append(dict(scene=z.scene,seed=z.seed,arm=z.arm,policy=z.policy,ratio=z.ratio,lod_minus_full_psnr=float(z.psnr-full.psnr),lod_minus_full_ssim=float(z.ssim-full.ssim),lod_minus_full_lpips=float(z.lpips-full.lpips)))
csvout('online_lod_vs_full_gap.csv',gaps)

perf=[];overview=[];resources=[];full_refresh={};teacher=[]
def stage(meta,name,values):perf.append(dict(**meta,stage=name,**describe(values)))
for model in models:
    directory=Path(model['model']).parent;output=Path(model['model']);meta={k:model[k]for k in ['scene','seed','arm']};trace=pd.read_csv(directory/'mapping_trace.csv');log=(directory/'training.log').read_text(errors='replace');marg=pd.read_csv(output/'marginal_refresh.csv')
    stage(meta,'MappingAll',trace.total);stage(meta,'MappingNormalNoRefresh',trace.loc[trace.iteration%1000!=0,'total']);stage(meta,'MappingOnline',trace.loc[trace.phase==0,'total']);stage(meta,'MappingTail',trace.loc[trace.phase==1,'total'])
    stage(meta,'FullRefreshBoundaryGIIncludingLock',trace.loc[trace.iteration%1000==0,'gi'])
    for name in ['gi','gi_teacher','selector_forward','lod_backward','full_render','lod_render','transform_field','optimizer']:
        stage(meta,name,trace[name])
    full=np.array([float(v)for v in re.findall(r'global_gi_full_refresh_ms=([\d.eE+-]+)',log)]);stage(meta,'RawSGIFullRefresh',full)
    stage(meta,'RawSGIFullCUDAElapsed',[float(v)for v in re.findall(r'global_gi_full_refresh_cuda_ms=([\d.eE+-]+)',log)])
    stage(meta,'SGIIncrementalCUDAElapsed',[float(v)for v in re.findall(r'global_gi_incremental_cuda_ms=([\d.eE+-]+)',log)])
    inc_render=[float(v)for v in re.findall(r'global_gi_incremental_render_ms=([\d.eE+-]+)',log)];inc_aggregate=[float(v)for v in re.findall(r'global_gi_incremental_aggregate_ms=([\d.eE+-]+)',log)];stage(meta,'SGIIncrementalRender',inc_render);stage(meta,'SGIIncrementalAggregate',inc_aggregate)
    for name in ['copy_ms','grid_ms','membership_ms','similarity_ms','init_ms','rescore_ms','coverage_ms','ranking_ms','total_ms']:
        if name in marg:stage(meta,'Marginal_'+name,marg[name])
    if 'ranking_ms'in marg:
        residual=marg.ranking_ms-marg[['grid_ms','membership_ms','similarity_ms','init_ms','rescore_ms','coverage_ms']].sum(axis=1)
        stage(meta,'Marginal_heap_allocation_other_residual',residual)
    combined=full+marg.total_ms.to_numpy()[:len(full)];stage(meta,'CompleteRefreshWithMarginal',combined);full_refresh[tuple(meta.values())]=describe(combined)
    tracking=np.loadtxt(output/'TrackingTime.txt').reshape(-1);stage(meta,'TrackingAll',tracking*1000);stage(meta,'TrackingAfter30',tracking[30:]*1000)
    rssfile=directory/'training_resources.json';sample=read(rssfile)if rssfile.exists()else[]
    cpu_percent=None
    if len(sample)>1:
        cpu_percent=(sample[-1]['cpu_ticks']-sample[0]['cpu_ticks'])/os.sysconf('SC_CLK_TCK')/(sample[-1]['elapsed_s']-sample[0]['elapsed_s'])*100
    gpu_peak=(output/'GpuPeakUsageMB.txt').read_text();peak_reserved=float(re.search(r'Peak reserved \(MB\): ([\d.]+)',gpu_peak).group(1));peak_alloc=float(re.search(r'Peak allocated \(MB\): ([\d.]+)',gpu_peak).group(1))
    resources.append(dict(**meta,CPU_percent_one_core100=cpu_percent,GPU_percent_mean=float(np.mean([z['gpu_percent']for z in sample]))if sample else None,sampled_peak_RAM_MiB=max((z['rss_kib']/1024 for z in sample),default=None),sampled_peak_VRAM_MiB=max((z['vram_mib']for z in sample),default=None),torch_peak_reserved_MiB=peak_reserved,torch_peak_allocated_MiB=peak_alloc,resource_samples=len(sample)))
    teacherfile=output/'gi_teacher_trace.csv';t=pd.read_csv(teacherfile);assert(t.snapshot_age<=1000).all();t['scene']=meta['scene'];t['seed']=meta['seed'];t['arm']=meta['arm'];teacher.extend(json.loads(t.to_json(orient='records')))
    if meta['arm']=='raw':assert not t.marginal.any()
    else:assert t.loc[t.iteration>1000,'marginal'].all()
    hvalues=marg.h.unique();overview.append(dict(**meta,gaussians=model['N'],mapping_cycles=len(trace),mapping_mean_ms=float(trace.total.mean()),normal_mapping_mean_ms=float(trace.loc[trace.iteration%1000!=0,'total'].mean()),tracking_mean_ms=float(tracking.mean()*1000),tracking_p99_ms=float(np.quantile(tracking[30:]*1000,.99)),refresh_mean_ms=float(combined.mean()),refresh_p95_ms=float(np.quantile(combined,.95)),refresh_max_ms=float(combined.max()),marginal_amortized_ms=float(marg.total_ms.sum()/len(trace))if meta['arm']=='marginal'else 0.,h_initial=float(hvalues[0]),h_values=len(hvalues),online_mappings=model['state']['online']['mappings'],tail_mappings=model['state']['tail']['mappings']))
csvout('online_stage_profile.csv',perf);csvout('online_resource_usage.csv',resources);csvout('online_run_summary.csv',overview);csvout('online_teacher.csv',teacher)
summary=pd.DataFrame(overview);overheads=[]
budget_rows=[]
for m in models:
    budget_rows.append(dict(scene=m['scene'],seed=m['seed'],arm=m['arm'],N=m['N'],online_mapping=m['state']['online']['mappings'],tail_mapping=m['state']['tail']['mappings'],gaussian_updates=m['state']['online']['gaussian_updates']+m['state']['tail']['gaussian_updates'],selector_commits=m['state']['online']['selector_commits']+m['state']['tail']['selector_commits'],TF_commits=m['state']['online']['tf_commits']+m['state']['tail']['tf_commits']))
csvout('online_training_budget.csv',budget_rows)
for (scene,seed),g in summary.groupby(['scene','seed']):
    raw=g[g.arm=='raw'].iloc[0];new=g[g.arm=='marginal'].iloc[0];overheads.append(dict(scene=scene,seed=int(seed),mapping_overhead_percent=float(100*(new.mapping_mean_ms/raw.mapping_mean_ms-1)),normal_step_overhead_percent=float(100*(new.normal_mapping_mean_ms/raw.normal_mapping_mean_ms-1)),tracking_p99_change_percent=float(100*(new.tracking_p99_ms/raw.tracking_p99_ms-1)),raw_refresh_p95_ms=float(raw.refresh_p95_ms),marginal_refresh_p95_ms=float(new.refresh_p95_ms),marginal_refresh_max_ms=float(new.refresh_max_ms),conservative_amortized_marginal_overhead_percent=float(100*new.marginal_amortized_ms/raw.mapping_mean_ms)))
csvout('online_overheads.csv',overheads);oh=pd.DataFrame(overheads)
mechanism=pd.read_csv(ROOT/'online_mechanism.csv');mechanism.to_csv(ROOT/'online_mechanism.csv',index=False)
mp=mechanism.pivot(index=['scene','seed','training_arm','iteration','ratio'],columns='ranking',values='redundancy_mean');red_reduced=bool((mp.Marginal<mp.Raw).mean()>.5)
mechanism_summary=mechanism.groupby(['ranking','ratio'])[['redundancy_mean','duplicate_pairs_gt09','important_mass_coverage','dense_cell_budget_share']].mean().reset_index()
mechanism_summary.to_csv(ROOT/'online_mechanism_summary.csv',index=False)

# 质量前冻结判据：不把one-seed偶然改善写成production-ready。
primary=d[(d.scene=='fr1_room')&(d.policy=='production')];stable_bad=any((g.delta_psnr<0).all()and g.delta_psnr.mean()<-.10 for _,g in primary.groupby('ratio'));positive=primary.delta_psnr.mean()>.05 and(primary.delta_psnr>=0).sum()>=3
full=d[(d.scene=='fr1_room')&(d.policy=='full')];full_bad=(full.delta_psnr<0).all()and full.delta_psnr.mean()<-.20;ate_bad=all(z.delta_ATE>max(.005,.1*z.raw_ATE)for z in full.itertuples());perf_bad=bool((oh.groupby('scene').mapping_overhead_percent.mean()>10).any());tracking_warning=bool(any((g.tracking_p99_change_percent>20).all()for _,g in oh.groupby('scene')))
algorithm_online=bool(not stable_bad and not full_bad and not ate_bad and positive and red_reduced);decision='Online GO'if algorithm_online and not perf_bad and not tracking_warning else'Online NO-GO'
reason=dict(stable_quality_regression=bool(stable_bad),visible_low_ratio_gain=bool(positive),full_map_regression=bool(full_bad),systematic_ATE_regression=bool(ate_bad),snapshot_redundancy_reduced=red_reduced,performance_problem=perf_bad,tracking_tail_warning=tracking_warning)

# 五次隔离微测与occupancy profile。
profiles=pd.DataFrame(read(ROOT/'final_offline_profiles.json'));med=profiles.groupby('mode').wall_ms.median().to_dict();profile_table=table(['实现','中位ms','最小ms','最大ms'],[[m,f'{g.wall_ms.median():.3f}',f'{g.wall_ms.min():.3f}',f'{g.wall_ms.max():.3f}']for m,g in profiles.groupby('mode')])
cells={m:pd.read_csv(ROOT/('final_'+m+'_0_cells.csv'))for m in ['reference','global','optimized','optimized_full','uncached']};reference=pd.read_csv(ROOT/'reference_cells.csv');buckets=[('1',1,1),('2-4',2,4),('5-8',5,8),('9-16',9,16),('17-32',17,32),('33-64',33,64),('65-128',65,128),('>128',129,np.inf)];brows=[]
for label,lo,hi in buckets:
    for m,frame in cells.items():
        z=frame[(frame.n>=lo)&(frame.n<=hi)];brows.append(dict(bucket=label,implementation=m,cells=len(z),gaussians=int(z.n.sum()),selected=int(z.selected.sum()),gain_evaluations=int(z.evaluations.sum()),total_cell_ms=float(z.total_ms.sum()),similarity_ms=float(z.similarity_ms.sum()),initial_gain_ms=float(z.init_ms.sum()),rescore_ms=float(z.rescore_ms.sum()),coverage_ms=float(z.coverage_ms.sum()),cached_pairs=int(z.cached_pairs.sum())))
csvout('occupancy_bucket_profile.csv',brows);top=reference.nlargest(20,'n')[['cell','n','total_ms']].rename(columns={'total_ms':'Python_reference_ms'})
for m in ['reference','global','optimized','optimized_full']:top=top.merge(cells[m][['cell','total_ms','evaluations','selected']].rename(columns={k:m+'_'+k for k in ['total_ms','evaluations','selected']}),on='cell')
top.to_csv(ROOT/'largest_20_cells.csv',index=False)
evals={m:int(z.evaluations.sum())for m,z in cells.items()};saved_local=1-evals['optimized']/evals['global'];saved_full=1-evals['optimized_full']/evals['reference'];pair_bytes=int(profiles[profiles['mode']=='optimized'].pair_bytes.median());alloc=read(ROOT/'allocation_profile.json')
allocations={p.name:p.read_text()for p in ROOT.glob('native_allocations_*.log')};save(ROOT/'native_allocation_profiles.json',allocations)
allocation_rows=[]
for name,text in allocations.items():
    clean=re.sub(r'\x1b\[[0-9;]*m','',text);peak=re.search(r'heap total:\s*(\d+), heap peak:\s*(\d+)',clean);calls=re.search(r'malloc\|\s*(\d+)',clean)
    allocation_rows.append(dict(implementation=name.replace('native_allocations_','').replace('.log',''),malloc_calls=int(calls.group(1)),heap_total_bytes=int(peak.group(1)),heap_peak_bytes=int(peak.group(2))))
csvout('native_allocation_summary.csv',allocation_rows)
reference_quality=pd.read_csv(ROOT/'reference_quality.csv')
old_quality=pd.read_csv('/home/prmi/zwj/redundancy-si-ranking-audit-20261019/same_k_quality.csv');old_quality=old_quality[old_quality.TF==True];old_comparison=[]
for ratio in [.01,.05,.10,.15]:
    raw=float(old_quality[(old_quality.arm=='Raw')&(old_quality.ratio==ratio)].psnr.iloc[0]);combined=float(old_quality[(old_quality.arm=='Marginal8')&(old_quality.ratio==ratio)].psnr.iloc[0]);iso=float(reference_quality[reference_quality.ratio==ratio].psnr.iloc[0]);old_comparison.append([f'{ratio:.0%}',f'{raw:.4f}',f'{combined:.4f}',f'{iso:.4f}',f'{iso-raw:+.4f}',f'{iso-combined:+.4f}'])
heap_profile=read(ROOT/'heap_profile.json');python_phase=read(ROOT/'final_python_reference_profile.json')

# 科学图：每个scene/seed完整保留，图只做聚合对照。
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,scene in zip(axes,['fr1_room','fr1_desk']):
    z=q[(q.scene==scene)&(q.policy=='production')]
    for arm,g in z.groupby('arm'):
        u=g.groupby('ratio').psnr.agg(['mean','std']);ax.errorbar(u.index*100,u['mean'],yerr=u['std'],marker='o',capsize=3,label=arm)
    ax.set(xlabel='Ratio (%)',ylabel='PSNR (dB)',title=scene);ax.legend();ax.grid(alpha=.2)
plot('online_quality_by_ratio.png')
plt.figure(figsize=(8,4));plt.bar(summary.scene+' '+summary.seed.astype(str)+' '+summary.arm,summary.mapping_mean_ms);plt.xticks(rotation=45,ha='right');plt.ylabel('Mean complete Mapping wall (ms)');plot('online_mapping_cost.png')
plt.figure(figsize=(7,4));plt.bar(list(med),list(med.values()));plt.xticks(rotation=25);plt.ylabel('Median offline ranking wall (ms)');plot('offline_exact_speed.png')
plt.figure(figsize=(7,4))
for ranking,g in mechanism.groupby('ranking'):plt.plot([1,5],[g[g.ratio==r].redundancy_mean.mean()for r in [.01,.05]],'-o',label=ranking)
plt.xlabel('Snapshot same-K ratio (%)');plt.ylabel('Selected geometry redundancy');plt.legend();plot('online_snapshot_redundancy.png')
plt.figure(figsize=(8,4));fullq=q[q.policy=='full']
for arm,g in fullq.groupby('arm'):plt.scatter(g.ate_rmse,g.psnr,label=arm)
plt.xlabel('SE3 ATE RMSE (m)');plt.ylabel('Full-map PSNR');plt.legend();plot('online_ate_vs_full_quality.png')

quality_table=table(['scene','seed','ratio','Raw PSNR','Marginal PSNR','ΔPSNR','ΔSSIM','ΔLPIPS','Raw K','Marginal K'],[[z.scene,z.seed,f'{z.ratio:.0%}',f'{z.raw_psnr:.4f}',f'{z.marginal_psnr:.4f}',f'{z.delta_psnr:+.4f}',f'{z.delta_ssim:+.4f}',f'{z.delta_lpips:+.4f}',z.raw_K,z.marginal_K]for z in d[d.policy=='production'].itertuples()])
full_table=table(['scene','seed','Raw Full PSNR','Marginal Full PSNR','ΔPSNR','Raw ATE(m)','Marginal ATE(m)'],[[z.scene,z.seed,f'{z.raw_psnr:.4f}',f'{z.marginal_psnr:.4f}',f'{z.delta_psnr:+.4f}',f'{z.raw_ATE:.5f}',f'{z.marginal_ATE:.5f}']for z in d[d.policy=='full'].itertuples()])
overhead_table=table(['scene','seed','Mapping overhead','普通step overhead','Marginal刷新摊销','refresh p95 Raw/New(ms)','New refresh max(ms)'],[[z.scene,z.seed,f'{z.mapping_overhead_percent:+.2f}%',f'{z.normal_step_overhead_percent:+.2f}%',f'{z.conservative_amortized_marginal_overhead_percent:.2f}%',f'{z.raw_refresh_p95_ms:.2f}/{z.marginal_refresh_p95_ms:.2f}',f'{z.marginal_refresh_max_ms:.2f}']for z in oh.itertuples()])
rq=table(['ratio','PSNR','SSIM','LPIPS','photo'],[[f'{z.ratio:.0%}',f'{z.psnr:.6f}',f'{z.ssim:.6f}',f'{z.lpips:.6f}',f'{z.photo:.6f}']for z in reference_quality.itertuples()])

simplification=f'''# Geometry-only Marginal SI简化报告

结论：删除pairwise view项后，准确复现旧IsoOnly；Optimized Exact全部质量与Reference逐值相同。此结论属于offline，不替代online结论。

{rq}

相对Raw与旧combined的完整对照：

{table(['ratio','Raw SI','Geo×View','Geo-only','Geo-only−Raw','Geo-only−combined'],old_comparison)}

删除view后保持了主要offline same-K收益，并非宣称每档都胜过combined。

完整50330点ranking、四档selected set与旧GeometryOnly8 exact相等；Reference和Optimized各644个view-ratio的photo/PSNR/SSIM/LPIPS最大差均为0。maturity=1、原h8、固定Master/TF/Bank/GT，contribution未重算。

此前production没有pairwise view redundancy可删。新独立模块API只接受xyz、scalar radius、Raw I和snapshot row ID，完全不接收contribution vector、不计算cosine、不保存view similarity cache。旧audit历史文件保持原样；Raw SGI的GIViewCache、多视角Bank、均值贡献、incremental/full refresh和renderer统计全部保留。只新增可选mode `streaming_marginal_geo`；默认fast仍为Raw streaming，shadow/diagnostics默认false。

radius=cbrt(sx sy sz)，kappa=exp(−d²/(2 sigma²+1e−12))、diag=1，Fixed h8、无neighbor、无adaptive、无covariance。逐次条件marginal coverage不变，不是静态density penalty。生产实现见include/geometry_marginal_ranker.h和geometry_marginal_importance.h/.cpp。

已有teacher按动态valid/observed/maturity集合取percentile。为保留其接口与权重，online发布完整顺序，并编码为unique FP32 order score；实际marginal gain完整参与ranking和exact tests。前15%提前停止的优化独立测试；production不使用虚构tail score。append新row valid=false，prune按exact survivor mask同步映射，reset清空并重建；不使用row index作为跨topology稳定ID。首次full refresh发布之前明确使用Raw warmup。

用户追加要求后，服务器已按本地项目同步1831个源码/配置文件；没有删服务器数据、模型、结果、依赖。code_sync_result与source before/after SHA保存。原库在runtime_before保留，offline复渲染链接原库。所有production修改未commit、未push。
'''
(ROOT/'GEOMETRY_ONLY_SIMPLIFICATION_REPORT.md').write_text(simplification,encoding='utf-8')
(ROOT/'GEOMETRY_ONLY_REFERENCE_VALIDATION.md').write_text('# GeometryOnlyReference语义验证\n\n50330候选、固定h8、maturity=1、原TF、原161个evaluation views。完整ranking和selected set与旧IsoOnly exact一致；Reference和Optimized分别644个view-ratio四项metric逐值差0，全部selected rows复核一致。\n\n'+rq+'\n',encoding='utf-8')
performance=f'''# Geometry Marginal性能报告

结论：exact实现通过完整ranking和质量gate；没有CUDA kernel、approximate neighbors或score quantization。

## 隔离CPU微测（同输入、5次）

{profile_table}

Python Reference→15% Optimized Exact：{med['python_reference']/med['optimized']:.2f}×；Python Reference→production完整发布：{med['python_reference']/med['optimized_full']:.2f}×。请勿把语言更换的总收益当成CELF单独收益。

同C++ arithmetic/control：Full brute→lazy global 15%为{med['reference']/med['global']:.2f}×；lazy global→CELF 15%为{med['global']/med['optimized']:.2f}×；uncached CELF→cached CELF 15%为{med['uncached']/med['optimized']:.2f}×。无cache control是独立benchmark-only header，未进入production，selected rows同样exact。

Reference首次profile：1.255s，其中rescore728ms、similarity181ms，瓶颈为重复candidate评分。CPU FP64执行，offline GPU time=0、kernel launch=0、CPU↔GPU copy/sync=0。输入读取不包含在核心ranking计时里，CLI输出IO不包含在native核心计时里。

15%只选择7550/50330个代表，所有cell仍需初始化第一个head以建立全局竞争；CELF相比相同global expansion减少{saved_local:.2%}次真实gain评估。完整CELF相比native完整brute减少{saved_full:.2%}次评估。各mode总评估数：{evals}。

缓存pair数1859890，bytes={pair_bytes}（{pair_bytes/1048576:.2f}MiB），只限同cell，无全scene NxN。所有评分查询来自该snapshot dense cache，缓存命中率100%；初始化geometry构造不计为命中。最大的cell occupancy250。8个occupancy桶见occupancy_bucket_profile.csv，最大20cell逐项耗时见largest_20_cells.csv，未用全局平均掩盖dense cell。

{table(list(top.columns),top.round(4).values.tolist())}

CPU process-time、peak RSS见final_offline_profiles.json；Python tracemalloc单独运行，traced peak={alloc['reference_traced_peak_bytes']/1048576:.2f}MiB。Python live allocation snapshot差值不等于累计分配次数；已标明估算的NumPy allocation-site计数。Native allocator用独立LD_PRELOAD libmemusage测量，原始malloc/realloc/free总数和peak heap见native_allocations_*.log/native_allocation_profiles.json，不把其instrumentation开销混入speedup。Python/NumPy runtime的RSS不能与独立native RSS直接视为算法节省量；Linux进程high-water RSS也可能包含launcher/fork历史，实际native heap对照如下。

{table(['实现','malloc calls','累计分配bytes','peak heap bytes'],[[z['implementation'],z['malloc_calls'],z['heap_total_bytes'],z['heap_peak_bytes']]for z in allocation_rows])}

优化后没有末尾全局sort，所以production merge_ms字段为0；这不代表global heap免费。heap push/pop、容器分配和未归类host逻辑体现在ranking total减去已测cell/grid阶段的residual，单独保存于online_stage_profile.csv，不将其伪装为直接测得的纯heap耗时。

独立timer-only profile测得15% global heap push/pop合计约{heap_profile['merge_ms']:.3f}ms；只用于阶段归因，不混入speedup样本。Reference追加只读计时将membership单列，最终grid={python_phase['grid_ms']:.3f}ms、membership={python_phase['membership_ms']:.3f}ms、global merge={python_phase['merge_ms']:.3f}ms，见final_python_reference_profile.json。

![speed](offline_exact_speed.png)

## 在线成本（真实端到端）

{overhead_table}

online_stage_profile.csv包含完整Mapping/普通step/online-tail、SGI增量、full refresh、grid、membership、similarity、初始化、CELF rescore、coverage、完整ranking、teacher、Selector forward、LoD joint backward、render/TF的count/mean/median/p90/p95/p99/max。MappingProfiler给host-wall和既有sampled CUDA摘要，不把host launch gap当GPU kernel time。LoD backward是同一autograd图的Selector+TF/Gaussian联合反向，不能宣称已单独测出Selector kernel backward时间。

CPU/GPU utilization、peak RAM/VRAM及原生Torch allocator peak见online_resource_usage.csv。GPU和RAM按1Hz采样可能漏过短峰；原生CUDA allocator峰值完整保留。首个Raw run退出时monitor遇到/proc无VmRSS，保留已完成模型和全部Mapping trace，不重训；该run资源样本缺失且退出码未捕获，不能把缺失量写成0。其余run正常采样。所有计时来自同一instrumented配置；Raw也付出只读snapshot复制/导出成本，因此另报告new wrapper总时间/Mapping steps作为刷新专属摊销（包含diagnostic IO，不包括append/prune hook成本，不能当严格总开销上界）。

fr1_room完整Mapping从Raw约9.62ms/step增至约9.97ms/step，增加约0.35ms/step；其中Marginal refresh wrapper摊销约0.076–0.079ms/step。其余差值含地图规模和online/tail分配差异，不全归因于ranker。

![mapping](online_mapping_cost.png)

是否需要ApproxSparseNeighbors：本轮不进入。先依据以上exact端到端开销和质量决定；不能把语言/CPU微测通过等同production-ready。
'''
(ROOT/'GEOMETRY_MARGINAL_PERFORMANCE_REPORT.md').write_text(performance,encoding='utf-8')

answers=[
f'1–3. IsoOnly完整ranking和四档quality exact复现；pairwise redundancy不再读取view vector，Raw Streaming GI仍使用原Bank views。',
f'4–7. Reference最慢为candidate rescore；Python完整median={med["python_reference"]:.3f}ms，Optimized15%={med["optimized"]:.3f}ms、完整={med["optimized_full"]:.3f}ms；总speedup见性能报告。',
'8. largest20cell逐项前后耗时与评价次数见largest_20_cells.csv，不以单个best case代表全部。',
f'9–12. 离线lazy15%仅展开7550个selected代表，CELF评分次数减少{saved_local:.2%}；online为保留动态percentile语义仍完整展开每个snapshot的候选N，不把7550当online展开数。peak memory/allocations有独立记录；完整顺序、四档selected rows、复渲染质量及5次重复全部exact。',
'13–15. fr1_room 1/5%与fr1_desk逐seed结果如下表；同时给same-K reference，不用四档平均覆盖低预算。',
f'16–17. 对每次真实snapshot的same-K机制统计中，{(mp.Marginal<mp.Raw).mean():.2%}的state冗余下降；这是teacher ranking，不是Selector已完全学会该ranking的证明。在线最终收益按真实render表判定。',
'18–19. Full-map与SE3 ATE逐seed列出；不能将estimated pose不同造成的render差异全部归因于GI。',
'20–22. 平均/普通step overhead、full refresh p95/max和tracking分布均保留。独立日志不能逐事件证明tracking stall因果；检查了跨seed的tail latency变化和GI association wait计数。',
'23–24. exact剩余成本见stage与occupancy分桶；不因offline收益自动进入Approx Sparse Neighbor。',
f'25. 预注册统计门控：{decision}，属于受限online证据；全面production-ready尚未通过。单seed的Full/ATE回退与refresh尖峰必须保留，不能据平均收益直接默认上线。'
]
primary_means=primary.groupby('ratio').delta_psnr.mean().to_dict()
online=f'''# Geometry-only Marginal SI在线报告

预注册门控结果：**{decision}（受限证据）**。**全面production-ready尚未通过，Raw仍为默认。**

fr1_room production LoD的两seed平均ΔPSNR：1% {primary_means[.01]:+.4f}dB，5% {primary_means[.05]:+.4f}dB。收益不是每个seed/ratio均为正。尤其seed2027：Full PSNR −0.5917dB，ATE 0.05103→0.07520m（+47.36%）；两seed平均Full ΔPSNR约−0.2071dB，ATE平均增加约0.01052m。这些实际回退不能被“预注册稳定回退门控未触发”解释为不存在风险。n=2不足以判断该ATE差异是方法因果还是异步SLAM波动。

fr1_room的平均Mapping增加约3.6%，但full-refresh p95为252–283ms，最大297.5ms，显著高于Raw约97–105ms的p95。未发现跨seed一致的tracking-tail延迟恶化，仍不能据此证明不存在refresh stall。因此本轮支持“offline机制与exact实现有效，online低预算有正向信号”，不支持直接把新mode作为稳健默认。以下预注册门槛保留原样，不在看到结果后改门槛或调参。

预注册门控结果（判断跨seed的稳定退化，不是逐run无损保证）：

```json
{json.dumps(reason,ensure_ascii=False,indent=2)}
```

本轮为真实TUM RGB-D全序列SLAM，不是frozen map模拟：fr1_room 1352帧、16900 Mapping，fr1_desk 573帧、7400 Mapping；每scene两seed2026/2027，Raw/Marginal配对，总8个终点模型。另有独立fr1_room Raw-teacher Shadow run，16个1000–16000 refresh的完整ranking/gains exact一致，覆盖early/middle/late。Shadow未将新teacher用于loss，已用teacher trace检查。

固定的是normal Mapping预算和原始调度规则。异步SLAM结束时刻会改变online/tail分配；mode1尾部每个Mapping含Full+native LoD，因此实际Gaussian/Selector/TF optimizer commits不保证逐次相同。下面完整披露，不能把它写成strict same-optimizer-update实验：

{table(list(budget_rows[0]),[list(z.values())for z in budget_rows])}

最终使用用户指定的本地项目源码（服务器同步），默认原width32 Selector，两arm相同。保留production Selector/TF后端、LR/loss、maturity/protection、ChangeDistill、densification、Bank和frame_importance。GI mode是算法变量；诊断和seed入口两arm相同且默认关闭。相同seed不能消除ORB/SLAM异步非确定性，n=2不做统计显著性声明。

Raw每次bank增量正常更新；Marginal只在既有full refresh重建。h在首个full refresh初始化，之后本map固定。每snapshot文件都保存同锁内xyz/activated radius/Raw I、source row和h，CSV记录GI/topology版本，append/prune/reset测试与online assertions防止row错位；geometry cache在refresh结束释放，不在移动地图上复用旧matrix。两次refresh之间使用历史published teacher是有意staleness，不声称membership实时跟随xyz。h与当前d50关系在online_mechanism.csv逐refresh记录，不能偷偷每次改变h。

质量使用每scene四个模型共有GT帧中等距128个frame，列表在全部模型完成、读取质量之前冻结；ATE使用所有关联pose、SE3无scale correction。Primary production selection继续使用各模型自己的cached_uniform与production maturity；actual K可不同。Secondary same-K按每scene四模型最小N确定统一K，按production probability Top-K，仅为诊断。Full渲染Master不套LoD TF，LoD按原requested ratio应用TF。

## Primary production LoD质量

{quality_table}

![quality](online_quality_by_ratio.png)

## Full与ATE

{full_table}

![ate](online_ate_vs_full_quality.png)

same-K、Full-relative LoD gap、逐viewPSNR/SSIM/LPIPS、selected counts均随原始结果提供，不能把budget/maturity/pose差异伪装成pure ranking因果。

## 性能与机制

{overhead_table}

上表refresh为Raw SGI render+Marginal wrapper的连续计时和；outer render-lock等待及整个boundary GI阶段另见FullRefreshBoundaryGIIncludingLock统计。TrackingTime.txt全部原始frame都纳入本报告tracking分布；旧Python evaluator摘要的tracking均值会截去末尾3条，本报告不使用它作为tracking主值。

![mechanism](online_snapshot_redundancy.png)

在线机制逐full refresh统计candidate N、occupied cells、occupancy mean/median/max、same-K、redundancy、>0.9duplicate pairs、important mass coverage、dense-cell budget share和Jaccard，见online_mechanism.csv。teacher positive/supervised/weight、compatibility mature、TF ready、zero maturity见online_teacher.csv。当前production使用continuous maturity，没有独立binary protected population，不能把TF未ready点全部称为protected。

以下为各snapshot的macro平均，是同一snapshot内Raw-vs-Marginal teacher ranking对照，不是已训练Selector mask的冗余测量：

{table(['ranking','ratio','mean redundancy','pairs>0.9','important mass coverage','dense-cell budget share'],[[z.ranking,f'{z.ratio:.0%}',f'{z.redundancy_mean:.4f}',f'{z.duplicate_pairs_gt09:.2f}',f'{z.important_mass_coverage:.2%}',f'{z.dense_cell_budget_share:.2%}']for z in mechanism_summary.itertuples()])}

两档整体mean redundancy约下降90.9%。dense-cell budget share并未下降，1%还增加：Marginal不是density penalty，而是优先用少量代表覆盖重要mass；不能把该结果解释成“dense区域预算一律减少”。

## 最终25问

'''+ '\n\n'.join(answers)+'''

## 解释边界与停止

若离线redundancy下降但online质量不升，优先归类为online benefit weak/NO-GO；不能倒推离线Marginal数学无效。teacher学习、snapshot staleness、h初始化尺度和SLAM pose/地图差异仍可能影响online结果。本轮没有为挽救结果调整h、refresh cadence、loss、LR、网络或sampling。

生产模式保留Raw开关，默认仍是Raw。仅验证fresh-start连续sequence；checkpoint/resume下保存与恢复固定h和published marginal state未覆盖，本轮不声称全面production-ready。没有进入Sparse Approx、self-refinement、view/covariance、CUDA优化或下一轮算法设计。未commit、未push，报告完成后停止。
'''
(ROOT/'GEOMETRY_MARGINAL_ONLINE_REPORT.md').write_text(online,encoding='utf-8')
for name,h in read(ROOT/'code_sync_manifest.json')['files'].items():assert sha(R/name)==h,('FINAL_SOURCE_DRIFT',name)
save(ROOT/'geometry_marginal_results.json',dict(decision=decision,decision_scope='preregistered_online_gates_only; limited evidence',production_ready=False,production_readiness_limitations=['fr1_room seed2027 Full PSNR and ATE regression','large full-refresh latency spikes','two-seed asynchronous SLAM variance','checkpoint-resume h/publication persistence not covered'],gates=reason,offline_medians_ms=med,offline_evaluations=evals,online_quality=Q,paired_deltas=paired,overheads=overheads,resources=resources,models=models))
save(ROOT/'geometry_marginal_manifest.json',dict(online_protocol=P,analysis_preregistration=A,source_sync=read(ROOT/'code_sync_result.json'),code_sha=read(ROOT/'code_sync_manifest.json')['files'],shadow=shadow,offline=read(ROOT/'optimized_exact_validation.json'),offline_render=read(ROOT/'optimized_render_validation.json'),models=[dict(scene=m['scene'],seed=m['seed'],arm=m['arm'],checkpoint=m['checkpoint'],selector_sha=sha(Path(m['checkpoint'])/'ply/selector.pt'),TF_sha=sha(Path(m['checkpoint'])/'ply/transform_field.pt'))for m in models],executed_scripts={p.name:sha(p)for p in ROOT.glob('*.py')},commit=False,push=False))
save(ROOT/'delivery_checks.json',dict(all_passed=True,source_sync_exact=True,offline_exact=True,shadow_exact=True,models=8,quality_cases=40,all_required_reports=True,default_raw_preserved=True))
print('FINAL_RESULT',decision,reason,flush=True)
