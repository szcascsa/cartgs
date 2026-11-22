"""执行原预注册判据，补充2×2机制对照；不按结果改变算法。"""
from common import *
from evaluate import write_csv
import csv,math
assert read(ROOT/'quality_complete.json')['all_passed']and read(ROOT/'diagnostic_checks.json')['all_passed']
p=read(ROOT/'preregistration.json');ratios=p['ratios'];metric=['psnr','ssim','lpips','photo'];frames=read(ROOT/'input_manifest.json')['evaluation_frames']
q=list(csv.DictReader(open(ROOT/'quality_summary.csv')));pv=list(csv.DictReader(open(ROOT/'quality_per_view.csv')))
Q={(a['arm'],float(a['ratio'])):{m:float(a[m])for m in metric}for a in q};V={(a['arm'],float(a['ratio']),int(a['frame_index'])):a for a in pv}
comparisons=[(a,'C_Hierarchical')for a in p['new_arms']]+[('F3_ProgressiveMarginal',a)for a in ['F1_MarginalTail','F2_ProgressiveResidual','B_FixedMarginal','A_Raw']]+[('F1_MarginalTail','A_Raw'),('F2_ProgressiveResidual','A_Raw')]
differences=[];paired=[];perdelta=[]
for a,ref in comparisons:
    for ratio in ratios:
        differences.append(dict(comparison=a+'_minus_'+ref,ratio=ratio,**{'delta_'+m:Q[a,ratio][m]-Q[ref,ratio][m]for m in metric}))
        for m in metric:
            d=np.array([float(V[a,ratio,f][m])-float(V[ref,ratio,f][m])for f in frames]);np.testing.assert_allclose(d.mean(),Q[a,ratio][m]-Q[ref,ratio][m],atol=1e-10,rtol=1e-10)
            paired.append(dict(comparison=a+'_minus_'+ref,ratio=ratio,metric=m,mean=float(d.mean()),median=float(np.median(d)),p10=float(np.quantile(d,.1)),p90=float(np.quantile(d,.9)),positive_delta_fraction=float((d>0).mean()),improved_view_fraction=float(((d>0)if m in ['psnr','ssim']else(d<0)).mean())))
        for f in frames:perdelta.append(dict(comparison=a+'_minus_'+ref,ratio=ratio,frame_index=f,KF_id=V[a,ratio,f]['KF_id'],**{'delta_'+m:float(V[a,ratio,f][m])-float(V[ref,ratio,f][m])for m in metric}))
write_csv('quality_deltas.csv',differences);write_csv('paired_statistics.csv',paired);write_csv('paired_per_view_deltas.csv',perdelta)
red={(a['arm'],float(a['ratio'])):a for a in csv.DictReader(open(ROOT/'selected_redundancy.csv'))}
def dps(a,ref='C_Hierarchical'):return np.array([Q[a,r]['psnr']-Q[ref,r]['psnr']for r in ratios])
f1=dps('F1_MarginalTail');f2=dps('F2_ProgressiveResidual');f3=dps('F3_ProgressiveMarginal');fb=dps('F3_ProgressiveMarginal','B_FixedMarginal')
reduction={a:[1-float(red[a,r]['mean_redundancy'])/float(red['C_Hierarchical',r]['mean_redundancy'])for r in ratios]for a in p['new_arms']}
F1red=all(reduction['F1_MarginalTail'][i]>=.1 and int(red['F1_MarginalTail',ratios[i]]['pairs_gt_0.9'])<=int(red['C_Hierarchical',ratios[i]]['pairs_gt_0.9'])for i in [2,3])
F1quality=bool(f1[2:].mean()>=-.03);F1=bool(F1red and F1quality)
F2low=bool(f2[0]>=.05);F2high=bool(np.all(f2[1:]>=-.03));F2=F2low and F2high
F3mean=bool(f3.mean()>=0);F3low=bool(f3[0]>0);F3red=bool(all(reduction['F3_ProgressiveMarginal'][i]>0 for i in [2,3]));F3=F3mean and F3low and F3red
bmean=np.mean([float(red['B_FixedMarginal',r]['mean_redundancy'])for r in ratios]);newmean=np.mean([float(red['F3_ProgressiveMarginal',r]['mean_redundancy'])for r in ratios]);bpairs=sum(int(red['B_FixedMarginal',r]['pairs_gt_0.9'])for r in ratios);newpairs=sum(int(red['F3_ProgressiveMarginal',r]['pairs_gt_0.9'])for r in ratios)
strongPS=bool(fb.mean()>=-.02);strongRed=bool(newmean<=1.1*bmean and newpairs<=max(1,int(np.ceil(1.1*bpairs))));strong=F3 and strongPS and strongRed
factorial=[]
for ratio in ratios:
    for m in metric:
        c=Q['C_Hierarchical',ratio][m];a=Q['F1_MarginalTail',ratio][m];b=Q['F2_ProgressiveResidual',ratio][m];z=Q['F3_ProgressiveMarginal',ratio][m]
        interaction=z-b-a+c
        values=np.array([float(V['F3_ProgressiveMarginal',ratio,f][m])-float(V['F2_ProgressiveResidual',ratio,f][m])-float(V['F1_MarginalTail',ratio,f][m])+float(V['C_Hierarchical',ratio,f][m])for f in frames])
        np.testing.assert_allclose(values.mean(),interaction,atol=1e-10,rtol=1e-10)
        factorial.append(dict(ratio=ratio,metric=m,tail_without_progressive_F1_minus_C=a-c,progressive_without_tail_F2_minus_C=b-c,tail_with_progressive_F3_minus_F2=z-b,progressive_with_tail_F3_minus_F1=z-a,interaction_F3_minus_F2_minus_F1_plus_C=interaction,paired_interaction_median=float(np.median(values)),paired_interaction_p10=float(np.quantile(values,.1)),paired_interaction_p90=float(np.quantile(values,.9))))
write_csv('factorial_mechanism_contrasts.csv',factorial)
recommendation='ADOPT BOTH'if F3 else'ADOPT MARGINAL TAIL'if F1 else'ADOPT PROGRESSIVE'if F2 else'KEEP CURRENT C'
result=dict(scene=p['scene'],F1='MARGINAL TAIL SUPPORTED'if F1 else'NOT SUPPORTED',F2='PROGRESSIVE SUPPORTED'if F2 else'MIXED'if F2low or F2high else'NOT SUPPORTED',F3='COMBINED SUPPORTED'if F3 else'NOT SUPPORTED',F1_supported=F1,F2_supported=F2,F3_supported=F3,strong_improvement=bool(strong),
    gates=dict(F1_high_redundancy=bool(F1red),F1_high_mean_PSNR=F1quality,F2_1percent_PSNR=F2low,F2_high_no_regression=F2high,F3_mean_PSNR=F3mean,F3_1percent_PSNR=F3low,F3_high_redundancy=F3red,strong_B_PSNR=strongPS,strong_B_redundancy=strongRed),
    delta_PSNR_vs_C={a:dps(a).tolist()for a in p['new_arms']},mean_delta_PSNR_vs_C={a:float(dps(a).mean())for a in p['new_arms']},F1_high_ratio_mean_delta_PSNR=float(f1[2:].mean()),F3_minus_F2_PSNR=dps('F3_ProgressiveMarginal','F2_ProgressiveResidual').tolist(),F3_minus_F1_PSNR=dps('F3_ProgressiveMarginal','F1_MarginalTail').tolist(),redundancy_reduction_vs_C=reduction,
    recommendation=recommendation,recommendation_scope='offline mechanism candidate only; current production C unchanged; no online claim',runtime_is_gate=False,no_parameter_search=True,optimizer_updates=0,no_commit=True,no_push=True)
save(ROOT/'mechanism_result.json',result)
for path,v in read(ROOT/'frozen_sources.json').items():assert sha(path)==v['sha256'],path
for path,v in read(ROOT/'prior_artifact_checks.json')['files'].items():assert sha(path)==v['sha256'],path
assert sha(read(ROOT/'input_manifest.json')['cameras'])==read(ROOT/'camera_checks.json')['camera_archive_sha256']
assert sha(ROOT/'preregistration.json')==read(ROOT/'preparation.json')['preregistration_sha256']
prior=Path(read(ROOT/'preparation.json')['prior'])
for name in ['progressive_ranker.py','fixed_marginal_reference.py','common.py','evaluate.py','experiment_data.py','audit_worker']:assert sha(ROOT/name)==sha(prior/name),name
checks=dict(all_passed=True,frozen_sources_and_prior_unchanged=True,preregistration_unchanged=True,math_source_unchanged=True,camera_archive_unchanged=True,tree_sha256=sha(ROOT/'structure_run1/sparse_hash_target_2048_tree.csv'),optimizer_updates=0,baseline_reproduction=read(ROOT/'baseline_reproduction.json'),quality=read(ROOT/'quality_complete.json'))
save(ROOT/'final_checks.json',checks)
save(ROOT/'summary.json',dict(scene=p['scene'],population=read(ROOT/'population.json'),mechanism=result,quality=[dict(arm=a['arm'],ratio=float(a['ratio']),K=int(a['K']),views=int(a['views']),**{m:float(a[m])for m in metric})for a in q],checks=checks))
print('MECHANISM_RESULT',json.dumps(result),flush=True)
