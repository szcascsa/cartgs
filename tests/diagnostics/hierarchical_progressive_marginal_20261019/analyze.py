"""逐条执行已注册机制gate；高budget SSIM/photo只记录，不额外调gate。"""
from pathlib import Path
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(n):return json.loads((ROOT/n).read_text())
def rows(n):return list(csv.DictReader(open(ROOT/n)))
def write(n,a):
    with open(ROOT/n,'w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(a[0]));w.writeheader();w.writerows(a)
def save(n,v):(ROOT/n).write_text(json.dumps(v,indent=2,allow_nan=False,default=lambda v:v.item())+'\n')
assert read('quality_complete.json')['all_passed']and read('diagnostic_checks.json')['all_passed']
p=read('preregistration.json');ratios=p['ratios'];q=rows('progressive_marginal_quality_summary.csv');pv=rows('progressive_marginal_per_view.csv');metric=['psnr','ssim','lpips','photo'];Q={(a['arm'],float(a['ratio'])):{m:float(a[m])for m in metric}for a in q};V={(a['arm'],float(a['ratio']),int(a['frame_index'])):a for a in pv};frames=read('input_manifest.json')['evaluation_frames']
comparisons=[(a,'C_Hierarchical')for a in p['new_arms']]+[('F3_ProgressiveMarginal',a)for a in ['F1_MarginalTail','F2_ProgressiveResidual','B_FixedMarginal','A_Raw']]
differences=[];paired=[];perdelta=[]
for first,second in comparisons:
    for ratio in ratios:
        differences.append({'comparison':first+'_minus_'+second,'ratio':ratio,**{'delta_'+m:Q[first,ratio][m]-Q[second,ratio][m]for m in metric}})
        for m in metric:
            d=np.array([float(V[first,ratio,f][m])-float(V[second,ratio,f][m])for f in frames]);assert len(d)==161
            paired.append({'comparison':first+'_minus_'+second,'ratio':ratio,'metric':m,'mean':float(d.mean()),'median':float(np.median(d)),'p10':float(np.quantile(d,.1)),'p90':float(np.quantile(d,.9)),'positive_delta_fraction':float((d>0).mean()),'improved_view_fraction':float(((d>0)if m in ['psnr','ssim']else(d<0)).mean())})
        for f in frames:perdelta.append({'comparison':first+'_minus_'+second,'ratio':ratio,'frame_index':f,'KF_id':V[first,ratio,f]['KF_id'],**{'delta_'+m:float(V[first,ratio,f][m])-float(V[second,ratio,f][m])for m in metric}})
write('quality_deltas.csv',differences);write('paired_statistics.csv',paired);write('paired_per_view_deltas.csv',perdelta)
red={(a['arm'],float(a['ratio'])):a for a in rows('selected_redundancy.csv')}
def dps(a,ref='C_Hierarchical'):return np.array([Q[a,r]['psnr']-Q[ref,r]['psnr']for r in ratios])
f1=dps('F1_MarginalTail');f2=dps('F2_ProgressiveResidual');f3=dps('F3_ProgressiveMarginal');fb=dps('F3_ProgressiveMarginal','B_FixedMarginal')
reduction={a:[1-float(red[a,r]['mean_redundancy'])/float(red['C_Hierarchical',r]['mean_redundancy'])for r in ratios]for a in p['new_arms']}
F1red=all(reduction['F1_MarginalTail'][i]>=.1 and int(red['F1_MarginalTail',ratios[i]]['pairs_gt_0.9'])<=int(red['C_Hierarchical',ratios[i]]['pairs_gt_0.9'])for i in [2,3])
F1quality=bool(f1[2:].mean()>=-.03);F1=bool(F1red and F1quality)
F2low=bool(f2[0]>=.05);F2high=bool(np.all(f2[1:]>=-.03));F2=bool(F2low and F2high)
F3mean=bool(f3.mean()>=0);F3low=bool(f3[0]>0);F3red=bool(all(reduction['F3_ProgressiveMarginal'][i]>0 for i in [2,3]));F3=F3mean and F3low and F3red
bmean=np.mean([float(red['B_FixedMarginal',r]['mean_redundancy'])for r in ratios]);newmean=np.mean([float(red['F3_ProgressiveMarginal',r]['mean_redundancy'])for r in ratios])
bpairs=sum(int(red['B_FixedMarginal',r]['pairs_gt_0.9'])for r in ratios);newpairs=sum(int(red['F3_ProgressiveMarginal',r]['pairs_gt_0.9'])for r in ratios)
strongPS=bool(fb.mean()>=-.02);strongRed=bool(newmean<=1.1*bmean and newpairs<=max(1,int(np.ceil(1.1*bpairs))));strong=F3 and strongPS and strongRed
q1='YES'if F1 and f1[2:].mean()>0 else'MIXED'if F1red or f1[2:].mean()>0 else'NO'
q2='YES'if F2low and f2[0]>=.09920282541594894 else'PARTIAL'if f2[0]>0 else'NO'
q3='NO'if np.all(f2[1:]>=-.03)else'YES'if np.all(f2[1:]<-.03)else'MIXED'
q4='YES'if F3 and np.all(f3>=0)else'MIXED'if f3.mean()>=0 else'NO';q5='EXCEEDS'if fb.mean()>=.02 else'MATCHES'if fb.mean()>=-.02 else'BELOW'
q6='both'if F1 and F2 else'residual ordering'if F1 else'depth scheduling'if F2 else'neither'
q7='ADOPT BOTH'if F3 else'ADOPT MARGINAL TAIL'if F1 else'ADOPT PROGRESSIVE'if F2 else'KEEP CURRENT C'
coverage=rows('spatial_budget_coverage.csv');cov={(a['arm'],float(a['ratio'])):a for a in coverage}
oldE=next(a for a in csv.DictReader(open('/home/prmi/zwj/hierarchical-si-quality-audit-20261019/quality_summary.csv'))if a['arm']=='E_L1Only'and float(a['ratio'])==.01)
result={'F1':'MARGINAL TAIL SUPPORTED'if F1 else'NOT SUPPORTED','F2':'PROGRESSIVE SUPPORTED'if F2 else'MIXED'if F2low or F2high else'NOT SUPPORTED','F3':'COMBINED SUPPORTED'if F3 else'NOT SUPPORTED',
    'strong_improvement':bool(strong),'gates':{'F1_high_redundancy':bool(F1red),'F1_high_mean_PSNR':F1quality,'F2_1percent_PSNR':F2low,'F2_high_no_regression':F2high,'F3_mean_PSNR':F3mean,'F3_1percent_PSNR':F3low,'F3_high_redundancy':F3red,'strong_B_PSNR':strongPS,'strong_B_redundancy':strongRed},
    'delta_PSNR_vs_C':{a:dps(a).tolist()for a in p['new_arms']},'delta_PSNR_F3_vs_B':fb.tolist(),'F3_mean_delta_C_PSNR':float(f3.mean()),'F3_mean_delta_B_PSNR':float(fb.mean()),'redundancy_reduction_vs_C':reduction,
    'strong_B_redundancy_detail':{'B_mean_of_four':bmean,'F3_mean_of_four':newmean,'B_pairs_gt09_total':bpairs,'F3_pairs_gt09_total':newpairs,'allowed_multiplier':1.1},
    'Q1':q1,'Q2':q2,'Q3':q3,'Q4':q4,'Q5':q5,'Q6':q6,'Q7':q7,'recommendation_scope':'offline mechanism only; no production change or online claim',
    'F2_vs_prior_L1only_1pct_PSNR':float(Q['F2_ProgressiveResidual',.01]['psnr']-float(oldE['psnr'])),
    'one_percent_spatial':{a:{k:float(cov[a,.01][k])for k in ['selected_roots','selected_final_leaves','hit_root_Raw_mass_fraction','hit_leaf_Raw_mass_fraction']}for a in ['C_Hierarchical','F2_ProgressiveResidual','F3_ProgressiveMarginal']},
    'high_budget_metric_policy':'Current task records SSIM/photo without a new universal downgrade gate; all differences saved. Previous overall QUALITY INCONCLUSIVE baseline remains historically correct.',
    'no_parameter_search':True,'no_commit':True,'no_push':True}
save('mechanism_result.json',result);print(json.dumps(result,indent=2,default=lambda v:v.item()),flush=True)
