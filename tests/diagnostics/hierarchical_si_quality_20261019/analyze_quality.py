"""执行预注册quality gates与paired统计，不更改任何算法/排序。"""
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
assert read('quality_complete.json')['all_passed'] and read('baseline_reproduction.json')['all_passed']
p=read('preregistration.json');ratios=p['ratios'];metrics=['psnr','ssim','lpips','photo'];quality=rows('quality_summary.csv');views=rows('quality_per_view.csv')
Q={(a['arm'],float(a['ratio'])):{m:float(a[m])for m in metrics}for a in quality}
V={(a['arm'],float(a['ratio']),int(a['frame_index'])):a for a in views};frames=read('input_manifest.json')['evaluation_frames']
comparisons=[('C_vs_Raw','C_Hierarchical','A_Raw'),('C_vs_FixedMarginal','C_Hierarchical','B_FixedMarginal'),('D_vs_C','D_RepresentativeOnly','C_Hierarchical'),('C_vs_L1Only','C_Hierarchical','E_L1Only')]
deltas=[];paired=[];view_delta=[]
for name,a,b in comparisons:
    for ratio in ratios:
        K=int(ratio*read('ranking_sanity.json')['N_candidate']+.5)
        deltas.append({'comparison':name,'ratio':ratio,'K':K,**{'delta_'+m:Q[a,ratio][m]-Q[b,ratio][m]for m in metrics}})
        for m in metrics:
            d=np.array([float(V[a,ratio,f][m])-float(V[b,ratio,f][m])for f in frames]);assert len(d)==161
            paired.append({'comparison':name,'ratio':ratio,'metric':m,'views':len(d),'mean':float(d.mean()),'median':float(np.median(d)),'p10':float(np.quantile(d,.1)),'p90':float(np.quantile(d,.9)),'positive_delta_fraction':float((d>0).mean()),'improved_view_fraction':float(((d>0)if m in ['psnr','ssim']else(d<0)).mean())})
        for f in frames:view_delta.append({'comparison':name,'ratio':ratio,'frame_index':f,'KF_id':V[a,ratio,f]['KF_id'],**{'delta_'+m:float(V[a,ratio,f][m])-float(V[b,ratio,f][m])for m in metrics}})
write('quality_deltas.csv',deltas);write('paired_per_view_statistics.csv',paired);write('paired_per_view_deltas.csv',view_delta)
wide=[]
for ratio in ratios:
    record={'ratio':ratio,'K':int(ratio*read('ranking_sanity.json')['N_candidate']+.5)}
    for m in metrics:
        for label,arm in [('Raw','A_Raw'),('FixedMarginal','B_FixedMarginal'),('Hierarchical','C_Hierarchical')]:record[label+'_'+m]=Q[arm,ratio][m]
        record['Hierarchical_minus_Raw_'+m]=Q['C_Hierarchical',ratio][m]-Q['A_Raw',ratio][m]
        record['Hierarchical_minus_FixedMarginal_'+m]=Q['C_Hierarchical',ratio][m]-Q['B_FixedMarginal',ratio][m]
    wide.append(record)
write('primary_quality_table.csv',wide)
def delta(a,b):return {m:np.array([Q[a,r][m]-Q[b,r][m]for r in ratios])for m in metrics}
CR=delta('C_Hierarchical','A_Raw');CB=delta('C_Hierarchical','B_FixedMarginal');DC=delta('D_RepresentativeOnly','C_Hierarchical');CE=delta('C_Hierarchical','E_L1Only')
def low_gate(d):return bool(d[:2].mean()>.10 or (d[0]>.20 and d[1]>=-.02) or (d[1]>.20 and d[0]>=-.02))
ssim_tol=p['metric_interpretation_fixed_before_render']['substantive_SSIM_drop'];lpips_tol=p['metric_interpretation_fixed_before_render']['substantive_LPIPS_worsening']
def tradeoff(d):return bool(((d['psnr']>0)&(d['ssim']<-ssim_tol)&(d['lpips']>lpips_tol)).sum()>=2)
mixed=(CR['psnr']>0)&((CR['ssim']<-ssim_tol)|(CR['lpips']>lpips_tol))
gateA=bool((CR['psnr']>=-.02).sum()>=3);gateB=low_gate(CR['psnr']);gateC=not tradeoff(CR)
mixed_downgrade=bool(mixed[:2].any()or mixed.sum()>=2)
primary='QUALITY GO'if gateA and gateB and gateC and not mixed_downgrade else 'QUALITY INCONCLUSIVE'if gateA and gateB and gateC else 'QUALITY NO-GO'
near=bool(CB['psnr'].mean()>=-.05 or (np.all(CB['psnr'][:2]>=-.10)and not tradeoff(CB)))
paired_index={(a['comparison'],a['ratio'],a['metric']):a for a in paired}
red={(a['arm'],float(a['ratio'])):a for a in rows('selected_redundancy.csv')}
red_gate=all(float(red['C_Hierarchical',r]['mean'])<=.9*float(red['A_Raw',r]['mean'])and int(red['C_Hierarchical',r]['unordered_pairs_gt_0.9'])<int(red['A_Raw',r]['unordered_pairs_gt_0.9'])for r in ratios[:2])
strong_gates={'all_PSNR_nonnegative_or_equivalent':bool(np.all(CR['psnr']>=-.02)),'low_ratio_gain':gateB,'LPIPS_majority_improved':bool((CR['lpips']<0).sum()>=3),'paired_low_bias_positive':all(paired_index['C_vs_Raw',r,'psnr']['improved_view_fraction']>=.60 and paired_index['C_vs_Raw',r,'psnr']['median']>0 for r in ratios[:2]),'redundancy_reduced':bool(red_gate),'near_or_above_FixedMarginal':near}
strong=bool(primary=='QUALITY GO'and all(strong_gates.values()))
quantitative_primary=primary;quantitative_strong=strong
# The user's original rule is stricter than the first quantitative implementation:
# mixed PSNR/SSIM/LPIPS direction must downgrade the conclusion, including high ratios.
# Preserve the preregistration and numerical gates; correct the interpretation without tuning.
direction_mix=(CR['psnr']>0)&((CR['ssim']<0)|(CR['lpips']>0))
user_mixed_guard=bool(direction_mix.any())
if primary=='QUALITY GO'and user_mixed_guard:primary='QUALITY INCONCLUSIVE';strong=False
caseA=quantitative_primary=='QUALITY GO'and near;caseB=quantitative_primary=='QUALITY GO'and not near;caseC=bool(np.all(np.abs(CR['psnr'])<=.02));caseD=bool(CR['psnr'][:2].mean()<-.02 or(CR['psnr'].mean()<-.02 and(CR['psnr']<-.02).sum()>=3))
caseE=low_gate(DC['psnr'])and not tradeoff(DC);caseF=bool(np.all(np.abs(CE['psnr'])<=.02)and np.all(np.abs(CE['ssim'])<=.002)and np.all(np.abs(CE['lpips'])<=.005))
cases={k:bool(v)for k,v in zip('ABCDEF',[caseA,caseB,caseC,caseD,caseE,caseF])}
if caseA and user_mixed_guard:cases['A']='CONDITIONAL'
low_composition=[a for a in rows('hierarchical_topk_composition.csv')if a['arm']=='C_Hierarchical'and float(a['ratio'])in ratios[:2]]
case_label=('+'.join('Case '+k for k,v in cases.items()if v)or 'Mixed: no single predefined A-D class')+(' (conditional: mixed metrics)'if user_mixed_guard else '')
result={'primary_decision':primary,'Strong_GO':strong,'quantitative_primary_before_user_mixed_guard':quantitative_primary,'quantitative_Strong_criteria_pass':quantitative_strong,'primary_gates':{'A':gateA,'B':gateB,'C_no_joint_systematic_tradeoff':gateC,'quantitative_mixed_downgrade':mixed_downgrade,'user_required_direction_mixed_downgrade':user_mixed_guard},'strong_gates':strong_gates,'cases':cases,'case_label':case_label,
        'Q1':'YES'if primary=='QUALITY GO'else 'NO'if primary=='QUALITY NO-GO'else 'INCONCLUSIVE',
        'Q2':'YES'if low_gate(CR['psnr'])and all(paired_index['C_vs_Raw',r,'psnr']['improved_view_fraction']>=.60 for r in ratios[:2])else 'MIXED'if np.any(CR['psnr'][:2]>0)else 'NO',
        'Q3':'EXCEEDS'if np.all(CB['psnr']>=0)and CB['psnr'].mean()>.05 else 'MATCHES'if near else 'BELOW',
        'Q4':'YES'if all(float(a['representative_fraction'])>=.9 for a in low_composition)else 'MIXED',
        'Q5':'NO'if caseF else 'YES'if ((CE['psnr']>.02)&(CE['lpips']<0)).sum()>=2 else 'NO',
        'Q6':case_label,'Q7':primary,
        'low_mean_delta_PSNR':float(CR['psnr'][:2].mean()),'mean_C_minus_B_PSNR':float(CB['psnr'].mean()),'diagnostic_D_minus_C_PSNR':DC['psnr'].tolist(),'diagnostic_C_minus_E_PSNR':CE['psnr'].tolist(),
        'mixed_ratios':[r for r,v in zip(ratios,direction_mix)if v],
        'decision_correction':'Original user clause requires downgrade on mixed PSNR/SSIM/LPIPS direction. Initial quantitative implementation was too narrow; retain original preregistration and all numerical gates, apply stricter original clause. No parameter, ranking or quality result changed.',
        'interpretation_limits':['Q3 MATCHES means predefined PSNR proximity, not exact selected-set or all-metric equivalence','Q5 YES is ratio-dependent observed value; report 1% regression explicitly','Single frozen scene; archived views participated in original SLAM; no online or production claim','Case E is not supported by these data; do not change residual']}
save('mechanism_cases.json',result)
print(json.dumps(result,indent=2,default=lambda v:v.item()),flush=True)
