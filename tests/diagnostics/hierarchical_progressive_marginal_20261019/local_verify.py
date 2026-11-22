"""Windows上独立aggregate/paired/gate核对，并检查production源码零修改。"""
from pathlib import Path
import json,csv,hashlib,math,statistics,subprocess
import numpy as np
TASK=Path(__file__).resolve().parent;REPO=TASK.parent/'mycartgs';ROOT=REPO/'tests/diagnostics/hierarchical_progressive_marginal_20261019'
def read(n):return json.loads((ROOT/n).read_text())
def rows(n):return list(csv.DictReader(open(ROOT/n)))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,digest in read('artifact_manifest.json').items():assert sha(ROOT/n)==digest,n
for n,digest in read('audit_before.json')['frozen_local_source'].items():assert sha(REPO/n)==digest,n
assert sha(ROOT/'mechanism_preregistration.json')=='a094f25781f8f5834beab3b21880fa16a01b4facababb12c3d526218505d4021'
q=rows('progressive_marginal_quality_summary.csv');pv=rows('progressive_marginal_per_view.csv');assert len(q)==24 and len(pv)==3864
Q={(a['arm'],float(a['ratio'])):a for a in q};V={(a['arm'],float(a['ratio']),int(a['frame_index'])):a for a in pv};assert len(V)==3864
for a in q:
    group=[v for v in pv if v['arm']==a['arm']and v['ratio']==a['ratio']];assert len(group)==161
    for m in ['psnr','ssim','lpips','photo']:assert abs(math.fsum(float(v[m])for v in group)/161-float(a[m]))<1e-12
for a in rows('paired_statistics.csv'):
    first,second=a['comparison'].split('_minus_');r=float(a['ratio']);m=a['metric'];d=sorted(float(V[first,r,f][m])-float(V[second,r,f][m])for f in read('input_manifest.json')['evaluation_frames'])
    assert abs(math.fsum(d)/161-float(a['mean']))<1e-12 and abs(statistics.median(d)-float(a['median']))<1e-12
    assert abs(d[16]-float(a['p10']))<1e-12 and abs(d[144]-float(a['p90']))<1e-12
    assert abs(sum(v>0 if m in ['psnr','ssim']else v<0 for v in d)/161-float(a['improved_view_fraction']))<1e-14
ranks=np.load(ROOT/'new_arm_rankings.npz');base=np.load(ROOT/'arm_rankings.npz');N=len(ranks['row_ids']);assert N==50330
ratios=[.01,.05,.1,.15]
for a in read('preregistration.json')['new_arms']:
    o=ranks[a+'_order'];np.testing.assert_array_equal(np.sort(o),np.arange(N));previous=set()
    for r in ratios:
        current=set(o[:int(r*N+.5)]);assert previous<=current;previous=current
    np.testing.assert_array_equal(np.lexsort((ranks['row_ids'],-ranks[a+'_score'])),o)
red={(a['arm'],float(a['ratio'])):a for a in rows('selected_redundancy.csv')}
def delta(a,b='C_Hierarchical'):return [float(Q[a,r]['psnr'])-float(Q[b,r]['psnr'])for r in ratios]
F1=all(float(red['F1_MarginalTail',r]['mean_redundancy'])<=.9*float(red['C_Hierarchical',r]['mean_redundancy'])and int(red['F1_MarginalTail',r]['pairs_gt_0.9'])<=int(red['C_Hierarchical',r]['pairs_gt_0.9'])for r in [.1,.15])and statistics.mean(delta('F1_MarginalTail')[2:])>=-.03
F2=delta('F2_ProgressiveResidual')[0]>=.05 and all(v>=-.03 for v in delta('F2_ProgressiveResidual')[1:])
F3=statistics.mean(delta('F3_ProgressiveMarginal'))>=0 and delta('F3_ProgressiveMarginal')[0]>0 and all(float(red['F3_ProgressiveMarginal',r]['mean_redundancy'])<float(red['C_Hierarchical',r]['mean_redundancy'])for r in [.1,.15])
M=read('mechanism_result.json');assert not F1 and F2 and F3
assert M['F1']=='NOT SUPPORTED'and M['F2']=='PROGRESSIVE SUPPORTED'and M['F3']=='COMBINED SUPPORTED'and not M['strong_improvement']
for a in ['new_ranking_checks.json','independent_replay_checks.json','final_validation.json']:assert read(a)['all_passed']
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();assert head==read('audit_before.json')['HEAD']
assert not subprocess.check_output(['git','diff','--name-only'],cwd=REPO,text=True).strip()
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).strip()
subprocess.run(['git','diff','--check'],cwd=REPO,check=True)
result={'all_passed':True,'HEAD_unchanged':head,'production_source_unchanged':True,'pre_registration_unchanged':True,'independent_aggregate_paired_and_gates':True,'same_K_and_nested':True,'F1':F1,'F2':F2,'F3':F3,'committed':False,'pushed':False}
(ROOT/'local_independent_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
