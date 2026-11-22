"""独立CSV求和/paired统计、complete ranking及冻结本地源码复核。"""
from pathlib import Path
import csv,json,hashlib,math,statistics,subprocess
import numpy as np
TASK=Path(__file__).resolve().parent;REPO=TASK.parent/'mycartgs';ROOT=REPO/'tests/diagnostics/hierarchical_si_quality_20261019'
def read(n):return json.loads((ROOT/n).read_text())
def rows(n):return list(csv.DictReader(open(ROOT/n)))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for n,d in read('artifact_manifest.json').items():assert sha(ROOT/n)==d,n
for n,d in read('audit_before.json')['frozen_local_source'].items():assert sha(REPO/n)==d,n
data=rows('quality_per_view.csv');summary=rows('quality_summary.csv');assert len(data)==3220 and len(summary)==20
keys={(r['arm'],float(r['ratio']),int(r['frame_index'])):r for r in data};assert len(keys)==3220
for a in summary:
    for m in ['psnr','ssim','lpips','photo']:
        v=[float(r[m])for r in data if r['arm']==a['arm']and r['ratio']==a['ratio']]
        assert len(v)==161 and abs(math.fsum(v)/161-float(a[m]))<1e-12
comparisons={'C_vs_Raw':('C_Hierarchical','A_Raw'),'C_vs_FixedMarginal':('C_Hierarchical','B_FixedMarginal'),'D_vs_C':('D_RepresentativeOnly','C_Hierarchical'),'C_vs_L1Only':('C_Hierarchical','E_L1Only')}
frames=read('input_manifest.json')['evaluation_frames']
for a in rows('paired_per_view_statistics.csv'):
    first,second=comparisons[a['comparison']];ratio=float(a['ratio']);m=a['metric']
    d=sorted(float(keys[first,ratio,f][m])-float(keys[second,ratio,f][m])for f in frames)
    assert abs(math.fsum(d)/161-float(a['mean']))<1e-12
    assert abs(statistics.median(d)-float(a['median']))<1e-12
    assert abs(d[16]-float(a['p10']))<1e-12 and abs(d[144]-float(a['p90']))<1e-12
    improved=sum(v>0 if m in ['psnr','ssim']else v<0 for v in d)/161
    assert abs(improved-float(a['improved_view_fraction']))<1e-14
r=np.load(ROOT/'arm_rankings.npz');sel=np.load(ROOT/'selected_rows.npz');N=len(r['row_ids'])
for arm in read('preregistration.json')['primary']+read('preregistration.json')['diagnostic']:
    order=r[arm+'_order'];assert len(order)==N and sorted(order)==list(range(N));previous=set()
    for ratio in [.01,.05,.1,.15]:
        K=int(ratio*N+.5);ids=r['row_ids'][order[:K]];assert previous<=set(ids);previous=set(ids)
        np.testing.assert_array_equal(np.sort(ids),sel[arm+'_'+str(ratio)])
Q={(a['arm'],float(a['ratio'])):a for a in summary}
for ratio in [.01,.05]:
    c=Q['C_Hierarchical',ratio];a=Q['A_Raw',ratio]
    assert float(c['psnr'])>float(a['psnr'])and float(c['ssim'])>float(a['ssim'])
    assert float(c['lpips'])<float(a['lpips'])and float(c['photo'])<float(a['photo'])
for ratio in [.1,.15]:
    c=Q['C_Hierarchical',ratio];a=Q['A_Raw',ratio]
    assert float(c['psnr'])>float(a['psnr'])and float(c['lpips'])<float(a['lpips'])
    assert float(c['ssim'])<float(a['ssim'])and float(c['photo'])>float(a['photo'])
assert read('mechanism_cases.json')['primary_decision']=='QUALITY INCONCLUSIVE'
assert not read('mechanism_cases.json')['Strong_GO']
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).strip()
subprocess.run(['git','diff','--check'],cwd=REPO,check=True)
result={'all_passed':True,'independent_aggregate_and_paired_checks':True,'complete_ranking_K_nested':True,'frozen_local_source_unchanged':True,
        'low_ratio_four_metrics_improve':True,'high_ratio_SSIM_photo_regress':True,'verdict':'QUALITY INCONCLUSIVE','staged_files':0}
(ROOT/'local_independent_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
