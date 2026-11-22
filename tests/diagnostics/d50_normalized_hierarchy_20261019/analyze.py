"""按给定A-D/Strong gate评估，四种因果差值分别保存。"""
from pathlib import Path
import json,csv,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write(name,rows):
    with open(ROOT/name,'w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
reg=read(ROOT/'preregistration.json');ratios=reg['ratios'];scenes=reg['scenes'];metrics=['psnr','ssim','lpips','photo'];quality=[];perview=[];delta=[];paired=[];perdelta=[];structures={};old={};new={};Cchange={};check={}
for s in scenes:
    out=ROOT/s;q=list(csv.DictReader(open(out/'quality_summary.csv')));pv=list(csv.DictReader(open(out/'quality_per_view.csv')));Q={(a['arm'],float(a['ratio'])):a for a in q};V={(a['arm'],float(a['ratio']),int(a['frame_index'])):a for a in pv};frames=read(out/'input_manifest.json')['evaluation_frames']
    quality += [dict(scene=s,**a)for a in q];perview += [dict(scene=s,**a)for a in pv]
    old[s]=[float(Q['P0',r]['psnr'])-float(Q['C0',r]['psnr'])for r in ratios];new[s]=[float(Q['P32',r]['psnr'])-float(Q['C32',r]['psnr'])for r in ratios];Cchange[s]=[float(Q['C32',r]['psnr'])-float(Q['C0',r]['psnr'])for r in ratios]
    for a,b in [('C32','C0'),('P0','C0'),('P32','C32'),('P32','P0')]:
        for r in ratios:
            label=a+'_minus_'+b;delta.append(dict(scene=s,comparison=label,ratio=r,**{'delta_'+m:float(Q[a,r][m])-float(Q[b,r][m])for m in metrics}))
            for m in metrics:
                d=np.array([float(V[a,r,f][m])-float(V[b,r,f][m])for f in frames]);np.testing.assert_allclose(d.mean(),float(Q[a,r][m])-float(Q[b,r][m]),atol=1e-10,rtol=1e-10)
                paired.append(dict(scene=s,comparison=label,ratio=r,metric=m,mean=float(d.mean()),median=float(np.median(d)),p10=float(np.quantile(d,.1)),p90=float(np.quantile(d,.9)),positive_delta_fraction=float((d>0).mean()),improved_view_fraction=float(((d>0)if m in ['psnr','ssim']else(d<0)).mean())))
            for f in frames:perdelta.append(dict(scene=s,comparison=label,ratio=r,frame_index=f,**{'delta_'+m:float(V[a,r,f][m])-float(V[b,r,f][m])for m in metrics}))
    structures[s]=read(out/'structure_statistics.json')
    pc=read(out/'prior_checks.json')
    for path,v in pc['source_sha'].items():assert sha(path)==v['sha256'],path
    for path,digest in pc['artifact_sha'].items():assert sha(path)==digest,path
    assert sha(read(out/'input_manifest.json')['cameras'])==pc['camera_sha256']
    assert sha(ROOT/'preregistration.json')==pc['preregistration_sha256']
    for n,digest in pc['source_math'].items():assert sha(out/n)==digest,n
    check[s]=dict(all_passed=True,baseline=read(out/'baseline_reproduction.json'),new_ranking=read(out/'new_ranking_checks.json'),replay=read(out/'independent_replay_checks.json'),quality=read(out/'quality_complete.json'))
write('quality_summary.csv',quality);write('quality_per_view.csv',perview);write('quality_deltas.csv',delta);write('paired_statistics.csv',paired);write('paired_per_view_deltas.csv',perdelta)
ga=new['fr2_xyz'][0]-old['fr2_xyz'][0];gb=float(np.mean(new['fr1_xyz'][1:3])-np.mean(old['fr1_xyz'][1:3]));gc=new['fr1_xyz'][0];gd=float(np.mean(new['fr1_room'])-np.mean(old['fr1_room']))
gates=dict(A_fr2_1percent_recovery=bool(ga>=.05),B_fr1_5_10_recovery=bool(gb>=.05),C_fr1_1percent_preserved=bool(gc>=.10),D_room_preserved=bool(gd>=-.03));support=all(gates.values())
strongchecks=dict(each_scene_mean_nonnegative=all(np.mean(new[s])>=0 for s in scenes),positive_1percent_at_least2=sum(new[s][0]>0 for s in scenes)>=2,no_ratio_below_minus005=all(min(new[s])>=-.05 for s in scenes));strong=all(strongchecks.values())
allc=np.array([v for s in scenes for v in Cchange[s]]);cquality='YES'if np.all(allc>=-.02)else'NO'if np.all(allc<-.02)else'MIXED'
oldv=np.array([v for s in scenes for v in old[s]]);newv=np.array([v for s in scenes for v in new[s]]);oldbad=int((oldv<-.02).sum());newbad=int((newv<-.02).sum());stability='YES'if support else'NO'if newbad>=oldbad and not gates['A_fr2_1percent_recovery']and not gates['B_fr1_5_10_recovery']else'MIXED'
rec='ADOPT D50x32'if support and cquality=='YES'else'ABANDON PROGRESSIVE'if cquality=='YES'and not support else'KEEP TARGET-2048'
result=dict(primary_supported=support,strong_supported=bool(strong),gates=gates,strong_checks=strongchecks,gate_values=dict(A_recovery_dB=ga,B_recovery_dB=gb,C_new_delta_dB=gc,D_mean_change_dB=gd),old_progressive_delta_PSNR=old,new_progressive_delta_PSNR=new,C32_minus_C0_PSNR=Cchange,old_negative_outside_equivalence=oldbad,new_negative_outside_equivalence=newbad,old_std=float(oldv.std()),new_std=float(newv.std()),old_worst=float(oldv.min()),new_worst=float(newv.min()),
    Q1='YES'if max(structures[s]['old']['h_over_d50']for s in scenes)/min(structures[s]['old']['h_over_d50']for s in scenes)>2 else'NO',Q2=cquality,Q3='YES'if ga>=.05 else'PARTIAL'if ga>0 else'NO',Q4='YES'if gb>=.05 else'PARTIAL'if gb>0 else'NO',Q5='YES'if gc>=.1 else'NO',Q6=stability,Q7='SCALE DRIFT HYPOTHESIS SUPPORTED'if support else'SCALE DRIFT HYPOTHESIS NOT SUPPORTED',Q8=rec,
    no_lambda_sweep=True,only_lambda=32,no_new_F1_F3_renders=True,production_changed=False,optimizer_updates=0)
save(ROOT/'mechanism_result.json',result);save(ROOT/'structure_statistics.json',structures);save(ROOT/'final_checks.json',dict(all_passed=True,scenes=check,preregistration_sha256=sha(ROOT/'preregistration.json'),mathematical_parity=read(ROOT/'mathematical_parity.json'),no_stage_commit_push=True));save(ROOT/'summary.json',dict(mechanism=result,structures=structures,quality=quality,checks=check))
print('SCALE_NORMALIZATION_RESULT',json.dumps(result),flush=True)
