"""只读复用已验收冻结诊断，核查教师是否选入零/低Full贡献点。"""
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[3]
REF=Path('/home/prmi/zwj/experiments/li_teacher_frozen_20261106')
sys.path.insert(0,str(REF/'source/research/li_teacher_frozen_20261106'))
import evaluate as e
import numpy as np
import torch


def quantiles(values):
    return dict(zip(['min','p10','p50','p90','max'],map(float,np.quantile(values,[0,.1,.5,.9,1]))))


@torch.no_grad()
def main():
    assert ROOT==Path('/home/prmi/zwj/experiments/li_teacher_contribution_audit_20261106')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    assert json.loads((REF/'reports/FINAL_INTEGRITY.json').read_text())['passed']
    inputs=json.loads((REF/'reports/INPUTS_BEFORE.json').read_text())
    inputs[str(REF/'build/li_teacher_frozen.so')]=json.loads((REF/'reports/BUILD.json').read_text())['sha256']
    inputs[str(REF/'evaluation/reports/PER_VIEW.csv')]=e.sha(REF/'evaluation/reports/PER_VIEW.csv')
    assert all(e.sha(Path(n))==h for n,h in inputs.items())
    e.save(ROOT/'reports/INPUTS_BEFORE.json',inputs)
    config=json.loads((REF/'configs/experiment.json').read_text())
    scene=e.BudgetScene(config['checkpoint'],config['dataset'],5000,0,None,'pytorch')
    model=scene.model;pc=scene.gaussians;anchors=model.anchors;binding=e.load_binding()
    gi=torch.jit.load(str(Path(config['checkpoint'])/'flexgs_importance.pt'),map_location='cuda').state_dict()['0']
    with (REF/'evaluation/reports/PER_VIEW.csv').open() as f:
        prior={(int(r['frame']),float(r['ratio']),r['selection']):r for r in csv.DictReader(f) if r['compensation']=='False'}
    assert len(prior)==4752
    all_rows=[];absolute=defaultdict(list);relative=defaultdict(list);started=time.monotonic();changed=defaultdict(int)
    out=ROOT/'evaluation/PER_VIEW.csv';assert not out.exists()
    with out.open('w',newline='') as f:
        writer=None
        for frame,v in enumerate(scene.views):
            c=scene.calibration
            camera=e.MiniCam(c['width'],c['height'],e.focal2fov(c['fx'],c['width']),e.focal2fov(c['fy'],c['height']),e.np.linalg.inv(e.np.asarray(v['pose'])))
            q=e.native_render(binding,camera,scene.full_data,scene.background,training=True)[2]/float(c['height']*c['width'])
            visible=e.anchor_frustum_mask(anchors.centers,anchors.offset_scales,camera.full_proj_transform,scene.renderer.occupied)
            rows=torch.nonzero(visible[anchors.rows]).flatten();compact=e.compact_gaussians(pc,rows)
            eligible=e.slot_eligibility(model,compact.get_opacity,rows)
            pool=q[rows][eligible];mean=pool.mean();positive=int((pool>0).sum());assert float(mean)>0
            for ratio in [.01,.05]:
                teacher=binding.teacher(q[rows],gi[rows],rows,eligible,ratio);assert teacher is not None
                for selection in ['base','view','teacher']:
                    if selection=='teacher':ids=rows[teacher.bool()]
                    else:
                        data,ids,_=scene.renderer.prepare(camera,ratio_budget=ratio,compensation=False,view_conditioned=selection=='view')
                        del data
                    p=prior[(frame,ratio,selection)];assert len(ids)==int(p['K'])
                    digest=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
                    changed[selection]+=digest!=p['selection_sha256']
                    if selection!='teacher':assert digest==p['selection_sha256']
                    values=q[ids];rel=values/mean
                    a=values.cpu().numpy();b=rel.cpu().numpy();key=(ratio,selection)
                    absolute[key].append(a);relative[key].append(b)
                    row=dict(frame=frame,ratio=ratio,selection=selection,K=len(ids),eligible_N=len(pool),positive_pool_N=positive,
                        zero_points=int((a==0).sum()),below_tenth_pool_mean=int((b<.1).sum()),below_pool_mean=int((b<1).sum()),
                        pool_mean_contribution=float(mean),selected_mean_contribution=float(a.mean()),selected_mean_relative=float(b.mean()),
                        selected_mass_fraction=float(values.sum()/pool.sum()),matches_prior_ids=digest==p['selection_sha256'])
                    row.update({'contribution_'+k:v for k,v in quantiles(a).items()})
                    row.update({'relative_'+k:v for k,v in quantiles(b).items()})
                    if writer is None:writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
                    writer.writerow(row);all_rows.append(row)
            if frame%40==0 or frame==791:
                f.flush();state=dict(state='running',frame=frame+1,frames=792,rows=len(all_rows),seconds=time.monotonic()-started)
                e.save(ROOT/'reports/STATUS.json',state);print(state,flush=True)
    summary=[]
    for key in sorted(absolute):
        a=np.concatenate(absolute[key]);b=np.concatenate(relative[key]);r,s=key
        item=dict(ratio=r,selection=s,total_selected_observations=len(a),zero_points=int((a==0).sum()),zero_fraction=float((a==0).mean()),
            below_tenth_pool_mean_points=int((b<.1).sum()),below_tenth_pool_mean_fraction=float((b<.1).mean()),
            below_pool_mean_fraction=float((b<1).mean()),mean_contribution=float(a.mean()),mean_relative=float(b.mean()))
        item.update({'contribution_'+k:v for k,v in quantiles(a).items()});item.update({'relative_'+k:v for k,v in quantiles(b).items()})
        summary.append(item)
    e.save(ROOT/'reports/SUMMARY.json',summary)
    with (ROOT/'evaluation/SUMMARY.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    assert all(e.sha(Path(n))==h for n,h in inputs.items());scene.renderer.check_state()
    e.save(ROOT/'reports/INPUTS_AFTER.json',inputs)
    e.save(ROOT/'reports/STATUS.json',dict(state='complete',passed=True,frames=792,rows=len(all_rows),
        selected_ids_changed_vs_prior=dict(changed),optimizer_updates=0,images_saved=0,seconds=time.monotonic()-started))
    print(json.dumps(dict(summary=summary,changed_ids=dict(changed)),indent=2))


if __name__=='__main__':
    try:main()
    except Exception as error:
        e.save(ROOT/'reports/STATUS.json',dict(state='failed',error=repr(error)));raise
