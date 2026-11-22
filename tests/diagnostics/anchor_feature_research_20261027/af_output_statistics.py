"""使用物理变换后的量研究Feature作用，不直接比较原14D欧氏距离。"""
import json
import numpy as np
import torch
from af_runtime import ROOT,Study,RATIOS,csv,old


def distribution(rows,values,info,metric):
    rows.append(dict(info,metric=metric,mean=float(values.mean()),median=float(values.median()),p90=float(values.quantile(.9)),maximum=float(values.max()),count=len(values)))


def main():
    torch.backends.cuda.matmul.allow_tf32=False;rows=[]
    with torch.no_grad():
        for scene in ['fr1_xyz','fr2_xyz']:
            s=Study(scene);initial=s.make(dict(feature='none'),7200)
            for name in ['Head8','FrozenBaseHead8','FrozenBaseGlobalHead8','FrozenBaseContextHead8']:
                tier='confirm_continue' if name=='Head8' else 'mechanism_continue' if name=='FrozenBaseHead8' and scene=='fr1_xyz' else 'frozen_mechanism'
                for seed in [7200,7201]:
                    p=ROOT/'results'/tier/scene/name/str(seed)/'complete.json';info=json.loads(p.read_text());net=s.make(info['config'],seed);net.load_state_dict(torch.load(info['checkpoint'],map_location='cuda')['state_dict'])
                    for r in RATIOS:
                        ids=s.fixture.sets[r,'same_k'];delta=net(r,ids);base=initial(r,ids);own=s.owners[ids];occupancy=s.counts[own]
                        def quaternion(value):
                            ident=torch.zeros_like(value[:,6:10]);ident[:,0]=1
                            return old.rt._normalize_quaternion(old.rt._multiply_quaternion(old.rt._normalize_quaternion(ident+value[:,6:10]),s.scene.g.get_rotation[ids]))
                        angle=2*torch.acos((quaternion(delta)*quaternion(base)).sum(1).abs().clamp(0,1))*180/np.pi
                        metrics=dict(xyz_world=(delta[:,:3]-base[:,:3]).norm(dim=1),xyz_anchor_units=((delta[:,:3]-base[:,:3])/s.anchor_scale[ids]).norm(dim=1),
                            logscale_change=(delta[:,3:6]-base[:,3:6]).norm(dim=1),rotation_geodesic_degrees=angle,
                            activated_opacity_change=((s.scene.g._opacity[ids]+delta[:,10:11]).sigmoid()-(s.scene.g._opacity[ids]+base[:,10:11]).sigmoid()).abs().flatten(),
                            sh_dc_change=(delta[:,11:14]-base[:,11:14]).norm(dim=1))
                        for label,mask in [('all',torch.ones_like(occupancy,dtype=torch.bool)),('one',occupancy==1),('two_three',(occupancy>=2)&(occupancy<=3)),('four_ten',occupancy>=4)]:
                            if not mask.any():continue
                            for metric,values in metrics.items():distribution(rows,values[mask],dict(scene=scene,model=name,seed=seed,ratio=r,bin=label),metric)
    csv(ROOT/'physical_output_changes.csv',rows);print('PHYSICAL_OUTPUT_STATS_COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
