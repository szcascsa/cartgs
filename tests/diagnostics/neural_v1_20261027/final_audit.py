"""最终数据完整性和共享参数核查，不扩展生产测试范围。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import run as r
from neural import audit_parameters

def main():
    rows=[];associations=[];model_signatures={};checks=[]
    for scene_name,source_seed,seeds,tier in [('fr1_xyz',3100,[5100,5101],'confirm'),('fr2_xyz',4100,[6100,6101],'fr2_confirm')]:
        scene=r.rt.Scene(scene_name);fixture=r.Fixture(scene,source_seed)
        counts=torch.bincount(scene.owners,minlength=scene.M)
        associations.append(dict(scene=scene_name,N=scene.N,M_stored=scene.M,occupied_anchors=int((counts>0).sum()),empty_anchors=int((counts==0).sum()),
            min_occupied=int(counts[counts>0].min()),max_occupied=int(counts.max()),occupancy_histogram=torch.bincount(counts).cpu().tolist(),
            valid_owner_range=bool((scene.owners>=0).all()&(scene.owners<scene.M).all()),frozen_map=True))
        for name in ['SpatialFiLM','SpatialOutputKnots']:
            for seed in seeds:
                d=r.ROOT/'results'/tier/scene_name/name/str(seed);info=json.loads((d/'complete.json').read_text());net=r.make(fixture,info['config'],seed)
                state=torch.load(info['checkpoint'],map_location='cuda')['state_dict'];net.load_state_dict(state)
                params=audit_parameters(net);signature=[(p['name'],p['shape']) for p in params]
                if name in model_signatures:assert signature==model_signatures[name]
                else:model_signatures[name]=signature
                for entry in params:rows.append(dict(scene=scene_name,config=name,seed=seed,**entry))
                assert set(state)=={name for name,p in net.named_parameters()},'state_dict contains non-parameter scene buffers'
                del net,state
        assert scene.unchanged();del fixture,scene;torch.cuda.empty_cache()
    for path in sorted((r.ROOT/'results').glob('*/*/*/*/formal/test_per_view.csv')):
        data=pd.read_csv(path);manifest=json.loads((path.parent/'manifest.json').read_text());scene_name=data.scene.iloc[0]
        spec=json.loads((r.ROOT/'results'/scene_name/'scene_manifest.json').read_text())
        ratios=sorted(data.ratio.unique());assert ratios==r.rt.RATIOS
        for policy in ['same_k','cached']:
            subset=data[data.policy==policy];counts=subset.groupby('ratio').size();assert counts.nunique()==1
            assert subset[['psnr','ssim','lpips','photo']].apply(np.isfinite).all().all()
            if policy=='same_k':
                for ratio,g in subset.groupby('ratio'):assert g.selected.nunique()==1 and int(g.selected.iloc[0])==spec['K'][str(ratio)]
        assert set(data.path).isdisjoint(spec['train_paths']+spec['validation_paths'])
        assert r.rt.sha(Path(manifest['checkpoint']))==manifest['checkpoint_sha256']
        checks.append(dict(path=str(path.relative_to(r.ROOT)),rows=len(data),sameK_pass=True,finite=True,disjoint_Bank=True,checkpoint_SHA_pass=True))
    r.rt.write_csv(r.ROOT/'NEURAL_V1_TRAINABLE_PARAMETERS.csv',rows)
    r.rt.save_json(r.ROOT/'final_validation.json',dict(topology=associations,shared_parameter_shapes_independent_of_N_M=True,
        no_candidate_identity_parameters=True,formal_metrics=checks,production_unchanged='verified separately by source SHA and Git'))
    print('FINAL_AUDIT_PASS',len(checks),len(rows),flush=True)

if __name__=='__main__':main()
