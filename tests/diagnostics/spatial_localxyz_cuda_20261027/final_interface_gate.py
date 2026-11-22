"""验收配置入口、共享参数清单、原Checkpoint和生产源码未修改。"""
import hashlib
import json
from pathlib import Path
import torch
from baseline_profile import ROOT,REFERENCE,r,fixture
from decoded_graph import make_render_execution
from implementation import make_execution
from parity import compare


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.backends.cuda.matmul.allow_tf32=False;rows=[];inventory=[];models=[]
    baseline=json.loads((REFERENCE/'environment_final.json').read_text());project=Path('/home/prmi/zwj/mycartgs')
    current={name:sha(project/name) for name in baseline['source_sha256']}
    unchanged=current==baseline['source_sha256'];assert unchanged
    (ROOT/'production_source_verification.json').write_text(json.dumps(dict(production_changed=False,all_match=unchanged,source_sha256=current,compared_to=str(REFERENCE/'environment_final.json')),indent=2))
    cfg=json.loads((ROOT/'spatial_localxyz_default.json').read_text())
    for scene in ['fr1_xyz','fr2_xyz']:
        sc,f,n,p=fixture(scene);same=r.make(f,n.cfg,5100);same.load_state_dict(p['state_dict'])
        ref=make_render_execution(n,sc.render_bases,'reference');ex=make_render_execution(same,sc.render_bases)
        ids=f.sets[.01,'same_k']
        with torch.no_grad():
            aa=ref(.01,ids);bb=ex(.01,ids)
            for j,(x,y) in enumerate(zip(aa,bb)):compare(x,y,f'{scene}/decoded/{j}',rows,'configured_default',len(ids))
            comp=make_execution(n)
            compare(comp(.01,ids),aa[0],scene+'/compensation_only',rows,'configured_default',len(ids))
        assert ex.method=='decoded_graph' and comp.method=='graph_cached55_fused'
        for name,value in n.named_parameters():inventory.append(dict(scene=scene,name=name,shape=str(list(value.shape)),numel=value.numel(),bytes=value.numel()*value.element_size(),dtype=str(value.dtype),scope='shared network weights; independent of N and M'))
        models.append(dict(scene=scene,N=sc.N,M=sc.M,K={str(ratio):len(f.sets[ratio,'same_k']) for ratio in r.rt.RATIOS},checkpoint=cfg['checkpoints'][scene],checkpoint_sha256=sha(cfg['checkpoints'][scene]),parameters=sum(v.numel() for v in n.parameters())))
        assert models[-1]['parameters']==33806 and sc.unchanged()
    assert [(v['name'],v['shape']) for v in inventory if v['scene']=='fr1_xyz']==[(v['name'],v['shape']) for v in inventory if v['scene']=='fr2_xyz']
    r.rt.write_csv(ROOT/'parameter_inventory.csv',inventory);r.rt.write_csv(ROOT/'configured_default_parity.csv',rows)
    (ROOT/'model_manifests.json').write_text(json.dumps(models,indent=2))
    (ROOT/'final_interface_gate.json').write_text(json.dumps(dict(pass_=all(v['pass_'] for v in rows),rows=len(rows),production_unchanged=unchanged,identity_parameters=False),indent=2));assert all(v['pass_'] for v in rows)
    print('FINAL_INTERFACE_PASS',len(rows),flush=True)


if __name__=='__main__':main()
