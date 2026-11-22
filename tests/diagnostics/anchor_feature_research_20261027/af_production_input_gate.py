"""直接调用当前生产Python的实时55D定义，核对研究缓存数学。"""
import torch
from af_runtime import ROOT,Study,RATIOS,save
torch.backends.cuda.matmul.allow_tf32=False;rows=[]
for scene in ['fr1_xyz','fr2_xyz']:
    s=Study(scene);net=s.make({'feature':'none'},7200);sc=s.scene;g=sc.g
    for r in RATIOS:
        ids=s.fixture.sets[r,'same_k'];owners=s.owners[ids]
        args=(g.get_xyz[ids],g.get_scaling[ids],g.get_rotation[ids],g.get_features_dc[ids],g._opacity[ids],sc.source.anchors.centers[owners],sc.source.anchors.offset_scales[owners])
        with torch.no_grad():
            live=net.base.input55(*args);cached=s.x[ids];output=net.base(*args,r);reference=net(r,ids)
            error=float((live-cached).abs().max());out_error=float((output-reference).abs().max())
            assert torch.allclose(live,cached,atol=1e-6,rtol=1e-5) and torch.allclose(output,reference,atol=1e-6,rtol=1e-5)
            rows.append(dict(scene=scene,ratio=r,physical55_max_abs=error,output_max_abs=out_error,passed=True))
save(ROOT/'production_input_gate.json',dict(rows=rows,scope='current production Python live inputs vs cached55; fixed matching population statistics; no claim of online or CUDA-gradient lifecycle validation'))
print('PRODUCTION_INPUT_GATE_PASS',len(rows),flush=True)
