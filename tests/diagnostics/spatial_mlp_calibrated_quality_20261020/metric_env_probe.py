"""定位原LPIPS评估的cuDNN设置差异，不修改metric/容差。"""
import json
import numpy as np
import torch
from run_quality import ROOT,OLD,renderer,read

module=renderer('fr1_room');info=read(ROOT/'fr1_room/input_manifest.json')
rows=np.load(OLD/'selected_sets.npz')['fr1_room_Raw_0.01'];scene=module.Scene();results=[]
try:
    metric=module.Metrics(module.tensors(info['cameras'])['gt'])
    scene.call(dict(method='images',rows=rows.tolist(),ratio=.01,frames=[0,1,2,3],apply_tf=True))
    images=torch.from_numpy(np.array(scene.mapping[:4])).cuda()
    for enabled in [False,True]:
        torch.backends.cudnn.allow_tf32=enabled
        values=metric.all(images,[0,1,2,3]);results.append(dict(cudnn_TF32=enabled,LPIPS_frame0=float(values[0,2])))
finally:scene.close()
(ROOT/'metric_environment_probe.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results))
