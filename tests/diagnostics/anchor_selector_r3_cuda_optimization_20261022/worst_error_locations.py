"""为最终完整链路误差保存最差位置、对应值及Gaussian/Anchor输入身份。"""
import json
from pathlib import Path
import torch
import pandas as pd
R=Path('/home/prmi/zwj/anchor-selector-r3-cuda-optimization-20261022');torch.set_num_threads(2);rows=[]
for case in ['Full','lod_0.010000','lod_0.050000','lod_0.100000','lod_0.150000']:
    a=torch.jit.load(str(R/'legacy_a'/(case+'_parity.pt')),map_location='cpu').state_dict()
    for run in ['legacy_b','adam','adam_decode','fused_a','fused_b']:
        b=torch.jit.load(str(R/run/(case+'_parity.pt')),map_location='cpu').state_dict()
        for name,x in a.items():
            y=b[name]
            if not x.numel() or x.shape!=y.shape:continue
            error=(x.double()-y.double()).abs();flat=int(error.reshape(-1).argmax());index=[];remaining=flat
            for size in reversed(x.shape):index.append(remaining%size);remaining//=size
            index=list(reversed(index));gaussian=None;anchor=None
            if index and x.shape[0]==len(a['owner']):gaussian=index[0]
            elif index and name in ['xyz','scale','rotation','opacity','dc','residual'] and 'selected_ids' in a:gaussian=int(a['selected_ids'][index[0]])
            elif index and name in ['gradient_6','parameter_after_6','parameter_before_6']:anchor=index[0]
            inputs={}
            if gaussian is not None:
                anchor=int(a['owner'][gaussian]);inputs=dict(gaussian_row=gaussian,uid=int(a['gaussian_uid'][gaussian]),
                    anchor_owner=anchor,base_xyz=a['parameter_before_0'][gaussian].tolist(),
                    base_logscale=a['parameter_before_4'][gaussian].tolist(),base_rotation=a['parameter_before_5'][gaussian].tolist(),
                    base_opacity_logit=a['parameter_before_3'][gaussian].tolist())
            elif anchor is not None:inputs=dict(anchor_row=anchor,anchor_center=a['anchor_centers'][anchor].tolist())
            rows.append(dict(case=case,run=run,tensor=name,worst_flat_index=flat,index=json.dumps(index),reference=float(x.reshape(-1)[flat]),
                candidate=float(y.reshape(-1)[flat]),max_abs=float(error.reshape(-1)[flat]),input_identity=json.dumps(inputs)))
pd.DataFrame(rows).to_csv(R/'worst_error_locations.csv',index=False)
print('PASS error locations',len(rows),flush=True)
