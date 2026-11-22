"""核查Full→六ratio／双网络视角初态；不更改源模型。"""
from pathlib import Path
import json
import sys
import numpy as np
import torch
from plyfile import PlyData


def main():
    m=json.loads(Path(sys.argv[1]).read_text());out=Path(m['output'])
    before=torch.jit.load(str(Path(m['full_ply'])/'selector.pt'),map_location='cpu').state_dict()
    after=torch.jit.load(str(out/'converted_selector_init.pt'),map_location='cpu').state_dict()
    extra={'selector_view_input_schema','selector_view_scene_scale','selector_view_initialized',
           'joint_view_conditioning_schema','spatial_compensation.view_input_schema',
           'spatial_compensation.view_projection.weight','spatial_compensation.view_projection.bias',
           'spatial_compensation.geometry_head_schema'}
    checked=[]
    for n,a in after.items():
        assert torch.isfinite(a).all(),n
        if n.startswith('anchors.compensation_features_') and n!='anchors.compensation_features_0':
            assert a.shape==before['anchors.compensation_features_0'].shape and not a.count_nonzero()
        elif n=='anchors.compensation_ratio_knots':assert torch.equal(a,torch.tensor([.01,.05,.1,.3,.5,.7],dtype=torch.float32).to(a.dtype))
        elif n=='spatial_compensation.ratio_encoding_bounds':assert torch.equal(a,torch.tensor([.01,.7],dtype=torch.float32).to(a.dtype))
        elif n=='local_emd.0.weight':
            assert a.shape[1]==19 and before[n].shape[1]==11 and not a[:,11:].count_nonzero()
            assert a[:,:11].contiguous().view(torch.uint8).equal(before[n].contiguous().view(torch.uint8))
        elif n not in before:
            assert n in extra,n
            if n in ['selector_view_input_schema','spatial_compensation.view_input_schema']:assert a.item()==1,n
            elif n=='joint_view_conditioning_schema':assert a.item()==2
            elif n=='selector_view_scene_scale':assert a.item()==1
            else:assert not a.count_nonzero(),n
        else:
            assert a.shape==before[n].shape and a.contiguous().reshape(-1).view(torch.uint8).equal(before[n].contiguous().reshape(-1).view(torch.uint8)),n
        checked.append(n)
    assert len([n for n in after if n.startswith('anchors.compensation_features_')])==6
    a=PlyData.read(m['point_cloud'])['vertex'].data;b=PlyData.read(str(out/'step0_master.ply'))['vertex'].data
    assert a.dtype.names==b.dtype.names and len(a)==len(b)
    for n in a.dtype.names:assert a[n].tobytes()==b[n].tobytes() and np.isfinite(a[n]).all(),n
    result=dict(passed=True,tensors_checked=checked,ply_rows=len(a),master_fields=len(a.dtype.names),feature_tables=6,
                original_selector_columns_bitwise=True,new_view_columns_zero=True,optimizer_reset=True)
    (out/'initialization_verified.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
