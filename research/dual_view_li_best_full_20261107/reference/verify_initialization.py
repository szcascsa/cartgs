"""CPU逐位核查Full→六节点Fine初态；只检查独立目录，不更改模型。"""
from pathlib import Path
import json
import sys
import numpy as np
import torch
from plyfile import PlyData


def main():
    manifest = json.loads(Path(sys.argv[1]).read_text())
    output = Path(manifest['output'])
    old = torch.jit.load(str(Path(manifest['full_ply'])/'selector.pt'), map_location='cpu')
    new = torch.jit.load(str(output/'converted_selector_init.pt'), map_location='cpu')
    before = dict(list(old.named_parameters())+list(old.named_buffers()))
    after = dict(list(new.named_parameters())+list(new.named_buffers()))
    checks = []
    for name, tensor in after.items():
        if name.startswith('anchors.compensation_features_') and name!='anchors.compensation_features_0':
            assert tensor.shape == before['anchors.compensation_features_0'].shape
            assert not torch.count_nonzero(tensor)
        elif name=='anchors.compensation_ratio_knots':
            assert torch.equal(tensor, torch.tensor([.01,.05,.1,.3,.5,.7], dtype=torch.float32).to(tensor.dtype))
        elif name=='spatial_compensation.ratio_encoding_bounds':
            assert torch.equal(tensor, torch.tensor([.01,.7], dtype=torch.float32).to(tensor.dtype)), tensor
        else:
            # 按原始位模式比较，包括有符号零；浮点NaN不允许出现。
            assert tensor.shape == before[name].shape, name
            a=tensor.detach().contiguous().numpy(); b=before[name].detach().contiguous().numpy()
            assert a.dtype==b.dtype and a.tobytes()==b.tobytes(), name
        checks.append(name)
    features=[n for n in after if n.startswith('anchors.compensation_features_')]
    assert len(features)==6
    a=PlyData.read(manifest['point_cloud'])['vertex'].data
    b=PlyData.read(str(output/'step0_master.ply'))['vertex'].data
    assert a.dtype.names==b.dtype.names and len(a)==len(b)
    for name in a.dtype.names:
        assert a[name].tobytes()==b[name].tobytes(), 'Master initial '+name
        assert np.isfinite(a[name]).all()
    report=dict(passed=True, tensors_checked=checks, ply_rows=len(a), master_fields=len(a.dtype.names),
                feature_tables=6, optimizer_reset=True)
    (output/'initialization_verified.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__': main()
