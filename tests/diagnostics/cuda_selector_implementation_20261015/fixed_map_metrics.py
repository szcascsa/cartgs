"""固定训练后的三个实际相机帧，分别计算两后端的 PSNR/SSIM/LPIPS。"""
import json
from pathlib import Path
import struct
import sys
import numpy as np
import torch
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

dest = Path(__file__).resolve().parent
prefix = sys.argv[1] if len(sys.argv)>1 else 'fixed_map'
assert prefix in ('fixed_map', 'joint_fixed_map')
metrics = [PeakSignalNoiseRatio(data_range=1.).cuda(), StructuralSimilarityIndexMeasure(data_range=1.).cuda(),
           LearnedPerceptualImagePatchSimilarity(normalize=True).cuda()]


def read(path):
    with path.open('rb') as stream:
        count, = struct.unpack('<q', stream.read(8))
        values = np.frombuffer(stream.read(), dtype='<f4').copy()
    assert len(values) == count == 3*240*320
    return torch.from_numpy(values).reshape(1,3,240,320).cuda().clamp(0,1)


rows = []
with torch.no_grad():
    for path in sorted(dest.glob(prefix+'_ratio*_legacy.bin')):
        stem = path.name.removesuffix('_legacy.bin')
        a = read(path)
        b = read(dest/(stem+'_cuda.bin'))
        gt = read(dest/(stem+'_gt.bin'))
        assert torch.equal(a,b), stem+' image mismatch'
        values = {}
        for name, image in [('legacy',a),('cuda',b)]:
            scores = {}
            for key, metric in zip(['PSNR','SSIM','LPIPS'], metrics):
                metric.reset()
                scores[key] = float(metric(image,gt))
            values[name] = scores
        assert values['legacy'] == values['cuda']
        rows.append(dict(case=stem, image_exact=True, **values))
assert len(rows) == 12
(dest/(prefix+'_quality.json')).write_text(json.dumps(dict(rows=rows,
    limit=('Three fixed frames, four ratios, all Gaussian/Selector/TF parameters updated; independent Selector/input VJPs with shared unchanged renderer/TF VJP' if prefix=='joint_fixed_map' else
           'Three fixed frames, four ratios, frozen map/TF and shared real renderer VJP; not full online SLAM or joint Gaussian/TF training')),indent=2)+'\n')
print('12 fixed-map PSNR/SSIM/LPIPS cases exact')
