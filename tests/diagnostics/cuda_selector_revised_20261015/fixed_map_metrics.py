"""四场景独立固定地图 equal-updates：质量及 legacy 自重复基线。"""
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pandas as pd
import torch
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

dest = Path(sys.argv[1])
metrics = [PeakSignalNoiseRatio(data_range=1.).cuda(), StructuralSimilarityIndexMeasure(data_range=1.).cuda(),
           LearnedPerceptualImagePatchSimilarity(normalize=True).cuda()]


def read(path):
    with path.open('rb') as stream:
        count, = struct.unpack('<q', stream.read(8))
        data = np.frombuffer(stream.read(), dtype='<f4').copy()
    assert count == data.size == 3*240*320
    return torch.from_numpy(data).reshape(1, 3, 240, 320).cuda().clamp(0, 1)


rows = []
with torch.no_grad():
    for path in sorted(dest.glob('independent_ratio*_gt.bin')):
        stem = path.name.removesuffix('_gt.bin')
        row = {'case': stem}
        gt = read(path)
        for backend in ['legacy', 'repeat', 'cuda']:
            image = read(dest/(stem+'_'+backend+'.bin'))
            row[backend] = {}
            for key, metric in zip(['psnr', 'ssim', 'lpips'], metrics):
                metric.reset()
                row[backend][key] = float(metric(image, gt))
        row['checks'] = {}
        for key, margin in dict(psnr=.02, ssim=.0002, lpips=.0002).items():
            repeat_error = abs(row['repeat'][key]-row['legacy'][key])
            delta = row['cuda'][key]-row['legacy'][key]
            threshold = max(margin, 2*repeat_error)
            row['checks'][key] = dict(delta=delta, legacy_repeat_error=repeat_error,
                                       threshold=threshold, passed=abs(delta) <= threshold)
        rows.append(row)
assert len(rows) == 12
summary = dict(quality=rows, quality_passed=all(c['passed'] for r in rows for c in r['checks'].values()),
               scope='Three fixed cameras at 320x240, four ratios, 1000 Full+LoD updates; independent Gaussian/Selector/TF graphs; no topology or SLAM transport')
summary['runs'] = {}
for label in ['legacy', 'repeat', 'cuda']:
    steps = pd.read_csv(dest/(label+'_steps.csv'))
    summary['runs'][label] = dict(committed_updates=len(steps), training_wall_seconds=float(steps.training_ms.sum()/1000),
        selector_forward_seconds=float(steps.selector_forward_ms.sum()/1000),
        average_hard_mismatch=float(steps.mismatch_ratio.mean()), p99_hard_mismatch=float(steps.mismatch_ratio.quantile(.99)),
        final_loss=float(steps.loss.iloc[-1]))
    if label != 'legacy':
        parity = pd.read_csv(dest/(label+'_parity.csv'))
        summary['runs'][label]['selector'] = {}
        for field in ['parameter', 'gradient', 'adam_m', 'adam_v']:
            part = parity[parity.field == field]
            summary['runs'][label]['selector'][field] = dict(max_relative_l2=float(part.relative_l2.max()),
                minimum_cosine=float(part.cosine.min()), max_abs=float(part.max_abs.max()))
summary['training_wall_speedup'] = summary['runs']['legacy']['training_wall_seconds']/summary['runs']['cuda']['training_wall_seconds']
(dest/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps({k: v for k, v in summary.items() if k != 'quality'}, indent=2))
if not summary['quality_passed']:
    raise SystemExit(1)
