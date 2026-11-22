"""复用冻结评价路径，仅渲染预先选定的五个视角；保持原选点与原图口径。"""
from pathlib import Path
import json
import runpy
import sys

ROOT = Path(sys.argv[1])
SCOPE = sys.argv[2]
metadata = json.loads((ROOT / 'EXPERIMENT.json').read_text())
BATCH = Path(metadata['batch'])
output = ROOT / {'full': 'full_0', 'global': 'global_0.5', 'view_ratio': 'local_0.5'}[SCOPE]
sys.path[:0] = [str(BATCH / 'source/python'), str(BATCH / 'evaluation_runtime')]

import cv2
import numpy as np
import torch
from evo.tools.settings import SETTINGS

SETTINGS.plot_backend = 'Agg'
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

if SCOPE == 'view_ratio':
    import anchor_frustum
    from anchor_budget import FixedBudgetRenderer

    class LocalRatioRenderer:
        def __init__(self, model, gaussians, ratio, config, temperature, protection):
            assert int(model.joint_view_conditioning_schema) == 2
            self.ratio = ratio
            self.renderer = FixedBudgetRenderer(model, gaussians,
                dict(config, temperature=temperature, protection_enabled=protection), seed=0, backend='pytorch')

        def prepare(self, camera):
            data, ids, stats = self.renderer.prepare(camera, ratio_budget=self.ratio, view_conditioned=True)
            assert stats['view_conditioned'] and len(ids) == min(stats['requested_budget'], stats['eligible_candidates'])
            stats['legacy_quantile_budget'] = False
            return data, ids, stats

    anchor_frustum.AnchorFrustumRenderer = LocalRatioRenderer

original_imwrite = cv2.imwrite


def save_png(filename, pixels, *args, **kwargs):
    target = Path(filename)
    if target.parent.name == 'gt' and SCOPE != 'full':
        return True
    if target.parent.name in ['gt', 'image']:
        target = target.with_suffix('.png')
    assert original_imwrite(str(target), pixels, *args, **kwargs), str(target)
    return True


cv2.imwrite = save_png
script = BATCH / 'source/python/run_metrics_only.py'
arguments = list(metadata['launch_arguments'][SCOPE])
arguments.remove('--no-save-images')
arguments[arguments.index('--output_path') + 1] = str(output)
if '--joint-local-ratio' in arguments:
    arguments.remove('--joint-local-ratio')
    arguments.append('--selector-anchor-frustum')
arguments += ['--evaluation-frame-set', str(ROOT / 'FRAME_SET.json')]
sys.argv = [str(script), *arguments]
values = runpy.run_path(str(script), run_name='__main__')
rows = []
for index, (trajectory, rgb_index) in enumerate(values['associations']):
    pose = np.asarray(values['poses'][trajectory])
    stats = values['frustum_stats'][index] if SCOPE == 'view_ratio' else values['selector_stats']
    record = dict(scope=SCOPE, gt_index=int(rgb_index), trajectory_index=int(trajectory),
        timestamp=float(values['tstamp'][trajectory]), rgb_path=values['gt_color_paths'][rgb_index],
        c2w=pose.tolist(), w2c=np.linalg.inv(pose).tolist(),
        psnr=float(values['psnr_list'][index]), ssim=float(values['ssim_list'][index]),
        lpips=float(values['lpips_list'][index]), K=int(stats['active_gaussians']),
        candidate_N=int(stats['candidate_gaussians']) if SCOPE == 'view_ratio' else None,
        requested_budget=int(stats['requested_budget']) if SCOPE == 'view_ratio' else None,
        image=str(output / 'image' / Path(values['gt_color_paths'][rgb_index]).with_suffix('.png').name))
    assert Path(record['image']).is_file()
    assert all(np.isfinite(record[name]) for name in ['psnr', 'ssim', 'lpips'])
    rows.append(record)
assert [r['gt_index'] for r in rows] == metadata['gt_indices']
(output / 'SELECTED_FRAME_RESULTS.json').write_text(json.dumps(rows, indent=2) + '\n')
print('RENDER_PASS', SCOPE, len(rows), flush=True)
